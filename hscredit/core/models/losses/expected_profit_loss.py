"""
期望利润损失函数

将收益、坏账损失和通过决策融合为连续可微的期望利润优化目标。
相比 ProfitMaxLoss 使用硬阈值决策，本模块通过 sigmoid 软通过门
实现全程可微的利润优化，梯度更平滑，收敛更稳定。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid 函数。"""
    pos_mask = x >= 0
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    z[~pos_mask] = exp_x / (1.0 + exp_x)
    return z


class ExpectedProfitLoss(BaseLoss):
    """期望利润损失函数，将收益、坏账损失和通过决策融合为连续可微的期望利润优化目标。

    相比 :class:`ProfitMaxLoss` 使用硬阈值决策（不可微），本损失通过 sigmoid
    软通过门实现全程可微的利润优化，梯度更平滑，收敛更稳定。

    数学形式::

        approve_i = σ((cutoff - p_i) / temperature)
        profit_i  = (1 - y_i) × revenue  -  y_i × default_cost
        E[profit] = mean(approve_i × profit_i)
        Loss      = -E[profit] + bce_weight × BCE(y, p)

    :param revenue: 好客户通过后的单位收益，默认 1.0
        - 内部经验: 可设为平均利息收入或客均贡献
    :param default_cost: 坏客户通过后的单位损失，默认 10.0
        - 内部经验: 通常为 revenue 的 5~20 倍，视业务坏账回收率而定
    :param cutoff: 软通过阈值，预测概率低于此值的样本倾向通过，默认 0.5
    :param temperature: sigmoid 平滑温度，越小通过决策越接近硬阈值，默认 0.1
        - 内部经验: 0.05~0.2 之间效果较好；太小梯度集中在阈值附近，太大退化为线性
    :param bce_weight: 基础交叉熵正则权重，防止利润梯度消失，默认 0.1
    :param name: 损失函数名称，默认 "expected_profit_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import ExpectedProfitLoss
    >>>
    >>> # 每笔贷款利息收益 100 元，坏账损失 1000 元
    >>> loss = ExpectedProfitLoss(
    ...     revenue=100,
    ...     default_cost=1000,
    ...     cutoff=0.5,
    ...     temperature=0.1
    ... )
    >>>
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 XGBoost 中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> bst = xgb.train({}, dtrain, obj=loss.to_xgboost(), num_boost_round=100)
    """

    def __init__(
        self,
        revenue: float = 1.0,
        default_cost: float = 10.0,
        cutoff: float = 0.5,
        temperature: float = 0.1,
        bce_weight: float = 0.1,
        name: str = "expected_profit_loss",
    ):
        super().__init__(name)
        self.revenue = revenue
        self.default_cost = default_cost
        self.cutoff = cutoff
        self.temperature = temperature
        self.bce_weight = bce_weight

    def _approve_gate(self, y_pred: np.ndarray) -> np.ndarray:
        """计算软通过概率 σ((cutoff - p) / T)。"""
        z = (self.cutoff - y_pred) / self.temperature
        return _sigmoid(z)

    def _sample_profit(self, y_true: np.ndarray) -> np.ndarray:
        """计算每个样本的利润标签（好客户正利润，坏客户负利润）。"""
        return (1 - y_true) * self.revenue - y_true * self.default_cost

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算期望利润损失。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 损失值（负期望利润 + BCE 正则）
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        approve = self._approve_gate(y_pred)
        profit = self._sample_profit(y_true)

        # 负期望利润
        profit_loss = -np.mean(approve * profit)

        # BCE 正则
        bce = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        return float(profit_loss + self.bce_weight * bce)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        推导::

            dL/dp_i = σ_i(1-σ_i)/T × profit_i  +  bce_weight × (p_i - y_i)

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        approve = self._approve_gate(y_pred)
        profit = self._sample_profit(y_true)

        # 利润梯度: dL/dp = -d(approve)/dp × profit
        # d(approve)/dp = -σ(1-σ)/T  →  dL/dp = σ(1-σ)/T × profit
        grad_profit = approve * (1 - approve) / self.temperature * profit

        # BCE 梯度
        grad_bce = self.bce_weight * (y_pred - y_true)

        return grad_profit + grad_bce

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算二阶导数。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 二阶导数数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        approve = self._approve_gate(y_pred)
        profit = self._sample_profit(y_true)

        # d²L/dp² = -profit × σ(1-σ)(1-2σ) / T²
        hess_profit = (
            -profit
            * approve
            * (1 - approve)
            * (1 - 2 * approve)
            / (self.temperature ** 2)
        )

        # BCE 二阶导
        hess_bce = self.bce_weight * y_pred * (1 - y_pred)

        hess = hess_profit + hess_bce

        # 确保非零正值
        return np.maximum(np.abs(hess), 1e-6)
