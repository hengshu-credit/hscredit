"""
期望利润损失函数

将收益、坏账损失和通过决策融合为连续可微的期望利润优化目标。
本模块通过 sigmoid 软通过门构造可微的利润代理，并用交叉熵正则约束概率拟合。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, nonnegative, positive, unit_interval


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

    本损失通过 sigmoid 软通过门构造可微的利润代理。
    :class:`ProfitMaxLoss` 则使用收益和坏账成本加权 BCE，两者目标不同。

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

    **引用**

    成本敏感学习与基于期望收益/损失的最优决策阈值见 Elkan, C. (2001). *The Foundations
    of Cost-Sensitive Learning.* IJCAI 2001.
    https://cseweb.ucsd.edu/~elkan/rescale.pdf 。本损失以 sigmoid 软通过门将利润目标
    转为可微形式，属业务驱动设计。
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
        nonnegative(revenue=revenue, default_cost=default_cost, bce_weight=bce_weight)
        unit_interval(cutoff=cutoff)
        positive(temperature=temperature)
        self.is_additive = True
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

    def __call__(self, y_true, y_pred) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedProfitLoss
        >>> loss = ExpectedProfitLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        """相对概率求导：利润项 + bce_weight × (p-y)/(p(1-p))。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedProfitLoss
        >>> loss = ExpectedProfitLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        gate = self._approve_gate(p)
        return gate * (1 - gate) / self.temperature * self._sample_profit(y) + self.bce_weight * bce_terms(y, p)[1]

    def hessian(self, y_true, y_pred):
        """返回真实概率二阶导；非凸区域可以为负，由训练适配器稳定化。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedProfitLoss
        >>> loss = ExpectedProfitLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        gate = self._approve_gate(p)
        profit_hess = -self._sample_profit(y) * gate * (1 - gate) * (1 - 2 * gate) / self.temperature**2
        return profit_hess + self.bce_weight * bce_terms(y, p)[2]

    def loss_values(self, y_true, y_pred):
        """负软通过利润与 BCE 正则的逐样本损失。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的数组，其均值等于本损失值；详见 BaseLoss.loss_values。

        **参考样例**

        >>> from hscredit.core.models.losses import ExpectedProfitLoss
        >>> loss = ExpectedProfitLoss()
        >>> result = loss.loss_values([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y, p = binary_inputs(y_true, y_pred)
        return -self._approve_gate(p) * self._sample_profit(y) + self.bce_weight * bce_terms(y, p)[0]
