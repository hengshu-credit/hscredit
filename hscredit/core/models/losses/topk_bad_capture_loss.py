"""
Top-K 坏样本捕获损失函数

专门优化 top x% 坏样本捕获率（Bad Capture Rate），
很适合策略、催收、名单筛选场景。通过软 top-k 选择门
实现全程可微的捕获率优化。
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


class TopKBadCaptureLoss(BaseLoss):
    """Top-K 坏样本捕获损失，优化头部 k% 的坏样本召回率。

    在策略、催收、反欺诈名单筛选场景中，核心目标是在有限资源
    （如 top 5% 的高风险名单）下尽可能多地覆盖坏样本。本损失
    直接面向该业务目标进行优化。

    核心机制:

    1. 计算预测分布的 top-k 分位阈值
    2. 通过 sigmoid 软门控判断每个样本是否 "进入" 头部
    3. 对未进入头部的坏样本（漏捕）施加重惩罚
    4. 对进入头部的好样本（误报）施加轻惩罚

    数学形式::

        τ_k       = percentile(p, 100 - k)
        gate_i    = σ((p_i - τ_k) / temperature)
        miss_cost = (1 - gate_i) × y_i × miss_penalty
        fa_cost   = gate_i × (1-y_i) × fa_penalty
        Loss      = mean(miss_cost + fa_cost) + bce_weight × BCE

    :param top_ratio: 头部样本占比，默认 0.05（top 5%）
        - 内部经验: 策略名单通常 3%~10%；催收通常 5%~20%
    :param miss_penalty: 漏捕坏样本惩罚倍数，默认 5.0
    :param fa_penalty: 误报好样本惩罚倍数，默认 1.0
    :param temperature: sigmoid 平滑温度，默认 0.05
        - 内部经验: 0.02~0.1 之间；太小梯度消失，太大 gate 退化
    :param bce_weight: 基础交叉熵正则权重，默认 0.1
    :param name: 损失函数名称，默认 "topk_bad_capture_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import TopKBadCaptureLoss
    >>>
    >>> # top 5% 名单，重点惩罚漏捕
    >>> loss = TopKBadCaptureLoss(top_ratio=0.05, miss_penalty=10.0)
    >>>
    >>> y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.05, 0.15, 0.25, 0.4])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 LightGBM 中使用
    >>> import lightgbm as lgb
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> bst = lgb.train(
    ...     {'objective': 'binary'},
    ...     train_data,
    ...     fobj=loss.to_lightgbm(),
    ...     num_boost_round=200
    ... )
    """

    def __init__(
        self,
        top_ratio: float = 0.05,
        miss_penalty: float = 5.0,
        fa_penalty: float = 1.0,
        temperature: float = 0.05,
        bce_weight: float = 0.1,
        name: str = "topk_bad_capture_loss",
    ):
        super().__init__(name)
        self.top_ratio = top_ratio
        self.miss_penalty = miss_penalty
        self.fa_penalty = fa_penalty
        self.temperature = temperature
        self.bce_weight = bce_weight

    def _topk_threshold(self, y_pred: np.ndarray) -> float:
        """计算 top-k 分位阈值。"""
        k_percentile = 100 * (1 - self.top_ratio)
        return float(np.percentile(y_pred, k_percentile))

    def _topk_gate(
        self,
        y_pred: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """计算软 top-k 门控值 σ((p - τ) / T)。"""
        z = (y_pred - threshold) / self.temperature
        return _sigmoid(z)

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算 Top-K 捕获损失。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 损失值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        threshold = self._topk_threshold(y_pred)
        gate = self._topk_gate(y_pred, threshold)

        # 漏捕损失: 坏样本没进头部
        miss_cost = (1 - gate) * y_true * self.miss_penalty

        # 误报损失: 好样本进了头部
        fa_cost = gate * (1 - y_true) * self.fa_penalty

        capture_loss = np.mean(miss_cost + fa_cost)

        # BCE 正则
        bce = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        return float(capture_loss + self.bce_weight * bce)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        对坏样本: 推高预测值以进入头部（负梯度）。
        对好样本: 抑制预测值以离开头部（正梯度）。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        threshold = self._topk_threshold(y_pred)
        gate = self._topk_gate(y_pred, threshold)

        # d(gate)/dp = gate(1-gate) / T
        d_gate = gate * (1 - gate) / self.temperature

        # 漏捕梯度: d/dp[(1-gate)*y*penalty] = -d_gate * y * penalty
        grad_miss = -d_gate * y_true * self.miss_penalty

        # 误报梯度: d/dp[gate*(1-y)*penalty] = d_gate * (1-y) * penalty
        grad_fa = d_gate * (1 - y_true) * self.fa_penalty

        # BCE 梯度
        grad_bce = self.bce_weight * (y_pred - y_true)

        return grad_miss + grad_fa + grad_bce

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

        threshold = self._topk_threshold(y_pred)
        gate = self._topk_gate(y_pred, threshold)

        # d²(gate)/dp² = gate(1-gate)(1-2gate) / T²
        d2_gate = gate * (1 - gate) * (1 - 2 * gate) / (self.temperature ** 2)

        # 二阶导
        hess_miss = -d2_gate * y_true * self.miss_penalty
        hess_fa = d2_gate * (1 - y_true) * self.fa_penalty

        # BCE 二阶导
        hess_bce = self.bce_weight * y_pred * (1 - y_pred)

        hess = hess_miss + hess_fa + hess_bce

        return np.maximum(np.abs(hess), 1e-6)
