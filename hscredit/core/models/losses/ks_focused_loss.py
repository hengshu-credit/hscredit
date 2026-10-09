"""
KS 导向损失函数

强化正负样本分布分离能力，直接面向风控核心评估指标 KS 值优化。
通过 Fisher 判别思想的可微代理，在交叉熵基础上最大化正负样本预测
分布的间距，并在分布重叠区域施加更大权重。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, nonnegative, positive, unit_interval


class KSFocusedLoss(BaseLoss):
    """KS 导向损失函数，强化好坏样本预测分布的分离能力。

    KS（Kolmogorov-Smirnov）统计量衡量正负样本累积分布的最大距离，
    是信贷风控模型评估的核心指标之一。本损失在标准交叉熵基础上增加
    分布分离惩罚项，通过以下机制逼近 KS 优化：

    1. **均值分离**: 最大化 mean(p|y=1) - mean(p|y=0)
    2. **分布聚集**: 减小类内预测方差，使分布更集中
    3. **重叠区聚焦**: 在正负分布重叠区域施加更大权重

    数学形式::

        overlap_i = exp(-(p_i-midpoint)^2 / (2*bandwidth^2))
        L = bce_weight × mean((1 + focus_weight × overlap_i) × BCE_i)
          - ks_weight × [μ₁ - μ₀]
          + var_weight × [σ₁² + σ₀²]

        midpoint = (μ₁ + μ₀) / 2；导数包含该均值对所有概率的依赖。
        导数按 n × 平均损失计算；训练框架仅使用 Hessian 的对角项。

    :param ks_weight: 分布分离项权重，默认 1.0
        - 内部经验: 0.5~2.0 之间，越大分布分离越强但可能牺牲概率校准
    :param bce_weight: 基础交叉熵权重，默认 1.0
    :param var_weight: 类内方差惩罚权重，默认 0.1
        - 内部经验: 适度值（0.05~0.3）可减少类内离散度，提升 KS
    :param focus_weight: 重叠区聚焦倍数，默认 2.0
        - 对预测值处于正负分布重叠区的样本额外加权
    :param bandwidth: 重叠区高斯核带宽，默认 0.1
    :param name: 损失函数名称，默认 "ks_focused_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import KSFocusedLoss
    >>>
    >>> loss = KSFocusedLoss(ks_weight=1.5, var_weight=0.2)
    >>> y_true = np.array([0, 0, 0, 1, 1, 1])
    >>> y_pred = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 LightGBM 中使用
    >>> import lightgbm as lgb
    >>> train_data = lgb.Dataset(X_train, label=y_train)
    >>> bst = lgb.train(
    ...     {'objective': loss.to_lightgbm(api='native'), 'metric': 'None'},
    ...     train_data,
    ...     feval=loss.metric().to_lightgbm(api='native', raw_score=True),
    ...     num_boost_round=200
    ... )

    **引用**

    KS（Kolmogorov–Smirnov）统计量见 https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test ；
    其在信用风险中作为核心区分度指标见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley。
    本损失为面向 KS 的可微代理（分布分离 + 重叠区聚焦），属业务驱动设计。
    """

    def __init__(
        self,
        ks_weight: float = 1.0,
        bce_weight: float = 1.0,
        var_weight: float = 0.1,
        focus_weight: float = 2.0,
        bandwidth: float = 0.1,
        name: str = "ks_focused_loss",
    ):
        super().__init__(name)
        nonnegative(ks_weight=ks_weight, bce_weight=bce_weight, var_weight=var_weight, focus_weight=focus_weight)
        positive(bandwidth=bandwidth)
        self.ks_weight = ks_weight
        self.bce_weight = bce_weight
        self.var_weight = var_weight
        self.focus_weight = focus_weight
        self.bandwidth = bandwidth

    def _distribution_stats(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """计算正负样本分布统计量。"""
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        p_pos = y_pred[pos_mask]
        p_neg = y_pred[neg_mask]

        n_pos = max(len(p_pos), 1)
        n_neg = max(len(p_neg), 1)

        mu_pos = np.mean(p_pos) if len(p_pos) > 0 else 0.5
        mu_neg = np.mean(p_neg) if len(p_neg) > 0 else 0.5

        var_pos = np.var(p_pos) if len(p_pos) > 1 else 0.0
        var_neg = np.var(p_neg) if len(p_neg) > 1 else 0.0

        midpoint = (mu_pos + mu_neg) / 2.0

        return {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "var_pos": var_pos,
            "var_neg": var_neg,
            "midpoint": midpoint,
        }

    def _overlap_weight(
        self,
        y_pred: np.ndarray,
        midpoint: float,
    ) -> np.ndarray:
        """计算重叠区聚焦权重（高斯核）。"""
        bw_sq = self.bandwidth**2 + 1e-12
        proximity = np.exp(-((y_pred - midpoint) ** 2) / (2 * bw_sq))
        return 1.0 + self.focus_weight * proximity

    def __call__(self, y_true, y_pred) -> float:
        """高斯重叠区加权 BCE + 均值分离 + 类内方差代理，不等于真实 KS。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import KSFocusedLoss
        >>> loss = KSFocusedLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[0]

    def gradient(self, y_true, y_pred):
        """包含数据依赖 midpoint 的完整链式导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import KSFocusedLoss
        >>> loss = KSFocusedLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[1]

    def hessian(self, y_true, y_pred):
        """返回真实概率 Hessian 对角项，非凸区可为负；不包含跨样本项。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import KSFocusedLoss
        >>> loss = KSFocusedLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        return self._terms(y_true, y_pred)[2]

    def _terms(self, y_true, y_pred):
        """返回标量及 n 倍均值的精确一阶导、Hessian 对角项。"""
        y, p = binary_inputs(y_true, y_pred)
        n = len(y)
        stats = self._distribution_stats(y, p)
        bce, bce_grad, bce_hess = bce_terms(y, p)
        pos = y == 1
        neg = ~pos
        count = np.where(pos, stats["n_pos"], stats["n_neg"])
        # midpoint 对每个概率的偏导，单类时缺失类的均值保持为 0.5。
        center_grad = 0.5 / count
        offset = p - stats["midpoint"]
        bandwidth_sq = self.bandwidth**2
        kernel = np.exp(-(offset**2) / (2 * bandwidth_sq))
        kernel_grad = -offset * kernel / bandwidth_sq
        kernel_hess = (offset**2 / bandwidth_sq**2 - 1 / bandwidth_sq) * kernel
        focus = self.focus_weight
        bce_value = np.mean((1 + focus * kernel) * bce)
        focus_grad = kernel * bce_grad + kernel_grad * bce - center_grad * np.sum(kernel_grad * bce)
        focus_hess = (
            kernel * bce_hess
            + 2 * (1 - center_grad) * kernel_grad * bce_grad
            + (1 - 2 * center_grad) * kernel_hess * bce
            + center_grad**2 * np.sum(kernel_hess * bce)
        )
        grad = self.bce_weight * (bce_grad + focus * focus_grad)
        hess = self.bce_weight * (bce_hess + focus * focus_hess)
        separation = stats["mu_neg"] - stats["mu_pos"]
        grad += n * self.ks_weight * np.where(pos, -1.0 / count, 1.0 / count)
        means = np.where(pos, stats["mu_pos"], stats["mu_neg"])
        grad += n * self.var_weight * 2 * (p - means) / count
        hess += n * self.var_weight * 2 * (1 - 1 / count) / count
        value = (
            self.bce_weight * bce_value
            + self.ks_weight * separation
            + self.var_weight * (stats["var_pos"] + stats["var_neg"])
        )
        return float(value), grad, hess
