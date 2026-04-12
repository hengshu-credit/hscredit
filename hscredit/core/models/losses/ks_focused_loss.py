"""
KS 导向损失函数

强化正负样本分布分离能力，直接面向风控核心评估指标 KS 值优化。
通过 Fisher 判别思想的可微代理，在交叉熵基础上最大化正负样本预测
分布的间距，并在分布重叠区域施加更大权重。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss


class KSFocusedLoss(BaseLoss):
    """KS 导向损失函数，强化好坏样本预测分布的分离能力。

    KS（Kolmogorov-Smirnov）统计量衡量正负样本累积分布的最大距离，
    是信贷风控模型评估的核心指标之一。本损失在标准交叉熵基础上增加
    分布分离惩罚项，通过以下机制逼近 KS 优化：

    1. **均值分离**: 最大化 mean(p|y=1) - mean(p|y=0)
    2. **分布聚集**: 减小类内预测方差，使分布更集中
    3. **重叠区聚焦**: 在正负分布重叠区域施加更大权重

    数学形式::

        L = bce_weight × BCE
          - ks_weight × [μ₁ - μ₀]
          + var_weight × [σ₁² + σ₀²]

        梯度在重叠区通过 focus_weight 放大。

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
    ...     {'objective': 'binary'},
    ...     train_data,
    ...     fobj=loss.to_lightgbm(),
    ...     num_boost_round=200
    ... )
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
        bw_sq = self.bandwidth ** 2 + 1e-12
        proximity = np.exp(-((y_pred - midpoint) ** 2) / (2 * bw_sq))
        return 1.0 + self.focus_weight * proximity

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算 KS 导向损失。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 损失值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        stats = self._distribution_stats(y_true, y_pred)

        # 基础 BCE
        bce = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        # 分布分离项（负号表示最大化间距）
        separation = -(stats["mu_pos"] - stats["mu_neg"])

        # 类内方差惩罚
        variance_penalty = stats["var_pos"] + stats["var_neg"]

        return float(
            self.bce_weight * bce
            + self.ks_weight * separation
            + self.var_weight * variance_penalty
        )

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        对于正样本: 推高预测值（增大 μ₁），在重叠区加权。
        对于负样本: 压低预测值（减小 μ₀），在重叠区加权。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        stats = self._distribution_stats(y_true, y_pred)
        overlap_w = self._overlap_weight(y_pred, stats["midpoint"])

        # BCE 梯度
        grad = self.bce_weight * (y_pred - y_true)

        # 分离梯度: 正样本往上推, 负样本往下压
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        ks_grad = np.zeros_like(y_pred)
        ks_grad[pos_mask] = -self.ks_weight / stats["n_pos"]
        ks_grad[neg_mask] = self.ks_weight / stats["n_neg"]

        # 方差梯度: 向类均值靠拢
        var_grad = np.zeros_like(y_pred)
        if stats["n_pos"] > 1:
            var_grad[pos_mask] = (
                2 * self.var_weight
                * (y_pred[pos_mask] - stats["mu_pos"])
                / stats["n_pos"]
            )
        if stats["n_neg"] > 1:
            var_grad[neg_mask] = (
                2 * self.var_weight
                * (y_pred[neg_mask] - stats["mu_neg"])
                / stats["n_neg"]
            )

        # 重叠区聚焦加权（仅对 KS 分离梯度）
        grad += overlap_w * ks_grad + var_grad

        return grad

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

        # 使用 BCE 二阶导作为基础近似
        hess = self.bce_weight * y_pred * (1 - y_pred)

        return np.maximum(hess, 1e-6)
