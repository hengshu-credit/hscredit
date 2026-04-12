"""
平衡 Focal Loss

在标准 Focal Loss 上加入基于有效样本数的类别平衡方案与标签平滑，
提供比固定 alpha 更稳定的不平衡处理能力。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss


class BalancedFocalLoss(BaseLoss):
    """平衡 Focal Loss，在 FocalLoss 上加入更稳定的类别平衡方案。

    核心改进（相比标准 FocalLoss）:

    1. **有效样本数加权**: 基于 *Class-Balanced Loss* 论文，用
       ``E_n = (1 - β^n) / (1 - β)`` 自动计算正负样本权重，
       比手动设置 alpha 更鲁棒。
    2. **标签平滑**: 将硬标签 {0,1} 软化为 {ε/2, 1-ε/2}，
       减少过拟合并提升梯度稳定性。

    数学形式::

        y_smooth = y × (1-ε) + ε/2
        p_t      = y_smooth × p + (1-y_smooth) × (1-p)
        w_t      = (1-β) / (1-β^n_t)      # 有效样本数的倒数
        Loss     = -w_t × (1-p_t)^γ × log(p_t)

    :param gamma: 聚焦参数，默认 2.0
        - gamma=0 退化为加权交叉熵
        - gamma 越大，易分类样本权重衰减越快
    :param beta: 有效样本数衰减因子，默认 0.999
        - 内部经验: 0.99~0.9999 之间；样本量越大建议 beta 越接近 1
    :param label_smoothing: 标签平滑系数，默认 0.0（不启用）
        - 内部经验: 0.01~0.1 之间可改善校准; 过大会降低区分度
    :param auto_alpha: 是否自动根据有效样本数计算 alpha，默认 True
        - 当 auto_alpha=False 时回退为固定 alpha
    :param alpha: 固定正样本权重，仅在 auto_alpha=False 时生效，默认 0.25
    :param name: 损失函数名称，默认 "balanced_focal_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import BalancedFocalLoss
    >>>
    >>> # 自动平衡（推荐）
    >>> loss = BalancedFocalLoss(gamma=2.0, beta=0.999, label_smoothing=0.05)
    >>>
    >>> y_true = np.array([0, 0, 0, 0, 1])  # 极不平衡
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.8])
    >>> loss_value = loss(y_true, y_pred)
    >>>
    >>> # 在 XGBoost 中使用
    >>> import xgboost as xgb
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> bst = xgb.train({}, dtrain, obj=loss.to_xgboost(), num_boost_round=100)

    **参考**

    Cui, Y., et al. "Class-Balanced Loss Based on Effective Number of
    Samples." CVPR 2019.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.999,
        label_smoothing: float = 0.0,
        auto_alpha: bool = True,
        alpha: float = 0.25,
        name: str = "balanced_focal_loss",
    ):
        super().__init__(name)
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.auto_alpha = auto_alpha
        self.alpha = alpha

    def _compute_alpha(self, y_true: np.ndarray) -> float:
        """根据有效样本数自动计算 alpha。"""
        if not self.auto_alpha:
            return self.alpha

        n_pos = max(int(np.sum(y_true == 1)), 1)
        n_neg = max(int(np.sum(y_true == 0)), 1)

        # 有效样本数
        e_pos = (1 - self.beta ** n_pos) / (1 - self.beta + 1e-12)
        e_neg = (1 - self.beta ** n_neg) / (1 - self.beta + 1e-12)

        # alpha 为负样本有效数占比（给少数类更大权重）
        alpha = e_neg / (e_pos + e_neg + 1e-12)
        return float(alpha)

    def _smooth_labels(self, y_true: np.ndarray) -> np.ndarray:
        """标签平滑。"""
        if self.label_smoothing <= 0:
            return y_true
        return y_true * (1 - self.label_smoothing) + self.label_smoothing / 2

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算平衡 Focal Loss。

        :param y_true: 真实标签, shape (n_samples,)
        :param y_pred: 预测概率, shape (n_samples,)
        :return: 平均损失值
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        alpha = self._compute_alpha(y_true)
        y_smooth = self._smooth_labels(y_true)

        # p_t 和 alpha_t
        p_t = np.where(y_smooth >= 0.5, y_pred, 1 - y_pred)
        alpha_t = np.where(y_smooth >= 0.5, alpha, 1 - alpha)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # 交叉熵
        ce = -np.log(np.clip(p_t, 1e-7, 1.0))

        loss = alpha_t * focal_weight * ce
        return float(np.mean(loss))

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """计算梯度。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: 梯度数组, shape (n_samples,)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        alpha = self._compute_alpha(y_true)
        y_smooth = self._smooth_labels(y_true)

        grad = np.zeros_like(y_pred, dtype=float)

        # 正样本 (y_smooth >= 0.5)
        pos_mask = y_smooth >= 0.5
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            a = alpha
            g = self.gamma
            grad[pos_mask] = a * (
                g * (1 - p) ** (g - 1) * np.log(np.clip(p, 1e-7, 1.0))
                - (1 - p) ** g / p
            )

        # 负样本 (y_smooth < 0.5)
        neg_mask = ~pos_mask
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            a = 1 - alpha
            g = self.gamma
            grad[neg_mask] = a * (
                -g * p ** (g - 1) * np.log(np.clip(1 - p, 1e-7, 1.0))
                + p ** g / (1 - p)
            )

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

        alpha = self._compute_alpha(y_true)
        y_smooth = self._smooth_labels(y_true)

        hess = np.zeros_like(y_pred, dtype=float)
        g = self.gamma

        pos_mask = y_smooth >= 0.5
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            a = alpha
            hess[pos_mask] = a * (
                g * (g - 1) * (1 - p) ** (g - 2) * np.log(np.clip(p, 1e-7, 1.0))
                + 2 * g * (1 - p) ** (g - 1) / p
                + (1 - p) ** g / (p ** 2)
            )

        neg_mask = ~pos_mask
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            a = 1 - alpha
            hess[neg_mask] = a * (
                g * (g - 1) * p ** (g - 2) * np.log(np.clip(1 - p, 1e-7, 1.0))
                + 2 * g * p ** (g - 1) / (1 - p)
                + p ** g / ((1 - p) ** 2)
            )

        # 确保正定
        return np.abs(hess) + 1e-6
