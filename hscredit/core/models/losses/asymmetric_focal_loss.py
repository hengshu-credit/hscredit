"""
不对称Focal Loss

针对风控极度不平衡数据，分别控制正负样本的聚焦强度。
"""

from __future__ import annotations

import numpy as np

from .base import BaseLoss


class AsymmetricFocalLoss(BaseLoss):
    """不对称 Focal Loss。

    与标准 Focal Loss 不同，该损失允许对正负样本使用不同的聚焦参数，
    从而更灵活地强调坏样本识别或抑制易分类好样本的影响。

    数学形式:
        - 正样本: -alpha * (1 - p)^gamma_pos * log(p)
        - 负样本: -(1 - alpha) * p^gamma_neg * log(1 - p)

    :param alpha: 正样本权重，默认 0.25
    :param gamma_pos: 正样本聚焦参数，默认 2.0
    :param gamma_neg: 负样本聚焦参数，默认 1.0
    :param clip_value: 对负样本概率进行裁剪，抑制极端易分类负样本影响，默认 0.0
    :param name: 损失函数名称，默认 "asymmetric_focal_loss"

    Example:
        >>> import numpy as np
        >>> from hscredit.core.models.losses import AsymmetricFocalLoss
        >>> loss = AsymmetricFocalLoss(alpha=0.7, gamma_pos=2.5, gamma_neg=1.0)
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> round(loss(y_true, y_pred), 6) >= 0
        True
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma_pos: float = 2.0,
        gamma_neg: float = 1.0,
        clip_value: float = 0.0,
        name: str = "asymmetric_focal_loss",
    ):
        super().__init__(name)
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_value = clip_value

    def _clip_probabilities(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
        if self.clip_value > 0:
            y_pred = np.minimum(y_pred + self.clip_value, 1 - 1e-7)
        return y_pred

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = self._clip_probabilities(y_pred)

        pos_loss = -self.alpha * y_true * ((1 - y_pred) ** self.gamma_pos) * np.log(y_pred)
        neg_loss = -(1 - self.alpha) * (1 - y_true) * (y_pred ** self.gamma_neg) * np.log(1 - y_pred)
        return float(np.mean(pos_loss + neg_loss))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = self._clip_probabilities(y_pred)
        grad = np.zeros_like(y_pred, dtype=float)

        pos_mask = y_true == 1
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            grad[pos_mask] = self.alpha * (
                self.gamma_pos * (1 - p) ** (self.gamma_pos - 1) * np.log(p)
                - ((1 - p) ** self.gamma_pos) / p
            )

        neg_mask = y_true == 0
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            grad[neg_mask] = (1 - self.alpha) * (
                -self.gamma_neg * (p ** (self.gamma_neg - 1)) * np.log(1 - p)
                + (p ** self.gamma_neg) / (1 - p)
            )

        return grad

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = self._clip_probabilities(y_pred)
        hess = np.zeros_like(y_pred, dtype=float)

        pos_mask = y_true == 1
        if np.any(pos_mask):
            p = y_pred[pos_mask]
            hess[pos_mask] = self.alpha * (
                self.gamma_pos * (self.gamma_pos - 1) * (1 - p) ** (self.gamma_pos - 2) * np.log(p)
                + 2 * self.gamma_pos * (1 - p) ** (self.gamma_pos - 1) / p
                + (1 - p) ** self.gamma_pos / (p ** 2)
            )

        neg_mask = y_true == 0
        if np.any(neg_mask):
            p = y_pred[neg_mask]
            hess[neg_mask] = (1 - self.alpha) * (
                self.gamma_neg * (self.gamma_neg - 1) * p ** (self.gamma_neg - 2) * np.log(1 - p)
                + 2 * self.gamma_neg * p ** (self.gamma_neg - 1) / (1 - p)
                + p ** self.gamma_neg / ((1 - p) ** 2)
            )

        return np.abs(hess) + 1e-6
