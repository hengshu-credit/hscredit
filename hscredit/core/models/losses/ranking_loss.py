"""
排序与头部效果导向损失函数

针对评分排序一致性与头部LIFT优化场景设计的损失函数。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseLoss


class OrdinalRankLoss(BaseLoss):
    """序数排序损失，兼顾概率拟合与好坏样本排序一致性。

    该损失在标准二元交叉熵基础上增加成对排序惩罚项，
    鼓励坏样本（label=1）的预测风险高于好样本（label=0）。

    :param rank_weight: 排序惩罚项权重，默认 1.0
    :param bce_weight: 交叉熵权重，默认 1.0
    :param temperature: 排序平滑温度，越小越强调排序间隔，默认 1.0
    :param max_pairs: 为控制计算开销，最多采样的正负样本对数，默认 20000
    :param random_state: 随机种子，保证采样对可复现，默认 42
    :param name: 损失函数名称，默认 "ordinal_rank_loss"

    Example:
        >>> import numpy as np
        >>> from hscredit.core.models.losses import OrdinalRankLoss
        >>> loss = OrdinalRankLoss(rank_weight=2.0, bce_weight=1.0)
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        >>> round(loss(y_true, y_pred), 6) >= 0
        True
    """

    def __init__(
        self,
        rank_weight: float = 1.0,
        bce_weight: float = 1.0,
        temperature: float = 1.0,
        max_pairs: int = 20000,
        random_state: int = 42,
        name: str = "ordinal_rank_loss",
    ):
        super().__init__(name)
        self.rank_weight = rank_weight
        self.bce_weight = bce_weight
        self.temperature = temperature
        self.max_pairs = max_pairs
        self.random_state = random_state

    def _prepare_pairs(
        self,
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        pos_grid = np.repeat(pos_idx, len(neg_idx))
        neg_grid = np.tile(neg_idx, len(pos_idx))

        if len(pos_grid) <= self.max_pairs:
            return pos_grid, neg_grid

        rng = np.random.default_rng(self.random_state)
        chosen = rng.choice(len(pos_grid), size=self.max_pairs, replace=False)
        return pos_grid[chosen], neg_grid[chosen]

    def _pairwise_rank_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) == 0:
            return 0.0

        diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
        return float(np.mean(np.log1p(np.exp(-diff))))

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        rank = self._pairwise_rank_loss(y_true, y_pred)
        return float(self.bce_weight * bce + self.rank_weight * rank)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        grad = self.bce_weight * (y_pred - y_true)

        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) == 0 or self.rank_weight == 0:
            return grad

        diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
        pair_grad = -1.0 / (1.0 + np.exp(diff))
        pair_grad = (self.rank_weight / len(pos_pairs)) * (pair_grad / self.temperature)

        np.add.at(grad, pos_pairs, pair_grad)
        np.add.at(grad, neg_pairs, -pair_grad)
        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

        hess = self.bce_weight * y_pred * (1 - y_pred)

        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) > 0 and self.rank_weight != 0:
            diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
            sig = 1.0 / (1.0 + np.exp(-diff))
            pair_hess = sig * (1 - sig)
            pair_hess = (self.rank_weight / len(pos_pairs)) * (pair_hess / (self.temperature ** 2))
            np.add.at(hess, pos_pairs, pair_hess)
            np.add.at(hess, neg_pairs, pair_hess)

        return np.maximum(hess, 1e-6)


class LiftFocusedLoss(BaseLoss):
    """头部 LIFT 导向损失，对高风险区间样本错误施加更大惩罚。

    该损失基于加权二元交叉熵，按照预测风险从高到低分配更大的样本权重，
    并在头部区间进一步放大坏样本的惩罚，提升模型在高风险头部样本上的区分能力。

    :param top_ratio: 头部样本占比，默认 0.10
    :param penalty_factor: 头部惩罚倍数，默认 3.0
    :param positive_class_boost: 头部坏样本额外增益倍数，默认 1.5
    :param base_weight: 非头部样本基础权重，默认 1.0
    :param name: 损失函数名称，默认 "lift_focused_loss"

    Example:
        >>> import numpy as np
        >>> from hscredit.core.models.losses import LiftFocusedLoss
        >>> loss = LiftFocusedLoss(top_ratio=0.2, penalty_factor=4.0)
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.7, 0.9])
        >>> round(loss(y_true, y_pred), 6) >= 0
        True
    """

    def __init__(
        self,
        top_ratio: float = 0.10,
        penalty_factor: float = 3.0,
        positive_class_boost: float = 1.5,
        base_weight: float = 1.0,
        name: str = "lift_focused_loss",
    ):
        super().__init__(name)
        self.top_ratio = top_ratio
        self.penalty_factor = penalty_factor
        self.positive_class_boost = positive_class_boost
        self.base_weight = base_weight

    def _get_sample_weights(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        n_samples = len(y_pred)
        if n_samples == 0:
            return np.array([], dtype=float)

        order = np.argsort(-y_pred)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_samples)

        head_count = max(1, int(np.ceil(n_samples * self.top_ratio)))
        top_mask = ranks < head_count

        weights = np.full(n_samples, self.base_weight, dtype=float)
        weights[top_mask] = self.base_weight * self.penalty_factor

        positive_top_mask = top_mask & (y_true == 1)
        weights[positive_top_mask] *= self.positive_class_boost
        return weights

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
        weights = self._get_sample_weights(y_true, y_pred)

        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.average(loss, weights=weights))

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
        weights = self._get_sample_weights(y_true, y_pred)
        return weights * (y_pred - y_true)

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
        weights = self._get_sample_weights(y_true, y_pred)
        hess = weights * y_pred * (1 - y_pred)
        return np.maximum(hess, 1e-6)
