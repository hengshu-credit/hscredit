"""
排序与头部效果导向损失函数

针对评分排序一致性与头部LIFT优化场景设计的损失函数。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseLoss
from ._loss_math import binary_inputs, bce_terms, nonnegative, positive, unit_interval


class OrdinalRankLoss(BaseLoss):
    """序数排序损失，兼顾概率拟合与好坏样本排序一致性。

    该损失在标准二元交叉熵基础上增加成对排序惩罚项，
    鼓励坏样本（label=1）的预测风险高于好样本（label=0）。
    导数按 n × 平均损失计算，返回 Hessian 的对角项；框架不使用跨样本二阶项。

    :param rank_weight: 排序惩罚项权重，默认 1.0
    :param bce_weight: 交叉熵权重，默认 1.0
    :param temperature: 排序平滑温度，越小越强调排序间隔，默认 1.0
    :param max_pairs: 为控制计算开销，最多采样的正负样本对数，默认 20000
    :param random_state: 随机种子，保证采样对可复现，默认 42
    :param name: 损失函数名称，默认 "ordinal_rank_loss"

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.models.losses import OrdinalRankLoss
    >>> loss = OrdinalRankLoss(rank_weight=2.0, bce_weight=1.0)
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> round(loss(y_true, y_pred), 6) >= 0
    True

    **引用**

    成对排序（pairwise ranking）优化与 AUC 的等价性见 Burges, C. et al. (2005). *Learning
    to Rank using Gradient Descent (RankNet).* ICML 2005，
    https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/ ；
    AUC 与 Wilcoxon–Mann–Whitney 统计量的关系见 Hanley & McNeil (1982)。
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
        nonnegative(rank_weight=rank_weight, bce_weight=bce_weight)
        positive(temperature=temperature, max_pairs=max_pairs)
        if not isinstance(max_pairs, (int, np.integer)):
            raise ValueError("max_pairs 必须为正整数。")
        self.rank_weight = rank_weight
        self.bce_weight = bce_weight
        self.temperature = temperature
        self.max_pairs = max_pairs
        self.random_state = random_state

    def _prepare_pairs(self, y_true):
        """直接采样组合编号，避免在 max_pairs 截断前构造平方级数组。"""
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)
        total = len(pos_idx) * len(neg_idx)
        if not total:
            return np.array([], dtype=int), np.array([], dtype=int)
        if total <= self.max_pairs:
            indices = np.arange(total)
        else:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(total, size=self.max_pairs, replace=False)
        return pos_idx[indices // len(neg_idx)], neg_idx[indices % len(neg_idx)]

    def _pairwise_rank_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) == 0:
            return 0.0

        diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
        return float(np.mean(np.logaddexp(0.0, -diff)))

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import OrdinalRankLoss
        >>> loss = OrdinalRankLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)

        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        rank = self._pairwise_rank_loss(y_true, y_pred)
        return float(self.bce_weight * bce + self.rank_weight * rank)

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import OrdinalRankLoss
        >>> loss = OrdinalRankLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)

        grad = self.bce_weight * bce_terms(y_true, y_pred)[1]

        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) == 0 or self.rank_weight == 0:
            return grad

        diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
        pair_grad = -np.exp(-np.logaddexp(0.0, diff))
        pair_grad = (len(y_true) * self.rank_weight / len(pos_pairs)) * (pair_grad / self.temperature)

        np.add.at(grad, pos_pairs, pair_grad)
        np.add.at(grad, neg_pairs, -pair_grad)
        return grad

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import OrdinalRankLoss
        >>> loss = OrdinalRankLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)

        hess = self.bce_weight * bce_terms(y_true, y_pred)[2]

        pos_pairs, neg_pairs = self._prepare_pairs(y_true)
        if len(pos_pairs) > 0 and self.rank_weight != 0:
            diff = (y_pred[pos_pairs] - y_pred[neg_pairs]) / self.temperature
            sig = np.exp(-np.logaddexp(0.0, -diff))
            pair_hess = sig * (1 - sig)
            pair_hess = (len(y_true) * self.rank_weight / len(pos_pairs)) * (pair_hess / (self.temperature**2))
            np.add.at(hess, pos_pairs, pair_hess)
            np.add.at(hess, neg_pairs, pair_hess)

        return hess


class LiftFocusedLoss(BaseLoss):
    """头部 LIFT 导向损失，对高风险区间样本错误施加更大惩罚。

    该损失基于加权二元交叉熵，按照预测风险从高到低分配更大的样本权重，
    并在头部区间进一步放大坏样本的惩罚，提升模型在高风险头部样本上的区分能力。
    排名权重在每次求导时固定，样本排名切换或同分边界处不可微。

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
        unit_interval(top_ratio=top_ratio)
        positive(
            top_ratio=top_ratio,
            penalty_factor=penalty_factor,
            positive_class_boost=positive_class_boost,
            base_weight=base_weight,
        )
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
        """计算本损失的平均值，越小越好。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: float；原始损失值，不根据调参方向改变符号。

        **参考样例**

        >>> from hscredit.core.models.losses import LiftFocusedLoss
        >>> loss = LiftFocusedLoss()
        >>> result = loss([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)
        weights = self._get_sample_weights(y_true, y_pred)

        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.average(loss, weights=weights))

    def gradient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """返回损失相对坏样本概率的一阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的一阶导数组，标度为样本数乘以平均损失；详见 BaseLoss.gradient。

        **参考样例**

        >>> from hscredit.core.models.losses import LiftFocusedLoss
        >>> loss = LiftFocusedLoss()
        >>> result = loss.gradient([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)
        weights = self._get_sample_weights(y_true, y_pred)
        return (weights / np.mean(weights)) * bce_terms(y_true, y_pred)[1]

    def hessian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """返回损失相对坏样本概率的二阶导数。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；不是原始分数。
        :return: 与输入等长的二阶导数组，标度同 BaseLoss.hessian；框架适配器再做链接函数转换。

        **参考样例**

        >>> from hscredit.core.models.losses import LiftFocusedLoss
        >>> loss = LiftFocusedLoss()
        >>> result = loss.hessian([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred = binary_inputs(y_true, y_pred)
        weights = self._get_sample_weights(y_true, y_pred)
        hess = (weights / np.mean(weights)) * bce_terms(y_true, y_pred)[2]
        return hess
