"""统一二分类排序指标与曲线的样本、分数方向和同分阈值口径。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import auc as curve_area, roc_curve


def _prepare_binary_scores(y_true, y_score, pos_label=1, allow_single_class=False, sample_weight=None):
    """按位置配对标签和分数，删除成对缺失样本，并明确正样本标签。"""
    target = np.asarray(y_true)
    score = np.asarray(y_score, dtype=float)
    if target.ndim != 1 or score.ndim != 1:
        raise ValueError("标签和分数必须是一维数组")
    if len(target) != len(score):
        raise ValueError(f"标签和分数的长度必须一致: {len(target)} != {len(score)}")
    valid = ~pd.isna(target) & ~pd.isna(score)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or len(weights) != len(target):
            raise ValueError("sample_weight必须是一维且与样本等长")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight必须是有限非负数")
        valid &= weights > 0
        weights = weights[valid]
    target, score = target[valid], score[valid]
    if len(target) == 0:
        raise ValueError("标签和分数没有可用的非缺失数据")
    if not np.isfinite(score).all():
        raise ValueError("分数必须是有限数值，不能包含正负无穷")
    labels = np.unique(target)
    single_binary_class = allow_single_class and len(labels) == 1 and labels[0] in (0, 1) and pos_label in (0, 1)
    if len(labels) != 2 and not single_binary_class:
        raise ValueError(f"标签必须是二分类标签（包含2个唯一值），当前有 {len(labels)} 个唯一值")
    if pos_label not in labels and not single_binary_class:
        raise ValueError(f"pos_label={pos_label!r} 不在标签 {labels.tolist()} 中")
    return (target == pos_label).astype(int), score, weights


def _binary_ranking_curve(target, score, sample_weight=None):
    """在每个不同分数的阈值处累计，返回覆盖率、好样本累计率和坏样本累计率。"""
    fpr, tpr, _ = roc_curve(target, score, sample_weight=sample_weight, drop_intermediate=False)
    positive_count = np.count_nonzero(target) if sample_weight is None else sample_weight[target == 1].sum()
    total = len(target) if sample_weight is None else sample_weight.sum()
    coverage = (tpr * positive_count + fpr * (total - positive_count)) / total
    return coverage, fpr, tpr


@dataclass(frozen=True)
class _BinaryRocResult:
    """同一次计算产生的 ROC 曲线、AUC 和累计样本覆盖率。"""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    coverage: np.ndarray
    auc: float
    auto_reversed: bool


def _binary_roc_statistics(y_true, y_score, pos_label=1, score_direction='auto', sample_weight=None):
    """统一计算 ROC 与曲线面积，方向、样本过滤、权重及同分处理只在这里实现。"""
    direction = str(score_direction).strip().lower()
    if direction not in {'auto', 'higher_risk', 'higher_safe'}:
        raise ValueError("score_direction 必须是 'auto'、'higher_risk' 或 'higher_safe'")
    target, score, weights = _prepare_binary_scores(y_true, y_score, pos_label, sample_weight=sample_weight)
    fpr, tpr, thresholds = roc_curve(target, score, sample_weight=weights, drop_intermediate=False)
    value = float(curve_area(fpr, tpr))
    auto_reversed = direction == 'auto' and value < 0.5
    if direction == 'higher_safe' or auto_reversed:
        fpr, tpr, thresholds = roc_curve(target, -score, sample_weight=weights, drop_intermediate=False)
        value = float(curve_area(fpr, tpr))
    if weights is None:
        positive_count = np.count_nonzero(target)
        total = len(target)
    else:
        positive_count = weights[target == 1].sum()
        total = weights.sum()
    coverage = (tpr * positive_count + fpr * (total - positive_count)) / total
    return _BinaryRocResult(fpr, tpr, thresholds, coverage, value, auto_reversed)
