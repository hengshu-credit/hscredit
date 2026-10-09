"""单变量筛选器的参数校验与轻量统计元数据。"""

from numbers import Real

import numpy as np
import pandas as pd


def validate_real(value, name, minimum=None, maximum=None, allow_infinite=False):
    """拒绝布尔值、NaN 和范围外实数；仅显式兼容入口允许无穷阈值。"""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or np.isnan(value)
        or (not allow_infinite and not np.isfinite(value))
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        bounds = f"，范围为 [{minimum}, {maximum}]" if minimum is not None or maximum is not None else ""
        raise ValueError(f"{name} 必须是有效{'实数' if allow_infinite else '有限实数'}{bounds}")


def validate_binary_target(y, name):
    """检查有监督风险指标使用的一维无缺失 0/1 目标。"""
    if y is None:
        raise ValueError(f"{name} 需要目标变量 y")
    values = np.asarray(y)
    if values.ndim != 1 or pd.isna(values).any() or not pd.Series(values).isin([0, 1]).all():
        raise ValueError(f"{name} 要求目标变量为无缺失的一维 0/1 标签")
    return values.astype(int)


def is_categorical(series):
    """统一识别 object、string、category、bool 字段。"""
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def record_counts(selector, X):
    """保存与实际指标输入一致的逐列样本量，不持有原始样本。"""
    selector.total_counts_ = pd.Series(len(X), index=X.columns, dtype=np.int64)
    selector.missing_counts_ = X.isna().sum().astype(np.int64)
    selector.valid_counts_ = selector.total_counts_ - selector.missing_counts_


def record_conditions(selector, index, **conditions):
    """保存每个实际筛选条件，报告可以区别阈值与排名限制。"""
    selector.condition_results_ = pd.DataFrame(conditions, index=index, dtype=bool)


def rank_scores(scores):
    """分数相同时按输入顺序稳定排名，无效值排最后。"""
    return scores.rank(method="first", ascending=False, na_option="bottom").astype(int)


def validate_k(k):
    """校验保留数量，布尔值不作为整数处理。"""
    if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) and k > 0:
        return
    if not isinstance(k, str) or k != "all":
        raise ValueError("k 必须是大于 0 的整数或 'all'")


def top_k_mask(scores, k):
    """复用得分的稳定 top-k 掩码。"""
    if k == "all":
        return np.ones(len(scores), dtype=bool)
    return rank_scores(pd.Series(scores)).to_numpy() <= int(k)
