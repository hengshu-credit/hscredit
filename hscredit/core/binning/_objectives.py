"""分箱目标的名称解析、参数绑定与自定义回调统计契约。"""

from functools import partial
from inspect import signature
from numbers import Real
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..metrics._binning import compute_bin_stats, _binary_stats_from_counts, woe_iv_vectorized

Objective = Union[str, Callable]
NATIVE_OBJECTIVES = ("iv", "ks", "gini", "entropy", "chi2")
COMPOSITE_OBJECTIVES = (
    "max_lift_iv",
    "min_lift_iv",
    "max_min_lift_sum_iv",
    "max_lift_distance_iv",
    "min_lift_distance_iv",
    "lift_distance_sum_iv",
    "max_ks_iv",
    "woe_variance_iv",
)
OBJECTIVE_METHODS = ("or_tools", "cp_sat", "genetic")


def pickle_callback_state(state):
    """仅在序列化状态中包装回调，加载后恢复原对象，不改构造参数或公开类型。"""
    from joblib import wrap_non_picklable_objects

    result = dict(state)
    parameter_dicts = {"objective_params", "kwargs", "x_params", "y_params"}
    for name, value in state.items():
        if callable(value) or (name in parameter_dicts and isinstance(value, dict) and value):
            result[name] = wrap_non_picklable_objects(value, keep_wrapper=False)
    return result


def resolve_objective(
    objective: Objective,
    objective_params: Optional[Dict] = None,
    custom_objective: Optional[Callable] = None,
    *,
    native_names=NATIVE_OBJECTIVES,
) -> Tuple[str, Optional[Callable]]:
    """保留构造参数原对象，独立解析名称或绑定 callable 的附加参数。"""
    if objective_params is not None and not isinstance(objective_params, dict):
        raise ValueError("objective_params 必须是参数字典或 None")
    params = dict(objective_params or {})
    if any(not isinstance(key, str) for key in params):
        raise ValueError("objective_params 的键必须是字符串")
    if custom_objective is not None and objective != "custom":
        raise ValueError("custom_objective 仅与 objective='custom' 配合；也可以直接传 objective=函数")
    if callable(objective):
        name, callback = getattr(objective, "__name__", type(objective).__name__), objective
    elif isinstance(objective, str) and objective in native_names:
        if params:
            raise ValueError(f"内置目标 '{objective}' 不接收 objective_params")
        return objective, None
    elif isinstance(objective, str) and objective == "custom":
        if not callable(custom_objective):
            raise ValueError("当 objective='custom' 时，必须提供 custom_objective 参数且为可调用对象")
        name, callback = "custom", custom_objective
    elif isinstance(objective, str) and objective in COMPOSITE_OBJECTIVES:
        from .or_binning import CustomObjectives

        factory = getattr(CustomObjectives, objective)
        try:
            signature(factory).bind(**params)
        except TypeError as exc:
            raise ValueError(f"目标 '{objective}' 的 objective_params 无效: {exc}") from exc
        return objective, factory(**params)
    else:
        raise ValueError(
            f"不支持的 objective: {objective!r}；可选 {list(native_names + COMPOSITE_OBJECTIVES)} 或自定义函数"
        )
    try:
        parameters = signature(callback)
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None:
        try:
            parameters.bind([], 0, 0, **params)
        except TypeError as exc:
            raise ValueError(
                "自定义 objective 必须接受 (bin_stats, total_good, total_bad) 及 objective_params"
            ) from exc
    return name, partial(callback, **params) if params else callback


def _callback_rows(table):
    """中文统计列作为标准字段，同时保留既有回调的英文计数/坏率键。"""
    rows = []
    for row in table.sort_values("分箱").to_dict("records"):
        count, good, bad = int(row["样本总数"]), int(row["好样本数"]), int(row["坏样本数"])
        rows.append(
            {
                **row,
                "count": count,
                "good": good,
                "bad": bad,
                "bad_rate": float(row["坏样本率"]),
                "good_rate": good / count if count else 0.0,
                "bad_dist": float(row["坏样本占比"]),
                "good_dist": float(row["好样本占比"]),
                "woe": float(row["分档WOE值"]),
                "iv": float(row["分档IV值"]),
                "lift": float(row["LIFT值"]),
            }
        )
    return rows


def objective_stats_from_counts(bin_stats, total_good, total_bad):
    """旧 OR 内部入口复用与 compute_bin_stats 相同的充分统计内核。"""
    good = np.asarray([stat["good"] for stat in bin_stats], dtype=float)
    bad = np.asarray([stat["bad"] for stat in bin_stats], dtype=float)
    if good.sum() != total_good or bad.sum() != total_bad:
        raise ValueError("自定义目标的分箱好坏计数与总体不一致")
    table = _binary_stats_from_counts(np.arange(len(bin_stats)), good, bad, round_digits=False)
    return _callback_rows(table)


def evaluate_callback(callback, bins, y, *, woe_clip=None):
    """所有回调都最大化一个有限实数；回调异常不能被当作约束冲突吞掉。"""
    target = np.asarray(y, dtype=int)
    table = compute_bin_stats(np.asarray(bins, dtype=int), target, round_digits=False, woe_clip=woe_clip)
    return _invoke_callback(callback, table, int(np.sum(target == 0)), int(np.sum(target == 1)))


def _invoke_callback(callback, table, total_good, total_bad):
    rows = _callback_rows(table)
    try:
        value = callback(rows, total_good, total_bad)
    except Exception as exc:
        raise ValueError(f"自定义分箱 objective 执行失败: {exc}") from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("自定义分箱 objective 必须返回有限实数，不能返回布尔值、NaN、无穷或数组")
    return float(value)


def make_objective_scorer(callback, x, y, *, woe_clip=None):
    """一次排序建立前缀计数；每个候选只聚合各箱，避免重新扫描全部训练行。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    order = np.argsort(x, kind="stable")
    values, target = x[order], y[order]
    prefix_bad = np.r_[0, np.cumsum(target == 1, dtype=int)]
    total_bad = int(prefix_bad[-1])
    total_good = int(len(target) - total_bad)

    def score(cuts):
        positions = np.searchsorted(values, np.sort(cuts), side="left")
        bounds = np.r_[0, positions, len(values)]
        bad = np.diff(prefix_bad[bounds])
        counts = np.diff(bounds)
        occupied = counts > 0
        table = _binary_stats_from_counts(
            np.flatnonzero(occupied),
            (counts - bad)[occupied],
            bad[occupied],
            round_digits=False,
            woe_clip=woe_clip,
        )
        return _invoke_callback(callback, table, total_good, total_bad)

    return score


def predefined_iv_total(bin_stats):
    """复合目标使用总体好坏分布的 IV，不能用箱内坏率减好率替代。"""
    if all("iv" in stat for stat in bin_stats):
        return float(sum(stat["iv"] for stat in bin_stats))
    return float(
        woe_iv_vectorized(
            np.asarray([stat["good"] for stat in bin_stats]),
            np.asarray([stat["bad"] for stat in bin_stats]),
        )[2]
    )
