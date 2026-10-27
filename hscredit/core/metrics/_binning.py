"""分箱统计内部实现.

提供高效的分箱后指标计算，供其他指标模块使用。
支持二元目标（0/1）、连续目标（金额、余额等）和金额加权模式。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Literal
from scipy import stats

from ._base import _woe_iv_vectorized
from ._semantics import format_bin_stats, metric_metadata, rejection_statistics, validate_amounts


def _safe_divide(numerator, denominator, default: float = 0.0) -> np.ndarray:
    """执行支持广播的安全除法，分母为零时返回默认值。"""
    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    result = np.full(numerator_array.shape, default, dtype=float)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array != 0,
    )
    return result


def _risk_oriented_cumulative_sums(
    bin_ids: np.ndarray,
    good_values: np.ndarray,
    bad_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """从正常分箱中风险较高的一端计算累计好坏样本量。"""
    bin_ids = np.asarray(bin_ids)
    good_values = np.asarray(good_values, dtype=float)
    bad_values = np.asarray(bad_values, dtype=float)

    normal_positions = np.flatnonzero(bin_ids >= 0)
    reserved_positions = np.flatnonzero(bin_ids < 0)
    if len(normal_positions) > 1:
        normal_bad_rates = _safe_divide(
            bad_values[normal_positions],
            good_values[normal_positions] + bad_values[normal_positions],
        )
        if normal_bad_rates[-1] > normal_bad_rates[0]:
            normal_positions = normal_positions[::-1]

    cumulative_order = np.concatenate([normal_positions, reserved_positions])
    cumulative_good = np.empty_like(good_values, dtype=float)
    cumulative_bad = np.empty_like(bad_values, dtype=float)
    cumulative_good[cumulative_order] = np.cumsum(good_values[cumulative_order])
    cumulative_bad[cumulative_order] = np.cumsum(bad_values[cumulative_order])
    return cumulative_good, cumulative_bad


def _normalize_curve_metric(metric: str) -> str:
    metric_norm = str(metric).strip().lower()
    aliases = {
        "lift": "lift",
        "badrate": "bad_rate",
        "bad_rate": "bad_rate",
        "坏样本率": "bad_rate",
    }
    if metric_norm not in aliases:
        raise ValueError("metric 必须是 'lift' 或 'bad_rate'")
    return aliases[metric_norm]


def _normalize_curve_trend(monotonic: str) -> str:
    trend_norm = str(monotonic).strip().lower()
    aliases = {
        "ascending": "ascending",
        "单增": "ascending",
        "increase": "ascending",
        "descending": "descending",
        "单减": "descending",
        "decrease": "descending",
        "valley": "valley",
        "u": "valley",
        "正u": "valley",
        "正u型": "valley",
        "peak": "peak",
        "inverted_u": "peak",
        "倒u": "peak",
        "倒u型": "peak",
    }
    if trend_norm not in aliases:
        raise ValueError("monotonic 必须是 ascending/descending/valley/peak")
    return aliases[trend_norm]


def _count_curve_violations(values: np.ndarray, trend: str) -> int:
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0
    tol = 1e-10
    diffs = np.diff(vals)
    if trend == "ascending":
        return int(np.sum(diffs < -tol))
    if trend == "descending":
        return int(np.sum(diffs > tol))
    if len(vals) < 3:
        return 0
    if trend == "peak":
        return min(
            int(np.sum(np.diff(vals[: pivot + 1]) < -tol)) + int(np.sum(np.diff(vals[pivot:]) > tol))
            for pivot in range(1, len(vals) - 1)
        )
    if trend == "valley":
        return min(
            int(np.sum(np.diff(vals[: pivot + 1]) > tol)) + int(np.sum(np.diff(vals[pivot:]) < -tol))
            for pivot in range(1, len(vals) - 1)
        )
    return 0


def _fit_monotone_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    trend: Literal["ascending", "descending"] = "descending",
) -> np.ndarray:
    """带单调约束的二次曲线最小二乘拟合.

    拟合 ``y ≈ a·x² + b·x + c`` ，通过约束保证拟合曲线在 ``x`` 的取值区间内
    严格单调（抛物线顶点落在区间之外），避免无约束 ``np.polyfit`` 的顶点落入
    区间内部，导致二次项系数无法反映"单调趋势下的弯曲程度"。

    约束原理：导数 ``y' = 2a·x + b`` 是线性函数，其区间最值在端点处取得，
    因此只需约束两端点的导数符号即可保证整个区间单调：

    - ``'descending'`` ： ``y'(x_min) ≤ 0`` 且 ``y'(x_max) ≤ 0`` （区间内只减不增）

    - ``'ascending'`` ： ``y'(x_min) ≥ 0`` 且 ``y'(x_max) ≥ 0`` （区间内只增不减）

    使用 SLSQP 约束优化求解（以无约束解为初值，允许不可行初值自动投影），
    优化失败时回退为无约束二次拟合。

    :param x: 自变量数组
    :param y: 因变量数组
    :param trend: 目标单调方向， ``'ascending'`` 或 ``'descending'``
    :return: 拟合系数 ``[a, b, c]``
    """
    from scipy.optimize import minimize

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # 无约束解：既作为 SLSQP 初值，也作为优化失败时的回退结果
    fallback = np.polyfit(x_arr, y_arr, 2)

    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))

    def objective(params):
        a, b, c = params
        return float(np.sum((y_arr - (a * x_arr**2 + b * x_arr + c)) ** 2))

    if trend == "descending":
        # 两端点导数均 <= 0：2a·x + b <= 0  =>  -(2a·x + b) >= 0
        constraints = [
            {"type": "ineq", "fun": lambda p: -(2.0 * p[0] * x_min + p[1])},
            {"type": "ineq", "fun": lambda p: -(2.0 * p[0] * x_max + p[1])},
        ]
    else:
        # 两端点导数均 >= 0：2a·x + b >= 0
        constraints = [
            {"type": "ineq", "fun": lambda p: 2.0 * p[0] * x_min + p[1]},
            {"type": "ineq", "fun": lambda p: 2.0 * p[0] * x_max + p[1]},
        ]

    try:
        result = minimize(
            objective,
            x0=np.asarray(fallback, dtype=float),
            constraints=constraints,
            method="SLSQP",
        )
        if result.success and np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=float)
    except Exception:
        pass
    return np.asarray(fallback, dtype=float)


def quadratic_curve_coefficient(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal["lift", "bad_rate"] = "lift",
    monotonic: Literal["ascending", "descending", "valley", "peak"] = "descending",
) -> float:
    """计算分箱曲线的二次项系数指标.

    基于分箱后的 ``LIFT值`` 或 ``坏样本率`` 序列做二次多项式拟合，
    返回经趋势方向标准化后的二次项系数。返回值越大，表示曲线越符合指定趋势
    且弯曲程度越明显，可作为“最优分箱”搜索的目标函数。

    **参数**

    :param bins: 分箱索引数组（整数）
    :param y: 目标变量 (0/1)，0=好样本，1=坏样本
    :param metric: 拟合曲线类型，默认 ``'lift'`` ：

        - ``'lift'`` ：使用各箱 LIFT 值序列

        - ``'bad_rate'`` ：使用各箱坏样本率序列

    :param monotonic: 目标趋势，默认 ``'descending'`` ：

        - ``'ascending'`` ：期望曲线单调递增（使用单调约束二次拟合，
          保证拟合曲线在区间内只增不减）

        - ``'descending'`` ：期望曲线单调递减（使用单调约束二次拟合，
          保证拟合曲线在区间内只减不增）

        - ``'valley'`` ：期望曲线先降后升（U 形，二次项系数为正）

        - ``'peak'`` ：期望曲线先升后降（倒 U 形，二次项系数为负）

    :return: 标准化后的二次项系数；曲线违反目标趋势时返回负值作为惩罚，
        有效箱数不足 3 或曲线为常数时返回 ``0.0``

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.metrics import quadratic_curve_coefficient
    >>> bins = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> y = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    >>> quadratic_curve_coefficient(bins, y, metric='lift', monotonic='descending')
    """
    metric = _normalize_curve_metric(metric)
    monotonic = _normalize_curve_trend(monotonic)

    table = compute_bin_stats(np.asarray(bins), np.asarray(y), round_digits=False)
    valid = table[table["分箱"] >= 0].reset_index(drop=True)
    if len(valid) < 3:
        return 0.0

    curve_values = (
        valid["LIFT值"].to_numpy(dtype=float) if metric == "lift" else valid["坏样本率"].to_numpy(dtype=float)
    )
    if np.allclose(curve_values, curve_values[0], atol=1e-12, rtol=0):
        return 0.0

    x_axis = np.linspace(-1.0, 1.0, len(curve_values), dtype=float)
    if monotonic in ("ascending", "descending"):
        # 单调趋势使用约束二次拟合：保证拟合曲线在区间内单调，
        # 避免无约束抛物线顶点落入区间内部导致系数方向与趋势相悖
        coef = float(_fit_monotone_quadratic(x_axis, curve_values, monotonic)[0])
    else:
        coef = float(np.polyfit(x_axis, curve_values, 2)[0])

    if monotonic == "peak":
        oriented_coef = -coef
    elif monotonic == "valley":
        oriented_coef = coef
    else:
        oriented_coef = abs(coef)

    violations = _count_curve_violations(curve_values, monotonic)
    return oriented_coef if violations == 0 else -abs(oriented_coef)


def _composite_binning_quality_components(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal["lift", "bad_rate"] = "lift",
    monotonic: Literal["ascending", "descending", "valley", "peak"] = "descending",
) -> dict:
    """拆解复合分箱评分的组成部分，供评分和搜索复用。"""
    metric = _normalize_curve_metric(metric)
    monotonic = _normalize_curve_trend(monotonic)

    table = compute_bin_stats(np.asarray(bins), np.asarray(y), round_digits=False)
    valid = table[table["分箱"] >= 0].reset_index(drop=True)
    if len(valid) == 0:
        return {
            "quadratic_score": 0.0,
            "head_score": 0.0,
            "tail_score": 0.0,
            "head_peak_bonus": 0.0,
            "tail_zero_bonus": 0.0,
            "tail_collapse_penalty": 0.0,
            "head_cumulative_gain": 0.0,
            "tail_compression_gain": 0.0,
            "marginal_return": 0.0,
            "marginal_decay_penalty": 0.0,
            "share_floor_bonus": 0.0,
            "spread": 0.0,
            "step_sum": 0.0,
            "tail_slope_bonus": 0.0,
            "monotonic_bonus": 0.0,
            "zero_pairs_penalty": 0.0,
            "n_bins_bonus": 0.0,
            "violations": 0,
            "n_bins": 0,
        }

    curve_values = (
        valid["LIFT值"].to_numpy(dtype=float) if metric == "lift" else valid["坏样本率"].to_numpy(dtype=float)
    )
    shares = valid["样本占比"].to_numpy(dtype=float)
    bad_rates = valid["坏样本率"].to_numpy(dtype=float)

    quadratic_score = quadratic_curve_coefficient(
        bins=np.asarray(bins),
        y=np.asarray(y),
        metric=metric,
        monotonic=monotonic,
    )

    head_value = float(curve_values[0]) if len(curve_values) > 0 else 0.0
    tail_value = float(curve_values[-1]) if len(curve_values) > 0 else 0.0
    head_share = float(shares[0]) if len(shares) > 0 else 0.0
    tail_share = float(shares[-1]) if len(shares) > 0 else 0.0

    head_score = head_value * np.sqrt(max(head_share, 1e-12))
    if monotonic == "descending":
        tail_score = max(0.0, 1.0 - tail_value) * np.sqrt(max(tail_share, 1e-12))
        oriented_diffs = curve_values[:-1] - curve_values[1:] if len(curve_values) > 1 else np.array([], dtype=float)
    elif monotonic == "ascending":
        tail_score = tail_value * np.sqrt(max(tail_share, 1e-12))
        oriented_diffs = curve_values[1:] - curve_values[:-1] if len(curve_values) > 1 else np.array([], dtype=float)
    else:
        tail_score = abs(tail_value) * np.sqrt(max(tail_share, 1e-12))
        oriented_diffs = np.abs(np.diff(curve_values)) if len(curve_values) > 1 else np.array([], dtype=float)

    shared_exposure = (shares[:-1] + shares[1:]) / 2.0 if len(shares) > 1 else np.array([], dtype=float)
    positive_margins = np.maximum(oriented_diffs, 0.0) if len(oriented_diffs) > 0 else np.array([], dtype=float)
    marginal_return = float(np.sum(positive_margins * shared_exposure)) if len(positive_margins) > 0 else 0.0

    if len(curve_values) > 0:
        head_window = max(1, min(2, len(curve_values)))
        tail_window = max(1, min(2, len(curve_values)))
        head_cumulative_gain = float(np.sum(curve_values[:head_window] * shares[:head_window]))
        if monotonic == "descending":
            tail_compression_gain = float(
                np.sum(np.maximum(1.0 - curve_values[-tail_window:], 0.0) * shares[-tail_window:])
            )
        elif monotonic == "ascending":
            tail_compression_gain = float(np.sum(curve_values[-tail_window:] * shares[-tail_window:]))
        else:
            tail_compression_gain = float(np.sum(np.abs(curve_values[-tail_window:]) * shares[-tail_window:]))
    else:
        head_cumulative_gain = 0.0
        tail_compression_gain = 0.0

    if len(positive_margins) > 1:
        marginal_decay_penalty = float(
            np.sum(np.maximum(positive_margins[1:] - positive_margins[:-1], 0.0) * shared_exposure[1:])
        )
    else:
        marginal_decay_penalty = 0.0

    share_floor_bonus = float(min(head_share, 0.08) + min(tail_share, 0.08))
    head_peak_bonus = float(max(head_value - 1.0, 0.0) ** 2 * np.sqrt(max(head_share, 1e-12)))
    tail_zero_bonus = float(max(1.0 - tail_value, 0.0) ** 2 * np.sqrt(max(tail_share, 1e-12)))
    tail_collapse_penalty = float(max(tail_value, 0.0) * max(tail_share, 1e-12))
    spread = float(np.max(curve_values) - np.min(curve_values)) if len(curve_values) > 1 else 0.0
    step_sum = float(np.sum(np.abs(np.diff(curve_values)))) if len(curve_values) > 1 else 0.0
    tail_slope_bonus = float(max(0.0, positive_margins[-1]) * tail_share) if len(positive_margins) > 0 else 0.0
    zero_pairs_penalty = (
        float(np.sum((bad_rates[:-1] <= 1e-12) & (bad_rates[1:] <= 1e-12))) * 0.5 if len(bad_rates) > 1 else 0.0
    )
    violations = _count_curve_violations(curve_values, monotonic)
    monotonic_bonus = 0.15 if violations == 0 else -0.25 * float(violations)
    n_bins_bonus = len(valid) * 0.04

    return {
        "quadratic_score": float(quadratic_score),
        "head_score": float(head_score),
        "tail_score": float(tail_score),
        "head_peak_bonus": float(head_peak_bonus),
        "tail_zero_bonus": float(tail_zero_bonus),
        "tail_collapse_penalty": float(tail_collapse_penalty),
        "head_cumulative_gain": float(head_cumulative_gain),
        "tail_compression_gain": float(tail_compression_gain),
        "marginal_return": float(marginal_return),
        "marginal_decay_penalty": float(marginal_decay_penalty),
        "share_floor_bonus": float(share_floor_bonus),
        "spread": float(spread),
        "step_sum": float(step_sum),
        "tail_slope_bonus": float(tail_slope_bonus),
        "monotonic_bonus": float(monotonic_bonus),
        "zero_pairs_penalty": float(zero_pairs_penalty),
        "n_bins_bonus": float(n_bins_bonus),
        "violations": int(violations),
        "n_bins": int(len(valid)),
    }


def composite_binning_quality(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal["lift", "bad_rate"] = "lift",
    monotonic: Literal["ascending", "descending", "valley", "peak"] = "descending",
) -> float:
    """计算复合分箱质量评分.

    在  :func:`quadratic_curve_coefficient`  之外，进一步将多项业务偏好加权汇总为
    单一评分，用于驱动“最优分箱”搜索，使分箱在保持单调趋势的同时兼顾头尾区分度
    与样本占比。显式纳入目标的分量包括：二次曲线得分、头部累计收益、尾部压降收益、
    样本占比加权边际收益、边际收益递减惩罚、头尾样本占比下限偏好、
    尾部塌陷/相邻零坏样本率惩罚等。

    **参数**

    :param bins: 分箱索引数组（整数）
    :param y: 目标变量 (0/1)，0=好样本，1=坏样本
    :param metric: 拟合曲线类型， ``'lift'`` 或 ``'bad_rate'`` ，默认 ``'lift'``
        （含义同  :func:`quadratic_curve_coefficient` ）

    :param monotonic: 目标趋势， ``'ascending'`` / ``'descending'`` / ``'valley'`` /
        ``'peak'`` ，默认 ``'descending'`` （含义同  :func:`quadratic_curve_coefficient` ）

    :return: 复合质量评分（float），越大表示分箱质量越好

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.metrics import composite_binning_quality
    >>> bins = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> y = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    >>> composite_binning_quality(bins, y, metric='lift', monotonic='descending')
    """
    comp = _composite_binning_quality_components(
        bins=np.asarray(bins),
        y=np.asarray(y),
        metric=metric,
        monotonic=monotonic,
    )
    return float(
        comp["quadratic_score"] * 1.0
        + comp["head_score"] * 1.05
        + comp["head_peak_bonus"] * 1.40
        + comp["tail_score"] * 0.12
        + comp["tail_zero_bonus"] * 1.25
        - comp["tail_collapse_penalty"] * 1.10
        + comp["head_cumulative_gain"] * 1.80
        + comp["tail_compression_gain"] * 0.55
        + comp["marginal_return"] * 1.05
        - comp["marginal_decay_penalty"] * 0.35
        + comp["share_floor_bonus"] * 0.25
        + comp["spread"] * 0.12
        + comp["step_sum"] * 0.03
        + comp["tail_slope_bonus"] * 0.15
        + comp["n_bins_bonus"]
        + comp["monotonic_bonus"]
        - comp["zero_pairs_penalty"]
    )


def _validate_bin_arrays(bins, y, target_type="binary"):
    """公共统计输入按位置对齐，禁止非法标签污染样本计数。"""
    bins = np.asarray(bins)
    try:
        y = np.asarray(y, dtype=np.float64)
        numeric_bins = np.asarray(bins, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("分箱索引和目标必须是有效数值") from exc
    if bins.ndim != 1 or y.ndim != 1 or len(bins) != len(y):
        raise ValueError("分箱索引和目标必须是一维等长数组")
    if not np.isfinite(numeric_bins).all() or np.any(numeric_bins != np.floor(numeric_bins)):
        raise ValueError("分箱索引必须是有限整数")
    if bins.dtype.kind not in "iu":
        if np.any(numeric_bins >= 2.0**63) or np.any(numeric_bins < -(2.0**63)):
            raise ValueError("分箱索引超出64位整数范围")
        bins = numeric_bins.astype(np.int64)
    if not np.isfinite(y).all():
        raise ValueError("目标不能包含缺失或无穷值")
    if target_type in {"binary", "amount_weighted"} and not np.isin(y, [0, 1]).all():
        raise ValueError("二分类目标只能包含0和1，1表示坏样本")
    return bins, y


def compute_bin_stats(
    bins: np.ndarray,
    y: np.ndarray,
    target_type: Literal["binary", "continuous", "amount_weighted"] = "binary",
    amount: Optional[np.ndarray] = None,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None,
) -> pd.DataFrame:
    """计算分箱统计信息（hscredit 全库统一的分箱指标计算入口）.

    一次性计算某一特征分箱后的全部统计指标。所有分箱器、IV/PSI/LIFT 指标、
    规则报告与可视化均复用本函数，以保证口径一致。支持三种目标类型。

    **参数**

    :param bins: 分箱索引数组（整数）。约定 ``-1`` 表示缺失值箱、 ``-2`` 表示特殊值箱，
        两者在输出中被排到正常分箱之后

    :param y: 目标变量，含义随 ``target_type`` 而变：

        - ``target_type='binary'`` ：0/1 数组（0=好样本，1=坏样本）

        - ``target_type='continuous'`` ：连续值数组（如逾期金额、余额）

        - ``target_type='amount_weighted'`` ：0/1 数组，并配合 ``amount`` 使用

    :param target_type: 目标变量类型，默认 ``'binary'`` ：

        - ``'binary'`` ：二分类，计算 样本/好坏数、坏样本率、WOE、IV、LIFT、KS 等

        - ``'continuous'`` ：连续目标，计算各箱均值/求和等金额统计，不计算 WOE/IV

        - ``'amount_weighted'`` ：基于二元标签但所有统计按 ``amount`` 加权（金额维度坏账）

    :param amount: 金额数组，仅 ``target_type='amount_weighted'`` 时必需，长度同 ``y``
    :param epsilon: 平滑参数，避免 WOE/IV 计算中除零或取对数为 ``±inf`` ，默认 ``1e-10``
    :param bin_labels: 可选的分箱区间标签列表，长度需与唯一分箱数一致；
        缺省时输出箱序号

    :param round_digits: 是否对浮点列做四舍五入格式化，默认 ``True`` ；
        作为中间计算（如指标搜索）时应设为 ``False`` 以保留精度

    :param woe_clip: WOE 值截断阈值，默认 ``None`` 不截断。
        当某箱无坏样本或无好样本时 WOE 可能趋于 ``±inf`` ，
        设置后将 WOE 限制在 ``[-woe_clip, woe_clip]`` ，避免评分卡分数异常

        截断后的 WOE 继续参与原有 IV 列计算；额外输出 ``原始分档WOE值`` 、
        ``原始分档IV值`` 和 ``原始指标IV值`` ，以便区分原始信息量与实际编码信息量。

    :return: 分箱统计 DataFrame（中文列名）。 ``binary`` 模式主要列包括：
        ``分箱`` / ``分箱标签`` / ``样本总数`` / ``好样本数`` / ``坏样本数`` /
        ``样本占比`` / ``坏样本率`` / ``WOE值`` / ``分档IV值`` / ``LIFT值`` /
        ``累积坏样本数`` / ``累积好样本数`` / ``分档KS值`` / ``坏账改善`` 等。
        累积类指标从正常分箱中坏样本率较高的一端开始计算，缺失值、特殊值等
        保留箱最后纳入累计，输出行顺序保持不变

    单类或某类别有效金额为0时，分档KS采用与 ``ks`` 相同的0兼容值；
    坏账改善保留原有兼容值：全拒绝且总体坏率大于0时为1，总体坏率为0时为0。
    全部拒绝时风险拒绝比为NaN；未拒绝时为0。不添加指标状态或原因说明列。

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.metrics import compute_bin_stats
    >>> bins = np.array([0, 0, 1, 1, 2, 2])
    >>>
    >>> # 二元目标（0/1）
    >>> y_binary = np.array([0, 1, 0, 1, 0, 1])
    >>> compute_bin_stats(bins, y_binary, target_type='binary')
    >>>
    >>> # 连续目标（逾期金额）
    >>> y_amount = np.array([0, 1000, 0, 2000, 0, 1500])
    >>> compute_bin_stats(bins, y_amount, target_type='continuous')
    >>>
    >>> # 金额加权（按逾期金额加权的坏账统计）
    >>> y_flag = np.array([0, 1, 0, 1, 0, 1])
    >>> amount = np.array([100, 1000, 200, 2000, 150, 1500])
    >>> compute_bin_stats(bins, y_flag, target_type='amount_weighted', amount=amount)

    **引用**

    WOE / IV 的定义见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley；
    本函数的 WOE 取 ``ln(坏样本占比 / 好样本占比)`` ，与 toad、scorecardpipeline 口径一致。
    """
    bins, y = _validate_bin_arrays(bins, y, target_type)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")
    if woe_clip is not None and (not np.isfinite(woe_clip) or woe_clip <= 0):
        raise ValueError("woe_clip 必须为有限正数或None")

    if target_type == "binary":
        return _compute_bin_stats_binary(bins, y, epsilon, bin_labels, round_digits, woe_clip)
    elif target_type == "continuous":
        return _compute_bin_stats_continuous(bins, y, epsilon, bin_labels, round_digits)
    elif target_type == "amount_weighted":
        if amount is None:
            raise ValueError("target_type='amount_weighted'时必须提供amount参数")
        amount = validate_amounts(amount, len(y))
        return _compute_bin_stats_amount_weighted(bins, y, amount, epsilon, bin_labels, round_digits, woe_clip)
    else:
        raise ValueError(f"target_type必须是'binary'/'continuous'/'amount_weighted'，得到: {target_type}")


def _compute_bin_stats_binary(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None,
) -> pd.DataFrame:
    """计算二元目标的分箱统计.

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :param woe_clip: WOE值截断阈值
    :return: 分箱统计DataFrame
    """
    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)

    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前

    # 使用 Python int 计算排序键，避免 numpy 2.x（NEP 50）下窄整型（如分类编码 int8）
    # 与大整数运算溢出抛出 OverflowError
    sort_order = np.asarray(sorted(range(len(sort_keys)), key=lambda i: sort_keys[i]), dtype=np.intp)
    old_to_new = np.empty(len(sort_order), dtype=np.intp)
    old_to_new[sort_order] = np.arange(len(sort_order))
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = old_to_new[bin_indices]

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)
    good_counts = np.bincount(bin_indices, weights=(y == 0).astype(int), minlength=n_bins)
    bad_counts = np.bincount(bin_indices, weights=y, minlength=n_bins)
    return _binary_stats_from_counts(unique_bins, good_counts, bad_counts, epsilon, bin_labels, round_digits, woe_clip)


def _binary_stats_from_counts(
    unique_bins,
    good_counts,
    bad_counts,
    epsilon=1e-10,
    bin_labels=None,
    round_digits=True,
    woe_clip=None,
    *,
    amount_weighted=False,
):
    """行级、分块聚合与金额加权共用的二分类派生指标内核。"""
    n_bins = len(unique_bins)
    counts = good_counts + bad_counts
    count_dtype = np.float64 if amount_weighted else np.int64

    bad_rate = _safe_divide(bad_counts, counts)

    # 计算WOE和IV
    woe, bin_iv, total_iv = _woe_iv_vectorized(good_counts, bad_counts, epsilon, woe_clip)

    # 计算占比
    total = counts.sum()
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()

    count_distr = counts / total if total > 0 else np.zeros(n_bins)
    good_distr = good_counts / total_good if total_good > 0 else np.zeros(n_bins)
    bad_distr = bad_counts / total_bad if total_bad > 0 else np.zeros(n_bins)

    # 计算总体坏样本率（用于LIFT）
    overall_bad_rate = total_bad / total if total > 0 else 0.0

    # 计算LIFT值
    lift = _safe_divide(bad_rate, overall_bad_rate)

    _, bad_improve, risk_reject = rejection_statistics(total, total_bad, counts, bad_counts)

    # 从正常分箱中风险较高的一端累计；缺失值、特殊值等保留箱最后纳入累计
    cum_good, cum_bad = _risk_oriented_cumulative_sums(unique_bins, good_counts, bad_counts)
    cum_total = cum_good + cum_bad

    cum_bad_rate = _safe_divide(cum_bad, cum_total)
    cum_lift = _safe_divide(cum_bad_rate, overall_bad_rate)
    _, cum_bad_improve, cum_risk_reject = rejection_statistics(total, total_bad, cum_total, cum_bad)

    # 计算KS值（使用cum_bad计算累积坏样本数）
    ks_values = np.zeros(n_bins)
    if total_good > 0 and total_bad > 0:
        ks_values = np.abs(cum_bad / total_bad - cum_good / total_good)

    # 构建DataFrame
    data = {"分箱": unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data["分箱标签"] = bin_labels

    data.update(
        {
            "样本总数": counts.astype(count_dtype),
            "好样本数": good_counts.astype(count_dtype),
            "坏样本数": bad_counts.astype(count_dtype),
            "样本占比": count_distr,
            "好样本占比": good_distr,
            "坏样本占比": bad_distr,
            "坏样本率": bad_rate,
            "分档WOE值": woe,
            "分档IV值": bin_iv,
            "指标IV值": total_iv,
            "LIFT值": lift,
            "坏账改善": bad_improve,
            "风险拒绝比": risk_reject,
            "累积LIFT值": cum_lift,
            "累积坏账改善": cum_bad_improve,
            "累计风险拒绝比": cum_risk_reject,
            "累积好样本数": cum_good.astype(count_dtype),
            "累积坏样本数": cum_bad.astype(count_dtype),
            "分档KS值": ks_values,
        }
    )

    df = pd.DataFrame(data)

    if woe_clip is not None:
        original_woe, original_iv, original_total_iv = _woe_iv_vectorized(good_counts, bad_counts, epsilon)
        df["原始分档WOE值"] = original_woe
        df["原始分档IV值"] = original_iv
        df["原始指标IV值"] = original_total_iv
    target_type = "amount_weighted" if amount_weighted else "binary"
    df.attrs["统计口径"] = metric_metadata(target_type, total_good=total_good, total_bad=total_bad, woe_clip=woe_clip)

    return format_bin_stats(df, target_type=target_type, round_digits=round_digits)


def _compute_bin_stats_continuous(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
) -> pd.DataFrame:
    """计算连续目标的分箱统计（如逾期金额、余额等）.

    :param bins: 分箱索引数组
    :param y: 目标变量（连续值，如逾期金额）
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :return: 分箱统计DataFrame
    """
    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)

    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前

    # 使用 Python int 计算排序键，避免 numpy 2.x（NEP 50）下窄整型（如分类编码 int8）
    # 与大整数运算溢出抛出 OverflowError
    sort_order = np.asarray(sorted(range(len(sort_keys)), key=lambda i: sort_keys[i]), dtype=np.intp)
    old_to_new = np.empty(len(sort_order), dtype=np.intp)
    old_to_new[sort_order] = np.arange(len(sort_order))
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = old_to_new[bin_indices]

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)

    # 计算每箱的样本数
    counts = np.bincount(bin_indices, minlength=n_bins)

    # 先按每箱真实观测中心化，再对残差做二次中心化；避免大金额下
    # E[y²]-E[y]² 的灾难性消减，也不需要每个箱重复扫描完整样本。
    first_positions = np.full(n_bins, len(y), dtype=np.intp)
    np.minimum.at(first_positions, bin_indices, np.arange(len(y), dtype=np.intp))
    origins = y[first_positions]
    offsets = y - origins[bin_indices]
    offset_sums = np.bincount(bin_indices, weights=offsets, minlength=n_bins)
    offset_means = _safe_divide(offset_sums, counts)
    y_mean = origins + offset_means
    y_sum = origins * counts + offset_sums
    deviations = offsets - offset_means[bin_indices]
    y_var = _safe_divide(np.bincount(bin_indices, weights=deviations**2, minlength=n_bins), counts)
    y_std = np.sqrt(np.maximum(y_var, 0))
    y_min = np.full(n_bins, np.inf)
    y_max = np.full(n_bins, -np.inf)
    np.minimum.at(y_min, bin_indices, y)
    np.maximum.at(y_max, bin_indices, y)

    # 计算占比
    total = counts.sum()
    total_y = y_sum.sum()

    count_distr = counts / total if total > 0 else np.zeros(n_bins)
    contribution_valid = bool(total_y > 0 and np.all(y >= 0))
    y_distr = y_sum / total_y if contribution_valid else np.full(n_bins, np.nan)

    # 计算总体均值（用于LIFT）
    overall_mean = total_y / total if total > 0 else 0.0

    # 计算LIFT值（连续目标的LIFT = 该箱均值 / 总体均值）
    lift = y_mean / overall_mean if contribution_valid else np.full(n_bins, np.nan)

    # 计算"改善"指标（连续目标的改善 = 该箱贡献度与该箱样本占比的差异）
    bad_improve = _safe_divide(y_distr - count_distr, count_distr, default=np.nan)

    # 按分箱顺序计算累积指标
    cum_counts = np.cumsum(counts)
    cum_y = np.cumsum(y_sum)
    cum_mean = np.where(cum_counts > 0, cum_y / cum_counts, 0.0)

    # 累积LIFT
    cum_lift = cum_mean / overall_mean if contribution_valid else np.full(n_bins, np.nan)

    # 累积占比
    cum_count_distr = cum_counts / total if total > 0 else np.zeros(n_bins)
    cum_y_distr = cum_y / total_y if contribution_valid else np.full(n_bins, np.nan)

    # 累积改善
    cum_bad_improve = _safe_divide(cum_y_distr - cum_count_distr, cum_count_distr, default=np.nan)

    # 计算KS值（对于连续目标，使用金额累积占比 vs 样本累积占比的差）
    ks_values = np.abs(cum_y_distr - cum_count_distr)

    # 构建DataFrame
    data = {"分箱": unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data["分箱标签"] = bin_labels

    data.update(
        {
            "样本总数": counts.astype(np.int64),
            "样本占比": count_distr,
            "目标值总和": y_sum,
            "目标值均值": y_mean,
            "目标值标准差": y_std,
            "目标值最小值": y_min,
            "目标值最大值": y_max,
            "目标值占比": y_distr,
            "平均LIFT值": lift,
            "贡献改善": bad_improve,
            "累积样本数": cum_counts.astype(int),
            "累积目标值": cum_y,
            "累积平均值": cum_mean,
            "累积LIFT值": cum_lift,
            "累积贡献改善": cum_bad_improve,
            "分档KS值": ks_values,
        }
    )

    df = pd.DataFrame(data)

    df.attrs["统计口径"] = {
        "目标类型": "continuous",
        "统计单位": "样本与连续目标值分别统计",
        "KS定义": "非负目标贡献累计占比与样本累计占比的差，非二分类KS",
        "贡献指标状态": "可计算" if contribution_valid else "不适用：目标含负数或没有正总量，贡献/LIFT/KS为空",
        "累计方向": "按分箱顺序累计",
        "展示舍入": False,
    }
    return format_bin_stats(df, target_type="continuous", round_digits=round_digits)


def _compute_bin_stats_amount_weighted(
    bins: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None,
) -> pd.DataFrame:
    """计算金额加权的分箱统计（基于二元标签，但按金额加权）.

    适用于风控场景：基于逾期金额加权的坏账统计分析。
    输出列名与binary模式保持一致（样本/好样本/坏样本表示金额），便于统一处理。

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param amount: 金额数组（如放款金额、余额等）
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :param woe_clip: WOE值截断阈值
    :return: 分箱统计DataFrame（列名与binary模式保持一致，便于对比）
    """
    bins = np.asarray(bins)
    y = np.asarray(y)
    amount = np.asarray(amount)

    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)

    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前

    # 使用 Python int 计算排序键，避免 numpy 2.x（NEP 50）下窄整型（如分类编码 int8）
    # 与大整数运算溢出抛出 OverflowError
    sort_order = np.asarray(sorted(range(len(sort_keys)), key=lambda i: sort_keys[i]), dtype=np.intp)
    old_to_new = np.empty(len(sort_order), dtype=np.intp)
    old_to_new[sort_order] = np.arange(len(sort_order))
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = old_to_new[bin_indices]

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)

    # 计算每个分箱的好金额和坏金额（金额口径的核心）
    good_amounts = np.bincount(bin_indices, weights=(y == 0).astype(float) * amount, minlength=n_bins)
    bad_amounts = np.bincount(bin_indices, weights=y.astype(float) * amount, minlength=n_bins)
    return _binary_stats_from_counts(
        unique_bins,
        good_amounts,
        bad_amounts,
        epsilon,
        bin_labels,
        round_digits,
        woe_clip,
        amount_weighted=True,
    )


def add_margins(table: pd.DataFrame) -> pd.DataFrame:
    """为分箱表添加合计行.

    在分箱统计表末尾追加一行“合计”，对原始计数列求和、累计计数列取总体值，
    对率值类列按总体重算。
    缺失值箱与特殊值箱被放在正常分箱之后、合计行之前。
    兼容单层表头与多级表头（MultiIndex），同时支持样本口径与金额口径。
    多特征分别合计，重复调用不重复累计已有合计行。累计全量拒绝的风险拒绝比为NaN。

    **参数**

    :param table: 分箱统计表，通常由  :func:`compute_bin_stats`  生成，
        需包含 ``分箱标签`` 列；为空表时原样返回。连续目标的总体矩由各箱统计合并，
        应传入 ``round_digits=False`` 的统计表以保留合计精度；已舍入的数据不能恢复原始精度

    :return: 在末尾添加 ``合计`` 行后的分箱表（不修改入参，返回新对象）

    **参考样例**

    >>> import numpy as np
    >>> from hscredit.core.metrics import compute_bin_stats, add_margins
    >>> table = compute_bin_stats(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    >>> add_margins(table)
    """
    if table.empty:
        return table

    # 查找分箱标签列（支持多级表头和单层表头）
    bin_label_col = None
    is_multi = isinstance(table.columns, pd.MultiIndex)

    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == "分箱标签":
            bin_label_col = col
            break

    if bin_label_col is None:
        return table

    group_cols = [col for col in table.columns if (col[1] if is_multi else col) in {"指标名称", "逾期标签"}]
    if group_cols and len(table[group_cols].drop_duplicates()) > 1:
        group_key = group_cols[0] if len(group_cols) == 1 else group_cols
        result = pd.concat(
            [add_margins(part) for _, part in table.groupby(group_key, sort=False, dropna=False)],
            ignore_index=True,
        )
        result.attrs = dict(table.attrs)
        return result
    # 合计为保留标签，重复调用不再次把已有合计当作原始数据累计。
    detail = table.loc[table[bin_label_col] != "合计"]
    if detail.empty:
        return table.copy()
    table = detail

    # 分离正常分箱、缺失值、特殊值
    normal_bins = []
    missing_bin = None
    special_bin = None

    for idx, row in table.iterrows():
        label = row[bin_label_col]
        if label == "missing":
            missing_bin = row
        elif label == "special":
            special_bin = row
        else:
            normal_bins.append(row)

    # 计算合计行
    total_row = table.iloc[0].copy()
    total_row[bin_label_col] = "合计"

    # 需要汇总的原始计数列（累计计数不能再次求和）
    numeric_cols = []
    cumulative_count_cols = []
    count_cols_by_name = {
        "样本总数": {},
        "好样本数": {},
        "坏样本数": {},
    }

    def _column_group(column):
        return column[0] if is_multi else None

    def _resolve_count_column(metric_name, group):
        grouped_columns = count_cols_by_name[metric_name]
        if group in grouped_columns:
            return grouped_columns[group]
        if len(grouped_columns) == 1:
            return next(iter(grouped_columns.values()))
        return None

    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ["样本总数", "好样本数", "坏样本数"]:
            numeric_cols.append(col)
            count_cols_by_name[col_name][_column_group(col)] = col
        if col_name in ["累积好样本数", "累积坏样本数"]:
            cumulative_count_cols.append(col)

    # 对每一列求和
    for col in numeric_cols:
        total_row[col] = table[col].sum()

    # 合计行的累计计数就是总体计数，而不是各行前缀和的总和
    cumulative_to_total = {"累积好样本数": "好样本数", "累积坏样本数": "坏样本数"}
    for cumulative_col in cumulative_count_cols:
        cumulative_name = cumulative_col[1] if is_multi else cumulative_col
        total_name = cumulative_to_total[cumulative_name]
        matching_total_col = _resolve_count_column(total_name, _column_group(cumulative_col))
        if matching_total_col is not None:
            total_row[cumulative_col] = total_row[matching_total_col]

    # 非空类别的总体占比为1；空类别不能凭空生成100%的占比。
    ratio_cols = []
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ["样本占比", "好样本占比", "坏样本占比"]:
            ratio_cols.append(col)

    for col in ratio_cols:
        name = col[1] if is_multi else col
        denominator_name = {"样本占比": "样本总数", "好样本占比": "好样本数", "坏样本占比": "坏样本数"}[name]
        count_col = _resolve_count_column(denominator_name, _column_group(col))
        total_row[col] = float(total_row[count_col] > 0) if count_col is not None else np.nan

    # 计算坏样本率
    bad_rate_cols = [col for col in table.columns if (col[1] if is_multi else col) == "坏样本率"]
    for bad_rate_col in bad_rate_cols:
        group = _column_group(bad_rate_col)
        bad_sample_col = _resolve_count_column("坏样本数", group)
        sample_total_col = _resolve_count_column("样本总数", group)
        if bad_sample_col is not None and sample_total_col is not None and total_row[sample_total_col] > 0:
            total_row[bad_rate_col] = total_row[bad_sample_col] / total_row[sample_total_col]
        else:
            # 缺少坏样本数/样本总数时无法重算总体坏样本率，置空避免沿用首行的错误值
            total_row[bad_rate_col] = np.nan

    # 合计表示总体基准；改善保留兼容值，累计全量拒绝的效率为NaN。
    ratio_metric_names = {"LIFT值", "坏账改善", "风险拒绝比", "累积LIFT值", "累积坏账改善", "累计风险拒绝比"}
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ratio_metric_names:
            group = _column_group(col)
            bad_sample_col = _resolve_count_column("坏样本数", group)
            sample_total_col = _resolve_count_column("样本总数", group)
            valid = (
                bad_sample_col is not None
                and sample_total_col is not None
                and total_row[bad_sample_col] > 0
                and total_row[sample_total_col] > 0
            )
            if col_name in {"LIFT值", "累积LIFT值"}:
                total_row[col] = 1.0 if valid else 0.0
            elif col_name in {"坏账改善", "风险拒绝比"}:
                total_row[col] = 0.0
            elif col_name == "累积坏账改善":
                total_row[col] = 1.0 if valid else 0.0
            else:
                total_row[col] = 0.0 if sample_total_col is not None and total_row[sample_total_col] == 0 else np.nan

    # WOE和IV值：分档WOE=0，分档IV=0，指标IV=各分档IV之和
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in {"分档WOE值", "原始分档WOE值"}:
            total_row[col] = 0.0
        elif col_name in {"分档IV值", "原始分档IV值"}:
            total_row[col] = 0.0
        elif col_name in {"指标IV值", "原始指标IV值"}:
            total_row[col] = table[col].iloc[0]

    # KS值：取最大KS
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == "分档KS值":
            total_row[col] = table[col].max()

    # 连续目标：合并各箱的原始矩，不能沿用首行或简单平均箱均值。
    for col in table.columns:
        if (col[1] if is_multi else col) != "目标值总和":
            continue
        group = _column_group(col)
        columns = {
            (candidate[1] if is_multi else candidate): candidate
            for candidate in table.columns
            if _column_group(candidate) == group
        }
        count_col = _resolve_count_column("样本总数", group)
        if count_col is None:
            continue
        counts = table[count_col].to_numpy(dtype=float)
        count = counts.sum()
        target_sum = table[col].sum()
        mean = target_sum / count if count else np.nan
        totals = {
            "目标值总和": target_sum,
            "目标值均值": mean,
            "累积样本数": count,
            "累积目标值": target_sum,
            "累积平均值": mean,
        }
        for metric, operation in [("目标值最小值", "min"), ("目标值最大值", "max"), ("目标值占比", "sum")]:
            if metric in columns:
                totals[metric] = (
                    table[columns[metric]].sum(min_count=1)
                    if operation == "sum"
                    else getattr(table[columns[metric]], operation)()
                )
        if "目标值标准差" in columns:
            totals["目标值标准差"] = np.nan
            if "目标值均值" in columns and count > 0:
                nonempty = counts > 0
                means = table[columns["目标值均值"]].to_numpy(dtype=float)[nonempty]
                stds = table[columns["目标值标准差"]].to_numpy(dtype=float)[nonempty]
                weights = counts[nonempty]
                offsets = means - means[0]
                centered = offsets - np.average(offsets, weights=weights)
                totals["目标值标准差"] = np.sqrt(np.average(stds**2 + centered**2, weights=weights))
        if "累积样本数" in columns:
            # 全量累计行保留原内核的零分母/连续目标改善口径。
            endpoint = table.iloc[int(np.argmax(table[columns["累积样本数"]].to_numpy()))]
            for destination, source in [
                ("平均LIFT值", "累积LIFT值"),
                ("累积LIFT值", "累积LIFT值"),
                ("贡献改善", "累积贡献改善"),
                ("累积贡献改善", "累积贡献改善"),
            ]:
                if source in columns:
                    totals[destination] = endpoint[columns[source]]
        for metric, value in totals.items():
            if metric in columns:
                total_row[columns[metric]] = value

    # 重新组合：正常分箱 -> 缺失值 -> 特殊值 -> 合计
    result_rows = []

    # 正常分箱
    for row in normal_bins:
        result_rows.append(row)

    # 缺失值
    if missing_bin is not None:
        result_rows.append(missing_bin)

    # 特殊值
    if special_bin is not None:
        result_rows.append(special_bin)

    # 合计
    result_rows.append(total_row)

    # 重建DataFrame
    result_table = pd.DataFrame(result_rows)
    result_table = result_table.reset_index(drop=True)
    result_table.attrs = dict(table.attrs)

    return result_table


def _ks_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """按分箱计算KS统计量.

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (max_ks, ks_array)
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    unique_bins = np.unique(bins)
    n_bins = len(unique_bins)

    total_good = (y == 0).sum()
    total_bad = y.sum()

    if total_good == 0 or total_bad == 0:
        return 0.0, np.zeros(n_bins)

    good_counts = np.array([((y == 0) & (bins == b)).sum() for b in unique_bins])
    bad_counts = np.array([y[bins == b].sum() for b in unique_bins])

    cum_good = np.cumsum(good_counts)
    cum_bad = np.cumsum(bad_counts)

    cum_good_rate = cum_good / total_good
    cum_bad_rate = cum_bad / total_bad

    ks_values = np.abs(cum_good_rate - cum_bad_rate)
    max_ks = ks_values.max()

    return max_ks, ks_values


def _chi2_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """按分箱计算卡方统计量.

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (chi2_stat, p_value, chi2_contrib)
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    contingency = pd.crosstab(bins, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 1.0, np.zeros(contingency.shape[0])

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    chi2_contrib = _safe_divide((contingency - expected) ** 2, expected).sum(axis=1)

    return chi2_stat, p_value, chi2_contrib


# 分箱优化相关的辅助函数，供binning模块使用
def woe_iv_vectorized(
    good_counts: np.ndarray, bad_counts: np.ndarray, epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """向量化计算WOE和IV（供binning模块使用）.

    :param good_counts: 每个箱的好样本数
    :param bad_counts: 每个箱的坏样本数
    :param epsilon: 平滑参数
    :return: (woe_array, bin_iv_array, total_iv)
    """
    return _woe_iv_vectorized(good_counts, bad_counts, epsilon)


def iv_for_splits(x: np.ndarray, y: np.ndarray, splits: List[float], epsilon: float = 1e-10) -> float:
    """计算给定分割点的IV值（供binning模块使用）.

    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits: 分割点列表
    :param epsilon: 平滑参数
    :return: IV值
    """
    bins = np.digitize(x, bins=splits, right=False)

    unique_bins = np.unique(bins)
    good_counts = np.array([((y == 0) & (bins == b)).sum() for b in unique_bins])
    bad_counts = np.array([y[bins == b].sum() for b in unique_bins])

    _, _, total_iv = _woe_iv_vectorized(good_counts, bad_counts, epsilon)
    return total_iv


def ks_for_splits(x: np.ndarray, y: np.ndarray, splits: List[float]) -> float:
    """计算给定分割点的KS值（供binning模块使用）.

    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits: 分割点列表
    :return: KS值
    """
    bins = np.digitize(x, bins=splits, right=False)
    max_ks, _ = _ks_by_bin(bins, y)
    return max_ks


def compare_splits_iv(
    x: np.ndarray, y: np.ndarray, splits1: List[float], splits2: List[float], epsilon: float = 1e-10
) -> Tuple[float, float, str]:
    """比较两组分割点的IV值（供binning模块使用）.

    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits1: 第一组分割点
    :param splits2: 第二组分割点
    :param epsilon: 平滑参数
    :return: (iv1, iv2, better)
    """
    iv1 = iv_for_splits(x, y, splits1, epsilon)
    iv2 = iv_for_splits(x, y, splits2, epsilon)

    better = "splits1" if iv1 >= iv2 else "splits2"
    return iv1, iv2, better


def compare_splits_ks(
    x: np.ndarray, y: np.ndarray, splits1: List[float], splits2: List[float]
) -> Tuple[float, float, str]:
    """比较两组分割点的KS值（供binning模块使用）.

    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits1: 第一组分割点
    :param splits2: 第二组分割点
    :return: (ks1, ks2, better)
    """
    ks1 = ks_for_splits(x, y, splits1)
    ks2 = ks_for_splits(x, y, splits2)

    better = "splits1" if ks1 >= ks2 else "splits2"
    return ks1, ks2, better
