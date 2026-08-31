"""分箱统计内部实现.

提供高效的分箱后指标计算，供其他指标模块使用。
支持二元目标（0/1）、连续目标（金额、余额等）和金额加权模式。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Literal
from scipy import stats

from ._base import _woe_iv_vectorized


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
        'lift': 'lift',
        'badrate': 'bad_rate',
        'bad_rate': 'bad_rate',
        '坏样本率': 'bad_rate',
    }
    if metric_norm not in aliases:
        raise ValueError("metric 必须是 'lift' 或 'bad_rate'")
    return aliases[metric_norm]



def _normalize_curve_trend(monotonic: str) -> str:
    trend_norm = str(monotonic).strip().lower()
    aliases = {
        'ascending': 'ascending', '单增': 'ascending', 'increase': 'ascending',
        'descending': 'descending', '单减': 'descending', 'decrease': 'descending',
        'valley': 'valley', 'u': 'valley', '正u': 'valley', '正u型': 'valley',
        'peak': 'peak', 'inverted_u': 'peak', '倒u': 'peak', '倒u型': 'peak',
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
    if trend == 'ascending':
        return int(np.sum(diffs < -tol))
    if trend == 'descending':
        return int(np.sum(diffs > tol))
    if len(vals) < 3:
        return 0
    if trend == 'peak':
        return min(
            int(np.sum(np.diff(vals[:pivot + 1]) < -tol)) + int(np.sum(np.diff(vals[pivot:]) > tol))
            for pivot in range(1, len(vals) - 1)
        )
    if trend == 'valley':
        return min(
            int(np.sum(np.diff(vals[:pivot + 1]) > tol)) + int(np.sum(np.diff(vals[pivot:]) < -tol))
            for pivot in range(1, len(vals) - 1)
        )
    return 0


def _fit_monotone_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    trend: Literal['ascending', 'descending'] = 'descending',
) -> np.ndarray:
    """带单调约束的二次曲线最小二乘拟合.

    拟合 ``y ≈ a·x² + b·x + c``，通过约束保证拟合曲线在 ``x`` 的取值区间内
    严格单调（抛物线顶点落在区间之外），避免无约束 ``np.polyfit`` 的顶点落入
    区间内部，导致二次项系数无法反映"单调趋势下的弯曲程度"。

    约束原理：导数 ``y' = 2a·x + b`` 是线性函数，其区间最值在端点处取得，
    因此只需约束两端点的导数符号即可保证整个区间单调：

    - ``'descending'``：``y'(x_min) ≤ 0`` 且 ``y'(x_max) ≤ 0``（区间内只减不增）
    - ``'ascending'``：``y'(x_min) ≥ 0`` 且 ``y'(x_max) ≥ 0``（区间内只增不减）

    使用 SLSQP 约束优化求解（以无约束解为初值，允许不可行初值自动投影），
    优化失败时回退为无约束二次拟合。

    :param x: 自变量数组
    :param y: 因变量数组
    :param trend: 目标单调方向，``'ascending'`` 或 ``'descending'``
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
        return float(np.sum((y_arr - (a * x_arr ** 2 + b * x_arr + c)) ** 2))

    if trend == 'descending':
        # 两端点导数均 <= 0：2a·x + b <= 0  =>  -(2a·x + b) >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda p: -(2.0 * p[0] * x_min + p[1])},
            {'type': 'ineq', 'fun': lambda p: -(2.0 * p[0] * x_max + p[1])},
        ]
    else:
        # 两端点导数均 >= 0：2a·x + b >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda p: 2.0 * p[0] * x_min + p[1]},
            {'type': 'ineq', 'fun': lambda p: 2.0 * p[0] * x_max + p[1]},
        ]

    try:
        result = minimize(
            objective,
            x0=np.asarray(fallback, dtype=float),
            constraints=constraints,
            method='SLSQP',
        )
        if result.success and np.all(np.isfinite(result.x)):
            return np.asarray(result.x, dtype=float)
    except Exception:
        pass
    return np.asarray(fallback, dtype=float)



def quadratic_curve_coefficient(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal['lift', 'bad_rate'] = 'lift',
    monotonic: Literal['ascending', 'descending', 'valley', 'peak'] = 'descending'
) -> float:
    """计算分箱曲线的二次项系数指标.

    基于分箱后的 ``LIFT值`` 或 ``坏样本率`` 序列做二次多项式拟合，
    返回经趋势方向标准化后的二次项系数。返回值越大，表示曲线越符合指定趋势
    且弯曲程度越明显，可作为“最优分箱”搜索的目标函数。

    **参数**

    :param bins: 分箱索引数组（整数）
    :param y: 目标变量 (0/1)，0=好样本，1=坏样本
    :param metric: 拟合曲线类型，默认 ``'lift'``：

        - ``'lift'``：使用各箱 LIFT 值序列
        - ``'bad_rate'``：使用各箱坏样本率序列

    :param monotonic: 目标趋势，默认 ``'descending'``：

        - ``'ascending'``：期望曲线单调递增（使用单调约束二次拟合，
          保证拟合曲线在区间内只增不减）
        - ``'descending'``：期望曲线单调递减（使用单调约束二次拟合，
          保证拟合曲线在区间内只减不增）
        - ``'valley'``：期望曲线先降后升（U 形，二次项系数为正）
        - ``'peak'``：期望曲线先升后降（倒 U 形，二次项系数为负）

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
    valid = table[table['分箱'] >= 0].reset_index(drop=True)
    if len(valid) < 3:
        return 0.0

    curve_values = valid['LIFT值'].to_numpy(dtype=float) if metric == 'lift' else valid['坏样本率'].to_numpy(dtype=float)
    if np.allclose(curve_values, curve_values[0], atol=1e-12, rtol=0):
        return 0.0

    x_axis = np.linspace(-1.0, 1.0, len(curve_values), dtype=float)
    if monotonic in ('ascending', 'descending'):
        # 单调趋势使用约束二次拟合：保证拟合曲线在区间内单调，
        # 避免无约束抛物线顶点落入区间内部导致系数方向与趋势相悖
        coef = float(_fit_monotone_quadratic(x_axis, curve_values, monotonic)[0])
    else:
        coef = float(np.polyfit(x_axis, curve_values, 2)[0])

    if monotonic == 'peak':
        oriented_coef = -coef
    elif monotonic == 'valley':
        oriented_coef = coef
    else:
        oriented_coef = abs(coef)

    violations = _count_curve_violations(curve_values, monotonic)
    return oriented_coef if violations == 0 else -abs(oriented_coef)


def _composite_binning_quality_components(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal['lift', 'bad_rate'] = 'lift',
    monotonic: Literal['ascending', 'descending', 'valley', 'peak'] = 'descending'
) -> dict:
    """拆解复合分箱评分的组成部分，供评分和搜索复用。"""
    metric = _normalize_curve_metric(metric)
    monotonic = _normalize_curve_trend(monotonic)

    table = compute_bin_stats(np.asarray(bins), np.asarray(y), round_digits=False)
    valid = table[table['分箱'] >= 0].reset_index(drop=True)
    if len(valid) == 0:
        return {
            'quadratic_score': 0.0,
            'head_score': 0.0,
            'tail_score': 0.0,
            'head_peak_bonus': 0.0,
            'tail_zero_bonus': 0.0,
            'tail_collapse_penalty': 0.0,
            'head_cumulative_gain': 0.0,
            'tail_compression_gain': 0.0,
            'marginal_return': 0.0,
            'marginal_decay_penalty': 0.0,
            'share_floor_bonus': 0.0,
            'spread': 0.0,
            'step_sum': 0.0,
            'tail_slope_bonus': 0.0,
            'monotonic_bonus': 0.0,
            'zero_pairs_penalty': 0.0,
            'n_bins_bonus': 0.0,
            'violations': 0,
            'n_bins': 0,
        }

    curve_values = valid['LIFT值'].to_numpy(dtype=float) if metric == 'lift' else valid['坏样本率'].to_numpy(dtype=float)
    shares = valid['样本占比'].to_numpy(dtype=float)
    bad_rates = valid['坏样本率'].to_numpy(dtype=float)

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
    if monotonic == 'descending':
        tail_score = max(0.0, 1.0 - tail_value) * np.sqrt(max(tail_share, 1e-12))
        oriented_diffs = curve_values[:-1] - curve_values[1:] if len(curve_values) > 1 else np.array([], dtype=float)
    elif monotonic == 'ascending':
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
        if monotonic == 'descending':
            tail_compression_gain = float(np.sum(np.maximum(1.0 - curve_values[-tail_window:], 0.0) * shares[-tail_window:]))
        elif monotonic == 'ascending':
            tail_compression_gain = float(np.sum(curve_values[-tail_window:] * shares[-tail_window:]))
        else:
            tail_compression_gain = float(np.sum(np.abs(curve_values[-tail_window:]) * shares[-tail_window:]))
    else:
        head_cumulative_gain = 0.0
        tail_compression_gain = 0.0

    if len(positive_margins) > 1:
        marginal_decay_penalty = float(np.sum(np.maximum(positive_margins[1:] - positive_margins[:-1], 0.0) * shared_exposure[1:]))
    else:
        marginal_decay_penalty = 0.0

    share_floor_bonus = float(min(head_share, 0.08) + min(tail_share, 0.08))
    head_peak_bonus = float(max(head_value - 1.0, 0.0) ** 2 * np.sqrt(max(head_share, 1e-12)))
    tail_zero_bonus = float(max(1.0 - tail_value, 0.0) ** 2 * np.sqrt(max(tail_share, 1e-12)))
    tail_collapse_penalty = float(max(tail_value, 0.0) * max(tail_share, 1e-12))
    spread = float(np.max(curve_values) - np.min(curve_values)) if len(curve_values) > 1 else 0.0
    step_sum = float(np.sum(np.abs(np.diff(curve_values)))) if len(curve_values) > 1 else 0.0
    tail_slope_bonus = float(max(0.0, positive_margins[-1]) * tail_share) if len(positive_margins) > 0 else 0.0
    zero_pairs_penalty = float(np.sum((bad_rates[:-1] <= 1e-12) & (bad_rates[1:] <= 1e-12))) * 0.5 if len(bad_rates) > 1 else 0.0
    violations = _count_curve_violations(curve_values, monotonic)
    monotonic_bonus = 0.15 if violations == 0 else -0.25 * float(violations)
    n_bins_bonus = len(valid) * 0.04

    return {
        'quadratic_score': float(quadratic_score),
        'head_score': float(head_score),
        'tail_score': float(tail_score),
        'head_peak_bonus': float(head_peak_bonus),
        'tail_zero_bonus': float(tail_zero_bonus),
        'tail_collapse_penalty': float(tail_collapse_penalty),
        'head_cumulative_gain': float(head_cumulative_gain),
        'tail_compression_gain': float(tail_compression_gain),
        'marginal_return': float(marginal_return),
        'marginal_decay_penalty': float(marginal_decay_penalty),
        'share_floor_bonus': float(share_floor_bonus),
        'spread': float(spread),
        'step_sum': float(step_sum),
        'tail_slope_bonus': float(tail_slope_bonus),
        'monotonic_bonus': float(monotonic_bonus),
        'zero_pairs_penalty': float(zero_pairs_penalty),
        'n_bins_bonus': float(n_bins_bonus),
        'violations': int(violations),
        'n_bins': int(len(valid)),
    }


def composite_binning_quality(
    bins: np.ndarray,
    y: np.ndarray,
    metric: Literal['lift', 'bad_rate'] = 'lift',
    monotonic: Literal['ascending', 'descending', 'valley', 'peak'] = 'descending'
) -> float:
    """计算复合分箱质量评分.

    在 :func:`quadratic_curve_coefficient` 之外，进一步将多项业务偏好加权汇总为
    单一评分，用于驱动“最优分箱”搜索，使分箱在保持单调趋势的同时兼顾头尾区分度
    与样本占比。显式纳入目标的分量包括：二次曲线得分、头部累计收益、尾部压降收益、
    样本占比加权边际收益、边际收益递减惩罚、头尾样本占比下限偏好、
    尾部塌陷/相邻零坏样本率惩罚等。

    **参数**

    :param bins: 分箱索引数组（整数）
    :param y: 目标变量 (0/1)，0=好样本，1=坏样本
    :param metric: 拟合曲线类型，``'lift'`` 或 ``'bad_rate'``，默认 ``'lift'``
        （含义同 :func:`quadratic_curve_coefficient`）
    :param monotonic: 目标趋势，``'ascending'`` / ``'descending'`` / ``'valley'`` /
        ``'peak'``，默认 ``'descending'``（含义同 :func:`quadratic_curve_coefficient`）
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
        comp['quadratic_score'] * 1.0
        + comp['head_score'] * 1.05
        + comp['head_peak_bonus'] * 1.40
        + comp['tail_score'] * 0.12
        + comp['tail_zero_bonus'] * 1.25
        - comp['tail_collapse_penalty'] * 1.10
        + comp['head_cumulative_gain'] * 1.80
        + comp['tail_compression_gain'] * 0.55
        + comp['marginal_return'] * 1.05
        - comp['marginal_decay_penalty'] * 0.35
        + comp['share_floor_bonus'] * 0.25
        + comp['spread'] * 0.12
        + comp['step_sum'] * 0.03
        + comp['tail_slope_bonus'] * 0.15
        + comp['n_bins_bonus']
        + comp['monotonic_bonus']
        - comp['zero_pairs_penalty']
    )


def compute_bin_stats(
    bins: np.ndarray,
    y: np.ndarray,
    target_type: Literal['binary', 'continuous', 'amount_weighted'] = 'binary',
    amount: Optional[np.ndarray] = None,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None
) -> pd.DataFrame:
    """计算分箱统计信息（hscredit 全库统一的分箱指标计算入口）.

    一次性计算某一特征分箱后的全部统计指标。所有分箱器、IV/PSI/LIFT 指标、
    规则报告与可视化均复用本函数，以保证口径一致。支持三种目标类型。

    **参数**

    :param bins: 分箱索引数组（整数）。约定 ``-1`` 表示缺失值箱、``-2`` 表示特殊值箱，
        两者在输出中被排到正常分箱之后
    :param y: 目标变量，含义随 ``target_type`` 而变：

        - ``target_type='binary'``：0/1 数组（0=好样本，1=坏样本）
        - ``target_type='continuous'``：连续值数组（如逾期金额、余额）
        - ``target_type='amount_weighted'``：0/1 数组，并配合 ``amount`` 使用

    :param target_type: 目标变量类型，默认 ``'binary'``：

        - ``'binary'``：二分类，计算 样本/好坏数、坏样本率、WOE、IV、LIFT、KS 等
        - ``'continuous'``：连续目标，计算各箱均值/求和等金额统计，不计算 WOE/IV
        - ``'amount_weighted'``：基于二元标签但所有统计按 ``amount`` 加权（金额维度坏账）

    :param amount: 金额数组，仅 ``target_type='amount_weighted'`` 时必需，长度同 ``y``
    :param epsilon: 平滑参数，避免 WOE/IV 计算中除零或取对数为 ``±inf``，默认 ``1e-10``
    :param bin_labels: 可选的分箱区间标签列表，长度需与唯一分箱数一致；
        缺省时输出箱序号
    :param round_digits: 是否对浮点列做四舍五入格式化，默认 ``True``；
        作为中间计算（如指标搜索）时应设为 ``False`` 以保留精度
    :param woe_clip: WOE 值截断阈值，默认 ``None`` 不截断。
        当某箱无坏样本或无好样本时 WOE 可能趋于 ``±inf``，
        设置后将 WOE 限制在 ``[-woe_clip, woe_clip]``，避免评分卡分数异常
    :return: 分箱统计 DataFrame（中文列名）。``binary`` 模式主要列包括：
        ``分箱`` / ``分箱标签`` / ``样本总数`` / ``好样本数`` / ``坏样本数`` /
        ``样本占比`` / ``坏样本率`` / ``WOE值`` / ``分档IV值`` / ``LIFT值`` /
        ``累积坏样本数`` / ``累积好样本数`` / ``分档KS值`` / ``坏账改善`` 等。
        累积类指标从正常分箱中坏样本率较高的一端开始计算，缺失值、特殊值等
        保留箱最后纳入累计，输出行顺序保持不变

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
    本函数的 WOE 取 ``ln(坏样本占比 / 好样本占比)``，与 toad、scorecardpipeline 口径一致。
    """
    bins = np.asarray(bins)
    y = np.asarray(y, dtype=np.float64)
    
    if target_type == 'binary':
        return _compute_bin_stats_binary(bins, y, epsilon, bin_labels, round_digits, woe_clip)
    elif target_type == 'continuous':
        return _compute_bin_stats_continuous(bins, y, epsilon, bin_labels, round_digits)
    elif target_type == 'amount_weighted':
        if amount is None:
            raise ValueError("target_type='amount_weighted'时必须提供amount参数")
        amount = np.asarray(amount, dtype=np.float64)
        return _compute_bin_stats_amount_weighted(bins, y, amount, epsilon, bin_labels, round_digits, woe_clip)
    else:
        raise ValueError(f"target_type必须是'binary'/'continuous'/'amount_weighted'，得到: {target_type}")


def _compute_bin_stats_binary(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None
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
    sort_order = np.argsort([int(sk[0]) * 10000 + int(sk[1]) for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)
    good_counts = np.bincount(bin_indices, weights=(y == 0).astype(int), minlength=n_bins)
    bad_counts = np.bincount(bin_indices, weights=y, minlength=n_bins)
    counts = good_counts + bad_counts

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

    # 坏账改善 = (全量坏样本率 - 拒绝后剩余样本坏样本率) / 全量坏样本率
    # 拒绝后剩余样本坏样本率 = (total_bad - bin_bad) / (total - bin_total)
    # 展开后与 (overall_bad_rate - bad_rate) / overall_bad_rate 等价
    other_bad = total_bad - bad_counts
    other_total = total - counts
    other_bad_rate = _safe_divide(other_bad, other_total)
    bad_improve = _safe_divide(overall_bad_rate - other_bad_rate, overall_bad_rate)

    # 风险拒绝比 = 坏账改善 / 当前箱样本占比
    # 反映"每拒绝1%样本能带来多少坏账改善"
    risk_reject = _safe_divide(bad_improve, count_distr)

    # 从正常分箱中风险较高的一端累计；缺失值、特殊值等保留箱最后纳入累计
    cum_good, cum_bad = _risk_oriented_cumulative_sums(unique_bins, good_counts, bad_counts)
    cum_total = cum_good + cum_bad

    cum_bad_rate = _safe_divide(cum_bad, cum_total)
    cum_lift = _safe_divide(cum_bad_rate, overall_bad_rate)
    # 累计坏账改善 = (全量坏样本率 - 累计拒绝后剩余样本坏样本率) / 全量坏样本率
    other_cum_bad = total_bad - cum_bad
    other_cum_total = total - cum_total
    other_cum_bad_rate = _safe_divide(other_cum_bad, other_cum_total)
    cum_bad_improve = _safe_divide(overall_bad_rate - other_cum_bad_rate, overall_bad_rate)
    # 累计风险拒绝比 = 累计坏账改善 / 累计样本占比
    cum_risk_reject = _safe_divide(cum_bad_improve, _safe_divide(cum_total, total))

    # 计算KS值（使用cum_bad计算累积坏样本数）
    cum_good_rate = cum_good / (total_good + epsilon)
    cum_bad_rate_ks = cum_bad / (total_bad + epsilon)
    ks_values = np.abs(cum_bad_rate_ks - cum_good_rate)

    # 构建DataFrame
    data = {'分箱': unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels

    data.update({
        '样本总数': counts.astype(int),
        '好样本数': good_counts.astype(int),
        '坏样本数': bad_counts.astype(int),
        '样本占比': count_distr,
        '好样本占比': good_distr,
        '坏样本占比': bad_distr,
        '坏样本率': bad_rate,
        '分档WOE值': woe,
        '分档IV值': bin_iv,
        '指标IV值': total_iv,
        'LIFT值': lift,
        '坏账改善': bad_improve,
        '风险拒绝比': risk_reject,
        '累积LIFT值': cum_lift,
        '累积坏账改善': cum_bad_improve,
        '累计风险拒绝比': cum_risk_reject,
        '累积好样本数': cum_good.astype(int),
        '累积坏样本数': cum_bad.astype(int),
        '分档KS值': ks_values,
    })

    df = pd.DataFrame(data)

    # 对浮点数列进行四舍五入格式化
    if round_digits:
        float_columns = {
            '样本占比': 6, '好样本占比': 6, '坏样本占比': 6,
            '坏样本率': 6, '分档WOE值': 6, '分档IV值': 6, '指标IV值': 6,
            'LIFT值': 4, '坏账改善': 4, '风险拒绝比': 4,
            '累积LIFT值': 4, '累积坏账改善': 4, '累计风险拒绝比': 4,
            '分档KS值': 6,
        }
        for col, digits in float_columns.items():
            if col in df.columns:
                df[col] = np.round(df[col], digits)

    return df


def _compute_bin_stats_continuous(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
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
    sort_order = np.argsort([int(sk[0]) * 10000 + int(sk[1]) for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)
    
    # 计算每箱的样本数
    counts = np.bincount(bin_indices, minlength=n_bins)
    
    # 计算每箱的目标值统计
    y_sum = np.bincount(bin_indices, weights=y, minlength=n_bins)
    y_mean = _safe_divide(y_sum, counts)
    
    # 计算每箱的方差和标准差
    y_squared_sum = np.bincount(bin_indices, weights=y**2, minlength=n_bins)
    y_var = _safe_divide(y_squared_sum, counts) - y_mean**2
    y_std = np.sqrt(np.maximum(y_var, 0))
    
    # 计算每箱的最小值和最大值
    y_min = np.array([y[bin_indices == i].min() if counts[i] > 0 else 0 for i in range(n_bins)])
    y_max = np.array([y[bin_indices == i].max() if counts[i] > 0 else 0 for i in range(n_bins)])
    
    # 计算占比
    total = counts.sum()
    total_y = y_sum.sum()
    
    count_distr = counts / total if total > 0 else np.zeros(n_bins)
    y_distr = y_sum / (total_y + epsilon) if total_y > 0 else np.zeros(n_bins)
    
    # 计算总体均值（用于LIFT）
    overall_mean = total_y / total if total > 0 else 0.0
    
    # 计算LIFT值（连续目标的LIFT = 该箱均值 / 总体均值）
    lift = np.where(y_mean > 0, y_mean / (overall_mean + epsilon), 0.0)
    
    # 计算"改善"指标（连续目标的改善 = 该箱贡献度与该箱样本占比的差异）
    bad_improve = np.where(
        count_distr > 0,
        (y_distr - count_distr) / (count_distr + epsilon),
        0.0
    )

    # 按分箱顺序计算累积指标
    cum_counts = np.cumsum(counts)
    cum_y = np.cumsum(y_sum)
    cum_mean = np.where(cum_counts > 0, cum_y / cum_counts, 0.0)

    # 累积LIFT
    cum_lift = np.where(cum_mean > 0, cum_mean / (overall_mean + epsilon), 0.0)
    
    # 累积占比
    cum_count_distr = cum_counts / total if total > 0 else np.zeros(n_bins)
    cum_y_distr = cum_y / (total_y + epsilon) if total_y > 0 else np.zeros(n_bins)
    
    # 累积改善
    cum_bad_improve = np.where(
        cum_count_distr > 0,
        (cum_y_distr - cum_count_distr) / (cum_count_distr + epsilon),
        0.0
    )

    # 计算KS值（对于连续目标，使用金额累积占比 vs 样本累积占比的差）
    ks_values = np.abs(cum_y_distr - cum_count_distr)

    # 构建DataFrame
    data = {'分箱': unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels

    data.update({
        '样本总数': counts.astype(int),
        '样本占比': count_distr,
        '目标值总和': y_sum,
        '目标值均值': y_mean,
        '目标值标准差': y_std,
        '目标值最小值': y_min,
        '目标值最大值': y_max,
        '目标值占比': y_distr,
        '平均LIFT值': lift,
        '贡献改善': bad_improve,
        '累积样本数': cum_counts.astype(int),
        '累积目标值': cum_y,
        '累积平均值': cum_mean,
        '累积LIFT值': cum_lift,
        '累积贡献改善': cum_bad_improve,
        '分档KS值': ks_values,
    })

    df = pd.DataFrame(data)

    # 对浮点数列进行四舍五入格式化
    if round_digits:
        float_columns = {
            '样本占比': 6,
            '目标值均值': 2, '目标值标准差': 2, '目标值最小值': 2, '目标值最大值': 2,
            '目标值占比': 6, '平均LIFT值': 4, '贡献改善': 4,
            '累积平均值': 2, '累积LIFT值': 4, '累积贡献改善': 4,
            '分档KS值': 6,
        }
        for col, digits in float_columns.items():
            if col in df.columns:
                df[col] = np.round(df[col], digits)

    return df


def _compute_bin_stats_amount_weighted(
    bins: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True,
    woe_clip: Optional[float] = None
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
    sort_order = np.argsort([int(sk[0]) * 10000 + int(sk[1]) for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])
    
    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]
    
    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted
    
    n_bins = len(unique_bins)
    
    # 计算每个分箱的好金额和坏金额（金额口径的核心）
    good_amounts = np.bincount(bin_indices, weights=(y == 0).astype(float) * amount, minlength=n_bins)
    bad_amounts = np.bincount(bin_indices, weights=y.astype(float) * amount, minlength=n_bins)
    amount_totals = good_amounts + bad_amounts
    
    # 计算占比（基于金额）
    total_amount = amount_totals.sum()
    total_good_amount = good_amounts.sum()
    total_bad_amount = bad_amounts.sum()
    
    amount_ratios = amount_totals / total_amount if total_amount > 0 else np.zeros(n_bins)
    good_amount_ratios = good_amounts / total_good_amount if total_good_amount > 0 else np.zeros(n_bins)
    bad_amount_ratios = bad_amounts / total_bad_amount if total_bad_amount > 0 else np.zeros(n_bins)
    
    # 金额口径坏账率 = 坏金额 / 总金额
    bad_rate = _safe_divide(bad_amounts, amount_totals)
    
    # 计算WOE和IV（基于金额占比）
    good_amounts_smooth = np.where(good_amounts == 0, epsilon, good_amounts)
    bad_amounts_smooth = np.where(bad_amounts == 0, epsilon, bad_amounts)
    total_good_smooth = good_amounts_smooth.sum()
    total_bad_smooth = bad_amounts_smooth.sum()
    
    good_distr = good_amounts_smooth / total_good_smooth if total_good_smooth > 0 else np.zeros(n_bins)
    bad_distr = bad_amounts_smooth / total_bad_smooth if total_bad_smooth > 0 else np.zeros(n_bins)
    
    woe = np.log(bad_distr / good_distr)
    
    # 截断极端WOE值，防止评分卡分数异常
    if woe_clip is not None:
        woe = np.clip(woe, -woe_clip, woe_clip)
    
    bin_iv = (bad_distr - good_distr) * woe
    total_iv = bin_iv.sum()
    
    # 计算LIFT值（金额口径）
    overall_bad_rate = total_bad_amount / total_amount if total_amount > 0 else 0.0
    lift = _safe_divide(bad_rate, overall_bad_rate)

    # 坏账改善 = (全量坏样本率 - 拒绝后坏样本率) / 全量坏样本率
    # 拒绝后坏样本率 = other_bad / other_total
    other_bad = total_bad_amount - bad_amounts
    other_total = total_amount - amount_totals
    other_bad_rate = _safe_divide(other_bad, other_total)
    bad_improve = _safe_divide(overall_bad_rate - other_bad_rate, overall_bad_rate)
    # 风险拒绝比 = 样本占比 = 该箱金额 / 全量金额
    risk_reject = amount_ratios

    # 金额口径与样本口径保持一致，从正常分箱中风险较高的一端累计
    cum_good, cum_bad = _risk_oriented_cumulative_sums(unique_bins, good_amounts, bad_amounts)
    cum_total = cum_good + cum_bad

    cum_bad_rate = _safe_divide(cum_bad, cum_total)
    cum_lift = _safe_divide(cum_bad_rate, overall_bad_rate)
    other_cum_bad = total_bad_amount - cum_bad
    other_cum_total = total_amount - cum_total
    # 累计坏账改善 = (全量坏样本率 - 累计拒绝后坏样本率) / 全量坏样本率
    other_cum_bad_rate = _safe_divide(other_cum_bad, other_cum_total)
    cum_bad_improve = _safe_divide(overall_bad_rate - other_cum_bad_rate, overall_bad_rate)
    # 累计风险拒绝比 = 累计样本占比
    cum_risk_reject = _safe_divide(cum_total, total_amount)
    
    # 计算KS值（基于金额累积占比）
    cum_good_rate = cum_good / (total_good_amount + epsilon)
    cum_bad_rate = cum_bad / (total_bad_amount + epsilon)
    ks_values = np.abs(cum_bad_rate - cum_good_rate)
    
    # 构建DataFrame（使用与样本口径统一的列名，便于统一处理）
    data = {'分箱': unique_bins}
    
    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels
    
    # 使用与样本口径相同的列名，便于统一处理
    data.update({
        '样本总数': np.round(amount_totals, 2),
        '好样本数': np.round(good_amounts, 2),
        '坏样本数': np.round(bad_amounts, 2),
        '样本占比': np.round(amount_ratios, 6),
        '好样本占比': np.round(good_amount_ratios, 6),
        '坏样本占比': np.round(bad_amount_ratios, 6),
        '坏样本率': np.round(bad_rate, 6),
        '分档WOE值': np.round(woe, 6),
        '分档IV值': np.round(bin_iv, 6),
        '指标IV值': np.round(total_iv, 6),
        'LIFT值': np.round(lift, 4),
        '坏账改善': np.round(bad_improve, 4),
        '风险拒绝比': np.round(risk_reject, 4),
        '累积LIFT值': np.round(cum_lift, 4),
        '累积坏账改善': np.round(cum_bad_improve, 4),
        '累计风险拒绝比': np.round(cum_risk_reject, 4),
        '累积好样本数': np.round(cum_good, 2),
        '累积坏样本数': np.round(cum_bad, 2),
        '分档KS值': np.round(ks_values, 6),
    })
    
    return pd.DataFrame(data)


def add_margins(table: pd.DataFrame) -> pd.DataFrame:
    """为分箱表添加合计行.

    在分箱统计表末尾追加一行“合计”，对原始计数列求和、累计计数列取总体值，
    对率值类列按总体重算。
    缺失值箱与特殊值箱被放在正常分箱之后、合计行之前。
    兼容单层表头与多级表头（MultiIndex），同时支持样本口径与金额口径。

    **参数**

    :param table: 分箱统计表，通常由 :func:`compute_bin_stats` 生成，
        需包含 ``分箱标签`` 列；为空表时原样返回
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
        if col_name == '分箱标签':
            bin_label_col = col
            break
    
    if bin_label_col is None:
        return table
    
    # 分离正常分箱、缺失值、特殊值
    normal_bins = []
    missing_bin = None
    special_bin = None
    
    for idx, row in table.iterrows():
        label = row[bin_label_col]
        if label == 'missing':
            missing_bin = row
        elif label == 'special':
            special_bin = row
        else:
            normal_bins.append(row)
    
    # 计算合计行
    total_row = table.iloc[0].copy()
    total_row[bin_label_col] = '合计'
    
    # 需要汇总的原始计数列（累计计数不能再次求和）
    numeric_cols = []
    cumulative_count_cols = []
    count_cols_by_name = {
        '样本总数': {},
        '好样本数': {},
        '坏样本数': {},
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
        if col_name in ['样本总数', '好样本数', '坏样本数']:
            numeric_cols.append(col)
            count_cols_by_name[col_name][_column_group(col)] = col
        if col_name in ['累积好样本数', '累积坏样本数']:
            cumulative_count_cols.append(col)
    
    # 对每一列求和
    for col in numeric_cols:
        total_row[col] = table[col].sum()

    # 合计行的累计计数就是总体计数，而不是各行前缀和的总和
    cumulative_to_total = {'累积好样本数': '好样本数', '累积坏样本数': '坏样本数'}
    for cumulative_col in cumulative_count_cols:
        cumulative_name = cumulative_col[1] if is_multi else cumulative_col
        total_name = cumulative_to_total[cumulative_name]
        matching_total_col = _resolve_count_column(total_name, _column_group(cumulative_col))
        if matching_total_col is not None:
            total_row[cumulative_col] = total_row[matching_total_col]
    
    # 计算占比类指标 = 1
    ratio_cols = []
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ['样本占比', '好样本占比', '坏样本占比']:
            ratio_cols.append(col)
    
    for col in ratio_cols:
        total_row[col] = 1.0
    
    # 计算坏样本率
    bad_rate_cols = [col for col in table.columns if (col[1] if is_multi else col) == '坏样本率']
    for bad_rate_col in bad_rate_cols:
        group = _column_group(bad_rate_col)
        bad_sample_col = _resolve_count_column('坏样本数', group)
        sample_total_col = _resolve_count_column('样本总数', group)
        if (
            bad_sample_col is not None
            and sample_total_col is not None
            and total_row[sample_total_col] > 0
        ):
            total_row[bad_rate_col] = total_row[bad_sample_col] / total_row[sample_total_col]
        else:
            # 缺少坏样本数/样本总数时无法重算总体坏样本率，置空避免沿用首行的错误值
            total_row[bad_rate_col] = np.nan
    
    # 合计行表示总体：LIFT=1，单箱拒绝改善=0；累计全量指标回到总体基准
    overall_metric_values = {
        'LIFT值': 1.0,
        '坏账改善': 0.0,
        '风险拒绝比': 0.0,
    }
    cumulative_metric_names = {'累积LIFT值', '累积坏账改善', '累计风险拒绝比'}
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in overall_metric_values:
            total_row[col] = overall_metric_values[col_name]
        elif col_name in cumulative_metric_names:
            group = _column_group(col)
            bad_sample_col = _resolve_count_column('坏样本数', group)
            sample_total_col = _resolve_count_column('样本总数', group)
            total_row[col] = (
                1.0
                if (
                    bad_sample_col is not None
                    and sample_total_col is not None
                    and total_row[bad_sample_col] > 0
                    and total_row[sample_total_col] > 0
                )
                else 0.0
            )
    
    # WOE和IV值：分档WOE=0，分档IV=0，指标IV=各分档IV之和
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '分档WOE值':
            total_row[col] = 0.0
        elif col_name == '分档IV值':
            total_row[col] = 0.0
        elif col_name == '指标IV值':
            total_row[col] = table[col].iloc[0]
    
    # KS值：取最大KS
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '分档KS值':
            total_row[col] = table[col].max()
    
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
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """向量化计算WOE和IV（供binning模块使用）.
    
    :param good_counts: 每个箱的好样本数
    :param bad_counts: 每个箱的坏样本数
    :param epsilon: 平滑参数
    :return: (woe_array, bin_iv_array, total_iv)
    """
    return _woe_iv_vectorized(good_counts, bad_counts, epsilon)


def iv_for_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[float],
    epsilon: float = 1e-10
) -> float:
    """计算给定分割点的IV值（供binning模块使用）.
    
    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits: 分割点列表
    :param epsilon: 平滑参数
    :return: IV值
    """
    bins = np.digitize(x, bins=splits, right=True)
    
    unique_bins = np.unique(bins)
    good_counts = np.array([((y == 0) & (bins == b)).sum() for b in unique_bins])
    bad_counts = np.array([y[bins == b].sum() for b in unique_bins])
    
    _, _, total_iv = _woe_iv_vectorized(good_counts, bad_counts, epsilon)
    return total_iv


def ks_for_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: List[float]
) -> float:
    """计算给定分割点的KS值（供binning模块使用）.
    
    :param x: 特征数组
    :param y: 目标变量 (0/1)
    :param splits: 分割点列表
    :return: KS值
    """
    bins = np.digitize(x, bins=splits, right=True)
    max_ks, _ = _ks_by_bin(bins, y)
    return max_ks


def compare_splits_iv(
    x: np.ndarray,
    y: np.ndarray,
    splits1: List[float],
    splits2: List[float],
    epsilon: float = 1e-10
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
    
    better = 'splits1' if iv1 >= iv2 else 'splits2'
    return iv1, iv2, better


def compare_splits_ks(
    x: np.ndarray,
    y: np.ndarray,
    splits1: List[float],
    splits2: List[float]
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
    
    better = 'splits1' if ks1 >= ks2 else 'splits2'
    return ks1, ks2, better
