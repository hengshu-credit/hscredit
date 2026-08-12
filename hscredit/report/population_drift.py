# -*- coding: utf-8 -*-
"""
客群偏移监控模块.

整合客群稳定性分析（PSI）、特征分布漂移、逾期率漂移等，
生成综合 Excel 监控报告，适用于模型上线后的定期监控场景。

用法::

    from hscredit.report import population_drift

    population_drift(
        expected=train_df,
        actual=prod_df,
        features=['age', 'income', 'credit_score'],
        target_col='fpd15',
        date_col='apply_month',
        output='客群偏移监控报告.xlsx',
    )
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from ..excel import ExcelWriter, dataframe2excel
from ..exceptions import ValidationError
from ..utils.parallel import ParallelWorkload, parallel_execute


def _population_feature_task(task):
    """计算单个特征的 PSI、分布和可选坏样本率表。"""
    expected, actual, feature, target_col, n_bins, window = task
    expected_values = expected[feature].dropna()
    actual_values = actual[feature].dropna()
    psi_value = _calc_psi(expected_values, actual_values, n_bins)
    distribution = _feature_distribution_compare(expected_values, actual_values, feature, n_bins)
    badrate = None
    if target_col and target_col in expected.columns and target_col in actual.columns:
        badrate = _badrate_compare(expected, actual, feature, target_col, n_bins)
    return feature, window, psi_value, distribution, badrate


def population_drift(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    features: List[str],
    target_col: Optional[str] = None,
    date_col: Optional[str] = None,
    score_col: Optional[str] = None,
    n_bins: int = 10,
    output: str = "客群偏移监控报告.xlsx",
    *,
    target: Optional[str] = None,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> str:
    """生成客群偏移综合监控结果（Excel）.

    报告包含以下Sheet:
    - **总览**: 各特征PSI汇总及稳定性等级
    - **特征分布对比**: 各特征在基准/实际数据集的分箱分布对比
    - **逾期率对比** (可选): 各特征分箱下的逾期率差异
    - **评分分布** (可选): 模型评分在两个数据集的分布对比

    :param expected: 基准数据集（如训练集）
    :param actual: 实际/监控数据集（如生产数据）
    :param features: 监控特征列表
    :param target_col: 目标变量列名（如有，则生成逾期率对比Sheet）
    :param date_col: 时间列名（如有，用于按时间拆分实际数据）
    :param score_col: 评分列名（如有，则生成评分分布Sheet）
    :param n_bins: 分箱数
    :param output: 输出文件路径
    :param target: target_col 的别名
    :return: 输出文件路径（即 ``output``）

    **参考样例**

    >>> from hscredit.report import population_drift
    >>> # 基础：仅特征分布 PSI 对比
    >>> population_drift(train_df, prod_df, features=['score', 'age', 'income'])
    >>>
    >>> # 含逾期率与评分分布对比
    >>> population_drift(
    ...     train_df, prod_df,
    ...     features=['score', 'age'],
    ...     target_col='FPD',
    ...     score_col='model_score',
    ...     output='客群偏移监控报告.xlsx',
    ... )
    """
    target_col = target or target_col

    valid_features = [feature for feature in features if feature in expected.columns and feature in actual.columns]
    windows = [(None, actual)]
    if date_col is not None:
        if date_col not in actual.columns:
            raise ValidationError(f"实际数据集缺少日期列: {date_col}")
        parsed_dates = pd.to_datetime(actual[date_col], errors="coerce")
        invalid_mask = actual[date_col].notna() & parsed_dates.isna()
        if invalid_mask.any():
            raise ValidationError(f"日期列 {date_col} 包含无效日期")
        if not parsed_dates.notna().any():
            raise ValidationError(f"日期列 {date_col} 没有有效日期")
        periods = parsed_dates.dt.to_period("M")
        ordered_periods = sorted(periods.dropna().unique())
        windows = [(str(period), actual.loc[periods == period].copy()) for period in ordered_periods]

    tasks = [(expected, window_data, feature, target_col, n_bins, window) for feature in valid_features for window, window_data in windows]
    feature_results = parallel_execute(
        _population_feature_task,
        tasks,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[f"{task[2]}:{task[5]}" if task[5] is not None else task[2] for task in tasks],
        default_backend="threading",
        workload=ParallelWorkload(
            task_count=len(tasks),
            rows=len(expected) + sum(len(window_data) for _, window_data in windows),
            columns=len(valid_features),
            data_bytes=int(expected.memory_usage(deep=True).sum()) + sum(int(window_data.memory_usage(deep=True).sum()) for _, window_data in windows),
            cost_per_item=10.0,
            capability="thread_safe",
            releases_gil=True,
            operation="客群漂移字段窗口计算",
        ),
    )

    writer = ExcelWriter()

    # ---------- Sheet 1: PSI 总览 ----------
    psi_rows = []
    for feat, window, psi_val, _, _ in feature_results:
        row = {
            "特征名": feat,
            "PSI": round(psi_val, 4),
            "稳定性": _psi_rating(psi_val),
        }
        if date_col is not None:
            row["时间窗口"] = window
            row = {"特征名": row["特征名"], "时间窗口": row["时间窗口"], "PSI": row["PSI"], "稳定性": row["稳定性"]}
        psi_rows.append(row)
    psi_df = pd.DataFrame(psi_rows)
    if not psi_df.empty and date_col is None:
        psi_df = psi_df.sort_values("PSI", ascending=False).reset_index(drop=True)
    dataframe2excel(psi_df, writer, sheet_name="PSI总览")

    # ---------- Sheet 2: 特征分布对比 ----------
    dist_rows = []
    for _, window, _, detail, _ in feature_results:
        if date_col is not None and not detail.empty:
            detail = detail.copy()
            detail.insert(1, "时间窗口", window)
        dist_rows.append(detail)
    if dist_rows:
        dist_df = pd.concat(dist_rows, ignore_index=True)
        dataframe2excel(dist_df, writer, sheet_name="特征分布对比")

    # ---------- Sheet 3: 逾期率对比 (可选) ----------
    if target_col and target_col in expected.columns and target_col in actual.columns:
        br_rows = []
        for _, window, _, _, br_detail in feature_results:
            if br_detail is not None:
                if date_col is not None and not br_detail.empty:
                    br_detail = br_detail.copy()
                    br_detail.insert(1, "时间窗口", window)
                br_rows.append(br_detail)
        if br_rows:
            br_df = pd.concat(br_rows, ignore_index=True)
            dataframe2excel(br_df, writer, sheet_name="逾期率对比")

    # ---------- Sheet 4: 评分分布 (可选) ----------
    if score_col and score_col in expected.columns and score_col in actual.columns:
        score_detail = _feature_distribution_compare(expected[score_col].dropna(), actual[score_col].dropna(), score_col, 20)
        dataframe2excel(score_detail, writer, sheet_name="评分分布对比")

    writer.save(output)
    return output


# ============================================================
# 内部工具函数
# ============================================================

_PSI_EPSILON = 1e-6


def _build_numeric_bin_edges(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int,
) -> Optional[np.ndarray]:
    """为两个数值序列构建统一的分箱边界。"""
    try:
        lower = float(min(expected.min(), actual.min()))
        upper = float(max(expected.max(), actual.max()))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(lower) or not np.isfinite(upper):
        return None

    try:
        bin_count = max(int(n_bins), 1)
    except (TypeError, ValueError):
        bin_count = 10

    if upper <= lower:
        return np.array([-np.inf, np.inf], dtype=float)

    breakpoints = np.linspace(lower, upper, bin_count + 1, dtype=float)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    return breakpoints


def _compute_distribution_percentages(
    expected: pd.Series,
    actual: pd.Series,
    breakpoints: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据统一分箱边界计算基准/实际占比。"""
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    return exp_pct, act_pct


def _format_bin_labels(breakpoints: np.ndarray) -> List[str]:
    """生成可读的区间标签。"""
    labels = []
    for i in range(len(breakpoints) - 1):
        lo = f"{breakpoints[i]:.2f}" if np.isfinite(breakpoints[i]) else "-∞"
        hi = f"{breakpoints[i + 1]:.2f}" if np.isfinite(breakpoints[i + 1]) else "+∞"
        labels.append(f"[{lo}, {hi})")
    return labels


def _calc_psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """计算 PSI."""
    breakpoints = _build_numeric_bin_edges(expected, actual, n_bins)
    if breakpoints is None:
        return 0.0

    exp_pct, act_pct = _compute_distribution_percentages(expected, actual, breakpoints)

    # 避免 0 值
    exp_pct = np.where(exp_pct == 0, _PSI_EPSILON, exp_pct)
    act_pct = np.where(act_pct == 0, _PSI_EPSILON, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def _psi_rating(psi: float) -> str:
    if psi < 0.1:
        return "稳定"
    elif psi < 0.25:
        return "轻微漂移"
    else:
        return "显著漂移"


def _feature_distribution_compare(expected: pd.Series, actual: pd.Series, feat_name: str, n_bins: int) -> pd.DataFrame:
    """对比单特征在两个数据集的分箱分布."""
    breakpoints = _build_numeric_bin_edges(expected, actual, n_bins)
    if breakpoints is None:
        return pd.DataFrame()

    exp_pct, act_pct = _compute_distribution_percentages(expected, actual, breakpoints)
    labels = _format_bin_labels(breakpoints)

    df = pd.DataFrame(
        {
            "特征名": feat_name,
            "分箱": labels,
            "基准占比": np.round(exp_pct, 4),
            "实际占比": np.round(act_pct, 4),
            "偏移量": np.round(act_pct - exp_pct, 4),
        }
    )
    return df


def _badrate_compare(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    feat: str,
    target_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """对比单特征分箱下的逾期率差异."""
    breakpoints = _build_numeric_bin_edges(expected[feat], actual[feat], n_bins)
    if breakpoints is None:
        return pd.DataFrame()

    exp_cut = pd.cut(expected[feat], bins=breakpoints, right=False, include_lowest=True)
    act_cut = pd.cut(actual[feat], bins=breakpoints, right=False, include_lowest=True)

    exp_br = expected.groupby(exp_cut, observed=False)[target_col].mean()
    act_br = actual.groupby(act_cut, observed=False)[target_col].mean()

    df = pd.DataFrame(
        {
            "特征名": feat,
            "分箱": [str(x) for x in exp_br.index],
            "基准逾期率": np.round(exp_br.values, 4),
            "实际逾期率": np.round(act_br.values, 4),
            "逾期率偏移": np.round(act_br.values - exp_br.values, 4),
        }
    )
    return df
