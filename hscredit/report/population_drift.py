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
        target='fpd15',
        date='apply_month',
        output='客群偏移监控报告.xlsx',
    )
"""

from hscredit.utils.arguments import analysis_inputs

import numpy as np
import pandas as pd
import warnings
from typing import List, Optional, Tuple

from ..excel import ExcelWriter, dataframe2excel
from ..exceptions import ValidationError
from ..utils.parallel import ParallelWorkload, parallel_execute
from ..core.metrics.monitoring import MonitoringBaseline
from ..core.metrics import compute_bin_stats
from .result import ReportGenerationError, ReportResult


def _population_feature_task(task):
    """计算单个特征的 PSI、分布和可选坏样本率表。"""
    expected, actual, feature, target_col, max_n_bins, window, baseline = task
    expected_values = expected[feature]
    actual_values = actual[feature]
    result = baseline.evaluate(actual_values)
    psi_value = result["PSI"]
    distribution = _feature_distribution_compare(expected_values, actual_values, feature, max_n_bins, baseline=baseline)
    badrate = None
    if target_col and target_col in expected.columns and target_col in actual.columns:
        badrate = _badrate_compare(expected, actual, feature, target_col, max_n_bins, baseline=baseline)
    return feature, window, psi_value, distribution, badrate


def _population_task_status(task):
    """并行计算保留字段与窗口的错误上下文，由主进程统一执行发布策略。"""
    try:
        return task[2], task[5], _population_feature_task(task), None
    except Exception as exc:
        return task[2], task[5], None, f"{type(exc).__name__}: {exc}"


@analysis_inputs("actual", roles=("date",))
def population_drift(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    features: List[str],
    target: Optional[str] = None,
    date: Optional[str] = None,
    score_col: Optional[str] = None,
    max_n_bins: int = 10,
    output: str = "客群偏移监控报告.xlsx",
    *,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
    mode: str = "best_effort",
    missing_date: str = "error",
    return_result: bool = False,
):
    """生成客群偏移综合监控结果（Excel）.

    报告包含以下Sheet:
    - **总览**: 各特征PSI汇总及稳定性等级
    - **特征分布对比**: 各特征在基准/实际数据集的分箱分布对比
    - **逾期率对比** (可选): 各特征分箱下的逾期率差异
    - **评分分布** (可选): 模型评分在两个数据集的分布对比

    :param expected: 基准数据集（如训练集）
    :param actual: 实际/监控数据集（如生产数据）
    :param features: 监控特征列表
    :param target: 目标变量列名（如有，则生成逾期率对比Sheet）
    :param date: 时间列名（如有，用于按时间拆分实际数据）
    :param score_col: 评分列名（如有，则生成评分分布Sheet）
    :param max_n_bins: 分箱数
    :param output: 输出文件路径
    :param mode: strict 拒绝缺失字段或失败章节；best_effort 披露状态并保留成功结果
    :param missing_date: 日期缺失策略：error 拒绝、drop 显式排除、separate 独立统计
    :param return_result: True 返回包含状态、样本口径和结果表的 ReportResult
    :return: 默认输出文件路径；return_result=True 时为结构化结果

    **参考样例**

    >>> from hscredit.report import population_drift
    >>> # 基础：仅特征分布 PSI 对比
    >>> population_drift(train_df, prod_df, features=['score', 'age', 'income'])
    >>>
    >>> # 含逾期率与评分分布对比
    >>> population_drift(
    ...     train_df, prod_df,
    ...     features=['score', 'age'],
    ...     target='FPD',
    ...     score_col='model_score',
    ...     output='客群偏移监控报告.xlsx',
    ... )
    """

    result = ReportResult(mode=mode, metadata={"报告类型": "客群偏移监控"})
    if missing_date not in {"error", "drop", "separate"}:
        raise ValidationError("missing_date 必须为 error、drop 或 separate")
    if not isinstance(expected, pd.DataFrame) or not isinstance(actual, pd.DataFrame):
        raise ValidationError("基准与实际数据必须为 DataFrame")
    if not expected.columns.is_unique or not actual.columns.is_unique:
        raise ValidationError("基准与实际数据列名不能重复")
    if isinstance(features, str) or not features or len(set(features)) != len(features):
        raise ValidationError("features 必须是非空且不重复的字段列表")

    def failure(section, reason):
        result.failure(section, reason)
        if mode == "strict":
            raise ReportGenerationError(f"{section}: {reason}", result)
        warnings.warn(f"{section}: {reason}", UserWarning, stacklevel=3)

    valid_features = []
    for feature in features:
        if feature not in expected.columns or feature not in actual.columns:
            failure(f"特征:{feature}", "基准或实际数据缺少请求字段")
        else:
            valid_features.append(feature)
    for name, column in (("逾期率对比", target), ("评分分布对比", score_col)):
        if column is not None and (column not in expected.columns or column not in actual.columns):
            failure(name, f"基准或实际数据缺少请求字段: {column}")
            if name == "逾期率对比":
                target = None
            else:
                score_col = None

    windows = [(None, actual)]
    included_actual = actual
    missing_rows = 0
    if date is not None:
        if date not in actual.columns:
            raise ValidationError(f"实际数据集缺少日期列: {date}")
        parsed_dates = pd.to_datetime(actual[date], errors="coerce")
        invalid_mask = actual[date].notna() & parsed_dates.isna()
        if invalid_mask.any():
            raise ValidationError(f"日期列 {date} 包含无效日期")
        missing_rows = int(parsed_dates.isna().sum())
        if missing_rows and missing_date == "error":
            raise ValidationError(f"日期列 {date} 包含 {missing_rows} 行缺失日期，请显式选择 drop 或 separate")
        if not parsed_dates.notna().any() and missing_date != "separate":
            raise ValidationError(f"日期列 {date} 没有有效日期")
        periods = parsed_dates.dt.to_period("M")
        ordered_periods = sorted(periods.dropna().unique())
        windows = [(str(period), actual.loc[periods == period].copy()) for period in ordered_periods]
        if missing_rows and missing_date == "separate":
            windows.append(("缺失日期", actual.loc[parsed_dates.isna()].copy()))
        if missing_rows and missing_date == "drop":
            included_actual = actual.loc[parsed_dates.notna()]
            warnings.warn(
                f"按 missing_date='drop' 排除 {missing_rows} 行缺失日期，详见样本口径", UserWarning, stacklevel=2
            )

    sample_table = pd.DataFrame(
        [
            {"数据集": "基准", "输入行数": len(expected), "纳入行数": len(expected), "排除行数": 0, "缺失日期行数": 0},
            {
                "数据集": "实际",
                "输入行数": len(actual),
                "纳入行数": sum(len(frame) for _, frame in windows),
                "排除行数": missing_rows if date is not None and missing_date == "drop" else 0,
                "缺失日期行数": missing_rows,
            },
        ]
    )
    result.metadata.update({"日期缺失策略": missing_date, "请求字段": list(features)})
    result.success("样本口径", sample_table)
    baselines = {}
    for feature in valid_features:
        try:
            baselines[feature] = MonitoringBaseline(max_n_bins=max_n_bins).fit(expected[feature])
        except Exception as exc:
            failure(f"特征:{feature}", f"基准拟合失败: {type(exc).__name__}: {exc}")
    tasks = [
        (expected, window_data, feature, target, max_n_bins, window, baselines[feature])
        for feature in valid_features
        if feature in baselines
        for window, window_data in windows
    ]
    task_results = parallel_execute(
        _population_task_status,
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
            data_bytes=int(expected.memory_usage(deep=True).sum())
            + sum(int(window_data.memory_usage(deep=True).sum()) for _, window_data in windows),
            cost_per_item=10.0,
            capability="thread_safe",
            releases_gil=True,
            operation="客群漂移字段窗口计算",
        ),
    )

    feature_results = []
    for feature, window, output_data, error in task_results:
        section = f"特征:{feature}" if window is None else f"特征:{feature} / 窗口:{window}"
        result.plan(section, input_rows=next(len(frame) for label, frame in windows if label == window))
        if error is not None:
            failure(section, error)
        else:
            feature_results.append(output_data)
            result.success(section)
    score_detail = None
    if score_col is not None:
        result.plan("评分分布对比", input_rows=len(included_actual))
        result.run(
            "评分分布对比",
            lambda: _feature_distribution_compare(expected[score_col], included_actual[score_col], score_col, 20),
        )
        if result.sections["评分分布对比"].status == "失败":
            warnings.warn(f"评分分布对比: {result.sections['评分分布对比'].reason}", UserWarning, stacklevel=2)
        else:
            score_detail = result["评分分布对比"]
    result.ensure_publishable()

    writer = ExcelWriter()

    # ---------- Sheet 1: PSI 总览 ----------
    psi_rows = []
    for feat, window, psi_val, _, _ in feature_results:
        row = {
            "特征名": feat,
            "PSI": round(psi_val, 4),
            "稳定性": _psi_rating(psi_val),
        }
        if date is not None:
            row["时间窗口"] = window
            row = {"特征名": row["特征名"], "时间窗口": row["时间窗口"], "PSI": row["PSI"], "稳定性": row["稳定性"]}
        psi_rows.append(row)
    psi_df = pd.DataFrame(psi_rows)
    if not psi_df.empty and date is None:
        psi_df = psi_df.sort_values("PSI", ascending=False).reset_index(drop=True)
    dataframe2excel(psi_df, writer, sheet_name="PSI总览")
    result["PSI总览"] = psi_df

    # ---------- Sheet 2: 特征分布对比 ----------
    dist_rows = []
    for _, window, _, detail, _ in feature_results:
        if date is not None and not detail.empty:
            detail = detail.copy()
            detail.insert(1, "时间窗口", window)
        dist_rows.append(detail)
    if dist_rows:
        dist_df = pd.concat(dist_rows, ignore_index=True)
        dataframe2excel(dist_df, writer, sheet_name="特征分布对比")
        result["特征分布对比"] = dist_df

    # ---------- Sheet 3: 逾期率对比 (可选) ----------
    if target and target in expected.columns and target in actual.columns:
        br_rows = []
        for _, window, _, _, br_detail in feature_results:
            if br_detail is not None:
                if date is not None and not br_detail.empty:
                    br_detail = br_detail.copy()
                    br_detail.insert(1, "时间窗口", window)
                br_rows.append(br_detail)
        if br_rows:
            br_df = pd.concat(br_rows, ignore_index=True)
            dataframe2excel(br_df, writer, sheet_name="逾期率对比")
            result.success("逾期率对比", br_df)

    # ---------- Sheet 4: 评分分布 (可选) ----------
    if score_detail is not None:
        dataframe2excel(score_detail, writer, sheet_name="评分分布对比")

    dataframe2excel(sample_table, writer, sheet_name="样本口径")
    dataframe2excel(result.status_table(), writer, sheet_name="执行状态")
    writer.save(output)
    result.artifacts.append({"路径": str(output), "完整": result.complete})
    return result if return_result else output


# ============================================================
# 内部工具函数
# ============================================================

_PSI_EPSILON = 1e-6


def _build_numeric_bin_edges(
    expected: pd.Series,
    actual: pd.Series,
    max_n_bins: int,
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
        bin_count = max(int(max_n_bins), 1)
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


def _calc_psi(expected: pd.Series, actual: pd.Series, max_n_bins: int = 10) -> float:
    """计算 PSI."""
    return MonitoringBaseline(max_n_bins=max_n_bins).fit(expected).evaluate(actual)["PSI"]


def _psi_rating(psi: float) -> str:
    if not np.isfinite(psi):
        return "数据不足或不可计算"
    if psi < 0.1:
        return "稳定"
    elif psi < 0.25:
        return "轻微漂移"
    else:
        return "显著漂移"


def _feature_distribution_compare(
    expected: pd.Series, actual: pd.Series, feat_name: str, max_n_bins: int, baseline=None
) -> pd.DataFrame:
    """对比单特征在两个数据集的分箱分布."""
    baseline = baseline or MonitoringBaseline(max_n_bins=max_n_bins).fit(expected)
    result = baseline.evaluate(actual)
    table = result["分箱明细"]
    exp_pct = table["期望占比"].to_numpy()
    act_pct = table["实际占比"].to_numpy()
    labels = table["分箱"].tolist()

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
    max_n_bins: int,
    baseline=None,
) -> pd.DataFrame:
    """对比单特征分箱下的逾期率差异."""
    baseline = baseline or MonitoringBaseline(max_n_bins=max_n_bins).fit(expected[feat])

    def rates(frame):
        valid = frame[target_col].notna()
        if not valid.any():
            return pd.Series(dtype=float)
        codes = baseline.transform(frame.loc[valid, feat])
        table = compute_bin_stats(codes, frame.loc[valid, target_col].to_numpy(), round_digits=False)
        return table.set_index("分箱")["坏样本率"]

    exp_br, act_br = rates(expected), rates(actual)
    codes = exp_br.index.union(act_br.index)
    exp_br, act_br = exp_br.reindex(codes), act_br.reindex(codes)

    df = pd.DataFrame(
        {
            "特征名": feat,
            "分箱": [baseline._label(int(x)) for x in codes],
            "基准逾期率": np.round(exp_br.values, 4),
            "实际逾期率": np.round(act_br.values, 4),
            "逾期率偏移": np.round(act_br.values - exp_br.values, 4),
        }
    )
    return df
