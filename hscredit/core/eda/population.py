"""客群分析与偏移监控模块.

提供客群画像、客群偏移分析、多期监控报告等功能，
面向策略分析人员的全链路客群稳定性监控需求。

主要函数:
- population_profile: 客群画像（特征统计 + 坏率，支持分组）
- population_shift_analysis: 两个数据集的客群偏移分析（PSI/均值变化）
- population_monitoring_report: 多期客群监控 Excel 报告
- segment_drift_analysis: 分客群、分时间的三维偏移矩阵
- feature_cross_segment_effectiveness: 特征在不同客群下的有效性矩阵
"""

import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ...utils.parallel import parallel_execute
from .utils import _eda_workload, validate_dataframe, psi_rating, iv_rating


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _quick_psi(base: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """两个分布之间的快速 PSI 计算."""
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)
    base = base[~np.isnan(base)]
    target = target[~np.isnan(target)]
    if len(base) == 0 or len(target) == 0:
        return np.nan
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(base, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    eps = 1e-8
    base_counts = np.histogram(base, bins=bin_edges)[0].astype(float)
    tgt_counts = np.histogram(target, bins=bin_edges)[0].astype(float)
    base_total = base_counts.sum()
    target_total = tgt_counts.sum()
    if base_total == 0 or target_total == 0:
        return np.nan
    base_pct = base_counts / base_total + eps
    tgt_pct = tgt_counts / target_total + eps
    return float(np.sum((tgt_pct - base_pct) * np.log(tgt_pct / base_pct)))


def _safe_badrate(y: pd.Series) -> Optional[float]:
    """安全计算坏率，目标为 0/1 时有效."""
    y = y.dropna()
    if len(y) == 0:
        return np.nan
    u = y.unique()
    if set(u).issubset({0, 1, 0.0, 1.0, True, False}):
        return float(y.mean())
    return np.nan


def _population_profile_worker(task):
    """计算一个分组内单个特征的画像行。"""
    feature_series, target_series, feature, group_value, percentiles, percentile_labels = task
    valid = pd.to_numeric(feature_series, errors="coerce").dropna()
    row: Dict[str, Any] = {
        "特征": feature,
        "分组": group_value,
        "样本数": len(feature_series),
        "缺失率(%)": round(feature_series.isna().mean() * 100, 2),
        "均值": round(float(valid.mean()), 4) if len(valid) > 0 else np.nan,
        "标准差": round(float(valid.std()), 4) if len(valid) > 1 else np.nan,
    }
    values = [
        round(float(np.percentile(valid, percentile * 100)), 4) if len(valid) > 0 else np.nan
        for percentile in percentiles
    ]
    for label, value in zip(percentile_labels, values):
        row[label] = value
    if target_series is not None:
        bad_rate = _safe_badrate(target_series)
        row["坏率(%)"] = round(bad_rate * 100, 4) if not np.isnan(bad_rate) else np.nan
    return row


def _population_shift_worker(task):
    """计算单个特征的两客群偏移行。"""
    (
        base_series,
        target_series,
        feature,
        psi_n_bins,
        warning_threshold,
        alert_threshold,
        base_bad_rate,
        target_bad_rate,
    ) = task
    base_col = pd.to_numeric(base_series, errors="coerce")
    target_col = pd.to_numeric(target_series, errors="coerce")
    psi_value = _quick_psi(base_col.dropna().values, target_col.dropna().values, psi_n_bins)
    mean_base = float(base_col.mean()) if base_col.notna().sum() > 0 else np.nan
    mean_target = float(target_col.mean()) if target_col.notna().sum() > 0 else np.nan
    mean_change = (
        round((mean_target - mean_base) / abs(mean_base) * 100, 2)
        if not np.isnan(mean_base) and not np.isnan(mean_target) and mean_base != 0
        else np.nan
    )
    if np.isnan(psi_value):
        level, suggestion = "未知", "数据不足，无法评估"
    elif psi_value < warning_threshold:
        level, suggestion = "稳定", "无需关注"
    elif psi_value < alert_threshold:
        level, suggestion = "轻微偏移", "建议关注，监控趋势"
    else:
        level, suggestion = "显著偏移", "强烈建议排查原因，可能影响模型效果"
    missing_base = round(base_series.isna().mean() * 100, 2)
    missing_target = round(target_series.isna().mean() * 100, 2)
    row: Dict[str, Any] = {
        "特征名": feature,
        "基准样本数": len(base_series),
        "目标样本数": len(target_series),
        "PSI": round(psi_value, 4) if not np.isnan(psi_value) else np.nan,
        "偏移等级": level,
        "基准均值": round(mean_base, 4) if not np.isnan(mean_base) else np.nan,
        "目标均值": round(mean_target, 4) if not np.isnan(mean_target) else np.nan,
        "均值变化(%)": mean_change,
        "基准缺失率(%)": missing_base,
        "目标缺失率(%)": missing_target,
        "缺失率变化(%)": round(missing_target - missing_base, 2),
        "建议": suggestion,
    }
    if base_bad_rate is not None:
        row["基准坏率(%)"] = round(base_bad_rate * 100, 4) if not np.isnan(base_bad_rate) else np.nan
        row["目标坏率(%)"] = round(target_bad_rate * 100, 4) if not np.isnan(target_bad_rate) else np.nan
    return row


def _monitoring_psi_worker(task):
    """计算监控期和特征的快速 PSI。"""
    base_series, compare_series, feature, label, n_bins = task
    value = _quick_psi(
        pd.to_numeric(base_series, errors="coerce").dropna().values,
        pd.to_numeric(compare_series, errors="coerce").dropna().values,
        n_bins,
    )
    return feature, label, round(value, 4) if not np.isnan(value) else np.nan


def _monitoring_distribution_worker(task):
    """生成单个 Top 偏移特征的多期分布明细。"""
    feature, base, comparisons, labels, n_bins = task
    base_col = pd.to_numeric(base.get(feature), errors="coerce").dropna()
    if len(base_col) == 0:
        return feature, pd.DataFrame()
    edges = np.unique(np.percentile(base_col, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 2:
        return feature, pd.DataFrame()
    rows = []
    for label, frame in [("基准", base)] + list(zip(labels, comparisons)):
        if feature not in frame.columns:
            continue
        counts = np.histogram(pd.to_numeric(frame[feature], errors="coerce").dropna().values, bins=edges)[0]
        total = counts.sum()
        for index, count in enumerate(counts):
            rows.append(
                {
                    "特征名": feature,
                    "期次": label,
                    "分箱": f"bin_{index + 1}",
                    "样本数": int(count),
                    "占比(%)": round(count / total * 100, 2) if total > 0 else 0,
                }
            )
    return feature, pd.DataFrame(rows)


def _segment_drift_worker(task):
    """计算一个客群、时期和特征的漂移行。"""
    base_frame, target_frame, feature, segment, period, base_period, target, n_bins = task
    psi_value = _quick_psi(
        pd.to_numeric(base_frame[feature], errors="coerce").dropna().values,
        pd.to_numeric(target_frame[feature], errors="coerce").dropna().values,
        n_bins,
    )
    row: Dict[str, Any] = {
        "特征名": feature,
        "客群": segment,
        "时间": period,
        "基准期": base_period,
        "PSI": round(psi_value, 4) if not np.isnan(psi_value) else np.nan,
        "偏移等级": psi_rating(psi_value) if not np.isnan(psi_value) else "未知",
    }
    if target is not None and target in target_frame.columns:
        base_bad_rate = _safe_badrate(base_frame[target])
        target_bad_rate = _safe_badrate(target_frame[target])
        row["基准坏率(%)"] = round(base_bad_rate * 100, 2) if not np.isnan(base_bad_rate) else np.nan
        row["当期坏率(%)"] = round(target_bad_rate * 100, 2) if not np.isnan(target_bad_rate) else np.nan
    return row


def _cross_segment_metric_worker(task):
    """计算一个特征在一个客群下的 IV/KS/AUC。"""
    series, target, feature, label, metric, min_segment_size = task
    from ..metrics import auc, iv, ks

    try:
        numeric = pd.to_numeric(series, errors="coerce")
        mask = numeric.notna() & target.notna()
        if mask.sum() < min_segment_size or target[mask].nunique() < 2:
            value = np.nan
        elif metric == "iv":
            value = round(float(iv(target[mask], numeric[mask])), 4)
        elif metric == "ks":
            value = round(float(ks(target[mask], numeric[mask])), 4)
        else:
            value = round(float(auc(target[mask], numeric[mask])), 4)
    except Exception:
        value = np.nan
    return feature, label, value


# ---------------------------------------------------------------------------
# 1. population_profile
# ---------------------------------------------------------------------------

def population_profile(
    df: pd.DataFrame,
    features: List[str],
    segment_col: Optional[str] = None,
    date_col: Optional[str] = None,
    target: Optional[str] = None,
    freq: str = 'M',
    percentiles: List[float] = [0.25, 0.5, 0.75],
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> pd.DataFrame:
    """客群画像分析.

    计算各特征的均值、分位数、缺失率，以及坏率（若提供标签），
    支持按客群维度（segment_col）或时间（date_col）分组对比。

    :param df: 输入数据
    :param features: 需要分析的特征列表
    :param segment_col: 客群分组列（如渠道、产品线），为 None 时不分组
    :param date_col: 时间列，为 None 时不按时间分组；与 segment_col 互斥，优先 segment_col
    :param target: 目标变量列名（0/1），提供时输出各组坏率
    :param freq: 时间聚合粒度，'M'=月，'Q'=季度，'Y'=年，仅 date_col 非 None 时有效
    :param percentiles: 分位数列表，默认 [0.25, 0.5, 0.75]
    :return: 客群画像 DataFrame

    **参考样例**

    >>> profile = population_profile(df, features=['age', 'income'], target='fpd15')
    >>> profile_by_seg = population_profile(df, features=['age'], segment_col='channel', target='fpd15')
    """
    validate_dataframe(df, required_cols=features)
    df = df.copy()

    # 确定分组列
    group_col: Optional[str] = None
    if segment_col is not None and segment_col in df.columns:
        group_col = segment_col
    elif date_col is not None and date_col in df.columns:
        freq_map = {'M': 'M', 'Q': 'Q', 'Y': 'A'}
        pf = freq_map.get(freq, 'M')
        df['__period__'] = pd.to_datetime(df[date_col]).dt.to_period(pf).astype(str)
        group_col = '__period__'

    pct_labels = [f'p{int(p * 100)}' for p in percentiles]
    stat_cols = ['均值', '标准差', '缺失率(%)'] + pct_labels + ['样本数']
    if target is not None and target in df.columns:
        stat_cols.append('坏率(%)')

    if group_col is not None:
        groups = list(df.groupby(group_col, sort=True))
    else:
        groups = [('全量', df)]

    def iter_tasks():
        for group_value, group_frame in groups:
            target_series = group_frame[target] if target is not None and target in group_frame.columns else None
            for feature in features:
                if feature in group_frame.columns:
                    yield (
                        group_frame[feature],
                        target_series,
                        feature,
                        group_value,
                        tuple(percentiles),
                        tuple(pct_labels),
                    )

    actual_task_count = sum(1 for _, group_frame in groups for feature in features if feature in group_frame.columns)
    rows = parallel_execute(
        _population_profile_worker,
        iter_tasks(),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[
            f"{group_value}:{feature}"
            for group_value, group_frame in groups
            for feature in features
            if feature in group_frame.columns
        ],
        default_backend="threading",
        workload=_eda_workload(df, actual_task_count, operation="客群画像", cost_per_item=5.0),
    )

    result = pd.DataFrame(rows)
    # 清理临时列
    if '__period__' in df.columns and group_col == '__period__':
        result = result.rename(columns={'分组': date_col or '时间周期'})
    return result


# ---------------------------------------------------------------------------
# 2. population_shift_analysis
# ---------------------------------------------------------------------------

def population_shift_analysis(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str],
    target: Optional[str] = None,
    psi_n_bins: int = 10,
    psi_threshold_warn: float = 0.1,
    psi_threshold_alert: float = 0.25,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> pd.DataFrame:
    """客群偏移分析.

    计算基准数据集与目标数据集之间各特征的 PSI、均值变化、坏率变化，
    输出偏移摘要表，标注偏移等级和建议。

    :param df_base: 基准数据集（如训练集/历史月份）
    :param df_target: 目标数据集（如生产数据/近期月份）
    :param features: 需要分析的特征列表
    :param target: 目标变量列名（0/1），提供时输出坏率变化
    :param psi_n_bins: PSI 分箱数，默认 10
    :param psi_threshold_warn: PSI 警告阈值，默认 0.1（黄色）
    :param psi_threshold_alert: PSI 告警阈值，默认 0.25（红色）
    :return: 偏移摘要 DataFrame，含 特征名/PSI/均值变化/偏移等级/建议

    **参考样例**

    >>> result = population_shift_analysis(train_df, prod_df, features=['age', 'income'])
    >>> print(result[['特征名', 'PSI', '偏移等级', '建议']])
    """
    validate_dataframe(df_base, required_cols=features)
    validate_dataframe(df_target, required_cols=features)

    valid_features = [feature for feature in features if feature in df_base.columns and feature in df_target.columns]
    base_bad_rate = _safe_badrate(df_base.get(target)) if target is not None else None  # type: ignore[arg-type]
    target_bad_rate = _safe_badrate(df_target.get(target)) if target is not None else None  # type: ignore[arg-type]
    rows = parallel_execute(
        _population_shift_worker,
        (
            (
                df_base[feature],
                df_target[feature],
                feature,
                psi_n_bins,
                psi_threshold_warn,
                psi_threshold_alert,
                base_bad_rate,
                target_bad_rate,
            )
            for feature in valid_features
        ),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=valid_features,
        default_backend="threading",
        workload=_eda_workload(
            df_base.loc[:, valid_features],
            len(valid_features),
            operation="客群偏移分析",
            cost_per_item=6.0,
            additional_data=(df_target.loc[:, valid_features],),
        ),
    )

    result = pd.DataFrame(rows)
    if not result.empty and 'PSI' in result.columns:
        result = result.sort_values('PSI', ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 3. population_monitoring_report
# ---------------------------------------------------------------------------

def population_monitoring_report(
    df_base: pd.DataFrame,
    df_compare_list: List[pd.DataFrame],
    compare_labels: List[str],
    features: List[str],
    target: Optional[str] = None,
    psi_n_bins: int = 10,
    top_drift_n: int = 10,
    output_path: str = 'population_monitor.xlsx',
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> str:
    """多期客群监控 Excel 报告.

    生成包含以下 Sheet 的 Excel 报告：
    - **总览**：各期相对基准的 PSI 汇总热力表，含稳定性等级
    - **趋势**：各期样本量和坏率趋势（若提供 target）
    - **偏移Top{top_drift_n}**：PSI 均值最高的特征详细对比分布

    :param df_base: 基准数据集
    :param df_compare_list: 各期对比数据集列表
    :param compare_labels: 各期标签（与 df_compare_list 一一对应）
    :param features: 监控特征列表
    :param target: 目标变量列名
    :param psi_n_bins: PSI 计算分箱数
    :param top_drift_n: 输出偏移最大的 N 个特征
    :param output_path: 输出 Excel 路径
    :return: 输出文件路径

    **参考样例**

    >>> path = population_monitoring_report(
    ...     df_base=train_df,
    ...     df_compare_list=[prod_2024q1, prod_2024q2],
    ...     compare_labels=['2024Q1', '2024Q2'],
    ...     features=['age', 'income', 'credit_score'],
    ...     target='fpd15',
    ...     output_path='monitor.xlsx',
    ... )
    """
    from ...excel import ExcelWriter, dataframe2excel

    assert len(df_compare_list) == len(compare_labels), \
        "df_compare_list 与 compare_labels 长度必须一致"

    writer = ExcelWriter()

    # ======================================================================
    # Sheet 1 - PSI 总览矩阵
    # ======================================================================
    psi_matrix_rows: Dict[str, Dict[str, Any]] = {f: {} for f in features}
    valid_pairs = []
    for label, compare_frame in zip(compare_labels, df_compare_list):
        for feature in features:
            psi_matrix_rows[feature][label] = np.nan
            if feature in df_base.columns and feature in compare_frame.columns:
                valid_pairs.append((label, compare_frame, feature))
    psi_results = parallel_execute(
        _monitoring_psi_worker,
        (
            (df_base[feature], compare_frame[feature], feature, label, psi_n_bins)
            for label, compare_frame, feature in valid_pairs
        ),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[f"{label}:{feature}" for label, _, feature in valid_pairs],
        default_backend="threading",
        workload=_eda_workload(
            df_base,
            len(valid_pairs),
            operation="客群监控PSI矩阵",
            cost_per_item=6.0,
            additional_data=tuple(df_compare_list),
        ),
    )
    for feature, label, value in psi_results:
        psi_matrix_rows[feature][label] = value

    psi_df = pd.DataFrame(psi_matrix_rows).T.reset_index().rename(columns={'index': '特征名'})
    # 添加均值 PSI 和等级
    numeric_cols = compare_labels
    psi_df['平均PSI'] = psi_df[numeric_cols].mean(axis=1).round(4)
    psi_df['稳定性'] = psi_df['平均PSI'].apply(psi_rating)
    psi_df = psi_df.sort_values('平均PSI', ascending=False).reset_index(drop=True)

    ws = writer.get_sheet_by_name('PSI总览')
    end_row, _ = writer.insert_value2sheet(ws, (2, 2), value='客群监控 - PSI总览', style='header_middle', end_space=(2, 30))
    end_row, _ = dataframe2excel(psi_df, writer, sheet_name=ws, start_row=end_row + 1,
                                  title='各期特征PSI（相对基准）')

    # ======================================================================
    # Sheet 2 - 趋势：样本量 + 坏率
    # ======================================================================
    trend_rows = []
    base_bad = _safe_badrate(df_base[target]) if target and target in df_base.columns else np.nan
    trend_rows.append({
        '期次': '基准',
        '样本数': len(df_base),
        '坏率(%)': round(base_bad * 100, 4) if not np.isnan(base_bad) else np.nan,
    })
    for label, df_cmp in zip(compare_labels, df_compare_list):
        br = _safe_badrate(df_cmp[target]) if target and target in df_cmp.columns else np.nan
        trend_rows.append({
            '期次': label,
            '样本数': len(df_cmp),
            '坏率(%)': round(br * 100, 4) if not np.isnan(br) else np.nan,
        })
    trend_df = pd.DataFrame(trend_rows)

    ws2 = writer.get_sheet_by_name('样本趋势')
    end_row2, _ = writer.insert_value2sheet(ws2, (2, 2), value='各期样本量与坏率趋势', style='header_middle', end_space=(2, 20))
    end_row2, _ = dataframe2excel(trend_df, writer, sheet_name=ws2, start_row=end_row2 + 1,
                                   percent_cols=['坏率(%)'])

    # ======================================================================
    # Sheet 3 - 偏移 Top N 详细分布对比
    # ======================================================================
    top_features = psi_df['特征名'].head(top_drift_n).tolist()
    ws3 = writer.get_sheet_by_name(f'偏移Top{top_drift_n}')
    cur_row = 2
    end_row3, _ = writer.insert_value2sheet(ws3, (cur_row, 2),
                                             value=f'偏移最大 Top{top_drift_n} 特征分布对比',
                                             style='header_middle', end_space=(cur_row, 30))
    cur_row = end_row3 + 1

    distribution_results = parallel_execute(
        _monitoring_distribution_worker,
        (
            (feature, df_base, tuple(df_compare_list), tuple(compare_labels), psi_n_bins)
            for feature in top_features
        ),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=top_features,
        default_backend="threading",
        workload=_eda_workload(
            df_base,
            len(top_features),
            operation="客群监控Top特征分布",
            cost_per_item=4.0,
            additional_data=tuple(df_compare_list),
        ),
    )
    for feat, dist_df in distribution_results:
        if not dist_df.empty:
            end_row3, _ = dataframe2excel(
                dist_df, writer, sheet_name=ws3,
                title=f'{feat} 分布对比', start_row=cur_row,
                percent_cols=['占比(%)'],
            )
            cur_row = end_row3 + 2

    writer.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. segment_drift_analysis
# ---------------------------------------------------------------------------

def segment_drift_analysis(
    df: pd.DataFrame,
    date_col: str,
    segment_col: str,
    features: List[str],
    target: Optional[str] = None,
    base_period: Optional[str] = None,
    freq: str = 'M',
    psi_n_bins: int = 10,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> pd.DataFrame:
    """分客群、分时间的特征偏移三维矩阵.

    计算每个 (客群, 时间) 组合相对于基准期（或第一期）的特征 PSI，
    输出长格式三维结果表，列为 [特征名, 客群, 时间, PSI, 偏移等级]。

    :param df: 输入数据（须含 date_col 和 segment_col）
    :param date_col: 日期列名
    :param segment_col: 客群分组列名
    :param features: 分析特征列表
    :param target: 目标变量（可选，用于输出各组坏率）
    :param base_period: 基准期字符串（如 '2024-01'），None 时取最早一期
    :param freq: 时间聚合频率，'M'=月，'Q'=季度，'Y'=年
    :param psi_n_bins: PSI 计算分箱数
    :return: 长格式 DataFrame，含 特征名/客群/时间/PSI/偏移等级

    **参考样例**

    >>> result = segment_drift_analysis(df, date_col='apply_month',
    ...     segment_col='channel', features=['age', 'income'], base_period='2024-01')
    >>> print(result.pivot_table(index=['客群', '时间'], columns='特征名', values='PSI'))
    """
    validate_dataframe(df, required_cols=[date_col, segment_col] + features)

    df = df.copy()
    freq_map = {'M': 'M', 'Q': 'Q', 'Y': 'A'}
    pf = freq_map.get(freq, 'M')
    df['__period__'] = pd.to_datetime(df[date_col]).dt.to_period(pf).astype(str)

    all_periods = sorted(df['__period__'].unique())
    if base_period is None:
        base_period = all_periods[0]

    segments = sorted(df[segment_col].dropna().unique())
    task_specs = []
    for seg in segments:
        seg_df = df[df[segment_col] == seg]
        base_seg = seg_df[seg_df['__period__'] == base_period]
        if len(base_seg) == 0:
            continue

        for period in all_periods:
            if period == base_period:
                continue
            period_seg = seg_df[seg_df['__period__'] == period]
            if len(period_seg) == 0:
                continue

            for feat in features:
                if feat in base_seg.columns:
                    task_specs.append((base_seg, period_seg, feat, seg, period, base_period, target, psi_n_bins))

    rows = parallel_execute(
        _segment_drift_worker,
        iter(task_specs),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[f"{segment}:{period}:{feature}" for _, _, feature, segment, period, *_ in task_specs],
        default_backend="threading",
        workload=_eda_workload(df, len(task_specs), operation="分客群分时间偏移", cost_per_item=6.0),
    )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['特征名', '客群', '时间']).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 5. feature_cross_segment_effectiveness
# ---------------------------------------------------------------------------

def feature_cross_segment_effectiveness(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    segment_col: str,
    metric: str = 'iv',
    n_bins: int = 10,
    min_segment_size: int = 50,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> pd.DataFrame:
    """特征在不同客群下的有效性矩阵.

    计算每个特征在每个客群下的 IV / KS / AUC，
    输出宽格式矩阵（行=特征，列=客群，格=指标值），
    用于快速发现特征在哪些客群有效、哪些失效。

    :param df: 输入数据
    :param features: 特征列表
    :param target: 目标变量列名（0/1）
    :param segment_col: 客群分组列名
    :param metric: 有效性指标，'iv' / 'ks' / 'auc'
    :param n_bins: 分箱数（iv/ks 计算使用）
    :param min_segment_size: 最小客群样本量，低于此值跳过
    :return: 宽格式有效性矩阵 DataFrame

    **参考样例**

    >>> matrix = feature_cross_segment_effectiveness(
    ...     df, features=['age', 'income'], target='fpd15',
    ...     segment_col='channel', metric='iv')
    >>> print(matrix)
    #        channel_A  channel_B  全量
    # age     0.15       0.08      0.12
    # income  0.22       0.19      0.20
    """
    validate_dataframe(df, required_cols=[target, segment_col] + features)
    metric = metric.lower()
    assert metric in ('iv', 'ks', 'auc'), "metric 须为 'iv' / 'ks' / 'auc'"

    segments = sorted(df[segment_col].dropna().unique())
    records: Dict[str, Dict[str, Any]] = {f: {} for f in features}
    task_specs = [(df[feature], df[target], feature, "全量", metric, min_segment_size) for feature in features]
    for seg in segments:
        seg_df = df[df[segment_col] == seg]
        if len(seg_df) < min_segment_size:
            continue
        task_specs.extend(
            (seg_df[feature], seg_df[target], feature, str(seg), metric, min_segment_size)
            for feature in features
        )

    metric_results = parallel_execute(
        _cross_segment_metric_worker,
        iter(task_specs),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[f"{label}:{feature}" for _, _, feature, label, *_ in task_specs],
        default_backend="loky" if metric == "iv" else "threading",
        workload=_eda_workload(
            df,
            len(task_specs),
            operation="特征跨客群有效性",
            cost_per_item=12.0 if metric == "iv" else 4.0,
            capability="process_safe" if metric == "iv" else "thread_safe",
        ),
    )
    for feature, label, value in metric_results:
        records[feature][label] = value

    result = pd.DataFrame(records).T
    result.index.name = '特征名'
    result = result.reset_index()
    # 按全量指标降序
    if '全量' in result.columns:
        result = result.sort_values('全量', ascending=False).reset_index(drop=True)
    return result
