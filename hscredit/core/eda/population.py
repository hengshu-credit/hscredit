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

from .utils import validate_dataframe, psi_rating, iv_rating


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
    base_pct = base_counts / base_counts.sum() + eps
    tgt_pct = tgt_counts / tgt_counts.sum() + eps
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

    Example:
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

    rows = []

    def _calc_stats(sub: pd.DataFrame, group_val: Any) -> None:
        for feat in features:
            if feat not in sub.columns:
                continue
            col = sub[feat]
            n = len(col)
            missing_rate = round(col.isna().mean() * 100, 2)
            col_num = pd.to_numeric(col, errors='coerce')
            valid = col_num.dropna()
            mean_val = round(float(valid.mean()), 4) if len(valid) > 0 else np.nan
            std_val = round(float(valid.std()), 4) if len(valid) > 1 else np.nan
            pct_vals = [round(float(np.percentile(valid, p * 100)), 4) if len(valid) > 0 else np.nan
                        for p in percentiles]
            row: Dict[str, Any] = {
                '特征': feat,
                '分组': group_val,
                '样本数': n,
                '缺失率(%)': missing_rate,
                '均值': mean_val,
                '标准差': std_val,
            }
            for label, val in zip(pct_labels, pct_vals):
                row[label] = val
            if target is not None and target in sub.columns:
                br = _safe_badrate(sub[target])
                row['坏率(%)'] = round(br * 100, 4) if not np.isnan(br) else np.nan
            rows.append(row)

    if group_col is not None:
        for gval, gdf in df.groupby(group_col, sort=True):
            _calc_stats(gdf, gval)
    else:
        _calc_stats(df, '全量')

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

    Example:
        >>> result = population_shift_analysis(train_df, prod_df, features=['age', 'income'])
        >>> print(result[['特征名', 'PSI', '偏移等级', '建议']])
    """
    validate_dataframe(df_base, required_cols=features)
    validate_dataframe(df_target, required_cols=features)

    rows = []
    for feat in features:
        if feat not in df_base.columns or feat not in df_target.columns:
            continue

        base_col = pd.to_numeric(df_base[feat], errors='coerce')
        tgt_col = pd.to_numeric(df_target[feat], errors='coerce')

        n_base = len(df_base)
        n_tgt = len(df_target)
        missing_base = round(df_base[feat].isna().mean() * 100, 2)
        missing_tgt = round(df_target[feat].isna().mean() * 100, 2)

        psi_val = _quick_psi(base_col.dropna().values, tgt_col.dropna().values, psi_n_bins)

        mean_base = float(base_col.mean()) if base_col.notna().sum() > 0 else np.nan
        mean_tgt = float(tgt_col.mean()) if tgt_col.notna().sum() > 0 else np.nan
        if not np.isnan(mean_base) and not np.isnan(mean_tgt) and mean_base != 0:
            mean_change_pct = round((mean_tgt - mean_base) / abs(mean_base) * 100, 2)
        else:
            mean_change_pct = np.nan

        # 偏移等级
        if np.isnan(psi_val):
            level = '未知'
            suggestion = '数据不足，无法评估'
        elif psi_val < psi_threshold_warn:
            level = '稳定'
            suggestion = '无需关注'
        elif psi_val < psi_threshold_alert:
            level = '轻微偏移'
            suggestion = '建议关注，监控趋势'
        else:
            level = '显著偏移'
            suggestion = '强烈建议排查原因，可能影响模型效果'

        row: Dict[str, Any] = {
            '特征名': feat,
            '基准样本数': n_base,
            '目标样本数': n_tgt,
            'PSI': round(psi_val, 4) if not np.isnan(psi_val) else np.nan,
            '偏移等级': level,
            '基准均值': round(mean_base, 4) if not np.isnan(mean_base) else np.nan,
            '目标均值': round(mean_tgt, 4) if not np.isnan(mean_tgt) else np.nan,
            '均值变化(%)': mean_change_pct,
            '基准缺失率(%)': missing_base,
            '目标缺失率(%)': missing_tgt,
            '缺失率变化(%)': round(missing_tgt - missing_base, 2),
            '建议': suggestion,
        }

        if target is not None:
            br_base = _safe_badrate(df_base.get(target))  # type: ignore[arg-type]
            br_tgt = _safe_badrate(df_target.get(target))  # type: ignore[arg-type]
            row['基准坏率(%)'] = round(br_base * 100, 4) if br_base is not None and not np.isnan(br_base) else np.nan
            row['目标坏率(%)'] = round(br_tgt * 100, 4) if br_tgt is not None and not np.isnan(br_tgt) else np.nan

        rows.append(row)

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

    Example:
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
    for label, df_cmp in zip(compare_labels, df_compare_list):
        for feat in features:
            if feat not in df_base.columns or feat not in df_cmp.columns:
                psi_matrix_rows[feat][label] = np.nan
                continue
            base_arr = pd.to_numeric(df_base[feat], errors='coerce').dropna().values
            cmp_arr = pd.to_numeric(df_cmp[feat], errors='coerce').dropna().values
            psi_val = _quick_psi(base_arr, cmp_arr, psi_n_bins)
            psi_matrix_rows[feat][label] = round(psi_val, 4) if not np.isnan(psi_val) else np.nan

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

    for feat in top_features:
        base_col = pd.to_numeric(df_base.get(feat), errors='coerce').dropna()  # type: ignore[arg-type]
        if len(base_col) == 0:
            continue
        edges = np.percentile(base_col, np.linspace(0, 100, psi_n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            continue

        dist_rows = []
        for label, df_cmp in [('基准', df_base)] + list(zip(compare_labels, df_compare_list)):  # type: ignore[list-item]
            if feat not in df_cmp.columns:
                continue
            arr = pd.to_numeric(df_cmp[feat], errors='coerce').dropna().values
            counts = np.histogram(arr, bins=edges)[0]
            total = counts.sum()
            for i, cnt in enumerate(counts):
                dist_rows.append({
                    '特征名': feat,
                    '期次': label,
                    '分箱': f'bin_{i + 1}',
                    '样本数': int(cnt),
                    '占比(%)': round(cnt / total * 100, 2) if total > 0 else 0,
                })

        dist_df = pd.DataFrame(dist_rows)
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

    Example:
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

    rows = []
    segments = sorted(df[segment_col].dropna().unique())

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
                if feat not in base_seg.columns:
                    continue
                base_arr = pd.to_numeric(base_seg[feat], errors='coerce').dropna().values
                tgt_arr = pd.to_numeric(period_seg[feat], errors='coerce').dropna().values
                psi_val = _quick_psi(base_arr, tgt_arr, psi_n_bins)

                row: Dict[str, Any] = {
                    '特征名': feat,
                    '客群': seg,
                    '时间': period,
                    '基准期': base_period,
                    'PSI': round(psi_val, 4) if not np.isnan(psi_val) else np.nan,
                    '偏移等级': psi_rating(psi_val) if not np.isnan(psi_val) else '未知',
                }
                if target is not None and target in period_seg.columns:
                    br_base = _safe_badrate(base_seg[target])
                    br_tgt = _safe_badrate(period_seg[target])
                    row['基准坏率(%)'] = round(br_base * 100, 2) if not np.isnan(br_base) else np.nan
                    row['当期坏率(%)'] = round(br_tgt * 100, 2) if not np.isnan(br_tgt) else np.nan
                rows.append(row)

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

    Example:
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

    from ..metrics import iv, ks, auc  # type: ignore[attr-defined]

    def _calc(col: pd.Series, y: pd.Series) -> float:
        try:
            col_num = pd.to_numeric(col, errors='coerce')
            mask = col_num.notna() & y.notna()
            if mask.sum() < min_segment_size or y[mask].nunique() < 2:
                return np.nan
            if metric == 'iv':
                return round(float(iv(y[mask], col_num[mask])), 4)
            elif metric == 'ks':
                return round(float(ks(y[mask], col_num[mask])), 4)
            else:
                return round(float(auc(y[mask], col_num[mask])), 4)
        except Exception:
            return np.nan

    segments = sorted(df[segment_col].dropna().unique())
    records: Dict[str, Dict[str, Any]] = {f: {} for f in features}

    # 全量
    for feat in features:
        records[feat]['全量'] = _calc(df[feat], df[target])

    # 各客群
    for seg in segments:
        seg_df = df[df[segment_col] == seg]
        if len(seg_df) < min_segment_size:
            continue
        for feat in features:
            records[feat][str(seg)] = _calc(seg_df[feat], seg_df[target])

    result = pd.DataFrame(records).T
    result.index.name = '特征名'
    result = result.reset_index()
    # 按全量指标降序
    if '全量' in result.columns:
        result = result.sort_values('全量', ascending=False).reset_index(drop=True)
    return result
