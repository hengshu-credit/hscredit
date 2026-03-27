# -*- coding: utf-8 -*-
"""策略分析图表.

提供策略人员需要的跨时间/客群/交叉特征有效性和偏移分析图，包括：
- 特征随时间趋势图 (feature_trend_by_time)
- 多特征偏移瀑布图 (feature_drift_comparison)
- 特征在不同客群下有效性对比 (feature_effectiveness_by_segment)
- 两特征交叉分析热力图 (feature_cross_heatmap)
- 多期客群偏移监控大图 (population_drift_monitor)
- 分客群评分效果对比图 (segment_scorecard_comparison)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils import DEFAULT_COLORS, get_or_create_ax, save_figure, setup_axis_style


def _quick_psi(base: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """快速计算 PSI（内部辅助函数）."""
    base = base[~np.isnan(base)]
    target = target[~np.isnan(target)]
    if len(base) == 0 or len(target) == 0:
        return 0.0
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(base, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    base_counts = np.histogram(base, bins=bin_edges)[0]
    target_counts = np.histogram(target, bins=bin_edges)[0]
    eps = 1e-8
    base_pct = base_counts / base_counts.sum() + eps
    target_pct = target_counts / target_counts.sum() + eps
    psi = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
    return float(psi)


def feature_trend_by_time(
    df: pd.DataFrame,
    feature: str,
    date_col: str,
    target: Optional[str] = None,
    stat: str = 'mean',
    freq: str = 'M',
    ax=None,
    figsize=(12, 5),
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """特征随时间的统计趋势图，检测特征偏移.

    :param df: 数据集
    :param feature: 特征列名
    :param date_col: 日期列名
    :param target: 目标变量（stat='badrate'时必需）
    :param stat: 'mean'/'median'/'psi'/'badrate'
    :param freq: 'M'月/'Q'季/'W'周
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = feature_trend_by_time(df, 'age', 'apply_date', stat='mean')
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['_period'] = df[date_col].dt.to_period(freq).astype(str)
    periods = sorted(df['_period'].unique())
    base_vals = df[df['_period'] == periods[0]][feature].dropna().values if periods else np.array([])

    values = []
    for p in periods:
        sub = df[df['_period'] == p]
        col = sub[feature].dropna()
        if stat == 'mean':
            values.append(float(col.mean()) if len(col) > 0 else np.nan)
        elif stat == 'median':
            values.append(float(col.median()) if len(col) > 0 else np.nan)
        elif stat == 'badrate':
            if target is None:
                raise ValueError("stat='badrate' 时需传入 target")
            values.append(float(sub[target].mean()) if len(sub) > 0 else np.nan)
        elif stat == 'psi':
            values.append(_quick_psi(base_vals, col.values))
        else:
            values.append(float(col.mean()) if len(col) > 0 else np.nan)

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    x = np.arange(len(periods))
    clean = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    if clean:
        xi, vi = zip(*clean)
        ax.plot(xi, vi, color=DEFAULT_COLORS[0], marker='o', linewidth=2, markersize=5)
        ax.fill_between(xi, vi, alpha=0.1, color=DEFAULT_COLORS[0])

    if stat == 'badrate':
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if stat == 'psi':
        ax.axhline(0.10, color='orange', linestyle='--', lw=0.8, alpha=0.7, label='PSI=0.10')
        ax.axhline(0.25, color='red', linestyle='--', lw=0.8, alpha=0.7, label='PSI=0.25')
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=30, ha='right', fontsize=9)
    stat_label = {'mean': '均值', 'median': '中位数', 'badrate': '坏率', 'psi': 'PSI'}.get(stat, stat)
    ax.set_ylabel(stat_label, fontsize=11)
    ax.set_title(title or f'{feature} {stat_label}趋势', fontsize=12, fontweight='bold')
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def feature_drift_comparison(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[str],
    top_n: int = 20,
    ax=None,
    figsize=(12, 8),
    title: str = '特征分布偏移（PSI）',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """多特征偏移瀑布图：颜色标注偏移等级（绿/黄/红）.

    :param df_base: 基准数据集（训练集）
    :param df_target: 目标数据集（测试集/OOT）
    :param features: 待分析特征列表
    :param top_n: 按PSI降序显示前N个
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = feature_drift_comparison(df_train, df_oot, model_features)
    """
    psi_vals = {}
    for feat in features:
        if feat not in df_base.columns or feat not in df_target.columns:
            continue
        try:
            psi_vals[feat] = _quick_psi(
                df_base[feat].dropna().values,
                df_target[feat].dropna().values,
            )
        except Exception:
            psi_vals[feat] = 0.0

    if not psi_vals:
        fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
        ax.text(0.5, 0.5, '无可用特征', ha='center', va='center', transform=ax.transAxes)
        return fig

    psi_series = pd.Series(psi_vals).sort_values(ascending=False).head(top_n)
    feats = psi_series.index.tolist()
    vals = psi_series.values
    bar_colors = ['#4CAF50' if v < 0.10 else '#FF9800' if v < 0.25 else '#F44336' for v in vals]

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    y_pos = np.arange(len(feats))
    bars = ax.barh(y_pos, vals, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.10, color='orange', linestyle='--', lw=1, alpha=0.7)
    ax.axvline(0.25, color='red', linestyle='--', lw=1, alpha=0.7)

    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='#4CAF50', label='稳定（PSI<0.10）'),
        mpatches.Patch(color='#FF9800', label='略变（0.10-0.25）'),
        mpatches.Patch(color='#F44336', label='不稳定（>0.25）'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    ax.set_xlabel('PSI值', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def feature_effectiveness_by_segment(
    df: pd.DataFrame,
    feature: str,
    target: str,
    segment_col: str,
    metric: str = 'iv',
    ax=None,
    figsize=(10, 6),
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """特征在不同客群下的有效性对比柱状图.

    :param df: 数据集
    :param feature: 特征列名
    :param target: 目标变量列名
    :param segment_col: 客群列名
    :param metric: 'iv' / 'ks' / 'auc'
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = feature_effectiveness_by_segment(df, 'age', 'fpd30', 'channel')
    """
    segments = sorted(df[segment_col].dropna().unique())
    values = []
    for seg in segments:
        sub = df[df[segment_col] == seg].dropna(subset=[feature, target])
        y = sub[target].values
        x = sub[feature].values
        if len(y) < 10:
            values.append(0.0)
            continue
        try:
            if metric == 'iv':
                from ..metrics.feature import iv as _iv
                values.append(_iv(y, x))
            elif metric == 'ks':
                from ..metrics.classification import ks as _ks
                values.append(_ks(y, x))
            elif metric == 'auc':
                from ..metrics.classification import auc as _auc
                values.append(_auc(y, x))
            else:
                values.append(0.0)
        except Exception:
            values.append(0.0)

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    x_pos = np.arange(len(segments))
    bars = ax.bar(x_pos, values, color=DEFAULT_COLORS[0], alpha=0.8, edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in segments], rotation=30, ha='right', fontsize=9)
    vmax = max(values) if values else 1.0
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title or f'{feature} {metric.upper()}（各{segment_col}客群）',
                 fontsize=12, fontweight='bold')
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def feature_cross_heatmap(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    target: str,
    stat: str = 'badrate',
    n_bins: int = 5,
    ax=None,
    figsize=(10, 8),
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """两特征交叉分析热力图：行=feature_x，列=feature_y，格=坏率/样本数/LIFT.

    :param df: 数据集
    :param feature_x: 行特征
    :param feature_y: 列特征
    :param target: 目标变量
    :param stat: 'badrate'/'count'/'lift'
    :param n_bins: 数值型特征分箱数
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = feature_cross_heatmap(df, 'age', 'income', 'fpd30')
    """
    import matplotlib.colors as mcolors
    df = df[[feature_x, feature_y, target]].dropna().copy()
    overall_br = df[target].mean()

    def _bin(series, nbins):
        if pd.api.types.is_numeric_dtype(series):
            try:
                return pd.qcut(series, q=nbins, duplicates='drop').astype(str)
            except Exception:
                return pd.cut(series, bins=nbins, duplicates='drop').astype(str)
        return series.astype(str)

    df['_bx'] = _bin(df[feature_x], n_bins)
    df['_by'] = _bin(df[feature_y], n_bins)

    if stat == 'count':
        matrix = df.groupby(['_bx', '_by'])[target].count().unstack().fillna(0)
        fmt, cmap, cbar_label = '{:.0f}', 'Blues', '样本数'
    elif stat == 'lift':
        br_mat = df.groupby(['_bx', '_by'])[target].mean().unstack().fillna(overall_br)
        matrix = br_mat / overall_br if overall_br > 0 else br_mat
        fmt, cmap, cbar_label = '{:.2f}', 'RdYlGn_r', 'LIFT值'
    else:
        matrix = df.groupby(['_bx', '_by'])[target].mean().unstack().fillna(overall_br)
        fmt, cmap, cbar_label = '{:.2%}', 'RdYlGn_r', '坏率'

    data = matrix.values
    row_labels = matrix.index.tolist()
    col_labels = matrix.columns.tolist()

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels([str(r) for r in row_labels], fontsize=8)
    ax.set_xlabel(feature_y, fontsize=11)
    ax.set_ylabel(feature_x, fontsize=11)
    vmax = data.max()
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = data[i, j]
            color = 'white' if v > vmax * 0.6 else 'black'
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7, color=color)
    ax.set_title(title or f'{feature_x} × {feature_y} 交叉（{cbar_label}）',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def population_drift_monitor(
    df_list: List[pd.DataFrame],
    labels: List[str],
    features: List[str],
    target: Optional[str] = None,
    top_n_drift: int = 5,
    figsize=(14, 10),
    title: str = '客群偏移监控',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """多期客群偏移监控大图（PSI热力图 + 趋势折线）.

    :param df_list: 多期数据集列表（第一期为基准）
    :param labels: 各期标签
    :param features: 特征列表
    :param target: 目标变量（可选）
    :param top_n_drift: 偏移最大的Top N特征
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = population_drift_monitor([df1,df2,df3], ['Q1','Q2','Q3'], feats)
    """
    import matplotlib.colors as mcolors
    base = df_list[0]
    n_periods = len(df_list)

    psi_matrix = pd.DataFrame(0.0, index=features, columns=labels)
    for df_i, lbl in zip(df_list, labels):
        for feat in features:
            if feat in base.columns and feat in df_i.columns:
                psi_matrix.loc[feat, lbl] = _quick_psi(
                    base[feat].dropna().values, df_i[feat].dropna().values)

    top_feats = psi_matrix.iloc[:, -1].sort_values(ascending=False).head(top_n_drift).index.tolist()

    fig, (ax_heat, ax_trend) = plt.subplots(2, 1, figsize=figsize,
                                             gridspec_kw={'height_ratios': [1, 1.5]})
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'psi', [(0, '#4CAF50'), (0.2, '#FF9800'), (1, '#F44336')], N=256)
    data = psi_matrix.values.astype(float)
    im = ax_heat.imshow(data, cmap=cmap, vmin=0, vmax=0.5, aspect='auto')
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01, label='PSI')
    ax_heat.set_xticks(np.arange(n_periods))
    ax_heat.set_yticks(np.arange(len(features)))
    ax_heat.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax_heat.set_yticklabels(features, fontsize=8)
    for i in range(len(features)):
        for j in range(n_periods):
            v = data[i, j]
            c = 'white' if v > 0.2 else 'black'
            ax_heat.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=c)
    ax_heat.set_title('PSI热力图', fontsize=12, fontweight='bold')

    colors = DEFAULT_COLORS + ['#9C27B0', '#00BCD4', '#795548']
    x = np.arange(n_periods)
    for ci, feat in enumerate(top_feats):
        means = [float(df_i[feat].mean()) if feat in df_i.columns else np.nan for df_i in df_list]
        base_mean = means[0] if means[0] and not np.isnan(means[0]) else 1.0
        norm = [m / base_mean if base_mean else m for m in means]
        ax_trend.plot(x, norm, marker='o', lw=2,
                      color=colors[ci % len(colors)], label=feat, markersize=5)
    ax_trend.axhline(1.0, color='gray', linestyle='--', lw=0.8, alpha=0.6)
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax_trend.set_ylabel('相对基准期均值', fontsize=10)
    ax_trend.set_title(f'偏移Top{top_n_drift}特征均值趋势', fontsize=12, fontweight='bold')
    ax_trend.legend(fontsize=8)
    setup_axis_style(ax_trend, hide_top_right=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def segment_scorecard_comparison(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    segment_col: str,
    metrics: List[str] = None,
    ax=None,
    figsize=(14, 6),
    title: str = '分客群评分效果对比',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """按客群分组的评分指标对比柱状图.

    :param df: 数据集
    :param score_col: 评分列名
    :param target: 目标变量列名
    :param segment_col: 客群列名
    :param metrics: 展示指标，默认 ['KS','AUC','LIFT@10%']
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = segment_scorecard_comparison(df, 'score', 'fpd30', 'channel')
    """
    if metrics is None:
        metrics = ['KS', 'AUC', 'LIFT@10%']
    segments = sorted(df[segment_col].dropna().unique())
    results = {m: [] for m in metrics}
    for seg in segments:
        sub = df[df[segment_col] == seg].dropna(subset=[score_col, target])
        y, s = sub[target].values, sub[score_col].values
        if len(y) < 10 or y.sum() == 0:
            for m in metrics:
                results[m].append(0.0)
            continue
        for m in metrics:
            try:
                ml = m.lower()
                if ml == 'ks':
                    from ..metrics.classification import ks as _ks
                    results[m].append(_ks(y, s))
                elif ml == 'auc':
                    from ..metrics.classification import auc as _auc
                    results[m].append(_auc(y, s))
                elif 'lift' in ml:
                    ratio = float(ml.split('@')[1].replace('%', '')) / 100 if '@' in ml else 0.10
                    from ..metrics.finance import lift_at
                    results[m].append(lift_at(y, s, ratios=ratio))
                else:
                    results[m].append(0.0)
            except Exception:
                results[m].append(0.0)

    n_metrics = len(metrics)
    x = np.arange(len(segments))
    width = 0.8 / n_metrics
    colors = DEFAULT_COLORS + ['#9C27B0', '#00BCD4']
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    for mi, m in enumerate(metrics):
        offset = (mi - n_metrics / 2 + 0.5) * width
        vals = results[m]
        vmax = max(vals) if vals else 1.0
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=colors[mi % len(colors)], label=m, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in segments], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('指标值', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


__all__ = [
    'feature_trend_by_time',
    'feature_drift_comparison',
    'feature_effectiveness_by_segment',
    'feature_cross_heatmap',
    'population_drift_monitor',
    'segment_scorecard_comparison',
]
