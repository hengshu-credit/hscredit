# -*- coding: utf-8 -*-
"""变量分析图表.

提供模型报告中变量分析专用图表，包括：
- IV值排名图 (variable_iv_plot)
- WOE折线+坏率柱状双轴图 (variable_woe_trend_plot)
- 特征PSI热力图 (variable_psi_heatmap)
- 分类重要性分组图 (variable_importance_grouped_plot)
- 缺失率vs坏率散点图 (variable_missing_badrate_plot)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils import DEFAULT_COLORS, get_or_create_ax, save_figure, setup_axis_style


# IV评级参考线
_IV_THRESHOLDS = [
    (0.02, '弱（< 0.02）', '#E0E0E0'),
    (0.10, '一般（0.02-0.10）', '#FFF9C4'),
    (0.30, '较强（0.10-0.30）', '#C8E6C9'),
    (float('inf'), '强（≥ 0.30）', '#A5D6A7'),
]

_PSI_THRESHOLDS = [(0.1, '稳定', '#4CAF50'), (0.25, '略变', '#FF9800'), (float('inf'), '不稳定', '#F44336')]


def variable_iv_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: str = '特征IV值排名',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """特征IV值横向柱状图.

    按IV降序排列，标注IV阈值参考线（0.02 / 0.10 / 0.30）。

    :param df: 数据集
    :param features: 特征列表
    :param target: 目标变量列名
    :param top_n: 显示前N个特征，默认20
    :param ax: matplotlib Axes，None时自动创建
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径，None时不保存
    :return: matplotlib Figure

    Example:
        >>> fig = variable_iv_plot(df, features=df.columns.tolist(), target='fpd30')
    """
    from ..metrics.feature import iv

    iv_vals = {}
    y = df[target].values
    for feat in features:
        if feat == target:
            continue
        try:
            iv_vals[feat] = iv(y, df[feat].values)
        except Exception:
            iv_vals[feat] = 0.0

    iv_series = pd.Series(iv_vals).sort_values(ascending=False).head(top_n)
    feat_names = iv_series.index.tolist()
    iv_values = iv_series.values

    # 颜色：按IV强度着色
    bar_colors = []
    for v in iv_values:
        if v < 0.02:
            bar_colors.append('#BDBDBD')
        elif v < 0.10:
            bar_colors.append('#FDD835')
        elif v < 0.30:
            bar_colors.append('#66BB6A')
        else:
            bar_colors.append('#1B5E20')

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)

    y_pos = np.arange(len(feat_names))
    bars = ax.barh(y_pos, iv_values, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.invert_yaxis()

    # 参考线
    for thresh, label, _ in [(0.02, 'IV=0.02', '#BDBDBD'), (0.10, 'IV=0.10', '#FDD835'),
                              (0.30, 'IV=0.30', '#66BB6A')]:
        ax.axvline(thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(thresh + 0.002, -0.8, label, fontsize=8, color='gray', va='top')

    # 数值标签
    for bar, v in zip(bars, iv_values):
        ax.text(v + max(iv_values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=9)

    ax.set_xlabel('IV值', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def variable_woe_trend_plot(
    bin_table: pd.DataFrame,
    feature: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """WOE折线图 + 坏率柱状图（双轴）.

    用于模型报告变量分析，展示每个分箱的WOE值和坏样本率。

    :param bin_table: 分箱统计表，需含「分箱标签」/「WOE」/「坏样本率」列
        （兼容 OptimalBinning.get_bin_table() 输出）
    :param feature: 特征名，用于标题
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题，None时自动生成
    :param save: 保存路径
    :return: Figure

    Example:
        >>> binner = OptimalBinning().fit(df[['age']], df['fpd30'])
        >>> tbl = binner.get_bin_table()['age']
        >>> fig = variable_woe_trend_plot(tbl, feature='age')
    """
    # 兼容列名
    label_col = next((c for c in ['分箱标签', 'bin_label', 'Bin'] if c in bin_table.columns), None)
    woe_col = next((c for c in ['分档WOE值', 'WOE', 'woe'] if c in bin_table.columns), None)
    br_col = next((c for c in ['坏样本率', 'bad_rate', 'BadRate'] if c in bin_table.columns), None)
    cnt_col = next((c for c in ['样本总数', 'count', 'Count'] if c in bin_table.columns), None)

    if woe_col is None or br_col is None:
        raise ValueError("bin_table 必须含 WOE 和坏样本率列")

    tbl = bin_table.dropna(subset=[woe_col, br_col]).copy()
    if label_col:
        x_labels = tbl[label_col].astype(str).tolist()
    else:
        x_labels = [str(i) for i in range(len(tbl))]

    woe_vals = tbl[woe_col].values
    br_vals = tbl[br_col].values
    x = np.arange(len(x_labels))

    fig, ax1 = get_or_create_ax(figsize=figsize, ax=ax)
    ax2 = ax1.twinx()

    # 柱状图：坏样本率
    bar_width = 0.6
    bars = ax1.bar(x, br_vals, width=bar_width, color=DEFAULT_COLORS[0],
                   alpha=0.35, label='坏样本率', zorder=2)
    ax1.set_ylabel('坏样本率', color=DEFAULT_COLORS[0], fontsize=10)
    ax1.tick_params(axis='y', labelcolor=DEFAULT_COLORS[0])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))

    # 折线图：WOE
    ax2.plot(x, woe_vals, color=DEFAULT_COLORS[1], marker='o',
             linewidth=2, markersize=6, label='WOE', zorder=3)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('WOE', color=DEFAULT_COLORS[1], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=DEFAULT_COLORS[1])

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9)

    feat_str = feature or ''
    ax1.set_title(title or f'{feat_str} WOE & 坏率分布', fontsize=12, fontweight='bold')

    # 图例合并
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

    setup_axis_style(ax1, hide_top_right=False)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def variable_psi_heatmap(
    psi_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (14, 8),
    title: str = '特征PSI热力图',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """特征PSI矩阵热力图.

    颜色反映偏移程度：绿色（PSI<0.1稳定）/ 橙色（0.1-0.25略变）/ 红色（>0.25不稳定）。

    :param psi_matrix: PSI矩阵，行=特征，列=时间周期或数据集，格=PSI值
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> psi_mat = batch_psi_analysis(df_train, df_test, features)
        >>> fig = variable_psi_heatmap(psi_mat)
    """
    import matplotlib.colors as mcolors

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)

    data = psi_matrix.values.astype(float)
    n_rows, n_cols = data.shape

    # 三段色：绿→橙→红
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'psi_cmap',
        [(0.0, '#4CAF50'), (0.1 / 0.5, '#FF9800'), (1.0, '#F44336')],
        N=256,
    )
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=0.5, aspect='auto')

    # 色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('PSI值', fontsize=10)
    for thresh, label, _ in _PSI_THRESHOLDS:
        if thresh < float('inf'):
            cbar.ax.axhline(thresh, color='white', linewidth=1.5)

    # 刻度
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(psi_matrix.columns.tolist(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(psi_matrix.index.tolist(), fontsize=9)

    # 格内数值
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            color = 'white' if v > 0.2 else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def variable_importance_grouped_plot(
    importance_df: pd.DataFrame,
    feature_col: str = 'feature',
    value_col: str = 'importance',
    group_col: Optional[str] = 'category',
    top_n: int = 30,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: str = '特征重要性（分类）',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """按特征类别分组的重要性横向柱状图.

    :param importance_df: 特征重要性 DataFrame，需含 feature_col 和 value_col 列
    :param feature_col: 特征名列名，默认 'feature'
    :param value_col: 重要性值列名，默认 'importance'
    :param group_col: 分组列名，默认 'category'；None 时不分组
    :param top_n: 显示前N个，默认30
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> imp_df = pd.DataFrame({'feature': feats, 'importance': imps, 'category': cats})
        >>> fig = variable_importance_grouped_plot(imp_df)
    """
    df = importance_df.copy()
    if value_col not in df.columns or feature_col not in df.columns:
        raise ValueError(f"importance_df 必须含 '{feature_col}' 和 '{value_col}' 列")

    df = df.sort_values(value_col, ascending=False).head(top_n)
    feats = df[feature_col].tolist()
    vals = df[value_col].values

    # 分组颜色
    if group_col and group_col in df.columns:
        groups = df[group_col].tolist()
        unique_groups = list(dict.fromkeys(groups))
        palette = plt.cm.get_cmap('tab10', len(unique_groups))
        group_colors = {g: palette(i) for i, g in enumerate(unique_groups)}
        bar_colors = [group_colors[g] for g in groups]
        legend_patches = [mpatches.Patch(color=group_colors[g], label=g) for g in unique_groups]
    else:
        bar_colors = [DEFAULT_COLORS[0]] * len(feats)
        legend_patches = []

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    y_pos = np.arange(len(feats))
    bars = ax.barh(y_pos, vals, color=bar_colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9)
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

    ax.set_xlabel('重要性', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def variable_missing_badrate_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
    title: str = '缺失率 vs 坏账率（缺失样本）',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """缺失率 vs 坏账率散点图.

    横轴=特征缺失率，纵轴=缺失样本坏账率。
    用于评估缺失值是否携带信息（坏率显著异于总体则说明缺失本身有预测价值）。

    :param df: 数据集
    :param features: 特征列表
    :param target: 目标变量列名
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = variable_missing_badrate_plot(df, features=feats, target='fpd30')
    """
    y = df[target].values
    overall_br = y.mean()
    rows = []
    for feat in features:
        if feat == target:
            continue
        miss_mask = df[feat].isna()
        miss_rate = miss_mask.mean()
        if miss_rate == 0:
            continue
        miss_br = y[miss_mask].mean() if miss_mask.sum() > 0 else np.nan
        rows.append({'特征': feat, '缺失率': miss_rate, '缺失坏率': miss_br})

    if not rows:
        fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
        ax.text(0.5, 0.5, '无缺失特征', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    stat_df = pd.DataFrame(rows)
    miss_rates = stat_df['缺失率'].values
    miss_brs = stat_df['缺失坏率'].values
    feat_names = stat_df['特征'].values

    # 颜色：缺失坏率与总体差异大的用红色标记
    diff = np.abs(miss_brs - overall_br)
    max_diff = diff.max() if diff.max() > 0 else 1.0
    colors = plt.cm.RdYlGn_r(diff / max_diff)

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    sc = ax.scatter(miss_rates, miss_brs, c=diff, cmap='RdYlGn_r',
                    s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.axhline(overall_br, color='gray', linestyle='--', linewidth=1,
               label=f'总体坏率 {overall_br:.2%}')

    # 标注差异大的特征
    top_idx = np.argsort(diff)[::-1][:10]
    for i in top_idx:
        ax.annotate(feat_names[i],
                    (miss_rates[i], miss_brs[i]),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=7, alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('|缺失坏率 - 总体坏率|', fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    ax.set_xlabel('缺失率', fontsize=11)
    ax.set_ylabel('缺失样本坏账率', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


__all__ = [
    'variable_iv_plot',
    'variable_woe_trend_plot',
    'variable_psi_heatmap',
    'variable_importance_grouped_plot',
    'variable_missing_badrate_plot',
]
