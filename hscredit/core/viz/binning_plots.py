# -*- coding: utf-8 -*-
"""
评分卡可视化函数.

提供常用的绘图功能，包括分箱图、KS/ROC曲线、分布图、PSI/CSI分析图等。

注：分箱统计计算已统一收口到hscredit.core.metrics.compute_bin_stats
"""

import logging
import re
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Union, Optional, List, Dict, Any

from .utils import (
    DEFAULT_COLORS, setup_axis_style, save_figure, 
    get_or_create_ax, format_bin_label
)

logger = logging.getLogger(__name__)

# 从统一metrics模块导入分箱统计计算
from ..metrics import compute_bin_stats
from ...exceptions import NotFittedError


def _is_feature_table(data):
    """判断是否为特征分箱统计表"""
    if not isinstance(data, pd.DataFrame):
        return False
    # 必须包含样本统计列
    stat_cols = ['好样本数', '坏样本数', '样本总数', '坏样本率']
    if not all(col in data.columns for col in stat_cols):
        return False
    # 分箱标识列：'分箱' 或 '分箱标签' 至少有一个
    return '分箱' in data.columns or '分箱标签' in data.columns


def _compute_bin_stats_from_raw_data(
    data: Union[pd.DataFrame, pd.Series],
    target: Union[str, pd.Series, np.ndarray],
    feature: Optional[str] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    min_bin_size: float = 0.01,
    rules: Optional[List] = None,
    **kwargs
) -> pd.DataFrame:
    """从原始数据计算分箱统计表
    
    此函数基于hscredit.core.binning.OptimalBinning进行分箱，
    使用hscredit.core.metrics.compute_bin_stats计算分箱统计。
    
    :param data: 特征数据（DataFrame 或 Series）
    :param target: 目标变量（列名或数据）
    :param feature: 特征列名（当 data 为 DataFrame 时需要）
    :param method: 分箱方法，可选：
        - 基础方法: 'uniform'(等宽), 'quantile'(等频), 'tree'(决策树), 'chi'(卡方)
        - 优化方法: 'best_ks'(最优KS), 'best_iv'(最优IV), 'mdlp'(信息论)
        - 高级方法: 'cart'(CART), 'monotonic'(单调性), 'genetic'(遗传算法),
                    'smooth'(平滑), 'kernel_density'(核密度)
        默认: 'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param rules: 自定义分箱边界（优先级高于method）
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 分箱统计表
    """
    # 处理输入数据
    if isinstance(data, pd.Series):
        X = data.copy()
        if feature is None:
            feature = data.name if data.name else 'feature'
    elif isinstance(data, pd.DataFrame):
        if feature is None:
            feature = data.columns[0]
        X = data[feature].copy()
    else:
        X = pd.Series(data)
        feature = 'feature'
    
    # 处理目标变量
    if isinstance(target, str):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            y = data[target].copy()
        else:
            raise ValueError(f"目标列 '{target}' 不在数据中")
    elif isinstance(target, (pd.Series, np.ndarray)):
        y = pd.Series(target)
    else:
        raise ValueError("target 必须是列名、Series 或数组")
    
    # 确保数据长度一致
    if len(X) != len(y):
        raise ValueError(f"特征数据长度 ({len(X)}) 与目标变量长度 ({len(y)}) 不一致")
    
    # 移除缺失值
    valid_mask = ~(pd.isna(X) | pd.isna(y))
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if len(X_valid) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")
    
    # 构建DataFrame
    df = pd.DataFrame({
        feature: X_valid,
        'target': y_valid
    })
    
    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning
    
    if rules is not None:
        # 使用自定义分箱规则
        binner = OptimalBinning(
            user_splits={feature: rules},
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            verbose=False,
            **kwargs
        )
    else:
        binner = OptimalBinning(
            method=method,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            verbose=False,
            **kwargs
        )
    
    binner.fit(df[[feature]], df['target'])
    
    # 应用分箱获取分箱索引
    bin_indices = binner.transform(df[[feature]], metric='indices').values.flatten()
    
    # 获取分箱标签
    bin_labels = None
    if feature in binner.bin_tables_:
        bin_table = binner.bin_tables_[feature]
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()
    
    # 使用统一的compute_bin_stats计算分箱统计
    stats_df = compute_bin_stats(bin_indices, y_valid.values, bin_labels=bin_labels, round_digits=False)
    
    # 构建结果DataFrame（兼容viz模块的期望格式）
    results = []
    for _, row in stats_df.iterrows():
        # 获取分箱标签
        if '分箱标签' in row and pd.notna(row['分箱标签']):
            label = row['分箱标签']
        else:
            label = f"Bin_{int(row['分箱'])}"
        
        results.append({
            '分箱': label,
            '分箱标签': label,
            '样本总数': int(row['样本总数']),
            '好样本数': int(row['好样本数']),
            '坏样本数': int(row['坏样本数']),
            '坏样本率': row['坏样本率'],
            '样本占比': row['样本占比'],
            '指标IV值': row.get('指标IV值', np.nan),
            '分档KS值': row.get('分档KS值', np.nan),
            'LIFT值': row.get('LIFT值', np.nan),
        })
    
    return pd.DataFrame(results)


def _is_missing_bin_label(label: Any) -> bool:
    """判断是否为缺失分箱标签。"""
    if pd.isna(label):
        return True
    text = str(label).strip().lower()
    return text in {'missing', 'nan', 'none', 'null', '缺失', '缺失值'}


def _is_special_bin_label(label: Any) -> bool:
    """判断是否为特殊值分箱标签。"""
    if pd.isna(label):
        return False
    text = str(label).strip().lower()
    return text in {'special', '特殊', '特殊值'}


def _is_total_bin_label(label: Any) -> bool:
    """判断是否为分箱表合计标签。"""
    if pd.isna(label):
        return False
    return str(label).strip() == '合计'


def _exclude_total_bin_rows(feature_table: pd.DataFrame) -> pd.DataFrame:
    """排除分箱表中的合计行。"""
    total_mask = pd.Series(False, index=feature_table.index)
    for label_col in ('分箱', '分箱标签'):
        if label_col in feature_table.columns:
            total_mask |= feature_table[label_col].apply(_is_total_bin_label)
    return feature_table.loc[~total_mask].copy()


def _is_interval_like_label(label: Any) -> bool:
    """判断分箱标签是否像数值区间。"""
    if pd.isna(label):
        return False
    text = str(label).strip()
    if _is_missing_bin_label(text):
        return False
    return bool(re.match(r'^[\(\[].*,.*[\)\]]$', text))


def _infer_numeric_feature_table(feature_table: pd.DataFrame) -> bool:
    """根据分箱标签粗略判断是否为数值型特征分箱表。"""
    label_col = '分箱标签' if '分箱标签' in feature_table.columns else '分箱'
    labels = feature_table[label_col].dropna() if label_col in feature_table.columns else pd.Series(dtype=object)
    labels = labels[~labels.apply(_is_missing_bin_label)]
    if labels.empty:
        return False
    interval_hits = labels.apply(_is_interval_like_label)
    return bool(interval_hits.any())


def _detect_bad_rate_trend(feature_table: pd.DataFrame) -> str:
    """判断坏样本率趋势，排除缺失值和特殊值分箱。"""
    if '坏样本率' not in feature_table.columns:
        return '未知'

    working = feature_table.copy()

    # 通过分箱索引排除缺失值(-1)和特殊值(-2)分箱
    if '分箱' in working.columns:
        bin_idx = pd.to_numeric(working['分箱'], errors='coerce')
        working = working[bin_idx.isna() | (bin_idx >= 0)]

    # 通过标签排除缺失值和特殊值分箱
    label_col = '分箱标签' if '分箱标签' in working.columns else '分箱'
    if label_col in working.columns:
        working = working[
            ~working[label_col].apply(_is_missing_bin_label)
            & ~working[label_col].apply(_is_special_bin_label)
        ]

    rates = pd.to_numeric(working['坏样本率'], errors='coerce').dropna().to_numpy()
    if len(rates) <= 1:
        return '未知'

    diffs = np.diff(rates)
    tol = 1e-6
    diffs = diffs[np.abs(diffs) > tol]
    if len(diffs) == 0:
        return '平稳'
    if np.all(diffs >= 0):
        return '上升'
    if np.all(diffs <= 0):
        return '下降'

    signs = np.sign(diffs)
    non_zero = signs[signs != 0]
    sign_changes = 0 if len(non_zero) <= 1 else int(np.sum(non_zero[1:] != non_zero[:-1]))
    if sign_changes == 1:
        return 'U型' if non_zero[0] < 0 < non_zero[-1] else '倒U型'
    return '波动'


def _build_bin_metric_summary(feature_table: pd.DataFrame, per_row: int = 2) -> str:
    """构建分箱图角标摘要。

    :param feature_table: 分箱统计表
    :param per_row: 每行展示的指标个数，默认 2（紧凑两列）；小图可用 1 使角标更窄
    """
    items = []

    if '指标IV值' in feature_table.columns:
        iv_values = pd.to_numeric(feature_table['指标IV值'], errors='coerce').dropna()
        if not iv_values.empty:
            items.append(f"IV {iv_values.iloc[-1]:.4f}")

    if _infer_numeric_feature_table(feature_table) and '分档KS值' in feature_table.columns:
        ks_values = pd.to_numeric(feature_table['分档KS值'], errors='coerce').dropna()
        if not ks_values.empty:
            items.append(f"KS {ks_values.max():.4f}")

    if 'LIFT值' in feature_table.columns:
        lift_values = pd.to_numeric(feature_table['LIFT值'], errors='coerce').dropna()
        if not lift_values.empty:
            items.append(f"LIFT {lift_values.min():.2f}~{lift_values.max():.2f}")

    trend = _detect_bad_rate_trend(feature_table)
    if trend != '未知':
        items.append(f"趋势 {trend}")

    if not items:
        return ''

    per_row = max(1, int(per_row))
    rows = []
    for i in range(0, len(items), per_row):
        rows.append('    '.join(items[i:i + per_row]))

    return '\n'.join(rows)


def _xtick_rotation_for_length(max_len: int, threshold: int = 5):
    """根据刻度文本最大长度决定横向分箱图 x 轴刻度的旋转角度与对齐方式.

    未超过 ``threshold`` 时保持垂直（90°）；超过后改为倾斜，且文本越长角度越小
    （越接近水平），下限 30°，避免过长文本纵向占用过多。

    :param max_len: 刻度文本的最大字符长度
    :param threshold: 触发倾斜的长度阈值
    :return: (rotation, horizontalalignment)
    """
    if max_len <= threshold:
        return 90, 'center'
    return max(30, 90 - (max_len - threshold) * 10), 'right'


def _auto_rotate_horizontal_xticks(fig, ax, threshold: int = 35) -> None:
    """横向分箱图 x 轴（样本数）刻度方向自适应.

    默认垂直展示，避免相邻数值刻度相互重叠；当刻度文本最大长度超过 ``threshold``
    时改为倾斜展示（角度由 :func:`_xtick_rotation_for_length` 动态决定）。

    :param fig: 刻度所在 Figure（需先绘制以读取自动生成的刻度文本）
    :param ax: 横向模式下承载样本数的 x 轴 Axes
    :param threshold: 触发倾斜的刻度文本长度阈值，未超过则保持垂直
    """
    fig.canvas.draw()
    lengths = [len(t.get_text()) for t in ax.get_xticklabels() if t.get_text()]
    if not lengths:
        return

    rotation, ha = _xtick_rotation_for_length(max(lengths), threshold)
    ax.tick_params(axis='x', labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment(ha)
        if ha == 'right':
            label.set_rotation_mode('anchor')


def bin_plot(
    data: Union[pd.DataFrame, pd.Series],
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    feature: Optional[str] = None,
    desc: str = "",
    figsize: tuple = (12, 7),
    colors: Optional[List[str]] = None,
    save: Optional[str] = None,
    anchor: float = 0.935,
    max_len: int = 35,
    fontdict: Optional[dict] = None,
    hatch: bool = True,
    ending: str = "分箱图",
    title: Optional[str] = None,
    n_bins: int = 10,
    method: str = 'quantile',
    rules: Optional[List] = None,
    show_data_points: bool = True,
    show_overall_bad_rate: bool = True,
    show_metric_summary: bool = True,
    show_rate_axis: bool = True,
    iv: bool = True,
    return_frame: bool = False,
    ax: Optional[Any] = None,
    orientation: str = 'horizontal',
    **kwargs
):
    """
    特征分箱可视化图.
    
    支持两种使用方式：
    
    **方式1：传入原始数据（toad 模式）**
    ```python
    # DataFrame + 列名
    bin_plot(df, x='feature_name', target='target')
    
    # Series + 目标数组
    bin_plot(df['feature'], target=df['target'])
    
    # 使用已创建的画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, col in enumerate(features):
        bin_plot(df[col], target=y, ax=axes[i], title=f'{col}分箱')
    ```
    
    **方式2：传入分箱统计表（scorecardpipeline 模式）**
    ```python
    # 传入已计算好的分箱统计表
    bin_plot(feature_table, desc="特征描述")
    ```

    :param data: 数据（DataFrame、Series 或分箱统计表）
    :param target: 目标变量（列名、Series 或数组）
    :param feature: 特征列名（当 data 为 DataFrame 且不明确时使用）
    :param desc: 特征中文描述
    :param figsize: 图像尺寸（创建新图时使用）
    :param colors: 配色方案
    :param save: 保存路径
    :param anchor: 图例位置
    :param max_len: 分箱标签最大长度
    :param fontdict: 字体样式
    :param hatch: 是否显示斜线
    :param ending: 标题后缀
    :param title: 完整标题（优先级高于 desc + ending）
    :param n_bins: 分箱数量（仅用于方式1）
    :param method: 分箱方法（仅用于方式1），可选 'quantile' 或 'uniform'
    :param rules: 自定义分箱边界（仅用于方式1）
    :param show_data_points: 是否显示数据点标记
    :param show_overall_bad_rate: 是否显示整体坏样本率参考线
    :param show_metric_summary: 是否显示左上角的指标角标（IV/KS/LIFT/趋势摘要），默认显示
    :param show_rate_axis: 是否显示坏样本率坐标轴刻度与标签（趋势线本身始终保留），默认显示
    :param iv: 是否显示 IV 值（暂不支持）
    :param return_frame: 是否返回分箱统计表
    :param ax: 可选的 matplotlib Axes 对象，用于在已有画布上绘图
    :param orientation: 图表方向，'horizontal'/'h'(横向，默认) 或 'vertical'/'v'(纵向)
    :param kwargs: 其他参数（兼容性）
    :return: matplotlib Figure 或 (Figure, DataFrame)，如果传入 ax 则返回 ax
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if fontdict is None:
        fontdict = {"color": "#000000"}

    # 判断输入类型并处理
    if _is_feature_table(data):
        # 方式2：传入的是分箱统计表
        feature_table = data.copy()
        # 兼容 feature_bin_stats 返回的格式（有 分箱标签 但无 分箱 列）
        if '分箱' not in feature_table.columns and '分箱标签' in feature_table.columns:
            feature_table['分箱'] = feature_table['分箱标签']
    else:
        # 方式1：传入的是原始数据，需要计算分箱统计
        if target is None:
            raise ValueError(
                "当传入原始数据时，必须提供 target 参数。\n"
                "用法示例:\n"
                "  bin_plot(df, x='feature', target='target')\n"
                "  bin_plot(df['feature'], target=df['target'])"
            )
        
        # 兼容 toad 的参数命名（x 参数）
        if 'x' in kwargs:
            feature = kwargs.pop('x')
        
        feature_table = _compute_bin_stats_from_raw_data(
            data=data,
            target=target,
            feature=feature,
            max_n_bins=n_bins,
            method=method,
            rules=rules,
        )

    # margins=True 生成的合计行不属于实际分箱，不参与绘图及指标计算
    feature_table = _exclude_total_bin_rows(feature_table)
    if feature_table.empty:
        raise ValueError("分箱表排除合计行后没有可绘制的分箱数据")

    # 处理分箱标签：优先显示具体分箱标签，而不是分箱索引
    plot_labels = None
    if '分箱标签' in feature_table.columns:
        candidate_labels = feature_table['分箱标签']
        if candidate_labels.notna().any():
            plot_labels = candidate_labels.astype(str)

    if plot_labels is None:
        plot_labels = feature_table['分箱'].astype(str)

    feature_table = feature_table.copy()
    feature_table['_plot_bin_label'] = plot_labels.apply(
        lambda x: format_bin_label(x, max_len)
    )

    # 判断方向
    orientation_key = orientation.lower()
    if orientation_key not in ['horizontal', 'h', '横向', 'vertical', 'v', '纵向']:
        raise ValueError("orientation 仅支持 'horizontal'/'h'/'横向' 或 'vertical'/'v'/'纵向'")
    is_horizontal = orientation_key in ['horizontal', 'h', '横向']

    # 统一排序：分离缺失值/特殊值分箱，数值型按区间下界升序，类别型保持原顺序
    label_col = '分箱标签' if '分箱标签' in feature_table.columns else '分箱'
    missing_mask = feature_table[label_col].apply(_is_missing_bin_label)
    special_mask = feature_table[label_col].apply(_is_special_bin_label)
    normal_rows = feature_table[~missing_mask & ~special_mask].copy()
    special_rows = feature_table[special_mask].copy()
    missing_rows = feature_table[missing_mask].copy()

    # 仅对数值型区间分箱排序，类别型分箱保持原有顺序
    is_numeric = _infer_numeric_feature_table(feature_table)
    if len(normal_rows) > 0 and is_numeric:
        def extract_lower_bound(bin_label):
            try:
                text = str(bin_label).strip()
                if text.startswith('(') or text.startswith('['):
                    left = text[1:].split(',')[0].strip()
                    if left in ('-inf', '-∞'):
                        return float('-inf')
                    return float(left)
            except Exception:
                pass
            return float('inf')

        normal_rows['_sort_key'] = normal_rows[label_col].apply(extract_lower_bound)
        normal_rows = normal_rows.sort_values('_sort_key').drop(columns=['_sort_key'])

    # 重新组合：普通分箱(升序) + 特殊值分箱 + 缺失值分箱(最后)
    feature_table = pd.concat([normal_rows, special_rows, missing_rows], ignore_index=True)

    # 保存升序排列的表用于 return_frame（不受横向反转影响）
    _sorted_table = feature_table.copy()

    if is_horizontal:
        # barh 第一行在底部、最后一行在顶部，反转使视觉从上到下为升序
        feature_table = feature_table.iloc[::-1].reset_index(drop=True)

    overall_bad_rate = float(feature_table['坏样本率'].mul(feature_table['样本总数']).sum() / feature_table['样本总数'].sum())
    axis_theme = colors[0]
    line_color = '#E85D4A'
    reference_color = '#4C8DFF'
    rate_fontdict = {
        'color': line_color,
        'fontsize': 10,
        'fontweight': 'semibold',
        'bbox': dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=line_color, linewidth=0.6, alpha=0.92)
    }

    # 获取或创建 Axes
    if ax is not None:
        ax1 = ax
        fig = ax.figure
        return_ax = True
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        return_ax = False

    if is_horizontal:
        # 横向柱状图（默认）—— 统一使用整数位置，与纵向模式保持一致
        y_pos = np.arange(len(feature_table))
        ax1.barh(y_pos, feature_table['好样本数'], color=colors[0], label='好样本',
                 hatch="/" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.barh(y_pos, feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1],
                 label='坏样本', hatch="\\" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feature_table['_plot_bin_label'])
        ax1.set_xlabel('样本数', color=axis_theme)

        ax2 = ax1.twiny()
        ax2.plot(feature_table['坏样本率'], y_pos, color=line_color, label='坏样本率', linestyle=(0, (4, 3)), linewidth=2.1,
                 marker='o' if show_data_points else None, markersize=5.5, markerfacecolor='white',
                 markeredgecolor=line_color, markeredgewidth=1.4)
        ax2.set_xlabel('坏样本率', color=axis_theme)
        ax2.set_xlim(left=0.)

        if show_overall_bad_rate:
            ax2.axvline(overall_bad_rate, color=reference_color, linestyle=(0, (2, 2)), linewidth=1.8, alpha=0.9,
                        label='整体坏样本率')

        x_right = max(ax2.get_xlim()[1], float(feature_table['坏样本率'].max()) * 1.15 if len(feature_table) > 0 else 0.1)
        ax2.set_xlim(right=x_right)
        x_offset = max((ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.012, 0.003)
        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.text(rate + x_offset, i, f'{rate:.2%}', va='center', ha='left', fontdict=rate_fontdict, clip_on=False)

        ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))
    else:
        # 纵向柱状图
        x_pos = np.arange(len(feature_table))
        width = 0.6

        ax1.bar(x_pos, feature_table['好样本数'], width, color=colors[0], label='好样本',
                hatch="/" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.bar(x_pos, feature_table['坏样本数'], width, bottom=feature_table['好样本数'], color=colors[1],
                label='坏样本', hatch="\\" if hatch else None, edgecolor='white' if hatch else None, alpha=0.92)
        ax1.set_ylabel('样本数', color=axis_theme)
        ax1.set_xticks(x_pos)
        # 分箱标签默认垂直展示，仅当标签非常长（阈值 18）时才按长度动态倾斜
        bin_labels = feature_table['_plot_bin_label'].tolist()
        label_rotation, label_ha = _xtick_rotation_for_length(
            max((len(str(l)) for l in bin_labels), default=0), threshold=18)
        ax1.set_xticklabels(bin_labels, rotation=label_rotation, ha=label_ha)
        if label_rotation != 90:
            for lbl in ax1.get_xticklabels():
                lbl.set_rotation_mode('anchor')

        ax2 = ax1.twinx()
        ax2.plot(x_pos, feature_table['坏样本率'], color=line_color, label='坏样本率', linestyle=(0, (4, 3)), linewidth=2.1,
                 marker='o' if show_data_points else None, markersize=5.5, markerfacecolor='white',
                 markeredgecolor=line_color, markeredgewidth=1.4)
        ax2.set_ylabel('坏样本率', color=axis_theme)
        ax2.set_ylim(bottom=0.)

        if show_overall_bad_rate:
            ax2.axhline(overall_bad_rate, color=reference_color, linestyle=(0, (2, 2)), linewidth=1.8, alpha=0.9,
                        label='整体坏样本率')

        y_top = max(float(feature_table['坏样本率'].max()) if len(feature_table) > 0 else 0.0, overall_bad_rate)
        ax2.set_ylim(top=max(ax2.get_ylim()[1], y_top * 1.18 if y_top > 0 else 0.1))
        y_offset = max(ax2.get_ylim()[1] * 0.015, 0.003)
        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.text(i, rate + y_offset, f'{rate:.2%}', ha='center', va='bottom', fontdict=rate_fontdict, clip_on=False)

        ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))

    setup_axis_style(ax1, [axis_theme], hide_top_right=False)
    setup_axis_style(ax2, [axis_theme], hide_top_right=False)
    ax1.tick_params(axis='both', colors=axis_theme)
    ax2.tick_params(axis='both', colors=axis_theme)
    ax1.grid(False)
    ax2.grid(False)

    if not show_rate_axis:
        # 隐藏坏样本率坐标轴刻度与标签（趋势线保留）：横向在顶部 x 轴，纵向在右侧 y 轴
        if is_horizontal:
            ax2.tick_params(axis='x', top=False, labeltop=False)
            ax2.set_xlabel('')
        else:
            ax2.tick_params(axis='y', right=False, labelright=False)
            ax2.set_ylabel('')

    # 横向模式：样本数 x 轴刻度默认垂直，文本过长时按最大长度动态倾斜
    if is_horizontal:
        _auto_rotate_horizontal_xticks(fig, ax1)

    _summary_source = feature_table.drop(columns=['_plot_bin_label'], errors='ignore')
    metric_summary = _build_bin_metric_summary(_summary_source) if show_metric_summary else ''
    if not return_ax:
        # 角标与图例字号随图尺寸自适应：默认 (12, 7) 不缩放，小图等比缩小以避免相互遮盖
        fig_w, fig_h = fig.get_size_inches()
        raw_scale = min(fig_w / 12.0, fig_h / 7.0)
        size_scale = float(np.clip(raw_scale, 0.5, 1.0))
        # 图较小时角标改为单列竖排，更窄以便挤入图例左侧而不遮盖
        if metric_summary and raw_scale < 0.65:
            metric_summary = _build_bin_metric_summary(_summary_source, per_row=1)
        summary_fontsize = float(np.clip(10.0 * size_scale, 6.0, 10.0))
        legend_fontsize = float(np.clip(10.0 * size_scale, 7.0, 10.0))

        if title is not None:
            fig.suptitle(f'{title}\n\n')
        else:
            fig.suptitle(f'{desc}{ending}\n\n')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend_labels = labels1 + labels2
        n_items = len(legend_labels)
        # 小图且需展示角标时，图例换行以缩小横向占用，为左上角角标让出空间
        if metric_summary and size_scale < 0.75 and n_items > 2:
            legend_ncol = int(np.ceil(n_items / 2))
        else:
            legend_ncol = n_items
        legend = fig.legend(handles1 + handles2, legend_labels, loc='upper center',
                            ncol=legend_ncol, bbox_to_anchor=(0.5, anchor),
                            frameon=False, fontsize=legend_fontsize)

        plt.tight_layout()
        if metric_summary:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_pos = ax1.get_position()
            legend_bbox = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            summary_text = fig.text(ax_pos.x0, legend_bbox.y0, metric_summary, ha='left', va='bottom',
                                    fontsize=summary_fontsize, color=axis_theme,
                                    bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                                              edgecolor=axis_theme, alpha=0.9, linewidth=0.8))
            # 角标右边界不越过居中图例左边界，必要时进一步缩小字号以免遮盖
            fig.canvas.draw()
            summary_bbox = summary_text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            available = legend_bbox.x0 - ax_pos.x0 - 0.015
            if available > 0.02 and summary_bbox.width > available:
                summary_text.set_fontsize(max(5.0, summary_fontsize * available / summary_bbox.width))
        save_figure(fig, save)

        if return_frame:
            return fig, _sorted_table.drop(columns=['_plot_bin_label'], errors='ignore')
        return fig
    else:
        if metric_summary:
            # 嵌入到已有画布时按子图实际尺寸自适应字号，避免在小子图上遮盖内容
            ax_pos = ax1.get_position()
            fig_w, fig_h = fig.get_size_inches()
            ax_w_in, ax_h_in = ax_pos.width * fig_w, ax_pos.height * fig_h
            ax_scale = min(ax_w_in / 7.5, ax_h_in / 4.5)
            summary_fontsize = float(np.clip(10.0 * ax_scale, 6.0, 10.0))
            if ax_scale < 0.65:
                metric_summary = _build_bin_metric_summary(_summary_source, per_row=1)
            ax2.text(0.0, 1.02, metric_summary, transform=ax2.transAxes, ha='left', va='bottom',
                     fontsize=summary_fontsize, color=axis_theme,
                     bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor=axis_theme, alpha=0.9, linewidth=0.8),
                     clip_on=False)
        if title is not None:
            ax1.set_title(title)
        else:
            ax1.set_title(f'{desc}{ending}')
        return ax1


def corr_plot(data, figure_size=None, fontsize=16, mask=False, save=None, 
              annot=True, max_len=35, linewidths=0.1, fmt='.2f', step=11, linecolor='white', 
              ax=None, figsize=(16, 8), **kwargs):
    """
    特征相关性热力图.

    :param data: 特征数据
    :param figure_size: 图像尺寸（创建新图时使用）
    :param fontsize: 字体大小
    :param mask: 是否只显示下三角
    :param save: 保存路径
    :param annot: 是否显示数值
    :param max_len: 特征名最大长度
    :param fmt: 数值格式
    :param step: 色阶步数
    :param linewidths: 边框宽度
    :param linecolor: 边框颜色
    :param ax: 可选的 matplotlib Axes 对象
    :return: matplotlib Figure 或 Axes
    """
    if max_len is None:
        corr = data.corr()
    else:
        corr = data.rename(columns={c: c if len(str(c)) <= max_len else f"{str(c)[:max_len]}..." 
                                   for c in data.columns}).corr()

    corr_mask = np.zeros_like(corr, dtype=bool)
    corr_mask[np.triu_indices_from(corr_mask)] = True

    # 获取或创建 Axes
    figsize = figure_size or figsize
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_ax = False
    else:
        fig = ax.figure
        return_ax = True

    map_plot = sns.heatmap(
        corr, cmap=sns.diverging_palette(267, 267, n=step, s=100, l=40),
        vmax=1, vmin=-1, center=0, square=True, linewidths=linewidths,
        annot=annot, fmt=fmt, linecolor=linecolor, robust=True, cbar=True,
        ax=ax, mask=corr_mask if mask else None, **kwargs
    )

    map_plot.tick_params(axis='x', labelrotation=270, labelsize=fontsize)
    map_plot.tick_params(axis='y', labelrotation=0, labelsize=fontsize)

    if not return_ax:
        save_figure(fig, save)
        return fig
    else:
        return ax


def ks_plot(score, target, title="", fontsize=14, figsize=(16, 8), save=None,
            colors=None, anchor=0.945, axes=None, ax=None, curve='both'):
    """
    KS曲线和ROC曲线.

    :param score: 预测分数或评分
    :param target: 真实标签
    :param title: 图表标题
    :param fontsize: 字体大小
    :param figsize: 图像尺寸（创建新图时使用）
    :param save: 保存路径
    :param colors: 配色方案
    :param anchor: 图例位置
    :param axes: 可选的 matplotlib Axes 对象数组 [ax1, ax2]
    :param ax: 可选的单个 Axes（配合 curve='ks'/'roc' 仅绘制单条曲线时使用）
    :param curve: 绘制内容，'both'(默认 KS+ROC) / 'ks'(仅KS曲线) / 'roc'(仅ROC曲线)。
        取 'ks' 或 'roc' 时仅需一个 Axes，便于嵌入到组合图中
    :return: matplotlib Figure 或 Axes（嵌入模式下返回所用 Axes）
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if curve not in ('both', 'ks', 'roc'):
        raise ValueError("curve 仅支持 'both'/'ks'/'roc'")

    # 兼容 axes 和 ax 参数
    axes = axes or (ax if isinstance(ax, (list, tuple, np.ndarray)) else None)

    # 转换 target 和 score 为 numpy 数组
    # 注意：函数签名是 ks_plot(score, target, ...)
    score_arr = np.asarray(score, dtype=float)
    target_arr = np.asarray(target)

    # 检查 target 是否为二分类
    unique_labels = np.unique(target_arr[~pd.isna(target_arr)])  # 排除 NaN
    if len(unique_labels) != 2:
        raise ValueError(
            f"target 必须是二分类标签（包含2个唯一值），当前有 {len(unique_labels)} 个唯一值。"
            f"请确保传入正确的 y_test 标签（如 0/1 或 True/False），而不是预测概率。"
        )

    # 确保 target 是数值型的 0/1 格式
    # 检查是否已经是 0/1 数值
    try:
        unique_numeric = [float(x) for x in unique_labels]
        is_01 = all(x in [0.0, 1.0] for x in unique_numeric)
    except (ValueError, TypeError):
        is_01 = False

    if is_01:
        # 已经是 0/1，直接转换类型
        target_arr = target_arr.astype(float)
    else:
        # 非 0/1 标签（如 -1/1, 'good'/'bad', True/False），映射为 0/1
        # 第一个唯一值映射为 0，第二个映射为 1
        label_0 = unique_labels[0]
        target_arr = np.where(target_arr == label_0, 0.0, 1.0)

    auc_value = roc_auc_score(target_arr, score_arr)

    if auc_value < 0.5:
        warnings.warn('评分AUC指标小于50%, 推断数据值越大, 正样本率越高, 将数据值转为负数后进行绘图')
        score_arr = -score_arr
        auc_value = 1 - auc_value

    df = pd.DataFrame({'label': target_arr, 'pred': score_arr})

    df_ks = df.sort_values('pred', ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / len(df.index)))) \
        .groupby('group')['label'].agg([lambda x: sum(x == 0), lambda x: sum(x == 1)]) \
        .reset_index().rename(columns={'<lambda_0>': 'good', '<lambda_1>': 'bad'}) \
        .assign(
            group=lambda x: (x.index + 1) / len(x.index),
            cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
            cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
        ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))

    need_ks = curve in ('both', 'ks')
    need_roc = curve in ('both', 'roc')

    def _is_single_ax(obj):
        return obj is not None and hasattr(obj, 'plot') and not hasattr(obj, '__len__')

    # 获取或创建 Axes
    if curve == 'both':
        if _is_single_ax(axes):
            # 传入的是单个 Axes，只用第一个子图，另建一个用于 ROC 曲线
            fig = axes.figure
            ax1 = axes
            ax2 = fig.add_subplot(122)
            return_axes = True
        elif axes is not None and hasattr(axes, '__len__') and len(axes) >= 2:
            ax1, ax2 = axes[0], axes[1]
            fig = ax1.figure
            return_axes = True
        else:
            fig, _ax = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = _ax[0], _ax[1]
            return_axes = False
    else:
        # 单曲线模式：仅需一个 Axes
        single = axes if _is_single_ax(axes) else (ax if _is_single_ax(ax) else None)
        if single is None and axes is not None and hasattr(axes, '__len__') and len(axes) >= 1:
            single = axes[0]
        if single is not None:
            target_ax = single
            fig = target_ax.figure
            return_axes = True
        else:
            fig, target_ax = plt.subplots(figsize=figsize)
            return_axes = False
        ax1 = target_ax if curve == 'ks' else None
        ax2 = target_ax if curve == 'roc' else None

    handles1, labels1, handles2, labels2 = [], [], [], []

    # KS曲线
    if need_ks:
        dfks = df_ks.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]

        ax1.plot(df_ks.group, df_ks.ks, color=colors[0], label="KS曲线")
        ax1.plot(df_ks.group, df_ks.cumgood, color=colors[1], label="累积好客户占比")
        ax1.plot(df_ks.group, df_ks.cumbad, color=colors[2], label="累积坏客户占比")
        ax1.fill_between(df_ks.group, df_ks.cumbad, df_ks.cumgood, color=colors[0], alpha=0.25)

        ax1.plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
        ax1.text(dfks['group'], dfks['ks'], f"KS: {round(dfks['ks'], 4)} at: {dfks.group:.2%}",
                 horizontalalignment='center', fontsize=fontsize)

        ax1.set_xlabel('% of Population', fontsize=fontsize)
        ax1.set_ylabel('% of Total Bad / Good', fontsize=fontsize)
        ax1.set_xlim((0, 1))
        ax1.set_ylim((0, 1))
        handles1, labels1 = ax1.get_legend_handles_labels()

    # ROC曲线
    if need_roc:
        fpr, tpr, thresholds = roc_curve(target_arr, score_arr)

        ax2.plot(fpr, tpr, color=colors[0], label="ROC Curve")
        ax2.stackplot(fpr, tpr, color=colors[0], alpha=0.25)
        ax2.plot([0, 1], [0, 1], color=colors[1], lw=2, linestyle=':')
        ax2.text(0.5, 0.5, f"AUC: {auc_value:.4f}", fontsize=fontsize,
                 horizontalalignment="center", transform=ax2.transAxes)

        ax2.set_xlabel("False Positive Rate", fontsize=fontsize)
        ax2.set_ylabel('True Positive Rate', fontsize=fontsize)
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0, 1))
        if curve == 'both':
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
        handles2, labels2 = ax2.get_legend_handles_labels()

    if not return_axes:
        if curve == 'both':
            if title:
                title += " "
            fig.suptitle(f"{title}K-S & ROC CURVE\n", fontsize=fontsize, fontweight="bold")
            fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center',
                       ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)
        else:
            single_ax = ax1 if curve == 'ks' else ax2
            if title:
                single_ax.set_title(f"{title}", fontsize=fontsize)
            handles, labels = handles1 + handles2, labels1 + labels2
            if labels:
                single_ax.legend(handles, labels, loc='best', frameon=False,
                                 fontsize=max(fontsize - 4, 8))
        plt.tight_layout()
        save_figure(fig, save)
        return fig
    else:
        if curve == 'both':
            return axes
        single_ax = ax1 if curve == 'ks' else ax2
        if title:
            single_ax.set_title(f"{title}", fontsize=fontsize)
        handles, labels = handles1 + handles2, labels1 + labels2
        if labels:
            single_ax.legend(handles, labels, loc='best', frameon=False,
                             fontsize=max(fontsize - 4, 7))
        return single_ax


def hist_plot(score, y_true=None, figsize=(15, 10), bins=30, save=None,
              labels=None, desc="", anchor=1.15, fontsize=14, kde=False, title=None,
              ax=None, **kwargs):
    """
    特征值分布直方图.

    :param score: 特征值
    :param y_true: 标签
    :param figsize: 图像尺寸（创建新图时使用）
    :param bins: 分箱数
    :param save: 保存路径
    :param labels: 图例标签
    :param desc: 描述
    :param anchor: 图例位置
    :param fontsize: 字体大小
    :param kde: 是否显示核密度估计
    :param title: 完整标题（优先级高于 desc）
    :param ax: 可选的 matplotlib Axes 对象
    :param kwargs: 其他参数
    :return: matplotlib Figure 或 Axes
    """
    if labels is None:
        labels = ["好样本", "坏样本"]

    target_unique = 1 if y_true is None else len(np.unique(y_true))

    if y_true is not None:
        if isinstance(labels, dict):
            y_true = y_true.map(labels)
            hue_order = list(labels.values())
        else:
            y_true = y_true.map({i: v for i, v in enumerate(labels)})
            hue_order = labels
    else:
        y_true = None
        hue_order = None

    # 获取或创建 Axes
    if ax is not None:
        return_ax = True
        fig = ax.figure
    else:
        return_ax = False
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    palette = sns.diverging_palette(340, 267, n=target_unique, s=100, l=40)

    # 处理 hue_order 参数
    if hue_order is not None:
        hue_order_final = hue_order[::-1]
    else:
        hue_order_final = None

    hist_kwargs = dict(
        x=score,
        hue=y_true,
        element="step",
        stat="probability",
        bins=bins,
        common_bins=True,
        common_norm=True,
        ax=ax,
        kde=kde,
    )
    if y_true is not None:
        hist_kwargs.update(palette=palette, hue_order=hue_order_final)
    else:
        hist_kwargs.update(color=kwargs.pop('color', DEFAULT_COLORS[0]))

    sns.histplot(**hist_kwargs, **kwargs)

    # 使用公共函数设置坐标轴样式
    setup_axis_style(ax)

    ax.set_xlabel("值域范围", fontsize=fontsize)
    ax.set_ylabel("样本占比", fontsize=fontsize)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 标题处理：优先使用 title 参数
    if title is not None:
        ax.set_title(f"{title}\n\n", fontsize=fontsize)
    else:
        ax.set_title(f"{desc + ' ' if desc else '特征'}分布情况\n\n", fontsize=fontsize)

    if y_true is not None:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, hue_order_final[:len(handles)] if hue_order_final else legend_labels,
                      loc='upper center', ncol=len(handles),
                      bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)
        else:
            ax.legend(hue_order,
                      loc='upper center', ncol=target_unique,
                      bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)

    if not return_ax:
        fig.tight_layout()
        save_figure(fig, save)
        return fig
    else:
        return ax


def psi_plot(expected, actual, y=None, labels=None, desc="", save=None, colors=None,
             figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True,
             max_len=None, hatch=True, title=None, **kwargs):
    """
    PSI稳定性分析图.

    支持两种输入方式：
    1. 直接传入原始分数数据（pd.Series 或单列 pd.DataFrame），自动分箱计算PSI
    2. 传入已计算好的分箱表（pd.DataFrame，含 '分箱' 列），直接绘图

    :param expected: 期望分布（原始数据或分箱表）
    :param actual: 实际分布（原始数据或分箱表）
    :param y: 目标变量（pd.Series），用于绘制坏样本率折线图。当传入原始数据时，
        应与 expected/actual 长度一致，按相同分箱计算坏样本率。
    :param labels: 标签
    :param desc: 描述
    :param save: 保存路径
    :param colors: 配色
    :param figsize: 图像尺寸
    :param anchor: 图例位置
    :param width: 柱宽
    :param result: 是否返回统计表
    :param plot: 是否绘图
    :param max_len: 标签最大长度
    :param hatch: 是否显示斜线
    :param title: 完整标题（优先级高于 desc）
    :return: pd.DataFrame when result=True
    """
    if labels is None:
        labels = ["预期", "实际"]
    if colors is None:
        colors = DEFAULT_COLORS

    # 统一提取一维数值数组
    def _to_series(data):
        if isinstance(data, np.ndarray):
            return pd.Series(data.ravel(), name=data.name if hasattr(data, 'name') else None)
        if isinstance(data, pd.Series):
            return data
        return data.iloc[:, 0]

    def _has_bins(data):
        if isinstance(data, np.ndarray):
            return False
        if isinstance(data, pd.Series):
            return False
        # 分箱表: 有 '分箱' + '样本总数' (分箱统计表)
        if "分箱" in data.columns and "样本总数" in data.columns:
            return True
        # psi_table 输出: 有 '分箱' + '期望样本数' + '实际样本数'
        if "分箱" in data.columns and "期望样本数" in data.columns and "实际样本数" in data.columns:
            return True
        return False

    exp_series = _to_series(expected)
    act_series = _to_series(actual)

    if _has_bins(expected) and _has_bins(actual):
        # 检查是否为 psi_table 输出格式（有 '期望样本数' 列）
        is_psi_table_fmt = (
            "期望样本数" in expected.columns and "实际样本数" in expected.columns and
            "期望样本数" in actual.columns and "实际样本数" in actual.columns
        )

        if is_psi_table_fmt:
            # psi_table 格式：重命名期望/实际列，再 merge
            exp_renamed = expected.rename(columns={
                "期望样本数": f"{labels[0]}样本数",
                "期望占比": f"{labels[0]}样本占比",
            })
            act_renamed = actual.rename(columns={
                "实际样本数": f"{labels[1]}样本数",
                "实际占比": f"{labels[1]}样本占比",
            })
            cols_to_keep = ["分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比",
                           f"{labels[1]}样本数", f"{labels[1]}样本占比"]
            exp_cols = {c: c for c in cols_to_keep if c in exp_renamed.columns}
            act_cols = {c: c for c in cols_to_keep if c in act_renamed.columns}
            df_psi = exp_renamed[list(exp_cols.values())].merge(
                act_renamed[list(act_cols.values())], on="分箱", how="outer"
            ).replace(np.nan, 0)
            # psi_table 输出没有坏样本率列，设为0避免绘图报错
            for lbl in labels:
                df_psi[f"{lbl}坏样本率"] = 0.0
        else:
            # 分箱表格式：分别重命名 '样本总数' + '样本占比' + '坏样本率' 列
            exp_renamed = expected.rename(columns={
                "样本总数": f"{labels[0]}样本数",
                "样本占比": f"{labels[0]}样本占比",
                "坏样本率": f"{labels[0]}坏样本率",
            })
            act_renamed = actual.rename(columns={
                "样本总数": f"{labels[1]}样本数",
                "样本占比": f"{labels[1]}样本占比",
                "坏样本率": f"{labels[1]}坏样本率",
            })
            df_psi = exp_renamed.merge(act_renamed, on="分箱", how="outer").replace(np.nan, 0)
        df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
        df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(
            df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"]
        )
        df_psi["分档PSI值"] = df_psi[f"{labels[1]}% - {labels[0]}%"] * df_psi[f"ln({labels[1]}% / {labels[0]}%)"]
        df_psi = df_psi.fillna(0).replace([np.inf, -np.inf], 0)
        df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
        df_psi["指标名称"] = desc
    else:
        # 路径B：传入原始分数，使用同一个分箱器计算 PSI 与坏样本率。

        # 创建统一 binner，避免 PSI 分布与坏样本率使用不同的分箱。
        from ..binning import OptimalBinning
        exp_values = exp_series.rename('value').reset_index(drop=True)
        act_values = act_series.rename('value').reset_index(drop=True)
        combined = pd.concat([exp_values, act_values], ignore_index=True)
        dummy_y = np.random.randint(0, 2, size=len(combined))
        binning_kwargs = dict(kwargs)
        method = binning_kwargs.pop('method', 'quantile')
        max_n_bins = binning_kwargs.pop('max_n_bins', 10)
        min_bin_size = binning_kwargs.pop('min_bin_size', 0.01)
        binner = OptimalBinning(
            method=method,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            verbose=False,
            **binning_kwargs,
        )
        binner.fit(combined.to_frame(), dummy_y)

        exp_frame = exp_values.to_frame()
        act_frame = act_values.to_frame()
        exp_bins = binner.transform(exp_frame, metric='indices').values.flatten()
        act_bins = binner.transform(act_frame, metric='indices').values.flatten()

        # 如果传入了 y，计算各分箱的坏样本率
        if y is not None:
            y_combined = pd.Series(np.asarray(y).reshape(-1), name='y').reset_index(drop=True)
            expected_length = len(exp_values) + len(act_values)
            if len(y_combined) != expected_length:
                raise ValueError(
                    f"y 长度应等于 expected 与 actual 长度之和: "
                    f"{len(y_combined)} != {expected_length}"
                )
            exp_y = y_combined.iloc[:len(exp_values)].to_numpy()
            act_y = y_combined.iloc[len(exp_values):].to_numpy()
            _bad_exp_map = pd.DataFrame({'bin': exp_bins, 'y': exp_y}).groupby('bin')['y'].mean().to_dict()
            _bad_act_map = pd.DataFrame({'bin': act_bins, 'y': act_y}).groupby('bin')['y'].mean().to_dict()
        else:
            _bad_exp_map = {}
            _bad_act_map = {}

        unique_bins = sorted(set(exp_bins) | set(act_bins))
        bin_table = binner.bin_tables_.get('value')

        def _bin_label(bin_idx):
            if bin_table is not None and 0 <= int(bin_idx) < len(bin_table) and '分箱标签' in bin_table.columns:
                return bin_table.iloc[int(bin_idx)]['分箱标签']
            return f"Bin_{bin_idx}"

        rows = []
        epsilon = 1e-10
        for bin_idx in unique_bins:
            exp_count = int(np.sum(exp_bins == bin_idx))
            act_count = int(np.sum(act_bins == bin_idx))
            exp_prop = max(exp_count / len(exp_bins), epsilon) if len(exp_bins) else epsilon
            act_prop = max(act_count / len(act_bins), epsilon) if len(act_bins) else epsilon
            rows.append({
                '分箱': _bin_label(bin_idx),
                f"{labels[0]}样本数": exp_count,
                f"{labels[1]}样本数": act_count,
                f"{labels[0]}样本占比": exp_prop,
                f"{labels[1]}样本占比": act_prop,
                '分档PSI值': (act_prop - exp_prop) * np.log(act_prop / exp_prop),
                '_bin_idx': bin_idx,
            })
        df_psi = pd.DataFrame(rows)
        # 坏样本率：优先使用 y 计算的值，否则设为 0
        for lbl, bad_map in [(labels[0], _bad_exp_map), (labels[1], _bad_act_map)]:
            if bad_map:
                df_psi[f"{lbl}坏样本率"] = df_psi['_bin_idx'].apply(
                    lambda b: bad_map.get(b, 0.0)
                )
            else:
                df_psi[f"{lbl}坏样本率"] = 0.0
        # 补充 result=True 需要的差值列
        df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
        df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(
            df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"]
        )
        df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
        df_psi["指标名称"] = desc
        df_psi = df_psi.drop(columns=['_bin_idx'])

    if plot:
        x = df_psi['分箱'].apply(
            lambda l: l if max_len is None or len(str(l)) < max_len else f"{str(l)[:max_len]}..."
        )
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_psi[f'{labels[0]}样本占比'], width,
                label=f'{labels[0]}样本占比', color=colors[0], hatch="/" if hatch else None,
                edgecolor='white' if hatch else None)
        ax1.bar(x_indexes + width / 2, df_psi[f'{labels[1]}样本占比'], width,
                label=f'{labels[1]}样本占比', color=colors[1], hatch="\\" if hatch else None,
                edgecolor='white' if hatch else None)

        ax1.set_ylabel('样本占比')
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x, df_psi[f"{labels[0]}坏样本率"], color=colors[0], 
                label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(x, df_psi[f"{labels[1]}坏样本率"], color=colors[1], 
                label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))
        ax2.scatter(x, df_psi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(x, df_psi[f"{labels[1]}坏样本率"], marker=".")
        ax2.set_ylabel('坏样本率')
        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        # 标题处理：优先使用 title 参数
        if title is not None:
            fig.suptitle(f"{title}\n\n")
        else:
            fig.suptitle(
                f"{desc + ' ' if desc else ''}{labels[0]} vs {labels[1]} "
                f"群体稳定性指数(PSI): {df_psi['分档PSI值'].sum():.4f}\n\n"
            )

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
                  ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        return df_psi[["指标名称", "分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", 
                      f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", 
                      f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", 
                      f"ln({labels[1]}% / {labels[0]}%)", "分档PSI值", "总体PSI值"]]


def dataframe_plot(df, row_height=0.4, font_size=14, header_color='#2639E9', 
                  row_colors=None, edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, 
                  ax=None, save=None, **kwargs):
    """
    将DataFrame转换为图像.

    :param df: 数据框
    :param row_height: 行高
    :param font_size: 字体大小
    :param header_color: 表头颜色
    :param row_colors: 行颜色
    :param edge_color: 边框颜色
    :param bbox: 边框
    :param header_columns: 表头列数
    :param ax: 坐标系
    :param save: 保存路径
    :return: matplotlib Figure
    """
    if row_colors is None:
        row_colors = ['#dae3f3', 'w']

    data = df.copy()
    for col in data.select_dtypes('datetime'):
        data[col] = data[col].dt.strftime("%Y-%m-%d")

    for col in data.select_dtypes('float'):
        data[col] = data[col].apply(lambda x: np.nan if pd.isnull(x) else round(x, 4))

    cols_width = [
        max(data[col].apply(lambda x: len(str(x).encode())).max(), 
            len(str(col).encode())) / 8. 
        for col in data.columns
    ]

    if ax is None:
        size = (sum(cols_width), (len(data) + 1) * row_height)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    else:
        fig = ax.get_figure()

    mpl_table = ax.table(
        cellText=data.values, colWidths=cols_width, bbox=bbox, 
        colLabels=data.columns, **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def distribution_plot(data, date="date", target="target", save=None, figsize=(10, 6), 
                    colors=None, freq="M", anchor=0.94, result=False, hatch=True,
                    overdue=None, dpds=None, title=None):
    """
    样本时间分布图.

    支持两种模式：
    1. 单目标模式：传入 target 列名，展示好/坏样本堆叠柱状图 + 坏样本率折线
    2. 多逾期口径模式：传入 overdue + dpds，展示样本总数柱状图 + 多条坏样本率折线

    :param data: 数据集
    :param date: 日期列名
    :param target: 目标列名（单目标模式使用）
    :param save: 保存路径
    :param figsize: 图像尺寸
    :param colors: 配色
    :param freq: 日期频率，'D'/'W'/'M'/'Q'
    :param anchor: 图例位置
    :param result: 是否返回统计表
    :param hatch: 是否显示斜线
    :param overdue: 逾期列名列表，如 ['dpd7', 'dpd15', 'dpd30']（多逾期口径模式）
    :param dpds: 逾期阈值列表，与 overdue 一一对应，如 [1, 1, 1]
    :param title: 图表标题
    :return: matplotlib Figure or pd.DataFrame

    Example:
        >>> # 单目标模式
        >>> distribution_plot(df, date='apply_date', target='target')

        >>> # 多逾期口径模式
        >>> distribution_plot(
        ...     df, date='apply_date',
        ...     overdue=['dpd7', 'dpd15', 'dpd30'], dpds=[1, 1, 1]
        ... )
    """
    if colors is None:
        colors = DEFAULT_COLORS

    df = data.copy()

    if 'time' not in str(df[date].dtype):
        df[date] = pd.to_datetime(df[date])

    # ---------- 多逾期口径模式 ----------
    if overdue is not None and dpds is not None:
        if len(overdue) != len(dpds):
            raise ValueError("overdue 和 dpds 长度必须一致")

        # 按日期聚合样本总数
        df_indexed = df.set_index(date)
        total_counts = df_indexed.resample(freq).size()
        total_counts.index = [i.strftime("%Y-%m-%d") for i in total_counts.index]

        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        total_counts.plot(kind='bar', ax=ax1, color=colors[0],
                         hatch="/" if hatch else None,
                         edgecolor='white' if hatch else None,
                         legend=False, label='样本总数')
        ax1.tick_params(axis='x', labelrotation=-90)
        ax1.set(xlabel=None)
        ax1.set_ylabel('样本数')

        if title is None:
            title = '不同时点多逾期口径坏样本率分布\n\n'
        else:
            title = f'{title}\n\n'
        ax1.set_title(title)

        ax2 = ax1.twinx()
        # 定义多条折线的样式
        line_styles = ['--', '-.', ':', '-', (0, (3, 1, 1, 1))]
        line_colors = colors[1:] if len(colors) > 1 else ['#E85D4A', '#4C8DFF', '#67C23A', '#E6A23C', '#909399']
        # 确保颜色足够
        while len(line_colors) < len(overdue):
            line_colors = line_colors + line_colors

        result_frames = []
        for i, (dpd_col, threshold) in enumerate(zip(overdue, dpds)):
            y_target = (df[dpd_col] >= threshold).astype(int)
            df_temp = df_indexed.copy()
            df_temp['_bad'] = y_target.values
            bad_rate = df_temp.resample(freq)['_bad'].mean()
            bad_rate.index = [idx.strftime("%Y-%m-%d") for idx in bad_rate.index]

            label = f"{dpd_col}>={threshold}"
            style = line_styles[i % len(line_styles)]
            color = line_colors[i % len(line_colors)]
            bad_rate.plot(ax=ax2, color=color, style=style, linewidth=2,
                         marker='o', markersize=4, markerfacecolor='white',
                         label=label)

            if result:
                rate_df = bad_rate.reset_index()
                rate_df.columns = ['日期', f'{label}_坏样本率']
                result_frames.append(rate_df.set_index('日期'))

        ax2.set_ylabel('坏样本率')
        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center',
                  ncol=min(len(labels1 + labels2), 6),
                  bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

        if result:
            counts_df = total_counts.reset_index()
            counts_df.columns = ['日期', '样本总数']
            merged = counts_df.set_index('日期')
            for rf in result_frames:
                merged = merged.join(rf, how='left')
            return merged.reset_index()

        return fig

    # ---------- 单目标模式（原逻辑） ----------
    temp = df.set_index(date).assign(
        好样本=lambda x: (x[target] == 0).astype(int),
        坏样本=lambda x: (x[target] == 1).astype(int),
    ).resample(freq).agg({"好样本": sum, "坏样本": sum})

    temp.index = [i.strftime("%Y-%m-%d") for i in temp.index]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    temp.plot(kind='bar', stacked=True, ax=ax1, color=colors[:2], 
             hatch="/" if hatch else None, edgecolor='white' if hatch else None, legend=False)
    ax1.tick_params(axis='x', labelrotation=-90)
    ax1.set(xlabel=None)
    ax1.set_ylabel('样本数')

    if title is None:
        title = '不同时点数据集样本分布情况\n\n'
    else:
        title = f'{title}\n\n'
    ax1.set_title(title)

    ax2 = ax1.twinx()
    (temp["坏样本"] / temp.sum(axis=1)).plot(
        ax=ax2, color=colors[-1], style="--", linewidth=2, label="坏样本率"
    )
    ax2.set_ylabel('坏样本率')
    ax2.yaxis.set_major_formatter(PercentFormatter(1))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
              ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        temp = temp.reset_index().rename(
            columns={date: "日期", "index": "日期", 0: "好样本", 1: "坏样本"}
        )
        temp["样本总数"] = temp["坏样本"] + temp["好样本"]
        temp["样本占比"] = temp["样本总数"] / temp["样本总数"].sum()
        temp["好样本占比"] = temp["好样本"] / temp["好样本"].sum()
        temp["坏样本占比"] = temp["坏样本"] / temp["坏样本"].sum()
        temp["坏样本率"] = temp["坏样本"] / temp["样本总数"]

        return temp[["日期", "样本总数", "样本占比", "好样本", "好样本占比", 
                    "坏样本", "坏样本占比", "坏样本率"]]

    return fig


# ==================== 多维度分箱趋势图 ====================


def _compute_feature_bin_stats(
    data: pd.DataFrame,
    feature: str,
    target: str,
    group_col: Optional[str] = None,
    group_value: Optional[Any] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    min_bin_size: float = 0.02,
    rules: Optional[Dict] = None,
    special_values: Optional[List] = None,
    **kwargs
) -> pd.DataFrame:
    """计算特征分箱统计.

    :param data: 输入数据
    :param feature: 特征列名
    :param target: 目标变量列名
    :param group_col: 分组列名
    :param group_value: 分组值
    :param method: 分箱方法
    :param max_n_bins: 最大分箱数
    :param min_bin_size: 最小箱占比
    :param rules: 预定义分箱规则
    :param special_values: 特殊值列表
    :return: 分箱统计表
    """
    # 筛选分组数据
    if group_col is not None and group_value is not None:
        df_sub = data[data[group_col] == group_value].copy()
    else:
        df_sub = data.copy()

    if len(df_sub) == 0:
        return pd.DataFrame()

    X = df_sub[feature].copy()
    y = df_sub[target].copy()

    # 移除缺失值
    valid_mask = ~(pd.isna(X) | pd.isna(y))
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(X_valid) == 0:
        return pd.DataFrame()

    # 使用 OptimalBinning 进行分箱
    from ..binning import OptimalBinning

    if rules is not None and feature in rules:
        binner = OptimalBinning(
            user_splits={feature: rules[feature]},
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            verbose=False,
            **kwargs
        )
    else:
        binner = OptimalBinning(
            method=method,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            verbose=False,
            **kwargs
        )

    try:
        binner.fit(X_valid.to_frame(), y_valid)
        bin_indices = binner.transform(X_valid.to_frame(), metric='indices').values.flatten()

        # 获取分箱标签
        bin_labels = None
        if feature in binner.bin_tables_:
            bin_table = binner.bin_tables_[feature]
            if '分箱标签' in bin_table.columns:
                bin_labels = bin_table['分箱标签'].tolist()

        # 计算分箱统计
        stats_df = compute_bin_stats(bin_indices, y_valid.values, bin_labels=bin_labels, round_digits=False)

        # 添加缺失值统计
        missing_count = (~valid_mask).sum()
        if missing_count > 0:
            missing_bad = y[~valid_mask].sum()
            missing_row = pd.DataFrame([{
                '分箱': -1,
                '分箱标签': 'Missing',
                '样本总数': missing_count,
                '好样本数': missing_count - missing_bad,
                '坏样本数': missing_bad,
                '坏样本率': missing_bad / missing_count if missing_count > 0 else 0,
                '样本占比': missing_count / len(df_sub),
            }])
            stats_df = pd.concat([stats_df, missing_row], ignore_index=True)

        # 计算指标
        total_bad = y_valid.sum()
        total_count = len(df_sub)

        # 计算 IV
        try:
            from ..metrics import iv as iv_metric
            iv_val = iv_metric(X_valid, y_valid)
        except Exception:
            iv_val = 0

        # 计算 KS
        try:
            from ..metrics import ks as ks_metric
            ks_val = ks_metric(X_valid, y_valid)
        except Exception:
            ks_val = 0

        # 添加统计列
        stats_df['iv_bin'] = iv_val / len(stats_df) if len(stats_df) > 0 else 0
        stats_df['ks_bin'] = ks_val
        stats_df['total_count'] = total_count
        stats_df['total_bad'] = total_bad
        stats_df['feature'] = feature

        return stats_df

    except Exception as e:
        warnings.warn(f"分箱计算失败: {e}")
        return pd.DataFrame()


def bin_trend_plot(
    data: pd.DataFrame,
    feature: str,
    target: str,
    dimension_cols: Optional[Union[str, List[str]]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    method: str = 'quantile',
    max_n_bins: int = 10,
    min_bin_size: float = 0.02,
    rules: Optional[Dict] = None,
    special_values: Optional[List] = None,
    shared_bins: Optional[Union[str, bool]] = 'max_samples',
    sort_by: Optional[str] = None,
    sort_order: str = 'asc',
    max_groups: Optional[int] = None,
    figsize: Optional[tuple] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_overall: bool = True,
    show_stats: bool = True,
    orientation: str = 'vertical',
    dpi: int = 150,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制特征分箱风险趋势图.

    该图表集成了特征在不同维度下的样本分布、坏率走势、统计指标等信息。
    支持按时间维度（自动聚合）或指定维度列进行分组展示。

    :param data: 输入数据
    :param feature: 特征列名
    :param target: 目标变量列名（0/1）
    :param dimension_cols: 维度列名（单维或多维），用于分组展示
    :param date_col: 日期列名，如提供则按日期分组
    :param date_freq: 日期聚合频率，'D'/'W'/'M'/'Q'，默认'M'
    :param method: 分箱方法，可选 'quantile'/'uniform'/'cart' 等
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 最小箱占比，默认0.02
    :param rules: 预定义分箱规则 {特征名: 分箱边界列表}
    :param special_values: 特殊值列表
    :param shared_bins: 各分组是否共享同一切分点，默认 'max_samples'
        - 'first': 使用第一个分组（最早时间/第一个维度值）的切分点
        - 'last': 使用最后一个分组（最近时间/最后一个维度值）的切分点
        - 'max_samples': 使用样本量最多的分组的切分点（默认）
        - False 或 None: 每个分组独立计算切分点
    :param sort_by: 排序列名，None表示不排序，默认按维度值排序
    :param sort_order: 排序方向，'asc'/'desc'
    :param max_groups: 最大展示分组数，None表示全部展示
    :param figsize: 图像尺寸，None时自动计算
    :param colors: 配色方案
    :param title: 图表标题
    :param show_overall: 是否显示整体样本面板
    :param show_stats: 是否显示统计指标
    :param orientation: 图表方向，'vertical'（纵向，默认）或 'horizontal'
    :param dpi: 图像分辨率
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure

    Example:
        >>> # 按月份查看特征趋势
        >>> fig = bin_trend_plot(
        ...     df, feature='age', target='bad', date_col='apply_date'
        ... )

        >>> # 按客群维度查看
        >>> fig = bin_trend_plot(
        ...     df, feature='score', target='bad', dimension_cols='customer_type'
        ... )

        >>> # 多维度交叉
        >>> fig = bin_trend_plot(
        ...     df, feature='income', target='bad',
        ...     dimension_cols=['region', 'channel']
        ... )

        >>> # 自定义分箱规则
        >>> fig = bin_trend_plot(
        ...     df, feature='score', target='bad',
        ...     rules={'score': [300, 500, 600, 700, 800]}
        ... )

        >>> # 各分组使用第一个分组的切分点
        >>> fig = bin_trend_plot(
        ...     df, feature='score', target='bad', date_col='apply_date',
        ...     shared_bins='first'
        ... )
    """
    if colors is None:
        colors = DEFAULT_COLORS

    orientation_key = orientation.lower()
    is_horizontal = orientation_key in ['horizontal', 'h', '横向']

    # 处理维度列
    if dimension_cols is not None:
        if isinstance(dimension_cols, str):
            dimension_cols = [dimension_cols]
    else:
        dimension_cols = []

    # 处理日期列
    if date_col is not None:
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])

        try:
            if date_freq == 'D':
                data['_date_group'] = data[date_col].dt.strftime('%Y-%m-%d')
            else:
                data['_date_group'] = data[date_col].dt.to_period(date_freq).astype(str)
        except Exception:
            warnings.warn(f"无法识别 date_freq={date_freq}，已回退为按月分组")
            data['_date_group'] = data[date_col].dt.to_period('M').astype(str)

        dimension_cols.append('_date_group')

    # 创建组合维度列
    if len(dimension_cols) > 0:
        data = data.copy()
        data['_group_key'] = data[dimension_cols].astype(str).agg('_'.join, axis=1)
        group_col = '_group_key'
    else:
        group_col = None

    # 处理 shared_bins：从指定分组提取切分点，统一应用到所有分组
    if shared_bins and group_col is not None and rules is None:
        groups = data[group_col].unique()
        _sort_by = sort_by if (sort_by is not None and sort_by in data.columns) else None
        if _sort_by is not None:
            _group_order = data.groupby(group_col)[_sort_by].first().sort_values(
                ascending=(sort_order == 'asc')
            ).index.tolist()
        else:
            _group_order = sorted(groups)

        _shared_bins = str(shared_bins).lower()
        if _shared_bins == 'first':
            ref_group = _group_order[0] if _group_order else None
        elif _shared_bins == 'last':
            ref_group = _group_order[-1] if _group_order else None
        else:  # 'max_samples' 或其他真值
            group_sizes = data.groupby(group_col).size()
            ref_group = group_sizes.idxmax()

        if ref_group is not None:
            from ..binning import OptimalBinning
            ref_data = data[data[group_col] == ref_group]
            _valid = ~(pd.isna(ref_data[feature]) | pd.isna(ref_data[target]))
            X_ref = ref_data.loc[_valid, feature]
            y_ref = ref_data.loc[_valid, target]
            if len(X_ref) > 0:
                _binner = OptimalBinning(
                    method=method, max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size, verbose=False, **kwargs
                )
                try:
                    _binner.fit(X_ref.to_frame(), y_ref)
                    _splits = _binner.splits_.get(feature, [])
                    if len(_splits) > 0:
                        rules = {feature: list(_splits)}
                except Exception:
                    pass  # 回退到独立分箱

    overall_stats = _compute_feature_bin_stats(
        data, feature, target,
        method=method, max_n_bins=max_n_bins, min_bin_size=min_bin_size,
        rules=rules, special_values=special_values, **kwargs
    )

    if overall_stats.empty:
        raise ValueError(f"无法计算特征 '{feature}' 的分箱统计")

    panel_stats = [('Overall', overall_stats.copy())] if show_overall else []

    if group_col is not None:
        groups = data[group_col].unique()
        if sort_by is not None and sort_by in data.columns:
            group_order = data.groupby(group_col)[sort_by].first().sort_values(
                ascending=(sort_order == 'asc')
            ).index.tolist()
        else:
            group_order = sorted(groups)

        if max_groups is not None and len(group_order) > max_groups:
            group_order = group_order[:max_groups]

        for group_val in group_order:
            stats = _compute_feature_bin_stats(
                data, feature, target,
                group_col=group_col, group_value=group_val,
                method=method, max_n_bins=max_n_bins, min_bin_size=min_bin_size,
                rules=rules, special_values=special_values, **kwargs
            )
            if not stats.empty:
                panel_stats.append((group_val, stats.copy()))

    if not panel_stats:
        raise ValueError("没有可用的分箱统计数据")

    n_panels = len(panel_stats)
    if is_horizontal:
        n_cols = 1
        n_rows = n_panels
    else:
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))

    if figsize is None:
        if is_horizontal:
            figsize = (10.5, max(4.8 * n_rows, 5.2))
        else:
            figsize = (max(5.2 * n_cols, 10.5), max(5.4 * n_rows, 5.2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    if title is None:
        title = f"{feature} - Risk Trend Analysis"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='white', label='好样本'),
        Patch(facecolor=colors[1], edgecolor='white', label='坏样本'),
        Line2D([0], [0], color='#E85D4A', linestyle=(0, (4, 3)), linewidth=2.1, marker='o', markersize=5, markerfacecolor='white', label='坏样本率'),
        Line2D([0], [0], color='#4C8DFF', linestyle=(0, (2, 2)), linewidth=1.8, label='整体坏样本率'),
    ]

    summary_cols = ['指标IV值', '分档KS值', 'LIFT值']
    panel_max_len = 22 if is_horizontal else 18

    for idx, (group_name, group_df) in enumerate(panel_stats):
        ax = axes_flat[idx]
        group_total = group_df['样本总数'].sum()
        group_bad = group_df['坏样本数'].sum()
        group_bad_rate = group_bad / group_total if group_total > 0 else 0.0
        panel_title = f"{group_name}\n({int(group_bad)}/{int(group_total)}, {group_bad_rate:.1%})"

        panel_df = group_df.copy()
        if not show_stats:
            panel_df = panel_df.drop(columns=summary_cols, errors='ignore')

        try:
            bin_plot(
                data=panel_df,
                ax=ax,
                title=panel_title,
                colors=colors,
                orientation='horizontal' if is_horizontal else 'vertical',
                max_len=panel_max_len,
                show_overall_bad_rate=True,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(panel_title)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(
        top=0.84 if n_rows > 1 else 0.80,
        bottom=0.08,
        left=0.06,
        right=0.98,
        hspace=0.62 if n_rows > 1 else 0.42,
        wspace=0.28,
    )

    if save:
        save_figure(fig, save)

    return fig


def batch_bin_trend_plot(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    dimension_cols: Optional[Union[str, List[str]]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    sort_by: str = 'iv',
    max_features: int = 10,
    figsize_per_feature: tuple = (12, 4),
    save_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, plt.Figure]:
    """批量绘制多个特征的风险趋势图.

    :param data: 输入数据
    :param features: 特征列表
    :param target: 目标变量列名
    :param dimension_cols: 维度列名
    :param date_col: 日期列名
    :param date_freq: 日期聚合频率
    :param sort_by: 排序指标，'iv'/'ks'/'auc'
    :param max_features: 最大绘制特征数
    :param figsize_per_feature: 每个特征的图尺寸
    :param save_dir: 保存目录
    :param kwargs: 其他参数传递给 bin_trend_plot
    :return: 特征名到 Figure 的字典
    """
    results = {}

    # 计算特征排序
    feature_scores = []
    for feat in features:
        try:
            stats = _compute_feature_bin_stats(data, feat, target, **kwargs)
            if not stats.empty:
                iv_val = stats['iv_bin'].sum()
                ks_val = stats['ks_bin'].max()
                score = iv_val if sort_by == 'iv' else ks_val
                feature_scores.append({'feature': feat, 'score': score, 'iv': iv_val, 'ks': ks_val})
        except Exception:
            pass

    if feature_scores:
        score_df = pd.DataFrame(feature_scores).sort_values('score', ascending=False)
        sorted_features = score_df['feature'].tolist()[:max_features]
    else:
        sorted_features = features[:max_features]

    # 批量绘制
    for i, feat in enumerate(sorted_features):
        logger.info("[%s/%s] 正在绘制 %s", i + 1, len(sorted_features), feat)

        try:
            fig = bin_trend_plot(
                data, feature=feat, target=target,
                dimension_cols=dimension_cols,
                date_col=date_col, date_freq=date_freq,
                figsize=figsize_per_feature,
                **kwargs
            )

            results[feat] = fig

            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{feat}_trend.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

        except Exception as e:
            warnings.warn(f"绘制特征 {feat} 失败: {e}")

    return results


# ==================== 多逾期天数分箱图 ====================

def _is_multiindex_bin_table(df: pd.DataFrame) -> bool:
    """检查是否为多级表头的分箱表（来自 feature_bin_stats）."""
    return isinstance(df.columns, pd.MultiIndex)


def _extract_target_names_from_bin_table(bin_table: pd.DataFrame) -> List[str]:
    """从多级表头分箱表中提取目标名称列表."""
    # 获取第一级列名（排除 '分箱详情'）
    level_0_names = bin_table.columns.get_level_values(0).unique()
    target_names = [name for name in level_0_names if name != '分箱详情']
    return target_names


def _get_stats_for_target(bin_table: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """从多级表头分箱表中提取指定目标的统计信息.
    
    :param bin_table: 多级表头分箱表
    :param target_name: 目标名称
    :return: 单目标的分箱统计表（标准格式）
    """
    # 获取分箱详情列和目标列
    common_cols = []
    target_cols = []
    
    for col_tuple in bin_table.columns:
        if col_tuple[0] == '分箱详情':
            common_cols.append(col_tuple[1])
        elif col_tuple[0] == target_name:
            target_cols.append(col_tuple[1])
    
    # 构建标准格式的分箱表
    stats_df = pd.DataFrame()
    
    # 添加公共列（重命名以匹配标准格式）
    col_mapping = {
        '分箱标签': '分箱',
        '样本总数': '样本总数',
        '样本占比': '样本占比',
        '指标名称': '特征',
        '指标含义': '描述'
    }
    
    for orig_col, std_col in col_mapping.items():
        if orig_col in common_cols:
            stats_df[std_col] = bin_table[('分箱详情', orig_col)].values
    
    # 添加目标列（重命名以匹配标准格式）
    target_col_mapping = {
        '好样本数': '好样本数',
        '坏样本数': '坏样本数',
        '坏样本率': '坏样本率',
        '累计好样本占比': '累计好样本占比',
        '累计坏样本占比': '累计坏样本占比',
        'Lift': 'Lift',
        'WOE值': 'WOE值',
        'IV值': 'IV值'
    }
    
    for orig_col, std_col in target_col_mapping.items():
        if orig_col in target_cols:
            stats_df[std_col] = bin_table[(target_name, orig_col)].values
    
    # 计算 KS 值（如果坏样本率和累计占比存在）
    if '累计坏样本占比' in stats_df.columns and '累计好样本占比' in stats_df.columns:
        stats_df['KS值'] = (stats_df['累计坏样本占比'] - stats_df['累计好样本占比']).abs()
    
    return stats_df


def bin_overdues_plot(
    data: pd.DataFrame,
    feature: Optional[str] = None,
    overdue: Optional[List[str]] = None,
    dpds: Optional[List[int]] = None,
    bin_table: Optional[pd.DataFrame] = None,
    method: str = 'quantile',
    max_n_bins: int = 10,
    min_bin_size: float = 0.02,
    rules: Optional[Dict] = None,
    shared_bins: Optional[Union[str, bool]] = 'max_samples',
    figsize: Optional[tuple] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_stats: bool = True,
    max_cols: int = 3,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制多个逾期天数的分箱图（横向展示）.

    支持两种输入方式：
    1. 原始数据 + overdue + dpds：根据原始数据计算分箱并绘图
    2. 分箱表（来自 feature_bin_stats）：直接解析多级表头分箱表并绘图

    :param data: 输入数据（原始数据模式）或分箱表（当传入 bin_table 时忽略）
    :param feature: 特征列名（原始数据模式需要）
    :param overdue: 逾期天数列名列表，如 ['dpd7', 'dpd15', 'dpd30']
    :param dpds: 逾期阈值列表，与 overdue 一一对应，如 [1, 1, 1]
        表示逾期天数>=该阈值时视为坏样本
    :param bin_table: 分箱表（来自 feature_bin_stats 的多级表头 DataFrame）
        传入后将直接使用分箱表绘图，忽略 data/overdue/dpds 参数
    :param method: 分箱方法，默认 'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 最小箱占比，默认0.02
    :param rules: 预定义分箱规则 {特征名: 分箱边界列表}
    :param shared_bins: 各逾期目标是否共享同一切分点，默认 'max_samples'
        - 'first': 使用第一个逾期定义的切分点
        - 'last': 使用最后一个逾期定义的切分点
        - 'max_samples': 使用有效样本量最多的逾期定义的切分点（默认）
        - False 或 None: 每个逾期定义独立计算切分点
    :param figsize: 图像尺寸，None时自动计算
    :param colors: 配色方案
    :param title: 图表总标题
    :param show_stats: 是否显示统计指标
    :param max_cols: 每行最多显示几个子图
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure

    Example:
        >>> # 方式1：使用原始数据
        >>> fig = bin_overdues_plot(
        ...     df,
        ...     feature='score',
        ...     overdue=['dpd7', 'dpd15', 'dpd30'],
        ...     dpds=[1, 1, 1],
        ...     max_n_bins=5
        ... )

        >>> # 方式2：使用 feature_bin_stats 生成的分箱表
        >>> from hscredit.report.feature_analyzer import feature_bin_stats
        >>> bin_table = feature_bin_stats(
        ...     df, 
        ...     feature='score', 
        ...     overdue=['MOB1', 'MOB3'], 
        ...     dpds=[0, 7]
        ... )
        >>> fig = bin_overdues_plot(bin_table=bin_table)
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 检查是否为分箱表模式
    if bin_table is not None:
        # 分箱表模式：直接解析多级表头分箱表
        if not _is_multiindex_bin_table(bin_table):
            raise ValueError("bin_table 必须是多级表头的分箱表（来自 feature_bin_stats）")
        
        # 提取目标名称列表
        target_names = _extract_target_names_from_bin_table(bin_table)
        
        if len(target_names) == 0:
            raise ValueError("分箱表中没有找到目标列（除了 '分箱详情'）")
        
        # 从分箱详情中提取特征名（使用第一个分箱行）
        if ('分箱详情', '指标名称') in bin_table.columns:
            feature = bin_table[('分箱详情', '指标名称')].iloc[0]
        else:
            feature = 'Feature'
        
        n_plots = len(target_names)
        
        # 计算行列数
        n_cols = min(max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # 自动计算图像尺寸
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # 处理单个子图的情况
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # 绘制每个目标的分箱图
        for idx, target_name in enumerate(target_names):
            ax = axes[idx]
            
            try:
                # 提取该目标的统计信息
                stats_df = _get_stats_for_target(bin_table, target_name)
                
                if stats_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(target_name)
                    continue
                
                # 格式化分箱标签
                if '分箱' in stats_df.columns:
                    stats_df['分箱'] = stats_df['分箱'].apply(lambda x: format_bin_label(x, 35))
                
                # 使用 bin_plot 绘制单个子图
                bin_plot(
                    data=stats_df,
                    ax=ax,
                    title=target_name,
                    colors=colors,
                    orientation='vertical'
                )
                
                # 添加统计信息（从分箱表中提取）
                if show_stats:
                    stats_parts = []
                    
                    # 计算 IV（IV值列的和）
                    if 'IV值' in stats_df.columns:
                        iv_val = stats_df['IV值'].sum()
                        stats_parts.append(f"IV: {iv_val:.3f}")
                    
                    # 获取 KS（KS值列的最大值）
                    if 'KS值' in stats_df.columns:
                        ks_val = stats_df['KS值'].max()
                        stats_parts.append(f"KS: {ks_val:.2f}")
                    
                    # 计算整体坏样本率
                    if '坏样本数' in stats_df.columns and '样本总数' in stats_df.columns:
                        total_bad = stats_df['坏样本数'].sum()
                        total_samples = stats_df['样本总数'].sum()
                        if total_samples > 0:
                            bad_rate = total_bad / total_samples
                            stats_parts.append(f"BadRate: {bad_rate:.2%}")
                    
                    if stats_parts:
                        stats_text = ", ".join(stats_parts)
                        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                               ha='right', va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(target_name)
        
        # 隐藏多余的子图
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        # 设置总标题
        if title is None:
            title = f"{feature} - Multi DPD Binning Analysis"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, save)
        
        return fig
    
    # 原始数据模式
    if feature is None:
        raise ValueError("原始数据模式需要提供 feature 参数")
    if overdue is None or dpds is None:
        raise ValueError("原始数据模式需要提供 overdue 和 dpds 参数")
    
    if len(overdue) != len(dpds):
        raise ValueError("overdue 和 dpds 长度必须一致")

    n_plots = len(overdue)

    # 计算行列数
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # 自动计算图像尺寸
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # 处理单个子图的情况
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_plots > 1 else [axes]

    # 计算全局分箱规则
    if rules is None or feature not in rules:
        global_rules = None
        if shared_bins:
            from ..binning import OptimalBinning
            _shared = str(shared_bins).lower()
            if _shared == 'first':
                ref_idx = 0
            elif _shared == 'last':
                ref_idx = len(overdue) - 1
            else:  # 'max_samples' 或其他真值
                valid_counts = []
                for dpd_col, threshold in zip(overdue, dpds):
                    y_tmp = (data[dpd_col] >= threshold).astype(int)
                    valid_counts.append((~(pd.isna(data[feature]) | pd.isna(y_tmp))).sum())
                ref_idx = int(np.argmax(valid_counts))

            dpd_col = overdue[ref_idx]
            threshold = dpds[ref_idx]
            y = (data[dpd_col] >= threshold).astype(int)
            valid_mask = ~(pd.isna(data[feature]) | pd.isna(y))
            X_valid = data.loc[valid_mask, feature]
            y_valid = y[valid_mask]

            binner = OptimalBinning(method=method, max_n_bins=max_n_bins, min_bin_size=min_bin_size, verbose=False)
            binner.fit(X_valid.to_frame(), y_valid)

            bin_edges = binner.splits_.get(feature, [])
            global_rules = {feature: list(bin_edges)} if len(bin_edges) > 0 else None
    else:
        global_rules = rules

    # 绘制每个逾期定义的分箱图
    for idx, (dpd_col, threshold) in enumerate(zip(overdue, dpds)):
        ax = axes[idx]

        try:
            # 创建二元目标变量
            y = (data[dpd_col] >= threshold).astype(int)

            # 计算分箱统计
            stats_df = _compute_bin_stats_from_raw_data(
                data=data,
                target=y,
                feature=feature,
                method=method,
                max_n_bins=max_n_bins,
                min_bin_size=min_bin_size,
                rules=global_rules.get(feature, None) if global_rules else None,
                **kwargs
            )

            if stats_df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{dpd_col} (>= {threshold})")
                continue

            # 格式化分箱标签
            stats_df['分箱'] = stats_df['分箱'].apply(lambda x: format_bin_label(x, 35))

            # 使用 bin_plot 绘制单个子图
            bin_plot(
                data=stats_df,
                ax=ax,
                title=f"{dpd_col} (>= {threshold})",
                colors=colors,
                orientation='vertical'
            )

            # 添加统计信息
            if show_stats:
                valid_mask = ~(pd.isna(data[feature]) | pd.isna(y))
                X_valid = data.loc[valid_mask, feature]
                y_valid = y[valid_mask]

                try:
                    from ..metrics import iv as iv_metric, ks as ks_metric
                    iv_val = iv_metric(X_valid, y_valid)
                    ks_val = ks_metric(X_valid, y_valid)
                    bad_rate = y_valid.mean()
                    stats_text = f"IV: {iv_val:.3f}, KS: {ks_val:.2f}, BadRate: {bad_rate:.2%}"
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                           ha='right', va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                except Exception:
                    pass

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{dpd_col} (>= {threshold})")

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # 设置总标题
    if title is None:
        title = f"{feature} - Multi DPD Binning Analysis"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        save_figure(fig, save)

    return fig


def _cross_heatmap_cell(
    ax,
    M: np.ndarray,
    base_color: str,
    *,
    fmt: str = '.1%',
    annot: bool = True,
    diverging: bool = False,
    fontsize: int = 10,
    axis_color: str = '#2639E9',
):
    """在指定 Axes 上绘制二维分箱交叉指标热力图（类似相关性图）.

    使用 imshow 将单元格中心对齐到整数坐标 (列=特征2 bin j -> x=j, 行=特征1 bin i -> y=nx-1-i)，
    从而与 ``bin_plot`` 的整数柱位置共用坐标系。所有数值以百分数标注。

    :param ax: 目标 Axes
    :param M: 指标矩阵，形状 (nx, ny)，M[i, j] 对应 特征1 bin i × 特征2 bin j
    :param base_color: 顺序型配色的基准色（diverging=False 时生效）
    :param fmt: 数值标注格式（百分数），如 '.1%'、'.2%'
    :param annot: 是否标注数值
    :param diverging: 是否使用以 0 为中心的发散配色（用于可正可负的指标，如坏账改善）
    :param fontsize: 标注字体大小
    :param axis_color: 坐标轴边框颜色
    :return: imshow 返回的 AxesImage
    """
    import copy as _copy
    import matplotlib.colors as mcolors

    nx, ny = M.shape
    # 行翻转：使图像第 r 行对应 y=r，即 特征1 bin (nx-1-r)，与 barh 分箱图保持一致
    A = np.flipud(M)
    finite = np.isfinite(M)

    if diverging:
        vabs = float(np.nanmax(np.abs(M))) if finite.any() else 1.0
        vabs = vabs if vabs > 0 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        cmap = sns.diverging_palette(240, 12, s=80, l=50, as_cmap=True)
    else:
        vmin = float(np.nanmin(M)) if finite.any() else 0.0
        vmax = float(np.nanmax(M)) if finite.any() else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-9
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.light_palette(base_color, as_cmap=True)

    cmap = _copy.copy(cmap)
    cmap.set_bad('#f0f0f0')

    im = ax.imshow(
        np.ma.masked_invalid(A), aspect='auto', cmap=cmap, norm=norm, origin='lower',
        extent=(-0.5, ny - 0.5, -0.5, nx - 0.5), interpolation='nearest',
    )

    if annot:
        for r in range(nx):
            for c in range(ny):
                v = A[r, c]
                if not np.isfinite(v):
                    continue
                rgba = cmap(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = 'white' if lum < 0.55 else '#222222'
                ax.text(c, r, format(float(v), fmt), ha='center', va='center',
                        fontsize=fontsize, color=text_color)

    # 白色网格线（参照相关性图风格）
    ax.set_xticks(np.arange(-0.5, ny, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nx, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', length=0)
    ax.set_xlim(-0.5, ny - 0.5)
    ax.set_ylim(-0.5, nx - 0.5)
    setup_axis_style(ax, [axis_color])
    ax.tick_params(axis='both', colors=axis_color)
    return im


def _set_cross_heat_ticklabels(
    ax,
    nx: int,
    ny: int,
    *,
    xlabels: Optional[List] = None,
    ylabels: Optional[List] = None,
    x_top: bool = False,
    y_right: bool = False,
    axis_color: str = '#2639E9',
    rotation: int = 35,
    fontsize: int = 9,
    max_len: int = 14,
):
    """设置交叉热力图的分箱刻度标签.

    :param xlabels: 特征2 分箱标签（按 bin 索引 0..ny-1 顺序），None 表示不显示
    :param ylabels: 特征1 分箱标签（按 bin 索引 0..nx-1 顺序），None 表示不显示
    :param x_top: 是否将 x 轴标签放到顶部
    :param y_right: 是否将 y 轴标签放到右侧
    """
    if xlabels is not None:
        ax.set_xticks(np.arange(ny))
        ax.set_xticklabels([format_bin_label(str(lbl), max_len) for lbl in xlabels],
                           rotation=rotation, ha='left' if x_top else 'right',
                           fontsize=fontsize, color=axis_color)
        if x_top:
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
    else:
        ax.set_xticks([])

    if ylabels is not None:
        ax.set_yticks(np.arange(nx))
        # 刻度位置 p 对应 特征1 bin (nx-1-p)
        labels = [format_bin_label(str(ylabels[nx - 1 - p]), max_len) for p in range(nx)]
        ax.set_yticklabels(labels, fontsize=fontsize, color=axis_color)
        if y_right:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
    else:
        ax.set_yticks([])


def bin_2d_plot(
    data,
    features: Optional[List[str]] = None,
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    *,
    binner=None,
    method: str = 'quantile',
    max_n_bins: int = 5,
    min_bin_size: Union[float, int] = 0.02,
    figsize: Optional[tuple] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    annot: bool = True,
    fontsize: int = 10,
    save: Optional[str] = None,
    binner_kwargs: Optional[Dict] = None,
):
    """两个变量交叉分箱联合分析图（3×3 布局）.

    布局（特征1 为行维度，特征2 为列维度）：

    .. list-table::
       :widths: 1 1 1
       :header-rows: 0

       * - KS 曲线
         - 特征2分箱图
         - 风险拒绝比
       * - 样本占比
         - 坏样本率
         - 特征1分箱图
       * - LIFT
         - 坏账改善
         - KS 曲线

    - 两个单变量分箱图（复用 :func:`bin_plot`，与交叉热力图共用坐标系）：特征2分箱图
      （纵向，bin 落在 x 轴）置于第1行中列，与同列热力图（坏样本率/坏账改善）按列对齐；
      特征1分箱图（横向，bin 落在 y 轴）置于第2行右列，与同行热力图（样本占比/坏样本率）
      按行对齐
    - 两个 KS 曲线复用 :func:`ks_plot` （``curve='ks'``，仅 KS 曲线，去掉 ROC），分置左上、右下角
    - 其余 5 格为两变量分箱交叉指标热力图（类似相关性图，均以百分数标注）：
      样本占比、坏样本率、LIFT、风险拒绝比、坏账改善

    支持两种输入方式：

    **方式1：原始数据**

    >>> bin_2d_plot(df, features=['特征1', '特征2'], target='target')

    **方式2：已拟合的 OptimalBinning2D**

    >>> from hscredit.core.binning import OptimalBinning2D
    >>> b = OptimalBinning2D(max_n_bins=5).fit(df, y=df['target'], features=['f1', 'f2'])
    >>> bin_2d_plot(b)

    :param data: DataFrame（方式1）或已拟合的 OptimalBinning2D（方式2）
    :param features: [特征1, 特征2]，特征1 为行维度，特征2 为列维度（方式1 需要）
    :param target: 目标列名或数组（方式1 需要）
    :param binner: 已拟合的 OptimalBinning2D（可选，优先级高于由 data 构造）
    :param method: 分箱方法（方式1 构造 OptimalBinning2D 时使用）
    :param max_n_bins: 最大分箱数（方式1）
    :param min_bin_size: 每箱最小样本占比（方式1）
    :param figsize: 图像尺寸，None 时根据分箱数自动计算
    :param colors: 配色方案
    :param title: 图表总标题
    :param annot: 热力图是否标注数值
    :param fontsize: 单元格字体大小
    :param save: 保存路径
    :param binner_kwargs: 透传给 OptimalBinning2D 的其他参数（方式1）
    :return: matplotlib Figure
    """
    from ..binning import OptimalBinning2D

    if colors is None:
        colors = DEFAULT_COLORS
    axis_color = colors[0]
    eps = 1e-10

    # ---------- 解析输入，获取已拟合的 OptimalBinning2D ----------
    if isinstance(data, OptimalBinning2D):
        b2d = data
    elif binner is not None and isinstance(binner, OptimalBinning2D):
        b2d = binner
    else:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("方式1的 data 必须为 DataFrame，或直接传入已拟合的 OptimalBinning2D")
        if features is None or len(features) != 2:
            raise ValueError("方式1需提供两个特征名: features=['特征1', '特征2']")
        if target is None:
            raise ValueError("方式1需提供 target（目标列名或数组）")
        _kw = dict(binner_kwargs or {})
        b2d = OptimalBinning2D(
            method=_kw.pop('method', method),
            max_n_bins=_kw.pop('max_n_bins', max_n_bins),
            min_bin_size=_kw.pop('min_bin_size', min_bin_size),
            **_kw,
        )
        if isinstance(target, str):
            y_series = data[target]
        else:
            y_series = pd.Series(np.asarray(target).reshape(-1), index=data.index, name='target')
        b2d.fit(data, y=y_series, features=list(features))

    if not getattr(b2d, '_is_fitted', False):
        raise NotFittedError("OptimalBinning2D 尚未拟合，请先调用 fit 方法")

    feat_x = b2d.feature_x_   # 特征1（行维度）
    feat_y = b2d.feature_y_   # 特征2（列维度）
    nx = b2d.n_bins_x_
    ny = b2d.n_bins_y_
    cross = b2d.cross_table_.copy()
    cross = cross[(cross['特征1分箱'] >= 0) & (cross['特征2分箱'] >= 0)].copy()
    X = b2d._X
    y_arr = np.asarray(b2d._y, dtype=float)

    # ---------- 计算交叉指标矩阵 M[i, j]（行=特征1 bin i, 列=特征2 bin j） ----------
    total = float(cross['样本总数'].sum())
    total_bad = float(cross['坏样本数'].sum())
    total_good = total - total_bad
    overall_bad_rate = total_bad / total if total > 0 else 0.0

    work = cross.copy()
    # 坏账改善 = (全量坏样本率 - 拒绝该格后剩余样本坏样本率) / 全量坏样本率
    other_bad = total_bad - work['坏样本数']
    other_total = total - work['样本总数']
    other_bad_rate = np.where(other_total > 0, other_bad / other_total, 0.0)
    work['坏账改善'] = np.where(
        overall_bad_rate > 0, (overall_bad_rate - other_bad_rate) / overall_bad_rate, 0.0)
    # 风险拒绝比 = 坏账改善 / 当前格样本占比
    work['风险拒绝比'] = np.where(
        work['样本占比'] > eps, work['坏账改善'] / work['样本占比'], 0.0)

    def _matrix(col):
        M = np.full((nx, ny), np.nan)
        for _, r in work.iterrows():
            M[int(r['特征1分箱']), int(r['特征2分箱'])] = r[col]
        return M

    M_prop = _matrix('样本占比')
    M_bad = _matrix('坏样本率')
    M_lift = _matrix('LIFT值')
    M_reject = _matrix('风险拒绝比')
    M_improve = _matrix('坏账改善')

    # ---------- 由交叉表聚合出单变量边缘分箱表（保证与热力图行/列严格对齐） ----------
    def _marginal(bin_col, label_col):
        grp = work.groupby(bin_col, sort=True).agg(
            样本总数=('样本总数', 'sum'),
            好样本数=('好样本数', 'sum'),
            坏样本数=('坏样本数', 'sum'),
        )
        grp['分箱标签'] = work.groupby(bin_col, sort=True)[label_col].first()
        grp = grp.reset_index().rename(columns={bin_col: '分箱'}).sort_values('分箱')
        grp['坏样本率'] = np.where(grp['样本总数'] > 0, grp['坏样本数'] / grp['样本总数'], 0.0)
        grp['样本占比'] = grp['样本总数'] / total if total > 0 else 0.0
        good_distr = grp['好样本数'] / total_good if total_good > 0 else 0.0
        bad_distr = grp['坏样本数'] / total_bad if total_bad > 0 else 0.0
        woe = np.log((bad_distr + eps) / (good_distr + eps))
        grp['分档WOE值'] = woe
        grp['分档IV值'] = (bad_distr - good_distr) * woe
        grp['指标IV值'] = grp['分档IV值'].sum()
        grp['LIFT值'] = np.where(grp['坏样本率'] > 0, grp['坏样本率'] / overall_bad_rate, 0.0)
        return grp

    marg_x = _marginal('特征1分箱', '特征1标签')   # 特征1
    marg_y = _marginal('特征2分箱', '特征2标签')   # 特征2
    xlabels = marg_y['分箱标签'].tolist()      # 特征2 (列, x), bin 索引 0..ny-1
    ylabels = marg_x['分箱标签'].tolist()      # 特征1 (行, y), bin 索引 0..nx-1

    # ---------- 构建画布 ----------
    if figsize is None:
        figsize = (max(15.0, 9.0 + 0.9 * ny), max(13.0, 8.0 + 0.8 * nx))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, left=0.105, right=0.975, top=0.910, bottom=0.180,
                          hspace=0.18, wspace=0.03)

    ax_ks_y = fig.add_subplot(gs[0, 0])     # 特征2 KS 曲线
    ax_marg_y = fig.add_subplot(gs[0, 1])   # 特征2分箱图（纵向）
    ax_reject = fig.add_subplot(gs[0, 2])   # 风险拒绝比
    ax_prop = fig.add_subplot(gs[1, 0])     # 样本占比
    ax_bad = fig.add_subplot(gs[1, 1])      # 坏样本率
    ax_marg_x = fig.add_subplot(gs[1, 2])   # 特征1分箱图（横向）
    ax_lift = fig.add_subplot(gs[2, 0])     # LIFT
    ax_improve = fig.add_subplot(gs[2, 1])  # 坏账改善
    ax_ks_x = fig.add_subplot(gs[2, 2])     # 特征1 KS 曲线

    def _cell_title(ax, text):
        ax.set_title(text, fontsize=fontsize + 2, color=axis_color, fontweight='semibold', pad=6)

    # ---------- 5 个交叉指标热力图 ----------
    _cross_heatmap_cell(ax_prop, M_prop, '#2639E9', fmt='.1%', annot=annot,
                        fontsize=fontsize, axis_color=axis_color)
    _cell_title(ax_prop, '样本占比')
    _cross_heatmap_cell(ax_bad, M_bad, '#E85D4A', fmt='.2%', annot=annot,
                        fontsize=fontsize, axis_color=axis_color)
    _cell_title(ax_bad, '坏样本率')
    _cross_heatmap_cell(ax_lift, M_lift, '#FE7715', fmt='.1%', annot=annot,
                        fontsize=fontsize, axis_color=axis_color)
    _cell_title(ax_lift, 'LIFT')
    _cross_heatmap_cell(ax_reject, M_reject, '#2639E9', fmt='.1%', annot=annot,
                        fontsize=fontsize, axis_color=axis_color)
    _cell_title(ax_reject, '风险拒绝比')
    _cross_heatmap_cell(ax_improve, M_improve, '#2639E9', fmt='.1%', annot=annot,
                        diverging=True, fontsize=fontsize, axis_color=axis_color)
    _cell_title(ax_improve, '坏账改善')

    # 热力图刻度标签：左列(样本占比/LIFT)显示 特征1(y)，底行(LIFT/坏账改善)显示 特征2(x)；
    # 坏样本率、风险拒绝比不显示刻度标签
    _set_cross_heat_ticklabels(ax_prop, nx, ny, ylabels=ylabels, axis_color=axis_color)
    _set_cross_heat_ticklabels(ax_bad, nx, ny, axis_color=axis_color)
    _set_cross_heat_ticklabels(ax_lift, nx, ny, xlabels=xlabels, ylabels=ylabels,
                               rotation=90, axis_color=axis_color)
    _set_cross_heat_ticklabels(ax_improve, nx, ny, xlabels=xlabels,
                               rotation=90, axis_color=axis_color)
    _set_cross_heat_ticklabels(ax_reject, nx, ny, axis_color=axis_color)

    # ---------- 两个单变量分箱图（复用 bin_plot，与热力图共用坐标系） ----------
    # 去坏样本率坐标轴刻度；紧凑布局下样本数刻度也隐藏
    # （分布形态由柱高、坏样本率由数值标注体现）
    bin_plot(marg_y, ax=ax_marg_y, orientation='vertical', colors=colors,
             show_metric_summary=False, show_rate_axis=False, title='分箱图')
    ax_marg_y_rate = fig.axes[-1]
    ax_marg_y.set_xlim(-0.5, ny - 0.5)
    # bin_2d_plot 内单独覆盖标题样式，不改变 bin_plot 的默认标题颜色
    _cell_title(ax_marg_y, '分箱图')
    # 特征2分箱标签统一由底行热力图垂直展示，边缘图关闭重复标签
    for marginal_ax in (ax_marg_y, ax_marg_y_rate):
        marginal_ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
    ax_marg_y.set_ylabel('')

    bin_plot(marg_x, ax=ax_marg_x, orientation='horizontal', colors=colors,
             show_metric_summary=False, show_rate_axis=False, title='分箱图')
    ax_marg_x_rate = fig.axes[-1]
    ax_marg_x.set_ylim(-0.5, nx - 0.5)
    _cell_title(ax_marg_x, '分箱图')
    for marginal_ax in (ax_marg_x, ax_marg_x_rate):
        marginal_ax.tick_params(axis='y', left=False, labelleft=False)
    ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_marg_x.set_xlabel('')

    # ---------- 两个 KS 曲线（仅 KS 曲线，去掉 ROC；按特征方向保留单侧坐标） ----------
    def _draw_feature_ks(ax, binner_1d, feat, show_x=False, show_y=False):
        try:
            woe = binner_1d.transform(X[[feat]], metric='woe')[feat].to_numpy(dtype=float)
            mask = np.isfinite(woe) & np.isfinite(y_arr)
            if mask.sum() > 0 and len(np.unique(y_arr[mask])) == 2:
                ks_plot(woe[mask], y_arr[mask], curve='ks', ax=ax,
                        fontsize=max(fontsize - 1, 9), title='', colors=colors)
            else:
                ax.text(0.5, 0.5, 'KS 不可用', ha='center', va='center', transform=ax.transAxes)
        except Exception as exc:  # pragma: no cover - 防御性兜底
            ax.text(0.5, 0.5, f'KS 异常: {exc}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
        # 去图例；特征2保留纵坐标，特征1保留横坐标
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        # 组合图只保留要求方向的数值刻度，去掉英文轴名以避免挤压相邻子图
        ax.set_xlabel('')
        ax.set_ylabel('')
        if not show_x:
            ax.set_xticks([])
        if not show_y:
            ax.set_yticks([])
        setup_axis_style(ax, [axis_color])
        ax.tick_params(axis='both', colors=axis_color, labelsize=max(fontsize - 2, 7))
        _cell_title(ax, f'{feat} KS曲线')

    _draw_feature_ks(ax_ks_y, b2d.binner_y_, feat_y, show_y=True)
    _draw_feature_ks(ax_ks_x, b2d.binner_x_, feat_x, show_x=True)

    # ---------- 总标题与全局轴标签 ----------
    if title is None:
        title = f'{feat_x} × {feat_y} 二维交叉分箱分析'
    fig.suptitle(title, fontsize=fontsize + 6, fontweight='bold', y=0.965)
    ax_improve.set_xlabel(f'特征2：{feat_y}', fontsize=fontsize + 3, color=axis_color,
                          fontweight='semibold', labelpad=12)
    # 使用轴标签的自动布局能力，将特征名称放到刻度标签外侧，避免固定位置造成重叠
    ax_prop.set_ylabel(f'特征1：{feat_x}', fontsize=fontsize + 3, color=axis_color,
                       fontweight='semibold', labelpad=12)

    save_figure(fig, save)
    return fig
