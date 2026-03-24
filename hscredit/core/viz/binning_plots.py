# -*- coding: utf-8 -*-
"""
评分卡可视化函数.

提供常用的绘图功能，包括分箱图、KS/ROC曲线、分布图、PSI/CSI分析图等。
"""

import re
import warnings
import os
import six
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Union, Optional, List, Any

from .utils import (
    DEFAULT_COLORS, setup_axis_style, save_figure, 
    get_or_create_ax, create_legend, format_bin_label
)


def _is_feature_table(data):
    """判断是否为特征分箱统计表"""
    required_cols = ['分箱', '好样本数', '坏样本数', '样本总数', '坏样本率']
    if isinstance(data, pd.DataFrame):
        return all(col in data.columns for col in required_cols)
    return False


def _compute_bin_stats_from_raw_data(
    data: Union[pd.DataFrame, pd.Series],
    target: Union[str, pd.Series, np.ndarray],
    feature: Optional[str] = None,
    n_bins: int = 10,
    method: str = 'quantile',
    rules: Optional[List] = None,
) -> pd.DataFrame:
    """从原始数据计算分箱统计表
    
    :param data: 特征数据（DataFrame 或 Series）
    :param target: 目标变量（列名或数据）
    :param feature: 特征列名（当 data 为 DataFrame 时需要）
    :param n_bins: 分箱数量
    :param method: 分箱方法
    :param rules: 自定义分箱边界
    :return: 分箱统计表
    """
    # 处理输入数据
    if isinstance(data, pd.Series):
        feature_name = data.name if data.name is not None else 'feature'
        X = data.values
    elif isinstance(data, pd.DataFrame):
        if feature is None:
            # 如果没有指定特征列，使用第一列
            feature = data.columns[0]
        feature_name = feature
        X = data[feature].values
    else:
        feature_name = 'feature'
        X = np.array(data)
    
    # 处理目标变量
    if isinstance(target, str):
        if isinstance(data, pd.DataFrame) and target in data.columns:
            y = data[target].values
        else:
            raise ValueError(f"目标列 '{target}' 不在数据中")
    elif isinstance(target, (pd.Series, np.ndarray)):
        y = np.array(target)
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
    
    # 分箱
    if rules is not None:
        # 使用自定义分箱边界
        bins = rules
        if not np.isinf(bins[-1]):
            bins = list(bins) + [np.inf]
        if not np.isinf(bins[0]) and bins[0] > -np.inf:
            bins = [-np.inf] + list(bins)
    else:
        # 自动分箱
        if method == 'quantile':
            bins = np.percentile(X_valid, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)  # 去重
            bins[0] = -np.inf
            bins[-1] = np.inf
        elif method == 'uniform':
            bins = np.linspace(np.min(X_valid), np.max(X_valid), n_bins + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf
        else:
            # 默认使用等频分箱
            bins = np.percentile(X_valid, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            bins[0] = -np.inf
            bins[-1] = np.inf
    
    # 生成分箱标签
    bin_labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            label = f"(-∞, {bins[i+1]:.2f})"
        elif i == len(bins) - 2:
            label = f"[{bins[i]:.2f}, +∞)"
        else:
            label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
        bin_labels.append(label)
    
    # 分配样本到各箱
    bin_indices = np.digitize(X_valid, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
    
    # 计算统计信息
    results = []
    for i, label in enumerate(bin_labels):
        mask = bin_indices == i
        total = np.sum(mask)
        good = np.sum((y_valid[mask] == 0)) if total > 0 else 0
        bad = np.sum((y_valid[mask] == 1)) if total > 0 else 0
        bad_rate = bad / total if total > 0 else 0
        
        results.append({
            '分箱': label,
            '样本总数': total,
            '好样本数': good,
            '坏样本数': bad,
            '坏样本率': bad_rate,
            '样本占比': total / len(X_valid)
        })
    
    return pd.DataFrame(results)


def bin_plot(
    data: Union[pd.DataFrame, pd.Series],
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    feature: Optional[str] = None,
    desc: str = "",
    figsize: tuple = (10, 6),
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
    iv: bool = True,
    return_frame: bool = False,
    ax: Optional[Any] = None,
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
    :param iv: 是否显示 IV 值（暂不支持）
    :param return_frame: 是否返回分箱统计表
    :param ax: 可选的 matplotlib Axes 对象，用于在已有画布上绘图
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
            n_bins=n_bins,
            method=method,
            rules=rules,
        )

    # 处理分箱标签
    feature_table["分箱"] = feature_table["分箱"].apply(
        lambda x: format_bin_label(x, max_len)
    )

    # 获取或创建 Axes
    if ax is not None:
        ax1 = ax
        fig = ax.figure
        return_ax = True
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        return_ax = False

    # 绘图
    ax1.barh(feature_table['分箱'], feature_table['好样本数'], color=colors[0], label='好样本', 
             hatch="/" if hatch else None, edgecolor='white' if hatch else None)
    ax1.barh(feature_table['分箱'], feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1], 
             label='坏样本', hatch="\\" if hatch else None, edgecolor='white' if hatch else None)
    ax1.set_xlabel('样本数')

    ax2 = ax1.twiny()
    ax2.plot(feature_table['坏样本率'], feature_table['分箱'], colors[2], label='坏样本率', linestyle='-.')
    ax2.set_xlabel('坏样本率: 坏样本数 / 样本总数')
    ax2.set_xlim(xmin=0.)

    if show_data_points:
        for i, rate in enumerate(feature_table['坏样本率']):
            ax2.scatter(rate, i, color=colors[2])

    if fontdict and fontdict.get("color"):
        for i, v in feature_table[['样本总数', '好样本数', '坏样本数', '坏样本率']].iterrows():
            ax1.text(v['样本总数'] / 2, i + len(feature_table) / 60, 
                    f"{int(v['好样本数'])}:{int(v['坏样本数'])}:{v['坏样本率']:.2%}", fontdict=fontdict)

    ax1.invert_yaxis()
    ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))
    
    if not return_ax:
        if title is not None:
            fig.suptitle(f'{title}\n\n')
        else:
            fig.suptitle(f'{desc}{ending}\n\n')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
                   ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        plt.tight_layout()
        save_figure(fig, save)

        if return_frame:
            return fig, feature_table
        return fig
    else:
        if title is not None:
            ax1.set_title(title)
        else:
            ax1.set_title(f'{desc}{ending}')
        return ax1


def corr_plot(data, figure_size=(16, 8), fontsize=16, mask=False, save=None, 
              annot=True, max_len=35, linewidths=0.1, fmt='.2f', step=11, linecolor='white', 
              ax=None, **kwargs):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
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
            colors=None, anchor=0.945, axes=None):
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
    :return: matplotlib Figure 或 Axes 数组
    """
    if colors is None:
        colors = DEFAULT_COLORS

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
        label_0, label_1 = unique_labels[0], unique_labels[1]
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

    # 获取或创建 Axes
    if axes is not None:
        # 检查是否为单个 Axes（使用 matplotlib 的 Axes 类型判断）
        if hasattr(axes, 'plot') and not hasattr(axes, '__len__'):
            # 传入的是单个 Axes，只用第一个子图
            fig = axes.figure
            ax1 = axes
            # 创建第二个 axes 用于 ROC 曲线
            ax2 = fig.add_subplot(122)
            return_axes = True
        elif hasattr(axes, '__len__') and len(axes) >= 2:
            # 传入的是 Axes 列表/数组
            ax1, ax2 = axes[0], axes[1]
            fig = ax1.figure
            return_axes = True
        else:
            # 其他情况，创建新图
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = ax[0], ax[1]
            return_axes = False
    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = ax[0], ax[1]
        return_axes = False

    # KS曲线
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
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    handles2, labels2 = ax2.get_legend_handles_labels()

    if not return_axes:
        if title:
            title += " "
        fig.suptitle(f"{title}K-S & ROC CURVE\n", fontsize=fontsize, fontweight="bold")

        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
                   ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        plt.tight_layout()
        save_figure(fig, save)
        return fig
    else:
        return axes


def hist_plot(score, y_true=None, figsize=(15, 10), bins=30, save=None,
              labels=None, desc="", anchor=1.11, fontsize=14, kde=False, title=None,
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

    sns.histplot(
        x=score, hue=y_true, element="step", stat="probability", bins=bins,
        common_bins=True, common_norm=True, palette=palette, hue_order=hue_order_final,
        ax=ax, kde=kde, **kwargs
    )

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
        ax.legend([t for t in hue_order for _ in range(2)] if kde else hue_order,
                  loc='upper center', ncol=target_unique * 2 if kde else target_unique,
                  bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)

    if not return_ax:
        fig.tight_layout()
        save_figure(fig, save)
        return fig
    else:
        return ax


def psi_plot(expected, actual, labels=None, desc="", save=None, colors=None, 
             figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, 
             max_len=None, hatch=True, title=None):
    """
    PSI稳定性分析图.

    :param expected: 期望分布
    :param actual: 实际分布
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

    expected = expected.rename(columns={
        "样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", 
        "坏样本率": f"{labels[0]}坏样本率"
    })
    actual = actual.rename(columns={
        "样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", 
        "坏样本率": f"{labels[1]}坏样本率"
    })
    df_psi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
    df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(
        df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"]
    )
    df_psi["分档PSI值"] = df_psi[f"{labels[1]}% - {labels[0]}%"] * df_psi[f"ln({labels[1]}% / {labels[0]}%)"]
    df_psi = df_psi.fillna(0).replace([np.inf, -np.inf], 0)
    df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
    df_psi["指标名称"] = desc

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

    mpl_table = ax.table(
        cellText=data.values, colWidths=cols_width, bbox=bbox, 
        colLabels=data.columns, **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_font_size(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
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
                    colors=None, freq="M", anchor=0.94, result=False, hatch=True):
    """
    样本时间分布图.

    :param data: 数据集
    :param date: 日期列名
    :param target: 目标列名
    :param save: 保存路径
    :param figsize: 图像尺寸
    :param colors: 配色
    :param freq: 日期频率
    :param anchor: 图例位置
    :param result: 是否返回统计表
    :param hatch: 是否显示斜线
    :return: matplotlib Figure or pd.DataFrame
    """
    if colors is None:
        colors = DEFAULT_COLORS

    df = data.copy()

    if 'time' not in str(df[date].dtype):
        df[date] = pd.to_datetime(df[date])

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
    ax1.set_title('不同时点数据集样本分布情况\n\n')

    ax2 = plt.twinx()
    (temp["坏样本"] / temp.sum(axis=1)).plot(
        ax=ax2, color=colors[-1], style="--", linewidth=2, label="坏样本率"
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
