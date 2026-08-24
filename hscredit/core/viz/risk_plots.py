# -*- coding: utf-8 -*-
"""
金融风控数据可视化函数.

提供金融建模和风控策略分析专用的可视化功能，包括：
- ROC曲线图 (roc_plot)
- Lift提升图 (lift_plot)
- Gain增益图 (gain_plot)
- 评分分布对比图 (score_dist_plot)
- 坏样本率趋势图 (bad_rate_trend_plot)
- 特征重要性图 (feature_importance_plot)
- 混淆矩阵图 (confusion_matrix_plot)
- PR曲线图 (pr_plot)
- 校准曲线图 (calibration_plot)
- Vintage账龄曲线图 (vintage_plot)
- 决策阈值分析图 (threshold_analysis_plot)
- 策略效果对比图 (strategy_compare_plot)

采用函数式API设计，与hscredit.core.viz模块风格保持一致。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve
from matplotlib.colors import to_hex
from matplotlib.ticker import PercentFormatter

from .utils import (
    DEFAULT_COLORS, setup_axis_style, save_figure,
    get_or_create_ax, BAD_RATE_COLOR, NEUTRAL_COLOR,
    get_series_colors, make_colormap, make_risk_cmap,
    _layout_top_center_legend,
)
from ..._lazy import LazyModule

# 延迟加载 seaborn：仅在首次实际绘图（访问 sns 属性）时才导入，
# 避免 import hscredit 时即触发 seaborn（及其 ipywidgets/IPython 依赖）的加载。
sns = LazyModule("seaborn")


# ==================== 模型评估图表 ====================

def roc_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "ROC Curve",
    colors: Optional[List[str]] = None,
    show_auc: bool = True,
    show_diagonal: bool = True,
    label: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制ROC曲线.
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param ax: matplotlib Axes对象，None时自动创建
    :param figsize: 图像尺寸，默认(8, 8)
    :param title: 图表标题
    :param colors: 配色方案
    :param show_auc: 是否显示AUC值
    :param show_diagonal: 是否显示对角线（随机猜测线）
    :param label: 曲线标签（多模型对比时使用）
    :param save: 保存路径
    :param kwargs: 其他参数传递给plt.plot
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = roc_plot(y_test, model.predict_proba(X_test)[:, 1])
    >>> 
    >>> # 多模型对比
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> roc_plot(y_test, model1.predict_proba(X_test)[:, 1], ax=ax, label='Model A')
    >>> roc_plot(y_test, model2.predict_proba(X_test)[:, 1], ax=ax, label='Model B')
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 绘制对角线
    if show_diagonal:
        ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle='--',
                lw=1, alpha=0.5, label='Random (AUC = 0.50)')
    
    # 绘制ROC曲线
    label_str = label if label else 'Model'
    if show_auc:
        label_str += f' (AUC = {roc_auc:.3f})'
    
    ax.plot(fpr, tpr, color=colors[0], lw=2, label=label_str, **kwargs)
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3)
    
    if save:
        save_figure(fig, save)
    
    return fig


def pr_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "Precision-Recall Curve",
    colors: Optional[List[str]] = None,
    show_ap: bool = True,
    show_baseline: bool = True,
    label: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制Precision-Recall曲线.
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸，默认(8, 8)
    :param title: 图表标题
    :param colors: 配色方案
    :param show_ap: 是否显示Average Precision
    :param show_baseline: 是否显示基线（随机猜测）
    :param label: 曲线标签
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = pr_plot(y_test, model.predict_proba(X_test)[:, 1])
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    # 计算基线（正样本比例）
    if show_baseline:
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color=NEUTRAL_COLOR, linestyle='--',
                   alpha=0.5, label=f'Baseline ({baseline:.2%})')
    
    # 绘制PR曲线
    label_str = label if label else 'Model'
    if show_ap:
        from sklearn.metrics import average_precision_score
        ap_score = average_precision_score(y_true, y_score)
        label_str += f' (AP = {ap_score:.3f})'
    
    ax.plot(recall, precision, color=colors[0], lw=2, label=label_str, **kwargs)
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3)
    
    if save:
        save_figure(fig, save)
    
    return fig


def lift_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Lift 提升图",
    colors: Optional[List[str]] = None,
    show_baseline: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制Lift提升图.
    
    Lift = (该分箱坏样本率) / (整体坏样本率)
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param n_bins: 分箱数，默认10
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_baseline: 是否显示基线（Lift=1）
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = lift_plot(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true 和 y_score 必须是一维数组")
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true 与 y_score 长度不一致: {len(y_true)} != {len(y_score)}")
    valid_mask = ~pd.isna(y_true) & ~pd.isna(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    if len(y_true) == 0:
        raise ValueError("y_true 和 y_score 没有可用的非缺失数据")
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2 or not set(unique_labels).issubset({0, 1, False, True}):
        raise ValueError("y_true 必须是包含 0/1 的二分类标签")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("分箱数 n_bins 必须是正整数")
    if n_bins > len(y_true):
        raise ValueError(f"分箱数 ({n_bins}) 不能大于有效样本数 ({len(y_true)})")

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 计算Lift
    overall_bad_rate = np.mean(y_true)
    
    # 按分数排序分箱
    sorted_indices = np.argsort(-y_score)  # 降序
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # 计算每个分箱的Lift
    lifts = []
    depths = []
    end = 0

    for bin_values in np.array_split(y_true_sorted, n_bins):
        end += len(bin_values)
        bin_bad_rate = np.mean(bin_values)
        lift = bin_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 1.0

        lifts.append(lift)
        depths.append((end / len(y_true)) * 100)
    
    # 绘制基线
    if show_baseline:
        ax.axhline(y=1, color=NEUTRAL_COLOR, linestyle='--',
                   alpha=0.5, label='基准线（Lift=1）')
    
    # 绘制Lift曲线
    ax.plot(depths, lifts, color=colors[0], marker='o', lw=2, markersize=6, **kwargs)
    
    # 绘制柱状图
    ax.bar(depths, lifts, width=8, alpha=0.3, color=colors[0], edgecolor=colors[0])
    
    # 设置图表属性
    ax.set_xlabel('样本深度（累计占比）', fontsize=12)
    ax.set_ylabel('Lift 值', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 105])
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save:
        save_figure(fig, save)
    
    return fig


def gain_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Cumulative Gain Chart",
    colors: Optional[List[str]] = None,
    show_baseline: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制累积Gain增益图.
    
    Gain表示捕获的坏样本比例。
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param n_bins: 分箱数，默认10
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_baseline: 是否显示基线（随机模型）
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = gain_plot(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 按分数排序
    sorted_indices = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    total_bads = np.sum(y_true)
    
    # 计算累积Gain
    bin_size = len(y_true) // n_bins
    cumulative_gains = [0]
    depths = [0]
    
    for i in range(n_bins):
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        captured_bads = np.sum(y_true_sorted[:end])
        gain = captured_bads / total_bads if total_bads > 0 else 0
        
        cumulative_gains.append(gain * 100)
        depths.append((end / len(y_true)) * 100)
    
    # 绘制基线（随机模型）
    if show_baseline:
        ax.plot([0, 100], [0, 100], color=NEUTRAL_COLOR, linestyle='--',
                alpha=0.5, label='Baseline (Random)')
    
    # 绘制Gain曲线
    ax.plot(depths, cumulative_gains, color=colors[0], marker='o', 
            lw=2, markersize=6, label='Model', **kwargs)
    ax.fill_between(depths, cumulative_gains, alpha=0.2, color=colors[0])
    
    # 设置图表属性
    ax.set_xlabel('% of Population (Cumulative)', fontsize=12)
    ax.set_ylabel('% of Bad Samples Captured', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    
    ax.legend(loc='lower right', frameon=True)
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3)
    
    if save:
        save_figure(fig, save)
    
    return fig


def confusion_matrix_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "混淆矩阵",
    cmap: Optional[Any] = None,
    normalize: Optional[str] = None,
    show_values: bool = True,
    show_metrics: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制混淆矩阵热力图.
    
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param cmap: 颜色映射
    :param normalize: 归一化方式，None/'true'/'pred'/'all'
    :param show_values: 是否显示数值
    :param show_metrics: 是否显示评估指标
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = confusion_matrix_plot(y_test, y_pred)
    >>> fig = confusion_matrix_plot(y_test, y_pred, normalize='true')
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    if cmap is None:
        cmap = make_colormap("hscredit_confusion", ["#F7F8FF", DEFAULT_COLORS[0]])
    
    # 计算混淆矩阵；标签来自真实值与预测值并集，兼容二分类和多分类。
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm_counts.copy()
    
    # 归一化
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    elif normalize is not None:
        raise ValueError("normalize 仅支持 None/'true'/'pred'/'all'")
    
    # 绘制热力图
    sns.heatmap(cm, annot=show_values, fmt='.2f' if normalize else 'd',
                cmap=cmap, square=True, ax=ax,
                xticklabels=[str(label) for label in labels],
                yticklabels=[str(label) for label in labels],
                **kwargs)
    
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 显示评估指标
    if show_metrics:
        accuracy = accuracy_score(y_true, y_pred)
        if len(labels) == 2:
            tn, fp, fn, tp = cm_counts.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics_text = (
            f'准确率: {accuracy:.3f} | 精确率: {precision:.3f} | '
            f'召回率: {recall:.3f} | F1: {f1:.3f}'
        )
        ax.set_title(f'{title}\n{metrics_text}', fontsize=12, fontweight='bold')
    
    if save:
        save_figure(fig, save)
    
    return fig


def calibration_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "校准曲线",
    colors: Optional[List[str]] = None,
    show_histogram: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制校准曲线（可靠性图）.
    
    评估模型预测概率的可靠性。
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param n_bins: 分箱数，默认10
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_histogram: 是否显示样本分布直方图
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = calibration_plot(y_test, model.predict_proba(X_test)[:, 1])
    """
    if colors is None:
        colors = DEFAULT_COLORS

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true 和 y_score 必须是一维数组")
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true 与 y_score 长度不一致: {len(y_true)} != {len(y_score)}")
    valid_mask = ~pd.isna(y_true) & ~pd.isna(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    if len(y_true) == 0:
        raise ValueError("y_true 和 y_score 没有可用的非缺失数据")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins 必须是正整数")

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    ax_hist = ax.twinx() if show_histogram else None

    bin_accuracies, mean_probabilities = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy='uniform',
    )
    
    # 绘制完美校准线
    ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle='--',
            label='完美校准')
    
    # 绘制校准曲线
    brier = brier_score_loss(y_true, y_score)
    ax.plot(mean_probabilities, bin_accuracies, 's-', color=colors[0],
            label=f'模型（Brier={brier:.3f}）', **kwargs)
    
    # 绘制样本分布直方图
    if show_histogram and ax_hist is not None:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_counts = np.histogram(y_score, bins=bin_boundaries)[0]
        ax_hist.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.3,
                    color=colors[1], edgecolor=colors[1])
        ax_hist.set_ylabel('样本数', fontsize=10, color=colors[1])
        ax_hist.tick_params(axis='y', labelcolor=colors[1])
    
    ax.set_xlabel('平均预测概率', fontsize=12)
    ax.set_ylabel('实际正样本率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', frameon=True)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3)
    
    if save:
        save_figure(fig, save)
    
    return fig


# ==================== 评分卡相关图表 ====================

def score_dist_plot(
    df: Union[pd.DataFrame, pd.Series],
    score_col: Optional[str] = None,
    target_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    n_bins: int = 30,
    kde: bool = True,
    show_stats: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制评分分布对比图（好/坏样本分布对比）.
    
    :param df: 数据DataFrame
    :param score_col: 评分列名
    :param target_col: 目标变量列名，None时不区分好坏
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param n_bins: 直方图分箱数
    :param kde: 是否显示核密度估计曲线
    :param show_stats: 是否显示统计信息
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = score_dist_plot(df, 'score', 'target')
    """
    created_ax = ax is None
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)

    fontsize = kwargs.pop('fontsize', 14)
    anchor = kwargs.pop('anchor', None)
    labels = kwargs.pop('labels', ["好样本", "坏样本"])

    # 支持两种调用方式：
    # 1. score_dist_plot(df, 'score', 'target')        # 原始：df + 列名
    # 2. score_dist_plot(scores_series, targets_series)  # 简化：直接传 Series
    if isinstance(df, pd.Series):
        score_series = df.dropna() if score_col is None and target_col is None else df
        target_series = score_col if isinstance(score_col, pd.Series) else None
        score_col = df.name or "评分"
    else:
        if target_col is not None:
            score_series = df[score_col]
            target_series = df[target_col]
        else:
            score_series = df[score_col]
            target_series = None
        score_col = score_col or "评分"

    # 对齐好/坏样本：复用 hist_plot 的 step + probability 风格
    has_target = target_series is not None
    if has_target:
        mask = score_series.notna() & target_series.notna()
        score_series = score_series[mask]
        target_series = target_series[mask]
        target_unique = len(np.unique(target_series))
        if isinstance(labels, dict):
            hue = target_series.map(labels)
            hue_order = list(labels.values())
        else:
            hue = target_series.map({i: v for i, v in enumerate(labels)})
            hue_order = labels
        hue_order_final = hue_order[::-1]
        palette = get_series_colors(target_unique)
        sns.histplot(
            x=score_series, hue=hue, element="step", stat="probability",
            bins=n_bins, common_bins=True, common_norm=True, ax=ax,
            kde=kde, palette=palette, hue_order=hue_order_final, **kwargs,
        )
    else:
        score_series = score_series.dropna()
        color = colors[0] if colors else DEFAULT_COLORS[0]
        sns.histplot(
            x=score_series, element="step", stat="probability",
            bins=n_bins, ax=ax, kde=kde, color=color, **kwargs,
        )

    # 坐标轴样式（与 hist_plot 一致）
    setup_axis_style(ax)
    ax.set_xlabel(f"{score_col}区间", fontsize=fontsize)
    ax.set_ylabel("样本占比", fontsize=fontsize)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 标题
    if title is None:
        title = f"{score_col}分布情况"
    if created_ax:
        title_artist = fig.suptitle(title, fontsize=fontsize)
    else:
        title_artist = ax.set_title(title, fontsize=fontsize)

    # KS 统计信息
    if has_target and show_stats:
        from ..metrics import ks_2samps as ks_metric
        good_scores = score_series[target_series == 0]
        bad_scores = score_series[target_series == 1]
        ks_val = ks_metric(good_scores, bad_scores)
        ax.text(0.98, 0.98, f'KS: {ks_val:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=fontsize - 2,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 图例（顶部居中，与 hist_plot 一致）
    if has_target:
        legend_anchor = 1.15 if anchor is None else anchor
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, hue_order_final[:len(handles)],
                      loc='upper center', ncol=len(handles),
                      bbox_to_anchor=(0.5, legend_anchor), frameon=False, fontsize=fontsize)
        else:
            ax.legend(hue_order, loc='upper center', ncol=target_unique,
                      bbox_to_anchor=(0.5, legend_anchor), frameon=False, fontsize=fontsize)

    if created_ax:
        fig.tight_layout()
    if created_ax and has_target and anchor is None:
        _layout_top_center_legend(fig, ax.get_legend(), title=title_artist, axes=[ax])

    if save:
        save_figure(fig, save)

    return fig


def score_bin_plot(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    n_bins: int = 10,
    bin_type: str = 'quantile',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    show_table: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制评分分箱效果图（分箱区间+坏样本率）.

    使用 bin_plot（横向） + dataframe_plot 实现。

    :param df: 数据DataFrame
    :param score_col: 评分列名
    :param target_col: 目标变量列名
    :param n_bins: 分箱数，默认10
    :param bin_type: 分箱方式，'quantile'(等频)或'uniform'(等宽)
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_table: 是否显示数据表格
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象

    **参考样例**

    >>> fig = score_bin_plot(df, 'score', 'target', n_bins=10)
    """
    # 导入需要的函数
    from .binning_plots import bin_plot, dataframe_plot

    if colors is None:
        colors = DEFAULT_COLORS
    if bin_type not in {'quantile', 'uniform'}:
        raise ValueError("bin_type 仅支持 'quantile' 或 'uniform'")

    # 提取数据
    score_series = df[score_col]
    target_series = df[target_col]

    # 传入 ax 时复用调用方画布；未传 ax 时按是否展示表格创建一栏或两栏布局。
    if ax is not None:
        fig_charts = ax.figure
        ax_chart = ax
        if show_table:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            ax_table = make_axes_locatable(ax_chart).append_axes("right", size="42%", pad=0.6)
        else:
            ax_table = None
    elif show_table:
        fig_charts, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={'width_ratios': [2.5, 1]},
        )
        ax_chart, ax_table = axes
    else:
        fig_charts, ax_chart = plt.subplots(figsize=figsize)
        ax_table = None

    # 统一由推荐入口 bin_plot 同时生成图形和统计表，避免两套分箱口径漂移。
    _, bin_stats = bin_plot(
        score_series,
        target=target_series,
        desc=title or f'{score_col}分箱',
        figsize=(figsize[0] * 0.65, figsize[1]),
        colors=colors,
        ax=ax_chart,
        orientation='horizontal',
        n_bins=n_bins,
        method=bin_type,
        show_data_points=True,
        show_overall_bad_rate=True,
        return_frame=True,
        save=None,
    )

    # 2) dataframe_plot 显示分箱统计表
    if show_table and ax_table is not None:
        label_col = '分箱标签' if '分箱标签' in bin_stats.columns else '分箱'
        table_df = bin_stats[[label_col, '样本总数', '坏样本数', '坏样本率']].copy()
        table_df.columns = ['评分区间', '样本总数', '坏样本数', '坏样本率']
        table_df['坏样本率'] = table_df['坏样本率'].apply(lambda x: f'{x:.2%}')

        ax_table.axis('off')
        dataframe_plot(
            table_df,
            row_height=0.35,
            font_size=10,
            header_color=colors[0],
            ax=ax_table,
            save=None,
        )
    fig_charts.tight_layout()

    if save:
        save_figure(fig_charts, save)

    return fig_charts


# ==================== 风控策略相关图表 ====================

def threshold_analysis_plot(
    y_true: Union[pd.Series, np.ndarray],
    y_score: Union[pd.Series, np.ndarray],
    thresholds: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: str = "Threshold Analysis",
    colors: Optional[List[str]] = None,
    metrics: List[str] = ['precision', 'recall', 'f1', 'approval_rate'],
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制决策阈值分析图.
    
    展示不同阈值下的各项评估指标，帮助选择最优决策阈值。
    
    :param y_true: 真实标签
    :param y_score: 预测概率分数
    :param thresholds: 阈值数组，None时自动生成
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param metrics: 要显示的指标列表
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = threshold_analysis_plot(y_test, y_score)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    # 计算各阈值下的指标
    results = {metric: [] for metric in metrics}
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        approval_rate = (tp + fp) / len(y_true)
        
        if 'precision' in metrics:
            results['precision'].append(precision)
        if 'recall' in metrics:
            results['recall'].append(recall)
        if 'f1' in metrics:
            results['f1'].append(f1)
        if 'approval_rate' in metrics:
            results['approval_rate'].append(approval_rate)
        if 'specificity' in metrics:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['specificity'].append(specificity)
        if 'accuracy' in metrics:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            results['accuracy'].append(accuracy)
    
    # 绘制各指标曲线
    metric_labels = {
        'precision': 'Precision',
        'recall': 'Recall (TPR)',
        'f1': 'F1 Score',
        'approval_rate': 'Approval Rate',
        'specificity': 'Specificity (TNR)',
        'accuracy': 'Accuracy'
    }
    
    for i, metric in enumerate(metrics):
        if metric in results:
            ax.plot(thresholds, results[metric], lw=2, 
                   color=colors[i % len(colors)], label=metric_labels.get(metric, metric))
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score / Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    ax.legend(loc='best', frameon=True)
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3)
    
    if save:
        save_figure(fig, save)
    
    return fig


def strategy_compare_plot(
    strategies: List[Dict[str, Any]],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: str = "Strategy Comparison",
    colors: Optional[List[str]] = None,
    metrics: List[str] = ['approval_rate', 'bad_rate', 'ks'],
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制多策略效果对比图.
    
    :param strategies: 策略列表，每项为包含策略指标的字典
        例如: [{'name': '策略A', 'approval_rate': 0.8, 'bad_rate': 0.05, 'ks': 0.45}, ...]
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param metrics: 要对比的指标
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> strategies = [
    ...     {'name': 'Current', 'approval_rate': 0.75, 'bad_rate': 0.08, 'ks': 0.40},
    ...     {'name': 'New', 'approval_rate': 0.80, 'bad_rate': 0.06, 'ks': 0.50}
    ... ]
    >>> fig = strategy_compare_plot(strategies)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)

    strategy_names = [s['name'] for s in strategies]
    n_strategies = len(strategy_names)
    n_metrics = len(metrics)
    if colors is None:
        colors = get_series_colors(n_strategies)
    
    # 设置柱状图位置
    x = np.arange(n_metrics)
    width = 0.8 / n_strategies
    
    # 绘制每组策略的柱状图
    for i, strategy in enumerate(strategies):
        values = [strategy.get(m, 0) for m in metrics]
        offset = (i - n_strategies/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=strategy['name'],
               color=colors[i % len(colors)], alpha=0.8)
    
    # 设置标签
    metric_labels = {
        'approval_rate': 'Approval Rate',
        'bad_rate': 'Bad Rate',
        'ks': 'KS Statistic',
        'auc': 'AUC',
        'iv': 'IV',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], rotation=45, ha='right')
    ax.legend(loc='best', frameon=True)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, strategy in enumerate(strategies):
        values = [strategy.get(m, 0) for m in metrics]
        offset = (i - n_strategies/2 + 0.5) * width
        for j, v in enumerate(values):
            ax.text(j + offset, v + 0.01, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    if save:
        save_figure(fig, save)
    
    return fig


def vintage_plot(
    df: pd.DataFrame,
    mob_col: str,
    target_col: str,
    vintage_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    max_mob: Optional[int] = None,
    show_heatmap: bool = False,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制Vintage账龄曲线图.
    
    展示不同放款月份的资产在不同账龄(MOB)时的逾期率表现。
    
    :param df: 数据DataFrame
    :param mob_col: MOB（账龄）列名
    :param target_col: 目标变量列名（逾期标识）
    :param vintage_col: 放款月份/批次列名，None时不区分批次
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param max_mob: 最大MOB显示值
    :param show_heatmap: 是否同时显示热力图
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = vintage_plot(df, 'mob', 'ever_dpd30', 'issue_month')
    """
    # 创建透视表
    if vintage_col:
        vintage_data = df.groupby([vintage_col, mob_col])[target_col].mean().reset_index()
        vintage_pivot = vintage_data.pivot(index=vintage_col, columns=mob_col, values=target_col)
    else:
        # 不区分批次，计算整体
        vintage_overall = df.groupby(mob_col)[target_col].mean()
        vintage_pivot = vintage_overall.to_frame().T
        vintage_pivot.index = ['Overall']
    if colors is None:
        colors = get_series_colors(len(vintage_pivot))
    
    # 限制最大MOB
    if max_mob:
        mob_cols = [c for c in vintage_pivot.columns if c <= max_mob]
        vintage_pivot = vintage_pivot[mob_cols]
    
    # 创建图表：传入ax时复用该ax绘制曲线，热力图作为附加面板挂载在同一figure上，
    # 而不是整体重新plt.subplots()丢弃调用方传入的ax
    if ax is not None:
        ax_line = ax
        fig = ax_line.get_figure()
        if show_heatmap:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            ax_heat = make_axes_locatable(ax_line).append_axes("right", size="40%", pad=0.8)
    elif show_heatmap:
        fig, (ax_line, ax_heat) = plt.subplots(1, 2, figsize=figsize,
                                               gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, ax_line = get_or_create_ax(figsize=figsize, ax=ax)
    
    # 绘制曲线
    mob_values = vintage_pivot.columns.values
    
    for i, (vintage, row) in enumerate(vintage_pivot.iterrows()):
        color = colors[i % len(colors)]
        ax_line.plot(mob_values, row.values * 100, 'o-', 
                    label=str(vintage), color=color, lw=2, markersize=4)
    
    ax_line.set_xlabel('Month on Book (MOB)', fontsize=12)
    ax_line.set_ylabel('Bad Rate (%)', fontsize=12)
    
    if title is None:
        title = 'Vintage Analysis'
    ax_line.set_title(title, fontsize=14, fontweight='bold')
    
    ax_line.legend(loc='upper left', frameon=True, title='Vintage')
    setup_axis_style(ax_line, colors, hide_top_right=True)
    ax_line.grid(True, alpha=0.3)
    
    # 绘制热力图
    if show_heatmap:
        sns.heatmap(vintage_pivot * 100, annot=True, fmt='.2f',
                   cmap=make_risk_cmap("hscredit_vintage"),
                   ax=ax_heat, cbar_kws={'label': 'Bad Rate (%)'})
        ax_heat.set_title('Vintage Heatmap', fontsize=12, fontweight='bold')
        ax_heat.set_xlabel('MOB', fontsize=10)
        ax_heat.set_ylabel('Vintage', fontsize=10)
    
    if save:
        save_figure(fig, save)
    
    return fig


def feature_importance_plot(
    features: List[str],
    importance: Union[List[float], np.ndarray],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Feature Importance",
    colors: Optional[List[str]] = None,
    top_n: Optional[int] = 20,
    horizontal: bool = True,
    show_values: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制特征重要性图.
    
    :param features: 特征名称列表
    :param importance: 特征重要性值列表
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param top_n: 显示前N个特征，None时显示全部
    :param horizontal: 是否水平显示
    :param show_values: 是否显示数值
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> features = ['age', 'income', 'score', ...]
    >>> importance = model.feature_importances_
    >>> fig = feature_importance_plot(features, importance, top_n=15)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 排序并选择Top N
    sorted_indices = np.argsort(importance)[::-1]
    if top_n:
        sorted_indices = sorted_indices[:top_n]
    
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = [importance[i] for i in sorted_indices]
    
    # 绘制
    if horizontal:
        y_pos = np.arange(len(sorted_features))
        bars = ax.barh(y_pos, sorted_importance, color=colors[0], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()  # 最高重要性在顶部
        ax.set_xlabel('Importance', fontsize=12)
    else:
        x_pos = np.arange(len(sorted_features))
        bars = ax.bar(x_pos, sorted_importance, color=colors[0], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_features, rotation=45, ha='right')
        ax.set_ylabel('Importance', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加数值标签
    if show_values:
        for bar, val in zip(bars, sorted_importance):
            if horizontal:
                ax.text(val + 0.01 * max(sorted_importance), bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01 * max(sorted_importance),
                       f'{val:.3f}', ha='center', fontsize=9)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
    
    if save:
        save_figure(fig, save)
    
    return fig


def approval_rate_trend_plot(
    df: pd.DataFrame,
    date_col: str,
    decision_col: Optional[str] = None,
    score_col: Optional[str] = None,
    threshold: Optional[float] = None,
    freq: str = 'M',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (14, 6),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    show_bad_rate: bool = True,
    target_col: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制审批通过率趋势图.
    
    :param df: 数据DataFrame
    :param date_col: 日期列名
    :param decision_col: 决策结果列名（通过/拒绝），None时使用score_col+threshold
    :param score_col: 评分列名（用于计算通过/拒绝）
    :param threshold: 通过阈值（分数>=threshold为通过）
    :param freq: 时间频率，'D'/'W'/'M'/'Q'
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_bad_rate: 是否同时显示逾期率趋势
    :param target_col: 目标变量列名（show_bad_rate=True时需要）
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = approval_rate_trend_plot(df, 'apply_date', decision_col='is_approved')
    >>> fig = approval_rate_trend_plot(df, 'apply_date', score_col='score', threshold=500)
    """
    df = df.copy()
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 确保日期格式正确
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 计算通过标识
    if decision_col:
        df['_approved'] = df[decision_col]
    elif score_col and threshold is not None:
        df['_approved'] = (df[score_col] >= threshold).astype(int)
    else:
        raise ValueError("必须提供decision_col或(score_col+threshold)")
    
    # 按时间聚合
    df['_period'] = df[date_col].dt.to_period(freq)
    
    # 仅当提供 target_col 时才聚合逾期率，避免对 None 列聚合导致 KeyError
    agg_spec = {'_approved': ['count', 'sum', 'mean']}
    if target_col:
        agg_spec[target_col] = 'mean'
    trend_data = df.groupby('_period').agg(agg_spec).reset_index()

    if target_col:
        trend_data.columns = ['period', 'total', 'approved_count', 'approval_rate', 'bad_rate']
    else:
        trend_data.columns = ['period', 'total', 'approved_count', 'approval_rate']
    trend_data['period'] = trend_data['period'].dt.to_timestamp()
    
    # 绘制审批率
    ax.plot(trend_data['period'], trend_data['approval_rate'] * 100,
            'o-', color=colors[0], lw=2, markersize=4, label='Approval Rate')
    ax.fill_between(trend_data['period'], trend_data['approval_rate'] * 100,
                    alpha=0.2, color=colors[0])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Approval Rate (%)', fontsize=12, color=colors[0])
    ax.tick_params(axis='y', labelcolor=colors[0])
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    # 绘制逾期率（双轴）
    if show_bad_rate and target_col:
        ax2 = ax.twinx()
        ax2.plot(trend_data['period'], trend_data['bad_rate'] * 100,
                's-', color=BAD_RATE_COLOR, lw=2, markersize=4, label='Bad Rate')
        ax2.set_ylabel('Bad Rate (%)', fontsize=12, color=BAD_RATE_COLOR)
        ax2.tick_params(axis='y', labelcolor=BAD_RATE_COLOR)
        ax2.yaxis.set_major_formatter(PercentFormatter())
    
    if title is None:
        title = 'Approval Rate Trend'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    if show_bad_rate and target_col:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
    else:
        ax.legend(loc='best', frameon=True)
    
    setup_axis_style(ax, colors)
    ax.grid(True, alpha=0.3)
    
    # 清理临时列
    df.drop(columns=['_approved', '_period'], inplace=True, errors='ignore')
    
    if save:
        save_figure(fig, save)
    
    return fig


def bad_rate_trend_plot(
    df: pd.DataFrame,
    date_col: str,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    del_grey: bool = False,
    dimension_col: Optional[str] = None,
    freq: str = 'M',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (14, 6),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    show_sample_count: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """绘制坏样本率趋势图（支持分维度和多逾期标签展示）.
    
    :param df: 数据DataFrame
    :param date_col: 日期列名
    :param target: 目标变量列名（单标签模式）
    :param overdue: 逾期天数字段名或列表，优先于 target
    :param dpds: 逾期定义天数或列表，与 overdue 配合生成标签
    :param del_grey: 是否排除逾期天数在 (0, dpd] 区间的灰样本
    :param dimension_col: 维度列名（如客户等级），None时不分维度
    :param freq: 时间频率，'D'/'W'/'M'/'Q'
    :param ax: matplotlib Axes对象
    :param figsize: 图像尺寸
    :param title: 图表标题
    :param colors: 配色方案
    :param show_sample_count: 是否显示样本数柱状图
    :param save: 保存路径
    :param kwargs: 其他参数
    :return: matplotlib Figure对象
    
    **参考样例**

    >>> fig = bad_rate_trend_plot(df, 'apply_date', target='target')
    >>> fig = bad_rate_trend_plot(df, 'apply_date', overdue='MOB1', dpds=[7, 30])
    """
    if 'target_col' in kwargs:
        raise TypeError("bad_rate_trend_plot 已统一使用 target 参数，请将 target_col 改为 target")
    df = df.copy()

    # 全库统一标签入口：overdue + dpds 显式传入时优先于 target。
    target_groups = {}
    using_overdue = overdue is not None
    if using_overdue:
        if dpds is None:
            raise ValueError("传入 overdue 参数时必须同时传入 dpds")
        overdue_cols = [overdue] if isinstance(overdue, str) else list(overdue)
        dpd_values = [dpds] if isinstance(dpds, (int, np.integer)) else list(dpds)
        if not overdue_cols or not dpd_values:
            raise ValueError("overdue 和 dpds 不能为空")
        for overdue_col in overdue_cols:
            if overdue_col not in df.columns:
                raise ValueError(f"数据集缺少逾期天数列: {overdue_col}")
            overdue_days = pd.to_numeric(df[overdue_col], errors='coerce')
            overdue_targets = []
            for dpd in dpd_values:
                label = f"{overdue_col}_{dpd}+"
                generated_target = (overdue_days > dpd).astype(float)
                if del_grey:
                    generated_target[(overdue_days > 0) & (overdue_days <= dpd)] = np.nan
                overdue_targets.append((label, generated_target))
            target_groups[overdue_col] = overdue_targets
    else:
        if target is None or target not in df.columns:
            raise ValueError("必须传入数据集中存在的 target 或 overdue+dpds 参数")
        target_values = pd.to_numeric(df[target], errors='coerce')
        invalid_values = target_values.dropna()[~target_values.dropna().isin([0, 1])]
        if not invalid_values.empty:
            raise ValueError(f"target 列 {target} 必须是 0/1 二分类标签")
        target_groups[target] = [(target, target_values)]

    all_targets = [
        (target_label, target_values)
        for grouped_targets in target_groups.values()
        for target_label, target_values in grouped_targets
    ]

    # 传入ax时复用该ax绘制主曲线，样本数柱状图作为附加面板挂载在同一figure上，
    # 而不是整体重新plt.subplots()丢弃调用方传入的ax
    if ax is not None:
        ax_line = ax
        fig = ax_line.get_figure()
        if show_sample_count:
            panel_position = ax_line.get_position().frozen()
            panel_gap = min(0.02, panel_position.height * 0.05)
            available_height = panel_position.height - panel_gap
            bar_height = available_height / 4.0
            line_height = available_height * 3.0 / 4.0
            ax_bar = fig.add_axes(
                [panel_position.x0, panel_position.y0, panel_position.width, bar_height],
                sharex=ax_line,
            )
            ax_line.set_position(
                [
                    panel_position.x0,
                    panel_position.y0 + bar_height + panel_gap,
                    panel_position.width,
                    line_height,
                ]
            )
    elif show_sample_count:
        fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=figsize,
                                              sharex=True,
                                              gridspec_kw={'height_ratios': [3, 1],
                                                          'hspace': 0.1})
    else:
        fig, ax_line = get_or_create_ax(figsize=figsize, ax=ax)
    
    # 日期先转为周期，再用离散位置绘制；刻度只来自真实分组，不交给日期定位器补点。
    if date_col not in df.columns:
        raise ValueError(f"数据集缺少日期列: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    freq = str(freq).upper()
    if freq not in {'D', 'W', 'M', 'Q'}:
        raise ValueError("freq 必须是 'D'/'W'/'M'/'Q' 之一")
    df['_period'] = df[date_col].dt.to_period(freq)
    periods = sorted(df['_period'].dropna().unique())
    positions = np.arange(len(periods))

    if colors is not None and not colors:
        raise ValueError("colors 至少需要包含一种颜色")

    def _unique_series_palette(requested_colors, required_count):
        palette = []
        normalized_colors = set()
        fallback_colors = [BAD_RATE_COLOR, *get_series_colors(required_count + 1)]
        for color in [*requested_colors, *fallback_colors]:
            normalized = to_hex(color).lower()
            if normalized in normalized_colors:
                continue
            palette.append(color)
            normalized_colors.add(normalized)
            if len(palette) >= required_count:
                return palette

        risk_cmap = make_risk_cmap()
        for position in np.linspace(0.0, 1.0, max(required_count * 2, 2)):
            color = to_hex(risk_cmap(position))
            normalized = color.lower()
            if normalized in normalized_colors:
                continue
            palette.append(color)
            normalized_colors.add(normalized)
            if len(palette) >= required_count:
                break
        return palette

    requested_series_colors = (
        [BAD_RATE_COLOR, *get_series_colors(len(all_targets) + 1)]
        if colors is None
        else list(colors)
    )
    series_colors = _unique_series_palette(requested_series_colors, len(all_targets))

    dimensions = None
    dimension_series_colors = None
    if dimension_col:
        if dimension_col not in df.columns:
            raise ValueError(f"数据集缺少维度列: {dimension_col}")
        dimensions = list(pd.unique(df[dimension_col].dropna()))
        required_dimension_colors = len(all_targets) * len(dimensions)
        requested_dimension_colors = (
            get_series_colors(required_dimension_colors)
            if colors is None
            else list(colors)
        )
        dimension_series_colors = _unique_series_palette(
            requested_dimension_colors,
            required_dimension_colors,
        )

    rate_axes = []
    line_handles = []
    line_labels = []
    line_styles = ['-', '--', '-.', ':']
    target_index = 0
    for axis_index, (axis_label, grouped_targets) in enumerate(target_groups.items()):
        rate_ax = ax_line if axis_index == 0 else ax_line.twinx()
        if axis_index > 0:
            rate_ax.spines['right'].set_position(('outward', 52 * (axis_index - 1)))
        rate_axes.append(rate_ax)
        axis_color = series_colors[target_index % len(series_colors)]

        for target_label, target_values in grouped_targets:
            color = series_colors[target_index % len(series_colors)]
            working = pd.DataFrame({
                '_period': df['_period'],
                '_target': target_values,
            }, index=df.index)
            if dimension_col:
                working['_dimension'] = df[dimension_col]
                for dimension_index, dimension in enumerate(dimensions):
                    dimension_rates = (
                        working[working['_dimension'] == dimension]
                        .groupby('_period', observed=False)['_target']
                        .mean()
                        .reindex(periods)
                    )
                    legend_label = (
                        str(dimension)
                        if not using_overdue and len(all_targets) == 1
                        else f"{target_label} · {dimension}"
                    )
                    color_index = target_index * len(dimensions) + dimension_index
                    line_color = dimension_series_colors[color_index % len(dimension_series_colors)]
                    line, = rate_ax.plot(
                        positions,
                        dimension_rates.to_numpy(dtype=float),
                        marker='o',
                        linestyle=line_styles[dimension_index % len(line_styles)],
                        color=line_color,
                        lw=2,
                        markersize=4,
                        label=legend_label,
                    )
                    line_handles.append(line)
                    line_labels.append(legend_label)
            else:
                bad_rates = (
                    working.groupby('_period', observed=False)['_target']
                    .mean()
                    .reindex(periods)
                )
                line, = rate_ax.plot(
                    positions,
                    bad_rates.to_numpy(dtype=float),
                    'o-',
                    color=color,
                    lw=2,
                    markersize=4,
                    label=(
                        target_label
                        if using_overdue or len(all_targets) > 1
                        else '坏样本率'
                    ),
                )
                line_handles.append(line)
                line_labels.append(line.get_label())
                if len(all_targets) == 1:
                    rate_ax.fill_between(
                        positions,
                        bad_rates.to_numpy(dtype=float),
                        alpha=0.12,
                        color=color,
                    )
            target_index += 1

        rate_ax.set_ylabel(
            f'{axis_label} 坏样本率' if using_overdue else '坏样本率',
            fontsize=12,
            color=axis_color,
        )
        rate_ax.tick_params(axis='y', colors=axis_color)
        rate_ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        if axis_index == 0:
            setup_axis_style(rate_ax, [axis_color], hide_top_right=True)
        else:
            rate_ax.spines['top'].set_visible(False)
            rate_ax.spines['bottom'].set_visible(False)
            rate_ax.spines['left'].set_visible(False)
            rate_ax.spines['right'].set_color(axis_color)
            rate_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        rate_ax.set_axisbelow(True)
        rate_ax.grid(False)

    # 多坐标轴必须共享同一自适应百分比范围，否则成比例的序列会被各自缩放成完全重合的像素轨迹。
    if len(rate_axes) > 1:
        plotted_rates = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float)
            for rate_ax in rate_axes
            for line in rate_ax.lines
        ])
        finite_rates = plotted_rates[np.isfinite(plotted_rates)]
        if finite_rates.size:
            rate_min = float(finite_rates.min())
            rate_max = float(finite_rates.max())
            rate_span = rate_max - rate_min
            padding = max(rate_span * 0.08, 0.01)
            lower_limit = max(0.0, rate_min - padding)
            upper_limit = min(1.0, rate_max + padding)
            if upper_limit <= lower_limit:
                upper_limit = min(1.0, lower_limit + 0.05)
                lower_limit = max(0.0, upper_limit - 0.05)
            for rate_ax in rate_axes:
                rate_ax.set_ylim(lower_limit, upper_limit)
    
    if title is None:
        title = '坏样本率趋势' + (f'（按{dimension_col}）' if dimension_col else '')
    title_artist = fig.suptitle(title, fontsize=14, fontweight='bold')
    max_legend_columns = max(1, int(fig.get_figwidth() // 2.2))
    legend = fig.legend(
        line_handles,
        line_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(line_labels), max_legend_columns),
        frameon=False,
    )
    ax_line.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 绘制样本数柱状图
    if show_sample_count:
        sample_counts = df.groupby('_period', observed=False).size().reindex(periods, fill_value=0)
        bar_color = DEFAULT_COLORS[0]
        ax_bar.bar(positions, sample_counts.to_numpy(), width=0.65, alpha=0.6, color=bar_color)
        ax_bar.set_ylabel('样本数', fontsize=10)
        ax_bar.set_xlabel('日期', fontsize=12)
        setup_axis_style(ax_bar, [bar_color], hide_top_right=True)
        ax_bar.set_axisbelow(True)
        ax_bar.grid(True, axis='y', alpha=0.3, linestyle='--')
    else:
        ax_line.set_xlabel('日期', fontsize=12)

    tick_axis = ax_bar if show_sample_count else ax_line
    tick_axis.set_xticks(positions)
    tick_axis.set_xticklabels([str(period) for period in periods])
    tick_axis.tick_params(axis='x', labelrotation=30)
    for label in tick_axis.get_xticklabels():
        label.set_horizontalalignment('right')
    if show_sample_count:
        for rate_ax in rate_axes:
            rate_ax.tick_params(axis='x', bottom=False, labelbottom=False)

    x_limits = (-0.5, len(periods) - 0.5) if periods else (-0.5, 0.5)
    for rate_ax in rate_axes:
        rate_ax.set_xlim(x_limits)
    if show_sample_count:
        ax_bar.set_xlim(x_limits)

    # 多个右侧坏率轴按实际渲染宽度收缩内容区，避免外移轴及标签被画布裁切。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    right_artists = []
    for rate_ax in rate_axes[1:]:
        right_artists.extend([rate_ax.yaxis.label, *rate_ax.get_yticklabels()])
    right_edges = [
        artist.get_window_extent(renderer).x1
        for artist in right_artists
        if artist.get_visible() and artist.get_text()
    ]
    padding_pixels = 6.0 * fig.dpi / 72.0
    if right_edges:
        overflow = max(right_edges) - (fig.bbox.x1 - padding_pixels)
    else:
        overflow = 0.0

    # twinx 轴共享位置；公共几何只计算一次并原样赋给上下所有面板，避免逐轴重复收窄。
    rate_position = ax_line.get_position().frozen()
    shrink_fraction = max(0.0, overflow) / float(fig.bbox.width) + (0.01 if overflow > 0 else 0.0)
    common_width = max(0.2, rate_position.width - shrink_fraction)
    common_x0 = rate_position.x0
    common_rate_position = (common_x0, rate_position.y0, common_width, rate_position.height)
    for rate_ax in rate_axes:
        rate_ax.set_position(common_rate_position)
    if show_sample_count:
        bar_position = ax_bar.get_position().frozen()
        ax_bar.set_position((common_x0, bar_position.y0, common_width, bar_position.height))

    _layout_top_center_legend(fig, legend, title=title_artist, axes=[ax_line])
    final_rate_position = ax_line.get_position().frozen()
    for rate_ax in rate_axes[1:]:
        rate_ax.set_position(final_rate_position.bounds)
    if show_sample_count:
        final_bar_position = ax_bar.get_position().frozen()
        ax_bar.set_position(
            (
                final_rate_position.x0,
                final_bar_position.y0,
                final_rate_position.width,
                final_bar_position.height,
            )
        )

    if save:
        save_figure(fig, save)
    
    return fig
