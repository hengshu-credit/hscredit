# -*- coding: utf-8 -*-
"""
金融风控数据可视化函数.

提供金融建模和风控策略分析专用的可视化功能，包括：
- ROC曲线图 (roc_plot)
- Lift提升图 (lift_plot)
- Gain增益图 (gain_plot)
- 评分分布对比图 (score_dist_plot)
- 逾期率趋势图 (bad_rate_trend_plot)
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
import seaborn as sns
from typing import Union, Optional, List, Dict, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, brier_score_loss
)
from matplotlib.ticker import PercentFormatter

from .utils import (
    DEFAULT_COLORS, setup_axis_style, save_figure,
    get_or_create_ax
)


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
    
    Example:
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
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')
    
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
    
    Example:
        >>> fig = pr_plot(y_test, model.predict_proba(X_test)[:, 1])
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = np.mean(precision)  # 简化的AP计算
    
    # 计算基线（正样本比例）
    if show_baseline:
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', 
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
    title: str = "Lift Chart",
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
    
    Example:
        >>> fig = lift_plot(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 计算Lift
    overall_bad_rate = np.mean(y_true)
    
    # 按分数排序分箱
    sorted_indices = np.argsort(-y_score)  # 降序
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # 计算每个分箱的Lift
    bin_size = len(y_true) // n_bins
    lifts = []
    depths = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        bin_bad_rate = np.mean(y_true_sorted[start:end])
        lift = bin_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 1.0
        
        lifts.append(lift)
        depths.append((end / len(y_true)) * 100)
    
    # 绘制基线
    if show_baseline:
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Baseline (Lift=1)')
    
    # 绘制Lift曲线
    ax.plot(depths, lifts, color=colors[0], marker='o', lw=2, markersize=6, **kwargs)
    
    # 绘制柱状图
    ax.bar(depths, lifts, width=8, alpha=0.3, color=colors[0], edgecolor=colors[0])
    
    # 设置图表属性
    ax.set_xlabel('Depth (% of Population)', fontsize=12)
    ax.set_ylabel('Lift', fontsize=12)
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
    
    Example:
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
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        captured_bads = np.sum(y_true_sorted[:end])
        gain = captured_bads / total_bads if total_bads > 0 else 0
        
        cumulative_gains.append(gain * 100)
        depths.append((end / len(y_true)) * 100)
    
    # 绘制基线（随机模型）
    if show_baseline:
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Baseline (Random)')
    
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
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
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
    
    Example:
        >>> fig = confusion_matrix_plot(y_test, y_pred)
        >>> fig = confusion_matrix_plot(y_test, y_pred, normalize='true')
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    # 绘制热力图
    sns.heatmap(cm, annot=show_values, fmt='.2f' if normalize else 'd',
                cmap=cmap, square=True, ax=ax,
                xticklabels=['Good', 'Bad'],
                yticklabels=['Good', 'Bad'],
                **kwargs)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 显示评估指标
    if show_metrics:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
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
    title: str = "Calibration Curve",
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
    
    Example:
        >>> fig = calibration_plot(y_test, model.predict_proba(X_test)[:, 1])
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 如果需要显示直方图，创建双轴
    if show_histogram:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax_hist = ax.twinx()
    else:
        fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
        ax_hist = None
    
    # 计算校准曲线
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_score > bin_lower) & (y_score <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(np.sum(in_bin))
    
    # 绘制完美校准线
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # 绘制校准曲线
    brier = brier_score_loss(y_true, y_score)
    ax.plot(bin_centers, bin_accuracies, 's-', color=colors[0],
            label=f'Model (Brier={brier:.3f})', **kwargs)
    
    # 绘制样本分布直方图
    if show_histogram and ax_hist is not None:
        ax_hist.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.3,
                    color=colors[1], edgecolor=colors[1])
        ax_hist.set_ylabel('Count', fontsize=10, color=colors[1])
        ax_hist.tick_params(axis='y', labelcolor=colors[1])
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
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
    
    Example:
        >>> fig = score_dist_plot(df, 'score', 'target')
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)

    if colors is None:
        colors = DEFAULT_COLORS

    if title is None:
        title = f'{score_col} Distribution'

    # 支持两种调用方式：
    # 1. score_dist_plot(df, 'score', 'target')        # 原始：df + 列名
    # 2. score_dist_plot(scores_series, targets_series)  # 简化：直接传 Series
    if isinstance(df, pd.Series):
        score_series = df
        target_series = score_col if isinstance(score_col, pd.Series) else None
        score_col = score_series.name or 'score'
    else:
        score_series = None
        target_series = None
    
    if score_series is not None:
        # 路径B：直接传 Series（不区分好坏 or 用 target_series）
        if target_series is not None:
            good_scores = score_series[target_series == 0].dropna()
            bad_scores = score_series[target_series == 1].dropna()
            sns.histplot(good_scores, bins=n_bins, kde=kde, ax=ax,
                         color=colors[0], alpha=0.6, label=f'Good (n={len(good_scores)})')
            sns.histplot(bad_scores, bins=n_bins, kde=kde, ax=ax,
                         color=colors[1], alpha=0.6, label=f'Bad (n={len(bad_scores)})')
            if show_stats:
                from ..metrics import ks_2samps as ks_metric
                ks_val = ks_metric(good_scores, bad_scores)
                ax.text(0.98, 0.98, f'KS: {ks_val:.3f}', transform=ax.transAxes,
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend(loc='upper right', frameon=True)
        else:
            sns.histplot(score_series.dropna(), bins=n_bins, kde=kde, ax=ax,
                         color=colors[0], alpha=0.6)
        _path_b_done = True
    else:
        _path_b_done = False

    if not _path_b_done and target_col is not None:
        # 路径A：原始方式 — df + 列名
        good_scores = df[df[target_col] == 0][score_col].dropna()
        bad_scores = df[df[target_col] == 1][score_col].dropna()

        sns.histplot(good_scores, bins=n_bins, kde=kde, ax=ax,
                     color=colors[0], alpha=0.6, label=f'Good (n={len(good_scores)})')
        sns.histplot(bad_scores, bins=n_bins, kde=kde, ax=ax,
                     color=colors[1], alpha=0.6, label=f'Bad (n={len(bad_scores)})')

        if show_stats:
            from ..metrics import ks_2samps as ks_metric
            ks_val = ks_metric(good_scores, bad_scores)
            stats_text = f'KS: {ks_val:.3f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif not _path_b_done:
        # 不区分好坏（df 方式）
        sns.histplot(df[score_col].dropna(), bins=n_bins, kde=kde, ax=ax,
                     color=colors[0], alpha=0.6)
    
    # legend / xlabel / title：路径A/B 共用（路径B已在内部处理过 legend）
    if not _path_b_done:
        ax.set_xlabel(score_col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if target_col is not None:
            ax.legend(loc='upper right', frameon=True)
    
    setup_axis_style(ax, colors, hide_top_right=True)
    
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

    Example:
        >>> fig = score_bin_plot(df, 'score', 'target', n_bins=10)
    """
    # 导入需要的函数
    from .binning_plots import bin_plot, dataframe_plot

    if colors is None:
        colors = DEFAULT_COLORS

    # 提取数据
    score_series = df[score_col]
    target_series = df[target_col]

    # 使用 bin_plot 绘制横向分箱图
    fig_charts, axes = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [2.5, 1]})
    ax_chart = axes[0]
    ax_table = axes[1]

    # 1) bin_plot（横向）
    from matplotlib.figure import Figure
    if isinstance(ax_chart, Figure):
        # ax_chart 实际上是 Figure，ax 参数传入的是子 axes
        pass

    # 让 bin_plot 在 ax_chart 上绘图
    _ = bin_plot(
        score_series,
        target=target_series,
        desc=title or f'{score_col}分箱',
        figsize=(figsize[0] * 0.65, figsize[1]),
        colors=colors,
        ax=ax_chart,
        orientation='horizontal',
        show_data_points=True,
        show_overall_bad_rate=True,
        save=None,
    )

    # 2) dataframe_plot 显示分箱统计表
    if show_table:
        # 先计算分箱统计
        if bin_type == 'quantile':
            bins = pd.qcut(df[score_col], q=n_bins, duplicates='drop')
        else:
            bins = pd.cut(df[score_col], bins=n_bins)

        bin_stats = df.groupby(bins).agg({
            target_col: ['count', 'sum', 'mean'],
            score_col: ['min', 'max']
        }).reset_index()
        bin_stats.columns = ['bin', 'count', 'bad_count', 'bad_rate', 'min_score', 'max_score']
        bin_stats['good_count'] = bin_stats['count'] - bin_stats['bad_count']
        bin_stats['bin_label'] = bin_stats.apply(
            lambda x: f'[{x["min_score"]:.0f}, {x["max_score"]:.0f})', axis=1
        )

        table_df = bin_stats[['bin_label', 'count', 'bad_count', 'bad_rate']].copy()
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
    else:
        ax_table.axis('off')

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
    
    Example:
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
    
    Example:
        >>> strategies = [
        ...     {'name': 'Current', 'approval_rate': 0.75, 'bad_rate': 0.08, 'ks': 0.40},
        ...     {'name': 'New', 'approval_rate': 0.80, 'bad_rate': 0.06, 'ks': 0.50}
        ... ]
        >>> fig = strategy_compare_plot(strategies)
    """
    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    strategy_names = [s['name'] for s in strategies]
    n_strategies = len(strategy_names)
    n_metrics = len(metrics)
    
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
    
    Example:
        >>> fig = vintage_plot(df, 'mob', 'ever_dpd30', 'issue_month')
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 创建透视表
    if vintage_col:
        vintage_data = df.groupby([vintage_col, mob_col])[target_col].mean().reset_index()
        vintage_pivot = vintage_data.pivot(index=vintage_col, columns=mob_col, values=target_col)
    else:
        # 不区分批次，计算整体
        vintage_overall = df.groupby(mob_col)[target_col].mean()
        vintage_pivot = vintage_overall.to_frame().T
        vintage_pivot.index = ['Overall']
    
    # 限制最大MOB
    if max_mob:
        mob_cols = [c for c in vintage_pivot.columns if c <= max_mob]
        vintage_pivot = vintage_pivot[mob_cols]
    
    # 创建图表
    if show_heatmap:
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
        sns.heatmap(vintage_pivot * 100, annot=True, fmt='.2f', cmap='YlOrRd',
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
    
    Example:
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
    
    Example:
        >>> fig = approval_rate_trend_plot(df, 'apply_date', decision_col='is_approved')
        >>> fig = approval_rate_trend_plot(df, 'apply_date', score_col='score', threshold=500)
    """
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
    
    trend_data = df.groupby('_period').agg({
        '_approved': ['count', 'sum', 'mean'],
        target_col: 'mean' if target_col else 'sum'
    }).reset_index()
    
    trend_data.columns = ['period', 'total', 'approved_count', 'approval_rate', 'bad_rate']
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
                's-', color=colors[1], lw=2, markersize=4, label='Bad Rate')
        ax2.set_ylabel('Bad Rate (%)', fontsize=12, color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])
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
    target_col: str,
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
    """绘制逾期率趋势图（支持分维度展示）.
    
    :param df: 数据DataFrame
    :param date_col: 日期列名
    :param target_col: 目标变量列名
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
    
    Example:
        >>> fig = bad_rate_trend_plot(df, 'apply_date', 'target')
        >>> fig = bad_rate_trend_plot(df, 'apply_date', 'target', dimension_col='customer_grade')
    """
    if show_sample_count:
        fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=figsize, 
                                              sharex=True,
                                              gridspec_kw={'height_ratios': [3, 1],
                                                          'hspace': 0.1})
    else:
        fig, ax_line = get_or_create_ax(figsize=figsize, ax=ax)
    
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 确保日期格式正确
    df[date_col] = pd.to_datetime(df[date_col])
    df['_period'] = df[date_col].dt.to_period(freq)
    
    # 计算趋势
    if dimension_col:
        trend_data = df.groupby(['_period', dimension_col]).agg({
            target_col: ['count', 'mean']
        }).reset_index()
        trend_data.columns = ['period', 'dimension', 'count', 'bad_rate']
        trend_data['period'] = trend_data['period'].dt.to_timestamp()
        
        # 绘制各维度曲线
        dimensions = trend_data['dimension'].unique()
        for i, dim in enumerate(dimensions):
            dim_data = trend_data[trend_data['dimension'] == dim]
            ax_line.plot(dim_data['period'], dim_data['bad_rate'] * 100,
                        'o-', label=str(dim), color=colors[i % len(colors)],
                        lw=2, markersize=4)
    else:
        trend_data = df.groupby('_period').agg({
            target_col: ['count', 'mean']
        }).reset_index()
        trend_data.columns = ['period', 'count', 'bad_rate']
        trend_data['period'] = trend_data['period'].dt.to_timestamp()
        
        ax_line.plot(trend_data['period'], trend_data['bad_rate'] * 100,
                    'o-', color=colors[0], lw=2, markersize=4, label='Bad Rate')
        ax_line.fill_between(trend_data['period'], trend_data['bad_rate'] * 100,
                            alpha=0.2, color=colors[0])
    
    ax_line.set_ylabel('Bad Rate (%)', fontsize=12)
    ax_line.yaxis.set_major_formatter(PercentFormatter())
    
    if title is None:
        title = f'Bad Rate Trend' + (f' by {dimension_col}' if dimension_col else '')
    ax_line.set_title(title, fontsize=14, fontweight='bold')
    
    ax_line.legend(loc='best', frameon=True)
    setup_axis_style(ax_line, colors, hide_top_right=True)
    ax_line.grid(True, alpha=0.3)
    
    # 绘制样本数柱状图
    if show_sample_count:
        overall_trend = df.groupby('_period').size().reset_index(name='count')
        overall_trend['period'] = overall_trend['_period'].dt.to_timestamp()
        
        ax_bar.bar(overall_trend['period'], overall_trend['count'],
                  width=20, alpha=0.6, color=colors[2])
        ax_bar.set_ylabel('Sample Count', fontsize=10)
        ax_bar.set_xlabel('Date', fontsize=12)
    
    # 清理临时列
    df.drop(columns=['_period'], inplace=True, errors='ignore')
    
    if save:
        save_figure(fig, save)
    
    return fig
