# -*- coding: utf-8 -*-
"""评分分析图表.

提供模型报告中评分分析专用图表，包括：
- KS曲线图，支持多数据集叠加 (score_ks_plot)
- 评分分布对比图 (score_distribution_comparison_plot)
- 评分分箱坏率图 (score_badrate_bin_plot)
- LIFT曲线图 (score_lift_plot)
- 审批通过率-坏率权衡曲线 (score_approval_badrate_curve)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils import DEFAULT_COLORS, get_or_create_ax, save_figure, setup_axis_style


def score_ks_plot(
    y_true=None,
    y_prob=None,
    datasets: Optional[Dict[str, Tuple]] = None,
    ax=None,
    figsize=(10, 6),
    title: str = 'KS曲线',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """KS曲线图，支持多数据集叠加.

    :param y_true: 真实标签（单数据集时使用）
    :param y_prob: 预测概率（单数据集时使用）
    :param datasets: {'训练集': (y_true, y_prob), '测试集': ...}
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = score_ks_plot(y_test, proba)
        >>> fig = score_ks_plot(datasets={'训练集': (y_tr, p_tr), '测试集': (y_te, p_te)})
    """
    if datasets is None:
        if y_true is None or y_prob is None:
            raise ValueError("需提供 y_true/y_prob 或 datasets")
        datasets = {'数据集': (y_true, y_prob)}

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    colors = DEFAULT_COLORS + ['#9C27B0', '#00BCD4', '#795548']

    for idx, (label, (yt, yp)) in enumerate(datasets.items()):
        yt = np.asarray(yt)
        yp = np.asarray(yp)
        sorted_idx = np.argsort(yp)[::-1]
        yt_sorted = yt[sorted_idx]
        n = len(yt)
        total_bad = yt.sum()
        total_good = n - total_bad
        tpr = np.cumsum(yt_sorted) / total_bad if total_bad > 0 else np.zeros(n)
        fpr = np.cumsum(1 - yt_sorted) / total_good if total_good > 0 else np.zeros(n)
        ks_vals = np.abs(tpr - fpr)
        ks = ks_vals.max()
        ks_idx = ks_vals.argmax()
        x_axis = np.linspace(0, 1, n)
        color = colors[idx % len(colors)]
        ax.plot(x_axis, tpr, color=color, linewidth=2,
                label=f'{label} 坏样本（KS={ks:.4f}）')
        ax.plot(x_axis, fpr, color=color, linewidth=1.5, linestyle='--',
                alpha=0.7, label=f'{label} 好样本')
        ax.vlines(x_axis[ks_idx], fpr[ks_idx], tpr[ks_idx],
                  color=color, linewidth=1.2, linestyle=':', alpha=0.8)
        ax.annotate(
            f'KS={ks:.4f}',
            xy=(x_axis[ks_idx], (tpr[ks_idx] + fpr[ks_idx]) / 2),
            xytext=(x_axis[ks_idx] + 0.03, (tpr[ks_idx] + fpr[ks_idx]) / 2),
            fontsize=9, color=color,
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4, label='随机')
    ax.set_xlabel('覆盖率（按概率降序）', fontsize=11)
    ax.set_ylabel('累积比例', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def score_distribution_comparison_plot(
    scores: Dict[str, Union[np.ndarray, pd.Series]],
    ax=None,
    figsize=(10, 5),
    title: str = '评分分布对比',
    bins: int = 50,
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """多数据集评分分布对比图（KDE + 直方图）.

    :param scores: {'训练集': score_array, '测试集': score_array}
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param bins: 直方图分箱数
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = score_distribution_comparison_plot({'训练集': s_tr, '测试集': s_te})
    """
    try:
        from scipy.stats import gaussian_kde
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    colors = DEFAULT_COLORS + ['#9C27B0', '#00BCD4']

    for idx, (label, arr) in enumerate(scores.items()):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        color = colors[idx % len(colors)]
        ax.hist(arr, bins=bins, alpha=0.25, color=color, density=True,
                label=f'{label}（n={len(arr):,}）')
        if _has_scipy and len(arr) > 2:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(arr, bw_method='scott')
            x_range = np.linspace(arr.min(), arr.max(), 300)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)

    ax.set_xlabel('评分', fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def score_badrate_bin_plot(
    y_true,
    score,
    n_bins: int = 10,
    ax=None,
    figsize=(12, 6),
    title: str = '评分分箱坏率',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """评分分箱坏率图：柱=样本量，折线=坏率.

    :param y_true: 真实标签
    :param score: 评分
    :param n_bins: 分箱数，默认10
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = score_badrate_bin_plot(y_test, model.predict_score(X_test))
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(s, quantiles)
    bin_edges = np.unique(bin_edges)
    labels_cut = pd.cut(s, bins=bin_edges, include_lowest=True)

    categories = labels_cut.categories if hasattr(labels_cut, 'categories') else labels_cut.cat.categories
    rows = []
    for i, bl in enumerate(categories):
        m = labels_cut == bl
        n = m.sum()
        n_bad = int(y[m].sum())
        br = n_bad / n if n > 0 else 0.0
        rows.append({'分箱': i + 1, '分箱标签': str(bl), '样本数': n,
                     '坏样本数': n_bad, '坏样本率': br})
    stat_df = pd.DataFrame(rows)

    x = np.arange(len(stat_df))
    overall_br = y.mean()

    fig, ax1 = get_or_create_ax(figsize=figsize, ax=ax)
    ax2 = ax1.twinx()

    ax1.bar(x, stat_df['样本数'].values, color=DEFAULT_COLORS[0], alpha=0.4,
            label='样本数', zorder=2)
    ax1.set_ylabel('样本数', color=DEFAULT_COLORS[0], fontsize=10)
    ax1.tick_params(axis='y', labelcolor=DEFAULT_COLORS[0])

    ax2.plot(x, stat_df['坏样本率'].values, color=DEFAULT_COLORS[1], marker='o',
             linewidth=2, markersize=6, label='坏样本率', zorder=3)
    ax2.axhline(overall_br, color='gray', linestyle='--', linewidth=1,
                label=f'整体坏率 {overall_br:.2%}')
    ax2.set_ylabel('坏样本率', color=DEFAULT_COLORS[1], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=DEFAULT_COLORS[1])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax1.set_xticks(x)
    ax1.set_xticklabels(stat_df['分箱标签'].tolist(), rotation=30, ha='right', fontsize=8)
    ax1.set_title(title, fontsize=13, fontweight='bold')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)
    setup_axis_style(ax1, hide_top_right=False)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def score_lift_plot(
    y_true=None,
    y_prob=None,
    ratios: List[float] = None,
    datasets: Optional[Dict[str, Tuple]] = None,
    ax=None,
    figsize=(10, 6),
    title: str = 'LIFT曲线',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """LIFT曲线图，支持多数据集叠加，标注关键点.

    :param y_true: 真实标签（单数据集）
    :param y_prob: 预测概率（单数据集）
    :param ratios: 覆盖率列表，默认 [0.01,0.03,0.05,0.10,0.20,0.30,0.50]
    :param datasets: {'训练集': (y_true, y_prob), ...}
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = score_lift_plot(y_test, proba)
    """
    if ratios is None:
        ratios = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
    if datasets is None:
        if y_true is None or y_prob is None:
            raise ValueError("需提供 y_true/y_prob 或 datasets")
        datasets = {'数据集': (y_true, y_prob)}

    fig, ax = get_or_create_ax(figsize=figsize, ax=ax)
    colors = DEFAULT_COLORS + ['#9C27B0', '#00BCD4', '#795548']
    highlight = {0.01, 0.05, 0.10}

    for idx, (label, (yt, yp)) in enumerate(datasets.items()):
        yt = np.asarray(yt)
        yp = np.asarray(yp)
        total = len(yt)
        total_bad = yt.sum()
        overall_br = total_bad / total if total > 0 else 0.0
        sorted_idx = np.argsort(yp)[::-1]
        yt_sorted = yt[sorted_idx]
        lifts = []
        for ratio in ratios:
            n_top = max(1, int(total * ratio))
            br = yt_sorted[:n_top].mean()
            lifts.append(br / overall_br if overall_br > 0 else 0.0)
        color = colors[idx % len(colors)]
        ax.plot(ratios, lifts, color=color, marker='o', linewidth=2,
                markersize=5, label=label)
        for r, lv in zip(ratios, lifts):
            if r in highlight:
                ax.annotate(f'{lv:.2f}', xy=(r, lv),
                            xytext=(r + 0.005, lv + 0.05),
                            fontsize=8, color=color)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
               label='随机（LIFT=1）')
    ax.set_xlabel('覆盖率', fontsize=11)
    ax.set_ylabel('LIFT值', fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    setup_axis_style(ax, hide_top_right=True)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def score_approval_badrate_curve(
    y_true,
    score,
    score_ascending: bool = True,
    n_points: int = 100,
    ax=None,
    figsize=(10, 6),
    title: str = '通过率 - 坏率权衡曲线',
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """审批通过率 vs 坏账率曲线（策略人员必备）.

    :param y_true: 真实标签
    :param score: 评分
    :param score_ascending: True=分数越高越安全（低分拒绝）
    :param n_points: 曲线采样点数
    :param ax: matplotlib Axes
    :param figsize: 图像尺寸
    :param title: 图标题
    :param save: 保存路径
    :return: Figure

    Example:
        >>> fig = score_approval_badrate_curve(y_test, model.predict_score(X_test))
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]
    n = len(y)
    overall_br = y.mean()

    thresholds = np.percentile(s, np.linspace(0, 100, n_points + 2)[1:-1])
    approval_rates, pass_brs, reject_brs = [], [], []
    for thr in thresholds:
        pass_mask = s >= thr if score_ascending else s <= thr
        n_pass = pass_mask.sum()
        n_reject = n - n_pass
        approval_rates.append(n_pass / n)
        pass_brs.append(y[pass_mask].mean() if n_pass > 0 else 0.0)
        reject_brs.append(y[~pass_mask].mean() if n_reject > 0 else 0.0)

    approval_rates = np.array(approval_rates)
    pass_brs = np.array(pass_brs)
    reject_brs = np.array(reject_brs)

    fig, ax1 = get_or_create_ax(figsize=figsize, ax=ax)
    ax2 = ax1.twinx()

    ax1.plot(approval_rates, pass_brs, color=DEFAULT_COLORS[1], linewidth=2,
             label='通过样本坏率')
    ax1.plot(approval_rates, reject_brs, color=DEFAULT_COLORS[0], linewidth=2,
             linestyle='--', label='拒绝样本坏率')
    ax1.axhline(overall_br, color='gray', linestyle=':', linewidth=1,
                label=f'整体坏率 {overall_br:.2%}')
    ax1.set_xlabel('通过率', fontsize=11)
    ax1.set_ylabel('坏账率', fontsize=11)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax2.fill_between(approval_rates, approval_rates * n, alpha=0.08,
                     color=DEFAULT_COLORS[2], label='通过样本数')
    ax2.set_ylabel('通过样本数', color=DEFAULT_COLORS[2], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=DEFAULT_COLORS[2])

    ax1.set_title(title, fontsize=13, fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper right')
    setup_axis_style(ax1, hide_top_right=False)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


__all__ = [
    'score_ks_plot',
    'score_distribution_comparison_plot',
    'score_badrate_bin_plot',
    'score_lift_plot',
    'score_approval_badrate_curve',
]
