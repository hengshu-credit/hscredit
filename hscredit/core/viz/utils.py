# -*- coding: utf-8 -*-
"""
可视化工具函数.

提供公共的可视化辅助函数，减少代码重复。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any


# 默认配色方案
DEFAULT_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]


def setup_axis_style(ax, colors: Optional[list] = None, hide_top_right: bool = False):
    """设置坐标轴样式.
    
    :param ax: matplotlib Axes 对象
    :param colors: 边框颜色
    :param hide_top_right: 是否隐藏顶部和右侧边框
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    color = colors[0] if colors else "#2639E9"
    
    ax.spines['top'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)
    
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def save_figure(fig, save_path: Optional[str] = None, dpi: int = 240):
    """保存图表.
    
    :param fig: matplotlib Figure 对象
    :param save_path: 保存路径
    :param dpi: 分辨率
    """
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, format="png", bbox_inches="tight")


def get_or_create_ax(figsize: Tuple[float, float] = (10, 6), 
                     ax: Optional[Any] = None,
                     return_fig: bool = True) -> Tuple[Any, ...]:
    """获取或创建 Axes 对象.
    
    如果传入 ax，则直接使用；否则创建新的 Figure 和 Axes。
    
    :param figsize: 图像尺寸（创建新图时使用）
    :param ax: 可选的 matplotlib Axes 对象
    :param return_fig: 是否返回 Figure 对象
    :return: 如果 return_fig=True 返回 (fig, ax)，否则返回 ax

    **参考样例**

    >>> # 方式1：自动创建新的 figure 和 ax
    >>> fig, ax = get_or_create_ax(figsize=(10, 6))
    >>>
    >>> # 方式2：使用传入的 ax
    >>> import matplotlib.pyplot as plt
    >>> _, axes = plt.subplots(2, 3, figsize=(18, 10))
    >>> for i, col in enumerate(features):
    ...     _, ax = get_or_create_ax(ax=axes[i])
    ...     # 绘图...
    """
    if ax is not None:
        if return_fig:
            # 尝试从 ax 获取 figure
            fig = ax.figure
            return fig, ax
        else:
            return ax
    
    # 创建新的 figure 和 ax
    fig, ax = plt.subplots(figsize=figsize)
    
    if return_fig:
        return fig, ax
    else:
        return ax


def create_legend(fig_or_ax, loc: str = 'upper center', 
                  bbox_to_anchor: Tuple[float, float] = (0.5, 0.98),
                  ncol: int = 2, frameon: bool = False,
                  handles: Optional[list] = None, 
                  labels: Optional[list] = None):
    """创建图例.
    
    :param fig_or_ax: Figure 或 Axes 对象
    :param loc: 图例位置
    :param bbox_to_anchor: 锚点位置
    :param ncol: 列数
    :param frameon: 是否显示边框
    :param handles: 图例句柄（可选）
    :param labels: 图例标签（可选）
    :return: Legend 对象
    """
    if handles is None or labels is None:
        if hasattr(fig_or_ax, 'get_legend_handles_labels'):
            handles, labels = fig_or_ax.get_legend_handles_labels()
        else:
            return None
    
    legend = fig_or_ax.legend(
        handles, labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=frameon
    )
    
    return legend


def format_bin_label(label: str, max_len: int = 35) -> str:
    """格式化分箱标签.
    
    :param label: 原始标签
    :param max_len: 最大长度
    :return: 格式化后的标签
    """
    import re
    
    if pd.isnull(label):
        return "缺失值"
    
    label_str = str(label)
    
    # 检查是否符合区间格式 [x, y)
    if re.match(r"^\[.*\)$", label_str):
        return label_str
    
    # 截断过长的标签
    if len(label_str) > max_len:
        return label_str[:max_len] + "..."
    
    return label_str


# 导入 pandas 用于类型检查
try:
    import pandas as pd
except ImportError:
    pd = None
