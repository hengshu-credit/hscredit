# -*- coding: utf-8 -*-
"""
可视化工具函数.

提供公共的可视化辅助函数，减少代码重复。
"""

import math
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple, Any

# 统一配色：以 style 模块为唯一来源，所有图表均引用这些常量，
# 后续扩充色阶时只需在 style.py 中调整一处即可全局生效。
from .style import (
    PRIMARY_COLORS,
    EXTENDED_COLORS,
    SEMANTIC_COLORS,
    GRADIENT_PALETTES,
)


# 默认配色方案：主题色 + 2 个副主题色（与 PRIMARY_COLORS 一致）
DEFAULT_COLORS = list(PRIMARY_COLORS)

# 语义色（与 bin_plot 参考样式一致）：坏样本率折线、整体基线参考线
BAD_RATE_COLOR = SEMANTIC_COLORS["bad_rate"]            # #F0556F 坏样本率折线（副主题红融粉）
REFERENCE_COLOR = SEMANTIC_COLORS["overall_baseline"]  # #2639E9 整体基线参考线（主题蓝）
# PSI/稳定性状态语义色
STABLE_COLOR = SEMANTIC_COLORS["stable"]
CHANGING_COLOR = SEMANTIC_COLORS["changing"]
UNSTABLE_COLOR = SEMANTIC_COLORS["unstable"]
NEUTRAL_COLOR = SEMANTIC_COLORS["neutral"]
# 风险渐变色阶（浅蓝→玫紫→粉→深红），用于热力图/风险等级连续着色
RISK_GRADIENT = list(GRADIENT_PALETTES["risk"])
# 主题蓝色阶（浅→深，主题色叠白），用于「数值越大越好/越强」的顺序着色（如 IV 强度）
BLUE_GRADIENT = list(GRADIENT_PALETTES["blue"])
# 蓝→紫→粉→红 连续色阶，用于热力图/条件格式
SEQUENTIAL_GRADIENT = list(GRADIENT_PALETTES["blue_purple_red"])
# 发散色阶，用于相关性、改善率等可正可负指标
DIVERGING_GRADIENT = list(GRADIENT_PALETTES["diverging"])


def _axes_top_boundary(axes: list, renderer: Any) -> float:
    """返回坐标轴、轴装饰和显式顶部头部元素共同占用的最高像素位置。"""
    tops = []
    for ax in axes:
        tops.append(ax.get_window_extent(renderer).y1)
        if ax.xaxis.get_visible():
            xaxis_bbox = ax.xaxis.get_tightbbox(renderer)
            if xaxis_bbox is not None and math.isfinite(xaxis_bbox.y1):
                tops.append(xaxis_bbox.y1)

        summaries = [
            artist
            for artist in [*ax.texts, *ax.artists]
            if artist.get_visible() and artist.get_gid() == 'bin-metric-summary'
        ]
        if summaries:
            title = ax.title
            if title.get_visible() and title.get_text().strip():
                title_bbox = title.get_window_extent(renderer)
                if math.isfinite(title_bbox.y1):
                    tops.append(title_bbox.y1)

            for text in summaries:
                text_bbox = text.get_window_extent(renderer)
                if math.isfinite(text_bbox.y1):
                    tops.append(text_bbox.y1)
    return max(tops)


def _layout_top_center_legend(
    fig: Any,
    legend: Any,
    *,
    title: Optional[Any] = None,
    axes: Optional[list] = None,
    min_gap_points: float = 6.0,
) -> None:
    """将图例等距放在标题与坐标轴上边界之间。"""
    if legend is None:
        return

    title = title or getattr(fig, "_suptitle", None)
    if title is None or not title.get_text().strip():
        return
    axes = [ax for ax in (axes or fig.axes) if ax.get_visible()]
    if not axes:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    title_bbox = title.get_window_extent(renderer)
    axes_top = _axes_top_boundary(axes, renderer)
    min_gap_pixels = float(min_gap_points) * fig.dpi / 72.0
    required_height = legend_bbox.height + 2.0 * min_gap_pixels
    available_height = title_bbox.y0 - axes_top

    # 空间不足时只压低坐标轴上边界，标题位置和坐标轴底边保持不变。
    deficit = required_height - available_height
    fig_height = float(fig.bbox.height)
    if deficit > 0 and math.isfinite(fig_height) and fig_height > 0:
        deficit_fraction = deficit / fig_height
        for ax in axes:
            position = ax.get_position()
            new_height = position.height - deficit_fraction
            if new_height > 0.05:
                ax.set_position((position.x0, position.y0, position.width, new_height))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bbox = title.get_window_extent(renderer)
        axes_top = _axes_top_boundary(axes, renderer)
        legend_bbox = legend.get_window_extent(renderer)

    target_center = (title_bbox.y0 + axes_top) / 2.0
    current_center = (legend_bbox.y0 + legend_bbox.y1) / 2.0
    shift_pixels = target_center - current_center
    if not math.isfinite(shift_pixels) or not math.isfinite(fig_height) or fig_height <= 0:
        return

    anchor_bbox = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    legend.set_bbox_to_anchor(
        (0.5, anchor_bbox.y0 + shift_pixels / fig_height),
        transform=fig.transFigure,
    )


def make_colormap(name: str, colors: Optional[list] = None, n: int = 256):
    """根据统一色板创建 matplotlib colormap.

    :param name: colormap 名称
    :param colors: 颜色列表，默认使用蓝紫粉红连续色阶
    :param n: 颜色采样数
    :return: LinearSegmentedColormap
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        colors or SEQUENTIAL_GRADIENT,
        N=n,
    )


def make_risk_cmap(name: str = "hscredit_risk", n: int = 256):
    """创建风险连续色阶（低风险蓝紫 → 高风险粉红）。

    :param name: colormap 名称，默认 ``"hscredit_risk"``
    :param n: 颜色采样数，默认 256
    :return: ``LinearSegmentedColormap``，可直接传给热力图的 ``cmap`` 参数
    """
    return make_colormap(name, RISK_GRADIENT, n=n)


def make_diverging_cmap(name: str = "hscredit_diverging", n: int = 256):
    """创建发散色阶（主题蓝 → 近白 → 副主题红），适合带正负/中心值的热力图。

    :param name: colormap 名称，默认 ``"hscredit_diverging"``
    :param n: 颜色采样数，默认 256
    :return: ``LinearSegmentedColormap``
    """
    return make_colormap(name, DIVERGING_GRADIENT, n=n)


def get_psi_color(value: float) -> str:
    """根据 PSI 取值返回统一语义色（用于稳定性图表着色）。

    :param value: PSI 值
    :return: 颜色十六进制字符串：

        - ``value < 0.10``：稳定色（STABLE_COLOR）
        - ``0.10 <= value < 0.25``：变化色（CHANGING_COLOR）
        - ``value >= 0.25``：不稳定色（UNSTABLE_COLOR）
    """
    if value < 0.10:
        return STABLE_COLOR
    if value < 0.25:
        return CHANGING_COLOR
    return UNSTABLE_COLOR


def get_series_colors(n: int) -> list:
    """获取 n 条数据系列的统一配色（主题色 + 副主题色 + 扩展色循环）.

    :param n: 需要的颜色数量
    :return: 长度为 n 的颜色列表，优先使用主题色与副主题色，再循环扩展色板
    """
    if n <= 0:
        return []
    base = list(EXTENDED_COLORS)
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def setup_axis_style(ax, colors: Optional[list] = None, hide_top_right: bool = False):
    """设置坐标轴样式.
    
    :param ax: matplotlib Axes 对象
    :param colors: 边框颜色
    :param hide_top_right: 是否隐藏顶部和右侧边框
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    color = colors[0] if colors else "#2639E9"
    
    for spine in ax.spines.values():
        spine.set_color(color)
    ax.tick_params(axis='both', colors=color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def save_figure(fig, save_path: Optional[str] = None, dpi: int = 240):
    """保存图表.
    
    :param fig: matplotlib Figure 对象
    :param save_path: 保存路径（为 None 时不保存，直接返回）；目录不存在时自动创建
    :param dpi: 分辨率，默认 240

    **参考样例**

    >>> from hscredit.core.viz import save_figure
    >>> save_figure(fig, 'output/分箱图.png', dpi=300)
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
