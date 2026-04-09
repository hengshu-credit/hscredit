# -*- coding: utf-8 -*-
"""
可视化统一样式系统.

提供主题管理、配色方案、字体配置等全局样式设置，
确保所有图表风格一致。

用法::

    from hscredit.core.viz import set_style, get_palette, get_font_sizes

    # 应用风控主题（推荐）
    set_style("risk")

    # 获取配色
    colors = get_palette("default")
    semantic = get_palette("semantic")

    # 获取字体大小层级
    fonts = get_font_sizes()  # {'title': 14, 'subtitle': 13, ...}
"""

import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


# ============================================================
# 配色方案
# ============================================================

# 主色板（3色）
PRIMARY_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]

# 扩展色板（6色）
EXTENDED_COLORS = PRIMARY_COLORS + ["#9C27B0", "#00BCD4", "#795548"]

# 语义色
SEMANTIC_COLORS = {
    "bad_rate": "#E85D4A",
    "overall_baseline": "#4C8DFF",
    "stable": "#4CAF50",       # PSI < 0.1
    "changing": "#FF9800",     # 0.1 <= PSI < 0.25
    "unstable": "#F44336",     # PSI >= 0.25
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E",
    "reference": "gray",
}

# 渐变色板（适合热力图/连续值）
GRADIENT_PALETTES = {
    "risk": ["#4CAF50", "#FFC107", "#FF9800", "#F44336"],  # 绿→黄→橙→红
    "blue": ["#E3F2FD", "#90CAF9", "#42A5F5", "#1565C0"],
    "diverging": ["#2639E9", "#FFFFFF", "#F76E6C"],
}

_PALETTES = {
    "default": PRIMARY_COLORS,
    "primary": PRIMARY_COLORS,
    "extended": EXTENDED_COLORS,
    "semantic": SEMANTIC_COLORS,
}


def get_palette(name: str = "default"):
    """获取配色方案.

    :param name: 方案名称，可选 'default'/'primary'(3色), 'extended'(6色), 'semantic'(语义色字典)
    :return: 颜色列表或字典
    """
    if name in _PALETTES:
        return _PALETTES[name]
    if name in GRADIENT_PALETTES:
        return GRADIENT_PALETTES[name]
    raise ValueError(f"未知配色方案 '{name}'，可选: {list(_PALETTES.keys()) + list(GRADIENT_PALETTES.keys())}")


# ============================================================
# 字体层级
# ============================================================

_FONT_SIZES = {
    "title": 14,
    "subtitle": 13,
    "axis_label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
    "small": 8,
}


def get_font_sizes() -> Dict[str, int]:
    """获取字体大小层级."""
    return dict(_FONT_SIZES)


# ============================================================
# 默认参数
# ============================================================

_DEFAULTS = {
    "dpi": 240,
    "figsize": (10, 6),
    "grid_alpha": 0.3,
    "bar_alpha": 0.5,
    "line_alpha": 0.85,
    "fontweight_title": "bold",
    "fontweight_label": "bold",
}


def get_defaults() -> dict:
    """获取全局默认参数."""
    return dict(_DEFAULTS)


# ============================================================
# 中文字体自动检测
# ============================================================

def _detect_cjk_fonts() -> List[str]:
    """根据操作系统检测可用的中文字体列表."""
    system = platform.system()
    if system == "Darwin":
        candidates = ["PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
    elif system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "SimSun"]
    else:  # Linux
        candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Droid Sans Fallback"]

    available = []
    try:
        from matplotlib.font_manager import fontManager
        system_fonts = {f.name for f in fontManager.ttflist}
        for font in candidates:
            if font in system_fonts:
                available.append(font)
    except Exception:
        pass
    return available


# ============================================================
# 主题定义
# ============================================================

_THEMES: Dict[str, dict] = {
    "risk": {
        "figure.dpi": 100,
        "savefig.dpi": 240,
        "figure.figsize": (10, 6),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2639E9",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    },
    "minimal": {
        "figure.dpi": 100,
        "savefig.dpi": 240,
        "figure.figsize": (10, 6),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    },
    "report": {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.figsize": (12, 7),
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.frameon": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    },
}

_current_theme: Optional[str] = None


def set_style(theme: str = "risk", chinese_font: bool = True):
    """设置全局可视化主题.

    :param theme: 主题名称，可选 'risk'(默认风控主题), 'minimal'(极简), 'report'(报告用)
    :param chinese_font: 是否自动配置中文字体支持
    :raises ValueError: 未知主题名称

    用法::

        from hscredit.core.viz import set_style

        set_style("risk")         # 标准风控主题
        set_style("report")       # 报告导出主题（高DPI）
        set_style("minimal")      # 极简主题
    """
    global _current_theme

    if theme not in _THEMES:
        raise ValueError(f"未知主题 '{theme}'，可选: {list(_THEMES.keys())}")

    # 重置为 matplotlib 默认，再叠加主题
    mpl.rcdefaults()

    params = dict(_THEMES[theme])

    # 中文字体
    if chinese_font:
        cjk_fonts = _detect_cjk_fonts()
        if cjk_fonts:
            params["font.sans-serif"] = cjk_fonts + ["DejaVu Sans", "Arial"]
            params["axes.unicode_minus"] = False

    mpl.rcParams.update(params)
    _current_theme = theme


def get_current_theme() -> Optional[str]:
    """获取当前已应用的主题名称."""
    return _current_theme


def reset_style():
    """重置为 matplotlib 默认样式."""
    global _current_theme
    mpl.rcdefaults()
    _current_theme = None
