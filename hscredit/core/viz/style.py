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
from typing import Dict, List, Optional


# ============================================================
# 配色方案
# ============================================================

# 主色板（主题色 + 2 个副主题色）
PRIMARY_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]

# 扩展色板：保留主色板顺序，其余颜色全部由两大主题色派生——
#   · 蓝系 = 主题色 #2639E9 叠不同透明度白（浅→深），非通用 blue；
#   · 红系 = 副主题色 #F76E6C 融合粉色 #E0249A 或主题蓝（→玫紫/粉红/危险红），非通用 red。
# 蓝、紫、粉、红渐变排布，同时相邻系列色差差距拉大，全程无黄/绿/棕分散色相。
EXTENDED_COLORS = PRIMARY_COLORS + [
    "#8892F3", "#9956A4", "#EA4585", "#5665EE", "#F0587A",
    "#B729AB", "#3C4DEB", "#E73B72", "#832EC2", "#BC5F8F",
    "#E0249A",
]

# 语义色：蓝系基于主题色 #2639E9，红系基于副主题色 #F76E6C 融合粉/蓝
SEMANTIC_COLORS = {
    "bad_rate": "#F0556F",     # 副主题红融合粉，坏样本率折线
    "overall_baseline": "#2639E9",
    "stable": "#7884F1",       # PSI < 0.1，主题蓝叠白
    "changing": "#9956A4",     # 0.1 <= PSI < 0.25，红融蓝→玫紫
    "unstable": "#E73B72",     # PSI >= 0.25，红融粉危险色
    "positive": "#2639E9",
    "negative": "#E04566",
    "neutral": "#8A8FA3",
    "reference": "#8A8FA3",
}

# 渐变色板（适合热力图/连续值）：蓝系叠白、红系融粉/蓝，相邻锚点色差刻意拉开，无黄/绿
GRADIENT_PALETTES = {
    # 低风险→高风险：浅蓝 → 玫紫 → 粉 → 深红，四档色相大跨度区分
    "risk": ["#8892F3", "#9956A4", "#EA4585", "#E43550"],
    # 主题蓝叠白（浅→深），用于「数值越大越好/越强」的顺序着色
    "blue": ["#E5E7FC", "#ADB4F7", "#707CF0", "#2639E9"],
    # 发散：主题蓝 → 近白 → 副主题红（融粉），用于可正可负指标
    "diverging": ["#2639E9", "#F6F7FE", "#F0556F"],
    # 蓝→紫→粉→红 分段类别色（红系由 #F76E6C 融粉/融蓝派生）
    "pink_purple": [
        "#8892F3", "#2639E9", "#832EC2", "#B729AB",
        "#E0249A", "#EC4983", "#F0587A", "#E43550",
    ],
    # 蓝→紫→粉→红 平滑连续色阶：色相单调递进(234→351°)且相邻锚点色差拉大，蓝系叠白、红系融粉，
    # 全程无黄/绿，适合热力图/条件格式色阶；作为 Excel 条件格式 condition_color 锚点时自动取首/中/尾构成三色异色阶
    "blue_purple_red": [
        "#A4ACF6", "#2639E9", "#8C2DBE", "#B729AB",
        "#E0249A", "#EC4983", "#E43550",
    ],
}

_PALETTES = {
    "default": PRIMARY_COLORS,
    "primary": PRIMARY_COLORS,
    "extended": EXTENDED_COLORS,
    "pink_purple": GRADIENT_PALETTES["pink_purple"],
    "semantic": SEMANTIC_COLORS,
}


def get_palette(name: str = "default"):
    """获取配色方案.

    :param name: 方案名称，可选 'default'/'primary'(3色), 'extended', 'pink_purple', 'semantic'(语义色字典)
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
