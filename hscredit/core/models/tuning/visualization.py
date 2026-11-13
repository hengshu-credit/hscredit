"""绑定调参 Study，并为 Optuna 原生图形应用 hscredit 样式。"""

from functools import wraps
import inspect
import json
from math import ceil
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tuning import ModelTuner


_PLOT_TITLES = {
    "plot_pareto_front": "多目标优化 · 帕累托前沿",
    "plot_optimization_history": "优化历史",
    "plot_param_importances": "超参数重要性",
    "plot_slice": "参数切片图",
    "plot_contour": "参数等高线图",
    "plot_parallel_coordinate": "平行坐标图",
    "plot_edf": "经验分布函数",
    "plot_hypervolume_history": "超体积变化",
    "plot_intermediate_values": "试验中间值",
    "plot_terminator_improvement": "终止判定改进量",
    "plot_timeline": "试验时间线",
}

# 只翻译 Optuna 自有坐标，不覆盖模型指标、搜索参数或调用者传入的标签。
_FIXED_AXES = {
    "plot_optimization_history": ("试验编号", None),
    "plot_hypervolume_history": ("试验编号", "超体积"),
    "plot_intermediate_values": ("迭代步数", "中间值"),
    "plot_param_importances": ("重要性", "超参数"),
    "plot_edf": (None, "累积概率"),
    "plot_timeline": ("时间", "试验编号"),
    "plot_terminator_improvement": ("试验编号", "终止判定改进量"),
}


def _readable_pareto_hover(text):
    # Optuna 用 <br> 连接 JSON 各行。只重写字符串，保留换行标记和参数中的字面反斜杠。
    if r"\u" not in text:
        return text
    return re.sub(r'"(?:[^"\\]|\\.)*"', lambda match: json.dumps(json.loads(match[0]), ensure_ascii=False), text)


def _style_pareto_marker(marker, palette):
    from ...viz.style import get_palette

    best = marker.colorbar.title.text == "Best Trial"
    marker.update(
        colorscale=get_palette("red" if best else "blue"),
        size=9 if best else 7,
        line=dict(width=1, color=palette[1] if best else palette[0]),
    )
    colorbar = marker.colorbar
    colorbar.update(title_text="前沿试验" if best else "普通试验", thickness=13)
    numbers = marker.color
    if numbers is not None and len(numbers):
        low, high = min(numbers), max(numbers)
        colorbar.update(tickmode="linear", tick0=low, dtick=max(1, ceil((high - low) / 6)))


def _style_plotly(figure, name):
    """在原图上设置外观，保留坐标、分组、悬停内容、色阶方向和交互配置。"""
    if not hasattr(figure, "update_layout"):
        return figure

    from matplotlib import rcParams
    from ...viz.style import get_font_sizes, get_palette

    palette = get_palette("extended")
    primary = palette[0]
    family = ", ".join([*rcParams["font.family"], "Microsoft YaHei", "sans-serif"])
    # Matplotlib 使用 pt，Plotly 使用 CSS px。
    sizes = {key: round(value * 4 / 3) for key, value in get_font_sizes().items()}
    title = _PLOT_TITLES.get(name, figure.layout.title.text)
    if name == "plot_rank" and title and title.startswith("Rank ("):
        title = "参数排名 " + title[5:]
    figure.update_layout(
        title=dict(text=f"<b>{title}</b>" if name in _PLOT_TITLES else title, x=0.5, xanchor="center", font=dict(size=sizes["title"])),
        font=dict(family=family, size=sizes["tick"]),
        legend_font_size=sizes["legend"],
        hoverlabel=dict(bgcolor="white", bordercolor=primary, font=dict(family=family, size=sizes["tick"])),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=palette,
    )
    if name == "plot_pareto_front":
        figure.update_layout(hoverlabel=dict(align="left", font=dict(family=f"Consolas, {family}", color="#29243D")))
    blue = get_palette("blue")
    axis = dict(
        showline=True,
        linecolor=primary,
        color=primary,
        title_font_size=sizes["axis_label"],
        tickfont_size=sizes["tick"],
        showgrid=True,
        gridcolor=blue[0],
        zeroline=False,
    )
    if any(trace.type == "scatter3d" for trace in figure.data):
        scene_axis = dict(axis, backgroundcolor="white")
        figure.update_scenes(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis)
    else:
        figure.update_xaxes(**axis)
        figure.update_yaxes(**axis)
        xlabel, ylabel = _FIXED_AXES.get(name, (None, None))
        if xlabel:
            figure.update_xaxes(title_text=xlabel)
        if ylabel:
            figure.update_yaxes(title_text=ylabel)

    semantic = get_palette("semantic")
    state_colors = {
        "COMPLETE": primary,
        "RUNNING": semantic["stable"],
        "FAIL": semantic["negative"],
        "PRUNED": palette[2],
        "WAITING": semantic["neutral"],
        "INFEASIBLE": semantic["neutral"],
    }
    for trace in figure.data:
        if name == "plot_pareto_front" and trace.text:
            trace.text = [_readable_pareto_hover(text) for text in trace.text]
        if name == "plot_timeline" and trace.type == "bar" and trace.name in state_colors:
            trace.marker.color = state_colors[trace.name]
        elif trace.type == "contour":
            trace.colorscale = blue
        elif trace.type == "parcoords":
            trace.line.colorscale = blue
        elif trace.type in {"scatter", "scatter3d"} and trace.marker.colorscale:
            # Rank 已把每个点转为 RGB，必须保留与这些点配套的原生色条。
            if name == "plot_rank":
                continue
            if name == "plot_pareto_front":
                _style_pareto_marker(trace.marker, palette)
            else:
                trace.marker.colorscale = blue
        elif trace.type in {"scatter", "scatter3d"} and isinstance(trace.marker.color, str) and trace.marker.color.lower() == "#cccccc":
            trace.marker.color = semantic["neutral"]
    return figure


def _style_matplotlib(result, name):
    """装饰原生 Axes（包括子图数组），保留各子图的尺度及关联色条。"""
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.colors import LinearSegmentedColormap
    from ...viz.style import get_defaults, get_font_sizes, get_palette
    from ...viz.utils import setup_axis_style

    axes = np.asarray(result, dtype=object).ravel()
    if not len(axes) or not isinstance(axes[0], Axes):
        return result
    figure = axes[0].figure
    figure.set_facecolor("white")
    fonts = get_font_sizes()
    palette = get_palette()
    neutral = get_palette("semantic")["neutral"]
    grid_alpha = get_defaults()["grid_alpha"]
    pareto_groups = {
        "Best Trial": (palette[1], "前沿试验"),
        "Trial": (palette[0], "普通试验"),
        "Feasible Trial": (palette[0], "可行试验"),
        "Infeasible Trial": (neutral, "不可行试验"),
    }
    blue = LinearSegmentedColormap.from_list("hscredit_blue", get_palette("blue"))
    title = _PLOT_TITLES.get(name)
    if title and figure._suptitle is not None:
        figure.suptitle(title, fontsize=fonts["title"], fontweight="bold")
    xlabel, ylabel = _FIXED_AXES.get(name, (None, None))
    for ax in axes:
        ax.set_facecolor("white")
        setup_axis_style(ax, hide_top_right=True)
        ax.grid(True, color=neutral, alpha=grid_alpha)
        ax.tick_params(labelsize=fonts["tick"])
        for loc in ("left", "center", "right"):
            if ax.get_title(loc=loc):
                ax.set_title(title or ax.get_title(loc=loc), loc=loc, fontsize=fonts["title"], fontweight="bold")
        ax.set_xlabel(xlabel or ax.get_xlabel(), fontsize=fonts["axis_label"])
        ax.set_ylabel(ylabel or ax.get_ylabel(), fontsize=fonts["axis_label"])
        if hasattr(ax, "zaxis"):
            ax.set_zlabel(ax.get_zlabel(), fontsize=fonts["axis_label"], color=palette[0])
        for collection in ax.collections:
            cmap = collection.get_cmap()
            if cmap.name in {"Blues", "Blues_r"}:
                collection.set_cmap(blue if cmap.name == "Blues" else blue.reversed())
            if name == "plot_pareto_front" and collection.get_label() in pareto_groups:
                color, label = pareto_groups[collection.get_label()]
                collection.set_color(color)
                collection.set_label(label)
        if name == "plot_pareto_front" and ax.get_legend() is not None:
            ax.legend(frameon=False, fontsize=fonts["legend"])
    return result


class _OptunaVisualization:
    """按安装版本公开全部绘图函数，保留原生参数、返回值和错误语义。"""

    def __init__(self, tuner: "ModelTuner", module, *, interactive=True):
        self._tuner = tuner
        self._module = module
        self._interactive = interactive

    def __dir__(self):
        return sorted(set(super().__dir__()) | {name for name in dir(self._module) if not name.startswith("_")})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "matplotlib" and self._interactive:
            return type(self)(self._tuner, self._module.matplotlib, interactive=False)
        try:
            function = getattr(self._module, name)
        except AttributeError as exc:
            raise AttributeError(f"当前 Optuna 可视化模块没有 {name!r}，请用 dir(tuner.visualization) 查看可用入口") from exc
        if not name.startswith("plot_") or not callable(function):
            return function

        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].name not in {"study", "studies"}:
            return function
        study_parameter = parameters[0]

        @wraps(function)
        def plot(*args, **kwargs):
            # 显式 study/studies 可用于原生的多 Study 对比，其余调用绑定当前搜索。
            if study_parameter.name not in kwargs:
                study = self._tuner.get_study()
                if study_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwargs[study_parameter.name] = study
                else:
                    args = (study, *args)
            if self._interactive:
                return _style_plotly(function(*args, **kwargs), name)
            # Optuna 的静态绘图会调用 plt.style.use("ggplot")，将其限制在本次调用内。
            from matplotlib import rc_context

            with rc_context():
                result = function(*args, **kwargs)
            return _style_matplotlib(result, name)

        plot.__signature__ = signature.replace(parameters=parameters[1:])
        return plot
