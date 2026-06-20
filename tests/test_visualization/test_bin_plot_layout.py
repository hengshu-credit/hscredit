"""测试分箱图的刻度方向、指标角标开关与小图自适应布局。"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from hscredit.core.binning import OptimalBinning2D
from hscredit.core.viz.binning_plots import bin_2d_plot, bin_plot, _xtick_rotation_for_length


def _numeric_feature_table():
    """构造含 IV/KS/LIFT 的数值型分箱表（样本数为短数值，刻度默认垂直）。"""
    return pd.DataFrame(
        {
            "分箱": [0, 1, 2],
            "分箱标签": ["[0, 100)", "[100, 200)", "[200, inf)"],
            "好样本数": [80, 60, 50],
            "坏样本数": [20, 40, 50],
            "样本总数": [100, 100, 100],
            "坏样本率": [0.2, 0.4, 0.5],
            "指标IV值": [0.1, 0.2, 0.35],
            "分档KS值": [0.1, 0.2, 0.15],
            "LIFT值": [0.7, 1.4, 1.8],
        }
    )


def _summary_text(fig):
    """返回左上角指标角标的 Text 对象（不存在则为 None）。"""
    for text in fig.texts:
        if "IV" in text.get_text():
            return text
    return None


# ---------------------------------------------------------------------------
# 需求1：横向模式 x 轴刻度默认垂直，过长时按最大长度动态倾斜
# ---------------------------------------------------------------------------
def test_xtick_rotation_for_length_logic():
    # 未超过阈值保持垂直
    assert _xtick_rotation_for_length(3) == (90, "center")
    assert _xtick_rotation_for_length(5) == (90, "center")
    # 超过阈值后倾斜，且文本越长角度越小
    assert _xtick_rotation_for_length(6) == (80, "right")
    assert _xtick_rotation_for_length(8) == (60, "right")
    # 长度极大时夹到下限 30°
    assert _xtick_rotation_for_length(50)[0] == 30


def test_horizontal_xticks_default_vertical():
    fig = bin_plot(_numeric_feature_table(), orientation="horizontal", figsize=(12, 7))
    bottom_axis = fig.axes[0]  # 横向模式下 ax1 承载底部样本数 x 轴
    rotations = [t.get_rotation() for t in bottom_axis.get_xticklabels()]
    assert rotations and all(r == 90 for r in rotations)


def test_vertical_orientation_bin_labels_default_vertical():
    # 纵向模式下分箱标签默认垂直展示（常规长度不旋转）
    fig = bin_plot(_numeric_feature_table(), orientation="vertical", figsize=(12, 7))
    rotations = [t.get_rotation() for t in fig.axes[0].get_xticklabels()]
    assert rotations and all(r == 90 for r in rotations)


def test_vertical_orientation_bin_labels_slant_when_very_long():
    # 分箱标签非常长（超过阈值 18）时改为倾斜
    table = _numeric_feature_table()
    table["分箱标签"] = [
        "[非常长的类别名称A, 非常长的类别名称B)",
        "[非常长的类别名称C, 非常长的类别名称D)",
        "[非常长的类别名称E, +inf)",
    ]
    fig = bin_plot(table, orientation="vertical", figsize=(12, 7), max_len=40)
    rotations = [t.get_rotation() for t in fig.axes[0].get_xticklabels()]
    assert rotations and all(30 <= r < 90 for r in rotations)


# ---------------------------------------------------------------------------
# 需求2：show_metric_summary 控制左上角角标是否展示，默认展示
# ---------------------------------------------------------------------------
def test_metric_summary_shown_by_default():
    fig = bin_plot(_numeric_feature_table(), figsize=(12, 7))
    assert _summary_text(fig) is not None


def test_metric_summary_can_be_hidden():
    fig = bin_plot(_numeric_feature_table(), figsize=(12, 7), show_metric_summary=False)
    assert _summary_text(fig) is None


# ---------------------------------------------------------------------------
# 需求3：小图时角标动态缩小，避免遮盖其他元素
# ---------------------------------------------------------------------------
def test_metric_summary_shrinks_on_small_figure():
    fs_default = _summary_text(bin_plot(_numeric_feature_table(), figsize=(12, 7))).get_fontsize()
    fs_small = _summary_text(bin_plot(_numeric_feature_table(), figsize=(6, 4))).get_fontsize()
    assert fs_small < fs_default


# ---------------------------------------------------------------------------
# 需求4：二维分箱图九宫格顺序、隐藏元素与紧凑间距
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bin_2d_figure():
    rng = np.random.RandomState(42)
    size = 400
    frame = pd.DataFrame(
        {
            "特征1": rng.normal(size=size),
            "特征2": rng.normal(size=size),
        }
    )
    probability = 1 / (1 + np.exp(-(frame["特征1"] + 0.6 * frame["特征2"])))
    target = pd.Series((rng.random_sample(size) < probability).astype(int), name="目标")
    binner = OptimalBinning2D(max_n_bins=4, max_n_bins_2d=6, min_bin_size=0.05)
    binner.fit(frame, target, features=["特征1", "特征2"])
    return bin_2d_plot(binner, figsize=(15, 13))


def test_bin_2d_plot_uses_requested_grid(bin_2d_figure):
    axes = bin_2d_figure.axes[:9]
    assert [ax.get_title() for ax in axes] == [
        "特征2 KS曲线",
        "分箱图",
        "风险拒绝比",
        "样本占比",
        "坏样本率",
        "分箱图",
        "LIFT",
        "坏账改善",
        "特征1 KS曲线",
    ]


def test_bin_2d_plot_hides_requested_axes_and_legends(bin_2d_figure):
    axes = bin_2d_figure.axes
    # 特征2 KS 仅显示纵坐标，特征1 KS 仅显示横坐标
    assert not axes[0].get_xticks().size
    assert axes[0].get_yticks().size
    assert axes[8].get_xticks().size
    assert not axes[8].get_yticks().size
    assert axes[0].get_legend() is axes[8].get_legend() is None

    reject_axis = axes[2]
    assert not reject_axis.get_xticks().size
    assert not reject_axis.get_yticks().size

    # 两个分箱图标题在 2D 图内使用主题蓝色，但不改变独立 bin_plot 的默认标题样式
    assert axes[1].title.get_color() == axes[5].title.get_color() == "#2639E9"
    standalone = bin_plot(_numeric_feature_table(), title="分箱图")
    assert standalone.axes[0].title.get_color() != "#2639E9"

    # 坏样本率副轴和特征1分箱标签不重复展示
    assert not any(label.get_visible() for label in axes[9].get_yticklabels())
    assert not any(label.get_visible() for label in axes[10].get_xticklabels())
    assert not any(label.get_visible() for label in axes[5].get_yticklabels())

    assert not any(label.get_visible() for label in axes[1].get_xticklabels())

    # 特征2分箱标签在底行热力图垂直展示
    feature_2_labels = [
        label
        for axis in (axes[6], axes[7])
        for label in axis.get_xticklabels()
        if label.get_visible()
    ]
    assert feature_2_labels
    assert all(label.get_rotation() == 90 for label in feature_2_labels)


def test_bin_2d_feature_1_name_does_not_overlap_ticks(bin_2d_figure):
    axis = bin_2d_figure.axes[3]
    assert axis.get_ylabel() == "特征1：特征1"
    bin_2d_figure.canvas.draw()
    renderer = bin_2d_figure.canvas.get_renderer()
    name_box = axis.yaxis.label.get_window_extent(renderer)
    tick_boxes = [label.get_window_extent(renderer) for label in axis.get_yticklabels() if label.get_visible()]
    assert tick_boxes
    assert name_box.x1 < min(box.x0 for box in tick_boxes)
    assert name_box.x0 >= 0


def test_bin_2d_plot_cells_are_tightly_spaced(bin_2d_figure):
    axes = bin_2d_figure.axes[:9]
    horizontal_gap = axes[1].get_position().x0 - axes[0].get_position().x1
    vertical_gap = axes[0].get_position().y0 - axes[3].get_position().y1
    assert horizontal_gap < 0.01
    assert vertical_gap < 0.05


def test_bin_2d_feature_2_labels_stay_inside_canvas(bin_2d_figure):
    axis = bin_2d_figure.axes[7]
    assert axis.get_xlabel() == "特征2：特征2"
    bin_2d_figure.canvas.draw()
    renderer = bin_2d_figure.canvas.get_renderer()
    boxes = [axis.xaxis.label.get_window_extent(renderer)]
    boxes.extend(label.get_window_extent(renderer) for label in axis.get_xticklabels() if label.get_visible())
    assert min(box.y0 for box in boxes) >= 0
