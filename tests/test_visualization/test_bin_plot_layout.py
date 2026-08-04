"""测试分箱图的刻度方向、指标角标开关与小图自适应布局。"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

matplotlib.use("Agg")

from hscredit.core.binning import OptimalBinning2D
import hscredit.core.viz.binning_plots as binning_plots
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
def _normalized_segment(segment):
    points = tuple(tuple(float(value) for value in point) for point in segment)
    return tuple(sorted(points))


def _boundary_collections(ax):
    return [
        artist
        for artist in ax.collections
        if (artist.get_gid() or "").startswith("bin-2d-boundary-")
    ]


def test_draw_2d_bin_boundaries_wraps_merged_region_without_internal_edges():
    """合并箱只绘制整体外缘，不能保留同箱格子间的彩色边。"""
    import matplotlib.pyplot as plt

    solution = np.array([[0, 0, 1], [0, 2, 1]])
    fig, ax = plt.subplots()
    try:
        artists = binning_plots._draw_2d_bin_boundaries(
            ax,
            solution,
            bin_colors={0: "#2639E9", 1: "#E0249A", 2: "#E43550"},
            expected_shape=(2, 3),
        )

        bin_zero = next(
            artist for artist in artists
            if artist.get_gid() == "bin-2d-boundary-0"
        )
        segments = {
            _normalized_segment(segment)
            for segment in bin_zero.get_segments()
        }

        assert len(segments) == 8
        assert _normalized_segment(((0.5, 0.5), (0.5, 1.5))) not in segments
        assert _normalized_segment(((-0.5, 0.5), (0.5, 0.5))) not in segments
        assert tuple(bin_zero.get_colors()[0]) == pytest.approx(to_rgba("#2639E9"))
    finally:
        plt.close(fig)


def test_draw_2d_bin_boundaries_rejects_invalid_solution_shape():
    """空映射和与热力图不一致的映射必须被明确拒绝。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="二维分箱映射必须是非空二维矩阵"):
            binning_plots._draw_2d_bin_boundaries(ax, np.array([]))
        with pytest.raises(ValueError, match="二维分箱映射形状"):
            binning_plots._draw_2d_bin_boundaries(
                ax,
                np.zeros((2, 3)),
                expected_shape=(3, 2),
            )
    finally:
        plt.close(fig)


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


def test_bin_2d_plot_draws_same_merged_bin_boundaries_on_five_metric_axes(bin_2d_figure):
    """五个交叉指标子图必须复用同一组最终箱轮廓和颜色。"""
    axes = bin_2d_figure.axes[:9]
    metric_axis_indexes = (2, 3, 4, 6, 7)
    reference = _boundary_collections(axes[metric_axis_indexes[0]])
    reference_gids = [artist.get_gid() for artist in reference]
    reference_colors = [artist.get_colors()[0] for artist in reference]

    assert reference
    for axis_index in metric_axis_indexes[1:]:
        actual = _boundary_collections(axes[axis_index])
        assert [artist.get_gid() for artist in actual] == reference_gids
        assert np.allclose(
            [artist.get_colors()[0] for artist in actual],
            reference_colors,
        )
    for axis_index in (0, 1, 5, 8):
        assert _boundary_collections(axes[axis_index]) == []


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


def test_bin_2d_plot_axis_bins_follow_numeric_interval_order():
    """组合图横纵坐标应按分箱索引对应的数值区间升序排列."""
    rng = np.random.RandomState(24)
    size = 500
    frame = pd.DataFrame({
        "x": rng.uniform(-20, 20, size),
        "y": rng.uniform(0, 100, size),
    })
    target = pd.Series(rng.randint(0, 2, size), name="目标")
    binner = OptimalBinning2D(
        user_splits_x=[-10, 0, 10],
        user_splits_y=[20, 40, 60, 80],
        max_n_bins_2d=8,
    ).fit(frame, target, features=["x", "y"])

    # 模拟一维分箱表被报表层重排，绘图仍必须按“分箱”列而不是表行序取标签。
    binner.binner_x_.bin_tables_["x"] = (
        binner.binner_x_.bin_tables_["x"].sample(frac=1, random_state=3).reset_index(drop=True)
    )
    binner.binner_y_.bin_tables_["y"] = (
        binner.binner_y_.bin_tables_["y"].sample(frac=1, random_state=4).reset_index(drop=True)
    )
    fig = bin_2d_plot(binner, figsize=(15, 13))
    axes = fig.axes[:9]

    # Matplotlib 返回 y 刻度时按坐标值从底到顶；组合图视觉顺序需反转为从上到下。
    x_labels = [
        label.get_text()
        for label in reversed(axes[3].get_yticklabels())
        if label.get_visible()
    ]
    y_labels = [label.get_text() for label in axes[7].get_xticklabels() if label.get_visible()]
    assert x_labels[0].startswith("[-inf,")
    assert x_labels[-1].endswith("+inf)")
    assert y_labels[0].startswith("[-inf,")
    assert y_labels[-1].endswith("+inf)")


def test_bin_2d_plot_includes_missing_bins_in_heatmaps():
    """缺失箱应作为最后一行/列显示，并纳入二维热力图统计."""
    rng = np.random.RandomState(31)
    size = 600
    frame = pd.DataFrame({
        "x": rng.uniform(-20, 20, size),
        "y": rng.uniform(0, 100, size),
    })
    frame.loc[:39, "x"] = np.nan
    frame.loc[40:79, "y"] = np.nan
    frame.loc[80:99, ["x", "y"]] = np.nan
    target = pd.Series(rng.randint(0, 2, size), name="目标")
    binner = OptimalBinning2D(
        max_n_bins=4,
        max_n_bins_2d=6,
        missing_separate=True,
    ).fit(frame, target, features=["x", "y"])

    fig = bin_2d_plot(binner, figsize=(15, 13))
    axes = fig.axes[:9]
    expected_shape = (binner.n_bins_x_ + 1, binner.n_bins_y_ + 1)

    for axis_index in (2, 3, 4, 6, 7):
        assert axes[axis_index].images[0].get_array().shape == expected_shape

    x_labels = [
        label.get_text()
        for label in reversed(axes[3].get_yticklabels())
        if label.get_visible()
    ]
    y_labels = [label.get_text() for label in axes[7].get_xticklabels() if label.get_visible()]
    assert x_labels[-1] == "缺失值"
    assert y_labels[-1] == "缺失值"

    sample_share = np.asarray(axes[3].images[0].get_array(), dtype=float)
    assert np.isclose(np.nansum(sample_share), 1.0)

    expected_boundary_gids = {
        f"bin-2d-boundary-{bin_id}"
        for bin_id in np.unique(binner.solution_)
    }
    for axis_index in (2, 3, 4, 6, 7):
        assert {
            artist.get_gid()
            for artist in _boundary_collections(axes[axis_index])
        } == expected_boundary_gids
