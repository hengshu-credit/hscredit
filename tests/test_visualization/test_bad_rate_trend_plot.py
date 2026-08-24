"""坏样本率趋势图公共契约与视觉样式回归测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from hscredit.core.viz import bad_rate_trend_plot


def _trend_data():
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(
                [
                    "2025-11-01",
                    "2025-11-10",
                    "2025-11-20",
                    "2025-12-01",
                    "2025-12-10",
                    "2025-12-20",
                    "2026-01-01",
                    "2026-01-10",
                    "2026-01-20",
                    "2026-02-01",
                    "2026-02-10",
                    "2026-02-20",
                ]
            ),
            "目标": [0] * 12,
            "MOB1": [0, 10, 40, 0, 0, 20, 0, 50, 60, 5, 8, 31],
        }
    )


def test_overdue_dpds_take_priority_and_share_one_axis_per_overdue():
    """错误采用 target 或同一 overdue 的 dpds 被拆成多轴时，本测试必须失败。"""
    fig = bad_rate_trend_plot(
        _trend_data(),
        date_col="日期",
        target="目标",
        overdue="MOB1",
        dpds=[7, 30],
        show_sample_count=False,
    )

    rate_axes = [ax for ax in fig.axes if ax.get_ylabel().endswith("坏样本率")]
    assert [ax.get_ylabel() for ax in rate_axes] == ["MOB1 坏样本率"]
    assert [line.get_label() for line in rate_axes[0].lines] == ["MOB1_7+", "MOB1_30+"]
    np.testing.assert_allclose(rate_axes[0].lines[0].get_ydata(), [2 / 3, 1 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(rate_axes[0].lines[1].get_ydata(), [1 / 3, 0, 2 / 3, 1 / 3])
    plt.close(fig)


def test_different_overdue_fields_use_different_rate_axes():
    """不同 overdue 字段被错误合并到同一坐标轴时，本测试必须失败。"""
    data = _trend_data().copy()
    data["MOB3"] = [0, 20, 60, 0, 10, 40, 0, 35, 80, 0, 8, 50]

    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue=["MOB1", "MOB3"],
        dpds=[7, 30],
        show_sample_count=False,
    )

    rate_axes = [ax for ax in fig.axes if ax.get_ylabel().endswith("坏样本率")]
    assert [ax.get_ylabel() for ax in rate_axes] == ["MOB1 坏样本率", "MOB3 坏样本率"]
    assert [[line.get_label() for line in ax.lines] for ax in rate_axes] == [
        ["MOB1_7+", "MOB1_30+"],
        ["MOB3_7+", "MOB3_30+"],
    ]
    plt.close(fig)


def test_date_ticks_match_real_groups_and_align_with_sample_bars():
    """让日期定位器插入额外刻度或使上下图错位时，本测试必须失败。"""
    fig = bad_rate_trend_plot(
        _trend_data(),
        date_col="日期",
        target="目标",
        freq="M",
        show_sample_count=True,
    )

    rate_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "坏样本率")
    count_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "样本数")
    assert count_ax.get_xticks().tolist() == [0, 1, 2, 3]
    assert [label.get_text() for label in count_ax.get_xticklabels()] == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]
    np.testing.assert_allclose(rate_ax.lines[0].get_xdata(), [0, 1, 2, 3])
    np.testing.assert_allclose(
        [patch.get_x() + patch.get_width() / 2 for patch in count_ax.patches],
        [0, 1, 2, 3],
    )
    plt.close(fig)


def test_explicit_colors_apply_to_hscredit_axes_bars_and_grid():
    """样本数面板遗留黑轴、实线网格或忽略显式颜色时，本测试必须失败。"""
    fig = bad_rate_trend_plot(
        _trend_data(),
        date_col="日期",
        target="目标",
        colors=["#112233", "#445566"],
        show_sample_count=True,
    )

    rate_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "坏样本率")
    count_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "样本数")
    assert to_hex(rate_ax.lines[0].get_color()).lower() == "#112233"
    assert {to_hex(patch.get_facecolor()).lower() for patch in count_ax.patches} == {"#2639e9"}
    assert rate_ax.get_xlabel() == ""
    assert count_ax.get_xlabel() == "日期"

    for styled_ax, expected_color in ((rate_ax, "#112233"), (count_ax, "#2639e9")):
        assert to_hex(styled_ax.spines["left"].get_edgecolor()).lower() == expected_color
        assert to_hex(styled_ax.spines["bottom"].get_edgecolor()).lower() == expected_color
        assert not styled_ax.spines["top"].get_visible()
        assert not styled_ax.spines["right"].get_visible()
        visible_gridlines = [line for line in styled_ax.get_ygridlines() if line.get_visible()]
        assert visible_gridlines
        assert {line.get_linestyle() for line in visible_gridlines} == {"--"}
        assert {line.get_alpha() for line in visible_gridlines} == {0.3}
    plt.close(fig)


def test_multilabel_legend_sits_between_title_and_content_without_axis_clipping():
    """图例留在内容内或外移坏率轴被画布裁切时，本测试必须失败。"""
    data = _trend_data().copy()
    data["MOB3"] = [0, 20, 60, 0, 10, 40, 0, 35, 80, 0, 8, 50]
    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue=["MOB1", "MOB3"],
        dpds=[7, 30],
        show_sample_count=True,
        figsize=(10, 6),
        title="多标签坏样本率趋势",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "多标签坏样本率趋势"
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert not legend.get_frame_on()
    assert [text.get_text() for text in legend.get_texts()] == [
        "MOB1_7+", "MOB1_30+", "MOB3_7+", "MOB3_30+"
    ]

    rate_axes = [ax for ax in fig.axes if ax.get_ylabel().endswith("坏样本率")]
    count_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "样本数")
    title_box = fig._suptitle.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    content_top = max(ax.get_window_extent(renderer).y1 for ax in rate_axes)
    assert title_box.y0 > legend_box.y1 > legend_box.y0 > content_top

    figure_box = fig.bbox
    for rate_ax in rate_axes:
        visible_text = [rate_ax.yaxis.label, *rate_ax.get_yticklabels()]
        for artist in visible_text:
            if artist.get_visible() and artist.get_text():
                artist_box = artist.get_window_extent(renderer)
                assert artist_box.x0 >= figure_box.x0
                assert artist_box.x1 <= figure_box.x1

    count_box = count_ax.get_window_extent(renderer)
    for rate_ax in rate_axes:
        rate_box = rate_ax.get_window_extent(renderer)
        np.testing.assert_allclose([rate_box.x0, rate_box.x1], [count_box.x0, count_box.x1], atol=0.5)

    line = rate_axes[0].lines[0]
    line_x = rate_axes[0].transData.transform(
        np.column_stack([line.get_xdata(), line.get_ydata()])
    )[:, 0]
    bar_centers = np.array([patch.get_x() + patch.get_width() / 2 for patch in count_ax.patches])
    bar_x = count_ax.transData.transform(np.column_stack([bar_centers, np.zeros_like(bar_centers)]))[:, 0]
    np.testing.assert_allclose(line_x, bar_x, atol=0.5)
    plt.close(fig)


def test_target_col_keyword_is_rejected_with_target_migration_message():
    """旧名称被 **kwargs 静默吞掉时，本测试必须失败。"""
    with pytest.raises(TypeError, match="target_col.*target"):
        bad_rate_trend_plot(
            _trend_data(),
            date_col="日期",
            target_col="目标",
            show_sample_count=False,
        )


def test_default_multilabel_lines_on_same_overdue_axis_use_distinct_hscredit_colors():
    """同一 overdue 轴内多个 dpds 退化成同色曲线时，本测试必须失败。"""
    fig = bad_rate_trend_plot(
        _trend_data(),
        date_col="日期",
        overdue="MOB1",
        dpds=[0, 7, 30],
        show_sample_count=False,
    )

    rate_axes = [ax for ax in fig.axes if ax.get_ylabel().endswith("坏样本率")]
    assert len(rate_axes) == 1
    line_colors = [to_hex(line.get_color()).lower() for line in rate_axes[0].lines]
    assert len(set(line_colors)) == 3
    assert to_hex(rate_axes[0].spines["left"].get_edgecolor()).lower() == line_colors[0]
    assert to_hex(rate_axes[0].yaxis.label.get_color()).lower() == line_colors[0]
    plt.close(fig)


def test_multilabel_axes_keep_distinct_series_visually_separated():
    """独立自动量程把成比例的多条曲线映射到相同像素轨迹时，本测试必须失败。"""
    dates = pd.to_datetime(np.repeat(["2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"], 6))
    data = pd.DataFrame(
        {
            "日期": dates,
            "MOB1": [
                0, 2, 8, 16, 35, 50,
                0, 0, 5, 10, 20, 40,
                1, 9, 18, 32, 45, 60,
                0, 3, 12, 28, 38, 70,
            ],
        }
    )
    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue="MOB1",
        dpds=[0, 7, 15, 30],
        show_sample_count=False,
    )
    fig.canvas.draw()

    rate_ax = fig.axes[0]
    display_trajectories = []
    for line in rate_ax.lines:
        points = np.column_stack([line.get_xdata(), line.get_ydata()])
        display_y = rate_ax.transData.transform(points)[:, 1]
        display_trajectories.append(tuple(np.round(display_y, 3)))
    assert len(set(display_trajectories)) == 4
    plt.close(fig)


def test_single_target_dimension_series_keep_distinct_colors():
    """单标签维度曲线退化成同色，仅靠线型区分时，本测试必须失败。"""
    data = _trend_data().copy()
    data["目标"] = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    data["渠道"] = ["线上", "线下", "线上"] * 4

    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        target="目标",
        dimension_col="渠道",
        show_sample_count=False,
    )

    rate_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "坏样本率")
    assert [line.get_label() for line in rate_ax.lines] == ["线上", "线下"]
    assert len({to_hex(line.get_color()).lower() for line in rate_ax.lines}) == 2
    plt.close(fig)


def test_overdue_missing_values_follow_library_binary_target_semantics():
    """逾期字段缺失被排除、导致分母与全库 overdue>dpd 口径不一致时，本测试必须失败。"""
    data = pd.DataFrame(
        {
            "日期": pd.to_datetime(["2026-01-01", "2026-01-15"]),
            "MOB1": [np.nan, 10],
        }
    )

    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue="MOB1",
        dpds=7,
        show_sample_count=False,
    )

    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0.5])
    plt.close(fig)


def test_same_overdue_multidpd_dimension_series_share_axis_and_keep_distinct_colors():
    """同一 overdue 的 dpds×维度被拆轴或退化成同色时，本测试必须失败。"""
    data = _trend_data().copy()
    data["渠道"] = ["线上", "线下", "线上"] * 4

    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue="MOB1",
        dpds=[7, 30],
        dimension_col="渠道",
        show_sample_count=False,
    )

    rate_axes = [ax for ax in fig.axes if ax.get_ylabel().endswith("坏样本率")]
    assert [ax.get_ylabel() for ax in rate_axes] == ["MOB1 坏样本率"]
    labels = [line.get_label() for ax in rate_axes for line in ax.lines]
    colors = [to_hex(line.get_color()).lower() for ax in rate_axes for line in ax.lines]
    assert labels == ["MOB1_7+ · 线上", "MOB1_7+ · 线下", "MOB1_30+ · 线上", "MOB1_30+ · 线下"]
    assert len(set(colors)) == 4
    plt.close(fig)


def test_supplied_axis_keeps_rate_and_count_panels_equal_width_without_overlap():
    """外部 ax 的定位器在 draw 时让上下面板重叠或宽度漂移时，本测试必须失败。"""
    data = _trend_data().copy()
    data["MOB3"] = [0, 20, 60, 0, 10, 40, 0, 35, 80, 0, 8, 50]
    fig, supplied_ax = plt.subplots(figsize=(10, 6))

    returned = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue=["MOB1", "MOB3"],
        dpds=[7, 30],
        ax=supplied_ax,
        show_sample_count=True,
    )
    returned.canvas.draw()
    renderer = returned.canvas.get_renderer()

    rate_box = supplied_ax.get_window_extent(renderer)
    count_ax = next(ax for ax in returned.axes if ax.get_ylabel() == "样本数")
    count_box = count_ax.get_window_extent(renderer)
    np.testing.assert_allclose([rate_box.x0, rate_box.x1], [count_box.x0, count_box.x1], atol=0.5)
    assert count_box.y1 < rate_box.y0
    plt.close(fig)


def test_short_custom_palette_is_extended_with_unique_hscredit_colors():
    """自定义颜色耗尽后循环复用、导致不同标签同色时，本测试必须失败。"""
    data = _trend_data().copy()
    data["MOB3"] = [0, 20, 60, 0, 10, 40, 0, 35, 80, 0, 8, 50]

    fig = bad_rate_trend_plot(
        data,
        date_col="日期",
        overdue=["MOB1", "MOB3"],
        dpds=[7, 30],
        colors=["#112233", "#445566"],
        show_sample_count=False,
    )

    line_colors = [
        to_hex(line.get_color()).lower()
        for ax in fig.axes
        for line in ax.lines
    ]
    assert line_colors[:2] == ["#112233", "#445566"]
    assert len(set(line_colors)) == 4
    plt.close(fig)
