"""测试所有嵌入式 ``bin_plot`` 调用共享的顶部标题与指标摘要布局。"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hscredit.core.viz.binning_plots import (
    batch_bin_trend_plot,
    bin_overdues_plot,
    bin_plot,
    bin_trend_plot,
)
from hscredit.core.viz.risk_plots import score_bin_plot
from hscredit.report.feature_analyzer import feature_efficiency_analysis


def _metric_table():
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


def _metric_summary(fig):
    return next(text for axis in fig.axes for text in axis.texts if "IV" in text.get_text())


def _axes_decoration_top(fig, renderer):
    tops = []
    for axis in fig.axes:
        tops.append(axis.get_window_extent(renderer).y1)
        for component in (axis.xaxis, axis.yaxis):
            bbox = component.get_tightbbox(renderer)
            if bbox is not None:
                tops.append(bbox.y1)
    return max(tops)


def _minimum_gap_pixels(fig):
    return 6.0 * fig.dpi / 72.0


def _metric_summary_artists(fig):
    summaries = []
    for axis in fig.axes:
        summaries.extend(
            artist
            for artist in [*axis.texts, *axis.artists]
            if artist.get_visible() and artist.get_gid() == "bin-metric-summary"
        )
    return summaries


def _same_panel(axis, owner):
    return np.allclose(axis.get_position().bounds, owner.get_position().bounds, atol=1e-8)


def _assert_embedded_bin_headers_are_clear(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = _minimum_gap_pixels(fig)
    summaries = _metric_summary_artists(fig)
    assert summaries

    for summary in summaries:
        owner = summary.axes
        panel_axes = [axis for axis in fig.axes if _same_panel(axis, owner)]
        decoration_tops = [axis.get_window_extent(renderer).y1 for axis in panel_axes]
        for axis in panel_axes:
            for component in (axis.xaxis, axis.yaxis):
                bbox = component.get_tightbbox(renderer)
                if bbox is not None:
                    decoration_tops.append(bbox.y1)

        summary_bbox = summary.get_window_extent(renderer)
        title_bbox = owner.title.get_window_extent(renderer)
        assert summary_bbox.y0 >= max(decoration_tops) + gap_pixels - 1.0
        assert title_bbox.y0 >= summary_bbox.y1 + gap_pixels - 1.0

    if fig.legends:
        legend_bbox = fig.legends[0].get_window_extent(renderer)
        panel_title_top = max(summary.axes.title.get_window_extent(renderer).y1 for summary in summaries)
        assert legend_bbox.y0 >= panel_title_top + gap_pixels - 1.0
        if fig._suptitle is not None:
            title_bbox = fig._suptitle.get_window_extent(renderer)
            assert title_bbox.y0 >= legend_bbox.y1 + gap_pixels - 1.0
    elif fig._suptitle is not None:
        figure_title_bbox = fig._suptitle.get_window_extent(renderer)
        panel_title_top = max(summary.axes.title.get_window_extent(renderer).y1 for summary in summaries)
        assert figure_title_bbox.y0 >= panel_title_top + gap_pixels - 1.0
        assert figure_title_bbox.y1 <= fig.bbox.y1 + 1.0


def _assert_full_width_metric_summaries(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    summaries = [artist for axis in fig.axes for artist in axis.artists if artist.get_gid() == "bin-metric-summary"]
    assert summaries

    for summary in summaries:
        axes_bbox = summary.axes.get_window_extent(renderer)
        summary_bbox = summary.get_window_extent(renderer)
        metric_text = summary.metric_text
        text_bbox = metric_text.get_window_extent(renderer)

        assert "\n" not in metric_text.get_text()
        assert summary_bbox.x0 == pytest.approx(axes_bbox.x0, abs=1.0)
        assert summary_bbox.x1 == pytest.approx(axes_bbox.x1, abs=1.0)
        assert (text_bbox.x0 + text_bbox.x1) / 2.0 == pytest.approx(
            (axes_bbox.x0 + axes_bbox.x1) / 2.0,
            abs=1.0,
        )
        assert text_bbox.x0 >= summary_bbox.x0 + 1.0
        assert text_bbox.x1 <= summary_bbox.x1 - 1.0
        assert metric_text.get_fontsize() == pytest.approx(summary.axes.xaxis.label.get_fontsize())


def _assert_horizontal_panel_rows_are_clear(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_pixels = _minimum_gap_pixels(fig)
    summary_by_axis = {summary.axes: summary for summary in _metric_summary_artists(fig)}
    panel_axes = sorted(summary_by_axis, key=lambda axis: axis.get_position().y0, reverse=True)
    assert len(panel_axes) > 1

    for upper_axis, lower_axis in zip(panel_axes, panel_axes[1:]):
        upper_bottoms = [upper_axis.get_window_extent(renderer).y0]
        xaxis_bbox = upper_axis.xaxis.get_tightbbox(renderer)
        if xaxis_bbox is not None:
            upper_bottoms.append(xaxis_bbox.y0)

        lower_header_tops = [lower_axis.title.get_window_extent(renderer).y1]
        lower_header_tops.append(summary_by_axis[lower_axis].get_window_extent(renderer).y1)
        assert min(upper_bottoms) >= max(lower_header_tops) + gap_pixels - 1.0


def _assert_unified_bin_legend(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "好样本",
        "坏样本",
        "坏样本率",
        "整体坏样本率",
    ]
    legend_bbox = legend.get_window_extent(renderer)
    assert (legend_bbox.x0 + legend_bbox.x1) / 2.0 == pytest.approx(
        (fig.bbox.x0 + fig.bbox.x1) / 2.0,
        abs=1.0,
    )
    assert all(axis.get_legend() is None for axis in fig.axes)


def _assert_axis_decorations_are_inside_canvas(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        if not axis.get_visible():
            continue
        for component in (axis.xaxis, axis.yaxis):
            bbox = component.get_tightbbox(renderer)
            if bbox is None:
                continue
            assert bbox.x0 >= fig.bbox.x0 - 1.0
            assert bbox.x1 <= fig.bbox.x1 + 1.0


def _trend_data():
    return pd.DataFrame(
        {
            "评分": np.tile(np.arange(10), 8),
            "目标": np.tile([0, 0, 0, 1, 0, 1, 1, 1, 0, 1], 8),
            "渠道": np.repeat(["A", "B"], 40),
        }
    )


def _overdue_bin_table():
    rows = {
        ("分箱详情", "分箱标签"): ["[0, 2)", "[2, 4)", "[4, inf)"],
        ("分箱详情", "样本总数"): [30, 30, 20],
        ("分箱详情", "样本占比"): [0.375, 0.375, 0.25],
        ("分箱详情", "指标名称"): ["评分"] * 3,
    }
    for target, bad_counts in (("MOB1 1+", [3, 9, 12]), ("MOB3 7+", [2, 8, 14])):
        good_counts = [30 - bad for bad in bad_counts[:2]] + [20 - bad_counts[2]]
        totals = [30, 30, 20]
        bad_rates = [bad / total for bad, total in zip(bad_counts, totals)]
        rows[(target, "好样本数")] = good_counts
        rows[(target, "坏样本数")] = bad_counts
        rows[(target, "坏样本率")] = bad_rates
        rows[(target, "累计好样本占比")] = [0.4, 0.75, 1.0]
        rows[(target, "累计坏样本占比")] = [0.15, 0.5, 1.0]
        rows[(target, "IV值")] = [0.03, 0.05, 0.07]
        rows[(target, "Lift")] = [0.6, 1.0, 1.8]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_embedded_bin_plot_header_artists_do_not_overlap(orientation):
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        bin_plot(_metric_table(), ax=ax, title="分组标题", orientation=orientation)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        summary = _metric_summary(fig)
        summary_bbox = summary.get_window_extent(renderer)
        title_bbox = ax.title.get_window_extent(renderer)
        gap_pixels = _minimum_gap_pixels(fig)

        assert summary_bbox.y0 >= _axes_decoration_top(fig, renderer) + gap_pixels - 1.0
        assert title_bbox.y0 >= summary_bbox.y1 + gap_pixels - 1.0
        assert summary.get_horizontalalignment() == "left"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_bin_trend_plot_reserves_space_for_every_metric_summary(orientation):
    fig = bin_trend_plot(
        _trend_data(),
        feature="评分",
        target="目标",
        dimension_cols="渠道",
        rules={"评分": [2, 4, 6, 8]},
        orientation=orientation,
        show_stats=True,
    )
    try:
        _assert_embedded_bin_headers_are_clear(fig)
        _assert_full_width_metric_summaries(fig)
        _assert_axis_decorations_are_inside_canvas(fig)
        if orientation == "horizontal":
            _assert_horizontal_panel_rows_are_clear(fig)
    finally:
        plt.close(fig)


def test_batch_bin_trend_plot_preserves_embedded_header_layout():
    figures = batch_bin_trend_plot(
        _trend_data(),
        features=["评分"],
        target="目标",
        dimension_cols="渠道",
        rules={"评分": [2, 4, 6, 8]},
        show_stats=True,
        max_features=1,
    )
    try:
        assert set(figures) == {"评分"}
        _assert_embedded_bin_headers_are_clear(figures["评分"])
        _assert_full_width_metric_summaries(figures["评分"])
        _assert_axis_decorations_are_inside_canvas(figures["评分"])
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_batch_bin_trend_plot_horizontal_rows_do_not_overlap():
    figures = batch_bin_trend_plot(
        _trend_data(),
        features=["评分"],
        target="目标",
        dimension_cols="渠道",
        rules={"评分": [2, 4, 6, 8]},
        orientation="horizontal",
        show_stats=True,
        max_features=1,
    )
    try:
        _assert_horizontal_panel_rows_are_clear(figures["评分"])
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_bin_overdues_plot_raw_mode_preserves_embedded_header_layout():
    data = _trend_data().rename(columns={"评分": "score"})
    data["MOB1"] = np.where(data["目标"].eq(1), 4, 0)
    data["MOB3"] = np.where(data["目标"].eq(1), 10, 0)
    fig = bin_overdues_plot(
        data,
        feature="score",
        overdue=["MOB1", "MOB3"],
        dpds=[1, 7],
        rules={"score": [2, 4, 6, 8]},
        show_stats=True,
    )
    try:
        _assert_embedded_bin_headers_are_clear(fig)
        _assert_full_width_metric_summaries(fig)
        _assert_axis_decorations_are_inside_canvas(fig)
        _assert_unified_bin_legend(fig)
        assert not any("BadRate:" in text.get_text() for axis in fig.axes for text in axis.texts)
    finally:
        plt.close(fig)


def test_bin_overdues_plot_table_mode_preserves_embedded_header_layout():
    table = _overdue_bin_table()
    fig = bin_overdues_plot(table, bin_table=table, show_stats=True)
    try:
        _assert_embedded_bin_headers_are_clear(fig)
        _assert_full_width_metric_summaries(fig)
        _assert_axis_decorations_are_inside_canvas(fig)
        _assert_unified_bin_legend(fig)
        assert not any("BadRate:" in text.get_text() for axis in fig.axes for text in axis.texts)
    finally:
        plt.close(fig)


def test_score_bin_plot_preserves_embedded_header_layout():
    data = _trend_data().rename(columns={"评分": "score", "目标": "target"})
    fig = score_bin_plot(data, "score", "target", n_bins=5, show_table=True)
    try:
        _assert_embedded_bin_headers_are_clear(fig)
        _assert_axis_decorations_are_inside_canvas(fig)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_feature_efficiency_comparison_preserves_embedded_header_layout(orientation):
    data = _trend_data().rename(columns={"评分": "score", "目标": "target"})
    result = feature_efficiency_analysis(
        data,
        feature="score",
        manual_rules=[2, 4, 6, 8],
        target="target",
        auto_method="quantile",
        comparison_orientation=orientation,
        n_jobs=1,
    )
    fig = result["comparison_figure"]
    try:
        _assert_embedded_bin_headers_are_clear(fig)
        _assert_axis_decorations_are_inside_canvas(fig)
    finally:
        plt.close(fig)
