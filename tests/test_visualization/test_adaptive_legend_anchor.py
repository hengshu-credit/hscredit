"""测试顶部居中图例随画布高度自适应，并尊重用户显式锚点。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hscredit.core.viz.binning_plots import (
    bin_plot,
    bin_trend_plot,
    distribution_plot,
    hist_plot,
    ks_plot,
    psi_plot,
)
from hscredit.core.viz.risk_plots import score_dist_plot


def _feature_table():
    return pd.DataFrame(
        {
            "分箱": ["[0, 1)", "[1, 2)"],
            "样本总数": [100, 120],
            "好样本数": [90, 96],
            "坏样本数": [10, 24],
            "坏样本率": [0.10, 0.20],
            "样本占比": [100 / 220, 120 / 220],
        }
    )


def _psi_table(counts):
    total = sum(counts)
    return pd.DataFrame(
        {
            "分箱": ["[0, 1)", "[1, 2)"],
            "样本总数": counts,
            "样本占比": [count / total for count in counts],
            "坏样本率": [0.10, 0.20],
        }
    )


def _score_data():
    return pd.Series(np.linspace(0.05, 0.95, 20), name="评分"), pd.Series([0, 1] * 10, name="目标")


def _distribution_data():
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(["2025-11-01", "2025-11-15", "2025-12-01", "2025-12-15"]),
            "目标": [0, 1, 0, 1],
        }
    )


def _trend_data():
    return pd.DataFrame(
        {
            "评分": np.tile(np.arange(10), 6),
            "目标": np.tile([0, 1], 30),
        }
    )


def _figure_legend_anchor(fig):
    assert fig.legends
    return fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted()).y0


def _axes_legend_anchor(fig):
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return legend.get_bbox_to_anchor().transformed(fig.axes[0].transAxes.inverted()).y0


def _top_layout_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title = fig._suptitle
    if title is None:
        title = next(ax.title for ax in fig.axes if ax.title.get_text().strip())
    legend = fig.legends[0] if fig.legends else fig.axes[0].get_legend()
    axes_tops = [ax.get_window_extent(renderer).y1 for ax in fig.axes]
    axes_tops.extend(
        bbox.y1
        for ax in fig.axes
        if (bbox := ax.xaxis.get_tightbbox(renderer)) is not None
    )
    axes_top = max(axes_tops)
    return title.get_window_extent(renderer), legend.get_window_extent(renderer), axes_top


def _assert_title_legend_axes_are_equally_spaced(fig, tolerance_pixels=2.0):
    title_bbox, legend_bbox, axes_top = _top_layout_boxes(fig)
    title_gap = title_bbox.y0 - legend_bbox.y1
    axes_gap = legend_bbox.y0 - axes_top

    assert title_gap > 0
    assert axes_gap > 0
    assert title_gap == pytest.approx(axes_gap, abs=tolerance_pixels)


def _bin_figure(height, anchor=None):
    kwargs = {} if anchor is None else {"anchor": anchor}
    return bin_plot(_feature_table(), figsize=(12, height), show_metric_summary=False, **kwargs)


def _ks_figure(height, anchor=None):
    score, target = _score_data()
    kwargs = {} if anchor is None else {"anchor": anchor}
    return ks_plot(score, target, figsize=(16, height), **kwargs)


def _hist_figure(height, anchor=None):
    score, target = _score_data()
    kwargs = {} if anchor is None else {"anchor": anchor}
    return hist_plot(score, target, figsize=(15, height), kde=False, **kwargs)


def _psi_figure(height, anchor=None):
    kwargs = {} if anchor is None else {"anchor": anchor}
    return psi_plot(_psi_table([100, 200]), _psi_table([120, 180]), figsize=(15, height), **kwargs)


def _distribution_figure(height, anchor=None):
    kwargs = {} if anchor is None else {"anchor": anchor}
    return distribution_plot(
        _distribution_data(), date="日期", target="目标", freq="M", figsize=(10, height), **kwargs
    )


def _score_dist_figure(height, anchor=None):
    score, target = _score_data()
    kwargs = {} if anchor is None else {"anchor": anchor}
    return score_dist_plot(score, target, figsize=(12, height), kde=False, show_stats=False, **kwargs)


def _trend_figure(height, anchor=None):
    kwargs = {} if anchor is None else {"anchor": anchor}
    return bin_trend_plot(
        _trend_data(),
        feature="评分",
        target="目标",
        rules={"评分": [2, 4, 6, 8]},
        figsize=(10.5, height),
        show_stats=False,
        **kwargs,
    )


def test_hist_plot_places_legend_between_title_and_axes_with_equal_gaps():
    fig = _hist_figure(6.0)

    _assert_title_legend_axes_are_equally_spaced(fig)

    plt.close(fig)


def test_bin_plot_legend_stays_clear_of_top_rate_axis():
    fig = bin_plot(_feature_table(), figsize=(10, 5))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = fig.legends[0].get_window_extent(renderer)
    rate_axis_bbox = fig.axes[1].xaxis.get_tightbbox(renderer)
    minimum_gap_pixels = 6.0 * fig.dpi / 72.0

    assert rate_axis_bbox is not None
    assert legend_bbox.y0 >= rate_axis_bbox.y1 + minimum_gap_pixels - 1.0

    plt.close(fig)


def test_ks_plot_places_legend_between_title_and_axes_with_equal_gaps():
    fig = _ks_figure(5.0)

    _assert_title_legend_axes_are_equally_spaced(fig)

    plt.close(fig)


@pytest.mark.parametrize(
    ("factory", "height"),
    [
        (_bin_figure, 3.5),
        (_bin_figure, 7.0),
        (_bin_figure, 12.0),
        (_ks_figure, 4.0),
        (_ks_figure, 8.0),
        (_ks_figure, 12.0),
        (_hist_figure, 5.0),
        (_hist_figure, 10.0),
        (_hist_figure, 15.0),
        (_psi_figure, 4.0),
        (_psi_figure, 8.0),
        (_psi_figure, 12.0),
        (_distribution_figure, 3.5),
        (_distribution_figure, 6.0),
        (_distribution_figure, 10.0),
        (_score_dist_figure, 3.5),
        (_score_dist_figure, 6.0),
        (_score_dist_figure, 10.0),
        (_trend_figure, 3.0),
        (_trend_figure, 5.4),
        (_trend_figure, 9.0),
    ],
)
def test_default_legend_is_equally_spaced_between_title_and_axes(factory, height):
    fig = factory(height)

    _assert_title_legend_axes_are_equally_spaced(fig)

    plt.close(fig)


@pytest.mark.parametrize(
    ("factory", "anchor_reader", "default_height", "custom_anchor"),
    [
        (_bin_figure, _figure_legend_anchor, 7.0, 0.81),
        (_ks_figure, _figure_legend_anchor, 8.0, 0.82),
        (_hist_figure, _axes_legend_anchor, 10.0, 1.07),
        (_psi_figure, _figure_legend_anchor, 8.0, 0.83),
        (_distribution_figure, _figure_legend_anchor, 6.0, 0.84),
        (_score_dist_figure, _axes_legend_anchor, 6.0, 1.08),
    ],
)
def test_explicit_legend_anchor_always_wins(factory, anchor_reader, default_height, custom_anchor):
    fig = factory(default_height / 2.0, anchor=custom_anchor)

    assert anchor_reader(fig) == pytest.approx(custom_anchor)

    plt.close(fig)


@pytest.mark.parametrize(
    ("factory", "height"),
    [
        (_hist_figure, 5.0),
        (_hist_figure, 15.0),
        (_score_dist_figure, 3.5),
    ],
)
def test_automatic_axes_legend_stays_inside_canvas_and_clear_of_title(factory, height):
    fig = factory(height)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    legend_bbox = ax.get_legend().get_window_extent(renderer)
    title_bbox = ax.title.get_window_extent(renderer)

    assert not legend_bbox.overlaps(title_bbox)
    assert legend_bbox.y1 <= fig.bbox.y1

    plt.close(fig)


def test_bin_trend_plot_respects_explicit_anchor():
    fig = bin_trend_plot(
        _trend_data(),
        feature="评分",
        target="目标",
        rules={"评分": [2, 4, 6, 8]},
        figsize=(10.5, 2.7),
        show_stats=False,
        anchor=0.79,
    )

    assert _figure_legend_anchor(fig) == pytest.approx(0.79)

    plt.close(fig)
