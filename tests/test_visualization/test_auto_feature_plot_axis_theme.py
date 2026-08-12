import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex

from hscredit.core.viz.binning_plots import corr_plot, distribution_plot, ks_plot
from hscredit.core.viz.utils import DEFAULT_COLORS


def _assert_axis_uses_theme_color(ax):
    expected = DEFAULT_COLORS[0].lower()

    for spine in ax.spines.values():
        assert to_hex(spine.get_edgecolor(), keep_alpha=False).lower() == expected

    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    assert tick_labels
    assert all(to_hex(label.get_color(), keep_alpha=False).lower() == expected for label in tick_labels)
    assert to_hex(ax.xaxis.label.get_color(), keep_alpha=False).lower() == expected
    assert to_hex(ax.yaxis.label.get_color(), keep_alpha=False).lower() == expected


def test_ks_plot_uses_theme_color_for_spines_ticks_and_axis_labels():
    fig = ks_plot(
        pd.Series([0.05, 0.20, 0.75, 0.95]),
        pd.Series([0, 0, 1, 1]),
    )
    fig.canvas.draw()

    assert len(fig.axes) == 2
    for ax in fig.axes:
        _assert_axis_uses_theme_color(ax)

    plt.close(fig)


def test_distribution_plot_uses_theme_color_on_both_y_axes():
    data = pd.DataFrame(
        {
            "日期": pd.to_datetime(["2025-11-01", "2025-11-15", "2025-12-01", "2025-12-15"]),
            "目标": [0, 1, 0, 1],
        }
    )

    fig = distribution_plot(data, date="日期", target="目标", freq="M")
    fig.canvas.draw()

    assert len(fig.axes) == 2
    for ax in fig.axes:
        _assert_axis_uses_theme_color(ax)

    plt.close(fig)


def test_corr_plot_uses_theme_color_for_heatmap_and_colorbar_axes():
    data = pd.DataFrame({"变量A": [1.0, 2.0, 3.0], "变量B": [3.0, 2.0, 1.0]})

    fig = corr_plot(data)
    fig.canvas.draw()

    assert len(fig.axes) == 2
    for ax in fig.axes:
        _assert_axis_uses_theme_color(ax)

    plt.close(fig)
