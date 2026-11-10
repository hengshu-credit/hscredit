import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from hscredit.core.viz.binning_plots import bin_plot, hist_plot, psi_plot
from hscredit.core.viz.style import EXTENDED_COLORS, PRIMARY_COLORS, SECONDARY_COLORS, SEMANTIC_COLORS, get_palette
from hscredit.core.viz.utils import REFERENCE_COLOR


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


def _psi_table(prefix_count):
    return pd.DataFrame(
        {
            "分箱": ["[0, 1)", "[1, 2)"],
            "样本总数": prefix_count,
            "样本占比": [prefix_count[0] / sum(prefix_count), prefix_count[1] / sum(prefix_count)],
            "坏样本率": [0.10, 0.20],
        }
    )


def test_extended_palette_keeps_primary_prefix_and_avoids_legacy_green_yellow_brown():
    assert EXTENDED_COLORS[: len(PRIMARY_COLORS)] == PRIMARY_COLORS
    assert len(EXTENDED_COLORS) > len(PRIMARY_COLORS)
    assert {"#4CAF50", "#FFC107", "#795548"}.isdisjoint({c.upper() for c in EXTENDED_COLORS})


def test_secondary_palette_preserves_brand_accents_and_shares_table_colors():
    from hscredit.excel.writer import _metric_colors

    palette = get_palette("secondary")
    assert set(PRIMARY_COLORS[1:]).issubset(palette)
    assert len(palette) == len(set(palette)) == 5
    assert all("#" + _metric_colors(metric, "2639E9")[-1] in palette for metric in ["坏样本率", "LIFT值", "坏账改善"])
    assert all(len(color) == 7 for color in palette)
    palette.append("#000000")
    assert get_palette("secondary") == SECONDARY_COLORS

    expected = {"#2639E9", "#F76E6C", "#C00000", "#D4145A", "#FE7715", "#FF8BCE", "#E52DD0", "#E52B88", "#EF4638", "#EC2D59"}
    assert len(get_palette("extended")) == 10
    assert set(get_palette("extended")) == expected
    assert set(get_palette("ordinary")).issubset(expected)


def test_reference_color_matches_semantic_baseline_and_bin_plot_line():
    assert REFERENCE_COLOR == SEMANTIC_COLORS["overall_baseline"]

    fig = bin_plot(_feature_table())
    reference_lines = [line for ax in fig.axes for line in ax.lines if line.get_label() == "整体坏样本率"]

    assert reference_lines
    assert to_hex(reference_lines[0].get_color(), keep_alpha=False).lower() == REFERENCE_COLOR.lower()
    plt.close(fig)


def test_psi_plot_returns_figure_and_saves_plain_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fig = psi_plot(_psi_table([100, 200]), _psi_table([120, 180]), save="psi.png")

    assert fig is not None
    assert (tmp_path / "psi.png").exists()
    plt.close(fig)


@pytest.mark.parametrize("labels", [None, {1: "坏样本", 0: "好样本"}])
def test_histogram_label_colors_match_the_actual_good_and_bad_distributions(labels):
    score = pd.Series([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    target = pd.Series([0, 0, 0, 1, 1, 1])
    fig = hist_plot(score, target, labels=labels, bins=[0, 0.5, 1])
    try:
        ax = fig.axes[0]
        assert len(ax.collections) == 2
        for collection in ax.collections:
            vertices = collection.get_paths()[0].vertices
            filled_x = vertices[vertices[:, 1] > 0, 0]
            expected = PRIMARY_COLORS[0 if filled_x.mean() < 0.5 else 1]
            assert to_hex(collection.get_facecolor()[0]).upper() == expected
        legend = ax.get_legend()
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        observed = {text.get_text(): to_hex(handle.get_facecolor()).upper() for text, handle in zip(legend.get_texts(), handles)}
        assert observed == {"好样本": PRIMARY_COLORS[0], "坏样本": PRIMARY_COLORS[1]}
    finally:
        plt.close(fig)
