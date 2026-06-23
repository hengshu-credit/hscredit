import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex

from hscredit.core.viz.binning_plots import bin_plot, psi_plot
from hscredit.core.viz.style import EXTENDED_COLORS, PRIMARY_COLORS, SEMANTIC_COLORS
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


def test_reference_color_matches_semantic_baseline_and_bin_plot_line():
    assert REFERENCE_COLOR == SEMANTIC_COLORS["overall_baseline"]

    fig = bin_plot(_feature_table())
    reference_lines = [
        line for ax in fig.axes for line in ax.lines
        if line.get_label() == "整体坏样本率"
    ]

    assert reference_lines
    assert to_hex(reference_lines[0].get_color(), keep_alpha=False).lower() == REFERENCE_COLOR.lower()
    plt.close(fig)


def test_psi_plot_returns_figure_and_saves_plain_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fig = psi_plot(_psi_table([100, 200]), _psi_table([120, 180]), save="psi.png")

    assert fig is not None
    assert (tmp_path / "psi.png").exists()
    plt.close(fig)
