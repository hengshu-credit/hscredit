import matplotlib.pyplot as plt
import pandas as pd

from hscredit.report import feature_efficiency_analysis
import hscredit.report.feature_analyzer as feature_analyzer_module


def _fake_bin_plot(data, ax=None, title=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    ax.bar([0, 1], [1, 2], color=["#4C8DFF", "#E85D4A"])
    if title:
        ax.set_title(title)
    return ax


def _fake_ks_plot(score, target, axes=None, **kwargs):
    axes[0].plot([0, 0.5, 1], [0, 0.4, 1])
    axes[1].plot([0, 0.3, 1], [0, 0.7, 1])
    return axes


def _fake_bin_trend_plot(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0.2, 0.8])
    ax.set_title(kwargs.get("title", "trend"))
    return fig


def test_feature_efficiency_analysis_returns_tables_figures_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", _fake_bin_plot)
    monkeypatch.setattr(feature_analyzer_module, "ks_plot", _fake_ks_plot)
    monkeypatch.setattr(feature_analyzer_module, "bin_trend_plot", _fake_bin_trend_plot)

    data = pd.DataFrame(
        {
            "score": [420, 455, 500, 530, 580, 615, 660, 710, 735, 780, 820, 860],
            "target": [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "apply_date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "channel": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"],
        }
    )

    result = feature_efficiency_analysis(
        data=data,
        feature="score",
        manual_rules=[460, 560, 700],
        target="target",
        auto_method="quantile",
        date_col="apply_date",
        group_cols="channel",
        output_dir=str(tmp_path),
        suffix="_case",
    )

    assert not result["manual_table"].empty
    assert not result["auto_table"].empty
    assert result["manual_rules"] == [460, 560, 700]
    assert isinstance(result["auto_rules"], list)
    assert len(result["comparison_figure"].axes) == 4
    assert set(result["trend_figures"].keys()) == {"manual", "auto"}
    assert set(result["saved_paths"].keys()) == {"comparison", "trend_manual", "trend_auto"}
    for path in result["saved_paths"].values():
        assert path.endswith(".png")


def test_feature_efficiency_analysis_supports_overdue_target_building(monkeypatch):
    monkeypatch.setattr(feature_analyzer_module, "bin_plot", _fake_bin_plot)
    monkeypatch.setattr(feature_analyzer_module, "ks_plot", _fake_ks_plot)
    monkeypatch.setattr(feature_analyzer_module, "bin_trend_plot", _fake_bin_trend_plot)

    data = pd.DataFrame(
        {
            "score": [420, 455, 500, 530, 580, 615, 660, 710],
            "MOB1": [0, 1, 5, 0, 4, 0, 7, 2],
            "segment": ["新客", "新客", "老客", "老客", "新客", "老客", "新客", "老客"],
        }
    )

    result = feature_efficiency_analysis(
        data=data,
        feature="score",
        manual_rules=[450, 600],
        overdue="MOB1",
        dpd=3,
        auto_method="tree",
        group_cols="segment",
    )

    assert result["target"] == "MOB1 3+"
    assert "manual" in result["trend_figures"]
    assert "auto" in result["trend_figures"]