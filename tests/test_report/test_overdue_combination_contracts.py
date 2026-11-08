"""overdue+dpds 组合标签和剔灰分母的跨入口契约测试。"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from hscredit.core.eda.target import bad_rate_overall
from hscredit.core.rules import Rule
from hscredit.core.viz.binning_plots import _is_feature_table, _get_stats_for_target, bin_overdues_plot
from hscredit.report import feature_bin_stats, feature_bin_stats_2d
from hscredit.report.model_report import ModelReport
from hscredit.report.rule_analysis import rule_swap_analysis


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "other": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "MOB1": [0, 8, 2, 2, 0, 8],
            "target": [1, 0, 1, 1, 1, 0],
        }
    )


def test_grey_filtered_multi_target_tables_keep_union_of_bins():
    table = feature_bin_stats(
        _data(),
        "score",
        overdue="MOB1",
        dpds=[0, 3],
        del_grey=True,
        rules=[2.5, 4.5],
        margins=True,
        n_jobs=1,
    )

    detail = table[("分箱详情", "分箱标签")]
    assert table[("MOB1_0+", "样本总数")].loc[detail != "合计"].sum() == 6
    assert table[("MOB1_3+", "样本总数")].loc[detail != "合计"].sum() == 4
    assert "[2.5, 4.5)" in set(detail)


def test_grey_filtered_2d_tables_keep_target_specific_denominators():
    table = feature_bin_stats_2d(
        _data(),
        ["score", "other"],
        overdue="MOB1",
        dpds=[0, 3],
        del_grey=True,
        margins=True,
        n_jobs=1,
    )

    detail = table[("分箱详情", "分箱标签")]
    assert table[("MOB1_0+", "样本总数")].loc[detail != "合计"].sum() == 6
    assert table[("MOB1_3+", "样本总数")].loc[detail != "合计"].sum() == 4


def test_binning_skips_an_unusable_all_grey_first_target():
    data = (
        _data()
        .drop(columns="MOB1")
        .assign(
            MOB1=[0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            MOB2=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        )
    )
    table = feature_bin_stats(
        data,
        "score",
        overdue=["MOB2", "MOB1"],
        dpds=[3],
        del_grey=True,
        n_jobs=1,
    )

    assert "MOB2_3+" in table.columns.get_level_values(0)
    assert "MOB1_3+" in table.columns.get_level_values(0)


def test_model_report_single_label_bin_table_filters_grey_rows():
    class Model:
        def predict_proba(self, X):
            probability = np.asarray(X["score"], dtype=float) / 10
            return np.column_stack([1 - probability, probability])

    report = ModelReport(
        Model(),
        datasets={"样本": _data()},
        features=["score"],
        overdue="MOB1",
        dpds=[0, 3],
        del_grey=True,
        n_jobs=1,
    )

    table = report.get_bin_table("样本", label="MOB1>3", margins=True)
    assert table.iloc[-1]["样本总数"] == 4
    assert table.iloc[-1]["坏样本数"] == 2


def test_model_report_combined_metrics_exclude_rows_grey_for_every_label():
    class Model:
        def predict_proba(self, X):
            probability = np.asarray(X["score"], dtype=float) / 10
            return np.column_stack([1 - probability, probability])

    data = pd.DataFrame({"score": [1.0, 2.0, 3.0], "MOB1": [0.0, 1.0, 4.0]})
    report = ModelReport(
        Model(),
        datasets={"样本": data},
        features=["score"],
        overdue="MOB1",
        dpds=[3, 7],
        del_grey=True,
        n_jobs=1,
    )

    assert report.get_metrics().loc[report.get_metrics()["统计项"] == "样本数", "样本"].iloc[0] == 2


def test_grey_filtered_bin_table_can_be_rendered_as_overdue_plot():
    table = feature_bin_stats(
        _data(),
        "score",
        overdue="MOB1",
        dpds=[0, 3],
        del_grey=True,
        rules=[2.5, 4.5],
        margins=True,
        n_jobs=1,
    )
    for label in ("MOB1_0+", "MOB1_3+"):
        stats = _get_stats_for_target(table, label)
        assert _is_feature_table(stats)
        assert stats["样本总数"].notna().all()

    figure = bin_overdues_plot(bin_table=table, n_jobs=1)
    assert not any("Error:" in text.get_text() for axis in figure.axes for text in axis.texts)


def test_rule_swap_analysis_rejects_mixed_target_modes():
    with pytest.raises(ValueError, match="不能同时指定"):
        rule_swap_analysis(_data(), score="score", target="target", overdue="MOB1", dpds=[3], n_jobs=1)


def test_long_bin_table_plot_reads_each_labels_rows_once(monkeypatch):
    from hscredit.core.viz import binning_plots
    import matplotlib.pyplot as plt

    table = feature_bin_stats(_data(), features=["score", "other"], overdue="MOB1", dpds=[0, 3], del_grey=True, long_format=True, margins=True, n_jobs=1)
    observed = {}
    original = binning_plots.bin_plot

    def record(data, *, title, **kwargs):
        observed[title] = data.copy()
        return original(data=data, title=title, **kwargs)

    monkeypatch.setattr(binning_plots, "bin_plot", record)
    with pytest.raises(ValueError, match="多特征"):
        bin_overdues_plot(bin_table=table)
    fig = bin_overdues_plot(bin_table=table, feature="score")
    try:
        assert list(observed) == ["MOB1_0+", "MOB1_3+"]
        for name, group in observed.items():
            expected = table.loc[table["指标名称"].eq("score") & table["逾期标签"].eq(name) & table["分箱标签"].ne("合计")]
            pd.testing.assert_frame_equal(group, expected)
        assert not any("Error:" in text.get_text() for ax in fig.axes for text in ax.texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "call",
    [
        lambda data: feature_bin_stats(data, "score", overdue="MOB1", dpds=[], n_jobs=1),
        lambda data: bad_rate_overall(data, overdue="MOB1", dpds=[]),
        lambda data: Rule("score > 2", n_jobs=1).report(data, overdue="MOB1", dpds=[]),
    ],
)
def test_empty_dpd_combination_is_rejected(call):
    with pytest.raises(ValueError, match="overdue 和 dpds 不能为空"):
        call(_data())
