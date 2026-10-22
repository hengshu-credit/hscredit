"""公共分析参数的名称、数据绑定和实际结果回归。"""

import inspect
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from hscredit.core.binning import OptimalBinning2D
from hscredit.core.eda.target import bad_rate_trend
from hscredit.core.viz import bin_plot, distribution_plot, psi_plot, roc_plot
from hscredit.report import ModelReport, feature_bin_stats, feature_bin_stats_2d
from hscredit.report.mining import MultiFeatureRuleMiner, SingleFeatureRuleMiner
from hscredit.report.rule_strategy import rule_group_compare
from hscredit.report.swap_analysis import ReferenceDataProvider, swap_analysis
from hscredit.utils.arguments import bind_column, normalize_features


@pytest.fixture
def frame():
    n = 120
    return pd.DataFrame(
        {
            "score": np.arange(n, dtype=float) ** 2,
            "age": np.arange(n) % 17,
            "target": np.arange(n) % 2,
            "loan": np.arange(n, dtype=float) + 10,
            "month": pd.date_range("2024-01-01", periods=n),
            "segment": np.where(np.arange(n) % 3, "A", "B"),
        },
        index=np.arange(n) + 100,
    )


@pytest.mark.parametrize("value", ["score", ["score"], ("score",), np.array(["score"])])
def test_features_single_and_arrays_share_results(frame, value):
    actual = feature_bin_stats(frame, features=value, target="target", method="quantile", max_n_bins=3, n_jobs=1)
    expected = feature_bin_stats(frame, feature="score", target="target", method="quantile", max_n_bins=3, n_jobs=1)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "as_values", [lambda s: s, lambda s: s.iloc[::-1], lambda s: s.to_numpy(), lambda s: s.tolist()]
)
def test_amount_data_matches_column_after_target_filtering(frame, as_values):
    target = frame.target.astype(float)
    target.iloc[::7] = -1
    options = dict(feature="score", target=target, del_grey=True, method="quantile", max_n_bins=3, n_jobs=1)
    expected = feature_bin_stats(frame, amount="loan", **options)
    actual = feature_bin_stats(frame, amount=as_values(frame.loan), **options)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("as_values", [lambda s: s, lambda s: s.to_numpy(), lambda s: s.tolist()])
def test_date_data_matches_column_and_freq(frame, as_values):
    expected = bad_rate_trend(frame, date="month", target="target", freq="M")
    actual = bad_rate_trend(frame, date=as_values(frame.month), target="target", freq="M")
    assert_frame_equal(actual, expected)


def test_group_and_amount_data_survive_rule_filtering(frame):
    from hscredit.core.rules import Rule

    options = dict(rule=Rule("score > 100"), target="target", n_jobs=1)
    expected = rule_group_compare(frame, group="segment", amount="loan", **options)
    actual = rule_group_compare(frame, group=frame.segment.iloc[::-1], amount=frame.loan.to_numpy(), **options)
    assert_frame_equal(actual, expected)


def test_two_dimensional_features_and_x1_x2_are_equivalent(frame):
    options = dict(target="target", method="quantile", max_n_bins=3, n_jobs=1)
    expected = feature_bin_stats_2d(frame, features=np.array(["score", "age"]), **options)
    actual = feature_bin_stats_2d(frame, x1="score", x2="age", **options)
    assert_frame_equal(actual, expected)
    binner = OptimalBinning2D(max_n_bins=3, n_jobs=1).fit(frame, x1="score", x2="age")
    assert binner.feature_x_ == "score"
    assert binner.feature_y_ == "age"
    with pytest.raises(ValueError, match="不能同时"):
        feature_bin_stats_2d(frame, features=["score", "age"], x1="score", x2="age", **options)
    with pytest.raises(ValueError, match="同时"):
        feature_bin_stats_2d(frame, x1="score", **options)


def test_cross_miner_accepts_features_array(frame):
    miner = MultiFeatureRuleMiner(features="score", method="quantile", n_jobs=1)
    miner.fit(frame, features=["score", "age"])
    expected = miner.generate_cross_matrix(x1="score", x2="age")
    assert_frame_equal(miner.generate_cross_matrix(features=np.array(["score", "age"])), expected)


def test_binding_rejects_wrong_shapes_and_indexes_without_mutation(frame):
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="等长"):
        bind_column(frame, [1, 2], "amount")
    with pytest.raises(ValueError, match="索引"):
        bind_column(frame, pd.Series(frame.loan.to_numpy()), "amount")
    with pytest.raises(ValueError, match="一维"):
        bind_column(frame, np.ones((len(frame), 1)), "group")
    collision = pd.Series(np.ones(len(frame)), index=frame.index, name="score")
    bound, column = bind_column(frame, collision, "amount")
    assert column != "score"
    assert bound[column].eq(1).all()
    assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="重复"):
        normalize_features(["score", "score"])


def test_bin_plot_honors_binner_parameters_and_result(frame):
    fig, table = bin_plot(frame, feature="score", target="target", max_n_bins=3, result=True)
    assert len(table) == 3
    plt.close(fig)
    fig, table = bin_plot(
        frame,
        feature="score",
        target="target",
        max_n_bins=10,
        max_bin_size=None,
        min_bin_size=0.4,
        min_n_bins=1,
        result=True,
    )
    assert len(table) <= 2
    plt.close(fig)
    for function in (bin_plot, psi_plot, distribution_plot):
        assert "result" in inspect.signature(function).parameters
        assert "return_frame" not in inspect.signature(function).parameters


def test_plot_dpi_reaches_image_export(tmp_path):
    from PIL import Image

    path = tmp_path / "roc.png"
    fig = roc_plot([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], dpi=150, save=path)
    plt.close(fig)
    with Image.open(path) as image:
        assert image.info["dpi"][0] == pytest.approx(150, abs=1)


def test_chi_threshold_uses_binner_name():
    binner = SingleFeatureRuleMiner(method="chi", min_chi2_threshold=9, n_jobs=1)._get_binning_instance()
    assert binner.kwargs["min_chi2_threshold"] == 9


def test_swap_config_and_effective_method_match(frame):
    actual = frame.copy()
    actual["swap_type"] = ["in_in", "in_out", "out_in", "out_out"] * (len(frame) // 4)
    methods = []
    original = ReferenceDataProvider._generate_bins

    def capture(self, values):
        methods.append(self.method)
        return original(self, values)

    with patch.object(ReferenceDataProvider, "_generate_bins", capture):
        result = swap_analysis(actual, frame, score_col="score", target="target", method="uniform", n_jobs=1)
    assert methods == ["uniform"]
    assert result.config.method == "uniform"


def test_model_report_amount_data_is_bound_without_polluting_cache(frame):
    model = LogisticRegression().fit(frame[["score", "age"]], frame.target)
    report = ModelReport(model, datasets={"train": frame}, target="target", features=["score", "age"], n_jobs=1)
    original = report._datasets["train"].X.copy()
    expected = report.get_bin_table(amount="loan", max_n_bins=3)
    actual = report.get_bin_table(amount=frame.loan.to_numpy(), max_n_bins=3)
    assert_frame_equal(actual, expected)
    changed = report.get_bin_table(amount=frame.loan.to_numpy()[::-1], max_n_bins=3)
    assert not changed.equals(actual)
    assert_frame_equal(report._datasets["train"].X, original)


def test_bin_params_priority_and_removed_keywords(frame):
    fig, table = bin_plot(frame, feature="score", target="target", bin_params={"max_n_bins": 3}, result=True)
    assert len(table) == 3
    plt.close(fig)
    fig, table = bin_plot(
        frame, feature="score", target="target", max_n_bins=4, bin_params={"max_n_bins": 3}, result=True
    )
    assert len(table) == 4
    plt.close(fig)
    with pytest.raises(TypeError, match="max_n_bins"):
        bin_plot(frame, feature="score", target="target", n_bins=3)


@pytest.mark.parametrize("selection", [{"features": "score"}, {"x1": "score", "x2": "age"}])
def test_skills_normalize_feature_inputs(frame, tmp_path, selection):
    from hscredit.skills_runtime import execute_skill

    operation = "optimal_binning_2d_fit" if "x1" in selection else "optimal_binning_fit"
    result = execute_skill(
        "hsbin",
        {
            "version": "1",
            "operation": operation,
            "inputs": {"data": {"kind": "object_ref", "ref": "data"}},
            "parameters": {**selection, "target": "target", "max_n_bins": 3, "method": "quantile", "n_jobs": 1},
            "output": {"directory": str(tmp_path), "name": operation, "overwrite": False},
            "environment": {"mode": "current", "install_missing": False},
        },
        objects={"data": frame},
    )
    assert result["status"] == "success"


def test_model_report_binds_auxiliary_data_per_dataset(frame, tmp_path):
    model = LogisticRegression().fit(frame[["score", "age"]], frame.target)
    datasets = {"train": frame.iloc[:60], "test": frame.iloc[60:90]}
    report = ModelReport(model, datasets=datasets, target="target", features="score", n_jobs=1)
    path = tmp_path / "auxiliary.xlsx"
    report.to_excel(
        path,
        with_plots=False,
        include_sample_records=False,
        date={key: data.month for key, data in datasets.items()},
        group={key: data.segment for key, data in datasets.items()},
        amount={key: data.loan for key, data in datasets.items()},
        freq="M",
    )
    assert path.is_file()
    for key, data in datasets.items():
        assert_frame_equal(report._datasets[key].X, data)


def test_skill_plot_resolves_date_data(frame, tmp_path):
    from hscredit.skills_runtime import execute_skill

    result = execute_skill(
        "hsviz",
        {
            "version": "1",
            "operation": "plot",
            "inputs": {"data": {"kind": "object_ref", "ref": "data"}, "date": {"kind": "object_ref", "ref": "date"}},
            "parameters": {"plot": "distribution_plot", "kwargs": {"target": "target", "freq": "M"}},
            "output": {"directory": str(tmp_path), "name": "dates", "format": "png", "overwrite": False},
            "environment": {"mode": "current", "install_missing": False},
        },
        objects={"data": frame, "date": frame.month},
    )
    assert result["status"] == "success"


@pytest.mark.parametrize("explicit", [False, True])
def test_predictor_binning_priority_survives_clone(frame, explicit):
    from sklearn.base import clone
    from hscredit.report import OverduePredictor

    config = {"method": "uniform", "max_n_bins": 3, "min_bin_size": 0.01}
    options = {"method": "quantile", "max_n_bins": 4} if explicit else {}
    predictor = OverduePredictor("score", target="target", bin_params=config, n_jobs=1, **options)
    expected_method, expected_bins = ("quantile", 4) if explicit else ("uniform", 3)
    assert predictor.method == expected_method
    assert predictor.max_n_bins == expected_bins
    copied = clone(predictor)
    predictor.fit(frame)
    copied.fit(frame)
    assert_frame_equal(copied.bin_table_, predictor.bin_table_)
    assert config == {"method": "uniform", "max_n_bins": 3, "min_bin_size": 0.01}
    assert predictor.binner_.method == expected_method


@pytest.mark.parametrize("alias, canonical", [("ME", "M"), ("QE", "Q"), ("YE", "Y"), ("2ME", "2M")])
def test_frequency_aliases_match_across_reports_and_plots(frame, alias, canonical):
    options = dict(date="month", target="target")
    assert_frame_equal(bad_rate_trend(frame, freq=alias, **options), bad_rate_trend(frame, freq=canonical, **options))
    assert_frame_equal(
        rule_group_compare(frame, rule="score > 100", freq=alias, n_jobs=1, **options),
        rule_group_compare(frame, rule="score > 100", freq=canonical, n_jobs=1, **options),
    )
    try:
        actual = distribution_plot(frame, freq=alias, result=True, **options)
        expected = distribution_plot(frame, freq=canonical, result=True, **options)
        assert_frame_equal(actual, expected)
    finally:
        plt.close("all")


@pytest.mark.parametrize("freq", ["D", "Y"])
def test_feature_stability_accepts_common_frequencies(frame, freq):
    from hscredit.core.eda.feature import feature_stability_over_time
    from hscredit.core.eda.stability import time_psi_tracking

    data = frame.copy()
    data["month"] = pd.date_range("2023-01-01", periods=len(data), freq="7D")
    stats = feature_stability_over_time(data, features="score", date="month", freq=freq, n_jobs=1)
    assert stats.iloc[0]["统计期数"] == data.month.dt.to_period(freq).nunique()
    psi = time_psi_tracking(data, features="score", date="month", freq=freq, n_jobs=1)
    assert len(psi) == data.month.dt.to_period(freq).nunique() - 1


@pytest.mark.parametrize("function", [bad_rate_trend, distribution_plot])
def test_invalid_frequency_fails_instead_of_changing_grouping(frame, function):
    with pytest.raises(ValueError, match="freq"):
        function(frame, date="month", target="target", freq="invalid")


def test_prediction_report_resolves_bin_params_before_defaults(frame):
    from hscredit.report.overdue_predictor import overdue_prediction_report

    expected = overdue_prediction_report(frame, "score", target="target", method="quantile", max_n_bins=3, n_jobs=1)
    actual = overdue_prediction_report(
        frame, "score", target="target", bin_params={"method": "quantile", "max_n_bins": 3}, n_jobs=1
    )
    assert_frame_equal(actual, expected)


def test_swap_bin_params_do_not_override_explicit_method(frame):
    actual = frame.assign(swap_type=["in_in", "in_out", "out_in", "out_out"] * (len(frame) // 4))
    config = {"method": "uniform", "max_n_bins": 3}
    report = swap_analysis(actual, frame, score_col="score", target="target", bin_params=config, n_jobs=1)
    assert report.config.method == "uniform"
    overridden = swap_analysis(
        actual, frame, score_col="score", target="target", method="quantile", bin_params=config, n_jobs=1
    )
    assert overridden.config.method == "quantile"


def test_model_report_invalid_frequency_cannot_publish_complete_report(frame, tmp_path):
    from hscredit.report.result import ReportGenerationError

    model = LogisticRegression().fit(frame[["score"]], frame.target)
    report = ModelReport(model, datasets={"train": frame}, target="target", features="score", n_jobs=1)
    path = tmp_path / "report.xlsx"
    path.write_bytes(b"previous-report")
    with pytest.raises(ReportGenerationError):
        report.to_excel(path, date="month", freq="invalid", with_plots=False, mode="strict")
    assert path.read_bytes() == b"previous-report"


@pytest.mark.parametrize("keyword", ["feature_names", "method", "method_kwargs"])
def test_model_report_rejects_removed_keywords_before_prediction(frame, keyword):
    model = LogisticRegression().fit(frame[["score"]], frame.target)
    with pytest.raises(TypeError, match=f"{keyword}.*已移除"):
        ModelReport(model, datasets={"train": frame}, target="target", **{keyword: "ignored"})


def test_benchmark_rejects_removed_method_list_keyword(frame):
    from hscredit.report.feature_analyzer import benchmark_binning_methods

    with pytest.raises(TypeError, match="hscredit_methods.*methods"):
        benchmark_binning_methods(frame, "score", target="target", hscredit_methods=["quantile"], n_jobs=1)


@pytest.mark.parametrize("entry", ["function", "dataframe", "series"])
@pytest.mark.parametrize("explicit", [False, True])
def test_summary_binning_priority_matches_direct_configuration(frame, entry, explicit):
    from hscredit.core.eda import feature_summary

    options = {"method": "quantile", "max_n_bins": 4} if explicit else {}
    config = {"method": "uniform", "max_n_bins": 3}
    expected = feature_summary(
        frame, features="score", y="target", force_numeric=["score"], n_jobs=1, **{**config, **options}
    )
    common = dict(force_numeric=["score"], n_jobs=1, bin_params=config, **options)
    if entry == "function":
        actual = feature_summary(frame, features="score", y="target", **common)
    elif entry == "dataframe":
        actual = frame.summary(features="score", y="target", **common)
    else:
        actual = frame.score.summary(y=frame.target, **common)
    assert_frame_equal(actual, expected)
    assert config == {"method": "uniform", "max_n_bins": 3}


def test_efficiency_analysis_uses_explicit_binning_options(frame):
    from hscredit.report import feature_efficiency_analysis

    options = dict(feature="score", target="target", method="quantile", max_n_bins=4, n_jobs=1)
    try:
        expected = feature_efficiency_analysis(frame, **options)
        actual = feature_efficiency_analysis(frame, bin_params={"method": "uniform", "max_n_bins": 3}, **options)
        assert_frame_equal(actual["auto_table"], expected["auto_table"])
    finally:
        plt.close("all")
