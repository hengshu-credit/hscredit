"""跨入口验证逾期比较符、边界标签、独立灰样本分母与参数透传。"""

import importlib
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hscredit.core.eda.target import bad_rate_overall, bad_rate_by_dimension, bad_rate_by_bins, bad_rate_trend
from hscredit.core.rules import Rule
from hscredit.core.viz.binning_plots import distribution_plot, bin_overdues_plot
from hscredit.report import (
    feature_bin_stats,
    feature_bin_stats_2d,
    feature_binning_summary,
    feature_group_binning_summary,
)
from hscredit.report.model_report import ModelReport
from hscredit.report.overdue_predictor import OverduePredictor
from hscredit.report.rule_analysis import ruleset_analysis, rule_swap_analysis
from hscredit.report.rule_strategy import swap_out_report
from hscredit.utils.overdue import make_overdue_target, overdue_grey_mask


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "评分": np.arange(8, dtype=float),
            "特征": [0, 1, 0, 1, 0, 1, 0, 1],
            "MOB 1": [-1, 0, 1, 2, 3, 4, 8, np.nan],
            "金额": np.arange(1, 9, dtype=float) * 100,
            "客群": ["A"] * 4 + ["B"] * 4,
            "日期": pd.to_datetime(["2026-01-01"] * 4 + ["2026-02-01"] * 4),
        },
        index=[10, 20, 30, 40, 50, 60, 70, 80],
    )


LABELS = {
    ">": [0, 0, 0, 0, 0, 1, 1, 0],
    ">=": [0, 0, 0, 0, 1, 1, 1, 0],
    "<": [1, 1, 1, 1, 0, 0, 0, 0],
    "<=": [1, 1, 1, 1, 1, 0, 0, 0],
}
GREY = {">": [30, 40, 50], ">=": [30, 40], "<": [], "<=": []}


def expected_target(data, op, del_grey):
    y = pd.Series(LABELS[op], index=data.index)
    return y.drop(index=GREY[op]) if del_grey else y


@pytest.mark.parametrize("op", LABELS)
@pytest.mark.parametrize("del_grey", [False, True])
def test_boundary_labels_match_across_eda_rules_and_binning(data, op, del_grey):
    original = data.copy(deep=True)
    expected = expected_target(data, op, del_grey)
    params = dict(overdue="MOB 1", dpds=3, del_grey=del_grey, overdue_operator=op)
    overall = bad_rate_overall(data, **params).iloc[0]
    assert overall["样本总数"] == len(expected)
    assert overall["坏样本数"] == expected.sum()

    for amount in [None, "金额"]:
        weights = pd.Series(1, index=expected.index) if amount is None else data.loc[expected.index, amount]
        table = feature_bin_stats(data, "评分", rules=[3], margins=True, amount=amount, n_jobs=1, **params)
        total = table.iloc[-1]
        assert total["样本总数"] == weights.sum()
        assert total["坏样本数"] == weights.loc[expected.eq(1)].sum()

        report = Rule("评分 >= 0", n_jobs=1).report(data, margins=True, amount=amount, **params)
        label = "MOB 1 3+" if op == ">" else f"MOB 1{op}3"
        total = report.loc[report[("分箱详情", "分箱")] == "合计"].iloc[0]
        count_group = label if del_grey and op in (">", ">=") else "分箱详情"
        assert total[(count_group, "样本总数")] == weights.sum()
        assert total[(label, "坏样本数")] == weights.loc[expected.eq(1)].sum()
    assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "op, dpd, expected, grey",
    [
        (">", 0, [0, 0, 1, 1, 1], []),
        (">=", 0, [0, 1, 1, 1, 1], []),
        ("<", 0, [1, 0, 0, 0, 0], []),
        ("<=", 0, [1, 1, 0, 0, 0], []),
        (">", 0.5, [0, 0, 0, 0, 1], [2, 3]),
        (">=", 0.5, [0, 0, 0, 1, 1], [2]),
        ("<", 0.5, [1, 1, 1, 0, 0], []),
        ("<=", 0.5, [1, 1, 1, 1, 0], []),
    ],
)
def test_zero_and_decimal_thresholds(op, dpd, expected, grey):
    values = pd.Series([-1, 0, 0.25, 0.5, 1], dtype="Float64")
    actual = make_overdue_target(values, dpd, overdue_operator=op)
    assert actual.tolist() == expected
    assert overdue_grey_mask(values, dpd, op).loc[lambda x: x].index.tolist() == grey
    masked = make_overdue_target(values, dpd, del_grey=True, overdue_operator=op)
    assert masked.isna().loc[lambda x: x].index.tolist() == grey


class ProbabilityModel:
    """提供与输入索引无关、可稳定复用的预测概率。"""

    def predict_proba(self, X):
        p = np.asarray(X.iloc[:, 0], dtype=float) / 10 + 0.1
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("op", LABELS)
def test_model_report_multilabel_counts_labels_and_config_roundtrip(data, op):
    report = ModelReport(
        ProbabilityModel(),
        datasets={"样本": data},
        feature_names=["评分"],
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=op,
        del_grey=True,
        n_jobs=1,
    )
    expected = expected_target(data, op, True)
    label = f"MOB 1{op}3"
    display = "MOB 1@3" if op == ">" else label
    summary = report.summary()
    assert summary.loc[display, ("样本数", "样本")] == len(expected)
    assert summary.loc[display, ("坏样本率", "样本")] == pytest.approx(expected.mean())
    table = report.get_bin_table("样本", max_n_bins=2, margins=True, labels=report._label_names)
    total = table.iloc[-1]
    group = label if op in (">", ">=") else "分箱详情"
    assert total[(group, "样本总数")] == len(expected)
    assert total[(label, "坏样本数")] == expected.sum()
    restored = ModelReport(**report.init_params_)
    assert_frame_equal(restored.summary(), summary)


def test_model_dict_operator_and_explicit_override(data):
    config = {"overdue": "MOB 1", "dpds": 3, "overdue_operator": "<="}
    kwargs = dict(model=ProbabilityModel(), datasets={"样本": data}, feature_names=["评分"], target=config, n_jobs=1)
    inherited = ModelReport(**kwargs)
    overridden = ModelReport(**kwargs, overdue_operator=">")
    assert inherited._datasets["样本"].y.tolist() == LABELS["<="]
    assert overridden._datasets["样本"].y.tolist() == LABELS[">"]
    assert config["overdue_operator"] == "<="


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_summary_group_and_2d_use_the_same_operator(data, op):
    params = dict(overdue="MOB 1", dpds=[3, 0], overdue_operator=op, del_grey=True, margins=True, n_jobs=1)
    label = f"MOB 1{op}3"
    expected = expected_target(data, op, True)
    table = feature_bin_stats_2d(data, ["评分", "特征"], max_n_bins=2, **params)
    group = label if op == ">=" else "分箱详情"
    assert table.iloc[-1][(group, "样本总数")] == len(expected)
    assert table.iloc[-1][(label, "坏样本数")] == expected.sum()
    tables, summary = feature_binning_summary(data, "评分", methods="quantile", max_n_bins=2, **params)
    assert summary.iloc[0][("坏样本数", label)] == expected.sum()
    _, group_summary = feature_group_binning_summary(
        data,
        "评分",
        methods="quantile",
        group_col="客群",
        max_n_bins=2,
        **params,
    )
    assert group_summary[("坏样本数", label)].sum() == expected.sum()
    assert label in tables["评分"]["quantile"].columns.get_level_values(0)


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_eda_dimension_trend_and_bins(data, op):
    expected = expected_target(data, op, True)
    params = dict(overdue="MOB 1", dpds=3, overdue_operator=op, del_grey=True)
    tables = [
        bad_rate_by_dimension(data, "客群", **params),
        bad_rate_trend(data, "日期", **params),
        bad_rate_by_bins(data, "评分", n_bins=2, **params),
    ]
    for table in tables:
        assert table["样本数"].sum() == len(expected)
        assert table["坏样本数"].sum() == expected.sum()


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_predictor_label_names_and_explicit_override(data, op):
    predictor = OverduePredictor(
        "评分",
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=op,
        del_grey=True,
        rules=[3],
        bin_params={"overdue_operator": ">"},
        n_jobs=1,
    ).fit(data)
    expected = expected_target(data, op, True)
    label = f"MOB 1{op}3"
    assert label in predictor.target_names_
    total = predictor.bin_table_.iloc[-1]
    assert total[(label, "坏样本数")] == expected.sum()
    group = label if op == ">=" else "分箱详情"
    assert total[(group, "样本总数")] == len(expected)


@pytest.mark.parametrize("op", ["<", "<="])
def test_lower_operators_del_grey_is_a_noop_in_reports(data, op):
    params = dict(overdue="MOB 1", dpds=[3, 0], overdue_operator=op, n_jobs=1)
    for fn, args in [(feature_bin_stats, (data, "评分")), (ruleset_analysis, (data, [Rule("评分 >= 3")]))]:
        kept = fn(*args, **params, del_grey=False)
        removed = fn(*args, **params, del_grey=True)
        assert_frame_equal(removed, kept)


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_distribution_and_bin_plot_statistics_match_operator(data, op, monkeypatch):
    params = dict(overdue=["MOB 1"], dpds=[3], overdue_operator=op, del_grey=True)
    result = distribution_plot(data, date="日期", result=True, **params)
    assert isinstance(result, pd.DataFrame)
    rates = result[f"MOB 1{op}3_坏样本率"].to_numpy()
    expected = expected_target(data, op, True)
    assert rates == pytest.approx(
        [
            expected.loc[expected.index.intersection(data.index[:4])].mean(),
            expected.loc[expected.index.intersection(data.index[4:])].mean(),
        ]
    )
    module = importlib.import_module("hscredit.core.viz.binning_plots")
    original = module._compute_bin_stats_from_raw_data
    captured = []

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value)
        return value

    monkeypatch.setattr(module, "_compute_bin_stats_from_raw_data", capture)
    bin_overdues_plot(data, feature="评分", rules={"评分": [3]}, n_jobs=1, **params)
    assert captured[0]["样本总数"].sum() == len(expected)
    assert captured[0]["坏样本数"].sum() == expected.sum()
    plt.close("all")


@pytest.mark.parametrize("invalid", ["==", "!=", "gt", "", None, [">"]])
def test_invalid_operator_rejected_before_binning(data, invalid):
    with pytest.raises(ValueError, match="overdue_operator.*仅支持"):
        feature_bin_stats(data, "评分", overdue="MOB 1", dpds=3, overdue_operator=invalid)


@pytest.mark.parametrize("op", [">=", "<", "<="])
def test_swap_pipeline_reference_and_actual_labels_agree(data, op):
    # 规则置换保留未表现样本的既有缺失值语义；本例用已表现数据核对剔灰分母。
    data = data.iloc[:-1]
    expected = pd.Series(LABELS[op][:-1], index=data.index).drop(index=GREY[op])
    result = rule_swap_analysis(
        data,
        score="评分",
        reference_data=data,
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=op,
        del_grey=True,
        bin_method="quantile",
        max_n_bins=2,
        bin_params={"overdue_operator": ">"},
        rules_out=[Rule("评分 >= 4")],
        n_jobs=1,
    )
    label = f"MOB 1{op}3"
    pipeline = result["swap_pipeline"]
    group = label if op == ">=" else "分箱详情"
    assert pipeline.iloc[0][(group, "样本总数")] == len(expected)
    assert label in pipeline.columns.get_level_values(0)


@pytest.mark.parametrize("op", [">=", "<="])
def test_swap_out_workbook_contains_consistent_labels(data, op, tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "策略.xlsx"
    swap_out_report(
        data,
        rules=Rule("评分 >= 4"),
        features=["评分"],
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=op,
        del_grey=True,
        group_col="客群",
        methods="quantile",
        n_jobs=1,
        save=str(path),
    )
    workbook = load_workbook(path, data_only=True)
    for name in ["策略迭代", "变量分箱"]:
        values = [cell.value for row in workbook[name] for cell in row]
        assert f"MOB 1{op}3" in values
        assert "MOB 1 3+" not in values
        assert "MOB 1_3+" not in values


@pytest.mark.parametrize("explicit, configured, expected_op", [(None, "<=", "<="), (">=", "<", ">="), (">", "<=", ">")])
def test_auto_feature_report_operator_precedence(data, explicit, configured, expected_op, tmp_path, monkeypatch):
    from hscredit.excel import ExcelWriter
    from hscredit.report import feature_analyzer

    data = data.assign(评分=data["评分"] + 1)
    writer = ExcelWriter(system="windows")
    original = feature_analyzer.feature_bin_stats
    tables = []

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        tables.append(value)
        return value

    monkeypatch.setattr(feature_analyzer, "feature_bin_stats", capture)
    feature_analyzer.auto_feature_analysis(
        data,
        features=["评分"],
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=explicit,
        del_grey=True,
        excel_writer=writer,
        sheet="标签验证",
        pictures=[],
        show_progress=False,
        output_dir=str(tmp_path),
        margins=True,
        n_jobs=1,
        bin_params={"method": "quantile", "max_n_bins": 2, "overdue_operator": configured},
    )
    label = "MOB 1_3+" if expected_op == ">" else f"MOB 1{expected_op}3"
    assert tables[0].iloc[-1][(label, "坏样本数")] == sum(LABELS[expected_op])
    values = [cell.value for row in writer.get_sheet_by_name("标签验证") for cell in row]
    display = "MOB 1@3" if expected_op == ">" else label
    assert display in values


@pytest.mark.parametrize("op", [">=", "<="])
def test_tree_report_and_rule_table_forward_operator(data, op):
    from hscredit.report.mining import DecisionTreeAnalyzer, ManualTreeExtractor

    fit_data = data.assign(target=LABELS[">"])
    tree = DecisionTreeAnalyzer(target="target", features=["评分"], tree_params={"max_depth": 1}, n_jobs=1).fit(
        fit_data
    )
    manual = ManualTreeExtractor(target="target", max_depth=1, min_samples_leaf=1, n_jobs=1).fit(
        fit_data, features=["评分"]
    )
    manual.manual_split(fit_data, feature="评分", threshold=3, node=0)
    for estimator in [tree, manual]:
        params = dict(overdue="MOB 1", dpds=[3, 0], overdue_operator=op, del_grey=True, leaf_only=True)
        report = estimator.report({"样本": data}, **params)["样本"]
        assert report[(f"MOB 1{op}3", "坏样本数")].sum() == sum(LABELS[op])
        table = estimator.get_rule_table(data, **params)
        assert (f"MOB 1{op}3", "坏样本数") in table.columns


@pytest.mark.parametrize("op", [">=", "<="])
def test_benchmark_fits_only_target_valid_rows(data, op):
    from hscredit.report.feature_analyzer import benchmark_binning_methods

    report = benchmark_binning_methods(
        data,
        "评分",
        overdue="MOB 1",
        dpds=[3],
        overdue_operator=op,
        del_grey=True,
        hscredit_methods=["quantile"],
        max_n_bins=2,
        prebinning=None,
        min_bin_size=0.01,
        monotonic=False,
        n_jobs=1,
    )
    assert f"MOB 1{op}3" in report.columns.get_level_values(0)
    assert report[(f"MOB 1{op}3", "错误信息")].isna().all()


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_operator_survives_parallel_tasks(data, backend):
    params = dict(overdue="MOB 1", dpds=[3, 0], overdue_operator=">=", del_grey=True, margins=True)
    for fn, args in [(feature_bin_stats, (data, ["评分", "特征"])), (Rule("评分 >= 3").report, (data,))]:
        expected = fn(*args, n_jobs=1, **params)
        actual = fn(*args, n_jobs=2, parallel_backend=backend, **params)
        assert_frame_equal(actual, expected)


@pytest.fixture(scope="module")
def loan_data():
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少真实放款数据 hscredit_yyp.xlsx")
    return pd.read_excel(
        path,
        usecols=[
            "衡枢鉴真分老客版",
            "近六个月非银多头机构数",
            "青云24",
            "FPD",
            "MOB1",
            "放款金额",
            "放款时间",
            "商品类别",
        ],
    )


@pytest.mark.parametrize("op", LABELS)
@pytest.mark.parametrize("del_grey", [False, True])
def test_real_loan_data_counts_and_amounts(loan_data, op, del_grey):
    days = loan_data["MOB1"]
    for amount in [None, "放款金额"]:
        table = feature_bin_stats(
            loan_data,
            "衡枢鉴真分老客版",
            overdue="MOB1",
            dpds=[7, 3, 0],
            overdue_operator=op,
            del_grey=del_grey,
            amount=amount,
            method="quantile",
            max_n_bins=3,
            margins=True,
            n_jobs=1,
        )
        total = table.iloc[-1]
        for dpd in [7, 3, 0]:
            bad = {">": days > dpd, ">=": days >= dpd, "<": days < dpd, "<=": days <= dpd}[op]
            valid = pd.Series(True, index=days.index)
            if del_grey and op == ">":
                valid = ~((days > 0) & (days <= dpd))
            elif del_grey and op == ">=":
                valid = ~((days > 0) & (days < dpd))
            weights = pd.Series(1, index=days.index) if amount is None else loan_data[amount]
            label = f"MOB1_{dpd}+" if op == ">" else f"MOB1{op}{dpd}"
            group = label if del_grey and op in (">", ">=") else "分箱详情"
            assert total[(group, "样本总数")] == pytest.approx(weights[valid].sum())
            assert total[(label, "坏样本数")] == pytest.approx(weights[valid & bad].sum())


@pytest.mark.parametrize("op", LABELS)
def test_swap_reference_rates_exclude_only_operator_grey(data, op):
    from hscredit.report.swap_analysis import swap_analysis, SwapType

    expected = expected_target(data, op, True)
    result = swap_analysis(
        data.assign(swap_type="in-in"),
        data,
        score_col="评分",
        overdue="MOB 1",
        dpds=[3],
        overdue_operator=op,
        del_grey=True,
        custom_bins=[-np.inf, np.inf],
        n_jobs=1,
    )
    label = "MOB 1_3+" if op == ">" else f"MOB 1{op}3"
    assert result.targets == [label]
    assert result.count_stats[SwapType.IN_IN][f"predicted_bad_rate_{label}"] == pytest.approx(expected.mean())


@pytest.mark.parametrize("op", [">=", "<="])
def test_efficiency_dataset_preserves_decimal_threshold(data, op):
    from hscredit.report.feature_analyzer import _prepare_efficiency_dataset, feature_efficiency_analysis

    data = data.assign(**{"MOB 1": data["MOB 1"] / 2})
    working, _, label = _prepare_efficiency_dataset(
        data,
        "评分",
        "target",
        overdue="MOB 1",
        dpd=1.5,
        overdue_operator=op,
        del_grey=True,
    )
    expected = expected_target(data, op, True)
    assert label == f"MOB 1{op}1.5"
    assert working[label].tolist() == expected.tolist()
    result = feature_efficiency_analysis(
        data,
        "评分",
        overdue="MOB 1",
        dpd=1.5,
        overdue_operator=op,
        del_grey=True,
        manual_rules=[3],
        auto_method="quantile",
        prebinning=None,
        max_n_bins=2,
        n_jobs=1,
    )
    assert isinstance(result, dict)
    plt.close("all")


def test_legacy_model_report_restores_default_operator(data):
    report = ModelReport(
        ProbabilityModel(),
        datasets={"样本": data},
        feature_names=["评分"],
        overdue="MOB 1",
        dpds=[3, 0],
        del_grey=True,
        n_jobs=1,
    )
    expected = report.summary()
    # 模拟新参数引入前的制品：实例状态和初始化配置均不含比较符。
    del report.overdue_operator
    report.init_params_.pop("overdue_operator")
    restored = pickle.loads(pickle.dumps(report))
    restored._invalidate_caches()
    assert restored.overdue_operator == ">"
    assert restored.init_params_["overdue_operator"] == ">"
    assert_frame_equal(restored.summary(), expected)
    restored.add_dataset(key="新增", label="新增", X=data)
    table = restored.get_bin_table("新增", labels=restored._label_names, margins=True, max_n_bins=2)
    assert table.iloc[-1][("MOB 1>3", "坏样本数")] == 2


def test_legacy_predictor_can_be_refitted_after_restore(data):
    predictor = OverduePredictor("评分", overdue="MOB 1", dpds=[3, 0], rules=[3], n_jobs=1).fit(data)
    expected = predictor.bin_table_.copy()
    del predictor.overdue_operator
    restored = pickle.loads(pickle.dumps(predictor))
    assert restored.get_params()["overdue_operator"] is None
    restored.fit(data)
    assert_frame_equal(restored.bin_table_, expected)


@pytest.mark.parametrize("op", LABELS)
def test_normalized_threshold_names_match_predictor_and_swap_tables(data, op):
    params = dict(overdue="MOB 1", dpds=[3.0, 0.0], overdue_operator=op, n_jobs=1)
    predictor = OverduePredictor("评分", rules=[3], **params).fit(data)
    label = "MOB 1_3+" if op == ">" else f"MOB 1{op}3"
    assert predictor.target_names_[0] == label
    assert predictor.bin_table_.iloc[-1][(label, "坏样本数")] == sum(LABELS[op])
    swap_data = data.dropna(subset=["MOB 1"])
    result = rule_swap_analysis(
        swap_data,
        score="评分",
        reference_data=swap_data,
        rules_out=[Rule("评分 >= 4")],
        max_n_bins=2,
        **params,
    )
    assert label in result["swap_pipeline"].columns.get_level_values(0)


@pytest.mark.parametrize("dpds", [[3.0, 0.0], (3, 0), np.array([3, 0]), pd.Index([3, 0])])
def test_model_report_normalizes_threshold_containers(data, dpds):
    report = ModelReport(
        ProbabilityModel(),
        datasets={"样本": data},
        feature_names=["评分"],
        overdue="MOB 1",
        dpds=dpds,
        overdue_operator=">=",
        n_jobs=1,
    )
    assert report._label_names == ["MOB 1>=3", "MOB 1>=0"]
    table = report.get_bin_table("样本", labels=report._label_names, margins=True, max_n_bins=2)
    assert table.iloc[-1][("MOB 1>=3", "坏样本数")] == 3


@pytest.mark.parametrize("op", [">=", "<="])
def test_efficiency_auto_kwargs_cannot_override_label_definition(data, op):
    from hscredit.report.feature_analyzer import feature_efficiency_analysis

    auto_kwargs = {"overdue_operator": ">", "del_grey": True}
    result = feature_efficiency_analysis(
        data,
        "评分",
        overdue="MOB 1",
        dpd=3,
        overdue_operator=op,
        del_grey=False,
        manual_rules=[3],
        auto_method="quantile",
        prebinning=None,
        max_n_bins=2,
        margins=True,
        n_jobs=1,
        auto_kwargs=auto_kwargs,
    )
    for key in ["manual_table", "auto_table"]:
        assert result[key].iloc[-1]["坏样本数"] == sum(LABELS[op])
        assert result[key].iloc[-1]["样本总数"] == len(data)
    assert auto_kwargs == {"overdue_operator": ">", "del_grey": True}
    plt.close("all")


def test_benchmark_accepts_scalar_decimal_threshold(data):
    from hscredit.report.feature_analyzer import benchmark_binning_methods

    params = dict(overdue="MOB 1", overdue_operator=">=", del_grey=True, hscredit_methods=["quantile"], n_jobs=1)
    scalar = benchmark_binning_methods(data, "评分", dpds=0.5, **params)
    sequence = benchmark_binning_methods(data, "评分", dpds=[0.5], **params)
    assert_frame_equal(scalar, sequence)
    assert ("MOB 1>=0.5", "坏样本率序列") in scalar.columns


@pytest.mark.parametrize("op", LABELS)
def test_normalized_thresholds_use_the_same_values_for_labels_and_comparisons(data, op):
    params = dict(overdue="MOB 1", overdue_operator=op, del_grey=True)
    expected = bad_rate_overall(data, dpds=[3, 0], **params)
    actual = bad_rate_overall(data, dpds=pd.Index(["3.0", "0", 3]), **params)
    assert_frame_equal(actual, expected)
    assert make_overdue_target(data["MOB 1"], "3.0", overdue_operator=op).tolist() == LABELS[op]


@pytest.mark.parametrize("container", [np.array, pd.Index, pd.Series])
def test_swap_threshold_containers_match_normalized_list(data, container):
    sample = data.dropna(subset=["MOB 1"])
    params = dict(
        score="评分",
        reference_data=sample,
        overdue="MOB 1",
        overdue_operator=">=",
        rules_out=[Rule("评分 >= 4")],
        del_grey=True,
        max_n_bins=2,
        n_jobs=1,
    )
    expected = rule_swap_analysis(sample, dpds=[3, 0], **params)
    actual = rule_swap_analysis(sample, dpds=container(["3.0", "0", "3"]), **params)
    for key in expected:
        assert_frame_equal(actual[key], expected[key])


@pytest.mark.parametrize("op", LABELS)
def test_auto_feature_summary_excludes_primary_grey_samples(data, op, monkeypatch, tmp_path):
    from hscredit.report import auto_feature_analysis

    original = pd.DataFrame.summary
    observed = []

    def capture(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        observed.append((frame.copy(), kwargs["y"].copy(), result.copy()))
        return result

    monkeypatch.setattr(pd.DataFrame, "summary", capture)
    auto_feature_analysis(
        data,
        features=["评分"],
        overdue="MOB 1",
        dpds=[3, 0],
        overdue_operator=op,
        del_grey=True,
        pictures=[],
        show_progress=False,
        bin_params={"method": "quantile", "max_n_bins": 2},
        n_jobs=1,
        excel_writer=str(tmp_path / "报告.xlsx"),
        output_dir=str(tmp_path),
    )
    frame, target, summary = observed[0]
    expected = expected_target(data, op, True)
    assert frame.index.tolist() == expected.index.tolist()
    assert target.tolist() == expected.tolist()
    assert summary.iloc[0]["样本数"] == len(expected)


def test_paired_overdue_plots_normalize_without_dropping_repeated_thresholds(data):
    sample = data.assign(MOB2=data["MOB 1"])
    figure = bin_overdues_plot(
        sample,
        feature="评分",
        overdue=["MOB 1", "MOB2"],
        dpds=["3.0", 3.0],
        overdue_operator=">=",
        rules={"评分": [3]},
        n_jobs=1,
    )
    assert [ax.get_title() for ax in figure.axes[:2]] == ["MOB 1 (>= 3)", "MOB2 (>= 3)"]
    actual = distribution_plot(sample, date="日期", overdue="MOB 1", dpds="3.0", overdue_operator=">=", result=True)
    expected = distribution_plot(sample, date="日期", overdue=["MOB 1"], dpds=[3], overdue_operator=">=", result=True)
    assert_frame_equal(actual, expected)
    plt.close("all")


@pytest.mark.parametrize("container", [np.array, pd.Index, pd.Series])
def test_auto_feature_analysis_normalizes_and_deduplicates_thresholds(data, container, tmp_path):
    from hscredit.excel import ExcelWriter
    from hscredit.report import auto_feature_analysis

    writer = ExcelWriter()
    expected_writer = ExcelWriter()
    try:
        params = dict(
            features=["评分"],
            overdue=pd.Index(["MOB 1"]),
            overdue_operator=">=",
            del_grey=True,
            pictures=[],
            show_progress=False,
            n_jobs=1,
            bin_params={"method": "quantile", "max_n_bins": 2},
            output_dir=str(tmp_path),
        )
        auto_feature_analysis(data, dpds=container(["3.0", "0", "3"]), excel_writer=writer, **params)
        auto_feature_analysis(data, dpds=[3, 0], excel_writer=expected_writer, **params)
        rows = list(writer.get_sheet_by_name("分析报告").values)
        assert rows == list(expected_writer.get_sheet_by_name("分析报告").values)
        assert any("MOB 1>=3" in row for row in rows)
        assert not any("MOB 1>=3.0" in row for row in rows)
    finally:
        writer.workbook.close()
        expected_writer.workbook.close()


@pytest.mark.parametrize("op", LABELS)
def test_real_loan_data_eda_model_rules_and_category_share_overdue_definition(loan_data, op):
    from sklearn.dummy import DummyClassifier

    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    model = DummyClassifier(strategy="prior").fit(loan_data[features], loan_data["FPD"])
    params = dict(overdue="MOB1", dpds=[7, 3, 0], overdue_operator=op, del_grey=True)
    report = ModelReport(model, datasets={"放款数据": loan_data}, feature_names=features, n_jobs=1, **params)
    summary = report.summary()
    overall = bad_rate_overall(loan_data, **params)
    by_date = bad_rate_trend(loan_data, "放款时间", **params)
    category_table = feature_bin_stats(loan_data, "商品类别", method="quantile", margins=True, n_jobs=1, **params)
    rule_table = Rule("放款金额 >= 0", n_jobs=1).report(loan_data, margins=True, **params)
    days = loan_data["MOB1"]
    for dpd in [7, 3, 0]:
        bad = {">": days > dpd, ">=": days >= dpd, "<": days < dpd, "<=": days <= dpd}[op]
        valid = pd.Series(True, index=days.index)
        if op == ">":
            valid = ~((days > 0) & (days <= dpd))
        elif op == ">=":
            valid = ~((days > 0) & (days < dpd))
        count, bad_count = int(valid.sum()), int((valid & bad).sum())
        label = f"MOB1{op}{dpd}"
        display = f"MOB1@{dpd}" if op == ">" else label
        assert summary.loc[display, ("样本数", "放款数据")] == count
        assert summary.loc[display, ("坏样本率", "放款数据")] == pytest.approx(bad_count / count)
        row = overall.loc[overall["标签"] == label].iloc[0]
        assert row["样本总数"] == count
        assert row["坏样本数"] == bad_count
        dated = valid & loan_data["放款时间"].notna()
        assert by_date[label]["样本数"].sum() == dated.sum()
        assert by_date[label]["坏样本数"].sum() == (dated & bad).sum()
        bin_label = f"MOB1_{dpd}+" if op == ">" else label
        rule_label = f"MOB1 {dpd}+" if op == ">" else label
        for table, target_label in [(category_table, bin_label), (rule_table, rule_label)]:
            total = table.iloc[-1]
            group = target_label if op in (">", ">=") else "分箱详情"
            assert total[(group, "样本总数")] == count
            assert total[(target_label, "坏样本数")] == bad_count
