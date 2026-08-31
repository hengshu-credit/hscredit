"""分析报告统一并行接口与确定性回归测试。"""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import hscredit.report.feature_analyzer as feature_analyzer_module
from hscredit.core.rules import Rule
from hscredit.exceptions import ParallelExecutionError, ValidationError
from hscredit.excel import ExcelWriter
from hscredit.report import (
    OverduePredictor,
    ReferenceDataProvider,
    SwapAnalyzer,
    ModelReport,
    QuickModelReport,
    auto_model_report,
    compare_models,
    auto_feature_analysis,
    create_swap_dataset,
    create_swap_dataset_from_rules,
    feature_bin_stats,
    feature_binning_summary,
    feature_efficiency_analysis,
    feature_group_binning_summary,
    multi_label_rule_analysis,
    overdue_prediction_report,
    population_drift,
    rule_group_compare,
    rule_group_hit_table,
    rule_report_table,
    rule_swap_analysis,
    rule_target_analysis,
    rule_target_table,
    ruleset_analysis,
    swap_analysis,
    swap_out_report,
)
from hscredit.report._sample_stats import build_group_distribution_table, build_sample_stats_table
from hscredit.report.swap_analysis import SwapRiskConfig


COMMON = ("n_jobs", "parallel_backend", "parallel_config")


class _CountingProbabilityModel:
    """可序列化、预测结果确定的报告测试模型。"""

    classes_ = np.array([0, 1])

    def __init__(self, fail_after=None):
        self.feature_names_in_ = np.asarray(["f0", "f1"])
        self.proba_calls = 0
        self.predict_calls = 0
        self.fail_after = fail_after

    def predict_proba(self, X):
        self.proba_calls += 1
        if self.fail_after is not None and self.proba_calls > self.fail_after:
            raise RuntimeError("预测阶段注入失败")
        values = np.asarray(X[self.feature_names_in_], dtype=float)
        logits = values[:, 0] * 0.7 - values[:, 1] * 0.2
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        self.predict_calls += 1
        values = np.asarray(X[self.feature_names_in_], dtype=float)
        return (values[:, 0] * 0.7 - values[:, 1] * 0.2 > 0).astype(int)

    def get_feature_importances(self):
        return pd.Series([0.7, 0.3], index=self.feature_names_in_)


def _task11_datasets():
    data = {}
    for position, name in enumerate(("train", "test", "oot")):
        offset = position * 0.15
        frame = pd.DataFrame(
            {
                "f0": np.linspace(-2 + offset, 2 + offset, 80),
                "f1": np.tile([-1.0, 0.0, 1.0, 2.0], 20),
                "target": np.tile([0, 0, 1, 1], 20),
                "MOB1": np.tile([0, 3, 7, 15], 20),
                "amount": np.arange(80, dtype=float) + 1000,
                "group": np.tile(["A", "B"], 40),
                "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            }
        )
        data[name] = frame
    return data


@pytest.fixture
def report_data():
    rng = np.random.default_rng(42)
    size = 120
    return pd.DataFrame(
        {
            "score": rng.normal(600, 60, size),
            "multi": rng.integers(0, 8, size),
            "target": rng.integers(0, 2, size),
            "MOB1": rng.choice([0, 1, 3, 7, 15], size=size),
            "amount": rng.uniform(500, 5000, size),
            "group": rng.choice(["A", "B", "C"], size=size),
            "date": pd.date_range("2025-01-01", periods=size, freq="D"),
        }
    )


@pytest.mark.parametrize(
    "entry",
    [
        feature_bin_stats,
        feature_binning_summary,
        feature_group_binning_summary,
        feature_efficiency_analysis,
        auto_feature_analysis,
        ruleset_analysis,
        multi_label_rule_analysis,
        rule_swap_analysis,
        rule_report_table,
        rule_target_analysis,
        rule_target_table,
        rule_group_hit_table,
        rule_group_compare,
        swap_out_report,
        create_swap_dataset,
        create_swap_dataset_from_rules,
        swap_analysis,
        overdue_prediction_report,
        population_drift,
        build_sample_stats_table,
        build_group_distribution_table,
    ],
)
def test_task10_public_functions_expose_common_parallel_parameters(entry):
    signature = inspect.signature(entry)
    for name in COMMON:
        assert name in signature.parameters
    assert signature.parameters["n_jobs"].default == -1
    assert signature.parameters["parallel_backend"].default is None
    assert signature.parameters["parallel_config"].default is None


@pytest.mark.parametrize("cls", [ReferenceDataProvider, SwapAnalyzer, OverduePredictor])
def test_task10_estimators_expose_common_parallel_parameters(cls):
    signature = inspect.signature(cls)
    for name in COMMON:
        assert name in signature.parameters


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_feature_reports_parallel_match_serial(report_data, backend):
    kwargs = dict(
        data=report_data,
        feature=["score", "multi"],
        methods=["quantile", "uniform"],
        target="target",
        max_n_bins=4,
        random_state=7,
    )
    serial_tables, serial_summary = feature_binning_summary(n_jobs=1, **kwargs)
    parallel_tables, parallel_summary = feature_binning_summary(
        n_jobs=2, parallel_backend=backend, **kwargs
    )
    assert list(parallel_tables) == list(serial_tables)
    for feature in serial_tables:
        assert list(parallel_tables[feature]) == list(serial_tables[feature])
        for method in serial_tables[feature]:
            pd.testing.assert_frame_equal(
                parallel_tables[feature][method], serial_tables[feature][method]
            )
    pd.testing.assert_frame_equal(parallel_summary, serial_summary)

    serial_stats = feature_bin_stats(
        report_data, ["score", "multi"], target="target", method="quantile", n_jobs=1
    )
    parallel_stats = feature_bin_stats(
        report_data,
        ["score", "multi"],
        target="target",
        method="quantile",
        n_jobs=2,
        parallel_backend=backend,
    )
    pd.testing.assert_frame_equal(parallel_stats, serial_stats)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_reports_parallel_match_serial(report_data, backend):
    rules = [Rule("score < 600", name="低分"), Rule("multi > 3", name="多头")]
    kwargs = dict(
        datasets=report_data,
        rules=rules,
        overdue="MOB1",
        dpds=[7, 3, 0],
        amount="amount",
    )
    serial = ruleset_analysis(n_jobs=1, **kwargs)
    parallel = ruleset_analysis(n_jobs=2, parallel_backend=backend, **kwargs)
    pd.testing.assert_frame_equal(parallel, serial)

    serial_group = rule_group_compare(
        report_data,
        rules[0],
        group_col="group",
        overdue="MOB1",
        dpds=[7, 3, 0],
        amount="amount",
        n_jobs=1,
    )
    parallel_group = rule_group_compare(
        report_data,
        rules[0],
        group_col="group",
        overdue="MOB1",
        dpds=[7, 3, 0],
        amount="amount",
        n_jobs=2,
        parallel_backend=backend,
    )
    pd.testing.assert_frame_equal(parallel_group, serial_group)


def _swap_inputs():
    reference = pd.DataFrame(
        {
            "score": np.arange(480, 560),
            "t1": [0, 1] * 40,
            "t2": [0, 0, 1, 1] * 20,
            "amount": np.arange(80) + 100,
        }
    )
    swap = pd.DataFrame(
        {
            "score": np.arange(500, 520),
            "swap_type": ["in-in", "in-out", "out-in", "out-out"] * 5,
            "amount": np.arange(20) + 100,
        }
    )
    return reference, swap


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_swap_analyzer_parallel_match_and_is_transactional(backend, monkeypatch):
    reference, swap = _swap_inputs()
    cfg = SwapRiskConfig(score_col="score", amount_col="amount", targets=["t1", "t2"])
    provider = ReferenceDataProvider(
        score_col="score", target_cols=["t1", "t2"], amount_col="amount", n_jobs=2,
        parallel_backend=backend,
    ).fit(reference)
    serial = SwapAnalyzer(cfg, provider, n_jobs=1).analyze(swap)
    analyzer = SwapAnalyzer(cfg, provider, n_jobs=2, parallel_backend=backend)
    parallel = analyzer.analyze(swap)
    pd.testing.assert_frame_equal(parallel.get_detail_report(), serial.get_detail_report())
    pd.testing.assert_frame_equal(parallel.summary_report_count, serial.summary_report_count)

    old_result = analyzer.result
    monkeypatch.setattr(provider, "predict_bad_rate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(Exception, match="boom"):
        analyzer.analyze(swap)
    assert analyzer.result is old_result


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_overdue_predictor_parallel_clone_pickle_and_parity(report_data, backend):
    config = {"batch_size": 1}
    predictor = OverduePredictor(
        feature="score",
        overdue="MOB1",
        dpds=[7, 3, 0],
        rules=[560, 620, 680],
        n_jobs=2,
        parallel_backend=backend,
        parallel_config=config,
    )
    assert predictor.parallel_config is config
    cloned = clone(predictor)
    assert cloned.get_params()["n_jobs"] == 2
    restored = pickle.loads(pickle.dumps(predictor))
    assert restored.parallel_backend == backend

    serial = clone(predictor).set_params(n_jobs=1, parallel_backend=None).fit(report_data)
    parallel = predictor.fit(report_data)
    pd.testing.assert_frame_equal(parallel.bin_table_, serial.bin_table_)
    pd.testing.assert_frame_equal(parallel.transform(report_data[["score"]]), serial.transform(report_data[["score"]]))
    assert parallel.parallel_config is config


def test_overdue_predictor_del_grey_uses_target_specific_totals():
    """逾期率预测器应显式透传 del_grey，并保留各 DPD 的独立样本基数。"""
    data = pd.DataFrame(
        {
            "score": np.arange(12, dtype=float),
            "MOB1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    predictor = OverduePredictor(
        feature="score",
        overdue="MOB1",
        dpds=[0, 3],
        rules=[5.5],
        del_grey=True,
        n_jobs=1,
    ).fit(data)

    total = predictor.bin_table_.loc[
        predictor.bin_table_[("分箱详情", "分箱标签")] == "合计"
    ].iloc[0]
    assert total[("MOB1_0+", "样本总数")] == 12
    assert total[("MOB1_3+", "样本总数")] == 9


def test_population_drift_parallel_excel_layout_matches_serial(report_data, tmp_path):
    actual = report_data.copy()
    actual["score"] += 10
    serial_path = tmp_path / "serial.xlsx"
    parallel_path = tmp_path / "parallel.xlsx"
    population_drift(
        report_data,
        actual,
        ["score", "multi"],
        target_col="target",
        score_col="score",
        output=str(serial_path),
        n_jobs=1,
    )
    population_drift(
        report_data,
        actual,
        ["score", "multi"],
        target_col="target",
        score_col="score",
        output=str(parallel_path),
        n_jobs=2,
        parallel_backend="threading",
    )
    serial = pd.ExcelFile(serial_path)
    parallel = pd.ExcelFile(parallel_path)
    assert parallel.sheet_names == serial.sheet_names
    for sheet in serial.sheet_names:
        pd.testing.assert_frame_equal(
            pd.read_excel(parallel_path, sheet_name=sheet, header=None),
            pd.read_excel(serial_path, sheet_name=sheet, header=None),
        )


def test_overdue_predictor_fit_is_transactional_and_successful_refit_is_fresh(report_data):
    predictor = OverduePredictor(
        feature="score", target="target", rules=[560, 620, 680], n_jobs=2,
    )
    with pytest.raises(ValueError, match="缺少目标变量字段"):
        predictor.fit(report_data.drop(columns="target"))
    for name in ("feature_names_in_", "target_names_", "bin_table_", "bin_rates_", "splits_"):
        assert not hasattr(predictor, name)

    predictor.fit(report_data)
    old_table = predictor.bin_table_.copy(deep=True)
    with pytest.raises(ValueError, match="缺少目标变量字段"):
        predictor.fit(report_data.drop(columns="target"))
    pd.testing.assert_frame_equal(predictor.bin_table_, old_table)

    predictor.set_coefficients(1.2)
    predictor.fit(report_data.assign(target=1 - report_data["target"]))
    fresh = OverduePredictor(
        feature="score", target="target", rules=[560, 620, 680], coefficients=1.2, n_jobs=2,
    ).fit(report_data.assign(target=1 - report_data["target"]))
    pd.testing.assert_frame_equal(predictor.bin_table_, fresh.bin_table_)
    assert not hasattr(predictor, "coefficients_")


def test_reference_provider_refit_failure_preserves_old_state(monkeypatch):
    reference, _ = _swap_inputs()
    provider = ReferenceDataProvider(score_col="score", target_cols=["t1", "t2"], n_jobs=2).fit(reference)
    old_bins = list(provider.bins)
    old_stats = pickle.loads(pickle.dumps(provider.bin_stats))

    import importlib

    module = importlib.import_module("hscredit.report.swap_analysis")

    original = module._reference_target_stats

    def fail_second(task):
        if task[1] == "t2":
            raise RuntimeError("target failed")
        return original(task)

    monkeypatch.setattr(module, "_reference_target_stats", fail_second)
    with pytest.raises(Exception, match="target failed"):
        provider.fit(reference)
    assert provider.bins == old_bins
    assert provider.bin_stats == old_stats


def test_single_and_empty_report_tasks_do_not_create_backend(report_data, monkeypatch):
    import hscredit.utils.parallel as parallel_module

    def forbidden(*args, **kwargs):
        raise AssertionError("单任务不应创建 joblib backend")

    monkeypatch.setattr(parallel_module, "_create_joblib_parallel", forbidden)
    feature_binning_summary(
        report_data, "score", methods="quantile", target="target", n_jobs=8,
    )
    build_sample_stats_table([], [], ["target"], n_jobs=8)
    OverduePredictor(
        "score", target="target", rules=[560, 620], n_jobs=8,
    ).fit(report_data).transform(report_data[["score"]])


def test_feature_binning_summary_preserves_explicit_child_parallel_config(report_data, monkeypatch):
    """方法级显式 child 配置不得被报告层继承默认值覆盖。"""
    captured = []
    original = feature_analyzer_module.parallel_execute

    def recording_execute(function, tasks, **kwargs):
        task_list = list(tasks)
        captured.extend(task[2] for task in task_list)
        assert "workload" in kwargs
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(feature_analyzer_module, "parallel_execute", recording_execute)
    child_config = {"batch_size": 1}
    feature_binning_summary(
        report_data,
        "score",
        methods="quantile",
        target="target",
        n_jobs=1,
        bin_params={
            "quantile": {
                "n_jobs": 3,
                "parallel_backend": "threading",
                "parallel_config": child_config,
            }
        },
    )

    assert len(captured) == 1
    assert captured[0]["n_jobs"] == 3
    assert captured[0]["parallel_backend"] == "threading"
    assert captured[0]["parallel_config"] is child_config


def test_rule_group_compare_declares_real_nested_children(report_data, monkeypatch):
    import hscredit.report.rule_strategy as module

    original = module.parallel_execute
    nested = []

    def capture(*args, **kwargs):
        nested.append(kwargs.get("has_parallel_children", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    rule_group_compare(
        report_data,
        Rule("score < 600"),
        group_col="group",
        overdue="MOB1",
        dpds=[7, 3, 0],
        n_jobs=4,
        parallel_backend="threading",
    )
    assert nested == [True]


def _simple_bin_table(data, feature):
    return feature_bin_stats(
        data,
        feature,
        target="target",
        method="quantile",
        max_n_bins=3,
        margins=True,
        n_jobs=1,
    )


def _workbook_matrix(writer):
    return {
        sheet: [
            [writer.workbook[sheet].cell(row=row, column=col).value for col in range(1, writer.workbook[sheet].max_column + 1)]
            for row in range(1, writer.workbook[sheet].max_row + 1)
        ]
        for sheet in writer.workbook.sheetnames
    }


def test_auto_feature_analysis_uses_ordered_shared_feature_executor(report_data, tmp_path, monkeypatch):
    import hscredit.report.feature_analyzer as module

    original = module.parallel_execute
    labels_seen = []

    def capture(function, tasks, **kwargs):
        labels_seen.append(list(kwargs.get("task_labels", [])))
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    writer = ExcelWriter(system="windows")
    auto_feature_analysis(
        report_data,
        features=["score", "multi"],
        target="target",
        pictures=[],
        output_dir=str(tmp_path / "images"),
        excel_writer=writer,
        n_jobs=2,
        parallel_backend="threading",
    )
    assert ["score", "multi"] in labels_seen


def test_auto_feature_analysis_compute_failure_has_zero_external_side_effects(report_data, tmp_path, monkeypatch):
    import hscredit.report.feature_analyzer as module

    original = module.feature_bin_stats

    def fail_second(data, feature, **kwargs):
        if feature == "multi":
            raise RuntimeError("feature failed")
        return original(data, feature, **kwargs)

    monkeypatch.setattr(module, "feature_bin_stats", fail_second)
    writer = ExcelWriter(system="windows")
    before = _workbook_matrix(writer)
    image_dir = tmp_path / "images"

    with pytest.raises(ParallelExecutionError, match="feature failed") as exc_info:
        auto_feature_analysis(
            report_data,
            features=["score", "multi"],
            target="target",
            pictures=[],
            output_dir=str(image_dir),
            excel_writer=writer,
            n_jobs=2,
            parallel_backend="threading",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert _workbook_matrix(writer) == before
    assert not image_dir.exists()


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_auto_feature_analysis_excel_layout_matches_serial(report_data, tmp_path, backend):
    serial_writer = ExcelWriter(system="windows")
    parallel_writer = ExcelWriter(system="windows")
    common = dict(
        data=report_data,
        features=["score", "multi"],
        target="target",
        pictures=[],
        output_dir=str(tmp_path / backend),
        bin_params={"method": "quantile", "max_n_bins": 4},
    )
    auto_feature_analysis(excel_writer=serial_writer, n_jobs=1, **common)
    auto_feature_analysis(
        excel_writer=parallel_writer,
        n_jobs=2,
        parallel_backend=backend,
        **common,
    )
    assert _workbook_matrix(parallel_writer) == _workbook_matrix(serial_writer)


def test_auto_feature_analysis_propagates_sample_stats_parallel_config(report_data, tmp_path, monkeypatch):
    import hscredit.report.feature_analyzer as module

    sample_original = module.build_sample_stats_table
    group_original = module.build_group_distribution_table
    calls = []

    def sample_capture(*args, **kwargs):
        calls.append(("sample", kwargs.get("n_jobs"), kwargs.get("parallel_backend"), kwargs.get("parallel_config")))
        return sample_original(*args, **kwargs)

    def group_capture(*args, **kwargs):
        calls.append(("group", kwargs.get("n_jobs"), kwargs.get("parallel_backend"), kwargs.get("parallel_config")))
        return group_original(*args, **kwargs)

    monkeypatch.setattr(module, "build_sample_stats_table", sample_capture)
    monkeypatch.setattr(module, "build_group_distribution_table", group_capture)
    monkeypatch.setattr(module, "distribution_plot", lambda *args, **kwargs: pd.DataFrame())
    config = {"batch_size": 1}
    writer = ExcelWriter(system="windows")
    writer.insert_pic2sheet = lambda worksheet, fig, insert_space, figsize=(600, 250): (
        insert_space[0] + 1,
        insert_space[1] + 1,
    )
    auto_feature_analysis(
        report_data,
        features=["score"],
        target="target",
        date="date",
        pictures=[],
        output_dir=str(tmp_path / "images"),
        excel_writer=writer,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=config,
    )
    assert calls == [
        ("sample", 2, "threading", config),
        ("group", 2, "threading", config),
    ]


def test_auto_feature_analysis_n_jobs_one_never_creates_backend(report_data, tmp_path, monkeypatch):
    import hscredit.utils.parallel as parallel_module

    monkeypatch.setattr(
        parallel_module,
        "_create_joblib_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("n_jobs=1 不应创建后端")),
    )
    auto_feature_analysis(
        report_data,
        features=["score", "multi"],
        target="target",
        pictures=[],
        output_dir=str(tmp_path / "images"),
        excel_writer=ExcelWriter(system="windows"),
        n_jobs=1,
    )


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_population_drift_month_windows_are_ordered_and_parallel_exact(tmp_path, backend):
    expected = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "y": [6, 5, 4, 3, 2, 1], "target": [0, 0, 1, 0, 1, 1]}
    )
    actual = pd.DataFrame(
        {
            "x": [2, 5, 3, 6, 1, 4],
            "y": [5, 2, 4, 1, 6, 3],
            "target": [0, 1, 0, 1, 0, 1],
            "放款时间": ["2025-02-02", "2025-01-03", "2025-02-10", "2025-01-20", "2025-01-01", "2025-02-20"],
        }
    )
    serial_path = tmp_path / "window_serial.xlsx"
    parallel_path = tmp_path / f"window_{backend}.xlsx"
    kwargs = dict(expected=expected, actual=actual, features=["x", "y"], target_col="target", date_col="放款时间")
    population_drift(output=str(serial_path), n_jobs=1, **kwargs)
    population_drift(output=str(parallel_path), n_jobs=2, parallel_backend=backend, **kwargs)

    serial = pd.ExcelFile(serial_path)
    parallel = pd.ExcelFile(parallel_path)
    assert parallel.sheet_names == serial.sheet_names
    for sheet in serial.sheet_names:
        pd.testing.assert_frame_equal(
            pd.read_excel(parallel_path, sheet_name=sheet, header=None),
            pd.read_excel(serial_path, sheet_name=sheet, header=None),
        )
    overview = pd.read_excel(serial_path, sheet_name="PSI总览", header=1)
    assert list(zip(overview["特征名"], overview["时间窗口"])) == [
        ("x", "2025-01"), ("x", "2025-02"), ("y", "2025-01"), ("y", "2025-02")
    ]


@pytest.mark.parametrize(
    "actual, message",
    [
        (pd.DataFrame({"x": [1, 2]}), "缺少日期列"),
        (pd.DataFrame({"x": [1, 2], "date": ["2025-01-01", "不是日期"]}), "包含无效日期"),
    ],
)
def test_population_drift_date_col_validation_is_chinese(tmp_path, actual, message):
    expected = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValidationError, match=message):
        population_drift(
            expected,
            actual,
            ["x"],
            date_col="date",
            output=str(tmp_path / "invalid.xlsx"),
        )


def _swap_parallel_case():
    data = pd.DataFrame(
        {
            "score_a": [510, 540, 570, 600, 630, 660, 690, 720],
            "score_b": [720, 690, 660, 630, 600, 570, 540, 510],
            "x": [0, 1, 2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )
    tables = {"a": _simple_bin_table(data, "score_a"), "b": _simple_bin_table(data, "score_b")}
    return data, tables


def test_rule_swap_independent_ready_bins_use_ordered_executor(monkeypatch):
    import hscredit.report.rule_analysis as module
    from hscredit.utils.parallel import parallel_execute as shared_parallel_execute

    data, tables = _swap_parallel_case()
    original = getattr(module, "parallel_execute", shared_parallel_execute)
    calls = []

    def capture(function, tasks, **kwargs):
        calls.append((function.__name__, list(kwargs.get("task_labels", []))))
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture, raising=False)
    rule_swap_analysis(
        data,
        score={"a": "score_a", "b": "score_b"},
        bin_table=tables,
        rules_in=[Rule("x >= 2", name="置入1"), Rule("x >= 5", name="置入2")],
        target="target",
        n_jobs=2,
        parallel_backend="threading",
    )
    assert any(name == "_swap_score_prediction_call" and labels == ["a", "b"] for name, labels in calls)
    assert any(name == "_swap_rule_mask_call" and labels == ["置入1", "置入2"] for name, labels in calls)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_swap_multi_score_parallel_matches_serial(backend):
    data, tables = _swap_parallel_case()
    kwargs = dict(
        data=data,
        score={"a": "score_a", "b": "score_b"},
        bin_table=tables,
        rules_base=[Rule("x == 0", name="基准")],
        rules_out=[Rule("x == 3", name="置出1"), Rule("x == 4", name="置出2")],
        rules_in=[Rule("x >= 5", name="置入1"), Rule("x >= 7", name="置入2")],
        target="target",
    )
    serial = rule_swap_analysis(n_jobs=1, **kwargs)
    parallel = rule_swap_analysis(n_jobs=2, parallel_backend=backend, **kwargs)
    pd.testing.assert_frame_equal(parallel["swap_pipeline"], serial["swap_pipeline"])
    pd.testing.assert_frame_equal(parallel["swap_result"], serial["swap_result"])


def test_rule_swap_independent_worker_failure_is_fail_fast(monkeypatch):
    import hscredit.report.rule_analysis as module

    data, tables = _swap_parallel_case()
    original = getattr(module, "_swap_rule_mask_call", None)

    def fail_second(task):
        if task[0] == "置入2":
            raise RuntimeError("rule failed")
        return original(task)

    monkeypatch.setattr(module, "_swap_rule_mask_call", fail_second, raising=False)
    with pytest.raises(ParallelExecutionError, match="rule failed") as exc_info:
        rule_swap_analysis(
            data,
            score={"a": "score_a", "b": "score_b"},
            bin_table=tables,
            rules_in=[Rule("x >= 2", name="置入1"), Rule("x >= 5", name="置入2")],
            target="target",
            n_jobs=2,
            parallel_backend="threading",
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_rule_swap_sequential_mode_is_disjoint_and_skips_rule_executor(monkeypatch):
    import hscredit.report.rule_analysis as module
    from hscredit.utils.parallel import parallel_execute as shared_parallel_execute

    data, tables = _swap_parallel_case()
    original = getattr(module, "parallel_execute", shared_parallel_execute)
    worker_names = []

    def capture(function, tasks, **kwargs):
        worker_names.append(function.__name__)
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture, raising=False)
    result = rule_swap_analysis(
        data,
        score={"a": "score_a", "b": "score_b"},
        bin_table=tables,
        rules_in=[Rule("x >= 2", name="置入1"), Rule("x >= 5", name="置入2")],
        target="target",
        rule_analysis_mode="sequential",
        n_jobs=2,
        parallel_backend="threading",
    )["swap_pipeline"]
    rows = result[result["规则分类"] == "OUT-IN置入"].set_index("指标名称")
    assert rows.loc["置入1", "样本总数"] == 6
    assert rows.loc["置入2", "样本总数"] == 0
    assert "_swap_rule_mask_call" not in worker_names


@pytest.mark.parametrize(
    "groups, dpds, expected",
    [(["A"] * 12, [7, 3], False), (["A", "B"] * 6, [7], False), (["A", "B"] * 6, [7, 3], True)],
)
def test_rule_group_nested_flag_requires_outer_and_inner_parallelism(monkeypatch, groups, dpds, expected):
    import hscredit.report.rule_strategy as module

    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": groups}
    )
    original = module.parallel_execute
    captured = []

    def capture(function, tasks, **kwargs):
        if function.__name__ == "_rule_group_report_call":
            captured.append(kwargs.get("has_parallel_children"))
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    rule_group_compare(
        data,
        Rule("score < 6"),
        group_col="group",
        overdue="MOB1",
        dpds=dpds,
        n_jobs=1,
    )
    assert captured == [expected]


def test_swap_out_nested_flags_require_multiple_outer_sections(monkeypatch):
    import hscredit.report.rule_strategy as module

    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A"] * 12}
    )
    original = module.parallel_execute
    captured = {}

    def capture(function, tasks, **kwargs):
        if function.__name__ in {"_swap_rule_report_call", "_swap_stability_call"}:
            captured[function.__name__] = kwargs.get("has_parallel_children")
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    swap_out_report(
        data,
        rules=[Rule("score < 6", name="低分")],
        features=[],
        group_col="group",
        overdue="MOB1",
        dpds=[7, 3],
        n_jobs=1,
    )
    assert captured == {"_swap_rule_report_call": False, "_swap_stability_call": False}


def _capture_rule_strategy_parallel(monkeypatch):
    import hscredit.report.rule_strategy as module

    original = module.parallel_execute
    captured = []

    def capture(function, tasks, **kwargs):
        task_list = list(tasks)
        name = function.__name__
        if name in {"_rule_group_report_call", "_swap_rule_report_call", "_swap_stability_call"}:
            child_jobs = [task[-1].get("n_jobs") for task in task_list]
            captured.append(
                {
                    "name": name,
                    "outer_jobs": kwargs.get("n_jobs"),
                    "child_jobs": child_jobs,
                    "has_children": kwargs.get("has_parallel_children"),
                    "task_count": len(task_list),
                }
            )
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    return captured


@pytest.mark.parametrize("n_jobs", [1, 2, 4, 0.5, -1])
def test_rule_group_budget_uses_resolved_total_and_real_inner_tasks(monkeypatch, n_jobs):
    from hscredit.utils.parallel import resolve_n_jobs, split_parallel_budget

    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame(
        {
            "score": np.arange(12),
            "MOB1": [0, 1, 3, 7, 15, 0] * 2,
            "group": ["A", "B"] * 6,
        }
    )
    rule_group_compare(
        data,
        Rule("score < 6"),
        group_col="group",
        overdue="MOB1",
        dpds=[7, 3],
        n_jobs=n_jobs,
        parallel_backend="threading",
    )
    total = resolve_n_jobs(n_jobs) or 1
    outer, child_budget = split_parallel_budget(total, 2, True)
    child = min(child_budget, 2)
    call = next(item for item in captured if item["name"] == "_rule_group_report_call")
    assert call == {
        "name": "_rule_group_report_call",
        "outer_jobs": outer,
        "child_jobs": [child, child],
        "has_children": True,
        "task_count": 2,
    }


@pytest.mark.parametrize(
    "dpds",
    [[7, 7], [None, 0], None],
)
def test_rule_group_duplicate_or_default_dpds_have_one_real_child(monkeypatch, dpds):
    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A", "B"] * 6}
    )
    rule_group_compare(
        data,
        Rule("score < 6"),
        group_col="group",
        overdue="MOB1",
        dpds=dpds,
        n_jobs=4,
        parallel_backend="threading",
    )
    call = next(item for item in captured if item["name"] == "_rule_group_report_call")
    assert call["outer_jobs"] == 2
    assert call["child_jobs"] == [1, 1]
    assert call["has_children"] is False


def test_rule_group_single_outer_gives_full_budget_to_deduplicated_inner(monkeypatch):
    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A"] * 12}
    )
    rule_group_compare(
        data,
        Rule("score < 6"),
        group_col="group",
        overdue=["MOB1"],
        dpds=[7, 3, 3],
        n_jobs=4,
        parallel_backend="threading",
    )
    call = next(item for item in captured if item["name"] == "_rule_group_report_call")
    assert call["outer_jobs"] == 1
    assert call["child_jobs"] == [2]
    assert call["has_children"] is False


def test_rule_group_budget_respects_active_parent(monkeypatch):
    from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET

    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A", "B"] * 6}
    )
    token = _ACTIVE_BUDGET.set(ParallelBudget(3, 1))
    try:
        rule_group_compare(
            data,
            Rule("score < 6"),
            group_col="group",
            overdue="MOB1",
            dpds=[7, 3],
            n_jobs=4,
            parallel_backend="threading",
        )
    finally:
        _ACTIVE_BUDGET.reset(token)
    call = next(item for item in captured if item["name"] == "_rule_group_report_call")
    assert call["outer_jobs"] == 2
    assert call["child_jobs"] == [1, 1]


@pytest.mark.parametrize("unique_dpds, expected_flag, expected_outer, expected_child", [
    ([7, 7], False, 3, 1),
    ([None, 0], False, 3, 1),
    ([7, 3], True, 2, 2),
])
def test_swap_out_rule_report_budget_uses_real_deduplicated_targets(
    monkeypatch, unique_dpds, expected_flag, expected_outer, expected_child
):
    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame({"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2})
    swap_out_report(
        data,
        rules=[Rule("score < 4", name="规则1"), Rule("score > 8", name="规则2")],
        features=[],
        overdue=["MOB1"],
        dpds=unique_dpds,
        n_jobs=4,
        parallel_backend="threading",
    )
    call = next(item for item in captured if item["name"] == "_swap_rule_report_call")
    assert call["outer_jobs"] == expected_outer
    assert call["child_jobs"] == [expected_child] * 3
    assert call["has_children"] is expected_flag


@pytest.mark.parametrize("unique_dpds, expected_flag, expected_outer, expected_child", [
    ([7, 7], False, 3, 1),
    ([None, 0], False, 3, 1),
    ([7, 3], True, 2, 2),
])
def test_swap_out_stability_budget_uses_real_deduplicated_targets(
    monkeypatch, unique_dpds, expected_flag, expected_outer, expected_child
):
    captured = _capture_rule_strategy_parallel(monkeypatch)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A"] * 12}
    )
    swap_out_report(
        data,
        rules=[Rule("score < 4", name="规则1"), Rule("score > 8", name="规则2")],
        features=[],
        group_col="group",
        overdue="MOB1",
        dpds=unique_dpds,
        n_jobs=4,
        parallel_backend="threading",
    )
    call = next(item for item in captured if item["name"] == "_swap_stability_call")
    assert call["outer_jobs"] == expected_outer
    assert call["child_jobs"] == [expected_child] * 3
    assert call["has_children"] is expected_flag


def _capture_executing_rule_configs(monkeypatch):
    from hscredit.utils.parallel import _ACTIVE_BUDGET, resolve_n_jobs

    original = Rule._parallel_execute
    captured = []
    kept_alive = []

    def capture(self, function, tasks, **kwargs):
        task_list = list(tasks)
        active = _ACTIVE_BUDGET.get()
        kept_alive.append(self)
        captured.append(
            {
                "rule": self,
                "rule_id": id(self),
                "n_jobs": self.n_jobs,
                "backend": self.parallel_backend,
                "config": self.parallel_config,
                "active": None if active is None else active.available,
                "resolved": resolve_n_jobs(
                    self.n_jobs,
                    task_count=len(task_list),
                    available_budget=None if active is None else active.available,
                ),
            }
        )
        return original(self, function, task_list, **kwargs)

    monkeypatch.setattr(Rule, "_parallel_execute", capture)
    return captured, kept_alive


def _assert_rule_unchanged(rule, snapshot):
    assert rule.n_jobs == snapshot["n_jobs"]
    assert rule.parallel_backend == snapshot["parallel_backend"]
    assert rule.parallel_config is snapshot["parallel_config"]
    assert rule.result_ is snapshot["result_"]
    assert rule._state is snapshot["_state"]


def _rule_snapshot(rule):
    return {
        "n_jobs": rule.n_jobs,
        "parallel_backend": rule.parallel_backend,
        "parallel_config": rule.parallel_config,
        "result_": rule.result_,
        "_state": rule._state,
    }


def test_rule_group_executes_isolated_rule_copies_with_planned_config(monkeypatch):
    from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET

    captured, _ = _capture_executing_rule_configs(monkeypatch)
    original_config = {"batch_size": 9}
    call_config = {"batch_size": 1}
    rule = Rule("score < 6", name="低分", n_jobs=9, parallel_backend="loky", parallel_config=original_config)
    snapshot = _rule_snapshot(rule)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A", "B"] * 6}
    )
    token = _ACTIVE_BUDGET.set(ParallelBudget(3, 1))
    try:
        rule_group_compare(
            data,
            rule,
            group_col="group",
            overdue="MOB1",
            dpds=[7, 3],
            n_jobs=4,
            parallel_backend="threading",
            parallel_config=call_config,
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert len(captured) == 2
    assert len({item["rule_id"] for item in captured}) == 2
    assert all(item["rule"] is not rule for item in captured)
    assert all(item["n_jobs"] == 1 and item["resolved"] == 1 for item in captured)
    assert all(item["backend"] == "threading" and item["config"] is call_config for item in captured)
    assert all(item["active"] == 1 for item in captured)
    _assert_rule_unchanged(rule, snapshot)


def test_swap_rule_report_count_and_amount_use_distinct_configured_copies(monkeypatch):
    captured, _ = _capture_executing_rule_configs(monkeypatch)
    original_config = {"batch_size": 9}
    call_config = {"batch_size": 1}
    rule = Rule("score < 6", name="低分", n_jobs=9, parallel_backend="loky", parallel_config=original_config)
    snapshot = _rule_snapshot(rule)
    data = pd.DataFrame(
        {
            "score": np.arange(12),
            "MOB1": [0, 1, 3, 7, 15, 0] * 2,
            "amount": np.arange(12) + 100,
        }
    )
    swap_out_report(
        data,
        rules=[rule],
        features=[],
        overdue="MOB1",
        dpds=[7, 3],
        amount="amount",
        n_jobs=4,
        parallel_backend="threading",
        parallel_config=call_config,
    )
    assert len(captured) == 2
    assert len({item["rule_id"] for item in captured}) == 2
    assert all(item["rule"] is not rule for item in captured)
    assert all(item["n_jobs"] == 2 and item["resolved"] == 2 for item in captured)
    assert all(item["backend"] == "threading" and item["config"] is call_config for item in captured)
    _assert_rule_unchanged(rule, snapshot)


def test_swap_stability_executes_section_rules_as_isolated_planned_copies(monkeypatch):
    from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET

    captured, _ = _capture_executing_rule_configs(monkeypatch)
    call_config = {"batch_size": 1}
    rules = [
        Rule("score < 4", name="规则1", n_jobs=9, parallel_backend="loky"),
        Rule("score > 8", name="规则2", n_jobs=9, parallel_backend="loky"),
    ]
    snapshots = [_rule_snapshot(rule) for rule in rules]
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A"] * 12}
    )
    token = _ACTIVE_BUDGET.set(ParallelBudget(3, 1))
    try:
        swap_out_report(
            data,
            rules=rules,
            features=[],
            group_col="group",
            overdue="MOB1",
            dpds=[7, 3],
            n_jobs=4,
            parallel_backend="threading",
            parallel_config=call_config,
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert len(captured) == 6  # rule report 三个 section + stability 三个 section
    assert len({item["rule_id"] for item in captured}) == 6
    assert all(item["n_jobs"] == 1 and item["resolved"] == 1 for item in captured)
    assert all(item["backend"] == "threading" and item["config"] is call_config for item in captured)
    for rule, snapshot in zip(rules, snapshots):
        _assert_rule_unchanged(rule, snapshot)


def test_ruleset_analysis_sequential_reports_use_configured_copies(monkeypatch):
    captured, _ = _capture_executing_rule_configs(monkeypatch)
    call_config = {"batch_size": 1}
    rule = Rule("score < 6", name="低分", n_jobs=9, parallel_backend="loky")
    snapshot = _rule_snapshot(rule)
    data = pd.DataFrame({"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2})
    ruleset_analysis(
        data,
        [rule],
        overdue="MOB1",
        dpds=[7, 3],
        n_jobs=4,
        parallel_backend="threading",
        parallel_config=call_config,
    )
    assert len(captured) == 2
    assert len({item["rule_id"] for item in captured}) == 2
    assert all(item["n_jobs"] == 2 and item["resolved"] == 2 for item in captured)
    assert all(item["backend"] == "threading" and item["config"] is call_config for item in captured)
    _assert_rule_unchanged(rule, snapshot)


def test_configured_rule_worker_failure_preserves_direct_cause(monkeypatch):
    original = Rule.report

    def fail(self, *args, **kwargs):
        if self.name == "失败规则":
            raise RuntimeError("configured rule failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Rule, "report", fail)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A", "B"] * 6}
    )
    with pytest.raises(ParallelExecutionError, match="configured rule failed") as exc_info:
        rule_group_compare(
            data,
            Rule("score < 6", name="失败规则", n_jobs=9, parallel_backend="loky"),
            group_col="group",
            overdue="MOB1",
            dpds=[7, 3],
            n_jobs=4,
            parallel_backend="threading",
        )
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_configured_parent_rule_group_parallel_matches_serial_and_stays_unchanged(backend):
    original_config = {"batch_size": 9}
    rule = Rule("score < 6", name="低分", n_jobs=9, parallel_backend="loky", parallel_config=original_config)
    snapshot = _rule_snapshot(rule)
    data = pd.DataFrame(
        {"score": np.arange(12), "MOB1": [0, 1, 3, 7, 15, 0] * 2, "group": ["A", "B"] * 6}
    )
    kwargs = dict(data=data, rule=rule, group_col="group", overdue="MOB1", dpds=[7, 3])
    serial = rule_group_compare(n_jobs=1, **kwargs)
    parallel = rule_group_compare(
        n_jobs=4,
        parallel_backend=backend,
        parallel_config={"batch_size": 1},
        **kwargs,
    )
    pd.testing.assert_frame_equal(parallel, serial)
    _assert_rule_unchanged(rule, snapshot)


# ---------------------------------------------------------------------------
# Task 11: 模型报告与模型比较
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [ModelReport, QuickModelReport, auto_model_report, compare_models],
)
def test_task11_model_report_entries_expose_common_parallel_parameters(entry):
    signature = inspect.signature(entry)
    names = list(signature.parameters)
    expected_slice = names[-4:-1] if names[-1] == "kwargs" else names[-3:]
    assert expected_slice == list(COMMON)
    assert signature.parameters["n_jobs"].default == -1
    assert signature.parameters["parallel_backend"].default is None
    assert signature.parameters["parallel_config"].default is None


def test_model_report_validates_config_before_prediction_and_preserves_identity():
    data = _task11_datasets()
    model = _CountingProbabilityModel()
    with pytest.raises(ValidationError):
        ModelReport(
            model,
            datasets=data,
            target="target",
            parallel_config={"未知配置": True},
        )
    assert model.proba_calls == 0
    assert model.predict_calls == 0

    config = {"batch_size": 1}
    report = ModelReport(
        model,
        datasets={"train": data["train"]},
        target="target",
        n_jobs=1.0,
        parallel_backend="threading",
        parallel_config=config,
    )
    assert report.parallel_config is config
    report.summary()
    assert report.parallel_config is config
    restored = pickle.loads(pickle.dumps(report))
    assert restored.n_jobs == 1.0
    assert restored.parallel_backend == "threading"
    assert restored.parallel_config == config


def test_model_report_computes_each_dataset_prediction_once():
    model = _CountingProbabilityModel()
    report = ModelReport(
        model,
        datasets=_task11_datasets(),
        target="target",
        n_jobs=1,
    )
    assert list(report._datasets) == ["train", "test", "oot"]
    assert model.proba_calls == 3
    assert model.predict_calls == 0

    report.summary()
    report.get_metrics()
    report.get_bin_table("train", max_n_bins=4)
    report.get_feature_importance()
    report.get_features_corr()
    assert model.proba_calls == 3
    assert model.predict_calls == 0


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_model_report_parallel_tables_match_serial_exactly(backend):
    datasets = _task11_datasets()
    kwargs = dict(datasets=datasets, overdue="MOB1", dpds=[7, 3, 0], feature_names=["f0", "f1"])
    serial = ModelReport(_CountingProbabilityModel(), n_jobs=1, **kwargs)
    parallel = ModelReport(
        _CountingProbabilityModel(),
        n_jobs=2,
        parallel_backend=backend,
        parallel_config={"batch_size": 1},
        **kwargs,
    )

    pd.testing.assert_frame_equal(parallel.summary(), serial.summary(), check_exact=True)
    for label in serial._label_names:
        pd.testing.assert_frame_equal(
            parallel.get_metrics(label), serial.get_metrics(label), check_exact=True
        )
    pd.testing.assert_frame_equal(
        parallel.get_bin_table("test", max_n_bins=4, labels=parallel._label_names),
        serial.get_bin_table("test", max_n_bins=4, labels=serial._label_names),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        parallel.get_feature_importance(), serial.get_feature_importance(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        parallel.get_features_describe(), serial.get_features_describe(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        parallel.get_features_corr(), serial.get_features_corr(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        parallel._get_monthly_metrics("date"), serial._get_monthly_metrics("date"), check_exact=True
    )


def test_model_report_does_not_change_trained_model_parameters():
    frame = _task11_datasets()["train"]
    X, y = frame[["f0", "f1"]], frame["target"]
    model = LogisticRegression(C=0.75, random_state=20260811, max_iter=300).fit(X, y)
    params_before = model.get_params(deep=True).copy()
    coef_before = model.coef_.copy()
    intercept_before = model.intercept_.copy()

    report = ModelReport(
        model,
        datasets=_task11_datasets(),
        target="target",
        n_jobs=2,
        parallel_backend="threading",
    )
    report.summary()

    assert model.get_params(deep=True) == params_before
    np.testing.assert_array_equal(model.coef_, coef_before)
    np.testing.assert_array_equal(model.intercept_, intercept_before)


def test_model_report_single_dataset_does_not_create_backend(monkeypatch):
    import hscredit.utils.parallel as parallel_module

    def forbidden(*args, **kwargs):
        raise AssertionError("单任务不应创建 joblib backend")

    monkeypatch.setattr(parallel_module, "_create_joblib_parallel", forbidden)
    report = ModelReport(
        _CountingProbabilityModel(),
        datasets={"train": _task11_datasets()["train"]},
        target="target",
        n_jobs=4,
        parallel_backend="threading",
    )
    assert list(report._datasets) == ["train"]
    assert not report.summary().empty


def test_model_report_multiple_datasets_invoke_shared_executor(monkeypatch):
    import hscredit.report.model_report as module

    original = module.parallel_execute
    calls = []

    def capture(function, tasks, **kwargs):
        task_list = list(tasks)
        calls.append((function.__name__, len(task_list), kwargs.copy()))
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    ModelReport(
        _CountingProbabilityModel(),
        datasets=_task11_datasets(),
        target="target",
        n_jobs=2,
        parallel_backend="threading",
    ).summary()
    assert any(name == "_build_report_dataset" and count == 3 for name, count, _ in calls)
    assert any(name == "_binary_metric_worker" and count == 3 for name, count, _ in calls)


def test_model_report_add_dataset_is_transactional_and_invalidates_old_cache():
    datasets = _task11_datasets()
    model = _CountingProbabilityModel(fail_after=1)
    report = ModelReport(model, datasets={"训练集": datasets["train"]}, target="target", n_jobs=1)
    old_summary = report.summary()
    old_keys = list(report._datasets)
    with pytest.raises(Exception, match="预测阶段注入失败"):
        report.add_dataset("test", "测试集", datasets["test"])
    assert list(report._datasets) == old_keys
    pd.testing.assert_frame_equal(report.summary(), old_summary, check_exact=True)

    model.fail_after = None
    report.add_dataset("test", "测试集", datasets["test"])
    assert list(report._datasets) == ["训练集", "test"]
    assert list(report.summary().columns.get_level_values("数据集").unique()) == ["训练集", "测试集"]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_compare_models_parallel_matches_serial_and_keeps_mapping_order(backend):
    frame = _task11_datasets()["train"]
    X = frame[["f0", "f1"]]
    y = frame["target"]
    models = {
        "逻辑回归B": LogisticRegression(random_state=7).fit(X, y),
        "逻辑回归A": LogisticRegression(C=0.5, random_state=7).fit(X, y),
    }
    serial = compare_models(models, X, y, X.iloc[::-1], y.iloc[::-1], n_jobs=1)
    parallel = compare_models(
        models,
        X,
        y,
        X.iloc[::-1],
        y.iloc[::-1],
        n_jobs=2,
        parallel_backend=backend,
        parallel_config={"batch_size": 1},
    )
    pd.testing.assert_frame_equal(parallel, serial, check_exact=True)
    assert list(parallel.index.get_level_values("模型名称").unique()) == list(models)


def test_compare_models_marks_real_nested_parallel_boundary(monkeypatch):
    import hscredit.report.model_report as module
    import hscredit.utils.parallel as parallel_module
    from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET

    monkeypatch.setattr(parallel_module, "get_physical_cpu_count", lambda: 8)
    original = module.parallel_execute
    captured = []

    def capture(function, tasks, **kwargs):
        task_list = list(tasks)
        active = _ACTIVE_BUDGET.get()
        captured.append(
            {
                "worker": function.__name__,
                "count": len(task_list),
                "has_children": kwargs.get("has_parallel_children"),
                "active": None if active is None else active.available,
            }
        )
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    frame = _task11_datasets()["train"]
    X, y = frame[["f0", "f1"]], frame["target"]
    models = {
        "A": LogisticRegression(random_state=1).fit(X, y),
        "B": LogisticRegression(C=0.8, random_state=2).fit(X, y),
    }
    token = _ACTIVE_BUDGET.set(ParallelBudget(4, 0))
    try:
        compare_models(
            models,
            X,
            y,
            X.iloc[::-1],
            y.iloc[::-1],
            n_jobs=-1,
            parallel_backend="threading",
        )
    finally:
        _ACTIVE_BUDGET.reset(token)

    outer = next(item for item in captured if item["worker"] == "_compare_model_worker")
    children = [item for item in captured if item["worker"] == "_build_report_dataset"]
    assert outer["has_children"] is True
    assert outer["active"] == 4
    assert len(children) == 2
    assert all(item["active"] == 2 and item["count"] == 2 for item in children)


@pytest.mark.parametrize(
    "requested_n_jobs, expected_total, expected_outer, expected_child",
    [
        (None, 1, 1, 1),
        (1, 1, 1, 1),
        (1.0, 1, 1, 1),
        (2, 2, 2, 1),
        (4, 4, 2, 2),
        (0.5, 4, 2, 2),
        (-1, 7, 2, 2),
    ],
)
def test_compare_models_actual_nested_budget_matrix(
    monkeypatch, requested_n_jobs, expected_total, expected_outer, expected_child
):
    import hscredit.report.model_report as module
    import hscredit.utils.parallel as parallel_module
    from hscredit.utils.parallel import _ACTIVE_BUDGET

    monkeypatch.setattr(parallel_module, "get_physical_cpu_count", lambda: 8)
    original = module.parallel_execute
    captured = []

    def capture(function, tasks, **kwargs):
        task_list = list(tasks)
        active = _ACTIVE_BUDGET.get()
        captured.append(
            (
                function.__name__,
                len(task_list),
                kwargs.get("n_jobs"),
                kwargs.get("has_parallel_children", False),
                None if active is None else active.available,
            )
        )
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    frame = _task11_datasets()["train"]
    X, y = frame[["f0", "f1"]], frame["target"]
    models = {
        "A": LogisticRegression(random_state=1).fit(X, y),
        "B": LogisticRegression(C=0.8, random_state=2).fit(X, y),
    }
    before = [pickle.dumps(model.__dict__) for model in models.values()]
    compare_models(
        models,
        X,
        y,
        X.iloc[::-1],
        y.iloc[::-1],
        n_jobs=requested_n_jobs,
        parallel_backend="threading",
    )

    outer = next(item for item in captured if item[0] == "_compare_model_worker")
    children = [item for item in captured if item[0] == "_build_report_dataset"]
    assert outer[2:] == (expected_outer, expected_total > 1, expected_total)
    assert len(children) == 2
    assert all(item[2] == expected_child and item[4] <= expected_total for item in children)
    assert [pickle.dumps(model.__dict__) for model in models.values()] == before


def test_compare_models_actual_nested_budget_respects_active_cap(monkeypatch):
    import hscredit.report.model_report as module
    import hscredit.utils.parallel as parallel_module
    from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET

    monkeypatch.setattr(parallel_module, "get_physical_cpu_count", lambda: 8)
    original = module.parallel_execute
    captured = []

    def capture(function, tasks, **kwargs):
        task_list = list(tasks)
        active = _ACTIVE_BUDGET.get()
        captured.append((function.__name__, kwargs.get("n_jobs"), None if active is None else active.available))
        return original(function, task_list, **kwargs)

    monkeypatch.setattr(module, "parallel_execute", capture)
    frame = _task11_datasets()["train"]
    X, y = frame[["f0", "f1"]], frame["target"]
    token = _ACTIVE_BUDGET.set(ParallelBudget(3, 1))
    try:
        compare_models(
            {"A": _CountingProbabilityModel(), "B": _CountingProbabilityModel()},
            X,
            y,
            X.iloc[::-1],
            y.iloc[::-1],
            n_jobs=4,
            parallel_backend="threading",
        )
    finally:
        _ACTIVE_BUDGET.reset(token)
    outer = next(item for item in captured if item[0] == "_compare_model_worker")
    children = [item for item in captured if item[0] == "_build_report_dataset"]
    assert outer == ("_compare_model_worker", 2, 3)
    assert len(children) == 2
    assert all(item[1:] == (1, 1) for item in children)


def test_compare_models_worker_failure_is_fail_fast_with_direct_cause():
    frame = _task11_datasets()["train"]
    X, y = frame[["f0", "f1"]], frame["target"]
    models = {
        "正常模型": _CountingProbabilityModel(),
        "失败模型": _CountingProbabilityModel(fail_after=0),
    }
    with pytest.raises(ParallelExecutionError, match="失败模型") as exc_info:
        compare_models(models, X, y, n_jobs=2, parallel_backend="threading")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "预测阶段注入失败" in str(exc_info.value.__cause__)


def test_model_report_excel_computes_before_render(monkeypatch, tmp_path):
    import hscredit.excel as excel_module

    report = ModelReport(
        _CountingProbabilityModel(),
        datasets={"train": _task11_datasets()["train"]},
        target="target",
        n_jobs=1,
    )
    events = []

    def fail_compute(**kwargs):
        events.append("compute")
        raise RuntimeError("预计算失败")

    class ForbiddenWriter:
        def __init__(self, *args, **kwargs):
            events.append("render")

    monkeypatch.setattr(report, "_precompute_excel_tables", fail_compute)
    monkeypatch.setattr(excel_module, "ExcelWriter", ForbiddenWriter)
    with pytest.raises(RuntimeError, match="预计算失败"):
        report.to_excel(str(tmp_path / "should_not_exist.xlsx"), with_plots=False)
    assert events == ["compute"]


_MODEL_REPORT_CACHE_NAMES = (
    "_metrics_cache",
    "_summary_cache",
    "_importance_cache",
    "_features_describe_cache",
    "_corr_cache",
    "_bin_table_cache",
    "_feature_bin_table_cache",
    "_lift_table_cache",
    "_monthly_metrics_cache",
    "_monthly_psi_cache",
    "_features_summary_cache",
)


def _cache_snapshot(report):
    snapshot = {}
    for name in _MODEL_REPORT_CACHE_NAMES:
        value = getattr(report, name)
        if isinstance(value, dict):
            snapshot[name] = (
                value,
                {key: (item, item.copy(deep=True)) for key, item in value.items()},
            )
        elif isinstance(value, pd.DataFrame):
            snapshot[name] = (value, value.copy(deep=True))
        else:
            snapshot[name] = (value, value)
    return snapshot


def _assert_cache_snapshot(report, snapshot):
    for name, expected in snapshot.items():
        current = getattr(report, name)
        assert current is expected[0], name
        if isinstance(current, dict):
            assert list(current) == list(expected[1]), name
            for key, (old_value, old_copy) in expected[1].items():
                assert current[key] is old_value, (name, key)
                pd.testing.assert_frame_equal(current[key], old_copy, check_exact=True)
        elif isinstance(current, pd.DataFrame):
            pd.testing.assert_frame_equal(current, expected[1], check_exact=True)


def _warm_report_caches(report):
    report.summary()
    report.get_metrics()
    report.get_bin_table("train", max_n_bins=4)
    report.get_feature_importance()
    report.get_features_describe()
    report.get_features_corr()
    report.get_feature_bin_table("f0", "train", max_n_bins=4)
    report._get_top_n_lift_table()
    report._get_monthly_metrics("date")
    report._get_monthly_psi_matrix("date")
    report._get_features_summary()


@pytest.mark.parametrize("warm", [False, True])
def test_model_report_late_excel_save_failure_restores_all_cache_objects(monkeypatch, tmp_path, warm):
    from hscredit.excel import ExcelWriter

    report = ModelReport(
        _CountingProbabilityModel(),
        datasets={key: value for key, value in list(_task11_datasets().items())[:2]},
        target="target",
        n_jobs=2,
        parallel_backend="threading",
    )
    if warm:
        _warm_report_caches(report)
    snapshot = _cache_snapshot(report)

    def fail_save(self, *args, **kwargs):
        raise RuntimeError("后期保存失败")

    monkeypatch.setattr(ExcelWriter, "save", fail_save)
    with pytest.raises(RuntimeError, match="后期保存失败"):
        report.to_excel(
            str(tmp_path / "late-save.xlsx"),
            n_bins=4,
            amount_col="amount",
            date_col="date",
            group_col="group",
            with_plots=False,
        )
    _assert_cache_snapshot(report, snapshot)


@pytest.mark.parametrize("warm", [False, True])
def test_model_report_to_dict_late_bin_failure_restores_all_cache_objects(monkeypatch, warm):
    report = ModelReport(
        _CountingProbabilityModel(),
        datasets={key: value for key, value in list(_task11_datasets().items())[:2]},
        target="target",
        n_jobs=1,
    )
    if warm:
        _warm_report_caches(report)
    snapshot = _cache_snapshot(report)
    original = report.get_bin_table
    calls = 0

    def fail_second_bin(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("后期分箱失败")
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "get_bin_table", fail_second_bin)
    with pytest.raises(RuntimeError, match="后期分箱失败"):
        report.to_dict()
    _assert_cache_snapshot(report, snapshot)
