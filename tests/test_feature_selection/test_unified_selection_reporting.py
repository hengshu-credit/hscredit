"""所有筛选器共用报告的字段、组合、快照和发布合同。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import hscredit.core.selectors as selectors
from hscredit.core.selectors.reporting import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    METRIC_COLUMNS,
    HISTORY_COLUMNS,
    SelectionReport,
    collect_selection_report,
)
from hscredit.exceptions import NotFittedError, ValidationError
from hscredit.report.selection_report import load_selection_report


@pytest.fixture
def data():
    rng = np.random.RandomState(18)
    frame = pd.DataFrame(rng.uniform(0.1, 3, size=(80, 4)), columns=["信号", "噪声", "强保", "强删"])
    target = (frame["信号"] + rng.normal(0, 0.5, 80) > 1.4).astype(int)
    return frame, target


def make_selector(name):
    model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=7, n_jobs=1)
    options = {"n_jobs": 1, "include": ["强保"], "exclude": ["强删"]}
    special = {
        "RegexSelector": {"pattern": "信号|噪声"},
        "TypeSelector": {"dtype_include": "number"},
        "CorrSelector": {"weights": {"信号": 1, "噪声": 2, "强保": 3, "强删": 4}, "binning_params": None},
        "FeatureImportanceSelector": {"estimator": model},
        "NullImportanceSelector": {"estimator": model, "n_runs": 2, "cv": 2},
        "RFESelector": {"estimator": model, "n_features_to_select": 2},
        "SequentialFeatureSelector": {"estimator": model, "n_features_to_select": 2, "cv": 2},
        "StepwiseSelector": {"estimator": "ols", "max_iter": 3},
        "BorutaSelector": {"estimator": model, "max_iter": 3, "n_estimators": 4},
        "PSISelector": {"n_splits": 2},
        "ScorecardFeatureSelection": {"iv_threshold": 0, "corr_threshold": None},
        "CompositeFeatureSelector": {
            "selectors": [selectors.NullSelector(n_jobs=1), selectors.VarianceSelector(n_jobs=1)]
        },
    }
    options.update(special.get(name, {}))
    return getattr(selectors, name)(**options)


_NAMES = [
    "NullSelector",
    "ModeSelector",
    "CardinalitySelector",
    "TypeSelector",
    "RegexSelector",
    "VarianceSelector",
    "Chi2Selector",
    "FTestSelector",
    "MutualInfoSelector",
    "IVSelector",
    "KSSelector",
    "LiftSelector",
    "PSISelector",
    "StabilityAwareSelector",
    "CorrSelector",
    "VIFSelector",
    "FeatureImportanceSelector",
    "NullImportanceSelector",
    "RFESelector",
    "SequentialFeatureSelector",
    "BorutaSelector",
    "StepwiseSelector",
    "CompositeFeatureSelector",
    "ScorecardFeatureSelection",
]


@pytest.mark.parametrize("name", _NAMES)
def test_all_selectors_share_full_details_and_metric_schema(name, data):
    X, y = data
    fitted = make_selector(name).fit(X, y)
    report = collect_selection_report(fitted)
    assert report.details.columns.tolist() == DETAIL_COLUMNS
    assert report.summary.columns.tolist() == SUMMARY_COLUMNS
    assert report.metrics.columns.tolist() == METRIC_COLUMNS
    assert report.history.columns.tolist() == HISTORY_COLUMNS
    root = report.details[report.details["阶段路径"] == "selector"]
    assert root["特征"].tolist() == X.columns.tolist()
    assert root.loc[root["特征"] == "强保", "决策来源"].item() == "强制保留"
    assert root.loc[root["特征"] == "强删", "决策来源"].item() == "强制剔除"
    assert root.loc[root["筛选结果"] == "保留", "特征"].tolist() == fitted.selected_features_
    for frame, empty in (
        (report.details, SelectionReport().details),
        (report.metrics, SelectionReport().metrics),
        (report.summary, SelectionReport().summary),
        (report.history, SelectionReport().history),
    ):
        assert frame.dtypes.equals(empty.dtypes)
    assert "[SUMMARY]" not in report.details["特征"].tolist()


def test_snapshot_remains_fit_configuration_and_is_fully_detached(data):
    X, _ = data
    selector = selectors.NullSelector(threshold=0.3, n_jobs=1).fit(X)
    expected = selector.get_selection_details()
    report = selector.get_selection_result()
    report.details.loc[:, "筛选结果"] = "损坏"
    meta = report.metadata
    meta["阶段快照"]["selector"]["输入字段"].clear()
    selector.set_params(threshold=0.9)
    pd.testing.assert_frame_equal(selector.get_selection_details(), expected)
    assert selector.get_selection_result().metadata["阶段快照"]["selector"]["拟合参数"]["threshold"] == 0.3


def test_mixed_type_labels_not_sorted_or_coerced():
    X = pd.DataFrame({1: [0.0, 1, 2], "1": [2.0, 1, 0], "z": [1.0, 1, 1]})
    composite = selectors.CompositeFeatureSelector([selectors.VarianceSelector(n_jobs=1)], n_jobs=1).fit(X)
    details = collect_selection_report(composite).details
    assert details.loc[details["阶段路径"] == "selector", "特征"].tolist() == [1, "1", "z"]
    assert details.loc[(details["阶段路径"] == "selector") & (details["筛选结果"] == "保留"), "特征"].tolist() == [
        1,
        "1",
    ]


def test_pipeline_and_nested_pipeline_are_read_without_refit(data, monkeypatch):
    X, y = data
    pipeline = Pipeline(
        [
            (
                "预筛",
                Pipeline([("空值", selectors.NullSelector(n_jobs=1)), ("方差", selectors.VarianceSelector(n_jobs=1))]),
            ),
            ("模型", LogisticRegression()),
        ]
    ).fit(X, y)
    monkeypatch.setattr(selectors.NullSelector, "fit", lambda *a, **k: pytest.fail("报告禁止重复fit"))
    report = collect_selection_report(pipeline)
    paths = report.summary["阶段路径"].tolist()
    assert len(paths) == len(set(paths))
    assert "pipeline/01_预筛/01_空值" in paths
    assert "pipeline/01_预筛/02_方差" in paths
    assert not any("模型" in path for path in paths)


def test_list_default_independent_and_explicit_relations(data):
    X, _ = data
    X["噪声"] = 1.0
    left = selectors.VarianceSelector(n_jobs=1).fit(X)
    right = selectors.VarianceSelector(n_jobs=1).fit(X)
    independent = collect_selection_report([("同名", left), ("同名", right)])
    assert independent.metadata["关系类型"] == "independent"
    assert "最终字段" not in independent.metadata
    assert independent.summary["阶段路径"].nunique() == 2
    with pytest.raises(ValidationError, match="输入字段"):
        collect_selection_report([left, right], relation="sequential")
    intersection = collect_selection_report([left, right], relation="intersection")
    assert intersection.summary.iloc[0]["剔除特征数"] == 1
    right.fit(left.transform(X))
    chain = collect_selection_report([left, right], relation="sequential")
    assert chain.summary.iloc[0]["选中特征数"] == 3


def test_unfitted_errors_and_legitimate_skip_is_not_failure(data):
    with pytest.raises(NotFittedError):
        collect_selection_report(selectors.NullSelector())
    partial = collect_selection_report([selectors.NullSelector()], strict=False)
    assert not partial.metadata["完整"]
    X, _ = data
    composite = selectors.CompositeFeatureSelector(
        [
            selectors.VarianceSelector(threshold=100, n_jobs=1),
            selectors.NullSelector(n_jobs=1),
        ],
        n_jobs=1,
    ).fit(X)
    report = collect_selection_report(composite)
    assert report.metadata["完整"]
    assert report.summary.iloc[-1]["执行状态"] == "未执行"
    assert "上游" in report.summary.iloc[-1]["停止原因"]


def test_scorecard_disabled_steps_have_explicit_status(data):
    X, y = data
    selector = selectors.ScorecardFeatureSelection(
        iv_threshold=None, corr_threshold=None, mode_threshold=None, n_jobs=1
    ).fit(X, y)
    report = collect_selection_report(selector)
    assert len(report.summary) == 5
    assert (report.summary["执行状态"] == "未执行").sum() == 3


def test_no_training_rows_are_captured_in_parameters(data):
    X, _ = data
    selector = selectors.PSISelector(oot_df=X.copy(), n_jobs=1).fit(X)
    params = collect_selection_report(selector).metadata["阶段快照"]["selector"]["拟合参数"]
    assert params["oot_df"]["数据内容"] == "未保存"
    assert params["oot_df"]["形状"] == X.shape


def test_snapshot_list_is_supported(data):
    X, _ = data
    snapshot = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X))
    report = collect_selection_report([snapshot, snapshot])
    assert report.summary["阶段名称"].tolist() == ["NullSelector", "NullSelector"]


def test_trace_budget_fails_before_cartesian_expansion(data):
    X, _ = data
    report = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X))
    with pytest.raises(ValidationError, match="预算"):
        report.get_feature_trace(max_rows=1)


def test_safe_excel_json_types_roundtrip_and_sheet_splitting(tmp_path):
    from openpyxl import load_workbook

    X = pd.DataFrame({1: [0.0, 1, 2], "1": [2.0, 1, 0], "=1+1": [1.0, 1, 1]})
    report = collect_selection_report(selectors.VarianceSelector(n_jobs=1).fit(X))
    result = report.save(tmp_path / "nested" / "筛选.xlsx", max_rows_per_sheet=2)
    restored = load_selection_report(result["路径"]["json"])
    pd.testing.assert_frame_equal(restored.details, report.details)
    pd.testing.assert_frame_equal(restored.metrics, report.metrics)
    workbook = load_workbook(result["路径"]["xlsx"], data_only=False)
    assert not any(cell.data_type == "f" for sheet in workbook for row in sheet for cell in row)
    assert any(cell.value == "=1+1" for sheet in workbook for row in sheet for cell in row)
    assert all(sheet.freeze_panes == "A2" and sheet.auto_filter.ref for sheet in workbook)
    detail_sheet = workbook["特征决策_1"]
    assert detail_sheet.column_dimensions["F"].width == 30
    assert detail_sheet["F2"].alignment.vertical == "top"
    assert all(item["行数"] <= 2 for item in result["工作表映射"])
    manifest = json.loads(Path(result["清单"]).read_text(encoding="utf-8"))
    assert all(item["sha256"] and item["bytes"] > 0 for item in manifest["制品"])


def test_nonfinite_and_tuple_fields_json_roundtrip(tmp_path):
    report = SelectionReport(
        details=[{"特征": (1, "x"), "指标值": np.inf}],
        metadata={"字段": [1, "1", (1, "x")], "特殊数": [np.nan, -np.inf]},
    )
    output = report.save(tmp_path / "nonfinite", formats=("json",))
    restored = load_selection_report(output["路径"]["json"])
    assert restored.details.iloc[0]["特征"] == (1, "x")
    assert restored.details.iloc[0]["指标值"] == np.inf
    assert restored.metadata["特殊数"][1] == -np.inf
    raw = Path(output["路径"]["json"]).read_text("utf-8")
    json.loads(raw, parse_constant=lambda value: pytest.fail("非法JSON常量"))


def test_publish_failure_preserves_previous_manifest_and_both_files(tmp_path, monkeypatch, data):
    import hscredit.report.selection_report as adapter

    X, _ = data
    report = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X))
    previous = report.save(tmp_path / "report")
    manifest = Path(previous["清单"]).read_bytes()
    contents = {kind: Path(path).read_bytes() for kind, path in previous["路径"].items()}
    monkeypatch.setattr(adapter, "_write_excel", lambda *a, **k: (_ for _ in ()).throw(OSError("模拟Excel失败")))
    with pytest.raises(OSError, match="模拟Excel失败"):
        report.save(tmp_path / "report", overwrite=True)
    assert Path(previous["清单"]).read_bytes() == manifest
    assert all(Path(previous["路径"][kind]).read_bytes() == content for kind, content in contents.items())
    assert not (tmp_path / ".report.publish.lock").exists()


def test_capacity_rejected_before_publish(tmp_path, data):
    X, _ = data
    report = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X))
    with pytest.raises(ValidationError, match="max_excel_cells"):
        report.save(tmp_path / "report", max_excel_cells=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", [[], Pipeline([("标准化", StandardScaler())])])
def test_no_selectors_is_not_a_successful_report(source, tmp_path):
    with pytest.raises(ValidationError, match="没有可收集"):
        collect_selection_report(source)
    report = collect_selection_report(source, strict=False)
    assert not report.metadata["完整"]
    with pytest.raises(ValidationError, match="allow_incomplete"):
        report.save(tmp_path / "partial")
    saved = report.save(tmp_path / "partial", allow_incomplete=True, formats=("json",))
    assert not saved["完整"]


@pytest.mark.parametrize("target_rm", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_target_passthrough_is_a_transform_boundary_not_a_selected_feature(target_rm, nested):
    X = pd.DataFrame({"目标": np.tile([0, 1], 10), "x": np.arange(20), "常量": np.ones(20)})
    selection = Pipeline(
        [
            ("空值", selectors.NullSelector(target="目标", n_jobs=1)),
            ("方差", selectors.VarianceSelector(target="目标", target_rm=target_rm, n_jobs=1)),
        ]
    )
    pipeline = Pipeline(
        [
            ("筛选", selection if nested else selectors.VarianceSelector(target="目标", target_rm=target_rm, n_jobs=1)),
            ("模型", LogisticRegression()),
        ]
    ).fit(X, X["目标"])
    report = collect_selection_report(pipeline)
    actual_output = ["x"] if target_rm else ["x", "目标"]
    assert report.metadata["完整"]
    assert "目标" not in report.details["特征"].tolist()
    assert report.summary.iloc[0]["输入特征数"] == 2
    assert report.summary.iloc[0]["选中特征数"] == 1
    assert report.summary.iloc[0]["剔除特征数"] == 1
    assert report.metadata["最终字段"] == ["x"]
    assert report.metadata["最终转换字段"] == actual_output
    snapshot = report.metadata["阶段快照"]["pipeline"]
    assert snapshot["转换输入字段"] == X.columns.tolist()
    assert snapshot["输出字段"] == ["x"]
    assert snapshot["转换输出字段"] == actual_output
    boundary = next(item for item in report.metadata["字段边界"] if item["类型"] == "模型")
    assert boundary["输入字段"] == actual_output
    assert boundary["非筛选字段"] == ([] if target_rm else ["目标"])


def test_target_passthrough_sequential_selector_list_preserves_feature_counts():
    X = pd.DataFrame({"目标": [0, 1, 0, 1], "x": [0, 1, 2, 3], "常量": [1, 1, 1, 1]})
    first = selectors.NullSelector(target="目标", n_jobs=1).fit(X)
    second = selectors.VarianceSelector(target="目标", n_jobs=1).fit(first.transform(X))
    report = collect_selection_report([first, second], relation="sequential")
    assert report.metadata["完整"]
    assert report.metadata["最终字段"] == ["x"]
    assert report.metadata["最终转换字段"] == ["x", "目标"]
    assert report.summary.iloc[0]["输入特征数"] == 2
    assert report.summary.iloc[0]["选中特征数"] == 1
    assert "目标" not in report.details["特征"].tolist()


def test_target_passthrough_does_not_disable_model_boundary_validation():
    X = pd.DataFrame({"x": np.arange(20), "目标": np.tile([0, 1], 10)})
    selector = selectors.NullSelector(target="目标", n_jobs=1).fit(X)
    model = LogisticRegression().fit(X[["x"]], X["目标"])
    pipeline = Pipeline([("筛选", selector), ("模型", model)])
    with pytest.raises(ValidationError, match="模型实际输入"):
        collect_selection_report(pipeline)
    assert not collect_selection_report(pipeline, strict=False).metadata["完整"]


def test_safe_parameter_snapshot_never_calls_arbitrary_repr(data):
    from hscredit.core.selectors.reporting import _parameters

    class PrivateObject:
        def __str__(self):
            pytest.fail("参数快照禁止展开对象自定义输出")

    assert _parameters(PrivateObject())["内容"] == "未保存"


def test_date_labels_keep_distinct_python_types(tmp_path):
    from datetime import date, datetime

    labels = [date(2026, 1, 1), datetime(2026, 1, 2, 3), pd.Timestamp("2026-01-03", tz="UTC")]
    report = SelectionReport(details=[{"特征": label} for label in labels], metadata={"字段": labels})
    saved = report.save(tmp_path / "types", formats=("json",))
    restored = load_selection_report(saved["路径"]["json"])
    actual = restored.metadata["字段"]
    assert [type(item) for item in actual] == [type(item) for item in labels]
    assert actual == labels


@pytest.mark.parametrize(
    "tamper",
    ["missing_column", "row_width", "missing_table", "bool_version", "completeness", "numeric_string", "bool_count"],
)
def test_loader_rejects_corrupt_protocol(tmp_path, tamper, data):
    X, _ = data
    saved = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X)).save(
        tmp_path / "report", formats=("json",)
    )
    path = Path(saved["路径"]["json"])
    payload = json.loads(path.read_text("utf-8"))
    if tamper == "missing_column":
        payload["表"]["details"]["列"].pop()
    elif tamper == "row_width":
        payload["表"]["details"]["行"][0].pop()
    elif tamper == "missing_table":
        del payload["表"]["metrics"]
    elif tamper == "bool_version":
        payload["协议版本"] = True
    elif tamper == "numeric_string":
        column = payload["表"]["metrics"]["列"].index("数值")
        payload["表"]["metrics"]["行"][0][column] = "损坏的数字"
    elif tamper == "bool_count":
        column = payload["表"]["summary"]["列"].index("输入特征数")
        payload["表"]["summary"]["行"][0][column] = True
    else:
        for pair in payload["元数据"]["值"]:
            if pair[0] == "完整":
                pair[1] = "true"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_selection_report(path)


def test_report_result_adapter_preserves_completeness(data):
    X, _ = data
    report = collect_selection_report(selectors.NullSelector(n_jobs=1).fit(X))
    result = report.to_report_result()
    assert result.complete
    assert set(result) == {"筛选汇总", "特征决策", "指标明细", "迭代记录"}
    result["特征决策"].iloc[0, 0] = 99
    assert report.details.iloc[0, 0] == 1
    assert not collect_selection_report([], strict=False).to_report_result().complete


@pytest.mark.parametrize(
    "name", ["CorrSelector", "RFESelector", "BorutaSelector", "StabilityAwareSelector", "SequentialFeatureSelector"]
)
def test_main_metric_threshold_and_direction_have_matching_units(name, data):
    X, y = data
    selector = make_selector(name).fit(X, y)
    details = selector.get_selection_details()
    if name in {"CorrSelector", "SequentialFeatureSelector"}:
        assert details["有效阈值"].isna().all()  # Correlation and feature-count budgets are not score thresholds.
    elif name == "RFESelector":
        assert details["有效阈值"].eq(1).all()
        assert set(details["指标名称"]) == {"RFE排名"}
    elif name == "BorutaSelector":
        assert details["有效阈值"].eq(selector.corrected_alpha_).all()
        assert set(details["指标名称"]) == {"显著性p值"}
        assert set(details["指标方向"]) == {"越小越优"}
    elif name == "StabilityAwareSelector":
        assert details["有效阈值"].eq(selector.score_threshold).all()
        assert set(details["指标名称"]) == {"稳定性综合评分"}


@pytest.mark.parametrize("as_frame", [False, True])
def test_report_constructor_detaches_nested_mutable_source(as_frame):
    subset = ["a"]
    rows = [{"候选子集": subset}]
    if as_frame:
        rows = pd.DataFrame(rows)
    report = SelectionReport(history=rows)
    subset.append("tamper")
    assert report.history.iloc[0]["候选子集"] == ["a"]
    view = report.history
    view.iloc[0]["候选子集"].append("tamper2")
    assert report.history.iloc[0]["候选子集"] == ["a"]


def test_nested_pipeline_preserves_trailing_transform_field_boundary():
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]})
    inner = Pipeline(
        [
            ("select", selectors.NullSelector(n_jobs=1)),
            ("encode", OneHotEncoder(sparse_output=False).set_output(transform="pandas")),
        ]
    )
    pipeline = Pipeline([("inner", inner), ("outerselect", selectors.NullSelector(n_jobs=1))]).fit(X)
    report = collect_selection_report(pipeline)
    assert report.metadata["最终字段"] == pipeline.get_feature_names_out().tolist()
    assert report.summary.iloc[0]["输入特征数"] == 2
    assert report.summary.iloc[0]["选中特征数"] == 8
    assert pd.isna(report.summary.iloc[0]["剔除特征数"])


def test_stability_reports_each_threshold_in_its_own_units(data):
    X, y = data
    selector = selectors.StabilityAwareSelector(iv_threshold=0.1, psi_threshold=0.2, score_threshold=0.3, n_jobs=1).fit(
        X, y
    )
    metrics = selector.get_selection_result().metrics
    for label, threshold, comparison in (("IV", 0.1, ">="), ("PSI", 0.2, "<="), ("综合分", 0.3, ">=")):
        part = metrics[(metrics["指标名称"] == label) & (metrics["统计口径"] == "拟合时保存")]
        assert len(part) == len(X.columns)
        assert part["阈值"].eq(threshold).all()
        assert part["比较符"].eq(comparison).all()


@pytest.mark.parametrize("name", ["Chi2Selector", "FTestSelector"])
def test_statistical_topk_conditions_include_rank_and_budget(name, data):
    X, y = data
    kwargs = {"k": 2, "n_jobs": 1}
    if name == "FTestSelector":
        kwargs["percentile"] = 50
    selector = getattr(selectors, name)(**kwargs).fit(X, y)
    metrics = selector.get_selection_result().metrics
    rows = metrics[metrics["指标名称"] == "数量限制达标"]
    assert rows["阈值"].eq(2).all()
    assert rows["比较符"].eq("<=").all()
    assert rows["数值"].tolist() == selector.ranks_.tolist()
    assert rows["条件结果"].tolist() == (rows["数值"] <= 2).tolist()
    if name == "FTestSelector":
        rows = metrics[metrics["指标名称"] == "百分位达标"]
        assert rows["阈值"].eq(selector.percentile_cutoff_).all()
        assert rows["数值"].tolist() == selector.percentile_values_.tolist()


def test_f_undefined_raw_statistic_not_presented_as_valid_zero():
    X = pd.DataFrame({"常量": [1.0] * 10, "变量": np.arange(10, dtype=float)})
    selector = selectors.FTestSelector(n_jobs=1).fit(X, np.tile([0, 1], 5))
    details = selector.get_selection_details().set_index("特征")
    assert pd.isna(details.loc["常量", "指标值"])
    assert details.loc["常量", "评估状态"] == "无效"
    metrics = selector.get_selection_result().metrics
    applied = metrics[(metrics["特征"] == "常量") & (metrics["指标名称"] == "实际F筛选统计量（原始NaN归0）")]
    assert applied["数值"].item() == 0.0


def test_unfitted_pipeline_model_has_unverifiable_boundary(data):
    X, _ = data
    pipeline = Pipeline([("selector", selectors.NullSelector(n_jobs=1).fit(X)), ("model", LogisticRegression())])
    with pytest.raises(ValidationError, match="无法验证实际入模"):
        collect_selection_report(pipeline)
    assert not collect_selection_report(pipeline, strict=False).metadata["完整"]
