"""筛选报告的双表输出、流程组合与独立快照归档契约。"""

import json
import inspect
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hscredit.core.selectors import (
    CompositeFeatureSelector,
    NullSelector,
    RFESelector,
    SelectionReport,
    VarianceSelector,
    collect_selection_report,
)
from hscredit.exceptions import NotFittedError, ValidationError
from hscredit.report import selection_report, load_selection_report


@pytest.fixture
def data():
    return pd.DataFrame({"评分": [1, 2, 3, 4], "缺失字段": [1, np.nan, 3, np.nan], "常量": [1, 1, 1, 1]})


@pytest.fixture
def selector(data):
    return NullSelector(threshold=0.4, n_jobs=1).fit(data)


def decisions(process):
    return process.loc[process["记录类型"] == "特征决策"]


def test_selection_report_public_signature_and_unfitted_input():
    import hscredit
    import hscredit.report as reports

    assert hscredit.selection_report is reports.selection_report is selection_report
    assert import_module("hscredit.report.selection_report").selection_report is selection_report
    parameters = inspect.signature(selection_report).parameters
    assert list(parameters) == ["selector", "relation", "strict"]
    assert parameters["selector"].default is inspect.Parameter.empty
    assert "auto_selection_report" not in reports.__all__
    assert "auto_selection_report" not in hscredit.__all__
    with pytest.raises(TypeError, match="selector"):
        selection_report()
    with pytest.raises(NotFittedError):
        selection_report(selector=NullSelector())


def test_base_is_the_only_report_implementation_and_legacy_api_is_removed():
    import importlib.util
    import hscredit.core.selectors as selectors
    import hscredit.core.selectors.base as base

    assert selectors.selection_report is selection_report is base.selection_report
    assert selectors.SelectionReport is base.SelectionReport
    assert selectors.collect_selection_report is base.collect_selection_report
    assert base.SelectionReport.__module__ == base.__name__
    assert base.selection_report.__module__ == base.__name__
    assert importlib.util.find_spec("hscredit.core.selectors.reporting") is None
    for name in ("get_selection_report", "get_selection_report_df"):
        assert not hasattr(NullSelector(), name)
    collector = base.SelectionReportCollector()
    for name in ("add_report", "get_summary", "to_dataframe", "to_excel", "reports"):
        assert not hasattr(collector, name)


def test_missing_fitted_snapshot_is_not_reconstructed(selector, monkeypatch):
    import hscredit.core.selectors.base as base

    del selector.selection_report_
    monkeypatch.setattr(base, "capture_selection_report", lambda *a: pytest.fail("不能重建旧版报告"))
    with pytest.raises(ValidationError, match="缺少拟合报告快照"):
        selector.report()
    with pytest.raises(ValidationError, match="缺少拟合报告快照"):
        selector.get_selection_result()
    summary, details = selector.report(strict=False)
    assert summary.attrs["完整"] is False
    assert details.empty


@pytest.mark.parametrize(
    "option", ["excel_writer", "writer_params", "return_result", "mode", "max_sheets", "by_selector"]
)
def test_removed_export_options_are_rejected(selector, option):
    with pytest.raises(TypeError, match=option):
        selection_report(selector, **{option: None})


def test_report_returns_two_frames_without_fit_transform_or_files(selector, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("报告不能重新筛选或写入文件")

    monkeypatch.setattr(NullSelector, "fit", unexpected)
    monkeypatch.setattr(NullSelector, "transform", unexpected)
    adapter = import_module("hscredit.report.selection_report")
    monkeypatch.setattr(adapter, "save_selection_report", unexpected)
    monkeypatch.chdir(tmp_path)
    result = selection_report(selector=selector)
    assert type(result) is tuple and len(result) == 2
    summary, process = result
    assert all(isinstance(table, pd.DataFrame) for table in result)
    assert summary.iloc[0]["输入特征数"] == 3
    assert summary.iloc[0]["选中特征数"] == 2
    assert summary.iloc[0]["剔除特征数"] == 1
    assert summary.iloc[0]["保留率"] == 2 / 3
    detail = decisions(process)
    assert detail["特征"].tolist() == ["评分", "缺失字段", "常量"]
    assert detail["指标值"].tolist() == [0, 0.5, 0]
    assert detail["有效阈值"].tolist() == [0.4, 0.4, 0.4]
    assert detail["筛选结果"].tolist() == ["保留", "剔除", "保留"]
    assert not list(tmp_path.iterdir())


def test_native_report_is_independent_of_report_facade_and_keeps_fit_snapshot(selector, monkeypatch):
    expected = selection_report(selector)
    facade = import_module("hscredit.report.selection_report")
    monkeypatch.setattr(facade, "selection_report", lambda *a, **k: pytest.fail("核心筛选器不依赖报告层"))
    monkeypatch.setattr(NullSelector, "fit", lambda *a, **k: pytest.fail("不得重新拟合"))
    monkeypatch.setattr(NullSelector, "transform", lambda *a, **k: pytest.fail("不得重新转换"))
    selector.set_params(threshold=0.9)
    summary, details = selector.report()
    for actual, original in zip((summary, details), expected):
        pd.testing.assert_frame_equal(actual, original)
    summary.attrs["阶段快照"].clear()
    details.loc[:, "筛选结果"] = "修改副本"
    assert selector.report()[0].attrs["阶段快照"]
    assert selector.selected_features_ == ["评分", "常量"]


def test_native_report_handles_unfitted_and_invalid_options():
    selector = NullSelector()
    with pytest.raises(NotFittedError):
        selector.report()
    summary, details = selector.report(strict=False)
    assert summary.attrs["完整"] is False and summary.attrs["诊断"]
    assert details.empty
    for source in (selector, SelectionReport()):
        with pytest.raises(ValidationError, match="strict"):
            source.report(strict="False")


def test_independent_selectors_keep_separate_counts(data):
    items = [
        ("宽松", NullSelector(threshold=0.6, n_jobs=1).fit(data)),
        ("严格", NullSelector(threshold=0.4, n_jobs=1).fit(data)),
    ]
    summary, process = selection_report(items)
    assert summary["阶段名称"].tolist() == ["宽松", "严格"]
    assert summary["选中特征数"].tolist() == [3, 2]
    assert summary["剔除特征数"].tolist() == [0, 1]
    assert "最终字段" not in summary.attrs
    assert decisions(process).groupby("阶段路径", sort=False).size().tolist() == [3, 3]


@pytest.mark.parametrize("kind", ["list", "composite", "pipeline", "nested_pipeline"])
def test_sequential_forms_preserve_each_stage_and_final_counts(data, kind):
    if kind == "list":
        first = NullSelector(threshold=0.4, n_jobs=1).fit(data)
        second = VarianceSelector(n_jobs=1).fit(first.transform(data))
        source = [("缺失", first), ("方差", second)]
    else:
        steps = [("缺失", NullSelector(threshold=0.4, n_jobs=1)), ("方差", VarianceSelector(n_jobs=1))]
        if kind == "composite":
            source = CompositeFeatureSelector(steps, n_jobs=1).fit(data)
        elif kind == "pipeline":
            source = Pipeline(steps).fit(data)
        else:
            source = Pipeline([("预筛", Pipeline(steps)), ("模型", LogisticRegression())]).fit(data, [0, 0, 1, 1])
    summary, process = selection_report(source, relation="sequential")
    root = summary.iloc[0]
    assert root["输入特征数"] == 3 and root["选中特征数"] == 1 and root["剔除特征数"] == 2
    actual = decisions(process)
    leaf = summary[summary["筛选器"].isin(["NullSelector", "VarianceSelector"])]
    assert leaf["输入特征数"].tolist() == [3, 2]
    assert leaf["剔除特征数"].tolist() == [1, 1]
    for path, expected in zip(leaf["阶段路径"], [["缺失字段"], ["常量"]]):
        dropped = actual[(actual["阶段路径"] == path) & (actual["筛选结果"] == "剔除")]
        assert dropped["特征"].tolist() == expected
    assert "模型" not in summary["阶段名称"].tolist()
    assert process["阶段路径"].drop_duplicates().tolist() == [
        p for p in summary["阶段路径"] if p in set(process["阶段路径"])
    ]


def test_mixed_single_composite_and_pipeline_inputs(data):
    def steps():
        return [("缺失", NullSelector(threshold=0.4, n_jobs=1)), ("方差", VarianceSelector(n_jobs=1))]

    source = [
        ("独立", NullSelector(threshold=0.6, n_jobs=1).fit(data)),
        ("组合", CompositeFeatureSelector(steps(), n_jobs=1).fit(data)),
        ("流水线", Pipeline(steps()).fit(data)),
    ]
    summary, process = selection_report(source)
    assert summary["阶段路径"].is_unique
    roots = summary[summary["父路径"].isna()]
    assert roots["阶段名称"].tolist() == ["独立", "组合", "流水线"]
    assert roots["选中特征数"].tolist() == [3, 1, 1]
    assert set(process["阶段路径"]) <= set(summary["阶段路径"])
    assert "最终字段" not in summary.attrs


def test_unconnected_list_cannot_be_described_as_sequential(data):
    items = [NullSelector(threshold=0.4, n_jobs=1).fit(data), VarianceSelector(n_jobs=1).fit(data.fillna(0))]
    with pytest.raises(ValidationError, match="上一阶段输出"):
        selection_report(items, relation="sequential")


@pytest.mark.parametrize("relation, count", [("intersection", 1), ("union", 3)])
def test_set_relations_have_one_root_decision(data, relation, count):
    items = [NullSelector(threshold=0.4, n_jobs=1).fit(data), VarianceSelector(n_jobs=1).fit(data.fillna(0))]
    summary, process = selection_report(items, relation=relation)
    assert summary.iloc[0]["选中特征数"] == count
    assert summary.iloc[0]["剔除特征数"] == 3 - count
    root = decisions(process).query("阶段路径 == 'list'")
    assert len(root) == 3 and (root["筛选结果"] == "保留").sum() == count


def test_iterative_events_and_multiple_metrics_are_not_cartesian_joined():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(60, 3)), columns=["甲", "乙", "丙"])
    target = (data["甲"] + data["乙"] > 0).astype(int)
    selector = RFESelector(LogisticRegression(), n_features_to_select=1, report_history="diagnostics", n_jobs=1).fit(
        data, target
    )
    snapshot = collect_selection_report(selector)
    summary, process = selection_report(selector)
    assert not snapshot.history.empty
    assert len(process) == len(snapshot.details) + len(snapshot.metrics) + len(snapshot.history)
    for kind, frame in (("特征决策", snapshot.details), ("指标明细", snapshot.metrics), ("迭代事件", snapshot.history)):
        actual = process.loc[process["记录类型"] == kind, frame.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, frame, check_flags=False)
    assert summary.iloc[0]["选中特征数"] == 1


def test_partial_and_empty_results_keep_schema_and_isolated_diagnostics(selector):
    complete = selection_report(selector)
    partial = collect_selection_report([], strict=False)
    with pytest.raises(ValidationError, match="不完整"):
        selection_report(partial)
    summary, process = selection_report(partial, strict=False)
    assert summary.empty and process.empty
    assert summary.columns.tolist() == complete[0].columns.tolist()
    pd.testing.assert_series_equal(process.dtypes, complete[1].dtypes)
    assert summary.attrs["完整"] is False and summary.attrs["诊断"]
    summary.attrs["诊断"].clear()
    assert process.attrs["诊断"] and partial.metadata["诊断"]


def test_result_does_not_share_mutable_cells_or_metadata():
    snapshot = SelectionReport(
        summary=[{"阶段路径": "selector", "阶段编号": 1}],
        details=[{"阶段路径": "selector", "特征": (1, "甲"), "指标值": 1 / 3}],
        history=[{"阶段路径": "selector", "补充信息": {"候选": [1, 2]}, "是否采纳": True}],
        metadata={"自定义": {"值": [1]}},
    )
    summary, process = selection_report(snapshot)
    assert decisions(process).iloc[0]["特征"] == (1, "甲")
    assert decisions(process).iloc[0]["指标值"] == 1 / 3
    process.loc[process["记录类型"] == "迭代事件", "补充信息"].iloc[0]["候选"].clear()
    summary.attrs["自定义"]["值"].clear()
    assert snapshot.history.iloc[0]["补充信息"] == {"候选": [1, 2]}
    assert process.attrs["自定义"]["值"] == [1]


@pytest.mark.parametrize(
    "options",
    [{"max_excel_cells": True}, {"max_sheets": True}, {"max_rows_per_sheet": False}, {"by_selector": "False"}],
)
def test_invalid_archive_limits_fail_before_creating_files(tmp_path, selector, options):
    with pytest.raises(ValidationError):
        collect_selection_report(selector).save(tmp_path / "归档", **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("formats", [None, 1, [["json"]], ["json", "json"], {"json": True}])
def test_invalid_formats_raise_chinese_validation(tmp_path, selector, formats):
    with pytest.raises(ValidationError, match="formats"):
        collect_selection_report(selector).save(tmp_path / "报告", formats=formats)
    assert not list(tmp_path.iterdir())


def test_numpy_integer_limits_produce_serializable_mapping(tmp_path, selector):
    saved = collect_selection_report(selector).save(
        tmp_path / "整数预算", max_rows_per_sheet=np.int64(2), max_sheets=np.int64(20)
    )
    assert all(item["行数"] <= 2 for item in saved["工作表映射"])
    json.dumps(saved, ensure_ascii=False)


@pytest.mark.parametrize(
    "value", [{"类型": "浮点特殊值", "值": "损坏"}, {"类型": "日期", "值": "无效日期"}, {"类型": "映射", "值": [1]}]
)
def test_invalid_type_tags_raise_validation_error(tmp_path, value):
    saved = SelectionReport(metadata={"自定义": 1}).save(tmp_path / "报告", formats="json")
    path = Path(saved["路径"]["json"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["元数据"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="JSON"):
        load_selection_report(path)


def test_real_workbook_null_rates_and_roundtrip(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("本地演示工作簿 hscredit_yyp.xlsx 未提供")
    fields = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "商品类别"]
    data = pd.read_excel(path, usecols=fields + ["FPD"])
    selector = NullSelector(target="FPD", threshold=0.5, n_jobs=1).fit(data)
    summary, process = selection_report(selector)
    detail = decisions(process).set_index("特征")
    for field in fields:
        expected = sum(pd.isna(value) for value in data[field]) / len(data)
        assert detail.loc[field, "指标值"] == expected
    assert "FPD" not in detail.index
    assert summary.iloc[0]["输入特征数"] == 4
    saved = collect_selection_report(selector).save(tmp_path / "快照", formats="json")
    restored = load_selection_report(saved["路径"]["json"])
    for actual, expected in zip(selection_report(restored), (summary, process)):
        pd.testing.assert_frame_equal(actual, expected)
