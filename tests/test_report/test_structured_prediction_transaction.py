"""结构化分箱预测和直接模型报告整批发布的契约。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning.spec import BinSpec
from hscredit.report import OverduePredictor, feature_bin_stats
from hscredit.report.result import ReportGenerationError
from hscredit.skills_runtime.artifacts import ArtifactTransaction
from tests.test_report.test_audit_report_contracts import small_model_report


def test_raw_predictor_reuses_exact_binner_at_rounded_boundary():
    boundary = 0.123456789
    train = pd.DataFrame({"x": [0.0, boundary - 1e-8, boundary, 1.0], "y": [0, 0, 1, 1]})
    predictor = OverduePredictor("x", target="y", rules=[boundary], n_jobs=1).fit(train)
    assert predictor.prediction_mode_ == "structured"
    assert hasattr(predictor, "binner_")
    prediction = predictor.predict(train[["x"]])
    np.testing.assert_allclose(prediction, [0.0, 0.0, 1.0, 1.0])


def test_structured_table_predictions_do_not_parse_display_labels():
    boundary = 0.123456789
    train = pd.DataFrame({"x": [0.0, 0.1, 0.2, 1.0], "y": [0, 0, 1, 1]})
    table, binners = feature_bin_stats(train, "x", target="y", rules=[boundary], return_binner=True, n_jobs=1)
    specs = BinSpec.from_binner(binners["x"], "x")
    table["分箱标签"] = ["显示成0.12的低分箱", "显示成0.12的高分箱"]
    test = pd.DataFrame({"x": [boundary - 1e-8, boundary, boundary + 1e-8]})
    with_specs = OverduePredictor("x", bin_specs=specs, table_mode="strict", n_jobs=1).fit(table)
    with_binner = OverduePredictor("x", binner=binners["x"], table_mode="strict", n_jobs=1).fit(table)
    np.testing.assert_allclose(with_specs.predict(test), [0.0, 1.0, 1.0])
    pd.testing.assert_series_equal(with_specs.predict(test), with_binner.predict(test))


def test_category_specs_preserve_comma_and_quote_values():
    table = pd.DataFrame({"分箱": [0, 1], "分箱标签": ["展示箱甲", "展示箱乙"], "坏样本率": [0.1, 0.9], "样本总数": [10, 10]})
    specs = {0: BinSpec("x", 0, kind="categories", values=("a,b",)), 1: BinSpec("x", 1, kind="categories", values=("a'b",))}
    predictor = OverduePredictor("x", bin_specs=specs, table_mode="strict").fit(table)
    np.testing.assert_allclose(predictor.predict(pd.DataFrame({"x": ["a,b", "a'b", "a"]})), [0.1, 0.9, 0.5])


def test_legacy_table_is_explicitly_marked_and_strict_rejects_it():
    table = pd.DataFrame({"分箱标签": ["[-inf, 0)", "[0, inf)"], "坏样本率": [0.1, 0.3]})
    with pytest.warns(FutureWarning, match="legacy"):
        predictor = OverduePredictor("x").fit(table)
    assert predictor.prediction_mode_ == "legacy_labels"
    with pytest.raises(ValueError, match="binner 或 bin_specs"):
        OverduePredictor("x", table_mode="strict").fit(table)


def test_multifeature_stats_return_fitted_binners_without_changing_default():
    data = pd.DataFrame({"x": [0, 1, 2, 3], "z": [2, 4, 1, 3], "y": [0, 0, 1, 1]})
    table, rules, binners = feature_bin_stats(data, ["x", "z"], target="y", method="quantile", return_rules=True, return_binner=True, n_jobs=1)
    assert set(binners) == {"x", "z"} == set(rules)
    assert "分箱" in table
    default = feature_bin_stats(data, "x", target="y", method="quantile", n_jobs=1)
    assert "分箱" not in default


def test_direct_report_transaction_publishes_manifest_and_preserves_older_versions(tmp_path):
    report = small_model_report()
    destination = tmp_path / "model.xlsx"
    destination.write_bytes(b"legacy-file-untouched")
    first = report.to_excel(str(destination), with_plots=True, show_lift=False, transactional=True, return_result=True)
    first_path = Path(first.artifacts[0]["路径"])
    assert first_path.is_file() and first_path != destination
    manifest_path = Path(first.metadata["完成清单"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["状态"] == "完成"
    assert any(item["type"] == "excel" for item in manifest["制品"])
    assert any(item["type"] == "image" for item in manifest["制品"])
    assert all(Path(item["path"]).is_file() for item in manifest["制品"])
    old_bytes = first_path.read_bytes()
    second = report.to_excel(str(destination), with_plots=False, transactional=True)
    assert isinstance(second, str) and Path(second).is_file()
    assert second != str(first_path)
    assert first_path.read_bytes() == old_bytes
    assert destination.read_bytes() == b"legacy-file-untouched"


def test_transaction_failure_after_image_stage_preserves_previous_bundle(tmp_path, monkeypatch):
    report = small_model_report()
    destination = tmp_path / "model.xlsx"
    first = report.to_excel(str(destination), with_plots=False, transactional=True, return_result=True)
    manifest_path = Path(first.metadata["完成清单"])
    old_manifest = manifest_path.read_bytes()
    old_workbook = Path(first.artifacts[0]["路径"])
    old_bytes = old_workbook.read_bytes()

    def failing_export(path, **kwargs):
        path = Path(path)
        asset_dir = path.parent / f"{path.stem}_assets"
        asset_dir.mkdir()
        (asset_dir / "partial.png").write_bytes(b"partial-image")
        raise RuntimeError("受控图片失败")
    monkeypatch.setattr(report, "_to_excel_impl", failing_export)
    with pytest.raises(ReportGenerationError, match="图片失败") as error:
        report.to_excel(str(destination), transactional=True)
    assert "核心数据计算" in error.value.result.sections
    assert manifest_path.read_bytes() == old_manifest
    assert old_workbook.read_bytes() == old_bytes
    assert not list(tmp_path.rglob("partial.png"))
    assert not list(tmp_path.glob(".hscredit-skill-*"))


def test_transaction_commit_failure_never_exposes_staging_as_published(tmp_path, monkeypatch):
    report = small_model_report()
    destination = tmp_path / "model.xlsx"
    first = report.to_excel(str(destination), with_plots=False, transactional=True, return_result=True)
    manifest_path = Path(first.metadata["完成清单"])
    before = manifest_path.read_bytes()
    def fail_commit(self):
        raise RuntimeError("受控整批提交失败")
    monkeypatch.setattr(ArtifactTransaction, "_commit", fail_commit)
    result = report.to_excel(str(destination), with_plots=False, transactional=True, mode="best_effort", return_result=True)
    assert result.artifacts == [] and not result.complete
    assert manifest_path.read_bytes() == before
