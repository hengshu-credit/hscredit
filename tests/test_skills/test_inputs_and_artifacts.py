"""Skills 输入解析和事务制品测试。"""

import pandas as pd
import pytest

from hscredit.skills_runtime.artifacts import ArtifactTransaction, summarize_dataframe
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.skills_runtime.io import InputResolver
from hscredit.skills_runtime.objects import ObjectRegistry


def test_input_resolver_reads_the_requested_excel_sheet(tmp_path):
    """防止 Excel 输入静默读取错误工作表。"""
    path = tmp_path / "sample.xlsx"
    expected = pd.DataFrame({"年龄": [22, 35], "target": [0, 1]})
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"忽略": [1]}).to_excel(writer, sheet_name="其他", index=False)
        expected.to_excel(writer, sheet_name="建模样本", index=False)

    result = InputResolver().resolve({"kind": "file", "path": str(path), "sheet_name": "建模样本"})

    pd.testing.assert_frame_equal(result, expected)


def test_input_resolver_reads_csv_options(tmp_path):
    """防止 CSV 编码和分隔符配置被忽略。"""
    path = tmp_path / "sample.csv"
    path.write_text("年龄;target\n22;0\n35;1\n", encoding="utf-8-sig")

    result = InputResolver().resolve(
        {"kind": "file", "path": str(path), "encoding": "utf-8-sig", "separator": ";"}
    )

    assert result.to_dict("records") == [{"年龄": 22, "target": 0}, {"年龄": 35, "target": 1}]


def test_object_registry_resolves_the_exact_reference(credit_frame):
    """防止对象引用被模糊匹配到其他对象。"""
    registry = ObjectRegistry({"data:credit": credit_frame, "data:other": object()})

    result = InputResolver(registry).resolve({"kind": "object_ref", "ref": "data:credit"})

    assert result is credit_frame


def test_untrusted_joblib_input_is_rejected(tmp_path):
    """防止未授权反序列化执行任意代码。"""
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not-loaded")

    with pytest.raises(SkillExecutionError) as exc_info:
        InputResolver().resolve({"kind": "file", "path": str(path)})

    assert exc_info.value.code == "ARTIFACT_UNTRUSTED"


def test_failed_artifact_transaction_does_not_publish_a_partial_file(tmp_path):
    """防止失败任务留下看似成功的半成品。"""
    final_path = tmp_path / "report.xlsx"

    with pytest.raises(RuntimeError):
        with ArtifactTransaction(
            {"directory": str(tmp_path), "name": "report", "overwrite": False}
        ) as transaction:
            transaction.stage_path("report.xlsx").write_bytes(b"partial")
            raise RuntimeError("render failed")

    assert not final_path.exists()
    assert not list(tmp_path.glob(".hscredit-skill-*"))


def test_artifact_transaction_refuses_to_overwrite_existing_file(tmp_path):
    """防止默认覆盖用户已有报告。"""
    final_path = tmp_path / "report.xlsx"
    final_path.write_bytes(b"original")

    with pytest.raises(SkillExecutionError) as exc_info:
        with ArtifactTransaction(
            {"directory": str(tmp_path), "name": "report", "overwrite": False}
        ) as transaction:
            staged = transaction.stage_path("report.xlsx")
            staged.write_bytes(b"replacement")
            transaction.publish(staged, "report.xlsx")

    assert exc_info.value.code == "ARTIFACT_EXISTS"
    assert final_path.read_bytes() == b"original"


def test_dataframe_summary_limits_preview_and_serializes_multiindex_columns():
    """防止大表或不可 JSON 化列名进入 Agent 上下文。"""
    frame = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        columns=pd.MultiIndex.from_tuples([("样本", "数量"), ("风险", "坏样本率")]),
    )

    summary = summarize_dataframe(frame, preview_rows=2)

    assert summary["rows"] == 3
    assert summary["columns"] == [["样本", "数量"], ["风险", "坏样本率"]]
    assert len(summary["preview"]) == 2
