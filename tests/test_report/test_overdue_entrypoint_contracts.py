"""所有逾期入口的标签参数、文档与自动参考集的原始行绑定契约。"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hscredit.core.rules import Rule
from hscredit.report.rule_analysis import rule_swap_analysis

ROOT = Path(__file__).resolve().parents[2]


def _overdue_entries():
    for package in ("core", "report"):
        for path in sorted((ROOT / "hscredit" / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            entries = [(node, node, node.name) for node in tree.body if isinstance(node, ast.FunctionDef)]
            for cls in tree.body:
                if isinstance(cls, ast.ClassDef) and not cls.name.startswith("_"):
                    entries.extend(
                        (node, cls if node.name == "__init__" else node, f"{cls.name}.{node.name}")
                        for node in cls.body
                        if isinstance(node, ast.FunctionDef)
                    )
            for node, documented, name in entries:
                arguments = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
                if {"overdue", "dpds"} <= arguments and (not node.name.startswith("_") or node.name == "__init__"):
                    yield f"{path.relative_to(ROOT)}:{name}", arguments, ast.get_docstring(documented) or ""


@pytest.mark.parametrize("entry", list(_overdue_entries()), ids=lambda entry: entry[0])
def test_all_overdue_entries_expose_and_document_label_controls(entry):
    """新增入口不能只暴露 overdue/dpds，或让 help() 仍声称只支持目标列名。"""
    _, arguments, doc = entry
    for name in ("target", "overdue_operator", "del_grey"):
        assert name in arguments
        assert f":param {name}:" in doc
    target_doc = re.search(r":param target:(.*?)(?=\n\s*:|\Z)", doc, re.S).group(1)
    for kind in ("Series", "ndarray", "list", "callable"):
        assert kind in target_doc
    assert "互斥" in target_doc


@pytest.fixture
def swap_frame():
    return pd.DataFrame(
        {
            "评分": np.arange(8, dtype=float),
            "标签": [0, 1, -1, 0, 1, np.nan, 0, 1],
            "MOB1": [0, 8, 2, 0, 8, np.nan, 0, 8],
        },
        index=[8, 3, 6, 1, 4, 7, 2, 5],
    )


@pytest.mark.parametrize("duplicate_index", [False, True])
@pytest.mark.parametrize("grey", [False, True])
@pytest.mark.parametrize("kind", ["column", "series", "array", "list", "callable", "dict"])
def test_implicit_swap_reference_keeps_original_label_rows(swap_frame, duplicate_index, grey, kind, monkeypatch):
    """各输入形式的参考分母独立核算，callable 仅在原始数据上执行一次。"""
    import hscredit.report.rule_analysis as module

    data = swap_frame.set_axis([0, 0, 1, 1, 2, 2, 3, 3]) if duplicate_index else swap_frame
    original = data.copy(deep=True)
    calls = []
    tables = []

    def label(frame):
        calls.append(frame.copy(deep=True))
        return frame["标签"]

    values = {
        "column": "标签",
        "series": data["标签"] if duplicate_index else data["标签"].sort_index(),
        "array": data["标签"].to_numpy(),
        "list": data["标签"].tolist(),
        "callable": label,
        "dict": {"自定义标签": label},
    }
    original_stats = module.feature_bin_stats

    def capture(*args, **kwargs):
        table = original_stats(*args, **kwargs)
        tables.append(table)
        return table

    monkeypatch.setattr(module, "feature_bin_stats", capture)
    result = rule_swap_analysis(
        data,
        score="评分",
        target=values[kind],
        del_grey=grey,
        rules_out=[Rule("评分 >= 6")],
        rules_in=[Rule("评分 == 5")],
        method="quantile",
        max_n_bins=2,
        n_jobs=1,
    )
    assert len(tables) == 1
    # 未表现样本不能参与参考坏率；非二值标签仅在剔灰时从参考分母移除。
    valid = data["标签"].isin([0, 1]) if grey else data["标签"].notna()
    total = tables[0].iloc[-1]
    assert total["样本总数"] == int(valid.sum())
    assert total["坏样本数"] == int(data["标签"].eq(1).sum())
    # OUT-IN 的缺失实际标签使用预测表现，不能随剔灰丢失；审批分母始终为输入全量。
    risk_valid = valid | data["评分"].eq(5)
    assert result["swap_pipeline"].iloc[0]["样本总数"] == (int(risk_valid.sum()) if grey else len(data))
    assert result["swap_pipeline"].iloc[0]["决策样本数"] == len(data)
    if kind in {"callable", "dict"}:
        assert len(calls) == 1
        assert_frame_equal(calls[0], original)
    assert_frame_equal(data, original)


@pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
@pytest.mark.parametrize("grey", [False, True])
def test_swap_reference_labels_match_explicit_target_modes(swap_frame, operator, grey):
    """自动参考集和显式参考集都按每个逾期标签独立剔灰与排除未表现行。"""
    options = dict(
        score="评分",
        overdue="MOB1",
        dpds=[7, 3, 0],
        overdue_operator=operator,
        del_grey=grey,
        method="quantile",
        max_n_bins=2,
        n_jobs=1,
        rules_out=[Rule("评分 >= 6")],
        rules_in=[Rule("评分 == 5")],
    )
    implicit = rule_swap_analysis(swap_frame, **options)
    explicit = rule_swap_analysis(swap_frame, reference_data=swap_frame, **options)
    for name in implicit:
        assert_frame_equal(implicit[name], explicit[name])


def test_multilabel_swap_reference_does_not_filter_other_labels(swap_frame, monkeypatch):
    """第一标签未表现的行仍可作为第二标签的已表现样本。"""
    import hscredit.report.rule_analysis as module

    original_stats = module.feature_bin_stats
    tables = []

    def capture(*args, **kwargs):
        table = original_stats(*args, **kwargs)
        tables.append(table)
        return table

    monkeypatch.setattr(module, "feature_bin_stats", capture)
    data = swap_frame.assign(MOB2=[8, 0, 0, 8, 0, 8, 8, 0])
    rule_swap_analysis(
        data,
        score="评分",
        overdue=["MOB1", "MOB2"],
        dpds=3,
        del_grey=True,
        rules_out=[Rule("评分 > 4")],
        method="quantile",
        max_n_bins=2,
        n_jobs=1,
    )
    total = tables[0].iloc[-1]
    assert total[("MOB1_3+", "样本总数")] == 6
    assert total[("MOB1_3+", "坏样本数")] == 3
    assert total[("MOB2_3+", "样本总数")] == 8
    assert total[("MOB2_3+", "坏样本数")] == 4


@pytest.mark.parametrize("entry", ["report", "model_report"])
@pytest.mark.parametrize("kind", ["series", "callable", "dict", ">", ">=", "<", "<="])
def test_model_workflow_forwards_target_controls(swap_frame, entry, kind):
    """继承统一工作流的模型入口使用相同标签定义，显式 target 不回退为模型目标列。"""
    from hscredit.core.models._workflow import ModelWorkflowMixin

    class Model(ModelWorkflowMixin):
        target = "标签"

        def predict_proba(self, data):
            probability = data["评分"].to_numpy() / 10
            return np.column_stack([1 - probability, probability])

    if kind in {">", ">=", "<", "<="}:
        options = dict(overdue="MOB1", dpds=3, overdue_operator=kind)
        days = swap_frame.MOB1
        values = {">": days.gt(3), ">=": days.ge(3), "<": days.lt(3), "<=": days.le(3)}[kind]
        valid = ~((days > 0) & (days <= 3)) if kind == ">" else pd.Series(True, index=days.index)
        if kind == ">=":
            valid = ~((days > 0) & (days < 3))
        expected = values[valid].astype(float)
    else:
        options = {
            "target": {
                "series": swap_frame["标签"],
                "callable": lambda df: df["标签"],
                "dict": {"指定标签": lambda df: df["标签"]},
            }[kind]
        }
        expected = swap_frame.loc[swap_frame["标签"].isin([0, 1]), "标签"]
    report = getattr(Model(), entry)(
        datasets={"样本": swap_frame},
        features=["评分"],
        del_grey=True,
        n_jobs=1,
        with_plots=False,
        verbose=False,
        **options,
    )
    np.testing.assert_equal(report._get_y("样本")[report._get_target_valid_mask("样本")], expected.to_numpy())
