"""其余 Skills 的公开操作和真实最小工作流回归。"""

from pathlib import Path

import pandas as pd

from hscredit.skills_runtime import execute_skill
from hscredit.skills_runtime.registry import ensure_default_operations


def _request(tmp_path, skill, operation, inputs, parameters=None, name=None):
    return {
        "version": "1",
        "operation": operation,
        "inputs": inputs,
        "parameters": parameters or {},
        "output": {"directory": str(tmp_path), "name": name or operation, "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }


def test_all_phase_two_skills_have_registered_operations():
    registry = ensure_default_operations()
    assert registry.list_operations("hscredit") == ("capabilities",)
    assert "model_fit" in registry.list_operations("hsmodel")
    assert "rule_report" in registry.list_operations("hsrule")
    assert "selector_fit" in registry.list_operations("hsselect")
    assert registry.list_operations("hsviz") == ("plot",)
    assert "dataframe_to_excel" in registry.list_operations("hsexcel")


def test_hscredit_capabilities_exposes_current_catalog(tmp_path):
    result = execute_skill("hscredit", _request(tmp_path, "hscredit", "capabilities", {}))
    assert set(result["summary"]["skills"]) >= {
        "hsbin",
        "hsreport",
        "hsmodel",
        "hsrule",
        "hsselect",
        "hsviz",
        "hsexcel",
    }


def test_hsmodel_fits_and_publishes_a_model(credit_frame, tmp_path):
    result = execute_skill(
        "hsmodel",
        _request(
            tmp_path,
            "hsmodel",
            "model_fit",
            {"data": {"kind": "object_ref", "ref": "data:credit"}},
            {
                "model": "LogisticRegression",
                "target": "target",
                "features": ["score", "age", "amount"],
                "model_params": {"max_iter": 200},
                "n_jobs": 1,
            },
            "lr",
        ),
        objects={"data:credit": credit_frame},
    )
    assert any(item["type"] == "hscredit-artifact" for item in result["artifacts"])
    assert any(item["type"] == "excel" for item in result["artifacts"])


def test_hsmodel_runs_autotuner_and_publishes_tuner(credit_frame, tmp_path):
    result = execute_skill(
        "hsmodel",
        _request(
            tmp_path,
            "hsmodel",
            "model_tune",
            {"data": {"kind": "object_ref", "ref": "data:credit"}},
            {
                "model_type": "lr",
                "n_trials": 1,
                "cv": 2,
                "n_jobs": 1,
                "features": ["score", "age", "amount"],
                "target": "target",
                "search_space": {"C": [0.1, 1.0]},
            },
            "tune",
        ),
        objects={"data:credit": credit_frame},
    )
    assert result["summary"]["trials"] == 1
    assert any(item["type"] == "hscredit-artifact" for item in result["artifacts"])


def test_hsrule_generates_rule_report(credit_frame, tmp_path):
    result = execute_skill(
        "hsrule",
        _request(
            tmp_path,
            "hsrule",
            "rule_report",
            {"data": {"kind": "object_ref", "ref": "data:credit"}},
            {"rules": ["score < 560"], "target": "target"},
            "rule",
        ),
        objects={"data:credit": credit_frame},
    )
    assert any(item["type"] == "excel" for item in result["artifacts"])


def test_hsselect_fits_and_publishes_selection_report(credit_frame, tmp_path):
    result = execute_skill(
        "hsselect",
        _request(
            tmp_path,
            "hsselect",
            "selector_fit",
            {"data": {"kind": "object_ref", "ref": "data:credit"}},
            {
                "selector": "VarianceSelector",
                "selector_params": {"threshold": 0.0},
                "target": "target",
                "features": ["score", "age", "amount"],
            },
            "selection",
        ),
        objects={"data:credit": credit_frame},
    )
    assert any(item["type"] == "json" for item in result["artifacts"])


def test_hsexcel_writes_a_workbook(credit_frame, tmp_path):
    result = execute_skill(
        "hsexcel",
        _request(
            tmp_path,
            "hsexcel",
            "dataframe_to_excel",
            {"data": {"kind": "object_ref", "ref": "data:credit"}},
            {"sheet_name": "数据"},
            "table",
        ),
        objects={"data:credit": credit_frame},
    )
    assert Path(next(item["path"] for item in result["artifacts"] if item["type"] == "excel")).is_file()


def test_hsviz_renders_a_public_plot(credit_frame, tmp_path):
    result = execute_skill(
        "hsviz",
        _request(
            tmp_path,
            "hsviz",
            "plot",
            {"score": {"kind": "object_ref", "ref": "score"}},
            {"plot": "hist_plot", "kwargs": {"bins": 5}},
            "hist",
        ),
        objects={"score": credit_frame["score"]},
    )
    assert any(item["type"] == "image" for item in result["artifacts"])
