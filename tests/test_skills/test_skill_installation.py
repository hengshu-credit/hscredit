"""Skill 目录、Schema 和仓库安装器测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).parents[2]
PLANNED = ("hscredit", "hsmodel", "hsrule", "hsselect", "hsviz", "hsexcel")


def _install(target, *args):
    return subprocess.run(
        [sys.executable, "skills/install.py", *args, "--target-dir", str(target)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_repository_installer_installs_only_the_requested_skill(tmp_path):
    """防止部分安装夹带用户未选择的 Skill。"""
    result = _install(tmp_path, "--skills", "hsbin")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "hsbin" / "SKILL.md").is_file()
    assert not (tmp_path / "hsreport").exists()


def test_repository_installer_suite_installs_only_implemented_skills(tmp_path):
    """防止规划中目录被当作已实现 Skill 安装。"""
    result = _install(tmp_path, "--suite", "hscredit")

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["hsbin", "hsreport"]


def test_repository_installer_preserves_an_existing_directory(tmp_path):
    """防止重复安装覆盖用户已经修改的 Skill。"""
    destination = tmp_path / "hsbin"
    destination.mkdir()
    marker = destination / "user.txt"
    marker.write_text("keep", encoding="utf-8")

    result = _install(tmp_path, "--skills", "hsbin")

    assert result.returncode != 0
    assert "已存在" in f"{result.stdout}\n{result.stderr}"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_planned_skill_directories_are_non_discoverable_placeholders():
    """防止空壳目录被 Agent 误识别为可调用 Skill。"""
    for name in PLANNED:
        directory = ROOT / "skills" / name
        assert directory.is_dir(), name
        assert sorted(path.name for path in directory.iterdir()) == [".gitkeep"]
        assert not (directory / "SKILL.md").exists()


@pytest.mark.parametrize(
    ("skill", "request_payload"),
    [
        (
            "hsbin",
            {
                "version": "1",
                "operation": "feature_bin_stats",
                "inputs": {"data": {"kind": "file", "path": "sample.csv"}},
                "parameters": {"feature": "score", "target": "target"},
                "output": {"directory": "outputs", "name": "stats", "overwrite": False},
            },
        ),
        (
            "hsreport",
            {
                "version": "1",
                "operation": "auto_feature_analysis",
                "inputs": {"data": {"kind": "file", "path": "sample.xlsx", "sheet_name": "样本"}},
                "parameters": {"features": ["score"], "target": "target"},
                "output": {"directory": "outputs", "name": "report", "overwrite": False},
            },
        ),
    ],
)
def test_skill_request_schema_accepts_a_real_request(skill, request_payload):
    """防止发布的 Schema 与真实运行时信封不一致。"""
    schema_path = ROOT / "skills" / skill / "schemas" / "request.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    errors = list(Draft202012Validator(schema).iter_errors(request_payload))

    assert errors == []
