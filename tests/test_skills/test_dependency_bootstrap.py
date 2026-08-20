"""Skills 依赖隔离和引导测试。"""

import json
import subprocess
import sys

import pytest

from hscredit.skills_runtime.bootstrap import install_requirement
from hscredit.skills_runtime.dependencies import environment_key, plan_environment, resolve_required_extras
from hscredit.skills_runtime.errors import SkillExecutionError


def test_current_environment_install_requires_explicit_permission():
    """防止缺失依赖时静默修改当前代码环境。"""
    request = {"environment": {"mode": "current", "install_missing": False}}

    with pytest.raises(SkillExecutionError) as exc_info:
        plan_environment("hsbin", "feature_bin_stats", request, missing_extras=("skills",))

    assert exc_info.value.code == "DEPENDENCY_MISSING"


def test_object_ref_cannot_be_moved_to_an_isolated_interpreter():
    """防止实时 Python 对象被错误假定为可跨解释器传递。"""
    request = {
        "inputs": {"binner": {"kind": "object_ref", "ref": "binner:fitted"}},
        "environment": {"mode": "isolated", "install_missing": True},
    }

    with pytest.raises(SkillExecutionError) as exc_info:
        plan_environment(
            "hsbin",
            "optimal_binning_transform",
            request,
            missing_extras=("skills",),
        )

    assert exc_info.value.code == "OBJECT_REF_REQUIRES_CURRENT_ENV"


def test_parquet_input_uses_only_the_operation_extras():
    """防止请求把任意包名注入自动安装列表。"""
    request = {
        "inputs": {"data": {"kind": "file", "path": "sample.parquet"}},
        "parameters": {"extras": ["malicious-package"]},
    }

    extras = resolve_required_extras("hsbin", "feature_bin_stats", request)

    assert extras == ("skills",)


def test_environment_key_is_order_independent():
    """防止相同依赖集合重复创建隔离环境。"""
    first = environment_key("C:/repo", ("skills", "tune"), "3.14")
    second = environment_key("C:/repo", ("tune", "skills"), "3.14")

    assert first == second
    assert len(first) == 16


def test_install_requirement_uses_a_real_isolated_venv(tmp_path):
    """防止隔离安装只创建目录却仍从当前解释器导入依赖。"""
    package_dir = tmp_path / "tiny_dep"
    module_dir = package_dir / "hscredit_skill_test_dep"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("VALUE = 73\n", encoding="utf-8")
    (package_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=77"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[project]",
                'name = "hscredit-skill-test-dep"',
                'version = "0.0.1"',
            ]
        ),
        encoding="utf-8",
    )
    environment_dir = tmp_path / "venv"

    python = install_requirement(environment_dir, str(package_dir))
    result = subprocess.run(
        [str(python), "-c", "import hscredit_skill_test_dep as dep; print(dep.VALUE)"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, json.dumps({"stdout": result.stdout, "stderr": result.stderr})
    assert result.stdout.strip() == "73"
    assert str(python) != sys.executable


@pytest.mark.parametrize("skill", ["hsbin", "hsreport"])
def test_self_contained_launcher_exposes_help_without_importing_hscredit(skill):
    """防止在线安装后的 launcher 在查看用法时就要求预装 hscredit。"""
    result = subprocess.run(
        [sys.executable, f"skills/{skill}/scripts/run.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "request" in result.stdout
    assert "--debug" in result.stdout
