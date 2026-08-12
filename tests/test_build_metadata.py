"""构建和安装元数据契约测试。"""

import runpy
import sys
from pathlib import Path

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_build_backend_has_no_legacy_helper_dependencies():
    build_system = _pyproject()["build-system"]

    assert build_system == {
        "requires": ["setuptools>=77"],
        "build-backend": "setuptools.build_meta",
    }


def test_project_uses_spdx_license_string():
    assert _pyproject()["project"]["license"] == "MIT"


def test_packaging_is_an_unversioned_direct_runtime_dependency():
    requirements = [Requirement(item) for item in _pyproject()["project"]["dependencies"]]
    packaging = [item for item in requirements if item.name.lower() == "packaging"]

    assert len(packaging) == 1
    assert str(packaging[0].specifier) == ""


def test_build_tools_are_not_runtime_dependencies():
    names = {Requirement(item).name.lower() for item in _pyproject()["project"]["dependencies"]}

    assert names.isdisjoint({"setuptools", "wheel", "build", "pkg-resources"})


def test_environment_validator_checks_packaging():
    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "validate_environment.py"))

    assert "packaging" in namespace["REQUIRED_MODULES"]


def test_ci_covers_setuptools_with_and_without_pkg_resources():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'setuptools-version: "77.0.3"' in workflow
    assert 'setuptools-version: "82.0.1"' in workflow
    assert 'setuptools-version: "latest"' in workflow


def test_ci_runs_full_suite_with_all_extras_on_every_supported_python():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]' in workflow
    assert workflow.count('pip install -e ".[all]"') == 2
    assert 'pip install -e ".[dev' not in workflow
    assert 'test-full:' not in workflow
    assert '-m "not slow and not integration"' not in workflow
    assert 'run: pytest tests/ --tb=short' in workflow
    assert 'HSCREDIT_STRICT_CI_SKIPS: "1"' in workflow


def test_pmml_extra_does_not_install_incompatible_sklearn_pandas():
    """PMML 导出使用 sklearn 原生组合器，不得再安装已不兼容新版 sklearn 的桥接包。"""
    names = {
        Requirement(item).name.lower()
        for item in _pyproject()["project"]["optional-dependencies"]["pmml"]
    }

    assert "sklearn-pandas" not in names
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "sklearn-pandas" not in requirements
