"""构建和安装元数据契约测试。"""

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
