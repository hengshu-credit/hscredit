"""Skills 运行时和根目录 Skill 的包发现边界测试。"""

from pathlib import Path

from packaging.requirements import Requirement
from setuptools import find_namespace_packages

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[2]


def test_parquet_engine_is_installed_with_the_base_package():
    """防止默认安装缺失 Parquet 读写引擎。"""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {Requirement(value).name.lower() for value in config["project"]["dependencies"]}
    extras = config["project"]["optional-dependencies"]

    assert "pyarrow" in dependencies
    assert "parquet" not in extras


def test_setuptools_packages_runtime_but_not_root_skill_directories():
    """防止 GitHub Skill 目录污染通用顶层 Python 命名空间。"""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = config["tool"]["setuptools"]["packages"]["find"]["exclude"]

    packages = find_namespace_packages(where=str(ROOT), exclude=excluded)

    assert "hscredit.skills_runtime" in packages
    assert not any(package == "skills" or package.startswith("skills.") for package in packages)
    assert not any(package == "build" or package.startswith("build.") for package in packages)
    assert not any(package == "scripts" or package.startswith("scripts.") for package in packages)
