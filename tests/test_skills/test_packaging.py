"""Skills 运行时和根目录 Skill 的包发现边界测试。"""

from pathlib import Path

from packaging.requirements import Requirement
from setuptools import find_namespace_packages

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[2]


def test_base_package_includes_parquet_skills_validation_and_ssh_support():
    """基础安装提供 Parquet、Skills 请求校验及加密 SSH 私钥支持。"""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {Requirement(value) for value in config["project"]["dependencies"]}
    extras = config["project"]["optional-dependencies"]

    assert any(req.name.lower() == "pyarrow" for req in dependencies)
    assert Requirement("jsonschema>=4.18") in dependencies
    assert Requirement("asyncssh[bcrypt]>=2.14.2,<3") in dependencies
    assert {"parquet", "skills", "db-ssh"}.isdisjoint(extras)


def test_setuptools_packages_runtime_but_not_root_skill_directories():
    """防止 GitHub Skill 目录污染通用顶层 Python 命名空间。"""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = config["tool"]["setuptools"]["packages"]["find"]["exclude"]

    packages = find_namespace_packages(where=str(ROOT), exclude=excluded)

    assert "hscredit.skills_runtime" in packages
    assert not any(package == "skills" or package.startswith("skills.") for package in packages)
    assert not any(package == "build" or package.startswith("build.") for package in packages)
    assert not any(package == "scripts" or package.startswith("scripts.") for package in packages)
