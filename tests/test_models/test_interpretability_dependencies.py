"""模型解释基础依赖的安装契约测试。"""

from importlib import metadata

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest


def _active_requirements(name, python_version, extra=""):
    """按声明支持的解释器和可选能力解析真实发布元数据。"""
    environment = dict(default_environment(), python_version=python_version, extra=extra)
    return [
        requirement
        for requirement in map(Requirement, metadata.requires("hscredit") or [])
        if requirement.name == name and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]


def test_shap_is_base_dependency_and_explain_extra_is_removed():
    """SHAP 应随基础包安装，且不再暴露 explain 可选依赖。"""
    requirements = metadata.requires("hscredit") or []
    assert _active_requirements("shap", "3.11")
    assert not any('extra == "explain"' in item for item in requirements)


@pytest.mark.parametrize("python_version", ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"])
def test_shap_and_xgboost_resolve_compatible_versions_for_supported_python(python_version):
    """旧 Python 保留可安装 SHAP，新 Python 支持向量截距且避开 0.52 类别误判。"""
    shap_requirements = _active_requirements("shap", python_version)
    xgboost_requirements = _active_requirements("xgboost", python_version, extra="boost")
    assert len(shap_requirements) == len(xgboost_requirements) == 1
    shap_versions = shap_requirements[0].specifier
    xgboost_versions = xgboost_requirements[0].specifier

    if python_version in {"3.9", "3.10"}:
        assert "0.49.1" in shap_versions
        assert "0.50.0" not in shap_versions
        assert "2.1.4" in xgboost_versions
        assert "3.0.5" in xgboost_versions
        assert "3.1.0" not in xgboost_versions
    else:
        assert "0.51.0" in shap_versions
        assert "0.49.1" not in shap_versions
        assert "0.52.0" not in shap_versions
        assert "3.4.1" in xgboost_versions


@pytest.mark.parametrize("python_version", ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"])
def test_pmml_restricts_sklearn_without_restricting_core_users(python_version):
    """仅 PMML extra 排除已验证不兼容的 sklearn；基础安装仍允许新版。"""
    core_requirements = _active_requirements("scikit-learn", python_version)
    pmml_requirements = _active_requirements("scikit-learn", python_version, extra="pmml")
    assert core_requirements and pmml_requirements
    assert all("1.9.1" in requirement.specifier for requirement in core_requirements)
    assert all("1.8.0" in requirement.specifier for requirement in pmml_requirements)
    assert not all("1.9.1" in requirement.specifier for requirement in pmml_requirements)
    exporter = _active_requirements("sklearn2pmml", python_version, extra="pmml")
    assert len(exporter) == 1
    assert "0.125.0" in exporter[0].specifier
    assert "0.124.0" not in exporter[0].specifier
