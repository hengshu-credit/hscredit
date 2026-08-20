"""模型解释基础依赖的安装契约测试。"""

from importlib import metadata


def test_shap_is_base_dependency_and_explain_extra_is_removed():
    """SHAP 应随基础包安装，且不再暴露 explain 可选依赖。"""
    requirements = metadata.requires("hscredit") or []
    shap_requirements = [item for item in requirements if item.lower().startswith("shap")]

    assert any("<0.50" in item and 'python_version < "3.14"' in item for item in shap_requirements)
    assert any("<0.53" in item and 'python_version >= "3.14"' in item for item in shap_requirements)
    assert not any('extra == "explain"' in item for item in requirements)
