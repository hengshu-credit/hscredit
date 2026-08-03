"""第三方依赖显式版本兼容测试。"""

from packaging.version import Version

from hscredit._compat import (
    _install_pandas_inf_option_alias,
    _install_pandas_string_methods_alias,
    needs_lightgbm_dask_pandas_compat,
    needs_lightgbm_sklearn_compat,
    needs_seaborn_pandas_compat,
    prepare_dependency,
    prepare_runtime_compatibility,
)


def _version(value):
    """把测试字符串转换为版本对象，None 保持不变。"""
    return None if value is None else Version(value)


def test_lightgbm_dask_pandas_matrix_has_explicit_boundaries():
    """LightGBM、Dask、Pandas 适配只在明确的半开区间启用。"""
    assert needs_lightgbm_dask_pandas_compat(_version("3.2.0"), _version("2.0.0"), _version("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(_version("3.1.1"), _version("2.0.0"), _version("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(_version("4.7.0"), _version("2.0.0"), _version("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(_version("3.3.5"), _version("1.5.3"), _version("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(_version("3.3.5"), _version("2.0.0"), _version("2023.2.0"))
    assert not needs_lightgbm_dask_pandas_compat(_version("3.3.5"), _version("2.0.0"), None)


def test_lightgbm_sklearn_matrix_has_explicit_boundaries():
    """LightGBM 与 scikit-learn 适配使用明确版本上下界。"""
    assert needs_lightgbm_sklearn_compat(_version("4.5.0"), _version("1.8.0"))
    assert not needs_lightgbm_sklearn_compat(_version("4.6.0"), _version("1.8.0"))
    assert not needs_lightgbm_sklearn_compat(_version("4.5.0"), _version("1.7.2"))
    assert not needs_lightgbm_sklearn_compat(None, _version("1.8.0"))


def test_seaborn_pandas_matrix_has_explicit_boundaries():
    """Seaborn 与 Pandas 适配只覆盖已知不兼容组合。"""
    assert needs_seaborn_pandas_compat(_version("0.11.0"), _version("2.0.0"))
    assert needs_seaborn_pandas_compat(_version("0.12.1"), _version("2.3.3"))
    assert not needs_seaborn_pandas_compat(_version("0.12.2"), _version("2.3.3"))
    assert not needs_seaborn_pandas_compat(_version("0.11.2"), _version("1.5.3"))
    assert not needs_seaborn_pandas_compat(None, _version("2.3.3"))


def test_prepare_dependency_uses_only_explicit_version_matrix(monkeypatch):
    """命中版本矩阵时只安装对应适配，不依赖导入失败来判断。"""
    import hscredit._compat as compat

    versions = {
        "lightgbm": Version("3.3.5"),
        "pandas": Version("2.3.3"),
        "dask": Version("2022.7.0"),
        "seaborn": Version("0.11.2"),
        "sklearn": Version("1.0.2"),
    }
    installed = []
    monkeypatch.setattr(compat, "installed_version", lambda name, distribution_name=None: versions.get(name))
    monkeypatch.setattr(compat, "_install_pandas_string_methods_alias", lambda: installed.append("strings"))
    monkeypatch.setattr(compat, "_install_pandas_inf_option_alias", lambda: installed.append("inf_option"))

    prepare_dependency("lightgbm")
    prepare_dependency("seaborn")

    assert installed == ["strings", "inf_option"]


def test_prepare_dependency_does_not_install_outside_version_matrix(monkeypatch):
    """不命中版本矩阵时不做任何补丁动作。"""
    import hscredit._compat as compat

    versions = {
        "lightgbm": Version("4.7.0"),
        "pandas": Version("2.3.3"),
        "dask": Version("2023.2.0"),
        "seaborn": Version("0.12.2"),
        "sklearn": Version("1.8.0"),
    }
    installed = []
    monkeypatch.setattr(compat, "installed_version", lambda name, distribution_name=None: versions.get(name))
    monkeypatch.setattr(compat, "_install_pandas_string_methods_alias", lambda: installed.append("strings"))
    monkeypatch.setattr(compat, "_install_pandas_inf_option_alias", lambda: installed.append("inf_option"))

    prepare_dependency("lightgbm")
    prepare_dependency("seaborn")

    assert installed == []


def test_pandas_compat_installers_are_idempotent():
    """重复准备兼容环境不会重复注册配置或改变别名语义。"""
    import pandas as pd
    from pandas._config.config import _registered_options
    from pandas.core.strings.accessor import StringMethods

    _install_pandas_string_methods_alias()
    _install_pandas_string_methods_alias()
    _install_pandas_inf_option_alias()
    _install_pandas_inf_option_alias()

    assert pd.core.strings.StringMethods is StringMethods
    assert "mode.use_inf_as_null" in _registered_options


def test_prepare_runtime_compatibility_prepares_supported_dependencies(monkeypatch):
    """包初始化入口统一准备 LightGBM 和 Seaborn 兼容。"""
    import hscredit._compat as compat

    prepared = []
    monkeypatch.setattr(compat, "prepare_dependency", prepared.append)

    prepare_runtime_compatibility()

    assert prepared == ["lightgbm", "seaborn"]
