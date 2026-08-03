"""第三方依赖显式版本兼容测试。"""

import importlib
from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(
    "module_name, attribute",
    [
        ("hscredit.core.models.boosting", "LightGBMRiskModel"),
        ("hscredit.core.models.tuning", "ModelTuner"),
    ],
)
def test_lazy_loader_preserves_real_import_error(monkeypatch, module_name, attribute):
    """依赖内部错误不能被伪装成公开属性不存在。"""
    module = importlib.import_module(module_name)
    monkeypatch.delattr(module, attribute, raising=False)
    marker = RuntimeError("真实依赖错误")

    def fail_import(*args, **kwargs):
        raise marker

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="真实依赖错误"):
        module.__getattr__(attribute)

    assert attribute not in module.__dict__


def test_lightgbm_fit_strategy_is_version_based():
    """LightGBM 4.0 是旧参数与 callbacks API 的唯一分界。"""
    from hscredit.core.models.boosting.lightgbm_model import _lightgbm_fit_api

    assert _lightgbm_fit_api(Version("3.3.5")) == "legacy"
    assert _lightgbm_fit_api(Version("3.99.0")) == "legacy"
    assert _lightgbm_fit_api(Version("4.0.0")) == "callbacks"
    assert _lightgbm_fit_api(Version("4.6.0")) == "callbacks"


def test_lightgbm_sklearn_adapter_renames_keyword_in_version_matrix():
    """命中矩阵时将旧参数名转为 sklearn 1.8 的新参数名。"""
    from hscredit._compat import install_lightgbm_sklearn_compat

    received = []

    def check_xy(*args, **kwargs):
        received.append(kwargs)
        return args

    compat_module = SimpleNamespace(_LGBMCheckXY=check_xy, _LGBMCheckArray=check_xy)
    sklearn_module = SimpleNamespace(_LGBMCheckXY=check_xy, _LGBMCheckArray=check_xy)
    lightgbm_module = SimpleNamespace(compat=compat_module, sklearn=sklearn_module)

    install_lightgbm_sklearn_compat(lightgbm_module, Version("4.5.0"), Version("1.8.0"))
    install_lightgbm_sklearn_compat(lightgbm_module, Version("4.5.0"), Version("1.8.0"))
    lightgbm_module.compat._LGBMCheckXY("X", force_all_finite=False)

    assert received == [{"ensure_all_finite": False}]


def test_lightgbm_sklearn_adapter_is_noop_outside_version_matrix():
    """越界版本保持 LightGBM 原绑定不变。"""
    from hscredit._compat import install_lightgbm_sklearn_compat

    def check_xy(*args, **kwargs):
        return args, kwargs

    compat_module = SimpleNamespace(_LGBMCheckXY=check_xy, _LGBMCheckArray=check_xy)
    sklearn_module = SimpleNamespace(_LGBMCheckXY=check_xy, _LGBMCheckArray=check_xy)
    lightgbm_module = SimpleNamespace(compat=compat_module, sklearn=sklearn_module)

    install_lightgbm_sklearn_compat(lightgbm_module, Version("4.6.0"), Version("1.8.0"))

    assert lightgbm_module.compat._LGBMCheckXY is check_xy
    assert lightgbm_module.sklearn._LGBMCheckArray is check_xy


def test_lazy_module_prepares_dependency_before_import(monkeypatch):
    """延迟加载重依赖前必须先按版本准备兼容环境。"""
    import hscredit._lazy as lazy

    events = []
    loaded = SimpleNamespace(name="seaborn")
    monkeypatch.setattr(lazy, "prepare_dependency", lambda name: events.append(("prepare", name)))
    monkeypatch.setattr(
        lazy.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or loaded,
    )

    proxy = lazy.LazyModule("seaborn")

    assert proxy._load() is loaded
    assert events == [("prepare", "seaborn"), ("import", "seaborn")]


def test_normalize_seaborn_inf_preserves_series_index_and_order(monkeypatch):
    """旧 Seaborn 组合只替换无穷值，不重排或重置索引。"""
    import numpy as np
    import pandas as pd
    import hscredit._compat as compat

    versions = {"seaborn": Version("0.11.2"), "pandas": Version("2.3.3")}
    monkeypatch.setattr(compat, "installed_version", lambda name, distribution_name=None: versions.get(name))
    score = pd.Series([0.1, np.inf, 0.4, -np.inf, 0.8], index=[5, 4, 3, 2, 1])
    original = score.copy(deep=True)

    normalized = compat.normalize_seaborn_inf(score)

    pd.testing.assert_series_equal(score, original)
    assert normalized.index.tolist() == [5, 4, 3, 2, 1]
    assert normalized.iloc[[0, 2, 4]].tolist() == [0.1, 0.4, 0.8]
    assert normalized.iloc[[1, 3]].isna().all()


def test_normalize_seaborn_inf_is_noop_outside_version_matrix(monkeypatch):
    """新 Seaborn 版本由上游原生处理，兼容层不复制数据。"""
    import pandas as pd
    import hscredit._compat as compat

    versions = {"seaborn": Version("0.12.2"), "pandas": Version("2.3.3")}
    monkeypatch.setattr(compat, "installed_version", lambda name, distribution_name=None: versions.get(name))
    score = pd.Series([1.0, 2.0], index=[2, 1])

    assert compat.normalize_seaborn_inf(score) is score
