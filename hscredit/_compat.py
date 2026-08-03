"""第三方依赖版本兼容注册中心。

本模块集中读取已安装发行版版本，并通过明确、有上下界的版本矩阵决定是否
安装兼容适配。兼容路径不使用异常捕获、函数签名探测、错误文本解析或失败重试。
"""

from functools import wraps
from importlib import metadata, util
from typing import Optional

from packaging.version import Version


def installed_version(
    import_name: str, distribution_name: Optional[str] = None
) -> Optional[Version]:
    """返回已安装模块对应发行版的版本，模块不存在时返回 None。"""
    if util.find_spec(import_name) is None:
        return None
    return Version(metadata.version(distribution_name or import_name))


def needs_lightgbm_dask_pandas_compat(
    lightgbm_version: Optional[Version],
    pandas_version: Optional[Version],
    dask_version: Optional[Version],
) -> bool:
    """判断是否需要为旧 Dask 恢复 Pandas 字符串访问器别名。"""
    return (
        lightgbm_version is not None
        and Version("3.2.0") <= lightgbm_version < Version("4.7.0")
        and pandas_version is not None
        and pandas_version >= Version("2.0.0")
        and dask_version is not None
        and dask_version < Version("2023.2.0")
    )


def needs_lightgbm_sklearn_compat(
    lightgbm_version: Optional[Version], sklearn_version: Optional[Version]
) -> bool:
    """判断是否需要转换 LightGBM 调用 scikit-learn 的校验参数名。"""
    return (
        lightgbm_version is not None
        and lightgbm_version < Version("4.6.0")
        and sklearn_version is not None
        and sklearn_version >= Version("1.8.0")
    )


def needs_seaborn_pandas_compat(
    seaborn_version: Optional[Version], pandas_version: Optional[Version]
) -> bool:
    """判断是否需要为旧 Seaborn 恢复已移除的 Pandas 配置项。"""
    return (
        seaborn_version is not None
        and Version("0.11.0") <= seaborn_version < Version("0.12.2")
        and pandas_version is not None
        and pandas_version >= Version("2.0.0")
    )


def _wrap_lightgbm_validation_keyword(func):
    """把旧 LightGBM 校验参数名转换为 scikit-learn 1.8 的新名称。"""
    if getattr(func, "_hscredit_finite_compat", False):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
        return func(*args, **kwargs)

    wrapper._hscredit_finite_compat = True
    return wrapper


def install_lightgbm_sklearn_compat(
    lightgbm_module, lightgbm_version, sklearn_version
) -> None:
    """按明确版本矩阵适配 LightGBM 内部 scikit-learn 校验调用。"""
    if not needs_lightgbm_sklearn_compat(lightgbm_version, sklearn_version):
        return

    for module in (lightgbm_module.compat, lightgbm_module.sklearn):
        for attribute in ("_LGBMCheckXY", "_LGBMCheckArray"):
            checker = getattr(module, attribute)
            setattr(module, attribute, _wrap_lightgbm_validation_keyword(checker))


def normalize_seaborn_inf(values):
    """为旧 Seaborn/Pandas 组合恢复“无穷值视为缺失”的行为。"""
    seaborn_version = installed_version("seaborn")
    pandas_version = installed_version("pandas")
    if not needs_seaborn_pandas_compat(seaborn_version, pandas_version):
        return values

    import numpy as np
    import pandas as pd

    if isinstance(values, pd.Series):
        return values.replace([np.inf, -np.inf], np.nan)

    array = np.asarray(values)
    return np.where(np.isinf(array), np.nan, array)


def _install_pandas_string_methods_alias() -> None:
    """为仍引用旧路径的 Dask 恢复 Pandas 字符串访问器别名。"""
    import pandas as pd
    from pandas.core.strings.accessor import StringMethods

    pd.core.strings.StringMethods = StringMethods


def _install_pandas_inf_option_alias() -> None:
    """为旧 Seaborn 幂等注册 Pandas 2 已移除的配置项。"""
    from pandas._config.config import _registered_options, register_option

    if "mode.use_inf_as_null" not in _registered_options:
        register_option(
            "mode.use_inf_as_null",
            False,
            doc="Compatibility option for seaborn<0.12.2",
        )


def prepare_dependency(name: str) -> None:
    """根据依赖名和已安装版本，安装命中的兼容适配。"""
    import_name = name.split(".", 1)[0]

    if import_name == "lightgbm":
        lightgbm_version = installed_version("lightgbm")
        pandas_version = installed_version("pandas")
        dask_version = installed_version("dask")
        if needs_lightgbm_dask_pandas_compat(
            lightgbm_version, pandas_version, dask_version
        ):
            _install_pandas_string_methods_alias()
        return

    if import_name == "seaborn":
        seaborn_version = installed_version("seaborn")
        pandas_version = installed_version("pandas")
        if needs_seaborn_pandas_compat(seaborn_version, pandas_version):
            _install_pandas_inf_option_alias()


def prepare_runtime_compatibility() -> None:
    """在 hscredit 公开模块加载前准备已知依赖组合。"""
    prepare_dependency("lightgbm")
    prepare_dependency("seaborn")
