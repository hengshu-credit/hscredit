"""第三方依赖版本兼容注册中心。

本模块集中读取已安装发行版版本，并通过明确、有上下界的版本矩阵决定是否
安装兼容适配。兼容路径不使用异常捕获、函数签名探测、错误文本解析或失败重试。
"""

from functools import wraps
from importlib import metadata, util
from typing import Optional

from packaging.version import Version


def installed_version(import_name: str, distribution_name: Optional[str] = None) -> Optional[Version]:
    """返回已安装模块对应发行版的版本，模块不存在时返回 None。"""
    if util.find_spec(import_name) is None:
        return None
    return Version(metadata.version(distribution_name or import_name))


def pandas_styler_map_api(pandas_version: Version) -> str:
    """pandas 2.1 起使用 Styler.map；2.0 使用 applymap（3.0 已移除）。"""
    if pandas_version < Version("2.0"):
        raise ValueError("规则表样式要求 pandas>=2.0")
    return "applymap" if pandas_version < Version("2.1") else "map"


def map_styler(styler, func, subset=None, **kwargs):
    """按 pandas 版本应用逐单元格样式，原样传递参数和用户函数异常。"""
    import pandas as pd

    method = pandas_styler_map_api(Version(pd.__version__))
    return getattr(styler, method)(func, subset=subset, **kwargs)


class _LegacyColormapRegistry:
    """为 Matplotlib 3.3/3.4 提供 pandas 使用的只读色图查询接口。"""

    def __init__(self, get_cmap):
        self.get_cmap = get_cmap

    def __getitem__(self, name):
        return self.get_cmap(name)


def install_matplotlib_colormap_compat(matplotlib_module, matplotlib_version: Version) -> None:
    """为旧 Matplotlib 补齐 pandas 2/3 的 Styler 色图查询接口。

    3.5 引入 colormaps 注册表，3.6.1 才引入注册表的 get_cmap。
    只适配声明支持的 3.3 <= version < 3.6.1，不改动现代版本。
    此适配在请求规则表样式时安装，不在包初始化时提前导入绘图库。
    """
    if not Version("3.3") <= matplotlib_version < Version("3.6.1"):
        return
    if getattr(matplotlib_module, "_hscredit_colormap_compat", False):
        return

    from copy import copy

    def get_cmap(cmap=None):
        # 旧版 cm.get_cmap 也接受 Colormap 和 None；返回副本以免改变全局色图。
        return copy(matplotlib_module.cm.get_cmap(cmap))

    if matplotlib_version < Version("3.5"):
        matplotlib_module.colormaps = _LegacyColormapRegistry(get_cmap)
    else:
        matplotlib_module.colormaps.get_cmap = get_cmap
    matplotlib_module._hscredit_colormap_compat = True


def pandas_apply_api(pandas_version: Version) -> str:
    """按明确版本边界选择 pandas apply 内部接口。

    pandas 2.1 引入 by_row，2.2 引入 engine/include_groups 并将 grouper
    移至 _grouper；3.0 移除 convert_dtype、GroupBy axis 和内置函数替换。
    版本规则同时用于参数构造和结果装配，不通过执行用户函数来探测接口。
    """
    if pandas_version < Version("2.0"):
        raise ValueError("pandas apply 并行代理要求 pandas>=2.0")
    if pandas_version < Version("2.1"):
        return "2.0"
    if pandas_version < Version("2.2"):
        return "2.1"
    if pandas_version < Version("3.0"):
        return "2.2"
    return "3.0"


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


def needs_lightgbm_sklearn_compat(lightgbm_version: Optional[Version], sklearn_version: Optional[Version]) -> bool:
    """判断是否需要转换 LightGBM 调用 scikit-learn 的校验参数名。"""
    return (
        lightgbm_version is not None
        and lightgbm_version < Version("4.6.0")
        and sklearn_version is not None
        and sklearn_version >= Version("1.6.0")
    )


def needs_lightgbm_verbosity_compat(lightgbm_version: Optional[Version]) -> bool:
    """判断是否需要保留 LightGBM 4.0–4.4 重置参数时的日志级别。"""
    return lightgbm_version is not None and Version("4.0.0") <= lightgbm_version < Version("4.5.0")


def needs_logistic_regression_parallel_compat(sklearn_version: Optional[Version]) -> bool:
    """sklearn 1.8 至 1.10 移除前不再为逻辑回归注入无效的 n_jobs。"""
    return sklearn_version is not None and Version("1.8") <= sklearn_version < Version("1.10")


def needs_seaborn_pandas_compat(seaborn_version: Optional[Version], pandas_version: Optional[Version]) -> bool:
    """判断是否需要为旧 Seaborn 恢复已移除的 Pandas 配置项。"""
    return (
        seaborn_version is not None
        and Version("0.11.0") <= seaborn_version < Version("0.12.2")
        and pandas_version is not None
        and pandas_version >= Version("2.0.0")
    )


def _wrap_lightgbm_validation_keyword(func):
    """把旧 LightGBM 校验参数名转换为 scikit-learn 1.6 起支持的新名称。"""
    if getattr(func, "_hscredit_finite_compat", False):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
        return func(*args, **kwargs)

    wrapper._hscredit_finite_compat = True
    return wrapper


def install_lightgbm_sklearn_compat(lightgbm_module, lightgbm_version, sklearn_version) -> None:
    """按明确版本矩阵适配 LightGBM 内部 scikit-learn 校验调用。"""
    if not needs_lightgbm_sklearn_compat(lightgbm_version, sklearn_version):
        return

    for module in (lightgbm_module.compat, lightgbm_module.sklearn):
        for attribute in ("_LGBMCheckXY", "_LGBMCheckArray"):
            checker = getattr(module, attribute)
            setattr(module, attribute, _wrap_lightgbm_validation_keyword(checker))


def install_lightgbm_verbosity_compat(lightgbm_module, lightgbm_version) -> None:
    """为旧 LightGBM 保留重置参数之前显式配置的日志级别。

    LightGBM 4.0–4.4 的自定义目标函数会执行 ``reset_parameter`` ，其 C++
    实现在没有日志参数时意外重置为 INFO，导致 ``verbose=False`` 失效。
    4.5.0 已通过上游 PR #6428 修复；这里仅给受影响版本补传已有日志设置，
    不修改全局 logger，也不拦截 Python 警告。显式重置日志级别仍优先。
    """
    if not needs_lightgbm_verbosity_compat(lightgbm_version):
        return
    reset_parameter = lightgbm_module.Booster.reset_parameter
    if getattr(reset_parameter, "_hscredit_verbosity_compat", False):
        return

    @wraps(reset_parameter)
    def wrapper(self, params):
        updated = dict(params)
        if "verbosity" not in updated and "verbose" not in updated:
            for name in ("verbosity", "verbose"):
                if name in self.params:
                    updated[name] = self.params[name]
                    break
        result = reset_parameter(self, updated)
        # 原生方法会更新 self.params；移除旧别名，确保后续重置沿用本次显式设置。
        if "verbosity" in updated:
            self.params.pop("verbose", None)
        elif "verbose" in updated:
            self.params.pop("verbosity", None)
        return result

    wrapper._hscredit_verbosity_compat = True
    lightgbm_module.Booster.reset_parameter = wrapper


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

    if import_name == "matplotlib":
        matplotlib_version = installed_version("matplotlib")
        if matplotlib_version is not None and Version("3.3") <= matplotlib_version < Version("3.6.1"):
            import matplotlib
            import matplotlib.cm

            install_matplotlib_colormap_compat(matplotlib, matplotlib_version)
        return

    if import_name == "lightgbm":
        lightgbm_version = installed_version("lightgbm")
        pandas_version = installed_version("pandas")
        dask_version = installed_version("dask")
        if needs_lightgbm_dask_pandas_compat(lightgbm_version, pandas_version, dask_version):
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
