"""超参数调优子包.

包含基于Optuna的模型自动调参工具:
- ModelTuner: 模型调优器
- AutoTuner: 自动调优器
- TuningObjective: 调优目标
- TuningSampler: 采样器码表（optuna 内置 + optunahub）
- normalize_search_space: 多框架搜索空间格式统一转换
- search_space: 跨框架同名符号入口（suggest_int / Real / uniform 等），
  支持 ``from hscredit.core.models.tuning.search_space import *`` 直接使用

懒加载：仅在首次访问上述类名时才真正导入 optuna。
"""

import importlib

__all__ = [
    "ModelTuner",
    "AutoTuner",
    "TuningObjective",
    "TuningSampler",
    "normalize_search_space",
    # search_space 同名符号
    "Dimension",
    "Real",
    "Integer",
    "Categorical",
    "IntDistribution",
    "FloatDistribution",
    "CategoricalDistribution",
    "suggest_int",
    "suggest_float",
    "suggest_categorical",
    "suggest_uniform",
    "suggest_discrete_uniform",
    "suggest_loguniform",
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "choice",
    "randint",
    "normal",
    "qnormal",
    "lognormal",
    "qlognormal",
]

# search_space 模块不依赖 optuna，可立即导入其符号到子包命名空间
from .search_space import (  # noqa: F401,E402
    Dimension,
    Real,
    Integer,
    Categorical,
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
    suggest_int,
    suggest_float,
    suggest_categorical,
    suggest_uniform,
    suggest_discrete_uniform,
    suggest_loguniform,
    uniform,
    loguniform,
    quniform,
    qloguniform,
    choice,
    randint,
    normal,
    qnormal,
    lognormal,
    qlognormal,
)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    module = importlib.import_module(".tuning", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
