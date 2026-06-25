"""超参数调优子包.

包含基于Optuna的模型自动调参工具:
- ModelTuner: 模型调优器
- AutoTuner: 自动调优器
- TuningObjective: 调优目标
- TuningSampler: 采样器码表（optuna 内置 + optunahub）

懒加载：仅在首次访问上述类名时才真正导入 optuna。
"""

import importlib

__all__ = [
    "ModelTuner",
    "AutoTuner",
    "TuningObjective",
    "TuningSampler",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    try:
        module = importlib.import_module(".tuning", __name__)
        value = getattr(module, name)
    except Exception:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}") from None
    globals()[name] = value
    return value
