"""超参数调优子包.

包含基于Optuna的模型自动调参工具:
- ModelTuner: 模型调优器
- AutoTuner: 自动调优器
- TuningObjective: 调优目标

懒加载：仅在首次访问上述类名时才真正导入 optuna。
"""

import importlib

__all__ = [
    "ModelTuner",
    "AutoTuner",
    "TuningObjective",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(".tuning", __name__)
        value = getattr(module, name)
    except Exception:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value
