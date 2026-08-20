"""提升树模型子包.

包含基于梯度提升的风控模型:
- XGBoost
- LightGBM
- CatBoost
- NGBoost

各模型均为懒加载：仅在首次访问对应类名时才真正导入
xgboost/lightgbm/catboost/ngboost，避免 `import hscredit` 时
背负全部重依赖的加载耗时。
"""

import importlib

__all__ = ["XGBoost", "LightGBM", "CatBoost", "NGBoost"]

_LAZY_SUBMODULES = {
    "XGBoost": ".xgboost_model",
    "LightGBM": ".lightgbm_model",
    "CatBoost": ".catboost_model",
    "NGBoost": ".ngboost_model",
}


def __getattr__(name):
    module_name = _LAZY_SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
