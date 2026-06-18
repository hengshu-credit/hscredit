"""提升树模型子包.

包含基于梯度提升的风控模型:
- XGBoostRiskModel
- LightGBMRiskModel
- CatBoostRiskModel
- NGBoostRiskModel

各模型均为懒加载：仅在首次访问对应类名时才真正导入
xgboost/lightgbm/catboost/ngboost，避免 `import hscredit` 时
背负全部重依赖的加载耗时。
"""

import importlib

__all__ = ["XGBoostRiskModel", "LightGBMRiskModel", "CatBoostRiskModel", "NGBoostRiskModel"]

_LAZY_SUBMODULES = {
    "XGBoostRiskModel": ".xgboost_model",
    "LightGBMRiskModel": ".lightgbm_model",
    "CatBoostRiskModel": ".catboost_model",
    "NGBoostRiskModel": ".ngboost_model",
}


def __getattr__(name):
    module_name = _LAZY_SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
    except Exception:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value
