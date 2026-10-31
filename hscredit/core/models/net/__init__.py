"""表格神经网络与预训练基础模型；访问模型类时才加载对应模块。"""

from importlib import import_module

__all__ = ["ResNet", "FTTransformer", "TabM", "TabPFN", "TabNet"]
_MODULES = {
    "ResNet": ".resnet",
    "FTTransformer": ".ft_transformer",
    "TabM": ".tabm",
    "TabPFN": ".tabpfn",
    "TabNet": ".tabnet",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
