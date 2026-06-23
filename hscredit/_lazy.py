"""轻量级延迟导入工具.

用于把 seaborn、statsmodels 等较重的可选/绘图依赖延迟到**首次实际使用时**才导入，
从而避免 ``import hscredit`` 时即加载这些库及其传递依赖（如 ipywidgets/IPython），
显著降低顶层导入耗时。

与模块级 ``__getattr__``（PEP 562）不同，本代理对象会被赋值给模块全局名（如 ``sns``），
因此模块内部函数中以裸名方式使用（``sns.heatmap(...)``）也能正确触发延迟导入，
不会出现 ``NameError``。
"""

import importlib
from types import ModuleType
from typing import Optional


class LazyModule(ModuleType):
    """延迟导入的模块代理.

    首次访问任意属性时才真正 ``import`` 目标模块，之后行为与真实模块一致。

    **参数**

    :param import_name: 目标模块的导入名，如 ``"seaborn"``、``"statsmodels.api"``

    **参考样例**

    >>> sns = LazyModule("seaborn")  # 此处不会导入 seaborn
    >>> sns.heatmap(...)             # 首次访问属性时才导入
    """

    def __init__(self, import_name: str):
        super().__init__(import_name)
        # 用对象 __dict__ 直接存储，避免触发自定义 __getattr__
        self.__dict__["_lazy_import_name"] = import_name
        self.__dict__["_lazy_module"] = None  # type: Optional[ModuleType]

    def _load(self) -> ModuleType:
        module = self.__dict__["_lazy_module"]
        if module is None:
            module = importlib.import_module(self.__dict__["_lazy_import_name"])
            self.__dict__["_lazy_module"] = module
        return module

    def __getattr__(self, attr):
        # __getattr__ 仅在常规查找失败时调用，因此 _lazy_* 等已存于 __dict__ 的名不会进入这里
        return getattr(self._load(), attr)
