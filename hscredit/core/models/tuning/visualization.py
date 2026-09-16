"""自动绑定 Study 的 Optuna 原生可视化入口。"""

from functools import wraps
import importlib
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tuning import ModelTuner


class _OptunaVisualization:
    """按安装版本公开全部绘图函数，保留原生参数、返回值和错误语义。"""

    def __init__(self, tuner: "ModelTuner", module_name: str = "optuna.visualization"):
        self._tuner = tuner
        self._module_name = module_name

    def _module(self):
        return importlib.import_module(self._module_name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | {name for name in dir(self._module()) if not name.startswith("_")})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "matplotlib" and self._module_name == "optuna.visualization":
            return type(self)(self._tuner, "optuna.visualization.matplotlib")
        try:
            function = getattr(self._module(), name)
        except AttributeError as exc:
            raise AttributeError(
                f"当前 Optuna 可视化模块没有 {name!r}，请用 dir(tuner.visualization) 查看可用入口"
            ) from exc
        if not name.startswith("plot_") or not callable(function):
            return function

        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].name not in {"study", "studies"}:
            return function
        study_parameter = parameters[0]

        @wraps(function)
        def plot(*args, **kwargs):
            # 显式 study/studies 可用于原生的多 Study 对比，其余调用绑定当前搜索。
            if study_parameter.name in kwargs:
                return function(*args, **kwargs)
            study = self._tuner.get_study()
            if study_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[study_parameter.name] = study
                return function(*args, **kwargs)
            return function(study, *args, **kwargs)

        plot.__signature__ = signature.replace(parameters=parameters[1:])
        return plot
