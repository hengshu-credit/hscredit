"""Optuna超参数调优接口 - 基于内部建模经验优化.

提供统一的超参数调优功能，支持所有风控模型。
搜索空间基于内部建模经验优化，适配不同样本量和特征数。

支持功能:
1. 单目标优化（如KS、AUC）
2. 多目标优化（如同时优化KS和稳定性）
3. 自定义评估指标
4. 指定trials_point评估超参数空间内特定点的模型效果

**依赖**
pip install optuna

**参考样例**
>>> from hscredit.core.models import XGBoost, ModelTuner
>>>
>>> # 定义搜索空间
>>> search_space = {
...     'max_depth': {'type': 'int', 'low': 3, 'high': 10},
...     'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
...     'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
... }
>>>
>>> # sklearn风格
>>> tuner = ModelTuner(
...     model_class=XGBoost,
...     search_space=search_space,
...     metric='ks',
...     direction='maximize'
... )
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)
>>>
>>> # scorecardpipeline风格
>>> tuner = ModelTuner(
...     model_class=XGBoost,
...     search_space=search_space,
...     metric='ks',
...     target='label'
... )
>>> best_params = tuner.fit(df, n_trials=100)
>>>
>>> # 多目标调优（KS + 稳定性）
>>> tuner = ModelTuner(
...     model_class=XGBoost,
...     search_space=search_space,
...     metric=['ks', 'ks_diff'],
...     direction=['maximize', 'minimize'],
... )
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)

>>> # 自定义metric
>>> def custom_metric(y_true, y_pred):
...     return some_score(y_true, y_pred)
>>> 
>>> tuner = ModelTuner(
...     model_class=XGBoost,
...     search_space=search_space,
...     metric=custom_metric,
...     direction='maximize'
... )
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)

>>> # 评估特定超参数点（sklearn风格）
    >>> trial_points = [
    ...     {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
    ...     {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200},
    ... ]
    >>> results = tuner.evaluate_trials(X_train, y_train, trial_points=trial_points)

    >>> # 评估特定超参数点（scorecardpipeline风格）
    >>> results = tuner.evaluate_trials(df, trial_points=trial_points)
"""

import copy
import inspect
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple, Type, Union
import numpy as np
import pandas as pd
from ....utils.parallel import resolve_n_jobs
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import get_scorer, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from optuna.trial import Trial
else:
    # 类型标注在运行时无需加载 Optuna 的 Trial 类；保留该名称可支持
    # inspect/get_type_hints 等运行时注解读取，同时避免 Pylance 将 optuna 变量当类型命名空间。
    Trial = Any

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.study import StudyDirection

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    StudyDirection = None


def _normalize_space_param(name: str, spec: Any) -> Dict[str, Any]:
    """将单个超参数定义统一为内部 DSL（optuna 风格字典）.

    支持的输入格式（按识别顺序）：

    - dict（hscredit/optuna 风格）：
      ``{'type': 'int'/'float'/'categorical', 'low':, 'high':, 'step':, 'log':, 'choices':}``
    - dict（hyperopt 风格 type）：
      ``{'type': 'uniform'/'loguniform'/'quniform'/'randint'/'choice', ...}``
    - tuple ``(low, high)``（bayesian-optimization / skopt 简写）：
      两端均为整数时为 int，否则为 float
    - tuple ``(low, high, prior)``（skopt 简写）：
      prior 为 ``'log-uniform'`` 时 float(log=True)，``'uniform'`` 时 float
    - list（sklearn 网格 / skopt Categorical / hyperopt choice 简写）：
      视为 categorical 的 choices
    - scipy.stats 冻结分布（sklearn RandomizedSearchCV 风格）：
      ``randint`` / ``loguniform``(reciprocal) / ``uniform``
    - optuna.distributions 分布对象（安装 optuna 时）：
      ``IntDistribution`` / ``FloatDistribution`` / ``CategoricalDistribution``
      及 optuna 旧版 ``IntUniformDistribution`` 等

    :param name: 参数名（用于错误提示）
    :param spec: 参数定义
    :return: 内部 DSL 字典，含 ``type`` 及相应键
    :raises ValueError: 无法识别的格式或取值非法时抛出
    """
    # 0. hscredit 维度对象（search_space 模块的 Real/Integer/Categorical/suggest_*/hp.* 返回值）
    # 对象自带 to_spec 方法时优先委托，再走统一字典解析路径
    if hasattr(spec, "to_spec") and callable(getattr(spec, "to_spec")):
        return _normalize_space_param(name, spec.to_spec())

    # 1. optuna 原生分布对象
    if OPTUNA_AVAILABLE and isinstance(spec, optuna.distributions.BaseDistribution):
        return _space_param_from_optuna(name, spec)

    # 2. scipy 冻结分布（sklearn RandomizedSearchCV 风格）
    # 鸭子类型识别 rv_frozen（rv_frozen 基类在 scipy 各版本中的暴露路径不稳定）
    if (
        hasattr(spec, "dist")
        and hasattr(spec, "args")
        and callable(getattr(spec, "rvs", None))
        and getattr(getattr(spec, "dist", None), "name", None) is not None
    ):
        return _space_param_from_scipy(name, spec)

    # 3. tuple：bayesian-optimization / skopt 简写
    if isinstance(spec, tuple):
        return _space_param_from_tuple(name, spec)

    # 4. list：sklearn 网格 / skopt Categorical / hyperopt choice 简写
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"参数 {name!r} 的 choices 不能为空列表")
        return {"type": "categorical", "choices": list(spec)}

    # 5. dict：hscredit DSL 或 hyperopt 风格
    if isinstance(spec, dict):
        return _space_param_from_dict(name, spec)

    raise ValueError(
        f"参数 {name!r} 的搜索空间定义无法识别: {spec!r}。"
        "支持 dict（optuna/hyperopt 风格）、tuple（bayesian-optimization/skopt 风格）、"
        "list（categorical 简写）、scipy.stats 分布或 optuna.distributions 分布对象"
    )


def _space_param_from_dict(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """解析字典形式（hscredit DSL 或 hyperopt 风格 type）的参数定义。"""
    param_type = spec.get("type")
    if param_type is None:
        raise ValueError(f"参数 {name!r} 的搜索空间字典缺少 'type' 键: {spec!r}")
    param_type = str(param_type).strip().lower()

    if param_type == "int":
        _check_bounds(name, spec, integer=True)
        result = {"type": "int", "low": int(spec["low"]), "high": int(spec["high"])}
        if spec.get("log", False):
            result["log"] = True
        elif "step" in spec:
            result["step"] = int(spec["step"])
        return result
    if param_type == "float":
        _check_bounds(name, spec, integer=False)
        result = {"type": "float", "low": float(spec["low"]), "high": float(spec["high"])}
        if spec.get("log", False):
            result["log"] = True
        if "step" in spec and spec["step"] is not None:
            result["step"] = float(spec["step"])
        return result
    if param_type == "categorical":
        choices = spec.get("choices", spec.get("options"))
        if not choices:
            raise ValueError(f"参数 {name!r} 的 categorical 类型必须提供非空 choices")
        return {"type": "categorical", "choices": list(choices)}

    # hyperopt 风格
    if param_type == "uniform":
        _check_bounds(name, spec, integer=False)
        return {"type": "float", "low": float(spec["low"]), "high": float(spec["high"])}
    if param_type == "loguniform":
        _check_bounds(name, spec, integer=False)
        return {"type": "float", "low": float(spec["low"]), "high": float(spec["high"]), "log": True}
    if param_type == "quniform":
        _check_bounds(name, spec, integer=False)
        if "q" not in spec:
            raise ValueError(f"参数 {name!r} 的 quniform 类型必须提供步长 'q'")
        return {
            "type": "float",
            "low": float(spec["low"]),
            "high": float(spec["high"]),
            "step": float(spec["q"]),
        }
    if param_type == "randint":
        low = int(spec.get("low", 0))
        if "high" not in spec:
            raise ValueError(f"参数 {name!r} 的 randint 类型必须提供 'high'")
        high = int(spec["high"])
        if low > high:
            raise ValueError(f"参数 {name!r} 的 low({low}) 不能大于 high({high})")
        return {"type": "int", "low": low, "high": high}
    if param_type == "choice":
        choices = spec.get("choices", spec.get("options"))
        if not choices:
            raise ValueError(f"参数 {name!r} 的 choice 类型必须提供非空 choices")
        return {"type": "categorical", "choices": list(choices)}

    # hyperopt 正态族（normal/lognormal/qnormal/qlognormal）
    # optuna 无原生正态采样，归一化为 'normal' DSL，由 _sample_params 逆 CDF 变换实现
    if param_type in ("normal", "qnormal", "lognormal", "qlognormal"):
        if "mu" not in spec or "sigma" not in spec:
            raise ValueError(f"参数 {name!r} 的 {param_type} 类型必须提供 'mu' 和 'sigma'")
        mu = float(spec["mu"])
        sigma = float(spec["sigma"])
        if sigma <= 0:
            raise ValueError(f"参数 {name!r} 的 sigma 必须 > 0")
        log = param_type in ("lognormal", "qlognormal")
        q = None
        if param_type in ("qnormal", "qlognormal"):
            if "q" not in spec:
                raise ValueError(f"参数 {name!r} 的 {param_type} 类型必须提供步长 'q'")
            q = float(spec["q"])
        # 截断区间 [mu-4σ, mu+4σ]，log 时映射到 exp
        lo, hi = mu - 4 * sigma, mu + 4 * sigma
        low = float(np.exp(lo)) if log else float(lo)
        high = float(np.exp(hi)) if log else float(hi)
        result = {"type": "normal", "mu": mu, "sigma": sigma, "low": low, "high": high}
        if q is not None:
            result["q"] = q
        if log:
            result["log"] = True
        return result

    raise ValueError(
        f"参数 {name!r} 的搜索空间类型未知: {param_type!r}，"
        "可选: 'int'/'float'/'categorical'（optuna 风格）或 "
        "'uniform'/'loguniform'/'quniform'/'randint'/'choice'（hyperopt 风格）或 "
        "'normal'/'qnormal'/'lognormal'/'qlognormal'（hyperopt 正态族）"
    )


def _space_param_from_tuple(name: str, spec: tuple) -> Dict[str, Any]:
    """解析元组形式（bayesian-optimization / skopt 简写）的参数定义。"""
    if len(spec) == 2:
        low, high = spec
        if low > high:
            raise ValueError(f"参数 {name!r} 的下界({low})不能大于上界({high})")
        # 两端均为整数（排除 bool）时视为整数参数，否则为浮点参数
        if (
            isinstance(low, (int, np.integer))
            and isinstance(high, (int, np.integer))
            and not isinstance(low, bool)
            and not isinstance(high, bool)
        ):
            return {"type": "int", "low": int(low), "high": int(high)}
        return {"type": "float", "low": float(low), "high": float(high)}
    if len(spec) == 3:
        low, high, prior = spec
        if low > high:
            raise ValueError(f"参数 {name!r} 的下界({low})不能大于上界({high})")
        prior_norm = str(prior).strip().lower()
        if prior_norm == "log-uniform":
            return {"type": "float", "low": float(low), "high": float(high), "log": True}
        if prior_norm == "uniform":
            return {"type": "float", "low": float(low), "high": float(high)}
        raise ValueError(f"参数 {name!r} 的元组第三元素（prior）仅支持 'uniform'/'log-uniform'，" f"当前为 {prior!r}")
    raise ValueError(f"参数 {name!r} 的元组形式仅支持 (low, high) 或 (low, high, prior)，" f"当前长度: {len(spec)}")


def _space_param_from_scipy(name: str, spec: Any) -> Dict[str, Any]:
    """解析 scipy 冻结分布（sklearn RandomizedSearchCV 风格）的参数定义。"""
    dist_name = getattr(getattr(spec, "dist", None), "name", None)
    args = getattr(spec, "args", ())

    if dist_name == "randint":
        # scipy randint 采样区间为 [low, high)，转为闭区间 [low, high-1]
        low, high = int(args[0]), int(args[1]) - 1
        if low > high:
            raise ValueError(f"参数 {name!r} 的 randint 分布区间为空: {spec!r}")
        return {"type": "int", "low": low, "high": high}
    if dist_name in ("loguniform", "reciprocal"):
        low, high = float(args[0]), float(args[1])
        if low <= 0 or high <= 0:
            raise ValueError(f"参数 {name!r} 的 loguniform 分布要求区间为正数: {spec!r}")
        return {"type": "float", "low": low, "high": high, "log": True}
    if dist_name == "uniform":
        # scipy uniform 参数为 (loc, scale)，采样区间 [loc, loc+scale]
        low, high = float(args[0]), float(args[0] + args[1])
        return {"type": "float", "low": low, "high": high}

    raise ValueError(f"参数 {name!r} 的 scipy 分布暂不支持: {dist_name!r}，" "仅支持 randint / loguniform / uniform")


def _space_param_from_optuna(name: str, spec: Any) -> Dict[str, Any]:
    """解析 optuna 分布对象的参数定义（含旧版分布类的兼容映射）。"""
    # optuna >= 2.4 的统一分布类
    if isinstance(spec, optuna.distributions.IntDistribution):
        result = {"type": "int", "low": int(spec.low), "high": int(spec.high)}
        if getattr(spec, "log", False):
            result["log"] = True
        else:
            step = int(getattr(spec, "step", 1))
            if step != 1:
                result["step"] = step
        return result
    if isinstance(spec, optuna.distributions.FloatDistribution):
        result = {"type": "float", "low": float(spec.low), "high": float(spec.high)}
        if getattr(spec, "log", False):
            result["log"] = True
        step = getattr(spec, "step", None)
        if step is not None:
            result["step"] = float(step)
        return result
    if isinstance(spec, optuna.distributions.CategoricalDistribution):
        return {"type": "categorical", "choices": list(spec.choices)}

    # optuna < 2.4 的旧版分布类
    legacy = optuna.distributions
    if hasattr(legacy, "IntLogUniformDistribution") and isinstance(spec, legacy.IntLogUniformDistribution):
        return {"type": "int", "low": int(spec.low), "high": int(spec.high), "log": True}
    if hasattr(legacy, "IntUniformDistribution") and isinstance(spec, legacy.IntUniformDistribution):
        result = {"type": "int", "low": int(spec.low), "high": int(spec.high)}
        step = int(getattr(spec, "step", 1))
        if step != 1:
            result["step"] = step
        return result
    if hasattr(legacy, "LogUniformDistribution") and isinstance(spec, legacy.LogUniformDistribution):
        return {"type": "float", "low": float(spec.low), "high": float(spec.high), "log": True}
    if hasattr(legacy, "DiscreteUniformDistribution") and isinstance(spec, legacy.DiscreteUniformDistribution):
        return {
            "type": "float",
            "low": float(spec.low),
            "high": float(spec.high),
            "step": float(spec.q),
        }
    if hasattr(legacy, "UniformDistribution") and isinstance(spec, legacy.UniformDistribution):
        return {"type": "float", "low": float(spec.low), "high": float(spec.high)}

    raise ValueError(f"参数 {name!r} 的 optuna 分布类型不支持: {type(spec).__name__}")


def _check_bounds(name: str, spec: Dict[str, Any], integer: bool) -> None:
    """校验字典形式参数定义的 low/high 边界。"""
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"参数 {name!r} 的搜索空间必须提供 'low' 和 'high': {spec!r}")
    low, high = spec["low"], spec["high"]
    if low > high:
        raise ValueError(f"参数 {name!r} 的下界({low})不能大于上界({high})")
    if integer:
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, np.integer))
            or not isinstance(high, (int, np.integer))
        ):
            raise ValueError(f"参数 {name!r} 的 int 类型要求整数边界: {spec!r}")


def _legacy_normalize_search_space(search_space: Optional[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
    """将多种超参数框架的搜索空间格式统一为内部 DSL（optuna 风格）.

    每个参数单独定义，支持以下框架的入参格式（无需安装对应库，
    仅需按其风格以 dict/tuple/list/分布对象表达）：

    - **hscredit/optuna 风格**：
      ``{'max_depth': {'type': 'int', 'low': 2, 'high': 4}}``
    - **optuna 分布对象**：``{'max_depth': optuna.distributions.IntDistribution(2, 4)}``
    - **bayesian-optimization 风格**：
      ``{'max_depth': (2, 4), 'learning_rate': (1e-3, 0.1)}``
    - **scikit-optimize 风格**：
      ``{'learning_rate': (1e-3, 0.1, 'log-uniform'), 'penalty': ['l1', 'l2']}``
    - **sklearn 风格**：
      ``{'C': [0.1, 1, 10]}`` 或 scipy 分布 ``{'C': scipy.stats.loguniform(1e-3, 1e1)}``
    - **hyperopt 风格**：
      ``{'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 0.1},
        'penalty': {'type': 'choice', 'choices': ['l1', 'l2']}}``

    :param search_space: 搜索空间字典，键为参数名；为 None 时原样返回 None
    :return: 统一为 ``{'type': 'int'/'float'/'categorical', ...}`` 形式的字典
    :raises ValueError: 格式无法识别或取值非法时抛出
    """
    if search_space is None:
        return None
    if not isinstance(search_space, dict):
        raise ValueError(f"search_space 必须是字典（参数名 -> 参数定义），当前类型: {type(search_space).__name__}")
    return {name: _normalize_space_param(name, spec) for name, spec in search_space.items()}


# 新适配器覆盖上方保留的旧解析实现；旧私有函数暂留用于兼容可能存在的内部导入，
# 所有公开入口与 ModelTuner 从这里开始统一走同一套格式、校验和采样语义。
from .space_adapter import SearchSpaceAdapter, normalize_search_space  # noqa: E402, F401


class TuningSampler:
    """采样器码表 - 统一管理 optuna 内置及 optunahub 提供的搜索器.

    通过字符串名称即可在 ``ModelTuner(sampler=...)`` 中选用不同搜索器，
    无需直接 import 对应的采样器类。

    **optuna 内置采样器** (``BUILTIN_SAMPLERS``)：

    - ``'tpe'``        : TPESampler，树结构 Parzen 估计（默认）
    - ``'random'``     : RandomSampler，随机搜索
    - ``'cmaes'``      : CmaEsSampler，CMA-ES 进化策略（依赖 cmaes）
    - ``'grid'``       : GridSampler，网格搜索（需 sampler_kwargs 传入 search_space）
    - ``'nsgaii'``     : NSGAIISampler，多目标遗传算法
    - ``'nsgaiii'``    : NSGAIIISampler，多目标遗传算法（多目标场景）
    - ``'qmc'``        : QMCSampler，准蒙特卡洛
    - ``'gp'``         : GPSampler，高斯过程贝叶斯优化
    - ``'bruteforce'`` : BruteForceSampler，穷举搜索

    **optunahub 采样器** (``OPTUNAHUB_SAMPLERS``，依赖 optunahub，按需联网下载)：

    - ``'auto'``       : AutoSampler，自动选择最优采样器
    - ``'hebo'``       : HEBOSampler，异方差贝叶斯优化
    - ``'smac'``       : SMACSampler，基于 SMAC3 的贝叶斯优化
    - ``'neldermead'`` : NelderMeadSampler，单纯形法

    **参考样例**

    >>> from hscredit.core.models import TuningSampler
    >>> TuningSampler.list_samplers()
    >>> sampler = TuningSampler.create('cmaes', seed=42)
    >>> sampler = TuningSampler.create('auto')   # optunahub
    """

    # optuna 内置采样器：name -> optuna.samplers 中的类名
    BUILTIN_SAMPLERS = {
        "tpe": "TPESampler",
        "random": "RandomSampler",
        "cmaes": "CmaEsSampler",
        "grid": "GridSampler",
        "nsgaii": "NSGAIISampler",
        "nsgaiii": "NSGAIIISampler",
        "qmc": "QMCSampler",
        "gp": "GPSampler",
        "bruteforce": "BruteForceSampler",
    }

    # optunahub 采样器：name -> (package 路径, 类名)
    OPTUNAHUB_SAMPLERS = {
        "auto": ("samplers/auto_sampler", "AutoSampler"),
        "hebo": ("samplers/hebo", "HEBOSampler"),
        "smac": ("samplers/smac_sampler", "SMACSampler"),
        "neldermead": ("samplers/nelder_mead", "NelderMeadSampler"),
    }

    @classmethod
    def list_samplers(cls) -> Dict[str, List[str]]:
        """列出所有支持的采样器名称.

        :return: {'内置': [...], 'optunahub': [...]}
        """
        return {
            "内置": list(cls.BUILTIN_SAMPLERS.keys()),
            "optunahub": list(cls.OPTUNAHUB_SAMPLERS.keys()),
        }

    @staticmethod
    def _instantiate(sampler_cls: Type, kwargs: Dict[str, Any]) -> Any:
        """按构造函数签名过滤 kwargs 后实例化采样器（如 seed 不被支持则丢弃）."""
        import inspect

        try:
            sig = inspect.signature(sampler_cls.__init__)
            accepted = set(sig.parameters)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except (TypeError, ValueError):
            accepted, accepts_var_kw = set(), True

        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return sampler_cls(**kwargs)

    @classmethod
    def create(
        cls,
        sampler: Union[str, Any, None] = "tpe",
        seed: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """按名称创建采样器实例.

        :param sampler: 采样器名称（见 BUILTIN_SAMPLERS / OPTUNAHUB_SAMPLERS），
            或已实例化的采样器对象（直接返回），或 None（默认 TPE）
        :param seed: 随机种子，若采样器支持则注入
        :param kwargs: 透传给采样器构造函数的额外参数
        :return: optuna 采样器实例
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请使用 pip install optuna 安装")

        # 已是采样器实例，直接返回
        if sampler is None:
            sampler = "tpe"
        if not isinstance(sampler, str):
            return sampler

        key = sampler.lower()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed

        if key in cls.BUILTIN_SAMPLERS:
            sampler_cls = getattr(optuna.samplers, cls.BUILTIN_SAMPLERS[key])
            return cls._instantiate(sampler_cls, kwargs)

        if key in cls.OPTUNAHUB_SAMPLERS:
            try:
                import optunahub
            except ImportError:
                raise ImportError(
                    f"使用 '{sampler}' 采样器需要 optunahub，请使用 " f"pip install optunahub 安装（或 pip install hscredit[tune]）"
                )
            package, class_name = cls.OPTUNAHUB_SAMPLERS[key]
            module = optunahub.load_module(package)
            sampler_cls = getattr(module, class_name, None)
            if sampler_cls is None:
                # 类名兜底：查找模块中以 Sampler 结尾的类
                sampler_cls = next(
                    (getattr(module, n) for n in dir(module) if n.endswith("Sampler")),
                    None,
                )
            if sampler_cls is None:
                raise ValueError(f"无法从 optunahub 包 '{package}' 中找到采样器类")
            return cls._instantiate(sampler_cls, kwargs)

        all_names = list(cls.BUILTIN_SAMPLERS) + list(cls.OPTUNAHUB_SAMPLERS)
        raise ValueError(f"未知采样器 '{sampler}'，可选: {all_names}")


def _calc_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算KS值（内部辅助函数）.

    :param y_true: 真实标签
    :param y_pred: 预测概率
    :return: KS值
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return abs(tpr - fpr).max()


def _calc_ks_with_diff(
    y_train: np.ndarray, y_train_pred: np.ndarray, y_val: np.ndarray, y_val_pred: np.ndarray
) -> Tuple[float, float]:
    """计算KS及训练/验证差异.

    :return: (验证集KS, KS差异)
    """
    ks_train = _calc_ks(y_train, y_train_pred)
    ks_val = _calc_ks(y_val, y_val_pred)
    ks_diff = abs(ks_train - ks_val)
    return ks_val, ks_diff


def _safe_index(data: Any, indices: np.ndarray) -> Any:
    """按行索引切分 pandas / numpy / list 数据."""
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


class TuningObjective:
    """内置调参目标函数集合.

    所有静态方法签名均为 ``(y_true, y_prob, **kwargs) -> float``，
    值越大越好（均已设计为 maximize 方向）。

    可通过字符串名称传给 ``ModelTuner(objective=...)``：
    - ``'ks'``            : 标准 KS（默认）
    - ``'auc'``           : ROC-AUC
    - ``'lift_head'``     : 头部 LIFT（高概率前 ratio 比例）
    - ``'lift_tail'``     : 尾部 LIFT（低概率前 ratio 比例的纯净度）
    - ``'lift_head_monotonic'`` : KS × (1 - 违反单调比例 × penalty)
    - ``'ks_with_lift_constraint'`` : 满足头部 LIFT 约束下的 KS
    - ``'head_ks'``       : 仅头部 ratio 比例样本的 KS
    - ``'approval_bad_rate'`` : 固定通过率下优化低风险通过客群坏率
    - ``'expected_profit'`` : 固定通过率下优化通过客群期望利润

    Example:
        >>> from hscredit.core.models import ModelTuner, XGBoost
        >>> tuner = ModelTuner(
        ...     model_class=XGBoost,
        ...     objective='lift_head',
        ...     objective_kwargs={'ratio': 0.05},
        ... )
        >>> tuner.fit(X_train, y_train, n_trials=50)
    """

    # 支持的字符串名称
    BUILTIN_OBJECTIVES = [
        "ks",
        "auc",
        "lift_head",
        "lift_tail",
        "lift_head_monotonic",
        "ks_with_lift_constraint",
        "head_ks",
        "ks_lift_combined",
        "tail_purity_ks",
        "approval_bad_rate",
        "expected_profit",
    ]

    @staticmethod
    def ks(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
        """标准 KS 目标."""
        return _calc_ks(y_true, y_prob)

    @staticmethod
    def auc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
        """ROC-AUC 目标."""
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return 0.0

    @staticmethod
    def lift_head(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ratio: float = 0.10,
        **kwargs,
    ) -> float:
        """头部 LIFT 目标：优化预测概率最高 ratio 比例样本的 LIFT 值.

        :param ratio: 覆盖率，默认 0.10（即 Top 10%）
        """
        total = len(y_true)
        n_top = max(1, int(total * ratio))
        sorted_idx = np.argsort(y_prob)[::-1]
        y_sorted = y_true[sorted_idx]
        overall_br = y_true.mean()
        if overall_br == 0:
            return 0.0
        top_br = y_sorted[:n_top].mean()
        return float(top_br / overall_br)

    @staticmethod
    def lift_tail(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ratio: float = 0.10,
        **kwargs,
    ) -> float:
        """尾部 LIFT 目标：优化预测概率最低 ratio 比例样本（低风险客群）的纯净度.

        纯净度定义为：(1 - 尾部坏率) / (1 - 整体坏率)，值越大表示尾部越纯净。

        :param ratio: 尾部覆盖率，默认 0.10
        """
        total = len(y_true)
        n_tail = max(1, int(total * ratio))
        sorted_idx = np.argsort(y_prob)  # 升序，低概率在前
        y_sorted = y_true[sorted_idx]
        overall_br = y_true.mean()
        if overall_br == 1.0:
            return 0.0
        tail_br = y_sorted[:n_tail].mean()
        tail_purity = (1.0 - tail_br) / (1.0 - overall_br) if (1.0 - overall_br) > 0 else 0.0
        return float(tail_purity)

    @staticmethod
    def lift_head_monotonic(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        penalty: float = 0.5,
        **kwargs,
    ) -> float:
        """头部单调 LIFT 目标：KS × (1 - 违反单调性比例 × penalty).

        单调性违反比例越低，目标越高；完全单调时等同于 KS 目标。

        :param n_bins: 分箱数，默认 10
        :param penalty: 单调性惩罚强度，默认 0.5
        """
        ks_val = _calc_ks(y_true, y_prob)
        try:
            total = len(y_true)
            n_bin = max(2, n_bins)
            bin_size = total // n_bin
            sorted_idx = np.argsort(y_prob)[::-1]
            y_sorted = y_true[sorted_idx]
            overall_br = y_true.mean()
            if overall_br == 0:
                return 0.0
            brs = []
            for i in range(n_bin):
                start = i * bin_size
                end = (i + 1) * bin_size if i < n_bin - 1 else total
                seg = y_sorted[start:end]
                brs.append(seg.mean() if len(seg) > 0 else 0.0)
            violations = sum(1 for i in range(1, len(brs)) if brs[i] > brs[i - 1] + 1e-8)
            n_pairs = n_bin - 1
            violation_ratio = violations / n_pairs if n_pairs > 0 else 0.0
            return float(ks_val * (1.0 - violation_ratio * penalty))
        except Exception:
            return float(ks_val)

    @staticmethod
    def ks_with_lift_constraint(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_lift_ratio: float = 0.05,
        min_lift_value: float = 2.0,
        **kwargs,
    ) -> float:
        """KS + LIFT 约束：满足头部 LIFT >= min_lift_value 前提下最大化 KS.

        若不满足约束，返回 0（惩罚）。

        :param min_lift_ratio: 头部覆盖率，默认 0.05（Top 5%）
        :param min_lift_value: 最低 LIFT 要求，默认 2.0
        """
        head_lift = TuningObjective.lift_head(y_true, y_prob, ratio=min_lift_ratio)
        if head_lift < min_lift_value:
            return 0.0  # 不满足约束，惩罚为0
        return _calc_ks(y_true, y_prob)

    @staticmethod
    def head_ks(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ratio: float = 0.30,
        **kwargs,
    ) -> float:
        """头部 KS：仅计算预测概率最高 ratio 比例样本的 KS（头部区分能力）.

        :param ratio: 头部覆盖率，默认 0.30
        """
        total = len(y_true)
        n_top = max(2, int(total * ratio))
        sorted_idx = np.argsort(y_prob)[::-1]
        y_top = y_true[sorted_idx[:n_top]]
        prob_top = y_prob[sorted_idx[:n_top]]
        if y_top.sum() == 0 or y_top.sum() == n_top:
            return 0.0
        try:
            return _calc_ks(y_top, prob_top)
        except Exception:
            return 0.0

    @staticmethod
    def ks_lift_combined(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ks_weight: float = 0.5,
        lift_ratio: float = 0.05,
        **kwargs,
    ) -> float:
        """KS + LIFT 联合目标：加权组合 KS 和头部 LIFT.

        score = ks_weight × KS + (1 - ks_weight) × normalized_LIFT

        :param ks_weight: KS 权重，默认 0.5
        :param lift_ratio: LIFT 覆盖率，默认 0.05
        """
        ks_val = _calc_ks(y_true, y_prob)
        lift_val = TuningObjective.lift_head(y_true, y_prob, ratio=lift_ratio)
        # 归一化 LIFT 到 [0, 1] 范围（假设最大合理 LIFT 为 10）
        norm_lift = min(lift_val / 10.0, 1.0)
        return float(ks_weight * ks_val + (1.0 - ks_weight) * norm_lift)

    @staticmethod
    def tail_purity_ks(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        tail_ratio: float = 0.30,
        **kwargs,
    ) -> float:
        """尾部纯净度 + 整体 KS 联合目标.

        适用于「放量优先」场景：确保通过（低风险）部分的坏率尽量低，同时保持整体区分度.
        score = 0.5 × KS + 0.5 × tail_purity

        :param tail_ratio: 尾部覆盖率，默认 0.30（即通过的低风险比例）
        """
        ks_val = _calc_ks(y_true, y_prob)
        purity = TuningObjective.lift_tail(y_true, y_prob, ratio=tail_ratio)
        # purity 已经在 [0, 1+] 范围
        norm_purity = min(purity, 1.0)
        return float(0.5 * ks_val + 0.5 * norm_purity)

    @staticmethod
    def approval_bad_rate(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        approval_rate: float = 0.30,
        bad_rate_weight: float = 1.0,
        **kwargs,
    ) -> float:
        """通过率坏率目标：固定通过率下最大化通过收益、惩罚通过坏率.

        默认把预测概率最低的 ``approval_rate`` 样本视为通过客群。
        score = approval_rate × (1 - 通过坏率 × bad_rate_weight)

        :param approval_rate: 通过率，默认 0.30
        :param bad_rate_weight: 坏率惩罚权重，默认 1.0
        """
        if not 0 < approval_rate <= 1:
            raise ValueError("approval_rate 必须在 (0, 1] 范围内")
        total = len(y_true)
        n_approved = max(1, int(total * approval_rate))
        approved_idx = np.argsort(y_prob)[:n_approved]
        approved_br = np.asarray(y_true)[approved_idx].mean()
        return float(approval_rate * (1.0 - approved_br * bad_rate_weight))

    @staticmethod
    def expected_profit(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        approval_rate: float = 0.30,
        good_profit: float = 1.0,
        bad_loss: float = 5.0,
        **kwargs,
    ) -> float:
        """期望利润目标：固定通过率下最大化通过客群单位样本收益.

        默认预测概率最低的样本为通过客群，好客户收益为 ``good_profit``，
        坏客户损失为 ``bad_loss``，拒绝样本收益记为 0。

        :param approval_rate: 通过率，默认 0.30
        :param good_profit: 通过好客户收益，默认 1.0
        :param bad_loss: 通过坏客户损失，默认 5.0
        """
        if not 0 < approval_rate <= 1:
            raise ValueError("approval_rate 必须在 (0, 1] 范围内")
        y_true = np.asarray(y_true)
        total = len(y_true)
        n_approved = max(1, int(total * approval_rate))
        approved_idx = np.argsort(y_prob)[:n_approved]
        approved_y = y_true[approved_idx]
        profit = np.where(approved_y == 1, -bad_loss, good_profit).sum()
        return float(profit / total)

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs,
    ):
        """按名称获取目标函数（偏函数形式）.

        :param name: 目标函数名称，见 BUILTIN_OBJECTIVES
        :param kwargs: 额外参数（如 ratio/penalty 等）
        :return: 可调用对象 (y_true, y_prob) -> float

        Example:
            >>> obj = TuningObjective.get('lift_head', ratio=0.05)
            >>> score = obj(y_true, y_prob)
        """
        name_lower = name.lower()
        if name_lower not in cls.BUILTIN_OBJECTIVES:
            raise ValueError(f"未知目标函数 '{name}'，可选: {cls.BUILTIN_OBJECTIVES}")
        func = getattr(cls, name_lower)
        if kwargs:
            import functools

            return functools.partial(func, **kwargs)
        return func


class Metric:
    """评估指标包装类.

    用于统一管理内置指标和自定义指标。

    :param metric: 指标名称(str)或自定义函数(Callable)
    :param name: 指标名称（用于显示）
    :param direction: 优化方向，'maximize'或'minimize'
    """

    # 内置指标映射
    BUILTIN_METRICS = {
        "auc": {"scorer": "roc_auc", "direction": "maximize"},
        "accuracy": {"scorer": "accuracy", "direction": "maximize"},
        "precision": {"scorer": "precision", "direction": "maximize"},
        "recall": {"scorer": "recall", "direction": "maximize"},
        "f1": {"scorer": "f1", "direction": "maximize"},
        "logloss": {"scorer": "neg_log_loss", "direction": "maximize"},
        "ks": {"scorer": None, "direction": "maximize"},  # 使用自定义计算
        "ks_diff": {"scorer": None, "direction": "minimize"},  # KS差异，需要特殊处理
        "lift_head": {"scorer": None, "direction": "maximize"},
        "lift_tail": {"scorer": None, "direction": "maximize"},
        "lift_head_monotonic": {"scorer": None, "direction": "maximize"},
        "ks_with_lift_constraint": {"scorer": None, "direction": "maximize"},
        "head_ks": {"scorer": None, "direction": "maximize"},
        "ks_lift_combined": {"scorer": None, "direction": "maximize"},
        "tail_purity_ks": {"scorer": None, "direction": "maximize"},
        "approval_bad_rate": {"scorer": None, "direction": "maximize"},
        "expected_profit": {"scorer": None, "direction": "maximize"},
    }

    def __init__(self, metric: Union[str, Callable], name: Optional[str] = None, direction: Optional[str] = None):
        self.metric = metric
        self._is_builtin = isinstance(metric, str)

        if self._is_builtin:
            metric_key = metric.lower()
            if metric_key not in self.BUILTIN_METRICS:
                raise ValueError(f"未知的内置指标: {metric}，可用指标: {list(self.BUILTIN_METRICS.keys())}")

            self.name = name or metric_key.upper()
            self.scorer = self.BUILTIN_METRICS[metric_key]["scorer"]
            self.direction = direction or self.BUILTIN_METRICS[metric_key]["direction"]
        else:
            # 自定义函数
            if not callable(metric):
                raise ValueError("自定义metric必须是可调用的函数")
            self.name = name or getattr(metric, "__name__", "custom_metric")
            self.scorer = metric
            if direction is None:
                raise ValueError("使用自定义metric时必须指定direction")
            self.direction = direction

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_train_pred: Optional[np.ndarray] = None,
    ) -> float:
        """计算指标值.

        :param y_true: 验证集真实标签
        :param y_pred: 验证集预测概率
        :param y_train: 训练集真实标签（用于ks_diff）
        :param y_train_pred: 训练集预测概率（用于ks_diff）
        :return: 指标值
        """
        if self._is_builtin and self.metric.lower() == "ks":
            return _calc_ks(y_true, y_pred)
        elif self._is_builtin and self.metric.lower() == "ks_diff":
            if y_train is None or y_train_pred is None:
                raise ValueError("计算ks_diff需要提供训练集预测结果")
            _, ks_diff = _calc_ks_with_diff(y_train, y_train_pred, y_true, y_pred)
            return ks_diff
        elif self._is_builtin and self.metric.lower() in TuningObjective.BUILTIN_OBJECTIVES:
            return TuningObjective.get(self.metric.lower())(y_true, y_pred)
        elif self._is_builtin:
            # 其他内置指标使用sklearn scorer
            if self.scorer is None:
                raise ValueError(f"指标 {self.metric} 没有对应的sklearn scorer")
            scorer = get_scorer(self.scorer)
            # sklearn scorer需要estimator，这里直接计算
            from sklearn.metrics import get_scorer_names

            if self.scorer in get_scorer_names():
                # 对于可以直接计算的指标
                if self.scorer == "roc_auc":
                    return roc_auc_score(y_true, y_pred)
                # 其他指标需要类别预测
                # 这里简化处理，实际使用时可能需要调整
                return scorer._score_func(y_true, y_pred > 0.5)
            return scorer._score_func(y_true, y_pred)
        else:
            # 自定义函数
            return self.scorer(y_true, y_pred)

    def __repr__(self):
        return f"Metric(name='{self.name}', direction='{self.direction}')"


class ModelTuner:
    """模型超参数调优器 - 支持单/多目标优化.

    基于Optuna实现贝叶斯优化超参数搜索。
    支持单目标优化和多目标优化（帕累托最优）。

    **参数**

    :param model_class: 模型类 (如XGBoost)
    :param search_space: 参数搜索空间，默认None则使用预定义空间
    :param fixed_params: 固定参数，不参与搜索
    :param metric: 优化指标（决定评估计算逻辑），可选:
        - 字符串: 'auc', 'ks', 'ks_diff', 'accuracy', 'precision', 'recall', 'f1', 'logloss'
        - 列表: 多个指标，用于多目标优化，如 ['ks', 'ks_diff']
        - 函数: 自定义评估函数，接收(y_true, y_pred)返回float
        - 列表的函数: 多个自定义函数
    :param direction: 优化方向，'maximize'或'minimize'，或列表（多目标时）
    :param metric_names: 指标显示名称列表（仅用于日志/报告/可视化的展示标签，
        不参与任何计算逻辑），默认 None 时从 metric 自动推断
        （内置字符串取其大写形式，自定义函数取其 ``__name__``）。
        与 metric 不重复：metric 决定"算什么"，metric_names 只决定"叫什么"
    :param cv: 交叉验证折数，默认5
    :param n_jobs: 当前 trial 中模型可使用的并行任务数，默认-1；
        trial 本身顺序执行，确保主动中断及时生效并让自适应采样器利用全部历史结果
    :param random_state: 随机种子，默认None
    :param verbose: 是否逐 Trial 输出得分、参数、当前最佳结果及最终摘要，默认False
    :param early_stopping_rounds: 早停轮数，默认20
    :param min_resource: 多目标优化时的最小资源，默认'auto'

    **搜索空间定义**

    搜索空间可使用参数字典，或使用每个维度都设置 ``name`` 的 skopt 维度列表。
    内部统一转换并由 Optuna Study 执行：

    - 整数参数: {'type': 'int', 'low': 1, 'high': 10, 'step': 1}
    - 浮点参数: {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True}
    - 类别参数: {'type': 'categorical', 'choices': ['a', 'b', 'c']}

    同时兼容多种超参数框架的入参格式（无需安装对应库），传入后自动归一化:

    - bayesian-optimization 风格: {'max_depth': (2, 4, int), 'booster': ('gbtree', 'dart')}
    - scikit-optimize 风格: [Real(1e-3, 0.1, prior='log-uniform', name='learning_rate')]
    - sklearn 风格: {'C': [0.1, 1, 10]} 或 scipy 分布 {'C': scipy.stats.loguniform(1e-3, 1e1)}
    - hyperopt 风格: {'learning_rate': loguniform('learning_rate', log(1e-3), log(0.1)),
      'penalty': choice('penalty', ['l1', 'l2'])}
    - optuna 分布对象: {'max_depth': optuna.distributions.IntDistribution(2, 4)}

    **内部建模经验**

    1. XGBoost参数经验:
       - max_depth: 风控场景通常2-4，防止过拟合
       - min_child_weight: 8-256（step 4），越大越保守
       - subsample: 0.35-0.85，colsample_bytree: 0.4-0.9
       - gamma: 0.0-32.0，reg_lambda: 32.0-128.0（强 L2 正则）
       - scale_pos_weight: 16.0-32.0（适配低坏率不平衡）
       - learning_rate: 0.0001-0.01，较小学习率更稳定
       - n_estimators: 32-256（step 16）

    2. LightGBM参数经验（与 XGBoost 对齐）:
       - num_leaves: 与max_depth相关，受 2**max_depth 上界约束
       - max_depth: 风控场景通常2-4，防止过拟合
       - learning_rate: 0.0001-0.01，较小学习率更稳定
       - min_child_samples: 8-256（step 4）

    3. LogisticRegression参数经验:
       - C: 0.01-32 离散网格，越小正则越强
       - penalty: 'l2'，class_weight: None/'balanced'/自定义权重字典
       - solver: liblinear/sag/lbfgs/newton-cg，max_iter: 16-256

    4. 评估指标:
       - 主要用KS评估模型区分能力
       - 同时考虑训练/测试KS差异防止过拟合

    **参考样例**

    >>> from hscredit.core.models import XGBoost, ModelTuner
    >>> # 单目标：最大化 KS
    >>> tuner = ModelTuner(XGBoost, metric='ks', direction='maximize', cv=5)
    >>> tuner.fit(X_train, y_train, n_trials=50)   # 返回最佳参数 best_params_
    >>> best_model = tuner.get_best_model()  # 已使用完整训练集重训
    >>>
    >>> # 多目标：同时优化 KS 与训练/测试 KS 差异（帕累托最优）
    >>> tuner = ModelTuner(
    ...     XGBoost,
    ...     metric=['ks', 'ks_diff'],
    ...     direction=['maximize', 'minimize'],
    ...     sampler='nsgaii',
    ... )
    >>> tuner.fit(X_train, y_train, n_trials=100)
    >>>
    >>> # 自定义搜索空间
    >>> space = {'max_depth': {'type': 'int', 'low': 2, 'high': 4},
    ...          'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True}}
    >>> tuner = ModelTuner(XGBoost, search_space=space, metric='auc')

    **引用**

    基于 Optuna 超参数优化框架（默认 TPE 采样器），见
    Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter
    Optimization Framework.* KDD；TPE 见 Bergstra, J. et al. (2011),
    *Algorithms for Hyper-Parameter Optimization*, NeurIPS。
    文档：https://optuna.readthedocs.io/
    """

    def __init__(
        self,
        model_class: Type,
        search_space: Optional[Any] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        metric: Union[str, Callable, List[Union[str, Callable]]] = "ks",
        direction: Union[str, List[str]] = "maximize",
        metric_names: Optional[List[str]] = None,
        objective: Union[str, Callable, None] = None,
        objective_kwargs: Optional[Dict[str, Any]] = None,
        eval_ratios: List[float] = None,
        trial_points: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        sampler: Union[str, Any, None] = "tpe",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        load_if_exists: bool = False,
        target: str = "target",
        cv: int = 5,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = False,
        early_stopping_rounds: int = 20,
        points_to_evaluate: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """初始化 ModelTuner.

        :param objective: 调参优化目标，支持字符串名称（见 TuningObjective.BUILTIN_OBJECTIVES）
            或自定义函数 (y_true, y_prob) -> float。
            若指定此参数，则覆盖 metric 参数。
            支持：'ks' / 'auc' / 'lift_head' / 'lift_tail' /
                   'lift_head_monotonic' / 'ks_with_lift_constraint' / 'head_ks'
        :param objective_kwargs: 透传给 TuningObjective 目标函数的额外参数，
            如 {'ratio': 0.05, 'penalty': 0.3}
        :param eval_ratios: 调参过程中额外追踪的 LIFT 覆盖率列表，
            如 [0.01, 0.03, 0.05, 0.10]，结果记录在 optimization_history_ 中
        :param trial_points: 预指定的超参数搜索点，``dict`` 或 ``list[dict]``。
            在 fit 创建 study 后通过 ``study.enqueue_trial`` 优先评估这些点
            （例如已知的经验最优配置或上一轮调优结果），随后再进行常规采样。
            每个 dict 的键应为搜索空间中的参数名，可只指定部分参数（其余由采样器补全）。
            也可在实例化后通过 :meth:`enqueue_trials` 追加。
        :param points_to_evaluate: Hyperopt 风格的初始搜索点，格式与 ``fmin`` 的
            ``points_to_evaluate`` 一致；内部与 ``trial_points`` 一并转换并入队。
        :param sampler: 搜索器，支持:
            - 字符串名称：见 :class:`TuningSampler`，如 'tpe'（默认）/'cmaes'/'random'/
              'gp'/'nsgaii' 等内置采样器，或 'auto'/'hebo'/'smac' 等 optunahub 采样器
            - 已实例化的 optuna 采样器对象（直接使用）
            - None：等价于 'tpe'
        :param sampler_kwargs: 透传给采样器构造函数的额外参数，如 {'n_startup_trials': 10}
        :param storage: optuna 存储 URL，如 ``'sqlite:///hscredit_tuning.db'``。
            指定后可配合 ``optuna-dashboard sqlite:///hscredit_tuning.db`` 实时查看
            调优进度；不指定则使用内存存储（进程结束即丢失）。
        :param study_name: study 名称，配合 storage 持久化时用于标识/复用同一 study。
        :param load_if_exists: storage 中已存在同名 study 时是否加载续跑，默认False。
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请使用 pip install optuna 安装")

        self.model_class = model_class
        self._space_adapter = SearchSpaceAdapter(search_space)
        self.search_space = self._space_adapter.space
        self.fixed_params = fixed_params or {}
        self._validate_lightgbm_leaf_point(self.fixed_params)
        self.objective = objective
        self.objective_kwargs = objective_kwargs or {}
        self.eval_ratios = eval_ratios or [0.01, 0.03, 0.05, 0.10]
        initial_points = self._normalize_trial_points(trial_points)
        initial_points.extend(self._normalize_trial_points(points_to_evaluate))
        self.trial_points: List[Dict[str, Any]] = []
        self._pending_trials: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool]] = []
        self._pending_public_trials: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool]] = []
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs or {}
        self.storage = storage
        self.study_name = study_name
        self.load_if_exists = load_if_exists
        self.target = target
        self.cv = cv
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

        # 若指定了 objective（TuningObjective 风格），将其转换为 metric callable
        if objective is not None:
            if isinstance(objective, str):
                objective_key = objective.lower()
                if objective_key in TuningObjective.BUILTIN_OBJECTIVES:
                    _obj_func = TuningObjective.get(objective_key, **self.objective_kwargs)
                    metric = _obj_func
                    direction = "maximize"
                    metric_names = metric_names or [objective_key]
                else:
                    # 可能是旧式 metric 字符串，直接透传
                    metric = objective
            elif callable(objective):
                metric = objective
                direction = "maximize"

        # 处理metric和direction
        self._setup_metrics(metric, direction, metric_names)

        # 存储结果
        self.study_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_scores_ = None  # 多目标时使用
        self.optimization_history_ = None
        self.pareto_front_ = None  # 多目标帕累托前沿

        # 存储数据信息用于自适应搜索空间
        self._n_samples = None
        self._n_features = None
        self._is_multi_objective = len(self.metrics) > 1

        for point in initial_points:
            self.enqueue_trial(point)

    def _setup_metrics(
        self,
        metric: Union[str, Callable, List[Union[str, Callable]]],
        direction: Union[str, List[str]],
        metric_names: Optional[List[str]],
    ):
        """设置评估指标."""
        # 统一转换为列表
        if not isinstance(metric, list):
            metrics_list = [metric]
        else:
            metrics_list = metric

        # 处理direction
        if not isinstance(direction, list):
            directions_list = [direction] * len(metrics_list)
        else:
            if len(direction) != len(metrics_list):
                raise ValueError("direction列表长度必须与metric列表长度相同")
            directions_list = direction

        # 处理metric_names
        if metric_names is None:
            metric_names = [None] * len(metrics_list)
        elif len(metric_names) != len(metrics_list):
            raise ValueError("metric_names列表长度必须与metric列表长度相同")

        # 创建Metric对象列表
        self.metrics = []
        for m, d, name in zip(metrics_list, directions_list, metric_names):
            if d not in ("maximize", "minimize"):
                raise ValueError("direction 只能是 'maximize' 或 'minimize'")
            if isinstance(m, Metric):
                if m.direction not in ("maximize", "minimize"):
                    raise ValueError("Metric.direction 只能是 'maximize' 或 'minimize'")
                self.metrics.append(m)
            else:
                self.metrics.append(Metric(m, name=name, direction=d))

        # 方便访问
        self.metric = self.metrics[0] if len(self.metrics) == 1 else self.metrics
        self.direction = directions_list[0] if len(directions_list) == 1 else directions_list
        self.directions = [m.direction for m in self.metrics]
        self.metric_names = [m.name for m in self.metrics]

    def _check_input(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
        """检查并处理输入数据.

        支持两种风格:
        1. fit(X, y): sklearn风格，直接使用传入的y
        2. fit(df): scorecardpipeline风格，从df中提取target列

        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :return: (X, y) 处理后的特征和目标
        """
        if y is None:
            # scorecardpipeline风格：从X中提取target
            if isinstance(X, pd.DataFrame):
                if self.target not in X.columns:
                    raise ValueError(f"X中不存在目标列 '{self.target}'，请检查target参数或传入y")
                y = X[self.target]
                X = X.drop(columns=[self.target])
            else:
                raise ValueError("当y为None时，X必须是包含目标列的DataFrame")

        return X, y

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """执行超参数调优.

        支持两种调用风格:

        **sklearn风格**::

            tuner.fit(X_train, y_train, n_trials=100)

        **scorecardpipeline风格** (在__init__中指定target)::

            tuner = ModelTuner(..., target='label')
            tuner.fit(df)  # df包含'label'列

        :param X: 特征矩阵，或包含目标列的DataFrame（scorecardpipeline风格）
        :param y: 目标变量，可选。如果为None，则从X中提取target列
        :param n_trials: 搜索次数，默认100
        :param timeout: 超时时间(秒)，默认None
        :param show_progress_bar: 是否显示进度条，默认True
        :param sample_weight: 样本权重，可选
        :return: 最佳参数字典
        """
        # 检查并处理输入
        X, y = self._check_input(X, y)

        # 记录数据信息
        self._n_samples = len(X)
        self._n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self._X = X
        self._y = y
        self._sample_weight = sample_weight

        # 如果没有指定搜索空间，使用自适应搜索空间
        if self.search_space is None:
            self.search_space = self._get_adaptive_search_space()
            self._space_adapter = SearchSpaceAdapter(self.search_space)
            self.search_space = self._space_adapter.space

        # 创建采样器（支持 optuna 内置及 optunahub 采样器，见 TuningSampler 码表）
        sampler = TuningSampler.create(self.sampler, seed=self.random_state, **self.sampler_kwargs)

        # 公共 study 参数（storage 指定后可用 optuna-dashboard 实时查看进度）
        common_kwargs = dict(sampler=sampler)
        if self.storage is not None:
            common_kwargs.update(
                storage=self.storage,
                study_name=self.study_name,
                load_if_exists=self.load_if_exists,
            )

        if self._is_multi_objective:
            # 多目标优化
            self.study_ = optuna.create_study(directions=self.directions, **common_kwargs)
        else:
            # 单目标优化
            self.study_ = optuna.create_study(direction=self.directions[0], **common_kwargs)

        # 入队预指定的超参数搜索点（优先评估）
        self._enqueue_trial_points()

        # Optuna 的并行 trial 使用线程池。Jupyter 主线程收到中断后，
        # 线程池会等待正在训练的 trial 完成，导致主动中断不能立即返回。
        # trial 顺序执行还可确保自适应采样器利用此前全部完成结果；
        # 调参总预算全部交给当前模型的原生并行参数。
        model_workers = max(1, int(self.n_jobs or 1))

        def objective(trial):
            # 从搜索空间采样参数
            params = self._sample_params(trial)
            params.update(self.fixed_params)
            params = self._apply_model_param_constraints(params)

            # 添加早停参数（仅当模型构造函数支持时，逻辑回归等不支持）
            self._inject_fit_params(params)
            self._inject_model_parallel_budget(params, model_workers)

            # 创建模型
            model = self.model_class(**params)

            # 评估模型
            return self._evaluate_model(model, X, y, sample_weight)

        # 运行优化
        self.study_.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar and self.verbose,
            n_jobs=1,
            callbacks=[self._print_trial_progress] if self.verbose else None,
            catch=(Exception,),
        )

        completed_trials = [
            trial
            for trial in self.study_.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
        ]
        if not completed_trials:
            raise ValueError("所有Trial均失败，请检查模型参数、数据和训练异常")

        # 保存结果
        self._save_results()

        self.optimization_history_ = self._build_public_history()

        if self.verbose:
            self._print_tuning_summary()

        return self.best_params_

    def _format_scores(self, values: Optional[Sequence[float]]) -> str:
        """将单目标或多目标得分格式化为稳定、易读的日志文本。"""
        if values is None:
            return "不可用"

        formatted = []
        for name, value in zip(self.metric_names, values):
            score = "不可用" if value is None else f"{float(value):.6f}"
            formatted.append(f"{name}={score}")
        return ", ".join(formatted) if formatted else "不可用"

    def _print_trial_progress(self, study: Any, trial: Any) -> None:
        """在 Trial 结束后立即输出本次结果和当前最佳结果。"""
        params = self._get_params_from_trial(trial)
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.values is None:
            print(f"[调参] Trial {trial.number} {trial.state.name} | 参数: {params}", flush=True)
            return

        if self._is_multi_objective:
            best_trial = self._select_best_pareto_trial(study.best_trials)
        else:
            best_trial = study.best_trial

        print(
            f"[调参] Trial {trial.number} 完成 | 得分: {self._format_scores(trial.values)} | "
            f"参数: {params} | 当前最佳: {self._format_scores(best_trial.values)} "
            f"(Trial {best_trial.number})",
            flush=True,
        )

    def _print_tuning_summary(self) -> None:
        """在调参正常完成并保存结果后输出最终摘要。"""
        completed_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in self.study_.trials)
        print(f"[调参] 调参完成 | 完成 Trial: {completed_trials}", flush=True)
        if self._is_multi_objective:
            print(f"[调参] 帕累托最优解: {len(self.study_.best_trials)}", flush=True)
        print(f"[调参] 最佳得分: {self._format_scores(self.best_scores_)}", flush=True)
        print(f"[调参] 最佳参数: {self.best_params_}", flush=True)

    def _inject_model_parallel_budget(self, params: Dict[str, Any], workers: int) -> None:
        """把调参总预算写入当前模型公开的最外层原生并行参数。"""
        try:
            signature = inspect.signature(self.model_class.__init__)
        except (TypeError, ValueError, AttributeError):
            return

        for parameter_name in ("n_jobs", "thread_count", "num_workers"):
            if parameter_name not in signature.parameters:
                continue
            configured = params.get(parameter_name)
            if configured is None or configured == -1:
                params[parameter_name] = workers
            else:
                try:
                    params[parameter_name] = min(max(1, int(configured)), workers)
                except (TypeError, ValueError):
                    params[parameter_name] = workers
            return

    def _evaluate_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, Tuple[float, ...]]:
        """评估模型，返回一个或多个指标值."""
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        fold_results = {i: [] for i in range(len(self.metrics))}

        for train_idx, val_idx in kf.split(X, y):
            X_train_fold, X_val_fold = _safe_index(X, train_idx), _safe_index(X, val_idx)
            y_train_fold, y_val_fold = _safe_index(y, train_idx), _safe_index(y, val_idx)
            sample_weight_fold = _safe_index(sample_weight, train_idx)

            try:
                fold_model = clone(model)
            except Exception:
                fold_model = copy.deepcopy(model)

            if sample_weight_fold is None:
                fold_model.fit(X_train_fold, y_train_fold)
            else:
                fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight_fold)

            y_train_pred = fold_model.predict_proba(X_train_fold)[:, 1]
            y_val_pred = fold_model.predict_proba(X_val_fold)[:, 1]

            for i, metric in enumerate(self.metrics):
                y_val_arr = y_val_fold.values if hasattr(y_val_fold, "values") else np.asarray(y_val_fold)
                y_train_arr = y_train_fold.values if hasattr(y_train_fold, "values") else np.asarray(y_train_fold)
                value = metric(y_val_arr, y_val_pred, y_train=y_train_arr, y_train_pred=y_train_pred)
                fold_results[i].append(value)

        results = [np.mean(fold_results[i]) for i in range(len(self.metrics))]
        if self._is_multi_objective:
            return tuple(results)
        return results[0]

    def _save_results(self):
        """保存优化结果."""
        if self._is_multi_objective:
            # 多目标优化
            self.pareto_front_ = self.study_.best_trials

            # 在帕累托前沿中按指标顺序做确定性选择：优先第一个主指标，
            # 主指标相同时再按后续指标方向排序。
            best_trial = self._select_best_pareto_trial(self.study_.best_trials)
            self.best_params_ = self._get_params_from_trial(best_trial)
            self.best_scores_ = list(best_trial.values)
            self.best_score_ = self.best_scores_[0]  # 第一个指标作为主指标
        else:
            # 单目标优化
            self.best_params_ = self._get_params_from_trial(self.study_.best_trial)
            self.best_score_ = self.study_.best_value
            self.best_scores_ = [self.best_score_]

        self.best_params_.update(self.fixed_params)
        self.best_params_ = self._apply_model_param_constraints(self.best_params_)

    def _select_best_pareto_trial(self, trials: Sequence[Any]) -> Any:
        """从帕累托前沿按主指标优先规则选择一个默认最优 trial."""
        if not trials:
            raise ValueError("没有可用的帕累托最优解")

        def sort_key(trial):
            values = trial.values or []
            key = []
            for value, direction in zip(values, self.directions):
                if value is None:
                    adjusted = float("-inf") if direction == "maximize" else float("inf")
                else:
                    adjusted = value if direction == "maximize" else -value
                key.append(adjusted)
            # trial.number 取负值，让完全同分时选择更早完成的 trial。
            key.append(-trial.number)
            return tuple(key)

        return max(trials, key=sort_key)

    def evaluate_trials(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        trial_points: Optional[List[Dict[str, Any]]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """评估指定超参数点的模型效果.

        无需运行完整调优，直接评估给定超参数配置的性能。

        支持两种调用风格:

        **sklearn风格**::

            results = tuner.evaluate_trials(X_train, y_train, trial_points)

        **scorecardpipeline风格** (在__init__中指定target)::

            tuner = ModelTuner(..., target='label')
            results = tuner.evaluate_trials(df, trial_points=trial_points)

        :param X: 特征矩阵，或包含目标列的DataFrame（scorecardpipeline风格）
        :param y: 目标变量，可选。如果为None，则从X中提取target列
        :param trial_points: 超参数点列表，每个点是一个参数字典
        :param sample_weight: 样本权重，可选
        :return: 包含评估结果的DataFrame
        """
        # 检查trial_points
        if trial_points is None:
            raise ValueError("trial_points不能为空，请提供要评估的超参数点列表")

        # 检查并处理输入
        X, y = self._check_input(X, y)

        results = []

        for i, params in enumerate(trial_points):
            if self.verbose:
                logger.info(f"评估 trial point {i+1}/{len(trial_points)}: {params}")

            # 合并固定参数
            full_params = dict(params)
            full_params.update(self.fixed_params)
            full_params = self._apply_model_param_constraints(full_params)
            self._inject_fit_params(full_params)

            # 创建模型并评估
            model = self.model_class(**full_params)
            metric_values = self._evaluate_model(model, X, y, sample_weight)

            if self._is_multi_objective:
                result = {"trial_id": i, **params, **{name: val for name, val in zip(self.metric_names, metric_values)}}
            else:
                result = {"trial_id": i, **params, self.metric_names[0]: metric_values}

            results.append(result)

        return pd.DataFrame(results)

    def evaluate_study_trials(
        self,
        trial_indices: Optional[Union[int, Sequence[int]]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """评估已完成 study 中指定 trial 的模型效果.

        从 ``self.study_.trials[i]`` 取出对应超参数重新评估，便于复核某次
        采样的稳定性、或在新数据集上对比若干历史 trial 的效果。

        与 :meth:`evaluate_trials` 的区别：本方法的超参数来自已学习完成的
        study（按 trial 索引取），而非外部传入的参数点；结果额外包含每个 trial
        的索引、状态及 study 记录的原始得分（``study记录值`` 列），便于与重新
        评估的得分对照。

        :param trial_indices: 要评估的 trial 索引，可选:
            - None: 评估全部已完成（COMPLETE）的 trial
            - int: 评估单个 trial，如 ``0`` 或 ``tuner.study_.best_trial.number``
            - 序列: 评估多个 trial，如 ``[0, 5, 10]``
        :param X: 特征矩阵，或包含目标列的DataFrame；默认复用 fit 时的训练数据
        :param y: 目标变量，可选；默认复用 fit 时的标签
        :param sample_weight: 样本权重，可选；默认复用 fit 时的样本权重
        :return: 包含评估结果的DataFrame，含 ``trial索引``/``trial状态``/超参数/
            重新评估指标/``study记录值`` 列

        Example:
            >>> tuner.fit(X_train, y_train, n_trials=100)
            >>> # 评估最优 trial 与前两个 trial
            >>> tuner.evaluate_study_trials([tuner.study_.best_trial.number, 0, 1])
            >>> # 在新数据集上复核全部 trial
            >>> tuner.evaluate_study_trials(X=X_oot, y=y_oot)
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优，再评估study中的trial")

        all_trials = self.study_.trials
        n_trials = len(all_trials)

        # 归一化 trial_indices
        if trial_indices is None:
            indices = [t.number for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not indices:
                raise ValueError("study中没有已完成（COMPLETE）的trial可供评估")
        elif isinstance(trial_indices, int):
            indices = [trial_indices]
        else:
            indices = list(trial_indices)

        # 校验索引合法性
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                raise ValueError(f"trial索引必须为整数，收到: {idx!r}")
            if idx < 0 or idx >= n_trials:
                raise ValueError(f"trial索引 {idx} 超出范围，study共有 {n_trials} 个trial（有效索引 0~{n_trials - 1}）")

        # 默认复用 fit 时的数据
        if X is None:
            if getattr(self, "_X", None) is None:
                raise ValueError("未提供X且fit时未缓存训练数据，请显式传入X/y")
            X, y = self._X, self._y
            if sample_weight is None:
                sample_weight = getattr(self, "_sample_weight", None)
        else:
            X, y = self._check_input(X, y)

        results = []

        for idx in indices:
            trial = all_trials[idx]
            params = self._get_params_from_trial(trial)

            if self.verbose:
                logger.info(f"评估 study trial #{idx} (state={trial.state.name}): {params}")

            # 合并固定参数并按模型签名注入早停参数
            full_params = dict(params)
            full_params.update(self.fixed_params)
            full_params = self._apply_model_param_constraints(full_params)
            self._inject_fit_params(full_params)

            # 创建模型并评估
            model = self.model_class(**full_params)
            metric_values = self._evaluate_model(model, X, y, sample_weight)

            # study 记录的原始得分（用于与重新评估结果对照）
            recorded = list(trial.values) if trial.values is not None else None

            result = {"trial索引": idx, "trial状态": trial.state.name, **params}
            if self._is_multi_objective:
                result.update({name: val for name, val in zip(self.metric_names, metric_values)})
                if recorded is not None:
                    result["study记录值"] = recorded
            else:
                result[self.metric_names[0]] = metric_values
                if recorded is not None:
                    result["study记录值"] = recorded[0]

            results.append(result)

        return pd.DataFrame(results)

    def _get_params_from_trial(self, trial) -> Dict[str, Any]:
        """从trial中获取参数."""
        return self._apply_model_param_constraints(self._space_adapter.public_params(trial))

    def _build_public_history(self) -> pd.DataFrame:
        """生成只包含模型最终参数名和值的 Optuna 历史表。"""
        history = self.study_.trials_dataframe()
        latent_columns = [column for column in history if column.startswith("params___hscredit__")]
        history = history.drop(columns=latent_columns, errors="ignore")
        for name in self.search_space:
            column = f"params_{name}"
            values = []
            for trial in self.study_.trials:
                params = self._space_adapter.public_params(trial)
                params.update(self.fixed_params)
                values.append(self._apply_model_param_constraints(params).get(name))
            history[column] = values
        return history

    def _get_adaptive_search_space(self) -> Dict[str, Dict[str, Any]]:
        """根据数据特征获取自适应搜索空间.

        基于内部建模经验，根据样本量和特征数调整搜索范围。
        """
        # 获取模型类型
        model_name = self.model_class.__name__.lower()

        if "xgboost" in model_name or "xgb" in model_name:
            return self._get_xgboost_search_space()
        elif "lightgbm" in model_name or "lgb" in model_name:
            return self._get_lightgbm_search_space()
        elif "catboost" in model_name or "cat" in model_name:
            return self._get_catboost_search_space()
        elif "randomforest" in model_name or "extratrees" in model_name or "rf" in model_name:
            # ExtraTrees 与 RandomForest 参数一致，共用搜索空间
            return self._get_randomforest_search_space()
        elif "gradientboosting" in model_name or "gbdt" in model_name:
            return self._get_gradientboosting_search_space()
        elif "ngboost" in model_name or "ngb" in model_name:
            return self._get_ngboost_search_space()
        elif "logistic" in model_name or model_name in ("lr",):
            return self._get_logisticregression_search_space()
        elif model_name == "svm":
            return self._get_svm_search_space()
        elif model_name == "decisiontreeclassifier":
            return self._get_decisiontree_search_space()
        else:
            # 默认使用XGBoost搜索空间
            return self._get_xgboost_search_space()

    def _get_xgboost_search_space(self) -> Dict[str, Dict[str, Any]]:
        """XGBoost搜索空间 - 基于内部建模经验.

        参考内部代码（强正则、浅树、小学习率以抑制风控样本过拟合）:
        - max_depth: 风控场景通常2-4，防止过拟合
        - min_child_weight: 8-256（step 4），叶子最小样本权重，越大越保守
        - subsample: 0.35-0.85，行采样
        - colsample_bytree: 0.4-0.9，列采样
        - gamma: 0.0-32.0，分裂最小损失下降，越大越保守
        - scale_pos_weight: 16.0-32.0，正样本权重（适配低坏率不平衡场景）
        - reg_alpha: 0.0-1.0（L1 正则）
        - reg_lambda: 32.0-128.0（L2 正则，强约束）
        - learning_rate: 0.0001-0.01，较小学习率更稳定
        - n_estimators: 32-256（step 16）

        固定项 ``objective='binary:logistic'`` / ``eval_metric='auc'`` /
        ``booster='gbtree'`` / ``importance_type='cover'`` 已是模型默认值，
        如需覆盖可通过 ``ModelTuner(fixed_params=...)`` 传入。
        """
        return {
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
            "n_estimators": {"type": "int", "low": 32, "high": 256, "step": 16},
            "min_child_weight": {"type": "int", "low": 8, "high": 256, "step": 4},
            "subsample": {"type": "float", "low": 0.35, "high": 0.85},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.9},
            "gamma": {"type": "float", "low": 0.0, "high": 32.0},
            "scale_pos_weight": {"type": "float", "low": 16.0, "high": 32.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 32.0, "high": 128.0},
        }

    def _get_lightgbm_search_space(self) -> Dict[str, Dict[str, Any]]:
        """LightGBM搜索空间 - 与XGBoost搜索空间对齐.

        参考内部代码（参数范围与 XGBoost 保持一致的建模经验）:
        - num_leaves: 与max_depth相关，受 ``2**max_depth`` 上界约束（见 _sample_params）
        - max_depth: 风控场景通常2-4，防止过拟合
        - min_child_samples: 8-256（step 4），叶子最小样本数，越大越保守
        - subsample: 0.35-0.85，行采样
        - colsample_bytree: 0.4-0.9，列采样
        - min_split_gain: 0.0-32.0，分裂最小增益（对应 XGBoost 的 gamma）
        - scale_pos_weight: 16.0-32.0，正样本权重
        - reg_alpha: 0.0-1.0（L1 正则）
        - reg_lambda: 32.0-128.0（L2 正则，强约束）
        - learning_rate: 0.0001-0.01，较小学习率更稳定
        - n_estimators: 32-256（step 16）
        """
        return {
            "num_leaves": {"type": "int", "low": 8, "high": 64},
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
            "n_estimators": {"type": "int", "low": 32, "high": 256, "step": 16},
            "min_child_samples": {"type": "int", "low": 8, "high": 256, "step": 4},
            "subsample": {"type": "float", "low": 0.35, "high": 0.85},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.9},
            "min_split_gain": {"type": "float", "low": 0.0, "high": 32.0},
            "scale_pos_weight": {"type": "float", "low": 16.0, "high": 32.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 32.0, "high": 128.0},
        }

    def _get_logisticregression_search_space(self) -> Dict[str, Dict[str, Any]]:
        """逻辑回归搜索空间 - 基于内部建模经验.

        参考内部代码:
        - C: 正则强度倒数，对数区间 0.01-32（越小正则越强）
        - penalty: 仅 'l2'（评分卡常用，兼容多数 solver）
        - class_weight: None / 'balanced' / 自定义正负样本权重字典（适配不平衡场景）
        - max_iter: 16-256（对数区间），迭代上限
        - solver: liblinear / sag / lbfgs / newton-cg

        .. note::
            ``class_weight`` 的字典候选会触发 optuna 关于非基础类型 categorical 的
            提示（内存存储下可正常工作）；若需持久化 study，可改用 None/'balanced'。
        """
        return {
            "C": {"type": "float", "low": 0.01, "high": 32.0, "log": True},
            "penalty": {"type": "categorical", "choices": ["l2"]},
            "class_weight": {
                "type": "categorical",
                "choices": [None, "balanced"] + [{1: i / 10.0, 0: 1 - i / 10.0} for i in range(1, 10, 2)],
            },
            "max_iter": {"type": "int", "low": 16, "high": 256, "log": True},
            "solver": {
                "type": "categorical",
                "choices": ["liblinear", "sag", "lbfgs", "newton-cg"],
            },
        }

    def _get_svm_search_space(self) -> Dict[str, Dict[str, Any]]:
        """SVC 搜索空间，始终保留 probability=True 的模型固定契约。"""
        return {
            "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"]},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            "degree": {"type": "int", "low": 2, "high": 5},
            "coef0": {"type": "float", "low": 0.0, "high": 1.0},
        }

    def _get_decisiontree_search_space(self) -> Dict[str, Dict[str, Any]]:
        """sklearn 决策树搜索空间。"""
        return {
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "min_samples_split": {"type": "int", "low": 2, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.05},
        }

    def _get_catboost_search_space(self) -> Dict[str, Dict[str, Any]]:
        """CatBoost搜索空间 - 基于风控场景优化.

        参考内部代码:
        - depth: 风控场景通常2-5，防止过拟合
        - learning_rate: 0.005-0.1，较小学习率更稳定
        - iterations: 50-500
        - l2_leaf_reg: 1e-8到10
        """
        return {
            "depth": {"type": "int", "low": 2, "high": 5},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "iterations": {"type": "int", "low": 50, "high": 500},
            "l2_leaf_reg": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "border_count": {"type": "int", "low": 32, "high": 255},
            "random_strength": {"type": "float", "low": 0.0, "high": 10.0},
        }

    def _get_randomforest_search_space(self) -> Dict[str, Dict[str, Any]]:
        """RandomForest搜索空间 - 基于内部建模经验.

        参考内部代码:
        - max_depth: 风控场景通常2-5，防止过拟合
        - n_estimators: 根据样本量调整
        """
        n_samples = self._n_samples or 10000

        # 根据样本量调整n_estimators
        if n_samples > 10000:
            n_estimators_high = 500
            n_estimators_low = 100
        else:
            n_estimators_high = 200
            n_estimators_low = 50

        return {
            "n_estimators": {"type": "int", "low": n_estimators_low, "high": n_estimators_high},
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        }

    def _get_ngboost_search_space(self) -> Dict[str, Dict[str, Any]]:
        """NGBoost搜索空间 - 基于风控场景优化.

        NGBoost 使用 CART 作为基学习器，参数名与其他 boosting 不同：
        - n_estimators: 自然梯度提升轮数，较小学习率需更多轮
        - learning_rate: 0.005-0.1，较小学习率更稳定
        - base_max_depth: 基学习器（CART）最大深度，风控场景通常2-4
        - minibatch_frac: 小批量采样比例（行采样）
        - col_sample: 特征采样比例
        """
        n_samples = self._n_samples or 10000
        if n_samples > 10000:
            n_estimators_low, n_estimators_high = 200, 800
        else:
            n_estimators_low, n_estimators_high = 100, 500

        return {
            "n_estimators": {"type": "int", "low": n_estimators_low, "high": n_estimators_high},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "base_max_depth": {"type": "int", "low": 2, "high": 4},
            "minibatch_frac": {"type": "float", "low": 0.5, "high": 1.0},
            "col_sample": {"type": "float", "low": 0.5, "high": 1.0},
        }

    def _get_gradientboosting_search_space(self) -> Dict[str, Dict[str, Any]]:
        """GradientBoosting搜索空间 - 基于风控场景优化.

        参考内部代码:
        - max_depth: 风控场景通常2-5，防止过拟合
        - learning_rate: 0.005-0.1，较小学习率更稳定
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        }

    @staticmethod
    def _normalize_trial_points(
        trial_points: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """将 trial_points 归一化为 list[dict].

        :param trial_points: ``None`` / 单个 dict / list[dict]
        :return: 参数点列表（可能为空）
        """
        if trial_points is None:
            return []
        if isinstance(trial_points, dict):
            return [dict(trial_points)]
        if isinstance(trial_points, (list, tuple)):
            for p in trial_points:
                if not isinstance(p, dict):
                    raise ValueError(f"trial_points 中每个元素必须为 dict，收到: {type(p).__name__}")
            return [dict(p) for p in trial_points]
        raise ValueError(f"trial_points 必须为 dict 或 list[dict]，收到: {type(trial_points).__name__}")

    def enqueue_trial(
        self,
        params: Dict[str, Any],
        user_attrs: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = False,
    ) -> "ModelTuner":
        """按 Optuna ``Study.enqueue_trial`` 风格追加一个手工搜索点。

        ``params`` 使用模型最终参数名和值。若某一声明需要内部潜变量采样，本方法
        会先完成逆变换，再把内部参数传给 Study；公开记录仍保留最终值。
        """
        public_point = dict(params)
        attrs = dict(user_attrs) if user_attrs is not None else None
        point_with_fixed = dict(public_point)
        point_with_fixed.update(self.fixed_params)
        self._validate_lightgbm_leaf_point(point_with_fixed)
        self.trial_points.append(public_point)
        if self.search_space is None:
            self._pending_public_trials.append((public_point, attrs, bool(skip_if_exists)))
            return self
        internal_point = self._space_adapter.to_internal_point(public_point)
        if self.study_ is not None:
            self.study_.enqueue_trial(internal_point, user_attrs=attrs, skip_if_exists=skip_if_exists)
            if self.verbose:
                logger.info(f"已入队手工搜索点: {public_point}")
        else:
            self._pending_trials.append((internal_point, attrs, bool(skip_if_exists)))
        return self

    def _ordered_point(self, values: Sequence[Any], source: str) -> Dict[str, Any]:
        """按搜索空间声明顺序把序列点转换为参数字典。"""
        values = list(values)
        names = self._space_adapter.names
        if len(values) != len(names):
            raise ValueError(f"{source} 搜索点维度数量为 {len(values)}，搜索空间要求 {len(names)}")
        return dict(zip(names, values))

    def enqueue_trials(
        self,
        trial_points: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        param_grid: Optional[Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]]] = None,
        x0: Optional[Sequence[Any]] = None,
        user_attrs: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = False,
    ) -> "ModelTuner":
        """按 Optuna、GridSearch 或 skopt 格式追加一个或多个搜索点。

        若 study 已创建（已调用过 fit），则立即通过 ``study.enqueue_trial`` 入队，
        在后续 ``fit`` 的采样中优先评估；否则缓存到 ``self.trial_points``，
        在下次 ``fit`` 创建 study 后入队。

        :param trial_points: Optuna/hscredit 格式，``dict`` 或 ``list[dict]``
        :param param_grid: GridSearch 格式，由 ``ParameterGrid`` 展开
        :param x0: skopt 格式，单个值序列或多个值序列，顺序与搜索空间一致
        :return: self，便于链式调用
        """
        supplied = sum(value is not None for value in (trial_points, param_grid, x0))
        if supplied != 1:
            raise ValueError("enqueue_trials 必须且只能提供 trial_points、param_grid 或 x0 中的一项")
        if param_grid is not None:
            points = [dict(point) for point in ParameterGrid(param_grid)]
        elif x0 is not None:
            raw = list(x0)
            if not raw:
                raise ValueError("x0 不能为空")
            first = raw[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                points = [self._ordered_point(row, "x0") for row in raw]
            else:
                points = [self._ordered_point(raw, "x0")]
        else:
            points = self._normalize_trial_points(trial_points)
        for point in points:
            self.enqueue_trial(point, user_attrs=user_attrs, skip_if_exists=skip_if_exists)
        return self

    def probe(
        self,
        params: Union[Dict[str, Any], Sequence[Any]],
        lazy: bool = True,
    ) -> "ModelTuner":
        """按 bayesian-optimization ``probe`` 风格追加一个搜索点。

        ``lazy`` 为兼容原方法保留；Optuna 后端无立即执行单点的等价操作，因此
        ``True`` 与 ``False`` 都会进入同一个 Study 队列，并在下一次 optimize 时执行。
        """
        del lazy
        point = dict(params) if isinstance(params, dict) else self._ordered_point(params, "probe")
        return self.enqueue_trial(point)

    def _enqueue_trial_points(self) -> None:
        """将 self.trial_points 入队到当前 study（fit 内部调用）."""
        for public_point, user_attrs, skip_if_exists in self._pending_public_trials:
            internal_point = self._space_adapter.to_internal_point(public_point)
            self._pending_trials.append((internal_point, user_attrs, skip_if_exists))
        self._pending_public_trials.clear()
        for point, user_attrs, skip_if_exists in self._pending_trials:
            self.study_.enqueue_trial(point, user_attrs=user_attrs, skip_if_exists=skip_if_exists)
            if self.verbose:
                logger.info(f"已入队预指定手工搜索点: {point}")
        self._pending_trials.clear()

    def _inject_fit_params(self, params: Dict[str, Any]) -> None:
        """按模型构造函数签名注入早停/验证集参数（原地修改 params）.

        Boosting 模型（XGBoost/LightGBM/CatBoost 等）将 ``early_stopping_rounds``
        与 ``validation_fraction`` 声明为显式构造参数，注入可启用调参过程中的早停；
        而逻辑回归、sklearn 集成模型（RandomForest/ExtraTrees）等不支持这些参数，
        直接注入会触发 TypeError。

        仅当参数是模型构造函数**显式声明**的命名参数时才注入：不依赖 ``**kwargs``，
        因为 SklearnRiskModel 子类虽有 ``**kwargs`` 但会在内部硬编码
        ``early_stopping_rounds=None`` 转发，经 ``**kwargs`` 再次传入会导致
        "multiple values for keyword argument" 冲突。

        :param params: 待注入的参数字典，将被原地更新
        """
        import inspect

        try:
            accepted = set(inspect.signature(self.model_class.__init__).parameters)
        except (TypeError, ValueError):
            accepted = set()

        fit_params = {
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_fraction": 0.2,
        }
        for name, value in fit_params.items():
            if name in accepted:
                params.setdefault(name, value)

    def _sample_params(self, trial: "Trial") -> Dict[str, Any]:
        """从搜索空间采样参数.

        :param trial: Optuna trial对象
        :return: 参数字典
        """
        params = self._space_adapter.sample(trial)
        return self._apply_model_param_constraints(params)

    def _uses_lightgbm_leaf_constraint(self) -> bool:
        """当前模型是否使用 LightGBM 的叶子数/深度约束。"""
        model_name = getattr(self.model_class, "__name__", "").lower()
        return "lightgbm" in model_name or "lgbm" in model_name

    def _leaf_limit(self, params: Dict[str, Any]) -> Optional[int]:
        """根据正的整数 max_depth 计算 LightGBM num_leaves 上限。"""
        if not self._uses_lightgbm_leaf_constraint():
            return None
        max_depth = params.get("max_depth")
        if isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer)):
            return None
        if max_depth <= 0:
            return None
        return 2 ** int(max_depth)

    def _apply_model_param_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """把模型关联约束应用到最终模型参数，不改变 Optuna 的稳定搜索分布。"""
        constrained = dict(params)
        limit = self._leaf_limit(constrained)
        num_leaves = constrained.get("num_leaves")
        if (
            limit is not None
            and isinstance(num_leaves, (int, np.integer))
            and not isinstance(num_leaves, (bool, np.bool_))
        ):
            constrained["num_leaves"] = min(int(num_leaves), limit)
        return constrained

    def _validate_lightgbm_leaf_point(self, params: Dict[str, Any]) -> None:
        """拒绝显式给出的无效 LightGBM 深度/叶子数组合。"""
        limit = self._leaf_limit(params)
        num_leaves = params.get("num_leaves")
        if limit is not None and isinstance(num_leaves, (int, np.integer)) and num_leaves > limit:
            raise ValueError(f"LightGBM 手工搜索点 num_leaves={num_leaves} 不能大于 " f"2**max_depth={limit}")

    def _sample_normal(self, trial: "Trial", param_name: str, param_config: Dict[str, Any]) -> float:
        """从截断正态/对数正态分布采样（hyperopt normal/lognormal 近似）。

        optuna 无原生正态采样，在 [0,1] 均匀采样后经逆 CDF 变换为目标分布，
        保证 optuna 可记录与复现。截断区间取 [mu-4σ, mu+4σ]，log 时为对数空间。

        :param trial: Optuna trial 对象
        :param param_name: 参数名
        :param param_config: 'normal' DSL 配置（含 mu/sigma/low/high/q/log）
        :return: 采样值
        """
        return self._space_adapter.sample_one(trial, param_name, param_config)

    def get_best_model(self) -> Any:
        """获取使用最佳参数的模型实例.

        :return: 训练好的模型实例
        """
        if self.best_params_ is None:
            raise ValueError("请先调用fit()进行调优")

        model = self.model_class(**self.best_params_)
        if self._sample_weight is None:
            model.fit(self._X, self._y)
        else:
            model.fit(self._X, self._y, sample_weight=self._sample_weight)
        return model

    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史.

        :return: 优化历史DataFrame
        """
        if self.optimization_history_ is None:
            raise ValueError("请先调用fit()进行调优")

        return self.optimization_history_

    def get_pareto_front(self) -> Optional[List]:
        """获取帕累托前沿（多目标优化时）.

        :return: 帕累托前沿上的trial列表
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        if not self._is_multi_objective:
            raise ValueError("单目标优化没有帕累托前沿")

        return self.study_.best_trials

    def _resolve_multi_objective_target(self, target: Optional[int]) -> Optional[int]:
        """多目标分析图/重要性默认使用第一个指标，并校验索引范围."""
        if not self._is_multi_objective:
            return target
        if target is None:
            return 0
        if not isinstance(target, (int, np.integer)):
            raise ValueError("target 必须是指标索引整数")
        if target < 0 or target >= len(self.metric_names):
            raise ValueError(f"target 超出范围，多目标指标索引有效范围为 0~{len(self.metric_names) - 1}")
        return int(target)

    def get_param_importance(self, target: Optional[int] = None) -> Optional[pd.Series]:
        """获取参数重要性.

        :param target: 多目标时指定要分析的指标索引，默认第一个
        :return: 参数重要性Series
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        try:
            target = self._resolve_multi_objective_target(target)
            if self._is_multi_objective:
                # 多目标优化时，可以指定特定目标
                importance = optuna.importance.get_param_importances(self.study_, target=lambda t: t.values[target])
            else:
                importance = optuna.importance.get_param_importances(self.study_)
            return pd.Series(importance)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"无法计算参数重要性: {e}")
            return None

    # ==================== 可视化方法 ====================

    def plot_optimization_history(self, target: Optional[int] = None, **kwargs):
        """绘制优化历史.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_optimization_history(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )

        return optuna.visualization.plot_optimization_history(self.study_, **kwargs)

    def plot_param_importances(self, target: Optional[int] = None, **kwargs):
        """绘制参数重要性.

        :param target: 多目标时指定要分析的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_param_importances(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )

        return optuna.visualization.plot_param_importances(self.study_, **kwargs)

    def plot_slice(self, target: Optional[int] = None, **kwargs):
        """绘制参数切片图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_slice(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )

        return optuna.visualization.plot_slice(self.study_, **kwargs)

    def plot_pareto_front(self, **kwargs):
        """绘制帕累托前沿（多目标优化时）.

        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        if not self._is_multi_objective:
            raise ValueError("只有多目标优化才能绘制帕累托前沿")

        return optuna.visualization.plot_pareto_front(self.study_, target_names=self.metric_names, **kwargs)

    def plot_contour(self, params: Optional[List[str]] = None, target: Optional[int] = None, **kwargs):
        """绘制参数等高线图.

        :param params: 要绘制的参数列表，默认前两个
        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        if params is None:
            params = list(self.search_space.keys())[:2]

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_contour(
                self.study_,
                params=params,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs,
            )

        return optuna.visualization.plot_contour(self.study_, params=params, **kwargs)

    def plot_parallel_coordinate(self, target: Optional[int] = None, **kwargs):
        """绘制平行坐标图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_parallel_coordinate(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )

        return optuna.visualization.plot_parallel_coordinate(self.study_, **kwargs)

    def plot_edf(self, target: Optional[int] = None, **kwargs):
        """绘制经验分布函数图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            return optuna.visualization.plot_edf(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )

        return optuna.visualization.plot_edf(self.study_, **kwargs)


class AutoTuner:
    """自动调优器 - 基于内部建模经验.

    为常见模型提供预定义的搜索空间，并根据数据特征自动调整。

    **参考样例**

    >>> from hscredit.core.models import AutoTuner
    >>>
    >>> # 自动根据数据特征选择搜索空间
    >>> tuner = AutoTuner.create('xgboost', metric='ks')
    >>> best_params = tuner.fit(X_train, y_train, n_trials=50)
    >>>
    >>> # 使用多目标优化（KS + 稳定性）
    >>> tuner = AutoTuner.create('lightgbm', metric=['ks', 'ks_diff'])
    >>> best_params = tuner.fit(X_train, y_train, n_trials=100)
    >>>
    >>> # 使用自定义指标
    >>> def my_metric(y_true, y_pred):
    ...     return custom_score(y_true, y_pred)
    >>>
    >>> tuner = AutoTuner.create('xgboost', metric=my_metric, direction='maximize')
    >>> best_params = tuner.fit(X_train, y_train, n_trials=100)
    """

    @classmethod
    def create(
        cls,
        model_type: str,
        metric: Union[str, Callable, List[Union[str, Callable]]] = "ks",
        direction: Union[str, List[str]] = "maximize",
        metric_names: Optional[List[str]] = None,
        target: str = "target",
        cv: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> ModelTuner:
        """创建自动调优器.

        :param model_type: 模型类型，可选:
            - 'xgboost' / 'xgb'
            - 'lightgbm' / 'lgb'
            - 'catboost' / 'cat'
            - 'ngboost' / 'ngb'
            - 'randomforest' / 'rf'
            - 'gradientboosting' / 'gbdt'
            - 'logisticregression' / 'lr'
            - 'svm' / 'svc'
            - 'decisiontree' / 'dt'
        :param metric: 优化指标，可以是字符串、函数或列表
        :param direction: 优化方向，单目标时str，多目标时list
        :param metric_names: 指标名称列表（多目标时用于显示）
        :param target: 目标列名，用于scorecardpipeline风格的fit，默认'target'
        :param cv: 交叉验证折数，默认5
        :param random_state: 随机种子
        :param verbose: 是否输出详细信息
        :param early_stopping_rounds: 早停轮数，默认20
        :param kwargs: 其他参数
        :return: ModelTuner实例
        """
        from .. import (
            XGBoost,
            LightGBM,
            CatBoost,
            NGBoost,
            RandomForest,
            ExtraTrees,
            GradientBoosting,
            LogisticRegression,
            SVM,
            DecisionTreeClassifier,
        )

        model_map = {
            "xgboost": XGBoost,
            "xgb": XGBoost,
            "lightgbm": LightGBM,
            "lgb": LightGBM,
            "catboost": CatBoost,
            "cat": CatBoost,
            "ngboost": NGBoost,
            "ngb": NGBoost,
            "randomforest": RandomForest,
            "rf": RandomForest,
            "extratrees": ExtraTrees,
            "et": ExtraTrees,
            "gradientboosting": GradientBoosting,
            "gbdt": GradientBoosting,
            "logisticregression": LogisticRegression,
            "lr": LogisticRegression,
            "svm": SVM,
            "svc": SVM,
            "decisiontree": DecisionTreeClassifier,
            "dt": DecisionTreeClassifier,
        }

        model_type = model_type.lower()
        if model_type not in model_map:
            raise ValueError(f"未知模型类型: {model_type}")

        model_class = model_map[model_type]

        return ModelTuner(
            model_class=model_class,
            search_space=None,
            metric=metric,
            direction=direction,
            metric_names=metric_names,
            target=target,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )  # 使用自适应搜索空间
