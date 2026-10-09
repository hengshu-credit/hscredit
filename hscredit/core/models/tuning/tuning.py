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
from ...._compat import installed_version, needs_logistic_regression_parallel_compat
from ....utils.parallel import resolve_n_jobs
from ....utils.serialization import ArtifactSerializableMixin
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import get_scorer, log_loss, roc_curve
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from ...metrics import auc as auc_metric
from .._contracts import (
    extract_target,
    positive_probability,
    split_sample_params,
    take_rows,
    validate_labels,
    validate_sample_weight,
)
from ..losses.base import BaseMetric
from ._retention import compact_fold, load_fold, resolve_retention, retain_fold, trial_path

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


from .space_adapter import SearchSpaceAdapter, normalize_search_space


def _normalize_space_param(name, spec):
    """旧私有入口委托统一空间校验，避免两套实现逐渐产生不同语义。"""
    return normalize_search_space({name: spec})[name]


# 保留既有私有导入路径，但只维护一套真实解析实现。
_space_param_from_dict = _normalize_space_param
_space_param_from_tuple = _normalize_space_param
_space_param_from_scipy = _normalize_space_param
_space_param_from_optuna = _normalize_space_param
_legacy_normalize_search_space = normalize_search_space


def _check_bounds(name, spec, integer):
    """兼容旧参数边界检查。"""
    _normalize_space_param(name, {**spec, "type": "int" if integer else "float"})


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
        """校验采样器参数后实例化；仅对不支持随机种子的采样器移除 seed。"""
        import inspect

        try:
            sig = inspect.signature(sampler_cls.__init__)
            accepted = set(sig.parameters)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except (TypeError, ValueError):
            accepted, accepts_var_kw = set(), True

        if not accepts_var_kw:
            unknown = set(kwargs) - accepted - {"seed"}
            if unknown:
                raise ValueError(f"采样器 {sampler_cls.__name__} 不支持参数 {sorted(unknown)}，请检查 sampler_kwargs")
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
        :param kwargs: 透传给采样器构造函数的额外参数；拼错或不支持的参数立即报错。
            传入已创建的采样器实例时不能再传 kwargs，实例自身管理随机种子。
        :return: optuna 采样器实例

        **参考样例**

        >>> sampler = TuningSampler.create("tpe", seed=42, n_startup_trials=5)
        >>> tuner = ModelTuner(LogisticRegression, sampler=sampler)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请使用 pip install optuna 安装")

        # 已是采样器实例，直接返回
        if sampler is None:
            sampler = "tpe"
        if not isinstance(sampler, str):
            if not isinstance(sampler, optuna.samplers.BaseSampler):
                raise TypeError("sampler 必须是支持的名称或已创建的 Optuna BaseSampler 实例")
            if kwargs:
                raise ValueError("传入采样器实例时不能再传 sampler_kwargs，请在创建实例时配置参数")
            return sampler

        key = sampler.lower()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed

        if key in cls.BUILTIN_SAMPLERS:
            sampler_cls = getattr(optuna.samplers, cls.BUILTIN_SAMPLERS[key], None)
            if sampler_cls is None:
                raise ImportError(f"当前Optuna版本不提供 {cls.BUILTIN_SAMPLERS[key]}，请升级Optuna或选择其它采样器")
            return cls._instantiate(sampler_cls, kwargs)

        if key in cls.OPTUNAHUB_SAMPLERS:
            try:
                import optunahub
            except ImportError:
                raise ImportError(
                    f"使用 '{sampler}' 采样器需要 optunahub，请使用 "
                    f"pip install optunahub 安装（或 pip install hscredit[tune]）"
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


# 旧模块路径继续导出，兼容历史导入和序列化记录。
from ._metrics import Metric, TuningObjective, _calc_ks, _calc_ks_with_diff  # noqa: E402,F401


def _safe_index(data: Any, indices: np.ndarray) -> Any:
    """按行位置切分 pandas、NumPy 或列表。"""
    return take_rows(data, indices)


class ModelTuner(ArtifactSerializableMixin):
    """模型超参数调优器 - 支持单/多目标优化.

    基于Optuna实现贝叶斯优化超参数搜索。
    支持单目标优化和多目标优化（帕累托最优）。

    **参数**

    :param model_class: 模型类（如 XGBoost）或未拟合的模型/Pipeline 实例。
    :param search_space: 参数搜索空间，默认None则使用预定义空间
    :param fixed_params: 固定参数，不参与搜索
    :param model_params: 来源模型实例的构造参数；搜索空间同名参数会覆盖它，
        ``fixed_params`` 的显式值具有最高优先级
    :param metric: 优化指标（决定评估计算逻辑），可选:
        - 字符串: 'auc', 'ks', 'ks_diff', 'accuracy', 'precision', 'recall', 'f1', 'logloss'
        - 列表: 多个指标，用于多目标优化，如 ['ks', 'ks_diff']
        - 函数: 自定义评估函数，接收(y_true, y_pred)返回float
        - 列表的函数: 多个自定义函数
    :param direction: 默认 None，按内置指标或 BaseMetric 自动确定方向。
        裸函数必须显式提供 'maximize'/'minimize'；多目标可逐项提供方向列表。
    :param metric_names: 指标显示名称列表（仅用于日志/报告/可视化的展示标签，
        不参与任何计算逻辑），默认 None 时从 metric 自动推断
        （内置字符串取其大写形式，自定义函数取其 ``__name__``）。
        与 metric 不重复：metric 决定"算什么"，metric_names 只决定"叫什么"
    :param cv: 交叉验证折数（默认5）、带 split 方法的分割器或 (训练位置, 验证位置) 序列。
    :param n_jobs: 当前 trial 中模型可使用的并行任务数，默认-1；
        trial 本身顺序执行，确保主动中断及时生效并让自适应采样器利用全部历史结果
    :param random_state: 随机种子，默认None
    :param verbose: 是否逐 Trial 输出得分、参数、当前最佳结果及最终摘要，默认False
    :param early_stopping_rounds: 早停轮数，默认20
    :param loss: HSCredit 提升树训练所用 BaseLoss；metric 控制折外评分，彼此独立。
    :param record_terminator_scores: 旧版 Optuna 终止分析兼容选项，默认None。
        Optuna 4.9及以后默认不调用弃用接口；True显式启用，False关闭专用CV记录。
        各折指标和常规搜索曲线不受影响。

    **属性**

    ``best_params_`` / ``best_score_`` 是最佳参数与主指标均值，
    ``best_scores_`` 保存所有目标，``best_trial_`` 保存选中的 Optuna Trial。
    ``study_`` 与 ``optimization_history_`` 可查看全过程；``best_model_`` 在第一次
    调用 ``get_best_model()`` 后生成。多目标默认按指标顺序优先选择帕累托解。

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

    预定义空间只是搜索起点，不保证适合每个数据集。实际空间在首次 fit 后保存在
    ``search_space`` 中；用户提供明确空间时优先使用该空间。部分范围会按样本数、
    特征数、坏样本比例调整；LightGBM 的叶子数受到 ``2**max_depth`` 上限约束。
    自定义训练损失不使用原生 ``scale_pos_weight``，自动空间会去掉该无效维度；
    类别权重应在损失参数或 ``sample_weight`` 中设置。
    常用评估组合为最大化 KS/AUC，同时最小化训练/验证 KS 差异。

    **参考样例**

    >>> from hscredit.core.models import XGBoost, ModelTuner
    >>> # 单目标：最大化 KS
    >>> tuner = ModelTuner(XGBoost, metric='ks', direction='maximize', cv=5)
    >>> tuner.fit(X_train, y_train, n_trials=50)   # 返回最佳参数 best_params_
    >>> best_model = tuner.get_best_model()  # 此时才使用完整训练集重训
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
        model_params: Optional[Dict[str, Any]] = None,
        metric: Union[str, Callable, List[Union[str, Callable]]] = "ks",
        direction: Optional[Union[str, List[Optional[str]]]] = None,
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
        study: Optional[Any] = None,
        pruner: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        fit_params_factory: Optional[Callable] = None,
        trial_objective: Optional[Callable] = None,
        artifact_dir: Optional[str] = None,
        store_models: bool = True,
        retention: Optional[str] = None,
        loss: Optional[Any] = None,
        record_terminator_scores: Optional[bool] = None,
    ):
        """初始化 ModelTuner.

        :param model_class: 模型类或模型/Pipeline 实例；实例参数作为搜索的默认配置。
        :param search_space: 参数字典、命名维度列表或函数 (trial) -> 模型参数字典；None 使用自动空间。
        :param fixed_params: 明确固定的模型参数，优先级高于搜索参数和实例配置。
        :param model_params: 模型默认构造参数；搜索结果会覆盖同名默认值。
        :param metric: 字符串、BaseMetric、BaseLoss、带方向的 Metric 或函数；列表表示多目标。
            None 时使用 loss.metric()，无 loss 时使用 KS。
        :param direction: None 按指标对象/内置名称推断；裸函数须声明优化方向。
        :param metric_names: 指标显示名称列表，需与指标数一致且名称不重复。
        :param target: fit(df) 时提取的标签列名；显式传 y 时也会删除该列避免泄漏。
        :param cv: 折数、分割器或 (训练索引, 验证索引) 序列，默认5。
        :param n_jobs: 每个模型的并行上限，默认 -1；Trial 按顺序运行。
        :param random_state: 交叉验证、采样器及未设置种子的模型共用的随机种子。
        :param verbose: 是否逐 Trial 打印得分及最终摘要，默认 False。
        :param early_stopping_rounds: 支持早停的模型使用此默认轮数；模型显式固定参数优先。
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
        :param study: 已有 Optuna Study；使用该 Study 自身的存储、采样器和剪枝器。
        :param pruner: 创建 Study 时传递的原生 Optuna 剪枝器。
        :param callbacks: 原生 Study.optimize 的试验完成回调。
        :param fit_params: 每次模型训练的原生参数，样本级权重等会随交叉验证切分。
        :param fit_params_factory: 函数 (trial, fold_index) -> dict；最终重训传入 (None, None)。
        :param trial_objective: 原生函数 (trial) -> 数值或多目标序列，完全接管试验内容。
        :param artifact_dir: 逐折保存试验制品的目录；None 时只保留内存记录。
        :param store_models: 是否保留各折模型，默认 True；False 仍保留预测和指标。
        :param retention: 结果保留策略：full、predictions、summary、best、disk。
            None 按 store_models 保持旧行为；disk 必须指定 artifact_dir，按需读取且不常驻折模型。
        :param loss: 用于模型训练的 BaseLoss 实例，与决定搜索优劣的 metric 分开。
            支持 HSCredit XGBoost、LightGBM、CatBoost；例如
            ``ModelTuner(XGBoost, loss=FocalLoss(), metric='auc')``。
            ``metric=None`` 时使用该损失的配套指标（越小越好）。
        :param record_terminator_scores: 是否额外记录旧 Optuna terminator 的专用CV数据。
            None（默认）仅在Optuna低于4.9且接口可用时记录；False关闭，True显式保留弃用兼容功能。
            常规各折指标、中间值、搜索历史始终保留。Optuna 4.9起该模块弃用，显式开启仍会收到其提示。

        **参考样例**

        >>> tuner = ModelTuner(XGBoost(n_estimators=100), search_space={'max_depth': [2, 3, 4]},
        ...                    loss=FocalLoss(), metric='auc', cv=3, random_state=42)
        >>> params = tuner.fit(X_train, y_train, n_trials=10)
        >>> best = tuner.get_best_model()
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请使用 pip install optuna 安装")

        instance_params = {}
        if record_terminator_scores is not None and not isinstance(record_terminator_scores, (bool, np.bool_)):
            raise ValueError("record_terminator_scores 必须是 True、False 或 None")
        self.record_terminator_scores = record_terminator_scores
        if not isinstance(model_class, type) and hasattr(model_class, "get_params"):
            instance_params = model_class.get_params(deep=False)
            model_class = type(model_class)
        self.model_class = model_class
        self.search_space_function = search_space if callable(search_space) else None
        self._space_adapter = SearchSpaceAdapter({} if callable(search_space) else search_space)
        self.search_space = self._space_adapter.space
        self.model_params = {**instance_params, **dict(model_params or {})}
        # params 字典中的旧值不能在 Trial 构造模型时重新覆盖搜索结果。
        native_params = self.model_params.pop("params", None)
        if isinstance(native_params, dict):
            self.model_params.update(native_params)
        self.model_params = self._canonical_params(self.model_params, prefer_alias=True)
        self.trial_objective = trial_objective
        self._explicit_fixed_params = dict(fixed_params or {})
        self.loss = loss
        self._configure_training_loss()
        self.fixed_params: Dict[str, Any] = {}
        self._refresh_fixed_params()
        self._validate_custom_loss_parameters()
        self.objective = objective
        self.objective_kwargs = objective_kwargs or {}
        self.eval_ratios = [0.01, 0.03, 0.05, 0.10] if eval_ratios is None else list(eval_ratios)
        for ratio in self.eval_ratios:
            if (
                isinstance(ratio, (bool, np.bool_))
                or not isinstance(ratio, (int, float, np.integer, np.floating))
                or not np.isfinite(float(ratio))
                or not 0 < float(ratio) <= 1
            ):
                raise ValueError("eval_ratios 中每个覆盖率必须是 (0, 1] 内的有限数值")
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
        self.cv = cv if isinstance(cv, (int, np.integer)) or hasattr(cv, "split") else tuple(cv)
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.pruner = pruner
        self.callbacks = list(callbacks or [])
        self.fit_params = dict(fit_params or {})
        self.fit_params_factory = fit_params_factory
        self.trial_objective = trial_objective
        self.artifact_dir = artifact_dir
        self.store_models = store_models
        self.retention = retention
        self.retention_ = resolve_retention(retention, store_models, artifact_dir)
        self.trial_results_ = {}
        self.best_model_ = None

        from ..losses.base import BaseLoss

        if isinstance(objective, BaseLoss):
            raise TypeError("objective 是旧版搜索指标参数；训练损失请使用 loss=，评估损失请使用 metric=loss.metric()")
        if metric is None:
            metric = "ks" if loss is None else loss.metric()
        # 若指定了 objective（TuningObjective 风格），将其转换为 metric callable
        if objective is not None:
            if isinstance(objective, str):
                objective_key = objective.lower()
                if objective_key in TuningObjective.BUILTIN_OBJECTIVES:
                    _obj_func = TuningObjective.get(objective_key, **self.objective_kwargs)
                    metric = _obj_func
                    if direction is None:
                        direction = "maximize"
                    metric_names = metric_names or [objective_key]
                else:
                    # 可能是旧式 metric 字符串，直接透传
                    metric = objective
            elif callable(objective):
                metric = objective

        # 处理metric和direction
        self._setup_metrics(metric, direction, metric_names)

        # 存储结果
        self.study_ = study
        self.best_params_ = None
        self.best_score_ = None
        self.best_scores_ = None  # 多目标时使用
        self.optimization_history_ = None
        self.pareto_front_ = None  # 多目标帕累托前沿

        # 存储数据信息用于自适应搜索空间
        self._n_samples = None
        self._n_features = None
        self._class_balance_ratio = None
        self._is_multi_objective = len(self.metrics) > 1

        for point in initial_points:
            self.enqueue_trial(point)

    def _configure_training_loss(self):
        """训练损失只通过明确支持该契约的模型传入，避免误当作搜索目标。"""
        from ..losses.base import BaseLoss

        if self.loss is None:
            return
        if not isinstance(self.loss, BaseLoss):
            raise TypeError("loss 必须是 BaseLoss 实例；原生目标回调请通过模型 objective 配置")
        model_module = getattr(self.model_class, "__module__", "")
        model_name = getattr(self.model_class, "__name__", "")
        if not model_module.startswith("hscredit.") or model_name not in {"XGBoost", "LightGBM", "CatBoost"}:
            raise ValueError(
                "loss 便捷参数仅支持 HSCredit XGBoost、LightGBM、CatBoost；Pipeline 请预配置最终模型的 objective"
            )
        if any(key in self._explicit_fixed_params for key in ("objective", "loss_function")):
            raise ValueError("loss 与 fixed_params 中的 objective/loss_function 不能同时设置")
        self._explicit_fixed_params["objective"] = self.loss

    @staticmethod
    def _validate_loss_sample_arrays(value):
        """拒绝未经过逐折对齐的金额等数组，防止长度恰好相同时发生静默错配。"""
        from ..losses.base import BaseLoss, LossMetric

        if isinstance(value, Metric):
            value = value.metric
        if isinstance(value, LossMetric):
            value = value.loss
        if isinstance(value, BaseLoss):
            for name in ("amounts_", "lgd", "ead", "rate", "cost"):
                item = getattr(value, name, None)
                if item is not None and np.asarray(item).ndim > 0:
                    raise ValueError(
                        "交叉验证不能直接使用绑定金额或金融参数数组的 loss/metric；"
                        "金额加权请使用 fit(sample_weight=金额, evaluation_weight=金额)，"
                        "复杂逐折损失请使用 trial_objective 自行切分和评估"
                    )
        elif isinstance(value, dict):
            for item in value.values():
                ModelTuner._validate_loss_sample_arrays(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                ModelTuner._validate_loss_sample_arrays(item)
        elif hasattr(value, "get_params") and not isinstance(value, type):
            ModelTuner._validate_loss_sample_arrays(value.get_params(deep=False))

    @staticmethod
    def _configuration_fingerprint(configuration):
        """按参数语义散列，避免 Notebook 函数经 cloudpickle 恢复后字节表示变化。"""
        import functools
        import types
        from joblib import hash as stable_hash

        active = set()

        def normalize(value):
            if value is None or isinstance(value, (str, bytes, bool, int, float, np.ndarray, np.generic)):
                return value
            if isinstance(value, type):
                return ("类型", value.__module__, value.__qualname__)
            if isinstance(value, types.ModuleType):
                return ("模块", value.__name__)
            if id(value) in active:
                return ("循环引用", type(value).__module__, type(value).__qualname__)
            active.add(id(value))
            try:
                if isinstance(value, dict):
                    return {key: normalize(item) for key, item in value.items()}
                if isinstance(value, (tuple, list)):
                    return tuple(normalize(item) for item in value)
                if isinstance(value, functools.partial):
                    return ("偏函数", normalize(value.func), normalize(value.args), normalize(value.keywords))
                if isinstance(value, types.CodeType):
                    return (
                        value.co_code,
                        normalize(value.co_consts),
                        value.co_names,
                        value.co_varnames,
                        value.co_argcount,
                        value.co_kwonlyargcount,
                    )
                if isinstance(value, types.FunctionType):
                    closure = tuple(cell.cell_contents for cell in (value.__closure__ or ()))
                    globals_used = {
                        name: value.__globals__[name] for name in value.__code__.co_names if name in value.__globals__
                    }
                    return (
                        "函数",
                        normalize(value.__code__),
                        normalize(value.__defaults__),
                        normalize(value.__kwdefaults__),
                        normalize(closure),
                        normalize(globals_used),
                    )
                if isinstance(value, types.MethodType):
                    return ("方法", normalize(value.__func__), normalize(value.__self__))
                if hasattr(value, "__dict__"):
                    return (type(value).__module__, type(value).__qualname__, normalize(vars(value)))
                return value
            finally:
                active.remove(id(value))

        return stable_hash(normalize(configuration))

    def _uses_custom_training_loss(self, params=None):
        """识别训练配置中的自定义损失，不把调参 objective 指标混入判断。"""
        from ..losses.base import BaseLoss

        params = {**self.model_params, **self._explicit_fixed_params} if params is None else params
        objective = params.get("objective", params.get("loss_function"))
        return isinstance(objective, BaseLoss) or callable(objective)

    def _validate_custom_loss_parameters(self):
        """自定义导数不消费内置正类权重，拒绝搜索没有效果的维度。"""
        if not self._uses_custom_training_loss():
            return
        keys = {"scale_pos_weight", "is_unbalance", "unbalance"}
        if keys.intersection(self.search_space or {}):
            raise ValueError(
                "自定义 loss 不使用原生 scale_pos_weight/is_unbalance，不能搜索这些参数；请通过损失参数或 sample_weight 设置类别权重"
            )
        params = {**self.model_params, **self._explicit_fixed_params}
        if (
            params.get("scale_pos_weight") not in (None, 1, "auto")
            or params.get("is_unbalance")
            or params.get("unbalance")
        ):
            raise ValueError(
                "自定义 loss 不使用原生 scale_pos_weight/is_unbalance；请通过损失参数或 sample_weight 设置类别权重"
            )

    def _refresh_fixed_params(self) -> None:
        """按实例默认值 < 搜索参数 < 显式固定参数合并模型配置。"""
        search_names = set(self.search_space or {})
        for canonical, aliases in getattr(self.model_class, "_parameter_aliases", {}).items():
            if search_names.intersection((canonical, *aliases)):
                search_names.update((canonical, *aliases))
        if self.search_space_function is not None or self.trial_objective is not None:
            search_names.update(self.model_params)
        self.fixed_params = {name: value for name, value in self.model_params.items() if name not in search_names}
        self.fixed_params.update(self._explicit_fixed_params)
        self._validate_lightgbm_leaf_point(self.fixed_params)

    def _setup_metrics(
        self,
        metric: Union[str, Callable, List[Union[str, Callable]]],
        direction: Optional[Union[str, List[Optional[str]]]],
        metric_names: Optional[List[str]],
    ):
        """设置评估指标."""
        # 统一转换为列表
        if not isinstance(metric, (list, tuple)):
            metrics_list = [metric]
        else:
            metrics_list = metric

        # 处理direction
        if not metrics_list:
            raise ValueError("metric 不能为空列表")
        if not isinstance(direction, (list, tuple)):
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
        from ..losses.base import BaseLoss

        for m, d, name in zip(metrics_list, directions_list, metric_names):
            if d not in (None, "maximize", "minimize"):
                raise ValueError("direction 只能是 'maximize' 或 'minimize'")
            if isinstance(m, BaseLoss):
                m = m.metric()
            if isinstance(m, Metric):
                if m.direction not in ("maximize", "minimize"):
                    raise ValueError("Metric.direction 只能是 'maximize' 或 'minimize'")
                if d is not None and d != m.direction:
                    raise ValueError(f"direction 与指标 {m.name} 自身声明的方向不一致")
                wrapped = copy.copy(m)
                wrapped.name = name or m.name
                self.metrics.append(wrapped)
            elif isinstance(m, BaseMetric):
                # 与 Metric 对象一致，指标对象自身声明评估方向，避免损失被默认最大化。
                if d is not None and d != m.direction:
                    raise ValueError(f"direction 与指标 {m.name} 自身声明的方向不一致")
                self.metrics.append(Metric(m, name=name))
            else:
                self.metrics.append(Metric(m, name=name, direction=d))

        # 方便访问
        self.metric = self.metrics[0] if len(self.metrics) == 1 else self.metrics
        self.directions = [m.direction for m in self.metrics]
        self.direction = self.directions[0] if len(self.directions) == 1 else self.directions
        self.metric_names = [m.name for m in self.metrics]
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("评估指标名称不能重复；请通过 metric_names 设置不同名称")

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
        X, y = extract_target(X, y, self.target)
        validate_labels(y)
        return X, y

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
        catch: Tuple[Type[Exception], ...] = (Exception,),
        gc_after_trial: bool = False,
        fit_params: Optional[Dict[str, Any]] = None,
        evaluation_weight: Optional[np.ndarray] = None,
        **optimize_kwargs,
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
        :param sample_weight: 仅用于训练的样本权重，不自动改变验证目标。
        :param evaluation_weight: 显式评估权重，按每折训练/验证位置切分；默认None保持未加权指标。
        :param groups: 传给自定义交叉验证分割器的分组标签。
        :param callbacks: 本次运行额外追加的 Study.optimize 回调。
        :param catch: 允许 Optuna 标记失败后继续执行的异常类型；默认保留历史行为。
        :param gc_after_trial: 是否在每次试验后执行垃圾回收。
        :param fit_params: 本次训练参数，覆盖构造时的同名 fit_params。
        :param optimize_kwargs: 透传 Optuna Study.optimize 的其他参数；n_jobs 由调参器管理。
        :return: 最佳参数字典（保持旧接口；并非 self）。最佳模型需调用 get_best_model()。

        相同数据再次 fit 会追加 Trial，并复用原交叉验证划分。数据、权重、固定参数
        或指标变化时应新建调参器和 study_name，避免混用不可比较的历史得分。

        **参考样例**

        >>> tuner = ModelTuner(XGBoost, metric=['auc', 'ks_diff'], random_state=42)
        >>> best_params = tuner.fit(X_train, y_train, n_trials=20)
        >>> best = tuner.get_best_model()
        >>> prediction = best.predict_proba(X_test)[:, 1]
        """
        # 检查并处理输入
        X, y = self._check_input(X, y)
        sample_weight = validate_sample_weight(sample_weight, len(y))
        evaluation_weight = validate_sample_weight(evaluation_weight, len(y))
        runtime_fit_params = {**self.fit_params, **dict(fit_params or {})}
        if self.trial_objective is None:
            self._validate_custom_loss_parameters()
            for value in (self.model_params, self.fixed_params, self.metrics, runtime_fit_params):
                self._validate_loss_sample_arrays(value)
            self._validate_native_loss({**self.model_params, **self.fixed_params})
        from joblib import hash as stable_hash

        data_signature = stable_hash((X, y, sample_weight, evaluation_weight, groups))
        configuration = (
            self.model_params,
            self._explicit_fixed_params,
            self.early_stopping_rounds,
            self.random_state,
            [(item.metric, item.name, item.direction) for item in self.metrics],
            runtime_fit_params,
        )
        configuration_signature = self._configuration_fingerprint(configuration)
        # 相同输入续跑时复用原始折，random_state=None 也不会改变验证口径。
        old_signature = getattr(self, "_data_signature_", None)
        if old_signature is not None and old_signature != data_signature and self.study_ is not None:
            raise ValueError("续跑数据或权重发生变化，不能混用历史 Trial 得分；请新建 ModelTuner 和 study_name")
        if (
            getattr(self, "_configuration_signature_", configuration_signature) != configuration_signature
            and self.study_ is not None
        ):
            raise ValueError("续跑固定参数、训练参数或指标发生变化；请新建 ModelTuner 和 study_name")
        cv_signature = stable_hash(self.cv)
        if getattr(self, "_cv_signature_", cv_signature) != cv_signature and self.study_ is not None:
            raise ValueError("续跑交叉验证配置发生变化；请新建 ModelTuner 和 study_name")
        existing_splits = getattr(self, "_cv_splits", None)
        cv_splits = (
            existing_splits
            if old_signature == data_signature and existing_splits is not None
            else self._validated_splits(X, y, groups)
        )
        self._validate_fold_weights(evaluation_weight, cv_splits)
        contract = {
            "数据": data_signature,
            "配置": configuration_signature,
            "交叉验证": cv_signature,
            "划分": stable_hash(cv_splits),
            "指标名称": self.metric_names,
            "指标方向": self.directions,
            "模型": f"{self.model_class.__module__}.{self.model_class.__qualname__}",
        }
        if self.study_ is not None:
            previous_contract = self.study_.user_attrs.get("HSCredit调参契约")
            if previous_contract is not None and previous_contract != contract:
                raise ValueError("Study 的数据、模型或评估口径与本次不同；请新建 ModelTuner 和 study_name")
        self.retention_ = resolve_retention(getattr(self, "retention", None), self.store_models, self.artifact_dir)

        # 记录数据信息
        self._n_samples = X.shape[0]
        self._n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
        y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        positive_count = int(np.sum(y_array == 1))
        negative_count = int(np.sum(y_array == 0))
        self._class_balance_ratio = (
            float(negative_count / positive_count) if positive_count > 0 and negative_count > 0 else 1.0
        )
        self._X = X
        self._y = y
        self._sample_weight = sample_weight
        self._evaluation_weight = evaluation_weight
        if self._evaluation_weight is not None:
            for metric in self.metrics:
                metric.validate_evaluation_weight()
        self._groups = groups
        self._training_data_released_ = False
        self._fit_params = runtime_fit_params
        self.best_model_ = None
        validate_sample_weight(sample_weight, len(y))

        self._cv_splits = cv_splits
        self._data_signature_ = data_signature
        self._cv_signature_ = cv_signature
        self._configuration_signature_ = configuration_signature

        # 如果没有指定搜索空间，使用自适应搜索空间
        if self.search_space is None:
            self.search_space = self._get_adaptive_search_space()
            if self._uses_custom_training_loss():
                for name in ("scale_pos_weight", "is_unbalance", "unbalance"):
                    self.search_space.pop(name, None)
            self._space_adapter = SearchSpaceAdapter(self.search_space)
            self.search_space = self._space_adapter.space
            self._refresh_fixed_params()

        if self.study_ is None:
            # 创建采样器（支持 optuna 内置及 optunahub 采样器，见 TuningSampler 码表）
            sampler = TuningSampler.create(self.sampler, seed=self.random_state, **self.sampler_kwargs)

            # 公共 study 参数（storage 指定后可用 optuna-dashboard 实时查看进度）
            common_kwargs = dict(
                sampler=sampler,
                pruner=self.pruner,
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

        if [direction.name.lower() for direction in self.study_.directions] != self.directions:
            raise ValueError("传入 Study 的优化方向与 ModelTuner 的 direction 不一致")

        previous_contract = self.study_.user_attrs.get("HSCredit调参契约")
        if previous_contract is not None and previous_contract != contract:
            raise ValueError("Study 的数据、模型或评估口径与本次不同；请新建 ModelTuner 和 study_name")
        self.study_.set_user_attr("HSCredit调参契约", contract)

        if hasattr(self.study_, "set_metric_names") and getattr(self.study_, "metric_names", None) != self.metric_names:
            # 包装器明确支持这个公开实验接口；只接管它自身的稳定性提示，
            # 保留中文曲线名称，不屏蔽弃用提示、用户警告或其它实验功能提示。
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"^optuna\.study\.study\.Study\.set_metric_names is experimental .*",
                    category=optuna.exceptions.ExperimentalWarning,
                )
                self.study_.set_metric_names(self.metric_names)

        # 入队预指定的超参数搜索点（优先评估）
        self._enqueue_trial_points()

        # Optuna 的并行 trial 使用线程池。Jupyter 主线程收到中断后，
        # 线程池会等待正在训练的 trial 完成，导致主动中断不能立即返回。
        # trial 顺序执行还可确保自适应采样器利用此前全部完成结果；
        # 调参总预算全部交给当前模型的原生并行参数。
        model_workers = max(1, int(self.n_jobs or 1))

        def objective(trial):
            self.trial_results_[trial.number] = {"试验编号": trial.number, "各折": [], "保留策略": self.retention_}
            try:
                if self.trial_objective is not None:
                    return self.trial_objective(trial)
                params = self._build_model_params(self._sample_params(trial), workers=model_workers)
                self.trial_results_[trial.number]["模型参数"] = params
                model = self._new_model(params)
                metric_values, diagnostics = self._evaluate_model(
                    model,
                    X,
                    y,
                    sample_weight,
                    return_diagnostics=True,
                    trial=trial,
                    evaluation_weight=self._evaluation_weight,
                )
                for name, value in diagnostics.items():
                    trial.set_user_attr(name, value)
                return metric_values
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                trial.set_user_attr("错误类型", type(exc).__name__)
                trial.set_user_attr("错误信息", str(exc))
                raise

        # 运行优化
        optimize_callbacks = [self._finish_trial] + self.callbacks + list(callbacks or [])
        if self.verbose:
            optimize_callbacks.append(self._print_trial_progress)
        try:
            self.study_.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress_bar,
                n_jobs=1,
                callbacks=optimize_callbacks,
                catch=catch,
                gc_after_trial=gc_after_trial,
                **optimize_kwargs,
            )
        finally:
            # 中断、回调错误和全失败同样保留已完成的搜索过程。
            # Optuna 在 KeyboardInterrupt 时可能不执行完成回调，仍需提交索引和状态。
            import sys

            interrupted = isinstance(sys.exc_info()[1], KeyboardInterrupt)
            for trial in self.study_.trials:
                record = self.trial_results_.get(trial.number)
                if record is not None and "状态" not in record and trial.state.name in {"COMPLETE", "FAIL", "PRUNED"}:
                    record.update(
                        状态=(
                            "中断"
                            if interrupted and trial.state.name == "FAIL"
                            else {"COMPLETE": "完成", "FAIL": "失败", "PRUNED": "剪枝"}[trial.state.name]
                        ),
                        得分=trial.values,
                        用户属性=dict(trial.user_attrs),
                    )
                    self._persist_trial(trial.number)
            self.optimization_history_ = self._build_public_history()
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in self.study_.trials):
                self._save_results()
            self._apply_retention()

        completed_trials = [
            trial
            for trial in self.study_.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
        ]
        if not completed_trials:
            failed_trials = [trial for trial in self.study_.trials if trial.state == optuna.trial.TrialState.FAIL]
            last_error = failed_trials[-1].user_attrs if failed_trials else {}
            detail = ""
            if last_error.get("错误信息"):
                detail = f"；最后错误: {last_error.get('错误类型', 'Exception')}: {last_error['错误信息']}"
            raise ValueError(f"所有Trial均失败，请检查模型参数、数据和训练异常{detail}")

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
            if (
                parameter_name == "n_jobs"
                and inspect.isclass(self.model_class)
                and issubclass(self.model_class, SklearnLogisticRegression)
                and needs_logistic_regression_parallel_compat(installed_version("sklearn", "scikit-learn"))
            ):
                # 不为无效参数注入预算；显式设置仍交给模型处理并保留其警告。
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

    def _inject_model_random_state(self, params: Dict[str, Any]) -> None:
        """在模型显式支持且调用者未固定时注入调参随机种子。"""
        if self.random_state is None or "random_state" in self._explicit_fixed_params:
            return
        try:
            accepted = inspect.signature(self.model_class.__init__).parameters
        except (TypeError, ValueError, AttributeError):
            return
        if "random_state" in accepted and params.get("random_state") is None:
            params["random_state"] = self.random_state

    def _build_model_params(self, params: Dict[str, Any], workers: Optional[int] = None) -> Dict[str, Any]:
        """构建 Trial 评估和最终重训共用的完整模型参数。"""
        full_params = dict(self.model_params)
        full_params.update(self._canonical_params(params))
        full_params.update(self._canonical_params(self.fixed_params))
        if self._uses_custom_training_loss(full_params):
            if full_params.get("scale_pos_weight") not in (None, 1, "auto") or full_params.get("is_unbalance"):
                raise ValueError("自定义 loss 不使用原生类别权重参数；请改用损失参数或 sample_weight")
            if "scale_pos_weight" in full_params:
                full_params["scale_pos_weight"] = 1
        full_params = self._apply_model_param_constraints(full_params)
        self._inject_fit_params(full_params)
        self._inject_model_random_state(full_params)
        self._inject_model_parallel_budget(full_params, workers or max(1, int(self.n_jobs or 1)))
        self._validate_loss_sample_arrays(full_params)
        self._validate_native_loss(full_params)
        return full_params

    def _validate_native_loss(self, params):
        """原生框架不能直接解释 BaseLoss 概率导数协议，须先显式使用适配器。"""
        from ..losses.base import BaseLoss

        if self.model_class.__module__.split(".")[0] in {"xgboost", "lightgbm", "catboost"}:
            if any(isinstance(params.get(name), BaseLoss) for name in ("objective", "loss_function")):
                raise ValueError("原生框架不能直接传 BaseLoss；请使用 HSCredit 模型和 loss=，或显式调用对应框架适配器")

    def _canonical_params(self, params, prefer_alias=False):
        """搜索前统一同义参数，避免旧别名重新覆盖本次采样值。"""
        result = dict(params)
        for canonical, aliases in getattr(self.model_class, "_parameter_aliases", {}).items():
            for alias in aliases:
                if alias in result:
                    value = result.pop(alias)
                    if prefer_alias or canonical not in result:
                        result[canonical] = value
        return result

    def _new_model(self, params):
        """按 sklearn 约定分别传递构造参数和 Pipeline 嵌套参数。"""
        direct = {name: value for name, value in params.items() if "__" not in name}
        nested = {name: value for name, value in params.items() if "__" in name}
        model = self.model_class(**copy.deepcopy(direct))
        if nested:
            model.set_params(**nested)
        return model

    def _splitter(self, X, y, groups=None):
        if isinstance(self.cv, (int, np.integer)):
            splitter = StratifiedKFold(n_splits=int(self.cv), shuffle=True, random_state=self.random_state)
        else:
            splitter = self.cv
        if hasattr(splitter, "split"):
            return splitter.split(X, y, groups)
        return iter(splitter)

    def _validated_splits(self, X, y, groups=None):
        """所有调参与重评估入口共用位置索引校验。"""
        if isinstance(self.cv, (bool, np.bool_)):
            raise ValueError("cv 必须是大于等于2的折数、交叉验证分割器或索引对序列")
        if isinstance(self.cv, (int, np.integer)) and self.cv < 2:
            raise ValueError("交叉验证折数 cv 必须大于等于2")
        splits = list(self._splitter(X, y, groups))
        if not splits:
            raise ValueError("交叉验证没有产生任何数据划分")
        normalized = []
        for train, valid in splits:
            pair = []
            for indices in (train, valid):
                indices = np.asarray(indices)
                if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
                    raise ValueError("交叉验证索引必须是非空的一维整数数组")
                if np.any(indices < 0) or np.any(indices >= len(y)):
                    raise ValueError("交叉验证索引超出训练数据范围")
                if len(np.unique(indices)) != len(indices):
                    raise ValueError("同一交叉验证折的索引不能重复")
                pair.append(indices)
            if np.intersect1d(*pair).size:
                raise ValueError("交叉验证的训练集和验证集不能包含相同样本")
            normalized.append(tuple(pair))
        return normalized

    @staticmethod
    def _final_estimator(model):
        """取得 Pipeline 最后一个估计器，保留其训练过程与早停配置。"""
        from sklearn.pipeline import Pipeline

        while isinstance(model, Pipeline):
            model = model.steps[-1][1]
        return model

    def _fold_fit_params(self, indices, trial=None, fold=None, n_samples=None, fit_params=None):
        # 先切分再复制，避免每折复制一份完整训练样本参数。
        size = self._n_samples if n_samples is None else n_samples
        source = getattr(self, "_fit_params", self.fit_params) if fit_params is None else fit_params
        params = copy.deepcopy(split_sample_params(source, indices, size))
        if self.fit_params_factory is not None:
            extra = self.fit_params_factory(trial, fold)
            if not isinstance(extra, dict):
                raise TypeError("fit_params_factory 必须返回训练参数字典")
            params.update(extra)
        return params

    @staticmethod
    def _route_sample_weight(model, params, sample_weight):
        """把显式训练权重传给最终估计器，兼容嵌套 Pipeline 的标准参数前缀。"""
        from sklearn.pipeline import Pipeline

        if sample_weight is None:
            return params
        prefix = ""
        while isinstance(model, Pipeline):
            name, model = model.steps[-1]
            prefix += name + "__"
        params[prefix + "sample_weight"] = sample_weight
        return params

    def _fit_fold(self, model, X, y, params):
        """原生提升器的早停使用训练折内部划分，外层验证折只用于评分。"""
        module = type(model).__module__.split(".")[0]
        if module not in {"xgboost", "lightgbm", "catboost", "ngboost"}:
            return model.fit(X, y, **params)
        configured = model.get_params(deep=False)
        rounds = params.get("early_stopping_rounds", configured.get("early_stopping_rounds"))
        if module not in {"xgboost", "lightgbm", "catboost", "ngboost"} or rounds is None:
            return model.fit(X, y, **params)
        if "eval_set" in params or "X_val" in params:
            return model.fit(X, y, **params)
        from sklearn.model_selection import train_test_split

        train_idx, valid_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=self.random_state
        )
        runtime = dict(params)
        validation_params = {}
        weight_names = {
            "xgboost": "sample_weight_eval_set",
            "lightgbm": "eval_sample_weight",
            "ngboost": "val_sample_weight",
        }
        for key in ("sample_weight", "base_margin", "init_score", "baseline"):
            if key in runtime:
                values = runtime[key]
                runtime[key] = _safe_index(values, train_idx)
                validation_params[key] = _safe_index(values, valid_idx)
        X_val, y_val = _safe_index(X, valid_idx), _safe_index(y, valid_idx)
        if module == "ngboost":
            runtime.update(X_val=X_val, Y_val=y_val)
        elif module == "catboost":
            from catboost import Pool

            pool_params = {
                "weight" if key == "sample_weight" else key: value for key, value in validation_params.items()
            }
            categorical = runtime.get("cat_features", configured.get("cat_features"))
            if categorical is not None:
                pool_params["cat_features"] = categorical
            runtime["eval_set"] = Pool(X_val, y_val, **pool_params)
        else:
            runtime["eval_set"] = [(X_val, y_val)]
        if "sample_weight" in validation_params and module in weight_names:
            value = validation_params["sample_weight"]
            runtime[weight_names[module]] = value if module == "ngboost" else [value]
        if "base_margin" in validation_params and module == "xgboost":
            runtime["base_margin_eval_set"] = [validation_params["base_margin"]]
        if "init_score" in validation_params and module == "lightgbm":
            runtime["eval_init_score"] = [validation_params["init_score"]]
        return model.fit(_safe_index(X, train_idx), _safe_index(y, train_idx), **runtime)

    def _validate_fold_weights(self, evaluation_weight, splits):
        """评估折不能只有零权样本；在拟合任何折模型之前拒绝。"""
        if evaluation_weight is None:
            return
        needs_training = any(metric._is_builtin and metric.metric.lower() == "ks_diff" for metric in self.metrics)
        for fold_index, (train_indices, validation_indices) in enumerate(splits):
            if np.sum(_safe_index(evaluation_weight, validation_indices)) <= 0:
                raise ValueError(f"第 {fold_index} 折验证集 evaluation_weight 总和必须大于0")
            if needs_training and np.sum(_safe_index(evaluation_weight, train_indices)) <= 0:
                raise ValueError(f"第 {fold_index} 折训练集评估权重总和为0，无法计算 ks_diff")

    def _evaluate_model(
        self,
        model,
        X,
        y,
        sample_weight=None,
        return_diagnostics=False,
        trial=None,
        evaluation_weight=None,
        cv_splits=None,
        fit_params=None,
    ):
        """交叉验证并保留逐折模型、预测、指标和学习曲线。"""
        evaluation_weight = validate_sample_weight(evaluation_weight, len(y))
        if evaluation_weight is not None:
            for metric in self.metrics:
                metric.validate_evaluation_weight()
        fold_results = {i: [] for i in range(len(self.metrics))}
        fold_lifts = {float(ratio): [] for ratio in self.eval_ratios}
        splits = cv_splits if cv_splits is not None else getattr(self, "_cv_splits", None)
        if splits is None:
            splits = self._validated_splits(X, y, groups=getattr(self, "_groups", None))
        if not splits:
            raise ValueError("交叉验证没有产生任何数据划分")
        self._validate_fold_weights(evaluation_weight, splits)
        for fold_index, (train_idx, val_idx) in enumerate(splits):
            X_train_fold, X_val_fold = _safe_index(X, train_idx), _safe_index(X, val_idx)
            y_train_fold, y_val_fold = _safe_index(y, train_idx), _safe_index(y, val_idx)
            sample_weight_fold = _safe_index(sample_weight, train_idx)
            validation_weight = _safe_index(evaluation_weight, val_idx)
            train_evaluation_weight = _safe_index(evaluation_weight, train_idx)
            try:
                fold_model = clone(model)
            except Exception:
                # 兼容既有的非 sklearn 自定义模型；模板尚未训练，每折仍使用独立对象。
                fold_model = copy.deepcopy(model)
            fit_kwargs = {}
            record = {"折编号": fold_index, "训练位置": np.asarray(train_idx), "验证位置": np.asarray(val_idx)}
            if trial is not None:
                self.trial_results_[trial.number]["各折"].append(record)
            try:
                fit_kwargs = self._fold_fit_params(
                    train_idx, trial, fold_index, n_samples=len(y), fit_params=fit_params
                )
                self._route_sample_weight(fold_model, fit_kwargs, sample_weight_fold)
                self._fit_fold(fold_model, X_train_fold, y_train_fold, fit_kwargs)
                y_train_pred = positive_probability(fold_model.predict_proba(X_train_fold), fold_model.classes_, 1)
                y_val_pred = positive_probability(fold_model.predict_proba(X_val_fold), fold_model.classes_, 1)
                y_val_arr, y_train_arr = np.asarray(y_val_fold), np.asarray(y_train_fold)
                record.update(
                    真实标签=y_val_arr, 预测概率=y_val_pred, 训练真实标签=y_train_arr, 训练预测概率=y_train_pred
                )
                scores = {}
                for i, metric in enumerate(self.metrics):
                    value = metric(
                        y_val_arr,
                        y_val_pred,
                        y_train=y_train_arr,
                        y_train_pred=y_train_pred,
                        sample_weight=validation_weight,
                        train_sample_weight=train_evaluation_weight,
                    )
                    if not np.isfinite(value):
                        raise ValueError(f"第 {fold_index} 折的 {metric.name} 指标不是有限数")
                    fold_results[i].append(value)
                    scores[metric.name] = float(value)
                record["指标"] = scores
                record["评估口径"] = "未加权" if evaluation_weight is None else "显式评估权重"
                record["补充LIFT口径"] = "按样本行，未应用评估权重"
                for ratio in fold_lifts:
                    fold_lifts[ratio].append(TuningObjective.lift_head(y_val_arr, y_val_pred, ratio=ratio))
                record["状态"] = "完成"
            except BaseException as exc:
                record.update(
                    状态="中断" if isinstance(exc, KeyboardInterrupt) else "失败",
                    错误类型=type(exc).__name__,
                    错误信息=str(exc),
                )
                raise
            finally:
                trained = self._final_estimator(fold_model)
                record["训练记录"] = getattr(trained, "training_summary_", {})
                record["评估曲线"] = getattr(trained, "evals_result_", {})
                record["最佳迭代"] = getattr(trained, "best_iteration_", getattr(trained, "best_iteration", None))
                trained_params = trained.get_params(deep=False) if hasattr(trained, "get_params") else {}
                record["启用早停"] = bool(
                    fit_kwargs.get("early_stopping_rounds", trained_params.get("early_stopping_rounds"))
                )
                if hasattr(trained, "get_best_iteration"):
                    record["最佳迭代"] = trained.get_best_iteration()
                if self.retention_ in {"full", "best", "disk"}:
                    record["模型"] = fold_model
                if trial is not None:
                    retain_fold(self, trial.number, record)
                if trial is not None and self.artifact_dir:
                    self._persist_trial(trial.number)
            if trial is not None:
                trial.set_user_attr(
                    "各折指标", [item.get("指标", {}) for item in self.trial_results_[trial.number]["各折"]]
                )
                if not self._is_multi_objective:
                    trial.report(float(np.mean(fold_results[0])), step=fold_index)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"第 {fold_index} 折后停止本次试验")
        results = [float(np.mean(fold_results[i])) for i in range(len(self.metrics))]
        if trial is not None and not self._is_multi_objective and len(fold_results[0]) > 1:
            self._report_terminator_scores(trial, fold_results[0])
        metric_result = tuple(results) if self._is_multi_objective else results[0]
        if not return_diagnostics:
            return metric_result
        diagnostics = {self._lift_metric_name(ratio): float(np.mean(values)) for ratio, values in fold_lifts.items()}
        return metric_result, diagnostics

    def _report_terminator_scores(self, trial, scores):
        """只为未弃用的旧版本或显式兼容请求调用终止分析接口。"""
        from packaging.version import Version

        enabled = getattr(self, "record_terminator_scores", None)
        if enabled is None:
            enabled = Version(optuna.__version__).release[:2] < (4, 9)
        if not enabled:
            return
        import importlib

        try:
            terminator = importlib.import_module("optuna.terminator")
        except ModuleNotFoundError as exc:
            if exc.name != "optuna.terminator":
                raise
            return
        report_cv = getattr(terminator, "report_cross_validation_scores", None)
        if report_cv is not None:
            report_cv(trial, [float(value) for value in scores])

    def _finish_trial(self, study, trial):
        record = self.trial_results_.setdefault(trial.number, {"试验编号": trial.number, "各折": []})
        states = {"COMPLETE": "完成", "FAIL": "失败", "PRUNED": "剪枝", "RUNNING": "运行中", "WAITING": "等待"}
        record.update(
            状态=states.get(trial.state.name, trial.state.name), 得分=trial.values, 用户属性=dict(trial.user_attrs)
        )
        self._apply_retention()
        self._persist_trial(trial.number)

    def _apply_retention(self):
        """best 档仅保留当前最佳试验的折模型，其他试验仍保留指标和诊断。"""
        if getattr(self, "retention_", "full") != "best" or self.study_ is None:
            return
        completed = [trial for trial in self.study_.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        best = (
            self._select_best_pareto_trial(self.study_.best_trials) if completed and self._is_multi_objective else None
        )
        if completed and not self._is_multi_objective:
            best = self.study_.best_trial
        for number, record in self.trial_results_.items():
            if best is not None and number == best.number:
                continue
            if record.get("保留策略") == "summary":
                continue
            if record.get("状态") in {"完成", "失败", "剪枝", "中断"}:
                record["各折"] = [compact_fold(fold) for fold in record["各折"]]
                record["保留策略"] = "summary"
                self._persist_trial(number)

    def _persist_trial(self, number):
        if self.artifact_dir is None:
            return
        from .._lifecycle import atomic_save_pickle

        path = trial_path(self, number)
        atomic_save_pickle(self.trial_results_[number], path, engine="cloudpickle")
        self.trial_results_[number]["制品路径"] = str(path)
        self.study_.set_user_attr(f"试验制品_{number}", str(path))

    def get_trial_result(self, number, *, load=True):
        """获取单次试验的逐折模型、预测、得分与训练记录。

        :param number: Optuna Trial 编号，通常使用 best_trial_.number。
        :param load: disk 策略是否读取完整折制品；False 返回轻量摘要。
        :return: 中文键字典。能否包含模型/预测取决于 retention。

        **参考样例**

        >>> result = tuner.get_trial_result(tuner.best_trial_.number)
        >>> result['各折'][0]['指标']
        """
        if number not in self.trial_results_:
            from ....utils import load_pickle

            path = self.study_.user_attrs.get(f"试验制品_{number}") if self.study_ is not None else None
            if self.artifact_dir is not None and self.study_ is not None:
                relocated = trial_path(self, number)
                if relocated.is_file():
                    path = relocated
            if not path:
                raise ValueError(f"试验 {number} 没有保存逐折结果；请配置 artifact_dir 或加载完整调参制品")
            self.trial_results_[number] = load_pickle(path, engine="cloudpickle")
        record = self.trial_results_[number]
        if load and any("制品路径" in fold for fold in record.get("各折", [])):
            return {**record, "各折": [load_fold(fold, self) for fold in record["各折"]]}
        return record

    def get_oof_predictions(self, trial_number=None):
        """返回折外预测；重复验证取平均，未参加验证的位置保留缺失值。

        :param trial_number: 试验编号；None 使用默认最佳试验。
        :return: 包含样本位置、真实标签、预测概率、验证次数的 DataFrame。
        :raises ValueError: 输入已释放或当前保留策略没有保存预测时抛出。

        **参考样例**

        >>> oof = tuner.get_oof_predictions()
        >>> oof.loc[oof['验证次数'] > 0, ['真实标签', '预测概率']]
        """
        if trial_number is None:
            if self.best_params_ is None:
                raise ValueError("请先完成至少一次成功试验")
            trial_number = self.best_trial_.number
        if getattr(self, "_training_data_released_", False):
            raise ValueError("训练数据已释放，无法组装完整标签的折外预测；请重新 fit")
        record = self.get_trial_result(trial_number, load=False)
        if not any("预测概率" in fold or "制品路径" in fold for fold in record.get("各折", [])):
            raise ValueError("当前保留策略未保留该试验的折外预测，请使用 full、predictions、disk 或最佳试验")
        sums, counts = np.zeros(self._n_samples), np.zeros(self._n_samples, dtype=int)
        for summary in record["各折"]:
            fold = load_fold(summary, self)
            if "预测概率" in fold:
                np.add.at(sums, fold["验证位置"], fold["预测概率"])
                np.add.at(counts, fold["验证位置"], 1)
        probability = np.divide(sums, counts, out=np.full(self._n_samples, np.nan), where=counts > 0)
        return pd.DataFrame(
            {
                "样本位置": np.arange(self._n_samples),
                "真实标签": np.asarray(self._y),
                "预测概率": probability,
                "验证次数": counts,
            }
        )

    def release_training_data(self):
        """显式释放续训输入和样本级 fit 参数；已缓存最佳模型仍可预测。

        不删除已保留的折结果或磁盘制品，不清理用户函数闭包。
        再次 fit 必须重新提供输入及所需的样本参数。

        :return: self。释放后不能重新训练或重建完整 OOF 表。

        **参考样例**

        >>> best_model = tuner.get_best_model()
        >>> tuner.release_training_data()
        >>> prediction = best_model.predict_proba(X_test)
        """
        from .._contracts import SAMPLE_PARAMETER_NAMES

        for name in ("_X", "_y", "_sample_weight", "_evaluation_weight", "_groups", "_cv_splits"):
            setattr(self, name, None)
        row_names = SAMPLE_PARAMETER_NAMES | {"eval_set", "X_val", "Y_val"}
        for name in ("fit_params", "_fit_params"):
            params = getattr(self, name, {})
            setattr(
                self, name, {key: value for key, value in params.items() if key.rsplit("__", 1)[-1] not in row_names}
            )
        self._training_data_released_ = True
        return self

    def save(self, path, **kwargs):
        """保存可继续调参的完整对象，包括 Study、输入、预测、模型和自定义函数。

        :param path: 本地制品路径，例如 'output/tuner.pkl'。
        :param kwargs: 传给保存函数的参数；默认 engine='cloudpickle'。
        :return: 保存后的文件路径。

        **参考样例**

        >>> path = tuner.save('output/tuner.pkl')
        >>> restored = ModelTuner.load(path)
        """
        kwargs.setdefault("engine", "cloudpickle")
        from .._lifecycle import atomic_save_pickle

        return atomic_save_pickle({**self.get_artifact_metadata(), "object": self}, path, **kwargs)

    @classmethod
    def load(cls, path, *, artifact_dir=None, **kwargs):
        """加载完整过程；移动折制品后可用 artifact_dir 指定新根目录。

        :param path: save 生成的本地制品路径。
        :param artifact_dir: 可选的折制品新目录；不修改 Trial 数据。
        :param kwargs: 传给制品加载器的参数。
        :return: 恢复后的 ModelTuner 实例。

        **参考样例**

        >>> restored = ModelTuner.load('output/tuner.pkl')
        >>> restored.fit(X_train, y_train, n_trials=10)
        """
        tuner = cls.load_artifact(path, **kwargs)
        if artifact_dir is not None:
            tuner.artifact_dir = str(artifact_dir)
        if not hasattr(tuner, "retention_"):
            tuner.retention_ = resolve_retention(
                getattr(tuner, "retention", None),
                getattr(tuner, "store_models", True),
                getattr(tuner, "artifact_dir", None),
            )
        return tuner

    @staticmethod
    def _lift_metric_name(ratio: float) -> str:
        """生成稳定的公开 LIFT 覆盖率名称。"""
        return f"LIFT@{float(ratio) * 100:g}%"

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

        self.best_trial_ = (
            self._select_best_pareto_trial(self.study_.best_trials)
            if self._is_multi_objective
            else self.study_.best_trial
        )
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
        evaluation_weight: Optional[np.ndarray] = None,
        groups: Optional[Any] = None,
        fit_params: Optional[Dict[str, Any]] = None,
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
        :param sample_weight: 仅训练权重。
        :param evaluation_weight: 显式验证指标权重，默认None，不继承之前fit的评估权重。
        :param groups: 本次交叉验证的分组标签，按当前数据重新划分。
        :param fit_params: 本次评估的训练参数。以构造时 fit_params 为基础，不继承旧 fit 的临时参数。
        :return: 包含评估结果的DataFrame
        """
        # 检查trial_points
        if trial_points is None:
            raise ValueError("trial_points不能为空，请提供要评估的超参数点列表")

        # 检查并处理输入
        X, y = self._check_input(X, y)
        sample_weight = validate_sample_weight(sample_weight, len(y))
        evaluation_weight = validate_sample_weight(evaluation_weight, len(y))
        cv_splits = self._validated_splits(X, y, groups)
        runtime = {**self.fit_params, **dict(fit_params or {})}
        for value in (self.model_params, self.fixed_params, self.metrics, runtime):
            self._validate_loss_sample_arrays(value)

        results = []

        for i, params in enumerate(trial_points):
            if self.verbose:
                logger.info(f"评估 trial point {i+1}/{len(trial_points)}: {params}")

            # 合并固定参数
            full_params = self._build_model_params(params)

            # 创建模型并评估
            model = self._new_model(full_params)
            metric_values = self._evaluate_model(
                model, X, y, sample_weight, evaluation_weight=evaluation_weight, cv_splits=cv_splits, fit_params=runtime
            )

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
        evaluation_weight: Optional[np.ndarray] = None,
        groups: Optional[Any] = None,
        fit_params: Optional[Dict[str, Any]] = None,
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
        :param evaluation_weight: 显式指标权重；仅X未传入时默认复用fit的评估权重。
        :param groups: 传入新X时使用的分组标签；不复用旧数据的CV位置。
        :param fit_params: 本次重评估训练参数；新数据仅继承构造参数，旧数据继承原 fit 参数。
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
        cv_splits = None
        if X is None:
            if getattr(self, "_X", None) is None:
                raise ValueError("未提供X且fit时未缓存训练数据，请显式传入X/y")
            X, y = self._X, self._y
            if sample_weight is None:
                sample_weight = getattr(self, "_sample_weight", None)
            if evaluation_weight is None:
                evaluation_weight = getattr(self, "_evaluation_weight", None)
            runtime = {**getattr(self, "_fit_params", self.fit_params), **dict(fit_params or {})}
        else:
            X, y = self._check_input(X, y)
            cv_splits = self._validated_splits(X, y, groups)
            runtime = {**self.fit_params, **dict(fit_params or {})}
        sample_weight = validate_sample_weight(sample_weight, len(y))
        for value in (self.model_params, self.fixed_params, self.metrics, runtime):
            self._validate_loss_sample_arrays(value)

        results = []

        for idx in indices:
            trial = all_trials[idx]
            params = self._get_params_from_trial(trial)

            if self.verbose:
                logger.info(f"评估 study trial #{idx} (state={trial.state.name}): {params}")

            # 合并固定参数并按模型签名注入早停参数
            full_params = self._build_model_params(params)

            # 创建模型并评估
            model = self._new_model(full_params)
            metric_values = self._evaluate_model(
                model, X, y, sample_weight, evaluation_weight=evaluation_weight, cv_splits=cv_splits, fit_params=runtime
            )

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
        if self.search_space_function is not None or self.trial_objective is not None:
            record = self.trial_results_.get(trial.number, {})
            if not record and trial.user_attrs.get("搜索参数需制品"):
                record = self.get_trial_result(trial.number)
            params = dict(record.get("搜索参数", trial.user_attrs.get("搜索参数", trial.params)))
        else:
            params = self._space_adapter.public_params(trial)
        return self._apply_model_param_constraints(params)

    def _build_public_history(self) -> pd.DataFrame:
        """生成只包含模型最终参数名和值的 Optuna 历史表。"""
        history = self.study_.trials_dataframe()
        latent_columns = [column for column in history if column.startswith("params___hscredit__")]
        history = history.drop(columns=latent_columns, errors="ignore")
        names = set(self.search_space or {})
        if self.search_space_function is not None or self.trial_objective is not None:
            names.update(name for trial in self.study_.trials for name in self._get_params_from_trial(trial))
        for name in sorted(names):
            column = f"params_{name}"
            values = []
            for trial in self.study_.trials:
                params = self._get_params_from_trial(trial)
                params.update(self.fixed_params)
                values.append(self._apply_model_param_constraints(params).get(name))
            history[column] = values
        for ratio in self.eval_ratios:
            public_name = self._lift_metric_name(float(ratio))
            internal_column = f"user_attrs_{public_name}"
            if internal_column in history:
                history[public_name] = history.pop(internal_column)
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
        elif model_name in ("svm", "svc"):
            return self._get_svm_search_space()
        elif model_name == "decisiontreeclassifier":
            return self._get_decisiontree_search_space()
        else:
            raise ValueError(f"无法为模型 {self.model_class.__name__} 自动生成搜索空间，请显式传入 search_space")

    def _get_class_weight_range(self) -> Tuple[float, float]:
        """根据训练标签负正样本比生成正样本权重范围。"""
        ratio = float(np.clip(self._class_balance_ratio or 1.0, 0.1, 100.0))
        return max(0.1, ratio * 0.5), min(100.0, max(ratio * 1.5, ratio * 0.5))

    def _get_xgboost_search_space(self) -> Dict[str, Dict[str, Any]]:
        """XGBoost搜索空间 - 基于内部建模经验.

        参考内部代码（强正则、浅树、小学习率以抑制风控样本过拟合）:
        - max_depth: 风控场景通常2-4，防止过拟合
        - min_child_weight: 8-256（step 4），叶子最小样本权重，越大越保守
        - subsample: 0.35-0.85，行采样
        - colsample_bytree: 0.4-0.9，列采样
        - gamma: 0.0-32.0，分裂最小损失下降，越大越保守
        - scale_pos_weight: 围绕训练集负正样本比的 0.5-1.5 倍自适应
        - reg_alpha: 0.0-1.0（L1 正则）
        - reg_lambda: 32.0-128.0（L2 正则，强约束）
        - learning_rate: 0.0001-0.01，较小学习率更稳定
        - n_estimators: 32-256（step 16）

        固定项 ``objective='binary:logistic'`` / ``eval_metric='auc'`` /
        ``booster='gbtree'`` / ``importance_type='cover'`` 已是模型默认值，
        如需覆盖可通过 ``ModelTuner(fixed_params=...)`` 传入。
        """
        class_weight_low, class_weight_high = self._get_class_weight_range()
        return {
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
            "n_estimators": {"type": "int", "low": 32, "high": 256, "step": 16},
            "min_child_weight": {"type": "int", "low": 8, "high": 256, "step": 4},
            "subsample": {"type": "float", "low": 0.35, "high": 0.85},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.9},
            "gamma": {"type": "float", "low": 0.0, "high": 32.0},
            "scale_pos_weight": {
                "type": "float",
                "low": class_weight_low,
                "high": class_weight_high,
            },
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 32.0, "high": 128.0},
        }

    def _get_lightgbm_search_space(self) -> Dict[str, Dict[str, Any]]:
        """LightGBM搜索空间 - 与XGBoost搜索空间对齐.

        参考风控小中型样本经验，并避免强正则导致所有样本预测相同:
        - num_leaves: 与max_depth相关，受 ``2**max_depth`` 上界约束（见 _sample_params）
        - max_depth: 风控场景通常2-4，防止过拟合
        - min_child_samples: 8-128（step 4），叶子最小样本数
        - subsample: 0.35-0.85，配合 subsample_freq=1 启用行采样
        - colsample_bytree: 0.4-0.9，列采样
        - min_split_gain: 0.0-1.0，保留小样本中的有效弱分裂
        - scale_pos_weight: 围绕训练集负正样本比的 0.5-1.5 倍自适应
        - reg_alpha: 0.0-1.0（L1 正则）
        - reg_lambda: 0.0-10.0（L2 正则）
        - learning_rate: 0.005-0.1，对数采样
        - n_estimators: 64-512（step 32）
        """
        class_weight_low, class_weight_high = self._get_class_weight_range()
        return {
            "num_leaves": {"type": "int", "low": 8, "high": 64},
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "n_estimators": {"type": "int", "low": 64, "high": 512, "step": 32},
            "min_child_samples": {"type": "int", "low": 8, "high": 128, "step": 4},
            "subsample": {"type": "float", "low": 0.35, "high": 0.85},
            "subsample_freq": {"type": "categorical", "choices": [1]},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 0.9},
            "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
            "scale_pos_weight": {
                "type": "float",
                "low": class_weight_low,
                "high": class_weight_high,
            },
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
        }

    def _get_logisticregression_search_space(self) -> Dict[str, Dict[str, Any]]:
        """逻辑回归搜索空间 - 基于内部建模经验.

        参考内部代码:
        - C: 正则强度倒数，对数区间 0.01-32（越小正则越强）
        - penalty: 仅 'l2'（评分卡常用，兼容多数 solver）
        - class_weight: None / 'balanced'（均可安全持久化到 Optuna storage）
        - max_iter: 16-256（对数区间），迭代上限
        - solver: liblinear / sag / lbfgs / newton-cg

        """
        return {
            "C": {"type": "float", "low": 0.01, "high": 32.0, "log": True},
            "penalty": {"type": "categorical", "choices": ["l2"]},
            "class_weight": {
                "type": "categorical",
                "choices": [None, "balanced"],
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
        - learning_rate: 0.005-0.05，避免小样本下概率过快饱和
        - base_max_depth: 基学习器（CART）最大深度，风控场景通常2-3
        - base_min_samples_leaf: 叶节点最小样本数，抑制单样本叶节点导致的奇异自然梯度
        - minibatch_frac: 小批量采样比例（行采样），至少保留70%样本
        - col_sample: 特征采样比例
        """
        n_samples = self._n_samples or 10000
        if n_samples > 10000:
            n_estimators_low, n_estimators_high = 200, 800
        else:
            n_estimators_low, n_estimators_high = 100, 500

        return {
            "n_estimators": {"type": "int", "low": n_estimators_low, "high": n_estimators_high},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.05, "log": True},
            "base_max_depth": {"type": "int", "low": 2, "high": 3},
            "base_min_samples_leaf": {"type": "int", "low": 5, "high": 20},
            "minibatch_frac": {"type": "float", "low": 0.7, "high": 1.0},
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
        trial_points: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
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

        :param params: 搜索点，可仅指定部分搜索参数，其他由采样器补全。
        :param user_attrs: 写入 Trial 的附加信息字典。
        :param skip_if_exists: 是否跳过已经存在的相同搜索点。
        :return: self；执行下一次 fit 时优先评估此点。

        **参考样例**

        >>> tuner.enqueue_trial({'max_depth': 3}, user_attrs={'来源': '人工经验'})
        >>> tuner.fit(X_train, y_train, n_trials=10)
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
        internal_point = (
            public_point
            if self.search_space_function is not None
            else self._space_adapter.to_internal_point(public_point)
        )
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
        :param user_attrs: 所有入队点共享的附加信息字典。
        :param skip_if_exists: 是否跳过已存在的相同搜索点。
        :return: self，便于链式调用

        **参考样例**

        >>> tuner.enqueue_trials(param_grid={'max_depth': [2, 3], 'learning_rate': [0.01, 0.05]})
        >>> tuner.fit(X_train, y_train, n_trials=4)
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

        :param params: 参数字典或按搜索空间声明顺序排列的值序列。
        :param lazy: 兼容参数；均在下次 fit 执行。
        :return: self。

        **参考样例**

        >>> tuner.probe({'max_depth': 3})
        >>> tuner.fit(X_train, y_train, n_trials=1)
        """
        del lazy
        point = dict(params) if isinstance(params, dict) else self._ordered_point(params, "probe")
        return self.enqueue_trial(point)

    def _enqueue_trial_points(self) -> None:
        """将 self.trial_points 入队到当前 study（fit 内部调用）."""
        for public_point, user_attrs, skip_if_exists in self._pending_public_trials:
            internal_point = (
                public_point
                if self.search_space_function is not None
                else self._space_adapter.to_internal_point(public_point)
            )
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
            if name in accepted and name not in self._explicit_fixed_params and params.get(name) is None:
                params[name] = value

    def _sample_params(self, trial: "Trial") -> Dict[str, Any]:
        """从搜索空间采样参数.

        :param trial: Optuna trial对象
        :return: 参数字典
        """
        if self.search_space_function is not None:
            params = self.search_space_function(trial)
            if not isinstance(params, dict):
                raise TypeError("search_space 函数必须返回模型参数字典")
            self.trial_results_.setdefault(trial.number, {})["搜索参数"] = params
            import json

            try:
                json.dumps(params, allow_nan=False)
            except (TypeError, ValueError):
                trial.set_user_attr("搜索参数需制品", True)
            else:
                trial.set_user_attr("搜索参数", params)
        else:
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

    def get_best_model(self, refit=False, full_data=True, **fit_params):
        """获取并缓存最佳模型；默认按各折最佳轮数使用全部输入重训。

        :param refit: True 时强制重新训练；默认复用已训练的 best_model_
        :param full_data: 是否关闭最终模型的内部验证划分，使用完整输入
        :param fit_params: 最终训练的额外原生参数
        :return: 已拟合模型。默认以各折早停最佳轮数中位数在完整输入重训。

        full_data=False 保留模型早停设置；更改 full_data 会自动重新拟合。
        已释放训练数据时仅允许复用此前缓存的相同模式模型。

        **参考样例**

        >>> best_model = tuner.get_best_model()
        >>> probability = best_model.predict_proba(X_test)[:, 1]
        >>> fresh_model = tuner.get_best_model(refit=True)
        """
        if self.best_params_ is None:
            raise ValueError("请先调用fit()进行调优")
        if (
            self.best_model_ is not None
            and not refit
            and not fit_params
            and getattr(self, "_best_model_full_data_", True) == full_data
        ):
            return self.best_model_
        if getattr(self, "_training_data_released_", False):
            raise ValueError("训练数据已释放，不能重训最佳模型；请重新 fit")
        params = self._build_model_params(self.best_params_)
        model = self._new_model(params)
        estimator = self._final_estimator(model)
        estimator_params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else dict(params)
        name = type(estimator).__name__.lower()
        boosting = any(
            key in name
            for key in (
                "xgboost",
                "xgbclassifier",
                "lightgbm",
                "lgbmclassifier",
                "catboost",
                "ngboost",
                "ngbclassifier",
            )
        )
        if full_data and boosting and hasattr(estimator, "set_params"):
            try:
                folds = self.get_trial_result(self.best_trial_.number, load=False).get("各折", [])
            except ValueError:
                folds = []
            iterations = [
                fold["最佳迭代"]
                for fold in folds
                if fold.get("启用早停") and fold.get("最佳迭代") is not None and fold["最佳迭代"] >= 0
            ]
            if iterations:
                offset = 0 if "lightgbm" in name or "lgbm" in name else 1
                estimator_params["iterations" if "catboost" in name else "n_estimators"] = max(
                    1, int(np.ceil(np.median(iterations))) + offset
                )
            if "early_stopping_rounds" in estimator_params:
                estimator_params["early_stopping_rounds"] = None
            if "validation_fraction" in estimator_params:
                estimator_params["validation_fraction"] = 0.0
            estimator.set_params(**estimator_params)
            params = model.get_params(deep=False)
        runtime = self._fold_fit_params(None)
        runtime.update(fit_params)
        self._route_sample_weight(model, runtime, self._sample_weight)
        if full_data and boosting:
            runtime.pop("early_stopping_rounds", None)
        self.refit_model_ = model
        self.refit_params_ = dict(params)
        self._fit_fold(model, self._X, self._y, runtime)
        self.best_model_ = model
        self._best_model_full_data_ = full_data
        if hasattr(model, "tuner"):
            model.tuner = self
        return model

    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史.

        :return: 优化历史DataFrame

        **参考样例**

        >>> history = tuner.get_optimization_history()
        >>> history.head()
        """
        if self.optimization_history_ is None:
            raise ValueError("请先调用fit()进行调优")

        return self.optimization_history_

    def get_pareto_front(self) -> Optional[List]:
        """获取帕累托前沿（多目标优化时）.

        :return: 帕累托前沿上的trial列表

        **参考样例**

        >>> pareto = tuner.get_pareto_front()
        >>> [(trial.number, trial.values) for trial in pareto]
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

        **参考样例**

        >>> importance = tuner.get_param_importance(target=0)
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
            public_importance = {self._space_adapter.to_public_name(name): value for name, value in importance.items()}
            return pd.Series(public_importance)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"无法计算参数重要性: {e}")
            return None

    # ==================== 可视化方法 ====================

    def get_study(self) -> Any:
        """获取完整的原生 Optuna Study，不过滤试验、指标、属性或中间值。

        两种训练入口使用同一契约：``model.tune(...)`` 之后可用
        ``model.tuner.get_study()``；直接使用 ModelTuner 时用 ``tuner.get_study()``。
        返回值与 ``study_`` 是同一对象，可交给任何 Optuna 原生分析和可视化函数。

        :return: 原生 optuna.study.Study 对象。

        **参考样例**

        >>> study = tuner.get_study()
        >>> study.trials_dataframe().head()
        """
        if self.study_ is None:
            raise ValueError("尚未创建超参数搜索 Study，请先调用 fit()、model.tune() 或传入已有 study")
        return self.study_

    @property
    def visualization(self):
        """当前 Optuna 版本的完整可视化入口，自动传入本次搜索的 Study。

        例如 ``tuner.visualization.plot_timeline()``、``plot_intermediate_values()``、
        ``plot_rank(params=[...])``；Matplotlib 后端使用
        ``tuner.visualization.matplotlib.plot_optimization_history()``。
        用 ``dir(tuner.visualization)`` 可查看当前版本的全部入口。

        此入口保留 Optuna 原生语义：多目标的 target 使用函数，超体积图需要
        reference_point；依赖、试验数量或指标条件不足时保留原生错误。
        既有 ``tuner.plot_*`` 便捷方法和整数 target 用法继续保留。

        :return: 自动绑定 Study 的 Optuna 可视化代理。

        **参考样例**

        >>> tuner.visualization.plot_timeline().show()
        """
        from .visualization import _OptunaVisualization

        return _OptunaVisualization(self)

    def _publicize_plot_figure(self, figure: Any) -> Any:
        """清理 Plotly 图对象中的 Optuna 内部潜变量名。"""
        if not hasattr(figure, "to_plotly_json"):
            return figure

        replacements = {self._space_adapter.latent_name(name): name for name in (self.search_space or {})}

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                for internal_name, public_name in replacements.items():
                    value = value.replace(internal_name, public_name)
                return value
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, tuple):
                return tuple(rewrite(item) for item in value)
            return value

        from plotly.graph_objects import Figure

        return Figure(rewrite(figure.to_plotly_json()))

    def _translate_plot_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """把可视化方法 kwargs 中的公开 params 转为 Optuna 内部名。"""
        translated = dict(kwargs)
        if self.search_space_function is not None or self.trial_objective is not None:
            return translated
        if translated.get("params") is not None:
            translated["params"] = [self._space_adapter.to_internal_name(name) for name in translated["params"]]
        return translated

    def plot_optimization_history(self, target: Optional[int] = None, **kwargs):
        """绘制优化历史.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象

        **参考样例**

        >>> tuner.plot_optimization_history(target=0).show()
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

        **参考样例**

        >>> tuner.plot_param_importances(target=0).show()
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        kwargs = self._translate_plot_params(kwargs)
        if self._is_multi_objective:
            figure = optuna.visualization.plot_param_importances(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )
        else:
            figure = optuna.visualization.plot_param_importances(self.study_, **kwargs)
        return self._publicize_plot_figure(figure)

    def plot_slice(self, target: Optional[int] = None, **kwargs):
        """绘制参数切片图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象

        **参考样例**

        >>> tuner.plot_slice(params=['max_depth'], target=0).show()
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        kwargs = self._translate_plot_params(kwargs)
        if self._is_multi_objective:
            figure = optuna.visualization.plot_slice(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )
        else:
            figure = optuna.visualization.plot_slice(self.study_, **kwargs)
        return self._publicize_plot_figure(figure)

    def plot_pareto_front(self, **kwargs):
        """绘制帕累托前沿（多目标优化时）.

        :param kwargs: 绘图参数
        :return: plotly图形对象

        **参考样例**

        >>> tuner.plot_pareto_front().show()
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

        **参考样例**

        >>> tuner.plot_contour(params=['max_depth', 'learning_rate']).show()
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        if params is None:
            params = list(self.search_space.keys())[:2]
            if self.search_space_function is not None or self.trial_objective is not None:
                params = sorted({name for trial in self.study_.trials for name in trial.params})[:2]
        internal_params = self._translate_plot_params({"params": params})["params"]

        target = self._resolve_multi_objective_target(target)
        if self._is_multi_objective:
            figure = optuna.visualization.plot_contour(
                self.study_,
                params=internal_params,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs,
            )
        else:
            figure = optuna.visualization.plot_contour(self.study_, params=internal_params, **kwargs)
        return self._publicize_plot_figure(figure)

    def plot_parallel_coordinate(self, target: Optional[int] = None, **kwargs):
        """绘制平行坐标图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象

        **参考样例**

        >>> tuner.plot_parallel_coordinate(params=['max_depth', 'learning_rate']).show()
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        target = self._resolve_multi_objective_target(target)
        kwargs = self._translate_plot_params(kwargs)
        if self._is_multi_objective:
            figure = optuna.visualization.plot_parallel_coordinate(
                self.study_, target=lambda t: t.values[target], target_name=self.metric_names[target], **kwargs
            )
        else:
            figure = optuna.visualization.plot_parallel_coordinate(self.study_, **kwargs)
        return self._publicize_plot_figure(figure)

    def plot_edf(self, target: Optional[int] = None, **kwargs):
        """绘制经验分布函数图.

        :param target: 多目标时指定要绘制的指标索引，默认第一个
        :param kwargs: 绘图参数
        :return: plotly图形对象

        **参考样例**

        >>> tuner.plot_edf(target=0).show()
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
        direction: Optional[Union[str, List[Optional[str]]]] = None,
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
        :param direction: 默认 None 按各指标推断；裸函数须明确方向，多目标可给列表。
        :param metric_names: 指标名称列表（多目标时用于显示）
        :param target: 目标列名，用于scorecardpipeline风格的fit，默认'target'
        :param cv: 折数、分割器或索引对序列，默认5折分层交叉验证。
        :param random_state: 随机种子
        :param verbose: 是否输出详细信息
        :param early_stopping_rounds: 早停轮数，默认20
        :param kwargs: 传给 ModelTuner 的其他参数，例如 search_space、fixed_params、
            loss、fit_params、retention、n_jobs。search_space 可覆盖自动空间。
        :return: ModelTuner实例

        **参考样例**

        >>> tuner = AutoTuner.create('lr', metric=['auc', 'ks_diff'], random_state=42,
        ...                          search_space={'C': [0.1, 1.0, 10.0]}, n_jobs=1)
        >>> best_params = tuner.fit(X_train, y_train, n_trials=10)
        >>> model = tuner.get_best_model()
        """
        import importlib

        model_map = {
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "lightgbm": "LightGBM",
            "lgb": "LightGBM",
            "catboost": "CatBoost",
            "cat": "CatBoost",
            "ngboost": "NGBoost",
            "ngb": "NGBoost",
            "randomforest": "RandomForest",
            "rf": "RandomForest",
            "extratrees": "ExtraTrees",
            "et": "ExtraTrees",
            "gradientboosting": "GradientBoosting",
            "gbdt": "GradientBoosting",
            "logisticregression": "LogisticRegression",
            "lr": "LogisticRegression",
            "svm": "SVM",
            "svc": "SVM",
            "decisiontree": "DecisionTreeClassifier",
            "dt": "DecisionTreeClassifier",
        }

        model_type = model_type.lower()
        if model_type not in model_map:
            raise ValueError(f"未知模型类型: {model_type}")

        model_class = getattr(importlib.import_module("hscredit.core.models"), model_map[model_type])

        return ModelTuner(
            model_class=model_class,
            search_space=kwargs.pop("search_space", None),
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
