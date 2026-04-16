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
>>> from hscredit.core.models import XGBoostRiskModel, ModelTuner
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
...     model_class=XGBoostRiskModel,
...     search_space=search_space,
...     metric='ks',
...     direction='maximize'
... )
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)
>>>
>>> # scorecardpipeline风格
>>> tuner = ModelTuner(
...     model_class=XGBoostRiskModel,
...     search_space=search_space,
...     metric='ks',
...     target='label'
... )
>>> best_params = tuner.fit(df, n_trials=100)
>>>
>>> # 多目标调优（KS + 稳定性）
>>> tuner = ModelTuner(
...     model_class=XGBoostRiskModel,
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
...     model_class=XGBoostRiskModel,
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

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import get_scorer, roc_auc_score, roc_curve

from sklearn.utils.validation import check_is_fitted

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


def _calc_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算KS值（内部辅助函数）.
    
    :param y_true: 真实标签
    :param y_pred: 预测概率
    :return: KS值
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return abs(tpr - fpr).max()


def _calc_ks_with_diff(y_train: np.ndarray, y_train_pred: np.ndarray,
                       y_val: np.ndarray, y_val_pred: np.ndarray) -> Tuple[float, float]:
    """计算KS及训练/验证差异.
    
    :return: (验证集KS, KS差异)
    """
    ks_train = _calc_ks(y_train, y_train_pred)
    ks_val = _calc_ks(y_val, y_val_pred)
    ks_diff = abs(ks_train - ks_val)
    return ks_val, ks_diff


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

    Example:
        >>> from hscredit.core.models import ModelTuner, XGBoostRiskModel
        >>> tuner = ModelTuner(
        ...     model_class=XGBoostRiskModel,
        ...     objective='lift_head',
        ...     objective_kwargs={'ratio': 0.05},
        ... )
        >>> tuner.fit(X_train, y_train, n_trials=50)
    """

    # 支持的字符串名称
    BUILTIN_OBJECTIVES = [
        'ks', 'auc', 'lift_head', 'lift_tail',
        'lift_head_monotonic', 'ks_with_lift_constraint', 'head_ks',
        'ks_lift_combined', 'tail_purity_ks',
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
        sorted_idx = np.argsort(y_prob)   # 升序，低概率在前
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
            violations = sum(
                1 for i in range(1, len(brs)) if brs[i] > brs[i - 1] + 1e-8
            )
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
            return 0.0   # 不满足约束，惩罚为0
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
            raise ValueError(
                f"未知目标函数 '{name}'，可选: {cls.BUILTIN_OBJECTIVES}"
            )
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
        'auc': {'scorer': 'roc_auc', 'direction': 'maximize'},
        'accuracy': {'scorer': 'accuracy', 'direction': 'maximize'},
        'precision': {'scorer': 'precision', 'direction': 'maximize'},
        'recall': {'scorer': 'recall', 'direction': 'maximize'},
        'f1': {'scorer': 'f1', 'direction': 'maximize'},
        'logloss': {'scorer': 'neg_log_loss', 'direction': 'maximize'},
        'ks': {'scorer': None, 'direction': 'maximize'},  # 使用自定义计算
        'ks_diff': {'scorer': None, 'direction': 'minimize'},  # KS差异，需要特殊处理
    }
    
    def __init__(
        self,
        metric: Union[str, Callable],
        name: Optional[str] = None,
        direction: Optional[str] = None
    ):
        self.metric = metric
        self._is_builtin = isinstance(metric, str)
        
        if self._is_builtin:
            metric_key = metric.lower()
            if metric_key not in self.BUILTIN_METRICS:
                raise ValueError(f"未知的内置指标: {metric}，可用指标: {list(self.BUILTIN_METRICS.keys())}")
            
            self.name = name or metric_key.upper()
            self.scorer = self.BUILTIN_METRICS[metric_key]['scorer']
            self.direction = direction or self.BUILTIN_METRICS[metric_key]['direction']
        else:
            # 自定义函数
            if not callable(metric):
                raise ValueError("自定义metric必须是可调用的函数")
            self.name = name or getattr(metric, '__name__', 'custom_metric')
            self.scorer = metric
            if direction is None:
                raise ValueError("使用自定义metric时必须指定direction")
            self.direction = direction
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_train_pred: Optional[np.ndarray] = None
    ) -> float:
        """计算指标值.
        
        :param y_true: 验证集真实标签
        :param y_pred: 验证集预测概率
        :param y_train: 训练集真实标签（用于ks_diff）
        :param y_train_pred: 训练集预测概率（用于ks_diff）
        :return: 指标值
        """
        if self._is_builtin and self.metric.lower() == 'ks':
            return _calc_ks(y_true, y_pred)
        elif self._is_builtin and self.metric.lower() == 'ks_diff':
            if y_train is None or y_train_pred is None:
                raise ValueError("计算ks_diff需要提供训练集预测结果")
            _, ks_diff = _calc_ks_with_diff(y_train, y_train_pred, y_true, y_pred)
            return ks_diff
        elif self._is_builtin:
            # 其他内置指标使用sklearn scorer
            if self.scorer is None:
                raise ValueError(f"指标 {self.metric} 没有对应的sklearn scorer")
            scorer = get_scorer(self.scorer)
            # sklearn scorer需要estimator，这里直接计算
            from sklearn.metrics import get_scorer_names
            if self.scorer in get_scorer_names():
                # 对于可以直接计算的指标
                if self.scorer == 'roc_auc':
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

    :param model_class: 模型类 (如XGBoostRiskModel)
    :param search_space: 参数搜索空间，默认None则使用预定义空间
    :param fixed_params: 固定参数，不参与搜索
    :param metric: 优化指标，可选:
        - 字符串: 'auc', 'ks', 'ks_diff', 'accuracy', 'precision', 'recall', 'f1', 'logloss'
        - 列表: 多个指标，用于多目标优化，如 ['ks', 'ks_diff']
        - 函数: 自定义评估函数，接收(y_true, y_pred)返回float
        - 列表的函数: 多个自定义函数
    :param direction: 优化方向，'maximize'或'minimize'，或列表（多目标时）
    :param metric_names: 指标名称列表（多目标时用于显示），默认自动推断
    :param cv: 交叉验证折数，默认5
    :param n_jobs: 并行任务数，默认-1
    :param random_state: 随机种子，默认None
    :param verbose: 是否输出详细信息，默认False
    :param early_stopping_rounds: 早停轮数，默认20
    :param min_resource: 多目标优化时的最小资源，默认'auto'

    **搜索空间定义**

    搜索空间是一个字典，每个参数定义如下:
    - 整数参数: {'type': 'int', 'low': 1, 'high': 10, 'step': 1}
    - 浮点参数: {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True}
    - 类别参数: {'type': 'categorical', 'choices': ['a', 'b', 'c']}

    **内部建模经验**

    1. XGBoost参数经验:
       - max_depth: 风控场景通常2-5，防止过拟合
       - min_child_weight: 样本>10000时10-2000，否则10-300
       - subsample/colsample_bytree: 0.6-0.9
       - reg_alpha/reg_lambda: 1e-8到100的对数尺度
       - learning_rate: 0.005-0.1，较小学习率更稳定

    2. LightGBM参数经验:
       - num_leaves: 与max_depth相关，通常20-64（2^5=32附近）
       - max_depth: 风控场景通常2-5，防止过拟合
       - learning_rate: 0.005-0.1，较小学习率更稳定
       - min_child_samples: 5-100

    3. 评估指标:
       - 主要用KS评估模型区分能力
       - 同时考虑训练/测试KS差异防止过拟合
    """

    def __init__(
        self,
        model_class: Type,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        metric: Union[str, Callable, List[Union[str, Callable]]] = 'ks',
        direction: Union[str, List[str]] = 'maximize',
        metric_names: Optional[List[str]] = None,
        objective: Union[str, Callable, None] = None,
        objective_kwargs: Optional[Dict[str, Any]] = None,
        eval_ratios: List[float] = None,
        target: str = 'target',
        cv: int = 5,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = False,
        early_stopping_rounds: int = 20
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
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna未安装，请使用 pip install optuna 安装"
            )

        self.model_class = model_class
        self.search_space = search_space
        self.fixed_params = fixed_params or {}
        self.objective = objective
        self.objective_kwargs = objective_kwargs or {}
        self.eval_ratios = eval_ratios or [0.01, 0.03, 0.05, 0.10]
        self.target = target
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

        # 若指定了 objective（TuningObjective 风格），将其转换为 metric callable
        if objective is not None:
            if isinstance(objective, str):
                if objective in TuningObjective.BUILTIN_OBJECTIVES:
                    _obj_func = TuningObjective.get(objective, **self.objective_kwargs)
                    metric = _obj_func
                    direction = 'maximize'
                    metric_names = metric_names or [objective]
                else:
                    # 可能是旧式 metric 字符串，直接透传
                    metric = objective
            elif callable(objective):
                metric = objective
                direction = 'maximize'

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

    def _setup_metrics(
        self,
        metric: Union[str, Callable, List[Union[str, Callable]]],
        direction: Union[str, List[str]],
        metric_names: Optional[List[str]]
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
        
        # 创建Metric对象列表
        self.metrics = []
        for m, d, name in zip(metrics_list, directions_list, metric_names):
            if isinstance(m, Metric):
                self.metrics.append(m)
            else:
                self.metrics.append(Metric(m, name=name, direction=d))
        
        # 方便访问
        self.metric = self.metrics[0] if len(self.metrics) == 1 else self.metrics
        self.direction = directions_list[0] if len(directions_list) == 1 else directions_list
        self.directions = [m.direction for m in self.metrics]
        self.metric_names = [m.name for m in self.metrics]

    def _check_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
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
        sample_weight: Optional[np.ndarray] = None
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
        self._n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        self._X = X
        self._y = y
        self._sample_weight = sample_weight
        
        # 如果没有指定搜索空间，使用自适应搜索空间
        if self.search_space is None:
            self.search_space = self._get_adaptive_search_space()
        
        # 创建Optuna study
        sampler = TPESampler(seed=self.random_state)
        
        if self._is_multi_objective:
            # 多目标优化
            self.study_ = optuna.create_study(
                directions=self.directions,
                sampler=sampler
            )
        else:
            # 单目标优化
            self.study_ = optuna.create_study(
                direction=self.directions[0],
                sampler=sampler
            )

        # 定义目标函数
        def objective(trial):
            # 从搜索空间采样参数
            params = self._sample_params(trial)
            params.update(self.fixed_params)
            
            # 添加早停参数
            params['early_stopping_rounds'] = self.early_stopping_rounds
            params['validation_fraction'] = 0.2
            
            # 创建模型
            model = self.model_class(**params)
            
            # 评估模型
            return self._evaluate_model(model, X, y, sample_weight)

        # 运行优化
        self.study_.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar and self.verbose
        )

        # 保存结果
        self._save_results()
            
        self.optimization_history_ = self.study_.trials_dataframe()

        if self.verbose:
            if self._is_multi_objective:
                print(f"\n找到 {len(self.study_.best_trials)} 个帕累托最优解")
                for i, trial in enumerate(self.study_.best_trials[:3]):  # 显示前3个
                    scores = ', '.join([f"{name}={val:.4f}" 
                                       for name, val in zip(self.metric_names, trial.values)])
                    print(f"  解 {i+1}: {scores}")
            else:
                print(f"\n最佳得分: {self.best_score_:.4f}")
            print(f"最佳参数: {self.best_params_}")

        return self.best_params_
    
    def _evaluate_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None
    ) -> Union[float, Tuple[float, ...]]:
        """评估模型，返回一个或多个指标值."""
        try:
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
            # 存储每折的结果
            fold_results = {i: [] for i in range(len(self.metrics))}
            
            for train_idx, val_idx in kf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                
                # 获取预测概率
                y_train_pred = model.predict_proba(X_train_fold)[:, 1]
                y_val_pred = model.predict_proba(X_val_fold)[:, 1]
                
                # 计算每个指标
                for i, metric in enumerate(self.metrics):
                    value = metric(
                        y_val_fold.values if hasattr(y_val_fold, 'values') else y_val_fold,
                        y_val_pred,
                        y_train=y_train_fold.values if hasattr(y_train_fold, 'values') else y_train_fold,
                        y_train_pred=y_train_pred
                    )
                    fold_results[i].append(value)
            
            # 返回各指标的平均值
            results = [np.mean(fold_results[i]) for i in range(len(self.metrics))]
            
            if self._is_multi_objective:
                return tuple(results)
            else:
                return results[0]
                
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Trial failed: {e}")
            # 返回极差的值
            bad_values = []
            for direction in self.directions:
                if direction == 'maximize':
                    bad_values.append(float('-inf'))
                else:
                    bad_values.append(float('inf'))
            
            if self._is_multi_objective:
                return tuple(bad_values)
            else:
                return bad_values[0]

    def _save_results(self):
        """保存优化结果."""
        if self._is_multi_objective:
            # 多目标优化
            self.pareto_front_ = self.study_.best_trials
            
            # 默认选择帕累托前沿的第一个解
            best_trial = self.study_.best_trials[0]
            self.best_params_ = self._get_params_from_trial(best_trial)
            self.best_scores_ = list(best_trial.values)
            self.best_score_ = self.best_scores_[0]  # 第一个指标作为主指标
        else:
            # 单目标优化
            self.best_params_ = self.study_.best_params.copy()
            self.best_score_ = self.study_.best_value
            self.best_scores_ = [self.best_score_]
        
        self.best_params_.update(self.fixed_params)

    def evaluate_trials(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        trial_points: Optional[List[Dict[str, Any]]] = None,
        sample_weight: Optional[np.ndarray] = None
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
                print(f"评估 trial point {i+1}/{len(trial_points)}: {params}")
            
            # 合并固定参数
            full_params = self.fixed_params.copy()
            full_params.update(params)
            full_params['early_stopping_rounds'] = self.early_stopping_rounds
            full_params['validation_fraction'] = 0.2
            
            # 创建模型并评估
            model = self.model_class(**full_params)
            metric_values = self._evaluate_model(model, X, y, sample_weight)
            
            if self._is_multi_objective:
                result = {
                    'trial_id': i,
                    **params,
                    **{name: val for name, val in zip(self.metric_names, metric_values)}
                }
            else:
                result = {
                    'trial_id': i,
                    **params,
                    self.metric_names[0]: metric_values
                }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _get_params_from_trial(self, trial) -> Dict[str, Any]:
        """从trial中获取参数."""
        params = {}
        for param_name in self.search_space.keys():
            params[param_name] = trial.params.get(param_name)
        return params
    
    def _get_adaptive_search_space(self) -> Dict[str, Dict[str, Any]]:
        """根据数据特征获取自适应搜索空间.
        
        基于内部建模经验，根据样本量和特征数调整搜索范围。
        """
        # 获取模型类型
        model_name = self.model_class.__name__.lower()
        
        if 'xgboost' in model_name or 'xgb' in model_name:
            return self._get_xgboost_search_space()
        elif 'lightgbm' in model_name or 'lgb' in model_name:
            return self._get_lightgbm_search_space()
        elif 'catboost' in model_name or 'cat' in model_name:
            return self._get_catboost_search_space()
        elif 'randomforest' in model_name or 'rf' in model_name:
            return self._get_randomforest_search_space()
        elif 'gradientboosting' in model_name or 'gbdt' in model_name:
            return self._get_gradientboosting_search_space()
        else:
            # 默认使用XGBoost搜索空间
            return self._get_xgboost_search_space()
    
    def _get_xgboost_search_space(self) -> Dict[str, Dict[str, Any]]:
        """XGBoost搜索空间 - 基于内部建模经验.
        
        参考内部代码:
        - max_depth: 风控场景通常2-5，防止过拟合
        - min_child_weight: 样本>10000时10-2000，否则10-300
        - subsample/colsample_bytree: 0.6-0.9
        - reg_alpha/reg_lambda: 1e-8到100
        - learning_rate: 0.005-0.1，较小学习率更稳定
        - n_estimators: 样本>10000时50-300，否则20-100
        """
        n_samples = self._n_samples or 10000
        
        # 根据样本量调整min_child_weight
        if n_samples > 10000:
            min_child_weight_high = 2000
        else:
            min_child_weight_high = 300
        
        # 根据样本量调整n_estimators
        if n_samples > 10000:
            n_estimators_high = 500
            n_estimators_low = 50
        else:
            n_estimators_high = 300
            n_estimators_low = 30
        
        return {
            'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True},
            'n_estimators': {'type': 'int', 'low': n_estimators_low, 'high': n_estimators_high},
            'min_child_weight': {'type': 'int', 'low': 10, 'high': min_child_weight_high},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'gamma': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
            'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 100.0, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 100.0, 'log': True},
        }
    
    def _get_lightgbm_search_space(self) -> Dict[str, Dict[str, Any]]:
        """LightGBM搜索空间 - 基于内部建模经验.
        
        参考内部代码:
        - num_leaves: 与max_depth相关，通常20-64（2^5=32附近）
        - max_depth: 风控场景通常2-5，防止过拟合
        - learning_rate: 0.005-0.1，较小学习率更稳定
        - n_estimators: 50-500
        - min_child_samples: 5-100
        - subsample/colsample_bytree: 0.6-1.0
        - reg_alpha/reg_lambda: 1e-8到10
        """
        return {
            'num_leaves': {'type': 'int', 'low': 20, 'high': 64},
            'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True},
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'min_split_gain': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
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
            'depth': {'type': 'int', 'low': 2, 'high': 5},
            'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True},
            'iterations': {'type': 'int', 'low': 50, 'high': 500},
            'l2_leaf_reg': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'border_count': {'type': 'int', 'low': 32, 'high': 255},
            'random_strength': {'type': 'float', 'low': 0.0, 'high': 10.0},
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
            'n_estimators': {'type': 'int', 'low': n_estimators_low, 'high': n_estimators_high},
            'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        }
    
    def _get_gradientboosting_search_space(self) -> Dict[str, Dict[str, Any]]:
        """GradientBoosting搜索空间 - 基于风控场景优化.
        
        参考内部代码:
        - max_depth: 风控场景通常2-5，防止过拟合
        - learning_rate: 0.005-0.1，较小学习率更稳定
        """
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'learning_rate': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True},
            'max_depth': {'type': 'int', 'low': 2, 'high': 5},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        }

    def _sample_params(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """从搜索空间采样参数.

        :param trial: Optuna trial对象
        :return: 参数字典
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            else:
                raise ValueError(f"未知参数类型: {param_type}")
            
            # LightGBM特殊处理：num_leaves不超过2^max_depth
            if param_name == 'max_depth' and 'num_leaves' in self.search_space:
                max_leaves = min(2 ** params['max_depth'], self.search_space['num_leaves']['high'])
                params['num_leaves'] = trial.suggest_int(
                    'num_leaves',
                    self.search_space['num_leaves']['low'],
                    max_leaves,
                    step=self.search_space['num_leaves'].get('step', 1)
                )

        return params

    def get_best_model(self) -> Any:
        """获取使用最佳参数的模型实例.

        :return: 训练好的模型实例
        """
        if self.best_params_ is None:
            raise ValueError("请先调用fit()进行调优")

        return self.model_class(**self.best_params_)

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

    def get_param_importance(self, target: Optional[int] = None) -> Optional[pd.Series]:
        """获取参数重要性.
        
        :param target: 多目标时指定要分析的指标索引，默认第一个
        :return: 参数重要性Series
        """
        if self.study_ is None:
            raise ValueError("请先调用fit()进行调优")

        try:
            if self._is_multi_objective and target is not None:
                # 多目标优化时，可以指定特定目标
                importance = optuna.importance.get_param_importances(
                    self.study_,
                    target=lambda t: t.values[target]
                )
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_optimization_history(
                self.study_, 
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_param_importances(
                self.study_,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_slice(
                self.study_,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        
        return optuna.visualization.plot_pareto_front(
            self.study_,
            target_names=self.metric_names,
            **kwargs
        )
    
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_contour(
                self.study_,
                params=params,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_parallel_coordinate(
                self.study_,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        
        if self._is_multi_objective and target is not None:
            return optuna.visualization.plot_edf(
                self.study_,
                target=lambda t: t.values[target],
                target_name=self.metric_names[target],
                **kwargs
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
        metric: Union[str, Callable, List[Union[str, Callable]]] = 'ks',
        direction: Union[str, List[str]] = 'maximize',
        metric_names: Optional[List[str]] = None,
        target: str = 'target',
        cv: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        early_stopping_rounds: int = 20,
        **kwargs
    ) -> ModelTuner:
        """创建自动调优器.

        :param model_type: 模型类型，可选:
            - 'xgboost'
            - 'lightgbm'
            - 'catboost'
            - 'randomforest'
            - 'gradientboosting'
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
            XGBoostRiskModel,
            LightGBMRiskModel,
            CatBoostRiskModel,
            RandomForestRiskModel,
            GradientBoostingRiskModel,
        )

        model_map = {
            'xgboost': XGBoostRiskModel,
            'xgb': XGBoostRiskModel,
            'lightgbm': LightGBMRiskModel,
            'lgb': LightGBMRiskModel,
            'catboost': CatBoostRiskModel,
            'cat': CatBoostRiskModel,
            'randomforest': RandomForestRiskModel,
            'rf': RandomForestRiskModel,
            'gradientboosting': GradientBoostingRiskModel,
            'gbdt': GradientBoostingRiskModel,
        }

        model_type = model_type.lower()
        if model_type not in model_map:
            raise ValueError(f"未知模型类型: {model_type}")

        model_class = model_map[model_type]

        return ModelTuner(
            model_class=model_class,
            search_space=None,  # 使用自适应搜索空间
            metric=metric,
            direction=direction,
            metric_names=metric_names,
            target=target,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )
