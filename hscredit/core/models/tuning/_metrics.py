"""搜索指标的数值、方向与业务口径；不依赖 Optuna。"""

import inspect
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_curve

from ...metrics import auc as auc_metric
from ..losses.base import BaseLoss, BaseMetric, LossMetric, _validate_binary_inputs
from ..losses.functional import make_metric
from ..losses.business_metrics import BadDebtMetric, TopKLiftMetric, _bounded_parameter, _selection_fraction


def _calc_ks(y_true, y_pred, sample_weight=None):
    y, p, weight = _validate_binary_inputs(y_true, y_pred, sample_weight)
    effective = np.ones(len(y)) if weight is None else weight
    if any(effective[y == label].sum() <= 0 for label in (0, 1)):
        raise ValueError("计算KS需要有效权重下同时存在好样本和坏样本")
    fpr, tpr, _ = roc_curve(y, p, pos_label=1, sample_weight=weight)
    return float(np.max(np.abs(tpr - fpr)))


def _calc_ks_with_diff(y_train, y_train_pred, y_val, y_val_pred, sample_weight=None, train_sample_weight=None):
    ks_train = _calc_ks(y_train, y_train_pred, train_sample_weight)
    ks_val = _calc_ks(y_val, y_val_pred, sample_weight)
    return ks_val, abs(ks_train - ks_val)


def _probability_inputs(y_true, y_prob, kwargs):
    unknown = set(kwargs) - {"sample_weight"}
    if unknown:
        raise TypeError(f"不支持的指标参数: {sorted(unknown)}")
    y, p, weight = _validate_binary_inputs(y_true, y_prob, kwargs.get("sample_weight"))
    return y, p, np.ones(len(y)) if weight is None else weight


class TuningObjective:
    """兼容既有字符串目标的业务评分集合，返回值均越大越好。

    **参数**

    本类无需实例化。各方法接收0/1标签和坏样本概率；``get`` 可绑定业务参数。

    **属性**

    ``BUILTIN_OBJECTIVES`` 列出名称。降低坏账率使用的评分函数与原始坏账率不同；
    要直接最小化坏账率，推荐 losses.BadDebtMetric。

    **参考样例**

    >>> metric = TuningObjective.get("lift_head", ratio=0.1)
    >>> metric.direction
    'maximize'
    >>> metric([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    2.0
    """

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
    def ks(y_true, y_prob, **kwargs):
        """计算 KS，越大越好。

        :param y_true: 0/1标签。
        :param y_prob: 一维坏样本概率。
        :param kwargs: 可传 sample_weight；两类的有效权重都须大于0。
        :return: [0, 1] 内的KS。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        return _calc_ks(y, p, weight)

    @staticmethod
    def auc(y_true, y_prob, score_direction="auto", **kwargs):
        """按统一评分方向计算 AUC，兼容既有字符串目标的自动方向。

        :param y_true: 0/1标签；必须同时包含有效好、坏样本。
        :param y_prob: [0, 1] 内的预测值。
        :param score_direction: auto（默认，兼容原有评分方向）、higher_risk（越大风险越高）
            或 higher_safe（越大风险越低）。固定坏样本概率评价可显式使用 higher_risk，
            或直接传入不会自动翻向的 AUCMetric 指标对象。
        :param kwargs: 可传 sample_weight。
        :return: [0, 1] 内的AUC；非法输入抛异常，不伪装成0分。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        return auc_metric(y, p, sample_weight=weight, score_direction=score_direction)

    @staticmethod
    def lift_head(y_true, y_prob, ratio=0.10, **kwargs):
        """计算高风险头部坏率 / 整体坏率，人数向上取整，同分边界等比例计入。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param ratio: 头部人数占比，(0, 1]，默认0.1。
        :param kwargs: 可传 sample_weight，只加权统计、不改变名单人数。
        :return: 头部提升倍数，越大越好；没有坏样本时返回0以兼容补充诊断。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        metric = TopKLiftMetric(top_ratio=ratio)
        return 0.0 if np.dot(weight, y) == 0 else metric(y, p, weight)

    @staticmethod
    def lift_tail(y_true, y_prob, ratio=0.10, **kwargs):
        """计算低风险尾部好率 / 整体好率，越大越好。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param ratio: 尾部人数占比，(0, 1]，默认0.1。
        :param kwargs: 可传 sample_weight。
        :return: 纯净度提升倍数；没有好样本时返回0。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        metric = TopKLiftMetric(top_ratio=ratio)
        return 0.0 if np.dot(weight, 1 - y) == 0 else metric(1 - y, 1 - p, weight)

    @staticmethod
    def lift_head_monotonic(y_true, y_prob, n_bins=10, penalty=0.5, **kwargs):
        """按概率等频分箱，以 KS × (1 - 违反坏率单调的比例 × penalty) 评分。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率；同分概率放在同一箱内。
        :param n_bins: 期望分箱数，至少2；实际箱数可因同分值减少。
        :param penalty: 违反单调性的惩罚系数，[0, 1]。
        :param kwargs: 本指标暂不支持 sample_weight。
        :return: 越大越好的惩罚后KS；使用统一分箱器和 compute_bin_stats。
        """
        y, p, _ = _probability_inputs(y_true, y_prob, kwargs)
        if kwargs.get("sample_weight") is not None:
            raise ValueError("分箱单调性指标暂不支持评估权重")
        if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
            raise ValueError("n_bins 必须是至少2的整数")
        _bounded_parameter(penalty, "单调性惩罚", 0, 1)
        ks = _calc_ks(y, p)
        if np.unique(p).size < 2:
            return ks
        import pandas as pd
        from ...binning import QuantileBinning
        from ...metrics import compute_bin_stats

        binner = QuantileBinning(max_n_bins=min(n_bins, len(y)), force_numerical=True, n_jobs=1)
        codes = binner.fit_transform(pd.DataFrame({"风险概率": p}), y)["风险概率"].to_numpy()
        table = compute_bin_stats(codes, y, round_digits=False).sort_values("分箱")
        bad_rates = table["坏样本率"].to_numpy()[::-1]
        violations = np.mean(np.diff(bad_rates) > 1e-8) if len(bad_rates) > 1 else 0.0
        return float(ks * (1 - violations * penalty))

    @staticmethod
    def ks_with_lift_constraint(y_true, y_prob, min_lift_ratio=0.05, min_lift_value=2.0, **kwargs):
        """满足头部提升约束时返回KS，否则返回0。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param min_lift_ratio: 头部人数占比，(0, 1]。
        :param min_lift_value: 最低提升倍数，非负有限数。
        :param kwargs: 可传 sample_weight。
        :return: 越大越好的KS评分；0仅表示不满足约束，不表示试验失败。
        """
        _bounded_parameter(min_lift_value, "最低提升倍数", 0, np.inf)
        head = TuningObjective.lift_head(y_true, y_prob, ratio=min_lift_ratio, **kwargs)
        return TuningObjective.ks(y_true, y_prob, **kwargs) if head >= min_lift_value else 0.0

    @staticmethod
    def head_ks(y_true, y_prob, ratio=0.30, **kwargs):
        """计算高风险头部内的KS，边界同分样本按比例加权。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param ratio: 头部人数占比，(0, 1]。
        :param kwargs: 可传 sample_weight。
        :return: [0, 1] 内的KS；头部只有单类时返回0。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        _bounded_parameter(ratio, "头部人数占比", 0, 1, include_lower=False)
        selected = weight * _selection_fraction(p, ratio, True)
        if any(selected[y == label].sum() <= 0 for label in (0, 1)):
            return 0.0
        return _calc_ks(y, p, selected)

    @staticmethod
    def ks_lift_combined(y_true, y_prob, ks_weight=0.5, lift_ratio=0.05, **kwargs):
        """计算 ks_weight × KS + (1-ks_weight) × min(头部LIFT/10, 1)。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param ks_weight: KS占比，[0, 1]，默认0.5。
        :param lift_ratio: 头部人数占比，(0, 1]。
        :param kwargs: 可传 sample_weight。
        :return: 越大越好的联合评分；LIFT除以10为沿用的业务缩放约定。
        """
        _bounded_parameter(ks_weight, "KS占比", 0, 1)
        ks = TuningObjective.ks(y_true, y_prob, **kwargs)
        lift = TuningObjective.lift_head(y_true, y_prob, ratio=lift_ratio, **kwargs)
        return float(ks_weight * ks + (1 - ks_weight) * min(lift / 10, 1))

    @staticmethod
    def tail_purity_ks(y_true, y_prob, tail_ratio=0.30, **kwargs):
        """计算 0.5 × KS + 0.5 × min(尾部好率提升, 1)，保留旧业务缩放约定。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率。
        :param tail_ratio: 低风险尾部人数占比，(0, 1]。
        :param kwargs: 可传 sample_weight。
        :return: 越大越好的联合评分。若需要完整尾部提升差异，请单用 lift_tail。
        """
        ks = TuningObjective.ks(y_true, y_prob, **kwargs)
        purity = TuningObjective.lift_tail(y_true, y_prob, ratio=tail_ratio, **kwargs)
        return float(0.5 * ks + 0.5 * min(purity, 1))

    @staticmethod
    def approval_bad_rate(y_true, y_prob, approval_rate=0.30, bad_rate_weight=1.0, **kwargs):
        """固定人数通过率下计算 approval_rate × (1 - 通过坏率 × bad_rate_weight)。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率；优先通过低风险客户。
        :param approval_rate: 通过人数占比，(0, 1]，默认0.3。
        :param bad_rate_weight: 坏率惩罚权重，非负有限数。
        :param kwargs: 可传 sample_weight。
        :return: 越大越好的审批评分；原始坏率应使用 BadDebtMetric 最小化。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        _bounded_parameter(bad_rate_weight, "坏率惩罚权重", 0, np.inf)
        bad_rate = BadDebtMetric(approval_rate)(y, p, weight)
        return float(approval_rate * (1 - bad_rate * bad_rate_weight))

    @staticmethod
    def expected_profit(y_true, y_prob, approval_rate=0.30, good_profit=1.0, bad_loss=5.0, **kwargs):
        """固定通过人数占比时的人均已实现利润，全体申请客户为分母。

        :param y_true: 0/1标签。
        :param y_prob: 坏样本概率；优先通过低风险客户，同分边界等比例计入。
        :param approval_rate: 通过人数占比，(0, 1]，默认0.3。
        :param good_profit: 通过好客户的收益，非负有限数。
        :param bad_loss: 通过坏客户的损失，非负有限数。
        :param kwargs: 可传 sample_weight。
        :return: 越大越好的利润；与使用固定概率阈值的 ProfitMetric 口径不同。
        """
        y, p, weight = _probability_inputs(y_true, y_prob, kwargs)
        _bounded_parameter(approval_rate, "通过人数占比", 0, 1, include_lower=False)
        _bounded_parameter(good_profit, "通过好客户收益", 0, np.inf)
        _bounded_parameter(bad_loss, "通过坏客户损失", 0, np.inf)
        selected = _selection_fraction(p, approval_rate, False)
        return float(np.average(selected * ((1 - y) * good_profit - y * bad_loss), weights=weight))

    @classmethod
    def get(cls, name, **kwargs):
        """按名称创建可在离线、训练和调参中复用的指标。

        :param name: BUILTIN_OBJECTIVES 中的名称，不区分大小写。
        :param kwargs: 绑定对应方法的业务参数；未知参数立即报错。
        :return: BaseMetric 对象，其 direction 为 maximize。

        >>> metric = TuningObjective.get("lift_head", ratio=0.05)
        >>> metric.name
        'LIFT_HEAD'
        """
        if not isinstance(name, str) or name.lower() not in cls.BUILTIN_OBJECTIVES:
            raise ValueError(f"未知目标函数 {name!r}，可选: {cls.BUILTIN_OBJECTIVES}")
        key = name.lower()
        function = getattr(cls, key)
        allowed = set(inspect.signature(function).parameters) - {"y_true", "y_prob", "kwargs"}
        if set(kwargs) - allowed:
            raise ValueError(f"指标 {key} 不支持参数: {sorted(set(kwargs)-allowed)}")
        metric = make_metric(function, name=key.upper(), greater_is_better=True, **kwargs)
        if key == "lift_head_monotonic":
            metric.supports_sample_weight = False
        return metric


class Metric:
    """统一管理搜索目标的名称、数值及优化方向。

    **参数**

    :param metric: 内置名称、BaseMetric、BaseLoss，或 ``(y_true, probability)->float`` 函数。
        BaseLoss 自动转换为配套指标。sklearn scorer 与框架回调需先还原为概率指标。
    :param name: 展示名称；多目标搜索中不可重名。
    :param direction: maximize 或 minimize；默认内置/指标对象自行声明。
        裸函数必须显式传方向，也可先用 make_metric 包装。

    **属性**

    name、direction、scorer；``log_loss`` 为正交叉熵，默认最小化。
    旧名 ``logloss`` 保留负交叉熵最大化语义，便于旧 Study 继续使用。
    字符串 ``auc`` 沿用 ``score_direction='auto'``；传入 ``AUCMetric()``
    或绑定 ``score_direction='higher_risk'`` 的目标时，保持显式风险方向。

    **参考样例**

    >>> Metric("log_loss").direction
    'minimize'
    >>> Metric("ks_diff").direction
    'minimize'
    """

    BUILTIN_METRICS = {
        "auc": {"direction": "maximize"},
        "ks": {"direction": "maximize"},
        "ks_diff": {"direction": "minimize"},
        "log_loss": {"direction": "minimize"},
        "logloss": {"direction": "maximize"},
        "neg_log_loss": {"direction": "maximize"},
        "brier": {"direction": "minimize"},
        **{key: {"direction": "maximize"} for key in ("accuracy", "precision", "recall", "f1")},
        **{key: {"direction": "maximize"} for key in TuningObjective.BUILTIN_OBJECTIVES},
    }

    def __init__(self, metric, name=None, direction=None):
        if direction not in (None, "maximize", "minimize"):
            raise ValueError("direction 必须为 'maximize' 或 'minimize'")
        if isinstance(metric, BaseLoss):
            metric = metric.metric()
        self.metric = metric
        self._is_builtin = isinstance(metric, str)
        if self._is_builtin:
            key = metric.lower()
            if key not in self.BUILTIN_METRICS:
                raise ValueError(f"未知的内置指标: {metric}，可用指标: {list(self.BUILTIN_METRICS)}")
            self.name = name or key.upper()
            self.direction = direction or self.BUILTIN_METRICS[key]["direction"]
            self.scorer = None
        elif isinstance(metric, BaseMetric):
            if direction is not None and direction != metric.direction:
                raise ValueError(f"direction 与指标 {metric.name} 自身声明的方向不一致")
            self.name = name or metric.name
            self.direction = direction or metric.direction
            self.scorer = metric.evaluate
        elif callable(metric):
            if hasattr(metric, "_score_func"):
                raise TypeError("metric接收(y_true, probability)函数；sklearn scorer只能用于cross_val_score")
            if direction is None:
                raise ValueError("自定义metric必须指定direction；也可用make_metric(..., greater_is_better=False/True)")
            self.name = name or getattr(metric, "__name__", "自定义指标")
            self.direction = direction
            self.scorer = metric
        else:
            raise TypeError("metric 必须是内置名称、BaseMetric、BaseLoss或可调用函数")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("指标名称必须是非空字符串")

    def validate_evaluation_weight(self):
        """在训练开始前拒绝不支持 sample_weight 的评估函数。

        :return: None；不支持时抛出中文 ValueError。
        """
        if self._is_builtin:
            accepted = self.metric.lower() != "lift_head_monotonic"
        elif isinstance(self.metric, LossMetric):
            accepted = type(self.metric.loss).loss_values is not BaseLoss.loss_values
        elif hasattr(self.metric, "supports_sample_weight"):
            accepted = self.metric.supports_sample_weight
        else:
            target = self.metric.__call__ if isinstance(self.metric, BaseMetric) else self.scorer
            unwrapped = target
            while isinstance(unwrapped, partial):
                unwrapped = unwrapped.func
            parameters = inspect.signature(target).parameters
            accepted = "sample_weight" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
            if unwrapped is TuningObjective.lift_head_monotonic:
                accepted = False
        if not accepted:
            raise ValueError(
                f"指标 {self.name} 暂不支持评估权重；请移除 evaluation_weight 或使用支持 sample_weight 的指标"
            )

    def __call__(self, y_true, y_pred, y_train=None, y_train_pred=None, sample_weight=None, train_sample_weight=None):
        """评价一个验证折，返回原始指标口径的有限标量。

        :param y_true: 验证标签，0为好样本、1为坏样本。
        :param y_pred: 验证集坏样本概率，不接受评分或类别预测。
        :param y_train: ks_diff 所需的训练标签。
        :param y_train_pred: ks_diff 所需的训练概率。
        :param sample_weight: 显式验证权重；默认不加权。
        :param train_sample_weight: ks_diff 所需训练评估权重。
        :return: float；优化方向只决定选优，不改变数值符号，旧 logloss 除外。

        >>> round(Metric("log_loss")([0, 1], [0.1, 0.9]), 3)
        0.105
        """
        y, p, weight = _validate_binary_inputs(y_true, y_pred, sample_weight)
        if weight is not None:
            self.validate_evaluation_weight()
        key = self.metric.lower() if self._is_builtin else None
        if key == "ks_diff":
            if y_train is None or y_train_pred is None:
                raise ValueError("计算ks_diff需要提供训练集预测结果")
            value = _calc_ks_with_diff(y_train, y_train_pred, y, p, weight, train_sample_weight)[1]
        elif key in TuningObjective.BUILTIN_OBJECTIVES:
            value = getattr(TuningObjective, key)(y, p, **({"sample_weight": weight} if weight is not None else {}))
        elif key in {"logloss", "log_loss", "neg_log_loss"}:
            value = float(log_loss(y, p, sample_weight=weight, labels=[0, 1]))
            if key != "log_loss":
                value = -value
        elif key == "brier":
            value = float(np.average((y - p) ** 2, weights=weight))
        elif self._is_builtin:
            functions = {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }
            kwargs = {} if key == "accuracy" else {"zero_division": 0}
            value = functions[key](y, p >= 0.5, sample_weight=weight, **kwargs)
        else:
            value = self.scorer(y, p, sample_weight=weight) if weight is not None else self.scorer(y, p)
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值，不能返回框架回调元组或数组")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"指标 {self.name} 必须返回单个有限数值") from exc
        if not np.isfinite(value):
            raise ValueError(f"指标 {self.name} 返回了非有限数值")
        return value

    def __repr__(self):
        return f"Metric(name={self.name!r}, direction={self.direction!r})"
