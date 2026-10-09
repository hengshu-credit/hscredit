"""与风控损失对应的真实业务评估指标。

统一约定 ``y=1`` 为坏样本、预测概率越高风险越高；低概率客户通过审批。
头部名单和固定通过率采用人数口径，边界同分样本等比例计入，避免结果受行顺序影响。
样本权重仅用于指标统计，不改变名单人数；通过率上限指标则按权重计算累计通过率。
"""

from numbers import Real
import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BaseLoss, BaseMetric, _validate_binary_inputs


def _bounded_parameter(value, name: str, lower: float, upper: float, include_lower: bool = True) -> float:
    """校验有限实数参数并返回浮点值。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name}必须是有限实数")
    valid_lower = value >= lower if include_lower else value > lower
    if not valid_lower or value > upper:
        bracket = "[" if include_lower else "("
        raise ValueError(f"{name}必须在{bracket}{lower}, {upper}]范围内")
    return float(value)


def _weighted_inputs(y_true, y_pred, sample_weight=None):
    """使用统一二分类输入契约，并补齐未提供的样本权重。"""
    y_true, y_pred, sample_weight = _validate_binary_inputs(y_true, y_pred, sample_weight=sample_weight)
    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=float)
    return y_true, y_pred, sample_weight


def _selection_fraction(probability: np.ndarray, ratio: float, higher: bool) -> np.ndarray:
    """按人数选取一端的样本，同分边界等比例占用剩余名额。"""
    count = min(len(probability), int(np.ceil(len(probability) * ratio)))
    rank_value = -probability if higher else probability
    threshold = np.partition(rank_value, count - 1)[count - 1]
    selected = rank_value < threshold
    tied = rank_value == threshold
    fractions = selected.astype(float)
    fractions[tied] = (count - np.count_nonzero(selected)) / np.count_nonzero(tied)
    return fractions


class AUCMetric(BaseMetric):
    """保持风险方向的 AUC，不会把反向模型自动翻转为高分。

    **参数**

    :param name: 指标名称，默认 ``"AUC曲线下面积"``。

    **属性**

    ``greater_is_better=True``；指标越大越好，有效权重下必须同时存在好、坏样本。

    **参考样例**

    >>> AUCMetric()([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    1.0
    """

    def __init__(self, name: str = "AUC曲线下面积"):
        super().__init__(name, greater_is_better=True)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算AUCMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import AUCMetric
        >>> metric = AUCMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        if np.sum(weight[y_true == 0]) <= 0 or np.sum(weight[y_true == 1]) <= 0:
            raise ValueError("计算AUC需要有效权重下同时存在好样本和坏样本")
        return float(roc_auc_score(y_true, y_pred, sample_weight=weight))


class TopKCaptureMetric(BaseMetric):
    """高风险头部名单捕获的坏样本数占全部坏样本数的比例。

    **参数**

    :param top_ratio: 头部人数占比，范围 ``(0, 1]``，默认 0.05；人数向上取整。
    :param name: 指标名称，默认 ``"头部坏样本捕获率"``。

    **属性**

    ``greater_is_better=True``；没有有效坏样本时无法定义捕获率，将抛出异常。
    同分边界按相同比例计入，权重用于加权坏样本数。

    **参考样例**

    >>> TopKCaptureMetric(top_ratio=0.5)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    1.0
    """

    def __init__(self, top_ratio: float = 0.05, name: str = "头部坏样本捕获率"):
        super().__init__(name, greater_is_better=True)
        self.top_ratio = _bounded_parameter(top_ratio, "头部人数占比", 0.0, 1.0, include_lower=False)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算TopKCaptureMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import TopKCaptureMetric
        >>> metric = TopKCaptureMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        bad_weight = weight * y_true
        total_bad = np.sum(bad_weight)
        if total_bad <= 0:
            raise ValueError("没有有效坏样本，无法计算头部坏样本捕获率")
        fraction = _selection_fraction(y_pred, self.top_ratio, higher=True)
        return float(np.sum(fraction * bad_weight) / total_bad)


class TopKLiftMetric(BaseMetric):
    """高风险头部名单坏样本率相对全量坏样本率的提升倍数。

    **参数**

    :param top_ratio: 头部人数占比，范围 ``(0, 1]``，默认 0.1；人数向上取整。
    :param name: 指标名称，默认 ``"头部提升倍数"``。

    **属性**

    ``greater_is_better=True``；名单和总体均使用加权坏样本率。
    名单权重为零或总体没有有效坏样本时抛出异常。

    **参考样例**

    >>> TopKLiftMetric(top_ratio=0.5)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    2.0
    """

    def __init__(self, top_ratio: float = 0.1, name: str = "头部提升倍数"):
        super().__init__(name, greater_is_better=True)
        self.top_ratio = _bounded_parameter(top_ratio, "头部人数占比", 0.0, 1.0, include_lower=False)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算TopKLiftMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import TopKLiftMetric
        >>> metric = TopKLiftMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        total_bad = np.sum(weight * y_true)
        if total_bad <= 0:
            raise ValueError("没有有效坏样本，无法计算头部提升倍数")
        selected_weight = weight * _selection_fraction(y_pred, self.top_ratio, higher=True)
        selected_total = np.sum(selected_weight)
        if selected_total <= 0:
            raise ValueError("头部名单的样本权重总和必须大于零")
        head_bad_rate = np.sum(selected_weight * y_true) / selected_total
        return float(head_bad_rate / (total_bad / np.sum(weight)))


class BadDebtMetric(BaseMetric):
    """固定人数通过率下，低风险通过客户的观察坏账率。

    **参数**

    :param approval_rate: 通过人数占比，范围 ``(0, 1]``，默认 0.3；人数向上取整。
    :param name: 指标名称，默认 ``"通过客户坏账率"``。

    **属性**

    ``greater_is_better=False``；越小越好。边界同分样本等比例通过，
    坏账率使用通过样本权重作为分母；通过样本权重为零时抛出异常。

    **参考样例**

    >>> BadDebtMetric(approval_rate=0.5)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    0.0
    """

    def __init__(self, approval_rate: float = 0.3, name: str = "通过客户坏账率"):
        super().__init__(name, greater_is_better=False)
        self.approval_rate = _bounded_parameter(approval_rate, "通过人数占比", 0.0, 1.0, include_lower=False)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算BadDebtMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import BadDebtMetric
        >>> metric = BadDebtMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        selected_weight = weight * _selection_fraction(y_pred, self.approval_rate, higher=False)
        selected_total = np.sum(selected_weight)
        if selected_total <= 0:
            raise ValueError("通过客户的样本权重总和必须大于零")
        return float(np.sum(selected_weight * y_true) / selected_total)


class ApprovalRateMetric(BaseMetric):
    """观察坏账率不超过上限时，所有风险阈值中的最大通过率。

    **参数**

    :param target_bad_debt_rate: 观察坏账率上限，范围 ``[0, 1]``，默认 0.05。
    :param name: 指标名称，默认 ``"坏账约束下最大通过率"``。

    **属性**

    ``greater_is_better=True``；相同预测概率的客户必须同时通过。
    样本权重同时用于累计通过率和累计坏账率。无可行阈值时返回 0。
    此值通过标签搜索评估集上的阈值，不能作为未见数据的坏账率保证。

    **参考样例**

    >>> ApprovalRateMetric(0.0)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    0.5
    """

    def __init__(self, target_bad_debt_rate: float = 0.05, name: str = "坏账约束下最大通过率"):
        super().__init__(name, greater_is_better=True)
        self.target_bad_debt_rate = _bounded_parameter(target_bad_debt_rate, "坏账率上限", 0.0, 1.0)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算ApprovalRateMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import ApprovalRateMetric
        >>> metric = ApprovalRateMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        order = np.argsort(y_pred, kind="stable")
        sorted_pred = y_pred[order]
        group_end = np.r_[sorted_pred[1:] != sorted_pred[:-1], True]
        cumulative_weight = np.cumsum(weight[order])[group_end]
        cumulative_bad = np.cumsum((weight * y_true)[order])[group_end]
        nonempty = cumulative_weight > 0
        feasible = nonempty & (cumulative_bad <= self.target_bad_debt_rate * cumulative_weight)
        if not np.any(feasible):
            return 0.0
        # 累计坏账率不一定单调：中途不可行的阈值之后仍可能再次满足上限。
        return float(np.max(cumulative_weight[feasible]) / np.sum(weight))


class ClassificationCostMetric(BaseMetric):
    """固定分类阈值下，全量样本的人均误判成本。

    **参数**

    :param fn_cost: 漏抓坏客户的单位成本，非负有限数，默认 1.0。
    :param fp_cost: 误拒好客户的单位成本，非负有限数，默认 1.0。
    :param threshold: 拒绝阈值，范围 ``[0, 1]``，默认 0.5；``p >= threshold`` 拒绝。
    :param name: 指标名称，默认 ``"人均误判成本"``。

    **属性**

    ``greater_is_better=False``；正确分类计零，错误分类按成本计入，并按样本权重平均。

    **参考样例**

    >>> ClassificationCostMetric(fn_cost=100, fp_cost=1)([0, 1], [0.8, 0.2])
    50.5
    """

    def __init__(self, fn_cost: float = 1.0, fp_cost: float = 1.0, threshold: float = 0.5, name: str = "人均误判成本"):
        super().__init__(name, greater_is_better=False)
        self.fn_cost = _bounded_parameter(fn_cost, "漏抓坏客户成本", 0.0, np.inf)
        self.fp_cost = _bounded_parameter(fp_cost, "误拒好客户成本", 0.0, np.inf)
        self.threshold = _bounded_parameter(threshold, "拒绝阈值", 0.0, 1.0)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算ClassificationCostMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import ClassificationCostMetric
        >>> metric = ClassificationCostMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        rejected = y_pred >= self.threshold
        cost = (y_true == 0) * rejected * self.fp_cost + (y_true == 1) * (~rejected) * self.fn_cost
        return float(np.average(cost, weights=weight))


class ProfitMetric(BaseMetric):
    """给定固定审批阈值时，全量申请客户的人均已实现利润。

    **参数**

    :param revenue: 好客户通过后的单位收益，非负有限数，默认 1.0。
    :param default_cost: 坏客户通过后的单位损失，非负有限数，默认 10.0。
    :param cutoff: 审批阈值，范围 ``[0, 1]``，默认 0.5；``p < cutoff`` 通过。
    :param name: 指标名称，默认 ``"人均审批利润"``。

    **属性**

    ``greater_is_better=True``；通过好客户计收益，通过坏客户计负损失，拒绝计零。
    分母为全量样本权重，因此各模型在同一评估集上可直接比较。

    **参考样例**

    >>> ProfitMetric(revenue=100, default_cost=1000)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    50.0
    """

    def __init__(
        self,
        revenue: float = 1.0,
        default_cost: float = 10.0,
        cutoff: float = 0.5,
        name: str = "人均审批利润",
    ):
        super().__init__(name, greater_is_better=True)
        self.revenue = _bounded_parameter(revenue, "好客户收益", 0.0, np.inf)
        self.default_cost = _bounded_parameter(default_cost, "坏客户损失", 0.0, np.inf)
        self.cutoff = _bounded_parameter(cutoff, "审批阈值", 0.0, 1.0)

    def __call__(self, y_true, y_pred, sample_weight=None) -> float:
        """计算ProfitMetric定义的实际业务指标。

        :param y_true: 一维 0/1 标签，1 为坏样本，不能为空。
        :param y_pred: 同形状坏样本概率，范围 [0, 1]；低概率表示低风险。
        :param sample_weight: 可选一维非负有限权重，须等长且总和大于 0。
            权重用于统计，名单人数和阈值规则详见本类说明。
        :return: float，单位和分母详见本类说明；优化方向见 direction 属性。

        **参考样例**

        >>> from hscredit.core.models.losses import ProfitMetric
        >>> metric = ProfitMetric()
        >>> value = metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        """
        y_true, y_pred, weight = _weighted_inputs(y_true, y_pred, sample_weight)
        profit = (1 - y_true) * self.revenue - y_true * self.default_cost
        return float(np.sum(weight * (y_pred < self.cutoff) * profit) / np.sum(weight))


def business_metric_for_loss(loss: BaseLoss) -> BaseMetric:
    """按损失类型构造业务指标；没有专用业务口径时使用该损失本身的评估指标。

    基于类型而非名称识别损失，用户修改 ``loss.name`` 不影响对应关系。
    指标复制构造时的业务参数，后续修改损失参数需要重新调用本函数。

    :param loss: BaseLoss 实例。
    :return: BaseMetric 实例。排序损失对应 AUC，KSFocusedLoss 对应 KS，
        成本、头部名单和审批损失对应其真实业务口径；其他损失返回 LossMetric。
    :raises TypeError: 参数不是 BaseLoss 实例。

    **参考样例**

    >>> from hscredit.core.models.losses import TopKBadCaptureLoss
    >>> metric = business_metric_for_loss(TopKBadCaptureLoss(top_ratio=0.5))
    >>> metric([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    1.0
    """
    from .custom_metrics import KSMetric
    from .expected_profit_loss import ExpectedProfitLoss
    from .ks_focused_loss import KSFocusedLoss
    from .ranking_auc_proxy_loss import RankingAUCProxyLoss
    from .ranking_loss import LiftFocusedLoss, OrdinalRankLoss
    from .risk_loss import ApprovalRateLoss, BadDebtLoss, ProfitMaxLoss
    from .topk_bad_capture_loss import TopKBadCaptureLoss
    from .weighted_loss import CostSensitiveLoss

    if not isinstance(loss, BaseLoss):
        raise TypeError("损失函数必须是BaseLoss实例")
    if isinstance(loss, (OrdinalRankLoss, RankingAUCProxyLoss)):
        return AUCMetric()
    if isinstance(loss, KSFocusedLoss):
        return KSMetric(name="KS区分度")
    if isinstance(loss, CostSensitiveLoss):
        return ClassificationCostMetric(fn_cost=loss.fn_cost, fp_cost=loss.fp_cost)
    if isinstance(loss, LiftFocusedLoss):
        return TopKLiftMetric(top_ratio=loss.top_ratio)
    if isinstance(loss, TopKBadCaptureLoss):
        return TopKCaptureMetric(top_ratio=loss.top_ratio)
    if isinstance(loss, BadDebtLoss):
        return BadDebtMetric(approval_rate=loss.target_approval_rate)
    if isinstance(loss, ApprovalRateLoss):
        return ApprovalRateMetric(target_bad_debt_rate=loss.target_bad_debt_rate)
    if isinstance(loss, ExpectedProfitLoss):
        return ProfitMetric(revenue=loss.revenue, default_cost=loss.default_cost, cutoff=loss.cutoff)
    if isinstance(loss, ProfitMaxLoss):
        return ProfitMetric(revenue=loss.interest_income, default_cost=loss.bad_debt_loss)
    return loss.metric()
