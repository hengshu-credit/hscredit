"""损失和指标的统一创建入口与可查询参数目录。"""

from copy import deepcopy
from inspect import signature

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .base import BaseLoss, BaseMetric
from .functional import make_metric


def _losses():
    from . import (
        AmountWeightedLoss,
        ApprovalRateLoss,
        AsymmetricFocalLoss,
        BadDebtLoss,
        BalancedFocalLoss,
        CostSensitiveLoss,
        ExpectedProfitLoss,
        ExpectedValueLoss,
        FocalLoss,
        KSFocusedLoss,
        LiftFocusedLoss,
        OrdinalRankLoss,
        ProfitMaxLoss,
        RankingAUCProxyLoss,
        TopKBadCaptureLoss,
        WeightedBCELoss,
    )

    return {
        "weighted_bce": (WeightedBCELoss, "类别加权交叉熵"),
        "cost_sensitive": (CostSensitiveLoss, "漏判与误判成本"),
        "focal": (FocalLoss, "聚焦难样本"),
        "balanced_focal": (BalancedFocalLoss, "类别平衡与难样本"),
        "asymmetric_focal": (AsymmetricFocalLoss, "非对称聚焦"),
        "bad_debt": (BadDebtLoss, "固定通过率降低坏账"),
        "approval_rate": (ApprovalRateLoss, "坏账约束下提高通过率"),
        "profit_max": (ProfitMaxLoss, "收益与坏账损失"),
        "expected_profit": (ExpectedProfitLoss, "预期利润代理目标"),
        "ordinal_rank": (OrdinalRankLoss, "顺序排序"),
        "lift_focused": (LiftFocusedLoss, "头部提升"),
        "ranking_auc_proxy": (RankingAUCProxyLoss, "排序与AUC代理目标"),
        "ks_focused": (KSFocusedLoss, "KS分布分离"),
        "topk_bad_capture": (TopKBadCaptureLoss, "头部坏样本捕获"),
        "amount_weighted": (AmountWeightedLoss, "金额加权"),
        "expected_value": (ExpectedValueLoss, "敞口与预期价值加权"),
    }


def _metrics():
    from . import (
        AUCMetric,
        ApprovalRateMetric,
        BadDebtMetric,
        ClassificationCostMetric,
        GiniMetric,
        KSMetric,
        PSIMetric,
        ProfitMetric,
        TopKCaptureMetric,
        TopKLiftMetric,
    )

    return {
        "auc": (AUCMetric, "AUC曲线下面积", "maximize"),
        "ks": (KSMetric, "KS区分度", "maximize"),
        "gini": (GiniMetric, "基尼系数", "maximize"),
        "psi": (PSIMetric, "群体稳定性", "minimize"),
        "topk_capture": (TopKCaptureMetric, "头部坏样本捕获率", "maximize"),
        "topk_lift": (TopKLiftMetric, "头部提升倍数", "maximize"),
        "bad_debt": (BadDebtMetric, "通过客户坏账率", "minimize"),
        "approval_rate": (ApprovalRateMetric, "坏账约束下最大通过率", "maximize"),
        "profit": (ProfitMetric, "人均利润", "maximize"),
        "classification_cost": (ClassificationCostMetric, "人均误判成本", "minimize"),
    }


def _entry(spec, entries, category):
    for key, entry in entries.items():
        if spec == key or spec == entry[0].__name__:
            return entry
    raise ValueError(f"未知{category}: {spec!r}；可选: {', '.join(entries)}")


def get_loss(spec, **params):
    """按名称或现有对象创建独立损失配置。

    :param spec: list_losses 中的标识、类名或 BaseLoss 实例。
    :param params: 损失构造参数；现有实例不接受覆盖，避免修改学习状态。
    :return: BaseLoss，可传给模型 objective 或 ModelTuner 的 loss。

    训练目标支持范围由算法决定；不能接收自定义梯度的模型仍可使用 loss.metric() 评估和选模。
    参数名称、默认值及用途通过 list_losses() 查看；金融逐行数组须与使用它的数据集按位置对齐。

    >>> loss = get_loss('cost_sensitive', fn_cost=5, fp_cost=1)
    >>> loss.metric(kind='business').direction
    'minimize'
    """
    if isinstance(spec, BaseLoss):
        if params:
            raise ValueError("损失实例不接受参数覆盖，请用损失名称重新创建")
        return deepcopy(spec)
    if not isinstance(spec, str):
        raise TypeError("损失必须是名称或 BaseLoss 实例")
    return _entry(spec, _losses(), "损失")[0](**params)


def get_metric(spec, *, kind="loss", **params):
    """按名称、损失、指标对象或概率函数创建统一指标。

    :param spec: list_metrics 中的标识/类名、BaseLoss、BaseMetric 或概率评估函数。
    :param kind: 仅损失对象支持 loss/business，等价于 loss.metric(kind=...)。
    :param params: 指标构造参数；自定义函数必须指定 greater_is_better。
    :return: BaseMetric，可共用于离线 evaluate、模型 eval_metric、调参 metric。

    指标方向由对象携带；kind='loss' 为可优化的代理损失，kind='business' 为对应的真实业务统计。
    raw margin 或信用分不能直接当作坏样本概率传入。多个数据集的金融权重应在各调用入口显式指定。

    >>> metric = get_metric('topk_capture', top_ratio=0.05)
    >>> metric.direction
    'maximize'
    """
    if isinstance(spec, BaseLoss):
        return spec.metric(kind=kind, **params)
    if kind != "loss":
        raise ValueError("kind 仅用于从损失对象生成指标")
    if isinstance(spec, BaseMetric):
        if params:
            raise ValueError("指标实例不接受参数覆盖，请用指标名称重新创建")
        return deepcopy(spec)
    if callable(spec):
        return make_metric(spec, **params)
    if not isinstance(spec, str):
        raise TypeError("指标必须是名称、损失/指标实例或概率评估函数")
    if spec in ("logloss", "brier"):
        function, label = (log_loss, "交叉熵") if spec == "logloss" else (brier_score_loss, "概率均方误差")
        if spec == "logloss":
            params.setdefault("labels", [0, 1])
        params.setdefault("name", label)
        return make_metric(function, greater_is_better=False, **params)
    cls, label, _ = _entry(spec, _metrics(), "指标")
    params.setdefault("name", label)
    if cls.__name__ == "GiniMetric":
        params.setdefault("score_direction", "higher_risk")
    return cls(**params)


def list_losses():
    """查询可用损失及如何配置参数。

    :return: 包含标识、类名、用途和参数签名的 DataFrame；不导入可选 boosting 后端。

    **参考样例**

    >>> catalog = list_losses()
    >>> cost_loss = get_loss('cost_sensitive', fn_cost=5, fp_cost=1)
    >>> business_metric = cost_loss.metric(kind='business')
    """
    return pd.DataFrame(
        [
            {"标识": key, "类名": cls.__name__, "用途": purpose, "参数": str(signature(cls))}
            for key, (cls, purpose) in _losses().items()
        ]
    )


def list_metrics():
    """查询统一指标的用途、参数默认值及优化方向。

    :return: 包含标识、类名、用途、优化方向、参数签名的 DataFrame。
        损失派生的业务指标也可通过 get_metric(loss, kind='business') 创建。

    **参考样例**

    >>> catalog = list_metrics()
    >>> capture = get_metric('topk_capture', top_ratio=0.05)
    >>> assert capture.direction == 'maximize'
    """
    rows = [
        {"标识": key, "类名": cls.__name__, "用途": purpose, "优化方向": direction, "参数": str(signature(cls))}
        for key, (cls, purpose, direction) in _metrics().items()
    ]
    rows.extend(
        {
            "标识": key,
            "类名": "CallableMetric",
            "用途": purpose,
            "优化方向": "minimize",
            "参数": str(signature(function)),
        }
        for key, function, purpose in (("logloss", log_loss, "交叉熵"), ("brier", brier_score_loss, "概率均方误差"))
    )
    return pd.DataFrame(rows)


__all__ = ["get_loss", "get_metric", "list_losses", "list_metrics"]
