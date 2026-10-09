"""损失配套业务指标的手算、方向、同分边界及加权语义验证。"""

import numpy as np
import pytest

from hscredit.core.models.losses import (
    ApprovalRateLoss,
    BadDebtLoss,
    CostSensitiveLoss,
    ExpectedProfitLoss,
    FocalLoss,
    KSFocusedLoss,
    KSMetric,
    LiftFocusedLoss,
    OrdinalRankLoss,
    ProfitMaxLoss,
    RankingAUCProxyLoss,
    TopKBadCaptureLoss,
)
from hscredit.core.models.losses.business_metrics import (
    AUCMetric,
    ApprovalRateMetric,
    BadDebtMetric,
    ClassificationCostMetric,
    ProfitMetric,
    TopKCaptureMetric,
    TopKLiftMetric,
    business_metric_for_loss,
)


def test_auc_preserves_risk_direction_and_half_credit_for_ties():
    metric = AUCMetric()
    assert metric([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metric([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert metric([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # 好样本权重 3、1，坏样本权重 2、1；正确排序的样本对权重是 6+3+1。
    assert metric([0, 1, 0, 1], [0.1, 0.4, 0.6, 0.9], [3, 2, 1, 1]) == pytest.approx(10 / 12)


def test_topk_counts_round_up_and_lift_uses_realized_head_fraction():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.9]
    # ceil(5 * .21) = 2；lift 的分母采用全体坏率 .4，不直接用 1/.21。
    assert TopKCaptureMetric(0.21)(y, p) == 1.0
    assert TopKLiftMetric(0.21)(y, p) == 2.5
    assert TopKCaptureMetric(1.0)(y, p) == 1.0
    assert TopKLiftMetric(1.0)(y, p) == 1.0


def test_head_ties_are_fractional_and_invariant_to_label_row_order():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    capture = TopKCaptureMetric(0.25)
    lift = TopKLiftMetric(0.25)
    rng = np.random.default_rng(17)
    for _ in range(12):
        order = rng.permutation(len(y))
        assert capture(y[order], p[order], w[order]) == pytest.approx(1 / 3)
        # 同分三人各计 1/3：头部坏率为 4/6，总体坏率为 4/10。
        assert lift(y[order], p[order], w[order]) == pytest.approx(5 / 3)


def test_constant_predictions_have_neutral_head_metrics():
    y = [0, 0, 1, 1, 1]
    p = [0.5] * len(y)
    w = [1, 2, 3, 4, 5]
    assert TopKCaptureMetric(0.21)(y, p, w) == pytest.approx(0.4)
    assert TopKLiftMetric(0.21)(y, p, w) == pytest.approx(1.0)


def test_bad_debt_approves_low_risk_and_shares_boundary_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.3, 0.9])
    w = np.array([2.0, 4.0, 2.0, 1.0])
    # 两个人的名额：第一人全选，同分二人各一半；加权坏率 2/(2+2+1)。
    metric = BadDebtMetric(0.5)
    assert metric(y, p, w) == pytest.approx(0.4)
    assert metric(y[::-1], p[::-1], w[::-1]) == pytest.approx(0.4)
    assert metric.greater_is_better is False
    assert BadDebtMetric(1.0)(y, p, w) == pytest.approx(5 / 9)


def test_approval_rate_scans_nonmonotone_bad_rates_and_keeps_ties_together():
    # 最低风险人恰好为坏样本；不能遇到第一个不可行阈值就停止。
    assert ApprovalRateMetric(0.4)([1, 0, 0], [0.1, 0.2, 0.3]) == 1.0
    # 同分的好坏客户不能根据标签拆开通过。
    assert ApprovalRateMetric(0.4)([0, 1], [0.5, 0.5]) == 0.0
    assert ApprovalRateMetric(0.5)([0, 1], [0.5, 0.5]) == 1.0
    assert ApprovalRateMetric(0.0)([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.5


def test_approval_rate_weighted_threshold_and_zero_weight_group():
    metric = ApprovalRateMetric(0.2)
    assert metric([1, 0], [0.1, 0.2], [1, 9]) == 1.0
    assert metric([1, 0, 1], [0.0, 0.2, 0.9], [0, 9, 1]) == 1.0
    assert ApprovalRateMetric(0.0)([1, 0, 1], [0.0, 0.2, 0.9], [0, 9, 1]) == 0.9


def test_profit_uses_low_probability_approval_and_all_applicants_denominator():
    metric = ProfitMetric(revenue=100, default_cost=1000, cutoff=0.5)
    y = [0, 1, 0, 1]
    p = [0.1, 0.2, 0.5, 0.9]
    assert metric(y, p) == -225.0
    assert metric(y, p, [2, 1, 3, 4]) == -80.0
    assert ProfitMetric(cutoff=0)(y, p) == 0.0
    assert metric.greater_is_better is True


def test_classification_cost_uses_real_errors_and_matches_explicit_loss_method():
    loss = CostSensitiveLoss(fn_cost=100, fp_cost=5)
    metric = ClassificationCostMetric(fn_cost=100, fp_cost=5)
    y = [0, 1, 0, 1]
    p = [0.5, 0.2, 0.1, 0.9]
    assert metric(y, p) == (5 + 100) / 4
    assert metric(y, p, [2, 1, 3, 4]) == (10 + 100) / 10
    assert metric(y, p) == loss.classification_cost(y, p)
    assert metric.greater_is_better is False
    assert loss(y, p) != metric(y, p)


@pytest.mark.parametrize("metric", [AUCMetric(), TopKCaptureMetric(), TopKLiftMetric()])
def test_missing_effective_bad_class_is_an_explicit_error(metric):
    with pytest.raises(ValueError, match="坏样本"):
        metric([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="坏样本"):
        metric([0, 1], [0.1, 0.9], [1, 0])


def test_single_class_business_metrics_keep_defined_meaning():
    with pytest.raises(ValueError, match="好样本"):
        AUCMetric()([1, 1], [0.1, 0.9])
    assert TopKCaptureMetric(0.5)([1, 1], [0.1, 0.9]) == 0.5
    assert TopKLiftMetric(0.5)([1, 1], [0.1, 0.9]) == 1.0
    assert BadDebtMetric()([0, 0], [0.1, 0.2]) == 0.0
    assert BadDebtMetric()([1, 1], [0.1, 0.2]) == 1.0
    assert ApprovalRateMetric()([0, 0], [0.1, 0.2]) == 1.0
    assert ApprovalRateMetric()([1, 1], [0.1, 0.2]) == 0.0
    assert ApprovalRateMetric(1)([1, 1], [0.1, 0.2]) == 1.0


def test_empty_selected_weight_is_not_a_reported_zero_bad_rate():
    with pytest.raises(ValueError, match="权重总和"):
        BadDebtMetric(0.5)([0, 1], [0.1, 0.9], [0, 1])
    with pytest.raises(ValueError, match="权重总和"):
        TopKLiftMetric(0.5)([1, 0], [0.1, 0.9], [1, 0])


@pytest.mark.parametrize(
    "metric",
    [
        AUCMetric(),
        TopKCaptureMetric(),
        TopKLiftMetric(),
        BadDebtMetric(),
        ApprovalRateMetric(),
        ProfitMetric(),
        ClassificationCostMetric(),
    ],
)
@pytest.mark.parametrize(
    "y,p,w",
    [
        ([], [], None),
        ([0, 1], [0.1], None),
        ([0, 2], [0.1, 0.2], None),
        ([0, 1], [0.1, np.nan], None),
        ([0, 1], [0.1, 1.1], None),
        ([0, 1], [0.1, 0.9], [1, -1]),
        ([0, 1], [0.1, 0.9], [1, np.inf]),
        ([0, 1], [0.1, 0.9], [0, 0]),
        ([0, 1], [0.1, 0.9], [1]),
    ],
)
def test_business_metrics_share_strict_binary_input_validation(metric, y, p, w):
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        metric(y, p, w)


@pytest.mark.parametrize("factory", [TopKCaptureMetric, TopKLiftMetric, BadDebtMetric])
@pytest.mark.parametrize("value", [0, -0.1, 1.1, np.inf, np.nan, True, "0.5"])
def test_invalid_population_ratio_is_rejected(factory, value):
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        factory(value)


@pytest.mark.parametrize(
    "kwargs",
    [{"revenue": -1}, {"default_cost": np.inf}, {"cutoff": 1.1}, {"cutoff": np.nan}],
)
def test_invalid_profit_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        ProfitMetric(**kwargs)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_invalid_bad_debt_constraint_is_rejected(value):
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        ApprovalRateMetric(value)


@pytest.mark.parametrize(
    "loss,expected_type,parameter,expected_value",
    [
        (OrdinalRankLoss(name="自定义名称"), AUCMetric, None, None),
        (RankingAUCProxyLoss(), AUCMetric, None, None),
        (KSFocusedLoss(), KSMetric, None, None),
        (CostSensitiveLoss(fn_cost=100, fp_cost=2), ClassificationCostMetric, "fn_cost", 100),
        (LiftFocusedLoss(top_ratio=0.25), TopKLiftMetric, "top_ratio", 0.25),
        (TopKBadCaptureLoss(top_ratio=0.2), TopKCaptureMetric, "top_ratio", 0.2),
        (BadDebtLoss(target_approval_rate=0.7), BadDebtMetric, "approval_rate", 0.7),
        (ApprovalRateLoss(target_bad_debt_rate=0.02), ApprovalRateMetric, "target_bad_debt_rate", 0.02),
        (ProfitMaxLoss(interest_income=100, bad_debt_loss=1000), ProfitMetric, "revenue", 100),
        (ExpectedProfitLoss(cutoff=0.3), ProfitMetric, "cutoff", 0.3),
    ],
)
def test_loss_mapping_uses_type_and_preserves_business_parameters(loss, expected_type, parameter, expected_value):
    metric = business_metric_for_loss(loss)
    assert isinstance(metric, expected_type)
    if parameter is not None:
        assert getattr(metric, parameter) == expected_value


def test_generic_loss_uses_loss_value_as_metric_and_rejects_unknown_type():
    loss = FocalLoss(name="任意名称")
    metric = business_metric_for_loss(loss)
    y, p = np.array([0, 1]), np.array([0.1, 0.8])
    assert metric.greater_is_better is False
    assert metric(y, p) == pytest.approx(loss(y, p))
    with pytest.raises(TypeError, match="损失函数"):
        business_metric_for_loss("focal")
