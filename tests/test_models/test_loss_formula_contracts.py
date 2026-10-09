"""训练导数、独立评估值及逐样本贡献必须对应同一个概率目标。"""

import numpy as np
import pytest

from hscredit.core.models.losses import (
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

Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
P = np.array([0.16, 0.43, 0.37, 0.81, 0.64])


def _losses():
    return [
        FocalLoss(gamma=0),
        FocalLoss(gamma=0.5),
        FocalLoss(gamma=2),
        AsymmetricFocalLoss(clip_value=0.1, gamma_pos=0.5, gamma_neg=2),
        BalancedFocalLoss(label_smoothing=0.2),
        BalancedFocalLoss(auto_alpha=False, label_smoothing=0.1),
        WeightedBCELoss(pos_weight=2, neg_weight=0.7),
        WeightedBCELoss(auto_balance=True),
        CostSensitiveLoss(fn_cost=3, fp_cost=0.7),
        AmountWeightedLoss(amounts=np.array([0, 3, 8, 10, 1]), floor_weight=0.2),
        ExpectedValueLoss(lgd=0.5, ead=np.array([3, 5, 8, 10, 2]), cost=0.1),
        ExpectedProfitLoss(revenue=2, default_cost=8),
        ProfitMaxLoss(interest_income=2, bad_debt_loss=8),
        BadDebtLoss(target_approval_rate=0.4),
        ApprovalRateLoss(target_bad_debt_rate=0.2),
        OrdinalRankLoss(rank_weight=2, temperature=0.3),
        LiftFocusedLoss(top_ratio=0.4),
        RankingAUCProxyLoss(rank_weight=2, margin=0.3, hard_mining_ratio=0.7),
        KSFocusedLoss(focus_weight=2),
        KSFocusedLoss(focus_weight=0),
        TopKBadCaptureLoss(top_ratio=0.3),
    ]


@pytest.mark.parametrize("loss", _losses(), ids=lambda loss: loss.name)
def test_probability_derivatives_match_whole_dataset_objective(loss):
    """有限差分每个概率，验证 d(n*mean_loss)/dp 及 Hessian 对角项。"""
    epsilon = 1e-5
    center = loss(Y, P)
    gradient = []
    hessian = []
    for index in range(len(Y)):
        plus, minus = P.copy(), P.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss, minus_loss = loss(Y, plus), loss(Y, minus)
        gradient.append(len(Y) * (plus_loss - minus_loss) / (2 * epsilon))
        hessian.append(len(Y) * (plus_loss - 2 * center + minus_loss) / epsilon**2)
    np.testing.assert_allclose(loss.gradient(Y, P), gradient, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(loss.hessian(Y, P), hessian, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("loss", _losses()[:13], ids=lambda loss: loss.name)
def test_sample_contributions_average_to_loss(loss):
    contributions = loss.loss_values(Y.tolist(), P.tolist())
    assert contributions.shape == Y.shape
    assert np.mean(contributions) == pytest.approx(loss(Y, P))


@pytest.mark.parametrize("loss", _losses(), ids=lambda loss: loss.name)
@pytest.mark.parametrize("probabilities", [[-0.1] * 5, [1.1] * 5, [np.nan] * 5, [[0.5]] * 5, []])
def test_invalid_probability_input_fails_clearly(loss, probabilities):
    with pytest.raises(ValueError):
        loss(Y, probabilities)


def test_focal_zero_gamma_is_class_weighted_bce():
    focal = FocalLoss(alpha=0.7, gamma=0)
    weighted = WeightedBCELoss(pos_weight=0.7, neg_weight=0.3)
    np.testing.assert_allclose(focal.loss_values(Y, P), weighted.loss_values(Y, P))
    np.testing.assert_allclose(focal.gradient(Y, P), weighted.gradient(Y, P))
    np.testing.assert_allclose(focal.hessian(Y, P), weighted.hessian(Y, P))


def test_asymmetric_clipping_only_suppresses_easy_negative_samples():
    loss = AsymmetricFocalLoss(clip_value=0.1)
    unclipped = AsymmetricFocalLoss()
    y = np.array([0, 1])
    p = np.array([0.05, 0.4])
    assert loss.gradient(y, p)[0] == 0
    assert loss.hessian(y, p)[0] == 0
    assert loss.loss_values(y, p)[1] == pytest.approx(unclipped.loss_values(y, p)[1])
    assert loss.gradient(y, p)[1] == pytest.approx(unclipped.gradient(y, p)[1])


def test_label_smoothing_changes_values_and_gradients():
    ordinary = BalancedFocalLoss(label_smoothing=0)
    smoothed = BalancedFocalLoss(label_smoothing=0.2)
    assert ordinary(Y, P) != pytest.approx(smoothed(Y, P))
    assert not np.allclose(ordinary.gradient(Y, P), smoothed.gradient(Y, P))


def test_auto_balance_does_not_reuse_weights_for_a_single_class():
    loss = WeightedBCELoss(auto_balance=True)
    loss(np.array([0, 0, 0, 1]), np.full(4, 0.5))
    assert loss.pos_weight == 3
    assert loss(np.ones(2), np.full(2, 0.5)) == pytest.approx(np.log(2))


def test_cost_surrogate_and_hard_cost_are_distinct_explicit_operations():
    loss = CostSensitiveLoss(fn_cost=10, fp_cost=2)
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    assert loss.classification_cost(y, p) == 3
    assert loss(y, p, threshold=0.5) == 3
    assert loss(y, p) == pytest.approx(WeightedBCELoss(10, 2)(y, p))


def test_business_approval_uses_low_bad_probability():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = BadDebtLoss()._compute_metrics(y, p, 0.5)
    assert metrics == {"approval_rate": 0.5, "bad_debt_rate": 0.0}
    assert ProfitMaxLoss(interest_income=2).profit(y, p) == 1
    assert ApprovalRateLoss(target_bad_debt_rate=0).approval_rate(y, p) == 0.5
    assert ApprovalRateLoss(target_bad_debt_rate=0.5).approval_rate(y, p) == 1


def test_hard_business_decisions_preserve_exact_probability_endpoints():
    y, p = np.array([0, 1]), np.array([0.0, 1.0])
    assert CostSensitiveLoss().classification_cost(y, p, threshold=1) == 0
    assert ProfitMaxLoss().profit(y, p, threshold=1) == 0.5


def test_approval_evaluation_handles_ties_without_label_order_leakage():
    loss = ApprovalRateLoss(target_bad_debt_rate=0)
    assert loss.approval_rate(np.array([0, 1]), np.array([0.2, 0.2])) == 0
    assert loss.approval_rate(np.array([1, 0]), np.array([0.2, 0.2])) == 0


@pytest.mark.parametrize(
    "loss",
    [
        AmountWeightedLoss(amounts=np.array([1, -1, 1, 1, 1])),
        AmountWeightedLoss(amounts=np.ones((5, 1))),
        AmountWeightedLoss(amounts=np.array([1, np.nan, 1, 1, 1])),
        AmountWeightedLoss(amounts=np.ones(4)),
        AmountWeightedLoss(amounts=np.zeros(5), floor_weight=0),
        ExpectedValueLoss(ead=np.ones(4)),
        ExpectedValueLoss(lgd=1.5),
        ExpectedValueLoss(rate=np.inf),
        ExpectedValueLoss(cost=-1),
    ],
)
def test_sample_parameters_are_validated(loss):
    with pytest.raises(ValueError):
        loss(Y, P)


@pytest.mark.parametrize(
    "constructor,parameters",
    [
        (FocalLoss, {"gamma": -1}),
        (AsymmetricFocalLoss, {"clip_value": 1}),
        (BalancedFocalLoss, {"beta": 1}),
        (WeightedBCELoss, {"pos_weight": 0, "neg_weight": 0}),
        (ExpectedProfitLoss, {"temperature": 0}),
        (OrdinalRankLoss, {"max_pairs": 1.5}),
        (RankingAUCProxyLoss, {"hard_mining_ratio": 0}),
        (KSFocusedLoss, {"bandwidth": 0}),
        (TopKBadCaptureLoss, {"top_ratio": 1.1}),
        (BadDebtLoss, {"target_approval_rate": 0}),
    ],
)
def test_invalid_hyperparameters_fail_at_construction(constructor, parameters):
    with pytest.raises(ValueError):
        constructor(**parameters)


@pytest.mark.parametrize("parameter", [None, "2", True, np.bool_(False), 1 + 2j, [], 10**1000])
def test_nonnumeric_hyperparameters_raise_chinese_value_error(parameter):
    with pytest.raises(ValueError, match="参数 gamma 必须"):
        FocalLoss(gamma=parameter)


@pytest.mark.parametrize(
    "labels,probabilities",
    [([0, "非数值标签"], [0.2, 0.8]), ([0, 1], [0.2, "非数值概率"]), ([0, 1], [0.2, object()])],
)
def test_nonnumeric_arrays_raise_chinese_value_error(labels, probabilities):
    with pytest.raises(ValueError, match="标签与预测概率必须是可转换为数值的数组"):
        FocalLoss()(labels, probabilities)


@pytest.mark.parametrize("ratio", [1e-9, np.float32(0.5), 1])
@pytest.mark.parametrize("loss_type", [BadDebtLoss, TopKBadCaptureLoss, LiftFocusedLoss])
def test_valid_selection_ratios_always_produce_a_working_business_metric(loss_type, ratio):
    keyword = "target_approval_rate" if loss_type is BadDebtLoss else "top_ratio"
    loss = loss_type(**{keyword: ratio})
    assert np.isfinite(loss.business_metric()(Y, P))


def test_pair_sampling_never_materializes_full_cartesian_product(monkeypatch):
    loss = OrdinalRankLoss(max_pairs=11)
    labels = np.r_[np.zeros(10000), np.ones(10000)]

    def unexpected_product(*args, **kwargs):
        raise AssertionError("不应构造正负样本全组合数组")

    monkeypatch.setattr(np, "repeat", unexpected_product)
    monkeypatch.setattr(np, "tile", unexpected_product)
    positive_indices, negative_indices = loss._prepare_pairs(labels)
    assert len(positive_indices) == len(negative_indices) == 11
    assert np.all(labels[positive_indices] == 1)
    assert np.all(labels[negative_indices] == 0)
