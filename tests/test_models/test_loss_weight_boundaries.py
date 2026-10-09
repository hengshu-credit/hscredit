"""损失接口的加权、行对齐与分批边界回归。"""

import numpy as np
import pytest
from scipy.special import logit
from hscredit.core.models.losses import (
    AmountWeightedLoss,
    ExpectedValueLoss,
    OrdinalRankLoss,
    KSFocusedLoss,
    TopKBadCaptureLoss,
    FocalLoss,
    KSMetric,
    GiniMetric,
)
from hscredit.core.models.losses.adapters import TabNetLossAdapter


@pytest.mark.parametrize(
    "loss",
    [
        AmountWeightedLoss(amounts=[1.0, 3.0], floor_weight=0),
        ExpectedValueLoss(lgd=0.8, ead=[1.0, 3.0], rate=0.2, floor_weight=0),
    ],
)
def test_combined_weights_preserve_constant_loss_and_match_direct_weighted_bce(loss):
    y = np.array([0.0, 1.0])
    weight = np.array([3.0, 1.0])
    assert loss.evaluate(y, [0.5, 0.5], sample_weight=weight) == pytest.approx(np.log(2))
    p = np.array([0.2, 0.6])
    intrinsic = loss._get_weights(2) if isinstance(loss, AmountWeightedLoss) else loss._get_weights(y)
    expected = np.average(-y * np.log(p) - (1 - y) * np.log1p(-p), weights=intrinsic * weight)
    assert loss.evaluate(y, p, sample_weight=weight) == pytest.approx(expected)
    margin = logit(p)
    gradient, _ = loss.to_xgboost(api="sklearn")(y, margin, sample_weight=weight)
    numerical = []
    for i in range(2):
        plus, minus = margin.copy(), margin.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        numerical.append(
            (loss.evaluate(y, plus, weight, raw_score=True) - loss.evaluate(y, minus, weight, raw_score=True)) / 2e-5
        )
    # 训练回调使用加权总和标度；评估采用加权均值。
    np.testing.assert_allclose(gradient / weight.sum(), numerical, rtol=1e-5)


@pytest.mark.parametrize("loss", [OrdinalRankLoss(), KSFocusedLoss(), TopKBadCaptureLoss()])
def test_global_loss_rejects_external_training_weights(loss):
    with pytest.raises(ValueError, match="整体样本分布"):
        loss.to_xgboost(api="sklearn")([0, 1], [-1, 1], sample_weight=[1, 2])
    with pytest.raises(ValueError, match="小批次"):
        TabNetLossAdapter(loss).loss_fn()


@pytest.mark.parametrize("loss", [AmountWeightedLoss(amounts=[1, 2]), ExpectedValueLoss(ead=[1, 2])])
def test_bound_arrays_cannot_silently_follow_cv_row_positions(loss):
    with pytest.raises(ValueError, match="每一折"):
        loss.to_scorer()


def test_scorer_validates_entire_probability_matrix():
    class Estimator:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.8, 0.8], [0.2, 0.7]])

    with pytest.raises(ValueError, match="合计为1"):
        FocalLoss().to_scorer()(Estimator(), None, [0, 1])


def test_existing_ks_gini_metrics_support_explicit_weights():
    y = [0, 0, 1, 1]
    p = [0.1, 0.7, 0.3, 0.9]
    weights = [1.0, 4.0, 2.0, 1.0]
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y, p, sample_weight=weights)
    assert KSMetric().evaluate(y, p, weights) == pytest.approx(np.max(np.abs(tpr - fpr)))
    assert GiniMetric(score_direction="higher_risk").evaluate(y, p, weights) == pytest.approx(
        2 * roc_auc_score(y, p, sample_weight=weights) - 1
    )
