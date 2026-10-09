"""所有损失的离线评估、训练回调、参数快照与 sklearn 调参契约。"""

import pickle

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

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
from hscredit.core.models.losses.business_metrics import AUCMetric, BadDebtMetric, ProfitMetric

LOSS_TYPES = [
    FocalLoss,
    AsymmetricFocalLoss,
    BalancedFocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    ExpectedProfitLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    RankingAUCProxyLoss,
    KSFocusedLoss,
    TopKBadCaptureLoss,
    AmountWeightedLoss,
    ExpectedValueLoss,
]
LABELS = np.array([0, 1, 0, 0, 1, 0], dtype=float)
PROBABILITY = np.array([0.1, 0.7, 0.35, 0.8, 0.2, 0.15])


class DatasetLike:
    """复现原生框架读取标签、样本权重的公开协议。"""

    def __init__(self, labels, weight=None):
        self.labels = labels
        self.weight = np.array([]) if weight is None else weight

    def get_label(self):
        return self.labels

    def get_weight(self):
        return self.weight


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_every_loss_exposes_equal_value_metric_and_explicit_minimization(loss_type):
    loss = loss_type()
    expected = loss(LABELS, PROBABILITY)
    metric = loss.metric()
    assert metric(LABELS, PROBABILITY) == pytest.approx(expected)
    assert metric.evaluate(LABELS, PROBABILITY) == pytest.approx(expected)
    assert loss.evaluate(LABELS, PROBABILITY) == pytest.approx(expected)
    assert loss.to_metric()(LABELS, PROBABILITY) == pytest.approx(expected)
    assert metric.direction == "minimize"
    assert metric.greater_is_better is False
    assert loss.metric(name="评估指标").name == "评估指标"


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_every_loss_evaluates_two_column_probability_and_explicit_raw_margin(loss_type):
    loss = loss_type()
    metric = loss.metric()
    expected = metric(LABELS, PROBABILITY)
    two_columns = np.column_stack((1 - PROBABILITY, PROBABILITY))
    assert metric.evaluate(LABELS, two_columns) == pytest.approx(expected)
    assert loss.evaluate(LABELS, two_columns) == pytest.approx(expected)
    assert metric.evaluate(LABELS, logit(PROBABILITY), raw_score=True) == pytest.approx(expected)
    assert loss.evaluate(LABELS, logit(PROBABILITY), raw_score=True) == pytest.approx(expected)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("raw_score", [False, True])
def test_native_and_sklearn_metric_callbacks_agree_with_offline_evaluation(loss_type, raw_score):
    metric = loss_type().metric()
    expected = metric(LABELS, PROBABILITY)
    prediction = logit(PROBABILITY) if raw_score else PROBABILITY
    data = DatasetLike(LABELS)
    xgb_name, xgb_value = metric.to_xgboost(raw_score=raw_score)(prediction, data)
    assert xgb_name == metric.name
    assert xgb_value == pytest.approx(expected)
    assert metric.to_xgboost(api="sklearn", raw_score=raw_score)(LABELS, prediction) == pytest.approx(expected)
    name, value, greater = metric.to_lightgbm(api="native", raw_score=raw_score)(prediction, data)
    assert name == metric.name
    assert value == pytest.approx(expected)
    assert greater is False
    assert metric.to_lightgbm(raw_score=raw_score)(LABELS, prediction) == (name, value, greater)


@pytest.mark.parametrize("metric", [WeightedBCELoss(pos_weight=3).metric(), AUCMetric(), BadDebtMetric(0.5)])
def test_weighted_native_callbacks_and_catboost_protocol(metric):
    weight = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 1.0])
    expected = metric.evaluate(LABELS, PROBABILITY, sample_weight=weight)
    margin = logit(PROBABILITY)
    data = DatasetLike(LABELS, weight)
    name, value = metric.to_xgboost(raw_score=True)(margin, data)
    assert name == metric.name
    assert value == pytest.approx(expected)
    assert metric.to_lightgbm(raw_score=True)(LABELS, margin, weight)[1] == pytest.approx(expected)
    callback = metric.to_catboost()
    error, mass = callback.evaluate([margin], LABELS, weight)
    assert mass == np.sum(weight)
    assert callback.get_final_error(error, mass) == pytest.approx(expected)
    assert callback.is_max_optimal() is metric.greater_is_better


def test_raw_margin_conversion_is_explicit_even_inside_probability_range():
    metric = WeightedBCELoss().metric()
    raw = np.array([0.2, 0.8])
    y = [0, 1]
    assert metric.evaluate(y, raw, raw_score=True) == pytest.approx(metric(y, expit(raw)))
    assert metric.evaluate(y, raw, raw_score=True) != pytest.approx(metric(y, raw))
    with pytest.raises(ValueError, match="raw_score"):
        metric.evaluate(y, [-1, 1])
    with pytest.raises(ValueError, match="一维"):
        metric.evaluate(y, [[-1, 1], [-2, 2]], raw_score=True)
    with pytest.raises(ValueError, match="两列概率"):
        metric.evaluate(y, [[0.2, 0.9], [0.1, 0.7]])


def test_validation_does_not_modify_training_auto_balance_parameters():
    loss = WeightedBCELoss(auto_balance=True)
    loss.gradient([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.7])
    assert loss.pos_weight == 3
    metric = loss.metric()
    metric([0, 1, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert loss.pos_weight == 3
    assert metric.loss.pos_weight == pytest.approx(1 / 3)
    loss.evaluate([0, 1], [0.2, 0.7])
    assert loss.pos_weight == 3


def test_metric_snapshot_does_not_follow_later_training_parameter_changes():
    loss = FocalLoss(alpha=0.8)
    metric = loss.metric()
    expected = metric(LABELS, PROBABILITY)
    loss.alpha = 0.2
    assert metric(LABELS, PROBABILITY) == pytest.approx(expected)
    assert loss(LABELS, PROBABILITY) != pytest.approx(expected)


def test_amount_validation_parameters_are_separate_from_training_array():
    train_amount = np.array([100.0, 200.0, 300.0, 1000.0])
    valid_amount = np.array([2000.0, 500.0])
    loss = AmountWeightedLoss(amounts=train_amount)
    metric = loss.metric(amounts=valid_amount)
    y, p = [0, 1], [0.2, 0.7]
    expected = AmountWeightedLoss(amounts=valid_amount)(y, p)
    assert metric(y, p) == pytest.approx(expected)
    assert loss.evaluate(y, p, amounts=valid_amount) == pytest.approx(expected)
    np.testing.assert_array_equal(loss.amounts_, train_amount)
    valid_amount[0] = 99
    assert metric(y, p) == pytest.approx(expected)
    with pytest.raises(ValueError, match="长度"):
        loss.metric()(y, p)


def test_scorer_chooses_class_one_column_and_only_it_changes_loss_sign():
    class ReverseClassesEstimator:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.column_stack((PROBABILITY, 1 - PROBABILITY))

    metric = FocalLoss().metric()
    estimator = ReverseClassesEstimator()
    value = metric(LABELS, PROBABILITY)
    assert metric.to_scorer()(estimator, None, LABELS) == pytest.approx(-value)
    assert AUCMetric().to_scorer()(estimator, None, LABELS) == pytest.approx(AUCMetric()(LABELS, PROBABILITY))
    scorer = pickle.loads(pickle.dumps(metric.to_scorer()))
    assert scorer(estimator, None, LABELS) == pytest.approx(-value)


def test_business_metric_direction_matches_optimization_goal():
    assert AUCMetric().direction == "maximize"
    assert ProfitMetric().direction == "maximize"
    assert BadDebtMetric().direction == "minimize"
    assert FocalLoss().metric().direction == "minimize"


def test_loss_scorer_works_with_cross_validation_and_grid_search():
    X, y = make_classification(n_samples=90, n_features=4, n_informative=3, n_redundant=0, random_state=17)
    loss = WeightedBCELoss(pos_weight=2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    scores = cross_val_score(LogisticRegression(max_iter=200), X, y, scoring=loss.to_scorer(), cv=cv)
    assert np.isfinite(scores).all()
    assert np.all(scores < 0)
    search = GridSearchCV(
        LogisticRegression(max_iter=200),
        {"C": [0.1, 1]},
        scoring={"损失": loss.to_scorer(), "区分度": AUCMetric().to_scorer()},
        refit="损失",
        cv=cv,
    ).fit(X, y)
    assert search.best_score_ == np.max(search.cv_results_["mean_test_损失"])
    assert search.best_score_ < 0
