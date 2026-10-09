"""指标的方向、输入、权重、同分处理及自定义函数接入契约。"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

from hscredit import make_metric
from hscredit.core.models.losses import AUCMetric, FocalLoss, KSFocusedLoss
from hscredit.core.models.tuning import Metric, ModelTuner, TuningObjective, TuningSampler


def test_default_metric_directions_are_inferred_separately():
    tuner = ModelTuner(LogisticRegression, metric=["auc", "ks_diff", "log_loss", "brier"], search_space={})
    assert tuner.directions == ["maximize", "minimize", "minimize", "minimize"]
    assert ModelTuner(LogisticRegression, metric=FocalLoss(), search_space={}).direction == "minimize"


def test_probability_auc_does_not_reward_reversed_predictions():
    y, p = [0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]
    assert Metric(AUCMetric())(y, p) == 0
    assert TuningObjective.auc(y, p, score_direction="higher_risk") == 0
    explicit_auc = TuningObjective.get("auc", score_direction="higher_risk")
    assert explicit_auc(y, p) == 0
    assert Metric(explicit_auc)(y, p) == 0


def test_string_auc_preserves_legacy_automatic_direction():
    y, p = [0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]
    assert Metric("auc")(y, p) == 1
    assert TuningObjective.auc(y, p) == 1
    assert TuningObjective.get("auc")(y, p) == 1


@pytest.mark.parametrize("score_direction", ["auto", "higher_risk", "higher_safe"])
@pytest.mark.parametrize("sample_weight", [None, [1, 4, 3, 2]])
def test_auc_direction_binding_and_weight_match_raw_ranking(score_direction, sample_weight):
    y, p = [0, 0, 1, 1], [0.8, 0.2, 0.3, 0.1]
    raw = roc_auc_score(y, p, sample_weight=sample_weight)
    expected = max(raw, 1 - raw) if score_direction == "auto" else (
        raw if score_direction == "higher_risk" else 1 - raw
    )
    metric = TuningObjective.get("auc", score_direction=score_direction)
    assert TuningObjective.auc(
        y, p, score_direction=score_direction, sample_weight=sample_weight
    ) == pytest.approx(expected)
    assert metric(y, p, sample_weight=sample_weight) == pytest.approx(expected)
    assert Metric(metric)(y, p, sample_weight=sample_weight) == pytest.approx(expected)
    assert Metric(AUCMetric())(y, p, sample_weight=sample_weight) == pytest.approx(raw)


def test_auc_direction_parameters_are_validated():
    with pytest.raises(ValueError, match="score_direction"):
        TuningObjective.auc([0, 1], [0.1, 0.9], score_direction="invalid")
    with pytest.raises(ValueError, match="不支持参数"):
        TuningObjective.get("auc", score_diretion="higher_risk")


@pytest.mark.parametrize("metric", [Metric("auc"), TuningObjective.auc, TuningObjective.get("auc")])
@pytest.mark.parametrize("y,p", [([0, 1], [0.1, np.nan]), ([0, 1], [-1, 2]), ([0, 0], [0.1, 0.2])])
def test_auc_errors_are_not_silently_converted_into_scores(metric, y, p):
    with pytest.raises(ValueError):
        metric(y, p)


def test_raw_log_loss_and_legacy_negative_logloss_are_both_explicit():
    y, p = [0, 1], [0.1, 0.9]
    assert Metric("log_loss")(y, p) == pytest.approx(log_loss(y, p))
    assert Metric("log_loss").direction == "minimize"
    assert Metric("logloss")(y, p) == pytest.approx(-log_loss(y, p))
    assert Metric("neg_log_loss")(y, p) == Metric("logloss")(y, p)
    assert Metric("logloss").direction == "maximize"


def test_classification_threshold_treats_probability_half_as_positive():
    assert Metric("accuracy")([1, 0], [0.5, 0.49]) == 1


def test_function_metric_reuses_business_parameters_and_weights():
    def cost(y, p, threshold=0.5, sample_weight=None):
        return float(np.average((p >= threshold) != y, weights=sample_weight))

    metric = make_metric(cost, name="错分成本", greater_is_better=False, threshold=0.7)
    assert metric([0, 1], [0.6, 0.8]) == 0
    assert metric([0, 1], [0.6, 0.6], sample_weight=[1, 3]) == 0.75
    assert Metric(metric).direction == "minimize"
    assert metric.to_lightgbm()([0, 1], [0.6, 0.8]) == ("错分成本", 0.0, False)


@pytest.mark.parametrize("bad", [("name", 0.4), [0.4], np.array([0.4]), np.nan, np.inf, True, None])
def test_custom_metric_rejects_invalid_return_protocol(bad):
    metric = make_metric(lambda y, p: bad, greater_is_better=False)
    with pytest.raises(ValueError):
        metric([0, 1], [0.2, 0.8])
    with pytest.raises(ValueError):
        Metric(lambda y, p: bad, direction="minimize")([0, 1], [0.2, 0.8])


def test_plain_function_requires_direction_and_sklearn_scorers_are_not_metrics():
    with pytest.raises(ValueError, match="direction"):
        ModelTuner(LogisticRegression, metric=lambda y, p: np.mean((y - p) ** 2))
    with pytest.raises(TypeError, match="scorer"):
        Metric(make_scorer(log_loss), direction="minimize")


def test_function_kwargs_and_unsupported_weight_are_rejected_early():
    with pytest.raises(ValueError, match="签名"):
        make_metric(lambda y, p: 0.0, greater_is_better=False, typo=1)
    metric = make_metric(lambda y, p: 0.0, name="指标", greater_is_better=False)
    with pytest.raises(ValueError, match="评估权重"):
        Metric(metric).validate_evaluation_weight()
    with pytest.raises(ValueError, match="sample_weight"):
        metric([0, 1], [0.2, 0.8], sample_weight=[1, 1])
    with pytest.raises(ValueError, match="评估权重"):
        Metric(KSFocusedLoss().metric()).validate_evaluation_weight()


def test_make_metric_supports_cv_and_tuning_without_defining_a_class():
    X, y = make_classification(n_samples=60, n_features=4, random_state=3)
    metric = make_metric(log_loss, name="交叉熵", greater_is_better=False, labels=[0, 1])
    scores = cross_val_score(LogisticRegression(), X, y, scoring=metric.to_scorer(), cv=2)
    assert np.isfinite(scores).all() and (scores < 0).all()
    tuner = ModelTuner(LogisticRegression, search_space={"C": [0.1, 1]}, metric=metric, cv=2, n_jobs=1)
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    assert tuner.direction == "minimize"
    assert tuner.best_score_ > 0


@pytest.mark.parametrize("name", TuningObjective.BUILTIN_OBJECTIVES)
def test_objective_list_inputs_and_ties_are_order_invariant(name):
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.2, 0.3, 0.3, 0.8, 0.8, 0.9, 0.9])
    permutation = np.array([1, 0, 3, 2, 5, 4, 7, 6])
    function = getattr(TuningObjective, name)
    expected = function(y.tolist(), p.tolist())
    assert np.isfinite(expected)
    assert function(y[permutation], p[permutation]) == pytest.approx(expected)


def test_objective_bound_parameters_keep_automatic_direction_and_reject_typos():
    metric = TuningObjective.get("lift_head", ratio=0.5)
    assert ModelTuner(LogisticRegression, search_space={}, metric=metric).direction == "maximize"
    with pytest.raises(ValueError, match="不支持参数"):
        TuningObjective.get("lift_head", ration=0.5)


def test_sampler_rejects_typos_and_invalid_objects():
    with pytest.raises(ValueError, match="sampler_kwargs"):
        TuningSampler.create("tpe", n_startup_trial=4)
    with pytest.raises(TypeError, match="BaseSampler"):
        TuningSampler.create(object())
    sampler = TuningSampler.create("random", seed=42)
    assert TuningSampler.create(sampler) is sampler
    with pytest.raises(ValueError, match="实例"):
        TuningSampler.create(sampler, typo=1)


def test_metric_object_cannot_be_accidentally_wrapped_with_reversed_direction():
    with pytest.raises(ValueError, match="方向不一致"):
        Metric(FocalLoss().metric(), direction="maximize")


def test_bare_and_partial_monotonic_objectives_reject_weights_before_fitting():
    from functools import partial

    for function in (TuningObjective.lift_head_monotonic, partial(TuningObjective.lift_head_monotonic, n_bins=3)):
        with pytest.raises(ValueError, match="评估权重"):
            Metric(function, direction="maximize").validate_evaluation_weight()
