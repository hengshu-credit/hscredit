import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from hscredit.core.models import AutoTuner, TuningObjective


optuna = pytest.importorskip("optuna")


def _small_binary_data(as_frame=False):
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    if as_frame:
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="target")
    return X, y


def test_model_tuner_supports_numpy_input_without_failed_trials():
    X, y = _small_binary_data(as_frame=False)
    tuner = AutoTuner.create("lr", metric="ks", cv=3, random_state=42)

    best_params = tuner.fit(X, y, n_trials=2, show_progress_bar=False)

    assert best_params
    assert np.isfinite(tuner.best_score_)
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in tuner.study_.trials)
    assert all(np.isfinite(trial.value) for trial in tuner.study_.trials)


def test_lightgbm_tuner_samples_num_leaves_after_max_depth_constraint():
    pytest.importorskip("lightgbm")
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create(
        "lightgbm",
        metric=["ks", "ks_diff"],
        direction=["maximize", "minimize"],
        cv=3,
        random_state=42,
    )

    tuner.fit(X, y, n_trials=3, show_progress_bar=False)

    assert tuner.pareto_front_
    assert len(tuner.best_scores_) == 2
    for trial in tuner.study_.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        max_depth = trial.params["max_depth"]
        num_leaves = trial.params["num_leaves"]
        assert num_leaves <= 2**max_depth


def test_multi_objective_default_best_trial_uses_primary_metric_first():
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create(
        "lr",
        metric=["ks", "ks_diff"],
        direction=["maximize", "minimize"],
        cv=3,
        random_state=42,
    )

    tuner.fit(X, y, n_trials=4, show_progress_bar=False)

    pareto = tuner.get_pareto_front()
    primary_scores = [trial.values[0] for trial in pareto]
    assert tuner.best_score_ == max(primary_scores)
    assert tuner._resolve_multi_objective_target(None) == 0


def test_metric_names_length_must_match_metric_list():
    with pytest.raises(ValueError, match="metric_names"):
        AutoTuner.create(
            "lr",
            metric=["ks", "ks_diff"],
            direction=["maximize", "minimize"],
            metric_names=["KS"],
        )


def test_business_tuning_objectives_are_available_as_metric_names():
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create("lr", metric="lift_head", cv=3, random_state=42)

    tuner.fit(X, y, n_trials=2, show_progress_bar=False)

    assert tuner.metric_names == ["LIFT_HEAD"]
    assert np.isfinite(tuner.best_score_)
    assert all(np.isfinite(trial.value) for trial in tuner.study_.trials)


def test_strategy_tuning_objectives_return_finite_values():
    y_true = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.05, 0.9, 0.2, 0.1, 0.8, 0.3])

    approval_score = TuningObjective.approval_bad_rate(y_true, y_prob, approval_rate=0.5)
    profit_score = TuningObjective.expected_profit(y_true, y_prob, approval_rate=0.5)

    assert np.isfinite(approval_score)
    assert np.isfinite(profit_score)
    assert TuningObjective.get("EXPECTED_PROFIT")(y_true, y_prob) == TuningObjective.expected_profit(y_true, y_prob)
