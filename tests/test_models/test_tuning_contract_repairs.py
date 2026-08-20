"""ModelTuner 交叉验证、失败状态与最佳模型契约测试。"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from hscredit.core.models import ModelTuner


class _SingleFitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bias=0.0):
        self.bias = bias

    def fit(self, X, y, sample_weight=None):
        if hasattr(self, "fit_size_"):
            raise RuntimeError("同一个模型实例被重复用于多个CV折")
        self.fit_size_ = len(y)
        self.classes_ = np.unique(y)
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), np.clip(self.rate_ + self.bias, 0.01, 0.99))
        return np.column_stack([1.0 - positive, positive])


class _BrokenClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, sample_weight=None):
        raise RuntimeError("训练实现故障")

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def _data():
    return make_classification(n_samples=60, n_features=4, random_state=43)


def test_cross_validation_uses_a_fresh_model_for_every_fold():
    X, y = _data()
    tuner = ModelTuner(
        _SingleFitClassifier,
        search_space={},
        metric="auc",
        cv=3,
        n_jobs=1,
        random_state=43,
    )

    params = tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    assert params == {}
    assert np.isfinite(tuner.best_score_)


def test_training_exceptions_mark_trial_failed_and_all_failed_is_explicit():
    import optuna

    X, y = _data()
    tuner = ModelTuner(
        _BrokenClassifier,
        search_space={},
        metric="auc",
        cv=2,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="所有Trial均失败"):
        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    assert tuner.study_.trials[0].state == optuna.trial.TrialState.FAIL


def test_get_best_model_returns_model_refitted_on_full_input():
    X, y = _data()
    tuner = ModelTuner(
        _SingleFitClassifier,
        search_space={},
        metric="auc",
        cv=2,
        n_jobs=1,
        random_state=43,
    )
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    best_model = tuner.get_best_model()

    assert best_model.fit_size_ == len(y)
    assert best_model.classes_.tolist() == [0, 1]


def test_risk_model_tune_exposes_one_shared_tuner_with_analysis_results():
    from hscredit.core.models import RandomForest

    X, y = _data()
    model = RandomForest(n_estimators=2, n_jobs=1, random_state=43)

    assert model.tuner is None

    best_model = model.tune(
        X,
        y,
        search_space={"max_depth": [2]},
        fixed_params={"n_estimators": 2, "n_jobs": 1},
        metric="auc",
        n_trials=1,
        cv=2,
        n_jobs=1,
    )

    assert model.tuner is best_model.tuner
    assert len(model.tuner.study_.trials) == 1
    assert not model.tuner.optimization_history_.empty
    assert model.tuner.best_params_["max_depth"] == 2
    assert np.isfinite(model.tuner.best_score_)


def test_extended_logistic_regression_tune_uses_the_same_public_tuner_contract():
    from hscredit.core.models import LogisticRegression

    X, y = _data()
    model = LogisticRegression(calculate_stats=False, solver="liblinear", random_state=43)

    assert model.tuner is None

    best_model = model.tune(
        X,
        y,
        search_space={"C": [1.0]},
        fixed_params={"calculate_stats": False, "max_iter": 50, "solver": "liblinear"},
        metric="auc",
        n_trials=1,
        cv=2,
        n_jobs=1,
    )

    assert model.tuner is best_model.tuner
    assert len(best_model.tuner.study_.trials) == 1
    assert best_model.tuner.best_params_["C"] == pytest.approx(1.0)
