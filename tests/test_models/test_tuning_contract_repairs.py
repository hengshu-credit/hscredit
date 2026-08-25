"""ModelTuner 交叉验证、失败状态与最佳模型契约测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from hscredit.core.models import ModelTuner
from hscredit.core.models.tuning.tuning import Metric


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


class _ConfiguredClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        bias=0.0,
        random_state=None,
        n_jobs=None,
        early_stopping_rounds=None,
        validation_fraction=0.0,
    ):
        self.bias = bias
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.rate_ = float(np.average(y, weights=sample_weight))
        self.fit_size_ = len(y)
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), np.clip(self.rate_ + self.bias, 0.01, 0.99))
        return np.column_stack([1.0 - positive, positive])


def _data():
    return make_classification(n_samples=60, n_features=4, random_state=43)


def test_logloss_metric_uses_probabilities_and_maximizes_negative_loss():
    metric = Metric("logloss")
    y_true = np.array([0, 1])

    good_score = metric(y_true, np.array([0.01, 0.99]))
    bad_score = metric(y_true, np.array([0.99, 0.01]))

    assert good_score > bad_score
    assert good_score == pytest.approx(-0.01005033585350145)


def test_callable_objective_preserves_explicit_minimize_direction():
    def calibration_error(y_true, y_prob):
        return float(np.mean(np.abs(y_true - y_prob)))

    tuner = ModelTuner(
        _SingleFitClassifier,
        search_space={},
        objective=calibration_error,
        direction="minimize",
        cv=2,
    )

    assert tuner.directions == ["minimize"]


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

    with pytest.raises(ValueError, match="训练实现故障"):
        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    failed_trial = tuner.study_.trials[0]
    assert failed_trial.state == optuna.trial.TrialState.FAIL
    assert failed_trial.user_attrs["错误类型"] == "RuntimeError"
    assert failed_trial.user_attrs["错误信息"] == "训练实现故障"


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


def test_fit_reuses_study_and_consumes_trial_enqueued_after_first_fit():
    X, y = _data()
    tuner = ModelTuner(
        _SingleFitClassifier,
        search_space={"bias": [0.0, 0.2]},
        metric="auc",
        cv=2,
        n_jobs=1,
        random_state=43,
    )
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    first_study = tuner.study_
    tuner.enqueue_trial({"bias": 0.2}, user_attrs={"来源": "人工排队"})

    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    assert tuner.study_ is first_study
    assert len(tuner.study_.trials) == 2
    assert tuner.study_.trials[1].params["bias"] == pytest.approx(0.2)
    assert tuner.study_.trials[1].user_attrs["来源"] == "人工排队"


def test_explicit_progress_bar_is_independent_from_verbose(monkeypatch):
    import optuna

    X, y = _data()
    tuner = ModelTuner(
        _SingleFitClassifier,
        search_space={},
        metric="auc",
        cv=2,
        verbose=False,
    )
    study = optuna.create_study(direction="maximize")
    original_optimize = study.optimize
    captured = {}

    def capture_optimize(*args, **kwargs):
        captured["show_progress_bar"] = kwargs["show_progress_bar"]
        kwargs["show_progress_bar"] = False
        return original_optimize(*args, **kwargs)

    monkeypatch.setattr(study, "optimize", capture_optimize)
    tuner.study_ = study

    tuner.fit(X, y, n_trials=1, show_progress_bar=True)

    assert captured["show_progress_bar"] is True


def test_get_best_model_reuses_tuner_runtime_configuration():
    X, y = _data()
    tuner = ModelTuner(
        _ConfiguredClassifier,
        search_space={"bias": [0.0]},
        metric="auc",
        cv=2,
        n_jobs=2,
        random_state=123,
        early_stopping_rounds=7,
    )
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    best_model = tuner.get_best_model()

    assert best_model.fit_size_ == len(y)
    assert best_model.random_state == 123
    assert best_model.n_jobs == 2
    assert best_model.early_stopping_rounds == 7
    assert best_model.validation_fraction == pytest.approx(0.2)


def test_eval_ratios_are_validated_and_empty_list_is_preserved():
    with pytest.raises(ValueError, match="eval_ratios"):
        ModelTuner(_SingleFitClassifier, search_space={}, eval_ratios=[0.0], cv=2)

    tuner = ModelTuner(_SingleFitClassifier, search_space={}, eval_ratios=[], cv=2)

    assert tuner.eval_ratios == []


def test_eval_ratios_are_recorded_as_public_lift_history_columns():
    X, y = _data()
    tuner = ModelTuner(
        _ConfiguredClassifier,
        search_space={"bias": [0.0]},
        metric="auc",
        eval_ratios=[0.2],
        cv=2,
        random_state=43,
    )

    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    assert "LIFT@20%" in tuner.optimization_history_.columns
    assert np.isfinite(tuner.optimization_history_.loc[0, "LIFT@20%"])
    assert tuner.study_.trials[0].user_attrs["LIFT@20%"] == pytest.approx(
        tuner.optimization_history_.loc[0, "LIFT@20%"]
    )


def test_analysis_apis_translate_latent_parameter_names(monkeypatch):
    import hscredit.core.models.tuning.tuning as tuning_module
    from types import SimpleNamespace

    tuner = ModelTuner(
        _ConfiguredClassifier,
        search_space={"bias": {"type": "quniform", "low": 0.0, "high": 0.4, "q": 0.1}},
        metric="auc",
        cv=2,
    )
    tuner.study_ = object()
    latent_name = tuner._space_adapter.latent_name("bias")
    captured = {}

    def fake_plot_contour(study, params, **kwargs):
        captured["params"] = params
        return "figure"

    monkeypatch.setattr(
        tuning_module,
        "optuna",
        SimpleNamespace(
            importance=SimpleNamespace(
                get_param_importances=lambda study, **kwargs: {latent_name: 1.0}
            ),
            visualization=SimpleNamespace(plot_contour=fake_plot_contour),
        ),
    )
    importance = tuner.get_param_importance()
    figure = tuner.plot_contour(params=["bias"])

    assert importance.index.tolist() == ["bias"]
    assert captured["params"] == [latent_name]
    assert figure == "figure"


def test_plotly_figures_do_not_expose_latent_parameter_names():
    plotly = pytest.importorskip("plotly.graph_objects")
    tuner = ModelTuner(
        _ConfiguredClassifier,
        search_space={"bias": {"type": "quniform", "low": 0.0, "high": 0.4, "q": 0.1}},
        metric="auc",
        cv=2,
    )
    latent_name = tuner._space_adapter.latent_name("bias")
    figure = plotly.Figure()
    figure.add_scatter(name=latent_name, hovertemplate=f"参数={latent_name}")
    figure.update_layout(title=latent_name, xaxis_title=latent_name)

    public_figure = tuner._publicize_plot_figure(figure)
    serialized = str(public_figure.to_plotly_json())

    assert latent_name not in serialized
    assert "bias" in serialized


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


def test_risk_model_tune_preserves_instance_params_target_and_sample_weight():
    from hscredit.core.models import RandomForest

    X, y = _data()
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    frame["label"] = y
    sample_weight = np.where(y == 1, 3.0, 1.0)
    model = RandomForest(
        n_estimators=3,
        max_depth=1,
        min_samples_split=4,
        n_jobs=1,
        random_state=43,
        target="label",
    )

    best_model = model.tune(
        frame,
        search_space={"min_samples_leaf": [1]},
        sample_weight=sample_weight,
        metric="auc",
        n_trials=1,
        cv=2,
        n_jobs=1,
        show_progress_bar=False,
    )

    assert best_model.n_estimators == 3
    assert best_model.max_depth == 1
    assert best_model.min_samples_split == 4
    assert best_model.target == "label"
    assert best_model.random_state == 43
    np.testing.assert_array_equal(model.tuner._sample_weight, sample_weight)


def test_logistic_tune_preserves_instance_params_and_sample_weight():
    from hscredit.core.models import LogisticRegression

    X, y = _data()
    sample_weight = np.where(y == 1, 2.0, 1.0)
    model = LogisticRegression(
        C=0.25,
        calculate_stats=False,
        max_iter=40,
        random_state=43,
        solver="liblinear",
    )

    best_model = model.tune(
        X,
        y,
        search_space={"tol": [1e-3]},
        sample_weight=sample_weight,
        metric="auc",
        n_trials=1,
        cv=2,
        n_jobs=1,
    )

    assert best_model.C == pytest.approx(0.25)
    assert best_model.calculate_stats is False
    assert best_model.max_iter == 40
    assert best_model.random_state == 43
    assert best_model.solver == "liblinear"
    np.testing.assert_array_equal(model.tuner._sample_weight, sample_weight)
