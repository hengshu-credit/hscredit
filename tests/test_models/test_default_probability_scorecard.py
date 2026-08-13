import importlib.util
import inspect

import numpy as np
import pytest
from sklearn.datasets import make_classification

from hscredit.core.models import (
    CatBoostRiskModel,
    ExtraTreesRiskModel,
    GradientBoostingRiskModel,
    LightGBMRiskModel,
    LogisticRegression,
    NGBoostRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
)


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=42,
    )
    return X, y


def test_random_forest_default_scorecard_uses_training_bad_odds(binary_data):
    X, y = binary_data

    model = RandomForestRiskModel(n_estimators=5, random_state=42).fit(X, y)

    assert model.scorecard_config_["method"] == "standard"
    assert model.scorecard_config_["pdo"] == 50
    assert model.scorecard_config_["base_score"] == 600
    assert model.scorecard_config_["lower"] == 0
    assert model.scorecard_config_["upper"] == 1000
    assert model.scorecard_config_["direction"] == "descending"
    assert model.scorecard_config_["rate"] == 2
    assert model.scorecard_config_["decimal"] == 0
    assert model.scorecard_config_["clip"] is True
    assert model.bad_rate_ == pytest.approx(np.mean(y == 1))
    assert model.base_odds_ == pytest.approx(model.bad_rate_ / (1 - model.bad_rate_))
    assert model.scorecard_.predict_score(proba=[model.bad_rate_])[0] == 600


def test_partial_scorecard_params_preserve_other_defaults():
    model = RandomForestRiskModel(scorecard_params={"pdo": 20})

    assert model.scorecard_params == {"pdo": 20}
    assert model.scorecard_config_["pdo"] == 20
    assert model.scorecard_config_["base_score"] == 600
    assert model.scorecard_config_["lower"] == 0
    assert model.scorecard_config_["upper"] == 1000
    assert model.scorecard_config_["direction"] == "descending"


def test_set_params_scorecard_override_is_applied_when_fitting(binary_data):
    X, y = binary_data
    model = RandomForestRiskModel(n_estimators=5, random_state=42)

    model.set_params(scorecard_params={"pdo": 25})
    model.fit(X, y)

    assert model.scorecard_config_["pdo"] == 25
    assert model.scorecard_config_["base_score"] == 600


@pytest.mark.parametrize(
    "scorecard_params, error_type, message",
    [
        ("pdo=20", TypeError, "scorecard_params.*字典"),
        ({"base_odds": 0.1}, ValueError, "不支持的评分卡参数.*base_odds"),
        ({"unknown": 1}, ValueError, "不支持的评分卡参数.*unknown"),
    ],
)
def test_invalid_scorecard_params_raise_chinese_error(scorecard_params, error_type, message):
    with pytest.raises(error_type, match=message):
        RandomForestRiskModel(scorecard_params=scorecard_params)


def test_predict_score_uses_fitted_probability_scorecard(binary_data):
    X, y = binary_data
    model = RandomForestRiskModel(n_estimators=5, random_state=42).fit(X, y)

    probability = model.predict_proba(X)[:, 1]
    expected = model.scorecard_.predict_score(proba=probability)
    scores = model.predict_score(X)

    np.testing.assert_array_equal(scores, expected)
    assert np.all((scores >= 0) & (scores <= 1000))
    assert np.all(scores == np.round(scores))
    low_probability_score = model.scorecard_.predict_score(proba=[0.05])[0]
    high_probability_score = model.scorecard_.predict_score(proba=[0.50])[0]
    assert low_probability_score > high_probability_score


@pytest.mark.parametrize("labels", [np.zeros(12, dtype=int), np.ones(12, dtype=int), np.tile([-1, 1], 6)])
def test_invalid_training_labels_fail_before_native_model_training(labels):
    X = np.arange(24, dtype=float).reshape(12, 2)
    model = RandomForestRiskModel(n_estimators=5, random_state=42)

    with pytest.raises(ValueError, match="训练标签必须同时包含 0 和 1"):
        model.fit(X, labels)

    assert model._model is None


def test_logistic_regression_uses_same_default_probability_scorecard(binary_data):
    X, y = binary_data
    model = LogisticRegression(
        calculate_stats=False,
        scorecard_params={"pdo": 20},
        max_iter=200,
        random_state=42,
    ).fit(X, y)

    assert model.scorecard_params == {"pdo": 20}
    assert model.scorecard_config_["pdo"] == 20
    assert model.scorecard_config_["base_score"] == 600
    assert model.base_odds_ == pytest.approx(np.mean(y == 1) / np.mean(y == 0))
    expected = model.scorecard_.predict_score(proba=model.predict_proba(X)[:, 1])
    np.testing.assert_array_equal(model.predict_score(X), expected)


@pytest.mark.parametrize(
    "model_class",
    [
        XGBoostRiskModel,
        LightGBMRiskModel,
        CatBoostRiskModel,
        NGBoostRiskModel,
        RandomForestRiskModel,
        ExtraTreesRiskModel,
        GradientBoostingRiskModel,
        LogisticRegression,
    ],
)
def test_every_risk_model_constructor_exposes_scorecard_params(model_class):
    """所有风险模型都应通过显式构造参数支持 sklearn clone 和调参透传。"""
    assert "scorecard_params" in inspect.signature(model_class.__init__).parameters


@pytest.mark.parametrize(
    "model_class, dependency, model_kwargs",
    [
        (XGBoostRiskModel, "xgboost", {"n_estimators": 3, "validation_fraction": 0}),
        (LightGBMRiskModel, "lightgbm", {"n_estimators": 3, "validation_fraction": 0}),
    ],
)
def test_installed_boosting_models_fit_default_scorecard(binary_data, model_class, dependency, model_kwargs):
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"未安装可选依赖 {dependency}")
    X, y = binary_data

    model = model_class(
        random_state=42,
        n_jobs=1,
        scorecard_params={"pdo": 25},
        **model_kwargs,
    ).fit(X, y)

    assert model.scorecard_config_["pdo"] == 25
    assert model.base_odds_ == pytest.approx(np.mean(y == 1) / np.mean(y == 0))
    np.testing.assert_array_equal(
        model.predict_score(X),
        model.scorecard_.predict_score(proba=model.predict_proba(X)[:, 1]),
    )


def test_tune_preserves_source_scorecard_params(binary_data):
    X, y = binary_data
    source = RandomForestRiskModel(
        n_estimators=5,
        random_state=42,
        scorecard_params={"pdo": 25},
    )

    tuned = source.tune(
        X,
        y,
        search_space={"max_depth": [2]},
        fixed_params={"n_estimators": 5},
        n_trials=1,
        cv=2,
        n_jobs=1,
        verbose=False,
    )

    assert tuned.scorecard_params == {"pdo": 25}
    assert tuned.scorecard_config_["pdo"] == 25
    assert tuned.scorecard_config_["base_score"] == 600
    assert tuned.base_odds_ == pytest.approx(np.mean(y == 1) / np.mean(y == 0))


def test_artifact_round_trip_preserves_fitted_scorecard(binary_data, tmp_path):
    X, y = binary_data
    model = RandomForestRiskModel(
        n_estimators=5,
        random_state=42,
        scorecard_params={"pdo": 25, "upper": 900},
    ).fit(X, y)
    expected = model.predict_score(X)

    path = model.save_artifact(tmp_path / "risk_model.joblib")
    restored = RandomForestRiskModel.load_artifact(path)

    assert restored.scorecard_params == {"pdo": 25, "upper": 900}
    assert restored.base_odds_ == pytest.approx(model.base_odds_)
    np.testing.assert_array_equal(restored.predict_score(X), expected)
