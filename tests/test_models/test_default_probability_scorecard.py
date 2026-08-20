import importlib.util
import inspect
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from hscredit.core.models import (
    CatBoost,
    ExtraTrees,
    GradientBoosting,
    LightGBM,
    LogisticRegression,
    NGBoost,
    RandomForest,
    XGBoost,
)
from hscredit.exceptions import NotFittedError


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

    model = RandomForest(n_estimators=5, random_state=42).fit(X, y)

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
    model = RandomForest(scorecard_params={"pdo": 20})

    assert model.scorecard_params == {"pdo": 20}
    assert model.scorecard_config_["pdo"] == 20
    assert model.scorecard_config_["base_score"] == 600
    assert model.scorecard_config_["lower"] == 0
    assert model.scorecard_config_["upper"] == 1000
    assert model.scorecard_config_["direction"] == "descending"


def test_set_params_scorecard_override_is_applied_when_fitting(binary_data):
    X, y = binary_data
    model = RandomForest(n_estimators=5, random_state=42)

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
        RandomForest(scorecard_params=scorecard_params)


def test_predict_score_uses_fitted_probability_scorecard(binary_data):
    X, y = binary_data
    model = RandomForest(n_estimators=5, random_state=42).fit(X, y)

    probability = model.predict_proba(X)[:, 1]
    expected = model.scorecard_.predict_score(proba=probability)
    scores = model.predict_score(X)

    np.testing.assert_array_equal(scores, expected)
    assert np.all((scores >= 0) & (scores <= 1000))
    assert np.all(scores == np.round(scores))
    low_probability_score = model.scorecard_.predict_score(proba=[0.05])[0]
    high_probability_score = model.scorecard_.predict_score(proba=[0.50])[0]
    assert low_probability_score > high_probability_score


def test_score_transformer_is_direct_shared_attribute_fitted_on_training_probability(binary_data):
    """数据驱动转换器不能只用单个坏样本率占位拟合。"""
    X, y = binary_data
    model = RandomForest(
        n_estimators=5,
        random_state=42,
        scorecard_params={"method": "quantile", "n_quantiles": 10},
    ).fit(X, y)
    probability = model.predict_proba(X)[:, 1]

    assert model.scorecard_.transformer_ is model.score_transformer_
    np.testing.assert_allclose(model.score_transformer_.train_proba_, probability)
    assert len(np.unique(model.score_transformer_.transformer_.quantile_values_)) > 1
    np.testing.assert_array_equal(model.predict_score(X), model.score_transformer_.predict(probability))


def test_clone_discards_fitted_transformer_but_copy_preserves_scores(binary_data):
    """clone 遵循 sklearn，copy/deepcopy 保留业务评分状态。"""
    X, y = binary_data
    model = RandomForest(n_estimators=5, random_state=42).fit(X, y)
    expected = model.predict_score(X)

    cloned = clone(model)
    shallow = copy.copy(model)
    deep = copy.deepcopy(model)

    assert not hasattr(cloned, "score_transformer_")
    np.testing.assert_array_equal(shallow.predict_score(X), expected)
    np.testing.assert_array_equal(deep.predict_score(X), expected)
    assert deep.score_transformer_ is not model.score_transformer_


def test_refit_replaces_score_transformer(binary_data):
    """重新拟合不能复用上一次训练分布。"""
    X, y = binary_data
    model = RandomForest(n_estimators=5, random_state=42).fit(X, y)
    first = model.score_transformer_

    model.fit(X[::-1], y[::-1])

    assert model.score_transformer_ is not first


@pytest.mark.parametrize("labels", [np.zeros(12, dtype=int), np.ones(12, dtype=int), np.tile([-1, 1], 6)])
def test_invalid_training_labels_fail_before_native_model_training(labels):
    X = np.arange(24, dtype=float).reshape(12, 2)
    model = RandomForest(n_estimators=5, random_state=42)

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
        XGBoost,
        LightGBM,
        CatBoost,
        NGBoost,
        RandomForest,
        ExtraTrees,
        GradientBoosting,
        LogisticRegression,
    ],
)
def test_every_risk_model_constructor_exposes_scorecard_params(model_class):
    """所有风险模型都应通过显式构造参数支持 sklearn clone 和调参透传。"""
    assert "scorecard_params" in inspect.signature(model_class.__init__).parameters


@pytest.mark.parametrize(
    "model_class, dependency, model_kwargs",
    [
        (XGBoost, "xgboost", {"n_estimators": 3, "validation_fraction": 0}),
        (LightGBM, "lightgbm", {"n_estimators": 3, "validation_fraction": 0}),
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
    source = RandomForest(
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
    model = RandomForest(
        n_estimators=5,
        random_state=42,
        scorecard_params={"pdo": 25, "upper": 900},
    ).fit(X, y)
    expected = model.predict_score(X)

    path = model.save_artifact(tmp_path / "risk_model.joblib")
    restored = RandomForest.load_artifact(path)

    assert restored.scorecard_params == {"pdo": 25, "upper": 900}
    assert restored.base_odds_ == pytest.approx(model.base_odds_)
    assert restored.scorecard_.transformer_ is restored.score_transformer_
    np.testing.assert_array_equal(restored.predict_score(X), expected)


def test_native_model_round_trip_restores_exact_score_transformer(binary_data, tmp_path):
    """原生模型导出若缺少转换器状态，加载后评分会漂移或不可用。"""
    X, y = binary_data
    X = pd.DataFrame(X, columns=[f"字段{i}" for i in range(X.shape[1])])
    model = RandomForest(
        n_estimators=5,
        random_state=42,
        scorecard_params={"method": "quantile", "n_quantiles": 10},
    ).fit(X, y)
    expected_probability = model.predict_proba(X)
    expected_score = model.predict_score(X)
    path = tmp_path / "forest.native"

    model.save_model(path)
    restored = RandomForest(n_estimators=5, random_state=42).load_model(path)

    assert (tmp_path / "forest.native.score_transformer.joblib").exists()
    assert restored.scorecard_.transformer_ is restored.score_transformer_
    assert restored.score_transformer_.method == "quantile"
    np.testing.assert_allclose(restored.predict_proba(X), expected_probability)
    np.testing.assert_array_equal(restored.predict_score(X), expected_score)


def test_lightgbm_native_round_trip_restores_wrapper_and_named_schema(binary_data, tmp_path):
    """LightGBM 原生恢复不能给只读 booster_ 赋值，也不能丢失字段契约。"""
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("未安装可选依赖 lightgbm")
    X, y = binary_data
    X = pd.DataFrame(X, columns=[f"字段{i}" for i in range(X.shape[1])])
    model = LightGBM(n_estimators=3, validation_fraction=0, random_state=42, verbose=-1).fit(X, y)
    expected_probability = model.predict_proba(X)
    expected_score = model.predict_score(X)
    path = tmp_path / "lightgbm.txt"

    model.save_model(path)
    restored = LightGBM(verbose=-1).load_model(path)

    assert restored.feature_names_in_ == list(X.columns)
    np.testing.assert_allclose(restored.predict_proba(X), expected_probability)
    np.testing.assert_array_equal(restored.predict_score(X), expected_score)


def test_native_model_without_sidecar_keeps_probability_but_not_score(binary_data, tmp_path):
    """缺失转换器制品时不能静默重建一个不同评分映射。"""
    X, y = binary_data
    model = RandomForest(n_estimators=5, random_state=42).fit(X, y)
    path = tmp_path / "forest.native"
    model.save_model(path)
    (tmp_path / "forest.native.score_transformer.joblib").unlink()

    restored = RandomForest(n_estimators=5, random_state=42).load_model(path)

    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    with pytest.raises(NotFittedError, match="评分"):
        restored.predict_score(X)
