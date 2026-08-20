"""Boosting 包装器训练与可选依赖契约回归测试。"""

import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError


def _binary_data():
    return make_classification(n_samples=100, n_features=4, random_state=23)


class _ProbabilityFake:
    """补齐真实分类器必备的二维概率接口。"""

    def predict_proba(self, X):
        positive = np.full(len(X), 0.5, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def test_importing_hscredit_does_not_import_shap_eagerly():
    code = "import sys, hscredit; assert 'shap' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_xgboost_without_early_stopping_uses_all_rows_and_forwards_fit_params(monkeypatch):
    from hscredit.core.models.boosting import xgboost_model as module

    class FakeXGBClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params
            self.best_iteration = None
            self.best_score = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.xgb, "XGBClassifier", FakeXGBClassifier)
    X, y = _binary_data()

    base_margin = np.zeros(len(y))
    module.XGBoost(validation_fraction=0.2).fit(X, y, base_margin=base_margin)

    fitted = FakeXGBClassifier.instance
    assert fitted.n_rows == len(y)
    assert "eval_set" not in fitted.fit_kwargs
    np.testing.assert_array_equal(fitted.fit_kwargs["base_margin"], base_margin)


def test_xgboost_auto_split_slices_row_fit_params_and_validation_weights(monkeypatch):
    from hscredit.core.models.boosting import xgboost_model as module

    class FakeXGBClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.best_iteration = None
            self.best_score = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.xgb, "XGBClassifier", FakeXGBClassifier)
    X, y = _binary_data()
    model = module.XGBoost(validation_fraction=0.2, early_stopping_rounds=2, random_state=23)
    model.fit(X, y, sample_weight=np.arange(1, 101), base_margin=np.arange(100))

    fitted = FakeXGBClassifier.instance
    assert fitted.n_rows == 80
    assert len(fitted.fit_kwargs["base_margin"]) == 80
    assert len(fitted.fit_kwargs["base_margin_eval_set"][0]) == 20
    assert len(fitted.fit_kwargs["sample_weight"]) == 80
    assert len(fitted.fit_kwargs["sample_weight_eval_set"][0]) == 20


def test_xgboost_prediction_reorders_training_fields_and_ignores_extra_fields(monkeypatch):
    from hscredit.core.models.boosting import xgboost_model as module

    class FakeXGBClassifier(_ProbabilityFake):
        def __init__(self, **params):
            self.best_iteration = None
            self.best_score = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            self.prediction_input = np.asarray(X)
            positive = np.full(len(X), 0.5)
            return np.column_stack([1.0 - positive, positive])

    monkeypatch.setattr(module.xgb, "XGBClassifier", FakeXGBClassifier)
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    model = module.XGBoost(validation_fraction=0).fit(train, np.array([0, 1]))

    prediction = pd.DataFrame({"b": [30.0], "extra": [99.0], "a": [10.0]})
    model.predict_proba(prediction)
    np.testing.assert_array_equal(model._model.prediction_input, [[10.0, 30.0]])

    with pytest.raises(ValueError, match="缺少训练字段.*b"):
        model.predict_proba(pd.DataFrame({"a": [10.0], "extra": [99.0]}))


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="XGBoost未安装")
def test_xgboost_ks_metric_is_actually_evaluated():
    from hscredit.core.models import XGBoost

    X, y = _binary_data()
    model = XGBoost(
        n_estimators=2,
        eval_metric="ks",
        validation_fraction=0,
        random_state=23,
    ).fit(X, y, eval_set=[(X, y)])

    assert "ks" in model.evals_result_["validation_0"]


def test_xgboost_predict_rejects_unfitted_model():
    from hscredit.core.models import XGBoost

    with pytest.raises(NotFittedError, match="尚未拟合"):
        XGBoost(n_estimators=1).predict_proba(np.zeros((2, 4)))


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="XGBoost未安装")
def test_xgboost_json_round_trip_preserves_probability_scorecard(tmp_path):
    from hscredit.core.models import XGBoost

    X, y = _binary_data()
    X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = XGBoost(
        n_estimators=2,
        validation_fraction=0,
        random_state=23,
        scorecard_params={"pdo": 35, "base_score": 650},
    ).fit(X, y)
    expected = model.predict_score(X.iloc[:5])

    path = tmp_path / "xgb.json"
    model.save(path)
    restored = XGBoost.load(path)

    assert (tmp_path / "xgb.native.score_transformer.joblib").exists()
    assert restored.scorecard_.transformer_ is restored.score_transformer_
    np.testing.assert_array_equal(restored.predict_score(X.iloc[:5]), expected)


def test_lightgbm_without_early_stopping_uses_all_rows_and_forwards_fit_params(monkeypatch):
    try:
        from hscredit.core.models.boosting import lightgbm_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入LightGBM: {exc}")

    class FakeLGBMClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params
            self.best_iteration_ = None
            self.best_score_ = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    X, y = _binary_data()

    module.LightGBM(validation_fraction=0.2).fit(X, y, init_model="sentinel")

    fitted = FakeLGBMClassifier.instance
    assert fitted.n_rows == len(y)
    assert "eval_set" not in fitted.fit_kwargs
    assert fitted.fit_kwargs["init_model"] == "sentinel"


def test_lightgbm_score_transformer_uses_wrapper_margin_normalization(monkeypatch):
    """自定义目标的一维 margin 必须先经 wrapper sigmoid 再拟合评分转换器。"""
    try:
        from hscredit.core.models.boosting import lightgbm_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入LightGBM: {exc}")

    class FakeLGBMClassifier:
        def __init__(self, **params):
            self.best_iteration_ = None
            self.best_score_ = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(module, "LIGHTGBM_AVAILABLE", True)
    monkeypatch.setattr(module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    X, y = _binary_data()

    model = module.LightGBM(validation_fraction=0).fit(X, y)

    np.testing.assert_allclose(model.score_transformer_.train_proba_, 0.5)


def test_lightgbm_ks_metric_is_forwarded_as_callable(monkeypatch):
    try:
        from hscredit.core.models.boosting import lightgbm_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入LightGBM: {exc}")

    class FakeLGBMClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params
            self.best_iteration_ = None
            self.best_score_ = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    X, y = _binary_data()
    module.LightGBM(n_estimators=2, eval_metric=["auc", "ks"], validation_fraction=0).fit(
        X, y, eval_set=[(X, y)]
    )

    fitted = FakeLGBMClassifier.instance
    assert fitted.params["metric"] == ["auc"]
    assert callable(fitted.fit_kwargs["eval_metric"])


def test_lightgbm_merges_user_callbacks_with_internal_callbacks(monkeypatch):
    try:
        from hscredit.core.models.boosting import lightgbm_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入LightGBM: {exc}")

    class FakeLGBMClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.best_iteration_ = None
            self.best_score_ = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    X, y = _binary_data()
    user_callback = object()
    module.LightGBM(
        n_estimators=2,
        early_stopping_rounds=2,
        validation_fraction=0,
    ).fit(X, y, eval_set=[(X, y)], callbacks=[user_callback])

    callbacks = FakeLGBMClassifier.instance.fit_kwargs["callbacks"]
    assert user_callback in callbacks
    assert len(callbacks) >= 2


def test_lightgbm_auto_split_slices_init_score_and_validation_weights(monkeypatch):
    try:
        from hscredit.core.models.boosting import lightgbm_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入LightGBM: {exc}")

    class FakeLGBMClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.best_iteration_ = None
            self.best_score_ = None
            self.evals_result_ = {}

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module.lgb, "LGBMClassifier", FakeLGBMClassifier)
    X, y = _binary_data()
    module.LightGBM(validation_fraction=0.2, early_stopping_rounds=2, random_state=23).fit(
        X, y, sample_weight=np.arange(1, 101), init_score=np.arange(100)
    )

    fitted = FakeLGBMClassifier.instance
    assert fitted.n_rows == 80
    assert len(fitted.fit_kwargs["init_score"]) == 80
    assert len(fitted.fit_kwargs["eval_init_score"][0]) == 20
    assert len(fitted.fit_kwargs["sample_weight"]) == 80
    assert len(fitted.fit_kwargs["eval_sample_weight"][0]) == 20


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="LightGBM未安装")
def test_lightgbm_ks_metric_is_actually_evaluated():
    from hscredit.core.models import LightGBM

    X, y = _binary_data()
    model = LightGBM(
        n_estimators=2,
        eval_metric="ks",
        validation_fraction=0,
        random_state=23,
    ).fit(X, y, eval_set=[(X, y)])

    assert any("ks" in metrics for metrics in model.evals_result_.values())


def test_catboost_without_early_stopping_uses_all_rows_and_forwards_fit_params(monkeypatch):
    from hscredit.core.models.boosting import catboost_model as module

    class FakeCatBoostClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

        def get_best_iteration(self):
            return None

        def get_best_score(self):
            return None

        def get_evals_result(self):
            return {}

    monkeypatch.setattr(module.cb, "CatBoostClassifier", FakeCatBoostClassifier)
    X, y = _binary_data()

    module.CatBoost(validation_fraction=0.2).fit(X, y, baseline=np.zeros(len(y)))

    fitted = FakeCatBoostClassifier.instance
    assert fitted.n_rows == len(y)
    assert "eval_set" not in fitted.fit_kwargs
    np.testing.assert_array_equal(fitted.fit_kwargs["baseline"], np.zeros(len(y)))


def test_catboost_ks_metric_is_configured_as_custom_metric(monkeypatch):
    from hscredit.core.models.boosting import catboost_model as module

    class FakeCatBoostClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params

        def fit(self, X, y, **kwargs):
            return self

        def get_best_iteration(self):
            return None

        def get_best_score(self):
            return None

        def get_evals_result(self):
            return {}

    monkeypatch.setattr(module.cb, "CatBoostClassifier", FakeCatBoostClassifier)
    X, y = _binary_data()
    module.CatBoost(iterations=2, eval_metric=["auc", "ks"], validation_fraction=0).fit(
        X, y, eval_set=[(X, y)]
    )

    params = FakeCatBoostClassifier.instance.params
    assert params["eval_metric"].__class__.__name__ == "CatBoostKSMetric"
    assert params["custom_metric"] == ["AUC"]


def test_catboost_auto_split_slices_baseline_and_uses_weighted_eval_pool(monkeypatch):
    from hscredit.core.models.boosting import catboost_model as module

    class FakePool:
        def __init__(self, X, y, **kwargs):
            self.X = X
            self.y = y
            self.kwargs = kwargs

    class FakeCatBoostClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

        def get_best_iteration(self):
            return None

        def get_best_score(self):
            return {}

        def get_evals_result(self):
            return {}

    monkeypatch.setattr(module.cb, "Pool", FakePool)
    monkeypatch.setattr(module.cb, "CatBoostClassifier", FakeCatBoostClassifier)
    X, y = _binary_data()
    module.CatBoost(validation_fraction=0.2, early_stopping_rounds=2, random_state=23).fit(
        X, y, sample_weight=np.arange(1, 101), baseline=np.arange(100)
    )

    fitted = FakeCatBoostClassifier.instance
    assert fitted.n_rows == 80
    assert len(fitted.fit_kwargs["baseline"]) == 80
    pool = fitted.fit_kwargs["eval_set"]
    assert len(pool.y) == 20
    assert len(pool.kwargs["baseline"]) == 20
    assert len(pool.kwargs["weight"]) == 20


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="CatBoost未安装")
def test_catboost_ks_metric_is_actually_evaluated():
    from hscredit.core.models import CatBoost

    X, y = _binary_data()
    model = CatBoost(
        iterations=2,
        eval_metric="ks",
        validation_fraction=0,
        random_state=23,
    ).fit(X, y, eval_set=[(X, y)])

    assert any("ks" in metrics for metrics in model.evals_result_.values())


def test_ngboost_forwards_fit_params(monkeypatch):
    try:
        from hscredit.core.models.boosting import ngboost_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入NGBoost: {exc}")

    class FakeNGBClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.params = params
            self.best_val_loss_itr = None

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module, "NGBClassifier", FakeNGBClassifier)
    X, y = _binary_data()
    sentinel = lambda *args: 0.0
    module.NGBoost(validation_fraction=0).fit(X, y, train_loss_monitor=sentinel)

    fitted = FakeNGBClassifier.instance
    assert fitted.n_rows == len(y)
    assert fitted.fit_kwargs["train_loss_monitor"] is sentinel


def test_ngboost_auto_split_preserves_validation_weights(monkeypatch):
    try:
        from hscredit.core.models.boosting import ngboost_model as module
    except Exception as exc:
        pytest.skip(f"当前环境无法导入NGBoost: {exc}")

    class FakeNGBClassifier(_ProbabilityFake):
        instance = None

        def __init__(self, **params):
            type(self).instance = self
            self.best_val_loss_itr = None

        def fit(self, X, y, **kwargs):
            self.n_rows = len(y)
            self.fit_kwargs = kwargs
            return self

    monkeypatch.setattr(module, "NGBClassifier", FakeNGBClassifier)
    X, y = _binary_data()
    module.NGBoost(validation_fraction=0.2, early_stopping_rounds=2, random_state=23).fit(
        X, y, sample_weight=np.arange(1, 101)
    )

    fitted = FakeNGBClassifier.instance
    assert fitted.n_rows == 80
    assert len(fitted.fit_kwargs["sample_weight"]) == 80
    assert len(fitted.fit_kwargs["val_sample_weight"]) == 20
