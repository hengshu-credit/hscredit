"""原生回归器与 Pipeline 在不同 sklearn 版本下保持回归调参语义。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit import ModelTuner


@pytest.mark.parametrize("pipeline", [False, True])
def test_native_regressor_tunes_signed_continuous_labels(pipeline):
    X = pd.DataFrame({"特征": np.linspace(-4, 4, 40)})
    y = 321.5 * X["特征"] + 27.25
    model = Pipeline([("标准化", StandardScaler()), ("回归", LinearRegression())]) if pipeline else LinearRegression()
    parameter = "回归__fit_intercept" if pipeline else "fit_intercept"
    tuner = ModelTuner(model, search_space={parameter: [True]}, metric="rmse", cv=None, n_jobs=1, random_state=42)
    tuner.fit(X.iloc[:30], y.iloc[:30], eval_set=(X.iloc[30:], y.iloc[30:]), n_trials=1, show_progress_bar=False)
    predicted = tuner.get_best_model().predict(X.iloc[30:])
    np.testing.assert_allclose(predicted, y.iloc[30:], rtol=1e-10, atol=1e-9)
    assert tuner.best_score_ < 1e-8


@pytest.mark.parametrize("as_class", [False, True])
def test_ngboost_tags_preserve_continuous_prediction(as_class):
    ngboost = pytest.importorskip("ngboost")
    options = {"n_estimators": 3, "natural_gradient": False, "verbose": False, "random_state": 42}
    X = pd.DataFrame({"特征": np.linspace(-3, 3, 40)})
    y = 250 * np.sin(X["特征"]) + 50
    native = ngboost.NGBRegressor(**options)
    tuner = ModelTuner(ngboost.NGBRegressor if as_class else native, model_params=options if as_class else None, search_space={"learning_rate": [0.1]}, metric="rmse", cv=None, n_jobs=1, random_state=42)
    tuner.fit(X.iloc[:30], y.iloc[:30], eval_set=(X.iloc[30:], y.iloc[30:]), n_trials=1, show_progress_bar=False)
    validation = tuner.get_validation_predictions()
    np.testing.assert_allclose(validation["真实标签"], y.iloc[30:])
    expected_rmse = np.sqrt(np.mean((validation["预测值"].to_numpy() - y.iloc[30:].to_numpy()) ** 2))
    assert tuner.best_score_ == pytest.approx(expected_rmse)
    # 连续值保留负值，不被分类概率协议裁剪到 [0, 1]。
    assert (tuner.get_best_model().predict(X.iloc[30:]) < 0).all()
