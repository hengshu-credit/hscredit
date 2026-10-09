"""真实放款数据验证 LightGBM 自定义目标的无告警概率预测及原生输出。"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.model_selection import cross_val_score, train_test_split

from hscredit.core.models.losses import FocalLoss


@pytest.fixture(scope="module")
def loans():
    """按仓库约定使用三个建模字段和 FPD 标签。"""
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    frame = pd.read_excel(path, usecols=features + ["FPD"], nrows=500)
    return train_test_split(
        frame[features].astype(float).fillna(0),
        frame["FPD"].to_numpy(),
        test_size=0.3,
        stratify=frame["FPD"],
        random_state=24,
    )


def _model(objective):
    pytest.importorskip("lightgbm")
    from hscredit import LightGBM

    return LightGBM(
        objective=objective, n_estimators=6, num_leaves=7, n_jobs=1, random_state=24,
        validation_fraction=0,
    )


@pytest.mark.parametrize("use_adapter", [False, True])
def test_custom_loss_fit_predict_and_cross_validation_do_not_warn(loans, use_adapter):
    X, X_valid, y, y_valid = loans
    loss = FocalLoss(alpha=0.7)
    objective = loss.to_lightgbm(api="sklearn") if use_adapter else loss
    model = _model(objective)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, eval_set=[(X_valid, y_valid)])
        probability = model.predict_proba(X_valid)
        array_probability = model.predict_proba(X_valid.to_numpy())
        labels = model.predict(X_valid)
        scores = cross_val_score(
            model, X, y, scoring=loss.metric().to_scorer(), cv=2, error_score="raise"
        )
    assert not caught, [str(item.message) for item in caught]
    raw_margin = model._model.predict_proba(X_valid, raw_score=True)
    np.testing.assert_allclose(probability[:, 1], expit(raw_margin))
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    np.testing.assert_allclose(array_probability, probability)
    np.testing.assert_array_equal(labels, model.classes_[np.argmax(probability, axis=1)])
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("objective", ["binary", FocalLoss(alpha=0.7)])
@pytest.mark.parametrize("flag", ["raw_score", "pred_leaf", "pred_contrib"])
def test_original_prediction_flags_and_iteration_parameters_are_preserved(loans, objective, flag):
    X, X_valid, y, _ = loans
    model = _model(objective).fit(X, y)
    options = {flag: True, "start_iteration": 1, "num_iteration": 2}
    expected = model._model.predict_proba(X_valid, **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.predict_proba(X_valid, **options)
    assert not caught, [str(item.message) for item in caught]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("objective", ["binary", FocalLoss(alpha=0.7)])
def test_probability_prediction_preserves_iteration_parameters_and_explicit_false_flags(loans, objective):
    X, X_valid, y, _ = loans
    model = _model(objective).fit(X, y)
    options = {"start_iteration": 1, "num_iteration": 2}
    margin = model._model.predict_proba(X_valid, raw_score=True, **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = model.predict_proba(
            X_valid, raw_score=False, pred_leaf=False, pred_contrib=False, **options
        )
    assert not caught, [str(item.message) for item in caught]
    np.testing.assert_allclose(probability[:, 1], expit(margin))


def test_loaded_custom_objective_preserves_probability_and_margin_outputs(loans, tmp_path):
    X, X_valid, y, _ = loans
    model = _model(FocalLoss(alpha=0.7)).fit(X, y)
    expected = model.predict_proba(X_valid)
    path = str(tmp_path / "custom_loss.txt")
    model.save_model(path)
    loaded = _model("binary").load_model(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = loaded.predict_proba(X_valid)
        margin = loaded.predict_proba(X_valid, raw_score=True)
    assert not caught, [str(item.message) for item in caught]
    np.testing.assert_allclose(probability, expected)
    np.testing.assert_allclose(probability[:, 1], expit(margin))


def test_extreme_custom_margins_do_not_overflow(loans, monkeypatch):
    X, X_valid, y, _ = loans
    model = _model(FocalLoss()).fit(X, y)
    monkeypatch.setattr(model._model, "predict_proba", lambda X, **kwargs: np.array([-1000.0, 1000.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = model.predict_proba(X_valid.iloc[:2])
    assert not caught, [str(item.message) for item in caught]
    np.testing.assert_array_equal(probability, [[1.0, 0.0], [0.0, 1.0]])


def test_unrelated_native_warning_is_not_suppressed(loans, monkeypatch):
    X, X_valid, y, _ = loans
    model = _model(FocalLoss()).fit(X, y)
    original_predict = model._model.predict_proba

    def predict_with_warning(*args, **kwargs):
        warnings.warn("真实预测警告", UserWarning)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model._model, "predict_proba", predict_with_warning)
    with pytest.warns(UserWarning, match="真实预测警告"):
        model.predict_proba(X_valid)
