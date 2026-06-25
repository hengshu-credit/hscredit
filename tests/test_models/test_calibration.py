"""概率校准统一接口测试."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hscredit.core.models.evaluation import (
    CalibratedModel,
    PlattCalibrator,
    ProbabilityCalibrator,
)


def _fitted_model_and_data():
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"特征{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="目标")
    model = LogisticRegression(max_iter=500).fit(X.iloc[:300], y.iloc[:300])
    return model, X.iloc[300:].copy(), y.iloc[300:].copy()


def test_base_calibrator_uses_sklearn_probability_first_order():
    y_prob = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator().fit(y_prob, y_true)

    transformed = calibrator.transform(y_prob)
    proba = calibrator.predict_proba(y_prob)
    assert transformed.shape == (6,)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], transformed)


def test_probability_calibrator_canonical_interface_and_report():
    model, X, y = _fitted_model_and_data()
    calibrator = ProbabilityCalibrator(
        method="platt",
        model=model,
        calib_ratio=None,
    ).fit(X, y)

    proba = calibrator.predict_proba(X)
    report = calibrator.report(X, y)

    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert report.columns.tolist() == ["指标", "校准前", "校准后", "改善值", "改善率"]
    assert report["指标"].tolist() == [
        "Brier分数",
        "期望校准误差(ECE)",
        "最大校准误差(MCE)",
    ]
    assert np.isfinite(report[["校准前", "校准后", "改善值"]].to_numpy()).all()


def test_probability_calibrator_keeps_legacy_fit_call_compatible():
    model, X, y = _fitted_model_and_data()
    calibrator = ProbabilityCalibrator(method="platt", calib_ratio=None)
    calibrator.fit(model, X, y)
    assert calibrator.predict_proba(X).shape == (len(X), 2)


def test_calibrated_model_matches_sklearn_probability_contract():
    model, X, y = _fitted_model_and_data()
    calibrator = ProbabilityCalibrator(
        model=model,
        calib_ratio=None,
    ).fit(X, y)
    calibrated_model = CalibratedModel(model, calibrator)

    proba = calibrated_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert calibrated_model.predict_score(X).shape == (len(X),)
    assert set(calibrated_model.evaluate(X, y)) == {"AUC", "KS", "Brier"}


def test_probability_calibrator_clone_preserves_parameters():
    calibrator = ProbabilityCalibrator(
        method="isotonic",
        calib_ratio=None,
        calibrator_params={"out_of_bounds": "clip"},
    )
    cloned = clone(calibrator)
    assert cloned.method == "isotonic"
    assert cloned.calib_ratio is None
    assert cloned.calibrator_params == {"out_of_bounds": "clip"}


def test_probability_calibrator_artifact_roundtrip(tmp_path):
    model, X, y = _fitted_model_and_data()
    calibrator = ProbabilityCalibrator(
        model=model,
        calib_ratio=None,
    ).fit(X, y)
    expected = calibrator.predict_proba(X)

    path = calibrator.save_artifact(tmp_path / "calibrator.joblib")
    restored = ProbabilityCalibrator.load_artifact(path)
    np.testing.assert_allclose(restored.predict_proba(X), expected)
