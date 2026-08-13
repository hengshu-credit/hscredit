"""概率校准数学、标签和数据划分契约回归测试。"""

import numpy as np
import pytest
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hscredit.core.models.evaluation.calibration import (
    BetaCalibrator,
    CalibratedModel,
    PlattCalibrator,
    ProbabilityCalibrator,
)


def test_calibration_metrics_include_zero_probability_in_first_bin():
    metrics = PlattCalibrator().compute_calibration_metrics(np.array([1]), np.array([0.0]))
    assert metrics["expected_calibration_error"] == pytest.approx(1.0)
    assert metrics["max_calibration_error"] == pytest.approx(1.0)


def test_calibration_metrics_honor_quantile_strategy():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.7, 0.95])

    uniform = PlattCalibrator(n_bins=3, strategy="uniform").compute_calibration_metrics(y_true, y_prob)
    quantile = PlattCalibrator(n_bins=3, strategy="quantile").compute_calibration_metrics(y_true, y_prob)

    assert uniform["expected_calibration_error"] != pytest.approx(quantile["expected_calibration_error"])


def test_platt_calibrator_operates_on_log_odds():
    y_prob = np.array([0.03, 0.1, 0.3, 0.6, 0.8, 0.97])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator().fit(y_prob, y_true)

    expected = calibrator.lr_.predict_proba(logit(y_prob).reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(calibrator.calibrate(y_prob), expected)


def test_beta_calibrator_uses_log_probability_features_and_stays_finite():
    y_prob = np.array([0.01, 0.05, 0.2, 0.55, 0.8, 0.98])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    calibrator = BetaCalibrator().fit(y_prob, y_true)

    calibrated = calibrator.calibrate(np.array([0.0, 0.2, 0.8, 1.0]))

    assert hasattr(calibrator, "lr_")
    assert np.isfinite(calibrated).all()
    assert np.all(np.diff(calibrated) >= 0)


@pytest.mark.parametrize(
    "labels",
    [np.array([-1, 1]), np.array(["正常", "逾期"])],
)
def test_probability_calibrator_preserves_binary_label_domain(labels):
    X, y01 = make_classification(n_samples=120, n_features=4, random_state=31)
    y = labels[y01]
    model = LogisticRegression(max_iter=300).fit(X, y)
    calibrator = ProbabilityCalibrator(model=model, method="platt", calib_ratio=None).fit(X, y)

    assert set(np.unique(calibrator.predict(X))).issubset(set(labels))
    assert calibrator.classes_.tolist() == model.classes_.tolist()


class _TrackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X, y):
        self.fit_size_ = len(y)
        self.classes_ = np.unique(y)
        self.rate_ = float(np.mean(np.asarray(y) == self.classes_[1]))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), np.clip(self.rate_ + self.offset, 0.01, 0.99))
        return np.column_stack([1.0 - positive, positive])


def test_automatic_calibration_split_refits_a_clone_on_training_partition():
    X, y = make_classification(n_samples=100, n_features=4, random_state=37)
    original = _TrackingClassifier().fit(X, y)

    calibrator = ProbabilityCalibrator(
        model=original,
        calib_ratio=0.2,
        random_state=37,
    ).fit(X, y)

    assert calibrator.model_ is not original
    assert calibrator.model_.fit_size_ == 80
    assert original.fit_size_ == 100


def test_calibrated_model_uses_its_explicit_base_model_for_predictions():
    X, y = make_classification(n_samples=100, n_features=4, random_state=41)
    calibration_model = LogisticRegression(max_iter=300).fit(X, y)
    serving_model = LogisticRegression(max_iter=300).fit(X, 1 - y)
    calibrator = ProbabilityCalibrator(model=calibration_model, calib_ratio=None).fit(X, y)
    wrapped = CalibratedModel(serving_model, calibrator)

    serving_positive = serving_model.predict_proba(X)[:, 1]
    expected = calibrator.calibrator_.calibrate(serving_positive)

    np.testing.assert_allclose(wrapped.predict_proba(X)[:, 1], expected)


def test_calibrated_model_evaluate_supports_string_labels():
    X, y01 = make_classification(n_samples=100, n_features=4, random_state=43)
    labels = np.array(["正常", "逾期"])
    y = labels[y01]
    model = LogisticRegression(max_iter=300).fit(X, y)
    calibrator = ProbabilityCalibrator(model=model, calib_ratio=None).fit(X, y)

    metrics = CalibratedModel(model, calibrator).evaluate(X, y)

    assert set(metrics) == {"AUC", "KS", "Brier"}
    assert np.isfinite(list(metrics.values())).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [({"n_bins": 0}, "n_bins"), ({"strategy": "bad"}, "strategy")],
)
def test_calibrator_rejects_invalid_binning_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PlattCalibrator(**kwargs)


@pytest.mark.parametrize("factory", [PlattCalibrator, BetaCalibrator])
@pytest.mark.parametrize(
    "probabilities",
    [np.array([0.2, np.nan]), np.array([-0.1, 0.8]), np.array([0.2, 1.1])],
)
def test_calibrator_rejects_invalid_probabilities(factory, probabilities):
    with pytest.raises(ValueError, match="概率"):
        factory().fit(probabilities, np.array([0, 1]))
