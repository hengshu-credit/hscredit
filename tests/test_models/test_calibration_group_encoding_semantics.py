"""校准拟合权重、实体隔离与默认折外编码的语义验证。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from hscredit import (
    ProbabilityCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    BetaCalibrator,
    HistogramCalibrator,
    TargetEncoder,
)


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator, BetaCalibrator])
def test_calibration_frequency_weights_match_repeated_rows(cls):
    p = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
    y = np.array([0, 1, 0, 1, 0, 1])
    weights = np.array([1, 4, 2, 1, 3, 2])
    weighted = cls().fit(p, y, sample_weight=weights)
    repeated = cls().fit(np.repeat(p, weights), np.repeat(y, weights))
    np.testing.assert_allclose(weighted.calibrate(p), repeated.calibrate(p), atol=1e-7)


def test_histogram_and_wrapper_use_calibration_weights_and_reject_unused_parameters():
    p = np.array([0.1, 0.2, 0.7, 0.8])
    y = np.array([0, 1, 0, 1])
    weights = [1, 3, 3, 1]
    calibrated = HistogramCalibrator(n_bins=2, strategy="uniform").fit(p, y, sample_weight=weights)
    np.testing.assert_allclose(calibrated.calibrate(p), [0.75, 0.75, 0.25, 0.25])
    X = pd.DataFrame({"x": np.arange(40)})
    y = np.arange(40) % 2
    base = LogisticRegression().fit(X, y)
    plain = ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y)
    weighted = ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y, sample_weight=np.where(y, 10, 1))
    assert not np.allclose(plain.predict_proba(X), weighted.predict_proba(X))
    with pytest.raises(ValueError, match="权重|sample_weight"):
        ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y, sample_weight=[-1])
    with pytest.raises(TypeError, match="不接受"):
        ProbabilityCalibrator(model=base, calib_ratio=None).fit(X, y, misspelled_parameter=True)


def test_integer_cv_respects_groups_and_custom_split_cannot_leak_entities():
    pytest.importorskip("optuna")
    from hscredit.core.models.tuning import ModelTuner

    X = pd.DataFrame({"x": np.arange(100)})
    y = np.arange(100) % 2
    groups = np.repeat(np.arange(20), 5)
    tuner = ModelTuner(LogisticRegression(), search_space={"C": [1.0]}, cv=3, n_jobs=1)
    splits = tuner._validated_splits(X, y, groups)
    assert all(not set(groups[a]).intersection(groups[b]) for a, b in splits)
    tuner.cv = KFold(3, shuffle=True, random_state=42)
    with pytest.raises(ValueError, match="groups"):
        tuner._validated_splits(X, y, groups)


def test_default_target_encoding_excludes_own_label_and_is_repeatable():
    X = pd.DataFrame({"id": [f"u{i}" for i in range(100)]})
    y = np.arange(100) % 2
    encoder = TargetEncoder(n_jobs=1)
    first = encoder.fit_transform(X, y)
    second = TargetEncoder(n_jobs=1).fit_transform(X, y)
    pd.testing.assert_frame_equal(first, second)
    assert roc_auc_score(y, first.id) < 0.65
    assert roc_auc_score(y, TargetEncoder(training_mode="in_sample", n_jobs=1).fit_transform(X, y).id) == 1


def test_internal_calibration_split_uses_correct_weights_on_both_partitions():
    from scipy.special import logit
    from sklearn.model_selection import train_test_split

    X = pd.DataFrame({"x": np.linspace(-3, 3, 80)})
    y = np.arange(80) % 2
    weights = np.arange(1, 81, dtype=float)
    train, calibration = train_test_split(np.arange(80), test_size=0.25, random_state=42, stratify=y)
    base = LogisticRegression().fit(X.iloc[train], y[train], sample_weight=weights[train])
    p = base.predict_proba(X.iloc[calibration])[:, 1]
    mapping = LogisticRegression(C=1.0, max_iter=1000).fit(
        logit(p).reshape(-1, 1), y[calibration], sample_weight=weights[calibration]
    )
    expected = mapping.predict_proba(logit(base.predict_proba(X)[:, 1]).reshape(-1, 1))
    actual = ProbabilityCalibrator(model=LogisticRegression(), calib_ratio=0.25, random_state=42).fit(
        X, y, sample_weight=weights
    )
    np.testing.assert_allclose(actual.predict_proba(X), expected, atol=1e-8)
