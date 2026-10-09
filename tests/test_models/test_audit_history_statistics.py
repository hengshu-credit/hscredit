"""训练记录有界保留及加权 LR 统计的审计回归。"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from hscredit.core.models import RandomForest, LogisticRegression, ScoreCard, RoundScoreCard
from hscredit.core.models.calibration.model import ProbabilityCalibrator
from hscredit.core.models.scorecard.model_scorecard import ProbabilityScoreCard
from hscredit.core.models._lifecycle import _fit_parameter_snapshot


@pytest.fixture
def data():
    rng = np.random.RandomState(12)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = (X.a + rng.normal(size=100) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("cls", [RandomForest, LogisticRegression, ScoreCard, RoundScoreCard, ProbabilityScoreCard, ProbabilityCalibrator])
def test_history_configuration_survives_clone(cls):
    estimator = cls(history_policy="diagnostics", max_history=3)
    copied = clone(estimator)
    assert copied.history_policy == "diagnostics"
    assert copied.max_history == 3
    assert copied.get_params()["max_history"] == 3
    assert clone(copied.set_params(max_history=2)).max_history == 2


def _assert_no_arrays(value):
    assert not isinstance(value, (np.ndarray, pd.DataFrame, pd.Series))
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_arrays(item)


def test_summary_history_is_bounded_without_eval_data(data):
    X, y = data
    model = RandomForest(n_estimators=2, n_jobs=1, max_history=3)
    for _ in range(10):
        model.fit(X, y, eval_set=[(X, y)], sample_weight=np.ones(len(y)))
    assert len(model.training_history_) == 3
    assert model.fit_revision_ == 10
    _assert_no_arrays(model.training_history_)
    _assert_no_arrays(model.scorecard_.training_history_)
    assert model.training_summary_["训练参数"]["eval_set"][0][0]["形状"] == [100, 3]
    assert "history_policy" not in model.get_native_params()


def test_full_history_is_explicit_and_can_keep_copy(data):
    X, y = data
    model = RandomForest(n_estimators=2, n_jobs=1, history_policy="full", max_history=1).fit(X, y, eval_set=[(X, y)])
    stored = model.training_summary_["训练参数"]["eval_set"][0][0]
    pd.testing.assert_frame_equal(stored, X)
    assert stored is not X


def test_summary_does_not_copy_callback_owned_state():
    class Callback:
        def __deepcopy__(self, memo):
            raise RuntimeError("不能复制")
    assert "对象类型" in _fit_parameter_snapshot(Callback())


def test_frequency_weight_standard_errors_equal_repeated_rows(data):
    X, y = data
    weights = np.arange(len(X)) % 3 + 1
    kwargs = dict(penalty=None, max_iter=1000, tol=1e-10, statistics_level="coef", positive_woe_coef=False)
    weighted = LogisticRegression(**kwargs).fit(X, y, sample_weight=weights)
    repeated = LogisticRegression(**kwargs).fit(X.iloc[np.repeat(np.arange(len(X)), weights)], y.iloc[np.repeat(np.arange(len(X)), weights)])
    np.testing.assert_allclose(weighted.coef_, repeated.coef_, atol=1e-7)
    np.testing.assert_allclose(weighted.std_err_coef_, repeated.std_err_coef_, atol=1e-7)
    np.testing.assert_allclose(weighted.std_err_intercept_, repeated.std_err_intercept_, atol=1e-7)
    assert weighted.statistics_status_["状态"] == "完成"
    assert np.isnan(weighted.vif_).all()


def test_sparse_statistics_budget_does_not_break_prediction(data):
    X, y = data
    model = LogisticRegression(statistics_level="full", max_dense_bytes=1)
    with pytest.warns(UserWarning, match="超过"):
        model.fit(sparse.csr_matrix(X), y)
    assert model.statistics_status_["状态"] == "超出预算"
    assert np.isfinite(model.predict_proba(sparse.csr_matrix(X))).all()
    assert not hasattr(model, "cov_matrix_")


def test_cost_and_regularized_statistics_are_explicit(data):
    X, y = data
    cost = LogisticRegression(weight_type="cost").fit(X, y, sample_weight=np.ones(len(y)))
    assert cost.statistics_status_["状态"] == "不适用"
    assert not hasattr(cost, "std_err_coef_")
    regularized = LogisticRegression(statistics_level="coef").fit(X, y)
    assert regularized.summary().attrs["统计状态"]["状态"] == "近似"
