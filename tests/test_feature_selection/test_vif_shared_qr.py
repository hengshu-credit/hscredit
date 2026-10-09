"""VIF共享QR对逐列带截距OLS的数值与筛选路径差分。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.selectors import VIFSelector
from hscredit.core.selectors.vif_selector import _compute_vif_single


@pytest.mark.parametrize("case", ["ordinary", "offset", "constant", "missing", "singular", "tiny", "huge_offset", "wide", "one", "two_rows"])
def test_shared_vif_matches_column_ols(case):
    rng = np.random.default_rng(51)
    X = rng.normal(size=(120, 8))
    if case == "offset":
        X += np.arange(8) * 1000
    elif case == "constant":
        X[:, 0] = 3.0
    elif case == "missing":
        X[::4, 1] = np.nan
    elif case == "singular":
        X[:, 1] = X[:, 0] + 1000
    elif case == "tiny":
        X[:, 1] *= 1e-12
    elif case == "huge_offset":
        X += 1e12
    elif case == "wide":
        X = X[:5]
    elif case == "one":
        X = X[:, :1]
    elif case == "two_rows":
        X = X[:2]
    frame = pd.DataFrame(X)
    selector = VIFSelector(n_jobs=1)
    actual = selector._compute_vif_all(frame)
    filled = frame.fillna(selector.missing).to_numpy()
    expected = [_compute_vif_single(filled, i) for i in range(filled.shape[1])]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_regular_matrix_uses_one_shared_factorization_not_column_regression(monkeypatch):
    import hscredit.core.selectors.vif_selector as module
    rng = np.random.default_rng(73)
    frame = pd.DataFrame(rng.normal(size=(1000, 20)))
    def forbidden(*args, **kwargs):
        raise AssertionError("正常满秩路径不应逐列OLS")
    monkeypatch.setattr(module, "_compute_vif_feature", forbidden)
    selector = VIFSelector(n_jobs=1)
    values = selector._compute_vif_all(frame)
    assert selector.vif_last_solver_ == "共享QR"
    assert np.isfinite(values).all()


def test_shared_vif_keeps_iterative_deletion_and_forced_include():
    class ReferenceVIF(VIFSelector):
        def _compute_vif_all(self, frame):
            filled = frame.fillna(self.missing).to_numpy()
            return pd.Series([_compute_vif_single(filled, i) for i in range(filled.shape[1])], index=frame.columns)
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(300, 7)), columns=list("abcdefg"))
    data["b"] = data.a + rng.normal(scale=0.03, size=len(data))
    data["d"] = data.c * 2 + rng.normal(scale=0.05, size=len(data))
    old = ReferenceVIF(threshold=4, include=["a"], n_jobs=1).fit(data)
    new = VIFSelector(threshold=4, include=["a"], n_jobs=1).fit(data)
    assert old.removed_features_ == new.removed_features_
    assert old.selected_features_ == new.selected_features_
    np.testing.assert_allclose(old.scores_, new.scores_, rtol=1e-8)


@pytest.mark.parametrize("seed", range(4))
def test_two_feature_numerical_tie_keeps_legacy_removal_order(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=200)
    data = pd.DataFrame({"a": x, "b": 3 * x + rng.normal(scale=0.1, size=200)})
    expected = pd.Series([_compute_vif_single(data.to_numpy(), i) for i in range(2)], index=data.columns).idxmax()
    actual = VIFSelector(threshold=4, n_jobs=1).fit(data)
    assert actual.removed_features_ == [expected]
