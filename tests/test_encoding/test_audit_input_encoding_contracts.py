"""2026-10 审计：输入隔离、训练编码、类别与缺失策略回归。"""

import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from hscredit.core.encoders import WOEEncoder, TargetEncoder, CatBoostEncoder, QuantileEncoder, OneHotEncoder, OrdinalEncoder
from hscredit.core.selectors import IVSelector
from hscredit.utils.data_contracts import prepare_xy, validate_target


@pytest.mark.parametrize("cls", [WOEEncoder, TargetEncoder, CatBoostEncoder, QuantileEncoder])
@pytest.mark.parametrize("index", [[10, 11, 12, 13], ["a", "b", "c", "d"], [1, 1, 2, 2]])
def test_input_apis_are_equivalent_and_position_aligned(cls, index):
    X = pd.DataFrame({"x": ["a", "a", "b", "b"]}, index=index)
    y = np.array([0, 0, 1, 1])
    df = X.assign(target=y)
    kwargs = {"n_jobs": 1, "target": "target"}
    if cls is CatBoostEncoder:
        kwargs["random_state"] = 42
    separate = cls(**kwargs).fit_transform(X, y)
    embedded = cls(**kwargs).fit_transform(df)
    mixed = cls(**kwargs).fit_transform(df, pd.Series(y, index=[3, 2, 1, 0]))
    pd.testing.assert_series_equal(separate.x, embedded.x)
    pd.testing.assert_series_equal(separate.x, mixed.x)
    estimator = cls(**kwargs).fit(df, y)
    assert estimator.cols_ == ["x"]
    assert list(estimator.feature_names_in_) == ["x"]


def test_selector_hybrid_target_is_not_learned():
    df = pd.DataFrame({"x": [0, 0, 1, 1], "target": [0, 0, 1, 1]})
    assert list(IVSelector(n_jobs=1).fit(df, df.target).feature_names_in_) == ["x"]


@pytest.mark.parametrize("y", [[0, 0.5, 1], [0, np.nan, 1], [0, np.inf, 1], [[0], [1], [0]], []])
def test_binary_invalid_targets_rejected_before_cast(y):
    with pytest.raises(ValueError):
        validate_target(y, target_type="binary")


def test_explicit_index_alignment_requires_exact_unique_keys():
    X = pd.DataFrame({"x": [1, 2]}, index=[10, 20])
    y = pd.Series([0, 1], index=[20, 10])
    assert prepare_xy(X, y).y.tolist() == [0, 1]
    assert prepare_xy(X, y, alignment="index").y.tolist() == [1, 0]
    with pytest.raises(ValueError, match="完全匹配"):
        prepare_xy(X, y.iloc[:1], alignment="index")
    with pytest.raises(ValueError, match="唯一"):
        prepare_xy(X.set_axis([1, 1]), y, alignment="index")


@pytest.mark.parametrize("cls", [WOEEncoder, TargetEncoder, CatBoostEncoder, QuantileEncoder, OrdinalEncoder])
@pytest.mark.parametrize("unknown", ["value", "return_nan", "error"])
def test_missing_nan_not_overridden_by_unknown_policy(cls, unknown):
    X = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    enc = cls(n_jobs=1, handle_missing="return_nan", handle_unknown=unknown).fit(X, [0, 0, 1, 1])
    assert pd.isna(enc.transform(pd.DataFrame({"x": [None]})).iloc[0, 0])


def test_catboost_ordered_training_is_not_in_sample_target():
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "target": [0, 0, 1, 1]})
    out = CatBoostEncoder(target="target", random_state=42, n_jobs=1).fit_transform(df)
    np.testing.assert_allclose(out.x, [.25, .5, .75, .5])


@pytest.mark.parametrize("cls", [TargetEncoder, WOEEncoder])
def test_oof_encoding_matches_fold_reference(cls):
    X = pd.DataFrame({"x": ["a", "a", "b", "b"] * 3}, index=[10] * 12)
    y = np.array([0, 1] * 6)
    groups = np.repeat([0, 1, 2], 4)
    cv = GroupKFold(3)
    enc = cls(training_mode="oof", cv=cv, n_jobs=1)
    output = enc.fit_transform(X, y, groups=groups)
    for train, valid in cv.split(X, y, groups):
        reference = cls(n_jobs=1).fit(X.iloc[train], y[train]).transform(X.iloc[valid])
        np.testing.assert_allclose(output.iloc[valid].x, reference.x)
    pd.testing.assert_frame_equal(enc.transform(X), cls(n_jobs=1).fit(X, y).transform(X))
    assert enc.oof_coverage_.all()


def test_time_oof_uncovered_prefix_is_nan_and_has_no_future():
    X = pd.DataFrame({"x": ["same"] * 12})
    y = np.arange(12, dtype=float)
    enc = TargetEncoder(training_mode="oof", cv=TimeSeriesSplit(3), n_jobs=1)
    with pytest.warns(UserWarning, match="未被折外"):
        result = enc.fit_transform(X, y)
    assert result.x.iloc[:3].isna().all()
    assert (result.x.iloc[3:] < y[3:]).all()


def test_onehot_dropped_class_known_and_names_unique():
    X = pd.DataFrame({"x": ["a", "b"]})
    OneHotEncoder(drop="first", handle_unknown="error", n_jobs=1).fit_transform(X)
    X = pd.DataFrame({"x": ["a-b", "a b", "missing", None], "x_a_b": [1, 2, 3, 4]})
    enc = OneHotEncoder(cols=["x"], n_jobs=1).fit(X)
    result = enc.transform(X)
    assert result.columns.is_unique
    assert len(enc.feature_names_) == 4
    assert result[enc.feature_names_].sum(axis=1).tolist() == [1] * 4
    assert not np.array_equal(result[enc.feature_names_].iloc[2], result[enc.feature_names_].iloc[3])
    restored = OneHotEncoder().import_mapping(json.loads(json.dumps(enc.export_mapping())))
    pd.testing.assert_frame_equal(result, restored.transform(X))


def test_onehot_sparse_grouping_and_budget():
    X = pd.DataFrame({"x": ["a"] * 5 + ["b", "c", "d", None]})
    enc = OneHotEncoder(n_jobs=1, min_frequency=2, max_categories=2, sparse_output=True, return_df=False)
    result = enc.fit_transform(X)
    assert sparse.isspmatrix_csr(result)
    assert result.shape == (9, 3)
    assert result.nnz == 9
    assert result[5].toarray().tolist() == result[7].toarray().tolist()
    with pytest.raises(ValueError, match="超过"):
        OneHotEncoder(n_jobs=1, max_output_bytes=1).fit_transform(X)


def test_sparse_frame_does_not_require_dense_matrix():
    X = pd.DataFrame({"x": [f"类别{i}" for i in range(1000)]})
    result = OneHotEncoder(n_jobs=1, sparse_output=True).fit_transform(X)
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in result.dtypes)
    assert result.memory_usage(deep=True).sum() < 100_000
