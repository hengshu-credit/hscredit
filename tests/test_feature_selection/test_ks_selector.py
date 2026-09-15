"""KS 筛选器的数值口径与统一接口测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import hscredit
from hscredit.core import selectors
from hscredit.core.binning import OptimalBinning
from hscredit.core.selectors import CompositeFeatureSelector, SelectionReportCollector
from hscredit.exceptions import NotFittedError, ValidationError


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "正向": [0, 0, 1, 1],
            "反向": [1, 1, 0, 0],
            "边界": [0, 1, 1, 2],
            "无效": [0, 1, 0, 1],
            "常量": [1, 1, 1, 1],
            "目标": [0, 0, 1, 1],
        },
        index=[10, 20, 30, 40],
    )


def test_raw_ks_threshold_and_chinese_report(sample_df):
    selector = selectors.KSSelector(target="目标", threshold=0.5, n_jobs=1)
    result = selector.fit_transform(sample_df)

    assert selector.scores_.to_dict() == pytest.approx(
        {"正向": 1.0, "反向": 1.0, "边界": 0.5, "无效": 0.0, "常量": 0.0}
    )
    assert selector.selected_features_ == ["正向", "反向", "边界"]
    pd.testing.assert_frame_equal(result, sample_df[["正向", "反向", "边界", "目标"]])
    assert selector.get_support().tolist() == [True, True, True, False, False]
    assert selector.get_feature_names_out().tolist() == selector.selected_features_
    report = selector.get_selection_report()
    assert report["输入特征数"] == 5
    assert report["选中特征数"] == 3
    assert report["筛选方法"] == "KS值筛选"
    assert set(selector.dropped_.columns) == {"特征", "剔除原因", "KS值", "阈值"}
    assert all("< 阈值" in reason for reason in selector.dropped_["剔除原因"])
    collector = SelectionReportCollector().add_report(selector)
    assert collector.get_summary()["最终特征数"] == 3


def test_tied_values_are_invariant_to_row_order(sample_df):
    first = selectors.KSSelector(target="目标").fit(sample_df)
    second = selectors.KSSelector(target="目标").fit(sample_df.iloc[[3, 0, 2, 1]])
    pd.testing.assert_series_equal(first.scores_, second.scores_)
    assert first.scores_["无效"] == 0


def test_external_y_has_priority_without_target_leakage(sample_df):
    external_y = np.array([0, 1, 0, 1])
    selector = selectors.KSSelector(target="目标", threshold=1).fit(sample_df, external_y)
    assert selector.selected_features_ == ["无效"]
    assert "目标" not in selector.scores_.index
    assert selector.n_features_in_ == 5


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_categories_use_bad_rate_order_independent_of_labels(dtype):
    X = pd.DataFrame({"类别": pd.Series(["甲", "丙", "乙", "丁"], dtype=dtype)})
    y = np.array([0, 1, 0, 1])
    selector = selectors.KSSelector(threshold=1).fit(X, y)
    assert selector.scores_["类别"] == 1
    assert selector.selected_features_ == ["类别"]


def test_missing_constant_and_single_class_valid_subset():
    X = pd.DataFrame(
        {
            "有缺失": pd.Series([0, pd.NA, 1, 2], dtype="Float64"),
            "全缺失": [np.nan] * 4,
            "单类别": [0, 1, np.nan, np.nan],
            "类别缺失": pd.Series(["甲", None, "乙", "乙"], dtype="category"),
        }
    )
    selector = selectors.KSSelector(threshold=0.1).fit(X, [0, 0, 1, 1])
    assert selector.scores_.to_dict() == {"有缺失": 1, "全缺失": 0, "单类别": 0, "类别缺失": 1}


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, "0.1", None, True])
def test_invalid_threshold_is_rejected_even_for_forced_features(sample_df, threshold):
    selector = selectors.KSSelector(target="目标", threshold=threshold, include=sample_df.columns[:-1].tolist())
    with pytest.raises(ValidationError, match="阈值"):
        selector.fit(sample_df)


@pytest.mark.parametrize("y", [None, [0, 0, 0, 0], [0, 1, 2, 0], [0, 1, np.nan, 0], [[0], [0], [1], [1]]])
def test_invalid_target_is_rejected(sample_df, y):
    with pytest.raises(ValidationError, match="目标变量"):
        selectors.KSSelector().fit(sample_df.drop(columns="目标"), y)


def test_forced_features_and_empty_selection(sample_df):
    selector = selectors.KSSelector(
        target="目标", threshold=1, include=["常量", "反向"], exclude=["反向"], force_drop=["正向"]
    ).fit(sample_df)
    assert selector.selected_features_ == ["常量"]
    assert set(selector.removed_features_) == {"正向", "反向", "边界", "无效"}
    assert set(selector.scores_.index) == {"边界", "无效"}
    empty = selectors.KSSelector(threshold=0.1).fit(sample_df[["常量"]], sample_df["目标"])
    assert empty.transform(sample_df[["常量"]]).shape == (4, 0)


def test_exports_clone_ndarray_and_pipeline(sample_df):
    from hscredit.core import KSSelector

    assert hscredit.KSSelector is KSSelector
    X = sample_df.drop(columns="目标").to_numpy()
    y = sample_df["目标"].to_numpy()
    selector = clone(KSSelector(threshold=0.5))
    pipeline = Pipeline([("筛选", selector), ("模型", LogisticRegression())]).fit(X, y)
    assert selector.transform(X).shape == (4, 3)
    np.testing.assert_array_equal(pipeline.predict(X), y)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_and_composite_match_serial(sample_df, backend):
    serial = selectors.KSSelector(target="目标", threshold=0.5, n_jobs=1).fit(sample_df)
    parallel = selectors.KSSelector(target="目标", threshold=0.5, n_jobs=2, parallel_backend=backend).fit(sample_df)
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    composite = CompositeFeatureSelector([parallel], target="目标").fit(sample_df)
    assert composite.selected_features_ == serial.selected_features_


@pytest.mark.parametrize("pretrained", [False, True])
def test_optional_binning_and_raw_transform(sample_df, pretrained):
    X, y = sample_df[["正向", "反向", "无效"]], sample_df["目标"]
    params = {"method": "uniform", "max_n_bins": 2, "n_jobs": 1}
    if pretrained:
        selector = selectors.KSSelector(threshold=0.5, binner=OptimalBinning(**params).fit(X, y))
    else:
        selector = selectors.KSSelector(threshold=0.5, binning_params=params)
    selector.fit(X, y)
    assert selector.scores_.to_dict() == pytest.approx({"正向": 1, "反向": 1, "无效": 0})
    pd.testing.assert_frame_equal(selector.transform(X), X[["正向", "反向"]])


def test_failed_refit_preserves_previous_result(sample_df):
    selector = selectors.KSSelector(target="目标").fit(sample_df)
    before = selector.scores_.copy()
    with pytest.raises(ValidationError, match="目标变量"):
        selector.fit(sample_df, [0, 0, 0, 0])
    pd.testing.assert_series_equal(selector.scores_, before)
    with pytest.raises(NotFittedError):
        selectors.KSSelector().transform(sample_df)


def test_binned_perfect_separation_keeps_threshold_one_without_divide_errors(sample_df):
    with np.errstate(divide="raise", invalid="raise"):
        selector = selectors.KSSelector(
            threshold=1,
            binning_params={"method": "uniform", "max_n_bins": 2, "n_jobs": 1},
        ).fit(sample_df[["正向", "反向"]], sample_df["目标"])
    assert selector.selected_features_ == ["正向", "反向"]
    assert selector.scores_.tolist() == [1, 1]
