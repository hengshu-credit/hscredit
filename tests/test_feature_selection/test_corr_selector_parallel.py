"""CorrSelector 分块并行相关计算测试。"""

from contextlib import contextmanager
import pickle
import threading
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import hscredit.core.selectors.corr_selector as corr_module
from hscredit.core.selectors import CorrSelector
from hscredit.exceptions import ValidationError


def _dense_reference(X, weights, threshold, method="pearson"):
    """以完整相关矩阵复现按 metric 顺序逐步保留的语义。"""
    feature_names = X.columns.tolist()
    weight_series = pd.Series(weights).reindex(feature_names).fillna(0.0)
    sort_idx = np.argsort(-weight_series.to_numpy(dtype=float), kind="stable")
    sorted_names = [feature_names[index] for index in sort_idx]
    corr_matrix = X[sorted_names].corr(method=method).abs()

    kept_indices = []
    records = []
    for index, name in enumerate(sorted_names):
        if not kept_indices:
            kept_indices.append(index)
            continue
        correlations = corr_matrix.iloc[kept_indices, index]
        conflicts = correlations[correlations > threshold]
        if conflicts.empty:
            kept_indices.append(index)
            continue
        value = conflicts.max()
        related = conflicts[conflicts == value].index[0]
        records.append((name, value, related, weight_series[name]))

    selected = [sorted_names[index] for index in kept_indices]
    return selected, records


@pytest.fixture
def correlated_frame():
    rng = np.random.RandomState(2026)
    first = rng.normal(size=160)
    second = rng.normal(size=160)
    X = pd.DataFrame(
        {
            "甲": first,
            "甲副本": first * 0.99 + rng.normal(scale=0.01, size=160),
            "乙": second,
            "乙副本": second * -0.98 + rng.normal(scale=0.02, size=160),
            "独立": rng.normal(size=160),
            "常量": np.ones(160),
        }
    )
    weights = {"甲": 6.0, "甲副本": 5.0, "乙": 4.0, "乙副本": 3.0, "独立": 2.0, "常量": 1.0}
    return X, weights


def _assert_matches_dense_reference(selector, X, weights, threshold, method="pearson"):
    expected_selected, expected_records = _dense_reference(X, weights, threshold, method)
    assert selector.selected_features_ == expected_selected
    assert selector.dropped_["特征"].tolist() == [record[0] for record in expected_records]
    assert selector.dropped_["相关特征"].tolist() == [record[2] for record in expected_records]
    np.testing.assert_allclose(
        selector.dropped_["最大相关系数"].to_numpy(),
        np.asarray([record[1] for record in expected_records]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_corr_selector_exposes_cloneable_block_size(correlated_frame):
    X, weights = correlated_frame
    selector = CorrSelector(
        weights=weights,
        binning_params=None,
        corr_block_size=3,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert selector.corr_block_size == 3
    assert CorrSelector().method == "spearman"
    assert clone(selector).corr_block_size == 3
    assert pickle.loads(pickle.dumps(selector)).corr_block_size == 3


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_block_parallel_pearson_matches_serial_and_dense_reference(correlated_frame, backend):
    X, weights = correlated_frame
    serial = CorrSelector(
        threshold=0.8,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=1,
    ).fit(X)
    parallel = CorrSelector(
        threshold=0.8,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend=backend,
    ).fit(X)

    assert parallel.selected_features_ == serial.selected_features_
    pd.testing.assert_series_equal(parallel.scores_, serial.scores_, check_exact=True)
    pd.testing.assert_frame_equal(parallel.dropped_, serial.dropped_, check_exact=True)
    _assert_matches_dense_reference(parallel, X, weights, 0.8)


def test_fast_pearson_does_not_build_full_dataframe_correlation(correlated_frame, monkeypatch):
    X, weights = correlated_frame

    def fail_full_corr(*args, **kwargs):
        raise AssertionError("无缺失 Pearson 路径不应构造完整 DataFrame.corr")

    monkeypatch.setattr(pd.DataFrame, "corr", fail_full_corr)
    selector = CorrSelector(
        threshold=0.8,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert selector.selected_features_


def test_finite_fast_path_defaults_to_shared_memory_threading(correlated_frame, monkeypatch):
    X, weights = correlated_frame
    observed_backends = []
    original = CorrSelector._parallel_execute

    def record_backend(self, *args, **kwargs):
        observed_backends.append(kwargs.get("default_backend"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CorrSelector, "_parallel_execute", record_backend)
    CorrSelector(
        threshold=0.8,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
    ).fit(X)

    assert observed_backends
    assert set(observed_backends) == {"threading"}


def test_default_spearman_uses_fast_block_path_and_matches_dense(correlated_frame, monkeypatch):
    X, weights = correlated_frame
    expected_selected, expected_records = _dense_reference(X, weights, 0.8, "spearman")

    def fail_full_corr(*args, **kwargs):
        raise AssertionError("无缺失 Spearman 路径不应构造完整 DataFrame.corr")

    monkeypatch.setattr(pd.DataFrame, "corr", fail_full_corr)
    selector = CorrSelector(
        threshold=0.8,
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert selector.method == "spearman"
    assert selector.selected_features_ == expected_selected
    assert selector.dropped_["相关特征"].tolist() == [record[2] for record in expected_records]
    np.testing.assert_allclose(
        selector.dropped_["最大相关系数"].to_numpy(),
        np.asarray([record[1] for record in expected_records]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_block_spearman_ranking_uses_multiple_worker_threads(monkeypatch):
    """特征数小于默认块大小时，排名阶段仍必须实际使用多个线程。"""
    rng = np.random.RandomState(20260807)
    X = pd.DataFrame(
        rng.normal(size=(1200, 32)),
        columns=[f"特征{index}" for index in range(32)],
    )
    weights = {column: float(32 - index) for index, column in enumerate(X.columns)}
    rank_thread_ids = set()
    lock = threading.Lock()
    original_rank = pd.Series.rank

    def recording_rank(series, *args, **kwargs):
        with lock:
            rank_thread_ids.add(threading.get_ident())
        time.sleep(0.01)
        return original_rank(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "rank", recording_rank)
    CorrSelector(
        method="spearman",
        weights=weights,
        binning_params=None,
        n_jobs=4,
        parallel_backend="threading",
    ).fit(X)

    assert len(rank_thread_ids) >= 2


def test_spearman_ranking_reuses_the_input_buffer():
    """超宽表排名必须分批原地写回，避免再分配一个完整矩阵。"""
    rng = np.random.RandomState(20260807)
    values = rng.normal(size=(120, 13))
    original = values.copy()
    selector = CorrSelector(
        method="spearman",
        binning_params=None,
        corr_block_size=4,
        n_jobs=3,
        parallel_backend="threading",
    )

    ranked = selector._rank_corr_values(values, "threading")

    assert ranked is values
    expected = pd.DataFrame(original).rank(method="average").to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(ranked, expected)


def test_threaded_spearman_ranking_starts_the_executor_once(monkeypatch):
    """共享内存排名不得按块反复创建线程池。"""
    rng = np.random.RandomState(20260807)
    values = rng.normal(size=(120, 13))
    selector = CorrSelector(
        method="spearman",
        binning_params=None,
        corr_block_size=4,
        n_jobs=3,
        parallel_backend="threading",
    )
    calls = []
    original_execute = selector._parallel_execute

    def recording_execute(*args, **kwargs):
        calls.append(len(args[1]))
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(selector, "_parallel_execute", recording_execute)
    selector._rank_corr_values(values, "threading")

    assert calls == [13]


def test_corr_threading_divides_total_budget_between_outer_and_native_threads(monkeypatch):
    """相关块任务与 BLAS 线程共享同一份 n_jobs 总预算。"""
    rng = np.random.RandomState(20260807)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=list("abcdef"))
    weights = {column: float(6 - index) for index, column in enumerate(X.columns)}
    observed_limits = []

    monkeypatch.setattr("hscredit.utils.parallel.get_physical_cpu_count", lambda: 1)

    @contextmanager
    def recording_limits(*, limits):
        observed_limits.append(limits)
        yield

    monkeypatch.setattr(corr_module, "threadpool_limits", recording_limits, raising=False)
    CorrSelector(
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=4,
        parallel_backend="threading",
    ).fit(X)

    assert observed_limits == [4, 2, 1]


def test_removed_rows_are_not_multiplied_in_later_correlation_blocks(monkeypatch):
    """前一块已剔除的特征不得继续进入后续跨块矩阵乘法。"""
    rng = np.random.RandomState(20260807)
    first = rng.normal(size=160)
    X = pd.DataFrame(
        {
            "保留": first,
            "剔除": first.copy(),
            "独立甲": rng.normal(size=160),
            "独立乙": rng.normal(size=160),
        }
    )
    weights = {"保留": 4.0, "剔除": 3.0, "独立甲": 2.0, "独立乙": 1.0}
    left_widths = []
    lock = threading.Lock()
    original_matmul = np.matmul

    def recording_matmul(left, right, *args, **kwargs):
        with lock:
            left_widths.append(left.shape[0])
        return original_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(corr_module.np, "matmul", recording_matmul)
    CorrSelector(
        threshold=0.7,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert sorted(left_widths) == [1, 2, 2]


@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_missing_values_and_all_methods_match_dense_reference(method, backend):
    X = pd.DataFrame(
        {
            "甲": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0],
            "甲副本": [1.0, 2.1, 3.0, np.nan, 5.1, 6.0, 7.1],
            "乙": [7.0, np.nan, 5.0, 4.0, 3.0, 2.0, 1.0],
            "独立": [2.0, 5.0, 1.0, 6.0, np.nan, 4.0, 3.0],
        }
    )
    weights = {"甲": 4.0, "甲副本": 3.0, "乙": 2.0, "独立": 1.0}
    selector = CorrSelector(
        threshold=0.75,
        method=method,
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend=backend,
    ).fit(X)

    _assert_matches_dense_reference(selector, X, weights, 0.75, method)


def test_internal_metric_binner_inherits_corr_parallel_configuration():
    parallel_config = {"batch_size": 4, "inner_max_num_threads": 1}
    selector = CorrSelector(
        binning_params={"method": "best_iv", "max_n_bins": 3},
        n_jobs=3,
        parallel_backend=None,
        parallel_config=parallel_config,
    )

    binner = selector._resolve_binner()

    assert binner.n_jobs == 3
    assert binner.parallel_backend == "loky"
    assert binner.parallel_config == parallel_config
    assert binner.parallel_config is not parallel_config


def test_internal_metric_binner_preserves_explicit_parallel_overrides():
    selector = CorrSelector(
        binning_params={
            "method": "best_iv",
            "n_jobs": 1,
            "parallel_backend": "threading",
            "parallel_config": {"batch_size": 2},
        },
        n_jobs=3,
        parallel_backend="loky",
        parallel_config={"batch_size": 8},
    )

    binner = selector._resolve_binner()

    assert binner.n_jobs == 1
    assert binner.parallel_backend == "threading"
    assert binner.parallel_config == {"batch_size": 2}


def test_fitted_internal_binner_receives_corr_parallel_configuration():
    X = pd.DataFrame(
        {
            "甲": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "乙": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    selector = CorrSelector(
        binning_params={"method": "uniform", "max_n_bins": 2, "min_n_bins": 2},
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
        corr_block_size=1,
    ).fit(X, y)

    assert selector._binner_instance.n_jobs == 2
    assert selector._binner_instance.parallel_backend == "threading"
    assert selector._binner_instance.parallel_config == {"batch_size": 1}


def test_generic_block_path_never_correlates_more_than_two_blocks(monkeypatch):
    rng = np.random.RandomState(9)
    X = pd.DataFrame(rng.normal(size=(40, 7)), columns=list("abcdefg"))
    weights = {column: float(10 - index) for index, column in enumerate(X.columns)}
    original_corr = pd.DataFrame.corr
    observed_widths = []

    def bounded_corr(frame, *args, **kwargs):
        observed_widths.append(frame.shape[1])
        if frame.shape[1] > 4:
            raise AssertionError("分块相关任务不应构造超过两个块的相关矩阵")
        return original_corr(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "corr", bounded_corr)
    CorrSelector(
        threshold=0.8,
        method="kendall",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert observed_widths
    assert max(observed_widths) <= 4


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "64"])
def test_corr_block_size_validation_is_chinese(value, correlated_frame):
    X, weights = correlated_frame
    with pytest.raises(ValidationError, match="corr_block_size.*正整数"):
        CorrSelector(
            weights=weights,
            binning_params=None,
            corr_block_size=value,
            n_jobs=1,
        ).fit(X)


def test_threshold_remains_strictly_greater_than():
    X = pd.DataFrame({"甲": [1.0, 2.0, 3.0, 4.0], "乙": [2.0, 4.0, 6.0, 8.0]})
    selector = CorrSelector(
        threshold=1.0,
        weights={"甲": 2.0, "乙": 1.0},
        binning_params=None,
        corr_block_size=1,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert selector.selected_features_ == ["甲", "乙"]
    assert selector.dropped_.empty


def test_correlation_chain_does_not_drop_feature_only_related_to_removed_feature():
    rng = np.random.RandomState(7)
    raw = rng.normal(size=(200, 3))
    raw -= raw.mean(axis=0)
    basis, _ = np.linalg.qr(raw)
    X = pd.DataFrame(
        {
            "A": basis[:, 0],
            "B": 0.8 * basis[:, 0] + 0.6 * basis[:, 1],
            "C": (
                0.3 * basis[:, 0]
                + (0.56 / 0.6) * basis[:, 1]
                + np.sqrt(1 - 0.3**2 - (0.56 / 0.6) ** 2) * basis[:, 2]
            ),
        }
    )

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights={"A": 3.0, "B": 2.0, "C": 1.0},
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(X)

    assert selector.selected_features_ == ["A", "C"]
    assert selector.dropped_["特征"].tolist() == ["B"]
    assert selector.dropped_["相关特征"].tolist() == ["A"]


def test_equal_metric_uses_original_column_order_as_stable_tie_breaker():
    X = pd.DataFrame(
        {
            "先出现": np.arange(20.0),
            "后出现": np.arange(20.0) * 2,
        }
    )

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights={"先出现": 1.0, "后出现": 1.0},
        binning_params=None,
        n_jobs=1,
    ).fit(X)

    assert selector.selected_features_ == ["先出现"]
    assert selector.dropped_["相关特征"].tolist() == ["先出现"]


def test_every_correlation_drop_references_retained_not_worse_metric_feature():
    X = pd.DataFrame(
        {
            "高": np.arange(30.0),
            "中": np.arange(30.0) * 1.5,
            "低": np.arange(30.0) * -2,
        }
    )
    weights = {"高": 3.0, "中": 2.0, "低": 1.0}

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights=weights,
        binning_params=None,
        corr_block_size=2,
        n_jobs=2,
        parallel_backend="loky",
    ).fit(X)

    for _, row in selector.dropped_.iterrows():
        assert row["相关特征"] in selector.selected_features_
        assert weights[row["相关特征"]] >= weights[row["特征"]]
        assert row["最大相关系数"] > selector.threshold


def test_force_drop_feature_cannot_eliminate_feature_that_will_be_retained():
    X = pd.DataFrame(
        {
            "将强制删除": np.arange(20.0),
            "应保留": np.arange(20.0) * 2,
        }
    )

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights={"将强制删除": 5.0, "应保留": 1.0},
        force_drop=["将强制删除"],
        binning_params=None,
        n_jobs=1,
    ).fit(X)

    assert selector.selected_features_ == ["应保留"]
    assert selector.dropped_["特征"].tolist() == ["将强制删除"]
    assert "强制剔除" in selector.dropped_["剔除原因"].iloc[0]


def test_force_include_has_priority_and_report_does_not_claim_lower_metric():
    X = pd.DataFrame(
        {
            "强制保留": np.arange(20.0),
            "高metric": np.arange(20.0) * 2,
        }
    )

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights={"强制保留": 1.0, "高metric": 5.0},
        include=["强制保留"],
        binning_params=None,
        n_jobs=1,
    ).fit(X)

    assert selector.selected_features_ == ["强制保留"]
    assert selector.dropped_["相关特征"].tolist() == ["强制保留"]
    assert "强制保留变量" in selector.dropped_["剔除原因"].iloc[0]
    assert "较低" not in selector.dropped_["剔除原因"].iloc[0]
