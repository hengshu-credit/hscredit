"""分箱器公共并行配置与结果一致性测试。"""

import inspect
import time

import hscredit.core.binning.optimal_binning as optimal_binning_module
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

from hscredit.core import binning
from hscredit.core.binning import BaseBinning, OptimalBinning, OptimalBinning2D, UniformBinning
from hscredit.core.metrics import compute_bin_stats
from hscredit.exceptions import ParallelExecutionError


PARALLEL_PARAMETERS = {
    "n_jobs": -1,
    "parallel_backend": None,
    "parallel_config": None,
}


FEATURE_DICT_STATE = (
    "splits_",
    "n_bins_",
    "bin_tables_",
    "feature_types_",
    "_cat_bins_",
    "_category_orders_",
    "_category_code_maps_",
    "_categorical_numeric_splits_",
    "_categorical_fit_context_",
    "tree_models_",
    "monotonic_trend_",
    "_actual_rates",
    "clip_bounds_",
)
FEATURE_SET_STATE = ("_categorical_encoded_features_",)


@pytest.fixture
def mixed_xy():
    index = pd.Index(range(100, 124), name="样本号")
    X = pd.DataFrame(
        {
            "数值一": np.tile(np.arange(1, 13), 2),
            "类别": np.tile(["甲", "乙", "丙", "丁"], 6),
            "数值二": np.tile(np.arange(12, 0, -1), 2),
        },
        index=index,
    )
    y = pd.Series(np.tile([0, 1, 0, 1, 1, 0], 4), index=index, name="目标")
    return X, y


def _assert_value_equal(left, right):
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_value_equal(left_item, right_item)
    elif isinstance(left, BaseEstimator):
        assert left.get_params(deep=True) == right.get_params(deep=True)
        if hasattr(left, "tree_"):
            for attribute in ("children_left", "children_right", "feature", "threshold", "value"):
                np.testing.assert_array_equal(getattr(left.tree_, attribute), getattr(right.tree_, attribute))
    elif isinstance(left, float) and np.isnan(left):
        assert isinstance(right, float) and np.isnan(right)
    else:
        assert left == right


def _assert_feature_state_equal(serial, parallel, columns):
    for state_name in FEATURE_DICT_STATE:
        if not hasattr(serial, state_name) and not hasattr(parallel, state_name):
            continue
        serial_state = getattr(serial, state_name)
        parallel_state = getattr(parallel, state_name)
        assert list(serial_state) == list(parallel_state), state_name
        assert list(parallel_state) == [feature for feature in columns if feature in serial_state], state_name
        for feature in serial_state:
            _assert_value_equal(serial_state[feature], parallel_state[feature])
    for state_name in FEATURE_SET_STATE:
        if hasattr(serial, state_name) or hasattr(parallel, state_name):
            assert getattr(serial, state_name) == getattr(parallel, state_name)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_uniform_fit_and_transform_match_serial(backend, mixed_xy):
    X, y = mixed_xy
    serial = UniformBinning(max_n_bins=4, n_jobs=1).fit(X, y)
    parallel = UniformBinning(max_n_bins=4, n_jobs=2, parallel_backend=backend).fit(X, y)

    _assert_feature_state_equal(serial, parallel, X.columns)
    for metric in ("indices", "bins", "woe"):
        serial_result = serial.transform(X, metric=metric)
        parallel_result = parallel.transform(X, metric=metric)
        pd.testing.assert_frame_equal(serial_result, parallel_result)
        assert parallel_result.index.equals(X.index)
        assert list(parallel_result.columns) == list(X.columns)
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in parallel.transform(X, metric="indices").dtypes)
    assert all(pd.api.types.is_object_dtype(dtype) for dtype in parallel.transform(X, metric="bins").dtypes)
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in parallel.transform(X, metric="woe").dtypes)


class DelayedUniformBinning(UniformBinning):
    """让首列稳定后完成，用于捕获线程写入顺序漂移。"""

    def _fit_feature(self, feature, X, y):
        if feature == "数值一":
            time.sleep(0.1)
        return super()._fit_feature(feature, X, y)


def test_threading_fit_state_keys_match_serial_input_order(mixed_xy):
    X, y = mixed_xy
    serial = DelayedUniformBinning(max_n_bins=4, n_jobs=1).fit(X, y)
    parallel = DelayedUniformBinning(max_n_bins=4, n_jobs=2, parallel_backend="threading").fit(X, y)

    _assert_feature_state_equal(serial, parallel, X.columns)


class FailingFeatureBinner(UniformBinning):
    """在成功特征完成后失败，用于验证整轮拟合原子性。"""

    def _fit_feature(self, feature, X, y):
        if feature == "坏特征":
            time.sleep(0.1)
            raise ValueError("坏特征不能拟合")
        return super()._fit_feature(feature, X, y)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_failed_feature_fit_does_not_commit_partial_state(backend):
    X = pd.DataFrame({"好特征": np.arange(12), "坏特征": np.arange(12, 24)})
    y = pd.Series(np.tile([0, 1], 6), name="目标")
    binner = FailingFeatureBinner(n_jobs=2, parallel_backend=backend)

    with pytest.raises(ParallelExecutionError, match="坏特征"):
        binner.fit(X, y)

    for state_name in FEATURE_DICT_STATE:
        if hasattr(binner, state_name):
            assert getattr(binner, state_name) == {}, state_name
    for state_name in FEATURE_SET_STATE:
        assert getattr(binner, state_name) == set(), state_name


BINNER_SMOKE_CASES = [
    pytest.param("UniformBinning", {}, id="uniform"),
    pytest.param("QuantileBinning", {}, id="quantile"),
    pytest.param("TreeBinning", {"max_depth": 2}, id="tree"),
    pytest.param("CartBinning", {}, id="cart"),
    pytest.param("ChiMergeBinning", {}, id="chi_merge"),
    pytest.param("BestKSBinning", {}, id="best_ks"),
    pytest.param("BestIVBinning", {}, id="best_iv"),
    pytest.param("MDLPBinning", {"max_candidates": 8}, id="mdlp"),
    pytest.param("ORBinning", {"n_prebins": 6, "max_candidates": 8, "time_limit": 1}, id="or_tools"),
    pytest.param("CPSATBinning", {"n_prebins": 6, "max_candidates": 8, "time_limit": 1}, id="cp_sat"),
    pytest.param("KMeansBinning", {"n_init": 2, "max_iter": 20}, id="kmeans"),
    pytest.param("MonotonicBinning", {"init_n_bins": 6}, id="monotonic"),
    pytest.param("GeneticBinning", {"population_size": 8, "generations": 2}, id="genetic"),
    pytest.param("SmoothBinning", {"n_prebins": 6}, id="smooth"),
    pytest.param("KernelDensityBinning", {"n_grid_points": 50}, id="kernel_density"),
    pytest.param("BestLiftBinning", {"n_prebins": 6}, id="best_lift"),
    pytest.param("TargetBadRateBinning", {}, id="target_bad_rate"),
    pytest.param("OptimalBinning", {"method": "uniform"}, id="optimal_factory"),
]


@pytest.mark.parametrize("class_name,algorithm_kwargs", BINNER_SMOKE_CASES)
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_exported_concrete_binner_smoke_matrix_matches_serial(
    class_name, algorithm_kwargs, backend, mixed_xy
):
    X, y = mixed_xy
    cls = getattr(binning, class_name)
    common = dict(max_n_bins=3, min_n_bins=2, min_bin_size=0.05, random_state=17)
    try:
        serial = cls(n_jobs=1, **common, **algorithm_kwargs).fit(X, y)
        parallel = cls(n_jobs=2, parallel_backend=backend, **common, **algorithm_kwargs).fit(X, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    _assert_feature_state_equal(serial, parallel, X.columns)
    for metric in ("indices", "bins", "woe"):
        pd.testing.assert_frame_equal(serial.transform(X, metric=metric), parallel.transform(X, metric=metric))


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_optimal_binning_2d_transform_matches_serial(backend, mixed_xy):
    X, y = mixed_xy
    features = ["数值一", "数值二"]
    serial = OptimalBinning2D(method="uniform", max_n_bins=3, n_jobs=1, random_state=17).fit(
        X[features], y
    )
    parallel = OptimalBinning2D(
        method="uniform", max_n_bins=3, n_jobs=2, parallel_backend=backend, random_state=17
    ).fit(X[features], y)

    for metric in ("indices", "bins", "woe", "event_rate"):
        pd.testing.assert_frame_equal(serial.transform(X[features], metric=metric), parallel.transform(X[features], metric=metric))


def test_all_exported_binners_expose_parallel_parameters():
    missing = []
    wrong_defaults = []
    for name in binning.__all__:
        cls = getattr(binning, name)
        if inspect.isclass(cls) and issubclass(cls, BaseBinning):
            params = inspect.signature(cls.__init__).parameters
            absent = set(PARALLEL_PARAMETERS) - set(params)
            if absent:
                missing.append(name)
                continue
            for parameter, expected in PARALLEL_PARAMETERS.items():
                if params[parameter].default != expected:
                    wrong_defaults.append((name, parameter, params[parameter].default))

    assert missing == []
    assert wrong_defaults == []


def test_optimal_binning_2d_exposes_parallel_parameters():
    params = inspect.signature(OptimalBinning2D.__init__).parameters
    assert {name: params[name].default for name in PARALLEL_PARAMETERS} == PARALLEL_PARAMETERS


def test_all_exported_binners_store_parallel_parameters():
    unavailable = []
    for name in binning.__all__:
        cls = getattr(binning, name)
        if not inspect.isclass(cls) or not (issubclass(cls, BaseBinning) or cls is OptimalBinning2D):
            continue
        if inspect.isabstract(cls):
            continue
        try:
            binner = cls(
                n_jobs=2,
                parallel_backend="threading",
                parallel_config={"batch_size": 1},
            )
        except ImportError:
            unavailable.append(name)
            continue
        params = binner.get_params(deep=False)
        assert params["n_jobs"] == 2, name
        assert params["parallel_backend"] == "threading", name
        assert params["parallel_config"] == {"batch_size": 1}, name

    assert set(unavailable) <= {"ORBinning", "CPSATBinning"}


def test_numpy_int8_category_sort_key_does_not_overflow():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    bins = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    result = compute_bin_stats(bins, y)
    assert not result.empty


def test_uniform_parallel_parameters_survive_sklearn_clone():
    binner = UniformBinning(
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )
    cloned = clone(binner)

    assert cloned.n_jobs == 2
    assert cloned.parallel_backend == "threading"
    assert cloned.parallel_config == {"batch_size": 1}


def test_optimal_binning_forwards_parallel_parameters_to_method_binner():
    X = pd.DataFrame(
        {
            "数值一": [1, 2, 3, 4, 5, 6, 7, 8],
            "数值二": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    binner = OptimalBinning(
        method="uniform",
        n_jobs=1,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    ).fit(X, y)

    assert binner._binner.n_jobs == 1
    assert binner._binner.parallel_backend == "threading"
    assert binner._binner.parallel_config == {"batch_size": 1}


def test_prebinning_empty_splits_fallback_inherits_parallel_parameters(monkeypatch):
    X = pd.DataFrame({"数值": [1, 2, 3, 4, 5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    parent = OptimalBinning(
        method="best_iv",
        max_n_bins=3,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )
    created = []

    class RecordingOptimalBinning(OptimalBinning):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(optimal_binning_module, "OptimalBinning", RecordingOptimalBinning)

    parent._fit_with_method_and_prebins(X, y, pre_splits={})

    assert len(created) == 1
    fallback = created[0]
    assert fallback.n_jobs == 2
    assert fallback.parallel_backend == "threading"
    assert fallback.parallel_config == {"batch_size": 1}


def test_optimal_binning_2d_forwards_parallel_parameters_to_axis_binners():
    X = pd.DataFrame(
        {
            "横轴": [1, 2, 3, 4, 5, 6, 7, 8],
            "纵轴": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="目标")
    binner = OptimalBinning2D(
        n_jobs=1,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    ).fit(X, y)

    for axis_binner in (binner.binner_x_, binner.binner_y_):
        assert axis_binner.n_jobs == 1
        assert axis_binner.parallel_backend == "threading"
        assert axis_binner.parallel_config == {"batch_size": 1}
