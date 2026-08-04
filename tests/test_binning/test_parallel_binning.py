"""分箱器公共并行配置与结果一致性测试。"""

import inspect
import pickle
import time

import hscredit.core.binning.optimal_binning as optimal_binning_module
import hscredit.core.binning.base as base_binning_module
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

from hscredit.core import binning
from hscredit.core.binning import BestIVBinning, BaseBinning, OptimalBinning, OptimalBinning2D, UniformBinning
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


EXECUTION_MODES = [
    pytest.param(1, None, id="serial"),
    pytest.param(2, "threading", id="threading"),
    pytest.param(2, "loky", id="loky"),
]


class RefitFailingUniformBinning(UniformBinning):
    """仅在重拟合指定类别特征时失败。"""

    def _fit_feature(self, feature, X, y):
        if feature == "坏分类":
            raise ValueError("坏分类重拟合失败")
        return super()._fit_feature(feature, X, y)


class PostFitFailingBestIVBinning(BestIVBinning):
    """在指定重拟合数据的后处理阶段失败。"""

    def _apply_post_fit_constraints(self, X, y, **kwargs):
        if "后处理失败" in X.columns:
            raise ValueError("后处理故意失败")
        return super()._apply_post_fit_constraints(X, y, **kwargs)


class FinalizeFailingUniformBinning(UniformBinning):
    """在指定类别重拟合的最终还原阶段失败。"""

    def _finalize_categorical_fit(self):
        if "最终失败" in self._categorical_fit_context_:
            raise ValueError("类别最终还原故意失败")
        return super()._finalize_categorical_fit()


class DelegatingUniformBinning(UniformBinning):
    """验证子类 fit 委托父类时只建立一个事务候选。"""

    def fit(self, X, y=None, **kwargs):
        return super().fit(X, y, **kwargs)


class RefinementFailingOptimalBinning(OptimalBinning):
    """若 lift_refine=False 在事务候选中丢失则立即失败。"""

    def _refine_splits_for_lift_stability(self, X, y):
        raise AssertionError("lift_refine=False 未保留")


class AxisWorkerFailingOptimalBinning(OptimalBinning):
    """在二维新 Y 轴的底层分箱 worker 阶段失败。"""

    def _fit_with_method(self, X, y):
        if "new_y" in X.columns:
            raise ValueError("二维轴 worker 故意失败")
        return super()._fit_with_method(X, y)


class AxisWorkerFailingOptimalBinning2D(OptimalBinning2D):
    """为新 Y 轴注入真实的失败 OptimalBinning。"""

    def _create_binner(self, is_x):
        binner = super()._create_binner(is_x)
        if not is_x and self.feature_y_ == "new_y":
            return AxisWorkerFailingOptimalBinning(**binner.get_params(deep=False), **binner.kwargs)
        return binner


class CrossTableFailingOptimalBinning2D(OptimalBinning2D):
    """在两轴成功后的交叉统计阶段失败。"""

    def _compute_cross_table(self):
        if self.feature_x_ == "new_x":
            raise ValueError("二维交叉统计故意失败")
        return super()._compute_cross_table()


class BinningTableFailingOptimalBinning2D(OptimalBinning2D):
    """在最终二维分箱表阶段失败。"""

    def _compute_binning_table(self):
        if self.feature_x_ == "new_x":
            raise ValueError("二维最终统计故意失败")
        return super()._compute_binning_table()


class UnregisteredStateUniformBinning(UniformBinning):
    """模拟具体算法未登记的按特征拟合输出。"""

    def fit(self, X, y=None, **kwargs):
        result = super().fit(X, y, **kwargs)
        self.audit_state_ = getattr(self, "audit_state_", {})
        self.audit_state_.update({feature: "已拟合" for feature in X.columns})
        return result


def _successful_initial_fit(binner):
    X = pd.DataFrame({"旧特征": np.arange(24, dtype=float)})
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    return binner.fit(X, y), y


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_failed_categorical_refit_restores_complete_previous_model(n_jobs, backend):
    binner, y = _successful_initial_fit(
        RefitFailingUniformBinning(n_jobs=n_jobs, parallel_backend=backend, random_state=17)
    )
    before = pickle.dumps(binner)
    refit = pd.DataFrame(
        {
            "新分类": np.tile(["甲", "乙", "丙"], 8),
            "坏分类": np.tile(["A", "B", "C", "D"], 6),
        }
    )

    with pytest.raises(ParallelExecutionError, match="坏分类"):
        binner.fit(refit, y)

    assert pickle.dumps(binner) == before
    assert binner._is_fitted is True


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_post_fit_failure_restores_complete_previous_model(n_jobs, backend):
    binner, y = _successful_initial_fit(
        PostFitFailingBestIVBinning(n_jobs=n_jobs, parallel_backend=backend, random_state=17)
    )
    before = pickle.dumps(binner)

    with pytest.raises(ValueError, match="后处理故意失败"):
        binner.fit(pd.DataFrame({"后处理失败": np.arange(24, dtype=float)}), y)

    assert pickle.dumps(binner) == before
    assert binner._is_fitted is True


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_finalize_failure_restores_complete_previous_model(n_jobs, backend):
    binner, y = _successful_initial_fit(
        FinalizeFailingUniformBinning(n_jobs=n_jobs, parallel_backend=backend, random_state=17)
    )
    before = pickle.dumps(binner)
    refit = pd.DataFrame({"最终失败": np.tile(["甲", "乙", "丙"], 8)})

    with pytest.raises(ValueError, match="类别最终还原故意失败"):
        binner.fit(refit, y)

    assert pickle.dumps(binner) == before
    assert binner._is_fitted is True


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_successful_refit_drops_stale_optional_feature_state(n_jobs, backend):
    binner = UniformBinning(
        left_clip=0.1,
        right_clip=0.9,
        n_jobs=n_jobs,
        parallel_backend=backend,
        random_state=17,
    )
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    binner.fit(pd.DataFrame({"特征": np.arange(24, dtype=float)}), y)
    assert "特征" in binner.clip_bounds_

    binner.fit(pd.DataFrame({"特征": np.full(24, np.nan)}), y)

    assert getattr(binner, "clip_bounds_", {}) == {}


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_fitted_binner_pickle_round_trip_preserves_parallel_results(n_jobs, backend, mixed_xy):
    X, y = mixed_xy
    fitted = UniformBinning(n_jobs=n_jobs, parallel_backend=backend, random_state=17).fit(X, y)
    restored = pickle.loads(pickle.dumps(fitted))

    _assert_feature_state_equal(fitted, restored, X.columns)
    pd.testing.assert_frame_equal(fitted.transform(X, metric="woe"), restored.transform(X, metric="woe"))


def test_delegating_fit_uses_single_candidate_transaction(monkeypatch, mixed_xy):
    X, y = mixed_xy
    clone_calls = []
    sklearn_clone = base_binning_module.clone

    def recording_clone(estimator):
        clone_calls.append(estimator)
        return sklearn_clone(estimator)

    monkeypatch.setattr(base_binning_module, "clone", recording_clone)
    fitted = DelegatingUniformBinning(n_jobs=1, random_state=17).fit(X, y)

    assert len(clone_calls) == 1
    assert fitted._is_fitted is True
    assert "_fit_transaction_active" not in fitted.__dict__


@pytest.mark.parametrize(
    "binner_cls,error_message",
    [
        pytest.param(AxisWorkerFailingOptimalBinning2D, "二维轴 worker 故意失败", id="axis-worker"),
        pytest.param(CrossTableFailingOptimalBinning2D, "二维交叉统计故意失败", id="cross-table"),
        pytest.param(BinningTableFailingOptimalBinning2D, "二维最终统计故意失败", id="binning-table"),
    ],
)
@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_2d_failed_refit_restores_complete_previous_model(binner_cls, error_message, n_jobs, backend):
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    old = pd.DataFrame({"old_x": np.arange(24, dtype=float), "old_y": np.arange(23, -1, -1, dtype=float)})
    new = pd.DataFrame({"new_x": np.arange(23, -1, -1, dtype=float), "new_y": np.arange(24, dtype=float)})
    binner = binner_cls(method="uniform", n_jobs=n_jobs, parallel_backend=backend, random_state=17).fit(old, y)
    before = pickle.dumps(binner)
    old_result = binner.transform(old, metric="woe")

    with pytest.raises(ValueError, match=error_message):
        binner.fit(new, y)

    assert pickle.dumps(binner) == before
    pd.testing.assert_frame_equal(binner.transform(old, metric="woe"), old_result)


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_2d_successful_refit_replaces_all_previous_state(n_jobs, backend):
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    old = pd.DataFrame({"old_x": np.arange(24, dtype=float), "old_y": np.arange(23, -1, -1, dtype=float)})
    new = pd.DataFrame({"new_x": np.arange(23, -1, -1, dtype=float), "new_y": np.arange(24, dtype=float)})
    binner = OptimalBinning2D(
        method="uniform", n_jobs=n_jobs, parallel_backend=backend, random_state=17
    ).fit(old, y)

    binner.fit(new, y)

    assert (binner.feature_x_, binner.feature_y_) == ("new_x", "new_y")
    assert binner.feature_names_in_.tolist() == ["new_x", "new_y"]
    assert list(binner._X.columns) == ["new_x", "new_y"]
    assert list(binner.binner_x_.splits_) == ["new_x"]
    assert list(binner.binner_y_.splits_) == ["new_y"]
    assert binner.transform(new, metric="indices").shape == (24, 1)


def test_2d_transaction_preserves_signature_clone_and_pickle(mixed_xy):
    X, y = mixed_xy
    features = ["数值一", "数值二"]
    params = inspect.signature(OptimalBinning2D.fit).parameters
    assert list(params) == ["self", "X", "y", "features"]
    source = OptimalBinning2D(method="uniform", max_n_bins=3, n_jobs=1, random_state=17)
    assert clone(source).get_params(deep=False) == source.get_params(deep=False)

    fitted = source.fit(X[features], y)
    restored = pickle.loads(pickle.dumps(fitted))

    pd.testing.assert_frame_equal(
        fitted.transform(X[features], metric="woe"), restored.transform(X[features], metric="woe")
    )


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
@pytest.mark.parametrize(
    "binner_factory,extra_state_name",
    [
        pytest.param(lambda **kwargs: UnregisteredStateUniformBinning(**kwargs), "audit_state_", id="base"),
        pytest.param(lambda **kwargs: OptimalBinning(method="uniform", **kwargs), None, id="optimal"),
    ],
)
def test_imported_rule_fit_candidate_excludes_previous_fitted_outputs(
    binner_factory, extra_state_name, n_jobs, backend
):
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    old = pd.DataFrame({"old": np.arange(24, dtype=float)})
    new = pd.DataFrame({"new": np.arange(24, dtype=float)})
    binner = binner_factory(n_jobs=n_jobs, parallel_backend=backend, random_state=17).fit(old, y)

    binner.import_rules({"new": [5.0, 10.0, 15.0]})
    assert set(binner.export_rules()) == {"old", "new"}
    assert binner.transform(old, metric="indices").shape == (24, 1)
    assert binner.transform(new, metric="indices").shape == (24, 1)

    binner.fit(new, y)

    assert binner.feature_names_in_.tolist() == ["new"]
    assert list(binner.splits_) == ["new"]
    assert list(binner.bin_tables_) == ["new"]
    assert "old" not in getattr(binner, "_woe_maps_", {})
    if extra_state_name is not None:
        assert list(getattr(binner, extra_state_name)) == ["new"]
    else:
        assert binner._binner is None
        np.testing.assert_array_equal(binner.splits_["new"], np.array([5.0, 10.0, 15.0]))
    assert binner.transform(new, metric="indices").shape == (24, 1)


@pytest.mark.parametrize(
    "binner_factory",
    [
        pytest.param(lambda: UniformBinning(n_jobs=1), id="base"),
        pytest.param(lambda: OptimalBinning(method="uniform", n_jobs=1), id="optimal"),
    ],
)
def test_import_rules_immediately_override_same_feature(binner_factory):
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    X = pd.DataFrame({"特征": np.arange(24, dtype=float)})
    binner = binner_factory().fit(X, y)

    binner.import_rules({"特征": [5.0, 10.0, 15.0]})

    np.testing.assert_array_equal(binner.splits_["特征"], np.array([5.0, 10.0, 15.0]))
    assert binner.transform(X, metric="indices").shape == (24, 1)


def test_loaded_woe_map_remains_available_after_imported_rule_fit():
    rules = {
        "特征": [5.0, 10.0],
        "_woe_maps_": {"特征": {"0": 0.1, "1": 0.2, "2": 0.3, "-1": -0.1}},
    }
    X = pd.DataFrame({"特征": [1.0, 7.0, 12.0, np.nan]})
    y = pd.Series([0, 1, 0, 1], name="目标")
    binner = OptimalBinning(method="uniform", n_jobs=1).load(rules)

    binner.fit(X, y)

    np.testing.assert_allclose(binner.transform(X, metric="woe")["特征"], [0.1, 0.2, 0.3, -0.1])


def test_load_update_true_keeps_direct_incremental_update_semantics():
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    old = pd.DataFrame({"old": np.arange(24, dtype=float)})
    new = pd.DataFrame({"new": np.arange(24, dtype=float)})
    binner = OptimalBinning(method="uniform", n_jobs=1).fit(old, y)

    binner.load({"new": [5.0, 10.0, 15.0]}, update=True)

    assert set(binner.export_rules()) == {"old", "new"}
    assert binner.transform(old, metric="indices").shape == (24, 1)
    assert binner.transform(new, metric="indices").shape == (24, 1)


@pytest.mark.parametrize("n_jobs,backend", EXECUTION_MODES)
def test_imported_rules_outside_fit_columns_do_not_skip_optimal_training(n_jobs, backend):
    y = pd.Series(np.tile([0, 1], 12), name="目标")
    old = pd.DataFrame({"old": np.arange(24, dtype=float)})
    new = pd.DataFrame({"new": np.arange(23, -1, -1, dtype=float)})
    binner = OptimalBinning(
        method="uniform", n_jobs=n_jobs, parallel_backend=backend, random_state=17
    ).fit(old, y)
    binner.import_rules({"未使用规则": [5.0, 10.0, 15.0]})

    binner.fit(new, y)

    assert list(binner.splits_) == ["new"]
    assert list(binner.bin_tables_) == ["new"]
    assert binner._binner is not None
    assert binner.transform(new, metric="indices").shape == (24, 1)


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
        cloned = clone(binner)
        restored = pickle.loads(pickle.dumps(binner))
        assert cloned.get_params(deep=False) == params, name
        assert restored.get_params(deep=False) == params, name
        if issubclass(cls, BaseBinning):
            assert inspect.signature(cls.fit) == inspect.signature(cls.fit.__wrapped__), name

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


def test_optimal_prebinning_parameters_survive_sklearn_clone():
    params = {"max_n_bins": 12}
    binner = OptimalBinning(prebinning="quantile", prebinning_params=params)
    cloned = clone(binner)

    assert cloned.prebinning_params == params


def test_optimal_fit_transaction_preserves_keyword_options(mixed_xy):
    X, y = mixed_xy
    source = OptimalBinning(method="uniform", lift_refine=False, or_time_limit=7, n_jobs=1)
    candidate = source._make_fit_transaction_candidate()
    binner = RefinementFailingOptimalBinning(method="best_iv", lift_refine=False, n_jobs=1).fit(X, y)

    assert candidate._fit_control_options["lift_refine"] is False
    assert candidate.kwargs["or_time_limit"] == 7
    assert binner._fit_control_options["lift_refine"] is False


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
        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            created.append(instance)
            return instance

    monkeypatch.setattr(optimal_binning_module, "OptimalBinning", RecordingOptimalBinning)

    parent._fit_with_method_and_prebins(X, y, pre_splits={})

    # 一个是工厂创建的临时分箱器，另一个是其 fit 事务的全新候选。
    assert len(created) == 2
    for fallback in created:
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
