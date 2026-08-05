"""特征筛选器统一前置分箱测试。"""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import OptimalBinning
from hscredit.core import selectors
from hscredit.core.selectors import CorrSelector, IVSelector, ScorecardFeatureSelection
from hscredit.core.selectors.base import BaseFeatureSelector
from hscredit.exceptions import ValidationError


class CaptureSelector(BaseFeatureSelector):
    """记录传入具体筛选逻辑的数据。"""

    def _fit_impl(self, X, y):
        self.fit_X_ = X.copy()
        self.selected_features_ = X.columns.tolist()
        self.scores_ = pd.Series(1.0, index=X.columns)


class RecordingBinner:
    """记录训练次数并返回可预测分箱 index 的测试分箱器。"""

    def __init__(self, fitted=False):
        self._is_fitted = fitted
        self.fit_calls = 0
        self.metrics = []
        self.fit_columns = []
        self.transform_columns = []

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.fit_columns.append(list(X.columns))
        self._is_fitted = True
        return self

    def transform(self, X, metric="indices"):
        self.metrics.append(metric)
        self.transform_columns.append(list(X.columns))
        return pd.DataFrame(
            {column: np.arange(len(X)) % 2 for column in X.columns},
            index=X.index,
        )


class NoTransformBinner:
    """只有训练能力、没有转换能力的非法分箱器。"""

    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y=None):
        self._is_fitted = True
        return self


@pytest.fixture
def sample_xy():
    X = pd.DataFrame(
        {
            "特征一": np.arange(20, dtype=float),
            "特征二": np.arange(20, dtype=float)[::-1],
        },
        index=pd.Index(range(100, 120), name="样本号"),
    )
    y = pd.Series([0, 1] * 10, index=X.index)
    return X, y


@pytest.fixture
def corr_xy():
    rng = np.random.RandomState(42)
    signal = np.linspace(-3.0, 3.0, 200)
    X = pd.DataFrame(
        {
            "主特征": signal,
            "相关特征": signal + rng.normal(0.0, 0.03, len(signal)),
            "随机特征": rng.normal(0.0, 1.0, len(signal)),
        }
    )
    y = pd.Series((signal + rng.normal(0.0, 0.4, len(signal)) > 0).astype(int))
    return X, y


def test_unfitted_binner_is_fitted_once_and_indices_reach_selector(sample_xy):
    """防止未训练分箱器未拟合，或原始值绕过 index 转换。"""
    X, y = sample_xy
    binner = RecordingBinner()

    selector = CaptureSelector(binner=binner).fit(X, y)

    assert binner.fit_calls == 1
    assert binner.metrics == ["indices"]
    assert selector.fit_X_["特征一"].tolist() == [0, 1] * 10
    assert selector.fit_X_.index.equals(X.index)


def test_fitted_binner_is_reused_without_refit(sample_xy):
    """防止已训练分箱规则被选择器意外覆盖。"""
    X, y = sample_xy
    binner = RecordingBinner(fitted=True)

    CaptureSelector(binner=binner).fit(X, y)

    assert binner.fit_calls == 0
    assert binner.metrics == ["indices"]


def test_binner_priority_ignores_invalid_binning_params(sample_xy):
    """防止低优先级参数覆盖或阻断显式 binner。"""
    X, y = sample_xy
    binner = RecordingBinner(fitted=True)

    selector = CaptureSelector(
        binner=binner,
        binning_params="该参数应被忽略",
    ).fit(X, y)

    assert selector._binner_instance is binner
    assert binner.fit_calls == 0


def test_binner_class_is_rejected_with_chinese_error(sample_xy):
    """防止类对象绕过实例配置并以默认参数构造。"""
    X, y = sample_xy

    with pytest.raises(ValidationError, match="分箱器实例"):
        CaptureSelector(binner=OptimalBinning).fit(X, y)


def test_binning_params_create_independent_optimal_binner(sample_xy):
    """防止参数字典未创建分箱器或被原地修改。"""
    X, y = sample_xy
    params = {"method": "uniform", "max_n_bins": 2, "min_n_bins": 2}
    snapshot = params.copy()

    selector = CaptureSelector(binning_params=params).fit(X, y)

    assert isinstance(selector._binner_instance, OptimalBinning)
    assert selector._binner_instance.method == "uniform"
    assert params == snapshot
    assert set(selector.fit_X_.stack().unique()).issubset({0, 1})


@pytest.mark.parametrize("bad_params", ["uniform", ["method", "uniform"], 3])
def test_invalid_binning_params_are_rejected(bad_params, sample_xy):
    """防止非字典参数进入 OptimalBinning 构造流程。"""
    X, y = sample_xy

    with pytest.raises(ValidationError, match="binning_params 分箱参数必须是字典"):
        CaptureSelector(binning_params=bad_params).fit(X, y)


def test_binner_without_transform_or_apply_is_rejected(sample_xy):
    """防止无转换能力的对象被静默当成未分箱数据。"""
    X, y = sample_xy

    with pytest.raises(ValidationError, match="transform 或 apply"):
        CaptureSelector(binner=NoTransformBinner()).fit(X, y)


def test_ordinary_selector_skips_forced_fields_before_binning_and_selection():
    X = pd.DataFrame(
        {
            "强制保留": np.arange(20, dtype=float),
            "普通字段": np.arange(20, dtype=float)[::-1],
            "显式剔除": np.ones(20),
            "强制删除": np.zeros(20),
        }
    )
    y = pd.Series([0, 1] * 10)
    binner = RecordingBinner()

    selector = CaptureSelector(
        include=["强制保留"],
        exclude=["显式剔除"],
        force_drop=["强制删除"],
        binner=binner,
    ).fit(X, y)

    assert binner.fit_columns == [["普通字段"]]
    assert binner.transform_columns == [["普通字段"]]
    assert selector.fit_X_.columns.tolist() == ["普通字段"]
    assert selector.selected_features_ == ["普通字段", "强制保留"]
    assert selector.feature_names_in_.tolist() == list(X.columns)
    assert selector.n_features_in_ == X.shape[1]
    assert set(selector.forced_dropped_) == {"显式剔除", "强制删除"}


def test_corr_include_participates_but_force_drop_skips_pre_binning():
    X = pd.DataFrame(
        {
            "强制保留": np.arange(20, dtype=float),
            "普通字段": np.arange(20, dtype=float) * 2,
            "强制删除": np.arange(20, dtype=float) * 3,
        }
    )
    y = pd.Series([0, 1] * 10)
    binner = RecordingBinner()

    selector = CorrSelector(
        threshold=0.7,
        method="pearson",
        weights={"强制保留": 1.0, "普通字段": 5.0, "强制删除": 10.0},
        include=["强制保留"],
        force_drop=["强制删除"],
        binner=binner,
        binning_params=None,
        n_jobs=1,
    ).fit(X, y)

    assert binner.fit_columns == [["强制保留", "普通字段"]]
    assert binner.transform_columns == [["强制保留", "普通字段"]]
    assert selector.selected_features_ == ["强制保留"]
    assert selector.dropped_.loc[
        selector.dropped_["特征"] == "普通字段", "相关特征"
    ].tolist() == ["强制保留"]


def test_all_forced_fields_skip_binner_and_finish_with_complete_metadata():
    X = pd.DataFrame({"保留": [1.0, 2.0], "删除": [3.0, 4.0]})
    binner = RecordingBinner()

    selector = CaptureSelector(
        include=["保留"],
        exclude=["删除"],
        binner=binner,
    ).fit(X)

    assert binner.fit_calls == 0
    assert binner.metrics == []
    assert selector.selected_features_ == ["保留"]
    assert selector.forced_dropped_ == ["删除"]
    assert selector.feature_names_in_.tolist() == ["保留", "删除"]
    assert selector.n_features_in_ == 2


def test_iv_selector_computes_iv_from_uniform_bin_indices():
    """防止 IVSelector 仍按连续原值而非分箱 index 计算。"""
    X = pd.DataFrame({"连续变量": np.arange(1, 9, dtype=float)})
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])

    selector = IVSelector(
        threshold=0.0,
        regularization=1.0,
        binning_params={"method": "uniform", "max_n_bins": 2, "min_n_bins": 2},
    ).fit(X, y)

    assert selector.scores_["连续变量"] == pytest.approx(0.462098, rel=1e-5)
    assert selector.transform(X).equals(X)


def test_corr_selector_uses_default_best_iv_binner_and_same_bin_tables(corr_xy):
    """防止 CorrSelector 使用另一套隐藏分箱器计算指标。"""
    X, y = corr_xy

    selector = CorrSelector(threshold=0.8).fit(X, y)

    binner = selector._binner_instance
    assert binner.method == "best_iv"
    assert binner.max_n_bins == 5
    assert binner.min_bin_size == 0.01
    expected = pd.Series(
        {
            column: binner.bin_tables_[column]["指标IV值"].iloc[0]
            for column in X.columns
        }
    )
    pd.testing.assert_series_equal(
        selector.scores_.sort_index(),
        expected.sort_index(),
        check_names=False,
    )


def test_corr_selector_without_target_skips_only_constructor_default():
    """防止默认监督分箱破坏 CorrSelector.fit(X) 兼容路径。"""
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

    selector = CorrSelector(threshold=0.8).fit(X)

    assert selector._binner_instance is None
    assert len(selector.selected_features_) == 1


def test_scorecard_outer_binner_is_not_reapplied_by_internal_corr(sample_xy):
    """防止组合筛选器对外层分箱 index 再次分箱。"""
    X, y = sample_xy
    binner = RecordingBinner()
    selector = ScorecardFeatureSelection(
        null_threshold=None,
        iv_threshold=0.0,
        corr_threshold=0.8,
        mode_threshold=None,
        binner=binner,
    )

    selector.fit(X, y)

    assert binner.fit_calls == 1
    assert selector.stage_selectors_["corr"]._binner_instance is None


def test_all_exported_selectors_expose_common_binning_parameters():
    """防止任一公开筛选器因构造签名缺失而无法使用统一分箱。"""
    missing = []
    for name in selectors.__all__:
        selector_class = getattr(selectors, name, None)
        if not inspect.isclass(selector_class):
            continue
        if not issubclass(selector_class, BaseFeatureSelector):
            continue
        parameters = inspect.signature(selector_class.__init__).parameters
        if "binner" not in parameters or "binning_params" not in parameters:
            missing.append(name)

    assert missing == []


def test_selector_clone_preserves_independent_binning_configuration():
    """防止 sklearn clone 丢失或共享可变分箱参数。"""
    selector = IVSelector(
        threshold=0.01,
        binning_params={"method": "uniform", "max_n_bins": 3},
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )

    cloned = clone(selector)

    assert cloned.binner is None
    assert cloned.binning_params == {"method": "uniform", "max_n_bins": 3}
    assert cloned.binning_params is not selector.binning_params
    assert cloned.n_jobs == 0.5
    assert cloned.parallel_backend == "threading"
    assert cloned.parallel_config == {"batch_size": 1}
    assert cloned.parallel_config is not selector.parallel_config
