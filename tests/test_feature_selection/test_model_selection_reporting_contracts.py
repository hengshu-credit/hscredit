"""模型型筛选的真实决策指标、条件变量与有界报告回归。"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from hscredit.core.selectors import (
    BorutaSelector,
    CorrSelector,
    FeatureImportanceSelector,
    NullImportanceSelector,
    RFESelector,
    SequentialFeatureSelector,
    StepwiseSelector,
    VIFSelector,
)

FIT_COLUMNS = []


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        FIT_COLUMNS.append(list(X.columns))
        self.classes_ = np.unique(y)
        self.coef_ = np.asarray(
            [[float(str(column).replace("x", "")) + 1 if str(column).startswith("x") else 0.1 for column in X.columns]]
        )
        return self

    def score(self, X, y):
        return 0.7


def finite_second(estimator, X, y):
    return np.nan if X.columns.tolist() == ["x0"] else 1.0


def all_invalid(estimator, X, y):
    return np.nan


@pytest.fixture
def xy():
    rng = np.random.RandomState(37)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X.x0 + X.x1 * 0.3 + rng.normal(size=80) > 0).astype(int)
    return X, y


def test_stepwise_compares_forced_baseline_before_adding_noise():
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=100), rng.normal(size=100)
    y = a + rng.normal(scale=0.1, size=100)
    X = pd.DataFrame({"a": a, "b": b})
    selector = StepwiseSelector(estimator="ols", direction="forward", criterion="aic", include=["a"], n_jobs=1).fit(
        X, y
    )
    assert selector.selected_features_ == ["a"]
    assert selector.initial_criterion_ == pytest.approx(sm.OLS(y, sm.add_constant(X[["a"]])).fit().aic)
    assert np.isnan(selector.scores_["b"])
    assert np.isnan(selector.p_values_["b"])


def test_stepwise_custom_model_does_not_invent_p_values(xy):
    X, y = xy
    selector = StepwiseSelector(
        LogisticRegression(), direction="forward", criterion="aic", max_features=2, n_jobs=1
    ).fit(X, y)
    assert selector.scores_.isna().all()
    assert selector.model_results_ is not None
    assert any(event["动作"] == "最终统计量" and not event["是否有效"] for event in selector.selection_events_)


def test_sequential_rejects_nan_and_supports_splitter(xy):
    X, y = xy
    selector = SequentialFeatureSelector(
        LogisticRegression(), n_features_to_select=1, cv=StratifiedKFold(2), scoring=finite_second, n_jobs=1
    ).fit(X.iloc[:, :2], y)
    assert selector.selected_features_ == ["x1"]
    assert selector.candidate_failures_ == 1
    assert np.isnan(selector.scores_["x0"])
    assert selector.scores_["x1"] == 1
    assert any(not event["是否有效"] for event in selector.selection_events_)


def test_sequential_all_invalid_preserves_previous_fit(xy):
    X, y = xy
    selector = SequentialFeatureSelector(LogisticRegression(), n_features_to_select=1, cv=2, n_jobs=1).fit(X, y)
    before = selector.selected_features_.copy()
    before_events = list(selector.selection_events_)
    selector.set_params(scoring=all_invalid)
    with pytest.raises(ValueError, match="没有有效"):
        selector.fit(X, y)
    assert selector.selected_features_ == before
    assert selector.selection_events_ == before_events


@pytest.mark.parametrize("kind", [RFESelector, SequentialFeatureSelector])
@pytest.mark.parametrize("budget", [1, 3])
def test_forced_variables_are_conditioned_and_count_toward_budget(xy, kind, budget):
    X, y = xy
    FIT_COLUMNS.clear()
    kwargs = {"cv": 2} if kind is SequentialFeatureSelector else {}
    selector = kind(RecordingClassifier(), n_features_to_select=budget, include=["x0"], n_jobs=1, **kwargs).fit(X, y)
    assert len(selector.selected_features_) == budget
    assert "x0" in selector.selected_features_
    assert all("x0" in columns for columns in FIT_COLUMNS)
    assert selector.selected_features_ == [name for name in X.columns if name in selector.selected_features_]
    if kind is SequentialFeatureSelector and budget == 1:
        assert FIT_COLUMNS == []
        assert selector.n_cv_splits_ == 0
    if kind is RFESelector and budget == 1:
        assert FIT_COLUMNS == [["x0"]]
        assert selector.scores_.drop("x0").isna().all()


def test_backward_sequential_skips_search_when_fixed_budget_is_full(xy):
    X, y = xy
    FIT_COLUMNS.clear()
    selector = SequentialFeatureSelector(
        RecordingClassifier(),
        n_features_to_select=1,
        include=["x0"],
        direction="backward",
        cv=1000,
        n_jobs=1,
    ).fit(X, y)
    assert selector.selected_features_ == ["x0"]
    assert FIT_COLUMNS == []
    assert selector.n_cv_splits_ == 0


@pytest.mark.parametrize("kind", [RFESelector, SequentialFeatureSelector])
def test_forced_budget_overflow_is_explicit(xy, kind):
    X, y = xy
    with pytest.raises(ValueError, match="总预算"):
        kind(LogisticRegression(), n_features_to_select=1, include=["x0", "x1"], n_jobs=1).fit(X, y)


@pytest.mark.parametrize("step", [1, 2, 0.4])
def test_rfe_matches_sklearn_and_retains_actual_importances(xy, step):
    X, y = xy
    model = LogisticRegression(max_iter=200)
    reference = RFE(model, n_features_to_select=2, step=step).fit(X, y)
    selector = RFESelector(model, n_features_to_select=2, step=step, report_history="full", n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(selector.ranking_.to_numpy(), reference.ranking_)
    assert selector.selected_features_ == X.columns[reference.support_].tolist()
    assert selector.decision_importances_.notna().all()
    assert "模型特征重要性" in selector.importance_history_["指标名称"].unique()
    assert selector.estimator_.n_features_in_ == 2


def test_vif_keeps_infinite_removal_value_and_stopping_state():
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0), "c": np.arange(40.0)})
    selector = VIFSelector(max_iter=1, n_jobs=1).fit(X)
    assert np.isinf(selector.dropped_["VIF值"].iloc[0])
    assert np.isinf(selector.decision_vif_[selector.removed_features_[0]])
    assert not selector.converged_
    assert selector.selection_stopping_reason_ == "达到最大迭代次数"
    assert selector.unresolved_features_


def test_importance_top_k_has_rank_reason_and_saved_model(xy):
    X, y = xy
    selector = FeatureImportanceSelector(LogisticRegression(), threshold=1, n_jobs=1).fit(X, y)
    assert selector.selection_mode_ == "前K个"
    assert "前 1 个" in selector.dropped_["剔除原因"].iloc[0]
    assert selector.estimator_.n_features_in_ == X.shape[1]
    assert sorted(selector.ranking_) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("runs", [0, -1, True, 1.2])
def test_null_importance_rejects_invalid_experiment_count(xy, runs):
    X, y = xy
    with pytest.raises(ValueError, match="n_runs"):
        NullImportanceSelector(LogisticRegression(), n_runs=runs, cv=2, n_jobs=1).fit(X, y)


def test_bounded_history_does_not_change_null_importance_result(xy):
    X, y = xy
    selector = NullImportanceSelector(LogisticRegression(), cv=2, n_runs=3, random_state=5, n_jobs=1)
    complete = clone(selector).fit(X, y)
    bounded = clone(selector).set_params(max_report_events=2, report_history="full").fit(X, y)
    pd.testing.assert_series_equal(complete.scores_, bounded.scores_)
    assert len(bounded.selection_events_) <= 2
    assert bounded.importance_runs_stored_ == 0
    assert bounded.importance_runs_truncated_ == 6


def test_boruta_has_effective_statistical_threshold_and_categorical_support():
    X = pd.DataFrame({"类别": pd.Series(["甲", "乙"] * 20, dtype="category"), "x0": np.arange(40.0)})
    y = np.tile([0, 1], 20)
    selector = BorutaSelector(n_estimators=5, max_iter=5, alpha=0.1, n_jobs=1).fit(X, y)
    assert selector.effective_threshold_ == 0.05
    assert selector.corrected_alpha_ == 0.05
    assert selector.support_.equals(selector.p_values_ <= 0.05)
    assert any(event["指标名称"] == "二项检验p值" for event in selector.selection_events_)


def test_corr_explains_conflicting_feature_without_calling_weight_correlation():
    X = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0), "c": np.tile([0.0, 1.0], 15)})
    selector = CorrSelector(weights={"a": 3, "b": 2, "c": 1}, n_jobs=1).fit(X)
    assert selector.scores_["b"] == 2
    assert selector.decision_correlations_["b"] == pytest.approx(1)
    assert selector.association_features_["b"] == "a"
    assert selector.score_name_ == "保留优先权重"


@pytest.mark.parametrize(
    "kind",
    [
        CorrSelector,
        VIFSelector,
        FeatureImportanceSelector,
        NullImportanceSelector,
        RFESelector,
        SequentialFeatureSelector,
        BorutaSelector,
        StepwiseSelector,
    ],
)
def test_model_selector_new_parameters_are_cloneable(kind):
    kwargs = (
        {"estimator": LogisticRegression()}
        if kind in {FeatureImportanceSelector, NullImportanceSelector, RFESelector, SequentialFeatureSelector}
        else {}
    )
    selector = kind(target_rm=True, report_history="full", max_report_events=123, max_report_bytes=4567, **kwargs)
    copied = clone(selector)
    assert copied.target_rm is True
    assert copied.report_history == "full"
    assert copied.max_report_events == 123
    assert copied.max_report_bytes == 4567


@pytest.mark.parametrize(
    "kind",
    [
        CorrSelector,
        VIFSelector,
        FeatureImportanceSelector,
        NullImportanceSelector,
        RFESelector,
        SequentialFeatureSelector,
        BorutaSelector,
        StepwiseSelector,
    ],
)
def test_byte_budget_changes_history_not_selection(xy, kind):
    X, y = xy
    kwargs = {"n_jobs": 1}
    if kind is CorrSelector:
        kwargs.update(weights={name: 1 for name in X.columns}, binning_params=None)
    elif kind is FeatureImportanceSelector:
        kwargs.update(estimator=LogisticRegression(), threshold=2)
    elif kind is NullImportanceSelector:
        kwargs.update(estimator=LogisticRegression(), n_runs=2, cv=2)
    elif kind is RFESelector:
        kwargs.update(estimator=LogisticRegression(), n_features_to_select=2)
    elif kind is SequentialFeatureSelector:
        kwargs.update(estimator=LogisticRegression(), n_features_to_select=2, cv=2)
    elif kind is BorutaSelector:
        kwargs.update(n_estimators=5, max_iter=3)
    elif kind is StepwiseSelector:
        kwargs.update(direction="forward", max_features=2)
    full = kind(**kwargs).fit(X, y)
    bounded = kind(**kwargs, max_report_bytes=64, report_history="full").fit(X, y)
    assert bounded.selected_features_ == full.selected_features_
    pd.testing.assert_series_equal(bounded.scores_, full.scores_)
    assert bounded.report_events_bytes_ <= 64
    assert bounded.report_events_truncated_ > 0
    assert (
        bounded.report_events_truncated_ == bounded.report_candidates_truncated_ + bounded.report_decisions_truncated_
    )
    assert bounded.selection_events_ == []
