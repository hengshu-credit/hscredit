"""统计筛选器报告证据、稳定策略与性能路径的专项回归。"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.model_selection import KFold

from hscredit.core.selectors import (
    CardinalitySelector,
    Chi2Selector,
    FTestSelector,
    IVSelector,
    KSSelector,
    LiftSelector,
    ModeSelector,
    MutualInfoSelector,
    NullSelector,
    PSISelector,
    RegexSelector,
    StabilityAwareSelector,
    TypeSelector,
    VarianceSelector,
)
from hscredit.core.selectors.iv_selector import _compute_iv_single
from hscredit.core.selectors.lift_selector import _compute_lift_single
from hscredit.core.selectors.psi_selector import _compute_psi_single


@pytest.fixture
def numeric_data():
    return pd.DataFrame(
        {"甲": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], "乙": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "常量": [1.0] * 6}
    ), np.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "factory",
    [
        NullSelector,
        ModeSelector,
        CardinalitySelector,
        TypeSelector,
        lambda **kw: RegexSelector(pattern="甲", **kw),
        VarianceSelector,
        Chi2Selector,
        FTestSelector,
        MutualInfoSelector,
        IVSelector,
        KSSelector,
        LiftSelector,
        PSISelector,
        StabilityAwareSelector,
    ],
)
def test_all_statistical_constructors_support_explicit_target_removal(factory, numeric_data):
    X, y = numeric_data
    for target_rm in (False, True):
        selector = factory(n_jobs=1, target="目标", target_rm=target_rm)
        assert clone(selector).target_rm == target_rm
        transformed = selector.fit_transform(X.assign(目标=y))
        assert ("目标" in transformed) is not target_rm


@pytest.mark.parametrize(
    "factory,parameter",
    [
        (NullSelector, "threshold"),
        (ModeSelector, "threshold"),
        (CardinalitySelector, "threshold"),
        (VarianceSelector, "threshold"),
        (Chi2Selector, "threshold"),
        (FTestSelector, "threshold"),
        (MutualInfoSelector, "threshold"),
        (IVSelector, "threshold"),
        (KSSelector, "threshold"),
        (LiftSelector, "ratio"),
        (PSISelector, "threshold"),
        (StabilityAwareSelector, "iv_weight"),
    ],
)
def test_nan_parameters_rejected_even_when_every_feature_forced(factory, parameter, numeric_data):
    X, y = numeric_data
    with pytest.raises(ValueError):
        factory(n_jobs=1, include=X.columns.tolist(), **{parameter: np.nan}).fit(X, y)


def test_real_variance_and_rule_specific_drop_reasons(numeric_data):
    X, y = numeric_data
    variance = VarianceSelector(n_jobs=1).fit(X)
    assert variance.scores_["甲"] == pytest.approx(11.666666666666666)
    assert variance.variances_["甲"] == variance.scores_["甲"]
    assert variance.ranges_["甲"] == 10
    regex = RegexSelector("甲", invert=True, n_jobs=1).fit(X)
    assert "匹配排除" in regex.dropped_.iloc[0]["剔除原因"]
    f_test = FTestSelector(k=1, n_jobs=1).fit(X, y)
    assert f_test.dropped_.set_index("特征").loc["乙", "剔除原因"] == "未进入前1名"
    chi = Chi2Selector(k=2, threshold=1e10, n_jobs=1).fit(X, y)
    assert "未进入" not in chi.dropped_.set_index("特征").loc["甲", "剔除原因"]


def test_null_mode_cardinality_metadata_and_boundary():
    X = pd.DataFrame({"有缺失": [1.0, 1.0, 1.0, 1.0, np.nan], "集中": [1, 1, 1, 1, 2]})
    null = NullSelector(threshold=0.2).fit(X)
    assert "有缺失" not in null.selected_features_
    assert null.missing_counts_["有缺失"] == 1
    mode = ModeSelector(threshold=0.8, dropna=True).fit(X)
    assert mode.mode_counts_["有缺失"] == 4
    assert mode.denominator_counts_["有缺失"] == 4
    assert "集中" not in mode.selected_features_
    card = CardinalitySelector(threshold=2, dropna=False).fit(X)
    assert card.cardinalities_.to_dict() == {"有缺失": 2, "集中": 2}


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_categorical_chi2_is_permutation_invariant_and_matches_contingency(dtype):
    X = pd.DataFrame({"类别": pd.Series(["a", "a", "a", "b", "b", "c", "c", "c"], dtype=dtype)})
    y = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 1, 2, 5, 6, 7]
    original = Chi2Selector(n_jobs=1).fit(X, y)
    permuted = Chi2Selector(n_jobs=1).fit(X.iloc[order].reset_index(drop=True), y[order])
    expected = chi2_contingency([[2, 1], [1, 1], [0, 3]], correction=False)
    assert original.scores_.iloc[0] == pytest.approx(expected[0])
    assert original.p_values_.iloc[0] == pytest.approx(expected[1])
    pd.testing.assert_series_equal(original.scores_, permuted.scores_)
    assert original.test_methods_.iloc[0] == "类别列联表卡方检验"


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_chi2_missing_removal_precedes_category_encoding(dtype):
    X = pd.DataFrame({"类别": pd.Series(["a", "b", None, "b", "a", "b"], dtype=dtype)})
    y = np.array([0, 1, 1, 0, 0, 1])
    excluded = Chi2Selector(missing=None, n_jobs=1).fit(X, y)
    separate = Chi2Selector(n_jobs=1).fit(X, y)
    assert excluded.effective_counts_.iloc[0] == 5
    assert separate.effective_counts_.iloc[0] == 6
    expected = chi2_contingency([[2, 0], [1, 2]], correction=False)[0]
    assert excluded.scores_.iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_f_test_requires_explicit_unordered_category_strategy(dtype):
    X = pd.DataFrame({"类别": pd.Series(["a", "b", "a", "b", "a", "b"], dtype=dtype)})
    y = [0, 0, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="无序类别"):
        FTestSelector(n_jobs=1).fit(X, y)
    legacy = FTestSelector(categorical_strategy="legacy_ordinal", n_jobs=1).fit(X, y)
    expected, probability = f_classif(pd.factorize(X["类别"])[0].reshape(-1, 1), y)
    assert legacy.scores_.iloc[0] == pytest.approx(expected[0])
    assert legacy.p_values_.iloc[0] == pytest.approx(probability[0])


def test_ordered_categorical_f_test_uses_declared_order():
    X = pd.DataFrame(
        {"等级": pd.Categorical(["中", "高", "低", "高", "低", "中"], categories=["低", "中", "高"], ordered=True)}
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    selected = FTestSelector(n_jobs=1).fit(X, y)
    expected, _ = f_classif(X["等级"].cat.codes.to_numpy().reshape(-1, 1), y)
    assert selected.scores_.iloc[0] == pytest.approx(expected[0])


@pytest.mark.parametrize("percentile", [25, 50, 75, 100])
def test_percentile_reuses_f_scores(monkeypatch, percentile):
    import hscredit.core.selectors.f_test_selector as module

    rng = np.random.RandomState(12)
    X, y = pd.DataFrame(rng.normal(size=(80, 8))), rng.randint(0, 2, 80)
    expected = SelectPercentile(percentile=percentile).fit(X, y).get_support()
    calls = []
    real = module.f_classif

    def capture(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "f_classif", capture)
    fitted = FTestSelector(percentile=percentile, n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(fitted.get_support(), expected)
    assert len(calls) == 1
    if percentile == 100:
        assert fitted.percentile_values_ is None
    else:
        reconstructed = fitted.percentile_values_.to_numpy() > fitted.percentile_cutoff_
        ties = np.flatnonzero(fitted.percentile_values_.to_numpy() == fitted.percentile_cutoff_)
        reconstructed[ties[: fitted.percentile_tie_slots_]] = True
        np.testing.assert_array_equal(reconstructed, expected)


def test_f_statistics_preserve_perfect_separation():
    with pytest.warns(RuntimeWarning):
        fitted = FTestSelector(n_jobs=1).fit(pd.DataFrame({"完美": [0, 0, 1, 1]}), [0, 0, 1, 1])
    assert np.isposinf(fitted.scores_.iloc[0])
    assert fitted.p_values_.iloc[0] == 0


def test_percentile_with_perfect_separation_keeps_highest_infinite_scores():
    X = pd.DataFrame({"完美甲": [0, 0, 1, 1], "完美乙": [0, 0, 1, 1], "弱": [0, 1, 0, 1], "常量": [1] * 4})
    with pytest.warns(RuntimeWarning):
        fitted = FTestSelector(percentile=50, n_jobs=1).fit(X, [0, 0, 1, 1])
    assert fitted.selected_features_ == ["完美甲", "完美乙"]


def test_contingency_chi2_rejects_continuous_labels():
    with pytest.raises(ValueError, match="离散分类"):
        Chi2Selector(n_jobs=1).fit(pd.DataFrame({"类别": ["a", "b", "c", "b"]}), [0.1, 0.2, 0.4, 0.3])


@pytest.mark.parametrize("regularization", [0.1, 1.0, 5.0])
def test_vectorized_iv_preserves_additive_smoothing(regularization):
    rng = np.random.RandomState(4)
    values = rng.randint(0, 43, 700).astype(float)
    values[::13] = np.nan
    target = rng.randint(0, 2, len(values))
    keep = ~np.isnan(values)
    data = pd.DataFrame({"x": values[keep], "y": target[keep]})
    groups = data.groupby("x")["y"].agg(["sum", "count"])
    events = (groups["sum"] + regularization) / (groups["sum"].sum() + len(groups) * regularization)
    good = groups["count"] - groups["sum"]
    good = (good + regularization) / (good.sum() + len(groups) * regularization)
    expected = ((events - good) * np.log(events / good)).sum()
    assert _compute_iv_single(values, target, regularization) == pytest.approx(expected)


def test_iv_continuous_high_cardinality_and_missing_metadata():
    X = pd.DataFrame({"连续": np.arange(10_000, dtype=float)})
    y = np.arange(len(X)) % 2
    fitted = IVSelector(n_jobs=1, threshold=float("-inf")).fit(X, y)
    assert fitted.category_counts_.iloc[0] == 10_000
    assert fitted.valid_counts_.iloc[0] == 10_000
    assert np.isfinite(fitted.scores_.iloc[0])


def test_fractional_lift_is_tie_permutation_invariant_and_retains_legacy():
    x = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 0, 0, 1, 1, 0])
    order = [0, 1, 2, 5, 3, 4]
    assert _compute_lift_single(x, y, 0.2) == pytest.approx(2)
    assert _compute_lift_single(x[order], y[order], 0.2) == pytest.approx(2)
    assert _compute_lift_single(x, y, 0.2, tie_policy="legacy") == pytest.approx(1.5)
    assert _compute_lift_single(x[order], y[order], 0.2, tie_policy="legacy") == pytest.approx(3)
    fitted = LiftSelector(ratio=0.2, n_jobs=1).fit(pd.DataFrame({"同值": x}), y)
    assert fitted.actual_coverage_.iloc[0] == pytest.approx(2 / 6)
    assert fitted.lift_detail_.iloc[0]["头部样本量"] == 2


def test_lift_missing_exclusion_and_binary_validation():
    X = pd.DataFrame({"有缺失": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]})
    y = [0, 0, 0, 0, 1, 1]
    fitted = LiftSelector(ratio=0.3, n_jobs=1).fit(X, y)
    assert fitted.effective_counts_.iloc[0] == 4
    assert fitted.scores_.iloc[0] == 0
    legacy = LiftSelector(ratio=0.3, n_jobs=1, tie_policy="legacy", missing_policy="legacy").fit(X, y)
    assert legacy.scores_.iloc[0] == 2
    with pytest.raises(ValueError, match="0/1"):
        LiftSelector(n_jobs=1).fit(X, [0, 0, 0, 0, 2, 2])


def test_psi_fold_summary_matches_kfold_without_retained_row_indices():
    X = pd.DataFrame({"连续": np.random.RandomState(7).normal(size=103)})
    fitted = PSISelector(n_splits=4, random_state=8, n_jobs=1).fit(X)
    expected = [
        _compute_psi_single(X.iloc[a, 0].to_numpy(), X.iloc[b, 0].to_numpy())
        for a, b in KFold(4, shuffle=True, random_state=8).split(X)
    ]
    np.testing.assert_allclose(fitted.fold_scores_.iloc[0].to_numpy(), expected)
    assert fitted.scores_.iloc[0] == pytest.approx(np.mean(expected))
    assert fitted.psi_std_.iloc[0] == pytest.approx(np.std(expected))
    assert fitted.reference_counts_.tolist() == [77, 77, 77, 78]
    assert fitted.comparison_counts_.tolist() == [26, 26, 26, 25]
    assert not hasattr(fitted, "splits_")


def test_stability_reports_actual_combined_threshold_and_conditions(numeric_data):
    X, y = numeric_data
    fitted = StabilityAwareSelector(oot_df=X, score_threshold=0.2, n_jobs=1).fit(X, y)
    assert fitted.threshold_ == 0.2
    assert list(fitted.condition_results_) == ["IV达标", "PSI达标", "综合评分达标"]
    assert fitted.reference_counts_.eq(len(X)).all()
    assert fitted.psi_normalizer_ == 1e-10


@pytest.mark.parametrize(
    "factory,removed,expected",
    [
        (Chi2Selector, ["categorical_strategy"], {"categorical_strategy": "legacy_ordinal"}),
        (FTestSelector, ["categorical_strategy"], {"categorical_strategy": "legacy_ordinal"}),
        (LiftSelector, ["tie_policy", "missing_policy"], {"tie_policy": "legacy", "missing_policy": "legacy"}),
    ],
)
def test_legacy_artifact_preserves_algorithm_policy(factory, removed, expected):
    original = factory(n_jobs=1)
    state = original.__getstate__()
    for key in removed:
        state.pop(key)
    restored = factory.__new__(factory)
    with pytest.warns(UserWarning, match="旧"):
        restored.__setstate__(state)
    for name, value in expected.items():
        assert getattr(restored, name) == value
    assert clone(restored).get_params()[removed[0]] == expected[removed[0]]
