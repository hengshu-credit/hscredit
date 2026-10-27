"""分箱算法审计反例：独立计数、置换不变性和公共参数契约。"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import (
    BestIVBinning,
    BestKSBinning,
    CPSATBinning,
    ChiMergeBinning,
    MDLPBinning,
    OptimalBinning,
    ORBinning,
    TargetBadRateBinning,
)
from hscredit.core.metrics import compute_bin_stats
from hscredit.core.metrics._binning import iv_for_splits, ks_for_splits
from hscredit.exceptions import ParallelExecutionError


def population(bad_counts, counts=None):
    """按每种数值的原始好坏人数构造可手算样本。"""
    counts = [20] * len(bad_counts) if counts is None else counts
    X = pd.DataFrame({"变量": np.repeat(np.arange(len(counts), dtype=float), counts)})
    y = pd.Series(np.concatenate([np.r_[np.ones(b), np.zeros(n - b)] for b, n in zip(bad_counts, counts)]))
    return X, y.astype(int)


def raw_groups(model, X, y):
    ids = model.transform(X)[X.columns[0]].to_numpy()
    return pd.DataFrame({"箱": ids, "标签": np.asarray(y)}).groupby("箱")["标签"].agg(["size", "sum", "mean"])


@pytest.mark.parametrize("method", ["cp_sat", "or_tools"])
def test_row_permutation_does_not_change_partition(method):
    X, y = population([2, 3, 4, 5, 6, 8, 10, 12, 13, 15, 16, 18])
    reference = None
    for order in [np.arange(len(y)), np.random.default_rng(42).permutation(len(y)), np.arange(len(y))[::-1]]:
        binner = OptimalBinning(
            method=method,
            max_n_bins=4,
            min_bin_size=0.1,
            monotonic=False,
            random_state=42,
            n_jobs=1,
            lift_refine=False,
        ).fit(X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True))
        ids = binner.transform(X)["变量"].to_numpy()
        if reference is None:
            reference = ids
        np.testing.assert_array_equal(ids, reference)


@pytest.mark.parametrize("method", ["chi", "best_iv", "best_ks", "cp_sat", "or_tools", "kmeans"])
def test_low_cardinality_keeps_feasible_boundaries(method):
    X, y = population([10, 50, 90], [100, 100, 100])
    binner = OptimalBinning(
        method=method,
        max_n_bins=5,
        min_n_bins=3,
        min_bin_size=0.1,
        max_bin_size=0.4,
        n_jobs=1,
        random_state=42,
    ).fit(X, y)
    assert raw_groups(binner, X, y)["size"].tolist() == [100, 100, 100]
    assert binner.n_bins_["变量"] == 3


@pytest.mark.parametrize("method", OptimalBinning.VALID_METHODS)
def test_common_precision_survives_clone_and_fit(method):
    binner = OptimalBinning._method_classes()[method](decimal=8, n_jobs=1)
    assert binner.decimal == clone(binner).decimal == 8


def test_control_options_survive_external_clone_and_set_params():
    binner = OptimalBinning(method="tree", lift_refine=False, lift_focus_weight=1.25)
    assert clone(binner)._fit_control_options == binner._fit_control_options
    binner.set_params(lift_refine=True, lift_focus_weight=2.0)
    assert clone(binner)._fit_control_options == binner._fit_control_options


def test_default_wrapper_does_not_run_lift_refinement(monkeypatch):
    X, y = population([2, 8, 4, 12, 17, 6])

    def unexpected(*args, **kwargs):
        pytest.fail("默认分箱不得调用额外 Lift 调整")

    monkeypatch.setattr(OptimalBinning, "_refine_splits_for_lift_stability", unexpected)
    OptimalBinning(method="tree", n_jobs=1).fit(X, y)


@pytest.mark.parametrize("method", ["uniform", "quantile", "tree", "best_iv", "best_ks"])
def test_decimal_is_display_precision_and_never_collapses_training_bins(method):
    X, y = population([2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
    X["变量"] *= 1e-6
    results = []
    for digits in [4, 8]:
        binner = OptimalBinning(method=method, decimal=digits, n_jobs=1, random_state=42).fit(X, y)
        assert np.all(np.diff(binner.splits_["变量"]) > 0)
        assert binner.n_bins_["变量"] == len(raw_groups(binner, X, y)) >= 2
        results.append(binner.transform(X)["变量"].to_numpy())
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("method", ["tree", "best_iv", "monotonic", "cp_sat", "quantile"])
def test_impossible_direction_is_not_returned_as_success(method):
    X, y = population([2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
    with pytest.raises((ValueError, ParallelExecutionError), match="约束|单调|可行"):
        OptimalBinning(
            method=method,
            min_n_bins=3,
            max_n_bins=4,
            monotonic="descending",
            n_jobs=1,
        ).fit(X, y)


@pytest.mark.parametrize("method", ["quantile", "tree", "chi", "best_iv", "or_tools", "cp_sat"])
def test_impossible_size_is_not_silently_relaxed(method):
    X, y = population([20, 2], [90, 10])
    with pytest.raises((ValueError, ParallelExecutionError), match="约束|样本|可行"):
        OptimalBinning(method=method, min_n_bins=2, min_bin_size=0.2, n_jobs=1).fit(X, y)


def test_search_metrics_use_the_same_left_closed_partition_as_transform():
    X, y = population([1, 7, 8], [10, 10, 10])
    x = X["变量"].to_numpy()
    yy = y.to_numpy()
    bins = np.searchsorted([1.0], x, side="right")
    table = compute_bin_stats(bins, yy, round_digits=False)
    expected_iv = float(table["分档IV值"].sum())
    expected_ks = float(table["分档KS值"].max())
    bad = np.cumsum(yy)
    good = np.arange(1, len(y) + 1) - bad
    assert iv_for_splits(x, yy, [1.0]) == pytest.approx(expected_iv)
    assert ks_for_splits(x, yy, [1.0]) == pytest.approx(expected_ks)
    assert BestIVBinning(min_bin_size=1)._calc_iv_fast(
        x, yy, good, bad, int((yy == 0).sum()), int(yy.sum()), [1.0]
    ) == pytest.approx(expected_iv)
    assert BestKSBinning(min_bin_size=1)._calc_ks_fast(
        x, yy, good, bad, int((yy == 0).sum()), int(yy.sum()), [1.0]
    ) == pytest.approx(expected_ks)


def test_chi_bin_cap_uses_statistical_merges_not_left_truncation():
    X, y = population([10, 90, 20, 80, 30, 70, 40, 60], [100] * 8)
    binner = ChiMergeBinning(max_n_bins=4, min_bin_size=0.05, n_jobs=1).fit(X, y)
    assert binner.splits_["变量"].tolist() != [4.0, 5.0, 6.0]
    assert binner.n_bins_["变量"] <= 4


@pytest.mark.parametrize("wrapped", [False, True])
def test_explicit_target_bad_rates_clone_and_fit(wrapped):
    X, y = population([10, 50, 90], [100] * 3)
    rates = [0.1, 0.5, 0.9]
    options = dict(target_bad_rates=rates, min_n_bins=3, max_n_bins=3, min_bin_size=0.1, n_jobs=1)
    binner = OptimalBinning(method="target_bad_rate", **options) if wrapped else TargetBadRateBinning(**options)
    clone(binner).fit(X, y)
    binner.fit(X, y)
    assert rates == [0.1, 0.5, 0.9]
    np.testing.assert_allclose(raw_groups(binner, X, y)["mean"], rates)


def test_mdlp_forced_minimum_does_not_force_maximum():
    x, y = np.arange(200.0), np.tile([0, 1], 100)
    binner = MDLPBinning(min_n_bins=2, max_n_bins=5, min_bin_size=1, force_min_bins=True)
    assert len(binner._mdlp_split_v3(x, y)) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_target_rate_strictness_rejects_unattainable_distinct_grades(wrapped):
    X, y = population([10, 10, 10], [20] * 3)
    opts = dict(target_bad_rates=[0.1, 0.9], min_n_bins=2, max_n_bins=2, n_jobs=1)
    cls = (lambda **kw: OptimalBinning(method="target_bad_rate", **kw)) if wrapped else TargetBadRateBinning
    with pytest.raises((ValueError, ParallelExecutionError), match="约束|可行"):
        cls(strict_mode=True, **opts).fit(X, y)
    model = cls(strict_mode=False, **opts).fit(X, y)
    assert len(raw_groups(model, X, y)) == 2


def test_prebin_cp_solution_matches_independent_exhaustive_partitions():
    X, y = population([2, 3, 8, 4, 13, 9, 18, 14, 15, 12, 19, 17])
    pre = OptimalBinning._method_classes()["quantile"](max_n_bins=8, min_bin_size=0.05, n_jobs=1)
    model = OptimalBinning(
        method="cp_sat",
        prebinning=pre,
        max_n_bins=4,
        min_n_bins=2,
        min_bin_size=0.05,
        monotonic=False,
        n_jobs=1,
    ).fit(X, y)
    candidates = model._prebinner.splits_["变量"]
    best = -np.inf
    for n_cuts in range(1, 4):
        for cuts in combinations(candidates, n_cuts):
            ids = np.searchsorted(cuts, X["变量"], side="right")
            groups = pd.DataFrame({"箱": ids, "标签": y}).groupby("箱")["标签"].agg(["size", "sum"])
            if groups["size"].min() < 12:
                continue
            bad = groups["sum"].to_numpy()
            good = groups["size"].to_numpy() - bad
            p, q = bad / bad.sum(), good / good.sum()
            best = max(best, float(np.sum((p - q) * np.log(p / q))))
    actual = compute_bin_stats(model.transform(X)["变量"], y, round_digits=False)["分档IV值"].sum()
    assert actual == pytest.approx(best, abs=1e-8)
    assert np.isin(model.splits_["变量"], candidates).all()
    assert model.solver_info_["变量"]["后端"] == "cp_sat"


def test_cp_sat_really_calls_solver_and_honors_worker_budget(monkeypatch):
    from ortools.sat.python import cp_model

    calls = []
    original = cp_model.CpSolver.Solve

    def tracked(self, *args, **kwargs):
        calls.append(self.parameters.num_search_workers)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "Solve", tracked)
    X, y = population([2, 4, 8, 16, 18, 19])
    model = CPSATBinning(monotonic=False, num_workers=1, n_jobs=1).fit(X, y)
    assert calls == [1]
    assert model.solver_info_["变量"]["求解状态"] in {"OPTIMAL", "FEASIBLE"}
    actual_iv = compute_bin_stats(model.transform(X)["变量"], y, round_digits=False)["分档IV值"].sum()
    assert model.solver_info_["变量"]["目标值"] == pytest.approx(actual_iv)


def test_mdlp_duplicate_values_and_specials_do_not_change_normal_partition():
    X, y = population([2, 3, 8, 4, 13, 9, 18, 14, 15, 12, 19, 17])
    model = MDLPBinning(force_min_bins=False, max_n_bins=3, min_bin_size=1, special_codes=[-999], n_jobs=1).fit(X, y)
    extra = pd.concat([X, pd.DataFrame({"变量": [-999.0] * 200})], ignore_index=True)
    labels = pd.concat([y, pd.Series([1] * 200)], ignore_index=True)
    extended = clone(model).fit(extra, labels)
    pd.testing.assert_frame_equal(model.transform(X), extended.transform(X))


def test_kmeans_does_not_change_global_random_state():
    from hscredit.core.binning import KMeansBinning

    X, y = population([1, 5, 8, 12, 16, 19])
    np.random.seed(917)
    state = np.random.get_state()
    KMeansBinning(random_state=42, n_jobs=1).fit(X, y)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_unsupported_lift_filter_is_rejected_instead_of_ignored():
    from hscredit.core.binning import BestLiftBinning

    with pytest.raises(ValueError, match="min_lift.*覆盖全部样本"):
        BestLiftBinning(min_lift=1.5)


def test_smoothing_selection_reaches_direct_algorithm():
    X, y = population([2, 4, 6, 12, 16, 18])
    model = OptimalBinning(method="smooth", smoothing_method="laplace", merge_criterion="chi2", n_jobs=1).fit(X, y)
    assert model._binner.method == "laplace"
    assert model._binner.merge_criterion == "chi2"


def test_smoothing_merge_criterion_changes_the_selected_statistical_objective(monkeypatch):
    from hscredit.core.binning import SmoothBinning

    table = pd.DataFrame({"count": [10, 10, 10], "smoothed_rate": [0.1, 0.5, 0.9]})
    model = SmoothBinning(min_n_bins=2, max_n_bins=2)
    monkeypatch.setattr(model, "_calculate_merge_iv_loss", lambda bins, y, index: [0.01, 0.03][index])
    monkeypatch.setattr(model, "_calculate_chi2", lambda bins, y, left, right: [3.0, 0.1][left])
    model.set_params(merge_criterion="iv")
    assert model._find_merge_candidate_conservative(None, None, table, 1, force=True) == 0
    model.set_params(merge_criterion="chi2")
    assert model._find_merge_candidate_conservative(None, None, table, 1, force=True) == 1


def test_kde_weight_selects_density_or_information_value(monkeypatch):
    from hscredit.core.binning import KernelDensityBinning

    X, y = population([1, 9, 9, 9], [10] * 4)
    cuts = np.array([0.5, 1.5, 2.5])
    model = KernelDensityBinning(max_n_bins=2, min_n_bins=2)
    monkeypatch.setattr(model, "_compute_kde_v3", lambda x: (cuts, np.array([0.3, 0.2, 0.1])))
    model.set_params(iv_weight=0.0)
    np.testing.assert_array_equal(model._select_best_valleys_by_iv_v3(cuts, X["变量"].to_numpy(), y), [2.5])
    model.set_params(iv_weight=1.0)
    np.testing.assert_array_equal(model._select_best_valleys_by_iv_v3(cuts, X["变量"].to_numpy(), y), [0.5])


@pytest.mark.parametrize("binner_class", [ORBinning, CPSATBinning])
def test_prebin_and_candidate_budgets_control_actual_candidate_pool(binner_class):
    X = pd.Series(np.arange(200.0), name="变量")
    y = pd.Series(np.tile([0, 1], 100))
    for prebins, cap, expected_max in [(5, 100, 4), (50, 7, 7)]:
        binner = binner_class(n_prebins=prebins, max_candidates=cap)
        result = binner._get_candidate_splits(X, y)
        candidates = result[0] if binner_class is CPSATBinning else result
        assert len(candidates) == expected_max


def test_refit_without_prebinning_does_not_reuse_previous_candidate_space():
    X, y = population([2, 3, 8, 4, 13, 9, 18, 14, 15, 12, 19, 17])
    binner = OptimalBinning(method="tree", prebinning="quantile", prebinning_params={"max_n_bins": 3}, n_jobs=1)
    binner.fit(X, y)
    binner.set_params(prebinning=None)
    binner.fit(X, y)
    fresh = OptimalBinning(method="tree", n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(binner.splits_["变量"], fresh.splits_["变量"])


def test_force_numeric_preserves_existing_non_numeric_coercion_contract():
    from hscredit.core.binning import UniformBinning

    X = pd.DataFrame({"数值": np.arange(20.0), "文本": ["甲", "乙"] * 10})
    y = pd.Series([0, 1] * 10)
    model = UniformBinning(force_numerical=True, n_jobs=1).fit(X, y)
    assert model.feature_types_ == {"数值": "numerical", "文本": "numerical"}
    assert model.transform(X).shape == X.shape


def test_cp_timeout_returns_only_verified_feasible_seed(monkeypatch):
    from ortools.sat.python import cp_model

    X, y = population([2, 4, 8, 16, 18, 19])
    monkeypatch.setattr(cp_model.CpSolver, "Solve", lambda *args: cp_model.UNKNOWN)
    with pytest.warns(RuntimeWarning, match="可行解"):
        model = CPSATBinning(monotonic="ascending", min_bin_size=0.1, n_jobs=1).fit(X, y)
    groups = raw_groups(model, X, y)
    assert groups["size"].min() >= 12
    assert np.all(np.diff(groups["mean"]) >= 0)
    assert model.solver_info_["变量"]["求解状态"] == "FEASIBLE"
    assert model.solver_info_["变量"]["求解器返回状态"] == "UNKNOWN"


@pytest.mark.parametrize("minimum", [None, 1])
def test_selectively_fixed_empty_boundary_is_preserved_or_explicitly_rejected(minimum):
    X = pd.DataFrame({"变量": np.arange(100.0)})
    y = pd.Series([0, 1] * 50)
    binner = OptimalBinning(
        method="quantile",
        user_splits={"变量": [-1.0, 50.0]},
        user_splits_fixed={"变量": [True, False]},
        min_bin_size=minimum,
        n_jobs=1,
    )
    if minimum is None:
        binner.fit(X, y)
        assert -1.0 in binner.splits_["变量"]
    else:
        with pytest.raises(ValueError, match="固定切点.*无法满足"):
            binner.fit(X, y)
