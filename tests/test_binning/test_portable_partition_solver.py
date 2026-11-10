"""无 OR-Tools 环境的分箱目标、硬约束和求解状态回归测试。"""

import builtins
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import CPSATBinning, ORBinning, OptimalBinning
from hscredit.core.binning._candidate_search import _is_monotonic, _score_partition
from hscredit.core.binning._contracts import BinningConstraintViolation
from hscredit.core.binning._partition_solver import solve_partition


@pytest.fixture
def without_ortools(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "ortools" or name.startswith("ortools."):
            raise ModuleNotFoundError("测试环境未安装 OR-Tools", name="ortools")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)


@pytest.fixture
def population():
    bad_counts = [1, 4, 6, 3, 8, 9]
    x = np.repeat(np.arange(len(bad_counts), dtype=float), 10)
    y = np.concatenate([np.r_[np.ones(bad), np.zeros(10 - bad)] for bad in bad_counts]).astype(int)
    return x, y, np.arange(0.5, 5, 1)


@pytest.mark.parametrize("objective", ["iv", "ks", "gini", "entropy", "chi2"])
@pytest.mark.parametrize(
    "constraints",
    [
        {},
        {"monotonic": "ascending"},
        {"monotonic": "peak", "fixed_splits": [2.5]},
        {"monotonic": "convex"},
        {"target_bad_rates": [0.25, 0.75], "merge_empty_bins": False},
        {"target_bad_rates": [0.25, 0.75], "strict_target": False},
    ],
)
def test_portable_solver_matches_exhaustive_feasible_optimum(without_ortools, population, objective, constraints):
    x, y, candidates = population
    expected = []
    for size in range(1, 4):
        for cuts in combinations(candidates, size):
            if not set(constraints.get("fixed_splits", [])).issubset(cuts):
                continue
            bins = np.searchsorted(cuts, x, side="right")
            counts = np.bincount(bins)
            bad = np.bincount(bins, weights=y)
            rates = bad / counts
            if counts.min() < 10 or counts.max() > 40 or not _is_monotonic(rates, constraints.get("monotonic"), 0.0):
                continue
            coverage = 0
            if "target_bad_rates" in constraints:
                thresholds = constraints["target_bad_rates"]
                coverage = len(np.unique(np.searchsorted(thresholds, rates, side="left")))
                if constraints.get("strict_target", True) and (coverage != len(counts) or (not constraints.get("merge_empty_bins", True) and coverage != len(thresholds) + 1)):
                    continue
            expected.append((coverage if not constraints.get("strict_target", True) else 0, _score_partition(bad, counts - bad, objective), -size, tuple(-v for v in cuts)))
    info = {}
    options = dict(objective=objective, min_n_bins=2, max_n_bins=4, min_samples=10, max_samples=40, info=info, **constraints)
    if not expected:
        with pytest.raises(BinningConstraintViolation):
            solve_partition(x, y, candidates, **options)
        return
    result = solve_partition(x, y, candidates, **options)
    best = max(expected)
    np.testing.assert_allclose(result, [-v for v in best[-1]])
    assert info["目标值"] == pytest.approx(best[1])
    assert info["求解状态"] == "OPTIMAL"
    assert info["后端"] in {"dynamic_programming", "enumeration"}


@pytest.mark.parametrize("factory", [ORBinning, CPSATBinning, lambda **kw: OptimalBinning(method="or_tools", **kw), lambda **kw: OptimalBinning(method="cp_sat", **kw)])
def test_public_binning_entry_points_work_without_native_solver(without_ortools, population, factory):
    x, y, _ = population
    data = pd.DataFrame({"风险分": x})
    model = factory(min_n_bins=2, max_n_bins=3, min_bin_size=0.1, monotonic="ascending", n_jobs=1).fit(data, y)
    transformed = model.transform(data)["风险分"]
    assert transformed.nunique() in (2, 3)
    rates = pd.Series(y).groupby(transformed).mean().to_numpy()
    assert np.all(np.diff(rates) >= 0)


def test_portable_solver_rejects_impossible_fixed_boundary(without_ortools, population):
    x, y, candidates = population
    with pytest.raises(BinningConstraintViolation):
        solve_partition(x, y, candidates[:2], fixed_splits=[4.5], min_n_bins=2, max_n_bins=3)


def test_portable_solver_preserves_empty_input_contract(without_ortools):
    assert solve_partition([], [], []) == []
    with pytest.raises(ValueError, match="等长"):
        solve_partition([], [1], [])


def test_portable_solver_reports_timeout_as_feasible(without_ortools, population, monkeypatch):
    from hscredit.core.binning import _candidate_search

    clock = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(_candidate_search, "perf_counter", lambda: next(clock))
    x, y, candidates = population
    info = {}
    with pytest.warns(RuntimeWarning, match="未证明"):
        cuts = solve_partition(x, y, candidates, objective="ks", time_limit=0.5, info=info)
    assert cuts == [0.5]
    assert info["求解状态"] == "FEASIBLE"


def test_portable_solver_raises_if_no_solution_before_timeout(without_ortools, population, monkeypatch):
    from hscredit.core.binning import _candidate_search

    clock = iter([0.0, 1.0])
    monkeypatch.setattr(_candidate_search, "perf_counter", lambda: next(clock))
    with pytest.raises(TimeoutError):
        solve_partition(*population, objective="ks", time_limit=0.5)
