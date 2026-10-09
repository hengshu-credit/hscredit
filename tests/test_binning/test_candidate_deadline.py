"""候选枚举与加性 IV 动态规划共享同一个期限。"""

import numpy as np
import pytest

import hscredit.core.binning._candidate_search as module


@pytest.mark.parametrize("objective,monotonic", [("iv", None), ("iv", "ascending"), ("ks", None), ("gini", None)])
def test_deadline_applies_to_every_search_branch(monkeypatch, objective, monotonic):
    ticks = iter([0.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks, 2.0))
    with pytest.raises(TimeoutError, match="time_limit"):
        module.search_candidate_splits(
            np.arange(20),
            np.arange(20) % 2,
            [4.5, 9.5, 14.5],
            objective=objective,
            monotonic=monotonic,
            min_n_bins=2,
            max_n_bins=4,
            min_samples=2,
            time_limit=1,
        )


@pytest.mark.parametrize("limit", [0, -1, np.inf, np.nan, True])
def test_invalid_time_limits_rejected(limit):
    with pytest.raises(ValueError, match="time_limit"):
        module.search_candidate_splits(
            np.arange(4),
            np.array([0, 1, 0, 1]),
            [1.5],
            objective="iv",
            min_n_bins=2,
            max_n_bins=2,
            min_samples=1,
            time_limit=limit,
        )
