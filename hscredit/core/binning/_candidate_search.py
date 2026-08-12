"""候选切分点约束搜索。

为 CP-SAT 难以直接表达的非线性 KS/Gini 目标和单调约束提供确定性搜索，
所有候选都来自预分箱边界，不会生成规则外切点。
"""

from itertools import combinations
from time import perf_counter
from typing import List, Optional, Sequence, Union

import numpy as np


def _is_monotonic(rates: np.ndarray, mode: Optional[Union[bool, str]], correlation: float) -> bool:
    if mode in (None, False, "none") or len(rates) < 2:
        return True
    if mode in (True, "auto", "auto_asc_desc", "auto_heuristic"):
        mode = "ascending" if correlation >= 0 else "descending"
    differences = np.diff(rates)
    if mode == "ascending":
        return bool(np.all(differences >= -1e-12))
    if mode == "descending":
        return bool(np.all(differences <= 1e-12))
    if mode in ("peak", "peak_heuristic", "concave"):
        return any(
            np.all(np.diff(rates[: pivot + 1]) >= -1e-12) and np.all(np.diff(rates[pivot:]) <= 1e-12)
            for pivot in range(len(rates))
        )
    if mode in ("valley", "valley_heuristic", "convex"):
        return any(
            np.all(np.diff(rates[: pivot + 1]) <= 1e-12) and np.all(np.diff(rates[pivot:]) >= -1e-12)
            for pivot in range(len(rates))
        )
    return True


def _score_partition(bad: np.ndarray, good: np.ndarray, objective: str) -> float:
    total_bad = float(bad.sum())
    total_good = float(good.sum())
    total = total_bad + total_good
    if total_bad <= 0 or total_good <= 0 or total <= 0:
        return -np.inf
    if objective == "iv":
        bad_dist = np.maximum(bad / total_bad, 1e-12)
        good_dist = np.maximum(good / total_good, 1e-12)
        return float(np.sum((bad_dist - good_dist) * np.log(bad_dist / good_dist)))
    if objective == "ks":
        return float(np.max(np.abs(np.cumsum(good) / total_good - np.cumsum(bad) / total_bad)))
    if objective == "gini":
        counts = bad + good
        rates = bad / np.maximum(counts, 1.0)
        impurity = np.sum((counts / total) * rates * (1.0 - rates))
        return float(1.0 - impurity)
    raise ValueError(f"候选搜索不支持目标: {objective}")


def _search_additive_iv(
    candidate_values: np.ndarray,
    positions: np.ndarray,
    prefix_bad: np.ndarray,
    prefix_good: np.ndarray,
    *,
    min_n_bins: int,
    max_n_bins: int,
    min_samples: int,
    max_samples: Optional[int],
) -> List[float]:
    """以动态规划求解任意候选边界组合的全局最优 IV 分区。"""
    boundaries = np.asarray([0, *positions.tolist(), int(prefix_bad.size - 1)], dtype=int)
    total_bad = float(prefix_bad[-1])
    total_good = float(prefix_good[-1])
    if total_bad <= 0 or total_good <= 0:
        return []

    n_boundaries = len(boundaries)
    states = {(0, 0): (0.0, tuple())}
    for n_bins in range(1, max_n_bins + 1):
        for end_index in range(1, n_boundaries):
            best = None
            for start_index in range(end_index):
                previous = states.get((n_bins - 1, start_index))
                if previous is None:
                    continue
                count = int(boundaries[end_index] - boundaries[start_index])
                if count < min_samples or (max_samples is not None and count > max_samples):
                    continue
                bad = float(prefix_bad[boundaries[end_index]] - prefix_bad[boundaries[start_index]])
                good = float(prefix_good[boundaries[end_index]] - prefix_good[boundaries[start_index]])
                bad_dist = max(bad / total_bad, 1e-12)
                good_dist = max(good / total_good, 1e-12)
                contribution = (bad_dist - good_dist) * np.log(bad_dist / good_dist)
                selected = previous[1] + (() if start_index == 0 else (start_index - 1,))
                candidate = (previous[0] + float(contribution), selected)
                key = (candidate[0], tuple(-candidate_values[index] for index in candidate[1]))
                if best is None or key > best[0]:
                    best = (key, candidate)
            if best is not None:
                states[(n_bins, end_index)] = best[1]

    best = None
    for n_bins in range(max(1, min_n_bins), max_n_bins + 1):
        result = states.get((n_bins, n_boundaries - 1))
        if result is None:
            continue
        values = [float(candidate_values[index]) for index in result[1]]
        key = (result[0], -len(values), tuple(-value for value in values))
        if best is None or key > best[0]:
            best = (key, values)
    return [] if best is None else best[1]


def search_candidate_splits(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    candidates: Sequence[float],
    *,
    objective: str,
    min_n_bins: int,
    max_n_bins: int,
    min_samples: int,
    max_samples: Optional[int] = None,
    monotonic: Optional[Union[bool, str]] = None,
    time_limit: float = 30.0,
) -> List[float]:
    """在候选边界子集中搜索满足硬约束的最优分箱。"""
    x_values = np.asarray(x_sorted, dtype=float)
    y_values = np.asarray(y_sorted, dtype=int)
    candidate_values = np.unique(np.sort(np.asarray(candidates, dtype=float)))
    if len(candidate_values) == 0:
        return []

    positions = np.searchsorted(x_values, candidate_values, side="right")
    valid = (positions > 0) & (positions < len(x_values))
    candidate_values = candidate_values[valid]
    positions = positions[valid]
    if len(candidate_values) == 0:
        return []

    prefix_bad = np.concatenate(([0], np.cumsum(y_values == 1, dtype=int)))
    prefix_good = np.concatenate(([0], np.cumsum(y_values == 0, dtype=int)))
    correlation = float(np.corrcoef(x_values, y_values)[0, 1]) if len(x_values) > 1 else 0.0
    if not np.isfinite(correlation):
        correlation = 0.0

    if objective == "iv" and monotonic in (None, False, "none"):
        return _search_additive_iv(
            candidate_values,
            positions,
            prefix_bad,
            prefix_good,
            min_n_bins=min_n_bins,
            max_n_bins=max_n_bins,
            min_samples=min_samples,
            max_samples=max_samples,
        )

    min_splits = max(0, min_n_bins - 1)
    max_splits = min(len(candidate_values), max(0, max_n_bins - 1))
    deadline = perf_counter() + max(0.01, float(time_limit))
    best_key = None
    best_values: List[float] = []

    for n_splits in range(min_splits, max_splits + 1):
        for selected in combinations(range(len(candidate_values)), n_splits):
            if perf_counter() >= deadline:
                return best_values
            boundaries = np.asarray([0, *[int(positions[index]) for index in selected], len(x_values)], dtype=int)
            counts = np.diff(boundaries)
            if np.any(counts < min_samples) or (max_samples is not None and np.any(counts > max_samples)):
                continue
            bad = np.asarray(
                [prefix_bad[end] - prefix_bad[start] for start, end in zip(boundaries[:-1], boundaries[1:])],
                dtype=float,
            )
            good = np.asarray(
                [prefix_good[end] - prefix_good[start] for start, end in zip(boundaries[:-1], boundaries[1:])],
                dtype=float,
            )
            rates = bad / np.maximum(bad + good, 1.0)
            if not _is_monotonic(rates, monotonic, correlation):
                continue
            score = _score_partition(bad, good, objective)
            values = [float(candidate_values[index]) for index in selected]
            key = (score, -len(values), tuple(-value for value in values))
            if best_key is None or key > best_key:
                best_key = key
                best_values = values
    return best_values
