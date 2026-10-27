"""候选切分点约束搜索。

为 CP-SAT 难以直接表达的非线性 KS/Gini 目标和单调约束提供确定性搜索，
所有候选都来自预分箱边界，不会生成规则外切点。
"""

from itertools import combinations
from time import perf_counter
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
from ..metrics._binning import woe_iv_vectorized
from ._partition_solver import prepare_candidates
from ._contracts import BinningConstraintViolation


def _is_monotonic(rates: np.ndarray, mode: Optional[Union[bool, str]], correlation: float) -> bool:
    if mode in (None, False, "none") or len(rates) < 2:
        return True
    if mode in (True, "auto", "auto_heuristic"):
        return _is_monotonic(rates, "peak", correlation) or _is_monotonic(rates, "valley", correlation)
    if mode == "auto_asc_desc":
        return _is_monotonic(rates, "ascending", correlation) or _is_monotonic(rates, "descending", correlation)
    differences = np.diff(rates)
    if mode == "ascending":
        return bool(np.all(differences >= -1e-12))
    if mode == "descending":
        return bool(np.all(differences <= 1e-12))
    if mode == "convex":
        return bool(np.all(np.diff(rates, n=2) >= -1e-12))
    if mode == "concave":
        return bool(np.all(np.diff(rates, n=2) <= 1e-12))
    if mode in ("peak", "peak_heuristic"):
        return any(
            np.all(np.diff(rates[: pivot + 1]) >= -1e-12) and np.all(np.diff(rates[pivot:]) <= 1e-12)
            for pivot in range(len(rates))
        )
    if mode in ("valley", "valley_heuristic"):
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
        return float(woe_iv_vectorized(good, bad)[2])
    if objective == "ks":
        return float(np.max(np.abs(np.cumsum(good) / total_good - np.cumsum(bad) / total_bad)))
    if objective == "gini":
        counts = bad + good
        rates = bad / np.maximum(counts, 1.0)
        impurity = np.sum((counts / total) * rates * (1.0 - rates))
        return float(1.0 - impurity)
    if objective == "entropy":
        counts = bad + good
        rates = bad / counts
        return float(sum(n / total * sum(p * np.log(p) for p in (r, 1 - r) if p > 0) for n, r in zip(counts, rates)))
    if objective == "chi2":
        counts = bad + good
        expected_bad, expected_good = counts * total_bad / total, counts * total_good / total
        return float(np.sum((bad - expected_bad) ** 2 / expected_bad + (good - expected_good) ** 2 / expected_good))
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
    deadline: float = float("inf"),
) -> List[float]:
    """以动态规划求解任意候选边界组合的全局最优 IV 分区。"""
    boundaries = np.asarray([0, *positions.tolist(), int(prefix_bad.size - 1)], dtype=int)
    total_bad = float(prefix_bad[-1])
    total_good = float(prefix_good[-1])
    if total_bad <= 0 or total_good <= 0:
        return []

    n_boundaries = len(boundaries)
    states = {(0, 0): (0.0, tuple())}
    timed_out = False
    for n_bins in range(1, max_n_bins + 1):
        for end_index in range(1, n_boundaries):
            if perf_counter() >= deadline:
                timed_out = True
                break
            best = None
            for start_index in range(end_index):
                if start_index % 64 == 0 and perf_counter() >= deadline:
                    timed_out = True
                    break
                previous = states.get((n_bins - 1, start_index))
                if previous is None:
                    continue
                count = int(boundaries[end_index] - boundaries[start_index])
                if count < min_samples or (max_samples is not None and count > max_samples):
                    continue
                bad = float(prefix_bad[boundaries[end_index]] - prefix_bad[boundaries[start_index]])
                good = float(prefix_good[boundaries[end_index]] - prefix_good[boundaries[start_index]])
                bad_dist = max(bad, 1e-10) / total_bad
                good_dist = max(good, 1e-10) / total_good
                contribution = (bad_dist - good_dist) * np.log(bad_dist / good_dist)
                selected = previous[1] + (() if start_index == 0 else (start_index - 1,))
                candidate = (previous[0] + float(contribution), selected)
                key = (candidate[0], tuple(-candidate_values[index] for index in candidate[1]))
                if best is None or key > best[0]:
                    best = (key, candidate)
            if best is not None:
                states[(n_bins, end_index)] = best[1]
            if timed_out:
                break
        if timed_out:
            break

    best = None
    for n_bins in range(max(1, min_n_bins), max_n_bins + 1):
        result = states.get((n_bins, n_boundaries - 1))
        if result is None:
            continue
        values = [float(candidate_values[index]) for index in result[1]]
        key = (result[0], -len(values), tuple(-value for value in values))
        if best is None or key > best[0]:
            best = (key, values)
    if timed_out:
        if best is None:
            raise TimeoutError("候选 IV 搜索达到 time_limit，尚未找到满足最小箱数的可行解")
        warnings.warn("候选 IV 搜索达到 time_limit，返回当前可行解，未证明全局最优", RuntimeWarning, stacklevel=2)
    if best is None:
        raise BinningConstraintViolation("候选边界上不存在满足箱数和样本数约束的可行分箱")
    return best[1]


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
    min_bad_rate: float = 0.0,
    score_callback=None,
) -> List[float]:
    """在候选边界子集中搜索满足硬约束的最优分箱。"""
    if isinstance(time_limit, (bool, np.bool_)) or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit 必须为有限正数")
    deadline = perf_counter() + float(time_limit)
    x_values, y_values, candidate_values, positions = prepare_candidates(x_sorted, y_sorted, candidates)

    prefix_bad = np.concatenate(([0], np.cumsum(y_values == 1, dtype=int)))
    prefix_good = np.concatenate(([0], np.cumsum(y_values == 0, dtype=int)))
    # 当前形状约束直接检查候选箱坏率，不需要对单类标签计算未定义的相关系数。
    correlation = 0.0

    if objective == "iv" and score_callback is None and monotonic in (None, False, "none") and min_bad_rate == 0:
        return _search_additive_iv(
            candidate_values,
            positions,
            prefix_bad,
            prefix_good,
            min_n_bins=min_n_bins,
            max_n_bins=max_n_bins,
            min_samples=min_samples,
            max_samples=max_samples,
            deadline=deadline,
        )

    min_splits = max(0, min_n_bins - 1)
    max_splits = min(len(candidate_values), max(0, max_n_bins - 1))
    best_key = None
    best_values: List[float] = []

    for n_splits in range(min_splits, max_splits + 1):
        for selected in combinations(range(len(candidate_values)), n_splits):
            if perf_counter() >= deadline:
                if best_key is None:
                    raise TimeoutError("候选搜索达到 time_limit，尚未找到满足约束的可行解")
                warnings.warn("候选搜索达到 time_limit，返回当前可行解，未证明全局最优", RuntimeWarning, stacklevel=2)
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
            if np.any(rates < min_bad_rate - 1e-12):
                continue
            if not _is_monotonic(rates, monotonic, correlation):
                continue
            values = [float(candidate_values[index]) for index in selected]
            score = score_callback(values) if score_callback is not None else _score_partition(bad, good, objective)
            key = (score, -len(values), tuple(-value for value in values))
            if best_key is None or key > best_key:
                best_key = key
                best_values = values
    if best_key is None:
        raise BinningConstraintViolation("候选边界上不存在同时满足箱数、样本数、坏率和单调约束的可行分箱")
    return best_values
