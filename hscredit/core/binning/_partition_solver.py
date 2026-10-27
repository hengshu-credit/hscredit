"""在预分箱边界上求解连续区间分区，不对求解结果做最近点映射。"""

from time import perf_counter
from itertools import combinations, islice
from typing import Dict, Optional
import warnings

import numpy as np

from ..metrics._binning import woe_iv_vectorized
from ._contracts import BinningConstraintViolation


def prepare_candidates(x, y, candidates):
    """稳定排序成对样本，并按最终左闭右开区间计算候选位置。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("分箱特征和标签必须是一维等长数组")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    candidates = np.unique(np.asarray(candidates, dtype=float))
    candidates = candidates[np.isfinite(candidates)]
    positions = np.searchsorted(x, candidates, side="left")
    valid = (positions > 0) & (positions < len(x))
    candidates, positions = candidates[valid], positions[valid]
    # 同一个样本边界只保留一个候选，避免零宽度预箱。
    _, indices = np.unique(positions, return_index=True)
    return x, y, candidates[indices], positions[indices]


def solve_partition(
    x,
    y,
    candidates,
    *,
    objective="iv",
    min_n_bins=2,
    max_n_bins=5,
    min_samples=1,
    max_samples=None,
    min_bad_rate=0.0,
    monotonic=False,
    time_limit=30.0,
    num_workers=1,
    random_state=None,
    fixed_splits=None,
    target_bad_rates=None,
    strict_target=True,
    merge_empty_bins=True,
    info: Optional[Dict] = None,
):
    """以 CP-SAT 的有向区间路径模型求解箱数、人数和风险形状约束。

    每条边代表完整的一个箱，流守恒保证所有预箱恰好覆盖一次。
    目标值按 1e9 比例整数化；求解状态和界限通过 info 返回。
    """
    from ortools.sat.python import cp_model

    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit 必须为有限正数")
    start_time = perf_counter()
    x, y, candidates, positions = prepare_candidates(x, y, candidates)
    if not len(x):
        return []
    bounds = np.r_[0, positions, len(x)]
    prefix_bad = np.r_[0, np.cumsum(y == 1, dtype=int)]
    total_bad, total_good = int(prefix_bad[-1]), int(len(y) - prefix_bad[-1])
    last = len(bounds) - 1
    model = cp_model.CpModel()
    incoming = [[] for _ in bounds]
    outgoing = [[] for _ in bounds]
    edges = {}
    rates = {}
    values = {}
    scale = 10**9
    rate_scale = 10**12
    fixed_positions = set(np.searchsorted(x, [] if fixed_splits is None else fixed_splits, side="left").tolist())
    # 在很短的预算内先验证一个可行分区，既用于 warm start，也防止短时限只耗在预求解上。
    from ._candidate_search import _is_monotonic, _score_partition

    seed = None
    seed_score = -np.inf
    seed_deadline = start_time + min(0.05, time_limit * 0.1)
    for size in range(min_n_bins, min(max_n_bins, last) + 1):
        balanced = tuple(np.rint(np.linspace(0, last, size + 1)[1:-1]).astype(int))
        attempts = [balanced]
        attempts.extend(islice(combinations(range(1, last), size - 1), 256))
        for selected in attempts:
            if perf_counter() >= seed_deadline:
                break
            if len(set(selected)) != size - 1 or not fixed_positions.issubset(set(bounds[list(selected)])):
                continue
            indices = np.asarray([0, *selected, last], dtype=int)
            counts = np.diff(bounds[indices])
            bad = np.diff(prefix_bad[bounds[indices]])
            if np.any(counts < max(1, min_samples)) or (max_samples is not None and np.any(counts > max_samples)):
                continue
            seed_rates = bad / counts
            if np.any(seed_rates < min_bad_rate - 1e-12) or not _is_monotonic(seed_rates, monotonic, 0.0):
                continue
            if target_bad_rates is not None and strict_target:
                regions = np.searchsorted(target_bad_rates, seed_rates, side="left")
                if len(np.unique(regions)) != size or (not merge_empty_bins and size != len(target_bad_rates) + 1):
                    continue
            score = _score_partition(bad, counts - bad, objective)
            if seed is None or score > seed_score:
                seed, seed_score = selected, score
        if perf_counter() >= seed_deadline:
            break
    for i in range(last):
        if perf_counter() - start_time >= time_limit:
            raise TimeoutError("分箱建模达到 time_limit，尚未生成满足全部约束的可行解")
        for j in range(i + 1, last + 1):
            if any(bounds[i] < position < bounds[j] for position in fixed_positions):
                continue
            count = int(bounds[j] - bounds[i])
            if count < max(1, min_samples) or (max_samples is not None and count > max_samples):
                continue
            bad = int(prefix_bad[bounds[j]] - prefix_bad[bounds[i]])
            good = count - bad
            rate = bad / count
            if rate < min_bad_rate - 1e-12:
                continue
            edge = model.NewBoolVar(f"区间_{i}_{j}")
            edges[i, j] = edge
            rates[i, j] = int(round(rate * rate_scale))
            incoming[j].append((i, j))
            outgoing[i].append((i, j))
            if objective == "iv":
                # 使用共享指标内核；无纯箱时是严格可加的 IV。
                _, contribution, _ = woe_iv_vectorized([good, total_good - good], [bad, total_bad - bad])
                value = float(contribution[0])
            elif objective == "gini":
                value = -count / len(y) * rate * (1.0 - rate)
            elif objective == "entropy":
                p = np.asarray([rate, 1 - rate])
                positive = p[p > 0]
                value = count / len(y) * float(np.sum(positive * np.log(positive)))
            elif objective == "chi2":
                expected_bad = count * total_bad / len(y)
                expected_good = count * total_good / len(y)
                value = (bad - expected_bad) ** 2 / expected_bad if expected_bad else 0.0
                value += (good - expected_good) ** 2 / expected_good if expected_good else 0.0
            elif objective == "ks":
                value = 0.0
            else:
                raise ValueError(f"区间求解器不支持目标: {objective}")
            values[i, j] = int(round(value * scale))

    def edge_sum(items):
        return sum(edges[key] for key in items)

    model.Add(edge_sum(outgoing[0]) == 1)
    model.Add(edge_sum(incoming[last]) == 1)
    active = {}
    left_rates, right_rates = {}, {}
    for j in range(1, last):
        active[j] = model.NewBoolVar(f"边界_{j}")
        model.Add(edge_sum(incoming[j]) == active[j])
        model.Add(edge_sum(outgoing[j]) == active[j])
        left_rates[j] = model.NewIntVar(0, rate_scale, f"左坏率_{j}")
        right_rates[j] = model.NewIntVar(0, rate_scale, f"右坏率_{j}")
        model.Add(left_rates[j] == sum(rates[key] * edges[key] for key in incoming[j]))
        model.Add(right_rates[j] == sum(rates[key] * edges[key] for key in outgoing[j]))
    n_bins = sum(edges.values())
    model.Add(n_bins >= min_n_bins)
    model.Add(n_bins <= max_n_bins)
    covered_regions = []
    if target_bad_rates is not None:
        thresholds = np.asarray(target_bad_rates, dtype=float)
        for region in range(len(thresholds) + 1):
            members = [
                edge
                for key, edge in edges.items()
                if np.searchsorted(thresholds, rates[key] / rate_scale, side="left") == region
            ]
            if strict_target:
                # 每个坏率档位只对应一个区间；空档是否允许由调用方明确选择。
                model.Add(sum(members) <= 1 if merge_empty_bins else sum(members) == 1)
            else:
                covered = model.NewBoolVar(f"坏率档位_{region}")
                model.Add(sum(members) >= covered)
                model.Add(sum(members) <= max_n_bins * covered)
                covered_regions.append(covered)

    if monotonic not in (False, None, "none"):
        mode = monotonic
        if mode == "auto_asc_desc":
            ascending = model.NewBoolVar("递增方向")
            for j in active:
                model.Add(left_rates[j] <= right_rates[j]).OnlyEnforceIf(ascending)
                model.Add(left_rates[j] >= right_rates[j]).OnlyEnforceIf(ascending.Not())
        elif mode in ("ascending", "descending"):
            for j in active:
                model.Add(left_rates[j] <= right_rates[j] if mode == "ascending" else left_rates[j] >= right_rates[j])
        elif mode in ("convex", "concave"):
            for (i, j), edge in edges.items():
                if i > 0 and j < last:
                    curvature = left_rates[i] + right_rates[j] - 2 * rates[i, j]
                    model.Add(curvature >= 0 if mode == "convex" else curvature <= 0).OnlyEnforceIf(edge)
        else:
            peak = model.NewBoolVar("单峰形状")
            if mode in ("peak", "peak_heuristic"):
                model.Add(peak == 1)
            elif mode in ("valley", "valley_heuristic"):
                model.Add(peak == 0)
            elif mode not in (True, "auto", "auto_heuristic"):
                raise ValueError(f"不支持的单调约束: {mode}")
            pivot = model.NewIntVar(0, last, "形状转折位置")
            for j in active:
                before = model.NewBoolVar(f"转折前_{j}")
                model.Add(j <= pivot).OnlyEnforceIf(before)
                model.Add(j > pivot).OnlyEnforceIf(before.Not())
                model.Add(left_rates[j] <= right_rates[j]).OnlyEnforceIf([peak, before])
                model.Add(left_rates[j] >= right_rates[j]).OnlyEnforceIf([peak, before.Not()])
                model.Add(left_rates[j] >= right_rates[j]).OnlyEnforceIf([peak.Not(), before])
                model.Add(left_rates[j] <= right_rates[j]).OnlyEnforceIf([peak.Not(), before.Not()])

    if objective == "ks":
        ks_terms = [0]
        for j, chosen in active.items():
            bad = int(prefix_bad[bounds[j]])
            good = int(bounds[j] - bad)
            distance = abs(bad / total_bad - good / total_good) if total_bad and total_good else 0.0
            ks_terms.append(int(round(distance * scale)) * chosen)
        value_var = model.NewIntVar(0, scale, "KS目标")
        model.AddMaxEquality(value_var, ks_terms)
        objective_expression = value_var
    else:
        objective_expression = sum(values[key] * edge for key, edge in edges.items())
    # 同目标值优先较少箱数，避免无收益的任意补箱。
    primary = objective_expression * (max_n_bins + 1) - n_bins
    if covered_regions:
        bound = sum(sorted((abs(value) for value in values.values()), reverse=True)[:max_n_bins])
        model.Maximize(sum(covered_regions) * (2 * bound * (max_n_bins + 1) + max_n_bins + 1) + primary)
    else:
        model.Maximize(primary)
    if seed is not None:
        path = [0, *seed, last]
        selected_edges = set(zip(path[:-1], path[1:]))
        for key, edge in edges.items():
            model.AddHint(edge, int(key in selected_edges))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.001, time_limit - (perf_counter() - start_time))
    solver.parameters.num_search_workers = max(1, int(num_workers or 1))
    if random_state is not None:
        solver.parameters.random_seed = int(random_state)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    use_seed = status == cp_model.UNKNOWN and seed is not None
    if info is not None:
        info.update(
            {
                "后端": "cp_sat",
                "求解状态": "FEASIBLE" if use_seed else status_name,
                "求解器返回状态": status_name,
                "可行解来源": "预先验证的候选分区" if use_seed else "CP-SAT",
                "目标函数": objective,
                "目标整数化比例": scale,
                "耗时秒": perf_counter() - start_time,
            }
        )
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            info.update({"整数目标值": solver.ObjectiveValue(), "整数目标上界": solver.BestObjectiveBound()})
    if status == cp_model.INFEASIBLE:
        raise BinningConstraintViolation("候选边界上不存在同时满足箱数、样本数、坏率及单调约束的可行分箱")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE) and not use_seed:
        if status == cp_model.MODEL_INVALID:
            raise ValueError(f"分箱约束模型无效: {model.Validate()}")
        raise TimeoutError("分箱求解达到 time_limit，尚未找到满足全部约束的可行解")
    if status == cp_model.FEASIBLE or use_seed:
        warnings.warn("分箱求解达到 time_limit，返回当前可行解，尚未证明最优", RuntimeWarning, stacklevel=2)
    result = (
        [float(candidates[j - 1]) for j in seed]
        if use_seed
        else [float(candidates[j - 1]) for j in active if solver.BooleanValue(active[j])]
    )
    if info is not None:
        from ._candidate_search import _score_partition

        bins = np.searchsorted(result, x, side="right")
        counts = np.bincount(bins, minlength=len(result) + 1)
        bad = np.bincount(bins, weights=y, minlength=len(counts))
        info["目标值"] = _score_partition(bad, counts - bad, objective)
    return result
