"""报告样本统计表构造工具."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.parallel import ParallelWorkload, parallel_execute


def _sample_stats_row(task):
    y_map, label_names, display_labels, is_multi, flat_total_col, dataset_label = task
    if is_multi:
        row = {}
        first_y = _as_array(y_map[label_names[0]])
        row[("统计详情", "样本总数")] = len(first_y)
        for label in label_names:
            y_arr = _as_array(y_map[label])
            n = len(y_arr)
            nb = int(np.nansum(y_arr))
            display = display_labels.get(label, label)
            row[("好样本数", display)] = n - nb
            row[("坏样本数", display)] = nb
            row[("坏样本率", display)] = float(np.nanmean(y_arr)) if n else 0.0
        return row
    y_arr = _as_array(y_map[label_names[0]])
    n = len(y_arr)
    nb = int(np.nansum(y_arr))
    return {
        "数据集": dataset_label,
        flat_total_col: n,
        "好样本数": n - nb,
        "坏样本数": nb,
        "坏样本率": float(np.nanmean(y_arr)) if n else 0.0,
    }


def _group_stats_row(task):
    dataset_label, group, y_map, mask, label_names, display_labels, is_multi, group_name = task
    if is_multi:
        row = {
            ("统计详情", "数据集"): dataset_label,
            ("统计详情", group_name): str(group),
        }
        first_y = _as_array(y_map[label_names[0]])[mask]
        row[("统计详情", "样本总数")] = int(len(first_y))
        for label in label_names:
            y_group = _as_array(y_map[label])[mask]
            n = len(y_group)
            nb = int(np.nansum(y_group))
            display = display_labels.get(label, label)
            row[("好样本数", display)] = n - nb
            row[("坏样本数", display)] = nb
            row[("坏样本率", display)] = float(np.nanmean(y_group)) if n else 0.0
        return (dataset_label, str(group)), row
    y_group = _as_array(y_map[label_names[0]])[mask]
    n = len(y_group)
    nb = int(np.nansum(y_group))
    return (dataset_label, str(group)), {
        "样本总数": n,
        "好样本数": n - nb,
        "坏样本数": nb,
        "坏样本率": float(np.nanmean(y_group)) if n else 0.0,
    }


def build_sample_stats_table(
    dataset_labels: Sequence[str],
    y_by_dataset: Sequence[Dict[str, Sequence[int]]],
    label_names: Sequence[str],
    display_labels: Optional[Dict[str, str]] = None,
    flat_total_col: str = "样本总数",
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """构造数据集样本统计表."""
    display_labels = display_labels or {}
    label_names = list(label_names)
    is_multi = len(label_names) > 1

    if is_multi:
        columns = [("统计详情", "样本总数")] + [(metric, display_labels.get(label, label)) for metric in ["好样本数", "坏样本数", "坏样本率"] for label in label_names]
        multi_cols = pd.MultiIndex.from_tuples(columns, names=["统计详情", ""])
        tasks = [(y_map, label_names, display_labels, True, flat_total_col, dataset_label) for dataset_label, y_map in zip(dataset_labels, y_by_dataset)]
        rows = parallel_execute(
            _sample_stats_row,
            tasks,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            task_labels=list(dataset_labels),
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=len(tasks),
                rows=sum(len(_as_array(y_map[label_names[0]])) for y_map in y_by_dataset),
                columns=len(label_names),
                cost_per_item=1.0,
                capability="vectorized",
                operation="报告样本统计",
            ),
        )
        result = pd.DataFrame(rows, index=list(dataset_labels), columns=multi_cols)
        result.index.name = "数据集"
        return result, [c for c in multi_cols if c[0] == "坏样本率"]

    tasks = [(y_map, label_names, display_labels, False, flat_total_col, dataset_label) for dataset_label, y_map in zip(dataset_labels, y_by_dataset)]
    rows = parallel_execute(
        _sample_stats_row,
        tasks,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=list(dataset_labels),
        default_backend="threading",
        workload=ParallelWorkload(
            task_count=len(tasks),
            rows=sum(len(_as_array(y_map[label_names[0]])) for y_map in y_by_dataset),
            columns=1,
            cost_per_item=1.0,
            capability="vectorized",
            operation="报告样本统计",
        ),
    )
    return pd.DataFrame(rows), ["坏样本率"]


def build_group_distribution_table(
    dataset_labels: Sequence[str],
    y_by_dataset: Sequence[Dict[str, Sequence[int]]],
    group_values_by_dataset: Sequence[Sequence[Any]],
    label_names: Sequence[str],
    display_labels: Optional[Dict[str, str]] = None,
    group_name: str = "数据分组",
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """构造数据集分组分布表."""
    display_labels = display_labels or {}
    label_names = list(label_names)
    is_multi = len(label_names) > 1
    tasks = []

    for dataset_label, y_map, group_values in zip(dataset_labels, y_by_dataset, group_values_by_dataset):
        gvals = pd.Series(group_values).fillna("缺失").astype(str).to_numpy()
        for group in _sorted_unique(gvals):
            mask = gvals == group
            tasks.append((dataset_label, group, y_map, mask, label_names, display_labels, is_multi, group_name))

    results = parallel_execute(
        _group_stats_row,
        tasks,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        task_labels=[f"{task[0]}:{task[1]}" for task in tasks],
        default_backend="threading",
        workload=ParallelWorkload(
            task_count=len(tasks),
            rows=sum(len(values) for values in group_values_by_dataset),
            columns=max(1, len(label_names)),
            cost_per_item=1.0,
            capability="vectorized",
            operation="报告分组样本统计",
        ),
    )
    index_tuples = [item[0] for item in results]
    rows = [item[1] for item in results]

    if not rows:
        return pd.DataFrame(), []

    index = pd.MultiIndex.from_tuples(index_tuples, names=["数据集", group_name])
    if is_multi:
        columns = [("统计详情", "数据集"), ("统计详情", group_name), ("统计详情", "样本总数")] + [(metric, display_labels.get(label, label)) for metric in ["好样本数", "坏样本数", "坏样本率"] for label in label_names]
        multi_cols = pd.MultiIndex.from_tuples(columns, names=["统计详情", ""])
        result = pd.DataFrame(rows, columns=multi_cols)
        return result, [c for c in multi_cols if c[0] == "坏样本率"]

    return pd.DataFrame(rows, index=index, columns=["样本总数", "好样本数", "坏样本数", "坏样本率"]), ["坏样本率"]


def _as_array(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _sorted_unique(values: Sequence[Any]) -> List[Any]:
    unique_values = pd.unique(pd.Series(values))
    try:
        return sorted(unique_values)
    except TypeError:
        return list(unique_values)
