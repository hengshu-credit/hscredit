"""报告样本统计表构造工具."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_sample_stats_table(
    dataset_labels: Sequence[str],
    y_by_dataset: Sequence[Dict[str, Sequence[int]]],
    label_names: Sequence[str],
    display_labels: Optional[Dict[str, str]] = None,
    flat_total_col: str = "样本总数",
) -> Tuple[pd.DataFrame, List[Any]]:
    """构造数据集样本统计表."""
    display_labels = display_labels or {}
    label_names = list(label_names)
    is_multi = len(label_names) > 1

    if is_multi:
        columns = [("统计详情", "样本总数")] + [
            (metric, display_labels.get(label, label))
            for metric in ["好样本数", "坏样本数", "坏样本率"]
            for label in label_names
        ]
        multi_cols = pd.MultiIndex.from_tuples(columns, names=["统计详情", ""])
        rows: List[Dict[Any, Any]] = []
        for y_map in y_by_dataset:
            row: Dict[Any, Any] = {}
            first_y = _as_array(y_map[label_names[0]])
            row[("统计详情", "样本总数")] = len(first_y)
            for label in label_names:
                y_arr = _as_array(y_map[label])
                n = len(y_arr)
                nb = int(np.nansum(y_arr))
                row[("好样本数", display_labels.get(label, label))] = n - nb
                row[("坏样本数", display_labels.get(label, label))] = nb
                row[("坏样本率", display_labels.get(label, label))] = float(np.nanmean(y_arr)) if n else 0.0
            rows.append(row)
        result = pd.DataFrame(rows, index=list(dataset_labels), columns=multi_cols)
        result.index.name = "数据集"
        return result, [c for c in multi_cols if c[0] == "坏样本率"]

    label = label_names[0]
    rows = []
    for dataset_label, y_map in zip(dataset_labels, y_by_dataset):
        y_arr = _as_array(y_map[label])
        n = len(y_arr)
        nb = int(np.nansum(y_arr))
        rows.append({
            "数据集": dataset_label,
            flat_total_col: n,
            "好样本数": n - nb,
            "坏样本数": nb,
            "坏样本率": float(np.nanmean(y_arr)) if n else 0.0,
        })
    return pd.DataFrame(rows), ["坏样本率"]


def build_group_distribution_table(
    dataset_labels: Sequence[str],
    y_by_dataset: Sequence[Dict[str, Sequence[int]]],
    group_values_by_dataset: Sequence[Sequence[Any]],
    label_names: Sequence[str],
    display_labels: Optional[Dict[str, str]] = None,
    group_name: str = "数据分组",
) -> Tuple[pd.DataFrame, List[Any]]:
    """构造数据集分组分布表."""
    display_labels = display_labels or {}
    label_names = list(label_names)
    is_multi = len(label_names) > 1
    index_tuples: List[tuple] = []
    rows: List[Dict[Any, Any]] = []

    for dataset_label, y_map, group_values in zip(dataset_labels, y_by_dataset, group_values_by_dataset):
        gvals = pd.Series(group_values).fillna("缺失").astype(str).to_numpy()
        for group in _sorted_unique(gvals):
            mask = gvals == group
            index_tuples.append((dataset_label, str(group)))
            if is_multi:
                row: Dict[Any, Any] = {
                    ("统计详情", "数据集"): dataset_label,
                    ("统计详情", group_name): str(group),
                }
                first_y = _as_array(y_map[label_names[0]])[mask]
                row[("统计详情", "样本总数")] = int(len(first_y))
                for label in label_names:
                    y_group = _as_array(y_map[label])[mask]
                    n = len(y_group)
                    nb = int(np.nansum(y_group))
                    display_label = display_labels.get(label, label)
                    row[("好样本数", display_label)] = n - nb
                    row[("坏样本数", display_label)] = nb
                    row[("坏样本率", display_label)] = float(np.nanmean(y_group)) if n else 0.0
            else:
                y_group = _as_array(y_map[label_names[0]])[mask]
                n = len(y_group)
                nb = int(np.nansum(y_group))
                row = {
                    "样本总数": n,
                    "好样本数": n - nb,
                    "坏样本数": nb,
                    "坏样本率": float(np.nanmean(y_group)) if n else 0.0,
                }
            rows.append(row)

    if not rows:
        return pd.DataFrame(), []

    index = pd.MultiIndex.from_tuples(index_tuples, names=["数据集", group_name])
    if is_multi:
        columns = [("统计详情", "数据集"), ("统计详情", group_name), ("统计详情", "样本总数")] + [
            (metric, display_labels.get(label, label))
            for metric in ["好样本数", "坏样本数", "坏样本率"]
            for label in label_names
        ]
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
