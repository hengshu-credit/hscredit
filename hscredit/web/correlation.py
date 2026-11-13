"""数据管理的全字段相关性矩阵，按所见行列投影字段，使用全部共同有效样本。"""

from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .targets import validate_columns


class CorrelationFields(BaseModel):
    """当前矩阵窗口内按展示顺序排列的字段。"""

    rows: List[str] = Field(min_items=1, max_items=64)
    columns: List[str] = Field(min_items=1, max_items=64)


def _numeric_values(series, kind):
    if kind == "date":
        dates = pd.to_datetime(series, errors="coerce", utc=True)
        series = dates.astype("int64").astype("Int64").where(dates.notna())
    if pd.api.types.is_integer_dtype(series.dtype) and series.notna().any():
        # 先平移再转浮点，保留大整数编号、精细时间戳之间的微小差值。
        low, high = int(series.min()), int(series.max())
        if pd.api.types.is_unsigned_integer_dtype(series.dtype):
            series = series - low
        else:
            series = series - ((low + high + 1) // 2)
    values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _categorical_association(left, right):
    valid = (left >= 0) & (right >= 0)
    left, right = left[valid], right[valid]
    left_counts, right_counts = np.bincount(left), np.bincount(right)
    dimension = min(np.count_nonzero(left_counts), np.count_nonzero(right_counts)) - 1
    if dimension < 1:
        return None
    # 只统计实际出现的类别对，避免为高基数字段分配稠密列联表。
    width = len(right_counts)
    cells, counts = np.unique(left.astype(np.int64) * width + right, return_counts=True)
    expected_factors = left_counts[cells // width].astype(float) * right_counts[cells % width]
    phi_squared = np.sum(counts.astype(float) ** 2 / expected_factors) - 1
    return float(np.sqrt(np.clip(phi_squared / dimension, 0, 1)))


def _correlation_ratio(groups, values):
    valid = (groups >= 0) & np.isfinite(values)
    groups, values = groups[valid], values[valid]
    counts = np.bincount(groups)
    present = counts > 0
    if present.sum() < 2 or values.min() == values.max():
        return None
    scale = np.max(np.abs(values))
    centered = values / scale
    centered -= centered.mean()
    group_sums = np.bincount(groups, weights=centered)[present]
    between = np.sum(group_sums**2 / counts[present])
    return float(np.sqrt(np.clip(between / np.sum(centered**2), 0, 1)))


def dataset_correlation(workspace, identifier, revision, row_page=1, column_page=1, page_size=16, *, selection=None):
    metadata = workspace.dataset_version(identifier, revision, include_examples=False)
    fields = metadata["columns"]

    def page_fields(page):
        offset = (page - 1) * page_size
        if offset >= len(fields):
            raise ValueError("相关性矩阵页码超出字段范围")
        return [field["name"] for field in fields[offset : offset + page_size]]

    kinds = {field["name"]: field["kind"] for field in fields}
    if selection is None:
        rows, columns = page_fields(row_page), page_fields(column_page)
    else:
        rows, columns = selection.rows, selection.columns
        validate_columns(kinds, rows)
        validate_columns(kinds, columns)
    names = list(dict.fromkeys(rows + columns))
    version = metadata.get("revision", 1)
    data = workspace.read_dataset(identifier, version, columns=names)
    categories = {name: pd.factorize(data[name])[0] for name in names if kinds[name] == "category"}
    numeric = pd.DataFrame({name: _numeric_values(data[name], kinds[name]) for name in names if name not in categories})
    pearson = numeric.corr(min_periods=2)
    cache = {}

    def coefficient(a, b):
        if a not in categories and b not in categories:
            value = pearson.at[a, b]
            return float(value) if np.isfinite(value) else None
        key = tuple(sorted((a, b)))
        if key not in cache:
            if a in categories and b in categories:
                cache[key] = _categorical_association(categories[a], categories[b])
            else:
                category, number = (a, b) if a in categories else (b, a)
                cache[key] = _correlation_ratio(categories[category], numeric[number].to_numpy())
        return cache[key]

    return {"revision": version, "rows": rows, "columns": columns, "total_fields": len(fields), "values": [[coefficient(a, b) for b in columns] for a in rows]}
