"""任务版本上的多级透视表，按实际出现的维度组合分页计算。"""

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .interactions import _axis_buckets, _axis_values
from .targets import validate_columns


class PivotRequest(BaseModel):
    class Config:
        extra = "forbid"

    rows: List[str] = Field(min_items=1)
    columns: List[str] = Field(min_items=1)
    value: Optional[str] = None
    aggregation: Literal["rows", "count", "nunique", "sum", "mean", "median", "min", "max"] = "rows"
    buckets: bool = True
    group: Optional[str] = None
    page: int = Field(1, ge=1)
    column_page: int = Field(1, ge=1)


def _dimension(series, axis, bins):
    if axis:
        values, levels = _axis_buckets(_axis_values(series, axis), axis, bins)
        levels = list(dict.fromkeys(levels))
        return pd.Categorical(values, categories=levels).codes, levels
    codes, values = pd.factorize(series, sort=True)
    levels = [str(v) for v in values]
    if (codes < 0).any():
        codes = np.where(codes < 0, len(levels), codes)
        levels.append(None)
    return codes, levels


def _page(dimensions, levels, page, size):
    keys = dimensions.drop_duplicates().sort_values(list(dimensions))
    ids = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(dimensions)) if len(keys) else np.empty(0, dtype=int)
    start = (page - 1) * size
    visible = keys.iloc[start:start + size]
    labels = [[levels[i][code] for i, code in enumerate(key)] for key in visible.itertuples(index=False, name=None)]
    return ids, start, labels, len(keys)


def _aggregate(values, method):
    if method == "rows":
        return values.size() if callable(values.size) else values.size
    if method == "sum":
        return values.sum(min_count=1)
    return getattr(values, method)()


def _number(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    return value if np.isfinite(value) else None


def pivot_table(workspace, job, request):
    interaction = (job.get("result") or {}).get("interaction")
    if not interaction:
        raise ValueError("请选择已完成的组合探索任务")
    dimensions = request.rows + request.columns
    source = workspace.dataset_version(job["dataset_id"], job["dataset_revision"], include_examples=False)
    available = {column["name"] for column in source["columns"]}
    validate_columns(available, dimensions)
    fields = list(dimensions)
    if request.aggregation != "rows":
        if request.value not in available:
            raise ValueError("请选择值字段")
        fields.append(request.value)
    label = interaction.get("target")
    if request.group is not None:
        if not label:
            raise ValueError("该任务未设置分组标签")
        fields.append(label)
    data = workspace.read_dataset(source["id"], job["dataset_revision"], columns=list(dict.fromkeys(fields)))
    axes = {axis["name"]: axis for axis in interaction["axes"]} if request.buckets else {}
    codes, levels = {}, []
    for i, name in enumerate(dimensions):
        codes[i], items = _dimension(data[name], axes.get(name), job["config"]["bins"])
        levels.append(items)
    keys = pd.DataFrame(codes, index=data.index)
    valid = keys.ge(0).all(axis=1)
    if request.group is not None:
        valid &= data[label].astype("string").fillna("（标签缺失）").eq(request.group)
    keys = keys.loc[valid]
    split = len(request.rows)
    r, r0, row_keys, height = _page(keys.iloc[:, :split], levels[:split], request.page, 30)
    c, c0, col_keys, width = _page(keys.iloc[:, split:], levels[split:], request.column_page, 12)
    cells = pd.DataFrame({"r": r, "c": c}, index=keys.index)
    if request.aggregation == "rows":
        cells["v"] = 1
    else:
        measure = data[request.value]
        if request.aggregation not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(measure):
            raise ValueError("该聚合方式需要数值类型的值字段")
        if pd.api.types.is_float_dtype(measure):
            measure = measure.astype("float64").replace([np.inf, -np.inf], np.nan)
        cells["v"] = measure
    # 合计始终在完整配对上重新聚合，包括非当前页的行列。
    in_rows = cells.r.between(r0, r0 + len(row_keys) - 1)
    in_cols = cells.c.between(c0, c0 + len(col_keys) - 1)
    def aggregate(mask, by):
        return _aggregate(cells.loc[mask].groupby(by, sort=False)["v"], request.aggregation)

    matrix = aggregate(in_rows & in_cols, ["r", "c"]).to_dict()
    row_totals, col_totals = aggregate(in_rows, "r"), aggregate(in_cols, "c")
    empty = 0 if request.aggregation in {"rows", "count", "nunique"} else None
    return {
        "row_keys": row_keys, "column_keys": col_keys, "row_count": height, "column_count": width,
        "values": [[_number(matrix.get((r0+i, c0+j), empty)) for j in range(len(col_keys))] for i in range(len(row_keys))],
        "row_totals": [_number(row_totals[r0+i]) for i in range(len(row_keys))],
        "column_totals": [_number(col_totals[c0+j]) for j in range(len(col_keys))],
        "total": _number(_aggregate(cells["v"], request.aggregation)), "samples": len(cells),
    }
