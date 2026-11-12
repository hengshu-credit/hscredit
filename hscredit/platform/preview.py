"""数据预览的字段取值查询、组合筛选与分页。"""

from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, StrictStr

from .storage import records


class PreviewFilter(BaseModel):
    query: str = ""
    filters: Dict[str, List[Optional[StrictStr]]] = Field(default_factory=dict)


def preview_values(series, query, limit=80):
    """返回可精确匹配的文本值；缺失值单独使用 JSON null 表示。"""
    values = series.drop_duplicates().astype("string")
    result = []
    if values.isna().any() and (not query or query in "缺失值"):
        result.append(None)
    values = values.dropna()
    empty = values.eq("")
    if empty.any():
        if not query or query in "空字符串":
            result.append("")
        values = values[~empty]
    if query:
        matches = values.str.contains(query, regex=False, case=False)
        exact = values.eq(query)
        values = pd.concat([values[exact], values[matches & ~exact]])
    return result + values.head(limit - len(result)).tolist()


def preview_rows(data, page, page_size, query, filters):
    mask = None
    for field, values in filters.items():
        if field not in data:
            raise ValueError("请选择数据集中存在的字段")
        column = data[field].astype("string")
        match = column.isin(values)
        if None in values:
            match |= column.isna()
        mask = match if mask is None else mask & match
    if mask is not None:
        data = data.loc[mask]
    if query and not data.empty:
        data = data.loc[data.astype(str).apply(lambda c: c.str.contains(query, regex=False)).any(axis=1)]
    start = (page - 1) * page_size
    return {"rows": records(data.iloc[start : start + page_size]), "total": len(data)}
