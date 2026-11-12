"""数据管理的多份划分配置、公共切分器适配与结果预览。"""

import math
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from hscredit.core.model_selection import group_train_test_split, oot_split, time_train_test_split

from .storage import records


class SplitPart(BaseModel):
    class Config:
        extra = "forbid"

    name: str = Field(min_length=1, max_length=150)
    ratio: Optional[float] = Field(default=None, gt=0, lt=1)


class SplitRequest(BaseModel):
    class Config:
        extra = "forbid"

    expected_revision: int = Field(ge=1)
    method: Literal["random", "time", "group", "oot"] = "random"
    parts: List[SplitPart] = Field(min_items=2, max_items=10)
    random_state: int = Field(default=42, ge=0, le=2147483647)
    stratify: Optional[str] = None
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    time_mode: Literal["ratio", "cutoff"] = "ratio"
    cutoffs: List[str] = Field(default_factory=list, max_items=9)
    gap: int = Field(default=0, ge=0, le=2000000)
    oot_start: Optional[str] = None
    oot_end: Optional[str] = None


def _counts(total, ratios):
    """最大余数法分配整数份额，避免逐次浮点取整放大偏差。"""
    exact = np.asarray(ratios) * total
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    if (counts < 1).any():
        raise ValueError("样本数或分组数不足，请减少数据集份数或调整比例、间隔")
    return counts


def split_frame(data, request):
    """对全量数据调用既有切分器，保留全部字段和行内容。"""
    names = [part.name.strip() for part in request.parts]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("每个结果数据集名称必须非空且互不重复")
    method = request.method
    field = request.time_col if method in {"time", "oot"} else request.group_col if method == "group" else request.stratify
    if method != "random" and not field:
        raise ValueError("请选择划分所需的时间或分组字段")
    if field:
        if field not in data:
            raise ValueError(f"划分字段不存在：{field}")
        if data[field].isna().any():
            raise ValueError(f"划分字段「{field}」包含缺失值，请先通过字段加工处理")
    # 平台保留原始日期字段，仅用切分器转换后的时间确定行归属。
    if method in {"time", "oot"} and pd.api.types.is_numeric_dtype(data[field]):
        raise ValueError("时间划分需要日期或日期文本字段，不能直接使用数值字段")
    if request.stratify and method != "random":
        raise ValueError("分层字段仅适用于随机划分")
    parts = []
    try:
        if method == "oot":
            if len(names) != 2 or not request.oot_start:
                raise ValueError("时间外划分需要两个结果数据集及时间外起始时间")
            parts = list(oot_split(data, field, request.oot_start, oot_end=request.oot_end, gap=request.gap))
        else:
            cutoffs = None
            if method == "time" and request.time_mode == "cutoff":
                if len(request.cutoffs) != len(names) - 1:
                    raise ValueError("时间边界数量必须比结果数据集数量少一个")
                cutoffs = [pd.Timestamp(value) for value in request.cutoffs]
                if any(pd.isna(value) for value in cutoffs) or any(a >= b for a, b in zip(cutoffs, cutoffs[1:])):
                    raise ValueError("时间边界必须为有效时间，并按从早到晚的顺序填写")
            else:
                ratios = [part.ratio for part in request.parts]
                if any(value is None or not math.isfinite(value) for value in ratios) or not math.isclose(sum(ratios), 1, rel_tol=0, abs_tol=1e-8):
                    raise ValueError("每份比例必须大于 0，且合计为 100%")
                total = data[field].nunique() if method == "group" else len(data) - (request.gap * (len(names) - 1) if method == "time" else 0)
                if total < len(names):
                    raise ValueError("样本数或分组数不足，请减少份数或时间间隔")
                counts = _counts(total, ratios)
            remaining = data
            # 从最后一份向前拆分，时间划分天然保持从早到晚的结果顺序。
            for index in range(len(names) - 1, 0, -1):
                if method == "time":
                    kwargs = {"cutoff": cutoffs[index - 1]} if cutoffs is not None else {"test_size": int(counts[index])}
                    remaining, part = time_train_test_split(remaining, field, gap=request.gap, **kwargs)
                elif method == "group":
                    remaining, part = group_train_test_split(remaining, groups=remaining[field], test_size=int(counts[index]), random_state=request.random_state)
                else:
                    remaining, part = train_test_split(remaining, test_size=int(counts[index]), random_state=request.random_state, stratify=remaining[field] if field else None)
                parts.insert(0, part)
            parts.insert(0, remaining)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"无法按当前配置划分，请检查比例、时间边界、分层类别样本数或分组数：{exc}") from exc
    if any(part.empty for part in parts):
        raise ValueError("划分后存在空数据集，请调整配置")
    return list(zip(names, parts))


def split_preview(data, parts, request):
    """展示实际结果和排除行数，比例以源数据集全部行为分母。"""
    result = []
    for name, part in parts:
        item = {"name": name, "rows": len(part), "fraction": len(part) / len(data), "preview": records(part.head(5))}
        if request.method in {"time", "oot"}:
            dates = pd.to_datetime(part[request.time_col])
            item.update(time_min=dates.min().isoformat(), time_max=dates.max().isoformat())
        if request.method == "group":
            item["groups"] = int(part[request.group_col].nunique())
        if request.method == "random" and request.stratify:
            counts = part[request.stratify].value_counts().rename_axis("类别").reset_index(name="样本数")
            item["strata"] = records(counts.head(50))
        result.append(item)
    return {"revision": request.expected_revision, "source_rows": len(data), "excluded_rows": len(data) - sum(len(part) for _, part in parts), "parts": result}
