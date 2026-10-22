"""分析入口的显式标签与逾期标签解析、灰客户处理和位置绑定。"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .input_utils import normalize_dpd_values
from .overdue import make_overdue_target, overdue_label, validate_overdue_operator

TargetValues = Union[pd.Series, np.ndarray, list]
TargetInput = Union[str, TargetValues, Callable, Mapping]


@dataclass(frozen=True)
class BoundLabel:
    """标签展示名与私有数据列的绑定，不覆盖用户字段。"""

    name: str
    column: str
    overdue: Optional[str] = None
    dpd: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class BoundTargets:
    """已经求值的标签定义；私有列随数据一起筛选和排序。"""

    labels: tuple
    mode: str = "target"
    operator: str = ">"
    source_columns: tuple = ()
    explicit_multi: bool = False

    def __len__(self):
        return len(self.labels)

    @property
    def columns(self):
        return [label.column for label in self.labels]

    def names(self, style="suffix"):
        return [
            (
                overdue_label(label.overdue, label.dpd, self.operator, style=style)
                if label.overdue is not None
                else label.name
            )
            for label in self.labels
        ]

    def single(self, position=0):
        return replace(self, labels=(self.labels[position],), explicit_multi=False)

    def y_map(self, data, style="suffix"):
        return {name: data[label.column] for name, label in zip(self.names(style), self.labels)}

    def needs_filtering(self, data):
        return bool(data[self.columns].isna().any().any())


def validate_target_mode(target=None, overdue=None, dpds=None):
    """两种模式互斥，逾期字段和阈值必须成对指定。"""
    if target is not None and (overdue is not None or dpds is not None):
        raise ValueError("target 与 overdue+dpds 不能同时指定，请选择一种标签模式")
    if (overdue is None) != (dpds is None):
        raise ValueError("overdue 和 dpds 必须同时传入")


def _explicit_values(data, value, name, del_grey, dataset_name, allow_missing=False):
    context = f"数据集“{dataset_name}”的" if dataset_name is not None else ""
    context += f"标签“{name}”"
    if callable(value):
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            try:
                signature.bind(data)
            except TypeError as exc:
                raise ValueError(f"{context}的 callable 必须能接收一个 DataFrame 位置参数: {exc}") from exc
        try:
            value = value(data.copy(deep=True))
        except Exception as exc:
            raise ValueError(f"{context}计算失败: {exc}") from exc
    if not isinstance(value, (pd.Series, np.ndarray, list)):
        raise TypeError(f"{context}必须是一维 Series、ndarray、list 或返回这些数据的 callable")
    if isinstance(value, pd.Series):
        if not value.index.equals(data.index):
            if not data.index.is_unique or not value.index.is_unique:
                raise ValueError(f"{context}含重复索引，必须与作用数据的索引完全同序")
            if len(value) != len(data) or not data.index.isin(value.index).all():
                raise ValueError(f"{context}索引必须与作用数据完全匹配")
            value = value.reindex(data.index)
        values = value.to_numpy()
    else:
        values = np.asarray(value)
    if values.ndim != 1 or len(values) != len(data):
        raise ValueError(f"{context}必须是一维且与作用数据等长（数据行数: {len(data)}）")
    series = pd.Series(values, index=data.index)
    # 非 0/1（包含缺失）为灰客户；不剔灰时统一按负样本 0 计。
    binary = series.isin([0, 1])
    result = series.eq(1).fillna(False).astype(float)
    if del_grey:
        return result.where(binary)
    return result.where(series.notna()) if allow_missing else result


def bind_targets(
    data,
    target=None,
    overdue=None,
    dpds=None,
    del_grey=False,
    overdue_operator=">",
    *,
    default_target=None,
    paired=False,
    dataset_name=None,
    allow_missing=False,
):
    """在标签原作用数据上求值一次，绑定私有列供所有后续子任务使用。"""
    validate_target_mode(target, overdue, dpds)
    validate_overdue_operator(overdue_operator)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("标签作用数据必须是 DataFrame")
    if not data.columns.is_unique:
        raise ValueError("标签作用数据的列名不能重复")
    if isinstance(target, BoundTargets):
        missing = [column for column in target.columns if column not in data]
        if missing:
            raise ValueError("已解析标签的绑定列缺失，筛选数据时必须保留标签列")
        return data, target

    definitions = []
    source_columns = []
    explicit_multi = isinstance(target, Mapping)
    if overdue is not None:
        overdue_columns = [overdue] if isinstance(overdue, str) else list(overdue)
        thresholds = normalize_dpd_values(dpds, deduplicate=not paired)
        if not overdue_columns or not thresholds:
            raise ValueError("overdue 和 dpds 不能为空")
        if paired and len(overdue_columns) != len(thresholds):
            raise ValueError("overdue 和 dpds 长度必须一致")
        pairs = (
            zip(overdue_columns, thresholds)
            if paired
            else ((column, threshold) for column in overdue_columns for threshold in thresholds)
        )
        seen = set()
        for column, threshold in pairs:
            if not isinstance(column, str) or not column:
                raise ValueError("overdue 必须为非空字段名或字段名列表")
            if (column, threshold) in seen:
                continue
            seen.add((column, threshold))
            if column not in data:
                if not allow_missing:
                    raise ValueError(f"数据集缺少逾期字段: {column}")
                values = pd.Series(np.nan, index=data.index)
            else:
                days = pd.to_numeric(data[column], errors="raise")
                values = make_overdue_target(days, threshold, del_grey, overdue_operator)
                if allow_missing:
                    values = values.where(days.notna())
            definitions.append((overdue_label(column, threshold, overdue_operator), values, column, threshold))
        source_columns = list(dict.fromkeys(overdue_columns))
        mode = "overdue"
    else:
        mode = "target"
        if target is None:
            target = default_target
        if target is None:
            raise ValueError("必须传入 target 或 overdue+dpds 参数")
        if isinstance(target, str):
            if target not in data and not allow_missing:
                raise ValueError(f"数据集缺少目标变量字段: {target}")
            source_columns = [target]
            inputs = {target: data[target] if target in data else pd.Series(np.nan, index=data.index)}
        elif isinstance(target, Mapping):
            if not target:
                raise ValueError("target 标签字典不能为空")
            # 旧配置不再具有特殊语义，也不允许字符串冒充标签数据。
            inputs = target
        else:
            inputs = {getattr(target, "name", None) or "target": target}
        for name, value in inputs.items():
            if not isinstance(name, str) or not name.strip() or name == "分箱详情":
                raise ValueError("标签名必须为非空字符串，且不能使用保留名称“分箱详情”")
            values = _explicit_values(data, value, name, del_grey, dataset_name, allow_missing)
            definitions.append((name, values, None, None))

    frame = data.copy(deep=False)
    labels: List[BoundLabel] = []
    for position, (name, values, column, threshold) in enumerate(definitions):
        bound_column = f"__hscredit_label_{position}__"
        while bound_column in frame:
            bound_column = "_" + bound_column
        frame[bound_column] = np.asarray(values, dtype=float)
        labels.append(BoundLabel(name, bound_column, column, threshold))
    return frame, BoundTargets(tuple(labels), mode, overdue_operator, tuple(source_columns), explicit_multi)


def target_count(target=None, overdue=None, dpds=None):
    """返回实际标签任务数，不执行 callable。"""
    if isinstance(target, BoundTargets):
        return len(target)
    if isinstance(target, Mapping):
        return len(target)
    if overdue is not None:
        return len([overdue] if isinstance(overdue, str) else overdue) * len(normalize_dpd_values(dpds))
    return 1


def reject_legacy_target_keywords(kwargs):
    """避免旧标签参数被算法的 **kwargs 静默吞掉。"""
    for name, replacement in (("target_col", "target"), ("dpd", "dpds")):
        if name in kwargs:
            raise TypeError(f"标签参数已统一，请将 {name} 改为 {replacement}")
