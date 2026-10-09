"""与显示标签无关、可持久化的分箱语义。"""

from dataclasses import dataclass, field, replace
import keyword
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def field_expression(feature):
    """返回 pandas 表达式字段引用；反引号字段无法无损表达时明确拒绝。"""
    name = str(feature)
    if "`" in name:
        raise ValueError("规则字段名不能包含反引号，请先重命名该字段")
    return name if name.isidentifier() and not keyword.iskeyword(name) else f"`{name}`"


def value_expression(value):
    """不引入 np 名称或无穷变量的标量字面量。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isinf(value):
        return "1e999" if value > 0 else "-1e999"
    if not isinstance(value, (str, bool, int, float, type(None))):
        raise ValueError(f"规则类别暂不支持 {type(value).__name__}，请先显式编码")
    return repr(value)


@dataclass(frozen=True)
class BinSpec:
    """分箱执行语义，标签精度不会改变边界。

    kind 支持 interval、categories、missing、other。other 表示不属于
    values 的非缺失值；exclude_values 用于将特殊值从普通数值箱排除。
    include_missing 可以将缺失合并到指定普通箱。
    """

    feature: str
    bin_id: int
    kind: str = "interval"
    lower: Optional[float] = None
    upper: Optional[float] = None
    closed_left: bool = True
    closed_right: bool = False
    values: Tuple[Any, ...] = field(default_factory=tuple)
    include_values: Tuple[Any, ...] = field(default_factory=tuple)
    known_values: Tuple[Any, ...] = field(default_factory=tuple)
    exclude_values: Tuple[Any, ...] = field(default_factory=tuple)
    include_missing: bool = False
    special: bool = False
    unknown: bool = False
    transform_ref: Optional[str] = None

    def __post_init__(self):
        if self.kind not in {"interval", "categories", "missing", "other"}:
            raise ValueError(f"不支持的分箱语义: {self.kind}")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("分箱下界不能大于上界")

    def mask(self, data):
        """在原始列上执行，不解析展示字符串。"""
        values = data[self.feature] if isinstance(data, pd.DataFrame) else data
        missing = values.isna()
        if self.kind == "missing":
            return missing
        if self.kind in {"categories", "other"}:
            result = values.isin(self.values)
            if self.kind == "other":
                result = ~result
        else:
            result = pd.Series(True, index=values.index)
            if self.lower is not None:
                result &= values.ge(self.lower) if self.closed_left else values.gt(self.lower)
            if self.upper is not None:
                result &= values.le(self.upper) if self.closed_right else values.lt(self.upper)
        result = result.fillna(False) & ~missing
        if self.exclude_values:
            result &= ~values.isin(self.exclude_values)
        if self.include_values:
            result |= values.isin(self.include_values)
        if self.unknown and self.kind != "other":
            result |= ~values.isin(self.known_values) & ~missing
        return result | missing if self.include_missing else result

    def to_expression(self):
        """生成可直接用于 Rule 的原始字段表达式。"""
        name = field_expression(self.feature)
        missing = f"({name}.isna())"
        if self.kind == "missing":
            return missing
        parts = []
        if self.kind in {"categories", "other"}:
            members = ", ".join(value_expression(value) for value in self.values)
            parts.append(f"({name} {'not in' if self.kind == 'other' else 'in'} [{members}])")
        else:
            if self.lower is not None:
                parts.append(f"({name} {'>=' if self.closed_left else '>'} {value_expression(self.lower)})")
            if self.upper is not None:
                parts.append(f"({name} {'<=' if self.closed_right else '<'} {value_expression(self.upper)})")
        parts.append(f"({name}.notna())")
        if self.exclude_values:
            members = ", ".join(value_expression(value) for value in self.exclude_values)
            parts.append(f"({name} not in [{members}])")
        expression = " & ".join(parts)
        if self.include_values:
            members = ", ".join(value_expression(value) for value in self.include_values)
            expression = f"({expression}) | ({name} in [{members}])"
        if self.unknown and self.kind != "other":
            members = ", ".join(value_expression(value) for value in self.known_values)
            expression = f"({expression}) | (({name} not in [{members}]) & ({name}.notna()))"
        return f"({expression}) | {missing}" if self.include_missing else expression

    @classmethod
    def from_binner(cls, binner, feature):
        """从本库已拟合分箱器取得箱号到语义的映射，不读取显示标签。

        使用 BaseBinning 的真实保留箱策略处理缺失/特殊/未知类别归属。
        对不提供该契约的外部分箱器明确拒绝，不能猜测字符串区间。
        """
        if feature not in getattr(binner, "splits_", {}):
            raise ValueError(f"分箱器不存在已拟合字段: {feature}")
        if not hasattr(binner, "_assign_base_feature_bins") or not hasattr(binner, "_apply_reserved_bin_policy"):
            raise ValueError("当前分箱器不支持结构化分箱契约")
        from ._categorical import is_missing_marker

        special = tuple(value for value in (binner.special_codes or []) if not is_missing_marker(value))

        def code_for(value):
            series = pd.Series([value])
            codes = binner._assign_base_feature_bins(feature, series)
            return int(binner._apply_reserved_bin_policy(feature, series, codes)[0])

        missing_target = code_for(np.nan)
        specs = {}
        categorical = binner.feature_types_.get(feature) == "categorical"
        if categorical:
            groups = binner._cat_bins_[feature]
            known = tuple(value for group in groups for value in group if not is_missing_marker(value)) + special
            for index, group in enumerate(groups):
                members = tuple(value for value in group if not is_missing_marker(value))
                specs[index] = cls(feature, index, kind="categories", values=members)
        else:
            splits = np.asarray(binner.splits_[feature], dtype=float)
            for index in range(len(splits) + 1):
                specs[index] = cls(
                    feature,
                    index,
                    lower=None if index == 0 else float(splits[index - 1]),
                    upper=None if index == len(splits) else float(splits[index]),
                    exclude_values=special,
                )
        for value in special:
            target = code_for(value)
            spec = specs.get(target, cls(feature, target, kind="categories", special=True))
            specs[target] = replace(spec, include_values=spec.include_values + (value,))
        if missing_target in specs:
            specs[missing_target] = replace(specs[missing_target], include_missing=True)
        else:
            specs[missing_target] = cls(feature, missing_target, kind="missing", special=missing_target == -2)
        unknown_target = binner.handle_unknown
        if categorical and unknown_target != "raise":
            unknown_target = int(unknown_target)
            spec = specs.get(unknown_target, cls(feature, unknown_target, kind="categories"))
            if spec.kind == "missing":
                spec = replace(spec, kind="categories", include_missing=True)
            specs[unknown_target] = replace(spec, unknown=True, known_values=known)
        return specs

    def label(self, precision=4):
        """只用于展示的中文标签。"""
        if self.special:
            return "特殊值"
        if self.kind == "missing":
            return "缺失"
        if self.kind == "other":
            return "其他"
        if self.kind == "categories":
            return str(list(self.values))
        lower = "-inf" if self.lower is None else format(self.lower, f".{precision}g")
        upper = "+inf" if self.upper is None else format(self.upper, f".{precision}g")
        return f"{'[' if self.closed_left else '('}{lower}, {upper}{']' if self.closed_right else ')'}"
