"""统一逾期标签的比较、灰客户区间和展示名称。"""

import operator

import pandas as pd

from .input_utils import normalize_dpd_values

_OVERDUE_OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def validate_overdue_operator(overdue_operator: str) -> str:
    """校验逾期比较符，避免无效参数被报告或分箱参数静默忽略。"""
    if not isinstance(overdue_operator, str) or overdue_operator not in _OVERDUE_OPERATORS:
        raise ValueError("overdue_operator 仅支持 '>'、'>='、'<'、'<='")
    return overdue_operator


def compare_overdue(values: pd.Series, dpd, overdue_operator: str = ">") -> pd.Series:
    """按指定比较符生成坏样本布尔标记，缺失值不满足比较条件。"""
    validate_overdue_operator(overdue_operator)
    return _OVERDUE_OPERATORS[overdue_operator](values, dpd).fillna(False)


def overdue_grey_mask(values: pd.Series, dpd, overdue_operator: str = ">") -> pd.Series:
    """返回灰客户标记：> 为 (0, dpd]，>= 为 (0, dpd)，< 和 <= 暂无灰客户。"""
    validate_overdue_operator(overdue_operator)
    if overdue_operator == ">":
        return (values.gt(0) & values.le(dpd)).fillna(False)
    if overdue_operator == ">=":
        return (values.gt(0) & values.lt(dpd)).fillna(False)
    return pd.Series(False, index=values.index)


def make_overdue_target(values: pd.Series, dpd, del_grey: bool = False, overdue_operator: str = ">") -> pd.Series:
    """生成 0/1 标签；启用剔灰时只将对应灰客户标记为 NaN，保留行索引。"""
    target = compare_overdue(values, dpd, overdue_operator).astype(int)
    if del_grey and overdue_operator in (">", ">="):
        target = target.astype(float).mask(overdue_grey_mask(values, dpd, overdue_operator))
    return target


def overdue_label(column: str, dpd, overdue_operator: str = ">", style: str = "suffix") -> str:
    """默认 > 沿用各入口的历史命名，其他比较符始终显式展示。"""
    validate_overdue_operator(overdue_operator)
    # 分箱入口会把 3.0 规范为 3；上层报告必须采用同一名称才能查找对应的坏样本率。
    dpd = normalize_dpd_values(dpd)[0]
    if overdue_operator == ">":
        if style == "suffix":
            return f"{column}_{dpd}+"
        if style == "space":
            return f"{column} {dpd}+"
        if style == "at":
            return f"{column}@{dpd}"
    return f"{column}{overdue_operator}{dpd}"
