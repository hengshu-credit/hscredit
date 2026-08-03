"""类别变量分箱的类型安全排序与规则辅助函数。"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


CategoryOrder = Optional[
    Union[
        Dict[str, Sequence[Any]],
        Callable[[str, pd.Series, pd.Series], Sequence[Any]],
    ]
]


def is_missing_marker(value: Any) -> bool:
    """判断标量是否为统一缺失标记。"""
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def typed_equal(left: Any, right: Any) -> bool:
    """按归一化后的原生类型和值比较两个类别。"""
    if is_missing_marker(left) or is_missing_marker(right):
        return is_missing_marker(left) and is_missing_marker(right)
    left = _normalize_scalar(left)
    right = _normalize_scalar(right)
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _contains_typed(values: Sequence[Any], candidate: Any) -> bool:
    return any(typed_equal(value, candidate) for value in values)


def _is_special(value: Any, special_codes: Optional[Sequence[Any]]) -> bool:
    return bool(special_codes) and _contains_typed(special_codes, value)


def unique_non_missing_typed(values: pd.Series, special_codes: Optional[Sequence[Any]] = None) -> List[Any]:
    """按首次出现顺序返回非缺失、非特殊的类型安全唯一类别。"""
    unique: List[Any] = []
    for value in values.tolist():
        if is_missing_marker(value) or _is_special(value, special_codes):
            continue
        value = _normalize_scalar(value)
        if not _contains_typed(unique, value):
            unique.append(value)
    return unique


def _typed_mask(values: pd.Series, category: Any) -> pd.Series:
    return values.map(lambda value: typed_equal(value, category)).astype(bool)


def _validate_supplied_order(feature: str, supplied: Sequence[Any], observed: Sequence[Any]) -> List[Any]:
    ordered = [_normalize_scalar(value) for value in list(supplied)]
    if any(is_missing_marker(value) for value in ordered):
        raise ValueError(f"特征 '{feature}' 的 category_order 不能包含缺失值")

    duplicates = [
        value
        for index, value in enumerate(ordered)
        if _contains_typed(ordered[:index], value)
    ]
    if duplicates:
        raise ValueError(f"特征 '{feature}' 的 category_order 包含重复类别: {duplicates}")

    missing = [value for value in observed if not _contains_typed(ordered, value)]
    unknown = [value for value in ordered if not _contains_typed(observed, value)]
    if missing or unknown:
        raise ValueError(
            f"特征 '{feature}' 的 category_order 必须完整覆盖训练类别；"
            f"缺少类别: {missing}，未知类别: {unknown}"
        )
    return ordered


def resolve_category_order(
    feature: str,
    x: pd.Series,
    y: pd.Series,
    category_order: CategoryOrder = None,
    special_codes: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """解析用户顺序，未提供时按坏样本率和首次出现顺序排序。"""
    observed = unique_non_missing_typed(x, special_codes)
    supplied = None
    if callable(category_order):
        supplied = category_order(feature, x.copy(), y.copy())
    elif isinstance(category_order, dict) and feature in category_order:
        supplied = category_order[feature]

    if supplied is not None:
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
            raise ValueError(f"特征 '{feature}' 的 category_order 必须返回类别序列")
        return _validate_supplied_order(feature, supplied, observed)

    ranked = []
    for first_seen, category in enumerate(observed):
        mask = _typed_mask(x, category)
        count = int(mask.sum())
        bad_rate = float(y.loc[mask].mean()) if count else 0.0
        ranked.append((bad_rate, first_seen, category))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]

