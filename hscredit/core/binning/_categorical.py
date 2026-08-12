"""类别变量分箱的类型安全排序与规则辅助函数。"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._contracts import HandleUnknown, MISSING_BIN, SPECIAL_BIN, UNKNOWN_BIN, is_missing_marker, validate_handle_unknown


CategoryOrder = Optional[
    Union[
        Dict[str, Sequence[Any]],
        Callable[[str, pd.Series, pd.Series], Sequence[Any]],
    ]
]


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

    duplicates = [value for index, value in enumerate(ordered) if _contains_typed(ordered[:index], value)]
    if duplicates:
        raise ValueError(f"特征 '{feature}' 的 category_order 包含重复类别: {duplicates}")

    missing = [value for value in observed if not _contains_typed(ordered, value)]
    unknown = [value for value in ordered if not _contains_typed(observed, value)]
    if missing or unknown:
        raise ValueError(f"特征 '{feature}' 的 category_order 必须完整覆盖训练类别；" f"缺少类别: {missing}，未知类别: {unknown}")
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


def encode_ordered_categories(
    x: pd.Series,
    ordered_categories: Sequence[Any],
    special_codes: Optional[Sequence[Any]] = None,
) -> pd.Series:
    """将普通类别映射为连续浮点编码，缺失值和特殊值保留为 NaN。"""
    encoded = pd.Series(np.nan, index=x.index, dtype=float, name=x.name)
    for code, category in enumerate(ordered_categories):
        encoded.loc[_typed_mask(x, category)] = float(code)
    if special_codes:
        for special in special_codes:
            encoded.loc[_typed_mask(x, special)] = np.nan
    return encoded


def restore_category_groups(ordered_categories: Sequence[Any], numeric_splits: Sequence[float]) -> List[List[Any]]:
    """将数值切分点还原为按数值箱顺序排列的类别组。"""
    categories = list(ordered_categories)
    if not categories:
        return []
    splits = np.asarray(list(numeric_splits), dtype=float)
    splits = splits[np.isfinite(splits)]
    splits = np.unique(np.sort(splits))
    bin_indices = np.digitize(np.arange(len(categories), dtype=float), splits)
    return [
        [category for category, actual_bin in zip(categories, bin_indices) if actual_bin == expected_bin]
        for expected_bin in sorted(set(bin_indices.tolist()))
    ]


def assign_category_groups(
    feature: str,
    x: pd.Series,
    groups: Sequence[Sequence[Any]],
    special_codes: Optional[Sequence[Any]] = None,
    missing_separate: bool = True,
    handle_unknown: HandleUnknown = UNKNOWN_BIN,
) -> np.ndarray:
    """按类型安全规则应用类别组。"""
    unknown_policy = validate_handle_unknown(handle_unknown)
    unknown_bin = UNKNOWN_BIN if unknown_policy == "raise" else unknown_policy
    bins = np.full(len(x), unknown_bin, dtype=int)
    user_owned = np.zeros(len(x), dtype=bool)
    missing_mask = x.map(is_missing_marker).to_numpy(dtype=bool)
    for bin_index, group in enumerate(groups):
        for category in group:
            if is_missing_marker(category):
                category_mask = missing_mask
            else:
                category_mask = _typed_mask(x, category).to_numpy(dtype=bool)
            bins[category_mask] = bin_index
            user_owned |= category_mask

    special_owned = np.zeros(len(x), dtype=bool)
    if special_codes:
        for special in special_codes:
            special_mask = _typed_mask(x, special).to_numpy(dtype=bool) & ~user_owned
            bins[special_mask] = SPECIAL_BIN
            special_owned |= special_mask

    if missing_separate:
        unresolved_missing = missing_mask & ~user_owned & ~special_owned
        bins[unresolved_missing] = MISSING_BIN
    if unknown_policy == "raise":
        unresolved_unknown = (bins == UNKNOWN_BIN) & ~missing_mask & ~special_owned
        if unresolved_unknown.any():
            unknown_values = unique_non_missing_typed(x.loc[unresolved_unknown])
            raise ValueError(f"特征 '{feature}' 在 transform 中出现训练期未知类别: {unknown_values}")
    return bins


def normalize_user_groups(
    feature: str,
    groups: Sequence[Sequence[Any]],
    observed: Optional[pd.Series] = None,
    special_codes: Optional[Sequence[Any]] = None,
    missing_separate: bool = True,
) -> List[List[Any]]:
    """校验并规范化用户提供的类别分箱规则。"""
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence) or len(groups) == 0:
        raise ValueError(f"特征 '{feature}' 的自定义类别分箱必须是非空 List[List]")

    normalized: List[List[Any]] = []
    ordinary_values: List[Any] = []
    missing_seen = False
    for group_index, group in enumerate(groups):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence) or len(group) == 0:
            raise ValueError(f"特征 '{feature}' 的第 {group_index} 个自定义箱不能为空，且必须是列表")
        normalized_group: List[Any] = []
        for value in list(group):
            if is_missing_marker(value):
                if missing_seen:
                    raise ValueError(f"特征 '{feature}' 的缺失值标记只能出现一次")
                missing_seen = True
                normalized_group.append(np.nan)
                continue
            value = _normalize_scalar(value)
            if _contains_typed(ordinary_values, value):
                raise ValueError(f"特征 '{feature}' 的类别 {value!r} 出现在多个自定义箱中")
            ordinary_values.append(value)
            normalized_group.append(value)
        normalized.append(normalized_group)

    if observed is not None:
        observed_values = unique_non_missing_typed(observed, special_codes)
        uncovered = [value for value in observed_values if not _contains_typed(ordinary_values, value)]
        unknown = [value for value in ordinary_values if not _contains_typed(observed_values, value)]
        if uncovered or unknown:
            raise ValueError(f"特征 '{feature}' 的自定义分箱必须完整覆盖训练类别；" f"未覆盖类别: {uncovered}，规则外类别: {unknown}")
    return normalized
