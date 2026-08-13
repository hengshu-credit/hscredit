"""分箱器公共参数与用户规则的严格契约。"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


MISSING_BIN = -1
SPECIAL_BIN = -2
UNKNOWN_BIN = -3

FixedFeatureValue = Union[bool, Sequence[bool]]
UserSplitsFixed = Optional[Union[bool, Mapping[str, FixedFeatureValue]]]
HandleUnknown = Union[int, Literal["value", "raise"]]


def is_missing_marker(value: Any) -> bool:
    """判断标量是否为统一缺失标记。"""
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def validate_handle_unknown(value: Any) -> HandleUnknown:
    """校验未知类别策略，并将整数箱号统一为 Python 整数。"""
    if isinstance(value, str):
        if value == "value":
            return UNKNOWN_BIN
        if value == "raise":
            return "raise"
        raise ValueError("handle_unknown 必须是整数箱号、字符串 'value' 或字符串 'raise'")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("handle_unknown 必须是整数箱号、字符串 'value' 或字符串 'raise'")
    return int(value)


def parse_numerical_user_splits(feature: str, values: Sequence[Any]) -> Tuple[np.ndarray, Optional[int]]:
    """解析严格递增的数值切分点及缺失值目标箱位置。"""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"特征 '{feature}' 的数值切分点必须是可迭代数值序列")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"特征 '{feature}' 的数值切分点必须是可迭代数值序列") from exc

    missing_positions = [index for index, value in enumerate(raw) if is_missing_marker(value)]
    if len(missing_positions) > 1:
        raise ValueError(f"特征 '{feature}' 的数值 user_splits 最多包含一个缺失标记")

    try:
        splits = np.asarray([value for value in raw if not is_missing_marker(value)], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"特征 '{feature}' 的数值切分点必须是严格递增的有限数值") from exc

    if not np.isfinite(splits).all() or (len(splits) > 1 and np.any(np.diff(splits) <= 0)):
        raise ValueError(f"特征 '{feature}' 的数值切分点必须是严格递增的有限数值")

    missing_bin = missing_positions[0] if missing_positions else None
    return splits, missing_bin


def _is_categorical_rule(values: Sequence[Any]) -> bool:
    """判断字段规则是否为类别型 List[List]。"""
    return bool(values) and all(not isinstance(group, (str, bytes)) and isinstance(group, Sequence) for group in values)


def _effective_rule_length(feature: str, values: Sequence[Any]) -> int:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"特征 '{feature}' 的 user_splits 必须是规则序列")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"特征 '{feature}' 的 user_splits 必须是规则序列") from exc
    if _is_categorical_rule(raw):
        return len(raw)
    return sum(not is_missing_marker(value) for value in raw)


def _expand_fixed_value(feature: str, value: FixedFeatureValue, expected_length: int) -> List[bool]:
    if isinstance(value, (bool, np.bool_)):
        return [bool(value)] * expected_length
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"特征 '{feature}' 的 user_splits_fixed 必须是布尔值或布尔序列")
    resolved = list(value)
    if len(resolved) != expected_length:
        raise ValueError(f"特征 '{feature}' 的 user_splits_fixed 长度必须等于有效用户节点数 " f"{expected_length}，实际为 {len(resolved)}")
    if not all(isinstance(item, (bool, np.bool_)) for item in resolved):
        raise ValueError(f"特征 '{feature}' 的 user_splits_fixed 只能包含布尔值")
    return [bool(item) for item in resolved]


def resolve_user_splits_fixed(
    user_splits: Mapping[str, Sequence[Any]],
    user_splits_fixed: UserSplitsFixed,
) -> Dict[str, List[bool]]:
    """将全局、字段级和节点级固定配置规范化为逐字段布尔掩码。"""
    if not isinstance(user_splits, Mapping):
        raise ValueError("user_splits 必须是字段规则字典")
    lengths = {feature: _effective_rule_length(feature, values) for feature, values in user_splits.items()}

    if user_splits_fixed is None or isinstance(user_splits_fixed, (bool, np.bool_)):
        default = bool(user_splits_fixed) if user_splits_fixed is not None else False
        return {feature: [default] * length for feature, length in lengths.items()}

    if not isinstance(user_splits_fixed, Mapping):
        raise ValueError("user_splits_fixed 必须是布尔值、字段配置字典或 None")
    unknown_features = sorted(set(user_splits_fixed) - set(user_splits))
    if unknown_features:
        raise ValueError(f"user_splits_fixed 包含未在 user_splits 中定义的字段: {unknown_features}")

    return {
        feature: _expand_fixed_value(
            feature,
            user_splits_fixed.get(feature, False),
            length,
        )
        for feature, length in lengths.items()
    }
