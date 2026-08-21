"""数据库字符串字段内容画像。

只分析用于建表的当前 DataFrame，不读取后续流式分块，也不修改原始值。
"""

import json
import math
from dataclasses import dataclass
import pandas as pd

_STRING_LENGTH_BUCKETS = (
    16,
    32,
    64,
    128,
    255,
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_767,
    65_533,
)


@dataclass(frozen=True)
class StringColumnProfile:
    """字符串列在当前建表样本中的内容画像。"""

    non_null_count: int
    all_strings: bool
    max_characters: int
    max_utf8_bytes: int
    all_json_documents: bool


def _is_json_document(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, (dict, list))


def profile_string_series(series: pd.Series) -> StringColumnProfile:
    """统计非空字符串长度，并严格识别整列 JSON 对象/数组。"""

    values = []
    for value in series:
        if value is None or value is pd.NA:
            continue
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                continue
        except (TypeError, ValueError):
            pass
        values.append(value)

    if not values:
        return StringColumnProfile(
            non_null_count=0,
            all_strings=True,
            max_characters=0,
            max_utf8_bytes=0,
            all_json_documents=False,
        )

    all_strings = all(isinstance(value, str) for value in values)
    string_values = [value for value in values if isinstance(value, str)]
    return StringColumnProfile(
        non_null_count=len(values),
        all_strings=all_strings,
        max_characters=max((len(value) for value in string_values), default=0),
        max_utf8_bytes=max(
            (len(value.encode("utf-8")) for value in string_values),
            default=0,
        ),
        all_json_documents=all_strings
        and bool(string_values)
        and all(_is_json_document(value) for value in string_values),
    )


def resolve_bounded_string_length(
    observed_length: int,
    *,
    maximum: int,
    headroom: float = 1.2,
) -> int:
    """按稳定档位为观察长度增加余量，同时不超过后端上限。"""

    if observed_length <= 0:
        return min(255, maximum)
    target = max(1, math.ceil(observed_length * headroom))
    for bucket in _STRING_LENGTH_BUCKETS:
        if bucket >= target and bucket <= maximum:
            return bucket
    return maximum


__all__ = [
    "StringColumnProfile",
    "profile_string_series",
    "resolve_bounded_string_length",
]
