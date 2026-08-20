"""数据库流式写入输入和批次结果。

规范化 DataFrame、DataFrame 分块、映射记录和位置记录，不预执行或重试用户迭代器。
"""

from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..exceptions import InputValidationError, ValidationError


@dataclass(frozen=True)
class BatchWriteResult:
    """单个已提交写入批次的统计。"""

    inserted: Optional[int] = None
    updated: Optional[int] = None
    skipped: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("inserted", "updated", "skipped"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValidationError(f"批次统计 {name} 必须是非负整数或 None")


def validate_sql_type(value: Any, *, database_type: str) -> str:
    """校验调用方覆盖的数据类型表达式，禁止注释、引号和语句分隔符。"""

    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{database_type} 数据类型必须是非空字符串")
    expression = value.strip()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_<>,() ")
    if expression[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" or any(
        character not in allowed for character in expression
    ):
        raise ValidationError(f"{database_type} 数据类型表达式不安全: {value!r}")

    pairs = {"(": ")", "<": ">"}
    closing = {value: key for key, value in pairs.items()}
    stack = []
    for character in expression:
        if character in pairs:
            stack.append(character)
        elif character in closing:
            if not stack or stack.pop() != closing[character]:
                raise ValidationError(f"{database_type} 数据类型括号不匹配: {value!r}")
    if stack:
        raise ValidationError(f"{database_type} 数据类型括号不匹配: {value!r}")
    return expression


def validate_column_mapping_keys(
    mapping: Mapping[Any, Any],
    columns: Sequence[Any],
    *,
    option_name: str,
    database_type: str,
) -> None:
    """拒绝引用输入数据中不存在字段的建表映射。"""

    unknown = sorted(set(mapping) - set(columns), key=str)
    if unknown:
        raise InputValidationError(f"{database_type} {option_name} 引用未知字段: {unknown}")


def validate_batch_size(batch_size: int) -> None:
    """校验写入批次大小。"""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("batch_size 必须是正整数")


def split_qualified_name(value: str) -> Tuple[str, ...]:
    """校验并拆分数据库对象限定名。"""

    if not isinstance(value, str) or not value.strip():
        raise ValidationError("数据库表名必须是非空字符串")
    parts = tuple(part.strip() for part in value.strip().split("."))
    if any(not part for part in parts):
        raise ValidationError(f"数据库表名格式无效: {value!r}")
    return parts


def _yield_frame_slices(
    frame: pd.DataFrame,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size].copy()


def _iter_dataframe_chunks(
    first: pd.DataFrame,
    remaining: Iterator[Any],
    batch_size: int,
    columns: Optional[Sequence[str]],
) -> Iterator[pd.DataFrame]:
    expected = tuple(columns) if columns is not None else None
    for chunk in chain((first,), remaining):
        if not isinstance(chunk, pd.DataFrame):
            raise InputValidationError("DataFrame 分块迭代器不能混入其他记录类型")
        current = tuple(chunk.columns)
        if expected is None:
            expected = current
        if current != expected:
            raise InputValidationError(f"DataFrame 分块字段不一致，期望 {list(expected)}，收到 {list(current)}")
        if len(chunk) == 0:
            continue
        yield from _yield_frame_slices(chunk, batch_size)


def _iter_mapping_rows(
    first: Mapping[str, Any],
    remaining: Iterator[Any],
    batch_size: int,
    columns: Optional[Sequence[str]],
) -> Iterator[pd.DataFrame]:
    expected = tuple(columns) if columns is not None else tuple(first.keys())
    if not expected:
        raise InputValidationError("映射记录必须至少包含一个字段")
    buffer: List[Mapping[str, Any]] = []
    for row in chain((first,), remaining):
        if not isinstance(row, Mapping):
            raise InputValidationError("映射记录迭代器不能混入其他记录类型")
        if set(row) != set(expected):
            raise InputValidationError(f"记录字段不一致，期望 {list(expected)}，收到 {list(row)}")
        buffer.append(row)
        if len(buffer) >= batch_size:
            yield pd.DataFrame.from_records(buffer, columns=expected)
            buffer = []
    if buffer:
        yield pd.DataFrame.from_records(buffer, columns=expected)


def _iter_positional_rows(
    first: Any,
    remaining: Iterator[Any],
    batch_size: int,
    columns: Optional[Sequence[str]],
) -> Iterator[pd.DataFrame]:
    if columns is None:
        raise InputValidationError("位置行记录迭代器必须通过 columns 指定字段名")
    expected = tuple(columns)
    if not expected or any(not isinstance(column, str) or not column for column in expected):
        raise InputValidationError("columns 必须是非空字段名序列")

    buffer: List[Any] = []
    for row in chain((first,), remaining):
        if isinstance(row, (str, bytes, bytearray, Mapping, pd.DataFrame)):
            raise InputValidationError("位置行记录必须是与 columns 等长的序列")
        try:
            values = tuple(row)
        except TypeError as exc:
            raise InputValidationError("位置行记录必须是可迭代序列") from exc
        if len(values) != len(expected):
            raise InputValidationError(f"位置行记录长度应为 {len(expected)}，收到 {len(values)}")
        buffer.append(values)
        if len(buffer) >= batch_size:
            yield pd.DataFrame.from_records(buffer, columns=expected)
            buffer = []
    if buffer:
        yield pd.DataFrame.from_records(buffer, columns=expected)


def iter_write_batches(
    data: Any,
    *,
    batch_size: int = 10_000,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """将支持的写入输入规范化为非空 DataFrame 批次。"""

    validate_batch_size(batch_size)
    if isinstance(data, pd.DataFrame):
        expected = tuple(columns) if columns is not None else tuple(data.columns)
        if tuple(data.columns) != expected:
            raise InputValidationError(f"DataFrame 字段与 columns 不一致: {list(data.columns)} != {list(expected)}")
        yield from _yield_frame_slices(data, batch_size)
        return

    try:
        iterator = iter(data)
    except TypeError as exc:
        raise InputValidationError("data 必须是 DataFrame、DataFrame 分块或行记录迭代器") from exc
    try:
        first = next(iterator)
    except StopIteration:
        return

    if isinstance(first, pd.DataFrame):
        yield from _iter_dataframe_chunks(first, iterator, batch_size, columns)
    elif isinstance(first, Mapping):
        yield from _iter_mapping_rows(first, iterator, batch_size, columns)
    else:
        yield from _iter_positional_rows(first, iterator, batch_size, columns)


__all__ = [
    "BatchWriteResult",
    "iter_write_batches",
    "split_qualified_name",
    "validate_batch_size",
    "validate_column_mapping_keys",
    "validate_sql_type",
]
