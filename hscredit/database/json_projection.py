"""JSON 字段投影参数规范。

把简洁的 ``源 JSON 字段 -> 输出字段 -> 路径或(路径, 默认值)`` 映射转换为
数据库适配器和流式结果处理共用的稳定结构。
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..exceptions import ValidationError


@dataclass(frozen=True)
class JsonProjectionField:
    """单个 JSON 路径投影字段。"""

    source_column: str
    output_column: str
    path: str
    default: Any = None


@dataclass(frozen=True)
class JsonProjection:
    """已经校验并保持字段顺序的 JSON 投影。"""

    columns: Tuple[str, ...]
    fields: Tuple[JsonProjectionField, ...]

    @property
    def defaults(self) -> Dict[str, Any]:
        """返回输出字段与缺失默认值的映射。"""

        return {field.output_column: field.default for field in self.fields}


def _validate_name(value: Any, *, option_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{option_name} 必须是非空字符串")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValidationError(f"{option_name} 包含不安全字符")
    return value


def _validate_json_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("$"):
        raise ValidationError("JSONPath 必须是以 $ 开头的非空字符串")
    unsafe_parts = ("'", "\\", ";", "\x00", "\r", "\n", "--", "/*", "*/")
    if any(part in value for part in unsafe_parts):
        raise ValidationError("JSONPath 包含不安全字符")
    return value


def normalize_json_projection(
    columns: Optional[Sequence[str]],
    json_fields: Optional[Mapping[str, Mapping[str, Any]]],
) -> Optional[JsonProjection]:
    """校验普通字段和 JSON 字段简写，并保留用户定义顺序。"""

    if columns is not None and isinstance(columns, (str, bytes)):
        raise ValidationError("columns 必须是字段名序列或 None")
    if json_fields is None:
        if columns is not None:
            raise ValidationError("columns 仅能与 json_fields 同时使用")
        return None
    if not isinstance(json_fields, Mapping) or not json_fields:
        raise ValidationError("json_fields 必须是非空映射")

    normalized_columns = tuple(_validate_name(column, option_name="columns 字段名") for column in (columns or ()))
    source_columns = tuple(_validate_name(source, option_name="JSON源字段名") for source in json_fields)
    normalized_sources = {source.casefold() for source in source_columns}
    leaked_sources = [column for column in normalized_columns if column.casefold() in normalized_sources]
    if leaked_sources:
        raise ValidationError(f"columns 不能包含 JSON源字段，否则会返回完整 JSON: {leaked_sources}")
    seen = set()
    for column in normalized_columns:
        if column in seen:
            raise ValidationError(f"输出字段名重复: {column!r}")
        seen.add(column)

    normalized_fields = []
    for source, raw_fields in zip(source_columns, json_fields.values()):
        if not isinstance(raw_fields, Mapping) or not raw_fields:
            raise ValidationError(f"json_fields[{source!r}] 必须是非空映射")
        for raw_output, raw_specification in raw_fields.items():
            output = _validate_name(raw_output, option_name="JSON输出字段名")
            if output in seen:
                raise ValidationError(f"输出字段名重复: {output!r}")
            seen.add(output)

            if isinstance(raw_specification, str):
                path = raw_specification
                default = None
            elif isinstance(raw_specification, tuple) and len(raw_specification) == 2:
                path, default = raw_specification
            else:
                raise ValidationError(f"JSON字段定义 {output!r} 必须是路径字符串或 (路径, 默认值) 二元组")
            normalized_fields.append(
                JsonProjectionField(
                    source_column=source,
                    output_column=output,
                    path=_validate_json_path(path),
                    default=default,
                )
            )

    return JsonProjection(
        columns=normalized_columns,
        fields=tuple(normalized_fields),
    )


__all__ = [
    "JsonProjectionField",
    "JsonProjection",
    "normalize_json_projection",
]
