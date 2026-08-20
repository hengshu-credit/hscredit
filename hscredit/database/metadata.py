"""数据库元数据宽表。

解析数据库/表目标，将适配器内部字段映射为稳定的中文列名，并保留数据库原始值。
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..exceptions import ValidationError

METADATA_COLUMN_MAP = (
    ("database_type", "数据库类型"),
    ("catalog", "目录"),
    ("database", "数据库名"),
    ("schema", "模式名"),
    ("table_name", "表名"),
    ("qualified_name", "完整表名"),
    ("table_type", "表类型"),
    ("table_comment", "表注释"),
    ("table_engine", "表引擎"),
    ("column_name", "字段名"),
    ("ordinal_position", "字段序号"),
    ("data_type", "数据类型"),
    ("full_data_type", "完整数据类型"),
    ("pandas_dtype", "Pandas类型"),
    ("nullable", "是否可空"),
    ("default_value", "默认值"),
    ("primary_key", "是否主键"),
    ("unique_key", "是否唯一键"),
    ("partition_key", "是否分区键"),
    ("sort_key", "是否排序键"),
    ("bucket_key", "是否分桶键"),
    ("column_comment", "字段注释"),
)

METADATA_COLUMNS_ZH = [chinese for _, chinese in METADATA_COLUMN_MAP]


@dataclass(frozen=True)
class QualifiedTarget:
    """未改写大小写的数据库对象限定名。"""

    raw: str
    parts: Tuple[str, ...]

    @classmethod
    def parse(cls, value: str) -> "QualifiedTarget":
        """解析以点分隔的数据库、模式和表目标。"""

        if not isinstance(value, str) or not value.strip():
            raise ValidationError("元数据目标必须是非空字符串")
        raw = value.strip()
        parts = tuple(part.strip() for part in raw.split("."))
        if any(not part for part in parts):
            raise ValidationError(f"元数据目标格式无效: {value!r}")
        return cls(raw=raw, parts=parts)


@dataclass
class MetadataInspection:
    """适配器元数据扫描结果。"""

    rows: Iterable[Mapping[str, Any]] = field(default_factory=tuple)
    errors: List[Any] = field(default_factory=list)


def parse_targets(
    targets: Optional[Sequence[str]],
) -> Optional[Tuple[QualifiedTarget, ...]]:
    """将目标参数规范化为限定名元组。"""

    if targets is None:
        return None
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, Sequence):
        raise ValidationError("元数据 targets 必须是字符串序列或 None")
    parsed = tuple(QualifiedTarget.parse(target) for target in targets)
    if not parsed:
        raise ValidationError("元数据目标列表不能为空")
    return parsed


def metadata_frame(inspection: MetadataInspection) -> pd.DataFrame:
    """将内部元数据映射为中文列名宽表。"""

    records = []
    for row in inspection.rows:
        records.append({chinese: row.get(internal) for internal, chinese in METADATA_COLUMN_MAP})
    frame = pd.DataFrame.from_records(records, columns=METADATA_COLUMNS_ZH)
    frame.attrs["错误"] = list(inspection.errors)
    return frame


__all__ = [
    "METADATA_COLUMN_MAP",
    "METADATA_COLUMNS_ZH",
    "QualifiedTarget",
    "MetadataInspection",
    "parse_targets",
    "metadata_frame",
]
