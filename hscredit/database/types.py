"""数据库模块公共类型。

定义连接池配置、数据库能力、流式查询状态和写入结果等稳定契约。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from ..exceptions import ValidationError

WRITE_MODES = frozenset({"a", "r", "o", "d"})


class StreamState(str, Enum):
    """流式查询状态。"""

    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class PoolOptions:
    """DBUtils 兼容的连接池配置。

    **参数**

    mincached、maxcached、maxshared、maxconnections 与 DBUtils ``PooledDB``
    含义一致，值为 0 时沿用 DBUtils 的“不限制”语义。
    """

    mincached: int = 0
    maxcached: int = 0
    maxshared: int = 0
    maxconnections: int = 0
    blocking: bool = False
    maxusage: Optional[int] = None
    setsession: Optional[Tuple[str, ...]] = None
    ping: int = 1

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "PoolOptions":
        """从映射创建并校验连接池配置。"""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("pool_options 必须是映射或 PoolOptions")

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(f"不支持的连接池参数: {unknown}")

        options = cls(**dict(value))
        options._validate()
        return options

    def _validate(self) -> None:
        integer_fields = (
            "mincached",
            "maxcached",
            "maxshared",
            "maxconnections",
            "ping",
        )
        for name in integer_fields:
            current = getattr(self, name)
            if isinstance(current, bool) or not isinstance(current, int) or current < 0:
                raise ValidationError(f"连接池参数 {name} 必须是非负整数")

        if self.maxusage is not None:
            if isinstance(self.maxusage, bool) or not isinstance(self.maxusage, int) or self.maxusage <= 0:
                raise ValidationError("连接池参数 maxusage 必须是正整数或 None")
        if not isinstance(self.blocking, bool):
            raise ValidationError("连接池参数 blocking 必须是布尔值")
        if self.maxcached and self.mincached > self.maxcached:
            raise ValidationError("连接池参数 mincached 不能大于 maxcached")
        if self.maxconnections and self.maxcached > self.maxconnections:
            raise ValidationError("连接池参数 maxcached 不能大于 maxconnections")
        if self.maxconnections and self.maxshared > self.maxconnections:
            raise ValidationError("连接池参数 maxshared 不能大于 maxconnections")
        if self.setsession is not None and not all(isinstance(sql, str) for sql in self.setsession):
            raise ValidationError("连接池参数 setsession 必须是 SQL 字符串序列")

    def to_dbutils_kwargs(self) -> Dict[str, Any]:
        """转换为 ``PooledDB`` 关键字参数。"""

        self._validate()
        result: Dict[str, Any] = {
            "mincached": self.mincached,
            "maxcached": self.maxcached,
            "maxshared": self.maxshared,
            "maxconnections": self.maxconnections,
            "blocking": self.blocking,
            "maxusage": self.maxusage,
            "ping": self.ping,
        }
        if self.setsession is not None:
            result["setsession"] = list(self.setsession)
        return result


@dataclass(frozen=True)
class DatabaseCapabilities:
    """数据库或目标表可保证的能力。"""

    transactions: bool = True
    streaming_read: bool = True
    native_bulk_write: bool = False
    metadata_export: bool = True
    write_modes: FrozenSet[str] = field(default_factory=lambda: frozenset({"o", "d"}))

    def __post_init__(self) -> None:
        normalized = frozenset(self.write_modes)
        invalid = sorted(normalized - WRITE_MODES)
        if invalid:
            raise ValidationError(f"不支持的写入模式: {invalid}")
        object.__setattr__(self, "write_modes", normalized)


@dataclass
class WriteResult:
    """流式写入结果。"""

    mode: str
    completed: bool
    rows_received: int = 0
    rows_inserted: Optional[int] = None
    rows_updated: Optional[int] = None
    rows_skipped: Optional[int] = None
    batches_committed: int = 0
    failed_batch: Optional[int] = None
    consistency: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in WRITE_MODES:
            raise ValidationError(f"mode 只支持 {sorted(WRITE_MODES)}，收到 {self.mode!r}")
        for name in ("rows_received", "batches_committed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} 必须是非负整数")


__all__ = [
    "WRITE_MODES",
    "StreamState",
    "PoolOptions",
    "DatabaseCapabilities",
    "WriteResult",
]
