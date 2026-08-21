"""数据库模块公共类型。

定义连接池配置、数据库能力、流式查询状态和写入结果等稳定契约。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from ..exceptions import ValidationError

WRITE_MODES = frozenset({"a", "r", "o", "d"})
#: ``Database.query``、``stream_query`` 和 ``read_query`` 共用的结果类型。
RESULT_TYPES = frozenset({"dataframe", "records", "rows"})


def validate_result_type(value: Any) -> str:
    """校验并返回 Database 模块统一的查询结果类型。

    :param value: 期望的结果类型。
    :return: ``dataframe``、``records`` 或 ``rows``。
    :raises ValidationError: 输入不是受支持的字符串。
    """

    if not isinstance(value, str) or value not in RESULT_TYPES:
        raise ValidationError(f"result 只支持 {sorted(RESULT_TYPES)}，收到 {value!r}")
    return value


class StreamState(str, Enum):
    """流式查询生命周期状态。

    ``running`` 表示仍可读取；``completed`` 表示自然耗尽；``interrupted`` 表示主动停止
    或键盘中断；``failed`` 表示读取失败；``closed`` 表示在耗尽前关闭。
    """

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
        """从映射创建并校验连接池配置。

        :param value: ``PoolOptions``、参数映射或 ``None``。
        :return: 已校验的不可变连接池配置。
        :raises ValidationError: 包含未知参数或参数范围无效。
        """

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
        """转换为 DBUtils ``PooledDB`` 关键字参数。

        :return: 可直接传给 ``PooledDB`` 的新字典。
        :rtype: dict
        :raises ValidationError: 当前配置组合无效。
        """

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
class RedisPoolOptions:
    """Redis 原生连接池配置。"""

    max_connections: Optional[int] = None
    blocking: bool = False
    timeout: Optional[float] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "RedisPoolOptions":
        """从映射创建 Redis 连接池配置。"""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("Redis pool_options 必须是映射或 RedisPoolOptions")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValidationError(f"不支持的 Redis 连接池参数: {unknown}")
        options = cls(**dict(value))
        options._validate()
        return options

    def _validate(self) -> None:
        if self.max_connections is not None:
            if (
                isinstance(self.max_connections, bool)
                or not isinstance(self.max_connections, int)
                or self.max_connections <= 0
            ):
                raise ValidationError("Redis 连接池参数 max_connections 必须是正整数或 None")
        if not isinstance(self.blocking, bool):
            raise ValidationError("Redis 连接池参数 blocking 必须是布尔值")
        if self.timeout is not None:
            if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or self.timeout < 0:
                raise ValidationError("Redis 连接池参数 timeout 必须是非负数或 None")
            if not self.blocking:
                raise ValidationError("Redis 连接池参数 timeout 仅在 blocking=True 时可用")

    def to_redis_kwargs(self) -> Dict[str, Any]:
        """转换为 redis-py 连接池参数。"""

        self._validate()
        result: Dict[str, Any] = {}
        if self.max_connections is not None:
            result["max_connections"] = self.max_connections
        if self.blocking and self.timeout is not None:
            result["timeout"] = float(self.timeout)
        return result


@dataclass(frozen=True)
class MongoPoolOptions:
    """PyMongo ``MongoClient`` 原生连接池配置。"""

    min_pool_size: int = 0
    max_pool_size: int = 100
    max_connecting: int = 2
    wait_queue_timeout_ms: Optional[int] = None
    max_idle_time_ms: Optional[int] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "MongoPoolOptions":
        """从映射创建 MongoDB 连接池配置。"""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("MongoDB pool_options 必须是映射或 MongoPoolOptions")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValidationError(f"不支持的 MongoDB 连接池参数: {unknown}")
        options = cls(**dict(value))
        options._validate()
        return options

    def _validate(self) -> None:
        for name in ("min_pool_size", "max_pool_size", "max_connecting"):
            current = getattr(self, name)
            if isinstance(current, bool) or not isinstance(current, int) or current < 0:
                raise ValidationError(f"MongoDB 连接池参数 {name} 必须是非负整数")
        if self.max_pool_size and self.min_pool_size > self.max_pool_size:
            raise ValidationError("MongoDB 连接池参数 min_pool_size 不能大于 max_pool_size")
        if self.max_connecting == 0:
            raise ValidationError("MongoDB 连接池参数 max_connecting 必须是正整数")
        for name in ("wait_queue_timeout_ms", "max_idle_time_ms"):
            current = getattr(self, name)
            if current is not None and (
                isinstance(current, bool) or not isinstance(current, int) or current < 0
            ):
                raise ValidationError(f"MongoDB 连接池参数 {name} 必须是非负整数或 None")

    def to_mongo_kwargs(self) -> Dict[str, Any]:
        """转换为 ``MongoClient`` 驼峰连接池参数。"""

        self._validate()
        result: Dict[str, Any] = {
            "minPoolSize": self.min_pool_size,
            "maxPoolSize": self.max_pool_size,
            "maxConnecting": self.max_connecting,
        }
        if self.wait_queue_timeout_ms is not None:
            result["waitQueueTimeoutMS"] = self.wait_queue_timeout_ms
        if self.max_idle_time_ms is not None:
            result["maxIdleTimeMS"] = self.max_idle_time_ms
        return result


@dataclass(frozen=True)
class DatabaseCapabilities:
    """数据库或目标表可保证的能力。

    **参数**

    transactions : bool
        是否支持事务提交和回滚。
    streaming_read : bool
        是否支持流式读取。
    native_bulk_write : bool
        是否具有后端原生批量写入通道。
    metadata_export : bool
        是否支持表结构扫描。
    write_modes : frozenset[str]
        可保证的 ``a/r/o/d`` 写入模式集合。
    """

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
    """流式写入结果。

    **参数**

    mode : {"a", "r", "o", "d"}
        本次写入模式。
    completed : bool
        是否完成全部批次和适配器收尾。
    rows_received、rows_inserted、rows_updated、rows_skipped : int, optional
        输入及后端报告的行数统计。
    batches_committed : int
        已成功提交的批次数。
    failed_batch : int, optional
        失败批次编号；准备阶段为 0，收尾阶段为已提交批次数加 1。
    consistency : str, optional
        后端最终一致性说明。
    details : dict
        适配器附加的原始统计信息。
    """

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


@dataclass(frozen=True)
class NoSQLWriteResult:
    """Redis 与 MongoDB 共用的写入、更新和删除结果。"""

    operation: str
    acknowledged: bool
    affected_count: Optional[int] = None
    matched_count: Optional[int] = None
    modified_count: Optional[int] = None
    identifiers: Tuple[Any, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in {"write", "delete"}:
            raise ValidationError("NoSQL operation 只支持 'write' 或 'delete'")
        if not isinstance(self.acknowledged, bool):
            raise ValidationError("NoSQL acknowledged 必须是布尔值")
        for name in ("affected_count", "matched_count", "modified_count"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValidationError(f"NoSQL {name} 必须是非负整数或 None")
        object.__setattr__(self, "identifiers", tuple(self.identifiers))
        object.__setattr__(self, "details", dict(self.details))


__all__ = [
    "WRITE_MODES",
    "RESULT_TYPES",
    "validate_result_type",
    "StreamState",
    "PoolOptions",
    "RedisPoolOptions",
    "MongoPoolOptions",
    "DatabaseCapabilities",
    "WriteResult",
    "NoSQLWriteResult",
]
