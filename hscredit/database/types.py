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
    :type value: Any
    :return: ``dataframe`` 、 ``records`` 或 ``rows`` 。
    :raises ValidationError: 输入不是受支持的字符串。
    """

    if not isinstance(value, str) or value not in RESULT_TYPES:
        raise ValidationError(f"result 只支持 {sorted(RESULT_TYPES)}，收到 {value!r}")
    return value


class StreamState(str, Enum):
    """流式查询生命周期状态。

    ``running`` 表示仍可读取； ``completed`` 表示自然耗尽； ``interrupted`` 表示主动停止
    或键盘中断； ``failed`` 表示读取失败； ``closed`` 表示在耗尽前关闭。
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

    参数与 DBUtils ``PooledDB`` 含义一致，各字段的零值语义见下方说明。

    :param mincached: 池初始化时预建的空闲连接数，0 表示不预建。 默认 ``0``。
    :param maxcached: 最多缓存的空闲连接数，0 表示不限制。 默认 ``0``。
    :param maxshared: 最多共享连接数，0 表示不共享。 默认 ``0``。
    :param maxconnections: 连接池允许的最大连接总数，0 表示不限制。 默认 ``0``。
    :param blocking: 达到连接上限时是否阻塞等待可用连接。 默认 ``False``。
    :param maxusage: 单连接最大复用次数；None 使用 DBUtils 默认行为。
    :param setsession: 新连接建立后执行的 SQL 语句序列；None 不额外设置会话。 默认 ``None``。
    :param ping: DBUtils ping 位掩码：0 不检查，1 获取连接时，2 创建游标时，4 执行时，7 每次检查。 默认 ``1``。
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

        :param value: 默认值为 ``None`` 。 ``PoolOptions`` 、参数映射或 ``None``
        :type value: Optional[Mapping[str, Any]]
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
    """Redis 原生连接池配置。

    :param max_connections: 原生池连接数上限，正整数；None 沿用 redis-py 默认值。
    :param blocking: 是否使用阻塞式连接池。 默认 ``False``。
    :param timeout: 等待连接的超时秒数，仅 blocking=True 可用；None 沿用原生池默认等待策略。
    """

    max_connections: Optional[int] = None
    blocking: bool = False
    timeout: Optional[float] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "RedisPoolOptions":
        """从映射创建 Redis 连接池配置。

        :param value: 同类型配置实例、配置映射或 None；未知字段及非法范围会抛出 ValidationError。 默认 ``None``。
        :return: 校验后的不可变连接池配置；None 创建默认配置。
        """

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
    """PyMongo ``MongoClient`` 原生连接池配置。

    :param min_pool_size: 每个服务端连接池最小连接数，非负整数。 默认 ``0``。
    :param max_pool_size: 每个服务端连接池最大连接数，0 表示不限制。 默认 ``100``。
    :param max_connecting: 池内可同时建立的连接数，必须为正整数。 默认 ``2``。
    :param wait_queue_timeout_ms: 等待可用连接的超时毫秒数；None 沿用驱动默认值。
    :param max_idle_time_ms: 空闲连接最大存活毫秒数；None 沿用驱动默认值。
    """

    min_pool_size: int = 0
    max_pool_size: int = 100
    max_connecting: int = 2
    wait_queue_timeout_ms: Optional[int] = None
    max_idle_time_ms: Optional[int] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]] = None) -> "MongoPoolOptions":
        """从映射创建 MongoDB 连接池配置。

        :param value: 同类型配置实例、配置映射或 None；未知字段及非法范围会抛出 ValidationError。 默认 ``None``。
        :return: 校验后的不可变连接池配置；None 创建默认配置。
        """

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
            if current is not None and (isinstance(current, bool) or not isinstance(current, int) or current < 0):
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

    :param transactions: 是否支持事务提交和回滚。
    :type transactions: bool
    :param streaming_read: 是否支持流式读取。
    :type streaming_read: bool
    :param native_bulk_write: 是否具有后端原生批量写入通道。
    :type native_bulk_write: bool
    :param metadata_export: 是否支持表结构扫描。
    :type metadata_export: bool
    :param write_modes: 可保证的 ``a/r/o/d`` 写入模式集合。
    :type write_modes: frozenset[str]
    :param atomic_replace: 是否提供显式暂存替换协议；True仍须运行时验证版本、表结构及可见性前提。
    :type atomic_replace: bool

    """

    transactions: bool = True
    streaming_read: bool = True
    native_bulk_write: bool = False
    metadata_export: bool = True
    write_modes: FrozenSet[str] = field(default_factory=lambda: frozenset({"o", "d"}))
    atomic_replace: bool = False

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

    :param mode: 本次写入模式。
    :type mode: {"a", "r", "o", "d"}
    :param completed: 是否完成全部批次和适配器收尾。
    :type completed: bool
    :param rows_received: 已接收的输入行数，默认 0；包含已提交和尚未成功提交的输入。
    :type rows_received: int, optional
    :param rows_inserted: 后端报告的新增行数；默认 None 表示后端不能提供可靠统计。
    :type rows_inserted: int, optional
    :param rows_updated: 后端报告的更新行数；默认 None 表示后端不能提供可靠统计。
    :type rows_updated: int, optional
    :param rows_skipped: 后端报告的跳过行数；默认 None 表示后端不能提供可靠统计。
    :type rows_skipped: int, optional
    :param batches_committed: 已成功提交的批次数。
    :type batches_committed: int
    :param failed_batch: 失败批次编号；准备阶段为 0，收尾阶段为已提交批次数加 1。
    :type failed_batch: int, optional
    :param consistency: 后端最终一致性说明。
    :type consistency: str, optional
    :param details: 适配器附加的原始统计信息。
    :type details: dict
    :param rows_staged: 原子替换入口确认写入暂存表的行数，默认 0，不表示目标已经更新。
    :type rows_staged: int
    :param batches_staged: 原子替换入口确认写入暂存表的批次数，默认 0，不表示目标已经更新。
    :type batches_staged: int
    :param target_published: 原子替换入口的未发布/已确认发布/未知状态；普通stream_write不使用此字段。
    :type target_published: bool, optional
    :param rows_published: 确认发布至目标的行数；发布结果未知时为None。
    :type rows_published: int, optional
    :param phase: preparing/staging/publishing/published/publication_unknown等操作阶段。
    :type phase: str, optional

    :param staging_table: 暂存表名称；未使用暂存协议时为 None。 默认 ``None``。
    :param backup_table: 旧目标的备份表名称；未建立备份时为 None。 默认 ``None``。
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
    staging_table: Optional[str] = None
    backup_table: Optional[str] = None
    rows_staged: int = 0
    batches_staged: int = 0
    target_published: Optional[bool] = None
    rows_published: Optional[int] = None
    phase: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in WRITE_MODES:
            raise ValidationError(f"mode 只支持 {sorted(WRITE_MODES)}，收到 {self.mode!r}")
        for name in ("rows_received", "batches_committed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} 必须是非负整数")


@dataclass(frozen=True)
class NoSQLWriteResult:
    """Redis 与 MongoDB 共用的写入、更新和删除结果。

    :param operation: 操作类型，``'write'`` 或 ``'delete'``。
    :param acknowledged: 后端是否确认执行结果。
    :param affected_count: 受影响的记录或 key 数；None 表示后端未提供。 默认 ``None``。
    :param matched_count: 更新条件命中的文档数；None 表示不提供。 默认 ``None``。
    :param modified_count: 实际修改的文档数；None 表示不提供。 默认 ``None``。
    :param identifiers: 写入生成或返回的记录标识元组，默认空元组。
    :param details: 后端附加统计信息，默认空映射。
    """

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
