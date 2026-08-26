"""数据库与 NoSQL 的类外快捷操作。

快捷函数接受以下任一 ``source``，并把实际操作委托给 :class:`Database` 的同名方法：

- 含 ``db_type`` 的连接配置映射；
- 已创建的 :class:`Database` 实例；
- PyMySQL、python-oracledb、Impyla、PyODPS 等原生 DB-API 连接。

配置创建的连接池在操作结束后自动关闭；传入的 Database 和原生连接均视为借用，不由
快捷函数关闭。配置创建的流式查询在查询流完成、停止或关闭时释放连接池。

**参考样例**

>>> from hscredit.database import read_query
>>> config = {"db_type": "mysql", "host": "127.0.0.1", "user": "risk", "password": "***"}
>>> frame = read_query(config, "SELECT id FROM events")
"""

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..exceptions import ValidationError
from .adapters.base import BaseDatabaseAdapter
from .adapters.dbapi import DBAPIAdapter
from .client import Database
from .exceptions import DatabaseCapabilityError
from .registry import canonical_adapter_name, get_adapter_class
from .types import PoolOptions


@dataclass
class _ResolvedSource:
    database: Database
    owned: bool

    def close(self) -> None:
        if self.owned:
            self.database.close()


class _BorrowedConnection:
    """屏蔽外部 DB-API 连接的 close，其余操作透明转发。"""

    def __init__(self, connection: Any):
        self._connection = connection

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class _BorrowedConnectionPool:
    """把一个外部连接适配为 DBAPIAdapter 所需的池接口。"""

    def __init__(self, connection: Any):
        self._connection = connection

    def connection(self) -> _BorrowedConnection:
        return _BorrowedConnection(self._connection)

    def close(self) -> None:
        return None


_CONNECTION_MODULE_TYPES = (
    ("pymysql", "mysql"),
    ("oracledb", "oracle"),
    ("cx_oracle", "oracle"),
    ("impala", "impala"),
    ("odps", "maxcompute"),
)


def _infer_db_type(connection: Any) -> Optional[str]:
    module_name = type(connection).__module__.casefold()
    for marker, db_type in _CONNECTION_MODULE_TYPES:
        if module_name == marker or module_name.startswith(marker + "."):
            return db_type
    return None


def _connection_kwargs(connection: Any, db_type: str) -> Mapping[str, Any]:
    values = {}
    if db_type in {"mysql", "starrocks"}:
        for target, candidates in {
            "host": ("host",),
            "user": ("user",),
            "database": ("db", "database"),
        }.items():
            for candidate in candidates:
                value = getattr(connection, candidate, None)
                if value is not None:
                    values[target] = value.decode() if isinstance(value, bytes) else value
                    break
    elif db_type == "oracle":
        user = getattr(connection, "username", None)
        if user is not None:
            values["user"] = user
    return values


def _driver_module(connection: Any) -> Any:
    root_module = type(connection).__module__.split(".", 1)[0]
    try:
        return importlib.import_module(root_module)
    except ImportError:
        return None


def _database_from_connection(connection: Any, db_type: Optional[str]) -> Database:
    resolved_type = db_type or _infer_db_type(connection)
    if not isinstance(resolved_type, str) or not resolved_type.strip():
        raise ValidationError("无法识别原生 DB-API 连接类型，请显式传入 db_type")
    canonical = canonical_adapter_name(resolved_type)
    adapter_class = get_adapter_class(canonical)
    if not issubclass(adapter_class, DBAPIAdapter):
        raise DatabaseCapabilityError(f"数据库 {canonical} 的原生客户端不是 DB-API 连接")

    pool_options_class = getattr(adapter_class, "pool_options_class", PoolOptions)
    pool_options = pool_options_class.from_mapping(None)
    adapter = adapter_class.__new__(adapter_class)
    BaseDatabaseAdapter.__init__(
        adapter,
        connect_kwargs=_connection_kwargs(connection, canonical),
        pool_options=pool_options,
        adapter_options={},
    )
    adapter.driver = _driver_module(connection)
    adapter._pool = _BorrowedConnectionPool(connection)

    database = Database.__new__(Database)
    database.database_type = canonical
    database.pool_options = pool_options
    database.adapter_options = {}
    database.adapter = adapter
    database._closed = False
    database._shortcut_native_connection = True
    return database


def _resolve_config(source: Mapping[str, Any], db_type: Optional[str]) -> _ResolvedSource:
    config = dict(source)
    configured_type = config.pop("db_type", None)
    if (
        configured_type is not None
        and db_type is not None
        and canonical_adapter_name(configured_type) != canonical_adapter_name(db_type)
    ):
        raise ValidationError(f"配置 db_type={configured_type!r} 与参数 db_type={db_type!r} 冲突")
    resolved_type = db_type or configured_type
    if not isinstance(resolved_type, str) or not resolved_type.strip():
        raise ValidationError("数据库配置必须包含非空 db_type，或显式传入 db_type")

    pool_options = config.pop("pool_options", None)
    adapter_options = config.pop("adapter_options", None)
    database = Database(
        resolved_type,
        pool_options=pool_options,
        adapter_options=adapter_options,
        **config,
    )
    return _ResolvedSource(database=database, owned=True)


def _resolve_source(source: Any, db_type: Optional[str] = None) -> _ResolvedSource:
    if isinstance(source, Database):
        if db_type is not None and source.database_type != canonical_adapter_name(db_type):
            raise ValidationError(f"Database 实例类型 {source.database_type!r} 与参数 db_type={db_type!r} 冲突")
        return _ResolvedSource(database=source, owned=False)
    if isinstance(source, Mapping):
        return _resolve_config(source, db_type)
    if callable(getattr(source, "cursor", None)):
        return _ResolvedSource(
            database=_database_from_connection(source, db_type),
            owned=True,
        )
    raise ValidationError("source 必须是数据库配置、Database 实例或原生 DB-API 连接")


def _invoke(source: Any, method_name: str, *args: Any, db_type: Optional[str] = None, **kwargs: Any) -> Any:
    resolved = _resolve_source(source, db_type)
    try:
        if (
            getattr(resolved.database, "_shortcut_native_connection", False)
            and resolved.database.database_type == "maxcompute"
            and method_name in {"export_schema", "stream_write", "write"}
        ):
            raise DatabaseCapabilityError(
                "MaxCompute 原生 DB-API 连接不包含 ODPS 元数据和 Tunnel 入口，" "请改为传入连接配置或 Database 实例"
            )
        method = getattr(resolved.database, method_name)
        return method(*args, **kwargs)
    finally:
        resolved.close()


def query(
    source: Any,
    sql: str,
    params: Any = None,
    result: str = "dataframe",
    *,
    db_type: Optional[str] = None,
) -> Any:
    """快捷执行一次性 SQL 查询。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param sql: 查询 SQL。
    :param params: 驱动绑定参数。
    :param result: ``dataframe``、``records`` 或 ``rows``。
    :param db_type: 原生连接无法自动识别时使用的数据库类型。
    :return: 对应格式的完整查询结果。
    """

    return _invoke(source, "query", sql, params=params, result=result, db_type=db_type)


def execute(
    source: Any,
    sql: str,
    params: Any = None,
    *,
    db_type: Optional[str] = None,
) -> Any:
    """快捷执行单条 DDL 或 DML SQL。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param sql: 要执行的 SQL。
    :param params: 驱动绑定参数。
    :param db_type: 可选数据库类型。
    :return: 适配器报告的影响行数或原生结果。
    """

    return _invoke(source, "execute", sql, params=params, db_type=db_type)


def executemany(
    source: Any,
    sql: str,
    values: Any,
    *,
    db_type: Optional[str] = None,
) -> Any:
    """快捷批量执行同一条 SQL。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param sql: 带驱动占位符的 SQL。
    :param values: 多组绑定值。
    :param db_type: 可选数据库类型。
    :return: 累计影响行数或原生结果。
    """

    return _invoke(source, "executemany", sql, values, db_type=db_type)


def stream_query(
    source: Any,
    sql: str,
    params: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷打开流式 SQL 查询；拥有的数据源随查询流关闭。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param sql: 查询 SQL。
    :param params: 驱动绑定参数。
    :param db_type: 可选数据库类型。
    :param options: 传递给 :meth:`Database.stream_query` 的分块、进度和 JSON 投影参数。
    :return: 可中断并可合并已读数据的 QueryStream。
    """

    resolved = _resolve_source(source, db_type)
    try:
        stream = resolved.database.stream_query(sql, params=params, **options)
    except Exception:
        resolved.close()
        raise
    if resolved.owned:
        stream._add_close_callback(resolved.close)
    return stream


def read_query(
    source: Any,
    sql: str,
    params: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷消费流式 SQL 查询并返回合并结果。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param sql: 查询 SQL。
    :param params: 驱动绑定参数。
    :param db_type: 可选数据库类型。
    :param options: 传递给 :meth:`Database.read_query` 的选项。
    :return: DataFrame、记录字典列表或原始行列表。
    """

    return _invoke(source, "read_query", sql, params=params, db_type=db_type, **options)


def export_schema(
    source: Any,
    targets: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷导出数据库表和字段元数据。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param targets: 数据库或 ``数据库.表`` 目标。
    :param db_type: 可选数据库类型。
    :param options: 输出路径和 ``excel_params``。
    :return: 中文字段元数据 DataFrame。
    """

    return _invoke(source, "export_schema", targets, db_type=db_type, **options)


def create_table(
    source: Any,
    data: Any,
    table_name: str,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷根据 DataFrame 创建表。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param data: 用于推断表结构的 DataFrame。
    :param table_name: 目标表限定名。
    :param db_type: 可选数据库类型。
    :param options: 传递给 :meth:`Database.create_table` 的方言参数。
    :return: 已执行 DDL 或适配器原生结果。
    """

    return _invoke(source, "create_table", data, table_name, db_type=db_type, **options)


def stream_write(
    source: Any,
    data: Any,
    table_name: str,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷流式写入 DataFrame 或记录迭代器。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param data: DataFrame、分块或记录迭代器。
    :param table_name: 目标表限定名。
    :param db_type: 可选数据库类型。
    :param options: 写入模式、批次、键字段和方言参数。
    :return: WriteResult 写入统计。
    """

    return _invoke(source, "stream_write", data, table_name, db_type=db_type, **options)


def read_one(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷读取单个 Redis key 或 MongoDB 文档。

    :param source: 数据库配置或 Database 实例。
    :param resource: Redis key 或 MongoDB collection。
    :param selector: MongoDB 查询条件。
    :param db_type: 可选数据库类型。
    :param options: 后端读取选项。
    :return: 单值或单个文档。
    """

    return _invoke(source, "read_one", resource, selector, db_type=db_type, **options)


def read_many(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷批量读取 Redis keys 或 MongoDB 文档。

    参数与 :func:`read_one` 一致，返回后端批量读取结果。
    """

    return _invoke(source, "read_many", resource, selector, db_type=db_type, **options)


def read(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷自适应执行单条或批量 NoSQL 读取。

    参数与 :func:`read_one` 一致，具体单条/批量语义由适配器根据输入和选项确定。
    """

    return _invoke(source, "read", resource, selector, db_type=db_type, **options)


def write_one(
    source: Any,
    resource: Any,
    data: Any,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷写入单个 Redis key 或 MongoDB 文档。

    :param source: 数据库配置或 Database 实例。
    :param resource: Redis key 或 MongoDB collection。
    :param data: 要写入的值或文档。
    :param db_type: 可选数据库类型。
    :param options: 后端写入选项。
    :return: 后端写入结果。
    """

    return _invoke(source, "write_one", resource, data, db_type=db_type, **options)


def write_many(
    source: Any,
    resource: Any,
    data: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷批量写入 Redis key-value 或 MongoDB 文档。

    参数与 :func:`write_one` 一致，``data`` 为批量输入。
    """

    return _invoke(source, "write_many", resource, data, db_type=db_type, **options)


def write(
    source: Any,
    resource: Any,
    data: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷写入 SQL 表，或自适应执行单条/批量 NoSQL 写入。

    SQL 用法为 ``write(source, 表名, 数据, ...)``，默认 ``mode="a"``；目标表不存在时
    根据首批数据自动创建。NoSQL 参数与 :func:`write_one` 一致，具体单条/批量语义由适配器确定。

    :param source: 数据库配置、Database 实例或原生 DB-API 连接。
    :param resource: SQL 目标表限定名、Redis key/映射或 MongoDB collection。
    :param data: SQL DataFrame/记录迭代器、Redis 值或 MongoDB 文档。
    :param db_type: 可选数据库类型。
    :param options: SQL 写入模式、批次、键字段和方言参数，或 NoSQL 后端选项。
    :return: SQL WriteResult 或 NoSQLWriteResult。
    """

    return _invoke(source, "write", resource, data, db_type=db_type, **options)


def delete_one(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷删除单个 Redis key 或 MongoDB 文档。

    :param source: 数据库配置或 Database 实例。
    :param resource: Redis key 或 MongoDB collection。
    :param selector: MongoDB 查询条件。
    :param db_type: 可选数据库类型。
    :param options: 后端删除选项。
    :return: 后端删除结果。
    """

    return _invoke(source, "delete_one", resource, selector, db_type=db_type, **options)


def delete_many(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷批量删除 Redis keys 或 MongoDB 文档。

    参数与 :func:`delete_one` 一致，返回后端批量删除结果。
    """

    return _invoke(source, "delete_many", resource, selector, db_type=db_type, **options)


def delete(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> Any:
    """快捷自适应执行单条或批量 NoSQL 删除。

    参数与 :func:`delete_one` 一致，具体单条/批量语义由适配器确定。
    """

    return _invoke(source, "delete", resource, selector, db_type=db_type, **options)


def exists(
    source: Any,
    resource: Any,
    selector: Any = None,
    *,
    db_type: Optional[str] = None,
    **options: Any,
) -> bool:
    """快捷判断 Redis key 或 MongoDB 文档是否存在。

    :param source: 数据库配置或 Database 实例。
    :param resource: Redis key 或 MongoDB collection。
    :param selector: MongoDB 查询条件。
    :param db_type: 可选数据库类型。
    :param options: 后端查询选项。
    :return: 是否存在。
    :rtype: bool
    """

    return bool(_invoke(source, "exists", resource, selector, db_type=db_type, **options))


__all__ = [
    "query",
    "execute",
    "executemany",
    "stream_query",
    "read_query",
    "export_schema",
    "create_table",
    "stream_write",
    "read_one",
    "read_many",
    "read",
    "write_one",
    "write_many",
    "write",
    "delete_one",
    "delete_many",
    "delete",
    "exists",
]
