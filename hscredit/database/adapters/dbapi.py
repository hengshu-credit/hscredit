"""DB-API 2.0 数据库共享实现。

提供 DBUtils 连接池、资源上下文、查询结果转换以及事务提交和回滚。
"""

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ...exceptions import DependencyError, ValidationError
from ..exceptions import DatabaseConnectionError, DatabaseQueryError
from ..types import DatabaseCapabilities, PoolOptions
from .base import BaseDatabaseAdapter

RESULT_TYPES = frozenset({"dataframe", "records", "rows"})


class DBAPIQueryResource:
    """持有一组用于流式查询的连接和游标。"""

    def __init__(self, connection: Any, cursor: Any):
        self.connection = connection
        self.cursor = cursor
        self.columns = [column[0] for column in (cursor.description or ())]
        self.closed = False

    def fetchmany(self, size: int) -> Sequence[Any]:
        """读取下一批原始记录。"""

        return self.cursor.fetchmany(size)

    def close(self) -> None:
        """先关闭游标，再归还连接。"""

        if self.closed:
            return
        try:
            self.cursor.close()
        finally:
            self.connection.close()
            self.closed = True


class DBAPIAdapter(BaseDatabaseAdapter):
    """基于 DBUtils 和 DB-API 2.0 的数据库适配器。"""

    capabilities = DatabaseCapabilities(
        transactions=True,
        streaming_read=True,
        native_bulk_write=False,
        metadata_export=True,
        write_modes={"o", "d"},
    )

    def __init__(
        self,
        *,
        connect_kwargs: Mapping[str, Any],
        pool_options: PoolOptions,
        adapter_options: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.driver = self.load_driver()
        self._pool = self._create_pool()

    def load_driver(self) -> Any:
        """加载底层 DB-API 驱动；子类必须实现。"""

        raise NotImplementedError

    def load_pool_class(self) -> Any:
        """按需加载 DBUtils ``PooledDB``。"""

        try:
            from dbutils.pooled_db import PooledDB
        except ImportError as exc:  # pragma: no cover - 由隔离导入测试覆盖用户行为
            raise DependencyError("缺少数据库连接池可选依赖 DBUtils，请安装对应 hscredit[db-*] 扩展") from exc
        return PooledDB

    def _create_pool(self) -> Any:
        pool_class = self.load_pool_class()
        kwargs = self.pool_options.to_dbutils_kwargs()
        kwargs.update(self.connect_kwargs)
        try:
            return pool_class(self.driver, **kwargs)
        except Exception as exc:
            raise DatabaseConnectionError(f"创建 {self.database_type} 数据库连接池失败") from exc

    def create_cursor(self, connection: Any, *, stream: bool = False) -> Any:
        """创建普通或流式游标。"""

        del stream
        return connection.cursor()

    @contextmanager
    def connection_cursor(self, *, stream: bool = False) -> Iterator[Tuple[Any, Any]]:
        """租借连接和游标，并确保按正确顺序释放。"""

        self.ensure_open()
        connection = None
        cursor = None
        try:
            connection = self._pool.connection()
            cursor = self.create_cursor(connection, stream=stream)
            yield connection, cursor
        except (DatabaseConnectionError, DatabaseQueryError):
            raise
        except Exception as exc:
            raise DatabaseConnectionError(f"获取 {self.database_type} 数据库连接或游标失败") from exc
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                finally:
                    if connection is not None:
                        connection.close()
            elif connection is not None:
                connection.close()

    @staticmethod
    def execute_cursor(cursor: Any, sql: str, params: Any = None) -> Any:
        """按 DB-API 约定执行带或不带参数的 SQL。"""

        if params is None:
            return cursor.execute(sql)
        return cursor.execute(sql, params)

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        """执行查询并转换为 DataFrame、记录字典或原始行。"""

        if result not in RESULT_TYPES:
            raise ValidationError(f"result 只支持 {sorted(RESULT_TYPES)}，收到 {result!r}")
        try:
            with self.connection_cursor() as (_, cursor):
                self.execute_cursor(cursor, sql, params)
                columns = [column[0] for column in (cursor.description or ())]
                rows = list(cursor.fetchall())
        except DatabaseConnectionError as exc:
            raise DatabaseQueryError("SQL查询失败") from exc
        except Exception as exc:
            raise DatabaseQueryError("SQL查询失败") from exc

        if result == "rows":
            return rows
        if result == "records":
            return [dict(zip(columns, row)) for row in rows]
        return pd.DataFrame.from_records(rows, columns=columns)

    @staticmethod
    def _rollback_quietly(connection: Any) -> None:
        try:
            connection.rollback()
        except Exception:
            return

    def execute(self, sql: str, params: Any = None) -> int:
        """执行单条 SQL，并在成功时提交、失败时回滚。"""

        try:
            with self.connection_cursor() as (connection, cursor):
                try:
                    self.execute_cursor(cursor, sql, params)
                    connection.commit()
                    return int(cursor.rowcount)
                except Exception as exc:
                    self._rollback_quietly(connection)
                    raise DatabaseQueryError("SQL执行失败") from exc
        except DatabaseQueryError:
            raise
        except DatabaseConnectionError as exc:
            raise DatabaseQueryError("SQL执行失败") from exc

    def executemany(self, sql: str, values: Iterable[Any]) -> int:
        """批量执行 SQL，并在成功时提交、失败时回滚。"""

        materialized: List[Any] = list(values)
        try:
            with self.connection_cursor() as (connection, cursor):
                try:
                    cursor.executemany(sql, materialized)
                    connection.commit()
                    return int(cursor.rowcount)
                except Exception as exc:
                    self._rollback_quietly(connection)
                    raise DatabaseQueryError("SQL批量执行失败") from exc
        except DatabaseQueryError:
            raise
        except DatabaseConnectionError as exc:
            raise DatabaseQueryError("SQL批量执行失败") from exc

    def open_stream(self, sql: str, params: Any = None) -> DBAPIQueryResource:
        """打开一个由调用方负责关闭的流式查询资源。"""

        self.ensure_open()
        connection = None
        cursor = None
        try:
            connection = self._pool.connection()
            cursor = self.create_cursor(connection, stream=True)
            self.execute_cursor(cursor, sql, params)
            return DBAPIQueryResource(connection, cursor)
        except Exception as exc:
            if cursor is not None:
                try:
                    cursor.close()
                finally:
                    if connection is not None:
                        connection.close()
            elif connection is not None:
                connection.close()
            raise DatabaseQueryError("打开流式SQL查询失败") from exc

    def count_rows(self, sql: str, params: Any = None) -> int:
        """执行统计 SQL 并返回第一列。"""

        rows = self.query(sql, params=params, result="rows")
        if not rows:
            return 0
        return int(rows[0][0])

    def close(self) -> None:
        """关闭连接池。"""

        if self.closed:
            return
        try:
            self._pool.close()
        finally:
            super().close()


__all__ = ["DBAPIAdapter", "DBAPIQueryResource", "RESULT_TYPES"]
