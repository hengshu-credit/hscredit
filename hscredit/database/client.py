"""数据库公共门面。"""

from typing import Any, Mapping, Optional

from ..exceptions import StateError, ValidationError
from .exceptions import DatabaseConnectionError
from .registry import canonical_adapter_name, get_adapter_class
from .stream import QueryStream
from .types import PoolOptions


class Database:
    """统一数据库连接与操作门面。

    **参数**

    database_type : str
        已注册数据库类型或别名。
    pool_options : mapping or PoolOptions, optional
        连接池配置。
    adapter_options : mapping, optional
        后端专有配置。
    **connect_kwargs
        直接传递给后端驱动的连接参数。

    **属性**

    adapter
        当前数据库适配器实例。

    **参考样例**

    >>> with Database("mysql", host="127.0.0.1", database="risk") as db:
    ...     rows = db.query("SELECT 1", result="rows")
    """

    def __init__(
        self,
        database_type: str,
        *,
        pool_options: Optional[Mapping[str, Any]] = None,
        adapter_options: Optional[Mapping[str, Any]] = None,
        **connect_kwargs: Any,
    ):
        self.database_type = canonical_adapter_name(database_type)
        self.pool_options = PoolOptions.from_mapping(pool_options)
        self.adapter_options = dict(adapter_options or {})
        self._closed = False

        adapter_class = get_adapter_class(self.database_type)
        try:
            self.adapter = adapter_class(
                connect_kwargs=dict(connect_kwargs),
                pool_options=self.pool_options,
                adapter_options=self.adapter_options,
            )
        except Exception as exc:
            raise DatabaseConnectionError(
                f"初始化 {self.database_type} 数据库适配器失败，请检查连接参数和可选依赖"
            ) from exc

    @property
    def closed(self) -> bool:
        """数据库门面是否已经关闭。"""

        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateError("数据库连接已经关闭")

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        """查询 SQL 数据。"""

        self._ensure_open()
        return self.adapter.query(sql, params=params, result=result)

    @staticmethod
    def _validate_stream_options(
        chunksize: int,
        progress: bool,
        retain: bool,
        total_rows: Optional[int],
    ) -> None:
        if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
            raise ValidationError("chunksize 必须是正整数")
        if not isinstance(progress, bool):
            raise ValidationError("progress 必须是布尔值")
        if not isinstance(retain, bool):
            raise ValidationError("retain 必须是布尔值")
        if total_rows is not None:
            if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 0:
                raise ValidationError("total_rows 必须是非负整数或 None")

    def stream_query(
        self,
        sql: str,
        params: Any = None,
        *,
        chunksize: int = 50_000,
        progress: bool = False,
        retain: bool = True,
        count_sql: Optional[str] = None,
        total_rows: Optional[int] = None,
    ) -> QueryStream:
        """打开按 DataFrame 分块返回的流式查询。"""

        self._ensure_open()
        self._validate_stream_options(chunksize, progress, retain, total_rows)

        resolved_total = total_rows
        if progress and resolved_total is None:
            resolved_count_sql = count_sql or self.adapter.build_count_sql(sql)
            resolved_total = self.adapter.count_rows(resolved_count_sql, params=params)

        resource = self.adapter.open_stream(sql, params=params)
        return QueryStream(
            resource,
            chunksize=chunksize,
            retain=retain,
            total_rows=resolved_total,
            progress=progress,
        )

    def read_query(
        self,
        sql: str,
        params: Any = None,
        *,
        chunksize: int = 50_000,
        progress: bool = False,
        count_sql: Optional[str] = None,
        total_rows: Optional[int] = None,
    ) -> Any:
        """消费流式查询，并在主动中断后返回已读取 DataFrame。"""

        stream = self.stream_query(
            sql,
            params=params,
            chunksize=chunksize,
            progress=progress,
            retain=True,
            count_sql=count_sql,
            total_rows=total_rows,
        )
        try:
            for _ in stream:
                pass
        except KeyboardInterrupt:
            stream.stop("KeyboardInterrupt")
        finally:
            if stream.state.value == "running":
                stream.close()
        return stream.to_dataframe()

    def close(self) -> None:
        """关闭适配器资源；重复调用不产生副作用。"""

        if self._closed:
            return
        try:
            self.adapter.close()
        finally:
            self._closed = True

    def __enter__(self) -> "Database":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Database(database_type={self.database_type!r}, closed={self.closed})"


__all__ = ["Database"]
