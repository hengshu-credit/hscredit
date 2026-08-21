"""数据库公共门面。"""

from pathlib import Path
from itertools import chain
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..exceptions import InputValidationError, StateError, ValidationError
from .exceptions import (
    DatabaseCapabilityError,
    DatabaseConnectionError,
    DatabaseMetadataError,
    DatabaseWriteError,
)
from .metadata import MetadataInspection, metadata_frame, parse_targets
from .registry import canonical_adapter_name, get_adapter_class
from .stream import QueryStream
from .types import PoolOptions, WRITE_MODES, WriteResult
from .writing import BatchWriteResult, iter_write_batches, split_qualified_name


class Database:
    """统一数据库连接与操作门面。

    **参数**

    database_type : str
        已注册数据库类型或别名。
    pool_options : mapping or PoolOptions, optional
        连接池配置。
    adapter_options : mapping, optional
        后端专有配置。
    ``**connect_kwargs``
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

    def execute(self, sql: str, params: Any = None) -> Any:
        """执行单条 SQL。"""

        self._ensure_open()
        return self.adapter.execute(sql, params=params)

    def executemany(self, sql: str, values: Any) -> Any:
        """批量执行 SQL。"""

        self._ensure_open()
        return self.adapter.executemany(sql, values)

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

    def export_schema(
        self,
        targets: Optional[Sequence[str]] = None,
        *,
        output: Optional[Any] = None,
        excel_params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """导出数据库表和字段元数据宽表。"""

        self._ensure_open()
        parsed_targets = parse_targets(targets)

        output_path = None
        if output is not None:
            output_path = Path(output)
            if output_path.suffix.lower() != ".xlsx":
                raise ValidationError("数据库表结构导出仅支持 .xlsx 文件")
        if excel_params is not None and not isinstance(excel_params, Mapping):
            raise ValidationError("excel_params 必须是映射或 None")

        try:
            inspection = self.adapter.inspect_schema(parsed_targets)
        except DatabaseMetadataError:
            raise
        except Exception as exc:
            raise DatabaseMetadataError("读取数据库表结构失败") from exc
        if not isinstance(inspection, MetadataInspection):
            raise DatabaseMetadataError("数据库适配器 inspect_schema() 必须返回 MetadataInspection")

        inspection = MetadataInspection(
            rows=list(inspection.rows),
            errors=list(inspection.errors),
        )
        if parsed_targets:
            missing_exact = []
            for target in parsed_targets:
                if len(target.parts) < 2:
                    continue
                expected = tuple(part.casefold() for part in target.parts)
                matched = False
                for row in inspection.rows:
                    candidate = tuple(
                        str(value).casefold()
                        for value in (
                            row.get("catalog"),
                            row.get("database"),
                            row.get("schema"),
                            row.get("table_name"),
                        )
                        if value not in (None, "")
                    )
                    if len(candidate) >= len(expected) and candidate[-len(expected) :] == expected:
                        matched = True
                        break
                if not matched:
                    missing_exact.append(target.raw)
            if missing_exact:
                raise DatabaseMetadataError(f"未找到精确指定的数据库表或无访问权限: {missing_exact}")

        frame = metadata_frame(inspection)
        if output_path is not None:
            from ..excel import dataframe2excel

            params = {
                "sheet_name": "表结构",
                "title": "数据库表结构",
                "index": False,
                "decimal": None,
                "auto_filter": True,
                "auto_width": True,
            }
            params.update(dict(excel_params or {}))
            try:
                dataframe2excel(frame, output_path, **params)
            except Exception as exc:
                raise DatabaseMetadataError(f"数据库表结构 Excel 导出失败: {output_path.name}") from exc
        return frame

    @staticmethod
    def _validate_dialect_options(
        dialect_options: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if dialect_options is None:
            return {}
        if not isinstance(dialect_options, Mapping):
            raise ValidationError("dialect_options 必须是映射或 None")
        return dict(dialect_options)

    @staticmethod
    def _validate_key_columns(
        key_columns: Optional[Sequence[str]],
        data_columns: Sequence[Any],
    ) -> Optional[Sequence[str]]:
        if key_columns is None:
            return None
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        normalized = tuple(key_columns)
        if not normalized or any(not isinstance(column, str) or not column for column in normalized):
            raise ValidationError("key_columns 必须是非空字段名序列")
        missing = [column for column in normalized if column not in data_columns]
        if missing:
            raise InputValidationError(f"写入数据缺少主键字段: {missing}")
        return normalized

    def create_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """根据 DataFrame 和方言参数创建数据库表。"""

        self._ensure_open()
        split_qualified_name(table_name)
        if not isinstance(data, pd.DataFrame) or len(data.columns) == 0:
            raise InputValidationError("create_table 的 data 必须是包含字段的 DataFrame")
        options = self._validate_dialect_options(dialect_options)
        return self.adapter.create_table(
            data,
            table_name,
            dialect_options=options,
        )

    @staticmethod
    def _add_optional_count(current: Optional[int], value: Optional[int]) -> Optional[int]:
        if current is None or value is None:
            return None
        return current + value

    def stream_write(
        self,
        data: Any,
        table_name: str,
        *,
        mode: str = "a",
        batch_size: int = 10_000,
        key_columns: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> WriteResult:
        """按批写入 DataFrame 或记录迭代器。"""

        self._ensure_open()
        if mode not in WRITE_MODES:
            raise ValidationError(f"mode 只支持 {sorted(WRITE_MODES)}，收到 {mode!r}")
        split_qualified_name(table_name)
        options = self._validate_dialect_options(dialect_options)
        batches = iter_write_batches(
            data,
            batch_size=batch_size,
            columns=columns,
        )
        try:
            first_batch = next(batches)
        except StopIteration as exc:
            raise InputValidationError("写入数据没有可用的有效数据行") from exc
        if len(first_batch) == 0 or len(first_batch.columns) == 0:
            raise InputValidationError("写入数据没有可用的有效数据行")

        resolved_keys = self._validate_key_columns(
            key_columns,
            first_batch.columns,
        )
        if mode in {"a", "r"}:
            resolved_keys = self.adapter.resolve_key_columns(
                table_name,
                resolved_keys,
                first_batch,
                dialect_options=options,
            )
            resolved_keys = self._validate_key_columns(
                resolved_keys,
                first_batch.columns,
            )
        result = WriteResult(
            mode=mode,
            completed=False,
            rows_inserted=0,
            rows_updated=0,
            rows_skipped=0,
        )
        try:
            self.adapter.prepare_write(
                table_name,
                mode,
                first_batch,
                key_columns=resolved_keys,
                dialect_options=options,
            )
        except (DatabaseCapabilityError, ValidationError, InputValidationError):
            raise
        except Exception as exc:
            result.failed_batch = 0
            raise DatabaseWriteError(
                f"写入目标表 {table_name!r} 的准备操作失败",
                result=result,
            ) from exc

        for batch_index, batch in enumerate(chain((first_batch,), batches), start=1):
            result.rows_received += len(batch)
            try:
                batch_result = self.adapter.write_batch(
                    table_name,
                    batch,
                    mode,
                    batch_index,
                    key_columns=resolved_keys,
                    dialect_options=options,
                )
            except Exception as exc:
                result.failed_batch = batch_index
                raise DatabaseWriteError(
                    f"写入目标表 {table_name!r} 的第 {batch_index} 批数据失败",
                    result=result,
                ) from exc
            if batch_result is None:
                batch_result = BatchWriteResult()
            if not isinstance(batch_result, BatchWriteResult):
                raise DatabaseWriteError(
                    "数据库适配器 write_batch() 必须返回 BatchWriteResult",
                    result=result,
                )
            result.rows_inserted = self._add_optional_count(
                result.rows_inserted,
                batch_result.inserted,
            )
            result.rows_updated = self._add_optional_count(
                result.rows_updated,
                batch_result.updated,
            )
            result.rows_skipped = self._add_optional_count(
                result.rows_skipped,
                batch_result.skipped,
            )
            result.batches_committed += 1

        try:
            self.adapter.finish_write(
                table_name,
                mode,
                result,
                dialect_options=options,
            )
        except Exception as exc:
            result.failed_batch = result.batches_committed + 1
            raise DatabaseWriteError(
                f"完成目标表 {table_name!r} 的写入收尾失败",
                result=result,
            ) from exc
        result.completed = True
        return result

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
