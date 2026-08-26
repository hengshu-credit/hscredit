"""ClickHouse 数据库适配器。

使用官方 clickhouse-connect 客户端的 DataFrame 流、参数化查询和原生批量插入。
"""

import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api import types as ptypes

from ...exceptions import DependencyError, InputValidationError, ValidationError
from ..exceptions import (
    DatabaseCapabilityError,
    DatabaseQueryError,
    DatabaseWriteError,
)
from ..metadata import MetadataInspection, QualifiedTarget
from ..type_inference import profile_string_series
from ..types import DatabaseCapabilities, PoolOptions, WriteResult, validate_result_type
from ..writing import BatchWriteResult, resolve_column_type, validate_column_mapping_keys
from .base import BaseDatabaseAdapter

_SAFE_ENGINE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ClickHouseQueryResource:
    """把 clickhouse-connect DataFrame 流适配为 QueryStream 资源。"""

    columns: Sequence[str] = ()

    def __init__(self, stream_context: Any):
        self.stream_context = stream_context
        self.closed = False
        if hasattr(stream_context, "__enter__"):
            self._entered = stream_context.__enter__()
        else:
            self._entered = stream_context
        self._iterator = iter(self._entered)

    def fetchmany(self, size: int) -> pd.DataFrame:
        del size
        try:
            return next(self._iterator)
        except StopIteration:
            return pd.DataFrame()

    def close(self) -> None:
        if self.closed:
            return
        try:
            if hasattr(self.stream_context, "__exit__"):
                self.stream_context.__exit__(None, None, None)
            elif hasattr(self.stream_context, "close"):
                self.stream_context.close()
        finally:
            self.closed = True


class ClickHouseAdapter(BaseDatabaseAdapter):
    """ClickHouse 官方 Python 客户端适配器。"""

    database_type = "clickhouse"
    identifier_quote = "`"
    capabilities = DatabaseCapabilities(
        transactions=False,
        streaming_read=True,
        native_bulk_write=True,
        metadata_export=True,
        write_modes={"a", "o", "d"},
    )

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """使用 ClickHouse SQL/JSON 函数兼容标量和对象/数组。"""

        exists = f"JSON_EXISTS({column_sql}, '{path}')"
        nested = f"JSON_QUERY({column_sql}, '{path}')"
        scalar = f"JSON_VALUE({column_sql}, '{path}')"
        return f"if({exists}, nullIf(coalesce(nullIf({nested}, ''), {scalar}), 'null'), NULL)"

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
        module = self.load_client_module()
        kwargs = dict(connect_kwargs)
        if "user" in kwargs and "username" not in kwargs:
            kwargs["username"] = kwargs.pop("user")
        try:
            self.client = module.get_client(**kwargs)
        except Exception as exc:
            from ..exceptions import DatabaseConnectionError

            raise DatabaseConnectionError("创建 ClickHouse 客户端失败") from exc

    def load_client_module(self) -> Any:
        try:
            import clickhouse_connect
        except ImportError as exc:
            raise DependencyError("缺少 ClickHouse 可选依赖，请安装: pip install hscredit[db-clickhouse]") from exc
        return clickhouse_connect

    def query(self, sql: str, params: Any = None, result: str = "dataframe") -> Any:
        validate_result_type(result)
        settings = self.adapter_options.get("query_settings")
        try:
            if result == "dataframe":
                return self.client.query_df(
                    sql,
                    parameters=params,
                    settings=settings,
                )
            query_result = self.client.query(
                sql,
                parameters=params,
                settings=settings,
            )
            rows = list(query_result.result_rows)
            if result == "rows":
                return rows
            columns = list(query_result.column_names)
            return [dict(zip(columns, row)) for row in rows]
        except Exception as exc:
            raise DatabaseQueryError("ClickHouse SQL查询失败") from exc

    def execute(self, sql: str, params: Any = None) -> Any:
        try:
            return self.client.command(sql, parameters=params)
        except Exception as exc:
            raise DatabaseQueryError("ClickHouse SQL执行失败") from exc

    def executemany(self, sql: str, values: Any) -> int:
        affected = 0
        for params in values:
            self.execute(sql, params=params)
            affected += 1
        return affected

    def open_stream(self, sql: str, params: Any = None) -> ClickHouseQueryResource:
        settings = self.adapter_options.get("query_settings")
        try:
            context = self.client.query_df_stream(
                sql,
                parameters=params,
                settings=settings,
            )
            return ClickHouseQueryResource(context)
        except Exception as exc:
            raise DatabaseQueryError("打开 ClickHouse 流式查询失败") from exc

    def count_rows(self, sql: str, params: Any = None) -> int:
        rows = self.query(sql, params=params, result="rows")
        return int(rows[0][0]) if rows else 0

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.client.close()
        finally:
            super().close()

    def _column_type(
        self,
        series: pd.Series,
        options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        dtype = series.dtype
        if ptypes.is_bool_dtype(dtype):
            return "UInt8"
        if ptypes.is_unsigned_integer_dtype(dtype):
            return "UInt64"
        if ptypes.is_integer_dtype(dtype):
            return "Int64"
        if ptypes.is_float_dtype(dtype):
            return "Float64"
        if ptypes.is_datetime64_any_dtype(dtype):
            return "DateTime64(6)"

        resolved_options = dict(options or {})
        profile = profile_string_series(series)
        if profile.all_json_documents and resolved_options.get("infer_json", True):
            json_type = resolved_options.get("json_type", "auto")
            if json_type == "JSON":
                return "JSON"
            if json_type == "String":
                return "String"
            version = str(getattr(self.client, "server_version", ""))
            match = re.match(r"^(\d+)\.(\d+)", version)
            if match and (int(match.group(1)), int(match.group(2))) >= (25, 3):
                return "JSON"
        return "String"

    @staticmethod
    def _quote_literal(value: Any) -> str:
        return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        options = dict(dialect_options or {})
        json_type = options.get("json_type", "auto")
        if json_type not in {"auto", "JSON", "String"}:
            raise ValidationError("ClickHouse json_type 只支持 auto、JSON 或 String")
        engine = str(options.get("engine", "MergeTree"))
        if not _SAFE_ENGINE.fullmatch(engine):
            raise ValidationError(f"ClickHouse engine 参数无效: {engine!r}")
        key_value = options.get("key_columns") or options.get("order_by") or ()
        keys = (key_value,) if isinstance(key_value, str) else tuple(key_value)
        missing = [key for key in keys if key not in data.columns]
        if missing:
            raise InputValidationError(f"ClickHouse 建表数据缺少排序字段: {missing}")
        column_types = dict(options.get("column_types") or {})
        comments = dict(options.get("column_comments") or {})
        validate_column_mapping_keys(
            column_types,
            data.columns,
            option_name="column_types",
            database_type="ClickHouse",
        )
        validate_column_mapping_keys(
            comments,
            data.columns,
            option_name="column_comments",
            database_type="ClickHouse",
        )
        definitions = []
        for column in data.columns:
            column_type = resolve_column_type(
                column_types,
                column,
                self._column_type(data[column], options),
                database_type="ClickHouse",
            )
            definition = f"{self.quote_identifier(str(column))} {column_type}"
            if column in comments:
                definition += f" COMMENT {self._quote_literal(comments[column])}"
            definitions.append(definition)

        engine_expression = engine
        version_column = options.get("version_column")
        if engine.lower() == "replacingmergetree" and version_column is not None:
            if version_column not in data.columns:
                raise InputValidationError(f"ClickHouse 数据缺少版本字段: {version_column!r}")
            engine_expression += f"({self.quote_identifier(str(version_column))})"
        elif engine.lower().endswith("mergetree"):
            engine_expression += "()" if options.get("explicit_engine_parentheses") else ""

        if_not_exists = " IF NOT EXISTS" if options.get("if_not_exists", True) else ""
        order_by = "(" + ", ".join(self.quote_identifier(str(key)) for key in keys) + ")" if keys else "tuple()"
        ddl = (
            f"CREATE TABLE{if_not_exists} {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definitions)
            + f"\n) ENGINE = {engine_expression}\nORDER BY {order_by}"
        )
        partition_value = options.get("partition_by")
        if partition_value is not None:
            partitions = (partition_value,) if isinstance(partition_value, str) else tuple(partition_value)
            if any(partition not in data.columns for partition in partitions):
                raise InputValidationError("ClickHouse partition_by 只能引用输入字段")
            expression = ", ".join(self.quote_identifier(str(partition)) for partition in partitions)
            ddl += f"\nPARTITION BY ({expression})"
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            ddl += f"\nCOMMENT {self._quote_literal(table_comment)}"
        return ddl

    def create_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        ddl = self.build_create_table_sql(data, table_name, dialect_options)
        self.execute(ddl)
        return ddl

    def _database_and_table(self, table_name: str) -> Tuple[str, str]:
        from ..writing import split_qualified_name

        parts = split_qualified_name(table_name)
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        database_name = self.connect_kwargs.get("database", "default")
        return str(database_name), parts[-1]

    def table_exists(self, table_name: str) -> bool:
        """通过 system.tables 只读判断目标表是否存在。"""

        database_name, table = self._database_and_table(table_name)
        rows = self.query(
            "SELECT 1 FROM system.tables "
            "WHERE database={database:String} AND name={table:String} LIMIT 1",
            params={"database": database_name, "table": table},
            result="rows",
        )
        return bool(rows)

    @staticmethod
    def is_table_already_exists_error(exc: BaseException) -> bool:
        """ClickHouse 57/TABLE_ALREADY_EXISTS 表示重复建表。"""

        current: Optional[BaseException] = exc
        while current is not None:
            code = getattr(current, "code", None)
            args = getattr(current, "args", ())
            if code == 57 or (args and args[0] == 57) or "TABLE_ALREADY_EXISTS" in str(current).upper():
                return True
            current = current.__cause__
        return False

    def get_table_engine(self, table_name: str) -> str:
        database_name, table = self._database_and_table(table_name)
        rows = self.query(
            "SELECT engine FROM system.tables " "WHERE database={database:String} AND name={table:String}",
            params={"database": database_name, "table": table},
            result="rows",
        )
        return str(rows[0][0]) if rows else "UNKNOWN"

    def get_sort_key_columns(self, table_name: str) -> Tuple[str, ...]:
        database_name, table = self._database_and_table(table_name)
        records = self.query(
            "SELECT name FROM system.columns "
            "WHERE database={database:String} AND table={table:String} "
            "AND is_in_sorting_key=1 ORDER BY position",
            params={"database": database_name, "table": table},
            result="records",
        )
        return tuple(str(record["name"]) for record in records)

    @staticmethod
    def _is_replacing(engine: str) -> bool:
        return engine.strip().lower().startswith("replacingmergetree")

    def capabilities_for_table(
        self,
        table_name: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        del table_name
        engine = str((table_metadata or {}).get("engine", "MergeTree"))
        modes = {"o", "d"}
        if self._is_replacing(engine):
            modes.add("r")
        else:
            modes.add("a")
        return DatabaseCapabilities(
            transactions=False,
            streaming_read=True,
            native_bulk_write=True,
            metadata_export=True,
            write_modes=modes,
        )

    def resolve_key_columns(
        self,
        table_name: str,
        key_columns: Optional[Sequence[str]],
        first_batch: pd.DataFrame,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Sequence[str]]:
        del first_batch
        if key_columns is not None:
            return tuple(key_columns)
        engine = str((dialect_options or {}).get("engine") or self.get_table_engine(table_name))
        if self._is_replacing(engine):
            keys = self.get_sort_key_columns(table_name)
            if not keys:
                raise DatabaseCapabilityError(f"ClickHouse ReplacingMergeTree 表 {table_name!r} 未发现排序键")
            return keys
        return None

    def prepare_write(
        self,
        table_name: str,
        mode: str,
        first_batch: pd.DataFrame,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        options = dict(dialect_options or {})
        self.validate_write(
            table_name,
            mode,
            first_batch,
            key_columns=key_columns,
            dialect_options=options,
        )
        quoted = self.quote_qualified_name(table_name)
        if mode == "o":
            self.execute(f"TRUNCATE TABLE {quoted}")
        elif mode == "d":
            if key_columns:
                options["key_columns"] = list(key_columns)
            options["if_not_exists"] = False
            ddl = self.build_create_table_sql(first_batch, table_name, options)
            self.execute(f"DROP TABLE IF EXISTS {quoted}")
            self.execute(ddl)

    def validate_write(
        self,
        table_name: str,
        mode: str,
        first_batch: pd.DataFrame,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
        table_exists: Optional[bool] = None,
    ) -> None:
        """无副作用校验 ClickHouse 引擎与冲突语义。"""

        del first_batch
        options = dict(dialect_options or {})
        engine = str(options.get("engine") or ("MergeTree" if table_exists is False else self.get_table_engine(table_name)))
        capabilities = self.capabilities_for_table(table_name, {"engine": engine})
        if mode not in capabilities.write_modes:
            if mode == "r":
                raise DatabaseCapabilityError(
                    f"ClickHouse 目标表 {table_name!r} 必须使用 ReplacingMergeTree 才能执行 r 模式"
                )
            raise DatabaseCapabilityError(f"ClickHouse ReplacingMergeTree 表 {table_name!r} 无法保证主键冲突时不覆盖")
        if mode == "a" and key_columns:
            raise DatabaseCapabilityError(f"ClickHouse MergeTree 表 {table_name!r} 无法原生保证主键冲突忽略语义")
        if mode == "r" and not key_columns:
            raise DatabaseCapabilityError("ClickHouse r 模式必须指定排序键")

    def write_batch(
        self,
        table_name: str,
        batch: pd.DataFrame,
        mode: str,
        batch_index: int,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> BatchWriteResult:
        del batch_index, key_columns
        settings = dict(self.adapter_options.get("insert_settings") or {})
        settings.update(dict((dialect_options or {}).get("insert_settings") or {}))
        try:
            summary = self.client.insert_df(table_name, batch, settings=settings or None)
        except Exception as exc:
            raise DatabaseWriteError("ClickHouse DataFrame 批量写入失败") from exc
        if mode == "r":
            return BatchWriteResult()
        written_rows = getattr(summary, "written_rows", None)
        inserted = int(written_rows) if written_rows is not None else None
        return BatchWriteResult(inserted=inserted, updated=0, skipped=0)

    def finish_write(
        self,
        table_name: str,
        mode: str,
        result: WriteResult,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        del table_name, dialect_options
        if mode == "r":
            result.consistency = "eventual"

    @staticmethod
    def _metadata_filter(
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not targets:
            return "c.database != 'system'", {}
        clauses = []
        params: Dict[str, Any] = {}
        for index, target in enumerate(targets):
            database_key = f"database_{index}"
            database_name = target.parts[-2] if len(target.parts) >= 2 else target.parts[0]
            params[database_key] = database_name
            if len(target.parts) == 1:
                clauses.append(f"c.database={{{database_key}:String}}")
            else:
                table_key = f"table_{index}"
                params[table_key] = target.parts[-1]
                clauses.append(f"(c.database={{{database_key}:String}} AND c.table={{{table_key}:String}})")
        return "(" + " OR ".join(clauses) + ")", params

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        where_sql, params = self._metadata_filter(targets)
        sql = f"""
SELECT
  c.database AS database,
  c.table AS table_name,
  'BASE TABLE' AS table_type,
  t.comment AS table_comment,
  t.engine AS table_engine,
  c.name AS column_name,
  c.position AS ordinal_position,
  c.type AS data_type,
  c.type AS full_data_type,
  c.default_expression AS default_value,
  c.is_in_primary_key AS primary_key,
  c.is_in_sorting_key AS sort_key,
  c.comment AS column_comment
FROM system.columns c
JOIN system.tables t ON t.database=c.database AND t.name=c.table
WHERE {where_sql}
ORDER BY c.database, c.table, c.position
""".strip()
        frame = self.query(sql, params=params or None, result="dataframe")
        normalized = []
        for row in frame.to_dict("records"):
            database_name = row.get("database")
            table = row.get("table_name")
            normalized.append(
                {
                    "database_type": self.database_type,
                    "catalog": None,
                    "database": database_name,
                    "schema": None,
                    "table_name": table,
                    "qualified_name": (f"{database_name}.{table}" if database_name and table else None),
                    "table_type": row.get("table_type"),
                    "table_comment": row.get("table_comment"),
                    "table_engine": row.get("table_engine"),
                    "column_name": row.get("column_name"),
                    "ordinal_position": row.get("ordinal_position"),
                    "data_type": row.get("data_type"),
                    "full_data_type": row.get("full_data_type"),
                    "pandas_dtype": None,
                    "nullable": None,
                    "default_value": row.get("default_value"),
                    "primary_key": row.get("primary_key"),
                    "unique_key": None,
                    "partition_key": None,
                    "sort_key": row.get("sort_key"),
                    "bucket_key": None,
                    "column_comment": row.get("column_comment"),
                }
            )
        return MetadataInspection(rows=normalized, errors=[])


__all__ = ["ClickHouseAdapter", "ClickHouseQueryResource"]
