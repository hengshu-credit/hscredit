"""HiveServer2 数据库适配器。

使用 Impyla DB-API 和 DBUtils 连接池，支持分块读取、Parquet 建表、严格事务表写入和元数据导出。
"""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api import types as ptypes

from ...exceptions import DependencyError, InputValidationError, ValidationError
from ..exceptions import DatabaseCapabilityError, DatabaseQueryError
from ..metadata import MetadataInspection, QualifiedTarget
from ..types import DatabaseCapabilities, PoolOptions
from ..writing import BatchWriteResult, resolve_column_type, validate_column_mapping_keys
from .dbapi import DBAPIAdapter

_SAFE_STORAGE = re.compile(r"^[A-Za-z0-9_]+$")


def parse_describe_formatted(
    rows: Iterable[Any],
    database_type: str,
    database_name: str,
    table_name: str,
) -> List[Dict[str, Any]]:
    """解析 Hive/Impala ``DESCRIBE FORMATTED`` 的列与原始表属性。"""

    materialized = list(rows)
    columns: List[Tuple[str, Any, Any]] = []
    table_values: Dict[str, Any] = {}
    reading_columns = True
    for raw in materialized:
        if isinstance(raw, Mapping):
            values = list(raw.values())
        else:
            values = list(raw)
        values += [None] * (3 - len(values))
        first, second, third = values[:3]
        label = "" if first is None else str(first).strip()
        if reading_columns:
            if not label:
                if columns:
                    reading_columns = False
                continue
            if label.startswith("#"):
                continue
            columns.append((label, second, third))
        elif label.endswith(":"):
            table_values[label[:-1].strip()] = second

    table_type = table_values.get("Table Type")
    table_comment = table_values.get("Comment")
    table_engine = table_values.get("InputFormat") or table_values.get("Storage Handler")
    result = []
    for position, (column_name, data_type, comment) in enumerate(columns, start=1):
        result.append(
            {
                "database_type": database_type,
                "catalog": None,
                "database": database_name,
                "schema": None,
                "table_name": table_name,
                "qualified_name": f"{database_name}.{table_name}",
                "table_type": table_type,
                "table_comment": table_comment,
                "table_engine": table_engine,
                "column_name": column_name,
                "ordinal_position": position,
                "data_type": data_type,
                "full_data_type": data_type,
                "pandas_dtype": None,
                "nullable": None,
                "default_value": None,
                "primary_key": None,
                "unique_key": None,
                "partition_key": None,
                "sort_key": None,
                "bucket_key": None,
                "column_comment": comment,
            }
        )
    return result


class HiveAdapter(DBAPIAdapter):
    """HiveServer2/Impyla 数据库适配器。"""

    database_type = "hive"
    identifier_quote = "`"
    default_port = 10_000
    default_auth_mechanism = "PLAIN"
    capabilities = DatabaseCapabilities(
        transactions=False,
        streaming_read=True,
        native_bulk_write=False,
        metadata_export=True,
        write_modes={"a", "o", "d"},
    )

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """使用 Hive ``GET_JSON_OBJECT`` 提取 JSON 路径。"""

        return f"GET_JSON_OBJECT({column_sql}, '{path}')"

    def __init__(
        self,
        *,
        connect_kwargs: Mapping[str, Any],
        pool_options: PoolOptions,
        adapter_options: Optional[Mapping[str, Any]] = None,
    ):
        resolved = dict(connect_kwargs)
        resolved.setdefault("port", self.default_port)
        resolved.setdefault("auth_mechanism", self.default_auth_mechanism)
        super().__init__(
            connect_kwargs=resolved,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )

    def load_driver(self) -> Any:
        try:
            from impala import dbapi
        except ImportError as exc:
            raise DependencyError(
                f"缺少 {self.database_type} 可选依赖，请安装: " f"pip install hscredit[db-{self.database_type}]"
            ) from exc
        return dbapi

    def create_cursor(self, connection: Any, *, stream: bool = False) -> Any:
        del stream
        cursor = connection.cursor()
        cursor.arraysize = int(self.adapter_options.get("arraysize", 1_000))
        return cursor

    @staticmethod
    def _quote_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _column_type(series: pd.Series) -> str:
        dtype = series.dtype
        if ptypes.is_bool_dtype(dtype):
            return "BOOLEAN"
        if ptypes.is_integer_dtype(dtype):
            return "BIGINT"
        if ptypes.is_float_dtype(dtype):
            return "DOUBLE"
        if ptypes.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        return "STRING"

    def _column_definitions(
        self,
        data: pd.DataFrame,
        options: Mapping[str, Any],
        *,
        inline_primary: bool = False,
    ) -> List[str]:
        key_value = options.get("key_columns") or options.get("primary_key") or ()
        keys = (key_value,) if isinstance(key_value, str) else tuple(key_value)
        missing = [key for key in keys if key not in data.columns]
        if missing:
            raise InputValidationError(f"建表数据缺少主键字段: {missing}")
        types = dict(options.get("column_types") or {})
        comments = dict(options.get("column_comments") or {})
        validate_column_mapping_keys(
            types,
            data.columns,
            option_name="column_types",
            database_type=self.database_type,
        )
        validate_column_mapping_keys(
            comments,
            data.columns,
            option_name="column_comments",
            database_type=self.database_type,
        )
        definitions = []
        for column in data.columns:
            column_type = resolve_column_type(
                types,
                column,
                self._column_type(data[column]),
                database_type=self.database_type,
            )
            definition = f"{self.quote_identifier(str(column))} {column_type}"
            if inline_primary and column in keys:
                definition += " PRIMARY KEY"
            comment = comments.get(column)
            if comment is not None:
                definition += f" COMMENT {self._quote_literal(comment)}"
            definitions.append(definition)
        return definitions

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        options = dict(dialect_options or {})
        storage = str(options.get("storage", "PARQUET")).upper()
        if not _SAFE_STORAGE.fullmatch(storage):
            raise ValidationError(f"Hive storage 参数无效: {storage!r}")
        definitions = self._column_definitions(data, options)
        if_not_exists = " IF NOT EXISTS" if options.get("if_not_exists", True) else ""
        ddl = (
            f"CREATE TABLE{if_not_exists} {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definitions)
            + "\n)"
        )
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            ddl += f"\nCOMMENT {self._quote_literal(table_comment)}"
        ddl += f"\nSTORED AS {storage}"
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

    def capabilities_for_table(
        self,
        table_name: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        del table_name
        transactional = bool((table_metadata or {}).get("transactional"))
        modes = {"a", "o", "d"}
        if transactional:
            modes.add("r")
        return DatabaseCapabilities(
            transactions=transactional,
            streaming_read=True,
            native_bulk_write=False,
            metadata_export=True,
            write_modes=modes,
        )

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
        capabilities = self.capabilities_for_table(table_name, options)
        if mode not in capabilities.write_modes:
            raise DatabaseCapabilityError(f"Hive 目标表 {table_name!r} 不是支持 MERGE 的事务表，无法执行 r 模式")
        if mode == "a" and key_columns:
            raise DatabaseCapabilityError(f"Hive 目标表 {table_name!r} 无法原生保证主键冲突忽略语义")
        if mode == "r" and not key_columns:
            raise DatabaseCapabilityError("Hive MERGE 必须指定 key_columns")
        quoted = self.quote_qualified_name(table_name)
        if mode == "o":
            self.execute(f"TRUNCATE TABLE {quoted}")
        elif mode == "d":
            create_options = dict(options)
            if key_columns:
                create_options["key_columns"] = list(key_columns)
            create_options["if_not_exists"] = False
            ddl = self.build_create_table_sql(first_batch, table_name, create_options)
            self.execute(f"DROP TABLE IF EXISTS {quoted}")
            self.execute(ddl)

    def _insert_sql(self, table_name: str, columns: Sequence[str], prefix: str = "INSERT INTO") -> str:
        quoted = ", ".join(self.quote_identifier(str(column)) for column in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        return f"{prefix} {self.quote_qualified_name(table_name)} ({quoted}) " f"VALUES ({placeholders})"

    def _merge_sql(
        self,
        table_name: str,
        columns: Sequence[str],
        key_columns: Sequence[str],
    ) -> str:
        source = ", ".join(f"%s AS {self.quote_identifier(str(column))}" for column in columns)
        condition = " AND ".join(
            f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
            for column in key_columns
        )
        keys = set(key_columns)
        updates = ", ".join(
            f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
            for column in columns
            if column not in keys
        )
        insert_columns = ", ".join(self.quote_identifier(str(column)) for column in columns)
        insert_values = ", ".join(f"source.{self.quote_identifier(str(column))}" for column in columns)
        sql = (
            f"MERGE INTO {self.quote_qualified_name(table_name)} target "
            f"USING (SELECT {source}) source ON ({condition})"
        )
        if updates:
            sql += f" WHEN MATCHED THEN UPDATE SET {updates}"
        return sql + f" WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})"

    @staticmethod
    def _dbapi_value(value: Any) -> Any:
        if value is None or value is pd.NA:
            return None
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

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
        del batch_index, dialect_options
        columns = [str(column) for column in batch.columns]
        sql = (
            self._merge_sql(table_name, columns, key_columns or ())
            if mode == "r"
            else self._insert_sql(table_name, columns)
        )
        values = [tuple(self._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
        affected = self.executemany(sql, values)
        inserted = int(affected) if int(affected) >= 0 else None
        if mode == "r":
            return BatchWriteResult()
        return BatchWriteResult(inserted=inserted, updated=0, skipped=0)

    @staticmethod
    def _metadata_filters(
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not targets:
            return "1=1", {}
        clauses = []
        params: Dict[str, Any] = {}
        for index, target in enumerate(targets):
            schema_key = f"schema_{index}"
            schema = target.parts[-2] if len(target.parts) >= 2 else target.parts[0]
            params[schema_key] = schema
            if len(target.parts) == 1:
                clauses.append(f"c.TABLE_SCHEMA=%({schema_key})s")
            else:
                table_key = f"table_{index}"
                params[table_key] = target.parts[-1]
                clauses.append(f"(c.TABLE_SCHEMA=%({schema_key})s AND c.TABLE_NAME=%({table_key})s)")
        return "(" + " OR ".join(clauses) + ")", params

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        where_sql, params = self._metadata_filters(targets)
        sql = f"""
SELECT
  c.TABLE_CATALOG,
  c.TABLE_SCHEMA,
  c.TABLE_NAME,
  t.TABLE_TYPE,
  c.COLUMN_NAME,
  c.ORDINAL_POSITION,
  c.DATA_TYPE,
  c.IS_NULLABLE,
  c.COLUMN_DEFAULT
FROM information_schema.columns c
LEFT JOIN information_schema.tables t
  ON t.TABLE_SCHEMA=c.TABLE_SCHEMA AND t.TABLE_NAME=c.TABLE_NAME
WHERE {where_sql}
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
""".strip()
        try:
            rows = self.query(sql, params=params or None, result="records")
        except DatabaseQueryError:
            return self._inspect_schema_describe(targets)

        normalized = []
        for raw in rows:
            row = {str(key).lower(): value for key, value in raw.items()}
            database_name = row.get("table_schema")
            table_name = row.get("table_name")
            normalized.append(
                {
                    "database_type": self.database_type,
                    "catalog": row.get("table_catalog"),
                    "database": database_name,
                    "schema": None,
                    "table_name": table_name,
                    "qualified_name": (f"{database_name}.{table_name}" if database_name and table_name else None),
                    "table_type": row.get("table_type"),
                    "table_comment": None,
                    "table_engine": None,
                    "column_name": row.get("column_name"),
                    "ordinal_position": row.get("ordinal_position"),
                    "data_type": row.get("data_type"),
                    "full_data_type": row.get("data_type"),
                    "pandas_dtype": None,
                    "nullable": row.get("is_nullable"),
                    "default_value": row.get("column_default"),
                    "primary_key": None,
                    "unique_key": None,
                    "partition_key": None,
                    "sort_key": None,
                    "bucket_key": None,
                    "column_comment": None,
                }
            )
        return MetadataInspection(rows=normalized, errors=[])

    def _inspect_schema_describe(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        errors: List[Any] = []
        rows: List[Dict[str, Any]] = []
        requested: Dict[str, Optional[set]] = {}
        if targets:
            for target in targets:
                database_name = target.parts[-2] if len(target.parts) >= 2 else target.parts[0]
                table_name = target.parts[-1] if len(target.parts) >= 2 else None
                requested.setdefault(database_name, set() if table_name else None)
                if table_name is not None and requested[database_name] is not None:
                    requested[database_name].add(table_name)
        else:
            for result in self.query("SHOW DATABASES", result="rows"):
                requested[str(result[0])] = None

        for database_name, table_names in requested.items():
            if table_names is None:
                table_names = {
                    str(result[0])
                    for result in self.query(
                        f"SHOW TABLES IN {self.quote_identifier(database_name)}",
                        result="rows",
                    )
                }
            for table_name in sorted(table_names):
                qualified = f"{database_name}.{table_name}"
                try:
                    described = self.query(
                        f"DESCRIBE FORMATTED {self.quote_qualified_name(qualified)}",
                        result="rows",
                    )
                    rows.extend(
                        parse_describe_formatted(
                            described,
                            self.database_type,
                            database_name,
                            table_name,
                        )
                    )
                except DatabaseQueryError as exc:
                    errors.append({"目标": qualified, "错误": str(exc)})
        return MetadataInspection(rows=rows, errors=errors)


__all__ = ["HiveAdapter", "parse_describe_formatted"]
