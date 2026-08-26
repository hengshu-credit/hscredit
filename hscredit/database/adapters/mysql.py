"""MySQL 数据库适配器。

使用 PyMySQL、DBUtils 和服务端游标实现池化查询、流式读取、建表、元数据扫描和四种写入模式。
"""

import math
import re
from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api import types as ptypes

from ...exceptions import DependencyError, InputValidationError, ValidationError
from ..exceptions import DatabaseCapabilityError, DatabaseQueryError
from ..metadata import MetadataInspection, QualifiedTarget
from ..type_inference import profile_string_series, resolve_bounded_string_length
from ..types import DatabaseCapabilities
from ..writing import (
    BatchWriteResult,
    resolve_column_type,
    split_qualified_name,
    validate_column_mapping_keys,
)
from .dbapi import DBAPIAdapter

_SAFE_OPTION = re.compile(r"^[A-Za-z0-9_]+$")


class MySQLAdapter(DBAPIAdapter):
    """MySQL/PyMySQL 数据库适配器。"""

    database_type = "mysql"
    identifier_quote = "`"
    capabilities = DatabaseCapabilities(
        transactions=True,
        streaming_read=True,
        native_bulk_write=False,
        metadata_export=True,
        write_modes={"a", "r", "o", "d"},
    )

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """使用 MySQL JSON 函数提取标量或嵌套 JSON 文本。"""

        extracted = f"JSON_EXTRACT({column_sql}, '{path}')"
        return f"CASE WHEN JSON_TYPE({extracted}) = 'NULL' THEN NULL ELSE JSON_UNQUOTE({extracted}) END"

    def load_driver(self) -> Any:
        """按需加载 PyMySQL。"""

        try:
            import pymysql
        except ImportError as exc:
            raise DependencyError("缺少 MySQL 可选依赖，请安装: pip install hscredit[db-mysql]") from exc
        return pymysql

    def create_cursor(self, connection: Any, *, stream: bool = False) -> Any:
        """流式读取时使用 PyMySQL ``SSCursor``。"""

        if stream:
            return connection.cursor(self.driver.cursors.SSCursor)
        return connection.cursor()

    @staticmethod
    def _validate_option_token(name: str, value: str) -> str:
        if not isinstance(value, str) or not _SAFE_OPTION.fullmatch(value):
            raise ValidationError(f"MySQL 建表参数 {name} 包含非法值: {value!r}")
        return value

    @staticmethod
    def _quote_literal(value: Any) -> str:
        return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"

    @staticmethod
    def _default_column_type(
        series: pd.Series,
        options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        dtype = series.dtype
        if ptypes.is_bool_dtype(dtype):
            return "BOOLEAN"
        if ptypes.is_integer_dtype(dtype):
            return "BIGINT"
        if ptypes.is_float_dtype(dtype):
            return "DOUBLE"
        if ptypes.is_datetime64_any_dtype(dtype):
            return "DATETIME"
        if ptypes.is_timedelta64_dtype(dtype):
            return "BIGINT"

        resolved_options = dict(options or {})
        profile = profile_string_series(series)
        if profile.all_json_documents and resolved_options.get("infer_json", True):
            return "JSON"
        if not profile.all_strings or profile.non_null_count == 0:
            return "VARCHAR(255)"

        varchar_limit = int(resolved_options.get("varchar_max_length", 255))
        if not 1 <= varchar_limit <= 65_535:
            raise ValidationError("MySQL varchar_max_length 必须位于 1 到 65535")
        charset = str(resolved_options.get("charset", "utf8mb4")).lower()
        default_widths = {
            "ascii": 1,
            "binary": 1,
            "latin1": 1,
            "ucs2": 2,
            "utf8": 3,
            "utf8mb3": 3,
            "utf8mb4": 4,
            "utf16": 4,
            "utf32": 4,
        }
        known_charset_width = default_widths.get(charset)
        configured_charset_width = resolved_options.get("charset_max_bytes_per_character")
        if (
            configured_charset_width is not None
            and known_charset_width is not None
            and int(configured_charset_width) < known_charset_width
        ):
            raise ValidationError(f"MySQL {charset} 字符集最大字节宽度不能小于 {known_charset_width}")
        charset_width = int(
            configured_charset_width if configured_charset_width is not None else (known_charset_width or 4)
        )
        if charset_width <= 0:
            raise ValidationError("MySQL charset_max_bytes_per_character 必须是正整数")
        varchar_byte_limit = int(resolved_options.get("varchar_max_bytes", 65_533))
        if not 1 <= varchar_byte_limit <= 65_533:
            raise ValidationError("MySQL varchar_max_bytes 必须位于 1 到 65533")
        headroom = float(resolved_options.get("string_length_headroom", 1.2))
        if headroom < 1:
            raise ValidationError("MySQL string_length_headroom 不能小于 1")
        safe_character_limit = min(
            varchar_limit,
            varchar_byte_limit // charset_width,
        )
        target_characters = math.ceil(profile.max_characters * headroom)
        target_bytes = math.ceil(profile.max_utf8_bytes * headroom)
        if target_characters <= safe_character_limit and target_bytes <= varchar_byte_limit:
            length = resolve_bounded_string_length(
                profile.max_characters,
                maximum=safe_character_limit,
                headroom=headroom,
            )
            return f"VARCHAR({length})"
        if target_bytes <= 65_535:
            return "TEXT"
        if target_bytes <= 16_777_215:
            return "MEDIUMTEXT"
        return "LONGTEXT"

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """根据 DataFrame 生成 MySQL 建表语句。"""

        options = dict(dialect_options or {})
        key_columns = tuple(options.get("key_columns") or options.get("primary_key") or ())
        if isinstance(options.get("primary_key"), str):
            key_columns = (options["primary_key"],)
        if isinstance(options.get("key_columns"), str):
            key_columns = (options["key_columns"],)
        missing_keys = [column for column in key_columns if column not in data.columns]
        if missing_keys:
            raise InputValidationError(f"MySQL 建表数据缺少主键字段: {missing_keys}")

        column_types = dict(options.get("column_types") or {})
        column_comments = dict(options.get("column_comments") or options.get("feature_map") or {})
        validate_column_mapping_keys(
            column_types,
            data.columns,
            option_name="column_types",
            database_type="MySQL",
        )
        validate_column_mapping_keys(
            column_comments,
            data.columns,
            option_name="column_comments",
            database_type="MySQL",
        )
        definitions: List[str] = []
        for column in data.columns:
            column_type = resolve_column_type(
                column_types,
                column,
                self._default_column_type(data[column], options),
                database_type="MySQL",
            )
            definition = f"{self.quote_identifier(str(column))} {column_type}"
            if column in key_columns:
                definition += " NOT NULL"
            comment = column_comments.get(column)
            if comment is not None:
                definition += f" COMMENT {self._quote_literal(comment)}"
            definitions.append(definition)

        if key_columns:
            quoted_keys = ", ".join(self.quote_identifier(column) for column in key_columns)
            definitions.append(f"PRIMARY KEY ({quoted_keys})")

        if_not_exists = " IF NOT EXISTS" if options.get("if_not_exists", True) else ""
        engine = self._validate_option_token("engine", options.get("engine", "InnoDB"))
        charset = self._validate_option_token("charset", options.get("charset", "utf8mb4"))
        ddl = (
            f"CREATE TABLE{if_not_exists} {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definitions)
            + f"\n) ENGINE={engine} DEFAULT CHARSET={charset}"
        )
        collate = options.get("collate")
        if collate is not None:
            ddl += f" COLLATE={self._validate_option_token('collate', collate)}"
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            ddl += f" COMMENT={self._quote_literal(table_comment)}"
        return ddl

    def create_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """创建 MySQL 表并返回实际执行的 DDL。"""

        ddl = self.build_create_table_sql(data, table_name, dialect_options)
        self.execute(ddl)
        return ddl

    def _schema_and_table(self, table_name: str) -> Tuple[str, str]:
        parts = split_qualified_name(table_name)
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        schema = self.connect_kwargs.get("database") or self.connect_kwargs.get("db")
        if not schema:
            raise ValidationError("未指定默认数据库，MySQL 表名必须使用 数据库名.表名")
        return str(schema), parts[-1]

    def table_exists(self, table_name: str) -> bool:
        """通过 information_schema 只读判断目标表是否存在。"""

        schema, table = self._schema_and_table(table_name)
        rows = self.query(
            "SELECT 1 FROM information_schema.tables "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s LIMIT 1",
            params=(schema, table),
            result="rows",
        )
        return bool(rows)

    def get_key_columns(self, table_name: str) -> Tuple[str, ...]:
        """读取目标表主键；没有主键时回退到第一个唯一索引。"""

        schema, table = self._schema_and_table(table_name)
        sql = """
SELECT INDEX_NAME, COLUMN_NAME, SEQ_IN_INDEX
FROM information_schema.statistics
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
  AND (INDEX_NAME='PRIMARY' OR NON_UNIQUE=0)
ORDER BY CASE WHEN INDEX_NAME='PRIMARY' THEN 0 ELSE 1 END,
         INDEX_NAME, SEQ_IN_INDEX
""".strip()
        rows = self.query(sql, params=(schema, table), result="records")
        grouped: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()
        for row in rows:
            index_name = row.get("INDEX_NAME", row.get("index_name"))
            column_name = row.get("COLUMN_NAME", row.get("column_name"))
            sequence = row.get("SEQ_IN_INDEX", row.get("seq_in_index", 0))
            if index_name is None or column_name is None:
                continue
            grouped.setdefault(str(index_name), []).append((int(sequence), str(column_name)))
        if not grouped:
            return ()
        selected = "PRIMARY" if "PRIMARY" in grouped else next(iter(grouped))
        return tuple(column for _, column in sorted(grouped[selected]))

    def resolve_key_columns(
        self,
        table_name: str,
        key_columns: Optional[Sequence[str]],
        first_batch: pd.DataFrame,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Sequence[str]]:
        """解析显式字段或目标表主键/唯一键。"""

        del first_batch, dialect_options
        if key_columns is not None:
            return tuple(key_columns)
        keys = self.get_key_columns(table_name)
        if not keys:
            raise DatabaseCapabilityError(f"MySQL 目标表 {table_name!r} 没有主键或唯一键，无法保证 a/r 冲突语义")
        return keys

    def build_insert_sql(
        self,
        table_name: str,
        columns: Sequence[str],
        *,
        mode: str,
        key_columns: Optional[Sequence[str]] = None,
    ) -> str:
        """生成 MySQL 批量插入或冲突覆盖 SQL。"""

        quoted_table = self.quote_qualified_name(table_name)
        quoted_columns = ", ".join(self.quote_identifier(str(column)) for column in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES ({placeholders})"
        if mode == "a" and not key_columns:
            raise DatabaseCapabilityError("MySQL a 模式必须指定主键或唯一键字段")
        if mode == "r":
            keys = set(key_columns or ())
            update_columns = [column for column in columns if column not in keys]
            if not update_columns:
                update_columns = list(key_columns or columns[:1])
            assignments = ", ".join(
                f"{self.quote_identifier(str(column))}=VALUES({self.quote_identifier(str(column))})"
                for column in update_columns
            )
            sql += f" ON DUPLICATE KEY UPDATE {assignments}"
        return sql

    def prepare_write(
        self,
        table_name: str,
        mode: str,
        first_batch: pd.DataFrame,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """准备 MySQL 四种写入模式。"""

        self.validate_write(
            table_name,
            mode,
            first_batch,
            key_columns=key_columns,
            dialect_options=dialect_options,
        )
        quoted_table = self.quote_qualified_name(table_name)
        if mode == "o":
            self.execute(f"TRUNCATE TABLE {quoted_table}")
        elif mode == "d":
            options = dict(dialect_options or {})
            if key_columns:
                options["key_columns"] = list(key_columns)
            options["if_not_exists"] = False
            ddl = self.build_create_table_sql(first_batch, table_name, options)
            self.execute(f"DROP TABLE IF EXISTS {quoted_table}")
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
        """无副作用校验 MySQL 写入模式和冲突键。"""

        del first_batch, dialect_options, table_exists
        self.require_write_mode(table_name, mode)
        if mode in {"a", "r"} and not key_columns:
            raise DatabaseCapabilityError(f"MySQL 目标表 {table_name!r} 未解析到主键，无法执行模式 {mode!r}")

    @staticmethod
    def _dbapi_value(value: Any) -> Any:
        if value is None or value is pd.NA:
            return None
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    @staticmethod
    def _mysql_error_code(error: BaseException) -> Optional[int]:
        current: Optional[BaseException] = error
        while current is not None:
            args = getattr(current, "args", ())
            if args and isinstance(args[0], int):
                return int(args[0])
            current = current.__cause__
        return None

    def is_table_already_exists_error(self, exc: BaseException) -> bool:
        """MySQL 1050 表示目标表已存在。"""

        return self._mysql_error_code(exc) == 1050

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
        """通过 PyMySQL ``executemany`` 写入一个 DataFrame 批次。"""

        del batch_index, dialect_options
        sql = self.build_insert_sql(
            table_name,
            [str(column) for column in batch.columns],
            mode=mode,
            key_columns=key_columns,
        )
        values = [tuple(self._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
        if mode == "a":
            inserted = 0
            skipped = 0
            try:
                with self.connection_cursor() as (connection, cursor):
                    try:
                        for value in values:
                            try:
                                self.execute_cursor(cursor, sql, value)
                            except Exception as exc:
                                if self._mysql_error_code(exc) != 1062:
                                    raise
                                skipped += 1
                            else:
                                inserted += 1
                        connection.commit()
                    except Exception as exc:
                        self._rollback_quietly(connection)
                        raise DatabaseQueryError("MySQL追加写入失败") from exc
            except DatabaseQueryError:
                raise
            except Exception as exc:
                raise DatabaseQueryError("MySQL追加写入失败") from exc
            return BatchWriteResult(
                inserted=inserted,
                updated=0,
                skipped=skipped,
            )
        affected = self.executemany(sql, values)
        if mode == "r":
            return BatchWriteResult()
        inserted = int(affected) if int(affected) >= 0 else None
        return BatchWriteResult(inserted=inserted, updated=0, skipped=0)

    @staticmethod
    def _metadata_filters(
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> Tuple[str, Tuple[Any, ...]]:
        if not targets:
            return (
                "t.TABLE_SCHEMA NOT IN ('information_schema','mysql','performance_schema','sys')",
                (),
            )
        clauses = []
        params: List[Any] = []
        for target in targets:
            if len(target.parts) == 1:
                clauses.append("t.TABLE_SCHEMA=%s")
                params.append(target.parts[0])
            else:
                clauses.append("(t.TABLE_SCHEMA=%s AND t.TABLE_NAME=%s)")
                params.extend(target.parts[-2:])
        return "(" + " OR ".join(clauses) + ")", tuple(params)

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        """一次查询导出 MySQL 可见表和字段元数据。"""

        where_sql, params = self._metadata_filters(targets)
        sql = f"""
SELECT
    t.TABLE_CATALOG AS catalog,
    t.TABLE_SCHEMA AS database_name,
    t.TABLE_NAME AS table_name,
    t.TABLE_TYPE AS table_type,
    t.TABLE_COMMENT AS table_comment,
    t.ENGINE AS table_engine,
    c.COLUMN_NAME AS column_name,
    c.ORDINAL_POSITION AS ordinal_position,
    c.DATA_TYPE AS data_type,
    c.COLUMN_TYPE AS full_data_type,
    c.IS_NULLABLE AS nullable,
    c.COLUMN_DEFAULT AS default_value,
    c.COLUMN_KEY AS column_key,
    c.COLUMN_COMMENT AS column_comment
FROM information_schema.tables t
JOIN information_schema.columns c
  ON c.TABLE_SCHEMA=t.TABLE_SCHEMA AND c.TABLE_NAME=t.TABLE_NAME
WHERE {where_sql}
ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
""".strip()
        rows = self.query(sql, params=params or None, result="records")
        normalized = []
        for row in rows:
            database_name = row.get("database_name")
            table_name = row.get("table_name")
            column_key = row.get("column_key")
            normalized.append(
                {
                    "database_type": self.database_type,
                    "catalog": row.get("catalog"),
                    "database": database_name,
                    "schema": None,
                    "table_name": table_name,
                    "qualified_name": (
                        f"{database_name}.{table_name}"
                        if database_name is not None and table_name is not None
                        else None
                    ),
                    "table_type": row.get("table_type"),
                    "table_comment": row.get("table_comment"),
                    "table_engine": row.get("table_engine"),
                    "column_name": row.get("column_name"),
                    "ordinal_position": row.get("ordinal_position"),
                    "data_type": row.get("data_type"),
                    "full_data_type": row.get("full_data_type"),
                    "pandas_dtype": None,
                    "nullable": row.get("nullable"),
                    "default_value": row.get("default_value"),
                    "primary_key": column_key == "PRI",
                    "unique_key": column_key,
                    "partition_key": None,
                    "sort_key": None,
                    "bucket_key": None,
                    "column_comment": row.get("column_comment"),
                }
            )
        return MetadataInspection(rows=normalized, errors=[])


__all__ = ["MySQLAdapter"]
