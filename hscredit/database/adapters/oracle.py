"""Oracle 数据库适配器。

使用 python-oracledb 原生连接池、数组游标、MERGE 和 Oracle 数据字典实现统一数据库契约。
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


class _OraclePoolWrapper:
    """把 python-oracledb 原生池适配为共享池接口。"""

    def __init__(self, pool: Any):
        self.pool = pool

    def connection(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self.pool.acquire()

    def close(self) -> None:
        self.pool.close()


class OracleAdapter(DBAPIAdapter):
    """Oracle/python-oracledb 数据库适配器。"""

    database_type = "oracle"
    identifier_quote = '"'
    capabilities = DatabaseCapabilities(
        transactions=True,
        streaming_read=True,
        native_bulk_write=True,
        metadata_export=True,
        write_modes={"a", "r", "o", "d"},
    )

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """同时兼容 Oracle JSON 标量和对象/数组路径。"""

        scalar = f"JSON_VALUE({column_sql}, '{path}' RETURNING CLOB NULL ON EMPTY NULL ON ERROR)"
        nested = f"JSON_QUERY({column_sql}, '{path}' RETURNING CLOB NULL ON EMPTY NULL ON ERROR)"
        return f"COALESCE({scalar}, {nested})"

    def load_driver(self) -> Any:
        try:
            import oracledb
        except ImportError as exc:
            raise DependencyError("缺少 Oracle 可选依赖，请安装: pip install hscredit[db-oracle]") from exc
        return oracledb

    def _create_pool(self) -> _OraclePoolWrapper:
        minimum = self.pool_options.mincached
        maximum = self.pool_options.maxconnections or self.pool_options.maxcached or max(minimum, 4)
        kwargs: Dict[str, Any] = dict(self.connect_kwargs)
        kwargs.update(
            {
                "min": minimum,
                "max": maximum,
                "increment": int(self.adapter_options.get("increment", 1)),
            }
        )
        if self.pool_options.blocking and hasattr(self.driver, "POOL_GETMODE_WAIT"):
            kwargs["getmode"] = self.driver.POOL_GETMODE_WAIT
        try:
            return _OraclePoolWrapper(self.driver.create_pool(**kwargs))
        except Exception as exc:
            from ..exceptions import DatabaseConnectionError

            raise DatabaseConnectionError("创建 Oracle 原生连接池失败") from exc

    def create_cursor(self, connection: Any, *, stream: bool = False) -> Any:
        del stream
        cursor = connection.cursor()
        cursor.arraysize = int(self.adapter_options.get("arraysize", 1_000))
        return cursor

    @staticmethod
    def _quote_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _default_column_type(
        series: pd.Series,
        options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        dtype = series.dtype
        if ptypes.is_bool_dtype(dtype):
            return "NUMBER(1)"
        if ptypes.is_integer_dtype(dtype):
            return "NUMBER(19)"
        if ptypes.is_float_dtype(dtype):
            return "BINARY_DOUBLE"
        if ptypes.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        if ptypes.is_timedelta64_dtype(dtype):
            return "INTERVAL DAY TO SECOND"

        resolved_options = dict(options or {})
        profile = profile_string_series(series)
        if profile.all_json_documents and resolved_options.get("infer_json", True):
            return "JSON" if resolved_options.get("native_json", False) else "CLOB"
        if not profile.all_strings or profile.non_null_count == 0:
            return "VARCHAR2(255 CHAR)"
        varchar_limit = int(resolved_options.get("varchar_max_length", 4_000))
        if not 1 <= varchar_limit <= 32_767:
            raise ValidationError("Oracle varchar_max_length 必须位于 1 到 32767")
        if profile.max_characters > varchar_limit:
            return "CLOB"
        length = resolve_bounded_string_length(
            profile.max_characters,
            maximum=varchar_limit,
        )
        return f"VARCHAR2({length} CHAR)"

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """根据 DataFrame 生成 Oracle 建表 DDL。"""

        options = dict(dialect_options or {})
        key_value = options.get("key_columns") or options.get("primary_key") or ()
        key_columns = (key_value,) if isinstance(key_value, str) else tuple(key_value)
        missing = [column for column in key_columns if column not in data.columns]
        if missing:
            raise InputValidationError(f"Oracle 建表数据缺少主键字段: {missing}")

        column_types = dict(options.get("column_types") or {})
        column_comments = dict(options.get("column_comments") or {})
        validate_column_mapping_keys(
            column_types,
            data.columns,
            option_name="column_types",
            database_type="Oracle",
        )
        validate_column_mapping_keys(
            column_comments,
            data.columns,
            option_name="字段注释",
            database_type="Oracle",
        )
        definitions: List[str] = []
        for column in data.columns:
            column_type = resolve_column_type(
                column_types,
                column,
                self._default_column_type(data[column], options),
                database_type="Oracle",
            )
            definition = f"{self.quote_identifier(str(column))} {column_type}"
            if column in key_columns:
                definition += " NOT NULL"
            definitions.append(definition)

        if key_columns:
            table_part = split_qualified_name(table_name)[-1]
            constraint_name = str(options.get("primary_key_name") or f"{table_part}_PK")
            quoted_keys = ", ".join(self.quote_identifier(column) for column in key_columns)
            definitions.append(f"CONSTRAINT {self.quote_identifier(constraint_name)} PRIMARY KEY ({quoted_keys})")
        return f"CREATE TABLE {self.quote_qualified_name(table_name)} (\n  " + ",\n  ".join(definitions) + "\n)"

    def create_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """创建 Oracle 表，并单独写入表/字段注释。"""

        options = dict(dialect_options or {})
        ddl = self.build_create_table_sql(data, table_name, options)
        self.execute(ddl)
        for column, comment in dict(options.get("column_comments") or {}).items():
            if column not in data.columns:
                raise InputValidationError(f"Oracle 字段注释引用未知字段: {column!r}")
            self.execute(
                f"COMMENT ON COLUMN {self.quote_qualified_name(table_name)}."
                f"{self.quote_identifier(str(column))} IS {self._quote_literal(comment)}"
            )
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            self.execute(
                f"COMMENT ON TABLE {self.quote_qualified_name(table_name)} " f"IS {self._quote_literal(table_comment)}"
            )
        return ddl

    def table_exists(self, table_name: str) -> bool:
        """通过 Oracle 数据字典只读判断目标表是否存在。"""

        owner, table = self._owner_and_table(table_name)
        rows = self.query(
            "SELECT 1 AS PRESENT FROM ALL_TABLES " "WHERE OWNER=:owner AND TABLE_NAME=:table_name",
            params={"owner": owner.upper(), "table_name": table.upper()},
            result="rows",
        )
        return bool(rows)

    def _owner_and_table(self, table_name: str) -> Tuple[str, str]:
        parts = split_qualified_name(table_name)
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        owner = self.connect_kwargs.get("schema") or self.connect_kwargs.get("user")
        if not owner:
            raise ValidationError("Oracle 表名必须使用 模式名.表名，或在连接中指定 user/schema")
        return str(owner).upper(), parts[-1]

    def get_key_columns(self, table_name: str) -> Tuple[str, ...]:
        owner, table = self._owner_and_table(table_name)
        sql = """
SELECT cc.OWNER, cc.TABLE_NAME, cc.COLUMN_NAME, cc.POSITION
FROM ALL_CONS_COLUMNS cc
JOIN ALL_CONSTRAINTS c
  ON c.OWNER=cc.OWNER AND c.CONSTRAINT_NAME=cc.CONSTRAINT_NAME
WHERE c.CONSTRAINT_TYPE='P'
  AND cc.OWNER=:owner AND cc.TABLE_NAME=:table_name
ORDER BY cc.POSITION
""".strip()
        rows = self.query(
            sql,
            params={"owner": owner.upper(), "table_name": table.upper()},
            result="records",
        )
        normalized = [{str(key).lower(): value for key, value in row.items()} for row in rows]
        return tuple(
            str(row["column_name"])
            for row in sorted(normalized, key=lambda row: int(row.get("position") or 0))
            if row.get("column_name") is not None
        )

    def resolve_key_columns(
        self,
        table_name: str,
        key_columns: Optional[Sequence[str]],
        first_batch: pd.DataFrame,
        *,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Sequence[str]]:
        del first_batch, dialect_options
        if key_columns is not None:
            return tuple(key_columns)
        keys = self.get_key_columns(table_name)
        if not keys:
            raise DatabaseCapabilityError(f"Oracle 目标表 {table_name!r} 没有主键，无法保证 a/r 冲突语义")
        return keys

    def build_merge_sql(
        self,
        table_name: str,
        columns: Sequence[str],
        key_columns: Sequence[str],
        *,
        mode: str,
    ) -> str:
        """生成逐行绑定、可由 executemany 批量执行的 MERGE SQL。"""

        if mode not in {"a", "r"}:
            raise ValidationError("Oracle MERGE 只支持 a/r 模式")
        if not key_columns:
            raise DatabaseCapabilityError("Oracle MERGE 必须指定主键字段")
        source_columns = ", ".join(
            f":{index} AS {self.quote_identifier(str(column))}" for index, column in enumerate(columns, start=1)
        )
        on_clause = " AND ".join(
            f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
            for column in key_columns
        )
        sql = (
            f"MERGE INTO {self.quote_qualified_name(table_name)} target "
            f"USING (SELECT {source_columns} FROM dual) source ON ({on_clause})"
        )
        if mode == "r":
            keys = set(key_columns)
            update_columns = [column for column in columns if column not in keys]
            if update_columns:
                assignments = ", ".join(
                    f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
                    for column in update_columns
                )
                sql += f" WHEN MATCHED THEN UPDATE SET {assignments}"
        insert_columns = ", ".join(self.quote_identifier(str(column)) for column in columns)
        insert_values = ", ".join(f"source.{self.quote_identifier(str(column))}" for column in columns)
        sql += f" WHEN NOT MATCHED THEN INSERT ({insert_columns}) " f"VALUES ({insert_values})"
        return sql

    @staticmethod
    def _oracle_error_code(error: BaseException) -> Optional[int]:
        current: Optional[BaseException] = error
        while current is not None:
            if getattr(current, "args", None):
                first = current.args[0]
                code = getattr(first, "code", None)
                if code is not None:
                    return int(code)
            current = current.__cause__
        return None

    def is_table_already_exists_error(self, exc: BaseException) -> bool:
        """Oracle ORA-00955 表示对象名已经存在。"""

        return self._oracle_error_code(exc) == 955

    def prepare_write(
        self,
        table_name: str,
        mode: str,
        first_batch: pd.DataFrame,
        *,
        key_columns: Optional[Sequence[str]] = None,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.validate_write(
            table_name,
            mode,
            first_batch,
            key_columns=key_columns,
            dialect_options=dialect_options,
        )
        quoted = self.quote_qualified_name(table_name)
        if mode == "o":
            self.execute(f"TRUNCATE TABLE {quoted}")
        elif mode == "d":
            options = dict(dialect_options or {})
            if key_columns:
                options["key_columns"] = list(key_columns)
            self.build_create_table_sql(first_batch, table_name, options)
            try:
                self.execute(f"DROP TABLE {quoted} PURGE")
            except DatabaseQueryError as exc:
                if self._oracle_error_code(exc) != 942:
                    raise
            self.create_table(first_batch, table_name, dialect_options=options)

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
        """无副作用校验 Oracle 写入模式和主键。"""

        del first_batch, dialect_options, table_exists
        self.require_write_mode(table_name, mode)
        if mode in {"a", "r"} and not key_columns:
            raise DatabaseCapabilityError(f"Oracle 目标表 {table_name!r} 未解析到主键，无法执行模式 {mode!r}")

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

    def _build_insert_sql(self, table_name: str, columns: Sequence[str]) -> str:
        quoted_columns = ", ".join(self.quote_identifier(str(column)) for column in columns)
        placeholders = ", ".join(f":{index}" for index in range(1, len(columns) + 1))
        return f"INSERT INTO {self.quote_qualified_name(table_name)} ({quoted_columns}) " f"VALUES ({placeholders})"

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
            self.build_merge_sql(table_name, columns, key_columns or (), mode=mode)
            if mode in {"a", "r"}
            else self._build_insert_sql(table_name, columns)
        )
        values = [tuple(self._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
        affected = self.executemany(sql, values)
        if mode == "a":
            inserted = max(min(int(affected), len(batch)), 0)
            return BatchWriteResult(inserted=inserted, updated=0, skipped=len(batch) - inserted)
        if mode == "r":
            return BatchWriteResult()
        return BatchWriteResult(inserted=len(batch), updated=0, skipped=0)

    @staticmethod
    def _metadata_filters(
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not targets:
            return "c.OWNER NOT IN ('SYS','SYSTEM')", {}
        clauses = []
        params: Dict[str, Any] = {}
        for index, target in enumerate(targets):
            owner_key = f"owner_{index}"
            params[owner_key] = target.parts[-2].upper() if len(target.parts) >= 2 else target.parts[0].upper()
            if len(target.parts) == 1:
                clauses.append(f"c.OWNER=:{owner_key}")
            else:
                table_key = f"table_{index}"
                params[table_key] = target.parts[-1].upper()
                clauses.append(f"(c.OWNER=:{owner_key} AND c.TABLE_NAME=:{table_key})")
        return "(" + " OR ".join(clauses) + ")", params

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        where_sql, params = self._metadata_filters(targets)
        sql = f"""
SELECT
    c.OWNER,
    c.TABLE_NAME,
    'TABLE' AS TABLE_TYPE,
    tc.COMMENTS AS TABLE_COMMENT,
    c.COLUMN_NAME,
    c.COLUMN_ID,
    c.DATA_TYPE,
    CASE
      WHEN c.DATA_TYPE='NUMBER' AND c.DATA_PRECISION IS NOT NULL
        THEN 'NUMBER(' || c.DATA_PRECISION || ',' || NVL(c.DATA_SCALE, 0) || ')'
      WHEN c.CHAR_LENGTH IS NOT NULL
        THEN c.DATA_TYPE || '(' || c.CHAR_LENGTH || ')'
      ELSE c.DATA_TYPE
    END AS FULL_DATA_TYPE,
    c.NULLABLE,
    c.DATA_DEFAULT,
    kc.CONSTRAINT_TYPE,
    cc.COMMENTS AS COLUMN_COMMENT
FROM ALL_TAB_COLUMNS c
LEFT JOIN ALL_TAB_COMMENTS tc
  ON tc.OWNER=c.OWNER AND tc.TABLE_NAME=c.TABLE_NAME
LEFT JOIN ALL_COL_COMMENTS cc
  ON cc.OWNER=c.OWNER AND cc.TABLE_NAME=c.TABLE_NAME AND cc.COLUMN_NAME=c.COLUMN_NAME
LEFT JOIN (
  SELECT cons.OWNER, cols.TABLE_NAME, cols.COLUMN_NAME, cons.CONSTRAINT_TYPE
  FROM ALL_CONSTRAINTS cons
  JOIN ALL_CONS_COLUMNS cols
    ON cols.OWNER=cons.OWNER AND cols.CONSTRAINT_NAME=cons.CONSTRAINT_NAME
  WHERE cons.CONSTRAINT_TYPE IN ('P','U')
) kc ON kc.OWNER=c.OWNER AND kc.TABLE_NAME=c.TABLE_NAME AND kc.COLUMN_NAME=c.COLUMN_NAME
WHERE {where_sql}
ORDER BY c.OWNER, c.TABLE_NAME, c.COLUMN_ID
""".strip()
        rows = self.query(sql, params=params or None, result="records")
        normalized_rows = []
        for raw in rows:
            row = {str(key).lower(): value for key, value in raw.items()}
            owner = row.get("owner")
            table = row.get("table_name")
            constraint = row.get("constraint_type")
            normalized_rows.append(
                {
                    "database_type": self.database_type,
                    "catalog": None,
                    "database": None,
                    "schema": owner,
                    "table_name": table,
                    "qualified_name": f"{owner}.{table}" if owner and table else None,
                    "table_type": row.get("table_type"),
                    "table_comment": row.get("table_comment"),
                    "table_engine": None,
                    "column_name": row.get("column_name"),
                    "ordinal_position": row.get("column_id"),
                    "data_type": row.get("data_type"),
                    "full_data_type": row.get("full_data_type"),
                    "pandas_dtype": None,
                    "nullable": row.get("nullable"),
                    "default_value": row.get("data_default"),
                    "primary_key": constraint == "P",
                    "unique_key": constraint,
                    "partition_key": None,
                    "sort_key": None,
                    "bucket_key": None,
                    "column_comment": row.get("column_comment"),
                }
            )
        return MetadataInspection(rows=normalized_rows, errors=[])


__all__ = ["OracleAdapter"]
