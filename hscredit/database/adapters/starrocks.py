"""StarRocks 数据库适配器。

普通 SQL 和流式读取复用 MySQL 协议，可选 Stream Load 提供大批量 UTF-8 CSV 写入。
"""

import base64
import json
import uuid
from collections.abc import MutableMapping
from typing import Any, List, Mapping, Optional, Sequence, Tuple
from urllib import request as urllib_request

import pandas as pd
from pandas.api import types as ptypes

from ...exceptions import DependencyError, InputValidationError, ValidationError
from ..exceptions import DatabaseCapabilityError, DatabaseQueryError, DatabaseWriteError
from ..metadata import MetadataInspection, QualifiedTarget
from ..type_inference import profile_string_series, resolve_bounded_string_length
from ..types import DatabaseCapabilities, PoolOptions
from ..writing import BatchWriteResult, resolve_column_type, validate_column_mapping_keys
from .mysql import MySQLAdapter

_RESOLVED_JSON_COLUMNS = "_hscredit_resolved_json_columns"


class StarRocksAdapter(MySQLAdapter):
    """StarRocks MySQL 协议与 Stream Load 适配器。"""

    database_type = "starrocks"

    def json_extract_expression(self, column_sql: str, path: str) -> str:
        """使用 StarRocks ``GET_JSON_STRING`` 提取 JSON 路径。"""

        return f"GET_JSON_STRING({column_sql}, '{path}')"

    def __init__(
        self,
        *,
        connect_kwargs: Mapping[str, Any],
        pool_options: PoolOptions,
        adapter_options: Optional[Mapping[str, Any]] = None,
    ):
        resolved = dict(connect_kwargs)
        resolved.setdefault("port", 9_030)
        super().__init__(
            connect_kwargs=resolved,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )

    def load_driver(self) -> Any:
        try:
            import pymysql
        except ImportError as exc:
            raise DependencyError("缺少 StarRocks 可选依赖，请安装: pip install hscredit[db-starrocks]") from exc
        return pymysql

    @staticmethod
    def _normalize_table_model(value: Any) -> str:
        normalized = str(value or "UNKNOWN").strip().upper().replace("_", " ")
        aliases = {
            "PRIMARY": "PRIMARY KEY",
            "UNIQUE": "UNIQUE KEY",
            "DUPLICATE": "DUPLICATE KEY",
            "AGGREGATE": "AGGREGATE KEY",
        }
        return aliases.get(normalized, normalized)

    def get_table_model(self, table_name: str) -> str:
        rows = self.query(
            f"SHOW CREATE TABLE {self.quote_qualified_name(table_name)}",
            result="records",
        )
        text = " ".join(str(value) for row in rows for value in row.values()).upper()
        for model in ("PRIMARY KEY", "UNIQUE KEY", "DUPLICATE KEY", "AGGREGATE KEY"):
            if model in text:
                return model
        return "UNKNOWN"

    def _resolve_write_model(
        self,
        table_name: str,
        options: Mapping[str, Any],
        *,
        table_exists: Optional[bool] = None,
        allow_missing_default: bool = False,
    ) -> str:
        configured = options.get("table_model") or options.get("model")
        if configured is not None:
            return self._normalize_table_model(configured)
        if table_exists is False:
            return "DUPLICATE KEY"
        try:
            return self._normalize_table_model(self.get_table_model(table_name))
        except DatabaseQueryError:
            if allow_missing_default:
                return "DUPLICATE KEY"
            raise

    def capabilities_for_table(
        self,
        table_name: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        del table_name
        model = self._normalize_table_model(
            (table_metadata or {}).get("table_model") or (table_metadata or {}).get("model")
        )
        modes = {"o", "d"}
        if model in {"PRIMARY KEY", "UNIQUE KEY"}:
            modes.add("r")
        return DatabaseCapabilities(
            transactions=True,
            streaming_read=True,
            native_bulk_write=True,
            metadata_export=True,
            write_modes=modes,
        )

    @staticmethod
    def _column_type(
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

        resolved_options = dict(options or {})
        profile = profile_string_series(series)
        if profile.all_json_documents and resolved_options.get("infer_json", True):
            return "JSON"
        if not profile.all_strings or profile.non_null_count == 0:
            return "VARCHAR(65533)"
        if profile.max_utf8_bytes > 65_533:
            raise InputValidationError(
                "StarRocks 单个字符串字段超过 VARCHAR/STRING 上限 65533 字节，请拆分字段或改用外部存储"
            )
        length = resolve_bounded_string_length(
            profile.max_utf8_bytes,
            maximum=65_533,
        )
        return f"VARCHAR({length})"

    @staticmethod
    def _quote_double(value: Any) -> str:
        return '"' + str(value).replace('"', '\\"') + '"'

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        options = dict(dialect_options or {})
        model = self._normalize_table_model(options.get("table_model") or options.get("model") or "DUPLICATE KEY")
        if model not in {"PRIMARY KEY", "UNIQUE KEY", "DUPLICATE KEY", "AGGREGATE KEY"}:
            raise ValidationError(f"不支持的 StarRocks 表模型: {model!r}")
        key_value = options.get("key_columns") or options.get("primary_key") or ()
        keys = (key_value,) if isinstance(key_value, str) else tuple(key_value)
        if not keys and model == "DUPLICATE KEY" and len(data.columns):
            keys = (str(data.columns[0]),)
        if not keys:
            raise DatabaseCapabilityError(f"StarRocks {model} 建表必须指定 key_columns")
        missing = [key for key in keys if key not in data.columns]
        if missing:
            raise InputValidationError(f"StarRocks 建表数据缺少键字段: {missing}")

        types = dict(options.get("column_types") or {})
        comments = dict(options.get("column_comments") or {})
        validate_column_mapping_keys(
            types,
            data.columns,
            option_name="column_types",
            database_type="StarRocks",
        )
        validate_column_mapping_keys(
            comments,
            data.columns,
            option_name="column_comments",
            database_type="StarRocks",
        )
        definitions = []
        for column in data.columns:
            column_type = resolve_column_type(
                types,
                column,
                self._column_type(data[column], options),
                database_type="StarRocks",
            )
            definition = f"{self.quote_identifier(str(column))} {column_type}"
            if column in keys and model == "PRIMARY KEY":
                definition += " NOT NULL"
            if column in comments:
                definition += f" COMMENT {self._quote_double(comments[column])}"
            definitions.append(definition)

        if_not_exists = " IF NOT EXISTS" if options.get("if_not_exists", True) else ""
        quoted_keys = ", ".join(self.quote_identifier(str(column)) for column in keys)
        ddl = (
            f"CREATE TABLE{if_not_exists} {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definitions)
            + f"\n) ENGINE=OLAP\n{model} ({quoted_keys})"
        )
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            ddl += f"\nCOMMENT {self._quote_double(table_comment)}"
        distribution = options.get("distribution_columns") or keys
        quoted_distribution = ", ".join(self.quote_identifier(str(column)) for column in distribution)
        ddl += f"\nDISTRIBUTED BY HASH({quoted_distribution})"
        buckets = options.get("buckets")
        if buckets is not None:
            if isinstance(buckets, bool) or not isinstance(buckets, int) or buckets <= 0:
                raise ValidationError("StarRocks buckets 必须是正整数")
            ddl += f" BUCKETS {buckets}"
        properties = dict(options.get("properties") or {})
        if properties:
            items = ", ".join(
                f"{self._quote_double(key)}={self._quote_double(value)}" for key, value in properties.items()
            )
            ddl += f"\nPROPERTIES ({items})"
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

    def _infer_json_columns(
        self,
        data: pd.DataFrame,
        options: Mapping[str, Any],
    ) -> Tuple[str, ...]:
        explicit_types = dict(options.get("column_types") or {})
        return tuple(
            str(column)
            for column in data.columns
            if resolve_column_type(
                explicit_types,
                column,
                self._column_type(data[column], options),
                database_type="StarRocks",
            ).upper()
            == "JSON"
        )

    def get_json_columns(self, table_name: str) -> Tuple[str, ...]:
        schema, table = self._schema_and_table(table_name)
        rows = self.query(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND UPPER(DATA_TYPE)='JSON' "
            "ORDER BY ORDINAL_POSITION",
            params=(schema, table),
            result="records",
        )
        return tuple(
            str(row.get("COLUMN_NAME", row.get("column_name")))
            for row in rows
            if row.get("COLUMN_NAME", row.get("column_name")) is not None
        )

    def get_key_columns(self, table_name: str) -> Tuple[str, ...]:
        schema, table = self._schema_and_table(table_name)
        sql = """
SELECT COLUMN_NAME, ORDINAL_POSITION
FROM information_schema.columns
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_KEY IN ('PRI','UNI')
ORDER BY ORDINAL_POSITION
""".strip()
        rows = self.query(sql, params=(schema, table), result="records")
        normalized = [{str(key).lower(): value for key, value in row.items()} for row in rows]
        return tuple(
            str(row["column_name"])
            for row in sorted(
                normalized,
                key=lambda item: int(item.get("ordinal_position") or 0),
            )
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
        del first_batch
        if key_columns is not None:
            return tuple(key_columns)
        options = dict(dialect_options or {})
        model = self._normalize_table_model(
            options.get("table_model") or options.get("model") or self.get_table_model(table_name)
        )
        if model in {"PRIMARY KEY", "UNIQUE KEY"}:
            keys = self.get_key_columns(table_name)
            if not keys:
                raise DatabaseCapabilityError(f"StarRocks {model} 表 {table_name!r} 未发现键字段")
            return keys
        return None

    def build_insert_sql(
        self,
        table_name: str,
        columns: Sequence[str],
        *,
        mode: str,
        key_columns: Optional[Sequence[str]] = None,
        json_columns: Optional[Sequence[str]] = None,
    ) -> str:
        del mode, key_columns
        quoted_columns = ", ".join(self.quote_identifier(str(column)) for column in columns)
        json_names = set(json_columns or ())
        placeholders = ", ".join("parse_json(%s)" if column in json_names else "%s" for column in columns)
        return f"INSERT INTO {self.quote_qualified_name(table_name)} ({quoted_columns}) " f"VALUES ({placeholders})"

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
        model = self._resolve_write_model(
            table_name,
            options,
            allow_missing_default=mode == "d",
        )
        self.validate_write(
            table_name,
            mode,
            first_batch,
            key_columns=key_columns,
            dialect_options={**options, "table_model": model},
        )
        quoted = self.quote_qualified_name(table_name)
        if mode == "o":
            json_columns = self.get_json_columns(table_name)
            self.execute(f"TRUNCATE TABLE {quoted}")
        elif mode == "d":
            options["table_model"] = model
            if key_columns:
                options["key_columns"] = list(key_columns)
            options["if_not_exists"] = False
            ddl = self.build_create_table_sql(first_batch, table_name, options)
            json_columns = self._infer_json_columns(first_batch, options)
            self.execute(f"DROP TABLE IF EXISTS {quoted}")
            self.execute(ddl)
        else:
            json_columns = self.get_json_columns(table_name)
        if isinstance(dialect_options, MutableMapping):
            dialect_options[_RESOLVED_JSON_COLUMNS] = tuple(json_columns)

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
        """无副作用校验 StarRocks 表模型和冲突语义。"""

        del first_batch
        options = dict(dialect_options or {})
        model = self._resolve_write_model(
            table_name,
            options,
            table_exists=table_exists,
        )
        capabilities = self.capabilities_for_table(
            table_name,
            {"table_model": model},
        )
        if mode not in capabilities.write_modes:
            raise DatabaseCapabilityError(
                f"StarRocks {model} 表 {table_name!r} 不支持模式 {mode!r}，"
                f"当前可用模式: {sorted(capabilities.write_modes)}"
            )
        if mode == "r" and not key_columns:
            raise DatabaseCapabilityError(f"StarRocks {model} 覆盖写入必须指定键字段")

    def _stream_load_request(
        self,
        table_name: str,
        batch: pd.DataFrame,
        batch_index: int,
        options: Mapping[str, Any],
    ) -> urllib_request.Request:
        schema, table = self._schema_and_table(table_name)
        host = self.connect_kwargs.get("host")
        if not host:
            raise ValidationError("StarRocks Stream Load 必须配置 host")
        secure = bool(self.adapter_options.get("secure", False))
        scheme = "https" if secure else "http"
        port = int(self.adapter_options.get("http_port", 8_030))
        url = options.get("stream_load_url") or (f"{scheme}://{host}:{port}/api/{schema}/{table}/_stream_load")
        data = batch.to_csv(
            index=False,
            header=False,
            na_rep="\\N",
            lineterminator="\n",
        ).encode("utf-8")
        label = str(options.get("label") or f"hscredit_{uuid.uuid4().hex}_{batch_index}")
        headers = {
            "Label": label,
            "Format": "csv",
            "Column_separator": ",",
            "Columns": ",".join(str(column) for column in batch.columns),
            "Expect": "100-continue",
        }
        user = self.connect_kwargs.get("user")
        password = self.connect_kwargs.get("password") or ""
        if user is not None:
            token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return urllib_request.Request(
            str(url),
            data=data,
            headers=headers,
            method="PUT",
        )

    def _send_stream_load(self, request: urllib_request.Request) -> Mapping[str, Any]:
        timeout = float(self.adapter_options.get("http_timeout", 300))
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

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
        options = dict(dialect_options or {})
        if options.get("stream_load"):
            request = self._stream_load_request(
                table_name,
                batch,
                batch_index,
                options,
            )
            try:
                payload = self._send_stream_load(request)
            except Exception as exc:
                raise DatabaseWriteError("StarRocks Stream Load 请求失败") from exc
            status = payload.get("Status")
            if status != "Success":
                message = payload.get("Message") or payload.get("ErrorURL") or status
                raise DatabaseWriteError(f"StarRocks Stream Load 失败: {message}")
            loaded = payload.get("NumberLoadedRows")
            inserted = int(loaded) if loaded is not None else None
            if mode == "r":
                return BatchWriteResult()
            return BatchWriteResult(
                inserted=inserted,
                updated=0,
                skipped=(max(len(batch) - inserted, 0) if inserted is not None else None),
            )

        if _RESOLVED_JSON_COLUMNS in options:
            json_columns = tuple(options[_RESOLVED_JSON_COLUMNS])
        else:
            json_columns = self.get_json_columns(table_name)
            if not json_columns:
                json_columns = self._infer_json_columns(batch, options)
        sql = self.build_insert_sql(
            table_name,
            [str(column) for column in batch.columns],
            mode=mode,
            key_columns=key_columns,
            json_columns=json_columns,
        )
        values = [tuple(self._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
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
            return "c.TABLE_SCHEMA NOT IN ('information_schema','sys')", ()
        clauses = []
        params: List[Any] = []
        for target in targets:
            if len(target.parts) == 1:
                clauses.append("c.TABLE_SCHEMA=%s")
                params.append(target.parts[0])
            elif len(target.parts) == 2:
                clauses.append("(c.TABLE_SCHEMA=%s AND c.TABLE_NAME=%s)")
                params.extend(target.parts)
            else:
                clauses.append("(c.TABLE_CATALOG=%s AND c.TABLE_SCHEMA=%s AND c.TABLE_NAME=%s)")
                params.extend(target.parts[-3:])
        return "(" + " OR ".join(clauses) + ")", tuple(params)

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        where_sql, params = self._metadata_filters(targets)
        sql = f"""
SELECT
  c.TABLE_CATALOG AS table_catalog,
  c.TABLE_SCHEMA AS table_schema,
  c.TABLE_NAME AS table_name,
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
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.TABLE_CATALOG=c.TABLE_CATALOG
 AND t.TABLE_SCHEMA=c.TABLE_SCHEMA
 AND t.TABLE_NAME=c.TABLE_NAME
WHERE {where_sql}
ORDER BY c.TABLE_CATALOG, c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
""".strip()
        rows = self.query(sql, params=params or None, result="records")
        normalized = []
        for raw in rows:
            row = {str(key).lower(): value for key, value in raw.items()}
            catalog = row.get("table_catalog")
            database_name = row.get("table_schema")
            table = row.get("table_name")
            column_key = row.get("column_key")
            qualified_parts = [part for part in (catalog, database_name, table) if part]
            normalized.append(
                {
                    "database_type": self.database_type,
                    "catalog": catalog,
                    "database": database_name,
                    "schema": None,
                    "table_name": table,
                    "qualified_name": ".".join(map(str, qualified_parts)) or None,
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


__all__ = ["StarRocksAdapter"]
