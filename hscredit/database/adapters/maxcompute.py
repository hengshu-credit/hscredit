"""MaxCompute 数据库适配器。

普通 SQL 使用 PyODPS DB-API，表写入和元数据使用原生 ODPS 入口。
"""

import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api import types as ptypes

from ...exceptions import DependencyError, InputValidationError, ValidationError
from ..exceptions import (
    DatabaseCapabilityError,
    DatabaseMetadataError,
    DatabaseQueryError,
    DatabaseWriteError,
)
from ..metadata import MetadataInspection, QualifiedTarget
from ..types import DatabaseCapabilities, PoolOptions
from ..writing import (
    BatchWriteResult,
    split_qualified_name,
    validate_column_mapping_keys,
    validate_sql_type,
)
from .dbapi import DBAPIAdapter


class MaxComputeAdapter(DBAPIAdapter):
    """MaxCompute/PyODPS 数据库适配器。"""

    database_type = "maxcompute"
    identifier_quote = "`"
    capabilities = DatabaseCapabilities(
        transactions=False,
        streaming_read=True,
        native_bulk_write=True,
        metadata_export=True,
        write_modes={"a", "o", "d"},
    )

    def __init__(
        self,
        *,
        connect_kwargs: Mapping[str, Any],
        pool_options: PoolOptions,
        adapter_options: Optional[Mapping[str, Any]] = None,
    ):
        self.odps_module = self.load_odps_module()
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.odps = self._create_odps_entry()

    def load_odps_module(self) -> Any:
        try:
            import odps
            import odps.dbapi
        except ImportError as exc:
            raise DependencyError("缺少 MaxCompute 可选依赖，请安装: pip install hscredit[db-maxcompute]") from exc
        odps.dbapi = odps.dbapi
        return odps

    def load_driver(self) -> Any:
        return self.odps_module.dbapi

    def _create_odps_entry(self) -> Any:
        values = dict(self.connect_kwargs)
        access_id = values.pop("access_id", None)
        secret = values.pop("access_key", values.pop("secret_access_key", None))
        allowed = {
            "project",
            "endpoint",
            "schema",
            "quota_name",
            "tunnel_endpoint",
            "region_name",
            "logview_host",
            "catalog_endpoint",
            "account",
            "app_account",
        }
        kwargs = {key: value for key, value in values.items() if key in allowed and value is not None}
        if access_id is not None:
            kwargs["access_id"] = access_id
        if secret is not None:
            kwargs["secret_access_key"] = secret
        try:
            return self.odps_module.ODPS(**kwargs)
        except Exception as exc:
            from ..exceptions import DatabaseConnectionError

            raise DatabaseConnectionError("创建 MaxCompute ODPS 入口失败") from exc

    def execute(self, sql: str, params: Any = None) -> int:
        """执行无事务的 MaxCompute SQL。"""

        try:
            with self.connection_cursor() as (_, cursor):
                self.execute_cursor(cursor, sql, params)
                return int(getattr(cursor, "rowcount", -1))
        except Exception as exc:
            if isinstance(exc, DatabaseQueryError):
                raise
            raise DatabaseQueryError("MaxCompute SQL执行失败") from exc

    def executemany(self, sql: str, values: Any) -> int:
        materialized = list(values)
        try:
            with self.connection_cursor() as (_, cursor):
                cursor.executemany(sql, materialized)
                return int(getattr(cursor, "rowcount", -1))
        except Exception as exc:
            raise DatabaseQueryError("MaxCompute SQL批量执行失败") from exc

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
            return "DATETIME"
        return "STRING"

    @staticmethod
    def _quote_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        options = dict(dialect_options or {})
        types = dict(options.get("column_types") or {})
        comments = dict(options.get("column_comments") or {})
        validate_column_mapping_keys(
            types,
            data.columns,
            option_name="column_types",
            database_type="MaxCompute",
        )
        validate_column_mapping_keys(
            comments,
            data.columns,
            option_name="column_comments",
            database_type="MaxCompute",
        )
        partition_value = options.get("partition_columns") or ()
        partition_columns = (partition_value,) if isinstance(partition_value, str) else tuple(partition_value)
        missing = [column for column in partition_columns if column not in data.columns]
        if missing:
            raise InputValidationError(f"MaxCompute 数据缺少分区字段: {missing}")

        def definition(column: Any) -> str:
            column_type = validate_sql_type(
                types.get(column) or self._column_type(data[column]),
                database_type="MaxCompute",
            )
            result = f"{self.quote_identifier(str(column))} {column_type}"
            if column in comments:
                result += f" COMMENT {self._quote_literal(comments[column])}"
            return result

        regular = [column for column in data.columns if column not in partition_columns]
        if not regular:
            raise InputValidationError("MaxCompute 表必须至少包含一个非分区字段")
        ddl = (
            f"CREATE TABLE {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definition(column) for column in regular)
            + "\n)"
        )
        table_comment = options.get("table_comment") or options.get("comment")
        if table_comment is not None:
            ddl += f"\nCOMMENT {self._quote_literal(table_comment)}"
        if partition_columns:
            ddl += "\nPARTITIONED BY (\n  " + ",\n  ".join(definition(column) for column in partition_columns) + "\n)"
        lifecycle = options.get("lifecycle")
        if lifecycle is not None:
            if isinstance(lifecycle, bool) or not isinstance(lifecycle, int) or lifecycle <= 0:
                raise ValidationError("MaxCompute lifecycle 必须是正整数")
            ddl += f"\nLIFECYCLE {lifecycle}"
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
        del table_name, first_batch
        if key_columns is not None:
            return tuple(key_columns)
        option_keys = (dialect_options or {}).get("key_columns")
        if option_keys is None:
            return None
        return (option_keys,) if isinstance(option_keys, str) else tuple(option_keys)

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
            raise DatabaseCapabilityError(f"MaxCompute 目标表 {table_name!r} 不是支持 MERGE 的事务表")
        if mode == "a" and key_columns:
            raise DatabaseCapabilityError(f"MaxCompute 目标表 {table_name!r} 无法原生保证主键冲突忽略语义")
        if mode == "r" and not key_columns:
            raise DatabaseCapabilityError("MaxCompute MERGE 必须指定 key_columns")
        if mode == "d":
            ddl = self.build_create_table_sql(first_batch, table_name, options)
            try:
                self.odps.delete_table(table_name, if_exists=True)
            except Exception as exc:
                raise DatabaseWriteError("删除 MaxCompute 目标表失败") from exc
            self.execute(ddl)

    def _merge_batch(
        self,
        table_name: str,
        batch: pd.DataFrame,
        batch_index: int,
        key_columns: Sequence[str],
        options: Mapping[str, Any],
    ) -> None:
        parts = split_qualified_name(table_name)
        staging_leaf = f"{parts[-1]}__hscredit_staging_{uuid.uuid4().hex[:12]}_{batch_index}"
        staging_name = ".".join(parts[:-1] + (staging_leaf,))
        try:
            self.odps.write_table(
                staging_name,
                batch,
                create_table=True,
                overwrite=True,
                table_kwargs={"lifecycle": int(options.get("staging_lifecycle", 1))},
            )
            keys = set(key_columns)
            condition = " AND ".join(
                f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
                for column in key_columns
            )
            updates = ", ".join(
                f"target.{self.quote_identifier(str(column))}=" f"source.{self.quote_identifier(str(column))}"
                for column in batch.columns
                if column not in keys
            )
            columns = ", ".join(self.quote_identifier(str(column)) for column in batch.columns)
            values = ", ".join(f"source.{self.quote_identifier(str(column))}" for column in batch.columns)
            sql = (
                f"MERGE INTO {self.quote_qualified_name(table_name)} target "
                f"USING {self.quote_qualified_name(staging_name)} source "
                f"ON ({condition})"
            )
            if updates:
                sql += f" WHEN MATCHED THEN UPDATE SET {updates}"
            sql += f" WHEN NOT MATCHED THEN INSERT ({columns}) VALUES ({values})"
            self.execute(sql)
        finally:
            self.odps.delete_table(staging_name, if_exists=True)

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
        try:
            if mode == "r":
                self._merge_batch(
                    table_name,
                    batch,
                    batch_index,
                    key_columns or (),
                    options,
                )
                return BatchWriteResult()
            self.odps.write_table(
                table_name,
                batch,
                overwrite=mode == "o" and batch_index == 1,
                create_table=False,
            )
        except DatabaseWriteError:
            raise
        except Exception as exc:
            raise DatabaseWriteError("MaxCompute 原生批量写入失败") from exc
        return BatchWriteResult(inserted=None, updated=0, skipped=0)

    @staticmethod
    def _table_identity(
        table: Any,
        default_project: Optional[str],
        default_schema: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], str]:
        project = getattr(table, "project", None) or default_project
        project = getattr(project, "name", project)
        schema = getattr(table, "schema_name", None) or default_schema
        return (
            str(project) if project is not None else None,
            str(schema) if schema is not None else None,
            str(table.name),
        )

    def _table_metadata_rows(
        self,
        table: Any,
        project: Optional[str],
        schema: Optional[str],
    ) -> List[Dict[str, Any]]:
        project_name, schema_name, table_name = self._table_identity(
            table,
            project,
            schema,
        )
        table_schema = table.table_schema
        columns = list(getattr(table_schema, "columns", ()) or ())
        partitions = list(getattr(table_schema, "partitions", ()) or ())
        result = []
        for position, column in enumerate(columns + partitions, start=1):
            is_partition = column in partitions
            qualified = ".".join(part for part in (project_name, schema_name, table_name) if part)
            result.append(
                {
                    "database_type": self.database_type,
                    "catalog": None,
                    "database": project_name,
                    "schema": schema_name,
                    "table_name": table_name,
                    "qualified_name": qualified,
                    "table_type": getattr(table, "type", None),
                    "table_comment": getattr(table, "comment", None),
                    "table_engine": None,
                    "column_name": getattr(column, "name", None),
                    "ordinal_position": position,
                    "data_type": str(getattr(column, "type", "")),
                    "full_data_type": str(getattr(column, "type", "")),
                    "pandas_dtype": None,
                    "nullable": getattr(column, "nullable", None),
                    "default_value": getattr(column, "default", None),
                    "primary_key": None,
                    "unique_key": None,
                    "partition_key": True if is_partition else False,
                    "sort_key": None,
                    "bucket_key": None,
                    "column_comment": getattr(column, "comment", None),
                }
            )
        return result

    def inspect_schema(
        self,
        targets: Optional[Sequence[QualifiedTarget]],
    ) -> MetadataInspection:
        default_project = self.connect_kwargs.get("project")
        default_schema = self.connect_kwargs.get("schema")
        rows: List[Dict[str, Any]] = []
        errors: List[Any] = []
        if not targets:
            targets_to_scan = [(str(default_project) if default_project is not None else None, default_schema, None)]
        else:
            targets_to_scan = []
            for target in targets:
                if len(target.parts) == 1:
                    targets_to_scan.append((target.parts[0], default_schema, None))
                elif len(target.parts) == 2:
                    targets_to_scan.append((target.parts[0], default_schema, target.parts[1]))
                else:
                    targets_to_scan.append((target.parts[-3], target.parts[-2], target.parts[-1]))

        for project, schema, table_name in targets_to_scan:
            if table_name is not None:
                try:
                    table = self.odps.get_table(
                        table_name,
                        project=project,
                        schema=schema,
                    )
                    rows.extend(self._table_metadata_rows(table, project, schema))
                except Exception as exc:
                    raise DatabaseMetadataError(f"读取 MaxCompute 表 {table_name!r} 元数据失败") from exc
                continue
            try:
                tables = self.odps.list_tables(project=project, schema=schema)
                for table in tables:
                    try:
                        rows.extend(self._table_metadata_rows(table, project, schema))
                    except Exception as exc:
                        errors.append(
                            {
                                "目标": getattr(table, "name", None),
                                "错误": str(exc),
                            }
                        )
            except Exception as exc:
                raise DatabaseMetadataError(f"扫描 MaxCompute Project {project!r} 元数据失败") from exc
        return MetadataInspection(rows=rows, errors=errors)


__all__ = ["MaxComputeAdapter"]
