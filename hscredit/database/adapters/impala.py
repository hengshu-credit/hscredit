"""Impala 数据库适配器。

复用 Impyla/DBUtils 查询能力，并按 Kudu 与文件表模型严格声明写入语义。
"""

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..exceptions import DatabaseCapabilityError
from ..types import DatabaseCapabilities, PoolOptions
from ..writing import BatchWriteResult
from .hive import HiveAdapter, _SAFE_STORAGE


class ImpalaAdapter(HiveAdapter):
    """Impala/Impyla 数据库适配器。"""

    database_type = "impala"
    default_port = 21_050
    default_auth_mechanism = "NOSASL"

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

    @staticmethod
    def _storage(options: Optional[Mapping[str, Any]]) -> str:
        return str((options or {}).get("storage", "PARQUET")).upper()

    def capabilities_for_table(
        self,
        table_name: str,
        table_metadata: Optional[Mapping[str, Any]] = None,
    ) -> DatabaseCapabilities:
        del table_name
        storage = self._storage(table_metadata)
        modes = {"a", "o", "d"}
        if storage == "KUDU":
            modes.add("r")
        return DatabaseCapabilities(
            transactions=False,
            streaming_read=True,
            native_bulk_write=False,
            metadata_export=True,
            write_modes=modes,
        )

    def build_create_table_sql(
        self,
        data: pd.DataFrame,
        table_name: str,
        dialect_options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        options = dict(dialect_options or {})
        storage = self._storage(options)
        if storage != "KUDU":
            return super().build_create_table_sql(data, table_name, options)
        if not _SAFE_STORAGE.fullmatch(storage):
            raise DatabaseCapabilityError(f"Impala storage 参数无效: {storage!r}")
        key_value = options.get("key_columns") or options.get("primary_key") or ()
        keys = (key_value,) if isinstance(key_value, str) else tuple(key_value)
        if not keys:
            raise DatabaseCapabilityError("Impala Kudu 建表必须指定 key_columns")
        definitions = self._column_definitions(data, options, inline_primary=True)
        partitions = int(options.get("partitions", 3))
        quoted_keys = ", ".join(self.quote_identifier(str(column)) for column in keys)
        if_not_exists = " IF NOT EXISTS" if options.get("if_not_exists", True) else ""
        return (
            f"CREATE TABLE{if_not_exists} {self.quote_qualified_name(table_name)} (\n  "
            + ",\n  ".join(definitions)
            + f"\n) PARTITION BY HASH ({quoted_keys}) PARTITIONS {partitions}\nSTORED AS KUDU"
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
        storage = self._storage(options)
        capabilities = self.capabilities_for_table(table_name, options)
        if mode not in capabilities.write_modes:
            raise DatabaseCapabilityError(f"Impala 目标表 {table_name!r} 不是 Kudu 表，无法执行 r/UPSERT 模式")
        if mode == "a" and key_columns and storage != "KUDU":
            raise DatabaseCapabilityError(f"Impala 非 Kudu 表 {table_name!r} 无法保证主键冲突忽略语义")
        if mode == "r" and not key_columns:
            raise DatabaseCapabilityError("Impala Kudu UPSERT 必须指定 key_columns")
        quoted = self.quote_qualified_name(table_name)
        if mode == "o":
            command = f"DELETE FROM {quoted}" if storage == "KUDU" else f"TRUNCATE TABLE {quoted}"
            self.execute(command)
        elif mode == "d":
            if key_columns:
                options["key_columns"] = list(key_columns)
            options["if_not_exists"] = False
            ddl = self.build_create_table_sql(first_batch, table_name, options)
            self.execute(f"DROP TABLE IF EXISTS {quoted}")
            self.execute(ddl)

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
        storage = self._storage(dialect_options)
        columns = [str(column) for column in batch.columns]
        prefix = "UPSERT INTO" if mode == "r" and storage == "KUDU" else "INSERT INTO"
        sql = self._insert_sql(table_name, columns, prefix=prefix)
        values = [tuple(self._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
        affected = self.executemany(sql, values)
        if mode == "r":
            return BatchWriteResult()
        inserted = int(affected) if int(affected) >= 0 else None
        skipped = len(batch) - inserted if storage == "KUDU" and inserted is not None else 0
        return BatchWriteResult(inserted=inserted, updated=0, skipped=skipped)


__all__ = ["ImpalaAdapter"]
