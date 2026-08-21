"""Hive 与 Impala 适配器连接、能力、DDL 和元数据测试。"""

import pandas as pd
import pytest

from hscredit.database import PoolOptions
from hscredit.database.adapters.hive import HiveAdapter, parse_describe_formatted
from hscredit.database.adapters.impala import ImpalaAdapter
from hscredit.database.exceptions import DatabaseCapabilityError
from hscredit.database.metadata import QualifiedTarget

from .fakes import FakeDBAPIDriver, FakeDBAPIState, FakePooledDB


class ObservableHiveAdapter(HiveAdapter):
    def __init__(self, *, connect_kwargs=None, adapter_options=None):
        self.state = FakeDBAPIState()
        super().__init__(
            connect_kwargs=connect_kwargs or {},
            pool_options=PoolOptions(maxconnections=2),
            adapter_options=adapter_options or {},
        )
        self.sql_calls = []
        self.metadata_rows = []

    def load_driver(self):
        return FakeDBAPIDriver(self.state)

    def load_pool_class(self):
        return FakePooledDB

    def execute(self, sql, params=None):
        self.sql_calls.append(("execute", sql, params))
        return 0

    def executemany(self, sql, values):
        materialized = list(values)
        self.sql_calls.append(("executemany", sql, materialized))
        return len(materialized)

    def query(self, sql, params=None, result="dataframe"):
        self.sql_calls.append(("query", sql, params, result))
        return list(self.metadata_rows)


class ObservableImpalaAdapter(ImpalaAdapter):
    def __init__(self, *, connect_kwargs=None, adapter_options=None):
        self.state = FakeDBAPIState()
        super().__init__(
            connect_kwargs=connect_kwargs or {},
            pool_options=PoolOptions(maxconnections=2),
            adapter_options=adapter_options or {},
        )
        self.sql_calls = []
        self.metadata_rows = []

    def load_driver(self):
        return FakeDBAPIDriver(self.state)

    def load_pool_class(self):
        return FakePooledDB

    def execute(self, sql, params=None):
        self.sql_calls.append(("execute", sql, params))
        return 0

    def executemany(self, sql, values):
        materialized = list(values)
        self.sql_calls.append(("executemany", sql, materialized))
        return len(materialized)

    def query(self, sql, params=None, result="dataframe"):
        self.sql_calls.append(("query", sql, params, result))
        return list(self.metadata_rows)


def test_hive_and_impala_apply_distinct_connection_defaults():
    hive = ObservableHiveAdapter(connect_kwargs={"host": "hive.internal"})
    impala = ObservableImpalaAdapter(connect_kwargs={"host": "impala.internal"})

    assert hive.state.pool_kwargs["port"] == 10000
    assert hive.state.pool_kwargs["auth_mechanism"] == "PLAIN"
    assert impala.state.pool_kwargs["port"] == 21050
    assert impala.state.pool_kwargs["auth_mechanism"] == "NOSASL"


def test_impyla_cursor_arraysize_uses_adapter_option():
    adapter = ObservableHiveAdapter(adapter_options={"arraysize": 2048})

    class Cursor:
        arraysize = None

    class Connection:
        def cursor(self):
            return Cursor()

    cursor = adapter.create_cursor(Connection(), stream=True)

    assert cursor.arraysize == 2048


def test_impala_kudu_exposes_conflict_modes():
    adapter = ObservableImpalaAdapter()

    capabilities = adapter.capabilities_for_table(
        "risk.events",
        {"storage": "KUDU"},
    )

    assert {"a", "r", "o", "d"}.issubset(capabilities.write_modes)


def test_impala_parquet_rejects_replace_mode():
    adapter = ObservableImpalaAdapter()

    with pytest.raises(DatabaseCapabilityError, match="Kudu"):
        adapter.prepare_write(
            "risk.events",
            "r",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"storage": "PARQUET"},
        )


def test_hive_non_transactional_table_rejects_replace_mode():
    adapter = ObservableHiveAdapter()

    with pytest.raises(DatabaseCapabilityError, match="事务表"):
        adapter.prepare_write(
            "risk.events",
            "r",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"transactional": False},
        )


def test_hive_default_ddl_is_parquet_and_preserves_comments():
    adapter = ObservableHiveAdapter()
    frame = pd.DataFrame({"id": [1], "name": ["A"]})

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.events",
        {
            "column_comments": {"name": "姓名"},
            "table_comment": "事件表",
        },
    )

    assert "CREATE TABLE IF NOT EXISTS `risk`.`events`" in ddl
    assert "`id` BIGINT" in ddl
    assert "`name` STRING COMMENT '姓名'" in ddl
    assert "COMMENT '事件表'" in ddl
    assert "STORED AS PARQUET" in ddl


def test_impala_kudu_ddl_requires_primary_key_and_partition():
    adapter = ObservableImpalaAdapter()
    frame = pd.DataFrame({"id": [1], "name": ["A"]})

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.events",
        {
            "storage": "KUDU",
            "key_columns": ["id"],
            "partitions": 5,
        },
    )

    assert "`id` BIGINT PRIMARY KEY" in ddl
    assert "PARTITION BY HASH (`id`) PARTITIONS 5" in ddl
    assert "STORED AS KUDU" in ddl


def test_impala_write_uses_insert_for_ignore_and_upsert_for_replace():
    adapter = ObservableImpalaAdapter()
    batch = pd.DataFrame({"id": [1], "name": ["A"]})

    adapter.write_batch(
        "risk.events",
        batch,
        "a",
        1,
        key_columns=["id"],
        dialect_options={"storage": "KUDU"},
    )
    adapter.write_batch(
        "risk.events",
        batch,
        "r",
        2,
        key_columns=["id"],
        dialect_options={"storage": "KUDU"},
    )

    writes = [call[1] for call in adapter.sql_calls if call[0] == "executemany"]
    assert writes[0].startswith("INSERT INTO")
    assert writes[1].startswith("UPSERT INTO")


def test_hive_metadata_query_preserves_raw_values_and_target_params():
    adapter = ObservableHiveAdapter()
    adapter.metadata_rows = [
        {
            "table_catalog": "hive",
            "table_schema": "risk",
            "table_name": "events",
            "table_type": "MANAGED_TABLE",
            "column_name": "id",
            "ordinal_position": 1,
            "data_type": "bigint",
            "is_nullable": "YES",
            "column_default": None,
        }
    ]

    inspection = adapter.inspect_schema(
        (
            QualifiedTarget.parse("risk"),
            QualifiedTarget.parse("audit.logs"),
        )
    )

    assert inspection.rows[0]["table_type"] == "MANAGED_TABLE"
    assert inspection.rows[0]["nullable"] == "YES"
    query_call = adapter.sql_calls[-1]
    assert "information_schema.columns" in query_call[1]
    assert query_call[2] == {
        "schema_0": "risk",
        "schema_1": "audit",
        "table_1": "logs",
    }


def test_describe_formatted_fallback_extracts_columns_and_raw_table_values():
    rows = [
        ("# col_name", "data_type", "comment"),
        ("id", "bigint", "编号"),
        ("name", "string", "姓名"),
        ("", None, None),
        ("Table Type:", "MANAGED_TABLE", None),
        ("Comment:", "事件表", None),
        ("InputFormat:", "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat", None),
    ]

    metadata = parse_describe_formatted(rows, "hive", "risk", "events")

    assert [row["column_name"] for row in metadata] == ["id", "name"]
    assert metadata[0]["table_type"] == "MANAGED_TABLE"
    assert metadata[0]["table_comment"] == "事件表"
    assert metadata[0]["table_engine"].endswith("MapredParquetInputFormat")


@pytest.mark.parametrize("adapter_class", [ObservableHiveAdapter, ObservableImpalaAdapter])
def test_hadoop_drop_mode_validates_ddl_before_drop(adapter_class):
    adapter = adapter_class()

    with pytest.raises(Exception, match="数据类型"):
        adapter.prepare_write(
            "risk.events",
            "d",
            pd.DataFrame({"id": [1]}),
            dialect_options={"column_types": {"id": "BIGINT); DROP TABLE x; --"}},
        )

    assert adapter.sql_calls == []


@pytest.mark.parametrize("adapter_class", [ObservableHiveAdapter, ObservableImpalaAdapter])
def test_hadoop_drop_mode_recreate_does_not_use_if_not_exists(adapter_class):
    adapter = adapter_class()

    adapter.prepare_write(
        "risk.events",
        "d",
        pd.DataFrame({"id": [1]}),
    )

    create_sql = [call[1] for call in adapter.sql_calls if call[0] == "execute"][-1]
    assert create_sql.startswith("CREATE TABLE `risk`.`events`")
    assert "IF NOT EXISTS" not in create_sql


@pytest.mark.parametrize("adapter_class", [ObservableHiveAdapter, ObservableImpalaAdapter])
def test_hadoop_json_strings_remain_unbounded_string(adapter_class):
    adapter = adapter_class()

    ddl = adapter.build_create_table_sql(
        pd.DataFrame({"payload": ['{"id": 1}']}),
        "risk.json_text",
    )

    assert "`payload` STRING" in ddl


@pytest.mark.parametrize("adapter_class", [ObservableHiveAdapter, ObservableImpalaAdapter])
def test_hadoop_explicit_empty_column_type_is_rejected(adapter_class):
    adapter = adapter_class()

    with pytest.raises(Exception, match="数据类型"):
        adapter.build_create_table_sql(
            pd.DataFrame({"id": [1]}),
            "risk.invalid_type",
            {"column_types": {"id": ""}},
        )


@pytest.mark.parametrize("adapter_class", [ObservableHiveAdapter, ObservableImpalaAdapter])
def test_hive_and_impala_json_projection_uses_get_json_object(adapter_class):
    adapter = adapter_class()

    sql = adapter.build_json_projection_sql(
        "select id, payload from events;",
        columns=["id"],
        json_fields={"payload": {"city": "$.address.city"}},
    )

    assert sql == (
        "SELECT `hscredit_json_source`.`id`, "
        "GET_JSON_OBJECT(`hscredit_json_source`.`payload`, '$.address.city') AS `city` "
        "FROM (select id, payload from events) `hscredit_json_source`"
    )
