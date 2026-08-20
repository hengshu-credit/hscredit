"""Oracle 适配器连接池、MERGE、DDL 和元数据测试。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from hscredit.database import Database, PoolOptions, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.adapters.oracle import OracleAdapter
from hscredit.database.metadata import QualifiedTarget


class ObservableOracleAdapter(OracleAdapter):
    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        BaseDatabaseAdapter.__init__(
            self,
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.driver = SimpleNamespace()
        self.sql_calls = []
        self.key_rows = connect_kwargs.get(
            "key_rows",
            [{"OWNER": "RISK", "TABLE_NAME": "EVENTS", "COLUMN_NAME": "ID", "POSITION": 1}],
        )
        self.metadata_rows = connect_kwargs.get("metadata_rows", [])

    def execute(self, sql, params=None):
        self.sql_calls.append(("execute", sql, params))
        return 0

    def executemany(self, sql, values):
        materialized = list(values)
        self.sql_calls.append(("executemany", sql, materialized))
        return len(materialized)

    def query(self, sql, params=None, result="dataframe"):
        self.sql_calls.append(("query", sql, params, result))
        if sql.startswith("SELECT cc.OWNER"):
            return list(self.key_rows)
        return list(self.metadata_rows)


@pytest.fixture(autouse=True)
def register_oracle_adapter():
    register_adapter("observable_oracle", ObservableOracleAdapter, replace=True)


@pytest.fixture
def adapter():
    return ObservableOracleAdapter(
        connect_kwargs={"user": "risk"},
        pool_options=PoolOptions(),
        adapter_options={},
    )


def test_oracle_count_wrapper_has_no_as_keyword(adapter):
    assert adapter.build_count_sql("select id from events;") == (
        "SELECT COUNT(1) FROM (select id from events) hscredit_count"
    )


def test_oracle_append_merge_has_no_matched_update(adapter):
    sql = adapter.build_merge_sql(
        "RISK.EVENTS",
        ["ID", "NAME"],
        ["ID"],
        mode="a",
    )

    assert "WHEN NOT MATCHED THEN INSERT" in sql
    assert "WHEN MATCHED THEN UPDATE" not in sql
    assert 'SELECT :1 AS "ID", :2 AS "NAME" FROM dual' in sql


def test_oracle_replace_merge_updates_only_non_key_columns(adapter):
    sql = adapter.build_merge_sql(
        "RISK.EVENTS",
        ["ID", "NAME"],
        ["ID"],
        mode="r",
    )

    assert 'WHEN MATCHED THEN UPDATE SET target."NAME"=source."NAME"' in sql
    assert 'target."ID"=source."ID"' not in sql.split("WHEN MATCHED")[1]


def test_oracle_resolves_primary_key_from_dictionary(adapter):
    keys = adapter.resolve_key_columns(
        "RISK.EVENTS",
        None,
        pd.DataFrame({"ID": [1], "NAME": ["A"]}),
        dialect_options={},
    )

    assert keys == ("ID",)
    assert adapter.sql_calls[-1][2] == {"owner": "RISK", "table_name": "EVENTS"}


def test_oracle_stream_write_uses_merge_with_discovered_key():
    database = Database("observable_oracle", user="risk")

    result = database.stream_write(
        pd.DataFrame({"ID": [1], "NAME": ["新值"]}),
        "RISK.EVENTS",
        mode="r",
    )

    write_call = next(call for call in database.adapter.sql_calls if call[0] == "executemany")
    assert 'MERGE INTO "RISK"."EVENTS"' in write_call[1]
    assert write_call[2] == [(1, "新值")]
    assert result.rows_inserted is None
    assert result.rows_updated is None


def test_oracle_create_table_maps_types_and_primary_key(adapter):
    frame = pd.DataFrame(
        {
            "ID": pd.Series([1], dtype="int64"),
            "ENABLED": pd.Series([True], dtype="bool"),
            "AMOUNT": pd.Series([1.5], dtype="float64"),
            "CREATED_AT": pd.to_datetime(["2026-08-20"]),
            "NAME": ["张三"],
        }
    )

    ddl = adapter.build_create_table_sql(
        frame,
        "RISK.EVENTS",
        {
            "key_columns": ["ID"],
            "column_comments": {"NAME": "姓名"},
            "table_comment": "事件表",
        },
    )

    assert '"ID" NUMBER(19) NOT NULL' in ddl
    assert '"ENABLED" NUMBER(1)' in ddl
    assert '"AMOUNT" BINARY_DOUBLE' in ddl
    assert '"CREATED_AT" TIMESTAMP' in ddl
    assert '"NAME" VARCHAR2(255 CHAR)' in ddl
    assert 'CONSTRAINT "EVENTS_PK" PRIMARY KEY ("ID")' in ddl

    adapter.create_table(
        frame,
        "RISK.EVENTS",
        dialect_options={
            "key_columns": ["ID"],
            "column_comments": {"NAME": "姓名"},
            "table_comment": "事件表",
        },
    )
    statements = [call[1] for call in adapter.sql_calls if call[0] == "execute"]
    assert any(statement.startswith("COMMENT ON COLUMN") for statement in statements)
    assert any(statement.startswith("COMMENT ON TABLE") for statement in statements)


def test_oracle_prepare_modes_clear_or_drop_then_create(adapter):
    frame = pd.DataFrame({"ID": [1]})

    adapter.prepare_write("RISK.EVENTS", "o", frame)
    assert adapter.sql_calls[-1] == (
        "execute",
        'TRUNCATE TABLE "RISK"."EVENTS"',
        None,
    )

    adapter.sql_calls.clear()
    adapter.prepare_write("RISK.EVENTS", "d", frame, key_columns=["ID"])
    assert adapter.sql_calls[0] == (
        "execute",
        'DROP TABLE "RISK"."EVENTS" PURGE',
        None,
    )
    assert adapter.sql_calls[1][1].startswith('CREATE TABLE "RISK"."EVENTS"')


def test_oracle_metadata_preserves_dictionary_values():
    row = {
        "OWNER": "RISK",
        "TABLE_NAME": "EVENTS",
        "TABLE_TYPE": "TABLE",
        "TABLE_COMMENT": "事件表",
        "COLUMN_NAME": "ID",
        "COLUMN_ID": 1,
        "DATA_TYPE": "NUMBER",
        "FULL_DATA_TYPE": "NUMBER(19,0)",
        "NULLABLE": "N",
        "DATA_DEFAULT": None,
        "CONSTRAINT_TYPE": "P",
        "COLUMN_COMMENT": "编号",
    }
    adapter = ObservableOracleAdapter(
        connect_kwargs={"metadata_rows": [row]},
        pool_options=PoolOptions(),
        adapter_options={},
    )

    inspection = adapter.inspect_schema((QualifiedTarget.parse("RISK.EVENTS"),))

    assert inspection.rows[0]["schema"] == "RISK"
    assert inspection.rows[0]["table_type"] == "TABLE"
    assert inspection.rows[0]["nullable"] == "N"
    assert inspection.rows[0]["primary_key"] is True
    assert adapter.sql_calls[-1][2] == {"owner_0": "RISK", "table_0": "EVENTS"}


def test_oracle_native_pool_maps_dbutils_style_limits():
    captured = {}

    class NativePool:
        def acquire(self):
            return object()

        def close(self):
            captured["closed"] = True

    class Driver:
        POOL_GETMODE_WAIT = "WAIT"

        @staticmethod
        def create_pool(**kwargs):
            captured.update(kwargs)
            return NativePool()

    class Adapter(OracleAdapter):
        def load_driver(self):
            return Driver()

    adapter = Adapter(
        connect_kwargs={"user": "risk", "password": "secret", "dsn": "db/service"},
        pool_options=PoolOptions(mincached=2, maxconnections=8, blocking=True),
        adapter_options={},
    )

    assert captured["min"] == 2
    assert captured["max"] == 8
    assert captured["increment"] == 1
    assert captured["getmode"] == "WAIT"
    assert captured["user"] == "risk"
    adapter.close()
    assert captured["closed"] is True


def test_oracle_drop_mode_validates_comments_before_drop(adapter):
    frame = pd.DataFrame({"ID": [1]})

    with pytest.raises(Exception, match="未知字段"):
        adapter.prepare_write(
            "RISK.EVENTS",
            "d",
            frame,
            dialect_options={"column_comments": {"MISSING": "注释"}},
        )

    assert adapter.sql_calls == []
