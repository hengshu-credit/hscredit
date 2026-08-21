"""ClickHouse 原生客户端、流式读取、DDL 和最终一致写入测试。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from hscredit.database import Database, PoolOptions, register_adapter
from hscredit.database.adapters.clickhouse import ClickHouseAdapter
from hscredit.database.exceptions import DatabaseCapabilityError
from hscredit.database.metadata import QualifiedTarget


class StreamContext:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.closed = True


class ObservableClickHouseClient:
    def __init__(self):
        self.queries = []
        self.commands = []
        self.inserts = []
        self.closed = False
        self.stream_context = None
        self.metadata_frame = pd.DataFrame()
        self.rows = []
        self.insert_summary = None

    def query_df_stream(self, sql, parameters=None, settings=None):
        self.queries.append(("stream", sql, parameters, settings))
        self.stream_context = StreamContext([pd.DataFrame({"number": [0, 1]}), pd.DataFrame({"number": [2]})])
        return self.stream_context

    def query_df(self, sql, parameters=None, settings=None):
        self.queries.append(("dataframe", sql, parameters, settings))
        return self.metadata_frame.copy()

    def query(self, sql, parameters=None, settings=None):
        self.queries.append(("rows", sql, parameters, settings))
        return SimpleNamespace(result_rows=list(self.rows), column_names=["value"])

    def command(self, sql, parameters=None):
        self.commands.append((sql, parameters))
        return 0

    def insert_df(self, table_name, frame, settings=None):
        self.inserts.append((table_name, frame.copy(), settings))
        return self.insert_summary

    def close(self):
        self.closed = True


class ObservableClickHouseAdapter(ClickHouseAdapter):
    clients = []

    def load_client_module(self):
        client = ObservableClickHouseClient()
        self.clients.append(client)
        return SimpleNamespace(get_client=lambda **kwargs: client)

    def get_table_engine(self, table_name):
        del table_name
        return self.connect_kwargs.get("table_engine", "MergeTree")

    def get_sort_key_columns(self, table_name):
        del table_name
        return tuple(self.connect_kwargs.get("sort_keys", ["id"]))


@pytest.fixture(autouse=True)
def register_clickhouse_adapter():
    ObservableClickHouseAdapter.clients.clear()
    register_adapter("observable_clickhouse", ObservableClickHouseAdapter, replace=True)


@pytest.fixture
def adapter():
    return ObservableClickHouseAdapter(
        connect_kwargs={"host": "clickhouse.internal", "database": "risk"},
        pool_options=PoolOptions(),
        adapter_options={},
    )


def test_clickhouse_progress_false_does_not_issue_count():
    database = Database("observable_clickhouse", host="clickhouse.internal")

    chunks = list(database.stream_query("select number from numbers(3)", progress=False))

    assert [chunk["number"].tolist() for chunk in chunks] == [[0, 1], [2]]
    assert database.adapter.client.queries == [("stream", "select number from numbers(3)", None, None)]
    assert database.adapter.client.stream_context.closed is True


def test_clickhouse_query_passes_parameters_to_native_client(adapter):
    adapter.client.metadata_frame = pd.DataFrame({"id": [2]})

    frame = adapter.query(
        "select id from events where id > {minimum:Int64}",
        params={"minimum": 1},
        result="dataframe",
    )

    assert frame["id"].tolist() == [2]
    assert adapter.client.queries[-1][2] == {"minimum": 1}


def test_clickhouse_replace_requires_replacing_merge_tree(adapter):
    with pytest.raises(DatabaseCapabilityError, match="ReplacingMergeTree"):
        adapter.prepare_write(
            "risk.events",
            "r",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"engine": "MergeTree"},
        )


def test_clickhouse_append_with_key_guarantee_is_rejected(adapter):
    with pytest.raises(DatabaseCapabilityError, match="主键冲突"):
        adapter.prepare_write(
            "risk.events",
            "a",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"engine": "MergeTree"},
        )


def test_clickhouse_replacing_merge_tree_write_marks_eventual_consistency():
    database = Database(
        "observable_clickhouse",
        host="clickhouse.internal",
        table_engine="ReplacingMergeTree",
        sort_keys=["id"],
    )

    result = database.stream_write(
        pd.DataFrame({"id": [1], "name": ["覆盖"]}),
        "risk.events",
        mode="r",
        dialect_options={"engine": "ReplacingMergeTree"},
    )

    assert database.adapter.client.inserts[0][0] == "risk.events"
    assert result.consistency == "eventual"
    assert result.rows_inserted is None
    assert result.rows_updated is None


def test_clickhouse_default_and_replacing_ddl(adapter):
    frame = pd.DataFrame({"id": [1], "created_at": pd.to_datetime(["2026-08-20"]), "name": ["A"]})

    default_ddl = adapter.build_create_table_sql(frame, "risk.events", {})
    replacing_ddl = adapter.build_create_table_sql(
        frame,
        "risk.events_current",
        {
            "engine": "ReplacingMergeTree",
            "key_columns": ["id"],
            "version_column": "created_at",
        },
    )

    assert "ENGINE = MergeTree" in default_ddl
    assert "ORDER BY tuple()" in default_ddl
    assert "`id` Int64" in default_ddl
    assert "`created_at` DateTime64(6)" in default_ddl
    assert "ENGINE = ReplacingMergeTree(`created_at`)" in replacing_ddl
    assert "ORDER BY (`id`)" in replacing_ddl


def test_clickhouse_metadata_uses_system_tables_and_preserves_values(adapter):
    adapter.client.metadata_frame = pd.DataFrame(
        [
            {
                "database": "risk",
                "table_name": "events",
                "table_type": "BASE TABLE",
                "table_comment": "事件表",
                "table_engine": "ReplacingMergeTree",
                "column_name": "id",
                "ordinal_position": 1,
                "data_type": "UInt64",
                "full_data_type": "UInt64",
                "default_value": None,
                "primary_key": 1,
                "sort_key": 1,
                "column_comment": "编号",
            }
        ]
    )

    inspection = adapter.inspect_schema((QualifiedTarget.parse("risk.events"),))

    assert inspection.rows[0]["table_engine"] == "ReplacingMergeTree"
    assert inspection.rows[0]["data_type"] == "UInt64"
    assert inspection.rows[0]["primary_key"] == 1
    query_call = adapter.client.queries[-1]
    assert "system.columns" in query_call[1]
    assert query_call[2] == {"database_0": "risk", "table_0": "events"}


def test_clickhouse_close_closes_native_client(adapter):
    adapter.close()
    adapter.close()

    assert adapter.client.closed is True


def test_clickhouse_native_writer_keeps_unknown_insert_count(adapter):
    result = adapter.write_batch(
        "risk.events",
        pd.DataFrame({"id": [1]}),
        "o",
        1,
    )

    assert result.inserted is None


def test_clickhouse_drop_mode_validates_ddl_before_drop(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.prepare_write(
            "risk.events",
            "d",
            pd.DataFrame({"id": [1]}),
            dialect_options={"column_types": {"id": "Int64); DROP TABLE risk.events; --"}},
        )

    assert adapter.client.commands == []


def test_clickhouse_drop_mode_recreate_does_not_use_if_not_exists(adapter):
    adapter.prepare_write(
        "risk.events",
        "d",
        pd.DataFrame({"id": [1]}),
        dialect_options={"engine": "MergeTree"},
    )

    create_sql = adapter.client.commands[-1][0]
    assert create_sql.startswith("CREATE TABLE `risk`.`events`")
    assert "IF NOT EXISTS" not in create_sql


def test_clickhouse_json_inference_respects_server_version_and_override(adapter):
    frame = pd.DataFrame({"payload": ['{"id": 1}']})

    adapter.client.server_version = "25.3.1"
    automatic = adapter.build_create_table_sql(frame, "risk.json_auto")
    forced_string = adapter.build_create_table_sql(
        frame,
        "risk.json_string",
        {"json_type": "String"},
    )

    assert "`payload` JSON" in automatic
    assert "`payload` String" in forced_string


def test_clickhouse_invalid_json_type_is_rejected_even_for_plain_text(adapter):
    with pytest.raises(Exception, match="json_type"):
        adapter.build_create_table_sql(
            pd.DataFrame({"note": ["plain text"]}),
            "risk.invalid_json_option",
            {"json_type": "JSNO"},
        )


def test_clickhouse_explicit_empty_column_type_is_rejected(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.build_create_table_sql(
            pd.DataFrame({"id": [1]}),
            "risk.invalid_type",
            {"column_types": {"id": ""}},
        )


def test_clickhouse_json_projection_handles_scalar_and_nested_values(adapter):
    sql = adapter.build_json_projection_sql(
        "select id, payload from events;",
        columns=["id"],
        json_fields={"payload": {"risk_tags": "$.risk.tags"}},
    )

    assert sql == (
        "SELECT `hscredit_json_source`.`id`, "
        "if(JSON_EXISTS(`hscredit_json_source`.`payload`, '$.risk.tags'), "
        "nullIf(coalesce(nullIf(JSON_QUERY(`hscredit_json_source`.`payload`, '$.risk.tags'), ''), "
        "JSON_VALUE(`hscredit_json_source`.`payload`, '$.risk.tags')), 'null'), NULL) AS `risk_tags` "
        "FROM (select id, payload from events) `hscredit_json_source`"
    )
