"""StarRocks 表模型、DDL、Stream Load 和元数据测试。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from hscredit.database import PoolOptions
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.adapters.starrocks import StarRocksAdapter
from hscredit.database.exceptions import DatabaseCapabilityError
from hscredit.database.metadata import QualifiedTarget


class ObservableStarRocksAdapter(StarRocksAdapter):
    def __init__(self, *, connect_kwargs=None, adapter_options=None):
        BaseDatabaseAdapter.__init__(
            self,
            connect_kwargs=connect_kwargs or {"database": "risk", "host": "fe.internal"},
            pool_options=PoolOptions(),
            adapter_options=adapter_options or {},
        )
        self.driver = SimpleNamespace()
        self.sql_calls = []
        self.metadata_rows = []
        self.table_model = (connect_kwargs or {}).get("table_model", "DUPLICATE KEY")
        self.key_rows = [{"COLUMN_NAME": "id", "ORDINAL_POSITION": 1}]
        self.stream_load_requests = []

    def execute(self, sql, params=None):
        self.sql_calls.append(("execute", sql, params))
        return 0

    def executemany(self, sql, values):
        materialized = list(values)
        self.sql_calls.append(("executemany", sql, materialized))
        return len(materialized)

    def query(self, sql, params=None, result="dataframe"):
        self.sql_calls.append(("query", sql, params, result))
        if sql.startswith("SHOW CREATE TABLE"):
            return [{"Table": "events", "Create Table": f"CREATE TABLE events {self.table_model}(id)"}]
        if "FROM information_schema.columns" in sql and "COLUMN_KEY" in sql and "JOIN" not in sql:
            return list(self.key_rows)
        return list(self.metadata_rows)

    def _send_stream_load(self, request):
        self.stream_load_requests.append(request)
        return {"Status": "Success", "NumberLoadedRows": 2}


@pytest.fixture
def adapter():
    return ObservableStarRocksAdapter()


def test_starrocks_primary_table_allows_replace_not_ignore(adapter):
    capabilities = adapter.capabilities_for_table(
        "risk.events",
        {"table_model": "PRIMARY KEY"},
    )

    assert "r" in capabilities.write_modes
    assert "a" not in capabilities.write_modes


def test_starrocks_duplicate_table_exposes_no_conflict_modes(adapter):
    capabilities = adapter.capabilities_for_table(
        "risk.events",
        {"table_model": "DUPLICATE KEY"},
    )

    assert "a" not in capabilities.write_modes
    assert "r" not in capabilities.write_modes


def test_starrocks_default_ddl_uses_explicit_key_and_distribution(adapter):
    frame = pd.DataFrame({"id": [1], "name": ["A"]})

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.events",
        {
            "key_columns": ["id"],
            "table_model": "PRIMARY KEY",
            "buckets": 8,
        },
    )

    assert "ENGINE=OLAP" in ddl
    assert "PRIMARY KEY (`id`)" in ddl
    assert "DISTRIBUTED BY HASH(`id`) BUCKETS 8" in ddl
    assert "`name` VARCHAR(16)" in ddl


def test_starrocks_insert_sql_uses_table_model_semantics(adapter):
    append_sql = adapter.build_insert_sql("risk.events", ["id", "name"], mode="a", key_columns=None)
    replace_sql = adapter.build_insert_sql("risk.events", ["id", "name"], mode="r", key_columns=["id"])

    assert append_sql.startswith("INSERT INTO")
    assert "IGNORE" not in append_sql
    assert replace_sql.startswith("INSERT INTO")
    assert "ON DUPLICATE" not in replace_sql


def test_starrocks_rejects_append_on_primary_table(adapter):
    with pytest.raises(DatabaseCapabilityError, match="PRIMARY KEY"):
        adapter.prepare_write(
            "risk.events",
            "a",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"table_model": "PRIMARY KEY"},
        )


def test_starrocks_stream_load_sends_utf8_csv_and_safe_headers():
    adapter = ObservableStarRocksAdapter(
        connect_kwargs={
            "database": "risk",
            "host": "fe.internal",
            "user": "loader",
            "password": "secret",
        },
        adapter_options={"http_port": 8030},
    )
    batch = pd.DataFrame({"id": [1, 2], "name": ["张三", pd.NA]})

    result = adapter.write_batch(
        "risk.events",
        batch,
        "o",
        1,
        dialect_options={"stream_load": True, "table_model": "DUPLICATE KEY"},
    )

    request = adapter.stream_load_requests[0]
    assert request.full_url == "http://fe.internal:8030/api/risk/events/_stream_load"
    assert request.get_method() == "PUT"
    assert request.headers["Columns"] == "id,name"
    assert "secret" not in repr(request.headers)
    assert "张三".encode("utf-8") in request.data
    assert b"\\N" in request.data
    assert result.inserted == 2


def test_starrocks_metadata_supports_catalog_database_table_filter(adapter):
    adapter.metadata_rows = [
        {
            "table_catalog": "default_catalog",
            "table_schema": "risk",
            "table_name": "events",
            "table_type": "BASE TABLE",
            "table_comment": "事件表",
            "table_engine": "StarRocks",
            "column_name": "id",
            "ordinal_position": 1,
            "data_type": "bigint",
            "full_data_type": "bigint",
            "nullable": "NO",
            "default_value": None,
            "column_key": "PRI",
            "column_comment": "编号",
        }
    ]

    inspection = adapter.inspect_schema((QualifiedTarget.parse("default_catalog.risk.events"),))

    assert inspection.rows[0]["catalog"] == "default_catalog"
    assert inspection.rows[0]["database_type"] == "starrocks"
    assert inspection.rows[0]["primary_key"] is True
    query_call = adapter.sql_calls[-1]
    assert query_call[2] == ("default_catalog", "risk", "events")


def test_starrocks_stream_load_failure_preserves_server_details_without_credentials(adapter):
    def fail(_request):
        return {
            "Status": "Fail",
            "Message": "column count mismatch",
            "ErrorURL": "http://fe/error/1",
        }

    adapter._send_stream_load = fail

    from hscredit.database.exceptions import DatabaseWriteError

    with pytest.raises(DatabaseWriteError, match="column count mismatch") as caught:
        adapter.write_batch(
            "risk.events",
            pd.DataFrame({"id": [1]}),
            "o",
            1,
            dialect_options={"stream_load": True, "table_model": "DUPLICATE KEY"},
        )

    assert "password" not in str(caught.value).lower()
    assert "secret" not in str(caught.value)


def test_starrocks_duplicate_table_rejects_append(adapter):
    with pytest.raises(DatabaseCapabilityError, match="DUPLICATE KEY"):
        adapter.prepare_write(
            "risk.events",
            "a",
            pd.DataFrame({"id": [1]}),
            dialect_options={"table_model": "DUPLICATE KEY"},
        )


def test_starrocks_drop_mode_validates_ddl_before_drop(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.prepare_write(
            "risk.events",
            "d",
            pd.DataFrame({"id": [1]}),
            dialect_options={
                "table_model": "DUPLICATE KEY",
                "column_types": {"id": "BIGINT); DROP TABLE x; --"},
            },
        )

    assert adapter.sql_calls == []


def test_starrocks_drop_mode_recreate_does_not_use_if_not_exists(adapter):
    adapter.prepare_write(
        "risk.events",
        "d",
        pd.DataFrame({"id": [1]}),
        dialect_options={"table_model": "DUPLICATE KEY"},
    )

    create_sql = [call[1] for call in adapter.sql_calls if call[0] == "execute"][-1]
    assert create_sql.startswith("CREATE TABLE `risk`.`events`")
    assert "IF NOT EXISTS" not in create_sql


def test_starrocks_string_inference_uses_bounded_varchar_and_json(adapter):
    frame = pd.DataFrame(
        {
            "description": ["a" * 300],
            "payload": ['{"id": 1}'],
        }
    )

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.string_types",
        {"table_model": "DUPLICATE KEY", "key_columns": ["description"]},
    )

    assert "`description` VARCHAR(512)" in ddl
    assert "`payload` JSON" in ddl


def test_starrocks_rejects_strings_beyond_single_column_limit(adapter):
    frame = pd.DataFrame({"description": ["a" * 70_000]})

    with pytest.raises(Exception, match="65533"):
        adapter.build_create_table_sql(frame, "risk.too_long")


def test_starrocks_sql_write_wraps_only_json_placeholders(adapter):
    batch = pd.DataFrame(
        {
            "id": [1],
            "payload": ['{"event": "apply"}'],
            "note": ["普通文本"],
        }
    )

    adapter.write_batch(
        "risk.events",
        batch,
        "o",
        1,
    )

    sql = adapter.sql_calls[-1][1]
    assert "VALUES (%s, parse_json(%s), %s)" in sql


def test_starrocks_json_columns_are_fixed_from_first_create_batch(adapter):
    first = pd.DataFrame({"id": [1], "payload": ['{"event": "apply"}']})
    second = pd.DataFrame({"id": [2], "payload": ["later plain text"]})
    options = {"table_model": "DUPLICATE KEY"}

    adapter.prepare_write(
        "risk.events",
        "d",
        first,
        key_columns=["id"],
        dialect_options=options,
    )
    adapter.write_batch("risk.events", first, "d", 1, dialect_options=options)
    adapter.write_batch("risk.events", second, "d", 2, dialect_options=options)

    writes = [call[1] for call in adapter.sql_calls if call[0] == "executemany"]
    assert len(writes) == 2
    assert all("VALUES (%s, parse_json(%s))" in sql for sql in writes)


def test_starrocks_later_json_does_not_change_first_plain_text_schema(adapter):
    first = pd.DataFrame({"id": [1], "payload": ["first plain text"]})
    second = pd.DataFrame({"id": [2], "payload": ['{"event": "later"}']})
    options = {"table_model": "DUPLICATE KEY"}

    adapter.prepare_write(
        "risk.events",
        "d",
        first,
        key_columns=["id"],
        dialect_options=options,
    )
    adapter.write_batch("risk.events", first, "d", 1, dialect_options=options)
    adapter.write_batch("risk.events", second, "d", 2, dialect_options=options)

    writes = [call[1] for call in adapter.sql_calls if call[0] == "executemany"]
    assert len(writes) == 2
    assert all("parse_json" not in sql for sql in writes)


def test_starrocks_interleaved_writes_keep_operation_scoped_json_schema(adapter):
    json_options = {"table_model": "DUPLICATE KEY"}
    plain_options = {"table_model": "DUPLICATE KEY"}
    json_first = pd.DataFrame({"id": [1], "payload": ['{"event": "json"}']})
    plain_first = pd.DataFrame({"id": [2], "payload": ["plain text"]})

    adapter.prepare_write(
        "risk.events",
        "d",
        json_first,
        key_columns=["id"],
        dialect_options=json_options,
    )
    adapter.prepare_write(
        "risk.events",
        "d",
        plain_first,
        key_columns=["id"],
        dialect_options=plain_options,
    )
    adapter.write_batch(
        "risk.events",
        json_first,
        "d",
        1,
        dialect_options=json_options,
    )
    adapter.write_batch(
        "risk.events",
        plain_first,
        "d",
        1,
        dialect_options=plain_options,
    )

    writes = [call[1] for call in adapter.sql_calls if call[0] == "executemany"]
    assert "parse_json(%s)" in writes[-2]
    assert "parse_json(%s)" not in writes[-1]


def test_starrocks_explicit_empty_column_type_is_rejected(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.build_create_table_sql(
            pd.DataFrame({"id": [1]}),
            "risk.invalid_type",
            {
                "table_model": "DUPLICATE KEY",
                "column_types": {"id": ""},
            },
        )


def test_starrocks_json_projection_uses_get_json_string(adapter):
    sql = adapter.build_json_projection_sql(
        "select id, payload from risk.events;",
        columns=["id"],
        json_fields={"payload": {"city": "$.address.city"}},
    )

    assert sql == (
        "SELECT `hscredit_json_source`.`id`, "
        "GET_JSON_STRING(`hscredit_json_source`.`payload`, '$.address.city') AS `city` "
        "FROM (select id, payload from risk.events) `hscredit_json_source`"
    )
