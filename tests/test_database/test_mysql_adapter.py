"""MySQL 适配器方言、DDL、元数据和写入模式测试。"""

from types import SimpleNamespace
from contextlib import contextmanager

import pandas as pd
import pytest

from hscredit.database import Database, PoolOptions, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.adapters.mysql import MySQLAdapter
from hscredit.database.writing import BatchWriteResult


class ServerSideCursor:
    pass


class ObservableMySQLAdapter(MySQLAdapter):
    """绕过外部连接，仅执行 MySQL 适配器真实方言逻辑。"""

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        BaseDatabaseAdapter.__init__(
            self,
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.driver = SimpleNamespace(cursors=SimpleNamespace(SSCursor=ServerSideCursor))
        self.sql_calls = []
        self.key_rows = connect_kwargs.get(
            "key_rows",
            [
                {"INDEX_NAME": "PRIMARY", "COLUMN_NAME": "id", "SEQ_IN_INDEX": 1},
                {"INDEX_NAME": "uniq_name", "COLUMN_NAME": "name", "SEQ_IN_INDEX": 1},
            ],
        )
        self.metadata_rows = connect_kwargs.get("metadata_rows", [])
        self.existing_tables = set(connect_kwargs.get("existing_tables", {"risk.events"}))

    def execute(self, sql, params=None):
        self.sql_calls.append(("execute", sql, params))
        return 0

    def executemany(self, sql, values):
        materialized = list(values)
        self.sql_calls.append(("executemany", sql, materialized))
        return len(materialized)

    def query(self, sql, params=None, result="dataframe"):
        self.sql_calls.append(("query", sql, params, result))
        if "information_schema.statistics" in sql:
            return list(self.key_rows)
        if sql.startswith("SELECT 1 FROM information_schema.tables"):
            qualified_name = ".".join(str(part) for part in params)
            return [(1,)] if qualified_name in self.existing_tables else []
        return list(self.metadata_rows)


@pytest.fixture(autouse=True)
def register_mysql_adapter():
    register_adapter("observable_mysql", ObservableMySQLAdapter, replace=True)


@pytest.fixture
def adapter():
    return ObservableMySQLAdapter(
        connect_kwargs={"database": "risk"},
        pool_options=PoolOptions(),
        adapter_options={},
    )


def test_mysql_append_preserves_existing_keys(adapter):
    sql = adapter.build_insert_sql(
        "risk.events",
        ["id", "name"],
        mode="a",
        key_columns=["id"],
    )

    assert sql == "INSERT INTO `risk`.`events` (`id`, `name`) VALUES (%s, %s)"
    assert "IGNORE" not in sql
    assert "ON DUPLICATE" not in sql


def test_mysql_replace_updates_only_non_key_columns(adapter):
    sql = adapter.build_insert_sql(
        "risk.events",
        ["id", "name"],
        mode="r",
        key_columns=["id"],
    )

    assert sql.endswith("ON DUPLICATE KEY UPDATE `name`=VALUES(`name`)")
    assert "`id`=VALUES(`id`)" not in sql


def test_mysql_identifier_quoting_escapes_backticks(adapter):
    assert adapter.quote_qualified_name("risk.ev`ents") == "`risk`.`ev``ents`"


def test_mysql_recognizes_only_duplicate_table_error(adapter):
    assert adapter.is_table_already_exists_error(RuntimeError(1050, "Table already exists")) is True
    assert adapter.is_table_already_exists_error(RuntimeError(1146, "Table does not exist")) is False


def test_mysql_resolves_primary_key_before_unique_index(adapter):
    keys = adapter.resolve_key_columns(
        "risk.events",
        None,
        pd.DataFrame({"id": [1], "name": ["A"]}),
        dialect_options={},
    )

    assert keys == ("id",)
    query_call = adapter.sql_calls[-1]
    assert query_call[0] == "query"
    assert query_call[2] == ("risk", "events")


def test_mysql_checks_table_existence_without_create_ddl(adapter):
    assert adapter.table_exists("risk.events") is True
    assert "information_schema.tables" in adapter.sql_calls[-1][1]
    assert adapter.sql_calls[-1][2] == ("risk", "events")

    assert adapter.table_exists("risk.missing_events") is False


def test_mysql_stream_write_uses_discovered_key_for_replace():
    database = Database("observable_mysql", database="risk")

    result = database.stream_write(
        pd.DataFrame({"id": [1], "name": ["新值"]}),
        "risk.events",
        mode="r",
    )

    write_call = next(call for call in database.adapter.sql_calls if call[0] == "executemany")
    assert "ON DUPLICATE KEY UPDATE `name`=VALUES(`name`)" in write_call[1]
    assert write_call[2] == [(1, "新值")]
    assert result.rows_inserted is None
    assert result.rows_updated is None


def test_mysql_prepare_modes_clear_or_drop_then_create(adapter):
    frame = pd.DataFrame({"id": [1], "name": ["A"]})

    adapter.prepare_write("risk.events", "o", frame, key_columns=["id"])
    assert adapter.sql_calls[-1] == (
        "execute",
        "TRUNCATE TABLE `risk`.`events`",
        None,
    )

    adapter.sql_calls.clear()
    adapter.prepare_write(
        "risk.events",
        "d",
        frame,
        key_columns=["id"],
        dialect_options={"table_comment": "事件表"},
    )
    assert adapter.sql_calls[0] == (
        "execute",
        "DROP TABLE IF EXISTS `risk`.`events`",
        None,
    )
    assert adapter.sql_calls[1][0] == "execute"
    assert adapter.sql_calls[1][1].startswith("CREATE TABLE `risk`.`events`")


def test_mysql_create_table_maps_types_keys_and_comments(adapter):
    frame = pd.DataFrame(
        {
            "id": pd.Series([1], dtype="int64"),
            "enabled": pd.Series([True], dtype="bool"),
            "amount": pd.Series([1.5], dtype="float64"),
            "created_at": pd.to_datetime(["2026-08-20"]),
            "name": ["张三"],
        }
    )

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.events",
        {
            "key_columns": ["id"],
            "column_comments": {"name": "姓名"},
            "table_comment": "事件表",
        },
    )

    assert "`id` BIGINT NOT NULL" in ddl
    assert "`enabled` BOOLEAN" in ddl
    assert "`amount` DOUBLE" in ddl
    assert "`created_at` DATETIME" in ddl
    assert "`name` VARCHAR(16) COMMENT '姓名'" in ddl
    assert "PRIMARY KEY (`id`)" in ddl
    assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" in ddl
    assert "COMMENT='事件表'" in ddl


def test_mysql_comments_escape_backslashes_before_quotes(adapter):
    ddl = adapter.build_create_table_sql(
        pd.DataFrame({"id": [1]}),
        "risk.events",
        {"table_comment": "前缀\\'; DROP TABLE x; --"},
    )

    assert "前缀\\\\''; DROP TABLE x; --" in ddl


def test_mysql_string_inference_uses_json_and_text_families(adapter):
    frame = pd.DataFrame(
        {
            "short_text": ["a" * 50],
            "long_text": ["b" * 300],
            "medium_text": ["c" * 70_000],
            "json_value": ['{"id": 1, "tags": ["a"]}'],
            "mixed_value": ['{"id": 1}普通文本'],
        }
    )

    ddl = adapter.build_create_table_sql(frame, "risk.string_types")

    assert "`short_text` VARCHAR(64)" in ddl
    assert "`long_text` TEXT" in ddl
    assert "`medium_text` MEDIUMTEXT" in ddl
    assert "`json_value` JSON" in ddl
    assert "`mixed_value` VARCHAR(16)" in ddl


def test_mysql_varchar_inference_respects_utf8mb4_byte_capacity(adapter):
    frame = pd.DataFrame({"description": ["衡" * 20_000]})

    ddl = adapter.build_create_table_sql(
        frame,
        "risk.utf8_text",
        {"varchar_max_length": 65_535, "charset": "utf8mb4"},
    )

    assert "`description` MEDIUMTEXT" in ddl


def test_mysql_explicit_empty_column_type_is_rejected(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.build_create_table_sql(
            pd.DataFrame({"id": [1]}),
            "risk.invalid_type",
            {"column_types": {"id": ""}},
        )


def test_mysql_known_charset_width_cannot_be_overridden_lower(adapter):
    with pytest.raises(Exception, match="utf8mb4"):
        adapter.build_create_table_sql(
            pd.DataFrame({"description": ["衡" * 20_000]}),
            "risk.unsafe_width",
            {
                "charset": "utf8mb4",
                "varchar_max_length": 65_535,
                "charset_max_bytes_per_character": 1,
            },
        )


def test_mysql_stream_cursor_uses_sscursor(adapter):
    calls = []

    class Connection:
        def cursor(self, cursor_class=None):
            calls.append(cursor_class)
            return object()

    adapter.create_cursor(Connection(), stream=True)

    assert calls == [ServerSideCursor]


def test_mysql_metadata_uses_one_parameterized_information_schema_query():
    metadata_rows = [
        {
            "catalog": "def",
            "database_name": "risk",
            "table_name": "events",
            "table_type": "BASE TABLE",
            "table_comment": "事件表",
            "table_engine": "InnoDB",
            "column_name": "id",
            "ordinal_position": 1,
            "data_type": "bigint",
            "full_data_type": "bigint unsigned",
            "nullable": "NO",
            "default_value": None,
            "column_key": "PRI",
            "column_comment": "编号",
        }
    ]
    adapter = ObservableMySQLAdapter(
        connect_kwargs={"metadata_rows": metadata_rows},
        pool_options=PoolOptions(),
        adapter_options={},
    )
    from hscredit.database.metadata import QualifiedTarget

    inspection = adapter.inspect_schema(
        (
            QualifiedTarget.parse("risk"),
            QualifiedTarget.parse("audit.logs"),
        )
    )

    assert len(inspection.rows) == 1
    assert inspection.rows[0]["database_type"] == "mysql"
    assert inspection.rows[0]["table_type"] == "BASE TABLE"
    assert inspection.rows[0]["primary_key"] is True
    query_call = adapter.sql_calls[-1]
    assert query_call[0] == "query"
    assert query_call[2] == ("risk", "audit", "logs")
    assert "information_schema.columns" in query_call[1]


def test_mysql_write_batch_converts_pandas_missing_values(adapter):
    batch = pd.DataFrame({"id": [1, 2], "name": ["A", pd.NA]})

    result = adapter.write_batch(
        "risk.events",
        batch,
        "o",
        1,
        key_columns=["id"],
    )

    assert isinstance(result, BatchWriteResult)
    assert adapter.sql_calls[-1][2] == [(1, "A"), (2, None)]
    assert result.inserted == 2
    assert result.skipped == 0


def test_mysql_append_skips_only_duplicate_key_error_without_update_branch(adapter):
    class DuplicateKeyError(Exception):
        pass

    class Connection:
        commit_calls = 0
        rollback_calls = 0

        def commit(self):
            self.commit_calls += 1

        def rollback(self):
            self.rollback_calls += 1

    class Cursor:
        def execute(self, sql, params=None):
            adapter.sql_calls.append(("execute", sql, params))
            if params[0] == 1:
                raise DuplicateKeyError(1062, "Duplicate entry")

    connection = Connection()

    @contextmanager
    def connection_cursor():
        yield connection, Cursor()

    adapter.connection_cursor = connection_cursor
    result = adapter.write_batch(
        "risk.events",
        pd.DataFrame({"id": [1, 2], "name": ["重复", "新增"]}),
        "a",
        1,
        key_columns=["id"],
    )

    assert result.inserted == 1
    assert result.skipped == 1
    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0
    assert all("ON DUPLICATE" not in call[1] for call in adapter.sql_calls)


def test_mysql_append_rolls_back_whole_batch_on_non_duplicate_error(adapter):
    class DriverError(Exception):
        pass

    class Connection:
        commit_calls = 0
        rollback_calls = 0

        def commit(self):
            self.commit_calls += 1

        def rollback(self):
            self.rollback_calls += 1

    class Cursor:
        def execute(self, sql, params=None):
            adapter.sql_calls.append(("execute", sql, params))
            if params[0] == 2:
                raise DriverError(1366, "Incorrect value")

    connection = Connection()

    @contextmanager
    def connection_cursor():
        yield connection, Cursor()

    adapter.connection_cursor = connection_cursor

    from hscredit.database.exceptions import DatabaseQueryError

    with pytest.raises(DatabaseQueryError, match="MySQL追加写入失败") as caught:
        adapter.write_batch(
            "risk.events",
            pd.DataFrame({"id": [1, 2], "name": ["先处理", "失败"]}),
            "a",
            1,
            key_columns=["id"],
        )

    message = str(caught.value)
    assert "执行SQL:\nINSERT INTO `risk`.`events` (`id`, `name`) VALUES (%s, %s)" in message
    assert "数据库错误: DriverError: (1366, 'Incorrect value')" in message
    assert "先处理" not in message
    assert caught.value.params == [(1, "先处理"), (2, "失败")]
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1


def test_mysql_drop_mode_validates_ddl_before_drop(adapter):
    frame = pd.DataFrame({"id": [1]})

    with pytest.raises(Exception, match="数据类型"):
        adapter.prepare_write(
            "risk.events",
            "d",
            frame,
            dialect_options={"column_types": {"id": "BIGINT); DROP TABLE risk.events; --"}},
        )

    assert adapter.sql_calls == []


def test_mysql_json_projection_uses_server_side_path_extraction(adapter):
    sql = adapter.build_json_projection_sql(
        "select id, payload from risk.events;",
        columns=["id"],
        json_fields={"payload": {"customer_id": "$.customer.id"}},
    )

    assert sql == (
        "SELECT `hscredit_json_source`.`id`, "
        "CASE WHEN JSON_TYPE(JSON_EXTRACT(`hscredit_json_source`.`payload`, '$.customer.id')) = 'NULL' "
        "THEN NULL ELSE JSON_UNQUOTE(JSON_EXTRACT(`hscredit_json_source`.`payload`, '$.customer.id')) END "
        "AS `customer_id` "
        "FROM (select id, payload from risk.events) `hscredit_json_source`"
    )
