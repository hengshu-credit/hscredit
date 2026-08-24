"""数据库快捷操作的数据源解析、委托和资源所有权测试。"""

import pandas as pd
import pytest

import hscredit.database.shortcuts as shortcut_api
from hscredit.database import Database, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.adapters.dbapi import DBAPIAdapter
from hscredit.database.exceptions import DatabaseCapabilityError, DatabaseQueryError
from hscredit.database.shortcuts import execute, query
from hscredit.database.metadata import MetadataInspection
from hscredit.database.writing import BatchWriteResult
from hscredit.exceptions import ValidationError


class ShortcutAdapter(BaseDatabaseAdapter):
    database_type = "shortcut"
    instances = []

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    def query(self, sql, params=None, result="dataframe"):
        self.calls.append(("query", sql, params, result))
        return {"sql": sql, "params": params, "result": result}

    def execute(self, sql, params=None):
        self.calls.append(("execute", sql, params))
        return 3

    def executemany(self, sql, values):
        values = list(values)
        self.calls.append(("executemany", sql, values))
        return len(values)

    def open_stream(self, sql, params=None):
        self.calls.append(("open_stream", sql, params))
        if self.connect_kwargs.get("fail_transform"):
            return PostFetchFailureResource()
        return FakeStreamResource(
            [(1,), (2,)],
            ["id"],
            fail_on_call=1 if self.connect_kwargs.get("fail_stream") else None,
        )

    def count_rows(self, sql, params=None):
        self.calls.append(("count_rows", sql, params))
        return 2

    def inspect_schema(self, targets):
        self.calls.append(("inspect_schema", targets))
        return MetadataInspection(
            rows=[
                {
                    "database_type": "shortcut",
                    "database": "risk",
                    "table_name": "events",
                    "column_name": "id",
                }
            ]
        )

    def create_table(self, data, table_name, *, dialect_options=None):
        self.calls.append(("create_table", data.copy(), table_name, dialect_options))
        return "CREATE TABLE"

    def prepare_write(self, table_name, mode, first_batch, *, key_columns=None, dialect_options=None):
        self.calls.append(("prepare_write", table_name, mode, first_batch.copy()))

    def write_batch(
        self,
        table_name,
        batch,
        mode,
        batch_index,
        *,
        key_columns=None,
        dialect_options=None,
    ):
        self.calls.append(("write_batch", table_name, mode, batch_index, batch.copy()))
        return BatchWriteResult(inserted=len(batch), updated=0, skipped=0)

    def finish_write(self, table_name, mode, result, *, dialect_options=None):
        self.calls.append(("finish_write", table_name, mode, result.rows_received))

    def _nosql(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return {"method": name, "args": args, "options": kwargs}

    def read_one(self, *args, **kwargs):
        return self._nosql("read_one", *args, **kwargs)

    def read_many(self, *args, **kwargs):
        return self._nosql("read_many", *args, **kwargs)

    def read(self, *args, **kwargs):
        return self._nosql("read", *args, **kwargs)

    def write_one(self, *args, **kwargs):
        return self._nosql("write_one", *args, **kwargs)

    def write_many(self, *args, **kwargs):
        return self._nosql("write_many", *args, **kwargs)

    def write(self, *args, **kwargs):
        return self._nosql("write", *args, **kwargs)

    def delete_one(self, *args, **kwargs):
        return self._nosql("delete_one", *args, **kwargs)

    def delete_many(self, *args, **kwargs):
        return self._nosql("delete_many", *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._nosql("delete", *args, **kwargs)

    def exists(self, *args, **kwargs):
        self._nosql("exists", *args, **kwargs)
        return True

    def close(self):
        self.close_calls += 1
        super().close()


class ShortcutDBAPIAdapter(DBAPIAdapter):
    database_type = "shortcut_dbapi"

    def load_driver(self):  # pragma: no cover - 原生连接快捷路径不得调用
        raise AssertionError("原生连接不应重新加载或创建驱动连接")


class FakeStreamResource:
    def __init__(self, rows, columns, *, fail_on_call=None):
        self.rows = list(rows)
        self.columns = list(columns)
        self.fail_on_call = fail_on_call
        self.position = 0
        self.fetch_calls = 0
        self.closed = False

    def fetchmany(self, size):
        self.fetch_calls += 1
        if self.fetch_calls == self.fail_on_call:
            raise RuntimeError("stream failed")
        start = self.position
        self.position += size
        return self.rows[start : start + size]

    def close(self):
        self.closed = True


class BrokenBatch:
    def __len__(self):
        return 1

    def __iter__(self):
        raise RuntimeError("transform failed")


class PostFetchFailureResource:
    columns = ["id"]

    def __init__(self):
        self.closed = False

    def fetchmany(self, size):
        del size
        return BrokenBatch()

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, rows=((1,), (2,))):
        self.rows = list(rows)
        self.description = [("id",)]
        self.position = 0
        self.closed = False
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.last_execute = (sql, params)
        self.rowcount = len(self.rows)

    def executemany(self, sql, values):
        self.last_executemany = (sql, list(values))
        self.rowcount = len(self.last_executemany[1])

    def fetchall(self):
        return list(self.rows)

    def fetchmany(self, size):
        start = self.position
        self.position += size
        return self.rows[start : start + size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursors = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def cursor(self, *args, **kwargs):
        del args, kwargs
        cursor = FakeCursor()
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1

    def close(self):
        self.close_calls += 1


class PyMySQLLikeConnection(FakeConnection):
    pass


PyMySQLLikeConnection.__module__ = "pymysql.connections"


class OracleLikeConnection(FakeConnection):
    pass


OracleLikeConnection.__module__ = "oracledb.connection"


class ImpalaLikeConnection(FakeConnection):
    pass


ImpalaLikeConnection.__module__ = "impala.hiveserver2"


class MaxComputeLikeConnection(FakeConnection):
    pass


MaxComputeLikeConnection.__module__ = "odps.dbapi"


@pytest.fixture(autouse=True)
def register_shortcut_adapter():
    ShortcutAdapter.instances.clear()
    register_adapter("shortcut", ShortcutAdapter, aliases=["shortcut_alias"], replace=True)
    register_adapter("shortcut_dbapi", ShortcutDBAPIAdapter, replace=True)


def test_config_source_creates_database_and_closes_it_after_query():
    result = query(
        {"db_type": "shortcut", "account": "reader"},
        "select id from events where id=%s",
        params=(7,),
        result="records",
    )

    adapter = ShortcutAdapter.instances[-1]
    assert result == {
        "sql": "select id from events where id=%s",
        "params": (7,),
        "result": "records",
    }
    assert adapter.connect_kwargs == {"account": "reader"}
    assert adapter.closed is True
    assert adapter.close_calls == 1


def test_database_source_is_borrowed_and_not_closed():
    database = Database("shortcut")

    assert execute(database, "delete from events") == 3

    assert database.closed is False
    assert database.adapter.close_calls == 0
    database.close()


@pytest.mark.parametrize(
    ("source", "db_type", "message"),
    [
        ({"host": "localhost"}, None, "db_type"),
        ({"db_type": "shortcut"}, "mysql", "冲突"),
        (Database, None, "source"),
    ],
)
def test_invalid_or_conflicting_sources_are_rejected(source, db_type, message):
    with pytest.raises(ValidationError, match=message):
        query(source, "select 1", db_type=db_type)


def test_native_dbapi_connection_is_borrowed_for_query_and_execute():
    connection = FakeConnection()

    records = query(connection, "select id from events", result="records", db_type="shortcut_dbapi")
    affected = execute(connection, "delete from events", db_type="shortcut_dbapi")
    batch_affected = shortcut_api.executemany(
        connection,
        "insert into events values (%s)",
        [(1,), (2,)],
        db_type="shortcut_dbapi",
    )

    assert records == [{"id": 1}, {"id": 2}]
    assert affected == 2
    assert batch_affected == 2
    assert connection.commit_calls == 2
    assert connection.rollback_calls == 0
    assert connection.close_calls == 0
    assert all(cursor.closed for cursor in connection.cursors)


def test_unknown_native_connection_requires_db_type():
    with pytest.raises(ValidationError, match="db_type"):
        query(FakeConnection(), "select 1")


def test_pymysql_connection_type_is_inferred_without_db_type():
    connection = PyMySQLLikeConnection()

    rows = query(connection, "select id from events", result="rows")

    assert rows == [(1,), (2,)]
    assert connection.close_calls == 0


@pytest.mark.parametrize(
    "connection_class",
    [OracleLikeConnection, ImpalaLikeConnection, MaxComputeLikeConnection],
)
def test_common_dbapi_connection_types_are_inferred_for_query(connection_class):
    connection = connection_class()

    rows = query(connection, "select id from events", result="rows")

    assert rows == [(1,), (2,)]
    assert connection.close_calls == 0


def test_pymysql_connection_can_explicitly_use_starrocks_dialect():
    connection = PyMySQLLikeConnection()

    rows = query(connection, "select id from events", result="rows", db_type="starrocks")

    assert rows == [(1,), (2,)]
    assert connection.close_calls == 0


def test_native_mysql_connection_supports_create_table_and_stream_write():
    connection = PyMySQLLikeConnection()

    ddl = shortcut_api.create_table(
        connection,
        pd.DataFrame({"id": [1]}),
        "risk.events",
    )
    result = shortcut_api.stream_write(
        connection,
        pd.DataFrame({"id": [1, 2]}),
        "risk.events",
        mode="d",
    )

    assert ddl.startswith("CREATE TABLE")
    assert result.completed is True
    assert result.rows_received == 2
    assert connection.close_calls == 0


def test_native_dbapi_stream_closes_cursor_but_not_connection():
    connection = FakeConnection()

    stream = shortcut_api.stream_query(
        connection,
        "select id from events",
        chunksize=1,
        db_type="shortcut_dbapi",
    )
    chunks = list(stream)
    frame = stream.to_dataframe()

    assert [chunk["id"].tolist() for chunk in chunks] == [[1], [2]]
    assert frame["id"].tolist() == [1, 2]
    assert connection.cursors[-1].closed is True
    assert connection.close_calls == 0


@pytest.mark.parametrize("method_name", ["export_schema", "stream_write"])
def test_native_maxcompute_connection_rejects_operations_that_require_odps_entry(method_name):
    connection = FakeConnection()

    with pytest.raises(DatabaseCapabilityError, match="MaxCompute.*配置|Database"):
        if method_name == "export_schema":
            shortcut_api.export_schema(connection, db_type="maxcompute")
        else:
            shortcut_api.stream_write(
                connection,
                pd.DataFrame({"id": [1]}),
                "risk.events",
                db_type="maxcompute",
            )

    assert connection.close_calls == 0


def test_db_type_aliases_are_compared_by_canonical_name():
    database = Database("shortcut")
    try:
        result = query(database, "select 1", db_type="shortcut_alias", result="rows")

        assert result["sql"] == "select 1"
    finally:
        database.close()


def test_sql_shortcuts_delegate_read_schema_create_and_stream_write():
    config = {"db_type": "shortcut"}

    frame = shortcut_api.read_query(config, "select id from events")
    schema = shortcut_api.export_schema(config)
    ddl = shortcut_api.create_table(config, pd.DataFrame({"id": [1]}), "risk.events")
    write_result = shortcut_api.stream_write(
        config,
        pd.DataFrame({"id": [1, 2]}),
        "risk.events",
        mode="d",
    )

    assert frame["id"].tolist() == [1, 2]
    assert schema[["数据库类型", "数据库名", "表名", "字段名"]].iloc[0].tolist() == [
        "shortcut",
        "risk",
        "events",
        "id",
    ]
    assert ddl == "CREATE TABLE"
    assert write_result.completed is True
    assert write_result.rows_inserted == 2


@pytest.mark.parametrize(
    ("method_name", "args", "options"),
    [
        ("read_one", ("events", {"id": 1}), {"projection": {"id": 1}}),
        ("read_many", ("events", {"state": "ok"}), {"limit": 10}),
        ("read", ("events", None), {"many": True}),
        ("write_one", ("events", {"id": 1}), {}),
        ("write_many", ("events", [{"id": 1}]), {}),
        ("write", ("events", {"id": 1}), {"many": False}),
        ("delete_one", ("events", {"id": 1}), {}),
        ("delete_many", ("events", {"state": "old"}), {}),
        ("delete", ("events", {"id": 1}), {"many": False}),
    ],
)
def test_nosql_shortcuts_delegate_to_same_named_database_methods(method_name, args, options):
    database = Database("shortcut")

    result = getattr(shortcut_api, method_name)(database, *args, **options)

    assert result == {"method": method_name, "args": args, "options": options}
    assert database.closed is False
    database.close()


def test_exists_shortcut_returns_boolean():
    database = Database("shortcut")

    assert shortcut_api.exists(database, "events", {"id": 1}) is True

    assert database.closed is False
    database.close()


def test_owned_stream_database_closes_on_completion_stop_and_close():
    completed = shortcut_api.stream_query({"db_type": "shortcut"}, "select id from events")
    completed_adapter = ShortcutAdapter.instances[-1]
    assert completed_adapter.closed is False
    list(completed)
    assert completed_adapter.close_calls == 1

    stopped = shortcut_api.stream_query({"db_type": "shortcut"}, "select id from events", chunksize=1)
    stopped_adapter = ShortcutAdapter.instances[-1]
    next(stopped)
    stopped.stop("完成抽样")
    stopped.close()
    assert stopped_adapter.close_calls == 1

    closed = shortcut_api.stream_query({"db_type": "shortcut"}, "select id from events")
    closed_adapter = ShortcutAdapter.instances[-1]
    closed.close()
    closed.close()
    assert closed_adapter.close_calls == 1


def test_owned_stream_database_closes_when_fetch_fails():
    stream = shortcut_api.stream_query(
        {"db_type": "shortcut", "fail_stream": True},
        "select id from events",
    )
    adapter = ShortcutAdapter.instances[-1]

    with pytest.raises(DatabaseQueryError, match="流式读取失败"):
        next(stream)

    assert adapter.close_calls == 1


def test_owned_stream_database_closes_when_batch_transform_fails():
    stream = shortcut_api.stream_query(
        {"db_type": "shortcut", "fail_transform": True},
        "select id from events",
    )
    adapter = ShortcutAdapter.instances[-1]

    with pytest.raises(DatabaseQueryError, match="流式读取失败") as caught:
        next(stream)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert adapter.close_calls == 1
