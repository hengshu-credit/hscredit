"""共享 DB-API 连接池、查询和事务测试。"""

import traceback

import pandas as pd
import pytest

from hscredit.database import PoolOptions
from hscredit.database.adapters.dbapi import DBAPIAdapter
from hscredit.database.exceptions import DatabaseQueryError
from hscredit.exceptions import ValidationError

from .fakes import FakeDBAPIDriver, FakeDBAPIState, FakePooledDB


class ObservableDBAPIAdapter(DBAPIAdapter):
    database_type = "fake_dbapi"

    def __init__(self, state, **kwargs):
        self.state = state
        super().__init__(**kwargs)

    def load_driver(self):
        return FakeDBAPIDriver(self.state)

    def load_pool_class(self):
        return FakePooledDB


@pytest.fixture
def state():
    return FakeDBAPIState(rows=[(2, "李四")], columns=["id", "name"])


@pytest.fixture
def adapter(state):
    return ObservableDBAPIAdapter(
        state,
        connect_kwargs={"host": "db.internal", "database": "risk"},
        pool_options=PoolOptions(maxconnections=4, blocking=True),
        adapter_options={},
    )


def test_query_binds_params_and_returns_dataframe(adapter, state):
    frame = adapter.query(
        "select id, name from users where id > %s",
        params=(1,),
        result="dataframe",
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict("records") == [{"id": 2, "name": "李四"}]
    assert state.cursors[-1].executed == (
        "select id, name from users where id > %s",
        (1,),
    )
    assert state.cursors[-1].closed is True
    assert state.connections[-1].close_calls == 1


def test_query_supports_records_rows_and_empty_dataframe(adapter, state):
    assert adapter.query("select id, name from users", result="records") == [{"id": 2, "name": "李四"}]
    assert adapter.query("select id, name from users", result="rows") == [(2, "李四")]

    state.rows = []
    empty = adapter.query("select id, name from users", result="dataframe")
    assert empty.empty
    assert empty.columns.tolist() == ["id", "name"]


def test_query_rejects_unknown_result_before_execution(adapter, state):
    before = len(state.cursors)

    with pytest.raises(ValidationError, match="result"):
        adapter.query("select 1", result="mapping")

    assert len(state.cursors) == before


def test_query_error_displays_sql_and_driver_error_without_params(adapter, state):
    driver_error = RuntimeError("1064 syntax error near prefixsensitive-tokensuffix")
    state.fail_execute = driver_error
    sql = "select id from users where token=%s"
    params = ("sensitive-token",)

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query(sql, params=params)

    message = str(caught.value)
    assert f"执行SQL:\n{sql}" in message
    assert "数据库错误: RuntimeError: [已脱敏：数据库错误包含绑定参数]" in message
    assert "sensitive-token" not in message
    assert "sensitive-token" not in "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert caught.value.sql == sql
    assert caught.value.params == params
    assert caught.value.driver_error is driver_error


def test_execute_commits_and_returns_rowcount(adapter, state):
    affected = adapter.execute("update users set name=%s", params=("王五",))

    assert affected == 1
    assert state.connections[-1].commit_calls == 1
    assert state.connections[-1].rollback_calls == 0
    assert state.connections[-1].close_calls == 1


def test_execute_rolls_back_and_preserves_original_error(adapter, state):
    driver_error = RuntimeError("driver failed")
    state.fail_execute = driver_error
    sql = "update users set name=%s"
    params = ("王五",)

    with pytest.raises(DatabaseQueryError, match="SQL执行失败") as caught:
        adapter.execute(sql, params=params)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert f"执行SQL:\n{sql}" in str(caught.value)
    assert "数据库错误: RuntimeError: driver failed" in str(caught.value)
    assert "王五" not in str(caught.value)
    assert caught.value.sql == sql
    assert caught.value.params == params
    assert caught.value.driver_error is driver_error
    assert state.connections[-1].commit_calls == 0
    assert state.connections[-1].rollback_calls == 1
    assert state.connections[-1].close_calls == 1
    assert state.cursors[-1].closed is True


def test_executemany_materializes_values_once_and_commits(adapter, state):
    values = ((value, f"用户{value}") for value in (3, 4))

    affected = adapter.executemany(
        "insert into users(id, name) values (%s, %s)",
        values,
    )

    assert affected == 2
    assert state.cursors[-1].executemany_call[1] == [(3, "用户3"), (4, "用户4")]
    assert state.connections[-1].commit_calls == 1


def test_executemany_error_keeps_materialized_params_out_of_message(adapter, state):
    driver_error = RuntimeError("1406 Data too long for column name")
    state.fail_executemany = driver_error
    sql = "insert into users(id, name) values (%s, %s)"
    values = [(3, "sensitive-name")]

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.executemany(sql, iter(values))

    message = str(caught.value)
    assert f"执行SQL:\n{sql}" in message
    assert "数据库错误: RuntimeError: 1406 Data too long for column name" in message
    assert "sensitive-name" not in message
    assert caught.value.params == values
    assert caught.value.driver_error is driver_error


def test_executemany_redacts_driver_echo_from_late_large_batch_row(adapter, state):
    secret = "late-sensitive-value"
    driver_error = RuntimeError(f"invalid value: {secret}")
    state.fail_executemany = driver_error
    values = [(index,) for index in range(299)] + [(secret,)]

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.executemany("insert into events values (%s)", values)

    assert secret not in str(caught.value)
    assert secret not in "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )


def test_query_redacts_deeply_nested_bound_value(adapter, state):
    secret = "deep-sensitive-value"
    params = secret
    for _ in range(8):
        params = [params]
    state.fail_execute = RuntimeError(f"invalid value: {secret}")

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query("select %s", params=params)

    assert secret not in str(caught.value)


def test_unprintable_bound_parameter_does_not_replace_database_error(adapter, state):
    class UnprintableParameter:
        def __str__(self):
            raise RuntimeError("parameter string conversion failed")

    driver_error = RuntimeError("database syntax failure")
    state.fail_execute = driver_error

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query("select %s", params=(UnprintableParameter(),))

    assert caught.value.driver_error is driver_error
    assert "SQL查询失败" in str(caught.value)
    assert "[已脱敏：数据库错误包含绑定参数]" in str(caught.value)


def test_invalid_utf8_bytes_parameter_is_conservatively_redacted(adapter, state):
    secret = b"\xff"
    state.fail_execute = RuntimeError("invalid bound value b'\\xff'")

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query("select %s", params=(secret,))

    message = str(caught.value)
    assert "b'\\xff'" not in message
    assert "[已脱敏：数据库错误包含绑定参数]" in message
    assert "b'\\xff'" not in "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )


@pytest.mark.parametrize(
    ("value", "echoed"),
    [
        (b"\x00", "b'\\x00'"),
        (b"secret\n", "b'secret\\n'"),
    ],
)
def test_valid_bytes_parameter_repr_is_conservatively_redacted(adapter, state, value, echoed):
    state.fail_execute = RuntimeError(f"invalid bound value {echoed}")

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query("select %s", params=(value,))

    message = str(caught.value)
    assert echoed not in message
    assert "[已脱敏：数据库错误包含绑定参数]" in message
    assert echoed not in "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )


def test_broken_mapping_parameter_does_not_replace_database_error(adapter, state):
    class BrokenMapping(dict):
        def values(self):
            raise RuntimeError("mapping traversal failed")

    driver_error = RuntimeError("database syntax failure")
    state.fail_execute = driver_error

    with pytest.raises(DatabaseQueryError) as caught:
        adapter.query("select %(value)s", params=BrokenMapping(value="secret"))

    assert caught.value.driver_error is driver_error
    assert "SQL查询失败" in str(caught.value)
    assert "[已脱敏：数据库错误包含绑定参数]" in str(caught.value)


def test_pool_receives_pool_and_connection_options_separately(adapter, state):
    del adapter

    assert state.pool_kwargs["host"] == "db.internal"
    assert state.pool_kwargs["database"] == "risk"
    assert state.pool_kwargs["maxconnections"] == 4
    assert state.pool_kwargs["blocking"] is True


def test_close_closes_pool_once(adapter, state):
    adapter.close()
    adapter.close()

    assert state.pool_closed is True
