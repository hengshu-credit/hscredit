"""共享 DB-API 连接池、查询和事务测试。"""

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
    assert adapter.query("select id, name from users", result="records") == [
        {"id": 2, "name": "李四"}
    ]
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


def test_execute_commits_and_returns_rowcount(adapter, state):
    affected = adapter.execute("update users set name=%s", params=("王五",))

    assert affected == 1
    assert state.connections[-1].commit_calls == 1
    assert state.connections[-1].rollback_calls == 0
    assert state.connections[-1].close_calls == 1


def test_execute_rolls_back_and_preserves_original_error(adapter, state):
    state.fail_execute = RuntimeError("driver failed")

    with pytest.raises(DatabaseQueryError, match="SQL执行失败") as caught:
        adapter.execute("update users set name=%s", params=("王五",))

    assert isinstance(caught.value.__cause__, RuntimeError)
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
