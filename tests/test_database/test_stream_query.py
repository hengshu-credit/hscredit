"""可中断流式读取、进度计数和部分结果测试。"""

import pandas as pd
import pytest

from hscredit.database import Database, StreamState, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.exceptions import DatabaseQueryError
from hscredit.exceptions import StateError, ValidationError


class ObservableStreamResource:
    def __init__(self, rows, columns, *, interrupt_on_call=None, fail_on_call=None):
        self.rows = list(rows)
        self.columns = list(columns)
        self.interrupt_on_call = interrupt_on_call
        self.fail_on_call = fail_on_call
        self.position = 0
        self.fetch_calls = 0
        self.closed = False

    def fetchmany(self, size):
        self.fetch_calls += 1
        if self.interrupt_on_call == self.fetch_calls:
            raise KeyboardInterrupt
        if self.fail_on_call == self.fetch_calls:
            raise RuntimeError("stream failed")
        if self.rows and isinstance(self.rows[0], pd.DataFrame):
            if self.position:
                return pd.DataFrame(columns=self.rows[0].columns)
            self.position = 1
            return self.rows[0]
        start = self.position
        self.position += size
        return self.rows[start : start + size]

    def close(self):
        self.closed = True


class ObservableStreamAdapter(BaseDatabaseAdapter):
    database_type = "observable_stream"

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        rows = connect_kwargs.get("rows", [(1,), (2,), (3,), (4,), (5,)])
        self.resource = ObservableStreamResource(
            rows,
            ["id"],
            interrupt_on_call=connect_kwargs.get("interrupt_on_call"),
            fail_on_call=connect_kwargs.get("fail_on_call"),
        )
        self.count_calls = []

    def open_stream(self, sql, params=None):
        self.stream_call = (sql, params)
        return self.resource

    def count_rows(self, sql, params=None):
        self.count_calls.append((sql, params))
        return len(self.resource.rows)


@pytest.fixture(autouse=True)
def register_stream_adapter():
    register_adapter("observable_stream", ObservableStreamAdapter, replace=True)


def test_progress_false_never_executes_count_query():
    database = Database("observable_stream")

    chunks = list(database.stream_query("select id from t", chunksize=2, progress=False))

    assert [chunk["id"].tolist() for chunk in chunks] == [[1, 2], [3, 4], [5]]
    assert database.adapter.count_calls == []
    assert database.adapter.resource.closed is True


def test_progress_true_counts_with_count_one_and_bound_params():
    database = Database("observable_stream")

    list(
        database.stream_query(
            "select id from t where id > %s;",
            params=(1,),
            chunksize=2,
            progress=True,
        )
    )

    assert database.adapter.count_calls == [
        (
            "SELECT COUNT(1) FROM (select id from t where id > %s) hscredit_count",
            (1,),
        )
    ]


def test_explicit_total_rows_skips_count_query():
    database = Database("observable_stream")

    list(database.stream_query("select id from t", progress=True, total_rows=5))

    assert database.adapter.count_calls == []


def test_custom_count_sql_is_executed_verbatim():
    database = Database("observable_stream")

    list(
        database.stream_query(
            "select id from t",
            progress=True,
            count_sql="select 5",
        )
    )

    assert database.adapter.count_calls == [("select 5", None)]


def test_read_query_returns_retained_rows_after_keyboard_interrupt():
    database = Database("observable_stream", interrupt_on_call=2)

    frame = database.read_query("select id from t", chunksize=2)

    assert frame["id"].tolist() == [1, 2]
    assert frame.attrs["completed"] is False
    assert frame.attrs["rows_read"] == 2
    assert frame.attrs["state"] == StreamState.INTERRUPTED.value
    assert frame.attrs["interrupt_reason"] == "KeyboardInterrupt"
    assert database.adapter.resource.closed is True


def test_stop_returns_current_retained_dataframe():
    database = Database("observable_stream")
    stream = database.stream_query("select id from t", chunksize=2)

    first = next(stream)
    stream.stop("用户主动停止")
    remaining = list(stream)
    partial = stream.to_dataframe()

    assert first["id"].tolist() == [1, 2]
    assert remaining == []
    assert partial["id"].tolist() == [1, 2]
    assert partial.attrs["state"] == StreamState.INTERRUPTED.value
    assert partial.attrs["interrupt_reason"] == "用户主动停止"


def test_retain_false_keeps_constant_memory_and_disables_merge():
    database = Database("observable_stream")
    stream = database.stream_query("select id from t", chunksize=2, retain=False)

    assert sum(len(chunk) for chunk in stream) == 5
    with pytest.raises(StateError, match="retain=False"):
        stream.to_dataframe()


def test_stream_failure_closes_resource_and_raises_query_error():
    database = Database("observable_stream", fail_on_call=2)
    stream = database.stream_query("select id from t", chunksize=2)

    assert next(stream)["id"].tolist() == [1, 2]
    with pytest.raises(DatabaseQueryError, match="流式读取失败") as caught:
        next(stream)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert stream.state is StreamState.FAILED
    assert database.adapter.resource.closed is True


@pytest.mark.parametrize("chunksize", [0, -1, 1.5, True])
def test_stream_query_rejects_invalid_chunksize(chunksize):
    database = Database("observable_stream")

    with pytest.raises(ValidationError, match="chunksize"):
        database.stream_query("select id from t", chunksize=chunksize)


def test_native_dataframe_chunks_are_not_reconstructed():
    native_chunk = pd.DataFrame({"id": [8, 9]})
    database = Database("observable_stream", rows=[native_chunk])

    chunks = list(database.stream_query("select id from native", chunksize=10))

    assert len(chunks) == 1
    assert chunks[0] is native_chunk
