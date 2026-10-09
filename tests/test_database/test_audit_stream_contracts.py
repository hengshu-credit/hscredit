"""流消费失败与返回批次的审计回归。"""
from types import SimpleNamespace
import pandas as pd
import pytest
from hscredit.database.client import Database
from hscredit.database.writing import BatchWriteResult
from hscredit.database.exceptions import DatabaseWriteError
from hscredit.database.adapters.clickhouse import ClickHouseQueryResource
from hscredit.database.stream import QueryStream


def test_iterator_failure_preserves_committed_counts():
    committed = []
    database = Database.__new__(Database)
    database._closed = False
    database.adapter = SimpleNamespace(table_exists=lambda *a: True,
        validate_write=lambda *a, **kw: None, prepare_write=lambda *a, **kw: None,
        ensure_table=lambda *a, **kw: None, finish_write=lambda *a, **kw: None,
        write_batch=lambda table, batch, *a, **kw: committed.append(len(batch)) or BatchWriteResult(inserted=len(batch)))
    def data():
        yield pd.DataFrame({'id': [1, 2]})
        yield pd.DataFrame({'other': [3]})
    with pytest.raises(DatabaseWriteError) as failure:
        database.stream_write(data(), 'fake_only', mode='o')
    assert sum(committed) == 2
    assert failure.value.result.batches_committed == 1
    assert failure.value.result.failed_batch == 2
    assert failure.value.result.rows_inserted == 2
    assert not failure.value.result.completed


def test_clickhouse_honors_user_chunk_size_and_does_not_drop_rows():
    resource = ClickHouseQueryResource(iter([pd.DataFrame({'x': range(7)}), pd.DataFrame({'x': [7, 8]})]))
    chunks = []
    stream = QueryStream(resource, chunksize=2, retain=True, result='records')
    for chunk in stream:
        assert len(chunk) <= 2
        chunks.extend(chunk)
    assert [row['x'] for row in chunks] == list(range(9))
    envelope = stream.to_envelope()
    assert envelope['完成'] is True
    assert envelope['已读取行数'] == 9
    assert envelope['结果'] == chunks
