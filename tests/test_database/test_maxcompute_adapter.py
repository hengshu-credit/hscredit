"""MaxCompute DB-API、原生写入、MERGE 和元数据测试。"""

from types import SimpleNamespace

import pandas as pd
import pytest

from hscredit.database import Database, PoolOptions, register_adapter
from hscredit.database.adapters.maxcompute import MaxComputeAdapter
from hscredit.database.exceptions import DatabaseCapabilityError, DatabaseWriteError
from hscredit.database.metadata import QualifiedTarget

from .fakes import FakeDBAPIDriver, FakeDBAPIState, FakePooledDB


class FakeColumn:
    def __init__(self, name, data_type, comment=None, nullable=None):
        self.name = name
        self.type = data_type
        self.comment = comment
        self.nullable = nullable


class FakeTable:
    def __init__(self, project="risk", schema="default", name="events"):
        self.project = project
        self.schema_name = schema
        self.name = name
        self.type = "MANAGED_TABLE"
        self.comment = "事件表"
        self.table_schema = SimpleNamespace(
            columns=[FakeColumn("id", "bigint", "编号", True)],
            partitions=[FakeColumn("pt", "string", "分区", False)],
        )


class FakeODPS:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.tables = [FakeTable()]

    def write_table(self, table_name, frame, **kwargs):
        self.calls.append(("write_table", table_name, frame.copy(), dict(kwargs)))

    def delete_table(self, table_name, **kwargs):
        self.calls.append(("delete_table", table_name, dict(kwargs)))

    def list_tables(self, project=None, schema=None):
        self.calls.append(("list_tables", project, schema))
        return iter(self.tables)

    def get_table(self, table_name, project=None, schema=None):
        self.calls.append(("get_table", table_name, project, schema))
        return self.tables[0]


class ObservableMaxComputeAdapter(MaxComputeAdapter):
    odps_entries = []

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        self.state = FakeDBAPIState()
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.sql_calls = []

    def load_odps_module(self):
        parent = self

        class ODPSFactory(FakeODPS):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                parent.odps_entries.append(self)

        return SimpleNamespace(
            dbapi=FakeDBAPIDriver(self.state),
            ODPS=ODPSFactory,
        )

    def load_pool_class(self):
        return FakePooledDB

    def execute(self, sql, params=None):
        self.sql_calls.append((sql, params))
        return 0


@pytest.fixture(autouse=True)
def register_maxcompute_adapter():
    ObservableMaxComputeAdapter.odps_entries.clear()
    register_adapter("observable_maxcompute", ObservableMaxComputeAdapter, replace=True)


@pytest.fixture
def adapter():
    return ObservableMaxComputeAdapter(
        connect_kwargs={
            "access_id": "id",
            "access_key": "secret",
            "project": "risk",
            "endpoint": "https://service.odps.example/api",
            "schema": "default",
        },
        pool_options=PoolOptions(maxconnections=2),
        adapter_options={},
    )


def test_maxcompute_declares_no_transactions(adapter):
    assert adapter.capabilities.transactions is False


def test_maxcompute_builds_dbapi_pool_and_native_entry_without_eager_dependency(adapter):
    assert adapter.state.pool_kwargs["project"] == "risk"
    assert adapter.odps.kwargs["project"] == "risk"
    assert adapter.odps.kwargs["schema"] == "default"


def test_maxcompute_append_with_primary_key_guarantee_is_rejected(adapter):
    with pytest.raises(DatabaseCapabilityError, match="主键冲突"):
        adapter.prepare_write(
            "risk.events",
            "a",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={},
        )


def test_maxcompute_replace_requires_transactional_table_and_keys(adapter):
    with pytest.raises(DatabaseCapabilityError, match="事务表"):
        adapter.prepare_write(
            "risk.events",
            "r",
            pd.DataFrame({"id": [1]}),
            key_columns=["id"],
            dialect_options={"transactional": False},
        )


def test_maxcompute_overwrite_only_applies_to_first_stream_batch():
    database = Database(
        "observable_maxcompute",
        access_id="id",
        access_key="secret",
        project="risk",
        endpoint="https://service.odps.example/api",
    )

    result = database.stream_write(
        iter([pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [2]})]),
        "risk.events",
        mode="o",
    )

    writes = [call for call in database.adapter.odps.calls if call[0] == "write_table"]
    assert writes[0][3]["overwrite"] is True
    assert writes[1][3]["overwrite"] is False
    assert result.rows_inserted is None


def test_maxcompute_drop_mode_recreates_schema_before_native_write(adapter):
    frame = pd.DataFrame({"id": [1], "name": ["A"]})

    adapter.prepare_write(
        "risk.events",
        "d",
        frame,
        dialect_options={"lifecycle": 30},
    )

    assert adapter.odps.calls[0] == (
        "delete_table",
        "risk.events",
        {"if_exists": True},
    )
    assert adapter.sql_calls[0][0].startswith("CREATE TABLE `risk`.`events`")
    assert "LIFECYCLE 30" in adapter.sql_calls[0][0]


def test_maxcompute_transactional_replace_uses_staging_merge_and_cleanup(adapter):
    batch = pd.DataFrame({"id": [1], "name": ["覆盖"]})

    result = adapter.write_batch(
        "risk.events",
        batch,
        "r",
        1,
        key_columns=["id"],
        dialect_options={"transactional": True},
    )

    assert result.inserted is None
    assert result.updated is None
    staging_write = next(call for call in adapter.odps.calls if call[0] == "write_table")
    assert staging_write[3]["create_table"] is True
    assert staging_write[3]["overwrite"] is True
    assert any(sql.startswith("MERGE INTO `risk`.`events`") for sql, _ in adapter.sql_calls)
    assert adapter.odps.calls[-1][0] == "delete_table"


def test_maxcompute_write_failure_reports_committed_batches():
    class FailingAdapter(ObservableMaxComputeAdapter):
        def write_batch(self, *args, **kwargs):
            batch_index = args[3]
            if batch_index == 2:
                raise RuntimeError("native write failed")
            return super().write_batch(*args, **kwargs)

    register_adapter("failing_maxcompute", FailingAdapter, replace=True)
    database = Database(
        "failing_maxcompute",
        access_id="id",
        access_key="secret",
        project="risk",
        endpoint="https://service.odps.example/api",
    )

    with pytest.raises(DatabaseWriteError) as caught:
        database.stream_write(
            iter([pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [2]})]),
            "risk.events",
            mode="a",
        )

    assert caught.value.result.batches_committed == 1
    assert caught.value.result.failed_batch == 2


def test_maxcompute_metadata_includes_columns_and_partition_fields(adapter):
    inspection = adapter.inspect_schema((QualifiedTarget.parse("risk.default.events"),))

    assert [row["column_name"] for row in inspection.rows] == ["id", "pt"]
    assert inspection.rows[0]["table_type"] == "MANAGED_TABLE"
    assert inspection.rows[0]["data_type"] == "bigint"
    assert inspection.rows[1]["partition_key"] is True
    assert adapter.odps.calls[-1] == (
        "get_table",
        "events",
        "risk",
        "default",
    )


def test_maxcompute_drop_mode_validates_ddl_before_delete(adapter):
    with pytest.raises(Exception, match="数据类型"):
        adapter.prepare_write(
            "risk.events",
            "d",
            pd.DataFrame({"id": [1]}),
            dialect_options={"column_types": {"id": "BIGINT); DROP TABLE x; --"}},
        )

    assert adapter.odps.calls == []
    assert adapter.sql_calls == []
