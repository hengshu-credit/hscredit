"""共享流式写入编排、输入规范化和破坏性顺序测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.database import Database, DatabaseCapabilities, WriteResult, register_adapter, write
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.exceptions import DatabaseCapabilityError, DatabaseWriteError
from hscredit.database.writing import BatchWriteResult, iter_write_batches, validate_sql_type
from hscredit.exceptions import InputValidationError, ValidationError


class ObservableWriteAdapter(BaseDatabaseAdapter):
    database_type = "observable_write"
    capabilities = DatabaseCapabilities(write_modes={"a", "r", "o", "d"})

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        super().__init__(
            connect_kwargs=connect_kwargs,
            pool_options=pool_options,
            adapter_options=adapter_options,
        )
        self.calls = []
        self.create_attempts = 0
        self.fail_batch = connect_kwargs.get("fail_batch")
        self.race_create = connect_kwargs.get("race_create", False)
        self.unsupported_modes = set(connect_kwargs.get("unsupported_modes", ()))
        self.tables = set(connect_kwargs.get("existing_tables", {"risk.events"}))

    class ConcurrentTableExistsError(RuntimeError):
        pass

    def table_exists(self, table_name):
        return table_name in self.tables

    def create_table(self, data, table_name, *, dialect_options=None):
        self.create_attempts += 1
        options = dict(dialect_options or {})
        if table_name in self.tables and options.get("if_not_exists"):
            return "CREATE TABLE"
        if self.race_create:
            self.tables.add(table_name)
            raise self.ConcurrentTableExistsError("concurrent creator won")
        self.calls.append(("create_table", table_name, data.copy(), options))
        self.tables.add(table_name)
        return "CREATE TABLE"

    def is_table_already_exists_error(self, exc):
        return isinstance(exc, self.ConcurrentTableExistsError)

    def validate_write(
        self,
        table_name,
        mode,
        first_batch,
        *,
        key_columns=None,
        dialect_options=None,
        table_exists=None,
    ):
        del first_batch, key_columns, dialect_options, table_exists
        if mode in self.unsupported_modes:
            raise DatabaseCapabilityError(f"测试适配器不支持模式 {mode}")
        self.require_write_mode(table_name, mode)

    def prepare_write(
        self,
        table_name,
        mode,
        first_batch,
        *,
        key_columns=None,
        dialect_options=None,
    ):
        self.require_write_mode(table_name, mode)
        if mode != "d" and table_name not in self.tables:
            raise RuntimeError("table missing")
        self.calls.append(
            (
                "prepare",
                table_name,
                mode,
                first_batch.copy(),
                tuple(key_columns or ()),
                dict(dialect_options or {}),
            )
        )
        if mode == "o":
            self.calls.append(("clear", table_name))
        elif mode == "d":
            self.calls.append(("drop", table_name))
            self.tables.discard(table_name)
            self.create_table(
                first_batch,
                table_name,
                dialect_options=dialect_options,
            )

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
        del key_columns, dialect_options
        self.calls.append(("write", table_name, mode, batch_index, batch.copy()))
        if self.fail_batch == batch_index:
            raise RuntimeError("batch failed")
        return BatchWriteResult(inserted=len(batch), updated=0, skipped=0)

    def finish_write(self, table_name, mode, result, *, dialect_options=None):
        del dialect_options
        self.calls.append(("finish", table_name, mode, result.batches_committed))


@pytest.fixture(autouse=True)
def register_write_adapter():
    register_adapter("observable_write", ObservableWriteAdapter, replace=True)


@pytest.mark.parametrize("mode", ["a", "r", "o", "d"])
def test_write_pipeline_accepts_defined_modes(mode):
    database = Database("observable_write")

    result = database.stream_write(
        pd.DataFrame({"id": [1, 2]}),
        "risk.events",
        mode=mode,
        batch_size=1,
    )

    assert result.mode == mode
    assert result.completed is True
    assert result.rows_received == 2
    assert result.rows_inserted == 2
    assert result.batches_committed == 2
    assert database.adapter.calls[0][0] == "prepare"
    assert database.adapter.calls[-1] == ("finish", "risk.events", mode, 2)


def test_sql_write_uses_append_mode_by_default():
    database = Database("observable_write")

    result = database.write(
        "risk.events",
        pd.DataFrame({"id": [1, 2]}),
        key_columns="id",
    )

    assert result.mode == "a"
    assert result.completed is True
    assert result.rows_inserted == 2


@pytest.mark.parametrize("mode", ["a", "r", "o", "d"])
def test_sql_write_creates_missing_table_for_every_mode(mode):
    database = Database("observable_write", existing_tables=())

    result = database.write(
        "risk.events",
        pd.DataFrame({"id": [1], "name": ["新记录"]}),
        mode=mode,
        key_columns="id",
    )

    create_calls = [call for call in database.adapter.calls if call[0] == "create_table"]
    assert len(create_calls) == 1
    assert create_calls[0][1] == "risk.events"
    assert create_calls[0][3]["key_columns"] == ["id"]
    assert result.completed is True


def test_sql_write_shortcut_uses_same_table_creation_pipeline():
    result = write(
        {"db_type": "observable_write", "existing_tables": ()},
        "risk.events",
        pd.DataFrame({"id": [1]}),
        key_columns="id",
    )

    assert result.mode == "a"
    assert result.completed is True


def test_existing_sql_table_does_not_attempt_create_ddl():
    database = Database("observable_write")

    database.write(
        "risk.events",
        pd.DataFrame({"id": [1]}),
        mode="o",
    )

    assert database.adapter.create_attempts == 0


def test_unsupported_mode_does_not_leave_created_empty_table():
    database = Database(
        "observable_write",
        existing_tables=(),
        unsupported_modes=("r",),
    )

    with pytest.raises(DatabaseCapabilityError, match="不支持模式"):
        database.write(
            "risk.events",
            pd.DataFrame({"id": [1]}),
            mode="r",
            key_columns="id",
        )

    assert database.adapter.create_attempts == 0
    assert database.adapter.tables == set()


def test_concurrent_table_creator_is_treated_as_successful_ensure():
    database = Database(
        "observable_write",
        existing_tables=(),
        race_create=True,
    )

    database.adapter.ensure_table(
        pd.DataFrame({"id": [1]}),
        "risk.events",
    )

    assert database.adapter.create_attempts == 1
    assert database.adapter.tables == {"risk.events"}


def test_post_create_failure_is_not_hidden_when_table_now_exists():
    class PostCreateFailureAdapter(ObservableWriteAdapter):
        def create_table(self, data, table_name, *, dialect_options=None):
            del data, dialect_options
            self.create_attempts += 1
            self.tables.add(table_name)
            raise RuntimeError("字段注释写入失败")

    register_adapter("post_create_failure", PostCreateFailureAdapter, replace=True)
    database = Database("post_create_failure", existing_tables=())

    with pytest.raises(RuntimeError, match="字段注释"):
        database.adapter.ensure_table(
            pd.DataFrame({"id": [1]}),
            "risk.events",
        )

    assert database.adapter.tables == {"risk.events"}


def test_sql_adapter_native_write_method_does_not_bypass_sql_pipeline():
    class SQLAdapterWithNativeWrite(ObservableWriteAdapter):
        def write(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("SQL 适配器不能走 NoSQL 原生 write 分支")

    register_adapter("sql_with_native_write", SQLAdapterWithNativeWrite, replace=True)
    database = Database("sql_with_native_write")

    result = database.write(
        "risk.events",
        pd.DataFrame({"id": [1]}),
        key_columns="id",
    )

    assert isinstance(result, WriteResult)
    assert result.completed is True


def test_write_pipeline_rejects_unknown_mode_before_consuming_data():
    consumed = []

    def rows():
        consumed.append(True)
        yield {"id": 1}

    database = Database("observable_write")

    with pytest.raises(ValidationError, match="mode"):
        database.stream_write(rows(), "risk.events", mode="x")

    assert consumed == []


def test_drop_mode_validates_first_batch_before_drop():
    database = Database("observable_write")
    bad_data = iter([pd.DataFrame()])

    with pytest.raises(InputValidationError, match="有效数据"):
        database.stream_write(bad_data, "risk.events", mode="d")

    assert database.adapter.calls == []


def test_qualified_identifier_rejects_empty_parts_before_writing():
    database = Database("observable_write")

    with pytest.raises(ValidationError, match="表名"):
        database.stream_write(
            pd.DataFrame({"id": [1]}),
            "risk..events",
            mode="a",
        )

    assert database.adapter.calls == []


def test_dataframe_is_split_into_requested_batch_size():
    database = Database("observable_write")
    frame = pd.DataFrame({"id": range(5)})

    result = database.stream_write(frame, "risk.events", mode="a", batch_size=2)

    batches = [call[-1] for call in database.adapter.calls if call[0] == "write"]
    assert [batch["id"].tolist() for batch in batches] == [[0, 1], [2, 3], [4]]
    assert result.batches_committed == 3


def test_dataframe_chunk_iterator_is_not_pre_run_or_retried():
    events = []

    def chunks():
        events.append("first")
        yield pd.DataFrame({"id": [1, 2]})
        events.append("second")
        yield pd.DataFrame({"id": [3]})

    database = Database("observable_write")
    result = database.stream_write(chunks(), "risk.events", mode="a", batch_size=10)

    assert events == ["first", "second"]
    assert result.rows_received == 3


def test_mapping_and_positional_rows_are_supported():
    mapping_batches = list(iter_write_batches([{"id": 1}, {"id": 2}], batch_size=10))
    positional_batches = list(iter_write_batches([(3, "A"), (4, "B")], batch_size=10, columns=["id", "name"]))

    assert mapping_batches[0].to_dict("records") == [{"id": 1}, {"id": 2}]
    assert positional_batches[0].to_dict("records") == [
        {"id": 3, "name": "A"},
        {"id": 4, "name": "B"},
    ]


def test_positional_rows_require_columns():
    with pytest.raises(InputValidationError, match="columns"):
        list(iter_write_batches([(1, "A")], batch_size=10))


def test_dataframe_chunks_must_keep_same_columns():
    chunks = [pd.DataFrame({"id": [1]}), pd.DataFrame({"name": ["A"]})]

    with pytest.raises(InputValidationError, match="字段"):
        list(iter_write_batches(chunks, batch_size=10))


def test_partial_failure_exposes_committed_batch_result():
    database = Database("observable_write", fail_batch=2)
    frame = pd.DataFrame({"id": [1, 2, 3]})

    with pytest.raises(DatabaseWriteError, match="第 2 批") as caught:
        database.stream_write(frame, "risk.events", mode="a", batch_size=1)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.result.completed is False
    assert caught.value.result.rows_received == 2
    assert caught.value.result.rows_inserted == 1
    assert caught.value.result.batches_committed == 1
    assert caught.value.result.failed_batch == 2


def test_create_table_delegates_validated_dataframe_and_options():
    database = Database("observable_write")
    frame = pd.DataFrame({"id": pd.Series([1], dtype="int64")})

    result = database.create_table(
        frame,
        "risk.events",
        dialect_options={"engine": "custom"},
    )

    assert result == "CREATE TABLE"
    call = database.adapter.calls[0]
    assert call[0:2] == ("create_table", "risk.events")
    assert call[2].equals(frame)
    assert call[3] == {"engine": "custom"}


def test_null_values_are_preserved_for_adapter_level_conversion():
    frame = pd.DataFrame({"value": [1.0, np.nan]})

    batches = list(iter_write_batches(frame, batch_size=10))

    assert np.isnan(batches[0].loc[1, "value"])


@pytest.mark.parametrize(
    "data_type",
    [
        "BIGINT); DROP TABLE risk.events; --",
        "VARCHAR(255) COMMENT '注入'",
        "String/*注入*/",
        "Nullable(String",
    ],
)
def test_sql_type_validator_rejects_ddl_injection_and_unbalanced_types(data_type):
    with pytest.raises(ValidationError, match="数据类型"):
        validate_sql_type(data_type, database_type="test")


@pytest.mark.parametrize(
    "data_type",
    ["BIGINT UNSIGNED", "DECIMAL(18, 2)", "ARRAY<STRING>", "Nullable(String)"],
)
def test_sql_type_validator_accepts_nested_backend_types(data_type):
    assert validate_sql_type(data_type, database_type="test") == data_type
