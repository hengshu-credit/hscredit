"""数据库公共契约与可选依赖隔离测试。"""

import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib

from hscredit.database import (
    RESULT_TYPES,
    DatabaseCapabilities,
    DatabaseCapabilityError,
    MongoPoolOptions,
    NoSQLWriteResult,
    PoolOptions,
    RedisPoolOptions,
    StreamState,
    WriteResult,
)
from hscredit.exceptions import HSCreditError, ValidationError


def test_database_package_imports_without_optional_drivers():
    code = "from hscredit import Database, WriteResult, DatabaseCapabilityError"

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_write_result_keeps_unknown_driver_counts_as_none():
    result = WriteResult(mode="a", completed=True, rows_received=3)

    assert result.rows_inserted is None
    assert result.rows_updated is None
    assert result.rows_skipped is None
    assert result.batches_committed == 0
    assert result.failed_batch is None


def test_database_capabilities_are_immutable_and_normalize_modes():
    capabilities = DatabaseCapabilities(write_modes={"a", "o"})

    assert capabilities.write_modes == frozenset({"a", "o"})
    assert capabilities.transactions is True
    assert capabilities.streaming_read is True


def test_pool_options_validate_positive_limits_and_known_keys():
    options = PoolOptions.from_mapping(
        {
            "mincached": 1,
            "maxcached": 5,
            "maxconnections": 10,
            "blocking": True,
        }
    )

    assert options.maxconnections == 10
    assert options.to_dbutils_kwargs()["blocking"] is True

    for invalid in (
        {"maxconnections": -1},
        {"mincached": 3, "maxcached": 2},
        {"unknown_pool_key": 1},
    ):
        try:
            PoolOptions.from_mapping(invalid)
        except ValidationError:
            pass
        else:  # pragma: no cover - 断言失败分支
            raise AssertionError(f"未拒绝非法连接池参数: {invalid}")


def test_stream_state_values_are_stable_public_contract():
    assert [state.value for state in StreamState] == [
        "running",
        "completed",
        "interrupted",
        "failed",
        "closed",
    ]


def test_database_query_result_types_are_shared_public_contract():
    assert RESULT_TYPES == frozenset({"dataframe", "records", "rows"})


def test_nosql_pool_options_translate_to_native_driver_names():
    redis_options = RedisPoolOptions.from_mapping(
        {"max_connections": 8, "blocking": True, "timeout": 2.5}
    )
    mongo_options = MongoPoolOptions.from_mapping(
        {
            "min_pool_size": 2,
            "max_pool_size": 20,
            "max_connecting": 4,
            "wait_queue_timeout_ms": 1500,
        }
    )

    assert redis_options.to_redis_kwargs() == {
        "max_connections": 8,
        "timeout": 2.5,
    }
    assert mongo_options.to_mongo_kwargs() == {
        "minPoolSize": 2,
        "maxPoolSize": 20,
        "maxConnecting": 4,
        "waitQueueTimeoutMS": 1500,
    }


def test_nosql_write_result_exposes_backend_neutral_counts_and_ids():
    result = NoSQLWriteResult(
        operation="write",
        acknowledged=True,
        affected_count=2,
        matched_count=3,
        identifiers=("a", "b"),
    )

    assert result.affected_count == 2
    assert result.matched_count == 3
    assert result.identifiers == ("a", "b")


def test_database_capability_error_belongs_to_hscredit_error_family():
    assert issubclass(DatabaseCapabilityError, HSCreditError)


def test_database_optional_dependencies_are_split_by_backend():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]

    expected_groups = {
        "db-mysql",
        "db-hive",
        "db-impala",
        "db-oracle",
        "db-starrocks",
        "db-clickhouse",
        "db-maxcompute",
        "db-redis",
        "db-mongodb",
        "database-all",
    }
    assert expected_groups.issubset(extras)
    assert any(item.startswith("DBUtils>=3.1.2") for item in extras["db-mysql"])
    assert any("python_version < '3.10'" in item for item in extras["db-clickhouse"])
    assert any(item.startswith("redis>=6.2") for item in extras["db-redis"])
    assert any(item.startswith("pymongo>=4.6") for item in extras["db-mongodb"])
    assert "database-all" in extras["all"][0]
