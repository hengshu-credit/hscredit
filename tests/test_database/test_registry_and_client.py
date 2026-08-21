"""数据库适配器注册表与公共门面测试。"""

import subprocess
import sys

import pytest

from hscredit.database import Database, available_adapters, register_adapter
from hscredit.exceptions import StateError, ValidationError

from .fakes import FakeAdapter


@pytest.fixture(autouse=True)
def register_fake_adapter():
    FakeAdapter.instances.clear()
    register_adapter("fake_contract", FakeAdapter, aliases=("fake-contract-alias",), replace=True)


def test_custom_adapter_is_closed_by_database_context():
    with Database(
        "fake_contract",
        token="sensitive-value",
        pool_options={"maxconnections": 3},
        adapter_options={"region": "local"},
    ) as database:
        assert isinstance(database.adapter, FakeAdapter)
        assert database.adapter.connect_kwargs == {"token": "sensitive-value"}
        assert database.adapter.pool_options.maxconnections == 3
        assert database.adapter.adapter_options == {"region": "local"}
        assert "sensitive-value" not in repr(database)

    assert database.adapter.closed is True


def test_adapter_alias_resolves_to_registered_class():
    database = Database("fake-contract-alias")

    assert isinstance(database.adapter, FakeAdapter)
    assert database.database_type == "fake_contract"


def test_database_delegates_query_without_changing_values():
    database = Database("fake_contract")

    result = database.query("select * from t where id=%s", params=(7,), result="records")

    assert result == {
        "sql": "select * from t where id=%s",
        "params": (7,),
        "result": "records",
    }


def test_database_delegates_execute_and_executemany():
    database = Database("fake_contract")

    affected = database.execute("delete from t where id=%s", params=(7,))
    inserted = database.executemany(
        "insert into t(id) values (%s)",
        ((8,), (9,)),
    )

    assert affected == 1
    assert inserted == 2
    assert database.adapter.calls == [
        ("execute", "delete from t where id=%s", (7,)),
        ("executemany", "insert into t(id) values (%s)", [(8,), (9,)]),
    ]


def test_close_is_idempotent_and_closed_database_rejects_operations():
    database = Database("fake_contract")

    database.close()
    database.close()

    assert database.adapter.calls.count(("close",)) == 1
    with pytest.raises(StateError, match="已经关闭"):
        database.query("select 1")


def test_unknown_database_lists_available_adapters():
    with pytest.raises(ValidationError, match="支持的数据库类型") as caught:
        Database("missing_database")

    assert "mysql" in str(caught.value)
    assert "fake_contract" in str(caught.value)
    assert "fake_contract" in available_adapters()


def test_register_adapter_rejects_duplicate_without_explicit_replace():
    with pytest.raises(ValidationError, match="已经注册"):
        register_adapter("fake_contract", FakeAdapter)


def test_importing_database_package_does_not_import_backend_drivers():
    code = """
import sys
import hscredit.database
drivers = ['pymysql', 'impala', 'oracledb', 'clickhouse_connect', 'odps', 'redis', 'pymongo']
loaded = [name for name in drivers if name in sys.modules]
assert not loaded, loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
