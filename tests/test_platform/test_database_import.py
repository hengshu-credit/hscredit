"""验证数据库查询导入、连接释放和快照在后续分析中的可用性。"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hscredit.database.adapters.mysql import MySQLAdapter
from hscredit.exceptions import DependencyError
from hscredit.web import create_app
from tests.test_database.fakes import FakeDBAPIDriver, FakeDBAPIState, FakePooledDB


@pytest.fixture
def source(tmp_path, monkeypatch):
    state = FakeDBAPIState(rows=[("0001", 640), ("0002", 720)], columns=["客户编号", "评分"])
    driver = FakeDBAPIDriver(state)
    driver.cursors = SimpleNamespace(SSCursor=object)
    monkeypatch.setattr(MySQLAdapter, "load_driver", lambda self: driver)
    monkeypatch.setattr(MySQLAdapter, "load_pool_class", lambda self: FakePooledDB)
    body = {
        "name": "评分样本",
        "db_type": "mysql",
        "address": "db.test",
        "port": 3306,
        "username": "reader",
        "password": "test-only-secret",
        "database": "risk",
        "sql": "WITH loan AS (SELECT customer_id FROM loans) SELECT loan.customer_id AS 客户编号, s.score AS 评分 FROM loan JOIN scores s ON s.id = loan.customer_id WHERE s.score >= 600",
    }
    with TestClient(create_app(tmp_path)) as client:
        yield client, state, body


def test_import_then_explore_without_reconnecting(source, tmp_path):
    client, state, body = source
    response = client.post("/api/datasets/database", json=body)
    assert response.status_code == 200, response.text
    asset = response.json()
    assert asset["rows"] == 2 and asset["revision"] == 1
    assert state.cursors[0].executed == (body["sql"], None)
    assert state.pool_kwargs["host"] == body["address"] and state.pool_kwargs["password"] == body["password"]
    assert state.pool_closed and state.cursors[0].closed and state.connections[0].close_calls
    metadata = (tmp_path / "datasets" / asset["id"] / "metadata.json").read_text(encoding="utf-8")
    assert all(value not in metadata for value in (body["password"], body["address"], body["sql"]))
    url = "/api/datasets/" + asset["id"]
    assert client.get(url + "/preview").json()["rows"] == [{"客户编号": "0001", "评分": 640}, {"客户编号": "0002", "评分": 720}]
    state.fail_execute = RuntimeError("数据源已断开")
    result = client.post(url + "/explore", json={"feature": "评分", "bins": 2}).json()
    assert result["valid"] == 2 and [r["样本数"] for r in result["distribution"]] == [1, 1]
    assert len(state.cursors) == 1


def test_failed_and_empty_queries_do_not_publish(source):
    client, state, body = source
    state.fail_execute = RuntimeError(body["password"])
    failure = client.post("/api/datasets/database", json=body)
    assert failure.status_code == 400 and body["password"] not in failure.text
    state.fail_execute = None
    state.rows.clear()
    empty = client.post("/api/datasets/database", json=body)
    assert empty.status_code == 400 and "为空" in empty.json()["detail"]
    assert client.get("/api/datasets").json() == []
    assert state.pool_closed and all(c.closed for c in state.cursors)


@pytest.mark.parametrize("sql", ["DELETE FROM loans", "SELECT 1; SELECT 2", "CREATE TABLE copy AS SELECT 1"])
def test_only_single_queries_reach_driver(source, sql):
    client, state, body = source
    response = client.post("/api/datasets/database", json={**body, "sql": sql})
    assert response.status_code == 400 and not state.cursors


def test_driver_install_hint_and_limit_leave_no_dataset(source, monkeypatch):
    client, state, body = source
    from hscredit.web import database_import

    monkeypatch.setattr(database_import, "MAX_DATASET_ROWS", 1)
    response = client.post("/api/datasets/database", json=body)
    assert response.status_code == 400 and state.pool_closed and state.cursors[0].closed

    def missing_driver(self):
        raise DependencyError("缺少驱动")

    monkeypatch.setattr(MySQLAdapter, "load_driver", missing_driver)
    response = client.post("/api/datasets/database", json=body)
    assert response.status_code == 400 and "hscredit[db-mysql]" in response.json()["detail"]
    assert client.get("/api/datasets").json() == []


def test_probe_executes_only_driver_check_without_dataset(source):
    client, state, body = source
    config = {k: v for k, v in body.items() if k not in {"sql", "name"}}
    state.columns, state.rows = ["1"], [(1,)]
    response = client.post("/api/datasets/database/test", json=config)
    assert response.status_code == 200
    assert state.cursors[-1].executed == ("SELECT 1", None) and state.pool_closed
    state.fail_execute = RuntimeError(body["password"])
    failure = client.post("/api/datasets/database/test", json=config)
    assert failure.status_code == 400 and body["password"] not in failure.text
    assert client.get("/api/datasets").json() == []


def test_invalid_ssh_credentials_never_open_database(source):
    client, state, body = source
    config = {k: v for k, v in body.items() if k not in {"sql", "name"}}
    tunnel = {"host": "localhost", "username": "reader", "auth": "key_password", "password": "ssh-secret", "private_key": "invalid-test-key", "passphrase": "key-secret"}
    response = client.post("/api/datasets/database/test", json={**config, "tunnel": tunnel})
    assert response.status_code == 400 and "私钥" in response.json()["detail"]
    assert all(value not in response.text for value in (tunnel["private_key"], tunnel["passphrase"], tunnel["password"]))
    assert not state.connections


def test_ssh_auth_uses_exact_credentials_for_selected_mode():
    import asyncssh
    from hscredit.web.database_tunnel import SSHTunnel

    key = asyncssh.generate_private_key("ssh-ed25519")
    encrypted = key.export_private_key(passphrase="test-key-passphrase").decode()
    for mode, methods in (("password", ["password"]), ("key", ["publickey"]), ("key_password", ["publickey", "password"])):
        config = SSHTunnel(host="jump.test", username="reader", auth=mode, password="test-ssh-password", private_key=encrypted, passphrase="test-key-passphrase", host_fingerprint=key.get_fingerprint())
        options = config.ssh_options(asyncssh)
        assert options["preferred_auth"] == methods
        assert options["password"] == ("test-ssh-password" if "password" in methods else None)
        assert options["client_keys"] == ([key] if "publickey" in methods else None)
        validate = options["client_factory"]().validate_host_public_key
        assert validate("jump.test", "127.0.0.1", 22, key)
        assert not validate("jump.test", "127.0.0.1", 22, asyncssh.generate_private_key("ssh-ed25519"))
