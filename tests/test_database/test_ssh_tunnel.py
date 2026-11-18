"""从公开 Database 构造入口验证 SSH 转发、读写与资源释放。"""

import asyncio
import socket
import ssl
from contextlib import asynccontextmanager
from types import SimpleNamespace

import asyncssh as ssh
import pytest

from hscredit.database import Database, DatabaseConnectionError
from hscredit.database.adapters.mysql import MySQLAdapter
from hscredit.database.tunnel import forward_database
from tests.test_database.fakes import FakeDBAPIDriver, FakeDBAPIState, FakePooledDB


@asynccontextmanager
async def jump_host(tmp_path):
    server_key = ssh.generate_private_key("ssh-ed25519")
    client_key = ssh.generate_private_key("ssh-ed25519")
    public_key = client_key.convert_to_public()
    private_key_path = tmp_path / "id_ed25519"
    private_key_path.write_bytes(client_key.export_private_key(passphrase="test-key-passphrase"))
    forwarded_destinations = []

    class JumpHost(ssh.SSHServer):
        def begin_auth(self, username):
            return True

        def public_key_auth_supported(self):
            return True

        def validate_public_key(self, username, key):
            return username == "jump_user" and key == public_key

        def connection_requested(self, dest_host, dest_port, orig_host, orig_port):
            forwarded_destinations.append((dest_host, dest_port))
            return dest_host == "127.0.0.1" and dest_port == target_port

    async def send_banner(reader, writer):
        writer.write(b"db-target")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    target = await asyncio.start_server(send_banner, "127.0.0.1", 0)
    async with target:
        target_port = target.sockets[0].getsockname()[1]
        server = await ssh.listen("127.0.0.1", 0, server_factory=JumpHost, server_host_keys=[server_key])
        async with server:
            config = {
                "host": "127.0.0.1",
                "port": server.get_port(),
                "username": "jump_user",
                "auth": "key",
                "private_key_path": private_key_path,
                "passphrase": "test-key-passphrase",
                "host_fingerprint": server_key.get_fingerprint(),
            }
            yield config, target_port, forwarded_destinations


@pytest.mark.parametrize("pool_fails", [False, True])
def test_mysql_pool_uses_the_tunnel_and_releases_it(tmp_path, monkeypatch, pool_fails):
    state = FakeDBAPIState(rows=[(42,)], columns=["amount"])
    driver = FakeDBAPIDriver(state)
    driver.cursors = SimpleNamespace(SSCursor=object)

    class RoutedPool(FakePooledDB):
        def __init__(self, creator, **kwargs):
            super().__init__(creator, **kwargs)
            self.check_forwarding()
            if pool_fails:
                raise RuntimeError("test pool initialization failure")

        def check_forwarding(self):
            with socket.create_connection((self.state.pool_kwargs["host"], self.state.pool_kwargs["port"]), timeout=3) as conn:
                with conn.makefile("rb") as stream:
                    assert stream.read() == b"db-target"

        def close(self):
            self.check_forwarding()
            super().close()

    monkeypatch.setattr(MySQLAdapter, "load_driver", lambda self: driver)
    monkeypatch.setattr(MySQLAdapter, "load_pool_class", lambda self: RoutedPool)

    def exercise(config, target_port):
        options = dict(host="127.0.0.1", port=target_port, user="db_user", password="db_password", tunnel=config)
        if pool_fails:
            with pytest.raises(DatabaseConnectionError, match="初始化 mysql"):
                Database("mysql", **options)
        else:
            with Database("mariadb", **options) as database:
                assert database.query("SELECT amount FROM loans", result="rows") == [(42,)]
                assert database.execute("UPDATE loans SET amount=%s", (42,)) == 1
                with database.stream_query("SELECT amount FROM loans", chunksize=1) as stream:
                    assert next(stream).to_dict("records") == [{"amount": 42}]
                assert state.connections[1].commit_calls == 1
                assert "test-key-passphrase" not in repr(database)
            database.close()
            assert database.closed and state.pool_closed

        assert state.pool_kwargs["user"] == "db_user"
        assert state.pool_kwargs["password"] == "db_password"
        assert "tunnel" not in state.pool_kwargs
        assert state.pool_kwargs["port"] != target_port
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", state.pool_kwargs["port"]), timeout=1).close()

    async def scenario():
        async with jump_host(tmp_path) as (config, target_port, destinations):
            await asyncio.to_thread(exercise, config, target_port)
            assert destinations == [("127.0.0.1", target_port)] * (1 if pool_fails else 2)

    asyncio.run(scenario())


def test_mysql_tunnel_keeps_original_hostname_and_user_tls_context():
    class RecordingContext(ssl.SSLContext):
        def wrap_socket(self, sock, **kwargs):
            return kwargs

    context = RecordingContext(ssl.PROTOCOL_TLS_CLIENT)
    listener = SimpleNamespace(get_port=lambda: 54321)

    async def forward_local_port(local_host, local_port, db_host, db_port):
        assert (local_host, local_port, db_host, db_port) == ("127.0.0.1", 0, "mysql.internal", 3306)
        return listener

    original = {"host": "mysql.internal", "ssl": context}
    _, forwarded, _ = asyncio.run(forward_database(SimpleNamespace(forward_local_port=forward_local_port), "mysql", original))
    assert forwarded["host"] == "127.0.0.1"
    assert forwarded["ssl"].wrap_socket(None, server_hostname="127.0.0.1") == {"server_hostname": "mysql.internal"}
    assert original == {"host": "mysql.internal", "ssl": context}
