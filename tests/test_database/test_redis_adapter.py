"""Redis 连接池与统一 NoSQL 读写契约测试。"""

from types import SimpleNamespace

import pytest

from hscredit.database import Database, NoSQLWriteResult, register_adapter
from hscredit.database.adapters.redis import RedisAdapter
from hscredit.exceptions import ValidationError


class ObservableRedisPool:
    instances = []

    def __init__(self, **kwargs):
        self.url = None
        self.kwargs = dict(kwargs)
        self.disconnected = False
        type(self).instances.append(self)

    @classmethod
    def from_url(cls, url, **kwargs):
        pool = cls(**kwargs)
        pool.url = url
        return pool

    def disconnect(self):
        self.disconnected = True


class ObservableBlockingRedisPool(ObservableRedisPool):
    instances = []


class ObservableRedisClient:
    instances = []

    def __init__(self, *, connection_pool):
        self.connection_pool = connection_pool
        self.values = {}
        self.calls = []
        self.closed = False
        type(self).instances.append(self)

    def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    def mget(self, keys):
        materialized = list(keys)
        self.calls.append(("mget", materialized))
        return [self.values.get(key) for key in materialized]

    def set(self, key, value, **kwargs):
        self.calls.append(("set", key, value, dict(kwargs)))
        self.values[key] = value
        return True

    def mset(self, mapping):
        materialized = dict(mapping)
        self.calls.append(("mset", materialized))
        self.values.update(materialized)
        return True

    def delete(self, *keys):
        self.calls.append(("delete", tuple(keys)))
        deleted = 0
        for key in keys:
            if key in self.values:
                deleted += 1
                del self.values[key]
        return deleted

    def exists(self, key):
        self.calls.append(("exists", key))
        return int(key in self.values)

    def close(self):
        self.closed = True


class ObservableRedisAdapter(RedisAdapter):
    def load_redis_module(self):
        return SimpleNamespace(
            ConnectionPool=ObservableRedisPool,
            BlockingConnectionPool=ObservableBlockingRedisPool,
            Redis=ObservableRedisClient,
        )


@pytest.fixture(autouse=True)
def register_redis_adapter():
    ObservableRedisPool.instances.clear()
    ObservableBlockingRedisPool.instances.clear()
    ObservableRedisClient.instances.clear()
    register_adapter("observable_redis", ObservableRedisAdapter, replace=True)


def test_redis_uses_configured_blocking_pool_and_exposes_native_client():
    database = Database(
        "observable_redis",
        url="redis://cache.internal:6379/3",
        decode_responses=True,
        pool_options={"max_connections": 8, "blocking": True, "timeout": 1.5},
    )

    pool = database.adapter.pool
    assert isinstance(pool, ObservableBlockingRedisPool)
    assert pool.url == "redis://cache.internal:6379/3"
    assert pool.kwargs == {
        "decode_responses": True,
        "max_connections": 8,
        "timeout": 1.5,
    }
    assert database.native_client is database.adapter.client
    assert database.adapter.capabilities.streaming_read is False
    assert database.adapter.capabilities.native_bulk_write is True
    assert database.adapter.capabilities.metadata_export is False


def test_redis_explicit_single_crud_returns_backend_neutral_result():
    database = Database("observable_redis")

    written = database.write_one("score:1", "720", ttl=60)

    assert written == NoSQLWriteResult(
        operation="write",
        acknowledged=True,
        affected_count=1,
        identifiers=("score:1",),
    )
    assert database.read_one("score:1") == "720"
    assert database.exists("score:1") is True
    deleted = database.delete_one("score:1")
    assert deleted.affected_count == 1
    assert database.exists("score:1") is False
    assert database.adapter.client.calls[0] == ("set", "score:1", "720", {"ex": 60})


def test_redis_adaptive_methods_dispatch_scalar_and_batch_inputs():
    database = Database("observable_redis")

    batch_result = database.write({"score:1": "720", "score:2": "680"})
    values = database.read(["score:2", "missing", "score:1"])
    deleted = database.delete(["score:1", "score:2"])

    assert batch_result.affected_count == 2
    assert values == ["680", None, "720"]
    assert deleted.affected_count == 2

    database.write("score:3", "700")
    assert database.read("score:3") == "700"
    assert database.delete("score:3").affected_count == 1


def test_redis_rejects_empty_batch_before_calling_driver():
    database = Database("observable_redis")

    with pytest.raises(ValidationError, match="不能为空"):
        database.write_many({})
    with pytest.raises(ValidationError, match="不能为空"):
        database.read_many([])

    assert database.adapter.client.calls == []


def test_redis_close_releases_client_and_pool_once():
    database = Database("observable_redis")
    client = database.adapter.client
    pool = database.adapter.pool

    database.close()
    database.close()

    assert client.closed is True
    assert pool.disconnected is True
