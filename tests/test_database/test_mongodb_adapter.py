"""MongoDB 连接池与统一 NoSQL 读写契约测试。"""

from types import SimpleNamespace

import pytest

from hscredit.database import Database, NoSQLWriteResult, register_adapter
from hscredit.database.adapters.mongodb import MongoDBAdapter
from hscredit.exceptions import ValidationError


class ObservableCursor:
    def __init__(self, documents, calls):
        self.documents = list(documents)
        self.calls = calls

    def sort(self, value):
        self.calls.append(("cursor.sort", value))
        return self

    def skip(self, value):
        self.calls.append(("cursor.skip", value))
        self.documents = self.documents[value:]
        return self

    def limit(self, value):
        self.calls.append(("cursor.limit", value))
        if value:
            self.documents = self.documents[:value]
        return self

    def __iter__(self):
        return iter(self.documents)


class ObservableCollection:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.one_result = None
        self.many_result = []

    def find_one(self, selector, projection=None, **options):
        self.calls.append(("find_one", dict(selector), projection, dict(options)))
        return self.one_result

    def find(self, selector, projection=None, **options):
        self.calls.append(("find", dict(selector), projection, dict(options)))
        return ObservableCursor(self.many_result, self.calls)

    def insert_one(self, document, **options):
        self.calls.append(("insert_one", dict(document), dict(options)))
        return SimpleNamespace(acknowledged=True, inserted_id="mongo-id-1")

    def insert_many(self, documents, **options):
        materialized = [dict(document) for document in documents]
        self.calls.append(("insert_many", materialized, dict(options)))
        return SimpleNamespace(
            acknowledged=True,
            inserted_ids=[f"mongo-id-{index}" for index in range(len(materialized))],
        )

    def update_one(self, selector, update, upsert=False, **options):
        self.calls.append(("update_one", dict(selector), dict(update), upsert, dict(options)))
        return SimpleNamespace(
            acknowledged=True,
            matched_count=1,
            modified_count=1,
            upserted_id=None,
        )

    def update_many(self, selector, update, upsert=False, **options):
        self.calls.append(("update_many", dict(selector), dict(update), upsert, dict(options)))
        return SimpleNamespace(
            acknowledged=True,
            matched_count=3,
            modified_count=2,
            upserted_id=None,
        )

    def replace_one(self, selector, replacement, upsert=False, **options):
        self.calls.append(("replace_one", dict(selector), dict(replacement), upsert, dict(options)))
        return SimpleNamespace(
            acknowledged=True,
            matched_count=1,
            modified_count=1,
            upserted_id=None,
        )

    def delete_one(self, selector, **options):
        self.calls.append(("delete_one", dict(selector), dict(options)))
        return SimpleNamespace(acknowledged=True, deleted_count=1)

    def delete_many(self, selector, **options):
        self.calls.append(("delete_many", dict(selector), dict(options)))
        return SimpleNamespace(acknowledged=True, deleted_count=4)


class ObservableMongoDatabase:
    def __init__(self, name):
        self.name = name
        self.collections = {}

    def __getitem__(self, collection_name):
        return self.collections.setdefault(collection_name, ObservableCollection(collection_name))


class ObservableMongoClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = dict(kwargs)
        self.databases = {}
        self.closed = False
        type(self).instances.append(self)

    def __getitem__(self, database_name):
        return self.databases.setdefault(database_name, ObservableMongoDatabase(database_name))

    def close(self):
        self.closed = True


class ObservableMongoDBAdapter(MongoDBAdapter):
    def load_pymongo_module(self):
        return SimpleNamespace(MongoClient=ObservableMongoClient)


@pytest.fixture(autouse=True)
def register_mongodb_adapter():
    ObservableMongoClient.instances.clear()
    register_adapter("observable_mongodb", ObservableMongoDBAdapter, replace=True)


@pytest.fixture
def database():
    return Database(
        "observable_mongodb",
        uri="mongodb://mongo.internal:27017/risk",
        database="risk",
        serverSelectionTimeoutMS=2000,
        pool_options={
            "min_pool_size": 2,
            "max_pool_size": 20,
            "max_connecting": 4,
            "wait_queue_timeout_ms": 1500,
        },
    )


def test_mongodb_uses_mongo_client_pool_options_and_exposes_native_client(database):
    client = database.adapter.client

    assert client.args == ("mongodb://mongo.internal:27017/risk",)
    assert client.kwargs == {
        "serverSelectionTimeoutMS": 2000,
        "minPoolSize": 2,
        "maxPoolSize": 20,
        "maxConnecting": 4,
        "waitQueueTimeoutMS": 1500,
    }
    assert database.native_client is client
    assert database.adapter.capabilities.streaming_read is False
    assert database.adapter.capabilities.native_bulk_write is True
    assert database.adapter.capabilities.metadata_export is False


def test_mongodb_explicit_read_methods_apply_projection_sort_skip_and_limit(database):
    collection = database.adapter.database["events"]
    collection.one_result = {"_id": 1, "score": 720}
    collection.many_result = [{"_id": 1}, {"_id": 2}, {"_id": 3}]

    one = database.read_one("events", {"_id": 1}, projection={"score": 1})
    many = database.read_many(
        "events",
        {"active": True},
        sort=[("created_at", -1)],
        skip=1,
        limit=1,
    )

    assert one == {"_id": 1, "score": 720}
    assert many == [{"_id": 2}]
    assert collection.calls == [
        ("find_one", {"_id": 1}, {"score": 1}, {}),
        ("find", {"active": True}, None, {}),
        ("cursor.sort", [("created_at", -1)]),
        ("cursor.skip", 1),
        ("cursor.limit", 1),
    ]


def test_mongodb_adaptive_read_defaults_many_and_limit_one_selects_single(database):
    collection = database.adapter.database["events"]
    collection.many_result = [{"_id": 1}, {"_id": 2}]
    collection.one_result = {"_id": 1}

    assert database.read("events", {"active": True}) == [{"_id": 1}, {"_id": 2}]
    assert database.read("events", {"active": True}, limit=1) == {"_id": 1}
    assert database.read("events", {"active": True}, many=False) == {"_id": 1}


def test_mongodb_adaptive_write_dispatches_document_and_document_sequence(database):
    one = database.write("events", {"score": 720})
    many = database.write("events", [{"score": 680}, {"score": 700}])

    assert one == NoSQLWriteResult(
        operation="write",
        acknowledged=True,
        affected_count=1,
        identifiers=("mongo-id-1",),
    )
    assert many.affected_count == 2
    assert many.identifiers == ("mongo-id-0", "mongo-id-1")


def test_mongodb_write_supports_update_one_update_many_and_replace(database):
    updated = database.write_one(
        "events",
        {"$set": {"score": 720}},
        selector={"_id": 1},
        mode="update",
        upsert=True,
    )
    updated_many = database.write_many(
        "events",
        {"$set": {"active": False}},
        selector={"expired": True},
        mode="update",
    )
    replaced = database.write_one(
        "events",
        {"_id": 1, "score": 730},
        selector={"_id": 1},
        mode="replace",
    )

    assert updated.matched_count == 1
    assert updated.modified_count == 1
    assert updated_many.matched_count == 3
    assert updated_many.affected_count == 2
    assert replaced.affected_count == 1


def test_mongodb_delete_is_safe_by_default_and_many_requires_opt_in(database):
    deleted_one = database.delete("events", {"expired": True})
    deleted_many = database.delete("events", {"expired": True}, many=True)

    assert deleted_one.affected_count == 1
    assert deleted_many.affected_count == 4
    assert database.adapter.database["events"].calls[-2:] == [
        ("delete_one", {"expired": True}, {}),
        ("delete_many", {"expired": True}, {}),
    ]


def test_mongodb_exists_uses_single_document_lookup(database):
    collection = database.adapter.database["events"]
    collection.one_result = {"_id": 1}

    assert database.exists("events", {"score": {"$gte": 700}}) is True
    assert collection.calls == [
        ("find_one", {"score": {"$gte": 700}}, {"_id": 1}, {}),
    ]


def test_mongodb_rejects_invalid_collection_documents_and_selector(database):
    with pytest.raises(ValidationError, match="集合名"):
        database.read_many("", {})
    with pytest.raises(ValidationError, match="limit"):
        database.read("events", {}, limit=True)
    with pytest.raises(ValidationError, match="文档序列不能为空"):
        database.write_many("events", [])
    with pytest.raises(ValidationError, match="selector"):
        database.delete_one("events", [1, 2])
    with pytest.raises(ValidationError, match="allow_all"):
        database.delete_many("events", {})


def test_mongodb_close_releases_mongo_client_once(database):
    client = database.adapter.client

    database.close()
    database.close()

    assert client.closed is True
