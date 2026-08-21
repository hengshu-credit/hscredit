"""真实 MongoDB 连接池和自适应读写集成测试。"""

import os
import uuid

import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_MONGODB_URI"),
    reason="未配置 HSCREDIT_TEST_MONGODB_URI",
)


def test_mongodb_real_pool_and_adaptive_crud():
    run_id = uuid.uuid4().hex
    collection = f"hscredit_database_it_{run_id[:12]}"
    database = Database(
        "mongodb",
        uri=os.environ["HSCREDIT_TEST_MONGODB_URI"],
        database=os.getenv("HSCREDIT_TEST_MONGODB_DATABASE", "hscredit_test"),
        pool_options={"min_pool_size": 0, "max_pool_size": 3},
    )
    try:
        inserted = database.write(
            collection,
            [
                {"_id": f"{run_id}-1", "score": 720},
                {"_id": f"{run_id}-2", "score": 680},
            ],
        )
        assert inserted.affected_count == 2
        assert database.read_one(collection, {"_id": f"{run_id}-1"})["score"] == 720
        assert len(database.read(collection, {"_id": {"$regex": f"^{run_id}"}})) == 2

        updated = database.write_one(
            collection,
            {"$set": {"score": 730}},
            selector={"_id": f"{run_id}-1"},
            mode="update",
        )
        assert updated.modified_count == 1
        assert database.delete(
            collection,
            {"_id": {"$regex": f"^{run_id}"}},
            many=True,
        ).affected_count == 2
    finally:
        database.delete_many(
            collection,
            {"_id": {"$regex": f"^{run_id}"}},
        )
        database.close()
