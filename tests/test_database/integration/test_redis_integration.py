"""真实 Redis 连接池和自适应读写集成测试。"""

import os
import uuid

import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_REDIS_URL"),
    reason="未配置 HSCREDIT_TEST_REDIS_URL",
)


def test_redis_real_pool_and_adaptive_crud():
    prefix = f"hscredit:database:it:{uuid.uuid4().hex}"
    keys = [f"{prefix}:1", f"{prefix}:2"]
    database = Database(
        "redis",
        url=os.environ["HSCREDIT_TEST_REDIS_URL"],
        decode_responses=True,
        pool_options={"max_connections": 3, "blocking": True, "timeout": 2},
    )
    try:
        written = database.write({keys[0]: "720", keys[1]: "680"})
        assert written.affected_count == 2
        assert database.read(keys) == ["720", "680"]
        assert database.exists(keys[0]) is True
        assert database.delete(keys).affected_count == 2
    finally:
        database.delete(keys)
        database.close()
