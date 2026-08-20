"""真实 HiveServer2 查询与元数据集成测试。"""

import os

import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_HIVE_HOST"),
    reason="未配置 HSCREDIT_TEST_HIVE_HOST",
)


def test_hive_real_pool_stream_and_metadata():
    database = Database(
        "hive",
        host=os.environ["HSCREDIT_TEST_HIVE_HOST"],
        port=int(os.getenv("HSCREDIT_TEST_HIVE_PORT", "10000")),
        database=os.getenv("HSCREDIT_TEST_HIVE_DATABASE", "default"),
        auth_mechanism=os.getenv("HSCREDIT_TEST_HIVE_AUTH", "PLAIN"),
        user=os.getenv("HSCREDIT_TEST_HIVE_USER"),
        password=os.getenv("HSCREDIT_TEST_HIVE_PASSWORD"),
    )
    try:
        frame = database.read_query("SELECT 1 AS id", chunksize=1)
        assert frame.iloc[0, 0] == 1
        assert not database.export_schema(targets=[os.getenv("HSCREDIT_TEST_HIVE_DATABASE", "default")]).empty
    finally:
        database.close()
