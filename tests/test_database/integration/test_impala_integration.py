"""真实 Impala 查询与元数据集成测试。"""

import os

import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_IMPALA_HOST"),
    reason="未配置 HSCREDIT_TEST_IMPALA_HOST",
)


def test_impala_real_pool_stream_and_metadata():
    database = Database(
        "impala",
        host=os.environ["HSCREDIT_TEST_IMPALA_HOST"],
        port=int(os.getenv("HSCREDIT_TEST_IMPALA_PORT", "21050")),
        database=os.getenv("HSCREDIT_TEST_IMPALA_DATABASE", "default"),
        auth_mechanism=os.getenv("HSCREDIT_TEST_IMPALA_AUTH", "NOSASL"),
    )
    try:
        frame = database.read_query("SELECT 1 AS id", chunksize=1)
        assert frame.iloc[0, 0] == 1
        assert not database.export_schema(targets=[os.getenv("HSCREDIT_TEST_IMPALA_DATABASE", "default")]).empty
    finally:
        database.close()
