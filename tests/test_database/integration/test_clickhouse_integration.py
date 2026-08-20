"""真实 ClickHouse 原生流和写入集成测试。"""

import os
import uuid

import pandas as pd
import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_CLICKHOUSE_HOST"),
    reason="未配置 HSCREDIT_TEST_CLICKHOUSE_HOST",
)


def test_clickhouse_real_native_stream_and_write():
    table = f"hscredit_db_it_{uuid.uuid4().hex[:12]}"
    database_name = os.getenv("HSCREDIT_TEST_CLICKHOUSE_DATABASE", "default")
    qualified = f"{database_name}.{table}"
    database = Database(
        "clickhouse",
        host=os.environ["HSCREDIT_TEST_CLICKHOUSE_HOST"],
        port=int(os.getenv("HSCREDIT_TEST_CLICKHOUSE_PORT", "8123")),
        username=os.getenv("HSCREDIT_TEST_CLICKHOUSE_USER", "default"),
        password=os.getenv("HSCREDIT_TEST_CLICKHOUSE_PASSWORD", ""),
        database=database_name,
        secure=os.getenv("HSCREDIT_TEST_CLICKHOUSE_SECURE", "false").lower() == "true",
    )
    try:
        database.stream_write(
            pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
            qualified,
            mode="d",
        )
        frame = database.read_query(
            f"SELECT id, name FROM {qualified} ORDER BY id",
            chunksize=1,
        )
        assert frame["id"].tolist() == [1, 2]
    finally:
        database.execute(f"DROP TABLE IF EXISTS {qualified}")
        database.close()
