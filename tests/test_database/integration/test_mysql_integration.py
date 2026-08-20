"""真实 MySQL 连接池、流式读取和写入模式集成测试。"""

import os
import uuid

import pandas as pd
import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_MYSQL_HOST"),
    reason="未配置 HSCREDIT_TEST_MYSQL_HOST",
)


def test_mysql_real_pool_stream_and_write_modes():
    table = f"hscredit_database_it_{uuid.uuid4().hex[:12]}"
    database_name = os.environ["HSCREDIT_TEST_MYSQL_DATABASE"]
    qualified = f"{database_name}.{table}"
    database = Database(
        "mysql",
        host=os.environ["HSCREDIT_TEST_MYSQL_HOST"],
        port=int(os.getenv("HSCREDIT_TEST_MYSQL_PORT", "3306")),
        user=os.environ["HSCREDIT_TEST_MYSQL_USER"],
        password=os.getenv("HSCREDIT_TEST_MYSQL_PASSWORD", ""),
        database=database_name,
        pool_options={"maxconnections": 3, "blocking": True},
    )
    try:
        original = pd.DataFrame({"id": [1], "name": ["原值"]})
        database.stream_write(
            original,
            qualified,
            mode="d",
            key_columns=["id"],
        )
        database.stream_write(
            pd.DataFrame({"id": [1, 2], "name": ["忽略", "新增"]}),
            qualified,
            mode="a",
        )
        after_append = database.query(f"SELECT id, name FROM `{table}` ORDER BY id")
        assert after_append.to_dict("records") == [
            {"id": 1, "name": "原值"},
            {"id": 2, "name": "新增"},
        ]

        database.stream_write(
            pd.DataFrame({"id": [1], "name": ["覆盖"]}),
            qualified,
            mode="r",
        )
        assert database.query(f"SELECT name FROM `{table}` WHERE id=1").iloc[0, 0] == "覆盖"

        stream = database.stream_query(
            f"SELECT id, name FROM `{table}` ORDER BY id",
            chunksize=1,
        )
        next(stream)
        stream.stop()
        assert len(stream.to_dataframe()) == 1
    finally:
        database.execute(f"DROP TABLE IF EXISTS `{table}`")
        database.close()
