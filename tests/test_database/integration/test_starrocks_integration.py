"""真实 StarRocks 查询、DDL 和写入集成测试。"""

import os
import uuid

import pandas as pd
import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_STARROCKS_HOST"),
    reason="未配置 HSCREDIT_TEST_STARROCKS_HOST",
)


def test_starrocks_real_pool_stream_and_table_models():
    table = f"hscredit_db_it_{uuid.uuid4().hex[:12]}"
    database_name = os.environ["HSCREDIT_TEST_STARROCKS_DATABASE"]
    qualified = f"{database_name}.{table}"
    database = Database(
        "starrocks",
        host=os.environ["HSCREDIT_TEST_STARROCKS_HOST"],
        port=int(os.getenv("HSCREDIT_TEST_STARROCKS_PORT", "9030")),
        user=os.getenv("HSCREDIT_TEST_STARROCKS_USER", "root"),
        password=os.getenv("HSCREDIT_TEST_STARROCKS_PASSWORD", ""),
        database=database_name,
    )
    try:
        database.stream_write(
            pd.DataFrame({"id": [1], "name": ["原值"]}),
            qualified,
            mode="d",
            key_columns=["id"],
            dialect_options={"table_model": "PRIMARY KEY"},
        )
        database.stream_write(
            pd.DataFrame({"id": [1], "name": ["覆盖"]}),
            qualified,
            mode="r",
            dialect_options={"table_model": "PRIMARY KEY"},
        )
        assert database.query(f"SELECT name FROM `{table}` WHERE id=1").iloc[0, 0] == "覆盖"
    finally:
        database.execute(f"DROP TABLE IF EXISTS `{table}`")
        database.close()
