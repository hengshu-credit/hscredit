"""真实 Oracle 流式读取和写入模式集成测试。"""

import os
import uuid

import pandas as pd
import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_ORACLE_DSN"),
    reason="未配置 HSCREDIT_TEST_ORACLE_DSN",
)


def test_oracle_real_pool_stream_and_merge_modes():
    table = f"HSCREDIT_DB_IT_{uuid.uuid4().hex[:10].upper()}"
    database = Database(
        "oracle",
        user=os.environ["HSCREDIT_TEST_ORACLE_USER"],
        password=os.environ["HSCREDIT_TEST_ORACLE_PASSWORD"],
        dsn=os.environ["HSCREDIT_TEST_ORACLE_DSN"],
        pool_options={"maxconnections": 3, "blocking": True},
    )
    try:
        database.stream_write(
            pd.DataFrame({"ID": [1], "NAME": ["原值"]}),
            table,
            mode="d",
            key_columns=["ID"],
        )
        database.stream_write(
            pd.DataFrame({"ID": [1, 2], "NAME": ["忽略", "新增"]}),
            table,
            mode="a",
        )
        database.stream_write(
            pd.DataFrame({"ID": [1], "NAME": ["覆盖"]}),
            table,
            mode="r",
        )
        frame = database.query(f'SELECT "ID", "NAME" FROM "{table}" ORDER BY "ID"')
        assert frame.to_dict("records") == [
            {"ID": 1, "NAME": "覆盖"},
            {"ID": 2, "NAME": "新增"},
        ]
    finally:
        database.execute(f'DROP TABLE "{table}" PURGE')
        database.close()
