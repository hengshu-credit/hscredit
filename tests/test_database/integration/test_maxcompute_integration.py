"""真实 MaxCompute 查询、原生写入和元数据集成测试。"""

import os
import uuid

import pandas as pd
import pytest

from hscredit.database import Database

pytestmark = pytest.mark.skipif(
    not os.getenv("HSCREDIT_TEST_MAXCOMPUTE_PROJECT"),
    reason="未配置 HSCREDIT_TEST_MAXCOMPUTE_PROJECT",
)


def test_maxcompute_real_query_write_and_metadata():
    table = f"hscredit_db_it_{uuid.uuid4().hex[:12]}"
    project = os.environ["HSCREDIT_TEST_MAXCOMPUTE_PROJECT"]
    qualified = f"{project}.{table}"
    database = Database(
        "maxcompute",
        access_id=os.environ["HSCREDIT_TEST_MAXCOMPUTE_ACCESS_ID"],
        access_key=os.environ["HSCREDIT_TEST_MAXCOMPUTE_ACCESS_KEY"],
        project=project,
        endpoint=os.environ["HSCREDIT_TEST_MAXCOMPUTE_ENDPOINT"],
    )
    try:
        database.stream_write(
            pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
            qualified,
            mode="d",
        )
        frame = database.read_query(f"SELECT id, name FROM {qualified}", chunksize=1)
        assert sorted(frame["id"].tolist()) == [1, 2]
        assert not database.export_schema(targets=[qualified]).empty
    finally:
        database.adapter.odps.delete_table(qualified, if_exists=True)
        database.close()
