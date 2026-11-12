"""验证多级透视、跨页合计与任务样本版本的绑定。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.platform.interactions import interactions
from hscredit.platform.pivot import PivotRequest, pivot_table
from hscredit.platform.storage import Workspace


@pytest.fixture
def example(tmp_path):
    ws = Workspace(tmp_path)
    n = np.arange(3600)
    frame = pd.DataFrame({"行1": n % 40, "行2": n % 3, "行3": n % 2, "列1": n % 17, "列2": n % 5, "列3": n % 7, "值": (n % 11).astype(float), "组": n % 2})
    frame.loc[::13, "值"] = np.nan
    root = ws.add_dataset(frame, "组合样本")
    result = interactions(frame, ["行1", "列1"], target="组", bins=4)
    job = {"dataset_id": root["id"], "dataset_revision": 1, "config": {"bins": 4}, "result": {"interaction": result}}
    ws.replace_dataset(root["id"], frame.assign(值=0), expected_revision=1, processing={"description": "更新值字段"})
    return ws, frame, job


@pytest.mark.parametrize("method", ["sum", "mean", "median", "min", "max", "count", "nunique", "rows"])
def test_six_dimensions_and_totals_cover_other_pages(example, method):
    ws, data, job = example
    req = PivotRequest(rows=["行1", "行2", "行3"], columns=["列1", "列2", "列3"], value="值", aggregation=method, buckets=False, page=2, column_page=2)
    out = pivot_table(ws, job, req)
    def expected(part):
        return len(part) if method == "rows" else getattr(part["值"], method)()
    assert out["row_count"] > 30 and out["column_count"] > 12
    assert out["total"] == pytest.approx(expected(data))
    for i, key in enumerate(out["row_keys"]):
        part = data.loc[(data[req.rows] == list(map(int, key))).all(axis=1)]
        assert out["row_totals"][i] == pytest.approx(expected(part))
        for j, col in enumerate(out["column_keys"]):
            cell = part.loc[(part[req.columns] == list(map(int, col))).all(axis=1)]
            if len(cell) and cell["值"].notna().any():
                assert out["values"][i][j] == pytest.approx(expected(cell))
            elif not len(cell):
                assert out["values"][i][j] == (0 if method in {"rows", "count", "nunique"} else None)


def test_density_and_pivot_agree_under_group_filter(example):
    ws, _, job = example
    req = PivotRequest(rows=["列1"], columns=["行1"], group="1")
    out = pivot_table(ws, job, req)
    expected = {(r["y"], r["x"]): r["样本数"] for r in job["result"]["interaction"]["pairs"][0]["density"] if r["标签"] == "1"}
    assert out["total"] == sum(expected.values())
    for i, (r,) in enumerate(out["row_keys"]):
        for j, (c,) in enumerate(out["column_keys"]):
            assert out["values"][i][j] == expected.get((r, c), 0)
    empty = pivot_table(ws, job, req.copy(update={"group": "不存在"}))
    assert empty["samples"] == 0 and empty["values"] == []


def test_numeric_precision_and_empty_cells(tmp_path):
    ws = Workspace(tmp_path)
    data = pd.DataFrame({"行": ["甲", "甲", "乙"], "列": ["是", "是", "否"], "值": np.array([100000000, 1, np.nan], dtype="float32")})
    root = ws.add_dataset(data, "精度")
    job = {"dataset_id": root["id"], "dataset_revision": 1, "config": {"bins": 2}, "result": {"interaction": interactions(data, ["行", "列"])}}
    out = pivot_table(ws, job, PivotRequest(rows=["行"], columns=["列"], aggregation="mean", value="值"))
    assert out["total"] == 50000000.5
    i = out["row_keys"].index(["乙"])
    assert all(cell is None for cell in out["values"][i])
