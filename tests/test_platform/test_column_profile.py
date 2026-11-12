"""验证表头摘要的全量口径、类型分布、版本绑定与按列读取。"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.platform import create_app
from hscredit.platform.analysis import explore
from hscredit.platform.storage import records


@pytest.fixture
def setup(tmp_path):
    frame = pd.DataFrame({"数值": [0.0] * 69 + [np.nan] + [100.0] * 9 + [np.inf], "文本": ["常见"] * 60 + [f"类别{i}" for i in range(19)] + [None], "时间": pd.to_datetime(["2025-01-01"] * 70 + ["2025-01-04"] * 9 + [None]), "布尔": pd.Series([True] * 50 + [False] * 20 + [None] * 10, dtype="boolean")})
    with TestClient(create_app(tmp_path)) as client:
        metadata = client.app.state.workspace.add_dataset(frame, "悬浮详情验证")
        yield client, metadata, frame


@pytest.mark.parametrize("column,kind", [("数值", "numeric"), ("文本", "category"), ("时间", "date"), ("布尔", "boolean")])
def test_profile_summary_and_distributions_use_all_rows(setup, column, kind):
    client, metadata, frame = setup
    response = client.get(f"/api/datasets/{metadata['id']}/column-profile", params={"column": column, "revision": 1})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["kind"] == kind
    assert result["summary"] == records(frame[[column]].summary(n_jobs=1, show_progress=False))[0]
    assert result["rows"] == 80
    assert result["summary"]["样本数"] == 80
    points = result["distribution"]
    if kind == "numeric":
        assert [p["count"] for p in points] == [69, 9]
        assert points[0]["lower"] == 0 and points[-1]["upper"] == 100
        assert result["invalid"] == 1 and result["missing"] == 1
    elif kind == "category":
        assert len(points) == 14
        assert points[0] == {"label": "常见", "count": 60}
        assert points[-2] == {"label": "其余类别（合计）", "count": 8}
        assert points[-1] == {"label": "缺失值（合计）", "count": 1}
        assert sum(p["count"] for p in points) == 80
    elif kind == "date":
        assert [p["label"] for p in points] == ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        assert [p["count"] for p in points] == [70, 0, 0, 9]
    else:
        assert points == [{"label": "真", "count": 50}, {"label": "假", "count": 20}, {"label": "缺失", "count": 10}]


def test_profile_binds_revision_and_reads_only_requested_column(setup, monkeypatch):
    client, metadata, frame = setup
    workspace = client.app.state.workspace
    original_read = pd.read_parquet
    projections = []

    def read(*args, **kwargs):
        projections.append(kwargs.get("columns"))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    workspace.replace_dataset(metadata["id"], frame.assign(数值=7), expected_revision=1, processing={"description": "更新数值"})
    url = f"/api/datasets/{metadata['id']}/column-profile"
    old = client.get(url, params={"column": "数值", "revision": 1}).json()
    new = client.get(url, params={"column": "数值", "revision": 2}).json()
    assert old["summary"]["唯一值数"] == 3
    assert new["summary"]["唯一值数"] == 1
    assert new["summary"]["平均值"] == 7
    assert [old["revision"], new["revision"]] == [1, 2]
    assert projections == [["数值"], ["数值"]]
    assert client.get(url, params={"column": "不存在"}).status_code == 400
    assert client.get(url, params={"column": "数值", "revision": 3}).status_code == 400


@pytest.mark.parametrize("column,values", [("数值", pd.Series([np.nan, np.nan], dtype=float)), ("布尔", pd.Series([None, None], dtype="boolean")), ("时间", pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")), ("文本", pd.Series([None, None], dtype="object"))])
def test_profile_all_missing_is_json_safe(setup, column, values):
    client, _, _ = setup
    metadata = client.app.state.workspace.add_dataset(values.to_frame(column), "全缺失")
    response = client.get(f"/api/datasets/{metadata['id']}/column-profile", params={"column": column})
    assert response.status_code == 200, response.text
    assert response.json()["missing"] == 2
    assert response.json()["summary"]["缺失率"] == 1


def test_date_strings_keep_invalid_values_outside_distribution(setup):
    client, _, _ = setup
    values = ["2025-01-01 10:00:00"] * 100 + ["2025-01-01 12:00:00", "错误时间", None]
    metadata = client.app.state.workspace.add_dataset(pd.DataFrame({"授信时间": values}), "时间字符串")
    response = client.get(f"/api/datasets/{metadata['id']}/column-profile", params={"column": "授信时间"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["kind"] == "date"
    assert result["invalid"] == 1 and result["missing"] == 1
    assert [p["count"] for p in result["distribution"]] == [100, 0, 1]


def test_default_explore_calls_summary_without_label_arguments(monkeypatch):
    original = pd.DataFrame.summary
    options = []

    def summary(self, *args, **kwargs):
        options.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "summary", summary)
    frame = pd.DataFrame({"评分": range(30), "标签": [0, 1] * 15})
    result = explore(frame)
    assert result["summary"] == records(original(frame, n_jobs=1, show_progress=False))
    assert all("target" not in kwargs and "y" not in kwargs for kwargs in options)
    assert "AUC" not in result["summary"][0]
