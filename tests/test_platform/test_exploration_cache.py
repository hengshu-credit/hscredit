"""验证字段概况按标签和数据版本复用，图表变化不重复计算。"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.platform import create_app
from hscredit.platform.analysis import explore
from hscredit.platform.exploration import explore_dataset
from hscredit.platform.storage import Workspace, records


@pytest.fixture
def setup(tmp_path, monkeypatch):
    frame = pd.DataFrame({"评分": range(40), "收入": [i * i for i in range(40)], "标签": [0, 1] * 20, "标签二": ["好", "坏", None, "好"] * 10})
    calls = []
    original = pd.DataFrame.summary

    def summary(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "summary", summary)
    with TestClient(create_app(tmp_path)) as client:
        metadata = client.app.state.workspace.add_dataset(frame, "字段概况缓存")
        yield client, metadata, frame, calls


def test_summary_reused_across_navigation_charts_rename_and_workspace_restart(setup):
    client, metadata, frame, calls = setup
    identifier = metadata["id"]
    url = f"/api/datasets/{identifier}/explore"
    first = client.get(url, params={"feature": "评分"}).json()
    assert len(calls) == 1
    repeated = client.post(url, json={"feature": "收入", "other": "评分", "bins": 5})
    assert repeated.status_code == 200
    assert repeated.json()["summary"] == first["summary"]
    assert len(repeated.json()["distribution"]) == 5
    assert repeated.json()["scatter_total"] == len(frame)
    assert repeated.json()["distribution"] != first["distribution"]
    assert client.patch(f"/api/datasets/{identifier}", json={"name": "重命名"}).status_code == 200
    assert client.get(url).json()["summary"] == first["summary"]
    restarted = Workspace(client.app.state.workspace.root)
    assert explore_dataset(restarted, identifier)["summary"] == first["summary"]
    assert len(calls) == 1


def test_management_summary_uses_all_fields_without_labels_and_pins_version(setup, monkeypatch):
    client, source, frame, calls = setup
    ws = client.app.state.workspace
    ws.update("datasets", source["id"], default_target="标签")
    expected = records(frame.summary(n_jobs=1, show_progress=False))
    url = f"/api/datasets/{source['id']}"
    labeled = client.post(url + "/explore", json={"targets": ["标签"], "positive_labels": {"标签": 1}}).json()
    assert "标签" not in {row["特征名"] for row in labeled["summary"]}
    calls.clear()
    monkeypatch.setattr("hscredit.platform.exploration.explore_charts", lambda *args, **kwargs: pytest.fail("字段概况不应生成图表"))
    result = client.get(url + "/summary", params={"revision": 1}).json()
    assert result["summary"] == expected
    assert {row["特征名"] for row in result["summary"]} == set(frame.columns)
    assert [col["label"] for col in result["summary_columns"]] == list(expected[0])
    assert len(calls) == 1 and "target" not in calls[0]
    ws.replace_dataset(source["id"], frame.iloc[:8], expected_revision=1, processing={"description": "截取样本"})
    assert client.get(url + "/summary", params={"revision": 2}).json()["rows"] == 8
    monkeypatch.setattr(ws, "read_dataset", lambda *args, **kwargs: pytest.fail("已缓存的版本不应读取数据文件"))
    assert client.get(url + "/summary", params={"revision": 1}).json() == result


def test_label_changes_and_data_revisions_compute_matching_public_results(setup):
    client, metadata, frame, calls = setup
    workspace = client.app.state.workspace
    identifier = metadata["id"]
    url = f"/api/datasets/{identifier}/explore"
    configurations = [{}, {"targets": ["标签"], "positive_labels": {"标签": 1}}, {"targets": ["标签"], "positive_labels": {"标签": 0}}, {"targets": ["标签", "标签二"], "positive_labels": {"标签": 0, "标签二": "坏"}}]
    for config in configurations:
        expected = explore(frame, **config)
        before = len(calls)
        actual = client.post(url, json=config).json()
        assert actual == expected
        assert len(calls) == before + 1
        assert client.post(url, json=config).json() == expected
        assert len(calls) == before + 1
    before = len(calls)
    assert client.get(url).status_code == 200  # 清空标签恢复已有无标签摘要。
    assert len(calls) == before
    updated = frame.assign(评分=100)
    workspace.replace_dataset(identifier, updated, expected_revision=1, processing={"description": "更新评分"})
    result = client.get(url).json()
    assert len(calls) == before + 1
    assert next(row for row in result["summary"] if row["特征名"] == "评分")["平均值"] == 100
    assert explore_dataset(workspace, identifier, revision=1) == explore(frame)
    second = workspace.add_dataset(frame.assign(评分=200), "另一数据集")
    result = client.get(f"/api/datasets/{second['id']}/explore").json()
    assert next(row for row in result["summary"] if row["特征名"] == "评分")["平均值"] == 200


def test_invalid_labels_do_not_publish_cache_and_chart_errors_do_not_poison_summary(setup):
    client, metadata, _, calls = setup
    url = f"/api/datasets/{metadata['id']}/explore"
    directory = client.app.state.workspace.path("datasets", metadata["id"]) / "summary-cache"
    assert client.post(url, json={"targets": ["标签"], "positive_labels": {"标签": 5}}).status_code == 400
    assert not directory.exists()
    assert client.get(url, params={"feature": "不存在"}).status_code == 400
    assert len(calls) == 1
    assert client.get(url, params={"feature": "评分"}).status_code == 200
    assert len(calls) == 1


def test_external_psi_uses_native_val_df_and_cache_binds_both_revisions(setup):
    client, source, frame, calls = setup
    ws = client.app.state.workspace
    shifted = pd.DataFrame({"评分": list(range(20, 60))})
    reference = ws.add_dataset(shifted, "时间外样本")
    url = f"/api/datasets/{source['id']}/explore"
    config = {"psi_dataset": {"id": reference["id"], "revision": 1}}
    expected = frame.summary(val_df=shifted, n_jobs=1, show_progress=False).set_index("特征名")
    response = client.post(url, json=config)
    assert response.status_code == 200, response.text
    summary = {row["特征名"]: row for row in response.json()["summary"]}
    assert summary["评分"]["PSI"] == pytest.approx(expected.loc["评分", "PSI"])
    assert summary["收入"]["PSI"] is None
    assert summary["评分"]["平均值"] == pytest.approx(frame["评分"].mean())
    count = len(calls)
    ws.replace_dataset(reference["id"], frame[["评分"]], expected_revision=1, processing={"description":"更新对照样本"})
    assert client.post(url, json=config).json()["summary"] == response.json()["summary"]
    assert len(calls) == count
    updated = client.post(url, json={"psi_dataset":{"id":reference["id"]}}).json()
    assert next(row["PSI"] for row in updated["summary"] if row["特征名"] == "评分") == pytest.approx(0)
    assert len(calls) == count + 1
