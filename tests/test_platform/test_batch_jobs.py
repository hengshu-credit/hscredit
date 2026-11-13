"""批量运行保留每份样本的行归属、版本和任务记录。"""

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app


@pytest.fixture
def assets(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = pd.DataFrame({"评分": range(200), "标签": [0, 1] * 100})
        root = client.app.state.workspace.add_dataset(frame, "全量")
        split = {"expected_revision": 1, "method": "random", "stratify": "标签", "parts": [{"name": "开发", "ratio": 0.65}, {"name": "留出", "ratio": 0.35}]}
        children = client.post(f"/api/datasets/{root['id']}/split/apply", json=split).json()["datasets"]
        yield client, [root, *children]


def submit(client, datasets, config):
    return client.post(
        "/api/jobs/batch",
        json={
            "datasets": [{"id": d["id"], "revision": d["revision"]} for d in datasets],
            "config": {"dataset_id": datasets[0]["id"], **config},
        },
    )


def test_batch_exploration_runs_separately_on_pinned_subsets(assets):
    client, datasets = assets
    response = submit(client, datasets, {"kind": "explore", "feature": "评分", "bins": 5})
    assert response.status_code == 202, response.text
    jobs = response.json()["jobs"]
    assert len({j["batch_id"] for j in jobs}) == 1
    assert len({j["id"] for j in jobs}) == 3
    assert [j["batch_position"] for j in jobs] == [0, 1, 2]
    # 排队后重新划分，已提交的任务仍使用旧版的 130 / 70 行。
    info = datasets[1]["splitting"]
    resplit = client.post(f"/api/datasets/{datasets[0]['id']}/split/apply", params={"plan_id": info["plan_id"], "plan_revision": 1}, json={"expected_revision": 1, "parts": [{"name": "开发", "ratio": 0.5}, {"name": "留出", "ratio": 0.5}]})
    assert resplit.status_code == 200
    for job, asset in zip(jobs, datasets):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            actual = client.get(f"/api/jobs/{job['id']}").json()
            if actual["status"] in {"成功", "失败"}:
                break
            time.sleep(0.1)
        assert actual["status"] == "成功", actual["message"]
        assert actual["dataset_id"] == actual["config"]["dataset_id"] == asset["id"]
        assert actual["dataset_revision"] == actual["config"]["dataset_revision"] == 1
        result = actual["result"]["exploration"]
        assert result["rows"] == sum(row["样本数"] for row in result["distribution"]) == asset["rows"]


@pytest.mark.parametrize(
    "config",
    [
        {"kind": "interaction", "features": ["评分"]},
        {"kind": "strategy", "workflow": True, "features": ["评分"], "target": "标签"},
        {"kind": "model", "workflow": True, "models": ["logistic"], "features": ["评分"], "target": "标签"},
        {"kind": "third_party", "features": ["评分"], "target": "标签"},
    ],
)
def test_batch_uses_each_dataset_as_primary(assets, config, monkeypatch):
    client, datasets = assets
    queued = []
    monkeypatch.setattr(client.app.state.jobs, "enqueue", queued.append)
    response = submit(client, datasets, config)
    assert response.status_code == 202, response.text
    jobs = response.json()["jobs"]
    assert queued == [j["id"] for j in jobs]
    assert [j["dataset_name"] for j in jobs] == ["全量", "开发", "留出"]
    for job, asset in zip(jobs, datasets):
        assert job["config"]["dataset_id"] == asset["id"]
        assert job["config"]["evaluation_datasets"] == []
        assert job["config"]["kind"] == config["kind"]


def test_invalid_batch_and_queue_capacity_do_not_partially_submit(assets, monkeypatch):
    client, datasets = assets
    queued = []
    monkeypatch.setattr(client.app.state.jobs, "enqueue", queued.append)
    bad = client.app.state.workspace.add_dataset(pd.DataFrame({"其他": [1, 2]}), "缺少评分")
    config = {"kind": "explore", "feature": "评分"}
    error = submit(client, [datasets[0], bad], config)
    assert error.status_code == 400 and bad["name"] in error.json()["detail"]
    assert submit(client, [datasets[0], datasets[0]], config).status_code == 400
    assert not queued and not client.get("/api/jobs").json()
    for _ in range(3):
        assert submit(client, datasets, config).status_code == 202
    assert submit(client, datasets, config).status_code == 429
    assert len(queued) == len(client.get("/api/jobs").json()) == 9
