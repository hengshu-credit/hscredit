"""真实进程停止、队列取消、结束竞争与探索任务回归。"""

import json
import sys
import time

import pandas as pd
import psutil
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app


def wait_until(check, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("等待任务状态超时")


def test_stop_kills_worker_and_child_and_cancels_queue(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path, max_concurrent_jobs=1)) as client:
        workspace, manager = client.app.state.workspace, client.app.state.jobs
        dataset = workspace.add_dataset(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), "停止验证")
        script = tmp_path / "slow_worker.py"
        script.write_text("import json,sys,subprocess,time\nfrom pathlib import Path\nd=Path(sys.argv[1])/'jobs'/sys.argv[2]\nchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n(d/'child.pid').write_text(str(child.pid))\nwhile True:time.sleep(.1)\n", encoding="utf-8")
        monkeypatch.setattr(manager, "_command", lambda identifier: [sys.executable, str(script), str(tmp_path), identifier])
        body = {"kind": "model", "dataset_id": dataset["id"], "features": ["x"], "target": "y"}
        first = client.post("/api/jobs", json=body).json()
        queued = client.post("/api/jobs", json=body).json()
        child_file = tmp_path / "jobs" / first["id"] / "child.pid"
        wait_until(child_file.exists)
        pid = client.get(f"/api/jobs/{first['id']}").json()["worker_pid"]
        child = int(child_file.read_text())
        assert psutil.pid_exists(pid) and psutil.pid_exists(child)
        assert client.post(f"/api/jobs/{queued['id']}/stop").json()["status"] == "已停止"
        stopped = client.post(f"/api/jobs/{first['id']}/stop")
        assert stopped.status_code == 202
        wait_until(lambda: client.get(f"/api/jobs/{first['id']}").json()["status"] == "已停止")
        assert not psutil.pid_exists(pid)
        assert not psutil.pid_exists(child)
        assert client.get(f"/api/jobs/{queued['id']}").json().get("worker_pid") is None
        assert client.get(f"/api/jobs/{first['id']}/files/report.xlsx").status_code == 404
        assert client.post(f"/api/jobs/{first['id']}/stop").json()["status"] == "已停止"
        # 已停止后数据集锁释放，用户可继续加工。
        payload = {"expected_revision": 1, "rules": [{"target": "z", "mode": "add", "expression": {"kind": "FIELD", "name": "x"}}]}
        assert client.post(f"/api/datasets/{dataset['id']}/processing/apply", json=payload).status_code == 200
        # 队列仍能继续执行，且任务完成后的停止不会丢弃已有结果。
        fast = tmp_path / "fast_worker.py"
        fast.write_text("import json,sys\nfrom pathlib import Path\nd=Path(sys.argv[1])/'jobs'/sys.argv[2]\n(d/'ok.txt').write_text('ok')\n(d/'outcome.json').write_text(json.dumps({'ok':True,'result':{'artifacts':['ok.txt'],'tables':{}}}))\n", encoding="utf-8")
        monkeypatch.setattr(manager, "_command", lambda identifier: [sys.executable, str(fast), str(tmp_path), identifier])
        third = client.post("/api/jobs", json=body).json()
        wait_until(lambda: client.get(f"/api/jobs/{third['id']}").json()["status"] == "成功")
        assert client.post(f"/api/jobs/{third['id']}/stop").json()["status"] == "成功"
        assert client.get(f"/api/jobs/{third['id']}/files/ok.txt").text == "ok"


def test_parallel_jobs_release_one_slot_and_keep_results_separate(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.app.state.workspace.add_dataset(pd.DataFrame({"x": [1, 2]}), "并发验证")
        script = tmp_path / "worker.py"
        script.write_text(
            "import json, sys, time\n" "from pathlib import Path\n" "job = sys.argv[1]\n" "Path('ready').touch()\n" "while not Path('finish').exists(): time.sleep(.05)\n" "Path('output.txt').write_text(job)\n" "Path('outcome.json').write_text(json.dumps({'ok': True, 'result': {'artifacts': ['output.txt']}}))\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(client.app.state.jobs, "_command", lambda job: [sys.executable, str(script), job])
        body = {"kind": "explore", "dataset_id": dataset["id"], "feature": "x"}
        first, second, third = [client.post("/api/jobs", json=body).json()["id"] for _ in range(3)]
        jobs = tmp_path / "jobs"

        def status(job):
            return client.get(f"/api/jobs/{job}").json()["status"]

        wait_until(lambda: all((jobs / job / "ready").exists() for job in (first, second)))
        assert [status(job) for job in (first, second, third)] == ["运行中", "运行中", "排队中"]
        assert client.post(f"/api/jobs/{first}/stop").json()["status"] == "停止中"
        wait_until(lambda: (jobs / third / "ready").exists())
        assert [status(job) for job in (first, second, third)] == ["已停止", "运行中", "运行中"]
        for job in (third, second):
            (jobs / job / "finish").touch()
            wait_until(lambda: status(job) == "成功")
            assert client.get(f"/api/jobs/{job}/files/output.txt").text == job


@pytest.mark.parametrize("limit", [0, 11, 1.5, True])
def test_invalid_concurrency_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="并发任务数"):
        create_app(tmp_path, max_concurrent_jobs=limit)


def test_service_shutdown_stops_parallel_and_queued_jobs(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path, max_concurrent_jobs=3)) as client:
        dataset = client.app.state.workspace.add_dataset(pd.DataFrame({"x": [1, 2]}), "关闭验证")
        monkeypatch.setattr(client.app.state.jobs, "_command", lambda identifier: [sys.executable, "-c", "import time;time.sleep(120)"])
        body = {"kind": "explore", "dataset_id": dataset["id"], "feature": "x"}
        jobs = [client.post("/api/jobs", json=body).json()["id"] for _ in range(4)]
        pids = [wait_until(lambda: client.get(f"/api/jobs/{job}").json().get("worker_pid")) for job in jobs[:-1]]
        assert len(set(pids)) == 3
    assert all(not psutil.pid_exists(pid) for pid in pids)
    for job in jobs:
        result = json.loads((tmp_path / "jobs" / job / "metadata.json").read_text("utf-8"))
        assert result["status"] == "已停止"
        if job == jobs[-1]:
            assert "worker_pid" not in result


def test_explore_job_matches_existing_explore_api(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.app.state.workspace.add_dataset(pd.DataFrame({"x": range(30), "y": [0, 1] * 15}), "探索任务")
        params = {"feature": "x", "targets": ["y"], "positive_labels": {"y": 1}, "bins": 5}
        expected = client.post(f"/api/datasets/{dataset['id']}/explore", json=params).json()
        cache = next((tmp_path / "datasets" / dataset["id"] / "summary-cache").glob("*.json"))
        cached_at = cache.stat().st_mtime_ns
        job = client.post("/api/jobs", json={"kind": "explore", "dataset_id": dataset["id"], **params}).json()
        wait_until(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] in {"成功", "失败"}, timeout=40)
        actual = client.get(f"/api/jobs/{job['id']}").json()
        assert actual["status"] == "成功", actual
        assert actual["result"]["exploration"] == expected
        assert cache.stat().st_mtime_ns == cached_at  # 独立进程复用接口已完成的摘要。
