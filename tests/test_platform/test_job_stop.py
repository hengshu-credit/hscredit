"""真实进程停止、队列取消、结束竞争与探索任务回归。"""

import json
import sys
import time

import pandas as pd
import psutil
from fastapi.testclient import TestClient

from hscredit.platform import create_app


def wait_until(check, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("等待任务状态超时")


def test_stop_kills_worker_and_child_and_cancels_queue(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
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


def test_service_shutdown_stops_active_process(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.app.state.workspace.add_dataset(pd.DataFrame({"x": [1, 2]}), "关闭验证")
        monkeypatch.setattr(client.app.state.jobs, "_command", lambda identifier: [sys.executable, "-c", "import time;time.sleep(120)"])
        job = client.post("/api/jobs", json={"kind": "explore", "dataset_id": dataset["id"], "feature": "x"}).json()
        pid = wait_until(lambda: client.get(f"/api/jobs/{job['id']}").json().get("worker_pid"))
    assert not psutil.pid_exists(pid)
    result = json.loads((tmp_path / "jobs" / job["id"] / "metadata.json").read_text("utf-8"))
    assert result["status"] == "已停止"


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
