"""索引子集的物理存储、重新划分、历史任务绑定和旧版副本兼容验证。"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app
from hscredit.web.storage import Workspace


def configuration(seed, revision=1):
    return {"expected_revision": revision, "method": "random", "random_state": seed, "parts": [{"name": "训练", "ratio": 0.7}, {"name": "测试", "ratio": 0.3}]}


def binding(child):
    return {key: child["splitting"][key] for key in ("plan_id", "plan_revision")}


def test_copy_and_split_keeps_source_plan_and_child_rows(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        ws = client.app.state.workspace
        frame = pd.DataFrame({"编号": range(120), "标签": [0, 1] * 60})
        source = ws.add_dataset(frame, "全部", defaults={"default_target": "标签"})
        url = f"/api/datasets/{source['id']}"
        original = client.post(url + "/split/apply", json=configuration(7)).json()["datasets"]
        train = original[0]
        train_frame = ws.read_dataset(train["id"])
        changed = frame.iloc[::-1].reset_index(drop=True)
        ws.replace_dataset(source["id"], changed, expected_revision=1, processing={"description": "调整行序"})
        assert client.post(url + "/copy", json={"expected_revision": 1}).status_code == 409
        copy = client.post(url + "/copy", json={"expected_revision": 2}).json()
        assert copy["name"] == "全部（副本）" and copy["revision"] == 1
        assert "parent_id" not in copy and "splitting" not in copy and copy["default_target"] == "标签"
        pd.testing.assert_frame_equal(ws.read_dataset(copy["id"]), changed)
        another = configuration(17)
        another["parts"] = [{"name": "开发", "ratio": 0.5}, {"name": "留出", "ratio": 0.5}]
        results = client.post(f"/api/datasets/{copy['id']}/split/apply", json=another).json()["datasets"]
        assert [d["rows"] for d in results] == [60, 60]
        assert all(d["parent_id"] == copy["id"] for d in results)
        saved = [d for d in Workspace(tmp_path).list("datasets") if d.get("parent_id") == source["id"]]
        assert sorted(saved, key=lambda d: d["splitting"]["part_index"]) == original
        repeat = client.post(url + "/copy", json={"expected_revision": 2}).json()
        assert repeat["name"] == "全部（副本 2）"
        denied = client.post(f"/api/datasets/{train['id']}/copy", json={"expected_revision": 1})
        assert denied.status_code == 400 and "仅原始数据集支持复制" in denied.text
        pd.testing.assert_frame_equal(ws.read_dataset(train["id"]), train_frame)


def test_repartition_preserves_identity_and_historical_inputs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        ws = client.app.state.workspace
        frame = pd.DataFrame({"编号": range(100), "日期": pd.date_range("2025-01-01", periods=100), "标签": [1, 0] * 50})
        root = ws.add_dataset(frame, "全部")
        # 分成多个 Parquet 行组，验证跨组取行、指定列及输出顺序。
        frame.to_parquet(ws.data_path(root), row_group_size=9, index=False)
        url = f"/api/datasets/{root['id']}/split"
        initial = client.post(url + "/apply", json=configuration(42)).json()["datasets"]
        previous = ws.read_dataset(initial[0]["id"])
        npys = list(tmp_path.rglob("*.npy"))
        assert len(npys) == 2
        assert sorted(np.concatenate([np.load(p, allow_pickle=False) for p in npys])) == list(range(100))
        assert len(list((tmp_path / "datasets").iterdir())) == 1
        assert len(list(tmp_path.rglob("*.parquet"))) == 1
        stored = json.loads(next((ws.path("datasets", root["id"]) / "splits").rglob("*.json")).read_text(encoding="utf-8"))
        assert all("values" not in col and "example_indices" in col for col in stored["columns"])
        assert ws.get("datasets", initial[0]["id"])["columns"] == initial[0]["columns"]
        pd.testing.assert_frame_equal(ws.read_dataset(initial[0]["id"], columns=["标签", "编号"]), previous[["标签", "编号"]])
        grand = configuration(9)
        grand["parts"] = [{"name": "内训练", "ratio": 0.5}, {"name": "内验证", "ratio": 0.5}]
        nested = client.post(f"/api/datasets/{initial[0]['id']}/split/apply", json=grand).json()["datasets"][0]
        nested_frame = ws.read_dataset(nested["id"])
        assert set(nested_frame["编号"]) < set(previous["编号"])

        # 更新源数据的值和行序，已有子集仍取版本 1；重新划分使用版本 2。
        updated = frame.iloc[::-1].reset_index(drop=True).assign(金额=10)
        ws.replace_dataset(root["id"], updated, expected_revision=1, processing={"description": "新增字段"})
        params = binding(initial[0])
        request = configuration(7, 2)
        request["parts"][0]["name"] = "新训练"
        assert client.post(url + "/preview", params=params, json=request).status_code == 200
        pd.testing.assert_frame_equal(ws.read_dataset(initial[0]["id"]), previous)
        response = client.post(url + "/apply", params=params, json=request)
        assert response.status_code == 200, response.text
        children = response.json()["datasets"]
        assert [d["id"] for d in children] == [d["id"] for d in initial]
        assert all(d["revision"] == 2 and d["parent_id"] == root["id"] for d in children)
        reloaded = Workspace(tmp_path)
        pd.testing.assert_frame_equal(reloaded.read_dataset(children[0]["id"], 1), previous)
        from hscredit.web.workflows import bound_frames

        frozen = bound_frames(reloaded, {"dataset_id": initial[0]["id"], "dataset_revision": 1, "test_dataset": {"id": initial[1]["id"], "revision": 1}})
        pd.testing.assert_frame_equal(frozen["训练集 · 训练"], previous)
        assert set(frozen["测试集 · 测试"]["编号"]) == set(frame["编号"]) - set(previous["编号"])
        pd.testing.assert_frame_equal(reloaded.read_dataset(nested["id"]), nested_frame)
        pd.testing.assert_frame_equal(pd.concat([reloaded.read_dataset(d["id"]) for d in children]).sort_values("编号").reset_index(drop=True), updated.sort_values("编号").reset_index(drop=True))
        assert "金额" in client.get(f"/api/datasets/{children[0]['id']}/preview").json()["rows"][0]
        assert client.post(url + "/apply", params=params, json=request).status_code == 409
        assert client.delete(f"/api/datasets/{root['id']}").status_code == 400
        assert len(list(tmp_path.rglob("*.parquet"))) == 2


def test_remove_output_and_failure_keep_published_plan(tmp_path, monkeypatch):
    import hscredit.web.subsets as subsets

    with TestClient(create_app(tmp_path)) as client:
        ws = client.app.state.workspace
        root = ws.add_dataset(pd.DataFrame({"a": range(120)}), "源")
        url = f"/api/datasets/{root['id']}/split"
        req = configuration(1)
        req["parts"] = [{"name": name, "ratio": ratio} for name, ratio in [("训练", 0.6), ("验证", 0.2), ("测试", 0.2)]]
        first = client.post(url + "/apply", json=req).json()["datasets"]
        before = ws.read_dataset(first[1]["id"])
        orig = subsets.write_json

        def fail(path, value):
            if path == ws.subsets.catalog:
                raise OSError("模拟方案发布失败")
            orig(path, value)

        params = binding(first[0])
        with monkeypatch.context() as m:
            m.setattr(subsets, "write_json", fail)
            with pytest.raises(OSError, match="模拟方案发布失败"):
                client.post(url + "/apply", params=params, json=configuration(17))
        assert len(list(tmp_path.rglob("*.npy"))) == 3
        pd.testing.assert_frame_equal(ws.read_dataset(first[1]["id"]), before)
        second = client.post(url + "/apply", params=params, json=configuration(17)).json()["datasets"]
        assert {d["id"] for d in ws.list("datasets")} == {root["id"], *(d["id"] for d in second)}
        with pytest.raises(FileNotFoundError):
            ws.read_dataset(first[1]["id"])
        pd.testing.assert_frame_equal(ws.read_dataset(first[1]["id"], 1), before)
        assert client.delete(f"/api/datasets/{root['id']}/split-plans/{params['plan_id']}", params={"plan_revision": 2}).status_code == 200
        assert [d["id"] for d in ws.list("datasets")] == [root["id"]]
        pd.testing.assert_frame_equal(ws.read_dataset(first[1]["id"], 1), before)


def test_legacy_copies_can_be_reopened_then_converted(tmp_path):
    from hscredit.web.splitting import SplitRequest, split_frame

    with TestClient(create_app(tmp_path)) as client:
        ws = client.app.state.workspace
        frame = pd.DataFrame({"x": range(100)})
        root = ws.add_dataset(frame, "旧源")
        config = configuration(31)
        legacy_meta = {"source_dataset_id": root["id"], "source_revision": 1, "config": config, "at": "2025-05-01"}
        legacy = [ws.add_dataset(part, name, splitting={**legacy_meta, "part_index": i}) for i, (name, part) in enumerate(split_frame(frame, SplitRequest(**config)))]
        before = ws.read_dataset(legacy[0]["id"])
        visible = ws.list("datasets")
        old = next(d for d in visible if d["id"] == legacy[0]["id"])
        assert old["parent_id"] == root["id"]
        url = f"/api/datasets/{root['id']}/split/apply"
        result = client.post(url, params=binding(old), json=configuration(30))
        assert result.status_code == 200, result.text
        current = result.json()["datasets"][0]
        assert current["id"] == old["id"] and current["storage"] == "indices" and current["revision"] == 2
        assert len(ws.list("datasets")) == 3
        pd.testing.assert_frame_equal(ws.read_dataset(current["id"], 1), before)
        assert not ws.read_dataset(current["id"]).equals(before)
