"""字段加工的真实计算、原子保存、历史版本与任务输入绑定验证。"""

import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app
from hscredit.web.processing import compile_expression, process_frame
from hscredit.web.storage import Workspace


def field(name):
    return {"kind": "FIELD", "name": name}


def literal(value, kind=None):
    return {"kind": "LITERAL", "value_type": kind or ("BOOLEAN" if isinstance(value, bool) else "STRING" if isinstance(value, str) else "NUMBER"), "value": value}


def operation(left, operator, right):
    return {"kind": "OPERATION", "terms": [{"operand": left}, {"operator": operator, "operand": right}]}


def function(name, *args):
    return {"kind": "FUNCTION", "name": name, "args": list(args)}


def rule(target, expression, mode="add"):
    return {"target": target, "mode": mode, "expression": expression}


@pytest.fixture
def setup(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = pd.DataFrame({"金额": [0.0] * 20 + [5.0] * 80, "渠道名称": ["线上", "线下"] * 50, "标签": [0, 1] * 50})
        metadata = client.app.state.workspace.add_dataset(frame, "字段加工验证")
        yield client, metadata, frame, tmp_path


def test_preview_and_apply_use_numexpr_and_full_population(setup, monkeypatch):
    client, metadata, frame, root = setup
    import hscredit.web.processing as processing

    original = processing.NumExprDerive
    calls = []

    def audited(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(processing, "NumExprDerive", audited)
    rules = [rule("金额", operation(field("金额"), "*", literal(2)), "replace"), rule("整列均值", function("mean", field("金额"))), rule("客群", function("where", operation(field("渠道名称"), "==", literal("线上")), literal("A"), literal("B")))]
    payload = {"expected_revision": 1, "rules": rules}
    url = f"/api/datasets/{metadata['id']}/processing"
    preview = client.post(url + "/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["rows"] == 100
    assert len(preview.json()["after"]) == 20
    assert all(row["整列均值"] == 8 for row in preview.json()["after"])
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"]), frame)
    saved = client.post(url + "/apply", json=payload)
    assert saved.status_code == 200, saved.text
    saved = saved.json()
    assert saved["revision"] == 2
    assert len(saved["columns"]) == 5
    actual = Workspace(root).read_dataset(metadata["id"])
    pd.testing.assert_series_equal(actual["金额"], frame["金额"] * 2)
    assert actual["整列均值"].eq(8).all()
    assert actual["客群"].tolist() == ["A", "B"] * 50
    assert calls and all(call["trusted_python"] is False for call in calls)
    assert all(name not in frame.columns for call in calls for name, _ in call["derivings"])
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"], 1), frame)
    assert saved["processing"]["rules"] == rules


def test_precedence_nested_logic_and_escaped_names():
    frame = pd.DataFrame({'a " quoted': [2, 5], "s": ["x", "y"]})
    tree = {"kind": "OPERATION", "terms": [{"operand": literal(1)}, {"operator": "+", "operand": literal(2)}, {"operator": "*", "operand": literal(3)}]}
    condition = operation(operation(field('a " quoted'), ">", literal(1)), "and", operation(field("s"), "==", literal("x")))
    rules = [rule("优先级", tree), rule("嵌套", operation(operation(literal(1), "+", literal(2)), "*", literal(3))), rule("标签", function("where", condition, literal(1), literal(0)))]
    result, _, _ = process_frame(frame, rules)
    assert result["优先级"].tolist() == [7, 7]
    assert result["嵌套"].tolist() == [9, 9]
    assert result["标签"].tolist() == [1, 0]
    assert list(result.columns[:2]) == list(frame.columns)


def test_multiple_replacements_and_prior_generated_fields():
    frame = pd.DataFrame({"x": [2, 3]})
    result, _, _ = process_frame(frame, [rule("x", operation(field("x"), "+", literal(1)), "replace"), rule("y", operation(field("x"), "*", literal(10))), rule("x", operation(field("y"), "+", field("x")), "replace")])
    assert result.to_dict("list") == {"x": [33, 44], "y": [30, 40]}
    assert frame["x"].tolist() == [2, 3]


@pytest.mark.parametrize(
    "rules",
    [
        [rule("金额", literal(1))],
        [rule("不存在", literal(1), "replace")],
        [rule("a", field("将来字段")), rule("将来字段", literal(1))],
        [rule("a", {"kind": "EXPRESSION", "code": "__import__('os').getcwd()"})],
        [rule("a", function("__import__", literal("os")))],
        [rule("a", operation(literal("x"), "*", literal(1000000)))],
        [rule("a", function("where", literal(True), literal("text"), literal(None, "MISSING")))],
    ],
)
def test_invalid_rules_never_change_dataset(setup, rules):
    client, metadata, frame, root = setup
    response = client.post(f"/api/datasets/{metadata['id']}/processing/apply", json={"expected_revision": 1, "rules": rules})
    assert response.status_code == 400, response.text
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"]), frame)
    assert Workspace(root).get("datasets", metadata["id"])["revision"] == 1


def test_missing_fill_infinite_preview_and_text_injection(setup):
    client, metadata, _, _ = setup
    rules = [rule("比值", operation(literal(1), "/", field("金额"))), rule("文本", literal("__import__('os').system('danger')"))]
    response = client.post(f"/api/datasets/{metadata['id']}/processing/preview", json={"expected_revision": 1, "rules": rules})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["statistics"][0]["无穷值数"] == 20
    assert result["after"][0]["比值"] == "正无穷"
    assert result["after"][0]["文本"] == rules[1]["expression"]["value"]
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    filled, _, _ = process_frame(frame, [rule("x", function("where", function("isnan", field("x")), literal(0), field("x")), "replace")])
    assert filled["x"].tolist() == [1, 0]


def test_save_copy_restore_and_stale_revision(setup):
    client, metadata, original, root = setup
    client.app.state.workspace.update("datasets", metadata["id"], default_target="标签", default_features=["金额"], source="hscredit_dataset")
    url = f"/api/datasets/{metadata['id']}/processing"
    payload = {"expected_revision": 1, "rules": [rule("新增", literal(42))]}
    copy = client.post(url + "/apply", json={**payload, "save_as": True, "name": "加工副本"}).json()
    assert copy["id"] != metadata["id"]
    assert copy["default_target"] == "标签"
    assert copy["default_features"] == ["金额"]
    assert copy.get("source") != "hscredit_dataset"
    assert Workspace(root).read_dataset(copy["id"])["新增"].eq(42).all()
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"]), original)
    assert client.post(url + "/apply", json=payload).json()["revision"] == 2
    assert client.post(url + "/apply", json=payload).status_code == 409
    assert client.post(url + "/preview", json=payload).status_code == 409
    restored = client.post(url + "/restore", json={"revision": 1, "expected_revision": 2})
    assert restored.status_code == 200, restored.text
    assert restored.json()["revision"] == 3
    assert len(restored.json()["history"]) == 3
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"]), original)
    assert Workspace(root).read_dataset(metadata["id"], 2)["新增"].eq(42).all()
    assert client.post(url + "/restore", json={"revision": 99, "expected_revision": 3}).status_code == 400


def test_failed_manifest_write_keeps_old_data(setup, monkeypatch):
    _, metadata, original, root = setup
    import hscredit.web.storage as storage

    real_write = storage.write_json

    def fail_commit(path, value):
        if path.name == "metadata.json":
            raise OSError("模拟磁盘写入失败")
        real_write(path, value)

    monkeypatch.setattr(storage, "write_json", fail_commit)
    workspace = Workspace(root)
    with pytest.raises(OSError, match="模拟磁盘"):
        workspace.replace_dataset(metadata["id"], original.assign(新字段=1), expected_revision=1, processing={"description": "测试"})
    pd.testing.assert_frame_equal(workspace.read_dataset(metadata["id"]), original)
    assert workspace.get("datasets", metadata["id"])["revision"] == 1
    assert len(list(workspace.path("datasets", metadata["id"]).glob("*.parquet"))) == 1


def test_active_tasks_prevent_overwrite_and_job_keeps_revision(setup, monkeypatch):
    import sys

    client, metadata, original, root = setup
    script = root / "capturing_worker.py"
    script.write_text("import sys,time\nfrom pathlib import Path\nfrom hscredit.web.storage import Workspace,write_json\nw=Workspace(sys.argv[1]);j=w.get('jobs',sys.argv[2]);d=w.path('jobs',j['id'])\nw.read_dataset(j['dataset_id'],j['dataset_revision']).to_parquet(d/'captured.parquet')\n(d/'started').touch()\nwhile not (w.root/'release').exists():time.sleep(.05)\nwrite_json(d/'outcome.json',{'ok':True,'result':{'artifacts':[],'tables':{}}})\n", encoding="utf-8")
    monkeypatch.setattr(client.app.state.jobs, "_command", lambda job_id: [sys.executable, str(script), str(root), job_id])
    config = {"kind": "model", "dataset_id": metadata["id"], "target": "标签", "features": ["金额"]}
    first = client.post("/api/jobs", json=config).json()
    assert first["dataset_revision"] == 1
    deadline = time.monotonic() + 20
    while not (root / "jobs" / first["id"] / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (root / "jobs" / first["id"] / "started").exists()
    payload = {"expected_revision": 1, "rules": [rule("金额", literal(10), "replace")]}
    try:
        assert client.post(f"/api/datasets/{metadata['id']}/processing/apply", json=payload).status_code == 409
    finally:
        (root / "release").touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and client.get(f"/api/jobs/{first['id']}").json()["status"] != "成功":
        time.sleep(0.05)
    assert client.post(f"/api/datasets/{metadata['id']}/processing/apply", json=payload).status_code == 200
    second = client.post("/api/jobs", json=first["config"]).json()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and client.get(f"/api/jobs/{second['id']}").json()["status"] != "成功":
        time.sleep(0.05)
    pd.testing.assert_frame_equal(pd.read_parquet(root / "jobs" / second["id"] / "captured.parquet"), original)
    assert Workspace(root).read_dataset(metadata["id"])["金额"].eq(10).all()


def test_legacy_dataset_without_revision_remains_editable(setup):
    client, metadata, frame, root = setup
    path = root / "datasets" / metadata["id"] / "metadata.json"
    old = json.loads(path.read_text("utf-8"))
    for key in ("revision", "data_file", "history"):
        old.pop(key, None)
    path.write_text(json.dumps(old, ensure_ascii=False), "utf-8")
    result = client.post(f"/api/datasets/{metadata['id']}/processing/apply", json={"expected_revision": 1, "rules": [rule("新列", literal(7))]})
    assert result.status_code == 200, result.text
    assert result.json()["revision"] == 2
    pd.testing.assert_frame_equal(Workspace(root).read_dataset(metadata["id"], 1), frame)


def test_expression_complexity_is_bounded():
    node = field("x")
    for _ in range(10):
        node = function("abs", node)
    with pytest.raises(ValueError, match="嵌套"):
        compile_expression(node, {"x": "numeric"})
