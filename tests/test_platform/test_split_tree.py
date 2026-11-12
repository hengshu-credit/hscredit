"""组合划分的真实行归属、首次命中、嵌套条件、保存与失败回滚。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.platform import create_app
from hscredit.platform.split_tree import SplitTreeRequest, condition_mask, execute_tree, validate_tree
from hscredit.platform.storage import Workspace


def leaf(name):
    return {"type": "dataset", "name": name}


def condition(field, op, value=None):
    return {"type": "leaf", "field": field, "operator": op, "value": value}


def group(*children, op="AND"):
    return {"type": "group", "op": op, "children": list(children)}


def decision(rules, otherwise):
    return {"type": "condition", "rules": [{"condition": c, "action": a} for c, a in rules], "otherwise": otherwise}


def split(method, children, **kwargs):
    return {"type": "split", "config": {"method": method, "parts": [{"name": f"分支{i}", "ratio": 1 / len(children)} for i in range(len(children))], **kwargs}, "children": children}


def scenario():
    random = split("random", [leaf("训练集"), leaf("测试集")], stratify="标签", random_state=42)
    time = split("time", [random, leaf("时间外集")], time_col="日期", time_mode="cutoff", cutoffs=["2025-03-01"])
    return decision([(group(condition("GROUP", "eq", "留出")), leaf("指定分组留出"))], time)


@pytest.fixture
def setup(tmp_path):
    frame = pd.DataFrame({"行号": range(200), "GROUP": ["留出"] * 40 + ["建模"] * 160, "客户": np.repeat(range(50), 4), "标签": [0, 1] * 100, "日期": pd.date_range("2025-01-01", periods=200), "金额": np.arange(200, dtype=float)})
    with TestClient(create_app(tmp_path)) as client:
        workspace = client.app.state.workspace
        metadata = workspace.add_dataset(frame, "源数据", defaults={"default_features": ["金额"], "default_target": "标签"})
        yield client, workspace, metadata, frame


def post(setup, tree, action="preview", revision=1):
    return setup[0].post(f"/api/datasets/{setup[2]['id']}/split-tree/{action}", json={"expected_revision": revision, "tree": tree})


def test_group_holdout_time_oot_random_train_test_end_to_end(setup, monkeypatch):
    import hscredit.platform.splitting as splitting

    calls = []
    for engine in ("time_train_test_split", "train_test_split"):
        original = getattr(splitting, engine)

        def audited(*args, _fn=original, _engine=engine, **kwargs):
            calls.append((_engine, len(args[0])))
            return _fn(*args, **kwargs)

        monkeypatch.setattr(splitting, engine, audited)
    tree = scenario()
    tree["otherwise"]["label"] = "常规渠道"
    before = post(setup, tree)
    assert before.status_code == 200, before.text
    assert len(setup[1].list("datasets")) == 1
    assert [p["rows"] for p in before.json()["parts"]] == [40, 10, 9, 141]
    saved = post(setup, tree, "apply")
    assert saved.status_code == 200, saved.text
    result = saved.json()
    assert before.json()["parts"] == result["parts"]
    assert calls == [("time_train_test_split", 160), ("train_test_split", 19)] * 2
    parts = {item["name"]: setup[1].read_dataset(item["id"]) for item in result["datasets"]}
    assert set(parts["指定分组留出"]["行号"]) == set(range(40))
    assert set(parts["时间外集"]["行号"]) == set(range(59, 200))
    assert set(pd.concat([parts["训练集"], parts["测试集"]])["行号"]) == set(range(40, 59))
    merged = pd.concat(parts.values()).sort_values("行号").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, setup[3])
    assert merged["行号"].is_unique
    assert result["excluded_rows"] == 0
    assert not result["empty_datasets"]
    for metadata in result["datasets"]:
        assert metadata["splitting"]["config"]["tree"] == tree
        assert metadata["splitting"]["source_revision"] == 1
        assert metadata["splitting"]["path"].startswith("全部样本")
        assert metadata["default_features"] == ["金额"]
        assert setup[0].get(f"/api/datasets/{metadata['id']}/preview").json()["total"] == metadata["rows"]
    pd.testing.assert_frame_equal(setup[1].read_dataset(setup[2]["id"]), setup[3])


def test_branch_statistics_analysis_and_repartition_use_bound_rows(setup):
    from hscredit.platform.workflows import bound_frames

    client, ws, source, frame = setup
    frame = frame.copy()
    frame.loc[44, "金额"] = np.nan
    ws.replace_dataset(source["id"], frame, expected_revision=1, processing={"description": "测试缺失值"})
    random = split("random", [leaf("训练"), leaf("测试")], random_state=17)
    branch = decision([(group(condition("行号", "lt", 50)), {"type": "exclude", "reason": "排除"})], random)
    tree = decision([(group(condition("行号", "lt", 40)), leaf("外部渠道"))], branch)
    result = post(setup, tree, "apply", revision=2).json()
    sibling = result["datasets"][0]
    groups = {tuple(d["group_path"]): d for d in client.get("/api/datasets").json() if "group_path" in d}
    parent, inner = groups[("otherwise",)], groups[("otherwise", "otherwise")]
    assert parent["rows"] == 160 and inner["rows"] == 150 and result["excluded_rows"] == 10
    assert client.post(f"/api/datasets/{parent['id']}/copy", json={"expected_revision": 1}).status_code == 400
    assert parent["missing_rate"] == pytest.approx(1 / (160 * len(frame.columns)))
    assert client.get(f"/api/datasets/{parent['id']}/column-profile", params={"column": "金额"}).json()["missing"] == 1
    previous = frame.iloc[50:].reset_index(drop=True)
    historical = {"dataset_id": inner["id"], "dataset_revision": 1, "evaluation_datasets": [{"id": sibling["id"], "revision": 1}]}
    pd.testing.assert_frame_equal(bound_frames(ws, historical)[f"训练集 · {inner['name']}"], previous)

    ws.replace_dataset(source["id"], frame.iloc[::-1].assign(金额=0), expected_revision=2, processing={"description": "改变当前源版本"})
    updated = deepcopy(branch)
    updated["rules"][0]["condition"]["children"][0]["value"] = 60
    req = {"expected_revision": 1, "tree": updated}
    params = {key: parent["splitting"][key] for key in ("plan_id", "plan_revision")}
    url = f"/api/datasets/{parent['id']}/split-tree"
    preview = client.post(url + "/preview", params=params, json=req).json()
    assert preview["source_rows"] == 200 and preview["excluded_rows"] == 20
    assert ws.read_dataset(inner["id"]).equals(previous)
    response = client.post(url + "/apply", params=params, json=req)
    assert response.status_code == 200, response.text
    saved = response.json()["datasets"]
    assert saved[0]["id"] == sibling["id"] and saved[0]["splitting"]["source_revision"] == 2
    assert saved[0]["splitting"]["config"]["tree"] == {**tree, "otherwise": updated}
    assert sum(d["rows"] for d in saved) == 180
    pd.testing.assert_frame_equal(ws.read_dataset(sibling["id"]), frame.iloc[:40])
    pd.testing.assert_frame_equal(ws.read_dataset(inner["id"]), frame.iloc[60:].reset_index(drop=True))
    pd.testing.assert_frame_equal(bound_frames(Workspace(ws.root), historical)[f"训练集 · {inner['name']}"], previous)
    assert client.post(url + "/apply", params=params, json=req).status_code == 409


def test_legacy_plan_materializes_groups_once_without_modifying_leaves(setup):
    client, ws, source, frame = setup
    request = SplitTreeRequest(expected_revision=1, tree=scenario())
    parts, preview = execute_tree(frame, request, source["columns"])
    for i, (name, output) in enumerate(parts):
        info = {"source_dataset_id": source["id"], "source_revision": 1, "config": request.dict(), "at": source["created_at"], "part_index": i, "path": preview["parts"][i]["path"]}
        ws.add_dataset(output, name, splitting=info)
    original_ids = {d["id"] for d in ws.list("datasets")}
    assert client.post(f"/api/datasets/{next(iter(original_ids - {source['id']}))}/copy", json={"expected_revision": 1}).status_code == 400
    ws.replace_dataset(source["id"], frame.iloc[::-1], expected_revision=1, processing={"description": "改变源版本"})
    all_rows = client.get("/api/datasets").json()
    added = [d for d in all_rows if d["id"] not in original_ids]
    assert sorted(d["rows"] for d in added) == [19, 160]
    for d in added:
        start, end = (40, 59) if d["rows"] == 19 else (40, 200)
        pd.testing.assert_frame_equal(ws.read_dataset(d["id"]), frame.iloc[start:end].reset_index(drop=True))
    assert {d["id"] for d in client.get("/api/datasets").json()} == {d["id"] for d in all_rows}
    assert len(list(ws.root.rglob("*.npy"))) == 2
    assert all(d.get("revision", 1) == 1 for d in all_rows if d["id"] != source["id"])


def test_first_match_and_nested_and_or_with_explicit_exclusion(setup):
    conditions = group(condition("行号", "lt", 10), group(condition("标签", "eq", 1), condition("行号", "eq", 2), op="OR"))
    tree = decision([(conditions, leaf("优先命中")), (group(condition("行号", "lt", 20)), leaf("后续命中"))], {"type": "exclude", "reason": "不在本次建模范围"})
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    result = response.json()
    frames = [setup[1].read_dataset(d["id"]) for d in result["datasets"]]
    assert frames[0]["行号"].tolist() == [1, 2, 3, 5, 7, 9]
    assert set(frames[1]["行号"]) == set(range(20)) - {1, 2, 3, 5, 7, 9}
    assert result["excluded_rows"] == 180
    assert result["trace"][-1]["reason"] == "不在本次建模范围"
    swapped = deepcopy(tree)
    swapped["rules"].reverse()
    request = SplitTreeRequest(expected_revision=1, tree=swapped)
    _, preview = execute_tree(setup[3], request, setup[2]["columns"])
    assert preview["empty_datasets"] == ["优先命中"]
    assert preview["parts"][0]["rows"] == 20


@pytest.mark.parametrize("op,value,expected", [("eq", 2, [2]), ("ne", 2, [0, 1, 3]), ("gt", 1, [2, 3]), ("ge", 2, [2, 3]), ("lt", 2, [0, 1]), ("le", 2, [0, 1, 2]), ("in", [1, 3], [1, 3]), ("not_in", [1, 3], [0, 2]), ("between", [1, 2], [1, 2]), ("is_null", None, [4]), ("not_null", None, [0, 1, 2, 3])])
def test_numeric_missing_values_have_explicit_routing(op, value, expected):
    frame = pd.DataFrame({"数值": [0, 1, 2, 3, np.nan]})
    mask = condition_mask(frame, condition("数值", op, value), {})
    assert frame.index[mask].tolist() == expected


def test_typed_dates_booleans_large_ids_text_and_literals():
    frame = pd.DataFrame({"日期": ["2025-01-01", "2025-02-01", None], "布尔": pd.Series([True, False, None], dtype="boolean"), "客户": [9007199254740992, 9007199254740993, 9007199254740994], "文本": ["a.b", "axb", "__import__('os')"]})
    assert condition_mask(frame, condition("日期", "ge", "2025-02-01"), {"日期": "date"}).tolist() == [False, True, False]
    assert condition_mask(frame, condition("布尔", "eq", "true"), {}).tolist() == [True, False, False]
    assert condition_mask(frame, condition("客户", "eq", "9007199254740993"), {}).tolist() == [False, True, False]
    assert condition_mask(frame, condition("客户", "eq", 9007199254740993), {}).tolist() == [False, True, False]
    assert condition_mask(frame, condition("文本", "contains", "."), {}).tolist() == [True, False, False]
    assert condition_mask(frame, condition("文本", "eq", "__import__('os')"), {}).tolist() == [False, False, True]


def test_group_random_and_oot_can_be_nested_and_gap_is_accounted(setup):
    grouped = split("group", [leaf("客户训练"), leaf("客户测试")], group_col="客户", random_state=17)
    tree = split("oot", [grouped, leaf("OOT")], time_col="日期", oot_start="2025-04-01", oot_end="2025-05-01", gap=2)
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    result = response.json()
    parts = [setup[1].read_dataset(d["id"]) for d in result["datasets"]]
    assert set(parts[0]["客户"]).isdisjoint(parts[1]["客户"])
    assert sum(len(p) for p in parts) + result["excluded_rows"] == 200
    assert result["excluded_rows"] == 81
    assert any(row.get("reason") == "时间间隔或时间外结束边界" for row in result["trace"])


def test_empty_branches_are_traced_but_not_saved(setup):
    tree = decision([(group(condition("行号", "gt", 1000)), split("random", [leaf("空训练"), leaf("空测试")]))], leaf("保留"))
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    assert response.json()["empty_datasets"] == ["空训练", "空测试"]
    assert len(response.json()["datasets"]) == 1
    assert len(setup[1].list("datasets")) == 2


@pytest.mark.parametrize("method,config", [("random", {"random_state": 17}), ("time", {"time_col": "日期", "time_mode": "cutoff", "cutoffs": ["2025-04-01"]}), ("group", {"group_col": "客户", "random_state": 17}), ("oot", {"time_col": "日期", "oot_start": "2025-04-01"})])
def test_removing_parent_output_keeps_sibling_membership(setup, method, config):
    nested = split("random", [leaf("子结果一"), leaf("子结果二")], random_state=19)
    tree = split(method, [nested, leaf("保留结果")], **config)
    original, _ = execute_tree(setup[3], SplitTreeRequest(expected_revision=1, tree=tree), setup[2]["columns"])
    removed_count = sum(len(frame) for name, frame in original if name != "保留结果")
    expected = dict(original)["保留结果"].reset_index(drop=True)
    tree["children"][0] = {"type": "exclude", "reason": "已移除结果：父级样本"}
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["datasets"]) == 1
    assert result["excluded_rows"] == removed_count
    assert result["datasets"][0]["name"] == "保留结果"
    pd.testing.assert_frame_equal(setup[1].read_dataset(result["datasets"][0]["id"]), expected)


def test_removed_condition_does_not_fall_through_to_other_results(setup):
    tree = decision([(group(condition("行号", "lt", 50)), {"type": "exclude", "reason": "已移除条件结果"})], leaf("其余"))
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["excluded_rows"] == 50
    actual = setup[1].read_dataset(result["datasets"][0]["id"])
    assert actual["行号"].tolist() == list(range(50, 200))


@pytest.mark.parametrize("method,config", [("random", {"stratify": "标签"}), ("group", {"group_col": "客户"}), ("time", {"time_col": "日期"})])
def test_tree_preview_preserves_single_split_details(setup, method, config):
    tree = split(method, [leaf("第一份"), leaf("第二份")], **config)
    response = post(setup, tree)
    assert response.status_code == 200, response.text
    for item in response.json()["parts"]:
        assert item["rows"] == 100
        assert item["fraction"] == 0.5
        if method == "random":
            assert item["strata"] == [{"类别": 0, "样本数": 50}, {"类别": 1, "样本数": 50}] or item["strata"] == [{"类别": 1, "样本数": 50}, {"类别": 0, "样本数": 50}]
        elif method == "group":
            assert item["groups"] == 25
        else:
            assert pd.Timestamp(item["time_max"]) - pd.Timestamp(item["time_min"]) == pd.Timedelta(days=99)


@pytest.mark.parametrize("first_method", ["random", "group", "time"])
def test_any_starting_method_with_independent_downstream_methods(setup, first_method):
    """不同起点与同级不同方法组合，预览和保存仍保持完整且唯一的真实行归属。"""
    early_late = split("time", [leaf("早期"), leaf("近期")], time_col="日期", time_mode="cutoff", cutoffs=["2025-02-20"])
    conditional = decision([(group(condition("标签", "eq", 1)), leaf("坏样本"))], split("random", [leaf("其余一"), leaf("其余二")], random_state=19))
    config = {"random": {"random_state": 17}, "group": {"group_col": "客户", "random_state": 17}, "time": {"time_col": "日期"}}[first_method]
    tree = split(first_method, [early_late, conditional], **config)
    before = post(setup, tree)
    assert before.status_code == 200, before.text
    response = post(setup, tree, "apply")
    assert response.status_code == 200, response.text
    result = response.json()
    assert [(p["name"], p["rows"]) for p in result["parts"]] == [(p["name"], p["rows"]) for p in before.json()["parts"]]
    frames = {d["name"]: setup[1].read_dataset(d["id"]) for d in result["datasets"]}
    combined = pd.concat(frames.values()).sort_values("行号").reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, setup[3])
    assert (frames["早期"]["日期"] < pd.Timestamp("2025-02-20")).all()
    assert (frames["近期"]["日期"] >= pd.Timestamp("2025-02-20")).all()
    assert frames["坏样本"]["标签"].eq(1).all()
    assert frames["其余一"]["标签"].eq(0).all()
    assert frames["其余二"]["标签"].eq(0).all()
    if first_method == "group":
        assert set(pd.concat([frames["早期"], frames["近期"]])["客户"]).isdisjoint(pd.concat([frames["坏样本"], frames["其余一"], frames["其余二"]])["客户"])


@pytest.mark.parametrize(
    "tree",
    [
        {"type": "execute", "code": "print(1)"},
        decision([(group(), leaf("a"))], leaf("b")),
        decision([(group(condition("行号", "eq", {"code": 1})), leaf("a"))], leaf("b")),
        decision([(group(condition("不存在", "eq", 1)), leaf("a"))], leaf("b")),
        decision([(group(condition("行号", "contains", "1")), leaf("a"))], leaf("b")),
        decision([(group(condition("行号", "between", [4, 1])), leaf("a"))], leaf("b")),
        decision([(group(condition("日期", "ge", "不是日期")), leaf("a"))], leaf("b")),
        decision([(group(condition("行号", "eq", 1)), leaf("a"))], leaf(" a ")),
        decision([(group(condition("行号", "eq", 1)), leaf("源数据"))], leaf("b")),
        decision([(group(condition("行号", "eq", 1)), {"type": "exclude", "reason": ""})], leaf("b")),
        decision([(group(condition("行号", "gt", 1000)), split("time", [leaf("a"), leaf("b")], time_col="不存在"))], leaf("c")),
        decision([(group(condition("行号", "gt", 1000)), split("random", [leaf("a"), leaf("b")], parts=[{"name": "1", "ratio": 0.2}, {"name": "2", "ratio": 0.2}]))], leaf("c")),
        decision([(group(condition("行号", "lt", 3)), split("random", [leaf("a"), leaf("b")], stratify="行号"))], leaf("c")),
        decision([(group(condition("行号", "not_null")), {"type": "exclude", "reason": "全部排除"})], leaf("b")),
    ],
)
def test_invalid_tree_never_publishes_partial_results(setup, tree):
    response = post(setup, tree, "apply")
    assert response.status_code in {400, 422}, response.text
    assert len(setup[1].list("datasets")) == 1


def test_depth_and_node_limits():
    tree = leaf("结尾")
    for i in range(9):
        tree = decision([(group(condition("行号", "eq", i)), leaf(str(i)))], tree)
    with pytest.raises(ValueError, match="8 层"):
        validate_tree(tree, 1)
    tree = decision([(group(*[condition("行号", "eq", i) for i in range(30)]), leaf(str(n))) for n in range(4)], leaf("其他"))
    with pytest.raises(ValueError, match="100 个"):
        validate_tree(tree, 1)


def test_stale_conflicting_and_failed_save_leave_source_intact(setup, monkeypatch):
    assert post(setup, scenario(), "apply", revision=2).status_code == 409
    original = np.save
    writes = 0

    def failed(frame, *args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("模拟组合划分写入失败")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(np, "save", failed)
    with pytest.raises(OSError, match="模拟组合划分"):
        post(setup, scenario(), "apply")
    assert len(setup[1].list("datasets")) == 1
    assert not list(setup[1].root.glob("split-*"))
    pd.testing.assert_frame_equal(setup[1].read_dataset(setup[2]["id"]), setup[3])


def test_revision_changes_during_tree_execution(setup, monkeypatch):
    import hscredit.platform.server as server

    original = server.execute_tree

    def changed(*args):
        result = original(*args)
        setup[1].update("datasets", setup[2]["id"], revision=2)
        return result

    monkeypatch.setattr(server, "execute_tree", changed)
    assert post(setup, scenario(), "apply").status_code == 409
    assert len(setup[1].list("datasets")) == 1


def test_real_loan_data_composition_keeps_every_original_value(tmp_path):
    frame = pd.read_excel(Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx")
    workspace = Workspace(tmp_path)
    metadata = workspace.add_dataset(frame, "真实贷款样例")
    frame = workspace.read_dataset(metadata["id"])
    amount = next(c for c in frame.select_dtypes(include="number") if frame[c].nunique() > 10)
    date = next(c["name"] for c in metadata["columns"] if c["kind"] == "date" and not frame[c["name"]].isna().any())
    random = split("random", [leaf("训练"), leaf("测试")], random_state=42)
    time = split("time", [random, leaf("OOT")], time_col=date)
    tree = decision([(group(condition(amount, "le", float(frame[amount].quantile(0.2)))), leaf("条件留出"))], time)
    request = SplitTreeRequest(expected_revision=1, tree=tree)
    parts, preview = execute_tree(frame, request, metadata["columns"])
    assert len(parts) == 4
    assert preview["excluded_rows"] == 0
    pd.testing.assert_frame_equal(pd.concat([part for _, part in parts]).sort_index(), frame)
    saved = workspace.add_split_datasets(parts, source=metadata, config=request.dict(), excluded_rows=0)
    for item, (_, expected) in zip(saved, parts):
        pd.testing.assert_frame_equal(workspace.read_dataset(item["id"]), expected.reset_index(drop=True))
