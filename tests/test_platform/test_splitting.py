"""数据集划分的行归属、分组隔离、时间边界及批量保存验证。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app
from hscredit.web.splitting import SplitRequest, split_frame
from hscredit.web.storage import Workspace
from hscredit.utils.datasets import hscredit_dataset


@pytest.fixture
def setup(tmp_path):
    frame = pd.DataFrame({"编号": range(120), "标签": [0, 1] * 60, "客户": np.repeat(np.arange(30), 4), "日期": pd.date_range("2025-01-01", periods=120), "金额": np.arange(120) * 1.5, "空值": np.nan})
    with TestClient(create_app(tmp_path)) as client:
        workspace = client.app.state.workspace
        source = workspace.add_dataset(frame, "原始数据", defaults={"default_target": "标签", "default_date": "日期", "default_features": ["金额"]})
        yield client, workspace, source, frame


def config(**kwargs):
    return {"expected_revision": 1, "method": "random", "parts": [{"name": name, "ratio": ratio} for name, ratio in zip(["训练集", "验证集", "测试集"], [0.6, 0.2, 0.2])], **kwargs}


def post(setup, action, payload):
    client, _, source, _ = setup
    return client.post(f"/api/datasets/{source['id']}/split/{action}", json=payload)


@pytest.mark.parametrize("method,options,engine", [("random", {"stratify": "标签"}, "train_test_split"), ("time", {"time_col": "日期"}, "time_train_test_split"), ("group", {"group_col": "客户"}, "group_train_test_split")])
def test_multi_split_preserves_rows_and_uses_existing_engines(setup, monkeypatch, method, options, engine):
    import hscredit.web.splitting as splitting

    client, workspace, source, frame = setup
    calls = []
    original = getattr(splitting, engine)

    def audited(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(splitting, engine, audited)
    payload = config(method=method, **options)
    preview = post(setup, "preview", payload)
    assert preview.status_code == 200, preview.text
    assert len(workspace.list("datasets")) == 1
    saved = post(setup, "apply", payload)
    assert saved.status_code == 200, saved.text
    result = saved.json()
    assert calls and len(calls) == 4
    assert result["parts"] == preview.json()["parts"]
    assert result["excluded_rows"] == 0
    datasets = result["datasets"]
    assert [d["name"] for d in datasets] == ["训练集", "验证集", "测试集"]
    assert [d["rows"] for d in datasets] == [72, 24, 24]
    reloaded = Workspace(workspace.root)
    parts = [reloaded.read_dataset(d["id"]) for d in datasets]
    merged = pd.concat(parts).sort_values("编号").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, frame)
    assert merged["编号"].is_unique
    pd.testing.assert_frame_equal(workspace.read_dataset(source["id"]), frame)
    for metadata, part, summary in zip(datasets, parts, result["parts"]):
        assert metadata["splitting"]["source_dataset_id"] == source["id"]
        assert metadata["splitting"]["source_revision"] == 1
        assert metadata["splitting"]["config"] == SplitRequest(**payload).dict()
        assert metadata["default_features"] == ["金额"]
        assert metadata["default_date"] == "日期"
        assert metadata["revision"] == 1
        assert [row["编号"] for row in summary["preview"]] == part["编号"].head(5).tolist()
        url = f"/api/datasets/{metadata['id']}"
        assert client.get(url + "/preview").json()["total"] == len(part)
        assert client.patch(url, json={"name": metadata["name"] + "独立重命名"}).status_code == 200
    if method == "random":
        assert all(part["标签"].mean() == 0.5 for part in parts)
    if method == "group":
        assert [len(part["客户"].unique()) for part in parts] == [18, 6, 6]
        assert all(set(a["客户"]).isdisjoint(b["客户"]) for i, a in enumerate(parts) for b in parts[i + 1 :])
    if method == "time":
        assert all(part["日期"].is_monotonic_increasing for part in parts)
        assert all(a["日期"].max() <= b["日期"].min() for a, b in zip(parts, parts[1:]))


def test_seed_reproducibility_and_many_parts(setup):
    frame = setup[3]
    payload = config(parts=[{"name": str(i), "ratio": 0.1} for i in range(10)])
    first = split_frame(frame, SplitRequest(**payload))
    second = split_frame(frame, SplitRequest(**payload))
    different = split_frame(frame, SplitRequest(**{**payload, "random_state": 7}))
    assert len(first) == 10
    assert all(a[1].equals(b[1]) for a, b in zip(first, second))
    assert not first[0][1].equals(different[0][1])
    assert sorted(pd.concat([part for _, part in first])["编号"]) == list(range(120))


def test_time_boundaries_and_gap_are_explicit(setup):
    payload = config(method="time", time_col="日期", time_mode="cutoff", cutoffs=["2025-02-01", "2025-03-01"], gap=2)
    response = post(setup, "apply", payload)
    assert response.status_code == 200, response.text
    result = response.json()
    parts = [setup[1].read_dataset(item["id"]) for item in result["datasets"]]
    assert result["excluded_rows"] == 4
    assert parts[1]["日期"].min() == pd.Timestamp("2025-02-01")
    assert parts[2]["日期"].min() == pd.Timestamp("2025-03-01")
    assert set(range(120)) - set(pd.concat(parts)["编号"]) == {29, 30, 57, 58}


def test_time_ratios_gap_and_unsorted_duplicate_times(setup):
    frame = setup[3].sample(frac=1, random_state=11)
    frame["日期"] = frame["日期"].dt.to_period("W").dt.start_time
    parts = split_frame(frame, SplitRequest(**config(method="time", time_col="日期", gap=3)))
    assert [len(p) for _, p in parts] == [68, 23, 23]
    assert all(a[1]["日期"].max() <= b[1]["日期"].min() for a, b in zip(parts, parts[1:]))
    assert pd.concat([p for _, p in parts])["编号"].is_unique


def test_oot_calls_core_and_reports_exclusions(setup, monkeypatch):
    import hscredit.web.splitting as splitting

    original = splitting.oot_split
    calls = []

    def audited(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(splitting, "oot_split", audited)
    payload = config(method="oot", time_col="日期", oot_start="2025-03-01", oot_end="2025-03-31", gap=2, parts=[{"name": "开发集"}, {"name": "时间外集"}])
    response = post(setup, "apply", payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert calls == [{"oot_end": "2025-03-31", "gap": 2}]
    assert [p["rows"] for p in result["parts"]] == [57, 31]
    assert result["excluded_rows"] == 32
    assert result["parts"][1]["time_min"].startswith("2025-03-01")
    assert result["parts"][1]["time_max"].startswith("2025-03-31")


@pytest.mark.parametrize(
    "changes",
    [
        {"parts": [{"name": "a", "ratio": 0.2}, {"name": "b", "ratio": 0.3}]},
        {"parts": [{"name": "a", "ratio": 0.5}, {"name": " a ", "ratio": 0.5}]},
        {"parts": [{"name": " ", "ratio": 0.5}, {"name": "b", "ratio": 0.5}]},
        {"parts": [{"name": "原始数据", "ratio": 0.5}, {"name": "b", "ratio": 0.5}]},
        {"parts": [{"name": "a", "ratio": 0.001}, {"name": "b", "ratio": 0.999}]},
        {"stratify": "不存在"},
        {"stratify": "编号"},
        {"stratify": "空值"},
        {"method": "group"},
        {"method": "group", "group_col": "空值"},
        {"method": "group", "group_col": "标签"},
        {"method": "time", "time_col": "日期", "gap": 100},
        {"method": "time", "time_col": "编号"},
        {"method": "time", "time_col": "日期", "time_mode": "cutoff", "cutoffs": ["2025-03-01", "2025-02-01"]},
        {"method": "time", "time_col": "日期", "time_mode": "cutoff", "cutoffs": ["NaT", "2025-02-01"]},
        {"method": "time", "time_col": "日期", "time_mode": "cutoff", "cutoffs": []},
        {"method": "oot", "time_col": "日期", "oot_start": "2025-02-01"},
    ],
)
def test_invalid_configs_do_not_write_any_results(setup, changes):
    response = post(setup, "apply", config(**changes))
    assert response.status_code in {400, 422}, response.text
    assert len(setup[1].list("datasets")) == 1


def test_stale_and_repeated_requests(setup):
    assert post(setup, "preview", config(expected_revision=2)).status_code == 409
    assert post(setup, "apply", config()).status_code == 200
    assert post(setup, "apply", config()).status_code == 400
    assert len(setup[1].list("datasets")) == 4


def test_revision_changed_during_calculation_is_rejected(setup, monkeypatch):
    import hscredit.web.server as server

    original = server.split_frame

    def changed(*args):
        result = original(*args)
        setup[1].update("datasets", setup[2]["id"], revision=2)
        return result

    monkeypatch.setattr(server, "split_frame", changed)
    assert post(setup, "apply", config()).status_code == 409
    assert len(setup[1].list("datasets")) == 1
    assert not list(setup[1].root.glob("split-*"))


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_batch_save_rolls_back_all_children(setup, monkeypatch, failure):
    workspace, source, frame = setup[1:]
    original = np.save if failure == "write" else Path.replace
    count = 0

    def broken(*args, **kwargs):
        nonlocal count
        count += 1
        if (failure == "write" and count == 2) or (failure == "publish" and args[1] == workspace.subsets.catalog):
            raise OSError("模拟磁盘失败")
        return original(*args, **kwargs)

    monkeypatch.setattr(np if failure == "write" else Path, "save" if failure == "write" else "replace", broken)
    request = SplitRequest(**config())
    with pytest.raises(OSError, match="模拟磁盘失败"):
        workspace.add_split_datasets(split_frame(frame, request), source=source, config=request.dict(), excluded_rows=0)
    assert len(workspace.list("datasets")) == 1
    assert len(list((workspace.root / "datasets").iterdir())) == 1
    assert not list(workspace.root.glob("split-*"))
    pd.testing.assert_frame_equal(workspace.read_dataset(source["id"]), frame)


def test_real_loan_data_round_trip(tmp_path):
    frame = hscredit_dataset(columns=["放款编码", "loan_amt", "授信时间", "流量渠道"]).sample(2000, random_state=42)
    workspace = Workspace(tmp_path)
    metadata = workspace.add_dataset(frame, "真实贷款数据")
    frame = workspace.read_dataset(metadata["id"])
    request = SplitRequest(**config())
    parts = split_frame(frame, request)
    result = workspace.add_split_datasets(parts, source=metadata, config=request.dict(), excluded_rows=0)
    assert sum(item["rows"] for item in result) == len(frame)
    for item, (_, expected) in zip(result, parts):
        pd.testing.assert_frame_equal(workspace.read_dataset(item["id"]), expected.reset_index(drop=True))
