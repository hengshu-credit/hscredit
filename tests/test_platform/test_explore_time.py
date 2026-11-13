"""时间探索的频率、全量计数、无效日期及接口和任务一致性。"""

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app
from hscredit.web.analysis import explore_charts
from hscredit.web.explore_time import exploration_offset, time_distribution
from hscredit.web.exploration import explore_dataset
from hscredit.web.storage import Workspace


def test_default_monthly_counts_keep_gaps_and_exclude_invalid_dates():
    series = pd.Series(["2025-01-01", "2025-01-31 23:59:59", "2025-03-01", "2025-03-31", None, "无效日期"])
    result = time_distribution(series)
    assert result["freq"] == "M"
    assert result["distribution"] == [{"区间": "2025-01", "样本数": 2}, {"区间": "2025-02", "样本数": 0}, {"区间": "2025-03", "样本数": 2}]
    assert (result["valid"], result["missing"], result["invalid"]) == (4, 1, 1)
    assert sum(p["样本数"] for p in result["distribution"]) + result["missing"] + result["invalid"] == len(series)


@pytest.mark.parametrize("freq", ["D", "W", "M", "ME", "MS", "Q", "QE", "Y", "YE", "h", "2W", "15D", "2MS", "B"])
def test_counts_match_pandas_resample_for_configured_frequencies(freq):
    dates = pd.Series(pd.date_range("2024-12-29 03:00", periods=150, freq="9h"))
    result = time_distribution(dates, freq)
    expected = pd.Series(1, index=pd.DatetimeIndex(dates)).resample(exploration_offset(freq)).size()
    assert [p["样本数"] for p in result["distribution"]] == expected.tolist()
    assert sum(p["样本数"] for p in result["distribution"]) == 150
    assert result["freq"] == freq


def test_timezone_and_all_invalid_dates():
    result = time_distribution(pd.Series(["2025-02-01T00:30:00+08:00", "2025-02-01T01:30:00+08:00"]))
    assert result["distribution"] == [{"区间": "2025-01", "样本数": 2}]
    assert "UTC" in result["distribution_note"]
    result = time_distribution(pd.Series([None, "日期未知"]))
    assert result["distribution"] == []
    assert (result["missing"], result["invalid"]) == (1, 1)


@pytest.mark.parametrize("freq", ["", "0D", "-2M", "不合法", "1D;print(1)", "D" * 33])
def test_invalid_frequencies_are_explicit(freq):
    with pytest.raises(ValueError, match="频率"):
        exploration_offset(freq)


def test_overly_fine_frequency_rejected_before_allocating_bins():
    with pytest.raises(ValueError, match="10000"):
        time_distribution(pd.Series(pd.to_datetime(["2020-01-01", "2025-01-01"])), "s")
    with pytest.raises(ValueError, match="10000"):
        time_distribution(pd.Series(pd.to_datetime(["1900-01-01", "2025-01-01"])), "B")


def test_charts_do_not_compute_a_second_statistics_table(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("字段统计应读取已有 summary，不能额外调用 describe")

    monkeypatch.setattr(pd.Series, "describe", unexpected)
    frame = pd.DataFrame({"数值": [1, 2, 3, np.nan, np.inf], "渠道": ["A", "A", "B", None, "B"]})
    result = explore_charts(frame, "数值", bins=3)
    assert "statistics" not in result
    assert sum(p["样本数"] for p in result["distribution"]) == 3
    assert (result["missing"], result["invalid"]) == (1, 1)
    category = explore_charts(frame, "渠道")
    assert sum(p["样本数"] for p in category["distribution"]) == 5
    assert "statistics" not in category
    assert "statistics" not in time_distribution(pd.Series(pd.date_range("2025-01-01", periods=3)))


def test_frequency_updates_chart_without_recomputing_summary(tmp_path, monkeypatch):
    import hscredit.web.exploration as exploration

    workspace = Workspace(tmp_path)
    dataset = workspace.add_dataset(pd.DataFrame({"时间": pd.date_range("2025-01-01", periods=80)}), "日期数据")
    calls = []

    def summary(data, *args):
        calls.append(len(data))
        return {"summary": [{"样本数": len(data)}]}

    monkeypatch.setattr(exploration, "explore_summary", summary)
    month = explore_dataset(workspace, dataset["id"], feature="时间")
    day = explore_dataset(workspace, dataset["id"], feature="时间", freq="D")
    assert calls == [80]
    assert len(month["distribution"]) == 3
    assert len(day["distribution"]) == 80
    assert month["summary"] == day["summary"]


def test_date_string_api_and_background_job_use_same_freq(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = pd.DataFrame({"授信时间": pd.date_range("2025-01-01", periods=80).astype(str), "数值": range(80)})
        dataset = client.app.state.workspace.add_dataset(frame, "日期字符串")
        url = f"/api/datasets/{dataset['id']}/explore"
        default = client.get(url, params={"feature": "授信时间"})
        assert default.status_code == 200, default.text
        assert default.json()["kind"] == "date"
        assert [p["样本数"] for p in default.json()["distribution"]] == [31, 28, 21]
        config = {"feature": "授信时间", "freq": "2W"}
        expected = client.post(url, json=config)
        assert expected.status_code == 200, expected.text
        assert client.get(url, params=config).json() == expected.json()
        for bad in ["0D", "-1W", "bad"]:
            assert client.post(url, json={**config, "freq": bad}).status_code == 400
            assert client.post("/api/jobs", json={"kind": "explore", "dataset_id": dataset["id"], **config, "freq": bad}).status_code == 400
        job = client.post("/api/jobs", json={"kind": "explore", "dataset_id": dataset["id"], **config}).json()
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            actual = client.get(f"/api/jobs/{job['id']}").json()
            if actual["status"] in {"成功", "失败"}:
                break
            time.sleep(0.05)
        assert actual["status"] == "成功", actual
        assert actual["result"]["exploration"] == expected.json()
