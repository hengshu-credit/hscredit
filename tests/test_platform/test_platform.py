"""验证平台 API、实际分析语义、持久化以及可下载模型的预测一致性。"""

import io
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from hscredit.platform import create_app, launch
from hscredit.platform.storage import Workspace


@pytest.fixture
def frame():
    rng = np.random.default_rng(52)
    x = rng.normal(size=240)
    y = (x + rng.normal(scale=0.4, size=240) > 0.25).astype(int)
    return pd.DataFrame({"评分": x, "渠道": np.where(x > 0, "线上", "线下"), "空字段": np.nan, "日期": pd.date_range("2025-01-01", periods=240), "金额": rng.integers(100, 10000, 240), "逾期天数": np.where(y, 30, 0), "标签": y})


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as value:
        yield value


def upload(client, frame, name="测试数据"):
    response = client.post("/api/datasets", files={"file": ("data.csv", frame.to_csv(index=False).encode("utf-8-sig"), "text/csv")}, data={"name": name})
    assert response.status_code == 200, response.text
    return response.json()


def run_job(client, identifier, kind, **kwargs):
    response = client.post("/api/jobs", json={"dataset_id": identifier, "kind": kind, "target": "标签", "positive_label": 1, "features": ["评分", "渠道"], **kwargs})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"成功", "失败"}:
            assert job["status"] == "成功", job.get("message", job)
            return job
        time.sleep(0.05)
    pytest.fail("后台任务超时")


def test_dataset_round_trip_preview_schema_and_persistence(client, frame, tmp_path):
    dataset = upload(client, frame, "<数据集>")
    assert dataset["rows"] == len(frame)
    columns = {c["name"]: c for c in dataset["columns"]}
    assert columns["日期"]["kind"] == "date"
    assert columns["标签"]["unique"] == 2
    assert columns["空字段"]["missing"] == len(frame)
    result = client.get(f"/api/datasets/{dataset['id']}/preview", params={"query": "线上", "page_size": 7}).json()
    assert len(result["rows"]) == 7
    assert result["total"] == int((frame["渠道"] == "线上").sum())
    assert all(row["渠道"] == "线上" for row in result["rows"])
    assert client.patch(f"/api/datasets/{dataset['id']}", json={"name": "新名称"}).status_code == 200
    assert Workspace(tmp_path).list("datasets")[0]["name"] == "新名称"
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 200
    assert not Workspace(tmp_path).list("datasets")


def test_preview_preserves_large_customer_identifiers(client, frame):
    frame["用户编码"] = 19859456400262761 + np.arange(len(frame), dtype=np.int64)
    dataset = upload(client, frame)
    row = client.get(f"/api/datasets/{dataset['id']}/preview").json()["rows"][0]
    assert row["用户编码"] == "19859456400262761"
    stored = client.app.state.workspace.read_dataset(dataset["id"])
    assert stored["用户编码"].iloc[0] == 19859456400262761


def test_preview_filters_before_pagination_and_preserves_value_identity(client):
    workspace = client.app.state.workspace
    frame = pd.DataFrame({"编号": np.arange(150) + 2**54, "渠道": [None, "", "缺失值"] * 50, "标签": [True, False] * 75})
    identifier = workspace.add_dataset(frame, "筛选样本")["id"]
    url = f"/api/datasets/{identifier}"
    old_id = str(frame["编号"].iloc[-1])
    workspace.replace_dataset(identifier, frame.assign(渠道="新渠道"), expected_revision=1, processing={"description": "更新渠道"})

    def options(field, query=""):
        return client.get(url + "/preview-values", params={"field": field, "revision": 1, "query": query}).json()

    assert len(options("编号")) == 80
    assert options("编号", old_id) == [old_id]
    assert options("渠道") == [None, "", "缺失值"]
    assert options("渠道", "空字符串") == [""]
    assert client.get(url + "/preview-values", params={"field": "渠道"}).json() == ["新渠道"]
    assert client.get(url + "/preview-values", params={"field": "不存在"}).status_code == 400

    def preview(filters, **kwargs):
        return client.post(url + "/preview", params={"revision": 1, **kwargs}, json={"filters": filters}).json()

    assert preview({"编号": [old_id]})["rows"] == [{"编号": old_id, "渠道": "缺失值", "标签": False}]
    result = preview({"渠道": [None, ""], "标签": ["True"]}, page=2, page_size=7)
    expected = frame.loc[frame["渠道"].isin([None, ""]) & frame["标签"], "编号"]
    assert result["total"] == len(expected)
    assert [row["编号"] for row in result["rows"]] == expected.iloc[7:14].astype(str).tolist()
    response = client.post(url + "/preview?revision=1", json={"filters": {"渠道": [None]}, "query": old_id})
    assert response.json() == {"rows": [], "total": 0}


def test_eda_counts_correlation_and_scatter(client, frame):
    identifier = upload(client, frame)["id"]
    result = client.get(f"/api/datasets/{identifier}/explore", params={"feature": "评分", "other": "金额", "bins": 8}).json()
    assert sum(row["样本数"] for row in result["distribution"]) == len(frame)
    assert len(result["scatter"]) == len(frame)
    corr = result["correlation"]
    idx = corr["columns"].index("评分")
    assert corr["values"][idx][idx] == pytest.approx(1)
    assert client.get(f"/api/datasets/{identifier}/explore", params={"feature": "渠道", "other": "评分"}).status_code == 400


@pytest.mark.parametrize("targets", [[], ["标签"], ["标签", "另一标签"]])
def test_explore_summary_matches_public_summary_and_label_populations(client, frame, targets):
    frame["另一标签"] = np.where(frame["金额"] > 5000, "坏", "好")
    frame.loc[::5, "标签"] = np.nan
    frame.loc[::7, "另一标签"] = None
    frame.loc[::11, "评分"] = np.nan
    identifier = upload(client, frame)["id"]
    workspace = client.app.state.workspace
    stored = workspace.read_dataset(identifier)
    original = stored.copy(deep=True)
    response = client.post(f"/api/datasets/{identifier}/explore", json={"targets": targets, "positive_labels": {name: "坏" if name == "另一标签" else 0 for name in targets}})
    assert response.status_code == 200, response.text
    result = response.json()
    normalized = stored.copy()
    for name in targets:
        normalized[name] = stored[name].eq("坏" if name == "另一标签" else 0).astype(float).where(stored[name].notna())
    expected = normalized.summary(target=targets or None, n_jobs=1)
    if len(targets) > 1:
        assert [c["label"] for c in result["summary_columns"]] == ["统计详情", *targets, "分布详情"]
        field_key, count_key = "统计详情 / 特征名", "统计详情 / 样本数"
        expected.columns = [f"{group} / {name}" for group, name in expected.columns]
        # 不取两个标签有效行的交集；每个标签指标与其单独分析完全一致。
        for name in targets:
            valid = normalized[name].notna()
            single = normalized.loc[valid, ["评分"]].summary(y=normalized.loc[valid, name], n_jobs=1)
            row = next(r for r in result["summary"] if r[field_key] == "评分")
            for metric in ["IV", "KS", "AUC"]:
                assert row[f"{name} / {metric}"] == pytest.approx(single[metric].iat[0])
    else:
        field_key, count_key = "特征名", "样本数"
        assert [c["key"] for c in result["summary_columns"]] == list(expected.columns)
        assert ("AUC" in expected.columns) == bool(targets)
    from hscredit.platform.storage import records

    assert result["summary"] == records(expected)
    assert all(row[count_key] == len(frame) for row in result["summary"])
    assert not set(targets).intersection(row[field_key] for row in result["summary"])
    pd.testing.assert_frame_equal(workspace.read_dataset(identifier), original)
    if not targets:
        assert client.get(f"/api/datasets/{identifier}/explore").json()["summary"] == result["summary"]


@pytest.mark.parametrize(
    "config,message",
    [
        ({"targets": ["不存在"]}, "字段"),
        ({"targets": ["标签", "标签"]}, "不重复"),
        ({"targets": ["评分"]}, "两个非空类别"),
        ({"targets": ["空字段"]}, "两个非空类别"),
        ({"targets": ["标签"], "positive_labels": {"标签": 9}}, "坏样本取值"),
        ({"targets": ["标签"], "positive_labels": {"标签": [1]}}, "坏样本取值"),
        ({"positive_labels": {"标签": 1}}, "已选择的标签"),
    ],
)
def test_explore_rejects_invalid_labels(client, frame, config, message):
    identifier = upload(client, frame)["id"]
    response = client.post(f"/api/datasets/{identifier}/explore", json=config)
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_explore_requires_a_non_label_feature(client):
    identifier = upload(client, pd.DataFrame({"标签": [0, 1, 0, 1]}))["id"]
    response = client.post(f"/api/datasets/{identifier}/explore", json={"targets": ["标签"]})
    assert response.status_code == 400
    assert "非标签字段" in response.json()["detail"]


@pytest.mark.parametrize("method", ["quantile", "uniform", "tree"])
def test_strategy_real_bins_and_rules(client, frame, method):
    identifier = upload(client, frame)["id"]
    job = run_job(client, identifier, "strategy", features=["评分"], method=method, min_lift=1.0)
    result = job["result"]
    bins = next(rows for name, rows in result["tables"].items() if "分箱" in name and rows)
    assert bins
    assert result["tables"]["规则明细"]
    for rule in result["tables"]["规则明细"]:
        assert rule["命中样本数"] >= 10
        assert rule["命中LIFT值"] >= 1
    response = client.get(f"/api/jobs/{job['id']}/files/report.xlsx")
    assert response.content.startswith(b"PK")
    workbook = pd.read_excel(io.BytesIO(response.content), sheet_name=None)
    assert len(workbook) >= 1
    assert "rules.xlsx" in job["artifacts"]


def test_model_search_holdout_and_export_prediction_semantics(client, frame, tmp_path):
    identifier = upload(client, frame)["id"]
    job = run_job(client, identifier, "model", features=["评分", "渠道", "空字段"], models=["logistic", "random_forest"], search=True, n_iter=2, search_spaces={"logistic": {"C": [0.1, 1]}, "random_forest": {"n_estimators": [10, 20]}})
    result = job["result"]
    scores = result["cv_scores"]
    assert len(scores) == 2
    assert result["best_model"] == max(scores, key=scores.get)
    assert all(value > 0.85 for value in scores.values())
    assert len(result["tables"]["交叉验证与搜索"]) == 4
    assert result["tables"]["模型对比"]
    assert set(result["visuals"]) == {"logistic", "random_forest"}
    for visual in result["visuals"].values():
        assert {image["title"] for image in visual["images"]} == {"ROC", "KS", "LIFT", "混淆矩阵"}
        assert visual["importance"]
        for image in visual["images"]:
            assert client.get(f"/api/jobs/{job['id']}/files/{image['file']}").content.startswith(b"\x89PNG")
    directory = tmp_path / "jobs" / job["id"]
    predictions = pd.read_csv(directory / f"predictions_{result['best_model']}.csv")
    estimator = joblib.load(directory / "model.joblib")
    actual = estimator.predict_proba(frame.loc[predictions["原始行号"], ["评分", "渠道", "空字段"]])[:, 1]
    np.testing.assert_allclose(actual, predictions["坏样本概率"], atol=1e-8)
    from hscredit.platform.models import model_catalog

    name = model_catalog()[result["best_model"]]["name"]
    metric = next(r for r in result["tables"][f"{name} / 评估指标"] if r["统计项"] == "AUC")
    assert metric["测试集"] == pytest.approx(roc_auc_score(predictions["真实标签"], actual))
    from sklearn.metrics import auc

    curve = result["curves"][result["best_model"]]
    assert auc(curve["fpr"], curve["tpr"]) == pytest.approx(metric["测试集"])
    unseen = frame.iloc[:2].copy()
    unseen["渠道"] = ["新渠道", None]
    assert np.isfinite(estimator.predict_proba(unseen)).all()
    assert client.get(f"/api/jobs/{job['id']}/files/model.joblib").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/metadata.json").status_code == 404


@pytest.mark.parametrize("label_mode", ["target", "overdue"])
def test_real_third_party_report(client, frame, label_mode):
    identifier = upload(client, frame)["id"]
    job = run_job(client, identifier, "third_party", features=["评分"], label_mode=label_mode, overdue=["逾期天数"], dpds=[7], date="日期", amount="金额", pictures=["bin"], corr=False)
    assert job["result"]["complete"] is True
    assert job["result"]["tables"]["章节状态"]
    response = client.get(f"/api/jobs/{job['id']}/files/report.xlsx")
    assert response.content.startswith(b"PK")
    workbook = pd.read_excel(io.BytesIO(response.content), sheet_name=None)
    assert sum(len(t) for t in workbook.values()) > 10


def test_security_validation_and_failed_job(client, frame):
    identifier = upload(client, frame)["id"]
    base = {"kind": "model", "dataset_id": identifier, "target": "标签", "features": ["评分"]}
    assert client.post("/api/jobs", json={**base, "features": ["标签"]}).status_code == 400
    assert client.post("/api/jobs", json={**base, "features": ["不存在"]}).status_code == 400
    assert client.post("/api/jobs", json={**base, "cv": 0}).status_code == 422
    assert client.post("/api/jobs", json={**base, "model_params": {"logistic": {"C": -1}}}).status_code == 400
    assert client.post("/api/jobs", json={**base, "search_spaces": {"logistic": {"C": []}}}).status_code == 400
    assert client.get("/api/datasets/not-an-id").status_code == 400
    assert client.get("/api/datasets", headers={"host": "evil.example"}).status_code == 403
    assert client.post("/api/jobs", json=base, headers={"origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/datasets", files={"file": ("code.pkl", b"evil")}).status_code == 400


@pytest.mark.parametrize("operator,valid,bad", [
    (">", 3, 2), (">=", 4, 3), ("<", 5, 3), ("<=", 5, 4),
])
def test_target_preview_boundaries_and_independent_populations(client, operator, valid, bad):
    data = pd.DataFrame({"评分": range(5), "M1": [0, 2, 7, 8, np.nan], "M3": [None, 30, 0, 8, 7]})
    identifier = upload(client, data)["id"]
    params = dict(label_mode="overdue", overdue=["M1", "M3"], dpds=[7, 7.0, 2.5],
                  overdue_operator=operator, del_grey=True, handle_missing_target="positive")
    response = client.post(f"/api/datasets/{identifier}/target-preview", json=params)
    assert response.status_code == 200, response.text
    first, _, third, _ = response.json()
    assert first["坏样本"] == bad
    assert first["有效样本"] == valid
    assert first["坏样本率"] == pytest.approx(bad / valid)
    assert third["有效样本"] + third["剔除样本"] == len(data)
    assert client.post(f"/api/datasets/{identifier}/target-preview", json={**params, "handle_missing_target":"raise"}).status_code == 400


def test_overdue_summary_matches_native_multi_label_summary(client, frame):
    frame["M3"] = np.arange(len(frame)) % 40
    frame.loc[1::5, "M3"] = np.nan
    frame = frame[["评分", "逾期天数", "M3"]]
    identifier = upload(client, frame)["id"]
    config = dict(overdue=["逾期天数", "M3"], dpds=[7, 15.5], overdue_operator=">=", del_grey=True, handle_missing_target="positive")
    job = run_job(client, identifier, "explore", label_mode="overdue", **config)
    summary = job["result"]["exploration"]
    expected = frame.summary(features=["评分"], **config, n_jobs=1)
    assert list(expected.columns.get_level_values(0).unique()) == [col["label"] for col in summary["summary_columns"]]
    for label in ["逾期天数>=7", "逾期天数>=15.5", "M3>=7", "M3>=15.5"]:
        assert summary["summary"][0][f"{label} / AUC"] == pytest.approx(expected[label]["AUC"].iloc[0])


def test_dictionary_preview_uses_native_grey_and_missing_and_preserves_order(client, frame):
    data = frame.iloc[:5].copy()
    data["MOB"] = [0, 7, 30, np.nan, 1]
    identifier = upload(client, data)["id"]
    url = f"/api/datasets/{identifier}/target-preview"
    target = {"近月": {"field": "MOB", "operator": ">=", "value": 7}, "手工": [1, 0, 2, None, 1]}
    response = client.post(url, json={"target": target, "del_grey": True, "handle_missing_target": "negative"})
    assert response.status_code == 200, response.text
    assert [(r["标签"], r["有效样本"], r["坏样本"]) for r in response.json()] == [("近月", 5, 2), ("手工", 4, 2)]
    assert client.post(url, json={"target": {"向量": [0, 1, 0, 1, 1]}}).json()[0]["坏样本"] == 3
    for invalid in [{"target": {}}, {"target": {"短": [1]}}, {"target": {"非向量": "MOB"}}, {"target": target, "overdue":["MOB"], "dpds":[7]}]:
        assert client.post(url, json=invalid).status_code in {400, 422}


def test_dictionary_summary_cache_is_order_sensitive(client, frame):
    identifier = upload(client, frame)["id"]
    url = f"/api/datasets/{identifier}/explore"
    target = {"30": {"field": "标签"}, "7": {"field": "逾期天数", "operator": ">", "value": 7}}
    for order in [list(target), list(reversed(target))]:
        response = client.post(url, json={"target": {k: target[k] for k in order}})
        assert response.status_code == 200, response.text
        assert [item["label"] for item in response.json()["summary_columns"]] == ["统计详情", *order, "分布详情"]
    response = client.post(url, json={"target": target, "target_order":["7","30"]})
    assert [item["label"] for item in response.json()["summary_columns"]][1:3] == ["7","30"]


def test_dictionary_model_uses_first_vector_on_original_rows(client, frame):
    identifier = upload(client, frame)["id"]
    primary = frame["标签"].astype(float).mask(frame.index % 8 == 0)
    target = {"自定义主标签": primary.astype(object).where(primary.notna(), None).tolist(), "反向标签": {"field":"标签", "operator":"==", "value":0}}
    job = run_job(client, identifier, "model", workflow=True, target=target, features=["评分"], models=["logistic"], cv=2)
    assert job["config"]["target_order"] == list(target)
    files = f"/api/jobs/{job['id']}/files"
    predictions = pd.read_csv(io.BytesIO(client.get(files+"/predictions_logistic.csv").content))
    np.testing.assert_array_equal(predictions["真实标签"], primary.loc[predictions["原始行号"]])
    assert client.get(files+"/model_metadata.json").json()["target"] == "自定义主标签"
    assert all(any(label+" / 评估指标" in name for name in job["result"]["tables"]) for label in target)


def test_token_access(tmp_path):
    with TestClient(create_app(tmp_path, token="a-long-platform-token")) as client:
        assert client.get("/api/datasets").status_code == 401
        assert client.get("/api/datasets", headers={"Authorization": "Bearer a-long-platform-token"}).status_code == 200


def test_brand_svg_has_browser_supported_media_type(tmp_path, monkeypatch):
    import mimetypes
    import re

    mimetypes.init()
    monkeypatch.setitem(mimetypes.types_map, ".svg", "image/svg")
    with TestClient(create_app(tmp_path)) as client:
        html = client.get("/").text
        logo_url = re.search(r'href="([^"]+\.svg)"', html).group(1)
        response = client.get(logo_url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert b"<svg" in response.content


def test_catalog_covers_public_model_families(client):
    models = client.get("/api/catalog").json()["models"]
    names = {m["class_name"] for m in models.values()}
    assert {"XGBoost", "LightGBM", "CatBoost", "NGBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "SVM", "DecisionTreeClassifier", "LogisticRegression", "ResNet", "FTTransformer", "TabM", "TabNet", "TabPFN", "ScoreCard", "RoundScoreCard", "ProbabilityScoreCard", "RulesClassifier", "ProbabilityCalibrator"} <= names
    for model in models.values():
        if not model["available"]:
            assert model["reason"]


def test_feature_psi_configuration_matches_core_metric():
    from hscredit.core.metrics import psi
    from hscredit.report.model_report import _feature_metric_worker

    train = pd.Series([np.nan] * 2000 + list(range(20)))
    test = pd.Series([np.nan] * 2000 + list(range(10, 30)))
    y = np.arange(len(train)) % 2
    original = _feature_metric_worker((y, train, test))
    configured = _feature_metric_worker((y, train, test, {"min_bin_size": 1}))
    assert np.isnan(original[2])
    assert np.isfinite(configured[2])
    assert configured[2] == pytest.approx(psi(train, test, min_bin_size=1))


def test_report_histogram_uses_isolated_thread_canvas():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from hscredit.core.viz import hist_plot
    from hscredit.core.viz.utils import _threaded_agg_rendering

    backend, figures = plt.get_backend(), plt.get_fignums()
    with _threaded_agg_rendering():
        result = hist_plot(pd.Series(np.arange(20)), title="工作线程直方图")
    figure = result if hasattr(result, "savefig") else result.figure
    assert isinstance(figure.canvas, FigureCanvasAgg)
    assert figure.canvas.manager is None
    assert plt.get_backend() == backend
    assert plt.get_fignums() == figures


def test_background_launch_and_dataframe_registration(tmp_path, frame):
    import urllib.request

    server = launch(port=0, data_dir=tmp_path, background=True, open_browser=False)
    try:
        assert server.url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(server.url + "api/health") as response:
            assert response.status == 200
        assert server.add_dataset(frame, name="内存数据")["rows"] == len(frame)
        with urllib.request.urlopen(server.url) as response:
            assert b'<div id="app">' in response.read()
    finally:
        server.stop()
    assert not server.thread.is_alive()


def test_demo_uses_shared_columns_and_preserves_invalid_targets(client, monkeypatch):
    from hscredit.utils import datasets

    calls = []

    def loader(*, columns):
        calls.append(columns)
        data = pd.DataFrame({c: [1, 2, 3, 4, 5] for c in columns})
        data["MOB6"] = [0, 31, None, -1, np.inf]
        return data

    monkeypatch.setattr(datasets, "hscredit_dataset", loader)
    first = client.post("/api/datasets/demo").json()
    second = client.post("/api/datasets/demo").json()
    assert calls == [datasets.HSCREDIT_EXAMPLE_COLUMNS]
    assert first["id"] == second["id"]
    assert first["rows"] == 5
    assert len(first["source_columns"]) == 88
    assert len(first["default_features"]) == 80
    assert not {"MOB1", "MOB3", "MOB6", "用户编码", "loan_amt"} & set(first["default_features"])
    data = client.app.state.workspace.read_dataset(first["id"])
    assert data[first["default_target"]].tolist()[:2] == [0, 1]
    assert data[first["default_target"]].iloc[2:].isna().all()


def test_failed_tasks_are_visible_and_not_downloadable(client, frame, monkeypatch, tmp_path):
    import sys

    identifier = upload(client, frame)["id"]
    script = tmp_path / "failing_worker.py"
    script.write_text("from hscredit.platform import _worker\nimport hscredit.platform.analysis as analysis\nimport sys\ndef fail(*args):\n    raise RuntimeError('测试计算错误')\nanalysis.train=fail\n_worker.run(*sys.argv[1:])\n", encoding="utf-8")
    monkeypatch.setattr(client.app.state.jobs, "_command", lambda job_id: [sys.executable, str(script), str(tmp_path), job_id])
    response = client.post("/api/jobs", json={"kind": "model", "dataset_id": identifier, "target": "标签", "features": ["评分"]})
    job_id = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] == "失败":
            break
        time.sleep(0.05)
    assert result["status"] == "失败", result
    assert "测试计算错误" in result["message"]
    assert client.get(f"/api/jobs/{job_id}/files/report.xlsx").status_code == 404
