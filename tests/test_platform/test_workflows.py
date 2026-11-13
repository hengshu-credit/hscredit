"""验证扩展工作流的规则语义、独立样本评估与服务鉴权。"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web.server import JobRequest, create_app
from hscredit.web.storage import Workspace
from hscredit.web.workflows import strategy_workflow, ruleset_workflow
from hscredit.web.interactions import interactions


def test_legacy_preprocessing_preserves_schema_and_training_statistics(tmp_path):
    from hscredit.web.analysis import _LegacyFramePreprocessor

    train = pd.DataFrame({"分数": [1.0, 3.0, np.nan] * 22, "空数值": np.nan, "渠道": ["常用"] * 6 + [f"渠道{i}" for i in range(60)], "空类别": None})
    prepare = _LegacyFramePreprocessor(["分数", "空数值"], ["渠道", "空类别"]).fit(train)
    sample = pd.DataFrame({"分数": [np.nan, 3.0], "空数值": [7.0, np.nan], "渠道": [None, "新渠道"], "空类别": [None, "新值"]}, index=[8, 2])
    result = prepare.transform(sample)
    assert result.index.tolist() == [8, 2]
    assert result.columns.tolist() == prepare.transform(train).columns.tolist()
    np.testing.assert_allclose(result["numeric__分数"], [0.0, 1.0 / np.sqrt(2.0 / 3.0)])
    np.testing.assert_array_equal(result["numeric__空数值"], [7.0, 0.0])
    channel = result.filter(like="category__渠道_")
    assert channel.shape[1] == 50
    assert channel.loc[8, "category__渠道_常用"] == 1
    assert channel.loc[2].sum() == 0
    np.testing.assert_array_equal(result.filter(like="category__空类别_").sum(axis=1), [1, 0])
    path = tmp_path / "prepare.joblib"
    joblib.dump(prepare, path)
    pd.testing.assert_frame_equal(result, joblib.load(path).transform(sample))


@pytest.fixture
def prepared(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    data = pd.DataFrame({"评分": x, "金额": rng.uniform(0, 100, 120), "渠道": np.where(x > 0, "A", "B"), "布尔": x > 0, "日期": pd.date_range("2025-01-01", periods=120), "标签": (x + rng.normal(size=120) > 0).astype(int)})
    workspace = Workspace(tmp_path / "workspace")
    training = workspace.add_dataset(data, "训练")
    evaluation = workspace.add_dataset(data.iloc[::-1].reset_index(drop=True), "评估")
    config = JobRequest(kind="strategy", workflow=True, dataset_id=training["id"], dataset_revision=1, features=["评分"], target="标签", evaluation_datasets=[{"id": evaluation["id"], "revision": 1}], max_n_bins=3).dict()
    output = tmp_path / "reports"
    output.mkdir()
    return workspace, data, config, output


@pytest.mark.parametrize("mode", ["single", "pair", "binary", "multiway"])
def test_strategy_modes_use_executable_rules_and_evaluate_other_samples(prepared, mode):
    from hscredit.core.rules import Rule

    workspace, data, config, output = prepared
    config["strategy_mode"] = mode
    if mode != "single":
        config["features"] = ["评分", "金额"]
    result = strategy_workflow(workspace, config, output, lambda *args: None)
    assert result["candidates"]
    assert any("评估集" in key for key in result["tables"])
    hits = [Rule(c["expression"]).predict(data).to_numpy() for c in result["candidates"]]
    np.testing.assert_array_equal(np.sum(hits, axis=0), np.ones(len(data)))
    if mode in {"binary", "multiway"}:
        assert "tree_report.xlsx" in result["artifacts"]
        assert any(k.startswith("报告预览") for k in result["tables"])


def test_rule_order_and_combination_report_are_real(prepared):
    workspace, data, config, output = prepared
    config.update(kind="ruleset", selected_rules=[{"name": "正分", "expression": "评分 > 0"}, {"name": "高金额", "expression": "金额 > 50"}], rule_logic="and")
    result = ruleset_workflow(workspace, config, output, lambda *args: None)
    model = joblib.load(output / "model.joblib")
    np.testing.assert_array_equal(model.predict(data), ((data["评分"] > 0) & (data["金额"] > 50)).astype(int))
    assert any("顺序累积" in key for key in result["tables"])
    assert "swap_out_report_2.xlsx" in result["artifacts"]


def test_interactions_all_pairs_density_and_label_missing(prepared):
    _, data, _, _ = prepared
    data.loc[0, "标签"] = np.nan
    result = interactions(data, ["评分", "渠道", "日期"], "标签", bins=5)
    assert len(result["pairs"]) == 3
    assert len(result["distributions"]) == 3
    assert "（标签缺失）" in {p["label"] for p in result["points"]}
    for pair in result["pairs"]:
        assert sum(row["样本数"] for row in pair["density"]) == len(data)


def test_multilabel_trends_categorical_fixed_bins(prepared):
    workspace, data, config, output = prepared
    data["第二标签"] = 1 - data["标签"]
    data.loc[0, "第二标签"] = np.nan
    asset = workspace.add_dataset(data, "多标签")
    config.update(dataset_id=asset["id"], evaluation_datasets=[], features=["渠道"], evaluation_targets=["第二标签"], date="日期", group="布尔")
    result = strategy_workflow(workspace, config, output, lambda *args: None)
    assert any("多标签分箱图" in image["title"] for image in result["images"])
    assert any("时间趋势" in image["title"] and "第二标签" in image["title"] for image in result["images"])
    assert any("分组趋势" in image["title"] for image in result["images"])
    assert any("分组多标签分箱图" in image["title"] for image in result["images"])
    assert any("多标签时间趋势分箱图" in image["title"] for image in result["images"])


def test_experiment_external_evaluation_scorecard_and_export_prediction(prepared):
    from hscredit.web.experiments import experiment

    workspace, data, config, output = prepared
    config.update(kind="model", models=["logistic"], model_params={"logistic": {"max_iter": 100}}, test_dataset=config["evaluation_datasets"][0], evaluation_datasets=[], amount="金额", scorecard=True, cv=2)
    result = experiment(workspace, config, output, lambda *args: None)
    model = joblib.load(output / "model.joblib")
    exported = pd.read_csv(output / "predictions_logistic.csv")
    evaluated = workspace.read_dataset(config["test_dataset"]["id"])
    np.testing.assert_allclose(model.predict_proba(evaluated[config["features"]])[:, 1], exported["坏样本概率"])
    assert np.isfinite(model.predict_score(data[config["features"]])).all()
    from openpyxl import load_workbook
    from openpyxl.utils.cell import range_boundaries

    book = load_workbook(output / "report_logistic.xlsx")
    performance = book["2-模型性能"]
    _, header, right, _ = range_boundaries(performance.auto_filter.ref)
    assert performance.cell(header - 1, 2).value == "订单口径"
    assert performance.cell(header + 1, right).value == pytest.approx(np.average(data["标签"], weights=data["金额"]), abs=5e-5)
    values = {c.value for row in book["5-模型参数"] for c in row}
    assert {"Std.Err", "P>|z|", "VIF", "刻度项", "评分区间", "numeric__评分"} <= values
    book.close()
    assert "model_logistic.joblib" in result["artifacts"]
    metrics = next(rows for name, rows in result["tables"].items() if name.startswith("logistic / 测试集") and name.endswith("评估指标"))
    from sklearn.metrics import roc_auc_score

    row = next(row for row in metrics if row["统计项"] == "AUC")
    assert next(v for k, v in row.items() if k.startswith("测试集")) == pytest.approx(roc_auc_score(evaluated["标签"], exported["坏样本概率"]))
    shared = result["tables"]["跨样本指标 / logistic / 标签"]
    assert [k for k in shared[0] if k != "统计项"] == ["训练集", "测试集 · 评估"]
    test_auc = next(row["测试集 · 评估"] for row in shared if row["统计项"] == "AUC")
    assert test_auc == pytest.approx(roc_auc_score(evaluated["标签"], exported["坏样本概率"]))


def test_overdue_training_primary_is_used_in_external_predictions_and_reports(prepared):
    from hscredit.web.experiments import experiment

    workspace, _, config, output = prepared
    external = config["evaluation_datasets"][0]
    config.update(kind="model", label_mode="overdue", target=None, overdue=["金额"], dpds=[25, 70], overdue_operator=">=", primary_index=1, features=["评分"], models=["logistic"], test_dataset=external, evaluation_datasets=[], cv=2)
    result = experiment(workspace, config, output, lambda *args: None)
    predictions = pd.read_csv(output / "predictions_logistic.csv")
    amounts = workspace.read_dataset(external["id"])["金额"]
    np.testing.assert_array_equal(predictions["真实标签"], amounts.ge(70).astype(int))
    for threshold in [25, 70]:
        assert any("测试集" in name and f"金额>={threshold} / 评估指标" in name for name in result["tables"])


def test_service_token_is_independent_persistent_rotatable_and_rules_are_safe(tmp_path):
    token = "platform-management-token"
    with TestClient(create_app(tmp_path, token=token)) as client:
        headers = {"Authorization": "Bearer " + token}
        response = client.post("/api/services", headers=headers, json={"name": "策略", "rules": [{"name": "低分", "expression": "评分 < 500"}]})
        assert response.status_code == 201, response.text
        service = response.json()
        endpoint = service["endpoint"]
        payload = {"records": [{"评分": 499}, {"评分": 500}]}
        assert client.post(endpoint, json=payload).status_code == 401
        assert client.post(endpoint, json=payload, headers=headers).status_code == 401
        auth = {"Authorization": "Bearer " + service["token"]}
        predicted = client.post(endpoint, json=payload, headers=auth)
        assert predicted.status_code == 200, predicted.text
        assert [p["预测"] for p in predicted.json()["predictions"]] == [1, 0]
        assert client.post(endpoint, json={"records": [{}]}, headers=auth).status_code == 422
        stored = json.loads((tmp_path / "services" / service["id"] / "metadata.json").read_text(encoding="utf-8"))
        assert service["token"] not in json.dumps(stored)
        assert "token_hash" not in client.get("/api/services", headers=headers).text
        assert client.get(f"/services/{service['id']}/openapi.json", headers=auth).status_code == 200
        rotated = client.post(f"/api/services/{service['id']}/rotate-token", headers=headers).json()
        assert client.post(endpoint, json=payload, headers=auth).status_code == 401
        auth = {"Authorization": "Bearer " + rotated["token"]}
        assert client.post(endpoint, json=payload, headers=auth).status_code == 200
        client.patch(f"/api/services/{service['id']}", headers=headers, json={"enabled": False})
        assert client.post(endpoint, json=payload, headers=auth).status_code == 503
        malicious = client.post("/api/services", headers=headers, json={"name": "非法", "rules": [{"name": "非法", "expression": "评分.__class__"}]})
        assert malicious.status_code == 400


def test_service_cors_is_limited_to_inference(client_with_service):
    client, service = client_with_service
    headers = {"Origin": "https://risk.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"}
    endpoint = service["endpoint"]
    response = client.options(endpoint, headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
    for auth, status in [({}, 401), ({"Authorization": "Bearer " + service["token"]}, 200)]:
        response = client.post(endpoint, headers={"Origin": headers["Origin"], **auth}, json={"records": [{"分": 499}]})
        assert response.status_code == status
        assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-origin" not in client.options("/api/services", headers=headers).headers


@pytest.fixture
def client_with_service(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/services", json={"name": "低分", "rules": [{"name": "低于500", "expression": "分 < 500"}]})
        assert response.status_code == 201
        yield client, response.json()


def test_composite_panels_use_full_marginal_and_pairwise_populations():
    n = 3100
    data = pd.DataFrame({"金额": np.arange(n, dtype=float), "渠道": ["线上", "线下"] * (n // 2), "日期": pd.date_range("2024-01-01", periods=n, freq="h"), "标签": [0, 0, 1, 1] * (n // 4)})
    data.loc[0, "金额"] = np.inf
    data.loc[1, "日期"] = pd.NaT
    result = interactions(data, ["渠道", "金额", "日期"], "标签")
    marginal = result["distributions"][1]
    assert marginal["five_number"] == data["金额"].iloc[1:].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    assert sum(r["有效数"] for r in marginal["by_group"]) == n - 1
    by_channel, _, by_month = result["pairs"]
    assert by_channel["valid"] == n - 1 and len(by_channel["points"]) == 3000
    assert sum(r["样本数"] for r in by_channel["means"]) == n - 1
    row = next(r for r in by_channel["means"] if r["维度"] == "线上" and r["标签"] == "0")
    assert row["均值"] == pytest.approx(data["金额"].iloc[4::4].mean())
    assert sum(r["样本数"] for r in by_month["means"]) == n - 2
    assert (by_month["dimension"], by_month["measure"]) == (2, 1)


def test_pair_population_independent_of_third_axis_and_column_names():
    data = pd.DataFrame({"标签": [1, 2, 3, 4], "样本数": [1, 1, 2, 2], "空列": [np.nan] * 4, "分组": [0, 0, 1, 1]})
    result = interactions(data, ["标签", "样本数", "空列"], "分组")
    assert result["valid"] == 0
    pair = result["pairs"][0]
    assert pair["valid"] == len(pair["points"]) == sum(r["样本数"] for r in pair["density"]) == 4
    assert sorted(p["values"] for p in pair["points"]) == [[1, 1], [2, 1], [3, 2], [4, 2]]


def test_exploration_free_text_retains_missing_and_counts_characters():
    data = pd.DataFrame({"备注": ["记录" + str(i) for i in range(40)] + [None]})
    result = interactions(data, ["备注"])
    assert result["axes"][0]["kind"] == "text"
    assert result["valid"] == 40
    assert sorted(point["values"][0] for point in result["points"]) == sorted(data["备注"].dropna().str.len())


def test_uploaded_inference_file_preserves_probabilities(tmp_path):
    from hscredit.core.models import LogisticRegression

    x = pd.DataFrame({"评分": np.arange(40)})
    model = LogisticRegression(calculate_stats=False).fit(x, np.arange(40) > 20)
    path = tmp_path / "inference.pkl"
    model.save_inference(path)
    with TestClient(create_app(tmp_path / "services-test")) as client:
        res = client.post("/api/services/upload", files={"file": (path.name, path.read_bytes())}, data={"name": "上传模型", "trusted": "true"})
        assert res.status_code == 201, res.text
        service = res.json()
        samples = x.iloc[[0, 19, 39]]
        result = client.post(service["endpoint"], headers={"Authorization": "Bearer " + service["token"]}, json={"records": samples.to_dict("records")})
        assert result.status_code == 200, result.text
        np.testing.assert_allclose([row["概率"] for row in result.json()["predictions"]], model.predict_proba(samples))


@pytest.mark.parametrize("key", ["scorecard", "roundscorecard"])
def test_native_scorecard_parameter_sheet_matches_fitted_card(prepared, key):
    from openpyxl import load_workbook
    from hscredit.web.analysis import model_pipeline
    from hscredit.report import ModelReport

    _, data, _, output = prepared
    X, y = data[["评分"]], data["标签"]
    pipeline = model_pipeline(key, {}, ["评分"], [], 42).fit(X, y)
    card = pipeline.steps[-1][1].model_
    path = output / "评分卡.xlsx"
    ModelReport(pipeline, X_train=X, y_train=y, n_jobs=1).to_excel(path, with_plots=False, mode="strict", max_n_bins=3)
    book = load_workbook(path)
    sheet = book["5-模型参数"]

    def column_below(header, count):
        cell = next(c for row in sheet for c in row if c.value == header)
        return [sheet.cell(cell.row + offset, cell.column).value for offset in range(1, count + 1)]

    summary = card.lr_model_.summary()
    np.testing.assert_allclose(column_below("Coef.", len(summary)), summary["Coef."])
    points = card.scorecard_points()
    np.testing.assert_allclose(column_below("对应分数", len(points)), points["对应分数"])
    rates = card.score_to_bad_rate_table(card.predict(X, input_type="raw"), y, max_n_bins=3)
    np.testing.assert_allclose(column_below("坏样本率", len(rates)), rates["坏样本数"] / rates["样本数"])
    assert "pdo" in column_below("刻度项", len(card.scorecard_scale()))
    book.close()
