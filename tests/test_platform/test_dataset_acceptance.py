"""以统一现金贷数据验收平台的探索、挖掘、评估与发布链路。"""

import io
import os
import time

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from hscredit.platform import create_app
from hscredit.platform.demo import DEMO_TARGET
from hscredit.utils.datasets import hscredit_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("HSCREDIT_RUN_DATASET_TESTS") != "1", reason="显式启用统一现金贷数据验收，避免常规测试自动下载数据"),
]

FEATURES = ["BASE_X187", "age_year"]
SECOND_TARGET = "MOB1大于7天"


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("hscredit_dataset"))) as client:
        ws = client.app.state.workspace
        data = hscredit_dataset(columns=[*FEATURES, "授信时间", "流量渠道", "MOB1", "MOB6"])
        for name, source, threshold in [(DEMO_TARGET, "MOB6", 30), (SECOND_TARGET, "MOB1", 7)]:
            values = data[source]
            data[name] = values.gt(threshold).astype(float).where(values.ge(0) & np.isfinite(values))
        full = ws.add_dataset(data, "统一样例 · 全量探索")
        # 固定抽样后按授信时间切成不重叠的角色，保留原数据中的无效标签。
        sample = data.sample(6000, random_state=42).sort_values("授信时间")
        train, test, other = (part.reset_index(drop=True) for part in (sample.iloc[:4000], sample.iloc[4000:5000], sample.iloc[5000:]))
        ids = [ws.add_dataset(frame, name)["id"] for frame, name in [(train, "训练"), (test, "测试"), (other, "评估")]]
        yield client, full, ids, (train, test, other)


def run(client, kind, dataset_id, **config):
    submitted = client.post("/api/jobs", json={"kind": kind, "workflow": True, "dataset_id": dataset_id, **config})
    assert submitted.status_code == 202, submitted.text
    job_id = submitted.json()["id"]
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in {"成功", "失败", "已停止"}:
            assert job["status"] == "成功", job["message"]
            return job
        time.sleep(0.4)
    pytest.fail("统一数据验收任务超时")


def test_full_population_interactions(assets):
    client, full, *_ = assets
    job = run(client, "interaction", full["id"], features=[FEATURES[0], "授信时间", "流量渠道"], target=DEMO_TARGET, bins=8)
    result = job["result"]["interaction"]
    assert result["total"] == full["rows"]
    assert len(result["pairs"]) == len(result["distributions"]) == 3
    assert len(result["points"]) == min(3000, result["valid"])
    for pair in result["pairs"]:
        assert sum(row["样本数"] for row in pair["density"]) == pair["valid"]
        assert {r["x"] for r in pair["density"]} <= set(pair["order"][0])
        assert {r["y"] for r in pair["density"]} <= set(pair["order"][1])


def test_mine_evaluate_and_publish_selected_bins(assets):
    from hscredit.core.rules import Rule

    client, _, (train, _, other), (_, _, evaluation) = assets
    options = dict(target=DEMO_TARGET, evaluation_datasets=[{"id": other}], max_n_bins=3)
    single = run(client, "strategy", train, strategy_mode="single", features=FEATURES[:1], evaluation_targets=[SECOND_TARGET], date="授信时间", freq="Q", group="流量渠道", **options)
    titles = [image["title"] for image in single["result"]["images"]]
    for section in ["分箱图", "时间趋势分箱图", "多标签分箱图", "多标签时间趋势分箱图", "分组趋势分箱图", "分组多标签分箱图"]:
        assert any("评估集1" in t and section in t for t in titles), section

    pair = run(client, "strategy", train, strategy_mode="pair", features=FEATURES, **options)
    rules = [{k: row[k] for k in ["name", "expression"]} for row in pair["result"]["candidates"][:2]]
    leaf_masks = [Rule(row["expression"]).predict(evaluation).to_numpy() for row in pair["result"]["candidates"]]
    np.testing.assert_array_equal(np.sum(leaf_masks, axis=0), np.ones(len(evaluation)))
    selected = run(client, "ruleset", train, selected_rules=rules[::-1], **options)
    assert any(key.startswith("报告预览 / swap_out_report_2.xlsx") for key in selected["result"]["tables"])

    service = client.post("/api/services", json={"name": "选中二维分箱", "source_job": selected["id"]})
    assert service.status_code == 201, service.text
    service = service.json()
    response = client.post(service["endpoint"], headers={"Authorization": "Bearer " + service["token"], "Origin": "https://risk.example"}, json={"records": evaluation[FEATURES].to_dict("records")})
    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-origin"] == "*"
    np.testing.assert_array_equal([r["预测"] for r in response.json()["predictions"]], leaf_masks[0] | leaf_masks[1])


@pytest.mark.parametrize("mode,edits", [("binary", [{"node": 0, "feature": "age_year", "threshold": 35}]), ("multiway", [{"node": 0, "feature": "age_year", "thresholds": [28, 36]}])])
def test_manual_tree_report(assets, mode, edits):
    from hscredit.core.rules import Rule

    client, _, (train, _, other), (_, _, evaluation) = assets
    tree = run(client, "strategy", train, strategy_mode=mode, target=DEMO_TARGET, features=FEATURES, tree_edits=edits, tree_depth=2, evaluation_datasets=[{"id": other}])
    assert "tree_report.xlsx" in tree["artifacts"]
    assert any(k.startswith("报告预览 / tree_report.xlsx") for k in tree["result"]["tables"])
    masks = [Rule(r["expression"]).predict(evaluation).to_numpy() for r in tree["result"]["candidates"]]
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(evaluation)))


def test_model_comparison_report_inference_and_shap(assets):
    client, _, (train, test, other), (_, test_data, evaluation) = assets
    common = dict(target=DEMO_TARGET, features=FEATURES, test_dataset={"id": test}, evaluation_datasets=[{"id": other}], cv=2)
    experiment = run(client, "model", train, models=["logistic"], scorecard=True, explain=True, evaluation_targets=[SECOND_TARGET], **common)
    assert any("SHAP" in img["title"] for img in experiment["result"]["visuals"]["logistic"]["images"])
    assert "report_logistic_label_1.xlsx" in experiment["artifacts"]
    assert "comparison.xlsx" not in experiment["artifacts"]
    explanation = run(client, "explain", other, source_job=experiment["id"], sample_rows=[2])
    assert explanation["result"]["images"]
    model = joblib.load(io.BytesIO(client.get("/api/jobs/" + experiment["id"] + "/files/model.joblib").content))
    service = client.post("/api/services", json={"name": "模型验收", "source_job": experiment["id"]}).json()
    pred = client.post(service["endpoint"], headers={"Authorization": "Bearer " + service["token"]}, json={"records": evaluation[FEATURES].to_dict("records")}).json()["predictions"]
    np.testing.assert_allclose([r["概率"] for r in pred], model.predict_proba(evaluation[FEATURES]))
    np.testing.assert_allclose([r["评分"] for r in pred], model.predict_score(evaluation[FEATURES]))
    valid = test_data.dropna(subset=[SECOND_TARGET])
    expected = roc_auc_score(valid[SECOND_TARGET], model.predict_proba(valid[FEATURES])[:, 1])
    table = next(v for k, v in experiment["result"]["tables"].items() if "测试集 · 测试 / " + SECOND_TARGET + " / 评估指标" in k)
    assert next(row["测试集 · 测试"] for row in table if row["统计项"] == "AUC") == pytest.approx(expected)

    comparison = run(client, "model", train, model_mode="compare", models=["logistic", "random_forest"], selector="iv", selector_threshold=0, search=True, n_iter=1, model_params={"random_forest": {"n_estimators": 10}}, **common)
    assert len(comparison["result"]["tables"]["模型对比"]) == 2
    assert "comparison.xlsx" in comparison["artifacts"]
