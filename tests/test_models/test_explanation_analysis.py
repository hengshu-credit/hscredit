"""模型解释结构化分析测试。"""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from hscredit.core.models.evaluation import ModelExplainer


@pytest.fixture()
def explained():
    values, y = make_classification(n_samples=70, n_features=4, random_state=11)
    X = pd.DataFrame(values, columns=["收入", "负债", "年龄", "查询次数"], index=range(100, 170))
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=11).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(20), random_state=11)
    return explainer, explainer.explain(X.tail(20))


def test_global_sample_and_representative_reports_have_chinese_audit_columns(explained):
    explainer, result = explained
    global_report = explainer.get_global_report(result)
    assert {"特征", "平均绝对SHAP值", "SHAP重要性占比", "正向影响占比", "Pearson相关系数"} <= set(global_report)
    assert global_report["SHAP重要性占比"].sum() == pytest.approx(1.0)
    sample_report = explainer.get_sample_report(result, sample_id=result.sample_ids[0], top_n=2)
    assert sample_report["贡献排名"].tolist() == [1, 2]
    selected = explainer.select_representative_samples(result, threshold=0.5)
    assert selected["样本索引"].is_unique
    assert {"选择理由", "模型输出", "风险排名", "阈值距离"} <= set(selected)


def test_correlation_clusters_interactions_and_stability_are_structured(explained):
    explainer, result = explained
    assert set(explainer.get_correlation_report(result, kind="shap_shap")) == set(result.feature_names)
    clusters = explainer.get_feature_clusters(result)
    assert {"特征", "叶序", "聚类编号"} <= set(clusters)
    interactions = explainer.get_feature_interactions(result=result, top_n=3)
    assert {"特征1", "特征2", "交互强度"} <= set(interactions)
    stability = explainer.get_stability_report(result, n_bootstrap=8, top_k=2, random_state=3)
    assert {"稳定性模式", "置信区间下限", "置信区间上限", "排名标准差", "Top-K入选率"} <= set(stability)
    assert stability["Top-K入选率"].between(0, 1).all()
