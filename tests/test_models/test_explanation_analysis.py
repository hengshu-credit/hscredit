"""模型解释结构化分析测试。"""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hscredit import ValidationError
from hscredit.core.models.explainability import ModelExplainer


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


def test_refit_stability_retrains_cloned_model_on_fixed_validation_data():
    values, y = make_classification(n_samples=50, n_features=4, random_state=31)
    X = pd.DataFrame(values, columns=list("abcd"))
    model = LogisticRegression(max_iter=300).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(12), random_state=31)
    table = explainer.get_stability_report(
        mode="refit",
        X_train=X.iloc[:40],
        y_train=y[:40],
        X_validation=X.iloc[40:],
        n_bootstrap=3,
        top_k=2,
        random_state=31,
    )
    assert set(table["稳定性模式"]) == {"重训Bootstrap"}


def test_single_feature_interactions_return_fixed_empty_schema():
    """单特征模型没有特征对，应返回可消费的空表而不是排序报错。"""
    X = pd.DataFrame({"收入": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]})
    y = [0, 0, 0, 1, 1, 1]
    model = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=9).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.iloc[:3], random_state=9)

    table = explainer.get_feature_interactions(X=X.iloc[3:])

    assert table.empty
    assert table.columns.tolist() == ["特征1", "特征2", "交互强度"]


def test_structured_analysis_rejects_invalid_limits(explained):
    """展示、聚类、交互和稳定性上限不能静默产生空或失真结果。"""
    explainer, result = explained

    with pytest.raises(ValidationError, match="top_n"):
        explainer.get_sample_report(result, top_n=0)
    with pytest.raises(ValidationError, match="max_clusters"):
        explainer.get_feature_clusters(result, max_clusters=0)
    with pytest.raises(ValidationError, match="top_n"):
        explainer.get_feature_interactions(result=result, top_n=0)
    with pytest.raises(ValidationError, match="confidence_level"):
        explainer.get_stability_report(result, n_bootstrap=3, confidence_level=1.5)
    with pytest.raises(ValidationError, match="top_k"):
        explainer.get_stability_report(result, n_bootstrap=3, top_k=0)
