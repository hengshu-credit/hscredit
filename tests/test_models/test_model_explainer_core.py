"""现代 ModelExplainer 核心语义测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hscredit import ValidationError
from hscredit.core.models import RandomForest
from hscredit.core.models.explainability import ModelExplainer
from hscredit.core.models.explainability.result import ExplanationResult, fingerprint_frame


def test_tree_explainer_returns_probability_result_with_additivity():
    X, y = make_classification(n_samples=80, n_features=4, random_state=7)
    X = pd.DataFrame(X, columns=["甲", "乙", "丙", "丁"])
    model = RandomForestClassifier(n_estimators=12, max_depth=3, random_state=7).fit(X, y)
    result = ModelExplainer(model, background_data=X.iloc[:20], random_state=7).explain(X.iloc[20:25])
    expected = model.predict_proba(X.iloc[20:25])[:, 1]
    assert isinstance(result, ExplanationResult)
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), expected, atol=1e-5)
    assert result.target_class == 1
    assert result.model_output == "probability"


def test_multiclass_requires_target_class():
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(X, y)
    with pytest.raises(ValidationError, match="多分类.*target_class"):
        ModelExplainer(model, background_data=X.head(20), target_class=None).explain(X.head())


def test_explain_rejects_empty_input_with_project_validation_error():
    """空输入应在 SHAP 后端前得到稳定中文错误。"""
    X, y = make_classification(n_samples=30, n_features=4, random_state=2)
    X = pd.DataFrame(X, columns=list("abcd"))
    model = RandomForestClassifier(n_estimators=4, random_state=2).fit(X, y)

    with pytest.raises(ValidationError, match="不能为空"):
        ModelExplainer(model, background_data=X.head()).explain(X.iloc[:0])


def test_legacy_cache_is_not_reused_for_different_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=3)
    X = pd.DataFrame(X, columns=list("abcd"))
    model = RandomForestClassifier(n_estimators=6, random_state=3).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(10))
    first = explainer.compute_shap_values(X.iloc[:3])
    second = explainer.compute_shap_values(X.iloc[3:6])
    assert explainer.last_result_.dataset_fingerprint == fingerprint_frame(X.iloc[3:6])
    assert not np.array_equal(first, second)


def test_binary_raw_target_class_zero_uses_class_zero_margin():
    """二分类一维 decision_function 的 class 0 解释必须反转方向。"""
    values, y = make_classification(n_samples=60, n_features=4, random_state=13)
    X = pd.DataFrame(values, columns=list("abcd"))
    model = LogisticRegression(max_iter=300).fit(X, y)
    explained = X.iloc[40:44]

    result = ModelExplainer(
        model,
        background_data=X.iloc[:12],
        algorithm="permutation",
        model_output="raw",
        target_class=0,
        random_state=13,
    ).explain(explained)

    np.testing.assert_allclose(
        result.base_values + result.values.sum(axis=1),
        -model.decision_function(explained),
        atol=1e-5,
    )


def test_score_auto_uses_supported_algorithm_and_high_score_is_low_risk():
    """树模型评分解释应自动走评分函数，并按高分低风险排序。"""
    values, y = make_classification(n_samples=60, n_features=4, random_state=14)
    X = pd.DataFrame(values, columns=list("abcd"))
    model = RandomForest(n_estimators=8, max_depth=3, random_state=14).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.iloc[:12], model_output="score", random_state=14)

    result = explainer.explain(X.iloc[40:45])
    selected = explainer.select_representative_samples(result, threshold=500)
    highest_risk = selected[selected["选择理由"].str.contains("最高风险")]

    assert result.explainer_type == "permutation"
    assert highest_risk.iloc[0]["模型输出"] == pytest.approx(selected["模型输出"].min())
