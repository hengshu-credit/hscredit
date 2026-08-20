"""现代 ModelExplainer 核心语义测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier

from hscredit import ValidationError
from hscredit.core.models.evaluation import ModelExplainer
from hscredit.core.models.evaluation.explanation import ExplanationResult, fingerprint_frame


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


def test_legacy_cache_is_not_reused_for_different_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=3)
    X = pd.DataFrame(X, columns=list("abcd"))
    model = RandomForestClassifier(n_estimators=6, random_state=3).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(10))
    first = explainer.compute_shap_values(X.iloc[:3])
    second = explainer.compute_shap_values(X.iloc[3:6])
    assert explainer.last_result_.dataset_fingerprint == fingerprint_frame(X.iloc[3:6])
    assert not np.array_equal(first, second)
