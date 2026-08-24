"""SHAP 解释器兼容性回归测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from hscredit.core.models import RandomForest
from hscredit.core.models.explainability import ModelExplainer


def _data():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=47,
    )
    return pd.DataFrame(X, columns=list("abcd")), y


def test_shap_importance_handles_binary_three_dimensional_output():
    pytest.importorskip("shap")
    X, y = _data()
    model = RandomForest(n_estimators=8, random_state=47).fit(X, y)
    explainer = ModelExplainer(model)

    values = explainer.compute_shap_values(X.head())
    importance = explainer.get_shap_importance()

    assert values.shape == (5, 4)
    assert importance.shape == (4,)
    assert np.isfinite(importance.to_numpy()).all()


def test_model_explainer_accepts_standard_sklearn_estimators():
    pytest.importorskip("shap")
    X, y = _data()
    model = RandomForestClassifier(n_estimators=8, random_state=47).fit(X, y)

    importance = ModelExplainer(model, feature_names=list(X.columns)).get_shap_importance(X.head())

    assert importance.index.tolist()
    assert set(importance.index) == set(X.columns)


def test_feature_interactions_handle_modern_multioutput_shape():
    pytest.importorskip("shap")
    X, y = _data()
    model = RandomForestClassifier(n_estimators=8, random_state=47).fit(X, y)
    explainer = ModelExplainer(model, feature_names=list(X.columns))
    interactions = np.zeros((3, 4, 4, 2), dtype=float)
    interactions[:, 0, 1, 1] = 2.0
    explainer._interaction_explainer = type(
        "InteractionExplainer",
        (),
        {"shap_interaction_values": lambda self, values: interactions},
    )()

    result = explainer.get_feature_interactions(X.head(3), top_n=2)

    assert result.iloc[0][["特征1", "特征2"]].tolist() == ["a", "b"]
    assert result.iloc[0]["交互强度"] == pytest.approx(2.0)
