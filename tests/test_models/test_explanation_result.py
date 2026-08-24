"""结构化 SHAP 解释结果测试。"""

import numpy as np
import pandas as pd
import pytest
import shap

from hscredit.core.models.explainability.result import ExplanationResult, fingerprint_frame


def test_explanation_result_preserves_chinese_columns_and_sample_ids():
    X = pd.DataFrame({"收入": [1.0, 2.0], "年龄": [20, 30]}, index=["A", "B"])
    explanation = shap.Explanation(
        values=np.array([[0.1, -0.2], [0.3, -0.4]]),
        base_values=np.array([0.5, 0.5]),
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )
    result = ExplanationResult.from_explanation(
        explanation,
        data=X,
        target_class=1,
        output_index=1,
        model_output="probability",
        explainer_type="tree",
        background_summary={"样本数": 2},
        metadata={"随机种子": 42},
    )
    assert result.values.shape == (2, 2)
    assert result.feature_names == ["收入", "年龄"]
    assert result.sample_ids.tolist() == ["A", "B"]
    assert result.position_for("B") == 1


def test_fingerprint_changes_when_values_or_schema_change():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    assert fingerprint_frame(X) != fingerprint_frame(X.assign(x=[1.0, 3.0]))
    assert fingerprint_frame(X) != fingerprint_frame(X.rename(columns={"x": "y"}))


def test_explanation_result_returns_defensive_data_and_value_copies():
    """冻结结果不能通过 DataFrame 或数组属性被原地篡改。"""
    X = pd.DataFrame({"收入": [1.0], "年龄": [20]})
    explanation = shap.Explanation(
        values=np.array([[0.1, -0.2]]),
        base_values=np.array([0.5]),
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )
    result = ExplanationResult.from_explanation(
        explanation,
        data=X,
        target_class=1,
        output_index=None,
        model_output="probability",
        explainer_type="tree",
        background_summary={},
        metadata={},
    )

    exposed_data = result.data
    exposed_values = result.values
    exposed_data.iloc[0, 0] = 999.0
    exposed_values[0, 0] = 999.0

    assert result.data.iloc[0, 0] == 1.0
    assert result.values[0, 0] == 0.1


def test_legacy_multioutput_base_values_are_not_confused_with_two_samples():
    """样本数等于输出数时，旧式多输出基准值仍应按目标输出选择。"""

    class LegacyExplanation:
        values = [np.array([[0.1], [0.2]]), np.array([[0.3], [0.4]])]
        base_values = np.array([0.2, 0.8])
        data = np.array([[1.0], [2.0]])
        feature_names = ["收入"]

    result = ExplanationResult.from_explanation(
        LegacyExplanation(),
        data=pd.DataFrame({"收入": [1.0, 2.0]}),
        target_class=1,
        output_index=1,
        model_output="probability",
        explainer_type="legacy",
        background_summary={},
        metadata={},
    )

    assert result.base_values.tolist() == pytest.approx([0.8, 0.8])
