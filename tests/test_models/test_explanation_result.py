"""结构化 SHAP 解释结果测试。"""

import numpy as np
import pandas as pd
import shap

from hscredit.core.models.evaluation.explanation import ExplanationResult, fingerprint_frame


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
