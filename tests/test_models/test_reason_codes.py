"""模型原因码风险方向测试。"""

import numpy as np
import pandas as pd
import shap

from hscredit.core.models.explainability.reason_codes import build_reason_codes
from hscredit.core.models.explainability.result import ExplanationResult


def test_model_reason_codes_only_include_adverse_contributions():
    X = pd.DataFrame({"负债": [8.0], "收入": [3.0], "年龄": [35]})
    explanation = shap.Explanation(values=np.array([[0.3, -0.2, 0.0]]), base_values=np.array([0.4]), data=X.to_numpy(), feature_names=list(X))
    result = ExplanationResult.from_explanation(
        explanation,
        data=X,
        target_class=1,
        output_index=None,
        model_output="probability",
        explainer_type="tree",
        background_summary={},
        metadata={"模型输出": (0.5,)},
    )
    table = build_reason_codes(
        result,
        keep=3,
        risk_direction="higher_output_higher_risk",
        reason_map={"负债": {"code": "R001", "description": "负债水平偏高"}},
    )
    assert (table["风险贡献"].dropna() > 0).all()
    assert table.iloc[0]["原因码"] == "R001"
    assert table["特征"].dropna().tolist() == ["负债"]


def test_reason_code_ties_keep_original_feature_order():
    """并列不利贡献必须保持原特征顺序，避免原因码抖动。"""
    X = pd.DataFrame({"甲": [1.0], "乙": [2.0]})
    explanation = shap.Explanation(
        values=np.array([[0.3, 0.3]]),
        base_values=np.array([0.4]),
        data=X.to_numpy(),
        feature_names=list(X),
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

    table = build_reason_codes(result, keep=2)

    assert table["特征"].tolist() == ["甲", "乙"]
