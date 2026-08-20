"""模型原因码风险方向测试。"""

import numpy as np
import pandas as pd
import shap

from hscredit.core.models.evaluation.explanation import ExplanationResult
from hscredit.core.models.evaluation.reason_codes import build_reason_codes


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
