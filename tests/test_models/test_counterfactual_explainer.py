"""受约束反事实解释测试。"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from hscredit.core.models import CounterfactualExplainer


def _fixture():
    reference = pd.DataFrame({"年龄": [20, 30, 40, 50, 60, 70], "收入": [1, 2, 4, 7, 10, 14], "负债": [12, 10, 8, 5, 3, 1]})
    y = [1, 1, 1, 0, 0, 0]
    return LogisticRegression().fit(reference, y), reference


def test_generic_counterfactual_respects_immutable_direction_and_change_limit():
    model, reference = _fixture()
    subject = reference.iloc[[0]]
    counter = CounterfactualExplainer(
        model,
        reference,
        constraints={"年龄": {"mutable": False}, "收入": {"direction": "increase_only", "min": 0}},
        random_state=7,
    )
    result = counter.generate(subject, target_probability=0.35, max_changes=2, top_n=3)
    assert result["说明"].str.contains("非因果建议").all()
    assert not (result["特征"] == "年龄").any()
    income = result[result["特征"] == "收入"]
    assert (income["新值"].astype(float) >= income["原值"].astype(float)).all()
    assert (result["变更特征数"] <= 2).all()


def test_counterfactual_returns_structured_failure_when_target_is_unreachable():
    model, reference = _fixture()
    counter = CounterfactualExplainer(model, reference, constraints={name: {"mutable": False} for name in reference.columns})
    result = counter.generate(reference.iloc[[0]], target_probability=0.0)
    assert result.loc[0, "是否达标"] == "否"
    assert "不可变" in result.loc[0, "失败原因"]
