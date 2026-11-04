"""受约束反事实解释测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from hscredit import ValidationError
from hscredit.core.models.explainability import CounterfactualExplainer


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


@pytest.mark.parametrize("target", [-0.1, 1.5, float("nan"), float("inf")])
def test_counterfactual_rejects_probability_outside_unit_interval(target):
    """非法概率目标不能被当作已经达标。"""
    model, reference = _fixture()
    counter = CounterfactualExplainer(model, reference)

    with pytest.raises(ValidationError, match="target_probability"):
        counter.generate(reference.iloc[[0]], target_probability=target)


def test_counterfactual_rejects_fractional_search_limits():
    """搜索上限必须在进入 range 前校验为正整数。"""
    model, reference = _fixture()
    counter = CounterfactualExplainer(model, reference)

    with pytest.raises(ValidationError, match="max_changes"):
        counter.generate(reference.iloc[[0]], target_probability=0.2, max_changes=1.5)


@pytest.mark.parametrize("config", [{"weight": -1}, {"mutable": "false"}])
def test_counterfactual_rejects_invalid_constraint_types(config):
    """负成本或非布尔可变性会破坏最小成本与约束语义。"""
    model, reference = _fixture()

    with pytest.raises(ValidationError, match="weight|mutable"):
        CounterfactualExplainer(model, reference, constraints={"收入": config})


def test_counterfactual_batches_candidate_predictions():
    """束搜索扩展候选应批量预测，避免每个候选单独调用模型。"""
    fitted, reference = _fixture()

    class CountingModel:
        classes_ = fitted.classes_

        def __init__(self):
            self.batch_sizes = []

        def predict_proba(self, X):
            self.batch_sizes.append(len(X))
            return fitted.predict_proba(X)

    model = CountingModel()
    counter = CounterfactualExplainer(model, reference)

    counter.generate(reference.iloc[[0]], target_probability=0.0, max_changes=2, beam_width=20)

    assert any(size > 1 for size in model.batch_sizes)


def test_counterfactual_probability_uses_explicit_positive_class_column():
    """非0/1标签模型必须按显式正类取概率列，不能默认取最后一列。"""

    class FixedClassifier:
        classes_ = ["坏", "好"]

        def predict_proba(self, X):
            return [[0.9, 0.1] for _ in range(len(X))]

    reference = pd.DataFrame({"收入": [1.0, 2.0, 3.0]})
    counter = CounterfactualExplainer(
        FixedClassifier(),
        reference,
        positive_class="坏",
        constraints={"收入": {"mutable": False}},
    )

    result = counter.generate(reference.iloc[[0]], target_probability=0.5)

    assert result.loc[0, "是否达标"] == "否"
    assert result.loc[0, "预测前值"] == pytest.approx(0.9)


def test_categorical_counterfactual_direction_is_replacement():
    """类别值不能按字符串大小错误标记为增加或减少。"""

    class CategoryClassifier:
        classes_ = [0, 1]

        def predict_proba(self, X):
            positive = X["等级"].map({"高": 0.9, "低": 0.1}).to_numpy()
            return np.column_stack([1 - positive, positive])

    reference = pd.DataFrame({"等级": ["高", "低"]})
    counter = CounterfactualExplainer(CategoryClassifier(), reference)

    result = counter.generate(reference.iloc[[0]], target_probability=0.2)

    assert result.loc[0, "变化方向"] == "替换"
