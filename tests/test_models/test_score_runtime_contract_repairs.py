"""评分转换、漂移校准和规则模型边界回归测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.models import RuleSet, RulesClassifier
from hscredit.core.models.scorecard.score_drift import (
    BinningRecalibrator,
    LinearDriftCalibrator,
    QuantileAligner,
)
from hscredit.core.models.scorecard.score_transformer import (
    BoxCoxScoreTransformer,
    LinearScoreTransformer,
    QuantileScoreTransformer,
    StandardScoreTransformer,
)
from hscredit.core.rules import Rule


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: StandardScoreTransformer(rate=1), "rate"),
        (lambda: StandardScoreTransformer(base_odds=0), "base_odds"),
        (lambda: QuantileScoreTransformer(n_quantiles=1), "n_quantiles"),
        (lambda: StandardScoreTransformer(direction="sideways"), "direction"),
        (lambda: StandardScoreTransformer(lower=700, upper=300), "lower"),
    ],
)
def test_score_transformers_reject_invalid_parameters(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_standard_score_transformer_keeps_probability_boundaries_finite():
    transformer = StandardScoreTransformer().fit(np.array([0.2, 0.8]))
    scores = transformer.transform(np.array([0.0, 1.0]))
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "transformer",
    [
        StandardScoreTransformer(),
        LinearScoreTransformer(),
        QuantileScoreTransformer(),
        BoxCoxScoreTransformer(lmbda=0),
    ],
)
def test_score_transformers_reject_invalid_probabilities(transformer):
    with pytest.raises(ValueError, match="概率"):
        transformer.fit(np.array([0.2, np.nan, 0.8]))
    with pytest.raises(ValueError, match="概率"):
        transformer.fit(np.array([-0.1, 0.8]))


def test_standard_score_transformer_rejects_empty_probabilities():
    with pytest.raises(ValueError, match="不能为空"):
        StandardScoreTransformer().fit(np.array([]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shift": 0}, "shift"),
        ({"shift": np.inf}, "shift"),
        ({"lmbda": np.nan}, "lmbda"),
    ],
)
def test_boxcox_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BoxCoxScoreTransformer(**kwargs)


def test_boxcox_keeps_probability_boundaries_finite():
    transformer = BoxCoxScoreTransformer(lmbda=0).fit(np.array([0.2, 0.8]))
    assert np.isfinite(transformer.transform(np.array([0.0, 1.0]))).all()


class _ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        positive = np.full(len(X), 0.5)
        return np.column_stack([1.0 - positive, positive])


class _FeatureProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        positive = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - positive, positive])


def test_binning_recalibrator_handles_constant_probabilities_in_logit_space():
    X = np.zeros((10, 2))
    reference_y = np.array([1, 1] + [0] * 8)
    current_y = np.array([1, 0] * 5)
    calibrator = BinningRecalibrator(n_bins=10).fit(
        _ConstantProbabilityModel(),
        X,
        current_y,
        X_reference=X,
        y_reference=reference_y,
    )

    calibrated = calibrator.predict_score(scores=np.array([0.5]))

    assert calibrator.n_bins_actual_ == 1
    assert calibrated[0] == pytest.approx(0.2, abs=1e-6)


def test_binning_recalibrator_assigns_shifted_current_scores_to_edge_bins():
    calibrator = BinningRecalibrator(n_bins=2).fit(
        _FeatureProbabilityModel(),
        np.array([[0.8], [0.9]]),
        np.array([1, 1]),
        X_reference=np.array([[0.2], [0.4]]),
        y_reference=np.array([0, 0]),
    )

    assert calibrator.cur_bad_rates_[-1] == pytest.approx(1.0)


def test_quantile_aligner_handles_tied_current_probabilities():
    X = np.zeros((10, 2))
    calibrator = QuantileAligner(n_quantiles=10).fit(_ConstantProbabilityModel(), X, X_reference=X)
    assert np.isfinite(calibrator.predict_score(scores=np.array([0.4, 0.5, 0.6]))).all()


def test_drift_detection_preserves_explicit_zero_threshold():
    X = np.zeros((10, 2))
    calibrator = LinearDriftCalibrator().fit(_ConstantProbabilityModel(), X, X_reference=X)
    result = calibrator.detect_drift(X, X, metric="psi", threshold=0.0)
    assert result["threshold"] == 0.0


def test_rules_classifier_rejects_negative_weights():
    model = RulesClassifier(rules=[Rule("age >= 18")], weights=[-1.0])
    with pytest.raises(ValueError, match="权重.*非负"):
        model.fit(pd.DataFrame({"age": [17, 20]}), np.array([0, 1]))


def test_rules_classifier_rejects_all_zero_weights():
    model = RulesClassifier(rules=[Rule("age >= 18")], weights=[0.0])
    with pytest.raises(ValueError, match="至少一个.*正数"):
        model.fit(pd.DataFrame({"age": [17, 20]}), np.array([0, 1]))


def test_rules_classifier_keeps_binary_classes_for_single_class_fit_data():
    model = RulesClassifier(rules=[Rule("age >= 18")]).fit(pd.DataFrame({"age": [17, 20]}), np.array([0, 0]))
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_rules_classifier_ignores_extra_fields_but_rejects_missing_training_fields():
    model = RulesClassifier(rules=[Rule("age >= 18")]).fit(
        pd.DataFrame({"age": [17, 20], "income": [1000, 2000]}),
        np.array([0, 1]),
    )

    result = model.predict(pd.DataFrame({"extra": [99], "income": [3000], "age": [30]}))
    assert result.tolist() == [1]

    with pytest.raises(ValueError, match="缺少训练字段.*income"):
        model.predict(pd.DataFrame({"age": [30], "extra": [99]}))


def test_reason_output_mode_always_returns_reasons():
    model = RulesClassifier(rules=[Rule("age >= 18")], output_mode="reason")
    result, reasons = model.predict(pd.DataFrame({"age": [17, 20]}))
    assert result.tolist() == [0, 1]
    assert len(reasons) == 2


def test_rules_classifier_rejects_non_binary_training_labels():
    model = RulesClassifier(rules=[Rule("age >= 18")])
    with pytest.raises(ValueError, match="0/1"):
        model.fit(pd.DataFrame({"age": [17, 20, 30]}), np.array([0, 1, 2]))


def test_rules_classifier_rejects_invalid_rule_objects():
    model = RulesClassifier(rules=[Rule("age >= 18"), "not-a-rule"])
    with pytest.raises(ValueError, match="Rule或RuleSet"):
        model.fit(pd.DataFrame({"age": [17, 20]}), np.array([0, 1]))


def test_rules_classifier_rejects_invalid_nested_rule_objects():
    nested = RuleSet(name="嵌套规则", rules=[Rule("age >= 18"), "not-a-rule"])
    model = RulesClassifier(rules=[nested])
    with pytest.raises(ValueError, match="Rule或RuleSet"):
        model.fit(pd.DataFrame({"age": [17, 20]}), np.array([0, 1]))
