"""模型评估的正类、显式指标和 sample_weight 契约。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hscredit.core.models import BaseRiskModel
from hscredit.core.models.calibration import CalibratedModel, ProbabilityCalibrator


class _FixedRiskModel(BaseRiskModel):
    """使用固定概率的最小真实 BaseRiskModel 测试实现。"""

    def __init__(self, scores, classes=(0, 1)):
        super().__init__(validation_fraction=0)
        self.scores = np.asarray(scores, dtype=float)
        self.classes_ = np.asarray(classes)
        self._is_fitted = True

    def fit(self, X, y=None, sample_weight=None, eval_set=None, **fit_params):
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        positive = self.scores[: len(X)]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return np.where(self.scores[: len(X)] >= 0.5, self.classes_[1], self.classes_[0])

    def get_feature_importances(self, importance_type="gain"):
        return pd.Series(dtype=float)


def test_explicit_metrics_do_not_append_unrequested_monotonicity_keys():
    """请求 AUC 时返回字典不能暗中增加三个单调性字段。"""
    model = _FixedRiskModel([0.9, 0.8, 0.7, 0.6])

    result = model.evaluate(np.zeros((4, 1)), np.array([1, 0, 1, 0]), metrics=["auc"])

    assert result == {"AUC": pytest.approx(0.75)}


def test_supported_weight_metric_uses_sample_weight():
    """AUC 权重必须改变正负样本对的贡献。"""
    model = _FixedRiskModel([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 0])
    weights = np.array([1.0, 10.0, 1.0, 1.0])

    result = model.evaluate(np.zeros((4, 1)), y, sample_weight=weights, metrics=["auc"])

    assert result["AUC"] == pytest.approx(6 / 11)


def test_unsupported_weight_metrics_warn_once_and_return_unweighted_results():
    """KS 与 Lift 忽略权重时必须集中 warning，不能静默或逐项报警。"""
    model = _FixedRiskModel([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 1, 0])
    weights = np.array([1.0, 10.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="KS.*LIFT@10%") as records:
        result = model.evaluate(
            np.zeros((4, 1)),
            y,
            sample_weight=weights,
            metrics=["ks", "lift@10%"],
        )

    assert len(records) == 1
    assert set(result) == {"KS", "LIFT@10%"}


def test_explicit_positive_class_selects_matching_probability_column():
    """字符串标签评估必须把指定正类转换为二值标签并取对应概率列。"""
    model = _FixedRiskModel([0.9, 0.8, 0.7, 0.6], classes=("正常", "逾期"))
    y = np.array(["逾期", "正常", "逾期", "正常"])

    result = model.evaluate(
        np.zeros((4, 1)),
        y,
        metrics=["auc"],
        positive_class="逾期",
    )

    assert result == {"AUC": pytest.approx(0.75)}


def test_sample_weight_validation_rejects_mismatched_or_negative_values():
    """非法权重不能被部分指标接受、部分指标忽略。"""
    model = _FixedRiskModel([0.9, 0.8, 0.7, 0.6])
    X = np.zeros((4, 1))
    y = np.array([1, 0, 1, 0])

    with pytest.raises(ValueError, match="sample_weight"):
        model.evaluate(X, y, sample_weight=np.ones(3), metrics=["auc"])
    with pytest.raises(ValueError, match="sample_weight"):
        model.evaluate(X, y, sample_weight=np.array([1.0, -1.0, 1.0, 1.0]), metrics=["auc"])


def test_calibrated_model_uses_the_same_metric_and_weight_warning_contract():
    """校准包装模型不能维护另一套 evaluate 参数和返回键。"""
    X, y = make_classification(n_samples=100, n_features=4, random_state=51)
    base_model = LogisticRegression(max_iter=300).fit(X[:60], y[:60])
    calibrator = ProbabilityCalibrator(model=base_model, calib_ratio=None).fit(X[60:80], y[60:80])
    model = CalibratedModel(base_model, calibrator)
    weights = np.linspace(1.0, 2.0, 20)

    with pytest.warns(UserWarning, match="KS") as records:
        result = model.evaluate(
            X[80:],
            y[80:],
            sample_weight=weights,
            metrics=["auc", "ks", "brier"],
        )

    assert len(records) == 1
    assert set(result) == {"AUC", "KS", "Brier"}
