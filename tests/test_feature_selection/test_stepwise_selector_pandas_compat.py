"""逐步回归筛选器的 pandas 兼容性测试."""

import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hscredit.core.selectors import StepwiseSelector


def test_forward_step_supports_string_indexed_pvalues(monkeypatch):
    """前向选择应按位置读取带字符串索引的 p 值."""
    selector = StepwiseSelector(direction="forward", criterion="aic")
    selector.history_ = []

    result = {
        "criterion": 10.0,
        "result": SimpleNamespace(),
        "p_values": pd.Series([0.2, 0.01], index=["const", "特征A"]),
        "params": None,
    }
    monkeypatch.setattr(selector, "_fit_model", lambda X, y, features: result)

    improved, selected, remaining, score = selector._forward_step(
        pd.DataFrame({"特征A": [0.0, 1.0]}),
        pd.Series([0, 1]),
        selected=[],
        remaining=["特征A"],
        best_score=np.inf,
        max_features=None,
    )

    assert improved is True
    assert selected == ["特征A"]
    assert remaining == []
    assert score == 10.0


def test_calculate_scores_supports_string_indexed_pvalues(monkeypatch):
    """特征得分应按位置读取带字符串索引的 p 值."""
    selector = StepwiseSelector()
    result = {
        "criterion": 10.0,
        "result": SimpleNamespace(),
        "p_values": pd.Series([0.2, 0.01], index=["const", "特征A"]),
        "params": None,
    }
    monkeypatch.setattr(selector, "_fit_model", lambda X, y, features: result)

    selector._calculate_scores(
        pd.DataFrame({"特征A": [0.0, 1.0], "特征B": [1.0, 0.0]}),
        pd.Series([0, 1]),
        selected=["特征A"],
    )

    assert selector.scores_["特征A"] == 0.99
    assert np.isnan(selector.scores_["特征B"])


def test_logit_perfect_separation_warning_is_treated_as_failed_candidate(monkeypatch):
    """完全分离候选应静默失败，不能把不可识别参数用于逐步选择."""
    stepwise_module = importlib.import_module("hscredit.core.selectors.stepwise_selector")

    class PerfectSeparationWarning(UserWarning):
        pass

    class FakeLogit:
        def __init__(self, y, X):
            self.y = y
            self.X = X

        def fit(self, **kwargs):
            warnings.warn(
                "Perfect separation or prediction detected, parameter may not be identified",
                PerfectSeparationWarning,
            )
            raise AssertionError("warning should be promoted to an exception")

    class FakeStatsmodels:
        Logit = FakeLogit

        @staticmethod
        def add_constant(X, has_constant="add"):
            return np.column_stack([np.ones(len(X)), X])

    monkeypatch.setattr(stepwise_module, "sm", FakeStatsmodels)
    selector = StepwiseSelector(estimator="logit", criterion="aic")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = selector._fit_model(
            pd.DataFrame({"特征A": [0.0, 1.0]}),
            pd.Series([0, 1]),
            ["特征A"],
        )

    assert result["result"] is None
    assert "PerfectSeparationWarning" in result["failure_reason"]
    assert caught == []
