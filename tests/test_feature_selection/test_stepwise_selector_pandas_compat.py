"""逐步回归筛选器的 pandas 兼容性测试."""

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
    assert selector.scores_["特征B"] == 0.0
