"""阈值权衡表不能静默截断输入或将方向拼写错误解释为反方向。"""

import numpy as np
import pytest

from hscredit.core.eda.strategy import approval_badrate_tradeoff


@pytest.mark.parametrize("y,score", [([0, 1], [10, 20, 30]), ([0, 1, 0], [10, 20]), ([[0], [1]], [10, 20])])
def test_tradeoff_rejects_mismatched_or_multidimensional_inputs(y, score):
    with pytest.raises(ValueError, match="一维|长度|等长"):
        approval_badrate_tradeoff(y, score, n_points=2)


@pytest.mark.parametrize("direction", ["HIGH", "typo", None])
def test_tradeoff_rejects_unrecognized_direction(direction):
    with pytest.raises(ValueError, match="score_low_risk"):
        approval_badrate_tradeoff([0, 1], [10, 20], score_low_risk=direction)


@pytest.mark.parametrize("points", [0, -1, True, 2.5])
def test_tradeoff_requires_positive_integer_points(points):
    with pytest.raises(ValueError, match="n_points"):
        approval_badrate_tradeoff([0, 1], [10, 20], n_points=points)


@pytest.mark.parametrize("score", [[], [np.nan, np.nan], [np.inf, 20]])
def test_tradeoff_rejects_empty_or_nonfinite_scores(score):
    with pytest.raises(ValueError, match="有效|有限|不能为空"):
        approval_badrate_tradeoff([0, 1] if score else [], score)


@pytest.mark.parametrize("direction", ["high", "low"])
def test_tradeoff_valid_thresholds_match_raw_decisions(direction):
    score = np.arange(10.0)
    y = np.arange(10) % 2
    table = approval_badrate_tradeoff(y, score, n_points=2, score_low_risk=direction)
    for _, row in table.iterrows():
        approved = score >= row["评分阈值"] if direction == "high" else score <= row["评分阈值"]
        assert row["通过人数"] == approved.sum()
        assert row["通过人群坏样本数"] == y[approved].sum()
        assert row["通过人数"] + row["拒绝人数"] == len(y)
