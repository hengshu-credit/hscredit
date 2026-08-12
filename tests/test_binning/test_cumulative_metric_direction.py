"""分箱累计指标风险方向回归测试。"""

import numpy as np

from hscredit.core.metrics import add_margins, compute_bin_stats


def _binary_samples(bin_counts):
    bins = []
    targets = []
    for bin_id, good_count, bad_count in bin_counts:
        bins.extend([bin_id] * (good_count + bad_count))
        targets.extend([0] * good_count + [1] * bad_count)
    return np.asarray(bins), np.asarray(targets)


def test_cumulative_metrics_start_from_the_higher_risk_normal_bin_end():
    """风险随箱值上升时仍从高风险端累计，而不是从第一行累计。"""
    bins, y = _binary_samples(
        [
            (0, 3, 1),
            (1, 2, 2),
            (2, 1, 3),
            (-1, 3, 1),
        ]
    )

    table = compute_bin_stats(
        bins,
        y,
        bin_labels=["missing", "低风险", "中风险", "高风险"],
        round_digits=False,
    )

    assert table["分箱"].tolist() == [0, 1, 2, -1]
    np.testing.assert_array_equal(table["累积好样本数"], [6, 3, 1, 9])
    np.testing.assert_array_equal(table["累积坏样本数"], [6, 5, 3, 7])
    np.testing.assert_allclose(table["累积LIFT值"], [8 / 7, 10 / 7, 12 / 7, 1])
    np.testing.assert_allclose(table["累积坏账改善"], [3 / 7, 3 / 7, 5 / 21, 1])
    np.testing.assert_allclose(table["累计风险拒绝比"], [4 / 7, 6 / 7, 20 / 21, 1])
    np.testing.assert_allclose(table["分档KS值"], [4 / 21, 8 / 21, 20 / 63, 0], atol=1e-10)


def test_cumulative_metrics_keep_forward_order_when_the_first_normal_bin_is_riskier():
    """风险随箱值下降时保持现有正向累计，避免无条件反转评分表。"""
    bins, y = _binary_samples(
        [
            (0, 1, 3),
            (1, 2, 2),
            (2, 3, 1),
            (-1, 3, 1),
        ]
    )

    table = compute_bin_stats(bins, y, round_digits=False)

    np.testing.assert_array_equal(table["累积好样本数"], [1, 3, 6, 9])
    np.testing.assert_array_equal(table["累积坏样本数"], [3, 5, 6, 7])
    np.testing.assert_allclose(table["累积LIFT值"], [12 / 7, 10 / 7, 8 / 7, 1])


def test_amount_weighted_cumulative_metrics_use_the_same_risk_direction():
    """金额口径与样本口径使用同一风险累计方向。"""
    bins, y = _binary_samples(
        [
            (0, 3, 1),
            (1, 2, 2),
            (2, 1, 3),
            (-1, 3, 1),
        ]
    )

    table = compute_bin_stats(
        bins,
        y,
        target_type="amount_weighted",
        amount=np.ones(len(y)),
        round_digits=False,
    )

    np.testing.assert_allclose(table["累积好样本数"], [6, 3, 1, 9])
    np.testing.assert_allclose(table["累积坏样本数"], [6, 5, 3, 7])
    np.testing.assert_allclose(table["累积LIFT值"], [8 / 7, 10 / 7, 12 / 7, 1], atol=1e-4)
    np.testing.assert_allclose(table["累计风险拒绝比"], [0.75, 0.5, 0.25, 1], atol=1e-4)


def test_add_margins_does_not_sum_prefix_counts_or_copy_a_bin_ratio():
    """合计行累计计数取总体值，拒绝指标不沿用首箱。"""
    bins, y = _binary_samples(
        [
            (0, 3, 1),
            (1, 2, 2),
            (2, 1, 3),
            (-1, 3, 1),
        ]
    )
    table = compute_bin_stats(
        bins,
        y,
        bin_labels=["missing", "低风险", "中风险", "高风险"],
        round_digits=False,
    )

    total = add_margins(table).iloc[-1]

    assert total["分箱标签"] == "合计"
    assert total["样本总数"] == 16
    assert total["好样本数"] == 9
    assert total["坏样本数"] == 7
    assert total["累积好样本数"] == 9
    assert total["累积坏样本数"] == 7
    assert total["LIFT值"] == 1
    assert total["坏账改善"] == 0
    assert total["风险拒绝比"] == 0
    assert total["累积LIFT值"] == table.iloc[-1]["累积LIFT值"]
    assert total["累积坏账改善"] == table.iloc[-1]["累积坏账改善"]
    assert total["累计风险拒绝比"] == table.iloc[-1]["累计风险拒绝比"]
