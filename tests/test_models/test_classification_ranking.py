"""分类排序指标的同分处理、方向和缺失样本口径回归测试。"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from hscredit.core.metrics import auc, gini, ks


@pytest.mark.parametrize(
    'target,score,expected',
    [
        ([0, 0, 1, 1], [1, 1, 1, 1], 0.0),
        ([0, 1, 0, 1, 0, 1], [1, 1, 2, 2, 3, 3], 0.0),
        ([0, 0, 1, 1], [1, 2, 2, 3], 0.5),
        ([0, 0, 1, 1], [1, 2, 3, 4], 1.0),
    ],
)
def test_ks_groups_ties_and_ignores_row_order_and_score_direction(target, score, expected):
    target, score = np.asarray(target), np.asarray(score)
    rng = np.random.RandomState(42)
    for _ in range(10):
        order = rng.permutation(len(target))
        for direction in (1, -1):
            assert ks(target[order], score[order] * direction) == pytest.approx(expected)
    assert ks(target, score) == pytest.approx(ks_2samp(score[target == 0], score[target == 1]).statistic)


@pytest.mark.parametrize('direction,expected', [('auto', 0.75), ('higher_risk', 0.25), ('higher_safe', 0.75)])
def test_auc_and_gini_respect_score_direction(direction, expected):
    target = np.array([0, 0, 1, 1])
    score = np.array([0.8, 0.2, 0.3, 0.1])
    assert auc(target, score, score_direction=direction) == pytest.approx(expected)
    assert gini(target, score, score_direction=direction) == pytest.approx(2 * expected - 1)
    assert auc(target, score) == pytest.approx(0.75)
    if direction != 'auto':
        oriented = score if direction == 'higher_risk' else -score
        assert auc(target, score, score_direction=direction) == pytest.approx(roc_auc_score(target, oriented))


@pytest.mark.parametrize('value', [0.5, 700.0])
def test_constant_scores_have_no_discrimination(value):
    target = [0, 0, 1, 1]
    score = np.full(4, value)
    assert ks(target, score) == 0.0
    assert auc(target, score) == 0.5
    assert gini(target, score) == 0.0


def test_metrics_share_missing_pair_filtering_and_explicit_positive_label():
    target = pd.Series(['好', '坏', '好', '坏', '好', None], index=[10, 20, 30, 40, 50, 60])
    score = pd.Series([0.1, 0.4, 0.4, 0.9, np.nan, 0.8], index=[60, 50, 40, 30, 20, 10])
    assert auc(target, score, pos_label='坏', score_direction='higher_risk') == pytest.approx(0.875)
    assert auc(target, score, pos_label='好', score_direction='higher_risk') == pytest.approx(0.125)
    assert ks(target, score, pos_label='坏') == pytest.approx(0.5)
    assert ks(target, score, pos_label='好') == pytest.approx(0.5)


@pytest.mark.parametrize('metric', [auc, ks])
@pytest.mark.parametrize(
    'target,score,message',
    [
        ([0, 1], [1], '长度'),
        ([0, 1], [[1], [2]], '一维'),
        ([0, 1], [np.nan, np.nan], '非缺失'),
        ([0, 1], [0.1, np.inf], '有限'),
        ([0, 1, 2], [0.1, 0.2, 0.3], '二分类'),
    ],
)
def test_ranking_metrics_reject_invalid_inputs(metric, target, score, message):
    with pytest.raises(ValueError, match=message):
        metric(target, score)


def test_auc_rejects_invalid_direction_and_keeps_single_class_ks_contract():
    with pytest.raises(ValueError, match='score_direction'):
        auc([0, 1], [0.1, 0.9], score_direction='invalid')
    with pytest.raises(ValueError, match='pos_label'):
        auc(['好', '坏'], [0.1, 0.9])
    with pytest.raises(ValueError, match='二分类'):
        auc([1, 1], [0.1, 0.9])
    assert ks([1, 1], [0.1, 0.9]) == 0.0
    assert ks([0, 0], [0.1, 0.9]) == 0.0


@pytest.mark.parametrize('score', [[0.8, 0.2, 0.3, 0.1], [0.1, 0.5, 0.5, 0.9], [0.5] * 4])
def test_all_auc_entry_points_share_direction_and_tie_handling(score):
    """报告、模型评估、调参、逐步筛选、Gini回调和ROC数据的AUC必须一致。"""
    from sklearn.metrics import auc as curve_area
    from hscredit.core.metrics import roc_curve
    from hscredit.core.models.base import _evaluate_binary_predictions
    from hscredit.core.models.losses import GiniMetric, KSMetric
    from hscredit.core.models.tuning.tuning import Metric, TuningObjective
    from hscredit.core.selectors import StepwiseSelector
    from hscredit.report.mining.manual_tree_extractor import _auc
    from hscredit.report.model_report import _binary_metric_worker
    from hscredit.core.eda.relationship import univariate_auc

    y, score = np.array([0, 0, 1, 1]), np.asarray(score)
    raw = roc_auc_score(y, score)
    expected = max(raw, 1 - raw)
    result = _evaluate_binary_predictions(y, score, score > 0.5, metrics=['auc', 'gini'])
    fpr, tpr, _ = roc_curve(y, score)
    values = [
        auc(y, score), result['AUC'], TuningObjective.auc(y, score), Metric('auc')(y, score),
        StepwiseSelector()._calculate_auc(y, score), _auc(y, score), _binary_metric_worker((y, score))[1],
        (GiniMetric()(y, score) + 1) / 2, curve_area(fpr, tpr),
    ]
    np.testing.assert_allclose(values, expected)
    assert result['Gini'] == pytest.approx(2 * expected - 1)
    assert GiniMetric(score_direction='higher_risk')(y, score) == pytest.approx(2 * raw - 1)
    assert KSMetric()(y, score) == ks(y, score)
    frame = pd.DataFrame({'特征': score, '目标': y})
    assert univariate_auc(frame, '特征', '目标')['AUC值'] == round(expected, 4)


@pytest.mark.parametrize('direction', ['auto', 'higher_risk', 'higher_safe'])
def test_weighted_auc_uses_same_filtered_pairs_and_roc_area(direction):
    from sklearn.metrics import auc as curve_area
    from hscredit.core.metrics import roc_curve

    y = np.array([0, 0, 1, 1, 1, np.nan, 0])
    score = np.array([0.8, 0.2, 0.3, 0.1, 0.99, 0.8, np.nan])
    weights = np.array([1, 4, 3, 2, 0, 5, 6])
    raw = roc_auc_score(y[:4], score[:4], sample_weight=weights[:4])
    expected = max(raw, 1 - raw) if direction == 'auto' else (raw if direction == 'higher_risk' else 1 - raw)
    value = auc(y, score, sample_weight=weights, score_direction=direction)
    assert value == pytest.approx(expected)
    assert value != pytest.approx(auc(y[:4], score[:4], score_direction=direction))
    fpr, tpr, thresholds = roc_curve(y, score, sample_weight=weights, score_direction=direction)
    assert curve_area(fpr, tpr) == value
    assert np.isinf(thresholds[0])
    oriented = -score[:4] if direction == 'higher_safe' or direction == 'auto' and raw < 0.5 else score[:4]
    for index, threshold in enumerate(thresholds):
        accepted = oriented >= threshold
        assert tpr[index] == pytest.approx(weights[:4][accepted & (y[:4] == 1)].sum() / weights[:4][y[:4] == 1].sum())
        assert fpr[index] == pytest.approx(weights[:4][accepted & (y[:4] == 0)].sum() / weights[:4][y[:4] == 0].sum())


@pytest.mark.parametrize('weights', [[1], [-1, 1, 1, 1], [1, np.inf, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
def test_weighted_auc_rejects_invalid_or_single_effective_class_weights(weights):
    with pytest.raises(ValueError):
        auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], sample_weight=weights)
