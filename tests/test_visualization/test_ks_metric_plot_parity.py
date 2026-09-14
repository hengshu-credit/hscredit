"""KS/ROC 曲线与公开指标在同分、方向、缺失样本下的一致性测试。"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import auc as curve_auc

from hscredit.core.metrics import auc, ks
from hscredit.core.viz import ks_plot, score_ks_plot


@pytest.mark.parametrize('direction', ['auto', 'higher_risk', 'higher_safe'])
@pytest.mark.parametrize(
    'target,score',
    [
        ([0, 0, 1, 1], [700, 700, 700, 700]),
        ([0, 1, 0, 1, 0, 1], [1, 1, 2, 2, 3, 3]),
        ([0, 0, 1, 1], [3, 2, 2, 1]),
    ],
)
def test_ks_and_roc_curves_match_metrics_and_reference_with_ties(direction, target, score):
    target, score = np.asarray(target), np.asarray(score)
    fig = ks_plot(score, target, score_direction=direction)
    try:
        ks_ax, roc_ax = fig.axes
        ks_line, good_line, bad_line = ks_ax.lines[:3]
        coverage = np.asarray(ks_line.get_xdata())
        expected_ks = ks_2samp(score[target == 0], score[target == 1]).statistic
        assert max(ks_line.get_ydata()) == pytest.approx(expected_ks)
        assert max(ks_line.get_ydata()) == pytest.approx(ks(target, score))
        np.testing.assert_allclose(
            ks_line.get_ydata(), np.abs(np.asarray(bad_line.get_ydata()) - good_line.get_ydata())
        )
        assert coverage[0] == 0.0
        assert coverage[-1] == 1.0
        assert len(coverage) == len(np.unique(score)) + 1
        assert np.all(np.diff(coverage) > 0)
        np.testing.assert_allclose(
            coverage, (np.asarray(bad_line.get_ydata()) * target.sum()
                       + np.asarray(good_line.get_ydata()) * (len(target) - target.sum())) / len(target)
        )
        assert any(f'KS: {round(expected_ks, 4)}' in text.get_text() for text in ks_ax.texts)
        expected_auc = auc(target, score, score_direction=direction)
        roc_line = roc_ax.lines[0]
        assert curve_auc(roc_line.get_xdata(), roc_line.get_ydata()) == pytest.approx(expected_auc)
        assert any(f'AUC: {expected_auc:.4f}' in text.get_text() for text in roc_ax.texts)
    finally:
        plt.close(fig)


def test_plot_and_metrics_use_same_missing_pairs_and_positive_label():
    target = np.array(['好', '坏', '好', '坏', '好', None])
    score = np.array([0.1, 0.4, 0.4, 0.9, np.nan, 0.8])
    fig = ks_plot(score, target, pos_label='坏', score_direction='higher_risk')
    try:
        assert max(fig.axes[0].lines[0].get_ydata()) == ks(target, score, pos_label='坏')
        assert any('AUC: 0.8750' in text.get_text() for text in fig.axes[1].texts)
    finally:
        plt.close(fig)


def test_multidataset_score_ks_plot_uses_same_thresholds_and_coverage():
    target = np.array([0, 0, 1, 1])
    datasets = {'常量': (target, np.ones(4)), '同分': (target, np.array([1, 2, 2, 3]))}
    fig = score_ks_plot(datasets=datasets)
    try:
        for index, (_, (y, score)) in enumerate(datasets.items()):
            bad_line, good_line = fig.axes[0].lines[index * 2:index * 2 + 2]
            difference = np.abs(np.asarray(bad_line.get_ydata()) - good_line.get_ydata())
            assert max(difference) == ks(y, score)
            assert f'KS={ks(y, score):.4f}' in bad_line.get_label()
            assert len(bad_line.get_xdata()) == len(np.unique(score)) + 1
            assert bad_line.get_xdata()[0] == 0.0
            assert bad_line.get_xdata()[-1] == 1.0
    finally:
        plt.close(fig)


@pytest.mark.parametrize('direction', ['auto', 'higher_risk', 'higher_safe'])
@pytest.mark.parametrize('weighted', [False, True])
def test_both_roc_plot_entries_share_exact_auc_curve_and_formatting(direction, weighted):
    from hscredit.core.metrics import roc_curve
    from hscredit.core.viz import roc_plot

    target = np.array(['好', '好', '坏', '坏', '好', None])
    score = np.array([0.8, 0.2, 0.3, 0.1, np.nan, 0.8])
    weights = np.array([1, 4, 3, 2, 5, 6]) if weighted else None
    options = dict(pos_label='坏', score_direction=direction, sample_weight=weights)
    expected = auc(target, score, **options)
    fpr, tpr, _ = roc_curve(target, score, **options)
    first = ks_plot(score, target, **options)
    second = roc_plot(target, score, show_diagonal=False, **options)
    try:
        for line in (first.axes[1].lines[0], second.axes[0].lines[0]):
            np.testing.assert_array_equal(line.get_xdata(), fpr)
            np.testing.assert_array_equal(line.get_ydata(), tpr)
            assert curve_auc(line.get_xdata(), line.get_ydata()) == expected
        assert first.axes[1].texts[0].get_text() == f'AUC: {expected:.4f}'
        assert f'AUC = {expected:.4f}' in second.axes[0].lines[0].get_label()
        assert max(first.axes[0].lines[0].get_ydata()) == pytest.approx(
            ks(target, score, pos_label='坏', sample_weight=weights)
        )
    finally:
        plt.close(first)
        plt.close(second)
