import numpy as np
import pytest
from sklearn.datasets import make_classification

from hscredit.core.models import LogisticRegression, ProbabilityScoreCard
from hscredit.exceptions import NotFittedError


def test_probability_scorecard_unfitted_predict_score_raises_not_fitted():
    card = ProbabilityScoreCard(model=None)

    with pytest.raises(NotFittedError, match="尚未拟合"):
        card.predict_score(proba=[0.1, 0.2])


def test_probability_scorecard_trains_fresh_hscredit_logistic_regression():
    """构造期评分配置不能让未训练模型被误判为已拟合。"""
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression(max_iter=500)

    card = ProbabilityScoreCard(model=model, method='standard').fit(X, y)

    assert card.model_ is model
    assert hasattr(model, 'coef_')
    assert np.all(np.isfinite(card.predict(X)))


def test_probability_scorecard_proba_only_scoring_report_and_roundtrip(tmp_path):
    proba = np.array([0.01, 0.03, 0.08, 0.15, 0.25, 0.4])
    y = np.array([0, 0, 0, 1, 1, 1])

    card = ProbabilityScoreCard(
        model=None,
        method='standard',
        base_odds=0.05,
        base_score=600,
        pdo=20,
        decimal=2,
    )
    card.fit(proba=proba)

    scores = card.predict_score(proba=proba)
    assert len(scores) == len(proba)
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[-1]

    report = card.report(scores=scores, y=y, n_bins=3)
    assert {'评分区间', '样本数', '坏样本率', 'KS'}.issubset(report.columns)

    path = tmp_path / 'probability_scorecard.pkl'
    card.save(str(path))
    loaded = ProbabilityScoreCard.load(str(path))
    np.testing.assert_allclose(loaded.predict_score(proba=proba), scores)


def test_probability_scorecard_report_keeps_single_bin_for_constant_scores():
    card = ProbabilityScoreCard(model=None, decimal=2)
    card.fit(proba=[0.1, 0.1, 0.1])

    report = card.report(scores=np.array([600.0, 600.0, 600.0]), y=np.array([0, 1, 0]), n_bins=5)

    assert len(report) == 1
    assert report.loc[0, '评分区间'] == '[600.00, 600.00]'
    assert report.loc[0, '样本数'] == 3
    assert report.loc[0, '坏样本率'] == '33.33%'
    assert report.loc[0, 'KS'] == 0.0
