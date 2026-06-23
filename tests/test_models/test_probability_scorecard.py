import numpy as np
import pytest

from hscredit.core.models import ProbabilityScoreCard
from hscredit.exceptions import NotFittedError


def test_probability_scorecard_unfitted_predict_score_raises_not_fitted():
    card = ProbabilityScoreCard(model=None)

    with pytest.raises(NotFittedError, match="尚未拟合"):
        card.predict_score(proba=[0.1, 0.2])


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
