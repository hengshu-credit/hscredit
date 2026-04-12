import numpy as np
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import LogisticRegression, ScoreCard
from hscredit.utils.datasets import germancredit


def _prepare_woe_data():
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    binner = OptimalBinning(method='target_bad_rate', max_n_bins=5)
    binner.fit(X_train, y_train)
    X_train_woe = binner.transform(X_train, metric='woe')

    return binner, X_train, X_test, X_train_woe, y_train, y_test


def test_logistic_regression_auto_normalizes_woe_coefficients():
    _, _, _, X_train_woe, y_train, _ = _prepare_woe_data()

    baseline_model = LogisticRegression(max_iter=1000, positive_woe_coef=False)
    baseline_model.fit(X_train_woe.copy(), y_train)
    assert (baseline_model.coef_[0] < 0).any()

    normalized_model = LogisticRegression(max_iter=1000)
    normalized_model.fit(X_train_woe.copy(), y_train)

    assert X_train_woe.attrs.get('hscredit_encoding') == 'woe'
    assert np.all(normalized_model.coef_[0] >= 0)
    assert hasattr(normalized_model, 'woe_coef_signs_')

    baseline_proba = baseline_model.predict_proba(X_train_woe)[:, 1]
    normalized_proba = normalized_model.predict_proba(X_train_woe)[:, 1]
    np.testing.assert_allclose(normalized_proba, baseline_proba, atol=1e-12)


def test_scorecard_positive_woe_coefficients_keep_scores_stable():
    binner, _, X_test, X_train_woe, y_train, _ = _prepare_woe_data()

    legacy_scorecard = ScoreCard(
        binner=binner,
        pdo=60,
        rate=2,
        base_odds=35,
        base_score=750,
        lr_kwargs={'positive_woe_coef': False, 'max_iter': 1000},
    )
    legacy_scorecard.fit(X_train_woe.copy(), y_train, input_type='woe')

    normalized_scorecard = ScoreCard(
        binner=binner,
        pdo=60,
        rate=2,
        base_odds=35,
        base_score=750,
    )
    normalized_scorecard.fit(X_train_woe.copy(), y_train, input_type='woe')

    assert np.all(normalized_scorecard.lr_model_.coef_[0] >= 0)

    reference_scores = legacy_scorecard.predict(X_test, input_type='raw')
    normalized_scores = normalized_scorecard.predict(X_test, input_type='raw')
    np.testing.assert_allclose(normalized_scores, reference_scores, atol=1e-10)

    reference_proba = legacy_scorecard.predict_proba(X_test)
    normalized_proba = normalized_scorecard.predict_proba(X_test)
    np.testing.assert_allclose(normalized_proba, reference_proba, atol=1e-12)