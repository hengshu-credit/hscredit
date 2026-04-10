import numpy as np
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard
from hscredit.utils.datasets import germancredit


def _train_scorecard(direction: str = 'descending'):
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    binner = OptimalBinning(method='target_bad_rate', max_n_bins=5)
    binner.fit(X_train, y_train)

    scorecard = ScoreCard(
        pdo=60,
        rate=2,
        base_odds=35,
        base_score=750,
        direction=direction,
        binner=binner,
    )
    scorecard.fit(binner.transform(X_train, metric='woe'), y_train, input_type='woe')
    return scorecard, binner, X_test


def test_target_bad_rate_preserves_categorical_labels_for_scorecard_export():
    scorecard, binner, _ = _train_scorecard()

    feature = 'status_of_existing_checking_account'

    assert feature in binner._cat_bins_
    assert all(isinstance(group, list) for group in binner._cat_bins_[feature])
    assert all(
        not str(label).startswith(('(', '['))
        for label in scorecard.rules_[feature]['bin_labels']
    )


def test_scorecard_export_load_with_meta_predict_consistency(tmp_path):
    scorecard, binner, X_test = _train_scorecard(direction='ascending')
    sample = X_test.iloc[:50].copy()
    reference = scorecard.predict(sample, input_type='raw')

    json_path = tmp_path / 'scorecard.json'
    scorecard.export(to_json=str(json_path), include_meta=True)

    loaded_scorecard = ScoreCard(binner=binner)
    loaded_scorecard.load(str(json_path))
    loaded_scores = loaded_scorecard.predict(sample, input_type='raw')

    assert np.max(np.abs(reference - loaded_scores)) < 0.05


def test_scorecard_python_deployment_code_matches_predict():
    scorecard, _, X_test = _train_scorecard(direction='ascending')
    sample = X_test.iloc[:50].copy()
    reference = scorecard.predict(sample, input_type='raw')

    namespace = {}
    exec(scorecard.export_deployment_code(language='python', decimal=12), namespace)
    deployed_scores = sample.apply(lambda row: namespace['calculate_score'](row.to_dict()), axis=1).to_numpy()

    np.testing.assert_allclose(reference, deployed_scores, atol=1e-9)