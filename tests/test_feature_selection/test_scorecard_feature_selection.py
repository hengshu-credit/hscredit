import numpy as np
import pandas as pd

from hscredit.core.selectors import ScorecardFeatureSelection


def _build_dataset() -> pd.DataFrame:
    size = 20
    target = [0, 0, 1, 1] * 5
    useful = [0, 2, 1, 3] * 5
    null_high = [np.nan] * 19 + [1.0]
    iv_low = [0, 1] * 10
    corr_keep = target.copy()
    corr_drop = target.copy()
    identical_high = [9] * 19 + [8]
    keep_me = [7] * 19 + [6]
    drop_me = [1, 3, 0, 2] * 5

    return pd.DataFrame({
        'useful': useful,
        'null_high': null_high,
        'iv_low': iv_low,
        'corr_keep': corr_keep,
        'corr_drop': corr_drop,
        'identical_high': identical_high,
        'keep_me': keep_me,
        'drop_me': drop_me,
        'target': target,
    })


def test_scorecard_feature_selection_dual_api_with_force_keep_and_drop():
    df = _build_dataset()

    selector = ScorecardFeatureSelection(
        null_threshold=0.95,
        iv_threshold=0.02,
        corr_threshold=0.7,
        mode_threshold=0.95,
        corr_weights={'corr_keep': 1.0, 'corr_drop': 0.1},
        include=['keep_me'],
        force_drop=['drop_me'],
        target='target',
    )

    selector.fit(df)

    assert 'useful' in selector.selected_features_
    assert 'corr_keep' in selector.selected_features_
    assert 'keep_me' in selector.selected_features_
    assert 'null_high' not in selector.selected_features_
    assert 'iv_low' not in selector.selected_features_
    assert 'corr_drop' not in selector.selected_features_
    assert 'identical_high' not in selector.selected_features_
    assert 'drop_me' not in selector.selected_features_
    assert 'target' in selector.select_columns

    transformed = selector.transform(df)
    assert list(transformed.columns) == ['useful', 'corr_keep', 'keep_me', 'target']

    drop_reasons = selector.dropped.set_index('variable')['rm_reason'].to_dict()
    assert drop_reasons['null_high'] == 'empty'
    assert drop_reasons['iv_low'] == 'iv'
    assert drop_reasons['corr_drop'] == 'corr'
    assert drop_reasons['drop_me'] == 'force_drop'


def test_scorecard_feature_selection_sklearn_api_without_target_passthrough():
    df = _build_dataset()
    X = df.drop(columns=['target'])
    y = df['target']

    selector = ScorecardFeatureSelection(
        null_threshold=0.95,
        iv_threshold=0.02,
        corr_threshold=0.7,
        mode_threshold=0.95,
        corr_weights={'corr_keep': 1.0, 'corr_drop': 0.1},
        target='target',
        target_rm=True,
    )

    transformed = selector.fit_transform(X, y)

    assert 'target' not in transformed.columns
    assert selector.selected_features_ == ['useful', 'corr_keep', 'drop_me']


def test_scorecard_feature_selection_identical_stage():
    df = _build_dataset()

    selector = ScorecardFeatureSelection(
        null_threshold=None,
        iv_threshold=None,
        corr_threshold=None,
        mode_threshold=0.95,
        target='target',
    )

    selector.fit(df)

    assert 'identical_high' not in selector.selected_features_
    assert 'keep_me' not in selector.selected_features_
    assert selector.stage_selectors_['identical'].selected_features_
    drop_reasons = selector.dropped.set_index('variable')['rm_reason'].to_dict()
    assert drop_reasons['identical_high'] == 'identical'
    assert drop_reasons['keep_me'] == 'identical'


def test_scorecard_feature_selection_can_disable_stages():
    df = _build_dataset()

    selector = ScorecardFeatureSelection(
        null_threshold=None,
        iv_threshold=None,
        corr_threshold=None,
        mode_threshold=None,
        target='target',
    )

    selector.fit(df)

    assert selector.selected_features_ == [
        'useful', 'null_high', 'iv_low', 'corr_keep', 'corr_drop',
        'identical_high', 'keep_me', 'drop_me'
    ]
    assert selector.dropped.empty