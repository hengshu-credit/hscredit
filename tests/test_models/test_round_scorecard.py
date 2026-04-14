import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import RoundScoreCard
from hscredit.core.models.scorecard.score_transformer import (
    StandardScoreTransformer,
    transform_probability_to_score,
)
from hscredit.utils.datasets import germancredit


def _train_round_scorecard(decimal: int = 2, direction: str = 'descending'):
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

    scorecard = RoundScoreCard(
        pdo=60,
        rate=2,
        base_odds=35,
        base_score=750,
        direction=direction,
        decimal=decimal,
        binner=binner,
    )
    X_train_woe = binner.transform(X_train, metric='woe')
    scorecard.fit(X_train_woe, y_train, input_type='woe')

    return scorecard, binner, X_test, X_train_woe


def _manual_score_from_points(scorecard: RoundScoreCard, X_raw: pd.DataFrame):
    points = scorecard.scorecard_points()
    base_score = float(points.loc[points['变量名称'] == '基础分', '对应分数'].iloc[0])
    special_codes = scorecard._get_deployment_special_codes()

    total = np.full(len(X_raw), base_score, dtype=float)
    feature_scores = {}
    matched_bins = {}

    for feature in scorecard.feature_names_:
        feature_points = points[points['变量名称'] == feature].reset_index(drop=True)
        feature_score_values = []
        feature_bin_labels = []

        for value in X_raw[feature].tolist():
            matched_score = None
            matched_label = None
            for _, row in feature_points.iterrows():
                bin_label = row['变量分箱']
                condition = scorecard._bin_label_to_python_condition('value', bin_label, special_codes=special_codes)
                matched = eval(condition, {'pd': pd}, {'value': value})

                if not matched and isinstance(bin_label, str) and ',' in bin_label and not bin_label.strip().startswith(('[', '(')):

                    categories = [item.strip() for item in bin_label.split(',') if item.strip()]
                    matched = str(value) in categories

                if matched:
                    matched_score = float(row['对应分数'])
                    matched_label = bin_label
                    break


            if matched_score is None:
                fallback_bins = [
                    (row['变量分箱'], float(row['对应分数']))
                    for _, row in feature_points.iterrows()
                ]
                matched_score = float(scorecard._get_deployment_default_score(fallback_bins))
                for label, score in fallback_bins:
                    if float(score) == matched_score and not scorecard._is_missing_descriptor(label) and not scorecard._is_special_descriptor(label):
                        matched_label = label
                        break

            if matched_score is None or matched_label is None:
                raise KeyError(f'未能在 scorecard_points 中找到特征 {feature} 值 {value!r} 的匹配分箱')

            feature_score_values.append(matched_score)
            feature_bin_labels.append(matched_label)


        mapped = np.asarray(feature_score_values, dtype=float)
        feature_scores[feature] = mapped
        matched_bins[feature] = pd.Series(feature_bin_labels, index=X_raw.index)
        total += mapped

    total = np.round(total, scorecard.decimal)
    return total, feature_scores, matched_bins, points



def test_score_transformer_decimal_parameter_name_unified():
    proba = np.array([0.1, 0.35, 0.8])

    transformer = StandardScoreTransformer(
        base_odds=0.05,
        base_score=600,
        pdo=20,
        decimal=2,
    )
    transformer.fit(proba)
    scores = transformer.predict(proba)

    assert transformer.decimal == 2
    assert np.all(scores == np.round(scores, 2))

    direct_scores = transform_probability_to_score(
        proba,
        method='standard',
        base_odds=0.05,
        base_score=600,
        pdo=20,
        decimal=2,
    )
    assert np.all(direct_scores == np.round(direct_scores, 2))


def test_round_scorecard_points_match_predict_descending():
    scorecard, binner, X_test, _ = _train_round_scorecard(decimal=2, direction='descending')

    raw_scores = scorecard.predict(X_test, input_type='raw')
    X_test_woe = binner.transform(X_test, metric='woe')[scorecard.feature_names_]
    woe_scores = scorecard.predict(X_test_woe, input_type='woe')


    assert np.all(raw_scores == np.round(raw_scores, 2))
    assert np.all(woe_scores == np.round(woe_scores, 2))
    np.testing.assert_allclose(raw_scores, woe_scores, atol=1e-9)

    manual_scores, feature_scores, _, points = _manual_score_from_points(scorecard, X_test)

    np.testing.assert_allclose(raw_scores, manual_scores, atol=1e-9)

    base_score = points.loc[points['变量名称'] == '基础分', '对应分数'].iloc[0]
    assert round(float(base_score), 2) == float(base_score)

    assert feature_scores
    for scores in feature_scores.values():
        assert np.all(scores == np.round(scores, 2))


def test_round_scorecard_consistency_for_details_and_deployment():
    scorecard, binner, X_test, _ = _train_round_scorecard(decimal=1, direction='ascending')
    sample = X_test.iloc[:60].copy()

    predicted = scorecard.predict(sample, input_type='raw')
    predicted_by_score = scorecard.predict_score(sample, input_type='raw')
    manual_scores, feature_scores, matched_bins, _ = _manual_score_from_points(scorecard, sample)


    np.testing.assert_allclose(predicted, predicted_by_score, atol=1e-9)
    np.testing.assert_allclose(predicted, manual_scores, atol=1e-9)

    detailed = scorecard.get_detailed_score(sample, include_reason=True)
    np.testing.assert_allclose(
        detailed[('样本信息', '总分')].to_numpy(dtype=float),
        predicted,
        atol=1e-9,
    )

    for feature in scorecard.feature_names_:
        np.testing.assert_allclose(
            detailed[(feature, '分数')].to_numpy(dtype=float),
            feature_scores[feature],
            atol=1e-9,
        )
        detailed_bins = detailed[(feature, '分箱')].astype(str).map(scorecard._normalize_rule_label)
        manual_bins = matched_bins[feature].astype(str).map(scorecard._normalize_rule_label)

        assert detailed_bins.tolist() == manual_bins.tolist()

    reasons = scorecard.get_reason(sample, keep=3)
    assert len(reasons) == len(sample)
    assert reasons['reason'].str.len().gt(0).all()

    namespace = {}
    exec(scorecard.export_deployment_code(language='python'), namespace)
    deployed_scores = sample.apply(lambda row: namespace['calculate_score'](row.to_dict()), axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(predicted, deployed_scores, atol=1e-9)

    transformed_bins = binner.transform(sample, metric='bins')[scorecard.feature_names_]
    assert transformed_bins.shape[0] == len(sample)
