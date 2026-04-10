import sys
import types

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


def test_scorecard_pmml_export_uses_lookup_transformer_for_string_categories(tmp_path, monkeypatch):
    scorecard, _, _ = _train_scorecard(direction='descending')
    captured = {}

    class FakeDataFrameMapper:
        def __init__(self, mapper, df_out=True):
            self.mapper = mapper
            self.df_out = df_out

    class FakeLookupTransformer:
        def __init__(self, mapping, default_value=0.0):
            self.mapping = mapping
            self.default_value = default_value

    class FakeExpressionTransformer:
        def __init__(self, expression):
            self.expression = expression

    class FakePMMLPipeline:
        def __init__(self, steps):
            self.steps = steps
            self.named_steps = dict(steps)

        def fit(self, X, y):
            captured['sample_df'] = X
            captured['sample_y'] = y
            return self

    def fake_sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=False):
        captured['pipeline'] = pipeline
        captured['pmml_file'] = pmml_file

    fake_sklearn_pandas = types.ModuleType('sklearn_pandas')
    fake_sklearn_pandas.DataFrameMapper = FakeDataFrameMapper

    fake_sklearn2pmml_module = types.ModuleType('sklearn2pmml')
    fake_sklearn2pmml_module.sklearn2pmml = fake_sklearn2pmml
    fake_sklearn2pmml_module.PMMLPipeline = FakePMMLPipeline

    fake_preprocessing = types.ModuleType('sklearn2pmml.preprocessing')
    fake_preprocessing.LookupTransformer = FakeLookupTransformer
    fake_preprocessing.ExpressionTransformer = FakeExpressionTransformer

    monkeypatch.setitem(sys.modules, 'sklearn_pandas', fake_sklearn_pandas)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml', fake_sklearn2pmml_module)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml.preprocessing', fake_preprocessing)

    scorecard.export_pmml(str(tmp_path / 'scorecard.pmml'))

    mapper = captured['pipeline'].named_steps['preprocessing'].mapper
    categorical_transformer = next(
        transformer
        for features, transformer in mapper
        if features == ['status_of_existing_checking_account']
    )

    assert isinstance(categorical_transformer, FakeLookupTransformer)
    assert 'no checking account' in categorical_transformer.mapping