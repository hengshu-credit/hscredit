import json
import re
import subprocess
import sys
import types
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard, RoundScoreCard
from hscredit.utils.datasets import germancredit


def test_scorecard_source_has_no_duplicate_dictionary_keys():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "hscredit"
        / "core"
        / "models"
        / "scorecard"
        / "scorecard.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            duplicates.append((node.lineno, repeated))
    assert duplicates == []


def _skip_if_java_too_old_for_sklearn2pmml():
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("PMML 集成测试需要 Java 运行时")

    version_output = result.stderr or result.stdout
    match = re.search(r'version "([^"]+)"', version_output)
    if not match:
        pytest.skip("无法识别 Java 版本，跳过 PMML 集成测试")

    version = match.group(1)
    if version.startswith("1."):
        major = int(version.split(".")[1])
    else:
        major = int(version.split(".", 1)[0])
    if major < 11:
        pytest.skip("当前 sklearn2pmml 运行时需要 Java 11+，本机 Java 版本过低")


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


@pytest.mark.parametrize('cls', [ScoreCard, RoundScoreCard])
def test_scorecard_export_load_without_binner_predicts_raw_by_rules(cls):
    """导出带元数据的评分卡应能在无 binner 的离线环境中按规则对原始数据评分."""
    scorecard, _, X_test = _train_scorecard(direction='descending')
    if cls is RoundScoreCard:
        scorecard = RoundScoreCard(scorecard=scorecard, decimal=2)

    sample = X_test.iloc[:50].copy()
    reference = scorecard.predict(sample, input_type='raw')

    rules = scorecard.export(include_meta=True, decimal=12)
    loaded_scorecard = cls(pdo=60, rate=2, base_odds=35, base_score=750)
    loaded_scorecard.load(rules)
    loaded_scores = loaded_scorecard.predict(sample, input_type='raw')

    np.testing.assert_allclose(reference, loaded_scores, atol=1e-9)


def test_scorecard_loads_toad_export_json_as_offline_model(tmp_path):
    """toad 0.1.5 的真实导出产物可直接作为离线评分卡导入。"""
    X = pd.DataFrame({
        'age': [18, 22, 25, 30, 35, 42, 48, 55, 60, 63] * 2,
        'city': ['A', 'A', 'B', 'C', 'A', 'B', 'D', 'D', 'C', 'E'] * 2,
    })
    exported_rules = {
        'age': {'[-inf ~ 38.5)': 739.64, '[38.5 ~ inf)': -297.39},
        'city': {'A': 336.6, 'B': 221.12, 'C': 221.12, 'E,D': 105.65},
    }
    reference = np.asarray([
        1076.2311719694264, 1076.2311719694264, 960.7572641665862,
        960.7572641665862, 1076.2311719694264, -76.27122619998207,
        -191.74513400282234, -191.74513400282234, -76.27122619998207,
        -191.74513400282234,
    ] * 2)
    json_path = tmp_path / 'toad_scorecard.json'
    json_path.write_text(json.dumps(exported_rules), encoding='utf-8')

    loaded_from_path = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750).load(str(json_path))
    loaded_from_dict = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750).load(exported_rules)

    np.testing.assert_allclose(loaded_from_path.predict(X, input_type='raw'), reference, atol=0.02)
    np.testing.assert_allclose(loaded_from_dict.predict(X, input_type='raw'), reference, atol=0.02)


def test_scorecard_loads_scorecardpipeline_export_labels():
    """兼容 scorecardpipeline 的中文区间、缺失值和 else 类别兜底标签."""
    rules = {
        'age': {
            '[负无穷 , 25)': 10,
            '[25 , 40)': 20,
            '[40 , 正无穷)': 30,
            '缺失值': -5,
        },
        'city': {
            'A,B': 3,
            'C': 7,
            'else': -2,
        },
    }
    sample = pd.DataFrame({
        'age': [18, 25, 45, np.nan],
        'city': ['A', 'C', 'Z', 'B'],
    })

    card = ScoreCard().load(rules)
    scores = card.predict(sample, input_type='raw')

    np.testing.assert_allclose(scores, np.array([13, 27, 28, -2], dtype=float))


def test_scorecard_loads_export_frame_records():
    """支持读取 export(to_frame=True) 或 DataFrame JSON records 风格的规则."""
    records = [
        {'name': 'age', 'value': '[负无穷 , 25)', 'score': 10},
        {'name': 'age', 'value': '[25 , 正无穷)', 'score': 20},
    ]
    sample = pd.DataFrame({'age': [18, 30]})

    card = ScoreCard().load(records)
    np.testing.assert_allclose(card.predict(sample, input_type='raw'), np.array([10, 20], dtype=float))


def test_scorecard_python_deployment_code_matches_predict():
    scorecard, _, X_test = _train_scorecard(direction='ascending')
    sample = X_test.copy()
    reference = scorecard.predict(sample, input_type='raw')

    namespace = {}
    decimal = 12
    exec(scorecard.export_deployment_code(language='python', decimal=decimal), namespace)
    deployed_scores = sample.apply(lambda row: namespace['calculate_score'](row.to_dict()), axis=1).to_numpy()

    np.testing.assert_allclose(reference, deployed_scores, atol=1e-9)
    assert namespace['feature_name_in_'] == scorecard.feature_names_
    assert namespace['feature_names_in_'] == scorecard.feature_names_
    assert namespace['n_features_in_'] == len(scorecard.feature_names_)
    assert namespace['pdo'] == scorecard.pdo
    assert namespace['rate'] == scorecard.rate
    assert namespace['base_odds'] == scorecard.base_odds
    assert namespace['base_score'] == scorecard.base_score
    assert namespace['step'] == scorecard.step
    assert namespace['lower'] == scorecard.lower
    assert namespace['upper'] == scorecard.upper
    assert namespace['direction'] == scorecard.direction_
    assert namespace['decimal'] == scorecard.decimal
    assert namespace['A_'] == float(scorecard.A_)
    assert namespace['B_'] == float(scorecard.B_)

    deployment_base_score, score_sign = scorecard._get_deployment_base_score_and_sign()
    assert namespace['intercept_score'] == round(float(deployment_base_score), decimal)
    assert namespace['deployment_base_score'] == round(float(deployment_base_score), decimal)
    assert namespace['score_sign'] == float(score_sign)


def test_scorecard_python_deployment_code_uses_categorical_default_bin_for_unseen_values():
    scorecard, _, X_test = _train_scorecard(direction='descending')
    sample = X_test.loc[[83]].copy()
    assert sample.iloc[0]['purpose'] == 'others'

    namespace = {}
    exec(scorecard.export_deployment_code(language='python', decimal=12), namespace)

    reference = scorecard.predict(sample, input_type='raw')[0]
    deployed = namespace['calculate_score'](sample.iloc[0].to_dict())

    assert abs(reference - deployed) < 1e-9


def test_scorecard_pmml_export_uses_expression_transformer_for_string_categories(tmp_path, monkeypatch):
    scorecard, _, _ = _train_scorecard(direction='descending')
    captured = {}

    class FakeLookupTransformer:
        def __init__(self, mapping, default_value=0.0):
            self.mapping = mapping
            self.default_value = default_value

    class FakeExpressionTransformer:
        def __init__(self, expression):
            self.expression = expression

    class FakeConcatTransformer:
        pass

    class FakeAggregateTransformer:
        def __init__(self, function):
            self.function = function

    class FakeAlias:
        def __init__(self, transformer, name, prefit=False):
            self.transformer = transformer
            self.name = name
            self.prefit = prefit

    class FakeCategoricalDomain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeContinuousDomain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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

    fake_sklearn2pmml_module = types.ModuleType('sklearn2pmml')
    fake_sklearn2pmml_module.sklearn2pmml = fake_sklearn2pmml
    fake_sklearn2pmml_module.PMMLPipeline = FakePMMLPipeline

    fake_decoration = types.ModuleType('sklearn2pmml.decoration')
    fake_decoration.Alias = FakeAlias
    fake_decoration.CategoricalDomain = FakeCategoricalDomain
    fake_decoration.ContinuousDomain = FakeContinuousDomain

    fake_preprocessing = types.ModuleType('sklearn2pmml.preprocessing')
    fake_preprocessing.AggregateTransformer = FakeAggregateTransformer
    fake_preprocessing.ConcatTransformer = FakeConcatTransformer
    fake_preprocessing.LookupTransformer = FakeLookupTransformer
    fake_preprocessing.ExpressionTransformer = FakeExpressionTransformer

    monkeypatch.setitem(sys.modules, 'sklearn2pmml', fake_sklearn2pmml_module)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml.decoration', fake_decoration)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml.preprocessing', fake_preprocessing)

    scorecard.export_pmml(str(tmp_path / 'scorecard.pmml'))

    mapper = captured['pipeline'].named_steps['preprocessing'].transformers
    categorical_steps = dict(next(
        transformer.steps
        for _, transformer, features in mapper
        if features == ['status_of_existing_checking_account']
    ))
    numeric_steps = dict(next(
        transformer.steps
        for _, transformer, features in mapper
        if (
            features == ['duration_in_month']
            and transformer.steps[-1][1].name == '__score_duration_in_month'
        )
    ))

    assert isinstance(categorical_steps['domain'], FakeCategoricalDomain)
    assert isinstance(categorical_steps['prepare'], FakeConcatTransformer)
    assert isinstance(categorical_steps['score'], FakeAlias)
    assert isinstance(categorical_steps['score'].transformer, FakeLookupTransformer)
    assert categorical_steps['score'].transformer.mapping['no checking account'] != 0.0
    assert categorical_steps['score'].transformer.default_value == pytest.approx(
        categorical_steps['score'].transformer.mapping['no checking account']
    )

    assert isinstance(numeric_steps['domain'], FakeContinuousDomain)
    assert isinstance(numeric_steps['prepare'], FakeAggregateTransformer)
    assert numeric_steps['prepare'].function == 'min'
    assert isinstance(numeric_steps['score'], FakeAlias)
    assert isinstance(numeric_steps['score'].transformer, FakeExpressionTransformer)
    assert 'X[0] < 6.5' in numeric_steps['score'].transformer.expression


def test_scorecard_pmml_export_tolerates_sklearn2pmml_none_len_bug(tmp_path, monkeypatch):
    scorecard, _, _ = _train_scorecard(direction='descending')

    class FakeExpressionTransformer:
        def __init__(self, expression):
            self.expression = expression

    class FakeLookupTransformer:
        def __init__(self, mapping, default_value=0.0):
            self.mapping = mapping
            self.default_value = default_value

    class FakeConcatTransformer:
        pass

    class FakeAggregateTransformer:
        def __init__(self, function):
            self.function = function

    class FakeAlias:
        def __init__(self, transformer, name, prefit=False):
            self.transformer = transformer
            self.name = name
            self.prefit = prefit

    class FakeCategoricalDomain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeContinuousDomain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLinearRegression:
        def __init__(self, fit_intercept=True):
            self.fit_intercept = fit_intercept

    class FakePMMLPipeline:
        def __init__(self, steps):
            self.steps = steps
            self.named_steps = dict(steps)

        def fit(self, X, y):
            return self

    def fake_sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=False):
        with open(pmml_file, 'w', encoding='utf-8') as handle:
            handle.write('<PMML/>')
        raise TypeError("object of type 'NoneType' has no len()")

    fake_linear_model = types.ModuleType('sklearn.linear_model')
    fake_linear_model.LinearRegression = FakeLinearRegression

    fake_sklearn2pmml_module = types.ModuleType('sklearn2pmml')
    fake_sklearn2pmml_module.sklearn2pmml = fake_sklearn2pmml
    fake_sklearn2pmml_module.PMMLPipeline = FakePMMLPipeline

    fake_decoration = types.ModuleType('sklearn2pmml.decoration')
    fake_decoration.Alias = FakeAlias
    fake_decoration.CategoricalDomain = FakeCategoricalDomain
    fake_decoration.ContinuousDomain = FakeContinuousDomain

    fake_preprocessing = types.ModuleType('sklearn2pmml.preprocessing')
    fake_preprocessing.AggregateTransformer = FakeAggregateTransformer
    fake_preprocessing.ConcatTransformer = FakeConcatTransformer
    fake_preprocessing.LookupTransformer = FakeLookupTransformer
    fake_preprocessing.ExpressionTransformer = FakeExpressionTransformer

    monkeypatch.setitem(sys.modules, 'sklearn.linear_model', fake_linear_model)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml', fake_sklearn2pmml_module)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml.decoration', fake_decoration)
    monkeypatch.setitem(sys.modules, 'sklearn2pmml.preprocessing', fake_preprocessing)

    pmml_path = tmp_path / 'scorecard.pmml'

    with pytest.warns(RuntimeWarning, match='continuing with the exported artifact'):
        scorecard.export_pmml(str(pmml_path))

    assert pmml_path.exists()
    assert pmml_path.read_text(encoding='utf-8') == '<PMML/>'


def test_scorecard_pmml_preprocessing_matches_reference_feature_scores(tmp_path):
    pytest.importorskip('sklearn2pmml')
    _skip_if_java_too_old_for_sklearn2pmml()

    scorecard, binner, X_test = _train_scorecard(direction='descending')
    sample = X_test.copy()

    pipeline = scorecard.export_pmml(str(tmp_path / 'scorecard.pmml'), debug=True)
    transformed = pipeline.named_steps['preprocessing'].transform(sample).astype(float)

    woe = binner.transform(sample, metric='woe')[scorecard.feature_names_]
    reference = scorecard._woe_to_score(woe, scorecard.feature_names_)

    np.testing.assert_allclose(np.asarray(transformed, dtype=float), reference, atol=1e-9)


def test_scorecard_pmml_predict_matches_reference_scores(tmp_path):
    pytest.importorskip('sklearn2pmml')
    pytest.importorskip('pypmml')
    _skip_if_java_too_old_for_sklearn2pmml()

    from pypmml import Model

    scorecard, _, X_test = _train_scorecard(direction='descending')
    sample = X_test.copy()
    reference = scorecard.predict(sample, input_type='raw')

    pmml_path = tmp_path / 'scorecard.pmml'
    scorecard.export_pmml(str(pmml_path))

    pmml_scores = Model.load(str(pmml_path)).predict(sample)['predicted_score'].to_numpy()

    np.testing.assert_allclose(reference, pmml_scores, atol=1e-9)
