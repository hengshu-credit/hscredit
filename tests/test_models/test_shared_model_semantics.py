"""模型公共契约的语义回归：等价输入、类别列、状态及配置往返。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as NativeLogisticRegression
from sklearn.tree import DecisionTreeClassifier

from hscredit.core.models import (
    LogisticRegression,
    ProbabilityCalibrator,
    ProbabilityScoreCard,
    RandomForest,
    ScoreCard,
    ScoreDriftCalibrator,
    ScoreTransformer,
)
from hscredit.core.models._contracts import FeatureSchema, split_sample_params


@pytest.fixture
def data():
    values, labels = make_classification(n_samples=100, n_features=4, random_state=41)
    return pd.DataFrame(values, columns=["年龄", "收入", "负债", "历史"], index=np.arange(100) + 1000), labels


@pytest.mark.parametrize("kind", ["forest", "lr", "calibration", "probability_card", "scorecard"])
def test_explicit_y_overrides_target_without_label_leakage(kind, data):
    X, y = data
    contaminated = X.assign(标签=1 - y)
    factories = {
        "forest": lambda: RandomForest(n_estimators=4, n_jobs=1, random_state=3, target="标签"),
        "lr": lambda: LogisticRegression(calculate_stats=False, target="标签"),
        "calibration": lambda: ProbabilityCalibrator(
            model=DecisionTreeClassifier(max_depth=3, random_state=3), target="标签", random_state=3
        ),
        "probability_card": lambda: ProbabilityScoreCard(model=NativeLogisticRegression(), target="标签"),
        "scorecard": lambda: ScoreCard(target="标签", calculate_stats=False),
    }
    explicit = factories[kind]().fit(contaminated, y)
    clean = factories[kind]().fit(X, y)
    kwargs = {"input_type": "woe"} if kind == "scorecard" else {}
    np.testing.assert_allclose(explicit.predict_proba(X, **kwargs), clean.predict_proba(X, **kwargs))
    np.testing.assert_allclose(explicit.predict_proba(X.iloc[:, ::-1], **kwargs), clean.predict_proba(X, **kwargs))
    assert "标签" not in (explicit.feature_names_ if kind == "scorecard" else explicit.feature_names_in_)


@pytest.mark.parametrize("kind", ["calibration", "probability_card", "scorecard"])
def test_failed_refit_does_not_serve_stale_predictions(kind, data):
    X, y = data
    models = {
        "calibration": ProbabilityCalibrator(model=DecisionTreeClassifier(random_state=2)),
        "probability_card": ProbabilityScoreCard(model=NativeLogisticRegression(), prefit=False),
        "scorecard": ScoreCard(calculate_stats=False),
    }
    model = models[kind].fit(X, y)
    with pytest.raises(ValueError):
        model.fit(X, y[:-1])
    assert model.training_summary_["状态"] == "失败"
    with pytest.raises(ValueError):
        model.predict_proba(X)


@pytest.mark.parametrize(
    "configured",
    [
        ProbabilityScoreCard(method="quantile", n_quantiles=7),
        ScoreTransformer(method="boxcox", lmbda=0.5),
        ScoreDriftCalibrator(method="quantile", n_quantiles=7),
        ScoreCard(clip=False),
    ],
)
def test_clone_preserves_all_extra_configuration(configured):
    copied = clone(configured)
    assert copied.get_params() == configured.get_params()
    extra = getattr(configured, configured._extra_params_attribute)
    for name, value in extra.items():
        assert copied.get_params()[name] == value


def test_scorecard_refit_matches_a_fresh_fit_and_refreshes_scale(data):
    X, y = data
    card = ScoreCard(calculate_stats=False).fit(X, y)
    card.set_params(pdo=20).fit(X, 1 - y)
    expected = ScoreCard(pdo=20, calculate_stats=False).fit(X, 1 - y)
    np.testing.assert_allclose(card.predict_proba(X, input_type="woe"), expected.predict_proba(X, input_type="woe"))
    np.testing.assert_allclose(card.predict_score(X, input_type="woe"), expected.predict_score(X, input_type="woe"))
    assert card.B_ == pytest.approx(20 / np.log(2))


class ReversedClassModel(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([1, 0])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack([p, 1 - p])


def test_probability_scorecard_uses_bad_class_not_column_position(data):
    X, y = data
    model = ReversedClassModel().fit(X, y)
    card = ProbabilityScoreCard(model=model).fit(X, y)
    expected = card.predict_score(proba=model.predict_proba(X)[:, 0])
    np.testing.assert_allclose(card.predict_score(X), expected)
    np.testing.assert_allclose(card.predict_score(proba=model.predict_proba(X)), expected)


def test_sample_parameter_slicing_is_positional_and_does_not_slice_categories(data):
    X, y = data
    weights = pd.Series(np.arange(len(y)) + 1, index=X.index[::-1])
    params = {"model__sample_weight": weights, "cat_features": list(range(len(y)))}
    sliced = split_sample_params(params, np.array([3, 1, 8]), len(y))
    assert sliced["model__sample_weight"].tolist() == [4, 2, 9]
    assert sliced["cat_features"] is params["cat_features"]
    assert len(params["model__sample_weight"]) == len(y)


def test_schema_preserves_categories_when_reconstructing_samples():
    frame = pd.DataFrame({"类别": pd.Categorical(["甲", "乙"], categories=["甲", "乙", "丙"]), "金额": [1.0, 2.0]})
    schema = FeatureSchema.from_data(frame)
    reconstructed = frame.iloc[0].to_frame().T
    aligned = schema.align(reconstructed, restore_dtypes=True)
    assert aligned.dtypes.equals(frame.dtypes)
    assert aligned["类别"].cat.categories.tolist() == ["甲", "乙", "丙"]


@pytest.mark.parametrize("model", [RandomForest(n_estimators=3, n_jobs=1), LogisticRegression(calculate_stats=False)])
def test_inference_export_preserves_predictions_without_mutating_training_object(model, data, tmp_path):
    X, y = data
    model.fit(X, y)
    model.tuner = {"原始数据": X}
    restored = type(model).load(model.save_inference(tmp_path / "inference.joblib"))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_allclose(restored.predict_score(X), model.predict_score(X))
    assert restored.tuner is None
    assert not hasattr(restored, "training_history_")
    assert not hasattr(restored.scorecard_, "training_history_")
    assert model.tuner["原始数据"] is X
    assert len(model.training_history_) == 1


@pytest.mark.parametrize("output,target", [("raw", 1), ("raw", 0), ("probability", 0), ("probability", 1)])
def test_tree_explanation_reconstructs_selected_class_and_output(output, target, data):
    from hscredit.core.models import ModelExplainer, XGBoost

    X, y = data
    model = XGBoost(n_estimators=4, max_depth=2, n_jobs=1, random_state=5).fit(X, y)
    explained = X.iloc[:4]
    result = ModelExplainer(model, background_data=X.iloc[:20], model_output=output, target_class=target).explain(
        explained
    )
    expected = (
        model.predict_proba(explained)[:, target]
        if output == "probability"
        else model.predict(explained, output_margin=True) * (1 if target else -1)
    )
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), expected, atol=1e-5)
    np.testing.assert_allclose(result.metadata["模型输出"], expected)


def test_linear_explanation_preserves_woe_sign_transformation():
    from hscredit.core.models import ModelExplainer

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["甲", "乙"])
    y = (X["甲"] + 0.4 * X["乙"] < 0).astype(int)
    model = LogisticRegression(calculate_stats=False, positive_woe_coef=True).fit(X, y)
    assert np.any(model.woe_coef_signs_ < 0)
    for target in (0, 1):
        result = ModelExplainer(model, background_data=X, model_output="raw", target_class=target).explain(X.head())
        expected = model.decision_function(X.head()) * (1 if target else -1)
        np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), expected, atol=1e-5)


def test_explanation_cache_is_invalidated_by_model_refit(data):
    from hscredit.core.models import ModelExplainer

    X, y = data
    model = RandomForest(n_estimators=4, n_jobs=1, random_state=4).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(20))
    first = explainer.explain(X.head())
    model.fit(X, 1 - y)
    second = explainer.explain(X.head())
    assert not np.allclose(first.values, second.values)
    np.testing.assert_allclose(
        second.base_values + second.values.sum(axis=1), model.predict_proba(X.head())[:, 1], atol=1e-5
    )


def test_scorecard_rejects_pipeline_that_did_not_train_lr_on_woe(data):
    from sklearn.pipeline import Pipeline
    from hscredit.core.binning import OptimalBinning

    X, y = data
    pipeline = Pipeline(
        [
            ("bin", OptimalBinning(method="quantile", max_n_bins=3, n_jobs=1)),
            ("lr", LogisticRegression(calculate_stats=False)),
        ]
    ).fit(X, y)
    card = ScoreCard(pipeline=pipeline, calculate_stats=False)
    with pytest.raises(ValueError, match="WOE 不一致"):
        card.predict_proba(X, input_type="raw")
    probability_card = ProbabilityScoreCard(model=pipeline).fit(X, y)
    np.testing.assert_allclose(probability_card.predict_proba(X), pipeline.predict_proba(X))


@pytest.mark.parametrize("name", ["LightGBM", "XGBoost"])
def test_custom_loss_weighted_training_matches_native_weighted_objective(name, data):
    from hscredit.core import models
    from hscredit.core.models import WeightedBCELoss

    X, y = data
    weights = np.where(y == 1, 5.0, 1.0)

    def weighted_objective(labels, margin, sample_weight=None):
        p = 1 / (1 + np.exp(-margin))
        sample_weight = weights if sample_weight is None else sample_weight
        return (p - labels) * sample_weight, p * (1 - p) * sample_weight

    cls = getattr(models, name)
    options = dict(n_estimators=4, n_jobs=1, random_state=4)
    actual = cls(objective=WeightedBCELoss(), **options).fit(X, y, sample_weight=weights)
    expected = cls(objective=weighted_objective, **options).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(actual.predict_proba(X), expected.predict_proba(X), atol=1e-7)
    if name == "LightGBM":
        np.testing.assert_array_equal(actual.predict(X, num_iteration=4), actual.predict(X))
