"""第二轮模型回归：真实推理映射、显式评估权重与旧状态迁移。"""

import pickle
import copy
import shutil
import sqlite3
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score

from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.models import LogisticRegression, RandomForest, ScoreCard
from hscredit.core.models.tuning.tuning import ModelTuner
from hscredit.core.models.tuning.tuning import Metric
from hscredit.exceptions import ParallelExecutionError


def _card(training_mode="oof"):
    rng = np.random.RandomState(12)
    X = pd.DataFrame({"x": np.arange(200, dtype=float)})
    y = (rng.uniform(size=200) < 0.1 + 0.8 * X.x / 200).astype(int)
    X.loc[1, "x"], X.loc[2, "x"] = np.nan, -999
    binner = OptimalBinning(method="quantile", max_n_bins=4, special_codes=[-999], n_jobs=1).fit(X, y)
    encoder = WOEEncoder(cols=["x"], regularization=50, training_mode=training_mode, cv=4, n_jobs=1)
    encoded = encoder.fit_transform(binner.transform(X, metric="bins"), y)
    card = ScoreCard(binner=binner, encoder=encoder, calculate_stats=False, decimal=8, clip=False).fit(encoded, y)
    return X, card


@pytest.mark.parametrize("mode", ["in_sample", "oof"])
def test_explicit_encoder_python_sql_and_pickle_match_raw_scores(mode):
    X, card = _card(mode)
    expected = card.predict(X, input_type="raw")
    namespace = {}
    exec(card.export_deployment_code("python"), namespace)
    actual = [namespace["calculate_score"](row) for row in X.to_dict("records")]
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    with sqlite3.connect(":memory:") as connection:
        X.to_sql("your_table", connection, index=False)
        sql = connection.execute(card.export_deployment_code("sql")).fetchall()
    np.testing.assert_allclose(np.asarray(sql)[:, 0], expected, atol=1e-7, rtol=0)
    restored = pickle.loads(pickle.dumps(card))
    np.testing.assert_allclose(restored.predict(X, input_type="raw"), expected)


@pytest.mark.integration
def test_explicit_encoder_java_matches_raw_scores(tmp_path):
    if not shutil.which("javac") or not shutil.which("java"):
        pytest.skip("Java 部署等价验收需要 javac 和 java")
    X, card = _card()
    sample = X.iloc[:8]
    java = card.export_deployment_code("java")
    values = ["null" if pd.isna(value) else f"Double.valueOf({float(value)!r})" for value in sample.x]
    main = "public static void main(String[] args) { Object[] values = {" + ",".join(values) + "};"
    main += 'for (Object value : values) { Map<String,Object> row = new java.util.HashMap<>(); row.put("x", value);'
    main += "System.out.println(calculate_score(row)); }}"
    path = tmp_path / "ScoreCard.java"
    path.write_text(java.rsplit("}", 1)[0] + main + "}", encoding="utf-8")
    subprocess.run(["javac", "-encoding", "UTF-8", str(path)], check=True, capture_output=True, timeout=60)
    output = subprocess.run(
        ["java", "-cp", str(tmp_path), "ScoreCard"], check=True, capture_output=True, text=True, timeout=60
    )
    np.testing.assert_allclose(
        np.fromstring(output.stdout, sep="\n"), card.predict(sample, input_type="raw"), atol=1e-7, rtol=0
    )


def _weighted_data():
    X = pd.DataFrame(np.random.RandomState(43).normal(size=(100, 2)), columns=["a", "b"])
    y = np.arange(100) % 2
    return X, y, np.where(y, 3, 1)


def test_frequency_score_anchor_matches_repeated_rows():
    X, y, weights = _weighted_data()
    params = dict(
        penalty=None, calculate_stats=False, weight_type="frequency", scorecard_params={"decimal": 8, "clip": False}
    )
    weighted = LogisticRegression(**params).fit(X, y, sample_weight=weights)
    repeated = LogisticRegression(**params).fit(X.iloc[np.repeat(np.arange(len(X)), weights)], np.repeat(y, weights))
    assert weighted.bad_rate_ == repeated.bad_rate_
    np.testing.assert_allclose(weighted.predict_score(X), repeated.predict_score(X), atol=1e-6)


def test_cost_weight_does_not_change_default_prior_and_accepts_explicit_business_prior():
    X, y, weights = _weighted_data()
    default = LogisticRegression(calculate_stats=False, weight_type="cost").fit(X, y, sample_weight=weights)
    assert default.bad_rate_ == 0.5
    explicit = LogisticRegression(calculate_stats=False, weight_type="cost", scorecard_params={"base_bad_rate": 0.2})
    explicit.fit(X, y, sample_weight=weights)
    assert explicit.bad_rate_ == 0.2


class FixedProbability(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        values = np.asarray(X)[:, 0]
        return np.column_stack([1 - values, values])


def _tuner(metric="auc", **kwargs):
    splits = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    return ModelTuner(FixedProbability, search_space={}, cv=splits, metric=metric, n_jobs=1, verbose=False, **kwargs)


def test_tuner_evaluation_weight_is_explicit_and_does_not_change_old_default():
    X = np.array([[0.1], [0.2], [0.3], [0.4]] * 2)
    y, weights = np.array([0, 1, 0, 1] * 2), np.array([1, 10, 1, 1] * 2)
    plain, weighted = _tuner(), _tuner()
    plain.fit(X, y, sample_weight=weights, n_trials=1, show_progress_bar=False)
    weighted.fit(X, y, sample_weight=weights, evaluation_weight=weights, n_trials=1, show_progress_bar=False)
    assert plain.best_score_ == pytest.approx(0.75)
    assert weighted.best_score_ == pytest.approx(roc_auc_score(y, X[:, 0], sample_weight=weights))
    assert weighted.evaluate_trials(X, y, trial_points=[{}], evaluation_weight=weights).AUC.iloc[0] == pytest.approx(
        weighted.best_score_
    )


def test_tuner_rejects_weight_for_metric_that_cannot_consume_it():
    tuner = _tuner(metric=lambda y, p: float(np.mean(p)), direction="maximize")
    with pytest.raises(ValueError, match="评估权重"):
        tuner.fit(
            np.array([[0.1], [0.2], [0.3], [0.4]] * 2),
            np.array([0, 1, 0, 1] * 2),
            evaluation_weight=np.ones(8),
            n_trials=1,
            show_progress_bar=False,
        )


@pytest.mark.parametrize("level", ["coef", "full"])
def test_no_intercept_summary_has_only_feature_rows(level):
    X, y, _ = _weighted_data()
    model = LogisticRegression(penalty=None, fit_intercept=False, statistics_level=level).fit(X, y)
    summary = model.summary()
    assert summary.index.tolist() == ["a", "b"]
    assert summary.shape[0] == 2


def test_postfit_woe_sign_change_transforms_statistics_without_data():
    rng = np.random.RandomState(15)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y = (rng.uniform(size=300) < 1 / (1 + np.exp(2 * X.a - X.b))).astype(int)
    model = LogisticRegression(penalty=None, positive_woe_coef=False, statistics_level="coef").fit(X, y)
    probability, covariance = model.predict_proba(X), model.cov_matrix_.copy()
    signs = np.r_[1.0, np.where(model.coef_[0] < 0, -1.0, 1.0)]
    model.ensure_positive_woe_coefficients()
    np.testing.assert_allclose(model.predict_proba(X), probability)
    np.testing.assert_allclose(model.z_coef_, model.coef_ / model.std_err_coef_)
    np.testing.assert_allclose(model.cov_matrix_, covariance * signs[:, None] * signs[None, :])


@pytest.mark.parametrize("model", [LogisticRegression(calculate_stats=False), RandomForest(n_estimators=2, n_jobs=1)])
def test_legacy_model_parameter_defaults_migrate_on_pickle_load(model, tmp_path):
    X, y, _ = _weighted_data()
    model.fit(X, y)
    expected = model.predict_proba(X)
    for name in ("history_policy", "max_history", "statistics_level", "max_dense_bytes", "weight_type"):
        model.__dict__.pop(name, None)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), expected)
    clone(restored).get_params()
    restored.fit(X, y)
    restored.save(tmp_path / "migrated.pkl")


@pytest.mark.parametrize("missing_separate", [False, True])
def test_multifeature_encoder_category_unknown_special_and_rule_roundtrip(missing_separate):
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"x": np.arange(120, dtype=float), "c": ["A", "B", "__UNKNOWN__", "a,b", "SPECIAL"] * 24})
    X.loc[0, "x"], X.loc[1, "x"], X.loc[2, "c"] = np.nan, -999, None
    y = (rng.uniform(size=len(X)) < 0.15 + 0.7 * np.arange(len(X)) / len(X)).astype(int)
    binner = OptimalBinning(
        method="quantile", max_n_bins=3, special_codes=[-999, "SPECIAL"], missing_separate=missing_separate, n_jobs=1
    ).fit(X, y)
    encoder = WOEEncoder(cols=["x", "c"], regularization=25, training_mode="oof", cv=3, n_jobs=1)
    encoded = encoder.fit_transform(binner.transform(X, metric="bins"), y)
    card = ScoreCard(binner=binner, encoder=encoder, calculate_stats=False, clip=False).fit(encoded, y)
    raw = pd.concat(
        [X, pd.DataFrame({"x": [0.0, 30.0, np.nan, -999.0], "c": ["new", None, "SPECIAL", "__UNKNOWN__"]})],
        ignore_index=True,
    )
    expected = card.predict(raw, input_type="raw")
    namespace = {}
    exec(card.export_deployment_code("python"), namespace)
    actual = np.array([namespace["calculate_score"](record) for record in raw.to_dict("records")])
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    with sqlite3.connect(":memory:") as connection:
        raw.to_sql("your_table", connection, index=False)
        sql_scores = np.asarray(connection.execute(card.export_deployment_code("sql")).fetchall())[:, 0]
    np.testing.assert_allclose(sql_scores, expected, atol=1e-7, rtol=0)
    restored = ScoreCard().load_rules(card.export())
    np.testing.assert_allclose(restored.predict(raw, input_type="raw"), expected, atol=1e-7, rtol=0)
    restored_external = ScoreCard().load_rules(card.export(), binner=binner)
    np.testing.assert_allclose(restored_external.predict(raw, input_type="raw"), expected, atol=1e-7, rtol=0)


def test_pretrained_scorecard_uses_same_explicit_encoder_mapping():
    X, card = _card()
    pretrained = ScoreCard(binner=card.binner, encoder=card.encoder, lr_model=card.lr_model_, clip=False)
    namespace = {}
    exec(pretrained.export_deployment_code("python"), namespace)
    np.testing.assert_allclose(
        [namespace["calculate_score"](r) for r in X.to_dict("records")],
        pretrained.predict(X, input_type="raw"),
        atol=1e-7,
        rtol=0,
    )


@pytest.mark.parametrize("metric", ["auc", "ks", "logloss", "accuracy", "precision", "recall", "f1", "ks_diff"])
def test_metric_explicit_frequency_weight_matches_replication(metric):
    labels, probability, weights = np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]), np.array([1, 4, 2, 1])
    evaluator = Metric(metric)
    actual = evaluator(
        labels,
        probability,
        y_train=labels,
        y_train_pred=probability,
        sample_weight=weights,
        train_sample_weight=weights,
    )
    expected = evaluator(
        np.repeat(labels, weights),
        np.repeat(probability, weights),
        y_train=np.repeat(labels, weights),
        y_train_pred=np.repeat(probability, weights),
    )
    assert actual == pytest.approx(expected)


def test_model_tune_forwards_evaluation_weight_and_retains_policy_in_summary():
    X, y, weights = _weighted_data()
    best = LogisticRegression(calculate_stats=False).tune(
        X,
        y,
        search_space={"C": [0.2]},
        cv=2,
        n_jobs=1,
        n_trials=1,
        metric="auc",
        sample_weight=weights,
        evaluation_weight=weights,
        retention="summary",
        show_progress_bar=False,
    )
    assert np.array_equal(best.tuner._evaluation_weight, weights)
    assert all(fold["评估口径"] == "显式评估权重" for fold in best.tuner.get_trial_result(0)["各折"])
    best.tuner.release_training_data()
    assert best.tuner._evaluation_weight is None


def test_standalone_evaluate_trials_accepts_weights_without_prior_fit():
    X, y, weights = _weighted_data()
    tuner = ModelTuner(FixedProbability, search_space={}, cv=2, metric="auc", n_jobs=1, verbose=False)
    values = pd.DataFrame({"p": 1 / (1 + np.exp(-X.a))})
    assert np.isfinite(tuner.evaluate_trials(values, y, trial_points=[{}], evaluation_weight=weights).AUC).all()


@pytest.mark.parametrize("model_name", ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "NGBoost"])
def test_risk_models_frequency_prior_routes_original_weights_and_not_native_parameters(model_name):
    import hscredit.core.models as models

    X, y, weights = _weighted_data()
    params = dict(n_jobs=1, random_state=42, weight_type="frequency")
    params["iterations" if model_name == "CatBoost" else "n_estimators"] = 2
    if model_name != "RandomForest":
        params.update(early_stopping_rounds=1, validation_fraction=0.2)
    model = getattr(models, model_name)(**params).fit(X, y, sample_weight=weights)
    assert model.bad_rate_ == np.average(y, weights=weights)
    assert "weight_type" not in model.get_native_params()
    assert clone(model).get_params()["weight_type"] == "frequency"


def test_scorecard_deployment_uses_exact_split_not_rounded_display_label():
    cut = 0.123456789
    X = pd.DataFrame({"x": np.r_[np.linspace(-1, 1, 100), np.nan, -999.0]})
    y = ((np.arange(len(X)) % 4 == 0) | (X.x > 0.5)).astype(int)
    binner = OptimalBinning(
        method="quantile",
        user_splits={"x": [cut]},
        user_splits_fixed=True,
        max_n_bins=2,
        special_codes=[-999.0],
        n_jobs=1,
    ).fit(X, y)
    encoder = WOEEncoder(cols=["x"], regularization=5, n_jobs=1)
    encoded = encoder.fit_transform(binner.transform(X, metric="bins"), y)
    card = ScoreCard(binner=binner, encoder=encoder, calculate_stats=False, clip=False).fit(encoded, y)
    raw = pd.DataFrame({"x": [cut - 1e-8, cut, cut + 1e-8, np.nan, -999.0]})
    expected = card.predict(raw, input_type="raw")
    namespace = {}
    exec(card.export_deployment_code("python"), namespace)
    np.testing.assert_allclose(
        [namespace["calculate_score"](row) for row in raw.to_dict("records")], expected, atol=1e-7, rtol=0
    )
    np.testing.assert_allclose(
        ScoreCard().load_rules(card.export()).predict(raw, input_type="raw"), expected, atol=1e-7, rtol=0
    )


def test_encoder_unknown_error_policy_survives_rule_export():
    X = pd.DataFrame({"c": ["A", "B", "C"] * 30})
    y = (np.arange(len(X)) % 4 == 0).astype(int)
    binner = OptimalBinning(method="quantile", max_n_bins=3, n_jobs=1).fit(X, y)
    encoder = WOEEncoder(cols=["c"], handle_unknown="error", n_jobs=1)
    encoded = encoder.fit_transform(binner.transform(X, metric="bins"), y)
    card = ScoreCard(binner=binner, encoder=encoder, calculate_stats=False).fit(encoded, y)
    for value in ("未见类别", None):
        raw = pd.DataFrame({"c": [value]})
        with pytest.raises((ValueError, ParallelExecutionError), match="未知"):
            card.predict(raw, input_type="raw")
        with pytest.raises(ValueError, match="未知"):
            ScoreCard().load_rules(card.export()).predict(raw, input_type="raw")
        with pytest.raises(ValueError, match="未知"):
            ScoreCard().load_rules(card.export(), binner=binner).predict(raw, input_type="raw")
    with pytest.raises(ValueError, match="无法表达"):
        card.export_deployment_code("sql")


def test_owned_callable_objectives_do_not_silently_ignore_evaluation_weight():
    from hscredit.core.models.tuning.tuning import TuningObjective

    labels, probability, weights = np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]), np.array([1, 4, 2, 1])
    evaluator = Metric(TuningObjective.ks, direction="maximize")
    assert evaluator(labels, probability, sample_weight=weights) == Metric("ks")(
        labels, probability, sample_weight=weights
    )
    # 头部指标现在明确支持评估权重，验证结果确实使用了权重。
    from functools import partial

    head = Metric(partial(TuningObjective.lift_head, ratio=0.5), direction="maximize")
    assert head(labels, probability, sample_weight=weights) == pytest.approx((1 / 3) / (5 / 8))
    assert head(labels, probability, sample_weight=weights) != head(labels, probability)


def test_frequency_prior_rejects_zero_weight_class_unless_explicit_prior():
    X, y, _ = _weighted_data()
    weights = np.where(y == 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="正权重|base_bad_rate"):
        LogisticRegression(calculate_stats=False, weight_type="frequency").fit(X, y, sample_weight=weights)
    explicit = LogisticRegression(
        calculate_stats=False, weight_type="frequency", scorecard_params={"base_bad_rate": 0.2}
    )
    explicit.fit(X, y, sample_weight=weights)
    assert explicit.bad_rate_ == 0.2
    assert np.isfinite(explicit.predict_score(X)).all()


def test_tuner_rejects_zero_mass_validation_fold_before_trials_start():
    X = np.array([[0.1], [0.2], [0.3], [0.4]] * 2)
    y, weights = np.array([0, 1, 0, 1] * 2), np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="第 0 折验证集 evaluation_weight 总和"):
        _tuner().fit(X, y, evaluation_weight=weights, n_trials=1, show_progress_bar=False)
    with pytest.raises(ValueError, match="第 0 折验证集 evaluation_weight 总和"):
        _tuner().evaluate_trials(X, y, trial_points=[{}], evaluation_weight=weights)


def _legacy_wrong_rules(card):
    legacy = copy.deepcopy(card)
    legacy.__dict__.pop("_rules_semantics_version_", None)
    for i, feature in enumerate(legacy.feature_names_):
        wrong = legacy.binner.bin_tables_[feature]["分档WOE值"].to_numpy() * legacy._get_feature_woe_sign(i)
        legacy.rules_[feature]["woe"] = wrong
        legacy.rules_[feature]["scores"] = np.array([legacy._woe_to_point(value, legacy.coef_[i]) for value in wrong])
    return legacy


def test_legacy_complete_card_rebuilds_wrong_cached_rules_before_export():
    X, card = _card()
    legacy = pickle.loads(pickle.dumps(_legacy_wrong_rules(card)))
    expected = legacy.predict(X, input_type="raw")
    namespace = {}
    exec(legacy.export_deployment_code("python"), namespace)
    np.testing.assert_allclose(
        [namespace["calculate_score"](row) for row in X.to_dict("records")], expected, atol=1e-7, rtol=0
    )
    assert legacy._rules_semantics_version_ == 2


def test_legacy_complete_card_without_verifiable_transformer_rejects_export():
    _, card = _card()
    legacy = _legacy_wrong_rules(card)
    legacy.binner = None
    with pytest.raises(ValueError, match="旧评分卡|重新拟合"):
        legacy.export_deployment_code("python")


def test_v2_rule_artifact_cannot_silently_downgrade_to_display_labels():
    _, card = _card()
    payload = card.export()
    payload["__meta__"].pop("rule_records")
    with pytest.raises(ValueError, match="完整结构化规则"):
        ScoreCard().load_rules(payload)
