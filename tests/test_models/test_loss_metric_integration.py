"""损失配套指标在模型训练、独立评估和交叉验证中的集成契约。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from hscredit.core.models.losses import GiniMetric, WeightedBCELoss


@pytest.fixture(scope="module")
def loan_data():
    """按仓库约定使用真实放款数据与三个建模字段。"""
    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    data = pd.read_excel(path, usecols=features + ["FPD"], nrows=500)
    X = data[features].astype(float).fillna(0)
    y = data["FPD"].to_numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=19)
    return X_train, X_valid, y_train, y_valid


def _model(name, **kwargs):
    from hscredit.core import models

    pytest.importorskip(name.lower())
    options = dict(n_estimators=5, n_jobs=1, random_state=19, validation_fraction=0)
    if name == "CatBoost":
        options.pop("n_estimators")
        options.update(iterations=5, allow_writing_files=False, use_best_model=False)
    return getattr(models, name)(**options, **kwargs)


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
@pytest.mark.parametrize("custom_objective", [False, True])
def test_loss_metric_training_curve_matches_probability_evaluation(name, custom_objective, loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    loss = WeightedBCELoss(pos_weight=1.7)
    metric = loss.metric()
    options = {"objective": loss} if custom_objective else {}
    model = _model(name, eval_metric=metric, **options).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    probability = model.predict_proba(X_valid)[:, 1]
    expected = metric.evaluate(y_valid, probability)
    dataset = next(key for key in model.evals_result_ if key.startswith("valid"))
    assert model.evals_result_[dataset][metric.name][-1] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
def test_custom_loss_automatically_selects_matching_metric(name, loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    loss = WeightedBCELoss()
    model = _model(name, objective=loss, eval_metric=None).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    dataset = next(key for key in model.evals_result_ if key.startswith("valid"))
    assert loss.metric().name in model.evals_result_[dataset]


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM"])
def test_training_metric_preserves_validation_weights(name, loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    loss = WeightedBCELoss(pos_weight=1.4)
    metric = loss.metric()
    weights = np.where(y_valid == 1, 6.0, 1.0)
    weight_param = "sample_weight_eval_set" if name == "XGBoost" else "eval_sample_weight"
    model = _model(name, objective=loss, eval_metric=metric).fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)], **{weight_param: [weights]}
    )
    dataset = next(key for key in model.evals_result_ if key.startswith("valid"))
    expected = metric.evaluate(y_valid, model.predict_proba(X_valid)[:, 1], sample_weight=weights)
    assert model.evals_result_[dataset][metric.name][-1] == pytest.approx(expected, abs=1e-6)


def test_lightgbm_accepts_mixed_metrics_and_selects_object_by_name(loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    metric = WeightedBCELoss().metric()
    model = _model(
        "LightGBM", eval_metric=["auc", metric], early_stopping_metric=metric.name, early_stopping_rounds=2
    ).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    assert set(model.evals_result_["valid_0"]) == {"auc", metric.name}


@pytest.mark.parametrize("loss_first", [True, False])
def test_lightgbm_default_early_stopping_preserves_requested_metric_order(loss_first, loan_data):
    from hscredit import LightGBM

    X_train, X_valid, y_train, y_valid = loan_data
    metric = WeightedBCELoss().metric()
    metrics = [metric, "auc"] if loss_first else ["auc", metric]
    model = LightGBM(
        n_estimators=40, n_jobs=1, random_state=19, learning_rate=0.05,
        eval_metric=metrics, early_stopping_rounds=3,
    ).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    curves = model.evals_result_["valid_0"]
    expected = np.argmin(curves[metric.name]) if loss_first else np.argmax(curves["auc"])
    assert model.best_iteration_ == expected + 1


def test_xgboost_maximizes_metric_when_early_stopping_is_requested(loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    metric = GiniMetric()
    model = _model("XGBoost", eval_metric=metric, early_stopping_rounds=2).fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)]
    )
    values = model.evals_result_["validation_0"][metric.name]
    assert float(model.best_score_) == pytest.approx(max(values))


def test_catboost_preserves_default_auc_for_custom_loss(loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    model = _model("CatBoost", objective=WeightedBCELoss()).fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)]
    )
    assert "AUC" in model.evals_result_["validation"]


@pytest.mark.parametrize("entry", ["constructor", "params", "fit"])
def test_catboost_metric_list_has_consistent_entry_points(entry, loan_data):
    X_train, X_valid, y_train, y_valid = loan_data
    metric = WeightedBCELoss(pos_weight=2).metric()
    metrics = [metric, "auc"]
    options, fit_options = {}, {}
    if entry == "constructor":
        options["eval_metric"] = metrics
    elif entry == "params":
        options["params"] = {"eval_metric": metrics}
    else:
        options["eval_metric"] = "auc"
        options["params"] = {"eval_metric": "Logloss"}
        fit_options["eval_metric"] = metrics
    model = _model("CatBoost", **options).fit(
        X_train, y_train, eval_set=[(X_valid, y_valid)], **fit_options
    )
    results = model.evals_result_["validation"]
    assert metric.name in results and "AUC" in results
    expected = metric.evaluate(y_valid, model.predict_proba(X_valid))
    assert results[metric.name][-1] == pytest.approx(expected)


def test_tuner_infers_each_metric_direction_in_multi_objective_search():
    from hscredit.core.models import ModelTuner

    loss_metric, business_metric = WeightedBCELoss().metric(), GiniMetric()
    tuner = ModelTuner(SklearnLogisticRegression, search_space={}, metric=[loss_metric, business_metric])
    assert tuner.directions == ["minimize", "maximize"]
    assert tuner.metric_names == [loss_metric.name, business_metric.name]


def test_loss_scorer_and_tuner_use_correct_direction(loan_data):
    from hscredit.core.models import ModelTuner
    from hscredit.core.models.tuning.tuning import Metric

    X_train, _, y_train, _ = loan_data
    metric = WeightedBCELoss().metric()
    tuner = ModelTuner(
        SklearnLogisticRegression, search_space={"C": [0.1, 1.0]},
        fixed_params={"max_iter": 1000}, metric=metric, cv=2, n_jobs=1,
    )
    assert tuner.directions == ["minimize"]
    assert tuner.direction == "minimize"
    assert tuner.metric_names == [metric.name]
    assert Metric(metric).direction == "minimize"
    scores = cross_val_score(SklearnLogisticRegression(max_iter=1000), X_train, y_train,
                             scoring=metric.to_scorer(), cv=2)
    assert np.isfinite(scores).all()
    assert np.all(scores < 0)
    tuner.fit(X_train, y_train, n_trials=1, show_progress_bar=False)
    assert np.isfinite(tuner.best_score_)
    assert tuner.best_score_ > 0
