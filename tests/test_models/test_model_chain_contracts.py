"""搜索、重训、评估和报告之间的参数传递契约。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit import LogisticRegression, ModelTuner, ProbabilityScoreCard, RandomForest, get_metric
from hscredit.report import ModelReport


@pytest.fixture
def frames():
    X, y = make_classification(n_samples=120, n_features=3, n_redundant=0, random_state=23)
    data = pd.DataFrame(X, columns=["评分", "多头", "收入"]).assign(FPD=y)
    return data.iloc[:70].copy(), data.iloc[70:95].copy(), data.iloc[95:].copy()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LogisticRegression(target="FPD", calculate_stats=False),
        lambda: RandomForest(target="FPD", n_estimators=3, n_jobs=1),
    ],
)
def test_model_target_flows_through_search_evaluation_and_report(factory, frames):
    train, valid, test = frames
    metric = get_metric("auc")
    tuner = ModelTuner(factory(), search_space={}, metric=metric, n_jobs=1, random_state=23)
    tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    assert best.target == "FPD"
    assert "FPD" not in best.feature_names_in_
    value = best.evaluate(test, metrics=metric)[metric.name]
    report = best.model_report(datasets={"独立测试": test}, metrics=metric, verbose=False)
    assert isinstance(report, ModelReport)
    assert report.get_metrics().set_index("统计项").loc[metric.name, "独立测试"] == pytest.approx(value)
    assert "调参记录" in report.get_training_process()
    compatible = best.report(datasets={"独立测试": test}, metrics=metric, verbose=False)
    assert compatible.get_metrics().set_index("统计项").loc[metric.name, "独立测试"] == pytest.approx(value)


def test_explicit_tuner_target_reaches_constructed_model(frames):
    train, valid, _ = frames
    tuner = ModelTuner(RandomForest, target="FPD", fixed_params={"n_estimators": 3}, search_space={}, n_jobs=1)
    tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().target == "FPD"


def test_pipeline_inherits_final_estimator_target(frames):
    train, valid, test = frames
    pipeline = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(target="FPD", calculate_stats=False))])
    tuner = ModelTuner(pipeline, search_space={}, metric="auc", n_jobs=1)
    tuner.fit(train, eval_set=(valid.drop(columns="FPD"), valid.FPD), n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    assert best.n_features_in_ == 3
    assert best.predict_proba(test.drop(columns="FPD")).shape == (25, 2)
    assert ModelReport(best, datasets={"测试": test}, n_jobs=1).get_metrics().shape[1] >= 2


def test_search_auto_probability_card_refits_fitted_base_model(frames):
    train, valid, _ = frames
    features = train.columns.drop("FPD")
    base = LogisticRegression(calculate_stats=False).fit(train[features], 1 - train.FPD)
    card = ProbabilityScoreCard(model=base, prefit=None, target="FPD")
    tuner = ModelTuner(card, target="FPD", search_space={}, metric="auc", n_jobs=1)
    tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    reference = LogisticRegression(calculate_stats=False).fit(train[features], train.FPD)
    np.testing.assert_allclose(best.predict_proba(valid[features]), reference.predict_proba(valid[features]))
    assert base.fit_revision_ == 1


def test_prefit_probability_card_rejects_ineffective_base_search(frames):
    train, valid, _ = frames
    base = LogisticRegression(calculate_stats=False).fit(train.drop(columns="FPD"), train.FPD)
    tuner = ModelTuner(
        ProbabilityScoreCard(model=base, prefit=True, target="FPD"),
        search_space={"model__C": [0.5, 2.0]},
        metric="auc",
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="prefit=True"):
        tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False, catch=())


def test_probability_card_tune_routes_optimizer_arguments(frames):
    train, valid, _ = frames
    model = ProbabilityScoreCard(model=LogisticRegression(calculate_stats=False), target="FPD")
    best = model.tune(
        train,
        search_space={"model__C": [1.0]},
        n_trials=1,
        eval_set={"验证": valid},
        catch=(),
        optimize_kwargs={"show_progress_bar": False},
        n_jobs=1,
    )
    assert best.target == "FPD"


def test_help_sees_public_model_class_documentation():
    from hscredit import RulesClassifier

    for cls in (LogisticRegression, ProbabilityScoreCard, RulesClassifier):
        assert cls.__doc__ and ":param target:" in cls.__doc__


def test_tuner_docstring_end_to_end_example_is_executable():
    import doctest
    import inspect

    doc = inspect.getdoc(ModelTuner.__init__).split("从数据、选参、最终模型到评估及 Excel 的可直接执行示例::", 1)[1]
    runner = doctest.DocTestRunner()
    sample = doctest.DocTestParser().get_doctest(doc, {}, "完整建模链路", None, 0)
    runner.run(sample)
    assert runner.failures == 0
    assert runner.tries >= 15


@pytest.mark.parametrize("kind", ["calibration", "scorecard", "pipeline"])
def test_frozen_components_do_not_silently_ignore_tuned_parameters(kind, frames):
    from hscredit import ProbabilityCalibrator, ScoreCard

    train, valid, _ = frames
    base = LogisticRegression(calculate_stats=False).fit(train.drop(columns="FPD"), train.FPD)
    if kind == "calibration":
        model = ProbabilityCalibrator(model=base, calib_ratio=None, target="FPD")
        key = "model__C"
    elif kind == "scorecard":
        model = ScoreCard(lr_model=base, target="FPD")
        key = "lr_model__C"
    else:
        model = Pipeline([("card", ProbabilityScoreCard(model=base, prefit=True, target="FPD"))])
        key = "card__model__C"
    tuner = ModelTuner(model, search_space={key: [0.5]}, metric="auc", n_jobs=1)
    with pytest.raises(ValueError, match="冻结底层模型"):
        tuner.fit(train, eval_set={"验证": valid}, n_trials=1, show_progress_bar=False, catch=())


def test_calibrated_report_keeps_input_schema_and_importance(frames):
    from hscredit import ProbabilityCalibrator, CalibratedModel

    train, valid, test = frames
    base = RandomForest(n_estimators=3, target="FPD", n_jobs=1).fit(train)
    cal = ProbabilityCalibrator(model=base, calib_ratio=None, target="FPD").fit(valid)
    model = CalibratedModel(base, cal)
    report = ModelReport(model, datasets={"测试": test}, n_jobs=1)
    assert list(report.feature_names) == ["评分", "多头", "收入"]
    assert not report.get_feature_importance().empty
