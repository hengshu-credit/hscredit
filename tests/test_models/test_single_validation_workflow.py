"""金融建模默认单次验证、显式多折和最终单模型契约。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit import ModelTuner, get_metric


class TrackingClassifier(ClassifierMixin, BaseEstimator):
    fits = []

    def __init__(self, bias=0.0):
        self.bias = bias

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.rows_ = len(y)
        self.weights_ = sample_weight
        type(self).fits.append(len(y))
        self.rate_ = np.average(y, weights=sample_weight)
        return self

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.rate_ + self.bias, 0.001, 0.999))
        return np.column_stack([1 - p, p])


@pytest.fixture
def data():
    TrackingClassifier.fits = []
    return pd.DataFrame({"甲": np.arange(80), "乙": np.arange(80) * 2}), np.tile([0, 1], 40)


def tuner(**kwargs):
    return ModelTuner(
        TrackingClassifier,
        search_space={"bias": [0.0]},
        metric=get_metric("logloss"),
        random_state=42,
        n_jobs=1,
        **kwargs,
    )


def fit(obj, X, y, **kwargs):
    obj.fit(X, y, n_trials=1, show_progress_bar=False, **kwargs)
    return obj


def test_default_holdout_is_one_fit_then_one_final_model(data):
    obj = fit(tuner(), *data)
    assert TrackingClassifier.fits == [64]
    assert obj.validation_summary_["验证方式"] == "单次留出"
    assert len(obj.get_validation_predictions()) == 16
    assert len(obj.get_oof_predictions()) == 80
    model = obj.get_best_model()
    assert model.rows_ == 80
    assert TrackingClassifier.fits == [64, 80]
    assert obj.get_best_model() is model


def test_external_validation_never_enters_final_training_and_weights_align(data):
    X, y = data
    external = pd.DataFrame({"乙": np.arange(100), "甲": np.arange(100)})
    valid_y = np.tile([0, 1], 50)
    valid_weight = np.arange(1, 101)
    obj = fit(
        tuner(),
        X,
        y,
        eval_set=[(external, valid_y)],
        sample_weight=np.arange(1, 81),
        evaluation_weight=np.arange(80, 0, -1),
        validation_weight=valid_weight,
    )
    record = obj.get_trial_result(0)["各折"][0]
    expected = get_metric("logloss").evaluate(valid_y, record["预测概率"], sample_weight=valid_weight)
    assert list(record["指标"].values()) == pytest.approx([expected])
    assert record["评估口径"] == "显式评估权重"
    assert obj.get_best_model().rows_ == 80
    assert len(obj.get_validation_predictions()) == 100
    assert (obj.get_validation_predictions()["验证来源"] == "独立验证集").all()
    with pytest.raises(ValueError, match="get_validation_predictions"):
        obj.get_oof_predictions()
    assert obj.evaluate_study_trials().iloc[0]["交叉熵"] == pytest.approx(expected)


def test_external_dataframe_target_and_reordered_columns(data):
    X, y = data
    external = X.iloc[:20, ::-1].copy()
    external["标签"] = y[:20]
    obj = fit(tuner(target="标签"), X, y, eval_set=external)
    assert list(obj._validation_data[0]) == list(X)
    assert len(obj.get_validation_predictions()) == 20


def test_explicit_cv_preserves_folds_and_final_single_model(data):
    obj = fit(tuner(cv=4), *data)
    assert TrackingClassifier.fits == [60] * 4
    assert len(obj.get_fold_models()) == 4
    assert (obj.get_validation_predictions()["验证次数"] == 1).all()
    assert obj.get_best_model().rows_ == 80
    assert TrackingClassifier.fits == [60] * 4 + [80]


def test_groups_stay_disjoint_in_default_single_split(data):
    obj = fit(tuner(), *data, groups=np.repeat(np.arange(20), 4))
    train, valid = obj._cv_splits[0]
    assert not set(train // 4).intersection(valid // 4)


@pytest.mark.parametrize("cv", [0, 1, True])
def test_invalid_cv_does_not_mean_training_only(data, cv):
    with pytest.raises(ValueError, match="cv"):
        fit(tuner(cv=cv), *data)


def test_validation_conflicts_and_missing_features_fail_before_fit(data):
    X, y = data
    with pytest.raises(ValueError, match="不能同时"):
        fit(tuner(cv=2), X, y, eval_set=[(X, y)])
    with pytest.raises(ValueError, match="一起提供"):
        fit(tuner(), X, y, validation_weight=np.ones(20))
    with pytest.raises(ValueError, match="缺少训练字段"):
        fit(tuner(), X, y, eval_set=[(X[["甲"]], y)])
    assert not TrackingClassifier.fits


def test_resume_save_and_release_external_validation(data, tmp_path):
    X, y = data
    obj = fit(tuner(retention="disk", artifact_dir=str(tmp_path / "trials")), X, y, eval_set=[(X[:20], y[:20])])
    obj.save(tmp_path / "tuner.pkl")
    restored = ModelTuner.load(tmp_path / "tuner.pkl")
    np.testing.assert_allclose(
        obj.get_validation_predictions()["预测概率"], restored.get_validation_predictions()["预测概率"]
    )
    fit(restored, X, y, eval_set=[(X[:20], y[:20])])
    assert len(restored.study_.trials) == 2
    with pytest.raises(ValueError, match="续跑数据"):
        fit(restored, X, y, eval_set=[(X[:22], y[:22])])
    best = restored.get_best_model()
    restored.release_training_data()
    assert restored.get_best_model() is best
    with pytest.raises(ValueError, match="释放"):
        restored.get_validation_predictions()


def test_lightgbm_uses_holdout_for_monitoring_without_inner_split(data):
    pytest.importorskip("lightgbm")
    from hscredit import LightGBM

    X, y = data
    model = LightGBM(n_estimators=12, min_child_samples=2, early_stopping_rounds=3, n_jobs=1, random_state=42)
    best = model.tune(
        X,
        y,
        search_space={"max_depth": [2]},
        eval_set=[(X[:20], y[:20])],
        metric=get_metric("logloss"),
        n_trials=1,
        retention="full",
        n_jobs=1,
    )
    trial_model = best.tuner.get_fold_models()[0]
    assert not hasattr(trial_model, "_eval_train_indices_")
    assert trial_model.evals_result_
    assert best.training_summary_["样本数"] == 80
    assert best.get_params()["early_stopping_rounds"] is None


def test_lr_direction_and_model_tune_validation_forwarding(data):
    from hscredit import LogisticRegression

    X, y = data
    best = LogisticRegression(calculate_stats=False, max_iter=100).tune(
        X,
        y,
        search_space={"C": [1.0]},
        metric=get_metric("logloss"),
        eval_set=[(X[:20], y[:20])],
        n_trials=1,
        n_jobs=1,
    )
    assert best.tuner.direction == "minimize"
    assert best.tuner.validation_summary_["验证方式"] == "独立验证集"


def test_pipeline_holdout_preserves_preprocessing_and_weight_routing(data):
    X, y = data
    model = Pipeline([("scale", StandardScaler()), ("model", TrackingClassifier())])
    obj = ModelTuner(model, search_space={}, metric=get_metric("logloss"), n_jobs=1)
    fit(obj, X, y, eval_set=[(X[:20], y[:20])], sample_weight=np.arange(1, 81))
    np.testing.assert_array_equal(obj.get_fold_models()[0][-1].weights_, np.arange(1, 81))
    assert obj.get_best_model()[-1].rows_ == 80


def test_point_evaluation_has_no_stale_validation_context(data):
    X, y = data
    obj = fit(tuner(), X, y, eval_set=[(X[:20], y[:20])])
    table = obj.evaluate_trials(X, y, trial_points=[{"bias": 0.0}], eval_set=[(X[:40], y[:40])])
    assert len(table) == 1
    assert TrackingClassifier.fits[-1] == 80
    obj.evaluate_trials(X, y, trial_points=[{"bias": 0.0}])
    assert TrackingClassifier.fits[-1] == 64


def test_unfitted_prediction_tables_have_actionable_errors():
    obj = tuner()
    with pytest.raises(ValueError, match="试验"):
        obj.get_oof_predictions()
    with pytest.raises(ValueError, match="fit"):
        obj.get_validation_predictions()


def test_auto_tuner_default_is_single_validation():
    from hscredit import AutoTuner

    assert AutoTuner.create("rf", n_jobs=1).cv is None


@pytest.mark.parametrize("mode", ["pairs", "named_pairs", "frames", "named_frames", "mixed", "pair", "frame"])
def test_eval_set_supports_report_dataset_forms_and_names(data, mode):
    from hscredit.report import ModelReport

    X, y = data
    first, last = X.iloc[:30].copy(), X.iloc[-20:].copy()
    first["标签"], last["标签"] = y[:30], y[-20:]
    choices = {
        "pairs": [(first, y[:30]), (last, y[-20:])],
        "named_pairs": {"开发集": (first, y[:30]), "验证集": (last, y[-20:])},
        "frames": [first, last],
        "named_frames": {"开发集": first, "验证集": last},
        "mixed": {"开发集": first, "验证集": (last.assign(标签=0), y[-20:])},
        "pair": (last, y[-20:]),
        "frame": last,
    }
    inputs = choices[mode]
    obj = fit(tuner(target="标签"), X, y, eval_set=inputs)
    if mode in ("pair", "frame"):
        report_inputs = [inputs]
    else:
        report_inputs = inputs
    report = ModelReport(obj.get_best_model(), datasets=report_inputs, target="标签", n_jobs=1)
    assert obj.validation_summary_["评估集名称"] == list(report._datasets)
    assert obj.validation_summary_["搜索评分数据集"] == list(report._datasets)[-1]
    assert (obj.get_validation_predictions()["数据集"] == list(report._datasets)[-1]).all()
    np.testing.assert_array_equal(obj.get_validation_predictions()["真实标签"], y[-20:])
    assert all(list(eval_X) == ["甲", "乙"] for eval_X, _ in obj._eval_set)
    assert obj.get_trial_result(0)["各折"][0]["评估集名称"] == list(report._datasets)


def test_search_scores_only_last_dataset_and_keeps_names_on_resume(data, tmp_path):
    X, y = data
    labels = np.array([0] * 5 + [1] * 15)
    metric = get_metric(lambda y, p: y.mean(), name="验证坏率", greater_is_better=False)
    obj = ModelTuner(TrackingClassifier, search_space={}, metric=metric, n_jobs=1)
    sets = {"训练监控": (X, y), "时间验证": (X[:20], labels)}
    fit(obj, X, y, eval_set=sets)
    assert obj.best_score_ == 0.75
    assert obj.evaluate_study_trials().iloc[0]["验证坏率"] == 0.75
    assert obj.evaluate_trials(X, y, trial_points=[{}], eval_set=sets).iloc[0]["验证坏率"] == 0.75
    obj.save(tmp_path / "named.pkl")
    restored = ModelTuner.load(tmp_path / "named.pkl")
    fit(restored, X, y, eval_set=sets)
    assert restored.validation_summary_["搜索评分数据集"] == "时间验证"
    changed = {"训练监控": (X + 1, y), "时间验证": (X[:20], labels)}
    with pytest.raises(ValueError, match="续跑数据"):
        fit(restored, X, y, eval_set=changed)


@pytest.mark.parametrize("invalid", [[], {}, {"": None}, {1: None, "1": None}, "wrong"])
def test_invalid_dataset_containers_fail_before_search(data, invalid):
    obj = tuner()
    with pytest.raises(ValueError, match="eval_set"):
        fit(obj, *data, eval_set=invalid)
    assert obj.study_ is None
    assert not TrackingClassifier.fits


@pytest.mark.parametrize("model_name", ["RandomForest", "LightGBM", "XGBoost"])
def test_direct_fit_shares_named_eval_set_contract(data, model_name):
    import hscredit

    X, y = data
    frame = X.assign(标签=y)
    model = getattr(hscredit, model_name)(n_estimators=3, target="标签", n_jobs=1, random_state=42)
    model.fit(frame, eval_set={"训练监控": frame, "独立验证": (frame.iloc[:20, ::-1].assign(标签=0), y[:20])})
    assert model.eval_names_ == ["训练监控", "独立验证"]
    assert model.training_summary_["评估集名称"] == model.eval_names_
    assert model.n_features_in_ == 2
    assert len(model.evals_result_) == 2
    model.fit(frame, eval_set=[frame[:30], frame[-20:]])
    assert model.eval_names_ == ["数据集1", "数据集2"]


@pytest.mark.parametrize("model_name", ["LightGBM", "XGBoost"])
def test_tuning_passes_all_monitor_sets_and_refits_only_training(data, model_name):
    import hscredit

    X, y = data
    model = getattr(hscredit, model_name)(n_estimators=5, n_jobs=1, random_state=42)
    best = model.tune(
        X,
        y,
        eval_set={"训练监控": (X, y), "独立验证": (X[:20], y[:20])},
        search_space={"max_depth": [2]},
        metric=get_metric("logloss"),
        n_trials=1,
        n_jobs=1,
    )
    fold = best.tuner.get_fold_models()[0]
    assert fold.eval_names_ == ["训练监控", "独立验证"]
    assert len(fold.evals_result_) == 2
    assert len(best.tuner.get_validation_predictions()) == 20
    assert best.training_summary_["样本数"] == len(y)
    assert best.evals_result_ == {}


def test_legacy_parameter_rejected_with_eval_set_hint(data):
    from hscredit import LightGBM, LogisticRegression

    for model in (LightGBM(), LogisticRegression(), tuner()):
        method = model.fit if isinstance(model, ModelTuner) else model.tune
        with pytest.raises(TypeError, match="eval_set"):
            method(*data, validation_data=data)


def test_ngboost_single_validation_backend_receives_last_named_set(data, monkeypatch):
    import hscredit.core.models.boosting.ngboost_model as module

    X, y = data
    seen = {}

    def native_fit(self, X, y, **kwargs):
        seen.update(kwargs)
        self.evals_result = {}
        return self

    monkeypatch.setattr(module.NGBClassifier, "fit", native_fit)
    monkeypatch.setattr(module.NGBClassifier, "predict_proba", lambda self, X: np.full((len(X), 2), 0.5))
    monkeypatch.setattr(module.NGBClassifier, "staged_predict_proba", lambda self, X: [np.full((len(X), 2), 0.5)])
    model = module.NGBoost(n_estimators=2, random_state=42, n_jobs=1)
    model.fit(X, y, eval_set={"训练监控": (X, y), "时间验证": (X[-20:], y[-20:])})
    np.testing.assert_array_equal(seen["X_val"], X[-20:].to_numpy())
    np.testing.assert_array_equal(seen["Y_val"], y[-20:])
    assert model.eval_names_ == ["训练监控", "时间验证"]
    assert set(model.get_eval_results()["数据集"]) == {"训练监控", "时间验证"}


def test_eval_set_accepts_plain_matrix_lists_without_confusing_dataset_lists(data):
    X, y = data
    obj = fit(tuner(), X.to_numpy(), y, eval_set=(X[:20].values.tolist(), y[:20].tolist()))
    assert len(obj.get_validation_predictions()) == 20


def test_eval_set_duplicate_fit_configuration_does_not_silently_override(data):
    from hscredit import LightGBM

    X, y = data
    obj = ModelTuner(LightGBM(n_estimators=3, n_jobs=1), search_space={}, n_jobs=1, fit_params={"eval_set": [(X, y)]})
    with pytest.raises(ValueError, match="重复配置"):
        fit(obj, X, y, eval_set=[(X[:20], y[:20])], catch=())
    assert obj.study_ is None


def test_named_eval_set_refit_can_explicitly_override_monitor_data(data):
    from hscredit import LightGBM

    X, y = data
    obj = ModelTuner(LightGBM(n_estimators=3, n_jobs=1), search_space={}, n_jobs=1)
    fit(obj, X, y, eval_set={"搜索验证": (X[:20], y[:20])})
    result = obj.get_best_model(full_data=False, eval_set={"重训验证": (X[-20:], y[-20:])})
    assert result.eval_names_ == ["重训验证"]
