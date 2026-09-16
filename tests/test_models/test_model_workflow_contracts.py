"""从使用者视角验证模型参数、训练过程、保存和 Optuna 的完整工作流。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hscredit.core.models import RandomForest, LogisticRegression, ModelTuner


@pytest.fixture
def data():
    X, y = make_classification(n_samples=90, n_features=4, random_state=12)
    return pd.DataFrame(X, columns=["年龄", "收入", "借款", "历史"]), y


@pytest.mark.parametrize(
    "name,extra",
    [
        ("XGBoost", {"max_bin": 32}),
        ("LightGBM", {"max_bin": 32}),
        ("CatBoost", {"thread_count": 1}),
        ("NGBoost", {"tol": 1e-4}),
        ("RandomForest", {"max_samples": 0.8}),
    ],
)
def test_clone_preserves_native_parameters_and_target(name, extra):
    from hscredit.core import models

    model = getattr(models, name)(n_estimators=3, n_jobs=1, target="FPD", **extra)
    copied = clone(model)
    assert is_classifier(model)
    assert copied.target == "FPD"
    for key, value in extra.items():
        assert copied.get_params()[key] == value
    copied.set_params(**extra)
    assert clone(copied).get_params()["target"] == "FPD"


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost", "NGBoost"])
def test_native_dict_is_not_mutated_and_set_params_wins(name, data):
    from hscredit.core import models

    cls = getattr(models, name)
    native = {"n_estimators": 3, "learning_rate": 0.03}
    model = cls(params=native, n_jobs=1)
    assert native == {"n_estimators": 3, "learning_rate": 0.03}
    model.set_params(learning_rate=0.08)
    model = clone(model).fit(*data)
    assert model.get_native_params()["learning_rate"] == 0.08
    assert native["learning_rate"] == 0.03


def test_pipeline_cross_validation_and_native_fit_arguments(data):
    model = Pipeline([("scale", StandardScaler()), ("model", RandomForest(n_estimators=4, n_jobs=1))])
    assert np.isfinite(cross_val_score(model, *data, scoring="roc_auc", cv=2)).all()
    with pytest.raises(TypeError):
        RandomForest(n_estimators=2, n_jobs=1).fit(*data, does_not_exist=True)


def test_failed_refit_records_error_and_invalidates_prediction(data):
    model = RandomForest(n_estimators=2, n_jobs=1).fit(*data)
    with pytest.raises(ValueError):
        model.fit(data[0], np.zeros(len(data[1])))
    assert [record["状态"] for record in model.training_history_] == ["完成", "失败"]
    assert model.training_summary_["错误类型"] == "ValueError"
    with pytest.raises(NotFittedError):
        model.predict_proba(data[0])


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
def test_validation_columns_are_aligned_and_importance_array_stays_in_feature_order(name, data):
    from hscredit.core import models

    X, y = data
    model = getattr(models, name)(n_estimators=4, n_jobs=1, random_state=12).fit(X, y, eval_set=[(X.iloc[:, ::-1], y)])
    np.testing.assert_allclose(model.feature_importances_, model.get_native_model().feature_importances_)
    assert model.evals_result_
    assert model.training_summary_["评估曲线"] == model.evals_result_


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM"])
@pytest.mark.parametrize("suffix", ["joblib", "json"])
def test_callable_metric_save_roundtrip_preserves_all_state(name, suffix, data, tmp_path):
    from hscredit.core import models

    cls = getattr(models, name)
    if name == "XGBoost":

        def metric(y, p):
            return float(np.mean((y - p) ** 2))

    else:

        def metric(y, p):
            return "误差", float(np.mean((y - p) ** 2)), False

    model = cls(n_estimators=4, n_jobs=1, eval_metric=metric, target="FPD").fit(*data, eval_set=[data])
    restored = cls.load(model.save(tmp_path / f"model.{suffix}"))
    np.testing.assert_allclose(restored.predict_proba(data[0]), model.predict_proba(data[0]))
    np.testing.assert_allclose(restored.predict_score(data[0]), model.predict_score(data[0]))
    assert restored.evals_result_ == model.evals_result_
    assert restored.target == "FPD"
    assert callable(restored.eval_metric)
    assert restored.training_summary_["状态"] == "完成"


def test_xgboost_ks_early_stopping_and_native_callbacks_survive_save(data, tmp_path):
    import xgboost
    from hscredit.core.models import XGBoost

    callback = xgboost.callback.EvaluationMonitor(period=100)
    model = XGBoost(
        n_estimators=5, n_jobs=1, eval_metric=["auc", "ks"], early_stopping_rounds=2, callbacks=[callback]
    ).fit(*data, eval_set=[data])
    restored = XGBoost.load(model.save(tmp_path / "ks.joblib"))
    assert restored.best_score_ is not None
    assert "ks" in restored.evals_result_["validation_0"]
    clone(restored).fit(*data, eval_set=[data])


def test_catboost_categorical_feature_after_target_and_native_aliases(data):
    from hscredit.core.models import CatBoost

    X, y = data
    X = X.assign(商品类别=np.where(X["年龄"] > 0, "甲", "乙"))
    X.insert(0, "FPD", y)
    model = CatBoost(
        n_estimators=3,
        max_depth=2,
        reg_lambda=2,
        cat_features=["商品类别"],
        target="FPD",
        n_jobs=1,
        auto_class_weights="Balanced",
    ).fit(X)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert model.get_native_params()["depth"] == 2
    assert model.get_native_params()["iterations"] == 3


def test_optuna_study_conditional_space_callbacks_and_resume(data, tmp_path):
    import optuna

    study = optuna.create_study(direction="maximize")
    observed = []

    def search(trial):
        depth = trial.suggest_int("max_depth", 1, 3)
        return {"max_depth": depth, "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, depth)}

    tuner = ModelTuner(
        RandomForest(n_estimators=3, n_jobs=1),
        search_space=search,
        study=study,
        metric="auc",
        cv=2,
        n_jobs=1,
        callbacks=[lambda study, trial: observed.append(trial.number)],
    )
    tuner.fit(*data, n_trials=2, show_progress_bar=False)
    best = tuner.get_best_model()
    assert tuner.get_best_model() is best
    assert observed == [0, 1]
    assert tuner.study_ is study
    assert len(tuner.get_trial_result(0)["各折"]) == 2
    assert tuner.get_oof_predictions()["预测概率"].notna().all()
    restored = ModelTuner.load(tuner.save(tmp_path / "tuner.joblib"))
    np.testing.assert_allclose(restored.get_best_model().predict_proba(data[0]), best.predict_proba(data[0]))
    restored.fit(*data, n_trials=1, show_progress_bar=False)
    assert len(restored.study_.trials) == 3


def test_tune_routes_optuna_options_and_nested_pipeline_parameters(data):
    import optuna

    study = optuna.create_study(direction="maximize")
    model = RandomForest(n_estimators=3, n_jobs=1, random_state=9)
    best = model.tune(
        *data,
        search_space={"max_depth": [2]},
        study=study,
        random_state=12,
        gc_after_trial=True,
        n_trials=1,
        cv=2,
        show_progress_bar=False,
        n_jobs=1,
    )
    assert best.tuner.study_ is study
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(calculate_stats=False))])
    tuner = ModelTuner(pipeline, search_space={"model__C": [0.1]}, cv=2, n_jobs=1)
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().named_steps["model"].C == 0.1


def test_group_cv_sample_weights_factory_and_disk_artifacts(data, tmp_path):
    X, y = data
    groups = np.arange(len(y)) % 3
    tuner = ModelTuner(
        RandomForest(n_estimators=3, n_jobs=1),
        search_space={},
        cv=GroupKFold(3),
        n_jobs=1,
        artifact_dir=str(tmp_path),
        fit_params_factory=lambda trial, fold: {},
    )
    tuner.fit(X, y, groups=groups, sample_weight=np.ones(len(y)), n_trials=1, show_progress_bar=False)
    for fold in tuner.get_trial_result(0)["各折"]:
        assert not set(groups[fold["训练位置"]]) & set(groups[fold["验证位置"]])
        assert fold["模型"].training_summary_["样本数"] == 60
    tuner.trial_results_.clear()
    assert len(tuner.get_trial_result(0)["各折"]) == 3


def test_pruned_and_failed_trials_keep_partial_history(data):
    import optuna

    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    tuner = ModelTuner(RandomForest(n_estimators=2, n_jobs=1), search_space={}, cv=2, n_jobs=1, pruner=AlwaysPrune())
    with pytest.raises(ValueError):
        tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.study_.trials[0].state == optuna.trial.TrialState.PRUNED
    assert not tuner.optimization_history_.empty
    assert "错误信息" not in tuner.study_.trials[0].user_attrs
    assert len(tuner.get_trial_result(0)["各折"]) == 1


def test_best_boosting_model_uses_full_input(data):
    from hscredit.core.models import LightGBM

    tuner = ModelTuner(LightGBM(n_estimators=5, n_jobs=1, early_stopping_rounds=2), search_space={}, cv=2, n_jobs=1)
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    best = tuner.get_best_model()
    assert best.validation_fraction == 0
    assert best.early_stopping_rounds is None
    assert "验证位置" not in best.training_summary_
    assert best.training_summary_["样本数"] == len(data[1])


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM"])
def test_native_aliases_override_wrapper_defaults(name, data):
    from hscredit.core import models

    aliases = {"eta": 0.02, "lambda": 2.0} if name == "XGBoost" else {"feature_fraction": 0.7, "lambda_l2": 2.0}
    model = getattr(models, name)(n_estimators=3, n_jobs=1, **aliases).fit(*data)
    native = model.get_native_params()
    assert native["reg_lambda"] == 2.0
    assert native["learning_rate" if name == "XGBoost" else "colsample_bytree"] == (0.02 if name == "XGBoost" else 0.7)


def test_native_xgboost_estimator_can_tune_with_early_stopping(data):
    from xgboost import XGBClassifier

    tuner = ModelTuner(
        XGBClassifier(n_estimators=4, n_jobs=1, early_stopping_rounds=2), search_space={}, cv=2, n_jobs=1
    )
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().predict_proba(data[0]).shape == (len(data[1]), 2)


def test_lightgbm_selected_custom_early_stopping_metric(data):
    from hscredit.core.models import LightGBM

    def constant_metric(y, p):
        return "固定指标", 0.5, True

    model = LightGBM(
        n_estimators=20,
        n_jobs=1,
        eval_metric=["auc", constant_metric],
        early_stopping_metric="固定指标",
        early_stopping_rounds=2,
    ).fit(*data, eval_set=[data])
    assert model.best_iteration_ == 1
    assert "固定指标" in next(iter(model.evals_result_.values()))


def test_native_training_verbosity_and_fit_metric_route(data):
    from hscredit.core.models import XGBoost, LightGBM

    for cls, metric in [(XGBoost, "auc"), (LightGBM, "auc")]:
        model = cls(n_estimators=4, n_jobs=1).fit(*data, eval_metric=metric, early_stopping_rounds=2, verbose=False)
        assert model.evals_result_


def test_logistic_reorders_target_and_recomputes_woe_directions(data):
    X, y = data
    model = LogisticRegression(calculate_stats=False, positive_woe_coef=True, target="FPD").fit(X.assign(FPD=1 - y), y)
    expected = model.predict_proba(X)
    np.testing.assert_allclose(model.predict_proba(X.iloc[:, ::-1].assign(FPD=y)), expected)
    model.fit(X, 1 - y)
    fresh = LogisticRegression(calculate_stats=False, positive_woe_coef=True).fit(X, 1 - y)
    np.testing.assert_allclose(model.predict_proba(X), fresh.predict_proba(X))


def test_sparse_training_and_warm_start_keep_native_behavior(data):
    from scipy.sparse import csr_matrix

    X, y = data
    model = RandomForest(n_estimators=2, n_jobs=1, warm_start=True).fit(csr_matrix(X), y)
    original = model.get_native_model()
    model.set_params(n_estimators=4).fit(csr_matrix(X), y)
    assert model.get_native_model() is original
    assert len(original.estimators_) == 4
    assert model.predict_proba(csr_matrix(X)).shape == (len(y), 2)


def test_custom_trial_objective_uses_native_optuna_protocol(data):
    def objective(trial):
        trial.set_user_attr("来源", "原生目标")
        return trial.suggest_float("max_features", 0.2, 0.8)

    tuner = ModelTuner(
        RandomForest(n_estimators=2, n_jobs=1), trial_objective=objective, metric="auc", search_space={}, cv=2, n_jobs=1
    )
    tuner.fit(*data, n_trials=2, show_progress_bar=False)
    assert tuner.best_params_["max_features"] == tuner.study_.best_params["max_features"]
    assert tuner.study_.best_trial.user_attrs["来源"] == "原生目标"


def test_conditional_space_with_model_object_uses_artifacts(data, tmp_path):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    def search(trial):
        depth = trial.suggest_int("depth", 1, 2)
        return {"estimator": DecisionTreeClassifier(max_depth=depth), "n_estimators": 2}

    tuner = ModelTuner(
        AdaBoostClassifier,
        search_space=search,
        cv=2,
        n_jobs=1,
        storage=f"sqlite:///{(tmp_path / 'study.db').as_posix()}",
        artifact_dir=str(tmp_path / "trials"),
    )
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert isinstance(tuner.best_params_["estimator"], DecisionTreeClassifier)
    tuner.trial_results_.clear()
    assert isinstance(tuner._get_params_from_trial(tuner.best_trial_)["estimator"], DecisionTreeClassifier)


def test_failed_serialization_keeps_previous_file(data, tmp_path):
    class Unserializable:
        def __reduce__(self):
            raise RuntimeError("拒绝序列化")

    model = RandomForest(n_estimators=2, n_jobs=1).fit(*data)
    path = tmp_path / "model.joblib"
    model.save(path)
    previous = path.read_bytes()
    model.extra = Unserializable()
    with pytest.raises(RuntimeError, match="拒绝序列化"):
        model.save(path)
    assert path.read_bytes() == previous


def test_logistic_evaluation_honors_requested_metrics_and_rejects_errors(data):
    model = LogisticRegression(calculate_stats=False).fit(*data)
    assert set(model.evaluate(*data, metrics=["auc", "ks"])) == {"AUC", "KS"}
    with pytest.raises(ValueError, match="不支持的评估指标"):
        model.evaluate(*data, metrics=["拼错的指标"])


@pytest.mark.parametrize(
    "name,initial,space,key,value",
    [
        ("XGBoost", {"eta": 0.01}, {"learning_rate": [0.15]}, "learning_rate", 0.15),
        ("LightGBM", {"num_iterations": 8}, {"n_estimators": [3]}, "n_estimators", 3),
        ("CatBoost", {"n_estimators": 8}, {"iterations": [3]}, "iterations", 3),
    ],
)
def test_tuning_does_not_restore_stale_native_aliases(name, initial, space, key, value, data):
    from hscredit.core import models

    model = getattr(models, name)(params=initial, n_jobs=1)
    best = model.tune(
        *data, search_space=space, n_trials=1, cv=2, n_jobs=1, early_stopping_rounds=None, show_progress_bar=False
    )
    assert best.get_native_params()[key] == value


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
def test_bare_native_model_load_restores_field_names_and_replaces_old_schema(name, data, tmp_path):
    from hscredit.core import models

    cls = getattr(models, name)
    X, y = data
    model = cls(n_estimators=3, n_jobs=1).fit(X, y)
    native = model.get_native_model()
    path = tmp_path / ("model.json" if name == "XGBoost" else "model.native")
    if name == "LightGBM":
        native.booster_.save_model(str(path))
    else:
        native.save_model(str(path))
    loaded = cls(n_estimators=2, n_jobs=1).fit(X.iloc[:, :2], y).load_model(str(path))
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:, ::-1]), model.predict_proba(X))
    assert loaded.feature_names_in_ == X.columns.tolist()


def test_lightgbm_custom_loss_native_roundtrip_preserves_probability_and_score(data, tmp_path):
    from scipy.special import expit
    from hscredit.core.models import LightGBM

    def loss(y, margin):
        probability = expit(margin)
        return probability - y, probability * (1 - probability)

    model = LightGBM(n_estimators=4, n_jobs=1, objective=loss).fit(*data)
    path = tmp_path / "custom.txt"
    model.save_model(str(path))
    restored = LightGBM().load_model(str(path))
    np.testing.assert_allclose(restored.predict_proba(data[0]), model.predict_proba(data[0]))
    np.testing.assert_allclose(restored.predict_score(data[0]), model.predict_score(data[0]))


def test_json_metadata_can_describe_nested_callable_parameters(data, tmp_path):
    from hscredit.core.models import LightGBM

    model = LightGBM(n_estimators=3, n_jobs=1, eval_metric=["auc", lambda y, p: ("自定义", 0.5, True)]).fit(
        *data, eval_set=[data]
    )
    restored = LightGBM.load(model.save(tmp_path / "nested.json"))
    assert callable(restored.get_params()["eval_metric"][1])


def test_boosting_pipeline_keeps_fold_history_and_refits_all_rows(data):
    from hscredit.core.models import LightGBM

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LightGBM(n_estimators=5, n_jobs=1, early_stopping_rounds=2)),
        ]
    )
    tuner = ModelTuner(pipeline, search_space={"model__max_depth": [2]}, cv=2, n_jobs=1)
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.get_trial_result(0)["各折"][0]["评估曲线"]
    best = tuner.get_best_model()
    assert best.named_steps["model"].validation_fraction == 0
    assert "验证位置" not in best.named_steps["model"].training_summary_
    assert best.predict_proba(data[0]).shape == (len(data[1]), 2)


def test_optuna_callable_space_retains_parameter_plots(data):
    pytest.importorskip("plotly")
    tuner = ModelTuner(
        RandomForest(n_estimators=2, n_jobs=1),
        search_space=lambda trial: {"max_depth": trial.suggest_int("max_depth", 1, 3)},
        cv=2,
        n_jobs=1,
    )
    tuner.fit(*data, n_trials=2, show_progress_bar=False)
    assert tuner.plot_slice(params=["max_depth"]).data


def test_catboost_pool_validation_is_not_broken_by_training_record(data, tmp_path):
    from catboost import Pool
    from hscredit.core.models import CatBoost

    X, y = data
    model = CatBoost(n_estimators=3, n_jobs=1).fit(X, y, eval_set=Pool(X, y))
    restored = CatBoost.load(model.save(tmp_path / "pool.joblib"))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    assert restored.training_summary_["训练参数"]["eval_set"]["容器类型"] == "CatBoost.Pool"


def test_legacy_custom_model_and_list_input_remain_supported(data):
    class CustomClassifier:
        def __init__(self, bias=0.0):
            self.bias = bias

        def fit(self, X, y):
            if hasattr(self, "rate"):
                raise RuntimeError("重复训练了同一个折模型")
            self.classes_ = np.array([0, 1])
            self.rate = float(np.mean(y))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), self.rate + self.bias)
            return np.column_stack([1 - p, p])

    X, y = data
    tuner = ModelTuner(CustomClassifier, search_space={"bias": [0.0]}, cv=2, n_jobs=1)
    tuner.fit(X.to_numpy().tolist(), y.tolist(), n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().predict_proba(X).shape == (len(y), 2)
