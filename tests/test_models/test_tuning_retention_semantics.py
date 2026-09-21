"""资源保留策略不得改变搜索、折外预测、最佳模型和持久化语义。"""

import numpy as np
import pandas as pd
import pytest
import optuna
from sklearn.datasets import make_classification

from hscredit.core.models import ModelTuner, RandomForest


@pytest.fixture
def data():
    X, y = make_classification(n_samples=70, n_features=4, random_state=31)
    return pd.DataFrame(X, columns=list("abcd")), y


def tune(data, directory, retention=None, **options):
    tuner = ModelTuner(
        RandomForest(n_estimators=3, n_jobs=1, random_state=3),
        search_space={"max_depth": [1, 3]},
        trial_points=[{"max_depth": 1}, {"max_depth": 3}],
        metric="auc",
        cv=2,
        n_jobs=1,
        random_state=3,
        retention=retention,
        artifact_dir=str(directory),
        **options,
    )
    tuner.fit(*data, n_trials=2, show_progress_bar=False)
    return tuner


@pytest.mark.parametrize("mode", ["full", "predictions", "summary", "best", "disk"])
def test_retention_preserves_trial_scores_and_best_model(mode, data, tmp_path):
    full = tune(data, tmp_path / "full")
    candidate = tune(data, tmp_path / mode, mode)
    assert candidate.best_params_ == full.best_params_
    assert [t.value for t in candidate.study_.trials] == [t.value for t in full.study_.trials]
    np.testing.assert_allclose(
        candidate.get_best_model().predict_proba(data[0]), full.get_best_model().predict_proba(data[0])
    )
    if mode == "summary":
        with pytest.raises(ValueError, match="未保留"):
            candidate.get_oof_predictions()
    else:
        pd.testing.assert_frame_equal(candidate.get_oof_predictions(), full.get_oof_predictions())
    for number, record in candidate.trial_results_.items():
        has_models = any("模型" in fold for fold in record["各折"])
        assert has_models == (mode == "full" or (mode == "best" and number == candidate.best_trial_.number))
        if mode in {"summary", "disk"}:
            assert not any("预测概率" in fold or "训练位置" in fold for fold in record["各折"])


def test_disk_materialization_is_lazy_and_oof_does_not_recache_models(data, tmp_path):
    tuner = tune(data, tmp_path, "disk")
    result = tuner.get_trial_result(0)
    assert all("模型" in fold for fold in result["各折"])
    assert all("模型" not in fold for fold in tuner.trial_results_[0]["各折"])
    tuner.get_oof_predictions(0)
    assert all("预测概率" not in fold for fold in tuner.trial_results_[0]["各折"])
    tuner.trial_results_.clear()
    assert all("模型" in fold for fold in tuner.get_trial_result(0)["各折"])
    assert all("模型" not in fold for fold in tuner.trial_results_[0]["各折"])


@pytest.mark.parametrize("mode", ["full", "predictions", "summary", "best", "disk"])
def test_retention_save_restore_and_resume(mode, data, tmp_path):
    tuner = tune(data, tmp_path / "folds", mode)
    expected = tuner.get_best_model().predict_proba(data[0])
    restored = ModelTuner.load(tuner.save(tmp_path / "tuner.joblib"))
    np.testing.assert_allclose(restored.get_best_model().predict_proba(data[0]), expected)
    assert restored.get_best_model().tuner is restored
    restored.fit(*data, n_trials=1, show_progress_bar=False)
    assert len(restored.study_.trials) == 3
    assert restored.retention_ == mode


def test_releasing_inputs_keeps_cached_model_but_requires_data_for_refit(data, tmp_path):
    tuner = tune(data, tmp_path, "summary")
    best = tuner.get_best_model()
    expected = best.predict_proba(data[0])
    tuner.release_training_data()
    assert tuner._X is None and tuner._y is None and tuner._cv_splits is None
    assert tuner.get_best_model() is best
    np.testing.assert_allclose(best.predict_proba(data[0]), expected)
    with pytest.raises(ValueError, match="训练数据已释放"):
        tuner.get_best_model(refit=True)
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.get_best_model().predict_proba(data[0]).shape == expected.shape


def test_legacy_store_models_false_keeps_predictions(data, tmp_path):
    tuner = tune(data, tmp_path, store_models=False)
    assert tuner.retention_ == "predictions"
    assert "训练预测概率" in tuner.get_trial_result(0)["各折"][0]
    assert "模型" not in tuner.get_trial_result(0)["各折"][0]


@pytest.mark.parametrize("mode", ["summary", "disk", "best"])
def test_pruning_keeps_partial_semantics(mode, data, tmp_path):
    class StopAfterFirstFold(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    tuner = ModelTuner(
        RandomForest(n_estimators=2, n_jobs=1),
        search_space={},
        cv=2,
        n_jobs=1,
        retention=mode,
        artifact_dir=str(tmp_path),
        pruner=StopAfterFirstFold(),
    )
    with pytest.raises(ValueError, match="所有Trial"):
        tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert tuner.study_.trials[0].state == optuna.trial.TrialState.PRUNED
    record = tuner.get_trial_result(0, load=False)
    assert record["状态"] == "剪枝"
    assert len(record["各折"]) == 1 and record["各折"][0]["指标"]


def test_invalid_retention_is_rejected_before_training():
    with pytest.raises(ValueError, match="retention"):
        ModelTuner(RandomForest, retention="unknown")
    with pytest.raises(ValueError, match="artifact_dir"):
        ModelTuner(RandomForest, retention="disk")


def test_disk_artifacts_can_be_relocated_without_recaching_models(data, tmp_path, monkeypatch):
    import shutil
    from hscredit import utils

    tuner = tune(data, tmp_path / "original", "disk")
    expected = tuner.get_oof_predictions(0)
    path = tuner.save(tmp_path / "tuner.joblib")
    shutil.copytree(tmp_path / "original", tmp_path / "relocated")
    restored = ModelTuner.load(path, artifact_dir=tmp_path / "relocated")
    restored.trial_results_.clear()
    read_paths = []
    original_load = utils.load_pickle

    def recording_load(path, **kwargs):
        read_paths.append(str(path))
        return original_load(path, **kwargs)

    monkeypatch.setattr(utils, "load_pickle", recording_load)
    pd.testing.assert_frame_equal(restored.get_oof_predictions(0), expected)
    assert read_paths and all("relocated" in path for path in read_paths)
    assert all("模型" not in fold for fold in restored.trial_results_[0]["各折"])


def test_missing_fold_is_reported_instead_of_retraining(data, tmp_path):
    tuner = tune(data, tmp_path, "disk")
    tuner.trial_results_[0]["各折"][0]["制品路径"] = str(tmp_path / "missing.pkl")
    with pytest.raises(ValueError, match="折制品不存在"):
        tuner.get_trial_result(0)
    assert len(tuner.study_.trials) == 2


@pytest.mark.parametrize("mode", ["full", "predictions", "disk"])
def test_repeated_validation_oof_uses_position_and_mean(mode, data, tmp_path):
    from sklearn.model_selection import StratifiedKFold

    X, y = data
    X.index = np.arange(len(X))[::-1] + 1000
    splits = list(StratifiedKFold(2, shuffle=True, random_state=9).split(X, y)) * 2
    tuner = ModelTuner(
        RandomForest(n_estimators=2, n_jobs=1, random_state=3),
        search_space={},
        cv=splits,
        n_jobs=1,
        retention=mode,
        artifact_dir=str(tmp_path),
    )
    tuner.fit(X, y, n_trials=1, show_progress_bar=False)
    actual = tuner.get_oof_predictions()
    expected = np.zeros(len(y))
    for fold in tuner.get_trial_result(0)["各折"]:
        expected[fold["验证位置"]] += fold["预测概率"] / 2
    np.testing.assert_allclose(actual["预测概率"], expected)
    np.testing.assert_array_equal(actual["真实标签"], y)
    assert actual["验证次数"].eq(2).all()


@pytest.mark.parametrize("mode", ["summary", "disk", "best"])
def test_interruption_preserves_partial_result_and_does_not_refit(mode, data, tmp_path):
    calls = []

    def factory(trial, fold):
        calls.append(fold)
        if fold == 1:
            raise KeyboardInterrupt("主动中断")
        return {}

    tuner = ModelTuner(
        RandomForest(n_estimators=2, n_jobs=1),
        search_space={},
        cv=2,
        n_jobs=1,
        retention=mode,
        artifact_dir=str(tmp_path),
        fit_params_factory=factory,
    )
    with pytest.raises(KeyboardInterrupt):
        tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert calls == [0, 1]
    assert tuner.best_model_ is None
    result = tuner.get_trial_result(0, load=False)
    assert result["状态"] == "中断"
    assert result["各折"][0]["状态"] == "完成"
    assert result["各折"][1]["错误类型"] == "KeyboardInterrupt"


@pytest.mark.parametrize("mode", ["full", "summary", "disk"])
def test_retention_preserves_early_stopping_refit_rounds(mode, data, tmp_path):
    from hscredit.core.models import LightGBM

    tuner = ModelTuner(
        LightGBM(n_estimators=8, min_child_samples=3, early_stopping_rounds=2, n_jobs=1, random_state=2),
        search_space={},
        metric="auc",
        cv=2,
        n_jobs=1,
        random_state=2,
        retention=mode,
        artifact_dir=str(tmp_path),
    )
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    rounds = [fold["最佳迭代"] for fold in tuner.get_trial_result(0, load=False)["各折"]]
    best = tuner.get_best_model()
    assert best.n_estimators == max(1, int(np.ceil(np.median(rounds))))
    assert best.early_stopping_rounds is None
    assert best.training_summary_["样本数"] == len(data[1])


@pytest.mark.parametrize("mode", ["predictions", "summary", "disk"])
def test_non_model_retention_really_releases_fold_objects(mode, data, tmp_path, monkeypatch):
    import gc
    import weakref

    references = []
    original_fit = ModelTuner._fit_fold

    def tracked_fit(self, model, X, y, params):
        result = original_fit(self, model, X, y, params)
        references.append(weakref.ref(model))
        return result

    monkeypatch.setattr(ModelTuner, "_fit_fold", tracked_fit)
    tuner = tune(data, tmp_path, mode)
    gc.collect()
    assert len(references) == 4
    assert all(reference() is None for reference in references)
    assert tuner.best_params_ is not None


@pytest.mark.parametrize("mode", ["summary", "best", "disk"])
def test_multiobjective_retention_keeps_pareto_and_best_selection(mode, data, tmp_path):
    options = dict(
        search_space={"max_depth": [1, 3]},
        trial_points=[{"max_depth": 1}, {"max_depth": 3}],
        metric=["auc", "ks_diff"],
        direction=["maximize", "minimize"],
        cv=2,
        n_jobs=1,
        random_state=2,
    )
    full = ModelTuner(RandomForest(n_estimators=3, n_jobs=1, random_state=2), **options)
    retained = ModelTuner(
        RandomForest(n_estimators=3, n_jobs=1, random_state=2), retention=mode, artifact_dir=str(tmp_path), **options
    )
    for tuner in (full, retained):
        tuner.fit(*data, n_trials=2, show_progress_bar=False)
    assert retained.best_params_ == full.best_params_
    assert [(trial.number, trial.values) for trial in retained.study_.best_trials] == [
        (trial.number, trial.values) for trial in full.study_.best_trials
    ]
    np.testing.assert_allclose(
        retained.get_best_model().predict_proba(data[0]), full.get_best_model().predict_proba(data[0])
    )
