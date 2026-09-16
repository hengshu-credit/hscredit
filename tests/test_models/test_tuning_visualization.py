"""两种调参入口均保留可供 Optuna 完整可视化使用的搜索过程。"""

import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from hscredit.core.models import LogisticRegression, ModelTuner, RandomForest

optuna = pytest.importorskip("optuna")
pytest.importorskip("plotly")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, n_redundant=0, random_state=44)
    return pd.DataFrame(X, columns=["年龄", "收入", "借款", "历史", "余额"]), y


@pytest.fixture(scope="module")
def tuner(data):
    tuner = ModelTuner(
        RandomForest(n_estimators=5, n_jobs=1, random_state=44),
        search_space={"max_depth": [2, 4], "min_samples_leaf": [1, 5]},
        metric="auc",
        cv=3,
        n_jobs=1,
        random_state=44,
    )
    tuner.enqueue_trials(param_grid={"max_depth": [2, 4], "min_samples_leaf": [1, 5]})
    tuner.fit(*data, n_trials=4, show_progress_bar=False)
    return tuner


@pytest.fixture(scope="module")
def multi_tuner(data):
    tuner = ModelTuner(
        RandomForest(n_estimators=5, n_jobs=1, random_state=44),
        search_space={"max_depth": [2, 4]},
        metric=["auc", "ks"],
        direction=["maximize", "maximize"],
        cv=2,
        n_jobs=1,
        random_state=44,
    )
    tuner.fit(*data, n_trials=3, show_progress_bar=False)
    return tuner


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _namespace(tuner, backend):
    return tuner.visualization if backend == "plotly" else tuner.visualization.matplotlib


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_every_installed_native_plot_is_discoverable_and_callable(tuner, backend):
    namespace = _namespace(tuner, backend)
    native = optuna.visualization if backend == "plotly" else optuna.visualization.matplotlib
    names = {name for name in dir(native) if name.startswith("plot_")}
    assert names <= set(dir(namespace))
    for name in names:
        function = getattr(namespace, name)
        assert callable(function)
        assert function.__wrapped__ is getattr(native, name)
        assert "study" not in inspect.signature(function).parameters
    assert namespace.is_available() == native.is_available()


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("plot_optimization_history", {}),
        ("plot_intermediate_values", {}),
        ("plot_timeline", {}),
        ("plot_rank", {"params": ["max_depth", "min_samples_leaf"]}),
        ("plot_slice", {"params": ["max_depth"]}),
        ("plot_contour", {"params": ["max_depth", "min_samples_leaf"]}),
        ("plot_parallel_coordinate", {"params": ["max_depth", "min_samples_leaf"]}),
        ("plot_param_importances", {}),
        ("plot_edf", {}),
    ],
)
def test_single_objective_native_plots_render_from_training_history(tuner, backend, name, kwargs):
    namespace = _namespace(tuner, backend)
    if not hasattr(namespace, name):
        pytest.skip(f"当前 Optuna 版本没有 {name}")
    result = getattr(namespace, name)(**kwargs)
    if backend == "plotly":
        assert result.to_plotly_json()["data"]
    else:
        assert np.asarray(result, dtype=object).size > 0


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_pareto_hypervolume_and_callable_target_preserve_native_semantics(multi_tuner, backend):
    namespace = _namespace(multi_tuner, backend)
    assert namespace.plot_pareto_front(target_names=["区分能力", "风险区分度"]) is not None
    assert namespace.plot_optimization_history(target=lambda trial: trial.values[1], target_name="KS") is not None
    if hasattr(namespace, "plot_hypervolume_history"):
        # reference_point 是 study 之后的原生位置参数，不应被当作 Study。
        assert namespace.plot_hypervolume_history([0.0, 0.0]) is not None


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_terminator_plot_can_read_automatically_recorded_cv_scores(tuner, backend):
    namespace = _namespace(tuner, backend)
    if not hasattr(namespace, "plot_terminator_improvement"):
        pytest.skip("当前 Optuna 版本没有终止改进图")
    from optuna.terminator import BaseImprovementEvaluator, CrossValidationErrorEvaluator

    class ImprovementEvaluator(BaseImprovementEvaluator):
        def evaluate(self, trials, study_direction):
            return 1.0 / len(trials)

    evaluator = CrossValidationErrorEvaluator()
    assert np.isfinite(evaluator.evaluate(tuner.get_study().trials, tuner.get_study().direction))
    assert (
        namespace.plot_terminator_improvement(
            plot_error=True,
            improvement_evaluator=ImprovementEvaluator(),
            error_evaluator=evaluator,
            min_n_trials=1,
        )
        is not None
    )


def test_both_training_entrypoints_share_complete_study_and_saved_visualizations(data, tmp_path, tuner):
    X, y = data
    model = RandomForest(n_estimators=3, n_jobs=1, random_state=44)
    best = model.tune(X, y, search_space={"max_depth": [2, 4]}, n_trials=2, cv=2, n_jobs=1, show_progress_bar=False)
    assert model.tuner is best.tuner
    assert best.tuner.get_study() is best.tuner.study_
    assert tuner.get_best_model().tuner is tuner
    trial = best.tuner.get_study().trials[0]
    assert set(trial.intermediate_values) == {0, 1}
    assert len(trial.user_attrs["各折指标"]) == 2
    assert trial.datetime_start is not None and trial.datetime_complete is not None
    assert model.tuner.visualization.plot_timeline().data

    restored = RandomForest.load(best.save(tmp_path / "best.joblib"))
    restored_study = restored.tuner.get_study()
    assert restored_study.trials[0].intermediate_values == trial.intermediate_values
    assert restored_study.trials[0].system_attrs == trial.system_attrs
    assert restored.tuner.visualization.plot_intermediate_values().data
    restored_tuner = ModelTuner.load(tuner.save(tmp_path / "tuner.joblib"))
    assert restored_tuner.visualization.plot_timeline().data
    assert restored_tuner.get_study().trials[0].system_attrs == tuner.get_study().trials[0].system_attrs


def test_logistic_tune_exposes_the_same_native_visualization_entrypoint(data):
    model = LogisticRegression(calculate_stats=False, max_iter=100, random_state=44)
    best = model.tune(*data, search_space={"C": [0.1, 1.0]}, n_trials=2, cv=2, n_jobs=1, show_progress_bar=False)
    assert model.tuner.get_study() is best.tuner.get_study()
    assert best.tuner.visualization.plot_intermediate_values().data


def test_raw_study_works_with_unwrapped_optuna_functions_and_multi_study_comparison(tuner):
    study = tuner.get_study()
    assert optuna.visualization.plot_timeline(study).data
    assert tuner.visualization.plot_edf(study=[study, study]).data
    assert study.sampler is tuner.study_.sampler
    assert study.pruner is tuner.study_.pruner


def test_namespace_keeps_native_errors_and_follows_new_functions(monkeypatch, tuner):
    def plot_future(study, *, marker):
        return study, marker

    monkeypatch.setattr(optuna.visualization, "plot_future", plot_future, raising=False)
    assert tuner.visualization.plot_future(marker="原生参数") == (tuner.get_study(), "原生参数")

    def plot_failure(study):
        raise ValueError("原生绘图条件不满足")

    monkeypatch.setattr(optuna.visualization, "plot_failure", plot_failure, raising=False)
    with pytest.raises(ValueError, match="原生绘图条件不满足"):
        tuner.visualization.plot_failure()


def test_unstarted_search_has_clear_error_but_can_inspect_visualization_availability():
    tuner = ModelTuner(RandomForest, search_space={})
    assert isinstance(tuner.visualization.is_available(), bool)
    with pytest.raises(ValueError, match="尚未创建"):
        tuner.get_study()
    with pytest.raises(ValueError, match="尚未创建"):
        tuner.visualization.plot_timeline()
    with pytest.raises(AttributeError, match="没有"):
        tuner.visualization.plot_does_not_exist()


def test_failed_and_pruned_trials_remain_visible(data):
    def objective(trial):
        trial.report(0.2, 0)
        if trial.number == 0:
            raise optuna.TrialPruned("已剪枝")
        raise RuntimeError("试验故障")

    tuner = ModelTuner(RandomForest, search_space={}, trial_objective=objective, cv=2, n_jobs=1)
    with pytest.raises(ValueError, match="所有Trial"):
        tuner.fit(*data, n_trials=2, show_progress_bar=False)
    study = tuner.get_study()
    assert [trial.state.name for trial in study.trials] == ["PRUNED", "FAIL"]
    assert study.trials[1].user_attrs["错误信息"] == "试验故障"
    assert tuner.visualization.plot_timeline().data
    assert tuner.visualization.plot_intermediate_values().data


def test_reloaded_database_study_can_be_visualized_without_refitting(data, tmp_path):
    storage = f"sqlite:///{(tmp_path / 'tuning.db').as_posix()}"
    tuner = ModelTuner(
        RandomForest(n_estimators=3, n_jobs=1),
        search_space={"max_depth": [2, 3]},
        metric="auc",
        cv=2,
        n_jobs=1,
        storage=storage,
        study_name="搜索过程",
    )
    tuner.fit(*data, n_trials=2, show_progress_bar=False)
    study = optuna.load_study(storage=storage, study_name="搜索过程")
    restored = ModelTuner(RandomForest, study=study)
    assert restored.get_study() is study
    assert restored.visualization.plot_intermediate_values().data
    assert len(study.trials[0].user_attrs["各折指标"]) == 2
    from optuna.terminator import CrossValidationErrorEvaluator

    assert np.isfinite(CrossValidationErrorEvaluator().evaluate(study.trials, study.direction))


def test_old_optuna_without_terminator_keeps_existing_training_contract(monkeypatch, data):
    import importlib

    original = importlib.import_module

    def import_without_terminator(name, *args, **kwargs):
        if name == "optuna.terminator":
            raise ModuleNotFoundError("旧版没有此模块", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_without_terminator)
    tuner = ModelTuner(RandomForest(n_estimators=2, n_jobs=1), search_space={}, cv=2, n_jobs=1)
    tuner.fit(*data, n_trials=1, show_progress_bar=False)
    assert len(tuner.get_study().trials[0].user_attrs["各折指标"]) == 2
    assert tuner.visualization.plot_intermediate_values().data
