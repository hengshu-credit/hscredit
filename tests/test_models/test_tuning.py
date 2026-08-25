import _thread
import threading
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin

from hscredit.core.models import AutoTuner, BaseRiskModel, ModelTuner, TuningObjective
from hscredit.core.models.tuning import normalize_search_space


optuna = pytest.importorskip("optuna")


_TUNER_MODEL_WORKERS = []
_TUNER_INTERRUPT_STARTED = threading.Event()
_TUNER_INTERRUPT_FIT_CALLS = 0


class _RecordingTunerEstimator(BaseEstimator, ClassifierMixin):
    """记录每个并行 trial 中底层模型实际获得的预算。"""

    def __init__(self, n_jobs=99):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        _TUNER_MODEL_WORKERS.append(self.n_jobs)
        self.classes_ = np.unique(y)
        self.positive_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_rate_, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class _VerboseTunerEstimator(BaseEstimator, ClassifierMixin):
    """提供可搜索参数的轻量分类器，用于验证调参过程输出。"""

    def __init__(self, marker=0, n_jobs=1):
        self.marker = marker
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.positive_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_rate_, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class _InterruptibleTunerEstimator(BaseEstimator, ClassifierMixin):
    """保持单次训练忙碌，便于模拟 Jupyter 向主线程发送中断。"""

    def __init__(self, n_jobs=1, fit_seconds=1.0):
        self.n_jobs = n_jobs
        self.fit_seconds = fit_seconds

    def fit(self, X, y):
        global _TUNER_INTERRUPT_FIT_CALLS
        _TUNER_INTERRUPT_FIT_CALLS += 1
        _TUNER_INTERRUPT_STARTED.set()
        deadline = time.perf_counter() + self.fit_seconds
        while time.perf_counter() < deadline:
            pass
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), 0.5, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class _InterruptRecordingRiskModel(BaseRiskModel):
    """记录最终模型是否在调参中断后被错误重训。"""

    fit_calls = 0

    def fit(self, X, y=None, sample_weight=None, eval_set=None, **fit_params):
        type(self).fit_calls += 1
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    def get_feature_importances(self):
        return pd.Series(dtype=float)


def test_model_tuner_trial_annotations_use_a_type_name_not_module_variable():
    """Pylance 不应把运行时可选模块变量 ``optuna`` 当作类型命名空间。"""
    assert ModelTuner._sample_params.__annotations__["trial"] == "Trial"
    assert ModelTuner._sample_normal.__annotations__["trial"] == "Trial"


def _small_binary_data(as_frame=False):
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    if as_frame:
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="target")
    return X, y


def test_model_tuner_supports_numpy_input_without_failed_trials():
    X, y = _small_binary_data(as_frame=False)
    tuner = AutoTuner.create("lr", metric="ks", cv=3, random_state=42)

    best_params = tuner.fit(X, y, n_trials=2, show_progress_bar=False)

    assert best_params
    assert np.isfinite(tuner.best_score_)
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in tuner.study_.trials)
    assert all(np.isfinite(trial.value) for trial in tuner.study_.trials)


def test_model_tuner_verbose_prints_each_trial_and_final_summary(capsys):
    """verbose=True 应独立于 Optuna 日志配置输出每次 Trial 和最终摘要。"""
    X, y = _small_binary_data(as_frame=True)
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    try:
        tuner = ModelTuner(
            _VerboseTunerEstimator,
            search_space={"marker": [0, 1]},
            metric="ks",
            cv=2,
            random_state=42,
            verbose=True,
        )
        tuner.fit(X, y, n_trials=2, show_progress_bar=False)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

    output = capsys.readouterr().out
    assert output.count("[调参] Trial ") == 2
    assert "得分: KS=" in output
    assert "参数: {'marker':" in output
    assert "当前最佳: KS=" in output
    assert "[调参] 调参完成 | 完成 Trial: 2" in output
    assert "[调参] 最佳得分: KS=" in output
    assert "[调参] 最佳参数:" in output


def test_model_tuner_verbose_false_does_not_print_tuning_lines(capsys):
    """verbose=False 不应产生 HSCredit 调参过程输出。"""
    X, y = _small_binary_data(as_frame=True)
    tuner = ModelTuner(
        _VerboseTunerEstimator,
        search_space={"marker": [0]},
        metric="ks",
        cv=2,
        random_state=42,
        verbose=False,
    )

    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    assert "[调参]" not in capsys.readouterr().out


def test_model_tuner_verbose_prints_all_multi_objective_scores(capsys):
    """多目标调参应输出全部指标、当前最佳结果和帕累托摘要。"""
    X, y = _small_binary_data(as_frame=True)
    tuner = ModelTuner(
        _VerboseTunerEstimator,
        search_space={"marker": [0]},
        metric=["ks", "ks_diff"],
        direction=["maximize", "minimize"],
        metric_names=["KS", "KS差异"],
        cv=2,
        random_state=42,
        verbose=True,
    )

    tuner.fit(X, y, n_trials=1, show_progress_bar=False)

    output = capsys.readouterr().out
    assert "得分: KS=" in output
    assert "KS差异=" in output
    assert "当前最佳:" in output
    assert "帕累托最优解: 1" in output


def test_model_tuner_runs_trials_sequentially_and_gives_model_full_budget(monkeypatch):
    """每个 trial 应利用全部历史结果，同时让当前模型使用完整 CPU 预算。"""
    X, y = _small_binary_data(as_frame=True)
    optimize_workers = []
    original_optimize = optuna.study.Study.optimize

    def recording_optimize(self, *args, **kwargs):
        optimize_workers.append(kwargs.get("n_jobs"))
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", recording_optimize)
    _TUNER_MODEL_WORKERS.clear()
    tuner = ModelTuner(
        _RecordingTunerEstimator,
        search_space={},
        metric="ks",
        cv=2,
        n_jobs=4,
        random_state=42,
    )

    tuner.fit(X, y, n_trials=2, show_progress_bar=False)

    assert optimize_workers == [1]
    assert _TUNER_MODEL_WORKERS
    assert set(_TUNER_MODEL_WORKERS) == {4}


def test_each_tuning_trial_observes_all_previous_completed_trials(monkeypatch):
    """顺序采样时，新 trial 开始前应能看到全部历史完成结果。"""
    X, y = _small_binary_data(as_frame=True)
    tuner = ModelTuner(
        _RecordingTunerEstimator,
        search_space={},
        metric="ks",
        cv=2,
        n_jobs=4,
        random_state=42,
    )
    observed_completed = []
    original_evaluate = tuner._evaluate_model

    def recording_evaluate(*args, **kwargs):
        observed_completed.append(
            sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in tuner.study_.trials)
        )
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(tuner, "_evaluate_model", recording_evaluate)

    tuner.fit(X, y, n_trials=3, show_progress_bar=False)

    assert observed_completed == [0, 1, 2]


def test_model_tuner_keyboard_interrupt_returns_without_waiting_for_parallel_trials(monkeypatch):
    """Jupyter 主线程中断必须停止当前训练，不能等待其他 trial 线程。"""
    global _TUNER_INTERRUPT_FIT_CALLS
    X, y = _small_binary_data(as_frame=True)
    _TUNER_INTERRUPT_STARTED.clear()
    _TUNER_INTERRUPT_FIT_CALLS = 0
    postprocessing_calls = []
    tuner = ModelTuner(
        _InterruptibleTunerEstimator,
        search_space={},
        metric="ks",
        cv=2,
        n_jobs=4,
        random_state=42,
    )
    monkeypatch.setattr(tuner, "_save_results", lambda: postprocessing_calls.append("保存结果"))
    monkeypatch.setattr(tuner, "_build_public_history", lambda: postprocessing_calls.append("构建历史"))

    def interrupt_main_when_training_starts():
        if _TUNER_INTERRUPT_STARTED.wait(timeout=2.0):
            time.sleep(0.05)
            _thread.interrupt_main()

    interrupter = threading.Thread(target=interrupt_main_when_training_starts, daemon=True)
    interrupter.start()
    started = time.perf_counter()

    with pytest.raises(KeyboardInterrupt):
        tuner.fit(X, y, n_trials=4, show_progress_bar=False)

    elapsed = time.perf_counter() - started
    interrupter.join(timeout=1.0)
    assert elapsed < 0.6
    assert _TUNER_INTERRUPT_FIT_CALLS == 1
    assert len(tuner.study_.trials) == 1
    assert postprocessing_calls == []


def test_risk_model_tune_does_not_refit_after_keyboard_interrupt(monkeypatch):
    """调参器中断后，便捷 tune API 不得继续训练所谓最佳模型。"""
    X, y = _small_binary_data(as_frame=True)
    _InterruptRecordingRiskModel.fit_calls = 0

    def interrupt_tuning(*args, **kwargs):
        raise KeyboardInterrupt("用户中断")

    monkeypatch.setattr(ModelTuner, "fit", interrupt_tuning)

    with pytest.raises(KeyboardInterrupt, match="用户中断"):
        _InterruptRecordingRiskModel(random_state=42).tune(X, y, n_trials=4, cv=2)

    assert _InterruptRecordingRiskModel.fit_calls == 0


def test_lightgbm_tuner_samples_num_leaves_after_max_depth_constraint():
    pytest.importorskip("lightgbm")
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create(
        "lightgbm",
        metric=["ks", "ks_diff"],
        direction=["maximize", "minimize"],
        cv=3,
        random_state=42,
    )

    tuner.fit(X, y, n_trials=3, show_progress_bar=False)

    assert tuner.pareto_front_
    assert len(tuner.best_scores_) == 2
    for trial in tuner.study_.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        public_params = tuner._get_params_from_trial(trial)
        max_depth = public_params["max_depth"]
        num_leaves = public_params["num_leaves"]
        assert num_leaves <= 2**max_depth


def test_lightgbm_grid_candidates_do_not_assume_low_high_fields():
    class LightGBMStub:
        pass

    tuner = ModelTuner(
        LightGBMStub,
        search_space={"max_depth": [2], "num_leaves": [8]},
        metric="ks",
        cv=2,
    )
    params = tuner._sample_params(optuna.trial.FixedTrial({"max_depth": 2, "num_leaves": 8}))

    assert params == {"max_depth": 2, "num_leaves": 4}


def test_num_leaves_constraint_is_not_applied_to_unrelated_models():
    class UnrelatedEstimator:
        pass

    tuner = ModelTuner(
        UnrelatedEstimator,
        search_space={
            "max_depth": {"type": "int", "low": 2, "high": 2},
            "num_leaves": {"type": "int", "low": 10, "high": 10},
        },
        metric="ks",
        cv=2,
    )
    params = tuner._sample_params(optuna.create_study().ask())

    assert params == {"max_depth": 2, "num_leaves": 10}


def test_ngboost_default_search_space_avoids_unstable_small_leaves_and_deep_trees():
    """默认空间不得再次采样已证实会令自然梯度概率饱和的激进组合。"""

    class NGBoostStub:
        pass

    tuner = ModelTuner(NGBoostStub, search_space=None, metric="ks", cv=2)
    tuner._n_samples = 184
    tuner._n_features = 8

    space = tuner._get_adaptive_search_space()

    assert space["learning_rate"]["high"] == pytest.approx(0.05)
    assert space["base_max_depth"] == {"type": "int", "low": 2, "high": 3}
    assert space["base_min_samples_leaf"]["low"] == 5
    assert space["minibatch_frac"]["low"] == pytest.approx(0.7)


def test_lightgbm_default_space_uses_class_ratio_and_effective_subsampling():
    class LightGBMStub:
        pass

    tuner = ModelTuner(LightGBMStub, search_space=None, metric="ks", cv=2)
    tuner._class_balance_ratio = 6.0

    space = tuner._get_adaptive_search_space()

    assert space["scale_pos_weight"] == {
        "type": "float",
        "low": pytest.approx(3.0),
        "high": pytest.approx(9.0),
    }
    assert space["subsample_freq"] == {"type": "categorical", "choices": [1]}
    assert space["min_split_gain"]["high"] <= 1.0
    assert space["reg_lambda"]["high"] <= 10.0
    assert space["learning_rate"]["log"] is True


def test_xgboost_default_class_weight_uses_observed_class_ratio():
    class XGBoostStub:
        pass

    tuner = ModelTuner(XGBoostStub, search_space=None, metric="ks", cv=2)
    tuner._class_balance_ratio = 4.0

    space = tuner._get_adaptive_search_space()

    assert space["scale_pos_weight"]["low"] == pytest.approx(2.0)
    assert space["scale_pos_weight"]["high"] == pytest.approx(6.0)


def test_native_svc_and_unknown_models_do_not_receive_xgboost_space():
    from sklearn.svm import SVC

    svc_space = ModelTuner(SVC)._get_adaptive_search_space()

    assert {"C", "kernel", "gamma"} <= set(svc_space)
    assert "n_estimators" not in svc_space

    class UnknownEstimator:
        pass

    with pytest.raises(ValueError, match="无法为模型.*自动生成搜索空间"):
        ModelTuner(UnknownEstimator)._get_adaptive_search_space()


def test_logistic_default_space_only_uses_persistable_class_weights():
    class LogisticRegressionStub:
        pass

    space = ModelTuner(LogisticRegressionStub)._get_adaptive_search_space()

    assert space["class_weight"]["choices"] == [None, "balanced"]


def test_lightgbm_rejects_explicit_invalid_manual_leaf_point():
    class LightGBMStub:
        pass

    tuner = ModelTuner(
        LightGBMStub,
        search_space={"max_depth": [2], "num_leaves": [4, 8]},
        metric="ks",
        cv=2,
    )

    with pytest.raises(ValueError, match="num_leaves.*2\*\*max_depth"):
        tuner.enqueue_trial({"max_depth": 2, "num_leaves": 8})


def test_multi_objective_default_best_trial_uses_primary_metric_first():
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create(
        "lr",
        metric=["ks", "ks_diff"],
        direction=["maximize", "minimize"],
        cv=3,
        random_state=42,
    )

    tuner.fit(X, y, n_trials=4, show_progress_bar=False)

    pareto = tuner.get_pareto_front()
    primary_scores = [trial.values[0] for trial in pareto]
    assert tuner.best_score_ == max(primary_scores)
    assert tuner._resolve_multi_objective_target(None) == 0


def test_metric_names_length_must_match_metric_list():
    with pytest.raises(ValueError, match="metric_names"):
        AutoTuner.create(
            "lr",
            metric=["ks", "ks_diff"],
            direction=["maximize", "minimize"],
            metric_names=["KS"],
        )


def test_business_tuning_objectives_are_available_as_metric_names():
    X, y = _small_binary_data(as_frame=True)
    tuner = AutoTuner.create("lr", metric="lift_head", cv=3, random_state=42)

    tuner.fit(X, y, n_trials=2, show_progress_bar=False)

    assert tuner.metric_names == ["LIFT_HEAD"]
    assert np.isfinite(tuner.best_score_)
    assert all(np.isfinite(trial.value) for trial in tuner.study_.trials)


def test_strategy_tuning_objectives_return_finite_values():
    y_true = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.05, 0.9, 0.2, 0.1, 0.8, 0.3])

    approval_score = TuningObjective.approval_bad_rate(y_true, y_prob, approval_rate=0.5)
    profit_score = TuningObjective.expected_profit(y_true, y_prob, approval_rate=0.5)

    assert np.isfinite(approval_score)
    assert np.isfinite(profit_score)
    assert TuningObjective.get("EXPECTED_PROFIT")(y_true, y_prob) == TuningObjective.expected_profit(y_true, y_prob)


class TestNormalizeSearchSpace:
    """多框架搜索空间格式统一为内部 DSL（optuna 风格）."""

    def test_native_dsl_passthrough(self):
        space = {
            "max_depth": {"type": "int", "low": 2, "high": 4, "step": 1},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        }
        normalized = normalize_search_space(space)
        assert normalized["max_depth"] == {"type": "int", "low": 2, "high": 4, "step": 1}
        assert normalized["learning_rate"] == {"type": "float", "low": 1e-3, "high": 0.1, "log": True}
        assert normalized["penalty"] == {"type": "categorical", "choices": ["l1", "l2"]}

    def test_bayesian_optimization_tuple_style(self):
        """bayesian-optimization 风格: (low, high) 元组."""
        normalized = normalize_search_space(
            {
                "max_depth": (2, 4),  # 两端整数 -> int
                "learning_rate": (1e-3, 0.1),  # 含浮点 -> float
            }
        )
        assert normalized["max_depth"] == {"type": "int", "low": 2, "high": 4}
        assert normalized["learning_rate"] == {"type": "float", "low": 1e-3, "high": 0.1}

    def test_skopt_tuple_and_list_style(self):
        """scikit-optimize 风格: (low, high, prior) 元组与 list 类别."""
        normalized = normalize_search_space(
            {
                "learning_rate": (1e-3, 0.1, "log-uniform"),
                "subsample": (0.5, 1.0, "uniform"),
                "penalty": ["l1", "l2"],
            }
        )
        assert normalized["learning_rate"] == {"type": "float", "low": 1e-3, "high": 0.1, "log": True}
        assert normalized["subsample"] == {"type": "float", "low": 0.5, "high": 1.0}
        assert normalized["penalty"] == {"type": "categorical", "choices": ["l1", "l2"]}

    def test_sklearn_list_and_scipy_distribution_style(self):
        """sklearn 风格: list 网格 + scipy 分布（RandomizedSearchCV）."""
        normalized = normalize_search_space(
            {
                "C": [0.1, 1.0, 10.0],
                "max_depth": scipy_stats.randint(2, 5),  # [2, 5) -> [2, 4]
                "learning_rate": scipy_stats.loguniform(1e-3, 1e-1),
                "subsample": scipy_stats.uniform(0.5, 0.5),  # [0.5, 1.0]
            }
        )
        assert normalized["C"] == {"type": "categorical", "choices": [0.1, 1.0, 10.0]}
        assert normalized["max_depth"] == {"type": "int", "low": 2, "high": 4}
        assert normalized["learning_rate"] == {
            "type": "float",
            "low": pytest.approx(1e-3),
            "high": pytest.approx(1e-1),
            "log": True,
        }
        assert normalized["subsample"] == {
            "type": "float",
            "low": pytest.approx(0.5),
            "high": pytest.approx(1.0),
        }

    def test_hyperopt_dict_style(self):
        """hyperopt 风格: uniform/loguniform/quniform/randint/choice."""
        normalized = normalize_search_space(
            {
                "subsample": {"type": "uniform", "low": 0.5, "high": 1.0},
                "learning_rate": {"type": "loguniform", "low": np.log(1e-3), "high": np.log(0.1)},
                "gamma": {"type": "quniform", "low": 0.0, "high": 5.0, "q": 0.5},
                "max_depth": {"type": "randint", "low": 2, "high": 4},
                "penalty": {"type": "choice", "choices": ["l1", "l2"]},
            }
        )
        assert normalized["subsample"] == {"type": "float", "low": 0.5, "high": 1.0}
        assert normalized["learning_rate"] == {
            "type": "float",
            "low": pytest.approx(1e-3),
            "high": pytest.approx(0.1),
            "log": True,
        }
        assert normalized["gamma"] == {"type": "quniform", "low": 0.0, "high": 5.0, "q": 0.5}
        assert normalized["max_depth"] == {"type": "int", "low": 2, "high": 4}
        assert normalized["penalty"] == {"type": "categorical", "choices": ["l1", "l2"]}

    def test_optuna_distribution_objects(self):
        """optuna 原生分布对象."""
        normalized = normalize_search_space(
            {
                "max_depth": optuna.distributions.IntDistribution(2, 4),
                "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.1, log=True),
                "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2"]),
            }
        )
        assert normalized["max_depth"] == {"type": "int", "low": 2, "high": 4}
        assert normalized["learning_rate"] == {"type": "float", "low": 1e-3, "high": 0.1, "log": True}
        assert normalized["penalty"] == {"type": "categorical", "choices": ["l1", "l2"]}

    def test_mixed_frameworks_in_one_space(self):
        """同一搜索空间内混用多种框架格式."""
        normalized = normalize_search_space(
            {
                "max_depth": (2, 4),
                "learning_rate": {"type": "loguniform", "low": 1e-3, "high": 0.1},
                "penalty": ["l1", "l2"],
            }
        )
        assert normalized["max_depth"]["type"] == "int"
        assert normalized["learning_rate"]["log"] is True
        assert normalized["penalty"]["type"] == "categorical"

    def test_none_passthrough(self):
        assert normalize_search_space(None) is None

    def test_invalid_specs_raise_chinese_value_error(self):
        with pytest.raises(ValueError, match="缺少 'type'"):
            normalize_search_space({"x": {"low": 1, "high": 2}})
        with pytest.raises(ValueError, match="未知|不支持|无法识别"):
            normalize_search_space({"x": {"type": "gaussian", "low": 1, "high": 2}})
        with pytest.raises(ValueError, match="不能大于"):
            normalize_search_space({"x": (5, 2)})
        with pytest.raises(ValueError, match="choices"):
            normalize_search_space({"x": []})
        with pytest.raises(ValueError, match="无法识别"):
            normalize_search_space({"x": 123})
        with pytest.raises(ValueError, match="必须设置 name"):
            normalize_search_space([("x", (1, 2))])

    def test_model_tuner_accepts_multi_framework_search_space(self):
        """ModelTuner 端到端：混合格式 search_space 可正常调优."""
        from sklearn.linear_model import LogisticRegression

        X, y = _small_binary_data(as_frame=True)
        tuner = ModelTuner(
            LogisticRegression,
            search_space={
                "C": (1e-3, 1.0, "log-uniform"),  # skopt 风格
                "max_iter": (100, 200),  # bayesian-optimization 风格
                "solver": ["liblinear", "lbfgs"],  # sklearn/skopt list 风格
            },
            metric="ks",
            cv=2,
            random_state=42,
        )
        tuner.fit(X, y, n_trials=2, show_progress_bar=False)

        assert tuner.best_params_
        assert set(tuner.best_params_) == {"C", "max_iter", "solver"}
        assert 1e-3 <= tuner.best_params_["C"] <= 1.0
        assert 100 <= tuner.best_params_["max_iter"] <= 200
        assert tuner.best_params_["solver"] in ("liblinear", "lbfgs")
        # 构造时 search_space 已被归一化为内部 DSL
        assert tuner.search_space["C"] == {"type": "float", "low": 1e-3, "high": 1.0, "log": True}
        assert tuner.search_space["max_iter"] == {"type": "int", "low": 100, "high": 200}
        assert tuner.search_space["solver"] == {"type": "categorical", "choices": ["liblinear", "lbfgs"]}


class TestManualSearchPoints:
    """各框架的手工搜索点最终都通过 Optuna enqueue_trial 执行。"""

    @staticmethod
    def _tuner():
        from sklearn.linear_model import LogisticRegression

        return ModelTuner(
            LogisticRegression,
            search_space={"C": [0.1, 1.0], "solver": ["liblinear", "lbfgs"]},
            fixed_params={"max_iter": 100},
            metric="ks",
            cv=2,
            random_state=42,
        )

    def test_optuna_enqueue_trial_preserves_attrs_and_runs_first(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = self._tuner()
        tuner.enqueue_trial({"C": 1.0, "solver": "liblinear"}, user_attrs={"来源": "经验值"})

        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

        assert tuner.study_.trials[0].params == {"C": 1.0, "solver": "liblinear"}
        assert tuner.study_.trials[0].user_attrs["来源"] == "经验值"
        assert {"LIFT@1%", "LIFT@3%", "LIFT@5%", "LIFT@10%"} <= set(
            tuner.study_.trials[0].user_attrs
        )

    def test_gridsearch_param_grid_expands_with_parameter_grid_order(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = self._tuner().enqueue_trials(param_grid={"C": [0.1, 1.0], "solver": ["liblinear"]})

        tuner.fit(X, y, n_trials=2, show_progress_bar=False)

        assert [trial.params for trial in tuner.study_.trials] == [
            {"C": 0.1, "solver": "liblinear"},
            {"C": 1.0, "solver": "liblinear"},
        ]

    def test_skopt_x0_uses_declared_dimension_order(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = self._tuner().enqueue_trials(x0=[[0.1, "liblinear"], [1.0, "lbfgs"]])

        tuner.fit(X, y, n_trials=2, show_progress_bar=False)

        assert [trial.params for trial in tuner.study_.trials] == [
            {"C": 0.1, "solver": "liblinear"},
            {"C": 1.0, "solver": "lbfgs"},
        ]

    def test_bayesian_probe_accepts_dict_and_ordered_sequence(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = self._tuner()
        tuner.probe(params={"C": 0.1, "solver": "liblinear"}, lazy=False)
        tuner.probe(params=[1.0, "lbfgs"], lazy=True)

        tuner.fit(X, y, n_trials=2, show_progress_bar=False)

        assert [trial.params for trial in tuner.study_.trials] == [
            {"C": 0.1, "solver": "liblinear"},
            {"C": 1.0, "solver": "lbfgs"},
        ]

    def test_hyperopt_points_to_evaluate_constructor_alias(self):
        from sklearn.linear_model import LogisticRegression

        X, y = _small_binary_data(as_frame=True)
        tuner = ModelTuner(
            LogisticRegression,
            search_space={"C": [0.1, 1.0], "solver": ["liblinear"]},
            fixed_params={"max_iter": 100},
            points_to_evaluate=[{"C": 1.0, "solver": "liblinear"}],
            metric="ks",
            cv=2,
            random_state=42,
        )

        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

        assert tuner.study_.trials[0].params == {"C": 1.0, "solver": "liblinear"}

    def test_transformed_point_is_inverted_for_optuna_but_public_results_stay_clean(self):
        from hscredit import qloguniform
        from sklearn.linear_model import LogisticRegression

        X, y = _small_binary_data(as_frame=True)
        tuner = ModelTuner(
            LogisticRegression,
            search_space={
                "C": qloguniform("C", np.log(0.1), np.log(2.0), 0.1),
                "solver": ["liblinear"],
            },
            fixed_params={"max_iter": 100},
            metric="ks",
            cv=2,
            random_state=42,
        )
        tuner.enqueue_trial({"C": 1.0, "solver": "liblinear"})

        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

        assert tuner.best_params_["C"] == pytest.approx(1.0)
        assert "__hscredit__C" in tuner.study_.trials[0].params
        assert not any("__hscredit__" in column for column in tuner.optimization_history_.columns)
        assert tuner.optimization_history_.loc[0, "params_C"] == pytest.approx(1.0)

    def test_partial_dict_point_leaves_other_parameters_to_sampler(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = self._tuner().enqueue_trial({"C": 1.0})

        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

        assert tuner.study_.trials[0].params["C"] == 1.0
        assert tuner.study_.trials[0].params["solver"] in {"liblinear", "lbfgs"}

    def test_sequence_points_require_all_dimensions_and_values_are_validated(self):
        tuner = self._tuner()
        with pytest.raises(ValueError, match="维度数量"):
            tuner.enqueue_trials(x0=[[0.1]])
        with pytest.raises(ValueError, match="不存在"):
            tuner.enqueue_trial({"unknown": 1})
        with pytest.raises(ValueError, match="choices"):
            tuner.probe(params=[0.1, "not-a-solver"])

    def test_dict_point_can_be_cached_before_adaptive_space_is_known(self):
        X, y = _small_binary_data(as_frame=True)
        tuner = AutoTuner.create(
            "lr",
            metric="ks",
            cv=2,
            random_state=42,
            trial_points={"C": 1.0},
        )

        tuner.fit(X, y, n_trials=1, show_progress_bar=False)

        assert tuner.study_.trials[0].params["C"] == 1.0


def test_all_declared_framework_styles_run_on_optuna_backend():
    """五类原声明格式都能直接驱动同一个 Optuna ModelTuner。"""
    from hscredit import (
        Categorical,
        Integer,
        Real,
        choice,
        loguniform,
        randint,
        suggest_categorical,
        suggest_float,
        suggest_int,
    )
    from sklearn.linear_model import LogisticRegression

    X, y = _small_binary_data(as_frame=True)
    spaces = {
        "optuna": {
            "C": suggest_float("C", 0.1, 1.0, log=True),
            "max_iter": suggest_int("max_iter", 80, 120),
            "solver": suggest_categorical("solver", ["liblinear", "lbfgs"]),
        },
        "gridsearch": {
            "C": [0.1, 1.0],
            "max_iter": [100],
            "solver": ["liblinear", "lbfgs"],
        },
        "skopt": [
            Real(0.1, 1.0, prior="log-uniform", name="C"),
            Integer(80, 120, name="max_iter"),
            Categorical(["liblinear", "lbfgs"], name="solver"),
        ],
        "bayesian-optimization": {
            "C": (0.1, 1.0, float),
            "max_iter": (80, 120, int),
            "solver": ("liblinear", "lbfgs"),
        },
        "hyperopt": {
            "C": loguniform("C", np.log(0.1), np.log(1.0)),
            "max_iter": randint("max_iter", 80, 121),
            "solver": choice("solver", ["liblinear", "lbfgs"]),
        },
    }

    for framework, space in spaces.items():
        tuner = ModelTuner(
            LogisticRegression,
            search_space=space,
            metric="ks",
            cv=2,
            random_state=42,
        )
        best = tuner.fit(X, y, n_trials=1, show_progress_bar=False)
        assert set(best) == {"C", "max_iter", "solver"}, framework
        assert len(tuner.study_.trials) == 1, framework
        assert tuner.study_.trials[0].state == optuna.trial.TrialState.COMPLETE, framework
