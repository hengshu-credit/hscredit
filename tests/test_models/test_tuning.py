import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin

from hscredit.core.models import AutoTuner, ModelTuner, TuningObjective
from hscredit.core.models.tuning import normalize_search_space


optuna = pytest.importorskip("optuna")


_TUNER_MODEL_WORKERS = []


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


def test_model_tuner_uses_trial_workers_and_splits_native_model_budget(monkeypatch):
    """调参器 n_jobs 必须进入 Optuna，并避免每个 trial 再次用满全部 CPU。"""
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

    assert optimize_workers == [2]
    assert _TUNER_MODEL_WORKERS
    assert set(_TUNER_MODEL_WORKERS) == {2}


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
        assert tuner.study_.trials[0].user_attrs == {"来源": "经验值"}

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
