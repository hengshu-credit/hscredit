import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification

from hscredit.core.models import AutoTuner, ModelTuner, TuningObjective
from hscredit.core.models.tuning import normalize_search_space


optuna = pytest.importorskip("optuna")


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
        max_depth = trial.params["max_depth"]
        num_leaves = trial.params["num_leaves"]
        assert num_leaves <= 2**max_depth


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
            'max_depth': {'type': 'int', 'low': 2, 'high': 4, 'step': 1},
            'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True},
            'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']},
        }
        normalized = normalize_search_space(space)
        assert normalized['max_depth'] == {'type': 'int', 'low': 2, 'high': 4, 'step': 1}
        assert normalized['learning_rate'] == {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True}
        assert normalized['penalty'] == {'type': 'categorical', 'choices': ['l1', 'l2']}

    def test_bayesian_optimization_tuple_style(self):
        """bayesian-optimization 风格: (low, high) 元组."""
        normalized = normalize_search_space({
            'max_depth': (2, 4),          # 两端整数 -> int
            'learning_rate': (1e-3, 0.1), # 含浮点 -> float
        })
        assert normalized['max_depth'] == {'type': 'int', 'low': 2, 'high': 4}
        assert normalized['learning_rate'] == {'type': 'float', 'low': 1e-3, 'high': 0.1}

    def test_skopt_tuple_and_list_style(self):
        """scikit-optimize 风格: (low, high, prior) 元组与 list 类别."""
        normalized = normalize_search_space({
            'learning_rate': (1e-3, 0.1, 'log-uniform'),
            'subsample': (0.5, 1.0, 'uniform'),
            'penalty': ['l1', 'l2'],
        })
        assert normalized['learning_rate'] == {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True}
        assert normalized['subsample'] == {'type': 'float', 'low': 0.5, 'high': 1.0}
        assert normalized['penalty'] == {'type': 'categorical', 'choices': ['l1', 'l2']}

    def test_sklearn_list_and_scipy_distribution_style(self):
        """sklearn 风格: list 网格 + scipy 分布（RandomizedSearchCV）."""
        normalized = normalize_search_space({
            'C': [0.1, 1.0, 10.0],
            'max_depth': scipy_stats.randint(2, 5),          # [2, 5) -> [2, 4]
            'learning_rate': scipy_stats.loguniform(1e-3, 1e-1),
            'subsample': scipy_stats.uniform(0.5, 0.5),      # [0.5, 1.0]
        })
        assert normalized['C'] == {'type': 'categorical', 'choices': [0.1, 1.0, 10.0]}
        assert normalized['max_depth'] == {'type': 'int', 'low': 2, 'high': 4}
        assert normalized['learning_rate'] == {
            'type': 'float', 'low': pytest.approx(1e-3), 'high': pytest.approx(1e-1), 'log': True,
        }
        assert normalized['subsample'] == {
            'type': 'float', 'low': pytest.approx(0.5), 'high': pytest.approx(1.0),
        }

    def test_hyperopt_dict_style(self):
        """hyperopt 风格: uniform/loguniform/quniform/randint/choice."""
        normalized = normalize_search_space({
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 0.1},
            'gamma': {'type': 'quniform', 'low': 0.0, 'high': 5.0, 'q': 0.5},
            'max_depth': {'type': 'randint', 'low': 2, 'high': 4},
            'penalty': {'type': 'choice', 'choices': ['l1', 'l2']},
        })
        assert normalized['subsample'] == {'type': 'float', 'low': 0.5, 'high': 1.0}
        assert normalized['learning_rate'] == {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True}
        assert normalized['gamma'] == {'type': 'float', 'low': 0.0, 'high': 5.0, 'step': 0.5}
        assert normalized['max_depth'] == {'type': 'int', 'low': 2, 'high': 4}
        assert normalized['penalty'] == {'type': 'categorical', 'choices': ['l1', 'l2']}

    def test_optuna_distribution_objects(self):
        """optuna 原生分布对象."""
        normalized = normalize_search_space({
            'max_depth': optuna.distributions.IntDistribution(2, 4),
            'learning_rate': optuna.distributions.FloatDistribution(1e-3, 0.1, log=True),
            'penalty': optuna.distributions.CategoricalDistribution(['l1', 'l2']),
        })
        assert normalized['max_depth'] == {'type': 'int', 'low': 2, 'high': 4}
        assert normalized['learning_rate'] == {'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True}
        assert normalized['penalty'] == {'type': 'categorical', 'choices': ['l1', 'l2']}

    def test_mixed_frameworks_in_one_space(self):
        """同一搜索空间内混用多种框架格式."""
        normalized = normalize_search_space({
            'max_depth': (2, 4),
            'learning_rate': {'type': 'loguniform', 'low': 1e-3, 'high': 0.1},
            'penalty': ['l1', 'l2'],
        })
        assert normalized['max_depth']['type'] == 'int'
        assert normalized['learning_rate']['log'] is True
        assert normalized['penalty']['type'] == 'categorical'

    def test_none_passthrough(self):
        assert normalize_search_space(None) is None

    def test_invalid_specs_raise_chinese_value_error(self):
        with pytest.raises(ValueError, match="缺少 'type'"):
            normalize_search_space({'x': {'low': 1, 'high': 2}})
        with pytest.raises(ValueError, match="未知|不支持|无法识别"):
            normalize_search_space({'x': {'type': 'gaussian', 'low': 1, 'high': 2}})
        with pytest.raises(ValueError, match="不能大于"):
            normalize_search_space({'x': (5, 2)})
        with pytest.raises(ValueError, match="choices"):
            normalize_search_space({'x': []})
        with pytest.raises(ValueError, match="无法识别"):
            normalize_search_space({'x': 123})
        with pytest.raises(ValueError, match="字典"):
            normalize_search_space([('x', (1, 2))])

    def test_model_tuner_accepts_multi_framework_search_space(self):
        """ModelTuner 端到端：混合格式 search_space 可正常调优."""
        from sklearn.linear_model import LogisticRegression

        X, y = _small_binary_data(as_frame=True)
        tuner = ModelTuner(
            LogisticRegression,
            search_space={
                'C': (1e-3, 1.0, 'log-uniform'),      # skopt 风格
                'max_iter': (100, 200),               # bayesian-optimization 风格
                'solver': ['liblinear', 'lbfgs'],     # sklearn/skopt list 风格
            },
            metric='ks',
            cv=2,
            random_state=42,
        )
        tuner.fit(X, y, n_trials=2, show_progress_bar=False)

        assert tuner.best_params_
        assert set(tuner.best_params_) == {'C', 'max_iter', 'solver'}
        assert 1e-3 <= tuner.best_params_['C'] <= 1.0
        assert 100 <= tuner.best_params_['max_iter'] <= 200
        assert tuner.best_params_['solver'] in ('liblinear', 'lbfgs')
        # 构造时 search_space 已被归一化为内部 DSL
        assert tuner.search_space['C'] == {'type': 'float', 'low': 1e-3, 'high': 1.0, 'log': True}
        assert tuner.search_space['max_iter'] == {'type': 'int', 'low': 100, 'high': 200}
        assert tuner.search_space['solver'] == {'type': 'categorical', 'choices': ['liblinear', 'lbfgs']}
