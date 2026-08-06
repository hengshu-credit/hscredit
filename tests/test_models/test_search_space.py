# -*- coding: utf-8 -*-
"""hscredit.core.models.tuning.search_space 跨框架同名符号回归测试."""
import numpy as np
import optuna
import pytest

optuna.logging.set_verbosity(optuna.logging.WARNING)

from hscredit.core.models.tuning.search_space import (  # noqa: F401
    Dimension,
    Real,
    Integer,
    Categorical,
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
    suggest_int,
    suggest_float,
    suggest_categorical,
    suggest_discrete_uniform,
    suggest_loguniform,
    uniform,
    loguniform,
    quniform,
    qloguniform,
    choice,
    randint,
    normal,
    qnormal,
    lognormal,
    qlognormal,
)
from hscredit.core.models.tuning import ModelTuner, normalize_search_space


def _norm(spec):
    return normalize_search_space({'p': spec})['p']


# ---------- skopt 风格 ----------
class TestSkoptStyle:
    def test_real_uniform(self):
        assert _norm(Real(0.1, 1.0)) == {'type': 'float', 'low': 0.1, 'high': 1.0}

    def test_real_log_uniform(self):
        assert _norm(Real(1e-3, 0.1, prior='log-uniform')) == {
            'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True
        }

    def test_integer(self):
        assert _norm(Integer(2, 6)) == {'type': 'int', 'low': 2, 'high': 6}

    def test_categorical(self):
        assert _norm(Categorical(['a', 'b'])) == {'type': 'categorical', 'choices': ['a', 'b']}


# ---------- optuna 分布对象风格 ----------
class TestOptunaDistributionStyle:
    def test_int_distribution(self):
        assert _norm(IntDistribution(2, 6)) == {'type': 'int', 'low': 2, 'high': 6}

    def test_int_distribution_log(self):
        assert _norm(IntDistribution(2, 256, log=True)) == {
            'type': 'int', 'low': 2, 'high': 256, 'log': True
        }

    def test_float_distribution(self):
        assert _norm(FloatDistribution(0.0, 1.0)) == {'type': 'float', 'low': 0.0, 'high': 1.0}

    def test_categorical_distribution(self):
        assert _norm(CategoricalDistribution(['x', 'y'])) == {
            'type': 'categorical', 'choices': ['x', 'y']
        }


# ---------- optuna suggest_* 风格 ----------
class TestOptunaSuggestStyle:
    def test_suggest_int(self):
        assert _norm(suggest_int('p', 2, 6)) == {'type': 'int', 'low': 2, 'high': 6}

    def test_suggest_int_step(self):
        assert _norm(suggest_int('p', 2, 10, step=2)) == {
            'type': 'int', 'low': 2, 'high': 10, 'step': 2
        }

    def test_suggest_float_log(self):
        assert _norm(suggest_float('p', 1e-3, 0.1, log=True)) == {
            'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True
        }

    def test_suggest_categorical(self):
        assert _norm(suggest_categorical('p', ['gbtree', 'dart'])) == {
            'type': 'categorical', 'choices': ['gbtree', 'dart']
        }

    def test_suggest_discrete_uniform(self):
        assert _norm(suggest_discrete_uniform('p', 0.0, 1.0, 0.1)) == {
            'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.1
        }

    def test_suggest_loguniform(self):
        assert _norm(suggest_loguniform('p', 1e-3, 0.1)) == {
            'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True
        }


# ---------- hyperopt hp.* 风格 ----------
class TestHyperoptStyle:
    def test_uniform(self):
        assert _norm(uniform('p', 0.6, 1.0)) == {'type': 'float', 'low': 0.6, 'high': 1.0}

    def test_loguniform(self):
        assert _norm(loguniform('p', 1e-3, 0.1)) == {
            'type': 'float', 'low': 1e-3, 'high': 0.1, 'log': True
        }

    def test_quniform(self):
        assert _norm(quniform('p', 0.0, 1.0, 0.1)) == {
            'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.1
        }

    def test_choice(self):
        assert _norm(choice('p', ['a', 'b', 'c'])) == {
            'type': 'categorical', 'choices': ['a', 'b', 'c']
        }

    def test_randint(self):
        # hyperopt randint(label, upper) 返回 [0, upper-1]
        assert _norm(randint('p', 5)) == {'type': 'int', 'low': 0, 'high': 4}

    def test_randint_with_low(self):
        assert _norm(randint('p', 10, low=2)) == {'type': 'int', 'low': 2, 'high': 9}


# ---------- hyperopt 正态族 ----------
class TestNormalFamily:
    def test_normal_spec(self):
        spec = _norm(normal('p', 0.1, 0.02))
        assert spec['type'] == 'normal'
        assert spec['mu'] == 0.1
        assert spec['sigma'] == 0.02
        assert 'low' in spec and 'high' in spec
        assert 'log' not in spec

    def test_lognormal_spec(self):
        spec = _norm(lognormal('p', -2.0, 0.5))
        assert spec['type'] == 'normal'
        assert spec.get('log') is True
        # 对数正态截断区间应为正数
        assert spec['low'] > 0 and spec['high'] > 0

    def test_qnormal_spec(self):
        spec = _norm(qnormal('p', 8, 2, 1))
        assert spec['type'] == 'normal'
        assert spec.get('q') == 1

    def test_qlognormal_spec(self):
        spec = _norm(qlognormal('p', -2.0, 0.5, 0.1))
        assert spec['type'] == 'normal'
        assert spec.get('log') is True
        assert spec.get('q') == 0.1

    def test_normal_sampling_in_range(self):
        """正态族采样值应落在截断区间内且可复现."""
        tuner = ModelTuner(
            model_class=None,
            search_space={'p': normal('p', 0.1, 0.02)},
            metric='ks',
            cv=2,
            random_state=42,
            verbose=False,
        )
        study = optuna.create_study(directions=['maximize'])
        values = []
        for _ in range(20):
            trial = study.ask()
            v = tuner._sample_normal(trial, 'p', tuner.search_space['p'])
            values.append(v)
        low, high = tuner.search_space['p']['low'], tuner.search_space['p']['high']
        assert all(low - 1e-6 <= v <= high + 1e-6 for v in values)


# ---------- 混合格式可整体归一化 ----------
class TestMixedFormat:
    def test_mixed_space_normalizable(self):
        space = {
            'max_depth': suggest_int('max_depth', 2, 6),
            'learning_rate': Real(1e-3, 0.1, prior='log-uniform'),
            'min_child_weight': Integer(1, 10),
            'booster': suggest_categorical('booster', ['gbtree', 'dart']),
            'subsample': uniform('subsample', 0.6, 1.0),
            'colsample_bytree': [0.6, 0.8, 1.0],
            'n_estimators': (50, 200),
            'reg_lambda': lognormal('reg_lambda', -2.0, 0.5),
        }
        norm = normalize_search_space(space)
        assert norm['max_depth']['type'] == 'int'
        assert norm['learning_rate'].get('log') is True
        assert norm['booster']['type'] == 'categorical'
        assert norm['colsample_bytree']['type'] == 'categorical'
        assert norm['n_estimators']['type'] == 'int'
        assert norm['reg_lambda']['type'] == 'normal'

    def test_dimension_subclass_repr(self):
        """维度对象 repr 不报错."""
        s = repr(Real(0.1, 1.0))
        assert 'Real' in s

    def test_dimension_base_to_spec_raises(self):
        with pytest.raises(NotImplementedError):
            Dimension().to_spec()
