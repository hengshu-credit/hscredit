"""搜索空间声明、边界转换与手工搜索点的行为回归。"""

import optuna
import pytest
from scipy import stats

from hscredit.core.models.tuning.search_space import (
    Categorical,
    FloatDistribution,
    IntDistribution,
    Integer,
    Real,
    lognormal,
    normal,
    qloguniform,
    qnormal,
    quniform,
    randint,
)
from hscredit.core.models.tuning.space_adapter import SearchSpaceAdapter, normalize_search_space


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IntDistribution(1.9, 5),
        lambda: IntDistribution(1, 5.9),
        lambda: IntDistribution(1, 5, step=1.9),
        lambda: Integer(True, 5),
        lambda: randint("depth", 1.9, 5),
        lambda: randint("depth", 5.9),
        lambda: randint("depth", low=True, upper=5),
        lambda: FloatDistribution(0, 1, step=True),
    ],
)
def test_declaration_rejects_lossy_integer_casts_and_boolean_steps(factory):
    with pytest.raises(ValueError, match="整数|数值"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Integer(5, 2),
        lambda: Real(0, 1, prior="log-uniform"),
        lambda: FloatDistribution(0.1, 1, log=True, step=0.1),
        lambda: Categorical([]),
        lambda: normal("p", 1, -1),
        lambda: qloguniform("p", 700, 1000, 1),
        lambda: lognormal("p", 700, 100),
        lambda: lognormal("p", -1000, 1),
        lambda: normal("p", 1e308, 1e308),
    ],
)
def test_invalid_declarations_fail_at_construction(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("distribution", [stats.randint(2, 5, loc=10), stats.randint(2, 5, 10)])
def test_scipy_randint_preserves_location_and_exclusive_upper(distribution):
    adapter = SearchSpaceAdapter({"p": distribution})
    assert adapter.space["p"] == {"type": "int", "low": 12, "high": 14}
    assert adapter.sample(optuna.trial.FixedTrial({"p": 14})) == {"p": 14}


@pytest.mark.parametrize("distribution", [stats.loguniform(1, 10, scale=2), stats.loguniform(1, 10, 0, 2)])
def test_scipy_loguniform_preserves_scale(distribution):
    assert normalize_search_space({"p": distribution})["p"] == {
        "type": "float",
        "low": 2.0,
        "high": 20.0,
        "log": True,
    }


def test_scipy_loguniform_rejects_unsupported_location_instead_of_changing_distribution():
    with pytest.raises(ValueError, match="非零 loc"):
        SearchSpaceAdapter({"p": stats.loguniform(1, 10, loc=5)})


@pytest.mark.parametrize("key", ["__hscredit__p", "", 1])
def test_parameter_names_cannot_collide_or_silently_coerce(key):
    with pytest.raises(ValueError, match="参数名"):
        normalize_search_space({"p": quniform("p", 0, 1, 0.1), key: (0, 1)})


@pytest.mark.parametrize("prior", [1, [[1], [2]], [float("inf"), 1], [0, 0], [-1, 2]])
def test_categorical_prior_validates_shape_and_values(prior):
    with pytest.raises(ValueError, match="prior"):
        Categorical(["a", "b"], prior=prior)


def test_categorical_prior_normalization_does_not_overflow():
    adapter = SearchSpaceAdapter({"p": Categorical(["a", "b"], prior=[1e308, 1e308])})
    assert adapter.space["p"]["prior"] == [0.5, 0.5]


@pytest.mark.parametrize("u, expected", [(0.0, "a"), (0.5, "b"), (1.0, "b")])
def test_zero_prior_categories_cannot_be_sampled_at_endpoints(u, expected):
    adapter = SearchSpaceAdapter({"p": Categorical(["zero", "a", "b", "zero2"], prior=[0, 1, 1, 0])})
    params = {"__hscredit__p": u}
    assert adapter.sample(optuna.trial.FixedTrial(params))["p"] == expected
    assert adapter.public_params(params)["p"] == expected


def test_quantized_normal_keeps_rounding_semantics_outside_raw_bounds():
    adapter = SearchSpaceAdapter({"p": qnormal("p", 0.1, 0.1, 0.3)})
    upper = stats.norm.cdf(4)
    sampled = adapter.sample(optuna.trial.FixedTrial({"__hscredit__p": upper}))["p"]
    assert sampled == pytest.approx(0.6)
    assert sampled > adapter.space["p"]["high"]
    internal = adapter.to_internal_point({"p": sampled})
    assert adapter.sample(optuna.trial.FixedTrial(internal))["p"] == pytest.approx(sampled)


def test_manual_quantized_point_requires_reachable_rounding_not_just_interval_intersection():
    adapter = SearchSpaceAdapter({"p": quniform("p", 0.5, 0.5, 1.0)})
    assert adapter.sample(optuna.trial.FixedTrial({"__hscredit__p": 0.5}))["p"] == 0.0
    with pytest.raises(ValueError, match="无法.*生成"):
        adapter.to_internal_point({"p": 1.0})


def test_large_float_manual_point_does_not_pass_with_fractional_step():
    adapter = SearchSpaceAdapter({"p": FloatDistribution(0, 200000, step=1)})
    with pytest.raises(ValueError, match="step"):
        adapter.to_internal_point({"p": 100000.4})


def test_negative_log_quantized_manual_point_gives_actionable_error():
    adapter = SearchSpaceAdapter({"p": qloguniform("p", -2, 2, 1)})
    with pytest.raises(ValueError, match="无法.*生成"):
        adapter.to_internal_point({"p": -1.0})


def test_package_and_space_declarations_keep_optuna_lazy():
    """定义空间及普通指标不应提前加载可选优化框架。"""
    import subprocess
    import sys

    source = """
import sys
import hscredit
from hscredit.core.models.tuning import Integer, Real, Categorical, normalize_search_space, make_metric
space = normalize_search_space({'深度': Integer(2, 4), '速率': Real(.01, .1), '选项': Categorical([1, 2])})
assert len(space) == 3
assert 'optuna' not in sys.modules
assert all(name not in sys.modules for name in ('xgboost', 'lightgbm', 'catboost', 'ngboost'))
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
