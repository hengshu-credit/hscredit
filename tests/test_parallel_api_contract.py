"""全项目统一并行 API 契约测试。"""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from hscredit.core import binning, encoders, selectors
from hscredit.core.binning.base import BaseBinning
from hscredit.core.encoders.base import BaseEncoder
from hscredit.core.models import rules as model_rules
from hscredit.core.rules import Rule
from hscredit.core import rules
from hscredit.core.selectors.base import BaseFeatureSelector
from hscredit.exceptions import ValidationError
from hscredit.report import mining, _sample_stats
import hscredit.report as report
from hscredit.report.mining.base import BaseRuleMiner
from hscredit.utils.parallel import (
    ParallelizableMixin,
    parallel_execute,
    resolve_n_jobs,
    validate_parallel_config,
)


COMMON_PARAMETERS = ("n_jobs", "parallel_backend", "parallel_config")

# 明确列出 Task 1-11 范围内的每个批量入口。下面的覆盖测试还会与各模块
# __all__ 双向核对，避免只抽查少数类而漏掉新增或删除的公开入口。
PUBLIC_PARALLEL_ENTRIES = {
    binning: {
        "BaseBinning",
        "UniformBinning",
        "QuantileBinning",
        "TreeBinning",
        "CartBinning",
        "ChiMergeBinning",
        "BestKSBinning",
        "BestIVBinning",
        "OptimalBinning",
        "MDLPBinning",
        "ORBinning",
        "CPSATBinning",
        "KMeansBinning",
        "MonotonicBinning",
        "GeneticBinning",
        "SmoothBinning",
        "KernelDensityBinning",
        "BestLiftBinning",
        "TargetBadRateBinning",
        "OptimalBinning2D",
    },
    selectors: {
        "BaseFeatureSelector",
        "TypeSelector",
        "RegexSelector",
        "NullSelector",
        "ModeSelector",
        "CardinalitySelector",
        "VarianceSelector",
        "CorrSelector",
        "VIFSelector",
        "IVSelector",
        "LiftSelector",
        "PSISelector",
        "FeatureImportanceSelector",
        "NullImportanceSelector",
        "RFESelector",
        "SequentialFeatureSelector",
        "StepwiseSelector",
        "BorutaSelector",
        "MutualInfoSelector",
        "Chi2Selector",
        "FTestSelector",
        "StabilityAwareSelector",
        "ScorecardFeatureSelection",
        "CompositeFeatureSelector",
    },
    encoders: {
        "BaseEncoder",
        "WOEEncoder",
        "TargetEncoder",
        "CountEncoder",
        "OneHotEncoder",
        "OrdinalEncoder",
        "QuantileEncoder",
        "CatBoostEncoder",
        "GBMEncoder",
        "CardinalityEncoder",
    },
    rules: {"Rule", "RuleFlow"},
    model_rules: {"RuleSet", "RulesClassifier"},
    mining: {
        "SingleFeatureRuleMiner",
        "MultiFeatureRuleMiner",
        "MultiLabelRuleMiner",
        "TreeRuleExtractor",
        "DecisionTreeAnalyzer",
        "ManualTreeExtractor",
        "RuleMetrics",
    },
    report: {
        "feature_bin_stats",
        "feature_binning_summary",
        "feature_group_binning_summary",
        "feature_efficiency_analysis",
        "auto_feature_analysis",
        "ruleset_analysis",
        "multi_label_rule_analysis",
        "rule_swap_analysis",
        "rule_report_table",
        "rule_target_analysis",
        "rule_target_table",
        "rule_group_hit_table",
        "rule_group_compare",
        "swap_out_report",
        "ReferenceDataProvider",
        "SwapAnalyzer",
        "create_swap_dataset",
        "create_swap_dataset_from_rules",
        "swap_analysis",
        "OverduePredictor",
        "overdue_prediction_report",
        "SingleFeatureRuleMiner",
        "MultiFeatureRuleMiner",
        "MultiLabelRuleMiner",
        "TreeRuleExtractor",
        "RuleMetrics",
        "ModelReport",
        "QuickModelReport",
        "auto_model_report",
        "compare_models",
        "population_drift",
    },
}

NON_BATCH_EXPORTS = {
    binning: {"CustomObjectives"},
    selectors: {"SelectionReportCollector"},
    encoders: set(),
    rules: {
        "get_columns_from_query",
        "optimize_expr",
        "beautify_expr",
        "get_expr_variables",
        "RuleState",
        "RuleStateError",
        "RuleUnAppliedError",
    },
    model_rules: {"LogicOperator", "RuleResult", "create_and_ruleset", "create_or_ruleset", "combine_rules"},
    mining: {"calculate_rule_metrics"},
    report: {"ExcelWriter", "dataframe2excel", "SwapAnalysisResult", "SwapRiskConfig", "SwapType", "calculate_rule_metrics"},
}


class _CloneableEncoder(BaseEncoder):
    """仅用于验证基类 sklearn 生命周期的最小编码器。"""

    def _fit(self, X, y=None):
        self.mapping_ = {}

    def _transform(self, X, y=None):
        return X


class _ExecutableBinning(BaseBinning):
    """仅用于验证基类特征任务委托的最小分箱器。"""

    def _fit_feature(self, feature, x, y):
        self.seen_.append(feature)

    def fit(self, X, y=None):
        self.seen_ = []
        self._fit_features(X, pd.Series(y, index=X.index), "_fit_feature")
        return self

    def transform(self, X, metric="indices"):
        return X


def _callable_signature(entry):
    return inspect.signature(entry.__init__ if inspect.isclass(entry) else entry)


def _all_parallel_entry_cases():
    cases = []
    for module, names in PUBLIC_PARALLEL_ENTRIES.items():
        cases.extend(pytest.param(getattr(module, name), id=f"{module.__name__}.{name}") for name in sorted(names))
    cases.extend(
        [
            pytest.param(_sample_stats.build_sample_stats_table, id="report.build_sample_stats_table"),
            pytest.param(_sample_stats.build_group_distribution_table, id="report.build_group_distribution_table"),
        ]
    )
    return cases


@pytest.mark.parametrize("entry", _all_parallel_entry_cases())
def test_every_public_parallel_entry_exposes_common_defaults(entry):
    """所有目标公开批量入口必须公开同名、同默认值参数。"""
    parameters = _callable_signature(entry).parameters

    assert parameters["n_jobs"].default == -1
    assert parameters["parallel_backend"].default is None
    assert parameters["parallel_config"].default is None


@pytest.mark.parametrize("module", PUBLIC_PARALLEL_ENTRIES)
def test_explicit_parallel_manifest_covers_every_exported_callable(module):
    """显式清单必须完整覆盖 __all__，标量辅助入口不得伪装成批量入口。"""
    exported_callables = {
        name
        for name in module.__all__
        if inspect.isclass(getattr(module, name)) or inspect.isfunction(getattr(module, name))
    }
    assert exported_callables == PUBLIC_PARALLEL_ENTRIES[module] | NON_BATCH_EXPORTS[module]

    detected_parallel = {
        name
        for name in exported_callables
        if all(parameter in _callable_signature(getattr(module, name)).parameters for parameter in COMMON_PARAMETERS)
    }
    assert detected_parallel == PUBLIC_PARALLEL_ENTRIES[module]


@pytest.mark.parametrize("cls", [BaseBinning, BaseEncoder, BaseFeatureSelector, BaseRuleMiner])
def test_parallel_base_classes_share_parallelizable_mixin(cls):
    """估计器基类不得各自维护独立执行器。"""
    assert issubclass(cls, ParallelizableMixin)


def test_base_constructor_preserves_parallel_parameters_without_resolution():
    """构造器必须原样保存比例预算和调用者配置。"""
    config = {"batch_size": 8}
    encoder = _CloneableEncoder(n_jobs=0.5, parallel_backend="threading", parallel_config=config)

    assert encoder.n_jobs == 0.5
    assert encoder.parallel_backend == "threading"
    assert encoder.parallel_config is config


def test_encoder_clone_and_pickle_preserve_parallel_configuration():
    """sklearn clone 与 pickle 必须保留并行配置值。"""
    config = {"batch_size": 8}
    encoder = _CloneableEncoder(
        cols=["类别"], n_jobs=0.5, parallel_backend="threading", parallel_config=config
    )

    cloned = clone(encoder)
    restored = pickle.loads(pickle.dumps(encoder))

    assert cloned.get_params()["parallel_config"] == config
    assert cloned.parallel_config is not config
    assert cloned.get_params()["n_jobs"] == 0.5
    assert restored.parallel_config == config
    assert restored.parallel_backend == "threading"


def _required_constructor_arguments(cls):
    estimator = LogisticRegression(max_iter=20)
    values = {
        "pattern": "特征",
        "estimator": estimator,
        "selectors": [],
        "expr": "特征 > 0",
        "rules": [Rule("特征 > 0")],
        "model": estimator,
        "feature": "特征",
    }
    signature = inspect.signature(cls.__init__)
    return {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is inspect.Parameter.empty and name in values
    }


def _sklearn_parallel_classes():
    unique = {}
    for module, names in PUBLIC_PARALLEL_ENTRIES.items():
        for name in names:
            entry = getattr(module, name)
            if inspect.isclass(entry) and issubclass(entry, BaseEstimator) and not inspect.isabstract(entry):
                unique[(entry.__module__, entry.__name__)] = entry
    return [pytest.param(cls, id=f"{module}.{name}") for (module, name), cls in sorted(unique.items())]


@pytest.mark.parametrize("cls", _sklearn_parallel_classes())
def test_all_applicable_estimators_support_get_params_clone_and_pickle(cls):
    """所有可实例化 sklearn 批量组件均须支持参数发现、克隆和序列化。"""
    config = {"batch_size": 2}
    instance = cls(
        **_required_constructor_arguments(cls),
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )

    parameters = instance.get_params(deep=False)
    cloned = clone(instance)
    restored = pickle.loads(pickle.dumps(instance))

    assert parameters["n_jobs"] == 0.5
    assert parameters["parallel_backend"] == "threading"
    assert parameters["parallel_config"] is config
    assert cloned.get_params(deep=False)["parallel_config"] == config
    assert restored.parallel_config == config


@pytest.mark.parametrize(
    "n_jobs,cpu_count,task_count,expected",
    [
        (None, 16, 99, None),
        (-1, 1, 99, 1),
        (-1, 8, 99, 7),
        (-1.0, 16, 99, 13),
        (1, 16, 99, 1),
        (1.0, 16, 99, 1),
        (3, 16, 99, 3),
        (3.0, 16, 99, 3),
        (0.25, 16, 99, 4),
        (0.26, 16, 99, 5),
        (20, 8, 99, 20),
        (-1, 16, 3, 3),
    ],
)
def test_n_jobs_resolution_contract(n_jobs, cpu_count, task_count, expected):
    """统一解析覆盖串行、自动、整数、整数浮点、比例和任务上限。"""
    assert resolve_n_jobs(n_jobs, task_count=task_count, cpu_count=cpu_count) == expected


@pytest.mark.parametrize("n_jobs", [True, False, 0, -2, 1.5, np.nan, np.inf, "2"])
def test_invalid_n_jobs_raises_chinese_validation_error(n_jobs):
    with pytest.raises(ValidationError, match="n_jobs 必须"):
        resolve_n_jobs(n_jobs, cpu_count=8)


def test_all_documented_parallel_config_keys_are_prevalidated_without_mutation():
    """支持键完整通过预校验，后端专属字典会复制且不修改调用者对象。"""
    config = {
        "prefer": "threads",
        "require": "sharedmem",
        "batch_size": 2,
        "pre_dispatch": "2*n_jobs",
        "max_nbytes": "64M",
        "mmap_mode": "r",
        "temp_folder": None,
        "timeout": 30.0,
        "verbose": 0,
        "inner_max_num_threads": 1,
        "backend_kwargs": {"idle_worker_timeout": 60},
    }
    original = pickle.loads(pickle.dumps(config))

    validated = validate_parallel_config(None, config)

    assert config == original
    assert validated == config
    assert validated is not config
    assert validated["backend_kwargs"] is not config["backend_kwargs"]


@pytest.mark.parametrize(
    "backend,config,match",
    [
        (None, {"n_jobs": 2}, "不能包含 n_jobs"),
        (None, {"backend": "loky"}, "不能包含 backend"),
        (None, {"未知配置": True}, "不支持的配置项"),
        (None, {"backend_kwargs": 1}, "backend_kwargs 必须为字典"),
    ],
)
def test_parallel_config_conflicts_raise_chinese_validation_error(backend, config, match):
    with pytest.raises(ValidationError, match=match):
        validate_parallel_config(backend, config)


def test_backend_conflict_is_rejected_before_worker_submission():
    """显式线程后端和 loky 专属线程限制冲突必须在 worker 前报中文错误。"""
    with pytest.raises(ValidationError, match="threading 后端不支持"):
        parallel_execute(
            abs,
            [-1, -2],
            n_jobs=2,
            parallel_backend="threading",
            parallel_config={"inner_max_num_threads": 1},
        )


def test_base_parallel_execute_preserves_order_for_threading_backend():
    encoder = _CloneableEncoder(n_jobs=2, parallel_backend="threading")
    assert encoder._parallel_execute(abs, [-3, -1, -2]) == [3, 1, 2]


def test_base_binning_feature_loop_uses_shared_parallel_configuration():
    binner = _ExecutableBinning(n_jobs=1, parallel_config={"未知配置": True})
    with pytest.raises(ValidationError, match="parallel_config"):
        binner.fit(pd.DataFrame({"特征A": [1, 2]}), [0, 1])


def test_base_encoder_parallel_parameters_do_not_change_dual_api():
    frame = pd.DataFrame({"类别": ["甲", "乙"], "FPD": [0, 1]})
    encoder = _CloneableEncoder(
        cols=["类别"], target="FPD", n_jobs=None, parallel_config={"batch_size": 1}
    )
    pd.testing.assert_frame_equal(encoder.fit_transform(frame), frame)
