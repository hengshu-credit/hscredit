"""规则挖掘模块统一并行执行测试。"""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.rules import Rule
from hscredit.exceptions import ParallelExecutionError, ValidationError
from hscredit.report import mining
from hscredit.report.mining import (
    DecisionTreeAnalyzer,
    ManualTreeExtractor,
    MultiFeatureRuleMiner,
    MultiLabelRuleMiner,
    RuleMetrics,
    SingleFeatureRuleMiner,
    TreeRuleExtractor,
)


PUBLIC_PARALLEL_CLASSES = (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    MultiLabelRuleMiner,
    TreeRuleExtractor,
    DecisionTreeAnalyzer,
    ManualTreeExtractor,
    RuleMetrics,
)


@pytest.fixture
def mining_data():
    """构造包含数值、类别和双标签的稳定小样本。"""
    rng = np.random.RandomState(17)
    size = 160
    score = rng.normal(size=size)
    debt = rng.normal(size=size)
    category = np.where(rng.rand(size) > 0.45, "A", "B")
    target = ((score + 0.7 * debt + (category == "B") * 0.4) > 0.3).astype(int)
    long_target = ((0.6 * score + debt + (category == "B") * 0.2) > 0.5).astype(int)
    return pd.DataFrame(
        {
            "score": score,
            "debt": debt,
            "category": category,
            "target": target,
            "long_target": long_target,
        }
    )


def _assert_miner_results_equal(left, right):
    assert list(left.results_) == list(right.results_)
    for feature in left.results_:
        pd.testing.assert_frame_equal(left.results_[feature], right.results_[feature])


def _normalize_tree_rules(rules):
    return [
        (
            tuple(
                (condition["feature"], condition["operator"], condition["threshold"])
                for condition in rule["conditions"]
            ),
            rule["predicted_class"],
            rule["class_probability"],
            rule["sample_count"],
            rule["tree_id"],
            rule["importance"],
        )
        for rule in rules
    ]


def test_all_exported_batch_mining_classes_expose_parallel_configuration():
    """删除任一公开类的统一参数时应失败。"""
    exported = {getattr(mining, name) for name in mining.__all__ if inspect.isclass(getattr(mining, name))}
    assert set(PUBLIC_PARALLEL_CLASSES) <= exported
    for cls in PUBLIC_PARALLEL_CLASSES:
        parameters = inspect.signature(cls.__init__).parameters
        assert parameters["n_jobs"].default == -1
        assert parameters["parallel_backend"].default is None
        assert parameters["parallel_config"].default is None


def test_single_feature_parallel_fit_matches_serial_for_both_backends(mining_data):
    """改变 feature 调度后端不得改变特征结果、顺序或 dtype。"""
    data = mining_data[["score", "debt", "category", "target"]]
    serial = SingleFeatureRuleMiner(
        method="quantile", max_n_bins=4, min_samples=2, min_lift=1.0, n_jobs=1
    ).fit(data)

    for backend in ("threading", "loky"):
        parallel = SingleFeatureRuleMiner(
            method="quantile",
            max_n_bins=4,
            min_samples=2,
            min_lift=1.0,
            n_jobs=2,
            parallel_backend=backend,
        ).fit(data)
        _assert_miner_results_equal(serial, parallel)
        assert [rule.expr for rule in serial.get_rules()] == [rule.expr for rule in parallel.get_rules()]


def test_multi_feature_parallel_combinations_match_serial(mining_data):
    """组合并行必须按 combinations 生成顺序提交并产生相同结果。"""
    data = mining_data[["score", "debt", "category", "target"]]
    kwargs = dict(max_n_bins=3, min_samples=2, min_lift=0.0)
    serial_miner = MultiFeatureRuleMiner(n_jobs=1, **kwargs).fit(data)
    parallel_miner = MultiFeatureRuleMiner(n_jobs=2, parallel_backend="threading", **kwargs).fit(data)

    serial = serial_miner.get_all_cross_rules(top_n=2, max_feature_pairs=3)
    parallel = parallel_miner.get_all_cross_rules(top_n=2, max_feature_pairs=3)
    pd.testing.assert_frame_equal(serial, parallel)
    assert list(serial_miner.cross_results_) == list(parallel_miner.cross_results_)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_multi_label_parallel_labels_match_serial(mining_data, backend):
    """标签外层并行及 feature 子并行不得改变规则集合和指标排序。"""
    kwargs = dict(
        labels=["target", "long_target"],
        label_names=["短期", "长期"],
        min_support=0.01,
        min_lift=1.0,
        max_rules=5,
        n_bins=3,
    )
    features = ["score", "debt"]
    serial = MultiLabelRuleMiner(n_jobs=1, **kwargs).fit(mining_data, features=features)
    parallel = MultiLabelRuleMiner(n_jobs=2, parallel_backend=backend, **kwargs).fit(
        mining_data, features=features
    )
    pd.testing.assert_frame_equal(serial.get_report(), parallel.get_report())
    pd.testing.assert_frame_equal(serial.get_effectiveness_matrix(), parallel.get_effectiveness_matrix())


def test_multi_label_declares_real_parallel_children(monkeypatch, mining_data):
    """label worker 内启动 Single feature 任务时必须声明真实子并行。"""
    import hscredit.utils.parallel as parallel_module

    original = parallel_module.parallel_execute
    nested_flags = []

    def recording_execute(function, tasks, **kwargs):
        nested_flags.append(kwargs.get("has_parallel_children", False))
        return original(function, tasks, **kwargs)

    monkeypatch.setattr(parallel_module, "parallel_execute", recording_execute)
    MultiLabelRuleMiner(
        labels=["target", "long_target"],
        min_support=0.01,
        min_lift=1.0,
        n_bins=3,
        n_jobs=1,
    ).fit(mining_data, features=["score", "debt"])
    assert nested_flags[0] is True
    assert False in nested_flags[1:]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_tree_extraction_parallel_trees_matches_serial(mining_data, backend):
    """独立树提取并行不得改变树序、规则序和指标。"""
    data = mining_data[["score", "debt", "target"]]
    kwargs = dict(algorithm="rf", n_estimators=4, max_depth=3, random_state=13)
    serial = TreeRuleExtractor(n_jobs=1, **kwargs).fit(data)
    parallel = TreeRuleExtractor(n_jobs=2, parallel_backend=backend, **kwargs).fit(data)
    assert _normalize_tree_rules(serial.extract_rules()) == _normalize_tree_rules(parallel.extract_rules())


def test_tree_analyzer_and_manual_tree_report_parallel_datasets_match_serial(mining_data):
    """自动树和人工树的数据集/节点报告并行必须保持容器与表格精确一致。"""
    train = mining_data.iloc[:120][["score", "debt", "target"]]
    test = mining_data.iloc[120:][["score", "debt", "target"]]
    datasets = {"训练": train, "测试": test}

    serial_analyzer = DecisionTreeAnalyzer(features=["score", "debt"], n_jobs=1).fit(train)
    parallel_analyzer = DecisionTreeAnalyzer(
        features=["score", "debt"], n_jobs=2, parallel_backend="threading"
    ).fit(train)
    assert serial_analyzer.evaluate([("测试", test)], metric_type="ks") == parallel_analyzer.evaluate(
        [("测试", test)], metric_type="ks"
    )
    serial_reports = serial_analyzer.report(datasets)
    parallel_reports = parallel_analyzer.report(datasets)
    assert list(serial_reports) == list(parallel_reports)
    for name in datasets:
        pd.testing.assert_frame_equal(serial_reports[name], parallel_reports[name])

    serial_manual = ManualTreeExtractor(max_depth=2, n_jobs=1).fit(train, features=["score", "debt"])
    parallel_manual = ManualTreeExtractor(
        max_depth=2, n_jobs=2, parallel_backend="loky"
    ).fit(train, features=["score", "debt"])
    serial_manual_reports = serial_manual.report(datasets)
    parallel_manual_reports = parallel_manual.report(datasets)
    for name in datasets:
        pd.testing.assert_frame_equal(serial_manual_reports[name], parallel_manual_reports[name])


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_metrics_parallel_rules_matches_serial(mining_data, backend):
    """批量规则指标评估必须保持输入规则顺序和精确结果。"""
    X = mining_data[["score", "debt"]]
    y = mining_data["target"]
    rules = [Rule("score > 0", name="正分"), Rule("debt <= 0", name="低负债")]
    serial = RuleMetrics(n_jobs=1).evaluate_rules(rules, X, y)
    parallel = RuleMetrics(n_jobs=2, parallel_backend=backend).evaluate_rules(rules, X, y)
    pd.testing.assert_frame_equal(serial, parallel)


class _FailingSingleFeatureMiner(SingleFeatureRuleMiner):
    fail_feature = None

    def _analyze_feature(self, feature):
        if feature == self.fail_feature:
            raise RuntimeError("故意失败")
        return super()._analyze_feature(feature)


def test_single_feature_fit_is_transactional_on_first_and_refit_failure(mining_data):
    """feature 失败不得留下部分新状态，重拟合失败必须保留旧状态。"""
    data = mining_data[["score", "debt", "target"]]
    first = _FailingSingleFeatureMiner(n_jobs=2, parallel_backend="threading")
    first.fail_feature = "debt"
    with pytest.raises(ParallelExecutionError) as error:
        first.fit(data)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert first.results_ == {}
    assert first._is_fitted is False

    refit = _FailingSingleFeatureMiner(n_jobs=2, parallel_backend="threading").fit(data)
    old_results = {feature: table.copy(deep=True) for feature, table in refit.results_.items()}
    refit.fail_feature = "debt"
    with pytest.raises(ParallelExecutionError):
        refit.fit(data)
    assert list(refit.results_) == list(old_results)
    for feature in old_results:
        pd.testing.assert_frame_equal(refit.results_[feature], old_results[feature])
    assert refit._is_fitted is True


def test_mining_parallel_configuration_validates_lazily_in_chinese(mining_data):
    """构造保存原始配置，执行时才由公共运行时抛中文 ValidationError。"""
    config = {"batch_size": 2}
    miner = SingleFeatureRuleMiner(n_jobs=0.5, parallel_config=config)
    assert miner.parallel_config is config
    clone(miner)
    pickle.loads(pickle.dumps(miner))

    invalid = SingleFeatureRuleMiner(n_jobs=0)
    with pytest.raises(ValidationError, match="n_jobs"):
        invalid.fit(mining_data[["score", "target"]])


def test_single_and_empty_mining_tasks_do_not_create_joblib_backend(monkeypatch, mining_data):
    """空任务及单任务即使默认 -1 也必须走公共串行路径。"""
    import hscredit.utils.parallel as parallel_module

    def forbidden_parallel(*args, **kwargs):
        raise AssertionError("不应创建 joblib 后端")

    monkeypatch.setattr(parallel_module, "_create_joblib_parallel", forbidden_parallel)
    SingleFeatureRuleMiner().fit(mining_data[["score", "target"]])
    SingleFeatureRuleMiner().fit(mining_data[["target"]])
