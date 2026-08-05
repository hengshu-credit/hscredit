"""规则挖掘模块统一并行执行测试。"""

import copy
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


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_tree_rule_reports_parallel_match_serial_and_fail_fast(mining_data, backend):
    """规则报告并行必须精确一致，任一规则失败不得静默跳过。"""
    data = mining_data[["score", "debt", "target"]]
    kwargs = dict(algorithm="rf", n_estimators=4, max_depth=3, random_state=13)
    serial = TreeRuleExtractor(n_jobs=1, **kwargs).fit(data)
    parallel = TreeRuleExtractor(n_jobs=2, parallel_backend=backend, **kwargs).fit(data)
    serial_rules = serial.get_rules(datasets=data, min_samples=1)
    parallel_rules = parallel.get_rules(datasets=data, min_samples=1)
    assert [rule.expr for rule in serial_rules] == [rule.expr for rule in parallel_rules]
    for left, right in zip(serial_rules, parallel_rules):
        assert left.metric_score_ == right.metric_score_
        assert left.metadata_ == right.metadata_

    parallel.extract_rules()
    parallel.rules_[0]["conditions"][0]["feature"] = "不存在特征"
    with pytest.raises(ParallelExecutionError) as error:
        parallel.get_rules(datasets=data, min_samples=1)
    assert error.value.__cause__ is not None


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


class _FailingBinner:
    def fit(self, X, y):
        raise ValueError("分箱器故意失败")


class _BinnerFailingSingleFeatureMiner(SingleFeatureRuleMiner):
    def _get_binning_instance(self):
        return _FailingBinner()


class _BinnerFailingMultiFeatureMiner(MultiFeatureRuleMiner):
    def _get_binning_instance(self, **override_params):
        return _FailingBinner()


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


def test_mining_does_not_hide_binner_failures_or_use_approximate_fallback(mining_data):
    """底层分箱失败必须保留直接 cause，不能退化为 quantile/原始值。"""
    data = mining_data[["score", "debt", "target"]]
    with pytest.raises(ParallelExecutionError) as single_error:
        _BinnerFailingSingleFeatureMiner(n_jobs=2, parallel_backend="threading").fit(data)
    assert isinstance(single_error.value.__cause__, ValueError)

    multi = _BinnerFailingMultiFeatureMiner(
        max_n_bins=2,
        min_samples=1,
        min_lift=0.0,
        n_jobs=2,
        parallel_backend="threading",
    ).fit(data)
    with pytest.raises(ParallelExecutionError) as multi_error:
        multi.get_all_cross_rules(top_n=1, max_feature_pairs=1)
    assert isinstance(multi_error.value.__cause__, ValueError)


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


@pytest.mark.parametrize(
    "estimator",
    [
        SingleFeatureRuleMiner(parallel_config={"batch_size": 2}),
        MultiFeatureRuleMiner(parallel_config={"batch_size": 2}),
        MultiLabelRuleMiner(
            labels=["短期", "长期"],
            label_names=["短期", "长期"],
            parallel_config={"batch_size": 2},
        ),
        TreeRuleExtractor(
            algorithm="RF",
            feature_trends={"score": 1},
            parallel_config={"batch_size": 2},
        ),
    ],
)
def test_sklearn_miners_clone_preserves_parallel_and_public_parameters(estimator):
    """构造函数规范化任何公开参数都会破坏 sklearn clone 契约。"""
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()


@pytest.mark.parametrize(
    ("labels", "label_names", "n_jobs", "backend", "config"),
    [
        (None, None, -1, None, None),
        ([], [], 1, None, {}),
        (["短期", "长期"], ["MOB1", "MOB3"], 0.5, "threading", {"batch_size": 2}),
    ],
)
def test_multi_label_clone_and_pickle_preserve_raw_constructor_parameters(
    labels, label_names, n_jobs, backend, config
):
    """None、空列表和用户容器均须原样保存，clone/pickle 后保持等值。"""
    miner = MultiLabelRuleMiner(
        labels=labels,
        label_names=label_names,
        n_jobs=n_jobs,
        parallel_backend=backend,
        parallel_config=config,
    )
    assert miner.labels is labels
    assert miner.label_names is label_names
    assert miner.parallel_config is config

    cloned = clone(miner)
    assert cloned.get_params() == miner.get_params()
    restored = pickle.loads(pickle.dumps(miner))
    assert restored.get_params() == miner.get_params()


@pytest.mark.parametrize(
    ("labels", "label_names", "config"),
    [
        (None, None, None),
        ([], [], {}),
        (["target", "long_target"], ["短期", "长期"], {"batch_size": 2}),
    ],
)
def test_multi_label_successful_fit_preserves_constructor_parameter_identity(
    mining_data, labels, label_names, config
):
    """成功拟合只能提交学习状态，不能用深拷贝替换调用方传入的公开参数。"""
    miner = MultiLabelRuleMiner(
        labels=labels,
        label_names=label_names,
        min_support=0.01,
        min_lift=1.0,
        max_rules=3,
        n_bins=3,
        n_jobs=1,
        parallel_config=config,
    ).fit(mining_data, features=["score"])

    fitted_params = miner.get_params(deep=False)
    assert miner.labels is labels
    assert miner.label_names is label_names
    assert miner.parallel_config is config
    assert fitted_params["labels"] is labels
    assert fitted_params["label_names"] is label_names
    assert fitted_params["parallel_config"] is config

    cloned = clone(miner)
    assert cloned.get_params() == miner.get_params()
    restored = pickle.loads(pickle.dumps(miner))
    assert restored.get_params() == miner.get_params()


def test_transactional_rule_miners_preserve_mutable_constructor_parameter_identity(mining_data):
    """所有基于临时副本拟合的 sklearn 规则挖掘器都必须保留公开可变参数引用。"""
    data = mining_data[["score", "debt", "target"]]
    exclude_cols = []
    special_codes = [-999]
    config = {"batch_size": 2}
    single = SingleFeatureRuleMiner(
        method="quantile",
        max_n_bins=3,
        min_samples=2,
        min_lift=0.0,
        exclude_cols=exclude_cols,
        special_codes=special_codes,
        n_jobs=1,
        parallel_config=config,
    ).fit(data)
    identity_results = {
        "single.exclude_cols": single.exclude_cols is exclude_cols,
        "single.special_codes": single.special_codes is special_codes,
        "single.parallel_config": single.parallel_config is config,
    }

    exclude_cols = []
    special_codes = [-999]
    config = {"batch_size": 2}
    multi = MultiFeatureRuleMiner(
        max_n_bins=3,
        min_samples=2,
        min_lift=0.0,
        exclude_cols=exclude_cols,
        special_codes=special_codes,
        n_jobs=1,
        parallel_config=config,
    ).fit(data)
    identity_results.update(
        {
            "multi.exclude_cols": multi.exclude_cols is exclude_cols,
            "multi.special_codes": multi.special_codes is special_codes,
            "multi.parallel_config": multi.parallel_config is config,
        }
    )

    exclude_cols = []
    feature_trends = {"score": 1}
    config = {"batch_size": 2}
    tree = TreeRuleExtractor(
        algorithm="dt",
        max_depth=2,
        exclude_cols=exclude_cols,
        feature_trends=feature_trends,
        n_jobs=1,
        parallel_config=config,
    ).fit(data)
    identity_results.update(
        {
            "tree.exclude_cols": tree.exclude_cols is exclude_cols,
            "tree.feature_trends": tree.feature_trends is feature_trends,
            "tree.parallel_config": tree.parallel_config is config,
        }
    )

    assert identity_results == {name: True for name in identity_results}

    for estimator in (single, multi, tree):
        assert clone(estimator).get_params() == estimator.get_params()
        assert pickle.loads(pickle.dumps(estimator)).get_params() == estimator.get_params()


def test_transactional_tree_helpers_preserve_mutable_constructor_parameter_identity(mining_data, tmp_path):
    """非 sklearn 树辅助器成功提交学习状态时也不能替换调用方配置对象。"""
    data = mining_data[["score", "debt", "target"]]
    features = ["score", "debt"]
    tree_params = {"max_depth": 2, "random_state": 7}
    config = {"batch_size": 2}
    analyzer = DecisionTreeAnalyzer(
        features=features,
        tree_params=tree_params,
        n_jobs=1,
        parallel_config=config,
    ).fit(data)
    identity_results = {
        "analyzer.features": analyzer.features is features,
        "analyzer.tree_params": analyzer.tree_params is tree_params,
        "analyzer.parallel_config": analyzer.parallel_config is config,
    }

    config = {"batch_size": 2}
    manual = ManualTreeExtractor(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        parallel_config=config,
    ).fit(data, features=features)
    identity_results["manual.parallel_config"] = manual.parallel_config is config
    assert identity_results == {name: True for name in identity_results}

    empty_features = []
    empty_tree_params = {}
    empty_config = {}
    inferred = DecisionTreeAnalyzer(
        features=empty_features,
        tree_params=empty_tree_params,
        n_jobs=1,
        parallel_config=empty_config,
    ).fit(data, features=["score", "debt"])
    assert inferred.features is empty_features
    assert inferred.tree_params is empty_tree_params
    assert inferred.parallel_config is empty_config
    assert inferred.features_ == ["score", "debt"]
    assert len(inferred.predict(data)) == len(data)

    model_path = tmp_path / "decision_tree_analyzer.pkl"
    inferred.save(str(model_path))
    restored = DecisionTreeAnalyzer.load(str(model_path))
    assert restored.features_ == inferred.features_
    np.testing.assert_array_equal(restored.predict(data), inferred.predict(data))


def test_multi_feature_successful_refit_discards_old_cross_and_binner_caches(mining_data):
    """第二批数据拟合成功后不得保留仅属于第一批特征的交叉结果或分箱器。"""
    first = mining_data.rename(columns={"score": "a", "debt": "b", "long_target": "c"})[
        ["a", "b", "c", "target"]
    ]
    second = mining_data.rename(columns={"score": "x", "debt": "y"})[["x", "y", "target"]]
    kwargs = dict(
        method="quantile",
        max_n_bins=3,
        min_samples=2,
        min_lift=0.0,
        n_jobs=1,
    )
    refitted = MultiFeatureRuleMiner(**kwargs).fit(first)
    refitted.get_all_cross_rules(top_n=2, max_feature_pairs=3)
    assert refitted.cross_results_
    assert refitted._binner_instances_

    refitted.fit(second)
    fresh = MultiFeatureRuleMiner(**kwargs).fit(second)
    assert refitted.cross_results_ == fresh.cross_results_ == {}
    assert refitted._binner_instances_ == fresh._binner_instances_ == {}

    refit_rules = refitted.get_all_cross_rules(top_n=2, max_feature_pairs=1)
    fresh_rules = fresh.get_all_cross_rules(top_n=2, max_feature_pairs=1)
    pd.testing.assert_frame_equal(refit_rules, fresh_rules)
    assert list(refitted.cross_results_) == list(fresh.cross_results_) == [("x", "y")]
    assert set(refitted._binner_instances_) == set(fresh._binner_instances_) <= {"x", "y"}


def test_tree_successful_refit_discards_rules_derived_from_old_model(mining_data):
    """重拟合必须使旧规则缓存失效，下一次 get_rules 只能从新模型重新提取。"""
    first = mining_data.rename(columns={"score": "old_score", "debt": "old_debt"})[
        ["old_score", "old_debt", "target"]
    ]
    second = mining_data.rename(columns={"score": "x", "debt": "y"})[["x", "y", "target"]]
    kwargs = dict(algorithm="dt", max_depth=2, random_state=19, n_jobs=1)
    refitted = TreeRuleExtractor(**kwargs).fit(first)
    assert refitted.extract_rules()

    refitted.fit(second)
    fresh = TreeRuleExtractor(**kwargs).fit(second)
    assert refitted.rules_ == fresh.rules_ == []

    refit_rules = refitted.get_rules(min_samples=1)
    fresh_rules = fresh.get_rules(min_samples=1)
    assert [rule.expr for rule in refit_rules] == [rule.expr for rule in fresh_rules]
    assert all("old_score" not in rule.expr and "old_debt" not in rule.expr for rule in refit_rules)


def test_tree_cross_algorithm_refit_removes_old_optional_learned_attributes(mining_data):
    """从监督树切换到孤立森林后，监督/卡方专属状态必须与 fresh 实例同样不存在。"""
    supervised = mining_data[["score", "debt", "target"]]
    unsupervised = mining_data.rename(columns={"score": "x", "debt": "y"})[["x", "y"]]
    refitted = TreeRuleExtractor(
        algorithm="chi2",
        n_estimators=3,
        max_depth=2,
        random_state=23,
        n_jobs=1,
    ).fit(supervised)
    refitted.extract_rules()
    assert hasattr(refitted, "chi2_bins_")
    assert hasattr(refitted, "y_train_")

    refitted.algorithm = "isf"
    refitted.fit(unsupervised)
    fresh = TreeRuleExtractor(
        algorithm="isf",
        n_estimators=3,
        max_depth=2,
        random_state=23,
        n_jobs=1,
    ).fit(unsupervised)
    for attribute in ("X_test_", "y_train_", "y_test_", "chi2_bins_"):
        assert hasattr(refitted, attribute) is hasattr(fresh, attribute) is False
    assert refitted.feature_names_ == fresh.feature_names_ == ["x", "y"]
    assert refitted.rules_ == fresh.rules_ == []
    assert _normalize_tree_rules(refitted.extract_rules()) == _normalize_tree_rules(fresh.extract_rules())


def test_single_successful_refit_matches_fresh_instance(mining_data):
    """Single 的第二次成功拟合状态必须等同 fresh 实例。"""
    first = mining_data[["score", "debt", "target", "long_target"]]
    second = mining_data.rename(
        columns={"score": "x", "debt": "y", "target": "next_target"}
    )[["x", "y", "next_target"]]

    single_kwargs = dict(
        target="target",
        method="quantile",
        max_n_bins=3,
        min_samples=2,
        min_lift=0.0,
        n_jobs=1,
    )
    refit_single = SingleFeatureRuleMiner(**single_kwargs).fit(first[["score", "debt", "target"]])
    refit_single.target = "next_target"
    refit_single.fit(second)
    fresh_single = SingleFeatureRuleMiner(**{**single_kwargs, "target": "next_target"}).fit(second)
    _assert_miner_results_equal(refit_single, fresh_single)
    assert refit_single.features_ == fresh_single.features_ == ["x", "y"]
    assert set(refit_single._binning_instances_) == set(fresh_single._binning_instances_) <= {"x", "y"}


def test_multi_label_successful_refit_matches_fresh_instance(mining_data):
    """MultiLabel 动态标签重拟合后不得保留旧目标或旧规则。"""
    first = mining_data[["score", "debt", "target", "long_target"]]
    second = mining_data.rename(
        columns={"score": "x", "debt": "y", "target": "next_target"}
    )[["x", "y", "next_target"]]
    labels = ["target", "long_target"]
    label_names = ["短期", "长期"]
    label_kwargs = dict(
        min_support=0.01,
        min_lift=1.0,
        max_rules=3,
        n_bins=3,
        n_jobs=1,
    )
    refit_labels = MultiLabelRuleMiner(labels=labels, label_names=label_names, **label_kwargs).fit(
        first, features=["score"]
    )
    labels[:] = ["next_target"]
    label_names[:] = ["新标签"]
    refit_labels.fit(second, features=["x", "y"])
    fresh_labels = MultiLabelRuleMiner(labels=labels, label_names=label_names, **label_kwargs).fit(
        second, features=["x", "y"]
    )
    assert refit_labels.target == fresh_labels.target == "next_target"
    pd.testing.assert_frame_equal(refit_labels.get_report(), fresh_labels.get_report())


def test_multi_label_target_is_not_a_public_constructor_parameter():
    """既有签名不公开 target，set_params 不应伪装支持不可 clone 的隐藏参数。"""
    assert "target" not in inspect.signature(MultiLabelRuleMiner.__init__).parameters
    miner = MultiLabelRuleMiner(labels=["target"])
    assert "target" not in miner.get_params(deep=False)
    with pytest.raises(ValueError, match="target"):
        miner.set_params(target="custom_target")


def test_multi_label_auto_target_tracks_current_labels_and_empty_fallback(mining_data):
    """自动模式每轮跟随当前首标签，空标签回退默认 target。"""
    labels = ["target", "long_target"]
    miner = MultiLabelRuleMiner(
        labels=labels,
        min_support=0.01,
        min_lift=1.0,
        n_bins=3,
        n_jobs=1,
    ).fit(mining_data, features=["score"])
    assert miner.target == "target"

    labels[:] = ["long_target"]
    miner.fit(mining_data, features=["score"])
    assert miner.target == "long_target"
    assert miner.labels is labels

    labels.clear()
    miner.fit(mining_data, features=["score"])
    assert miner.target == "target"
    assert miner.labels is labels


def test_multi_label_direct_target_assignment_enables_explicit_mode(mining_data):
    """直接赋值 target（即使值等于旧自动值）后，fit/refit 均不得被 labels 覆盖。"""
    labels = ["target", "long_target"]
    miner = MultiLabelRuleMiner(
        labels=labels,
        min_support=0.01,
        min_lift=1.0,
        n_bins=3,
        n_jobs=1,
    )
    miner.target = "target"
    labels[:] = ["long_target"]
    miner.fit(mining_data, features=["score"])
    assert miner.target == "target"

    labels[:] = ["target", "long_target"]
    miner.target = "custom_target"
    miner.fit(mining_data, features=["debt"])
    assert miner.target == "custom_target"
    assert miner.labels is labels
    report_columns = set(miner.get_report().columns.astype(str))
    assert "target_LIFT" in report_columns
    assert "long_target_LIFT" in report_columns

    restored = pickle.loads(pickle.dumps(miner))
    labels_after_pickle = restored.labels
    labels_after_pickle[:] = ["long_target"]
    restored.fit(mining_data, features=["score"])
    assert restored.target == "custom_target"

    copied = copy.deepcopy(miner)
    copied.labels[:] = ["long_target"]
    copied.fit(mining_data, features=["score"])
    assert copied.target == "custom_target"

    cloned = clone(miner)
    assert cloned.get_params() == miner.get_params()
    assert cloned.target == labels[0]


def test_multi_label_failed_refit_preserves_auto_and_explicit_target_modes(mining_data):
    """失败 working 副本不能提交 target 值或自动/显式模式标记。"""
    auto_labels = ["target"]
    auto = MultiLabelRuleMiner(labels=auto_labels, n_jobs=1).fit(
        mining_data, features=["score"]
    )
    auto_labels[:] = ["missing_label"]
    with pytest.raises(ValueError, match="missing_label"):
        auto.fit(mining_data, features=["score"])
    assert auto.target == "target"
    auto_labels[:] = ["long_target"]
    auto.fit(mining_data, features=["score"])
    assert auto.target == "long_target"

    explicit_labels = ["target"]
    explicit = MultiLabelRuleMiner(labels=explicit_labels, n_jobs=1)
    explicit.target = "custom_target"
    explicit_labels[:] = ["missing_label"]
    with pytest.raises(ValueError, match="missing_label"):
        explicit.fit(mining_data, features=["score"])
    assert explicit.target == "custom_target"
    explicit_labels[:] = ["long_target"]
    explicit.fit(mining_data, features=["score"])
    assert explicit.target == "custom_target"


def test_tree_helpers_successful_refit_matches_fresh_instance(mining_data):
    """DTA、MTE 的第二次成功拟合状态必须等同 fresh 实例。"""
    first = mining_data[["score", "debt", "target"]]
    second = mining_data.rename(columns={"score": "x", "debt": "y"})[["x", "y", "target"]]
    analyzer_features = ["score", "debt"]
    analyzer_kwargs = dict(features=analyzer_features, tree_params={"max_depth": 2}, n_jobs=1)
    refit_analyzer = DecisionTreeAnalyzer(**analyzer_kwargs).fit(first[["score", "debt", "target"]])
    _ = refit_analyzer._tree_info
    refit_analyzer.fit(second, features=["x", "y"])
    fresh_analyzer = DecisionTreeAnalyzer(**analyzer_kwargs).fit(second, features=["x", "y"])
    assert refit_analyzer.features_ == fresh_analyzer.features_ == ["x", "y"]
    pd.testing.assert_frame_equal(refit_analyzer._df_rules, fresh_analyzer._df_rules)
    np.testing.assert_array_equal(refit_analyzer.predict(second), fresh_analyzer.predict(second))

    manual_kwargs = dict(max_depth=2, min_samples_split=2, min_samples_leaf=1, n_jobs=1)
    refit_manual = ManualTreeExtractor(**manual_kwargs).fit(
        first[["score", "debt", "target"]], features=["score", "debt"]
    )
    refit_manual.fit(second, features=["x", "y"])
    fresh_manual = ManualTreeExtractor(**manual_kwargs).fit(second, features=["x", "y"])
    assert refit_manual._feature_list == fresh_manual._feature_list == ["x", "y"]
    assert refit_manual._manual_split_nodes == fresh_manual._manual_split_nodes == set()
    pd.testing.assert_frame_equal(refit_manual._df_rules, fresh_manual._df_rules)


def test_single_and_empty_mining_tasks_do_not_create_joblib_backend(monkeypatch, mining_data):
    """空任务及单任务即使默认 -1 也必须走公共串行路径。"""
    import hscredit.utils.parallel as parallel_module

    def forbidden_parallel(*args, **kwargs):
        raise AssertionError("不应创建 joblib 后端")

    monkeypatch.setattr(parallel_module, "_create_joblib_parallel", forbidden_parallel)
    SingleFeatureRuleMiner().fit(mining_data[["score", "target"]])
    SingleFeatureRuleMiner().fit(mining_data[["target"]])
