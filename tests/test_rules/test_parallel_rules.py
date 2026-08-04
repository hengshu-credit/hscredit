"""规则执行统一并行配置与结果一致性测试。"""

import inspect
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.models import LogicOperator, RuleSet, RulesClassifier
from hscredit.core.models.rules import rule_classifier as rule_classifier_module
from hscredit.core.rules import Rule, RuleFlow
from hscredit.exceptions import ParallelExecutionError, ValidationError
from hscredit.utils.parallel import ParallelBudget, _ACTIVE_BUDGET


@pytest.fixture
def rule_data():
    return pd.DataFrame(
        {
            "score": [420, 560, 680, 610, 450, 720],
            "multi": [2, 8, 9, 1, 6, 10],
            "商品类别": ["手机", "礼包", "手机", "家电", "礼包", "家电"],
            "target": [1, 1, 0, 0, 1, 0],
            "MOB1": [8, 4, 0, 2, 7, 1],
            "MOB2": [5, 0, 1, 6, 3, 9],
            "放款金额": [1000.0, 2000.0, 1500.0, 3000.0, 1200.0, 800.0],
            "放款时间": pd.to_datetime(
                ["2024-01-02", "2024-01-15", "2024-02-03", "2024-02-18", "2024-03-02", "2024-03-20"]
            ),
        },
        index=pd.Index([31, 12, 44, 7, 28, 19], name="订单索引"),
    )


@pytest.mark.parametrize("cls", [Rule, RuleFlow, RuleSet, RulesClassifier])
def test_rule_components_expose_common_parallel_defaults(cls):
    params = inspect.signature(cls.__init__).parameters
    assert params["n_jobs"].default == -1
    assert params["parallel_backend"].default is None
    assert params["parallel_config"].default is None


def test_rule_components_preserve_config_and_support_clone_pickle():
    config = {"batch_size": 2}
    rule = Rule(
        "score < 500",
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )
    flow = RuleFlow(
        [rule],
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )
    ruleset = RuleSet(
        rules=[rule],
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )
    classifier = RulesClassifier(
        rules=[rule],
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )

    for component in (rule, flow, ruleset, classifier):
        assert component.n_jobs == 0.5
        assert component.parallel_backend == "threading"
        assert component.parallel_config is config
        restored = pickle.loads(pickle.dumps(component))
        assert restored.n_jobs == 0.5
        assert restored.parallel_backend == "threading"
        assert restored.parallel_config == config

    cloned = clone(classifier)
    assert cloned.n_jobs == 0.5
    assert cloned.parallel_backend == "threading"
    assert cloned.parallel_config == config
    assert cloned.parallel_config is not config


def test_rule_composition_uses_left_operand_parallel_configuration():
    left_config = {"batch_size": 1}
    left = Rule("score < 500", n_jobs=2, parallel_backend="threading", parallel_config=left_config)
    right = Rule("multi > 6", n_jobs=3, parallel_backend="loky", parallel_config={"batch_size": 3})

    for combined in (left & right, left | right, left ^ right, ~left):
        assert combined.n_jobs == 2
        assert combined.parallel_backend == "threading"
        assert combined.parallel_config is left_config

    left_set = RuleSet(
        name="左规则集",
        rules=[left],
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=left_config,
    )
    combined_set = left_set | right
    assert combined_set.n_jobs == 2
    assert combined_set.parallel_backend == "threading"
    assert combined_set.parallel_config is left_config


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_flow_parallel_mode_matches_serial(rule_data, backend):
    serial = RuleFlow(
        [Rule("score < 500", name="低分"), Rule("multi > 6", name="多头")],
        mode="parallel",
        n_jobs=1,
    ).predict(rule_data)
    parallel = RuleFlow(
        [Rule("score < 500", name="低分"), Rule("multi > 6", name="多头")],
        mode="parallel",
        n_jobs=2,
        parallel_backend=backend,
    ).predict(rule_data)

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)


def test_rule_flow_serial_mode_keeps_strict_filter_order_when_n_jobs_exceeds_one(rule_data):
    first = Rule("score < 500", name="低分")
    second = Rule("multi > 6", name="多头")
    seen = []
    original_first = first.predict
    original_second = second.predict

    def predict_first(data):
        seen.append(("低分", data.index.tolist()))
        return original_first(data)

    def predict_second(data):
        seen.append(("多头", data.index.tolist()))
        return original_second(data)

    first.predict = predict_first
    second.predict = predict_second
    result = RuleFlow(
        [first, second],
        mode="serial",
        n_jobs=2,
        parallel_backend="threading",
    ).predict(rule_data)

    assert seen == [("低分", [31, 12, 44, 7, 28, 19]), ("多头", [12, 44, 7, 19])]
    assert result["命中规则"].tolist() == ["低分", "多头", "多头", "", "低分", "多头"]


def _assert_rule_results_equal(left, right):
    np.testing.assert_array_equal(left[0], right[0])
    assert len(left[1]) == len(right[1])
    for left_detail, right_detail in zip(left[1], right[1]):
        assert left_detail.rule_name == right_detail.rule_name
        assert left_detail.expression == right_detail.expression
        assert left_detail.matched == right_detail.matched
        assert left_detail.matched_count == right_detail.matched_count
        np.testing.assert_array_equal(left_detail.matched_indices, right_detail.matched_indices)
        left_sub = left_detail.details.get("sub_results", [])
        right_sub = right_detail.details.get("sub_results", [])
        assert [detail.rule_name for detail in left_sub] == [detail.rule_name for detail in right_sub]
        assert [detail.expression for detail in left_sub] == [detail.expression for detail in right_sub]
        assert [detail.matched_count for detail in left_sub] == [detail.matched_count for detail in right_sub]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_nested_ruleset_masks_match_serial_in_declared_order(rule_data, backend):
    def build(n_jobs, selected_backend):
        nested = RuleSet(
            name="嵌套规则",
            logic=LogicOperator.AND,
            rules=[Rule("score < 650", name="分数"), Rule("multi > 3", name="多头")],
            n_jobs=n_jobs,
            parallel_backend=selected_backend,
        )
        return RuleSet(
            name="外层规则",
            logic=LogicOperator.OR,
            rules=[nested, Rule("`商品类别` == '家电'", name="家电")],
            n_jobs=n_jobs,
            parallel_backend=selected_backend,
        )

    serial = build(1, None).evaluate(rule_data)
    parallel = build(2, backend).evaluate(rule_data)
    _assert_rule_results_equal(serial, parallel)
    assert [detail.rule_name for detail in parallel[1]] == ["嵌套规则", "家电"]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rules_classifier_prediction_probability_and_masks_match_serial(rule_data, backend):
    def build(n_jobs, selected_backend, output_mode):
        ruleset = RuleSet(
            name="风险组合",
            logic="and",
            rules=[Rule("score < 650", name="分数"), Rule("multi > 3", name="多头")],
            n_jobs=n_jobs,
            parallel_backend=selected_backend,
        )
        return RulesClassifier(
            rules=[ruleset, Rule("`商品类别` == '家电'", name="家电")],
            logic="or",
            output_mode=output_mode,
            weights=[2.0, 1.0],
            threshold=0.5,
            n_jobs=n_jobs,
            parallel_backend=selected_backend,
        ).fit(rule_data.drop(columns="target"), rule_data["target"])

    serial = build(1, None, "both")
    parallel = build(2, backend, "both")
    serial_final, serial_masks = serial.predict(rule_data.drop(columns="target"))
    parallel_final, parallel_masks = parallel.predict(rule_data.drop(columns="target"))
    np.testing.assert_array_equal(serial_final, parallel_final)
    pd.testing.assert_frame_equal(serial_masks, parallel_masks, check_exact=True)
    np.testing.assert_array_equal(
        serial.predict_proba(rule_data.drop(columns="target")),
        parallel.predict_proba(rule_data.drop(columns="target")),
    )


def test_rules_classifier_failed_refit_restores_old_learned_state(rule_data):
    classifier = RulesClassifier(rules=[Rule("score < 500")], n_jobs=2, parallel_backend="threading")
    classifier.fit(rule_data[["score", "multi"]], rule_data["target"])
    old_state = (
        classifier.n_features_in_,
        classifier.feature_names_in_.copy(),
        classifier.classes_.copy(),
        classifier.weights_.copy(),
        classifier._is_fitted,
    )

    with pytest.raises(ValueError, match="不存在的特征"):
        classifier.fit(rule_data[["multi"]], rule_data["target"])

    assert classifier.n_features_in_ == old_state[0]
    assert classifier.feature_names_in_ == old_state[1]
    np.testing.assert_array_equal(classifier.classes_, old_state[2])
    assert classifier.weights_ == old_state[3]
    assert classifier._is_fitted is old_state[4]


def test_rules_classifier_first_fit_failure_leaves_no_learned_state(rule_data):
    classifier = RulesClassifier(rules=[Rule("score < 500")], n_jobs=2, parallel_backend="threading")

    with pytest.raises(ValueError, match="不存在的特征"):
        classifier.fit(rule_data[["multi"]], rule_data["target"])

    for attribute in ("n_features_in_", "feature_names_in_", "classes_", "weights_", "_is_fitted"):
        assert not hasattr(classifier, attribute)


def test_rules_classifier_failed_prediction_preserves_learned_and_rule_state(rule_data):
    good_rule = Rule("score < 500", name="正常规则")
    classifier = RulesClassifier(rules=[good_rule], n_jobs=2, parallel_backend="threading").fit(
        rule_data[["score", "multi"]], rule_data["target"]
    )
    classifier.predict(rule_data[["score", "multi"]])
    old_result = good_rule.result().copy()
    old_features = classifier.feature_names_in_.copy()
    old_classes = classifier.classes_.copy()
    classifier.rules.append(Rule("missing_feature > 0", name="失败规则"))

    with pytest.raises(ParallelExecutionError, match="失败规则"):
        classifier.predict(rule_data[["score", "multi"]])

    pd.testing.assert_series_equal(good_rule.result(), old_result, check_exact=True)
    assert classifier.feature_names_in_ == old_features
    np.testing.assert_array_equal(classifier.classes_, old_classes)


@pytest.mark.parametrize("fitted", [False, True])
def test_rules_classifier_predict_postprocessing_failure_is_transactional(monkeypatch, fitted):
    train = pd.DataFrame({"score": [1, -1]}, index=[10, 20])
    changed = pd.DataFrame({"score": [-1, 1]}, index=[10, 20])
    rule = Rule("score > 0", name="分数规则")
    classifier = RulesClassifier(
        rules=[rule],
        output_mode="reason",
        n_jobs=2,
        parallel_backend="threading",
    )
    if fitted:
        classifier.fit(train).predict(train, return_reason=True)
        old_result = rule.result().copy()
        old_state = (
            classifier.n_features_in_,
            classifier.feature_names_in_.copy(),
            classifier.classes_.copy(),
            classifier.weights_.copy(),
            classifier._is_fitted,
        )

    def fail_after_workers(*args, **kwargs):
        raise RuntimeError("输出后处理失败")

    monkeypatch.setattr(classifier, "_generate_reasons", fail_after_workers)
    with pytest.raises(RuntimeError, match="输出后处理失败"):
        classifier.predict(changed, return_reason=True)

    if fitted:
        pd.testing.assert_series_equal(rule.result(), old_result, check_exact=True)
        assert classifier.n_features_in_ == old_state[0]
        assert classifier.feature_names_in_ == old_state[1]
        np.testing.assert_array_equal(classifier.classes_, old_state[2])
        assert classifier.weights_ == old_state[3]
        assert classifier._is_fitted is old_state[4]
    else:
        assert rule.result_ is None
        for attribute in ("n_features_in_", "feature_names_in_", "classes_", "weights_", "_is_fitted"):
            assert not hasattr(classifier, attribute)


@pytest.mark.parametrize("fitted", [False, True])
def test_rules_classifier_predict_proba_postprocessing_failure_is_transactional(monkeypatch, fitted):
    train = pd.DataFrame({"score": [1, -1]}, index=[10, 20])
    changed = pd.DataFrame({"score": [-1, 1]}, index=[10, 20])
    rule = Rule("score > 0", name="分数规则")
    classifier = RulesClassifier(rules=[rule], n_jobs=2, parallel_backend="threading")
    if fitted:
        classifier.fit(train).predict_proba(train)
        old_result = rule.result().copy()
        old_state = (
            classifier.n_features_in_,
            classifier.feature_names_in_.copy(),
            classifier.classes_.copy(),
            classifier.weights_.copy(),
            classifier._is_fitted,
        )

    def fail_after_workers(*args, **kwargs):
        raise RuntimeError("概率后处理失败")

    monkeypatch.setattr(rule_classifier_module.np, "clip", fail_after_workers)
    with pytest.raises(RuntimeError, match="概率后处理失败"):
        classifier.predict_proba(changed)

    if fitted:
        pd.testing.assert_series_equal(rule.result(), old_result, check_exact=True)
        assert classifier.n_features_in_ == old_state[0]
        assert classifier.feature_names_in_ == old_state[1]
        np.testing.assert_array_equal(classifier.classes_, old_state[2])
        assert classifier.weights_ == old_state[3]
        assert classifier._is_fitted is old_state[4]
    else:
        assert rule.result_ is None
        for attribute in ("n_features_in_", "feature_names_in_", "classes_", "weights_", "_is_fitted"):
            assert not hasattr(classifier, attribute)


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf, "1", True, object()])
def test_rules_classifier_rejects_non_finite_or_non_numeric_weights_before_workers(monkeypatch, weight):
    classifier = RulesClassifier(
        rules=[Rule("score > 0")],
        weights=[weight],
        n_jobs=2,
        parallel_backend="threading",
    )

    def worker_must_not_run(task):
        raise AssertionError("权重校验后不应启动worker")

    monkeypatch.setattr(rule_classifier_module, "_classifier_component_worker", worker_must_not_run)
    with pytest.raises(ValidationError, match="weights"):
        classifier.predict(pd.DataFrame({"score": [1, -1]}))


def test_rules_classifier_accepts_finite_numpy_scalar_weights():
    classifier = RulesClassifier(rules=[Rule("score > 0")], weights=[np.float64(1.5)], n_jobs=1)
    np.testing.assert_array_equal(
        classifier.predict(pd.DataFrame({"score": [1, -1]})),
        np.array([1, 0]),
    )


class _ConcurrencyProbeRule(Rule):
    """通过跨线程/进程独占文件探测并发 worker 数。"""

    def __init__(self, expr, name, lock_path):
        super().__init__(expr, name=name)
        self.lock_path = str(lock_path)

    def predict(self, X):
        descriptor = None
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with open(f"{self.lock_path}.conflict", "w", encoding="utf-8") as stream:
                stream.write("并发超出预算")
        else:
            time.sleep(0.1)
        finally:
            if descriptor is not None:
                os.close(descriptor)
                os.unlink(self.lock_path)
        return super().predict(X)


def _deep_parallel_ruleset(lock_path, backend):
    leaf = RuleSet(
        name="叶规则集",
        rules=[
            _ConcurrencyProbeRule("score > -10", name="叶规则1", lock_path=lock_path),
            _ConcurrencyProbeRule("score < 10", name="叶规则2", lock_path=lock_path),
        ],
        n_jobs=2,
        parallel_backend=backend,
    )
    middle = RuleSet(
        name="中间规则集",
        rules=[leaf],
        n_jobs=1,
        parallel_backend=backend,
    )
    outer = RuleSet(
        name="外层规则集",
        logic="or",
        rules=[middle, Rule("score == 999", name="外层规则")],
        n_jobs=2,
        parallel_backend=backend,
    )
    return outer


@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize("owner", ["ruleset", "classifier"])
def test_deep_parallel_descendants_are_detected_and_respect_active_budget(tmp_path, owner, backend):
    lock_path = tmp_path / f"{owner}-{backend}.lock"
    outer = _deep_parallel_ruleset(lock_path, backend)
    if owner == "ruleset":
        component = outer
        assert component._has_parallel_children() is True
    else:
        component = RulesClassifier(
            rules=[outer, Rule("score == -999", name="分类器规则")],
            n_jobs=2,
            parallel_backend=backend,
        )
        assert component._has_parallel_rule_children() is True
    token = _ACTIVE_BUDGET.set(ParallelBudget(2, 0))
    try:
        data = pd.DataFrame({"score": [0, 1]})
        if owner == "ruleset":
            component.evaluate(data)
        else:
            component.predict(data)
    finally:
        _ACTIVE_BUDGET.reset(token)

    assert not os.path.exists(f"{lock_path}.conflict")


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_root_explicit_positive_workers_are_not_budget_capped(monkeypatch, backend):
    captured = {}

    def capture_execute(function, tasks, **kwargs):
        captured.update(kwargs)
        return [function(task) for task in tasks]

    monkeypatch.setattr(rule_classifier_module, "parallel_execute", capture_execute)
    ruleset = RuleSet(
        name="根规则集",
        rules=[Rule("score > 0"), Rule("score < 10")],
        n_jobs=2,
        parallel_backend=backend,
    )
    ruleset.evaluate(pd.DataFrame({"score": [1, -1]}))

    assert captured["n_jobs"] == 2
    assert captured["parallel_backend"] == backend


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_multilabel_dpd_amount_report_matches_serial(rule_data, backend):
    kwargs = {
        "overdue": ["MOB1", "MOB2"],
        "dpds": [7, 3, 0],
        "amount": "放款金额",
        "margins": True,
    }
    serial = Rule("score < 600", n_jobs=1).report(rule_data, **kwargs)
    parallel = Rule("score < 600", n_jobs=2, parallel_backend=backend).report(rule_data, **kwargs)

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert isinstance(parallel.columns, pd.MultiIndex)
    assert parallel.columns.equals(serial.columns)
    assert parallel.index.equals(serial.index)
    assert parallel.dtypes.equals(serial.dtypes)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_rule_flow_group_reports_match_serial(rule_data, backend):
    def build(n_jobs, selected_backend):
        return RuleFlow(
            [Rule("score < 500", name="低分"), Rule("multi > 6", name="多头")],
            mode="parallel",
            n_jobs=n_jobs,
            parallel_backend=selected_backend,
        )

    kwargs = {"date_col": "放款时间", "freq": "M", "group_cols": "商品类别"}
    serial_flow = build(1, None)
    parallel_flow = build(2, backend)
    pd.testing.assert_frame_equal(
        serial_flow.report(rule_data, **kwargs),
        parallel_flow.report(rule_data, **kwargs),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        serial_flow.summary(rule_data, **kwargs),
        parallel_flow.summary(rule_data, **kwargs),
        check_exact=True,
    )
