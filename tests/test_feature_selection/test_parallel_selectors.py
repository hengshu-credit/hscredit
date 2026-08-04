"""特征筛选器统一并行 API 与确定性测试。"""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from hscredit.core import selectors
from hscredit.core.selectors import (
    BorutaSelector,
    CardinalitySelector,
    Chi2Selector,
    CompositeFeatureSelector,
    FTestSelector,
    FeatureImportanceSelector,
    IVSelector,
    LiftSelector,
    ModeSelector,
    MutualInfoSelector,
    NullSelector,
    NullImportanceSelector,
    PSISelector,
    RegexSelector,
    ScorecardFeatureSelection,
    SequentialFeatureSelector,
    StabilityAwareSelector,
    StepwiseSelector,
    TypeSelector,
    VarianceSelector,
    VIFSelector,
)
from hscredit.core.selectors.base import BaseFeatureSelector
from hscredit.exceptions import ValidationError
from hscredit.utils import parallel as parallel_utils
from hscredit.utils.parallel import _current_parallel_budget


class MeanDifferenceImportanceClassifier(BaseEstimator, ClassifierMixin):
    """以正负样本均值差提供确定性重要性的轻量分类器。"""

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y)
        self.classes_ = np.unique(target)
        positive = values[target == self.classes_[-1]]
        negative = values[target == self.classes_[0]]
        self.feature_importances_ = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class PartialFailureSelector(BaseFeatureSelector):
    """用于验证失败拟合不会泄漏部分状态。"""

    def __init__(self, fail=False, **kwargs):
        super().__init__(**kwargs)
        self.fail = fail

    def _fit_impl(self, X, y):
        self.selected_features_ = [X.columns[0]]
        self.scores_ = pd.Series([1.0], index=[X.columns[0]])
        if self.fail:
            raise RuntimeError("测试拟合失败")


class TransactionalBinner(BaseEstimator):
    """记录拟合和转换副作用，验证外部分箱器的事务边界。"""

    def __init__(self):
        self.fit_calls = 0
        self.metrics = []

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.fitted_ = True
        return self

    def transform(self, X, metric="indices"):
        self.metrics.append(metric)
        return X.copy()


class TransactionalFailureSelector(BaseFeatureSelector):
    """在原地修改嵌套学习状态后可控失败的真实筛选器。"""

    def __init__(self, fail=False, binner=None):
        super().__init__(binner=binner)
        self.fail = fail

    def _fit_impl(self, X, y):
        if not hasattr(self, "learned_state"):
            self.learned_state = {"history": []}
        self.learned_state["history"].append(list(X.columns))
        self.selected_features_ = list(X.columns)
        self.scores_ = pd.Series(1.0, index=X.columns)
        if self.fail:
            raise RuntimeError("事务测试失败")


class BudgetAwareClassifier(BaseEstimator, ClassifierMixin):
    """要求内部 n_jobs 与共享 ContextVar 子预算精确一致的分类器。"""

    def __init__(self, n_jobs=99, worker_budget=2, root_budget=4):
        self.n_jobs = n_jobs
        self.worker_budget = worker_budget
        self.root_budget = root_budget

    def fit(self, X, y):
        budget = _current_parallel_budget()
        expected = self.worker_budget if budget.depth > 0 else self.root_budget
        if budget.available != expected or self.n_jobs != expected:
            raise RuntimeError(
                f"内部预算错误: context={budget.available}, n_jobs={self.n_jobs}, expected={expected}"
            )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)

    def score(self, X, y):
        return 0.5


class NoIdentityParallelFeatureImportanceSelector(FeatureImportanceSelector):
    """禁止把联合模型的重要性标量映射伪装成并行计算。"""

    def _parallel_execute(self, *args, **kwargs):
        raise AssertionError("联合模型结果不得进入 identity 并行执行器")


class NoIdentityParallelBorutaSelector(BorutaSelector):
    """禁止把 Boruta 的廉价标量阈值比较伪装成并行计算。"""

    def _parallel_execute(self, *args, **kwargs):
        raise AssertionError("Boruta 标量比较不得进入 identity 并行执行器")


@pytest.fixture
def selector_xy():
    rng = np.random.RandomState(17)
    size = 120
    signal = rng.normal(size=size)
    X = pd.DataFrame(
        {
            "特征甲": signal,
            "特征乙": signal * 0.65 + rng.normal(scale=0.35, size=size),
            "特征丙": rng.normal(size=size),
        }
    )
    y = pd.Series((signal + rng.normal(scale=0.6, size=size) > 0).astype(int))
    return X, y


def test_all_exported_selectors_default_to_auto_parallel():
    """防止任一公开筛选器遗漏统一参数或退回串行默认值。"""
    missing = []
    for name in selectors.__all__:
        selector_class = getattr(selectors, name, None)
        if not inspect.isclass(selector_class) or not issubclass(selector_class, BaseFeatureSelector):
            continue
        parameters = inspect.signature(selector_class.__init__).parameters
        if parameters.get("n_jobs") is None or parameters["n_jobs"].default != -1:
            missing.append(name)
        if not {"parallel_backend", "parallel_config"} <= set(parameters):
            missing.append(name)
    assert missing == []


def test_all_exported_concrete_selectors_support_clone_and_pickle():
    """防止具体筛选器的构造链修改公共参数并破坏 sklearn/序列化协议。"""
    required = {
        "RegexSelector": {"pattern": "特征"},
        "FeatureImportanceSelector": {"estimator": MeanDifferenceImportanceClassifier()},
        "NullImportanceSelector": {"estimator": MeanDifferenceImportanceClassifier()},
        "RFESelector": {"estimator": LogisticRegression(max_iter=100)},
        "SequentialFeatureSelector": {"estimator": LogisticRegression(max_iter=100)},
        "CompositeFeatureSelector": {"selectors": [NullSelector()]},
    }
    failures = []
    config = {"batch_size": 1}
    for name in selectors.__all__:
        selector_class = getattr(selectors, name, None)
        if (
            not inspect.isclass(selector_class)
            or selector_class is BaseFeatureSelector
            or not issubclass(selector_class, BaseFeatureSelector)
        ):
            continue
        try:
            selector = selector_class(
                **required.get(name, {}),
                n_jobs=2,
                parallel_backend="threading",
                parallel_config=config,
            )
            assert selector.get_params(deep=False)["parallel_config"] is config
            cloned = clone(selector)
            assert cloned.n_jobs == 2
            assert cloned.parallel_backend == "threading"
            assert cloned.parallel_config == config
            restored = pickle.loads(pickle.dumps(selector))
            assert restored.n_jobs == 2
            assert restored.parallel_backend == "threading"
            assert restored.parallel_config == config
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    assert failures == []


def test_boruta_preserves_default_estimator_constructor_parameter():
    """默认估计器应在拟合时创建，构造参数必须原样保存以兼容 sklearn。"""
    selector = BorutaSelector()

    assert selector.estimator is None
    assert clone(selector).estimator is None


def test_joint_model_selectors_do_not_schedule_identity_parallel_tasks(selector_xy):
    """真正昂贵的联合拟合保持串行时，不应为廉价标量映射创建并行任务。"""
    X, y = selector_xy
    importance = NoIdentityParallelFeatureImportanceSelector(
        MeanDifferenceImportanceClassifier(), threshold=0.0, n_jobs=2
    ).fit(X, y)
    boruta = NoIdentityParallelBorutaSelector(
        MeanDifferenceImportanceClassifier(), max_iter=1, n_jobs=2
    ).fit(X, y)

    assert importance.scores_.index.tolist() == X.columns.tolist()
    assert boruta.scores_.index.tolist() == X.columns.tolist()


def test_selector_parallel_params_support_get_params_clone_and_pickle():
    """防止公共配置在 sklearn 参数协议或序列化中丢失。"""
    config = {"batch_size": 1, "pre_dispatch": "all"}
    selector = IVSelector(
        threshold=0.01,
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )

    assert selector.parallel_config is config
    assert selector.get_params(deep=False)["n_jobs"] == 0.5
    assert selector.get_params(deep=False)["parallel_backend"] == "threading"

    cloned = clone(selector)
    assert cloned.n_jobs == 0.5
    assert cloned.parallel_backend == "threading"
    assert cloned.parallel_config == config
    assert cloned.parallel_config is not config

    restored = pickle.loads(pickle.dumps(selector))
    assert restored.get_params(deep=False) == selector.get_params(deep=False)


@pytest.mark.parametrize(
    "selector_factory",
    [
        lambda n, backend: IVSelector(threshold=0.0, n_jobs=n, parallel_backend=backend),
        lambda n, backend: LiftSelector(threshold=0.0, n_jobs=n, parallel_backend=backend),
        lambda n, backend: VIFSelector(threshold=100.0, n_jobs=n, parallel_backend=backend),
    ],
)
def test_selector_scores_and_columns_match_serial(selector_factory, selector_xy):
    """防止代表性特征评分在并行后发生数值或列顺序漂移。"""
    X, y = selector_xy
    serial = selector_factory(1, None).fit(X, y)
    parallel = selector_factory(2, "threading").fit(X, y)

    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    assert serial.selected_features_ == parallel.selected_features_


@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize(
    "selector_factory",
    [
        pytest.param(lambda n, b: NullSelector(threshold=1.0, n_jobs=n, parallel_backend=b), id="null"),
        pytest.param(
            lambda n, b: CardinalitySelector(threshold=1000, n_jobs=n, parallel_backend=b),
            id="cardinality",
        ),
        pytest.param(
            lambda n, b: TypeSelector(dtype_include="number", n_jobs=n, parallel_backend=b), id="type"
        ),
        pytest.param(
            lambda n, b: RegexSelector(pattern="^特征", n_jobs=n, parallel_backend=b), id="regex"
        ),
        pytest.param(lambda n, b: VarianceSelector(threshold=0.0, n_jobs=n, parallel_backend=b), id="variance"),
        pytest.param(lambda n, b: ModeSelector(threshold=1.0, n_jobs=n, parallel_backend=b), id="mode"),
        pytest.param(
            lambda n, b: PSISelector(threshold=10.0, n_splits=3, n_jobs=n, parallel_backend=b), id="psi"
        ),
        pytest.param(
            lambda n, b: StabilityAwareSelector(
                iv_threshold=0.0,
                psi_threshold=10.0,
                score_threshold=-10.0,
                random_state=31,
                n_jobs=n,
                parallel_backend=b,
            ),
            id="stability",
        ),
        pytest.param(
            lambda n, b: MutualInfoSelector(
                threshold=0.0, random_state=31, n_jobs=n, parallel_backend=b
            ),
            id="mutual_info",
        ),
        pytest.param(lambda n, b: Chi2Selector(threshold=0.0, n_jobs=n, parallel_backend=b), id="chi2"),
        pytest.param(lambda n, b: FTestSelector(threshold=0.0, n_jobs=n, parallel_backend=b), id="f_test"),
    ],
)
def test_converted_worker_selectors_match_serial_exactly(selector_factory, selector_xy, backend):
    """所有已转换按列 worker 的分数、选择结果和列顺序须跨后端精确一致。"""
    X, y = selector_xy
    serial = selector_factory(1, None).fit(X, y)
    parallel = selector_factory(2, backend).fit(X, y)

    pd.testing.assert_series_equal(serial.scores_, parallel.scores_, check_exact=True)
    assert serial.selected_features_ == parallel.selected_features_
    assert parallel.scores_.index.tolist() == X.columns.tolist()


def test_selector_loky_matches_serial_and_remains_pickleable(selector_xy):
    """防止 loky worker 或已拟合产物无法跨进程序列化。"""
    X, y = selector_xy
    serial = IVSelector(threshold=0.0, n_jobs=1).fit(X, y)
    parallel = IVSelector(threshold=0.0, n_jobs=2, parallel_backend="loky").fit(X, y)

    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    assert pickle.loads(pickle.dumps(parallel)).selected_features_ == serial.selected_features_


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_null_importance_experiments_match_serial(selector_xy, backend):
    """防止随机实验因 worker 调度共享 RNG 或改变归并顺序。"""
    X, y = selector_xy
    factory = lambda n, b: NullImportanceSelector(
        MeanDifferenceImportanceClassifier(),
        threshold=-1.0,
        cv=3,
        n_runs=3,
        random_state=23,
        n_jobs=n,
        parallel_backend=b,
    )
    serial = factory(1, None).fit(X, y)
    parallel = factory(2, backend).fit(X, y)

    pd.testing.assert_frame_equal(serial.actual_importance_runs_, parallel.actual_importance_runs_)
    pd.testing.assert_frame_equal(serial.null_importance_runs_, parallel.null_importance_runs_)
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    assert serial.selected_features_ == parallel.selected_features_


def test_round_based_selector_matches_serial(selector_xy):
    """防止逐轮候选并行改变轮次、排名或稳定平局顺序。"""
    X, y = selector_xy
    factory = lambda n, backend: SequentialFeatureSelector(
        LogisticRegression(max_iter=300, random_state=7),
        n_features_to_select=2,
        direction="forward",
        cv=3,
        n_jobs=n,
        parallel_backend=backend,
    )
    serial = factory(1, None).fit(X, y)
    parallel = factory(2, "threading").fit(X, y)

    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    assert serial.selected_features_ == parallel.selected_features_


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_sequential_candidate_estimator_receives_child_budget_and_preserves_ties(
    selector_xy, monkeypatch, backend
):
    """轮内候选模型必须使用 child budget，且不得改变稳定首候选平局顺序。"""
    monkeypatch.setattr(parallel_utils, "get_physical_cpu_count", lambda: 5)
    X, y = selector_xy
    estimator = BudgetAwareClassifier(n_jobs=99, worker_budget=2, root_budget=4)

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=2,
        direction="forward",
        cv=2,
        n_jobs=-1,
        parallel_backend=backend,
    ).fit(X, y)

    assert selector.selected_features_ == ["特征甲", "特征乙"]
    assert [item["特征"] for item in selector.selection_history_] == ["特征甲", "特征乙"]
    assert estimator.n_jobs == 99


def test_sequential_single_candidate_receives_full_budget(monkeypatch):
    """外层因单候选而串行时，内部模型应获得未预留的完整预算。"""
    monkeypatch.setattr(parallel_utils, "get_physical_cpu_count", lambda: 5)
    X = pd.DataFrame({"唯一特征": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0, 1, 0, 1])

    selector = SequentialFeatureSelector(
        BudgetAwareClassifier(n_jobs=99, worker_budget=4, root_budget=4),
        n_features_to_select=1,
        cv=2,
        n_jobs=-1,
        parallel_backend="threading",
    ).fit(X, y)

    assert selector.selected_features_ == ["唯一特征"]


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_stepwise_candidate_estimator_receives_child_budget_and_preserves_tie_order(
    selector_xy, monkeypatch, backend
):
    """Stepwise 当前轮并行模型与最终串行模型须分别使用 child/full budget。"""
    monkeypatch.setattr(parallel_utils, "get_physical_cpu_count", lambda: 5)
    X, y = selector_xy
    estimator = BudgetAwareClassifier(n_jobs=99, worker_budget=2, root_budget=4)

    selector = StepwiseSelector(
        estimator=estimator,
        direction="forward",
        criterion="auc",
        max_features=1,
        p_enter=0.0,
        max_iter=2,
        n_jobs=-1,
        parallel_backend=backend,
    ).fit(X, y)

    assert selector.selected_features_ == ["特征甲"]
    assert selector.history_[0]["feature"] == "特征甲"
    assert estimator.n_jobs == 99


def test_successive_fit_replaces_old_dropped_state():
    """防止第二次成功拟合沿用第一次的剔除报告。"""
    selector = ModeSelector(threshold=0.75, n_jobs=1)
    selector.fit(pd.DataFrame({"甲": [1, 1, 1, 2], "乙": [0, 1, 2, 3]}))
    assert selector.dropped_["特征"].tolist() == ["甲"]

    selector.fit(pd.DataFrame({"甲": [0, 1, 2, 3], "乙": [3, 2, 1, 0]}))

    assert selector.selected_features_ == ["甲", "乙"]
    assert selector.dropped_.empty
    assert selector.removed_features_ == []


def test_failed_first_fit_does_not_leave_partial_state():
    """防止首次拟合异常后对象伪装成部分拟合状态。"""
    selector = PartialFailureSelector(fail=True)

    with pytest.raises(RuntimeError, match="测试拟合失败"):
        selector.fit(pd.DataFrame({"甲": [0, 1], "乙": [1, 0]}))

    assert not hasattr(selector, "selected_features_")
    assert not hasattr(selector, "scores_")
    assert not hasattr(selector, "_is_fitted")


def test_failed_refit_restores_previous_complete_state():
    """防止失败重拟合把旧结果和新 partial state 混合。"""
    selector = PartialFailureSelector(fail=False).fit(pd.DataFrame({"甲": [0, 1], "乙": [1, 0]}))
    selected_before = list(selector.selected_features_)
    scores_before = selector.scores_.copy()
    selector.fail = True

    with pytest.raises(RuntimeError, match="测试拟合失败"):
        selector.fit(pd.DataFrame({"丙": [0, 1], "丁": [1, 0]}))

    assert selector.selected_features_ == selected_before
    pd.testing.assert_series_equal(selector.scores_, scores_before)
    assert selector._is_fitted is True


def test_failed_refit_rolls_back_in_place_nested_learned_state():
    """失败重拟合不得污染已有模型中未以下划线结尾的嵌套学习状态。"""
    X = pd.DataFrame({"甲": [0, 1], "乙": [1, 0]})
    selector = TransactionalFailureSelector().fit(X)
    learned_state = selector.learned_state
    selector.fail = True
    before = pickle.dumps(selector)

    with pytest.raises(RuntimeError, match="事务测试失败"):
        selector.fit(pd.DataFrame({"丙": [0, 1], "丁": [1, 0]}))

    assert pickle.dumps(selector) == before
    assert selector.learned_state is learned_state


def test_failed_fit_does_not_mutate_external_unfitted_binner():
    """候选拟合失败时，外部未拟合分箱器及 selector 必须逐字节保持不变。"""
    X = pd.DataFrame({"甲": [0, 1], "乙": [1, 0]})
    binner = TransactionalBinner()
    selector = TransactionalFailureSelector(fail=True, binner=binner)
    binner_before = pickle.dumps(binner)
    selector_before = pickle.dumps(selector)

    with pytest.raises(RuntimeError, match="事务测试失败"):
        selector.fit(X)

    assert pickle.dumps(binner) == binner_before
    assert pickle.dumps(selector) == selector_before
    assert selector.binner is binner


def test_failed_composite_fit_restores_original_children():
    """后序阶段失败时，Composite 与所有外部子筛选器状态必须原子回滚。"""
    X = pd.DataFrame({"甲": [0, 1], "乙": [1, 0]})
    first = TransactionalFailureSelector()
    second = TransactionalFailureSelector(fail=True)
    children = [first, second]
    selector = CompositeFeatureSelector(children, n_jobs=2, parallel_backend="threading")
    selector_before = pickle.dumps(selector)
    children_before = [pickle.dumps(child) for child in children]

    with pytest.raises(RuntimeError, match="事务测试失败"):
        selector.fit(X)

    assert pickle.dumps(selector) == selector_before
    assert [pickle.dumps(child) for child in children] == children_before
    assert selector.selectors is children


def test_successful_fit_preserves_external_binner_and_composite_child_identity():
    """成功提交仍须保持公开外部对象身份，并向这些对象提交学习状态。"""
    X = pd.DataFrame({"甲": [0, 1], "乙": [1, 0]})
    binner = TransactionalBinner()
    child = TransactionalFailureSelector()
    children = [child]

    fitted = TransactionalFailureSelector(binner=binner).fit(X)
    composite = CompositeFeatureSelector(children).fit(X)

    assert fitted.binner is binner
    assert fitted._binner_instance is binner
    assert binner.fit_calls == 1
    assert composite.selectors is children
    assert child._is_fitted is True


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_stepwise_current_round_candidates_match_serial(selector_xy, backend):
    """防止逐步回归候选并行改变轮次决策、历史或最终得分。"""
    X, y = selector_xy
    factory = lambda n, b: StepwiseSelector(
        estimator=LogisticRegression(max_iter=300, random_state=7),
        direction="forward",
        criterion="auc",
        max_features=2,
        p_enter=0.0,
        max_iter=3,
        n_jobs=n,
        parallel_backend=b,
    )
    serial = factory(1, None).fit(X, y)
    parallel = factory(2, backend).fit(X, y)

    assert serial.selected_features_ == parallel.selected_features_
    assert serial.history_ == parallel.history_
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)


def test_composite_stages_receive_full_config_and_preserve_input_order():
    """防止组合阶段拆分预算、丢失配置或以 set 顺序输出交集。"""
    config = {"batch_size": 1}
    first = RegexSelector(pattern="^(特征甲|特征乙)$")
    second = NullSelector(threshold=1.0)
    selector = CompositeFeatureSelector(
        [first, second],
        strategy="intersection",
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=config,
    ).fit(pd.DataFrame({"特征甲": [0, 1], "特征乙": [1, 0], "特征丙": [1, 1]}))

    assert selector.selected_features_ == ["特征甲", "特征乙"]
    for child in (first, second):
        assert child.n_jobs == 2
        assert child.parallel_backend == "threading"
        assert child.parallel_config is config


def test_scorecard_stages_receive_full_parallel_config(selector_xy):
    """防止评分卡顺序阶段只收到 n_jobs 而丢失后端配置。"""
    X, y = selector_xy
    config = {"batch_size": 1}
    selector = ScorecardFeatureSelection(
        null_threshold=1.0,
        iv_threshold=None,
        corr_threshold=None,
        mode_threshold=None,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=config,
    ).fit(X, y)

    child = selector.stage_selectors_["empty"]
    assert child.n_jobs == 2
    assert child.parallel_backend == "threading"
    assert child.parallel_config is config


@pytest.mark.parametrize("selector", [ModeSelector(), TypeSelector()])
def test_selector_uses_shared_parallel_config_validation(selector):
    """防止筛选器绕过共享执行器而忽略非法并行配置。"""
    selector.set_params(n_jobs=2, parallel_config={"未知配置": 1})

    with pytest.raises(ValidationError, match="不支持的配置项"):
        selector.fit(pd.DataFrame({"特征甲": [0, 1], "特征乙": [1, 0]}))


def test_stability_random_split_matches_serial(selector_xy):
    """防止 PSI 随机拆分误把半量训练特征用于全量标签 IV。"""
    X, y = selector_xy
    factory = lambda n, backend: StabilityAwareSelector(
        iv_threshold=0.0,
        psi_threshold=10.0,
        score_threshold=-10.0,
        random_state=31,
        n_jobs=n,
        parallel_backend=backend,
    )
    serial = factory(1, None).fit(X, y)
    parallel = factory(2, "threading").fit(X, y)

    pd.testing.assert_series_equal(serial.iv_scores_, parallel.iv_scores_)
    pd.testing.assert_series_equal(serial.psi_scores_, parallel.psi_scores_)
    assert serial.selected_features_ == parallel.selected_features_


@pytest.mark.parametrize("random_state", [None, -7, 2**40 + 3])
def test_mutual_info_column_seeds_are_exact_across_backends(random_state):
    """按列种子须归一化；None 固定以 0 为基准，禁止依赖全局 RNG。"""
    rng = np.random.RandomState(31)
    base = rng.randint(0, 6, size=180).astype(float)
    X = pd.DataFrame(
        {
            "特征甲": base,
            "特征乙": np.roll(base, 3),
            "特征丙": rng.randint(0, 5, size=len(base)).astype(float),
        }
    )
    y = pd.Series((base >= 3).astype(int))

    def fit(n_jobs, backend):
        return MutualInfoSelector(
            threshold=0.0,
            n_neighbors=3,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=backend,
        ).fit(X, y)

    serial = fit(1, None)
    threading = fit(2, "threading")
    loky = fit(2, "loky")

    pd.testing.assert_series_equal(serial.scores_, threading.scores_, check_exact=True)
    pd.testing.assert_series_equal(serial.scores_, loky.scores_, check_exact=True)
    assert serial.selected_features_ == threading.selected_features_ == loky.selected_features_
    assert serial.scores_.index.tolist() == X.columns.tolist()
