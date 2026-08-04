# Unified Parallel Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every batch-capable binner, selector, encoder, rule component, rule miner, and report entry point a deterministic, resource-aware joblib execution path with `n_jobs=-1` enabled by default.

**Architecture:** A shared runtime in `hscredit/utils/parallel.py` validates public configuration, resolves conservative physical-CPU budgets, propagates real nested budgets through thread and process workers, and returns ordered results. Estimator base classes expose a small mixin while concrete algorithms submit pure feature/rule/label tasks and commit returned state on the main thread.

**Tech Stack:** Python 3.9+, joblib 1.0+, pandas, NumPy, scikit-learn estimator conventions, pytest, openpyxl.

## Global Constraints

- Public parallel parameters are `n_jobs=-1`, `parallel_backend=None`, and `parallel_config=None`.
- Automatic `-1` uses at most 80% of physical CPUs and always leaves one CPU unused when more than one exists.
- Positive integers and integer-valued floats are fixed worker counts; `0 < n_jobs < 1` is a physical-CPU ratio rounded upward; `1` and `1.0` both mean one worker.
- `None` remains accepted as a legacy serial value.
- Split nested budgets only when outer workers actually launch concurrent inner workers; sequential phases and rounds do not reserve child workers.
- Serial and parallel execution call the same worker implementation and preserve input order, fitted state, report layout, and numerical precision.
- Do not shard floating-point reductions by row, sample fewer records, reduce candidates, shorten iterations, or use approximate algorithms.
- All public errors, DataFrame columns, reports, and new docstrings are Chinese.
- Preserve Python 3.9, joblib 1.0, sklearn clone/Pipeline, and artifact serialization compatibility.
- Merge the current selector binning lifecycle implementation; do not revert or duplicate it.
- Use `examples/hscredit_yyp.xlsx` for final single-feature, multi-feature, multi-label, amount, date, and category verification.

---

### Task 1: Conservative CPU and configuration resolver

**Files:**
- Modify: `hscredit/utils/parallel.py`
- Modify: `hscredit/utils/__init__.py`
- Modify: `hscredit/exceptions.py`
- Modify: `tests/test_utils/test_parallel.py`
- Create: `tests/test_utils/test_parallel_runtime.py`

**Interfaces:**
- Produces: `get_physical_cpu_count() -> int`
- Produces: `resolve_n_jobs(n_jobs, task_count=None, *, cpu_count=None, available_budget=None) -> Optional[int]`
- Produces: `validate_parallel_config(parallel_backend, parallel_config) -> Dict[str, Any]`
- Produces: `ParallelExecutionError(StateError)`

- [ ] **Step 1: Write resolver tests**

```python
@pytest.mark.parametrize(
    ("cpus", "expected"),
    [(1, 1), (2, 1), (8, 7), (16, 13)],
)
def test_auto_workers_use_eighty_percent_and_leave_one_cpu(cpus, expected):
    assert resolve_n_jobs(-1, cpu_count=cpus) == expected


@pytest.mark.parametrize(
    ("value", "cpus", "expected"),
    [(1, 8, 1), (1.0, 8, 1), (2.0, 8, 2), (0.25, 8, 2), (0.26, 8, 3)],
)
def test_explicit_worker_forms(value, cpus, expected):
    assert resolve_n_jobs(value, cpu_count=cpus) == expected


def test_task_count_caps_workers():
    assert resolve_n_jobs(-1, task_count=2, cpu_count=16) == 2


@pytest.mark.parametrize("value", [True, 0, -2, 1.5, "2", object()])
def test_invalid_n_jobs_raises_chinese_validation_error(value):
    with pytest.raises(ValidationError, match="n_jobs"):
        resolve_n_jobs(value, cpu_count=8)
```

- [ ] **Step 2: Run resolver tests and verify RED**

Run: `pytest tests/test_utils/test_parallel.py tests/test_utils/test_parallel_runtime.py -k "workers or n_jobs" -v`

Expected: failures show that floats, task caps, physical CPU ratios, and validation are not implemented.

- [ ] **Step 3: Implement the resolver**

```python
NJobs = Optional[Union[int, float]]


def resolve_n_jobs(
    n_jobs: NJobs,
    task_count: Optional[int] = None,
    *,
    cpu_count: Optional[int] = None,
    available_budget: Optional[int] = None,
) -> Optional[int]:
    if n_jobs is None:
        return None
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, numbers.Real):
        raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")
    cpus = max(1, int(cpu_count or get_physical_cpu_count()))
    value = float(n_jobs)
    if value == -1:
        workers = available_budget or (1 if cpus == 1 else min(cpus - 1, math.ceil(cpus * 0.8)))
    elif value.is_integer() and value >= 1:
        workers = int(value)
    elif 0 < value < 1:
        workers = math.ceil(cpus * value)
    else:
        raise ValidationError("n_jobs 必须为 -1、正整数或 0 到 1 之间的小数")
    if task_count is not None:
        workers = min(workers, max(1, int(task_count)))
    return max(1, workers)
```

Use `joblib.cpu_count(only_physical_cores=True)` when supported, then `joblib.cpu_count()`, `os.cpu_count()`, and finally 1.

- [ ] **Step 4: Write and implement configuration validation tests**

```python
def test_parallel_config_rejects_duplicate_worker_and_backend_sources():
    with pytest.raises(ValidationError, match="n_jobs"):
        validate_parallel_config(None, {"n_jobs": 2})
    with pytest.raises(ValidationError, match="backend"):
        validate_parallel_config("loky", {"backend": "threading"})


def test_parallel_config_preserves_supported_joblib_values():
    source = {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}
    assert validate_parallel_config("loky", source) == source
    assert source == {"batch_size": 4, "pre_dispatch": "2*n_jobs", "mmap_mode": "r"}
```

Validate the exact approved key set and nested `backend_kwargs`; return a new dictionary without mutating caller input. Add and export `ParallelExecutionError` for worker failures.

- [ ] **Step 5: Run Task 1 tests**

Run: `pytest tests/test_utils/test_parallel.py tests/test_utils/test_parallel_runtime.py -v --tb=short`

Expected: all Task 1 tests pass, including updated model expectation for the 80% rule.

- [ ] **Step 6: Commit Task 1**

```bash
git add hscredit/utils/parallel.py hscredit/utils/__init__.py hscredit/exceptions.py tests/test_utils/test_parallel.py tests/test_utils/test_parallel_runtime.py
git commit -m "feat: add resource-aware parallel configuration"
```

### Task 2: Ordered executor and real nested budgets

**Files:**
- Modify: `hscredit/utils/parallel.py`
- Modify: `hscredit/utils/__init__.py`
- Modify: `tests/test_utils/test_parallel_runtime.py`

**Interfaces:**
- Produces: `ParallelBudget(available: int, depth: int)`
- Produces: `split_parallel_budget(available, task_count, has_parallel_children) -> Tuple[int, int]`
- Produces: `parallel_execute(function, tasks, *, n_jobs=-1, parallel_backend=None, parallel_config=None, task_labels=None, default_backend=None, has_parallel_children=False) -> List[Any]`
- Produces: `ParallelizableMixin._parallel_execute(...) -> List[Any]`

- [ ] **Step 1: Write ordered, serial-parity, and failure tests**

```python
def _square(value):
    return value * value


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_parallel_execute_preserves_submission_order(backend):
    assert parallel_execute(
        _square, [3, 1, 2], n_jobs=2, parallel_backend=backend
    ) == [9, 1, 4]


def test_serial_and_parallel_call_the_same_worker():
    serial = parallel_execute(_square, range(8), n_jobs=1)
    parallel = parallel_execute(_square, range(8), n_jobs=2, parallel_backend="threading")
    assert parallel == serial


def test_worker_failure_has_chinese_context_and_original_cause():
    def fail(_):
        raise KeyError("boom")
    with pytest.raises(ParallelExecutionError, match="特征A") as error:
        parallel_execute(fail, [1], task_labels=["特征A"])
    assert isinstance(error.value.__cause__, KeyError)
```

- [ ] **Step 2: Run the ordered executor tests and verify RED**

Run: `pytest tests/test_utils/test_parallel_runtime.py -k "submission_order or same_worker or failure" -v`

Expected: import failures because the executor and error type do not exist.

- [ ] **Step 3: Implement nested-budget math**

```python
def split_parallel_budget(available, task_count, has_parallel_children):
    available = max(1, int(available))
    if not has_parallel_children:
        return min(available, max(1, task_count)), 1
    outer = min(max(1, task_count), math.ceil(math.sqrt(available)))
    return outer, max(1, available // outer)
```

Add tests proving `(13, 100, True) == (4, 3)`, `(13, 100, False) == (13, 1)`, one outer task with children receives child budget 13, and sequential Composite/Stepwise labels do not trigger a split.

- [ ] **Step 4: Implement context propagation and execution**

Use a module-level `ContextVar[Optional[ParallelBudget]]`. Wrap every submitted task in a module-level picklable function that sets the child budget before calling the worker and resets it afterward. Materialize tasks and labels once, use direct iteration for one worker, and use joblib's ordered list return for multiple workers.

```python
def _run_with_budget(function, task, label, child_budget, depth):
    token = _ACTIVE_BUDGET.set(ParallelBudget(child_budget, depth))
    try:
        return function(task)
    except Exception as exc:
        raise ParallelExecutionError(f"并行任务 '{label}' 执行失败: {exc}") from exc
    finally:
        _ACTIVE_BUDGET.reset(token)
```

Use `joblib.parallel_backend` for `inner_max_num_threads` and backend-specific options so the implementation remains compatible with joblib 1.0.

- [ ] **Step 5: Test actual thread and process propagation**

Define a module-level test worker returning the active budget. Assert both `threading` and `loky` workers see the expected child budget, and a sequential parent followed by an inner parallel call receives the full current budget.

Run: `pytest tests/test_utils/test_parallel_runtime.py -v --tb=short`

Expected: all runtime tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add hscredit/utils/parallel.py hscredit/utils/__init__.py tests/test_utils/test_parallel_runtime.py
git commit -m "feat: coordinate ordered nested parallel tasks"
```

### Task 3: Shared estimator contract and existing EDA migration

**Files:**
- Modify: `hscredit/core/binning/base.py`
- Modify: `hscredit/core/selectors/base.py`
- Modify: `hscredit/core/encoders/base.py`
- Modify: `hscredit/report/mining/base.py`
- Modify: `hscredit/core/eda/_feature_summary.py`
- Modify: `tests/test_eda/test_feature_summary_parallel.py`
- Create: `tests/test_parallel_api_contract.py`

**Interfaces:**
- Consumes: `ParallelizableMixin`
- Produces: base constructor attributes `n_jobs`, `parallel_backend`, `parallel_config`
- Produces: one shared resolver for EDA and target modules

- [ ] **Step 1: Write base API and clone tests**

```python
@pytest.mark.parametrize("cls", [BaseBinning, BaseEncoder, BaseFeatureSelector, BaseRuleMiner])
def test_parallel_base_signatures_expose_common_parameters(cls):
    params = inspect.signature(cls.__init__).parameters
    assert params["n_jobs"].default == -1
    assert params["parallel_backend"].default is None
    assert params["parallel_config"].default is None


def test_encoder_clone_preserves_parallel_config_identity_contract():
    config = {"batch_size": 8}
    encoder = CountEncoder(n_jobs=0.5, parallel_backend="threading", parallel_config=config)
    cloned = clone(encoder)
    assert cloned.get_params()["parallel_config"] == config
    assert cloned.parallel_config is not config
```

- [ ] **Step 2: Run API tests and verify RED**

Run: `pytest tests/test_parallel_api_contract.py -v`

Expected: constructors lack the unified parameters and BaseRuleMiner is not parallelizable.

- [ ] **Step 3: Add the mixin contract to base classes**

Each base stores parameters unchanged and delegates execution through:

```python
def _parallel_execute(self, function, tasks, **kwargs):
    return parallel_execute(
        function,
        tasks,
        n_jobs=self.n_jobs,
        parallel_backend=self.parallel_backend,
        parallel_config=self.parallel_config,
        **kwargs,
    )
```

Do not resolve `n_jobs` in constructors.

- [ ] **Step 4: Replace EDA's duplicate CPU resolver**

Make `_feature_summary.py` call the shared resolver and executor while retaining its batching and progress behavior. Update tests so 16 physical CPUs resolve to 13 workers, two tasks cap at two, and invalid inputs raise the shared Chinese error.

Run: `pytest tests/test_eda/test_feature_summary_parallel.py tests/test_parallel_api_contract.py -v --tb=short`

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add hscredit/core/binning/base.py hscredit/core/selectors/base.py hscredit/core/encoders/base.py hscredit/report/mining/base.py hscredit/core/eda/_feature_summary.py tests/test_eda/test_feature_summary_parallel.py tests/test_parallel_api_contract.py
git commit -m "refactor: share parallel estimator contract"
```

### Task 4: Binning public API and NumPy categorical prerequisite

**Files:**
- Modify: `hscredit/core/metrics/_binning.py`
- Modify: `hscredit/core/binning/base.py`
- Modify: `hscredit/core/binning/best_iv_binning.py`
- Modify: `hscredit/core/binning/best_ks_binning.py`
- Modify: `hscredit/core/binning/best_lift_binning.py`
- Modify: `hscredit/core/binning/cart_binning.py`
- Modify: `hscredit/core/binning/chi_merge_binning.py`
- Modify: `hscredit/core/binning/cp_sat_binning.py`
- Modify: `hscredit/core/binning/genetic_binning.py`
- Modify: `hscredit/core/binning/kernel_density_binning.py`
- Modify: `hscredit/core/binning/kmeans_binning.py`
- Modify: `hscredit/core/binning/mdlp_binning.py`
- Modify: `hscredit/core/binning/monotonic_binning.py`
- Modify: `hscredit/core/binning/optimal_binning.py`
- Modify: `hscredit/core/binning/optimal_binning_2d.py`
- Modify: `hscredit/core/binning/or_binning.py`
- Modify: `hscredit/core/binning/quantile_binning.py`
- Modify: `hscredit/core/binning/smooth_binning.py`
- Modify: `hscredit/core/binning/target_bad_rate_binning.py`
- Modify: `hscredit/core/binning/tree_binning.py`
- Modify: `hscredit/core/binning/uniform_binning.py`
- Create: `tests/test_binning/test_parallel_binning.py`

**Interfaces:**
- Produces: every exported binner constructor accepts the common parameters
- Produces: NumPy-safe categorical sorting keys using Python `int`

- [ ] **Step 1: Write exported-binner signature and overflow regression tests**

```python
def test_all_exported_binners_expose_parallel_parameters():
    missing = []
    for name in binning.__all__:
        cls = getattr(binning, name)
        if inspect.isclass(cls) and issubclass(cls, BaseBinning):
            params = inspect.signature(cls.__init__).parameters
            if not {"n_jobs", "parallel_backend", "parallel_config"} <= set(params):
                missing.append(name)
    assert missing == []


def test_numpy_int8_category_sort_key_does_not_overflow():
    x = pd.Series(["a", "b", "c", "a", "b", "c"])
    y = pd.Series([0, 1, 0, 1, 0, 1])
    result = compute_bin_stats(x, y, bins=np.array([0, 1, 2, 0, 1, 2], dtype=np.int8))
    assert not result.empty
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_binning/test_parallel_binning.py -k "exported or int8" -v`

Expected: signatures are missing parameters and NumPy 2.x raises the documented overflow.

- [ ] **Step 3: Apply explicit constructor forwarding**

Every concrete constructor appends and forwards this exact block without moving existing positional parameters:

```python
n_jobs: Union[int, float] = -1,
parallel_backend: Optional[str] = None,
parallel_config: Optional[Dict[str, Any]] = None,
```

Constructors already using `**kwargs` must still declare these names explicitly for inspectability and sklearn parameter discovery.

- [ ] **Step 4: Fix the categorical sort key minimally**

At the documented `_binning.py` sorting expression, convert each NumPy bin scalar to Python `int` before multiplying by 10000. Add no unrelated formatting changes.

- [ ] **Step 5: Run API and categorical suites**

Run: `pytest tests/test_binning/test_parallel_binning.py tests/test_binning/test_categorical_methods.py tests/test_binning/test_categorical_binning_complete.py -v --tb=short`

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add hscredit/core/metrics/_binning.py hscredit/core/binning tests/test_binning/test_parallel_binning.py
git commit -m "feat: expose parallel configuration on binners"
```

### Task 5: Transactional binning fit and ordered transform

**Files:**
- Modify: `hscredit/core/binning/base.py`
- Modify: `hscredit/core/binning/best_iv_binning.py`
- Modify: `hscredit/core/binning/best_ks_binning.py`
- Modify: `hscredit/core/binning/best_lift_binning.py`
- Modify: `hscredit/core/binning/cart_binning.py`
- Modify: `hscredit/core/binning/chi_merge_binning.py`
- Modify: `hscredit/core/binning/cp_sat_binning.py`
- Modify: `hscredit/core/binning/genetic_binning.py`
- Modify: `hscredit/core/binning/kernel_density_binning.py`
- Modify: `hscredit/core/binning/kmeans_binning.py`
- Modify: `hscredit/core/binning/mdlp_binning.py`
- Modify: `hscredit/core/binning/monotonic_binning.py`
- Modify: `hscredit/core/binning/optimal_binning.py`
- Modify: `hscredit/core/binning/optimal_binning_2d.py`
- Modify: `hscredit/core/binning/or_binning.py`
- Modify: `hscredit/core/binning/quantile_binning.py`
- Modify: `hscredit/core/binning/smooth_binning.py`
- Modify: `hscredit/core/binning/target_bad_rate_binning.py`
- Modify: `hscredit/core/binning/tree_binning.py`
- Modify: `hscredit/core/binning/uniform_binning.py`
- Modify: `tests/test_binning/test_parallel_binning.py`

**Interfaces:**
- Produces: `_fit_features(X, y, method_name) -> None` using isolated per-feature state
- Produces: `_transform_features(X, transform_one) -> pd.DataFrame`
- Produces: ordered transaction merge for all feature-scoped state

- [ ] **Step 1: Write serial/parallel state parity tests**

```python
@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_uniform_fit_and_transform_match_serial(backend, mixed_xy):
    X, y = mixed_xy
    serial = UniformBinning(max_n_bins=4, n_jobs=1).fit(X, y)
    parallel = UniformBinning(
        max_n_bins=4, n_jobs=2, parallel_backend=backend
    ).fit(X, y)
    assert serial.splits_.keys() == parallel.splits_.keys()
    for feature in X.columns:
        np.testing.assert_array_equal(serial.splits_[feature], parallel.splits_[feature])
        pd.testing.assert_frame_equal(serial.bin_tables_[feature], parallel.bin_tables_[feature])
    pd.testing.assert_frame_equal(serial.transform(X), parallel.transform(X))


def test_failed_feature_fit_does_not_commit_partial_state(mixed_xy):
    binner = FailingFeatureBinner(n_jobs=2, parallel_backend="threading")
    with pytest.raises(ParallelExecutionError, match="坏特征"):
        binner.fit(*mixed_xy)
    assert binner.splits_ == {}
    assert binner.bin_tables_ == {}
```

Add a parametrized smoke matrix covering every exported concrete binner with a small compatible numeric/categorical fixture and fixed `random_state`.

- [ ] **Step 2: Run representative tests and verify RED**

Run: `pytest tests/test_binning/test_parallel_binning.py -k "match_serial or partial_state" -v`

Expected: loky loses worker mutations and failed threaded fit leaves partial state.

- [ ] **Step 3: Implement isolated feature state**

Define the exact feature-scoped registry in `BaseBinning`:

```python
_FEATURE_DICT_STATE = (
    "splits_", "n_bins_", "bin_tables_", "feature_types_", "_cat_bins_",
    "_category_orders_", "_category_code_maps_", "_categorical_numeric_splits_",
    "_categorical_fit_context_",
)
_FEATURE_SET_STATE = ("_categorical_encoded_features_",)
```

Each task creates a shallow estimator copy, replaces all registered mutable state with empty containers, invokes the named single-feature method, and returns only that feature's state. The main estimator validates every result and merges in original feature order only after every task succeeds.

Move local `_fit_one` closures into named `_fit_feature` methods where necessary so both threading and process backends can invoke the same isolated method. Keep algorithm bodies unchanged.

- [ ] **Step 4: Centralize ordered transforms**

Make transform workers read-only and return `(feature, Series/DataFrame)`. Concatenate in `X.columns` order and restore original index and documented dtypes. Update each override to call the base helper instead of maintaining a private loop.

- [ ] **Step 5: Test all binners and both backends**

Run: `pytest tests/test_binning/test_parallel_binning.py -v --tb=short`

Run: `pytest tests/test_binning/ -m "not slow and not integration" -v --tb=short`

Expected: all binning tests pass for both serial and parallel paths.

- [ ] **Step 6: Commit Task 5**

```bash
git add hscredit/core/binning tests/test_binning/test_parallel_binning.py
git commit -m "feat: parallelize binning transactions"
```

### Task 6: Selector API and per-feature/candidate parallelism

**Files:**
- Modify: `hscredit/core/selectors/base.py`
- Modify: `hscredit/core/selectors/boruta_selector.py`
- Modify: `hscredit/core/selectors/cardinality_selector.py`
- Modify: `hscredit/core/selectors/chi2_selector.py`
- Modify: `hscredit/core/selectors/corr_selector.py`
- Modify: `hscredit/core/selectors/f_test_selector.py`
- Modify: `hscredit/core/selectors/importance_selector.py`
- Modify: `hscredit/core/selectors/iv_selector.py`
- Modify: `hscredit/core/selectors/lift_selector.py`
- Modify: `hscredit/core/selectors/mode_selector.py`
- Modify: `hscredit/core/selectors/mutual_info_selector.py`
- Modify: `hscredit/core/selectors/null_importance_selector.py`
- Modify: `hscredit/core/selectors/null_selector.py`
- Modify: `hscredit/core/selectors/psi_selector.py`
- Modify: `hscredit/core/selectors/regex_selector.py`
- Modify: `hscredit/core/selectors/rfe_selector.py`
- Modify: `hscredit/core/selectors/scorecard_feature_selection.py`
- Modify: `hscredit/core/selectors/sequential_selector.py`
- Modify: `hscredit/core/selectors/stability_selector.py`
- Modify: `hscredit/core/selectors/stepwise_selector.py`
- Modify: `hscredit/core/selectors/type_selector.py`
- Modify: `hscredit/core/selectors/variance_selector.py`
- Modify: `hscredit/core/selectors/vif_selector.py`
- Modify: `tests/test_feature_selection/test_selector_binning.py`
- Create: `tests/test_feature_selection/test_parallel_selectors.py`

**Interfaces:**
- Produces: common parallel parameters on all exported selectors
- Produces: ordered parallel feature metrics, random experiments, and per-round candidates

- [ ] **Step 1: Write public API and representative parity tests**

```python
def test_all_exported_selectors_default_to_auto_parallel():
    missing = []
    for name in selectors.__all__:
        cls = getattr(selectors, name, None)
        if inspect.isclass(cls) and issubclass(cls, BaseFeatureSelector):
            params = inspect.signature(cls.__init__).parameters
            if params.get("n_jobs") is None or params["n_jobs"].default != -1:
                missing.append(name)
            if not {"parallel_backend", "parallel_config"} <= set(params):
                missing.append(name)
    assert missing == []


@pytest.mark.parametrize("selector_factory", [
    lambda n, b: IVSelector(threshold=0.0, n_jobs=n, parallel_backend=b),
    lambda n, b: LiftSelector(threshold=0.0, n_jobs=n, parallel_backend=b),
    lambda n, b: VIFSelector(threshold=100.0, n_jobs=n, parallel_backend=b),
])
def test_selector_scores_and_columns_match_serial(selector_factory, selector_xy):
    X, y = selector_xy
    serial = selector_factory(1, None).fit(X, y)
    parallel = selector_factory(2, "threading").fit(X, y)
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_)
    assert serial.selected_features_ == parallel.selected_features_
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_selection/test_parallel_selectors.py -v`

Expected: default/signature tests fail and existing selector-local Parallel calls bypass shared configuration.

- [ ] **Step 3: Forward the common constructor parameters**

Explicitly add the approved block to every exported selector and forward it to `BaseFeatureSelector`. Preserve existing `binner` and `binning_params` precedence and sklearn clone behavior.

- [ ] **Step 4: Replace independent feature loops**

Replace direct `Parallel` calls in IV, Lift, Mode, PSI, Stability, VIF, and Boruta with `_parallel_execute`; convert remaining Null, Cardinality, Type, Regex, Variance, MutualInfo, Chi2, FTest, and feature-importance column loops where each score is independent. Return `(feature, score)` and construct Series in input order.

- [ ] **Step 5: Parallelize only current-round candidates and independent experiments**

RFE, Sequential, and Stepwise rounds remain sequential. Within a round, clone the estimator per candidate, evaluate candidates through `_parallel_execute`, and choose the winner using the existing stable tie-break order. Null Importance uses a task seed derived from `random_state` and experiment ordinal. Composite stages remain sequential and each receives the full current budget.

- [ ] **Step 6: Verify selector parity and lifecycle integration**

Run: `pytest tests/test_feature_selection/test_selector_binning.py tests/test_feature_selection/test_parallel_selectors.py -v --tb=short`

Run: `pytest tests/test_feature_selection/ -v --tb=short`

Expected: all selector tests pass and the existing outer-binner reuse tests remain unchanged.

- [ ] **Step 7: Commit Task 6**

```bash
git add hscredit/core/selectors tests/test_feature_selection
git commit -m "feat: parallelize feature selectors"
```

### Task 7: Encoder mapping and transform parallelism

**Files:**
- Modify: `hscredit/core/encoders/base.py`
- Modify: `hscredit/core/encoders/cardinality_encoder.py`
- Modify: `hscredit/core/encoders/catboost_encoder.py`
- Modify: `hscredit/core/encoders/count_encoder.py`
- Modify: `hscredit/core/encoders/gbm_encoder.py`
- Modify: `hscredit/core/encoders/one_hot_encoder.py`
- Modify: `hscredit/core/encoders/ordinal_encoder.py`
- Modify: `hscredit/core/encoders/quantile_encoder.py`
- Modify: `hscredit/core/encoders/target_encoder.py`
- Modify: `hscredit/core/encoders/woe_encoder.py`
- Create: `tests/test_encoding/test_parallel_encoders.py`

**Interfaces:**
- Produces: common parallel parameters on all 9 encoders
- Produces: `_fit_columns` and `_transform_columns` ordered helpers in `BaseEncoder`

- [ ] **Step 1: Write API, mapping, transform, and clone tests**

```python
@pytest.mark.parametrize("encoder_factory", [
    lambda n, b: CountEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
    lambda n, b: WOEEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
    lambda n, b: TargetEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
    lambda n, b: OneHotEncoder(cols=["a", "b"], n_jobs=n, parallel_backend=b),
])
def test_encoder_parallel_fit_transform_matches_serial(encoder_factory, encoder_xy):
    X, y = encoder_xy
    serial = encoder_factory(1, None).fit(X, y)
    parallel = encoder_factory(2, "threading").fit(X, y)
    assert serial.export_mapping() == parallel.export_mapping()
    pd.testing.assert_frame_equal(serial.transform(X), parallel.transform(X))
```

Add signature coverage for all names in `encoders.__all__`, categorical missing/unknown values, output dtypes, OneHot column order, loky execution, and sklearn clone.

- [ ] **Step 2: Run encoder tests and verify RED**

Run: `pytest tests/test_encoding/test_parallel_encoders.py -v`

Expected: constructors reject the new parameters.

- [ ] **Step 3: Add explicit API and ordered base helpers**

Each encoder declares and forwards the common block. `BaseEncoder` helpers submit `(ordinal, column)` tasks, return mapping/state or transformed columns, and commit only after all tasks succeed.

- [ ] **Step 4: Convert all nine encoders**

Keep each encoder's single-column math unchanged. For CatBoost ordered encoding, parallelize columns but retain row order within a column. For GBMEncoder, parallelize independent column models and declare real parallel children when the underlying GBM model concurrently uses workers. For OneHot, concatenate returned blocks using learned feature/category order.

- [ ] **Step 5: Run encoding suites**

Run: `pytest tests/test_encoding/test_parallel_encoders.py tests/test_encoding/ -v --tb=short`

Expected: all encoder tests pass for serial, threading, and loky representative cases.

- [ ] **Step 6: Commit Task 7**

```bash
git add hscredit/core/encoders tests/test_encoding/test_parallel_encoders.py
git commit -m "feat: parallelize feature encoders"
```

### Task 8: Rule execution and classifier parallelism

**Files:**
- Modify: `hscredit/core/rules/rule.py`
- Modify: `hscredit/core/rules/rule_flow.py`
- Modify: `hscredit/core/models/rules/rule_classifier.py`
- Modify: `tests/test_rules/test_rule_report.py`
- Modify: `tests/test_rules/test_rule_flow.py`
- Modify: `tests/test_models/test_rule_classifier.py`
- Create: `tests/test_rules/test_parallel_rules.py`

**Interfaces:**
- Produces: common parallel parameters on `Rule`, `RuleFlow`, `RuleSet`, and `RulesClassifier`
- Produces: ordered independent-rule prediction and report calculations

- [ ] **Step 1: Write rule parity tests**

```python
def test_rule_flow_parallel_mode_matches_serial(rule_data):
    rules = [Rule("score < 600", name="低分"), Rule("多头 > 3", name="多头")]
    serial = RuleFlow(rules, mode="parallel", n_jobs=1).predict(rule_data)
    parallel = RuleFlow(
        rules, mode="parallel", n_jobs=2, parallel_backend="threading"
    ).predict(rule_data)
    pd.testing.assert_frame_equal(serial, parallel)


def test_rules_classifier_parallel_prediction_matches_serial(rule_data, target):
    ruleset = create_or_ruleset(["score < 600", "多头 > 3"])
    serial = RulesClassifier(ruleset, n_jobs=1).fit(rule_data, target)
    parallel = RulesClassifier(ruleset, n_jobs=2, parallel_backend="loky").fit(rule_data, target)
    np.testing.assert_array_equal(serial.predict(rule_data), parallel.predict(rule_data))
```

Also compare Rule multi-label/DPD reports and ensure serial RuleFlow mode remains sequential even with `n_jobs>1`.

- [ ] **Step 2: Run rule tests and verify RED**

Run: `pytest tests/test_rules/test_parallel_rules.py -v`

Expected: constructors reject unified configuration or ignore it.

- [ ] **Step 3: Implement independent-rule execution**

Store common parameters unchanged. Use the shared executor for RuleFlow parallel mode and RuleSet/RulesClassifier rule masks. Aggregate masks in declared rule order with the existing `LogicOperator`; do not parallelize serial RuleFlow filtering.

- [ ] **Step 4: Implement report label/group parallelism**

For Rule reports, create independent tasks for target/DPD/group slices only where they are simultaneously computable. Build totals and MultiIndex columns on the main thread in current order.

- [ ] **Step 5: Run rule and classifier suites**

Run: `pytest tests/test_rules/ tests/test_models/test_rule_classifier.py -v --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 8**

```bash
git add hscredit/core/rules hscredit/core/models/rules tests/test_rules tests/test_models/test_rule_classifier.py
git commit -m "feat: parallelize rule execution"
```

### Task 9: Rule-mining parallelism

**Files:**
- Modify: `hscredit/report/mining/base.py`
- Modify: `hscredit/report/mining/manual_tree_extractor.py`
- Modify: `hscredit/report/mining/metrics.py`
- Modify: `hscredit/report/mining/multi_feature.py`
- Modify: `hscredit/report/mining/multi_label.py`
- Modify: `hscredit/report/mining/single_feature.py`
- Modify: `hscredit/report/mining/tree_extractor.py`
- Modify: `tests/test_rules/test_mining.py`
- Create: `tests/test_rules/test_parallel_mining.py`

**Interfaces:**
- Produces: common configuration on all public miner/analyzer classes
- Produces: feature, combination, label, tree, rule, and dataset evaluation tasks

- [ ] **Step 1: Write miner API and parity tests**

Instantiate `SingleFeatureRuleMiner`, `MultiFeatureRuleMiner`, `MultiLabelRuleMiner`, `TreeRuleExtractor`, `DecisionTreeAnalyzer`, and `ManualTreeExtractor` with serial and parallel configurations. With fixed seeds, compare normalized rule expressions, rule order, metric scores, tree rules, and evaluation tables.

```python
def _normalized_rules(miner):
    return [(rule.to_expression(), rule.metric_score) for rule in miner.rules_]


def test_single_feature_miner_matches_serial(mining_xy):
    X, y = mining_xy
    serial = SingleFeatureRuleMiner(random_state=7, n_jobs=1).fit(X, y)
    parallel = SingleFeatureRuleMiner(
        random_state=7, n_jobs=2, parallel_backend="threading"
    ).fit(X, y)
    assert _normalized_rules(parallel) == _normalized_rules(serial)
```

- [ ] **Step 2: Run miner tests and verify RED**

Run: `pytest tests/test_rules/test_parallel_mining.py -v`

Expected: public classes lack configuration and mining loops remain serial.

- [ ] **Step 3: Implement ordered mining tasks**

SingleFeature submits features; MultiFeature submits combinations in generated order; MultiLabel submits labels while each label miner receives a real child budget only when labels and child feature tasks overlap; TreeRuleExtractor submits independent trees/rule evaluations. Manual tree construction remains sequential while independent dataset and node report calculations use the executor.

- [ ] **Step 4: Verify rule order and nested budgets**

Monkeypatch the shared executor in MultiLabelRuleMiner to capture `has_parallel_children=True`; assert SingleFeature direct calls use the full root budget and sequential manual-tree levels do not split it.

Run: `pytest tests/test_rules/test_mining.py tests/test_rules/test_parallel_mining.py -v --tb=short`

Expected: all mining tests pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add hscredit/report/mining tests/test_rules/test_mining.py tests/test_rules/test_parallel_mining.py
git commit -m "feat: parallelize rule mining"
```

### Task 10: Feature, rule, swap, overdue, and drift reports

**Files:**
- Modify: `hscredit/report/feature_analyzer.py`
- Modify: `hscredit/report/rule_analysis.py`
- Modify: `hscredit/report/rule_strategy.py`
- Modify: `hscredit/report/swap_analysis.py`
- Modify: `hscredit/report/overdue_predictor.py`
- Modify: `hscredit/report/population_drift.py`
- Modify: `hscredit/report/_sample_stats.py`
- Modify: `tests/test_report/test_feature_binning_summary.py`
- Modify: `tests/test_report/test_feature_efficiency_analysis.py`
- Modify: `tests/test_report/test_feature_report_layout.py`
- Modify: `tests/test_report/test_population_drift.py`
- Modify: `tests/test_report/test_rule_analysis.py`
- Modify: `tests/test_report/test_rule_strategy.py`
- Modify: `tests/test_report/test_swap_analysis.py`
- Create: `tests/test_report/test_parallel_reports.py`

**Interfaces:**
- Produces: keyword parallel parameters on every public report entry point and report estimator
- Produces: parallel calculation followed by deterministic main-thread rendering

- [ ] **Step 1: Write signature and DataFrame parity tests**

Introspect all exported report functions/classes and check the common parameters. Compare serial and parallel results for feature binning summaries, feature efficiency, ruleset analysis, rule swap analysis, SwapAnalyzer, OverduePredictor, and the internal tables used by population drift.

```python
def test_feature_binning_summary_parallel_matches_serial(report_data):
    kwargs = dict(features=["score", "多头"], target="target")
    serial = feature_binning_summary(report_data, n_jobs=1, **kwargs)
    parallel = feature_binning_summary(
        report_data, n_jobs=2, parallel_backend="threading", **kwargs
    )
    pd.testing.assert_frame_equal(serial, parallel)
```

- [ ] **Step 2: Run report tests and verify RED**

Run: `pytest tests/test_report/test_parallel_reports.py -v`

Expected: signatures reject parallel parameters.

- [ ] **Step 3: Separate compute and render phases**

Feature functions submit feature/target/method calculations and assemble tables in declared order. Rule functions submit rule/label/DPD calculations. Swap and overdue components submit independent target/DPD calculations. Population drift submits feature/window calculations. Excel calls and plot creation remain on the main thread.

- [ ] **Step 4: Propagate true nested intent**

Mark `has_parallel_children=True` only when simultaneously active outer workers invoke binners/miners/report calculations that themselves launch workers. Sequential report sections and ordered Excel sheets use the full current budget when their calculation begins.

- [ ] **Step 5: Verify existing report layouts**

Run: `pytest tests/test_report/test_feature_binning_summary.py tests/test_report/test_feature_efficiency_analysis.py tests/test_report/test_rule_analysis.py tests/test_report/test_rule_strategy.py tests/test_report/test_swap_analysis.py tests/test_report/test_population_drift.py tests/test_report/test_parallel_reports.py -v --tb=short`

Expected: DataFrames and existing Excel layout assertions pass.

- [ ] **Step 6: Commit Task 10**

```bash
git add hscredit/report tests/test_report
git commit -m "feat: parallelize analytical reports"
```

### Task 11: Model reports and model comparison

**Files:**
- Modify: `hscredit/report/model_report.py`
- Modify: `tests/test_report/test_model_report.py`
- Modify: `tests/test_report/test_parallel_reports.py`

**Interfaces:**
- Produces: common configuration on `ModelReport`, `QuickModelReport`, `auto_model_report`, and `compare_models`
- Produces: parallel dataset/metric/model computation with sequential rendering

- [ ] **Step 1: Write summary and comparison parity tests**

```python
def test_compare_models_parallel_matches_serial(models, model_xy):
    X, y = model_xy
    serial = compare_models(models, X, y, n_jobs=1)
    parallel = compare_models(models, X, y, n_jobs=2, parallel_backend="threading")
    pd.testing.assert_frame_equal(serial, parallel)
```

Add ModelReport tests for multiple named datasets and overdue/DPD labels; compare summary, discrimination, calibration, stability, feature tables, and Excel sheet order.

- [ ] **Step 2: Run model-report tests and verify RED**

Run: `pytest tests/test_report/test_model_report.py tests/test_report/test_parallel_reports.py -k "model or compare" -v`

Expected: public APIs lack the configuration.

- [ ] **Step 3: Cache predictions and parallelize independent calculations**

Compute each dataset's model score once, then submit independent metric/table calculations. `compare_models` submits model instances in input mapping order and passes child budgets only when each model report runs concurrent internal calculations. Do not call plotting libraries from workers.

- [ ] **Step 4: Verify complete model-report suite**

Run: `pytest tests/test_report/test_model_report.py tests/test_report/test_parallel_reports.py -v --tb=short`

Expected: all tests pass and Excel layout remains stable.

- [ ] **Step 5: Commit Task 11**

```bash
git add hscredit/report/model_report.py tests/test_report/test_model_report.py tests/test_report/test_parallel_reports.py
git commit -m "feat: parallelize model reports"
```

### Task 12: Documentation, real-data parity, performance, and full verification

**Files:**
- Create: `docs/parallelism.md`
- Create: `tests/test_parallel_real_data.py`
- Create: `tests/test_parallel_performance.py`
- Modify: `tests/test_parallel_api_contract.py`

**Interfaces:**
- Produces: user documentation and executable acceptance evidence

- [ ] **Step 1: Document exact public behavior**

Document `-1`, fixed counts, integer-valued floats, ratios, `None`, backends, config keys, real nested budgets, deterministic random-state guidance, memory mapping, and examples:

```python
binner = OptimalBinning(n_jobs=-1)
encoder = WOEEncoder(n_jobs=0.5, parallel_backend="threading")
report = ModelReport(
    model,
    datasets=data,
    n_jobs=4,
    parallel_backend="loky",
    parallel_config={"batch_size": 2, "max_nbytes": "64M", "mmap_mode": "r"},
)
```

- [ ] **Step 2: Add real-workbook serial/parallel acceptance**

Load `examples/hscredit_yyp.xlsx` once. Compare serial and parallel results for:

```python
FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
TARGET = "FPD"
OVERDUE = ["MOB1"]
DPDS = [7, 3, 0]
AMOUNT = "放款金额"
DATE = "放款时间"
CATEGORY = "商品类别"
```

Exercise at least one binner, selector, encoder, rule analysis, rule miner, feature report, overdue report, and model report. Assert exact table equality except explicitly documented third-party floating tolerances.

- [ ] **Step 3: Add reproducible slow benchmarks**

Create deterministic wide numeric/categorical data, many rules, and three labels. Warm each workload once, measure three serial and three parallel runs, and compare medians. Skip speed gates below four physical cores; otherwise assert at least one CPU-heavy wide workflow reaches 1.2x and no other representative parallel median exceeds 1.05x serial.

- [ ] **Step 4: Run focused target-module tests**

Run: `pytest tests/test_utils/test_parallel.py tests/test_utils/test_parallel_runtime.py tests/test_parallel_api_contract.py tests/test_binning/ tests/test_feature_selection/ tests/test_encoding/ tests/test_rules/ tests/test_report/ -m "not slow and not integration" -v --tb=short`

Expected: zero failures.

- [ ] **Step 5: Run the full non-slow suite**

Run: `pytest tests/ -m "not slow and not integration" --tb=short`

Expected: zero new failures; record exact pass/fail/skip counts. The NumPy categorical prerequisite should remove the 12 documented baseline failures.

- [ ] **Step 6: Run real data and performance acceptance**

Run: `pytest tests/test_parallel_real_data.py -m integration -v --tb=short`

Run: `pytest tests/test_parallel_performance.py -m slow -v --tb=short`

Expected: real-data parity passes; performance meets the core-dependent gates or reports a documented skip on fewer than four physical cores.

- [ ] **Step 7: Run packaging and diff checks**

Run: `python -m build`

Run: `python -m twine check dist/*`

Run: `git diff --check`

Run: `git status --short`

Expected: build and twine exit 0, no whitespace errors, and only intended files remain.

- [ ] **Step 8: Commit Task 12**

```bash
git add docs/parallelism.md tests/test_parallel_real_data.py tests/test_parallel_performance.py tests/test_parallel_api_contract.py
git commit -m "docs: verify unified parallel execution"
```
