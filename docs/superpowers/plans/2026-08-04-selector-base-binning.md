# Selector Base Binning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize pre-selection binning in `BaseFeatureSelector`, expose `binner` and `binning_params` on every selector, and make IV/correlation selection consume `OptimalBinning` indices without duplicate binning.

**Architecture:** `BaseFeatureSelector` resolves the effective binner using `binner > binning_params`, trains only unfitted instances, and passes `metric="indices"` output into `_fit_impl` while leaving public `transform` raw-valued. Concrete selectors only forward the two common parameters; `CorrSelector` supplies its default `best_iv` configuration and reads metric weights from the same fitted binner's `bin_tables_`.

**Tech Stack:** Python 3.9+, pandas, NumPy, scikit-learn estimator conventions, pytest.

## Global Constraints

- Work directly on the existing `main` branch as explicitly authorized by the user; do not create a worktree or branch.
- Keep all user-visible errors, DataFrame columns, reports, and public docstrings in Chinese.
- `binner` accepts a configured instance only and has priority over `binning_params`.
- `binning_params={}` creates default `OptimalBinning`; `binning_params=None` disables parameter-driven binning except for `CorrSelector()`'s constructor default.
- `CorrSelector()` defaults to `method="best_iv"`, `max_n_bins=5`, `min_bin_size=0.01`, and `missing_separate=True`.
- A fitted binner is reused without refitting; an unfitted binner is trained before index transformation.
- Public selector `transform` continues returning selected raw columns, not bin indices.
- Preserve Python 3.9 compatibility and sklearn `get_params`, `set_params`, `clone`, and Pipeline behavior.
- Validate with `examples/hscredit_yyp.xlsx`, target `FPD`, and the specified HSCredit feature columns.

---

### Task 1: Centralize binner resolution and lifecycle in BaseFeatureSelector

**Files:**
- Modify: `hscredit/core/selectors/base.py:579-903`
- Create: `tests/test_feature_selection/test_selector_binning.py`

**Interfaces:**
- Consumes: `binner: Optional[Any]`, `binning_params: Optional[Dict[str, Any]]`, processed `X`, and optional `y`.
- Produces: `_resolve_binner() -> Optional[Any]`, `_is_binner_fitted(binner: Any) -> bool`, `_should_apply_binner(y) -> bool`, and `_apply_binner(X, y) -> pd.DataFrame`; the resolved instance is exposed as `_binner_instance` after `fit`.

- [ ] **Step 1: Write failing lifecycle and priority tests**

```python
class CaptureSelector(BaseFeatureSelector):
    def _fit_impl(self, X, y):
        self.fit_X_ = X.copy()
        self.selected_features_ = X.columns.tolist()
        self.scores_ = pd.Series(1.0, index=X.columns)


class RecordingBinner:
    def __init__(self, fitted=False):
        self._is_fitted = fitted
        self.fit_calls = 0
        self.metrics = []

    def fit(self, X, y=None):
        self.fit_calls += 1
        self._is_fitted = True
        return self

    def transform(self, X, metric="indices"):
        self.metrics.append(metric)
        return pd.DataFrame(
            {column: np.arange(len(X)) % 2 for column in X.columns},
            index=X.index,
        )


def test_unfitted_binner_is_fitted_once_and_indices_reach_selector(sample_xy):
    X, y = sample_xy
    binner = RecordingBinner()
    selector = CaptureSelector(binner=binner).fit(X, y)

    assert binner.fit_calls == 1
    assert binner.metrics == ["indices"]
    assert selector.fit_X_.iloc[:, 0].tolist() == [0, 1] * (len(X) // 2)


def test_fitted_binner_is_reused_without_refit(sample_xy):
    X, y = sample_xy
    binner = RecordingBinner(fitted=True)
    CaptureSelector(binner=binner).fit(X, y)
    assert binner.fit_calls == 0


def test_binner_priority_ignores_invalid_binning_params(sample_xy):
    X, y = sample_xy
    binner = RecordingBinner(fitted=True)
    selector = CaptureSelector(binner=binner, binning_params="ignored").fit(X, y)
    assert selector._binner_instance is binner


def test_binner_class_is_rejected_with_chinese_error(sample_xy):
    X, y = sample_xy
    with pytest.raises(ValidationError, match="分箱器实例"):
        CaptureSelector(binner=OptimalBinning).fit(X, y)
```

The `sample_xy` fixture is a deterministic 20-row numeric DataFrame with balanced binary `y`; expected transformed values are literal alternating indices so the test fails if raw input reaches `_fit_impl`.

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run: `pytest tests/test_feature_selection/test_selector_binning.py -k "unfitted_binner or fitted_binner or binner_priority or binner_class" -v`

Expected: collection or construction fails because `binning_params` is not accepted, and the fitted-instance test exposes the existing `_is_fitted` detection bug.

- [ ] **Step 3: Implement minimal base behavior**

```python
def _resolve_binner(self) -> Optional[Any]:
    if self.binner is not None:
        if isinstance(self.binner, type):
            raise ValidationError("binner 必须传入配置好的分箱器实例，不能传入分箱器类")
        return self.binner
    if self.binning_params is None:
        return None
    if not isinstance(self.binning_params, dict):
        raise ValidationError("binning_params 分箱参数必须是字典")
    from ..binning import OptimalBinning
    return OptimalBinning(**dict(self.binning_params))


def _is_binner_fitted(self, binner: Any) -> bool:
    for name in ("_is_fitted", "is_fitted_", "fitted_"):
        if hasattr(binner, name):
            return bool(getattr(binner, name))
    try:
        check_is_fitted(binner)
    except SklearnNotFittedError:
        return False
    return True


def _should_apply_binner(self, y) -> bool:
    return self.binner is not None or self.binning_params is not None
```

Add `binning_params` to `BaseFeatureSelector.__init__`, store it unchanged for sklearn compatibility, and replace the current class/instance branches in `_apply_binner`. Use `inspect.signature` to determine whether `transform` accepts `metric` instead of catching arbitrary internal `TypeError`. Reorder DataFrame output to input columns when column sets match; reconstruct ndarray output with the input index/columns; raise `ValidationError` for missing conversion methods or incompatible output shape/columns.

- [ ] **Step 4: Add validation tests for generated OptimalBinning and malformed outputs**

```python
def test_binning_params_create_independent_optimal_binner(sample_xy):
    X, y = sample_xy
    params = {"method": "uniform", "max_n_bins": 2, "min_n_bins": 2}
    snapshot = params.copy()
    selector = CaptureSelector(binning_params=params).fit(X, y)
    assert isinstance(selector._binner_instance, OptimalBinning)
    assert selector._binner_instance.method == "uniform"
    assert params == snapshot


@pytest.mark.parametrize("bad_params", ["uniform", ["method", "uniform"], 3])
def test_invalid_binning_params_are_rejected(bad_params, sample_xy):
    X, y = sample_xy
    with pytest.raises(ValidationError, match="binning_params 分箱参数必须是字典"):
        CaptureSelector(binning_params=bad_params).fit(X, y)


def test_binner_without_transform_or_apply_is_rejected(sample_xy):
    X, y = sample_xy
    with pytest.raises(ValidationError, match="transform 或 apply"):
        CaptureSelector(binner=object()).fit(X, y)
```

- [ ] **Step 5: Run Task 1 tests and base-selector regressions**

Run: `pytest tests/test_feature_selection/test_selector_binning.py -v --tb=short`

Run: `pytest tests/test_feature_selection/test_all_selectors.py tests/test_feature_selection/test_selector_model_compat.py -v --tb=short`

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add hscredit/core/selectors/base.py tests/test_feature_selection/test_selector_binning.py
git commit -m "feat: centralize selector binning lifecycle"
```

### Task 2: Make IVSelector and CorrSelector consume base binning

**Files:**
- Modify: `hscredit/core/selectors/iv_selector.py:89-205`
- Modify: `hscredit/core/selectors/corr_selector.py:35-260`
- Modify: `hscredit/core/selectors/scorecard_feature_selection.py:36-230`
- Modify: `tests/test_feature_selection/test_selector_binning.py`
- Test: `tests/test_feature_selection/test_scorecard_feature_selection.py`

**Interfaces:**
- Consumes: `BaseFeatureSelector._binner_instance`, binned index DataFrame passed to `_fit_impl`, and each binner's `bin_tables_`.
- Produces: `IVSelector(..., binner=None, binning_params=None)`; `CorrSelector.DEFAULT_BINNING_PARAMS`; `_metric_weights_from_binner(feature_names) -> pd.Series` without constructing a second binner.

- [ ] **Step 1: Write failing IV and Corr integration tests**

```python
def test_iv_selector_computes_iv_from_uniform_bin_indices():
    X = pd.DataFrame({"连续变量": np.arange(1, 9, dtype=float)})
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    selector = IVSelector(
        threshold=0.0,
        regularization=1.0,
        binning_params={"method": "uniform", "max_n_bins": 2, "min_n_bins": 2},
    ).fit(X, y)
    assert selector.scores_["连续变量"] == pytest.approx(0.462098, rel=1e-5)
    assert selector.transform(X).equals(X)


def test_corr_selector_uses_default_best_iv_binner_and_same_bin_tables(corr_xy):
    X, y = corr_xy
    selector = CorrSelector(threshold=0.8).fit(X, y)
    binner = selector._binner_instance
    assert binner.method == "best_iv"
    assert binner.max_n_bins == 5
    assert binner.min_bin_size == 0.01
    expected = pd.Series({
        column: binner.bin_tables_[column]["IV值"].sum()
        for column in X.columns
    })
    pd.testing.assert_series_equal(
        selector.scores_.sort_index(), expected.sort_index(), check_names=False
    )


def test_corr_selector_without_target_skips_only_constructor_default():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    selector = CorrSelector(threshold=0.8).fit(X)
    assert selector._binner_instance is None
    assert len(selector.selected_features_) == 1
```

`corr_xy` uses at least 100 deterministic rows with two correlated numeric columns and balanced target values so `best_iv` produces stable bin tables.

- [ ] **Step 2: Run Task 2 tests and verify RED**

Run: `pytest tests/test_feature_selection/test_selector_binning.py -k "iv_selector or corr_selector" -v`

Expected: `IVSelector` rejects the new parameters; `CorrSelector` has no base-managed binner and still creates its internal local binner.

- [ ] **Step 3: Implement IV forwarding and documentation**

Add the common arguments at the end of `IVSelector.__init__`, forward them unchanged, and document the three supported calls:

```python
IVSelector(binning_params={"method": "best_iv", "max_n_bins": 5})
IVSelector(binner=OptimalBinning(method="best_iv", max_n_bins=5))
IVSelector(binner=trained_binner, binning_params={"ignored": True})
```

Keep `_fit_impl` focused on the values it receives; no binner-specific branch belongs in `IVSelector`.

- [ ] **Step 4: Replace CorrSelector's private binning lifecycle**

```python
DEFAULT_BINNING_PARAMS = {
    "method": "best_iv",
    "max_n_bins": 5,
    "min_bin_size": 0.01,
    "missing_separate": True,
}

def _metric_weights_from_binner(self, feature_names: List[str]) -> pd.Series:
    binner = getattr(self, "_binner_instance", None)
    tables = getattr(binner, "bin_tables_", {}) if binner is not None else {}
    metric_key = self.metric.lower()
    if metric_key not in _METRIC_COL_MAP:
        raise ValidationError(f"不支持的指标 '{self.metric}'，可选: {list(_METRIC_COL_MAP)}")
    column_name, agg_func = _METRIC_COL_MAP[metric_key]
    return pd.Series({
        feature: (
            tables[feature][column_name].agg(agg_func)
            if feature in tables and column_name in tables[feature]
            else 0.0
        )
        for feature in feature_names
    })
```

Use a module-level immutable-by-convention default dictionary and copy it per constructor. Treat `binning_params=None` passed explicitly as disabling Corr's default; track whether the effective dictionary equals the default so sklearn-cloned default selectors preserve the no-target skip. Override only `_should_apply_binner(y)` to skip constructor-default supervised binning when `y is None`; all creation, fitting, and conversion remain in `BaseFeatureSelector`. Remove the `OptimalBinning` import/creation from the old `_compute_metric_weights` path.

- [ ] **Step 5: Prevent duplicate binning inside ScorecardFeatureSelection**

When constructing its internal `CorrSelector`, pass the caller's dictionary when `corr_binning_params` is explicit. Otherwise pass `binning_params=None` when the outer selector already binned the data or `corr_weights` has already been resolved; omit the keyword only when no binning and no metric/user weights are available, allowing the Corr default.

Add a regression test with a recording outer binner and enabled correlation stage; assert the outer binner fits once and the internal Corr selector has `_binner_instance is None`.

- [ ] **Step 6: Run focused selector suites**

Run: `pytest tests/test_feature_selection/test_selector_binning.py tests/test_feature_selection/test_scorecard_feature_selection.py -v --tb=short`

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add hscredit/core/selectors/iv_selector.py hscredit/core/selectors/corr_selector.py hscredit/core/selectors/scorecard_feature_selection.py tests/test_feature_selection/test_selector_binning.py tests/test_feature_selection/test_scorecard_feature_selection.py
git commit -m "feat: select features from shared bin indices"
```

### Task 3: Expose common binning parameters on every selector

**Files:**
- Modify: `hscredit/core/selectors/base.py`
- Modify: `hscredit/core/selectors/boruta_selector.py`
- Modify: `hscredit/core/selectors/cardinality_selector.py`
- Modify: `hscredit/core/selectors/chi2_selector.py`
- Modify: `hscredit/core/selectors/f_test_selector.py`
- Modify: `hscredit/core/selectors/importance_selector.py`
- Modify: `hscredit/core/selectors/lift_selector.py`
- Modify: `hscredit/core/selectors/mode_selector.py`
- Modify: `hscredit/core/selectors/mutual_info_selector.py`
- Modify: `hscredit/core/selectors/null_importance_selector.py`
- Modify: `hscredit/core/selectors/null_selector.py`
- Modify: `hscredit/core/selectors/psi_selector.py`
- Modify: `hscredit/core/selectors/regex_selector.py`
- Modify: `hscredit/core/selectors/rfe_selector.py`
- Modify: `hscredit/core/selectors/sequential_selector.py`
- Modify: `hscredit/core/selectors/stability_selector.py`
- Modify: `hscredit/core/selectors/stepwise_selector.py`
- Modify: `hscredit/core/selectors/type_selector.py`
- Modify: `hscredit/core/selectors/variance_selector.py`
- Modify: `hscredit/core/selectors/vif_selector.py`
- Modify: `tests/test_feature_selection/test_selector_binning.py`

**Interfaces:**
- Consumes: the Task 1 base signature.
- Produces: every exported `BaseFeatureSelector` subclass constructor explicitly includes `binner` and `binning_params` and forwards both unchanged.

- [ ] **Step 1: Write a failing public-signature and clone test**

```python
def test_all_exported_selectors_expose_common_binning_parameters():
    missing = []
    for name in selectors.__all__:
        cls = getattr(selectors, name, None)
        if not inspect.isclass(cls) or not issubclass(cls, BaseFeatureSelector):
            continue
        parameters = inspect.signature(cls.__init__).parameters
        if "binner" not in parameters or "binning_params" not in parameters:
            missing.append(name)
    assert missing == []


def test_selector_clone_preserves_binning_configuration():
    selector = IVSelector(
        threshold=0.01,
        binning_params={"method": "uniform", "max_n_bins": 3},
    )
    cloned = clone(selector)
    assert cloned.binner is None
    assert cloned.binning_params == {"method": "uniform", "max_n_bins": 3}
    assert cloned.binning_params is not selector.binning_params
```

- [ ] **Step 2: Run the API tests and verify RED**

Run: `pytest tests/test_feature_selection/test_selector_binning.py -k "all_exported or clone_preserves" -v`

Expected: the exported-selector test lists every constructor that has not yet forwarded the two common parameters.

- [ ] **Step 3: Apply the explicit forwarding pattern**

For every listed constructor, append these parameters without shifting existing positional arguments:

```python
binner: Optional[Any] = None,
binning_params: Optional[Dict[str, Any]] = None,
```

Add `Any` and `Dict` to each module's existing `typing` import, then add both keywords to its `super().__init__` call:

```python
binner=binner,
binning_params=binning_params,
```

For `CompositeFeatureSelector`, retain its existing `binner` position and add only `binning_params` adjacent to it. Do not change selector-specific behavior or reorder any pre-existing public parameter.

- [ ] **Step 4: Run API, clone, and full feature-selection tests**

Run: `pytest tests/test_feature_selection/test_selector_binning.py -v --tb=short`

Run: `pytest tests/test_feature_selection/ -v --tb=short`

Expected: all collected feature-selection tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add hscredit/core/selectors tests/test_feature_selection/test_selector_binning.py
git commit -m "feat: expose binning on all selectors"
```

### Task 4: Regression and real-workbook verification

**Files:**
- Verify: `hscredit/core/selectors/`
- Verify: `tests/test_feature_selection/`
- Verify: `examples/hscredit_yyp.xlsx`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: fresh regression and real-data evidence for the final handoff.

- [ ] **Step 1: Run focused tests together**

Run: `pytest tests/test_feature_selection/ -v --tb=short`

Expected: all feature-selection tests pass.

- [ ] **Step 2: Run the non-slow, non-integration suite**

Run: `pytest tests/ -m "not slow and not integration" --tb=short`

Expected: no new failures beyond the repository baseline documented in `AGENTS.md`; record exact passed, failed, and skipped counts.

- [ ] **Step 3: Verify the real workbook**

```powershell
@'
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.core.selectors import CorrSelector, IVSelector

path = "examples/hscredit_yyp.xlsx"
features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
df = pd.read_excel(path)
data = df[features + ["FPD"]].dropna(subset=["FPD"])
X = data[features]
y = data["FPD"]

iv = IVSelector(
    threshold=0.02,
    binning_params={"method": "best_iv", "max_n_bins": 5, "min_bin_size": 0.01},
).fit(X, y)
assert isinstance(iv._binner_instance, OptimalBinning)
assert iv._binner_instance.method == "best_iv"
assert iv.transform(X).columns.tolist() == iv.selected_features_

trained = OptimalBinning(method="best_iv", max_n_bins=5, min_bin_size=0.01).fit(X, y)
rules_before = trained.export_rules()
reused = IVSelector(threshold=0.02, binner=trained).fit(X, y)
assert trained.export_rules() == rules_before
assert reused._binner_instance is trained

corr = CorrSelector(threshold=0.8).fit(X, y)
assert corr._binner_instance.method == "best_iv"
assert corr._binner_instance.min_bin_size == 0.01
assert set(corr.scores_.index) == set(features)
print({"rows": len(X), "iv_selected": iv.selected_features_, "corr_selected": corr.selected_features_})
'@ | python -
```

Expected: exit code 0, the trained binner rules remain unchanged, and selected feature names print successfully.

- [ ] **Step 4: Check formatting and final diff**

Run: `git diff --check`

Run: `git status --short`

Run: `git log --oneline -5`

Expected: no whitespace errors, only planned files are changed or committed, and no unrelated user files were modified.
