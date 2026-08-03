# Method-Aware Categorical Binning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make categorical variables use the selected binning method through stable ordinal encoding, while fixing custom rules, constraints, missing values, unknown categories, and rule round-trips.

**Architecture:** Add an internal categorical adapter that detects and orders categories, converts them to stable ordinal codes, and restores numeric split results to `List[List]` rules. Direct binners continue to run their existing numeric algorithms and constraints; `OptimalBinning` delegates to those method-specific results instead of replacing them with one shared bad-rate merger.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn estimator APIs, pytest, openpyxl/pandas Excel loading.

## Global Constraints

- Preserve all unrelated dirty-worktree changes.
- All user-facing errors, labels, and validation output must be Chinese.
- Black line length is 120.
- Numeric-coded categories remain opt-in through `cat_cutoff`.
- Missing indices are `-1`, special-value indices are `-2`, and unknown-category indices are `-3`.
- Unknown-category WOE defaults to `0.0` when `handle_unknown="value"`.
- Do not stringify category keys; native `1` and string `"1"` are different categories.
- Automatic category ordering defaults to bad-rate ascending with first-seen tie breaking.
- Explicit user ordering and custom rules must preserve original category value types.
- The “工资” column in `examples/hscredit_hsk.xlsx` is the primary real-data example for explicit category ordering and custom bins.

---

## File Structure

- Create `hscredit/core/binning/_categorical.py`: category-key comparison, ordering, validation, encoding, restoration, and custom-rule normalization.
- Modify `hscredit/core/binning/base.py`: public parameters, common validation, fit context, finalization, unknown labels/WOE, export/import/update lifecycle.
- Modify `hscredit/core/binning/optimal_binning.py`: parameter forwarding, custom-rule behavior, removal of shared regrouping, prebin state transfer.
- Modify the 17 direct binner modules under `hscredit/core/binning/`: finalize method-aware categorical fits and route categorical assignment through the common matcher.
- Create `tests/test_binning/test_categorical_adapter.py`: focused adapter, validation, missing, typed-value, and unknown-category tests.
- Create `tests/test_binning/test_categorical_methods.py`: all-method behavioral and constraint matrix.
- Extend `tests/test_binning/test_categorical_rules.py`: custom-rule and rule-lifecycle regression tests.
- Create `scripts/validate_hsk_categorical_binning.py`: reproducible real-workbook validation with the “工资” example.
- Modify `docs/api/binning.rst`: document `category_order`, `handle_unknown`, custom missing groups, and method semantics.

---

### Task 1: Shared Category Model and Parameter Validation

**Files:**
- Create: `hscredit/core/binning/_categorical.py`
- Modify: `hscredit/core/binning/base.py`
- Test: `tests/test_binning/test_categorical_adapter.py`

**Interfaces:**
- Produces: `is_missing_marker(value) -> bool`
- Produces: `resolve_category_order(feature, x, y, category_order, special_codes) -> list`
- Produces: `encode_ordered_categories(feature, x, ordered_categories, special_codes) -> pd.Series`
- Produces: `restore_category_groups(ordered_categories, numeric_splits) -> list[list]`
- Produces: `normalize_user_groups(feature, groups, observed, special_codes, missing_separate) -> list[list]`
- Produces: Base parameters `category_order` and `handle_unknown`.

- [ ] **Step 1: Write failing constructor and ordering tests**

```python
def test_default_category_order_uses_bad_rate_and_first_seen_ties():
    x = pd.Series(["B", "A", "C", "B", "A", "C"], name="grade")
    y = pd.Series([0, 0, 1, 1, 1, 1])
    order = resolve_category_order("grade", x, y, None, None)
    assert order == ["B", "A", "C"]


def test_category_order_preserves_int_and_string_keys():
    x = pd.Series([1, "1", 1, "1"], dtype=object, name="mixed")
    y = pd.Series([0, 1, 0, 1])
    order = resolve_category_order("mixed", x, y, {"mixed": [1, "1"]}, None)
    assert order == [1, "1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_n_bins": 0},
        {"min_n_bins": 3, "max_n_bins": 2},
        {"min_bin_size": 0},
        {"max_bin_size": -0.1},
        {"min_bad_rate": -0.1},
        {"cat_cutoff": 0},
        {"woe_clip": -1},
        {"handle_unknown": "ignore"},
    ],
)
def test_invalid_common_parameters_raise_chinese_value_error(kwargs):
    with pytest.raises(ValueError):
        OptimalBinning(**kwargs)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py -v`

Expected: collection/import failures for the new helper functions and constructor failures not being raised.

- [ ] **Step 3: Implement typed missing and category ordering helpers**

Implement `_categorical.py` with a private typed key that combines Python type and equality without calling `str()`. Treat `None`, `np.nan`, and `pd.NA` as one missing marker. For default ordering, aggregate `sum`, `count`, first-seen position, then sort by `(bad_rate, first_seen)`.

```python
CategoryOrder = Optional[
    Union[
        Dict[str, Sequence[Any]],
        Callable[[str, pd.Series, pd.Series], Sequence[Any]],
    ]
]


def resolve_category_order(feature, x, y, category_order, special_codes):
    observed = unique_non_missing_typed(x, special_codes)
    supplied = get_supplied_order(feature, x, y, category_order)
    if supplied is not None:
        validate_exact_typed_coverage(feature, supplied, observed)
        return list(supplied)
    return order_by_bad_rate_with_first_seen_ties(x, y, observed, special_codes)
```

- [ ] **Step 4: Add BaseBinning parameters and centralized validation**

Add `category_order=None` and `handle_unknown="value"` to `BaseBinning.__init__`, store them unchanged for sklearn cloning, and validate all common parameters listed in the spec. Update `_detect_feature_type` to use `pd.api.types.is_bool_dtype(series.dtype)` so pandas nullable BooleanDtype is categorical.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py -v`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit the shared category model**

```powershell
git add hscredit/core/binning/_categorical.py hscredit/core/binning/base.py tests/test_binning/test_categorical_adapter.py
git commit -m "feat: add typed categorical binning adapter"
```

---

### Task 2: Method-Aware Fit Lifecycle for Direct Binners

**Files:**
- Modify: `hscredit/core/binning/base.py`
- Modify: `hscredit/core/binning/uniform_binning.py`
- Modify: `hscredit/core/binning/quantile_binning.py`
- Modify: `hscredit/core/binning/tree_binning.py`
- Modify: `hscredit/core/binning/chi_merge_binning.py`
- Modify: `hscredit/core/binning/best_ks_binning.py`
- Modify: `hscredit/core/binning/best_iv_binning.py`
- Modify: `hscredit/core/binning/mdlp_binning.py`
- Modify: `hscredit/core/binning/or_binning.py`
- Modify: `hscredit/core/binning/cp_sat_binning.py`
- Modify: `hscredit/core/binning/cart_binning.py`
- Modify: `hscredit/core/binning/kmeans_binning.py`
- Modify: `hscredit/core/binning/monotonic_binning.py`
- Modify: `hscredit/core/binning/genetic_binning.py`
- Modify: `hscredit/core/binning/smooth_binning.py`
- Modify: `hscredit/core/binning/kernel_density_binning.py`
- Modify: `hscredit/core/binning/best_lift_binning.py`
- Modify: `hscredit/core/binning/target_bad_rate_binning.py`
- Test: `tests/test_binning/test_categorical_methods.py`

**Interfaces:**
- Consumes: Task 1 ordering/encoding/restoration helpers.
- Produces: `BaseBinning._prepare_categorical_fit(X, y) -> pd.DataFrame`
- Produces: `BaseBinning._finalize_categorical_fit() -> None`
- Produces: `_category_orders_`, `_category_code_maps_`, and `_categorical_numeric_splits_` fitted state.

- [ ] **Step 1: Write the all-method RED test**

```python
@pytest.mark.parametrize("binner_cls", DIRECT_BINNER_CLASSES)
def test_direct_binner_uses_numeric_method_for_categories(binner_cls):
    X, y = make_method_difference_category_data()
    binner = binner_cls(max_n_bins=3, min_n_bins=2, random_state=7)
    binner.fit(X, y)
    rules = binner.export_rules()["category"]
    assert binner.feature_types_["category"] == "categorical"
    assert isinstance(rules, list) and all(isinstance(group, list) for group in rules)
    assert binner.n_bins_["category"] == len(rules)
    assert sum(len(group) for group in rules) == X["category"].nunique()
    assert binner.get_bin_table("category")["样本总数"].sum() == len(X)
```

Use small deterministic data and method-specific constructor overrides for algorithms that require parameters such as OR-Tools time limits.

- [ ] **Step 2: Run representative methods and confirm RED**

Run: `python -m pytest tests/test_binning/test_categorical_methods.py -k "Uniform or BestIV or MDLP or Chi" -v`

Expected: inconsistent rules, counts, or `n_bins_` values.

- [ ] **Step 3: Prepare categorical columns as ordinal numeric data**

In `BaseBinning._check_input`, after target validation and before returning, call `_prepare_categorical_fit` unless `_defer_categorical_adapter` is true or `force_numerical` is true. Save the original frame and target in private fit-only state. Mark encoded feature names so `_detect_feature_type` returns `numerical` even when `cat_cutoff` is set.

```python
self._categorical_fit_context_[feature] = CategoricalFitContext(
    original=x.copy(),
    ordered_categories=order,
    encoded=encoded,
)
X_fit[feature] = encoded
```

- [ ] **Step 4: Restore each numeric result to category groups after constraints**

Implement `_finalize_categorical_fit` to read the final numeric splits, restore `List[List]`, set `_cat_bins_`, `splits_`, `n_bins_`, and `feature_types_`, then recompute the bin table from original typed values. Preserve numeric splits in `_categorical_numeric_splits_` for diagnostics and serialization.

- [ ] **Step 5: Call finalization in every direct fit method**

Insert `self._finalize_categorical_fit()` after each method's numeric post-fit constraints/statistics and immediately before `_is_fitted = True`. Do not alter the method's numeric split calculation.

- [ ] **Step 6: Run representative tests and confirm GREEN**

Run: `python -m pytest tests/test_binning/test_categorical_methods.py -k "Uniform or BestIV or MDLP or Chi" -v`

Expected: category rules are complete and each binner's state is internally consistent.

- [ ] **Step 7: Run all 17 direct methods**

Run: `python -m pytest tests/test_binning/test_categorical_methods.py -v`

Expected: all available methods pass; optional-method tests skip only when their declared dependency is unavailable.

- [ ] **Step 8: Commit method-aware fitting**

```powershell
git add hscredit/core/binning tests/test_binning/test_categorical_methods.py
git commit -m "fix: apply native binning methods to categories"
```

---

### Task 3: Shared Category Assignment, Missing Groups, and Unknown Values

**Files:**
- Modify: `hscredit/core/binning/_categorical.py`
- Modify: `hscredit/core/binning/base.py`
- Modify: all 17 direct binner assignment helpers listed in Task 2.
- Test: `tests/test_binning/test_categorical_adapter.py`
- Test: `tests/test_binning/test_categorical_methods.py`

**Interfaces:**
- Produces: `assign_category_groups(feature, x, groups, special_codes, missing_separate, handle_unknown) -> np.ndarray`
- Produces: `BaseBinning._assign_categorical_bins(feature, x) -> np.ndarray`

- [ ] **Step 1: Write failing typed, missing, and unknown tests**

```python
def test_unknown_category_never_falls_into_bin_zero():
    X = pd.DataFrame({"city": ["A", "B", "A", "B"]})
    y = pd.Series([0, 1, 0, 1])
    binner = OptimalBinning(method="uniform", max_n_bins=2).fit(X, y)
    unseen = pd.DataFrame({"city": ["C"]})
    assert binner.transform(unseen, metric="indices").iloc[0, 0] == -3
    assert binner.transform(unseen, metric="bins").iloc[0, 0] == "unknown"
    assert binner.transform(unseen, metric="woe").iloc[0, 0] == 0.0


def test_mixed_int_and_string_categories_get_different_bins():
    X = pd.DataFrame({"value": pd.Series([1, "1", 1, "1"], dtype=object)})
    y = pd.Series([0, 1, 0, 1])
    binner = OptimalBinning(method="uniform", max_n_bins=2).fit(X, y)
    got = binner.transform(pd.DataFrame({"value": [1, "1"]}), metric="indices")["value"]
    assert got.iloc[0] != got.iloc[1]
```

Add parameterized tests for explicit `[np.nan]`, mixed `["c4", np.nan]`, `None`, and `pd.NA` groups.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py -k "unknown or mixed or missing" -v`

Expected: unknown categories map to 0, string/int values collide, or missing-group variants fail.

- [ ] **Step 3: Implement one typed assignment function**

Initialize normal values to `-3`, match each group by typed equality, then apply explicit missing-group, implicit missing `-1`, and special-value `-2` precedence. Raise a Chinese error listing distinct unknown values when `handle_unknown="error"`.

- [ ] **Step 4: Route each direct assignment helper through BaseBinning**

At the start of each `_assign_bins`, `_apply_bins`, or `_apply_splits`, return `_assign_categorical_bins` when the fitted feature is categorical and `_cat_bins_` contains the feature. Keep existing numerical logic unchanged.

- [ ] **Step 5: Add unknown labels and WOE mapping**

Update `_assign_bin_labels`, bin-stat labels, and `_enrich_woe_map` so `-3` maps to `unknown` and neutral WOE `0.0`. Ensure pandas index alignment is preserved when mapping bins to WOE.

- [ ] **Step 6: Run all assignment tests and confirm GREEN**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_methods.py -v`

Expected: all category assignment metrics pass for direct and wrapper binners.

- [ ] **Step 7: Commit shared assignment behavior**

```powershell
git add hscredit/core/binning tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_methods.py
git commit -m "fix: handle missing and unknown categories safely"
```

---

### Task 4: OptimalBinning Delegation and Custom Group Semantics

**Files:**
- Modify: `hscredit/core/binning/optimal_binning.py`
- Modify: `hscredit/core/binning/_categorical.py`
- Test: `tests/test_binning/test_categorical_rules.py`
- Test: `tests/test_binning/test_categorical_methods.py`

**Interfaces:**
- Consumes: direct method-aware fitting from Task 2.
- Produces: `OptimalBinning(category_order=..., handle_unknown=...)`.
- Produces: strict and non-strict atomic custom-group fitting.

- [ ] **Step 1: Write failing wrapper and custom-rule tests**

```python
def test_optimal_binning_does_not_replace_native_category_result():
    X, y = make_method_difference_category_data()
    wrapped = OptimalBinning(method="best_iv", max_n_bins=3).fit(X, y)
    direct = BestIVBinning(max_n_bins=3).fit(X, y)
    assert wrapped.export_rules()["category"] == direct.export_rules()["category"]


@pytest.mark.parametrize(
    "groups",
    [
        [["c1", "c2"], ["c3"], [np.nan]],
        [["c1", "c2"], ["c3"], ["c4", np.nan]],
    ],
)
def test_custom_groups_support_explicit_missing_position(groups):
    X, y = make_custom_group_data()
    binner = OptimalBinning(user_splits={"category": groups}, strict_user_splits=True)
    binner.fit(X, y)
    assert binner.export_rules()["category"] == groups
```

Add failure tests for empty groups, duplicate normal categories, duplicate missing markers, uncovered training categories, and missing values omitted while `missing_separate=False`.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_binning/test_categorical_rules.py -v`

Expected: invalid rules are silently accepted or explicit missing placement is lost.

- [ ] **Step 3: Forward public category parameters to every method**

Add `category_order` and `handle_unknown` to `OptimalBinning.__init__`, `base_params`, `full_params`, method-specific parameter dictionaries, and prebinning construction. Remove method allowlists that currently drop `cat_cutoff`; pass it consistently to all constructors that inherit BaseBinning.

- [ ] **Step 4: Remove wrapper bad-rate regrouping**

Delete the `_regroup_categorical_features` call and obsolete `_group_categories_by_badrate` implementation. Copy the direct binner's category mappings, numeric splits, feature types, and tables without changing group boundaries.

- [ ] **Step 5: Normalize and validate strict custom rules**

Use `normalize_user_groups` before assigning `_cat_bins_`. Preserve explicit missing placement and set `n_bins_` to the number of ordinary groups, including a group that contains missing plus ordinary categories.

- [ ] **Step 6: Implement non-strict atomic group merging**

Encode each user group as one ordered atomic code, fit the selected method on those codes with `cat_cutoff=None`, restore numeric boundaries by concatenating whole input groups, and never split a user group. Recompute the table on original data.

- [ ] **Step 7: Verify direct/wrapper parity and custom rules**

Run: `python -m pytest tests/test_binning/test_categorical_rules.py tests/test_binning/test_categorical_methods.py -v`

Expected: direct and wrapper rules agree for automatic fitting; custom rule validations and both missing forms pass.

- [ ] **Step 8: Commit OptimalBinning integration**

```powershell
git add hscredit/core/binning/optimal_binning.py hscredit/core/binning/_categorical.py tests/test_binning/test_categorical_rules.py tests/test_binning/test_categorical_methods.py
git commit -m "fix: preserve categorical method and custom rules"
```

---

### Task 5: Constraint Enforcement and Rule Lifecycle

**Files:**
- Modify: `hscredit/core/binning/base.py`
- Modify: `hscredit/core/binning/optimal_binning.py`
- Test: `tests/test_binning/test_categorical_adapter.py`
- Test: `tests/test_binning/test_categorical_rules.py`

**Interfaces:**
- Produces: consistent `export_rules`, `import_rules`, `update`, prebin transfer, and constraint diagnostics.

- [ ] **Step 1: Write failing constraint and round-trip tests**

```python
def test_feasible_category_max_bin_size_is_enforced():
    X, y = make_constraint_category_data()
    binner = OptimalBinning(method="best_iv", max_n_bins=4, max_bin_size=0.30).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")
    assert ordinary["样本占比"].max() <= 0.30 + 1e-12


def test_infeasible_single_category_max_size_raises():
    X, y = make_dominant_category_data()
    with pytest.raises(ValueError, match="category.*max_bin_size"):
        OptimalBinning(max_bin_size=0.30).fit(X, y)


def test_export_import_preserves_mixed_missing_group():
    rules = {"工资": [["1000-3000"], ["3000-5000", np.nan]]}
    fitted = make_fitted_wage_binner(rules)
    loaded = OptimalBinning()
    loaded.import_rules(fitted.export_rules())
    assert loaded.export_rules()["工资"] == rules["工资"]
```

Add tests for `min_bin_size`, `min_bad_rate`, descending monotonic order, `cat_cutoff` propagation, prebin `_cat_bins_`, and invalid import rules.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_rules.py -k "constraint or round_trip or prebin" -v`

Expected: ignored constraints, lost category state, or inconsistent round-trips.

- [ ] **Step 3: Verify restored groups against final numeric constraints**

After restoration, calculate ordinary group counts and bad rates on original rows. Raise a Chinese `ValueError` when a constraint cannot be met without splitting an atomic category or user group. Include feature name, parameter name, configured limit, and observed value.

- [ ] **Step 4: Preserve monotonic ordering and numeric split diagnostics**

Use the method's final numeric bin order as category-group order. For explicit `monotonic="descending"`, assert restored bad rates follow descending order rather than applying the old unconditional ascending regroup.

- [ ] **Step 5: Fix export/import/update/prebin transfer**

Normalize imported `List[List]` rules, preserve missing placement, initialize neutral WOE maps when statistics are unavailable, and copy `_category_orders_`, `_category_code_maps_`, `_categorical_numeric_splits_`, `_cat_bins_`, and `n_bins_` through prebinning and wrapper delegation.

- [ ] **Step 6: Run constraint and lifecycle tests**

Run: `python -m pytest tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_rules.py -v`

Expected: all parameter, feasibility, and round-trip tests pass.

- [ ] **Step 7: Commit constraints and lifecycle fixes**

```powershell
git add hscredit/core/binning/base.py hscredit/core/binning/optimal_binning.py tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_rules.py
git commit -m "fix: enforce categorical constraints and rule lifecycle"
```

---

### Task 6: “工资” Real-Data Validation, Documentation, and Full Regression

**Files:**
- Create: `scripts/validate_hsk_categorical_binning.py`
- Modify: `docs/api/binning.rst`
- Test: `tests/test_binning/test_categorical_methods.py`

**Interfaces:**
- Consumes: all completed categorical APIs.
- Produces: deterministic Chinese validation output and documented public examples.

- [ ] **Step 1: Add the real “工资” ordering validation**

The script must load `examples/hscredit_hsk.xlsx`, select `target`, infer the non-missing wage categories, and build an explicit order from the workbook values. It must run both automatic and explicit ordering and verify the saved order.

```python
df = pd.read_excel(path)
wage_order = [value for value in df["工资"].dropna().drop_duplicates().tolist()]
binner = OptimalBinning(
    method=method,
    max_n_bins=5,
    cat_cutoff=10,
    category_order={"工资": wage_order},
)
binner.fit(df.drop(columns="target"), df["target"])
assert binner._category_orders_["工资"] == wage_order
```

Also validate strict custom wage groups with missing alone and missing merged into the final wage group.

- [ ] **Step 2: Add the 17-method workbook matrix**

For each available method, fit all features, transform `indices`, `bins`, and `woe`, and assert row count, column count, total bin-table counts, absence of silent `-3` on training data, and non-empty exported rules for categorical features. Print one Chinese summary row per method.

- [ ] **Step 3: Run workbook validation**

Run: `python scripts/validate_hsk_categorical_binning.py examples/hscredit_hsk.xlsx`

Expected: all available methods report success; optional dependencies are explicitly reported as skipped, not silently swallowed.

- [ ] **Step 4: Document the public behavior**

Add examples for default bad-rate ordering, explicit `category_order={"工资": [...]}`, callable ordering, custom `[np.nan]`, custom `["工资档", np.nan]`, `handle_unknown`, and the statement that each method uses its native numeric criterion after ordinal encoding.

- [ ] **Step 5: Run focused and full binning tests**

Run: `python -m pytest tests/test_binning -q`

Expected: all tests pass with no new warnings.

- [ ] **Step 6: Run broader non-slow regression**

Run: `python -m pytest tests -m "not slow and not integration" -q`

Expected: all collected non-slow, non-integration tests pass. If an unrelated dirty-worktree test fails, record the exact test and prove the binning-specific suite remains green.

- [ ] **Step 7: Run formatting and lint checks for changed files**

Run: `python -m black --check hscredit/core/binning tests/test_binning scripts/validate_hsk_categorical_binning.py`

Run: `python -m flake8 hscredit/core/binning tests/test_binning scripts/validate_hsk_categorical_binning.py`

Expected: both commands pass or only pre-existing unrelated violations are documented with exact paths.

- [ ] **Step 8: Commit validation and docs**

```powershell
git add scripts/validate_hsk_categorical_binning.py docs/api/binning.rst tests/test_binning
git commit -m "test: validate categorical binning on hsk data"
```

---

## Final Verification Checklist

- [ ] `git diff --check` passes for all task changes.
- [ ] Automatic categories use bad-rate or user-supplied ordering.
- [ ] The selected method determines category split boundaries.
- [ ] `[['c1', 'c2'], ['c3'], [np.nan]]` works.
- [ ] `[['c1', 'c2'], ['c3'], ['c4', np.nan]]` works.
- [ ] “工资” explicit ordering and custom rules work on `hscredit_hsk.xlsx`.
- [ ] Unknown categories never silently enter bin 0.
- [ ] All category constraints either succeed or raise a precise Chinese infeasibility error.
- [ ] Direct binners and `OptimalBinning` have consistent fitted state.
- [ ] Rule export/import/update and prebinning preserve category metadata.
- [ ] `tests/test_binning` passes.
- [ ] Real-workbook 17-method validation passes for installed methods.
