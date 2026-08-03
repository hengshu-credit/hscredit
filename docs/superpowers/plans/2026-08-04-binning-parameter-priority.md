# Binning Parameter Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make explicit axis and outer `feature_summary` parameters override params dictionaries, while deriving two-dimensional monotonic direction from the one-dimensional binning contract.

**Architecture:** `OptimalBinning2D` will resolve one immutable effective configuration per axis before constructing its internal `OptimalBinning`, then reuse the fitted binner's monotonic result in the two-dimensional merge. `feature_summary` will merge pass-through configuration first and overlay its three public convenience parameters last.

**Tech Stack:** Python 3.9+, pandas, NumPy, scikit-learn estimator conventions, pytest.

## Global Constraints

- Keep the public constructor and function signatures backward compatible.
- Use Chinese for user-visible errors, warnings, DataFrame columns, docstrings, and report content.
- Parameter priority is `explicit _x/_y > x_params/y_params > global` in `OptimalBinning2D`.
- Parameter priority is `outer arguments > binning_params` in `feature_summary`.
- `ascending` means bad rate rises as feature value rises; `descending` means bad rate falls as feature value rises.
- Preserve unrelated user changes in `examples/01_binning.ipynb` and `tests/test_binning/test_categorical_adapter.py`.
- Validate changed behavior with `examples/hscredit_yyp.xlsx`, using `衡枢鉴真分老客版`, `近六个月非银多头机构数`, `青云24`, and target `FPD`.

---

### Task 1: Resolve OptimalBinning2D axis parameters once

**Files:**
- Modify: `hscredit/core/binning/optimal_binning_2d.py:80-105,943-990`
- Test: `tests/test_binning/test_optimal_binning_2d.py:445-493`

**Interfaces:**
- Consumes: Existing global parameters, `x_params` / `y_params`, and explicit `_x` / `_y` constructor attributes.
- Produces: `_resolve_axis_params(self, is_x: bool) -> Dict[str, Any]` and `_create_binner(self, is_x: bool) -> OptimalBinning` using the resolved dictionary.

- [ ] **Step 1: Add failing priority tests**

```python
def test_explicit_axis_params_override_params_and_global():
    binner = OptimalBinning2D(
        max_n_bins=8,
        method="quantile",
        monotonic="descending",
        max_n_bins_x=3,
        method_x="uniform",
        monotonic_x="ascending",
        x_params={
            "max_n_bins": 4,
            "method": "mdlp",
            "monotonic": "descending",
            "min_n_bins": 1,
        },
    )

    axis_binner = binner._create_binner(is_x=True)

    assert axis_binner.max_n_bins == 3
    assert axis_binner.method == "uniform"
    assert axis_binner.monotonic == "ascending"
    assert axis_binner.min_n_bins == 1


def test_axis_params_override_global_when_explicit_axis_params_are_absent():
    binner = OptimalBinning2D(
        max_n_bins=8,
        method="quantile",
        monotonic=False,
        y_params={"max_n_bins": 4, "method": "uniform", "monotonic": "descending"},
    )

    axis_binner = binner._create_binner(is_x=False)

    assert axis_binner.max_n_bins == 4
    assert axis_binner.method == "uniform"
    assert axis_binner.monotonic == "descending"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `pytest tests/test_binning/test_optimal_binning_2d.py::TestOptimalBinning2DCustom::test_explicit_axis_params_override_params_and_global tests/test_binning/test_optimal_binning_2d.py::TestOptimalBinning2DCustom::test_axis_params_override_global_when_explicit_axis_params_are_absent -v`

Expected: the explicit-axis test fails because `x_params` currently overwrites the explicit `_x` attributes after construction.

- [ ] **Step 3: Implement one-pass axis resolution**

```python
def _resolve_axis_params(self, is_x: bool) -> Dict[str, Any]:
    params = {
        "target": self.target,
        "max_n_bins": self.max_n_bins,
        "min_bin_size": self.min_bin_size,
        "method": self.method,
        "monotonic": self.monotonic,
        "missing_separate": self._get_missing_separate(is_x),
        "random_state": self.random_state,
        "decimal": self.decimal,
        "woe_clip": self.woe_clip,
        "verbose": self.verbose,
    }
    extra_params = self.x_params if is_x else self.y_params
    valid_names = set(OptimalBinning().get_params(deep=False))
    for key, value in dict(extra_params or {}).items():
        if key not in valid_names:
            warnings.warn(f"OptimalBinning 无此参数 '{key}'，将忽略")
        else:
            params[key] = value

    explicit = {
        "max_n_bins": self.max_n_bins_x if is_x else self.max_n_bins_y,
        "min_bin_size": self.min_bin_size_x if is_x else self.min_bin_size_y,
        "method": self.method_x if is_x else self.method_y,
        "monotonic": self.monotonic_x if is_x else self.monotonic_y,
        "user_splits": self.user_splits_x if is_x else self.user_splits_y,
        "special_codes": self.special_codes_x if is_x else self.special_codes_y,
    }
    params.update({key: value for key, value in explicit.items() if value is not None})
    return params


def _create_binner(self, is_x: bool) -> OptimalBinning:
    return OptimalBinning(**self._resolve_axis_params(is_x))
```

Remove the post-construction `setattr` loop and the separate post-construction `user_splits` assignment in `fit`, because the resolved constructor configuration now carries those values.

- [ ] **Step 4: Update Chinese parameter documentation and comments**

Document the exact priority as `显式 _x/_y 参数 > x_params/y_params > 全局参数`, and explain that all effective values are resolved before constructing the internal binner so constructor validation cannot be bypassed.

- [ ] **Step 5: Run the Task 1 tests and the full OptimalBinning2D module**

Run: `pytest tests/test_binning/test_optimal_binning_2d.py -v --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add hscredit/core/binning/optimal_binning_2d.py tests/test_binning/test_optimal_binning_2d.py
git commit -m "fix: honor explicit 2d axis parameters"
```

### Task 2: Reuse one-dimensional monotonic direction in the 2D merge

**Files:**
- Modify: `hscredit/core/binning/optimal_binning_2d.py:1165-1364`
- Test: `tests/test_binning/test_optimal_binning_2d.py:744-824`

**Interfaces:**
- Consumes: `_resolve_axis_params(is_x)`, fitted `binner_x_` / `binner_y_`, `monotonic_trend_`, and ordinary-bin bad rates.
- Produces: `_resolve_axis_monotonic_trend(self, is_x: bool) -> Optional[str]`; `_merge_2d_bins` uses it for `trend_x` and `trend_y`.

- [ ] **Step 1: Add failing fitted-trend tests**

```python
def test_2d_auto_monotonic_direction_reuses_fitted_axis_trend():
    binner = OptimalBinning2D(monotonic="auto_asc_desc")
    binner.feature_x_ = "x"
    binner.binner_x_ = SimpleNamespace(
        monotonic="auto_asc_desc",
        monotonic_trend_={"x": "descending"},
        bin_tables_={},
    )

    assert binner._resolve_axis_monotonic_trend(is_x=True) == "descending"


def test_2d_non_directional_axis_trend_is_not_reinterpreted():
    binner = OptimalBinning2D(monotonic="auto")
    binner.feature_y_ = "y"
    binner.binner_y_ = SimpleNamespace(
        monotonic="auto",
        monotonic_trend_={"y": "peak"},
        bin_tables_={},
    )

    assert binner._resolve_axis_monotonic_trend(is_x=False) is None
```

Add `from types import SimpleNamespace` to the test imports.

- [ ] **Step 2: Run the fitted-trend tests and verify RED**

Run: `pytest tests/test_binning/test_optimal_binning_2d.py -k "reuses_fitted_axis_trend or non_directional_axis_trend" -v`

Expected: both tests fail because `_resolve_axis_monotonic_trend` does not exist.

- [ ] **Step 3: Implement shared monotonic resolution**

```python
def _resolve_axis_monotonic_trend(self, is_x: bool) -> Optional[str]:
    value = self._resolve_axis_params(is_x).get("monotonic", False)
    if value in (False, None, "", "none"):
        return None
    if isinstance(value, str):
        value = value.lower()
    if value in ("ascending", "descending"):
        return value

    binner = self.binner_x_ if is_x else self.binner_y_
    feature = self.feature_x_ if is_x else self.feature_y_
    fitted_trend = getattr(binner, "monotonic_trend_", {}).get(feature)
    if fitted_trend in ("ascending", "descending"):
        return fitted_trend
    if fitted_trend is not None:
        return None

    table = getattr(binner, "bin_tables_", {}).get(feature, pd.DataFrame())
    ordinary = table.loc[table["分箱"] >= 0] if "分箱" in table else table
    rates = ordinary.get("坏样本率", pd.Series(dtype=float)).to_numpy(dtype=float)
    if len(rates) < 2:
        return None
    resolved = binner._resolve_monotonic_target_mode(rates, value)
    return resolved if resolved in ("ascending", "descending") else None
```

Replace `_resolve_2d_trend(...)` calls in `_merge_2d_bins` with `_resolve_axis_monotonic_trend(is_x=True/False)` and remove the endpoint-based `_resolve_2d_trend` implementation.

- [ ] **Step 4: Add direction-contract assertions**

```python
def test_2d_direction_names_follow_base_binning_bad_rate_contract():
    solution = np.array([[0], [1]])
    ascending_counts = {0: (1.0, 9.0), 1: (8.0, 2.0)}
    descending_counts = {0: (8.0, 2.0), 1: (1.0, 9.0)}
    binner = OptimalBinning2D()
    binner.n_bins_x_ = 2
    binner.n_bins_y_ = 1

    assert not binner._monotonic_violations(solution, ascending_counts, "ascending", None)
    assert not binner._monotonic_violations(solution, descending_counts, "descending", None)
    assert binner._monotonic_violations(solution, ascending_counts, "descending", None)
```

- [ ] **Step 5: Run Task 2 tests and the full OptimalBinning2D module**

Run: `pytest tests/test_binning/test_optimal_binning_2d.py -v --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add hscredit/core/binning/optimal_binning_2d.py tests/test_binning/test_optimal_binning_2d.py
git commit -m "fix: align 2d monotonic direction"
```

### Task 3: Make feature_summary outer parameters authoritative

**Files:**
- Modify: `hscredit/core/eda/_feature_summary.py:19-38,887-897`
- Modify: `hscredit/core/eda/overview.py:128-240`
- Test: `tests/test_eda/test_feature_summary_parallel.py:190-215,590-680`

**Interfaces:**
- Consumes: `_normalize_binning_config(binning_method, max_n_bins, random_state, binning_params)`.
- Produces: an independent effective dictionary where outer `binning_method`, `max_n_bins`, and `random_state` override duplicate `binning_params` keys while extension keys remain.

- [ ] **Step 1: Reverse the existing priority regression test**

```python
def test_outer_binning_args_override_binning_params_without_mutation():
    params = {
        "method": "uniform",
        "max_n_bins": 3,
        "random_state": 99,
        "min_bin_size": 0.05,
    }
    snapshot = params.copy()

    result = feature_summary_impl._normalize_binning_config(
        binning_method="quantile",
        max_n_bins=10,
        random_state=42,
        binning_params=params,
    )

    assert result["method"] == "quantile"
    assert result["max_n_bins"] == 10
    assert result["random_state"] == 42
    assert result["min_bin_size"] == 0.05
    assert params == snapshot
```

Update the high-level IV test so its expected `OptimalBinning` uses outer `method="quantile"`, `max_n_bins=10`, and `random_state=42`, while retaining `min_bin_size` from `binning_params`.

- [ ] **Step 2: Run the priority tests and verify RED**

Run: `pytest tests/test_eda/test_feature_summary_parallel.py -k "outer_binning_args_override or custom_binning_params_control_iv" -v`

Expected: failures show that `binning_params` still overrides the outer values.

- [ ] **Step 3: Reverse the merge order**

```python
effective = dict(binning_params or {})
# 外层便捷参数表达调用者在当前入口的最终选择，优先于 params 中的同名透传值。
effective.update(
    {
        "method": binning_method,
        "max_n_bins": max_n_bins,
        "random_state": random_state,
    }
)
```

- [ ] **Step 4: Update public documentation and examples**

Change the `feature_summary` docstring to state `binning_method/max_n_bins/random_state > binning_params`, and update the example so `binning_params` demonstrates extension-only values such as `user_splits`, `strict_user_splits`, and `min_bin_size`.

- [ ] **Step 5: Run feature_summary tests**

Run: `pytest tests/test_eda/test_feature_summary_parallel.py -v --tb=short`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add hscredit/core/eda/_feature_summary.py hscredit/core/eda/overview.py tests/test_eda/test_feature_summary_parallel.py
git commit -m "fix: prioritize feature summary outer parameters"
```

### Task 4: Focused regression and real-data verification

**Files:**
- Verify: `hscredit/core/binning/optimal_binning_2d.py`
- Verify: `hscredit/core/eda/_feature_summary.py`
- Verify: `examples/hscredit_yyp.xlsx`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: fresh test and real-data evidence for the final handoff.

- [ ] **Step 1: Run both focused suites together**

Run: `pytest tests/test_binning/test_optimal_binning_2d.py tests/test_eda/test_feature_summary_parallel.py -v --tb=short`

Expected: all selected tests pass.

- [ ] **Step 2: Run the non-slow, non-integration regression suite**

Run: `pytest tests/ -m "not slow and not integration" --tb=short`

Expected: no new failures beyond repository baseline; record exact passed, failed, and skipped counts.

- [ ] **Step 3: Verify the real workbook**

Run this PowerShell command with the repository Python:

```powershell
@'
import pandas as pd
from hscredit.core.binning import OptimalBinning2D
from hscredit.core.eda import feature_summary

path = "examples/hscredit_yyp.xlsx"
features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
df = pd.read_excel(path)
data = df[features + ["FPD"]].dropna(subset=["FPD"])

binner = OptimalBinning2D(
    max_n_bins=6,
    max_n_bins_x=3,
    monotonic="auto_asc_desc",
    monotonic_x="descending",
    x_params={"max_n_bins": 4, "monotonic": "ascending"},
    y_params={"max_n_bins": 4},
).fit(data, y=data["FPD"], features=features[:2])
assert binner.binner_x_.max_n_bins == 3
assert binner.binner_x_.monotonic == "descending"
assert binner.binner_y_.max_n_bins == 4
assert binner.get_bin_table().shape[0] > 0

summary = feature_summary(
    data[features],
    y=data["FPD"],
    features=features,
    binning_method="quantile",
    max_n_bins=5,
    random_state=42,
    binning_params={"method": "uniform", "max_n_bins": 3, "random_state": 99, "min_bin_size": 0.02},
    n_jobs=1,
)
assert set(features).issubset(set(summary["特征名"]))
print({"rows": len(data), "bins_2d": binner.n_bins_2d_, "summary_rows": len(summary)})
'@ | python -
```

Expected: exit code 0 and printed positive row/bin counts.

- [ ] **Step 4: Inspect the final diff and whitespace**

Run: `git diff --check && git status --short && git diff --stat HEAD~3..HEAD`

Expected: no whitespace errors; only the planned implementation/test/docs files and the user's pre-existing unrelated changes are present.
