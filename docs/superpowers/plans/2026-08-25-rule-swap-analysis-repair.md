# rule_swap_analysis Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `rule_swap_analysis` so its stage funnel, production pass-rate calibration, observed/predicted risk mix, amount metrics, and multi-target outputs match the approved business definition.

**Architecture:** Keep the public function and return keys in `hscredit/report/rule_analysis.py`, but replace the current row-diff pipeline with explicit stage masks and atomic quadrant masks. Build pass-rate transitions from stage masks, build risk series per target from actual `y` except predicted OUT-IN risk, then render flat single-target or MultiIndex multi-target reports.

**Tech Stack:** Python 3.9+, pandas, NumPy, pytest, existing `Rule`, `feature_bin_stats`, and parallel utilities.

**Spec:** `docs/superpowers/specs/2026-08-25-rule-swap-analysis-design.md`

## Global Constraints

- Preserve the public `rule_swap_analysis` entry point and return keys `swap_pipeline` and `swap_result`.
- All user-visible DataFrame columns and errors remain Chinese.
- All binning comes from `hscredit.core.binning` through existing `feature_bin_stats` output.
- All rule evaluation uses `hscredit.core.rules.Rule`.
- `0 < sample_survival_rate <= 1`; no integer-expanded production denominator.
- Only OUT-IN uses predicted bad probability by default; all other atomic groups use actual `y`.
- Existing unrelated working-tree changes must not be modified or committed.

---

### Task 1: Stage masks and calibrated pass-rate funnel

**Files:**
- Modify: `hscredit/report/rule_analysis.py:67-657`
- Create: `tests/test_report/test_rule_swap_analysis_regression.py`

**Interfaces:**
- Consumes: existing `_evaluate_swap_rule_masks`, `_combine_swap_masks`, `Rule.predict`.
- Produces: `_SwapStages` and `_build_swap_stages(data, rules_base, rules_out, rules_in, mode, n_jobs, parallel_backend, parallel_config)`, plus calibrated shared pipeline rows consumed by Tasks 2-4.

- [ ] **Step 1: Write failing stage-invariant tests**

```python
def _swap_case():
    data = pd.DataFrame({
        "score": [10, 20, 30, 40, 50, 60],
        "base": [1, 0, 0, 0, 0, 0],
        "out": [0, 1, 0, 0, 0, 0],
        "x": [0, 0, 1, 2, 3, 4],
        "target": [1, 0, 0, 1, 0, 0],
        "MOB1": [10, 0, 8, 4, 0, 12],
        "amount": [100, 200, np.nan, 400, 500, 600],
        "approved_amount": [np.nan, np.nan, 250, np.nan, np.nan, np.nan],
    })
    table = pd.DataFrame({
        "分箱标签": ["[-inf, 35)", "[35, +inf)"],
        "坏样本率": [0.25, 0.50],
    })
    kwargs = {
        "data": data,
        "score": "score",
        "bin_table": table,
        "rules_base": [Rule("base == 1", name="基础拒绝")],
        "rules_out": [Rule("out == 1", name="本次置出")],
        "rules_in": [Rule("x == 1", name="本次置入")],
        "target": "target",
        "n_jobs": 1,
    }
    return data, table, kwargs

def test_rule_swap_uses_total_pass_as_out_in_parent_and_conserves_samples():
    _, _, kwargs = _swap_case()
    result = rule_swap_analysis(**kwargs)
    rows = result["swap_pipeline"].set_index(["规则分类", "指标名称"])
    assert rows.loc[("IN-IN通过", ""), "样本总数"] == 3
    assert rows.loc[("OUT-IN置入", "合计"), "样本总数"] == 1
    assert rows.loc[("ALL-IN置换", ""), "样本总数"] == 4

def test_rule_swap_independent_total_deduplicates_overlapping_rules():
    _, _, kwargs = _swap_case()
    kwargs["rules_in"] = [Rule("x >= 2", name="规则A"), Rule("x >= 3", name="规则B")]
    result = rule_swap_analysis(rule_analysis_mode="independent", **kwargs)
    rows = result["swap_pipeline"].set_index(["规则分类", "指标名称"])
    assert rows.loc[("OUT-IN置入", "规则A"), "样本总数"] == 3
    assert rows.loc[("OUT-IN置入", "规则B"), "样本总数"] == 2
    assert rows.loc[("OUT-IN置入", "合计"), "样本总数"] == 3

def test_rule_swap_survival_rate_calibrates_exact_production_funnel():
    _, _, kwargs = _swap_case()
    result = rule_swap_analysis(sample_survival_rate=0.7, **kwargs)
    pipeline = result["swap_pipeline"]
    assert pipeline.iloc[0]["生产通过率"] == pytest.approx(70.0)
```

- [ ] **Step 2: Run stage tests and verify RED**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "parent or deduplicates or survival" -v`

Expected: FAIL because OUT-IN currently uses the post-base parent, IN-IN is not split, ALL-IN double-counts, and production pass rate is integer-expanded.

- [ ] **Step 3: Implement explicit stage masks**

```python
@dataclass
class _SwapStages:
    base_masks: List[pd.Series]
    out_masks: List[pd.Series]
    in_masks: List[pd.Series]
    out_out: pd.Series
    in_out: pd.Series
    in_in: pd.Series
    out_in: pd.Series
    s1: pd.Series
    s2: pd.Series

def _build_swap_stages(data, rules_base, rules_out, rules_in, mode, n_jobs, parallel_backend, parallel_config):
    def expand_local_union(masks):
        expanded = pd.Series(False, index=data.index)
        for mask in masks:
            expanded.loc[mask.index] |= mask.astype(bool)
        return expanded

    base_masks = _evaluate_swap_rule_masks(
        rules_base, data, mode, n_jobs, parallel_backend, parallel_config
    )
    out_out = _combine_swap_masks(base_masks, data.index)
    s1 = ~out_out
    out_masks = _evaluate_swap_rule_masks(
        rules_out, data.loc[s1], mode, n_jobs, parallel_backend, parallel_config
    )
    in_out = expand_local_union(out_masks)
    s2 = s1 & ~in_out
    in_masks = _evaluate_swap_rule_masks(
        rules_in, data.loc[s2], mode, n_jobs, parallel_backend, parallel_config
    )
    out_in = expand_local_union(in_masks)
    in_in = s2 & ~out_in
    return _SwapStages(
        base_masks, out_masks, in_masks, out_out, in_out, in_in, out_in, s1, s2
    )
```

Build rows from explicit `before_mask` and `after_mask`. Compute `生产通过率 = sample_survival_rate * after_mask.sum() / len(data) * 100`; compute changes against the declared stage parent, never adjacent displayed rows.

- [ ] **Step 4: Run stage tests and verify GREEN**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "parent or deduplicates or survival" -v`

Expected: PASS.

- [ ] **Step 5: Commit stage implementation**

```bash
git add hscredit/report/rule_analysis.py tests/test_report/test_rule_swap_analysis_regression.py
git commit -m "fix: rebuild rule swap stage funnel"
```

### Task 2: Actual risk, predicted OUT-IN risk, and swap_result

**Files:**
- Modify: `hscredit/report/rule_analysis.py:244-738,975-1526`
- Test: `tests/test_report/test_rule_swap_analysis_regression.py`

**Interfaces:**
- Consumes: `_SwapStages`, normalized score bin tables, `target` or `overdue`/`dpds`.
- Produces: `_resolve_target_series(data, target, overdue, dpds)`, `_resolve_risk_uplifts(out_in_uplift, risk_uplifts)`, per-target raw/adjusted risk series, and corrected `swap_result`.

- [ ] **Step 1: Write failing risk-source and summary tests**

```python
def test_non_out_in_uses_actual_y_and_out_in_uses_predicted_uplift():
    _, _, kwargs = _swap_case()
    result = rule_swap_analysis(out_in_uplift=2.0, **kwargs)
    rows = result["swap_pipeline"].set_index(["规则分类", "指标名称"])
    assert rows.loc[("IN-IN通过", ""), "原始坏样本数"] == pytest.approx(1.0)
    assert rows.loc[("OUT-IN置入", "合计"), "原始坏样本数"] == pytest.approx(0.25)
    assert rows.loc[("OUT-IN置入", "合计"), "调整后坏样本数"] == pytest.approx(0.5)

def test_swap_result_compares_in_in_with_all_in_and_ignores_display_order():
    _, _, kwargs = _swap_case()
    normal = rule_swap_analysis(reverse_order=False, **kwargs)["swap_result"]
    reverse = rule_swap_analysis(reverse_order=True, **kwargs)["swap_result"]
    pd.testing.assert_frame_equal(normal, reverse)
    pass_row = normal.set_index("指标").loc["通过率"]
    assert pass_row["变化后"] > pass_row["变化前"]

def test_risk_uplifts_can_adjust_other_atomic_groups():
    _, _, kwargs = _swap_case()
    result = rule_swap_analysis(risk_uplifts={"in_in": 1.5}, **kwargs)
    row = result["swap_pipeline"].query("规则分类 == 'IN-IN通过'").iloc[0]
    assert row["调整后坏样本数"] == pytest.approx(row["原始坏样本数"] * 1.5)
```

- [ ] **Step 2: Run risk tests and verify RED**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "actual_y or swap_result or risk_uplifts" -v`

Expected: FAIL because order metrics currently predict every group, no `risk_uplifts` parameter exists, and summary reads the first OUT-IN detail row.

- [ ] **Step 3: Implement target and risk resolution**

```python
def _resolve_target_series(data, target, overdue, dpds):
    if target is not None:
        return {target: pd.to_numeric(data[target], errors="coerce")}
    overdue_cols = [overdue] if isinstance(overdue, str) else list(overdue)
    thresholds = [dpds] if isinstance(dpds, int) else list(dpds)
    return {
        f"{col}_{dpd}+": (pd.to_numeric(data[col], errors="coerce") > dpd).where(data[col].notna()).astype(float)
        for col in overdue_cols for dpd in thresholds
    }

def _resolve_risk_uplifts(out_in_uplift, risk_uplifts):
    result = {"out_out": 1.0, "in_out": 1.0, "in_in": 1.0, "out_in": float(out_in_uplift)}
    result.update(risk_uplifts or {})
    return result
```

For each target, validate actual labels outside `stages.out_in`, use predicted probability inside `stages.out_in`, multiply atomic masks by their configured uplift, and aggregate row risk from sample-level series. Build `swap_result` from IN-IN and ALL-IN state masks before applying `reverse_order`.

- [ ] **Step 4: Run risk tests and verify GREEN**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "actual_y or swap_result or risk_uplifts" -v`

Expected: PASS.

- [ ] **Step 5: Commit risk implementation**

```bash
git add hscredit/report/rule_analysis.py tests/test_report/test_rule_swap_analysis_regression.py
git commit -m "fix: separate observed and predicted swap risk"
```

### Task 3: Amount metrics and OUT-IN amount filling

**Files:**
- Modify: `hscredit/report/rule_analysis.py:660-738,975-1174`
- Test: `tests/test_report/test_rule_swap_analysis_regression.py`

**Interfaces:**
- Consumes: atomic masks and per-target raw/adjusted risk series from Task 2.
- Produces: effective amount series and additive amount columns without replacing order counts.

- [ ] **Step 1: Write failing amount tests**

```python
def test_amount_keeps_order_counts_and_adds_mixed_risk_amounts():
    data, _, kwargs = _swap_case()
    result = rule_swap_analysis(amount="amount", **kwargs)
    full = result["swap_pipeline"].iloc[0]
    assert full["样本总数"] == len(data)
    assert full["样本总额"] == pytest.approx(data["amount"].sum())
    assert full["生产通过率"] <= 100.0

def test_out_in_amount_column_then_fill_has_precedence():
    _, _, kwargs = _swap_case()
    result = rule_swap_analysis(
        amount="amount",
        out_in_amount_col="approved_amount",
        out_in_amount_fill=1000.0,
        **kwargs,
    )
    out_in = result["swap_pipeline"].query("规则分类 == 'OUT-IN置入' and 指标名称 == '合计'").iloc[0]
    assert out_in["样本总额"] == pytest.approx(250.0)
```

- [ ] **Step 2: Run amount tests and verify RED**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k amount -v`

Expected: FAIL because current code writes amount into `样本总数` and ignores OUT-IN fill parameters.

- [ ] **Step 3: Implement effective amount and additive columns**

```python
effective_amount = pd.to_numeric(data[amount], errors="coerce")
if out_in_amount_col:
    candidate = pd.to_numeric(data[out_in_amount_col], errors="coerce")
    effective_amount.loc[stages.out_in] = candidate.loc[stages.out_in].combine_first(
        effective_amount.loc[stages.out_in]
    )
if out_in_amount_fill is not None:
    effective_amount.loc[stages.out_in] = effective_amount.loc[stages.out_in].fillna(out_in_amount_fill)
```

Add `样本总额`, raw/adjusted bad amount, raw/adjusted amount bad rate, and `金额占比`; retain order `样本总数` and count-calibrated production pass rate.

- [ ] **Step 4: Run amount tests and verify GREEN**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k amount -v`

Expected: PASS.

- [ ] **Step 5: Commit amount implementation**

```bash
git add hscredit/report/rule_analysis.py tests/test_report/test_rule_swap_analysis_regression.py
git commit -m "fix: preserve order and amount swap metrics"
```

### Task 4: Multi-target output and robust bin prediction

**Files:**
- Modify: `hscredit/report/rule_analysis.py:1177-1526`
- Test: `tests/test_report/test_rule_swap_analysis_regression.py`

**Interfaces:**
- Consumes: normalized bin tables and target names from Task 2.
- Produces: `_extract_bad_rate_cols(table, target_names)`, robust per-target score prediction, and MultiIndex pipeline output.

- [ ] **Step 1: Write failing multi-target and bin-edge tests**

```python
def test_multi_dpd_pipeline_exposes_each_nonzero_target():
    _, _, kwargs = _swap_case()
    kwargs["bin_table"] = pd.DataFrame(
        [
            ["[-inf, 35)", 0.10, 0.20, 0.30],
            ["[35, +inf)", 0.40, 0.50, 0.60],
        ],
        columns=pd.MultiIndex.from_tuples([
            ("分箱详情", "分箱标签"),
            ("MOB1_7+", "坏样本率"),
            ("MOB1_3+", "坏样本率"),
            ("MOB1_0+", "坏样本率"),
        ]),
    )
    kwargs.pop("target")
    result = rule_swap_analysis(overdue="MOB1", dpds=[7, 3, 0], **kwargs)
    pipeline = result["swap_pipeline"]
    assert isinstance(pipeline.columns, pd.MultiIndex)
    assert {"MOB1_7+", "MOB1_3+", "MOB1_0+"}.issubset(pipeline.columns.get_level_values(0))
    assert pipeline[("MOB1_7+", "调整后坏样本率")].max() > 0

@pytest.mark.parametrize("label_col", ["分箱标签", "分箱"])
def test_single_bin_and_legacy_label_predict_constant_bad_rate(label_col):
    _, _, kwargs = _swap_case()
    table = pd.DataFrame({label_col: ["[-inf, +inf)"], "坏样本率": [0.25]})
    kwargs["bin_table"] = table
    result = rule_swap_analysis(**kwargs)
    assert result["swap_pipeline"].iloc[0]["原始坏样本率"] > 0

def test_multi_score_bin_keys_and_negative_weights_are_rejected_in_chinese():
    data, table, kwargs = _swap_case()
    with pytest.raises(ValueError, match="评分名必须完全一致"):
        rule_swap_analysis(
            data=data.assign(s1=data["score"], s2=data["score"]),
            score={"a": "s1", "b": "s2"},
            bin_table={"a": table},
            rules_in=kwargs["rules_in"],
            target="target",
            n_jobs=1,
        )
```

- [ ] **Step 2: Run multi-target/bin tests and verify RED**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "multi_dpd or single_bin or bin_keys" -v`

Expected: FAIL because multi-target risk is zero, zero-split tables return zero, and score/bin keys are not validated.

- [ ] **Step 3: Implement target-aware bin lookup**

```python
def _extract_bad_rate_cols(table, target_names):
    if not isinstance(table.columns, pd.MultiIndex):
        return {target_names[0]: "坏样本率"}
    return {
        target: next(col for col in table.columns if col[0] == target and "坏样本率" in col[1])
        for target in target_names
    }
```

Parse labels from plain `分箱标签`/`分箱`, MultiIndex detail columns, or MultiIndex index. Map a single numeric bin to a constant probability; map missing scores to the explicit missing bin or the weighted non-missing mean. Validate score/bin/weight keys before prediction. Render shared columns under `分箱详情` and target metrics under their target groups when more than one target exists.

- [ ] **Step 4: Run multi-target/bin tests and verify GREEN**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "multi_dpd or single_bin or bin_keys" -v`

Expected: PASS.

- [ ] **Step 5: Commit multi-target implementation**

```bash
git add hscredit/report/rule_analysis.py tests/test_report/test_rule_swap_analysis_regression.py
git commit -m "fix: support multi-target swap risk prediction"
```

### Task 5: Validation, documentation, and full verification

**Files:**
- Modify: `hscredit/report/rule_analysis.py:975-1174`
- Modify: `README.md:176-205`
- Test: `tests/test_report/test_rule_swap_analysis_regression.py`
- Test: `tests/test_report/test_parallel_reports.py:691-773`

**Interfaces:**
- Consumes: all prior task interfaces.
- Produces: final public contract and verified examples.

- [ ] **Step 1: Write failing validation tests**

```python
@pytest.mark.parametrize("rate", [0.0, -0.1, 1.1])
def test_sample_survival_rate_must_be_in_unit_interval(rate):
    _, _, kwargs = _swap_case()
    with pytest.raises(ValueError, match="样本集幸存比例必须位于"):
        rule_swap_analysis(sample_survival_rate=rate, **kwargs)

def test_non_out_in_missing_target_is_rejected_in_chinese():
    data, _, kwargs = _swap_case()
    kwargs["data"] = data.assign(target=data["target"].mask(data.index == 4))
    with pytest.raises(ValueError, match="非OUT-IN样本缺少实际表现"):
        rule_swap_analysis(**kwargs)
```

- [ ] **Step 2: Run validation tests and verify RED**

Run: `pytest tests/test_report/test_rule_swap_analysis_regression.py -k "unit_interval or missing_target" -v`

Expected: FAIL because current validation accepts invalid survival rates and silently treats missing actual risk as predicted/good.

- [ ] **Step 3: Implement validation and update public docs**

Validate all configured columns, rule presence, survival rate, finite non-negative uplifts, exact bin-table keys, and finite non-negative weights before expensive work. Update the docstring and README example to explain stage parents, production pass-rate calibration, actual/predicted risk sources, and multi-target output.

- [ ] **Step 4: Run focused and regression verification**

Run:

```bash
pytest tests/test_report/test_rule_swap_analysis_regression.py -v --tb=short
pytest tests/test_report/test_parallel_reports.py -k rule_swap -v --tb=short
pytest tests/test_report/ -m "not slow and not integration" -q --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 5: Run real-data verification**

Run a script using `examples/hscredit_yyp.xlsx`, `score='衡枢鉴真分老客版'`, `target='FPD'`, `overdue=['MOB1']`, `dpds=[7, 3, 0]`, and `amount='放款金额'`. Assert sample conservation, exact 70% calibrated input rate for `sample_survival_rate=0.7`, non-zero target risk, ALL-IN equals total-pass count, and every displayed production pass rate is at most 100%.

- [ ] **Step 6: Commit final validation and docs**

```bash
git add hscredit/report/rule_analysis.py tests/test_report/test_rule_swap_analysis_regression.py tests/test_report/test_parallel_reports.py README.md
git commit -m "fix: complete rule swap analysis repair"
```
