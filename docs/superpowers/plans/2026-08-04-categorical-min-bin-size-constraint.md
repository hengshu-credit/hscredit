# Categorical Min Bin Size Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在类别分箱最终校验前合并仍可修复的小箱，使 `uniform`、`genetic` 等方法真正满足 `min_bin_size`，同时保持 `min_bin_size=1` 表示 1 条样本。

**Architecture:** 保留各算法在类别有序编码上生成的初始切分点，在 `BaseBinning._finalize_categorical_fit` 中复用现有 `_adjust_splits_for_bin_size_constraints` 做统一样本量约束调整，再还原 `List[List]` 类别规则并执行硬校验。测试通过真实分箱器观察最终普通箱样本数，不模拟内部方法。

**Tech Stack:** Python 3.9+、pandas、NumPy、pytest、scikit-learn 兼容估计器接口。

## Global Constraints

- 所有用户可见错误与报告列名使用中文。
- `0 < min_bin_size < 1` 按比例计算，沿用现有向下取整逻辑并保证至少为 1。
- `min_bin_size >= 1` 按绝对样本数计算；`min_bin_size=1` 表示至少 1 条样本。
- 不改变数值型分箱行为、缺失箱、特殊值箱、未知类别索引或严格用户规则。
- 使用 `examples/hscredit_yyp.xlsx` 的 `商品类别` 与 `target_demo=(MOB1 > 3)` 做真实数据验证。

---

### Task 1: 锁定可行小箱必须先合并的行为

**Files:**
- Modify: `tests/test_binning/test_categorical_adapter.py`

**Interfaces:**
- Consumes: `UniformBinning.fit(X, y)`、`GeneticBinning.fit(X, y)`、`BaseBinning.get_bin_table(feature)`。
- Produces: 回归测试 `test_category_min_bin_size_merges_feasible_sparse_bins` 和边界测试 `test_min_bin_size_one_means_one_sample`。

- [ ] **Step 1: 添加稀有类别测试数据和失败测试**

```python
def _make_sparse_category_data():
    categories = []
    targets = []
    for category, count, bad_count in [
        ("A", 4, 0),
        ("B", 2, 0),
        ("C", 20, 2),
        ("D", 141, 17),
        ("E", 803, 143),
    ]:
        categories.extend([category] * count)
        targets.extend([1] * bad_count + [0] * (count - bad_count))
    return pd.DataFrame({"category": categories}), pd.Series(targets, name="target")


@pytest.mark.parametrize("binner_cls", [UniformBinning, GeneticBinning])
def test_category_min_bin_size_merges_feasible_sparse_bins(binner_cls):
    X, y = _make_sparse_category_data()
    binner = binner_cls(
        min_n_bins=2,
        max_n_bins=5,
        min_bin_size=0.01,
        random_state=42,
    ).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")

    assert ordinary["样本总数"].min() >= 9
    assert {value for group in binner.export_rules()["category"] for value in group} == set(X["category"])
```

- [ ] **Step 2: 添加 `min_bin_size=1` 的绝对样本数边界测试**

```python
def test_min_bin_size_one_means_one_sample():
    X = pd.DataFrame({"category": ["A", "B", "C", "D"]})
    y = pd.Series([0, 0, 1, 1], name="target")
    binner = UniformBinning(
        min_n_bins=2,
        max_n_bins=4,
        min_bin_size=1,
    ).fit(X, y)
    ordinary = binner.get_bin_table("category").query("分箱 >= 0")

    assert ordinary["样本总数"].min() == 1
```

- [ ] **Step 3: 运行测试并确认 RED**

Run: `pytest tests/test_binning/test_categorical_adapter.py::test_category_min_bin_size_merges_feasible_sparse_bins -v`

Expected: 两个参数化用例均在 `fit` 中因小箱样本数低于 9 而抛出 `ValueError`。边界测试在修改前应通过，用于防止修复误改参数语义。

### Task 2: 在类别规则还原前执行公共样本量调整

**Files:**
- Modify: `hscredit/core/binning/base.py:385-402`
- Test: `tests/test_binning/test_categorical_adapter.py`

**Interfaces:**
- Consumes: `encode_ordered_categories(...) -> pd.Series`、`BaseBinning._get_min_samples(n_samples) -> int`、`BaseBinning._get_max_samples(n_samples) -> Optional[int]`、`BaseBinning._adjust_splits_for_bin_size_constraints(...) -> np.ndarray`。
- Produces: `_finalize_categorical_fit()` 在规则还原前完成可行的最小/最大箱样本量调整。

- [ ] **Step 1: 写入最小生产代码**

在 `_ensure_categorical_minimum_bins` 之后、保存 `_categorical_numeric_splits_` 之前加入：

```python
order = self._category_orders_.get(feature, [])
encoded = encode_ordered_categories(original, order, self.special_codes)
numeric_splits = self._adjust_splits_for_bin_size_constraints(
    encoded,
    y,
    numeric_splits,
    BaseBinning._get_min_samples(self, len(y)),
    BaseBinning._get_max_samples(self, len(y)),
)
numeric_splits = self._round_splits(numeric_splits)
```

随后继续使用调整后的 `numeric_splits` 更新 `self.splits_`、`_categorical_numeric_splits_`、类别组和统计表。

- [ ] **Step 2: 运行定向测试并确认 GREEN**

Run: `pytest tests/test_binning/test_categorical_adapter.py::test_category_min_bin_size_merges_feasible_sparse_bins tests/test_binning/test_categorical_adapter.py::test_min_bin_size_one_means_one_sample -v`

Expected: `3 passed`。

- [ ] **Step 3: 运行类别分箱回归测试**

Run: `pytest tests/test_binning/test_categorical_adapter.py tests/test_binning/test_categorical_methods.py tests/test_binning/test_categorical_binning_complete.py -v --tb=short`

Expected: 全部通过；不可行 `max_bin_size` 用例仍抛出预期错误。

- [ ] **Step 4: 检查差异并提交实现**

```bash
git diff --check
git diff -- hscredit/core/binning/base.py tests/test_binning/test_categorical_adapter.py
git add hscredit/core/binning/base.py tests/test_binning/test_categorical_adapter.py
git commit -m "fix: merge sparse categorical bins before validation"
```

### Task 3: 真实数据与全量回归验证

**Files:**
- Verify: `examples/hscredit_yyp.xlsx`
- Verify: `hscredit/core/binning/base.py`
- Verify: `tests/test_binning/test_categorical_adapter.py`

**Interfaces:**
- Consumes: `OptimalBinning.VALID_METHODS`、`OptimalBinning.fit`、`OptimalBinning.get_bin_table`。
- Produces: 对用户报告场景和项目非慢速测试的验证证据。

- [ ] **Step 1: 遍历真实数据的全部类别分箱方法**

Run the following script from the repository root:

```python
import pandas as pd
from hscredit.core.binning import OptimalBinning

df = pd.read_excel("examples/hscredit_yyp.xlsx")
df["target_demo"] = (df["MOB1"] > 3).astype(int)
feature = "商品类别"

for method in OptimalBinning.VALID_METHODS:
    binner = OptimalBinning(
        method=method,
        max_n_bins=5,
        min_bin_size=0.01,
        random_state=42,
    ).fit(df[[feature]], df["target_demo"])
    ordinary = binner.get_bin_table(feature).query("分箱 >= 0")
    assert ordinary["样本总数"].min() >= 9, (method, ordinary["样本总数"].tolist())
```

Expected: 所有 `VALID_METHODS` 成功，无断言失败；`uniform` 和 `genetic` 不再抛错。

- [ ] **Step 2: 运行完整非慢速、非集成测试**

Run: `pytest tests/ -m "not slow and not integration" -v --tb=short`

Expected: 本次改动不新增失败；如仓库基线仍含已知失败，逐项确认均与本次修改无关并报告实际数量。

- [ ] **Step 3: 检查最终工作区范围**

```bash
git status --short
git diff --check HEAD^
git show --stat --oneline HEAD
```

Expected: 实现提交仅包含 `base.py` 与类别适配器测试；用户原有 `examples/01_binning.ipynb` 修改未被纳入。
