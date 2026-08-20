# HSCredit Model Interpretability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SHAP 变为 HSCredit 基础依赖，并交付可审计的全局、单样本、稳定性、原因码、反事实和 Excel 模型解释能力。

**Architecture:** 以 `ExplanationResult` 作为唯一结构化解释载体，`ModelExplainer` 负责解释器选择与分析，绘图、原因码和反事实分别放入独立模块；旧 `interpretability.py` 保留为兼容门面。`ModelReport` 通过显式 `explain_config` 按需追加第 7 个解释工作表，不改变现有 1–6 页。

**Tech Stack:** Python 3.9+、NumPy、Pandas、scikit-learn、SHAP、SciPy、Matplotlib、Seaborn、openpyxl、pytest。

**Spec:** `docs/superpowers/specs/2026-08-21-model-interpretability-design.md`

## Global Constraints

- SHAP 基础依赖：`shap>=0.49.1,<0.50; python_version < '3.14'`。
- Python 3.14 SHAP 依赖：`shap>=0.51,<0.53; python_version >= '3.14'`。
- 删除 `explain` extra，不保留 `explain = []`，`all` extra 不再引用它。
- 不引入 LIME 依赖、导入、API 或文档。
- 所有用户可见 DataFrame 列名、图标题、错误信息和报告内容使用中文。
- 保留 `compute_shap_values()` 等旧接口，并修复类别选择、陈旧缓存和交互总和缺陷。
- 二分类默认解释标签 `1`；多分类必须显式指定 `target_class`。
- SHAP、Numba 和绘图库继续延迟导入，不增加 `import hscredit` 的重依赖加载。
- `ModelReport` 的解释计算默认关闭；启用后追加 `7-模型解释`，现有 1–6 页名称不变。
- 反事实结果必须遵守约束并注明“模型条件下的非因果建议”。
- 修改后使用 `examples/hscredit_yyp.xlsx` 与仓库约定字段完成真实数据验证。
- 不改动或提交当前工作区中与模型解释无关的用户文件。

---

### Task 1: 基础依赖、安装契约与 NumPy 2 回归保护

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/installation.md`
- Modify: `hscredit/core/models/evaluation/interpretability.py`
- Create: `tests/test_models/test_interpretability_dependencies.py`
- Modify: `tests/test_binning/test_binning.py`

**Interfaces:**
- Consumes: PEP 508 marker syntax and current `compute_bin_stats(bins, y, ...)` API.
- Produces: ordinary HSCredit installation with SHAP, no `explain` extra, and an executable NumPy 2 categorical-bin regression test.

- [ ] **Step 1: Write dependency and regression tests**

```python
def test_shap_is_base_dependency_and_explain_extra_is_removed():
    requirements = importlib.metadata.requires("hscredit") or []
    shap_requirements = [item for item in requirements if item.lower().startswith("shap")]
    assert any("<0.50" in item and "python_version < \"3.14\"" in item for item in shap_requirements)
    assert any("<0.53" in item and "python_version >= \"3.14\"" in item for item in shap_requirements)
    assert not any('extra == "explain"' in item for item in requirements)


def test_compute_bin_stats_accepts_numpy_int8_bin_indices():
    bins = np.array([0, 1, 2, -1, -2, 0], dtype=np.int8)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    result = compute_bin_stats(bins, y)
    assert result["分箱"].notna().all()
```

- [ ] **Step 2: Run tests and confirm the metadata test fails**

Run: `pytest tests/test_models/test_interpretability_dependencies.py tests/test_binning/test_binning.py -q`

Expected: dependency test fails because SHAP is optional and `explain` still exists; NumPy regression test passes if the existing explicit `int(...)` fix is already present.

- [ ] **Step 3: Move SHAP into base dependencies and update installation copy**

```toml
dependencies = [
    # existing dependencies remain in their current order
    "shap>=0.49.1,<0.50; python_version < '3.14'",
    "shap>=0.51,<0.53; python_version >= '3.14'",
]

[project.optional-dependencies]
all = [
    "hscredit[boost,net,pmml,tune,skills,database-all,dev,docs]"
]
```

Delete the `explain` table entry, remove the two `hscredit[explain]` documentation rows, and change the lazy-load error to `SHAP加载失败，请检查当前 Python、NumPy 与 SHAP 版本是否兼容`.

- [ ] **Step 4: Install the editable project and run the dependency tests**

Run: `python -m pip install -e ".[dev]"`

Run: `pytest tests/test_models/test_interpretability_dependencies.py tests/test_binning/test_binning.py -q`

Expected: PASS and `python -c "import shap; print(shap.__version__)"` prints a 0.49.x version on the current Python 3.11 environment.

- [ ] **Step 5: Commit the dependency contract**

```bash
git add pyproject.toml README.md docs/installation.md hscredit/core/models/evaluation/interpretability.py tests/test_models/test_interpretability_dependencies.py tests/test_binning/test_binning.py
git commit -m "build: make shap a base dependency"
```

---

### Task 2: ExplanationResult、输入规范化与数据指纹

**Files:**
- Create: `hscredit/core/models/evaluation/explanation.py`
- Create: `tests/test_models/test_explanation_result.py`

**Interfaces:**
- Consumes: `shap.Explanation`, pandas DataFrame/Index and NumPy arrays.
- Produces: `ExplanationResult`, `coerce_explanation_frame()`, `fingerprint_frame()`, `normalize_explanation_output()`.

- [ ] **Step 1: Write failing tests for shape, identity and fingerprints**

```python
def test_explanation_result_preserves_chinese_columns_and_sample_ids():
    X = pd.DataFrame({"收入": [1.0, 2.0], "年龄": [20, 30]}, index=["A", "B"])
    explanation = shap.Explanation(
        values=np.array([[0.1, -0.2], [0.3, -0.4]]),
        base_values=np.array([0.5, 0.5]),
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )
    result = ExplanationResult.from_explanation(
        explanation,
        data=X,
        target_class=1,
        output_index=1,
        model_output="probability",
        explainer_type="tree",
        background_summary={"样本数": 2},
        metadata={"随机种子": 42},
    )
    assert result.values.shape == (2, 2)
    assert result.feature_names == ["收入", "年龄"]
    assert result.sample_ids.tolist() == ["A", "B"]
    assert result.position_for("B") == 1


def test_fingerprint_changes_when_values_or_schema_change():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    assert fingerprint_frame(X) != fingerprint_frame(X.assign(x=[1.0, 3.0]))
    assert fingerprint_frame(X) != fingerprint_frame(X.rename(columns={"x": "y"}))
```

- [ ] **Step 2: Run tests and confirm imports fail**

Run: `pytest tests/test_models/test_explanation_result.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `ExplanationResult`.

- [ ] **Step 3: Implement the immutable result contract**

```python
@dataclass(frozen=True)
class ExplanationResult:
    explanation: Any
    data: pd.DataFrame
    sample_ids: pd.Index
    target_class: Any
    output_index: Optional[int]
    model_output: str
    explainer_type: str
    background_summary: Mapping[str, Any]
    dataset_fingerprint: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_explanation(cls, explanation, *, data, target_class, output_index,
                         model_output, explainer_type, background_summary, metadata):
        frame = coerce_explanation_frame(data, feature_names=explanation.feature_names)
        normalized = normalize_explanation_output(explanation, output_index=output_index)
        return cls(normalized, frame, frame.index.copy(), target_class, output_index,
                   model_output, explainer_type, MappingProxyType(dict(background_summary)),
                   fingerprint_frame(frame), MappingProxyType(dict(metadata)))

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.explanation.values)
```

`normalize_explanation_output()` must handle a list of class arrays, `(样本, 特征, 输出)` arrays and already selected two-dimensional explanations while applying the same class selection to base values.

- [ ] **Step 4: Run result tests**

Run: `pytest tests/test_models/test_explanation_result.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the structured result**

```bash
git add hscredit/core/models/evaluation/explanation.py tests/test_models/test_explanation_result.py
git commit -m "feat: add structured explanation results"
```

---

### Task 3: ModelExplainer 核心、类别/尺度语义与旧 API 兼容

**Files:**
- Create: `hscredit/core/models/evaluation/explainer.py`
- Modify: `hscredit/core/models/evaluation/interpretability.py`
- Modify: `hscredit/core/models/evaluation/__init__.py`
- Modify: `hscredit/core/models/__init__.py`
- Modify: `hscredit/__init__.py`
- Create: `tests/test_models/test_model_explainer_core.py`
- Modify: `tests/test_models/test_interpretability_contract_repairs.py`

**Interfaces:**
- Consumes: `ExplanationResult.from_explanation()`, estimator `predict_proba()`, optional native tree/linear estimators.
- Produces: `ModelExplainer.explain(X, *, max_samples=None, check_additivity=True) -> ExplanationResult` and compatible `compute_shap_values()` / `get_shap_importance()`.

- [ ] **Step 1: Write failing core tests**

```python
def test_tree_explainer_returns_probability_result_with_additivity():
    X, y = make_classification(n_samples=80, n_features=4, random_state=7)
    X = pd.DataFrame(X, columns=["甲", "乙", "丙", "丁"])
    model = RandomForestClassifier(n_estimators=12, max_depth=3, random_state=7).fit(X, y)
    result = ModelExplainer(model, background_data=X.iloc[:20], random_state=7).explain(X.iloc[20:25])
    expected = model.predict_proba(X.iloc[20:25])[:, 1]
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), expected, atol=1e-5)
    assert result.target_class == 1
    assert result.model_output == "probability"


def test_multiclass_requires_target_class():
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(X, y)
    with pytest.raises(ValidationError, match="多分类.*target_class"):
        ModelExplainer(model, background_data=X.head(20), target_class=None).explain(X.head())


def test_legacy_cache_is_not_reused_for_different_data():
    first = explainer.compute_shap_values(X.iloc[:3])
    second = explainer.compute_shap_values(X.iloc[3:6])
    assert explainer.last_result_.dataset_fingerprint == fingerprint_frame(X.iloc[3:6])
    assert not np.array_equal(first, second)
```

- [ ] **Step 2: Run core tests and confirm they fail**

Run: `pytest tests/test_models/test_model_explainer_core.py tests/test_models/test_interpretability_contract_repairs.py -q`

Expected: FAIL because the new constructor fields, structured result and public exports do not exist.

- [ ] **Step 3: Implement lazy explainer selection and selected-class prediction**

```python
class ModelExplainer:
    def __init__(self, model, feature_names=None, background_data=None,
                 algorithm="auto", model_output="probability", target_class=1,
                 max_background=200, random_state=42, explainer_type=None):
        self.model = model
        self.algorithm = explainer_type or algorithm
        self.model_output = model_output
        self.target_class = target_class
        self.max_background = validate_positive_int(max_background, "max_background")
        self.random_state = random_state
        self.last_result_ = None

    def explain(self, X, *, max_samples=None, check_additivity=True):
        frame = coerce_explanation_frame(X, feature_names=self.feature_names)
        frame = deterministic_sample(frame, max_samples, self.random_state)
        class_index, class_label = self._resolve_target_class()
        background = self._resolve_background(frame)
        backend, explainer_type = self._build_explainer(background, class_index)
        explanation = self._call_explainer(backend, frame, class_index, check_additivity)
        result = ExplanationResult.from_explanation(
            explanation, data=frame, target_class=class_label, output_index=class_index,
            model_output=self._actual_model_output, explainer_type=explainer_type,
            background_summary=self._background_summary(background), metadata=self._metadata(frame),
        )
        self.last_result_ = result
        return result
```

Tree models use `TreeExplainer(..., model_output="probability", feature_perturbation="interventional")` when possible. Linear models requested on probability scale and pipelines use `PermutationExplainer` over a one-dimensional selected-class prediction function. Explicit `kernel` remains supported but is never the automatic fallback.

- [ ] **Step 4: Move the old class to compatibility wrappers and expose public imports**

```python
# interpretability.py
from .explainer import ModelExplainer

# evaluation/__init__.py and models/__init__.py lazy names
_LAZY_EXPLANATION_NAMES = {"ModelExplainer", "ExplanationResult"}

def __getattr__(name):
    if name in _LAZY_EXPLANATION_NAMES:
        value = getattr(importlib.import_module(".evaluation", __name__), name)
        globals()[name] = value
        return value
```

Keep `compute_shap_values()` returning a two-dimensional array, `get_shap_importance()` returning a Series named `SHAP重要性`, and expose `last_result_` as the only cached structured result.

- [ ] **Step 5: Run core and existing interpretability tests**

Run: `pytest tests/test_models/test_model_explainer_core.py tests/test_models/test_interpretability_report.py tests/test_models/test_interpretability_contract_repairs.py tests/test_models/test_model_public_names.py -q`

Expected: PASS with no SHAP skips.

- [ ] **Step 6: Commit the core explainer**

```bash
git add hscredit/core/models/evaluation/explainer.py hscredit/core/models/evaluation/interpretability.py hscredit/core/models/evaluation/__init__.py hscredit/core/models/__init__.py hscredit/__init__.py tests/test_models/test_model_explainer_core.py tests/test_models/test_interpretability_contract_repairs.py
git commit -m "feat: modernize shap model explainer"
```

---

### Task 4: 全局、单样本、代表样本、相关性、聚类与交互分析

**Files:**
- Modify: `hscredit/core/models/evaluation/explainer.py`
- Create: `tests/test_models/test_explanation_analysis.py`

**Interfaces:**
- Consumes: `ExplanationResult` and optional model-native feature importance.
- Produces: `get_global_report()`, `get_sample_report()`, `select_representative_samples()`, `get_correlation_report()`, `get_feature_clusters()`, `get_feature_interactions()`, `get_approximate_interactions()`.

- [ ] **Step 1: Write failing analysis tests**

```python
def test_global_and_sample_reports_have_chinese_audit_columns(fitted_explainer, result):
    global_report = fitted_explainer.get_global_report(result)
    assert {"特征", "平均绝对SHAP值", "SHAP重要性占比", "正向影响占比", "Pearson相关系数"} <= set(global_report)
    assert global_report["SHAP重要性占比"].sum() == pytest.approx(1.0)
    sample_report = fitted_explainer.get_sample_report(result, sample_id=result.sample_ids[0], top_n=2)
    assert sample_report["样本索引"].nunique() == 1
    assert sample_report["贡献排名"].tolist() == [1, 2]


def test_representative_samples_are_deduplicated_and_traceable(fitted_explainer, result):
    selected = fitted_explainer.select_representative_samples(result, threshold=0.5)
    assert selected["样本索引"].is_unique
    assert {"选择理由", "模型输出", "风险排名", "阈值距离"} <= set(selected)


def test_interaction_strength_is_sample_mean_not_sum(tree_explainer, X):
    once = tree_explainer.get_feature_interactions(X)
    twice = tree_explainer.get_feature_interactions(pd.concat([X, X], ignore_index=True))
    pd.testing.assert_series_equal(once["交互强度"], twice["交互强度"], check_names=False, rtol=1e-6)
```

- [ ] **Step 2: Run analysis tests and confirm missing methods**

Run: `pytest tests/test_models/test_explanation_analysis.py -q`

Expected: FAIL with missing analysis methods.

- [ ] **Step 3: Implement deterministic table builders**

```python
def get_global_report(self, result=None):
    result = self._require_result(result)
    values = result.values
    mean_abs = np.mean(np.abs(values), axis=0)
    table = pd.DataFrame({
        "特征": result.feature_names,
        "平均绝对SHAP值": mean_abs,
        "SHAP重要性占比": np.divide(mean_abs, mean_abs.sum(), out=np.zeros_like(mean_abs), where=mean_abs.sum() != 0),
        "平均SHAP值": values.mean(axis=0),
        "正向影响占比": (values > 0).mean(axis=0),
        "负向影响占比": (values < 0).mean(axis=0),
    })
    return self._append_native_importance_and_correlations(table, result).sort_values(
        ["平均绝对SHAP值", "特征"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
```

Implement sample lookup by label or position, representative selection with merged reasons, Pearson/Spearman correlations with finite-value filtering, SciPy hierarchical clustering using deterministic leaf order, exact tree interactions with `mean(abs(values), axis=0)`, and approximate non-tree interactions from binned dependence strength.

- [ ] **Step 4: Run analysis tests and existing interaction tests**

Run: `pytest tests/test_models/test_explanation_analysis.py tests/test_models/test_interpretability_contract_repairs.py -q`

Expected: PASS.

- [ ] **Step 5: Commit structured analysis**

```bash
git add hscredit/core/models/evaluation/explainer.py tests/test_models/test_explanation_analysis.py
git commit -m "feat: add global and local explanation analysis"
```

---

### Task 5: Bootstrap 解释稳定性

**Files:**
- Modify: `hscredit/core/models/evaluation/explainer.py`
- Create: `tests/test_models/test_explanation_stability.py`

**Interfaces:**
- Consumes: fixed `ExplanationResult`, or cloneable estimator plus train/validation data.
- Produces: `get_stability_report(result=None, *, mode, X_train=None, y_train=None, X_validation=None, n_bootstrap=100, confidence_level=0.95, top_k=10, random_state=None) -> pd.DataFrame`.

- [ ] **Step 1: Write failing sample/refit stability tests**

```python
def test_sample_stability_reports_confidence_and_top_k_rate(explainer, result):
    table = explainer.get_stability_report(result, mode="sample", n_bootstrap=20, top_k=2, random_state=3)
    assert {"稳定性模式", "置信区间下限", "置信区间上限", "排名标准差", "Top-K入选率"} <= set(table)
    assert set(table["稳定性模式"]) == {"样本Bootstrap"}
    assert table["Top-K入选率"].between(0, 1).all()


def test_refit_stability_retrains_clone_on_bootstrap_data(X, y):
    model = LogisticRegression(max_iter=300).fit(X, y)
    explainer = ModelExplainer(model, background_data=X.head(15), random_state=5)
    table = explainer.get_stability_report(
        mode="refit", X_train=X, y_train=y, X_validation=X.tail(10),
        n_bootstrap=3, top_k=2, random_state=5,
    )
    assert set(table["稳定性模式"]) == {"重训Bootstrap"}


def test_refit_mode_never_silently_falls_back(explainer):
    with pytest.raises(ValidationError, match="训练数据"):
        explainer.get_stability_report(mode="refit", n_bootstrap=3)
```

- [ ] **Step 2: Run stability tests and confirm missing implementation**

Run: `pytest tests/test_models/test_explanation_stability.py -q`

Expected: FAIL with missing `get_stability_report`.

- [ ] **Step 3: Implement sample and refit bootstrap paths**

```python
def get_stability_report(self, result=None, *, mode="sample", X_train=None, y_train=None,
                         X_validation=None, n_bootstrap=100, confidence_level=0.95,
                         top_k=10, random_state=None):
    rng = np.random.default_rng(self.random_state if random_state is None else random_state)
    if mode == "sample":
        result = self._require_result(result)
        importance_runs = [
            np.abs(result.values[rng.integers(0, len(result.data), len(result.data))]).mean(axis=0)
            for _ in range(n_bootstrap)
        ]
        label = "样本Bootstrap"
    elif mode == "refit":
        self._validate_refit_inputs(X_train, y_train, X_validation)
        importance_runs = self._refit_importance_runs(X_train, y_train, X_validation, n_bootstrap, rng)
        label = "重训Bootstrap"
    else:
        raise ValidationError("mode 必须是 'sample' 或 'refit'")
    return self._summarize_stability(importance_runs, confidence_level, top_k, label)
```

Clone the estimator for every refit, preserve the validation set, skip no failed run, and raise a Chinese error with bootstrap index and original exception chain.

- [ ] **Step 4: Run stability and core tests**

Run: `pytest tests/test_models/test_explanation_stability.py tests/test_models/test_model_explainer_core.py -q`

Expected: PASS.

- [ ] **Step 5: Commit stability analysis**

```bash
git add hscredit/core/models/evaluation/explainer.py tests/test_models/test_explanation_stability.py
git commit -m "feat: add shap stability analysis"
```

---

### Task 6: 中文解释图与综合总览

**Files:**
- Create: `hscredit/core/models/evaluation/explanation_plots.py`
- Modify: `hscredit/core/models/evaluation/explainer.py`
- Modify: `hscredit/core/models/evaluation/interpretability.py`
- Create: `tests/test_models/test_explanation_plots.py`

**Interfaces:**
- Consumes: `ExplanationResult` and structured analysis tables.
- Produces: `plot_decision`, `plot_heatmap`, `plot_distribution`, `plot_correlation`, `plot_feature_clustering`, `plot_interaction_heatmap`, `plot_interaction_bubble`, `plot_importance_overview`, `plot_explanation_overview`, plus compatible existing plot methods.

- [ ] **Step 1: Write no-GUI failing plot tests**

```python
@pytest.mark.parametrize("method_name", [
    "plot_decision", "plot_heatmap", "plot_distribution", "plot_correlation",
    "plot_feature_clustering", "plot_importance_overview", "plot_explanation_overview",
])
def test_new_plots_return_figures_without_showing(method_name, explainer, result, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    kwargs = {"feature": result.feature_names[0]} if method_name == "plot_distribution" else {}
    figure = getattr(explainer, method_name)(result, show=False, **kwargs)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert shown == []


def test_plot_titles_are_chinese(explainer, result):
    figure = explainer.plot_importance_overview(result, show=False)
    assert any("SHAP" in axis.get_title() and "重要性" in axis.get_title() for axis in figure.axes)
```

- [ ] **Step 2: Run plot tests and confirm missing methods**

Run: `pytest tests/test_models/test_explanation_plots.py -q`

Expected: FAIL with missing plot methods.

- [ ] **Step 3: Implement stateless plotting functions and delegating methods**

```python
def plot_importance_overview(result, *, max_display=20, figsize=(12, 7), show=True):
    frame = result.data
    values = result.values
    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:max_display]
    fig, (violin_ax, bar_ax) = plt.subplots(1, 2, figsize=figsize)
    _draw_shap_violin(violin_ax, values[:, order], frame.iloc[:, order], result.feature_names)
    _draw_importance_bar(bar_ax, values[:, order], result.feature_names)
    violin_ax.set_title("SHAP贡献分布")
    bar_ax.set_title("SHAP特征重要性")
    if show:
        plt.show()
    return fig
```

All SHAP-native calls must use `show=False`; capture `plt.gcf()` before optionally showing. Close no figure owned by the caller. Delegate old summary/dependence/force/waterfall methods to the same normalized result so they never recompute for a matching fingerprint.

- [ ] **Step 4: Run all interpretability plot tests**

Run: `pytest tests/test_models/test_explanation_plots.py tests/test_models/test_interpretability_report.py tests/test_visualization -q`

Expected: PASS without opening windows.

- [ ] **Step 5: Commit plots**

```bash
git add hscredit/core/models/evaluation/explanation_plots.py hscredit/core/models/evaluation/explainer.py hscredit/core/models/evaluation/interpretability.py tests/test_models/test_explanation_plots.py
git commit -m "feat: add chinese shap visualizations"
```

---

### Task 7: 通用原因码与评分卡不利原因修复

**Files:**
- Create: `hscredit/core/models/evaluation/reason_codes.py`
- Modify: `hscredit/core/models/evaluation/explainer.py`
- Modify: `hscredit/core/models/scorecard/scorecard.py`
- Create: `tests/test_models/test_reason_codes.py`
- Modify: `tests/test_models/test_scorecard_refactored.py`
- Modify: `tests/test_models/test_round_scorecard.py`

**Interfaces:**
- Consumes: local SHAP contributions or scorecard component-score deltas.
- Produces: `ModelExplainer.get_reason_codes()`, `ScoreCard.get_reason_codes()`, `RoundScoreCard.get_reason_codes()`, corrected legacy `get_reason()`.

- [ ] **Step 1: Write failing directionality tests**

```python
def test_model_reason_codes_only_include_adverse_contributions(result_with_signed_values):
    table = build_reason_codes(
        result_with_signed_values,
        keep=3,
        risk_direction="higher_output_higher_risk",
        reason_map={"负债": {"code": "R001", "description": "负债水平偏高"}},
    )
    assert (table["风险贡献"] > 0).all()
    assert table["原因码"].iloc[0] == "R001"
    assert "非因果" not in "".join(table["原因描述"].astype(str))


def test_scorecard_legacy_reason_never_uses_score_raising_feature(fitted_scorecard, X):
    codes = fitted_scorecard.get_reason_codes(X, keep=3)
    assert (codes["分数影响"] < 0).all()
    legacy = fitted_scorecard.get_reason(X, keep=3)
    assert not legacy["reason"].str.contains("提升").any()
```

- [ ] **Step 2: Run reason tests and confirm adverse-direction failure**

Run: `pytest tests/test_models/test_reason_codes.py tests/test_models/test_scorecard_refactored.py tests/test_models/test_round_scorecard.py -q`

Expected: FAIL because structured reason APIs do not exist and legacy code sorts absolute deltas.

- [ ] **Step 3: Implement the generic reason-code builder**

```python
def build_reason_codes(result, *, keep=3, risk_direction="higher_output_higher_risk",
                       feature_map=None, reason_map=None):
    sign = 1.0 if risk_direction == "higher_output_higher_risk" else -1.0
    rows = []
    for position, sample_id in enumerate(result.sample_ids):
        adverse = sign * result.values[position]
        order = [idx for idx in np.argsort(adverse)[::-1] if adverse[idx] > 0][:keep]
        for rank, idx in enumerate(order, 1):
            rows.append(_reason_row(result, position, sample_id, idx, rank, adverse[idx],
                                    feature_map or {}, reason_map or {}, risk_direction))
    return pd.DataFrame(rows, columns=REASON_CODE_COLUMNS)
```

Validate both risk direction values and mapping schema. Preserve one audit row with `原因状态="无不利贡献"` for samples with no adverse contribution.

- [ ] **Step 4: Implement scorecard structured reasons and delegate legacy strings**

```python
def get_reason_codes(self, X, keep=3, feature_map=None, reason_map=None):
    feature_names, feature_values, score_deltas = self._reason_components(X)
    return _build_scorecard_reason_codes(
        feature_names, feature_values, score_deltas, keep=keep,
        feature_map=feature_map, reason_map=reason_map,
    )

def get_reason(self, X, keep=3):
    codes = self.get_reason_codes(X, keep=keep)
    return _reason_codes_to_legacy_frame(codes, index=_input_index(X))
```

Only negative score deltas are adverse. `RoundScoreCard` obtains deltas from rounded component scores and uses its score formatter.

- [ ] **Step 5: Run reason and scorecard tests**

Run: `pytest tests/test_models/test_reason_codes.py tests/test_models/test_scorecard_refactored.py tests/test_models/test_round_scorecard.py -q`

Expected: PASS.

- [ ] **Step 6: Commit reason codes**

```bash
git add hscredit/core/models/evaluation/reason_codes.py hscredit/core/models/evaluation/explainer.py hscredit/core/models/scorecard/scorecard.py tests/test_models/test_reason_codes.py tests/test_models/test_scorecard_refactored.py tests/test_models/test_round_scorecard.py
git commit -m "feat: add adverse model reason codes"
```

---

### Task 8: 受约束通用与评分卡反事实解释

**Files:**
- Create: `hscredit/core/models/evaluation/counterfactual.py`
- Modify: `hscredit/core/models/evaluation/__init__.py`
- Modify: `hscredit/core/models/__init__.py`
- Create: `tests/test_models/test_counterfactual_explainer.py`

**Interfaces:**
- Consumes: fitted classifier or scorecard, reference DataFrame and per-feature constraint dictionaries.
- Produces: `CounterfactualExplainer.generate(X, *, target_probability=None, target_score=None, max_changes=3, top_n=5, beam_width=50) -> pd.DataFrame`.

- [ ] **Step 1: Write failing constraint and feasibility tests**

```python
def test_generic_counterfactual_respects_immutable_bounds_and_direction(model, reference):
    subject = reference.iloc[[0]].copy()
    counter = CounterfactualExplainer(
        model,
        reference_data=reference,
        constraints={
            "年龄": {"mutable": False},
            "收入": {"min": 0, "direction": "increase_only", "weight": 2.0},
            "职业": {"allowed": ["稳定", "普通"]},
        },
        random_state=7,
    )
    result = counter.generate(subject, target_probability=0.35, max_changes=2, top_n=3)
    assert result["说明"].str.contains("非因果建议").all()
    assert not (result["特征"] == "年龄").any()
    income = result[result["特征"] == "收入"]
    assert (income["新值"].astype(float) >= income["原值"].astype(float)).all()
    assert (result["变更特征数"] <= 2).all()


def test_counterfactual_returns_structured_failure_when_target_is_unreachable(model, reference):
    counter = CounterfactualExplainer(
        model, reference_data=reference,
        constraints={name: {"mutable": False} for name in reference.columns},
    )
    result = counter.generate(reference.iloc[[0]], target_probability=0.0)
    assert result.loc[0, "是否达标"] == "否"
    assert "不可变" in result.loc[0, "失败原因"]


def test_scorecard_counterfactual_uses_reachable_bins(fitted_scorecard, reference):
    result = CounterfactualExplainer(fitted_scorecard, reference).generate(
        reference.iloc[[0]], target_score=fitted_scorecard.predict(reference.iloc[[0]])[0] + 20,
        max_changes=2,
    )
    observed_values = {str(value) for value in reference.to_numpy().ravel()}
    assert set(result["新值"].dropna().map(str)).issubset(observed_values)
    assert (result["预测后值"] >= result["目标值"]).all()
```

- [ ] **Step 2: Run counterfactual tests and confirm missing class**

Run: `pytest tests/test_models/test_counterfactual_explainer.py -q`

Expected: FAIL because `CounterfactualExplainer` does not exist.

- [ ] **Step 3: Implement validation, candidate generation and deterministic beam search**

```python
class CounterfactualExplainer:
    def __init__(self, model, reference_data, constraints=None, random_state=42):
        self.model = model
        self.reference_data = _validate_reference_frame(reference_data)
        self.constraints = _normalize_constraints(self.reference_data, constraints or {})
        self.random_state = random_state
        self._candidates = _build_candidates(self.reference_data, self.constraints)

    def generate(self, X, *, target_probability=None, target_score=None,
                 max_changes=3, top_n=5, beam_width=50):
        frame = _validate_subject_frame(X, self.reference_data.columns)
        objective = _resolve_objective(self.model, target_probability, target_score)
        rows = []
        for sample_id, subject in frame.iterrows():
            plans = self._search_scorecard(subject, objective, max_changes, beam_width)
            if _is_scorecard(self.model) else self._search_generic(subject, objective, max_changes, beam_width)
            rows.extend(_format_counterfactual_rows(sample_id, subject, plans[:top_n], objective))
        return pd.DataFrame(rows, columns=COUNTERFACTUAL_COLUMNS)
```

Numeric candidates are `{min, q05, q10, q25, q50, q75, q90, q95, max}` after bounds/direction filtering. Categorical candidates are observed or explicitly allowed values. Scorecard search uses actual reachable bin representatives and `predict_score`; generic search uses selected-class `predict_proba`. Rank feasible plans by `(变更特征数, 总成本, 目标超额, canonical values)`.

- [ ] **Step 4: Run counterfactual and public import tests**

Run: `pytest tests/test_models/test_counterfactual_explainer.py tests/test_models/test_model_public_names.py -q`

Expected: PASS and repeated runs with the same seed are identical.

- [ ] **Step 5: Commit counterfactual explanations**

```bash
git add hscredit/core/models/evaluation/counterfactual.py hscredit/core/models/evaluation/__init__.py hscredit/core/models/__init__.py tests/test_models/test_counterfactual_explainer.py tests/test_models/test_model_public_names.py
git commit -m "feat: add constrained counterfactual explanations"
```

---

### Task 9: ModelReport 结构化解释与第 7 工作表

**Files:**
- Create: `hscredit/report/model_explanation.py`
- Modify: `hscredit/report/model_report.py`
- Create: `tests/test_report/test_model_explanation_report.py`
- Modify: `tests/test_report/test_model_report.py`

**Interfaces:**
- Consumes: `ModelExplainer`, report datasets and explicit `explain_config`.
- Produces: `ModelReport.get_model_explanation() -> Dict[str, Any]`, optional `模型解释` node in `to_dict()`, optional `7-模型解释` Excel sheet.

- [ ] **Step 1: Write failing opt-in, dictionary and workbook tests**

```python
def test_model_report_does_not_compute_explanation_by_default(report):
    assert report.explain_config["enabled"] is False
    assert "模型解释" not in report.to_dict()


def test_model_report_explanation_dict_is_structured(model, X, y):
    report = ModelReport(
        model, X_train=X, y_train=y, n_jobs=1,
        explain_config={"enabled": True, "data": X.head(12), "background_data": X.head(20),
                        "max_samples": 12, "n_bootstrap": 5},
    )
    explanation = report.get_model_explanation()
    assert {"元信息", "全局解释", "稳定性", "代表样本", "样本解释", "原因码"} <= set(explanation)
    assert "模型解释" in report.to_dict()


def test_model_report_appends_seventh_sheet_without_renaming_existing_sheets(report, tmp_path):
    report.explain_config.update({"enabled": True, "data": report._datasets["训练集"].X,
                                  "background_data": report._datasets["训练集"].X,
                                  "n_bootstrap": 3})
    output = report.to_excel(tmp_path / "解释报告.xlsx", with_plots=False)
    workbook = load_workbook(output)
    assert workbook.sheetnames[:7] == ["目录", "1-基本信息", "2-模型性能", "3-入模变量分析",
                                      "4-稳定性分析", "5-模型参数", "6-模型部署需求"]
    assert workbook.sheetnames[7] == "7-模型解释"
```

- [ ] **Step 2: Run report tests and confirm missing configuration**

Run: `pytest tests/test_report/test_model_explanation_report.py -q`

Expected: FAIL because `explain_config` and `get_model_explanation()` do not exist.

- [ ] **Step 3: Implement validated configuration and transaction-aware cache**

```python
DEFAULT_EXPLAIN_CONFIG = {
    "enabled": False,
    "data": None,
    "background_data": None,
    "target_class": 1,
    "model_output": "probability",
    "max_samples": 500,
    "representative_count": 6,
    "stability_mode": "sample",
    "n_bootstrap": 100,
    "random_state": 42,
    "on_explain_error": "raise",
}

def build_model_explanation(model, config):
    explainer = ModelExplainer(model, background_data=config["background_data"],
                               target_class=config["target_class"],
                               model_output=config["model_output"],
                               random_state=config["random_state"])
    result = explainer.explain(config["data"], max_samples=config["max_samples"])
    representatives = explainer.select_representative_samples(result)
    return {
        "元信息": dict(result.metadata),
        "全局解释": explainer.get_global_report(result),
        "稳定性": explainer.get_stability_report(result, mode=config["stability_mode"],
                                                 n_bootstrap=config["n_bootstrap"]),
        "代表样本": representatives,
        "样本解释": {sample_id: explainer.get_sample_report(result, sample_id=sample_id)
                     for sample_id in representatives["样本索引"]},
        "原因码": explainer.get_reason_codes(result),
        "解释结果": result,
    }
```

Include explanation cache fields in ModelReport cache snapshots so a failed report restores the pre-call state. `on_explain_error="warn"` returns a structured `失败原因`; `raise` preserves the original exception chain.

- [ ] **Step 4: Append the Excel sheet and directory row**

```python
if self.explain_config["enabled"]:
    contents.loc[len(contents)] = {"序号": 7, "内容": "7-模型解释", "备注": "SHAP贡献、稳定性及样本原因"}
    explanation = self.get_model_explanation()
    ws = writer.get_sheet_by_name("7-模型解释")
    end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="七、模型解释", style="header_middle")
    end_row = write_model_explanation_sheet(writer, ws, explanation, start_row=end_row + 2, with_plots=with_plots)
```

Write metadata, global table, correlation/interaction tables, stability, representative samples, local tables, reasons and the non-causal notice. Insert figures only when `with_plots=True`; all tables remain when false.

- [ ] **Step 5: Run report tests and workbook reopen tests**

Run: `pytest tests/test_report/test_model_explanation_report.py tests/test_report/test_model_report.py tests/test_report/test_model_report_method.py -q`

Expected: PASS; default workbook remains seven sheets including directory, enabled workbook has eight sheets including directory and `7-模型解释`.

- [ ] **Step 6: Commit report integration**

```bash
git add hscredit/report/model_explanation.py hscredit/report/model_report.py tests/test_report/test_model_explanation_report.py tests/test_report/test_model_report.py
git commit -m "feat: add model explanation report sheet"
```

---

### Task 10: 文档、可执行示例、构建、回归与真实数据验收

**Files:**
- Modify: `README.md`
- Modify: `docs/installation.md`
- Modify: `docs/api/models.rst`
- Create: `docs/articles/model-interpretability.md`
- Create: `examples/27_model_interpretability.py`
- Modify: `tests/test_models/test_interpretability_dependencies.py`
- Create: `tests/test_models/test_interpretability_example.py`

**Interfaces:**
- Consumes: all public APIs delivered by Tasks 1–9 and a caller-supplied xlsx dataset.
- Produces: runnable Chinese CLI example, public documentation, validated wheel metadata and real-data evidence.

- [ ] **Step 1: Write a failing end-to-end example test with controlled data**

```python
def test_interpretability_example_runs_and_writes_explanation_sheet(tmp_path):
    source = tmp_path / "样例数据.xlsx"
    output = tmp_path / "模型解释报告.xlsx"
    frame = pd.DataFrame({
        "衡枢鉴真分老客版": np.linspace(300, 900, 80),
        "近六个月非银多头机构数": np.tile(np.arange(8), 10),
        "青云24": np.linspace(0.1, 0.9, 80),
        "FPD": np.tile([0, 1], 40),
    })
    frame.to_excel(source, index=False)
    completed = subprocess.run(
        [sys.executable, "examples/27_model_interpretability.py",
         "--input", str(source), "--output", str(output), "--max-samples", "20",
         "--bootstrap", "3"],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "7-模型解释" in load_workbook(output, read_only=True).sheetnames
```

- [ ] **Step 2: Run the executable example test and confirm the script is missing**

Run: `pytest tests/test_models/test_interpretability_example.py -q`

Expected: FAIL because `examples/27_model_interpretability.py` does not exist.

- [ ] **Step 3: Add runnable Chinese documentation and example**

```python
from hscredit.core.models import CounterfactualExplainer, ModelExplainer, RandomForest

model = RandomForest(random_state=42).fit(X_train, y_train)
explainer = ModelExplainer(model, background_data=X_train, random_state=42)
result = explainer.explain(X_test, max_samples=200)
print(explainer.get_global_report(result))
print(explainer.get_sample_report(result, sample_id=result.sample_ids[0]))
print(explainer.get_reason_codes(result, keep=3))

counter = CounterfactualExplainer(model, X_train, constraints={"年龄": {"mutable": False}})
print(counter.generate(X_test.iloc[[0]], target_probability=0.20))
```

Give the script explicit `--input`, `--output`, `--max-samples` and `--bootstrap` arguments. It must validate the four required columns, train a deterministic model, produce global/local/reason/counterfactual console output, write a report and reopen it to assert `7-模型解释`. Document output scales, target class, two stability modes, reason directions, constraints, report opt-in and the non-causal limitation. Add the article to the active `docs/api/models.rst` toctree.

- [ ] **Step 4: Run focused interpretability and report suites**

Run: `pytest tests/test_models/test_explanation_result.py tests/test_models/test_model_explainer_core.py tests/test_models/test_explanation_analysis.py tests/test_models/test_explanation_stability.py tests/test_models/test_explanation_plots.py tests/test_models/test_reason_codes.py tests/test_models/test_counterfactual_explainer.py tests/test_report/test_model_explanation_report.py -q`

Expected: PASS with no skips caused by missing SHAP.

- [ ] **Step 5: Run full non-slow/non-integration regression**

Run: `pytest tests/ -m "not slow and not integration" --tb=short`

Expected: no newly introduced failures; any pre-existing unrelated failure must be listed with its unchanged test name and traceback signature.

- [ ] **Step 6: Build and inspect wheel metadata**

Run: `python -m build`

Run: `python -m twine check dist/*`

Run: `python -c "from importlib.metadata import metadata; print(metadata('hscredit').get_all('Requires-Dist'))"`

Expected: wheel and sdist pass twine; `Requires-Dist` contains both marker-specific SHAP requirements and no `extra == "explain"`.

- [ ] **Step 7: Run the repository real-data validation script**

Run: `python examples/27_model_interpretability.py`

Expected: using `examples/hscredit_yyp.xlsx`, target `FPD` and features `衡枢鉴真分老客版`、`近六个月非银多头机构数`、`青云24`, the script writes an Excel report, reopens it, asserts `7-模型解释` exists, and prints global rows, one local reason set and one constraint-valid counterfactual result.

- [ ] **Step 8: Commit documentation and validation assets**

```bash
git add README.md docs/installation.md docs/api/models.rst docs/articles/model-interpretability.md examples/27_model_interpretability.py tests/test_models/test_interpretability_dependencies.py tests/test_models/test_interpretability_example.py
git commit -m "docs: document model interpretability workflow"
```

- [ ] **Step 9: Record final verification evidence without generated artifacts**

Run: `git status --short`

Expected: no generated `dist/`, Excel report or plot asset is staged; only the user's pre-existing unrelated changes may remain outside the implementation worktree.
