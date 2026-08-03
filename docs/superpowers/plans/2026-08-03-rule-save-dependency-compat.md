# Rule.save And Dependency Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `Rule.save`，并以显式版本矩阵兼容当前 LightGBM/Dask/Pandas、LightGBM/scikit-learn 和 Seaborn/Pandas 组合，同时保持 hscredit 公开 API 一致。

**Architecture:** 新建单一职责的 `hscredit._compat`，只读取已安装发行版版本并根据有上下界的规则安装幂等适配器。LightGBM、Seaborn 与懒加载入口调用该层；兼容分支不使用异常捕获、签名探测、错误文本解析或失败重试。`Rule.save` 直接复用现有 Excel 核心，不复制 DataFrame 写入逻辑。

**Tech Stack:** Python 3.9+、Pandas、packaging、importlib.metadata、LightGBM、Dask、scikit-learn、Seaborn、openpyxl、pytest。

## Global Constraints

- 兼容分支必须由明确版本号决定，禁止使用 `try/except`、`inspect.signature()`、错误文本或失败重试选择兼容路径。
- 保持现有 hscredit 对外类名、函数名和基本输出语义。
- 不收紧第三方依赖范围。
- 保留 Pandas MultiIndex、多层级列、Index 和输入行序。
- 不引入或规划 Polars。
- 仅修改本计划列出的文件；保留工作树内其他任务的未提交改动。

## File Map

- Create `hscredit/_compat.py`: 版本读取、纯版本谓词、适配器注册与幂等应用。
- Create `tests/test_dependency_compat.py`: 版本边界、适配动作、懒加载错误语义测试。
- Modify `hscredit/__init__.py`: 在可选重依赖加载前执行轻量运行时兼容准备。
- Modify `hscredit/_lazy.py`: Seaborn 真正加载前调用同一兼容入口。
- Modify `hscredit/core/models/boosting/lightgbm_model.py`: 显式 LightGBM 版本分支、scikit-learn 适配、移除异常驱动 fit 重试。
- Modify `hscredit/core/models/boosting/__init__.py`: 保留真实导入异常，不伪装为属性不存在。
- Modify `hscredit/core/models/tuning/__init__.py`: 保留真实导入异常。
- Modify `hscredit/core/models/__init__.py`: 禁止缓存 `None`。
- Modify `hscredit/core/viz/binning_plots.py`: 旧 Seaborn 组合下按旧语义处理正负无穷。
- Modify `tests/test_visualization/test_hist_plot_theme_color.py`: 增加无穷值回归。
- Modify `hscredit/core/rules/rule.py`: 通过 `dataframe2excel` 实现 `Rule.save`。
- Create `tests/test_rules/test_rule_save.py`: 路径、writer、多层级列/Index、参数和错误测试。

---

### Task 1: 显式版本矩阵与适配注册中心

**Files:**
- Create: `hscredit/_compat.py`
- Create: `tests/test_dependency_compat.py`
- Modify: `hscredit/__init__.py`

**Interfaces:**
- Produces: `installed_version(import_name: str, distribution_name: Optional[str] = None) -> Optional[Version]`
- Produces: `needs_lightgbm_dask_pandas_compat(lightgbm, pandas, dask) -> bool`
- Produces: `needs_lightgbm_sklearn_compat(lightgbm, sklearn) -> bool`
- Produces: `needs_seaborn_pandas_compat(seaborn, pandas) -> bool`
- Produces: `prepare_dependency(name: str) -> None`
- Produces: `prepare_runtime_compatibility() -> None`

- [ ] **Step 1: 写版本边界失败测试**

```python
from packaging.version import Version

from hscredit._compat import (
    needs_lightgbm_dask_pandas_compat,
    needs_lightgbm_sklearn_compat,
    needs_seaborn_pandas_compat,
)


def V(value):
    return None if value is None else Version(value)


def test_lightgbm_dask_pandas_matrix_has_explicit_boundaries():
    assert needs_lightgbm_dask_pandas_compat(V("3.2.0"), V("2.0.0"), V("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(V("3.1.1"), V("2.0.0"), V("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(V("4.7.0"), V("2.0.0"), V("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(V("3.3.5"), V("1.5.3"), V("2023.1.1"))
    assert not needs_lightgbm_dask_pandas_compat(V("3.3.5"), V("2.0.0"), V("2023.2.0"))
    assert not needs_lightgbm_dask_pandas_compat(V("3.3.5"), V("2.0.0"), None)


def test_lightgbm_sklearn_matrix_has_explicit_boundaries():
    assert needs_lightgbm_sklearn_compat(V("4.5.0"), V("1.8.0"))
    assert not needs_lightgbm_sklearn_compat(V("4.6.0"), V("1.8.0"))
    assert not needs_lightgbm_sklearn_compat(V("4.5.0"), V("1.7.2"))


def test_seaborn_pandas_matrix_has_explicit_boundaries():
    assert needs_seaborn_pandas_compat(V("0.11.0"), V("2.0.0"))
    assert needs_seaborn_pandas_compat(V("0.12.1"), V("2.3.3"))
    assert not needs_seaborn_pandas_compat(V("0.12.2"), V("2.3.3"))
    assert not needs_seaborn_pandas_compat(V("0.11.2"), V("1.5.3"))
```

- [ ] **Step 2: 运行测试确认因模块/函数不存在而失败**

Run: `pytest tests/test_dependency_compat.py -v`

Expected: FAIL during collection because `hscredit._compat` does not exist.

- [ ] **Step 3: 实现版本读取和纯版本谓词**

```python
from importlib import metadata, util
from typing import Optional

from packaging.version import Version


def installed_version(import_name: str, distribution_name: Optional[str] = None) -> Optional[Version]:
    if util.find_spec(import_name) is None:
        return None
    return Version(metadata.version(distribution_name or import_name))


def needs_lightgbm_dask_pandas_compat(lightgbm, pandas, dask):
    return (
        lightgbm is not None
        and Version("3.2.0") <= lightgbm < Version("4.7.0")
        and pandas is not None
        and pandas >= Version("2.0.0")
        and dask is not None
        and dask < Version("2023.2.0")
    )


def needs_lightgbm_sklearn_compat(lightgbm, sklearn):
    return lightgbm is not None and lightgbm < Version("4.6.0") and sklearn is not None and sklearn >= Version("1.8.0")


def needs_seaborn_pandas_compat(seaborn, pandas):
    return (
        seaborn is not None
        and Version("0.11.0") <= seaborn < Version("0.12.2")
        and pandas is not None
        and pandas >= Version("2.0.0")
    )
```

- [ ] **Step 4: 写适配动作失败测试**

```python
def test_prepare_dependency_uses_only_explicit_version_matrix(monkeypatch):
    import hscredit._compat as compat

    installed = []
    monkeypatch.setattr(compat, "installed_version", lambda name, distribution_name=None: {
        "lightgbm": Version("3.3.5"), "pandas": Version("2.3.3"),
        "dask": Version("2022.7.0"), "seaborn": Version("0.11.2"),
        "sklearn": Version("1.0.2"),
    }.get(name))
    monkeypatch.setattr(compat, "_install_pandas_string_methods_alias", lambda: installed.append("strings"))
    monkeypatch.setattr(compat, "_install_pandas_inf_option_alias", lambda: installed.append("inf_option"))

    compat.prepare_dependency("lightgbm")
    compat.prepare_dependency("seaborn")

    assert installed == ["strings", "inf_option"]
```

另写不命中上下界的版本组合，断言两个安装函数都不被调用；直接调用两个安装函数两次，验证
真实别名/配置项注册幂等。这样测试不依赖 `import hscredit` 已经初始化过的全局状态，也能在不同
CI 依赖版本上稳定执行。

- [ ] **Step 5: 运行适配动作测试确认失败**

Run: `pytest tests/test_dependency_compat.py -v`

Expected: version predicates pass; adapter tests fail because preparation functions do not exist.

- [ ] **Step 6: 实现幂等适配器和运行时入口**

```python
def _install_pandas_string_methods_alias() -> None:
    import pandas as pd
    from pandas.core.strings.accessor import StringMethods
    pd.core.strings.StringMethods = StringMethods


def _install_pandas_inf_option_alias() -> None:
    from pandas._config.config import _registered_options, register_option
    if "mode.use_inf_as_null" not in _registered_options:
        register_option("mode.use_inf_as_null", False, doc="Compatibility option for seaborn<0.12.2")


def prepare_runtime_compatibility() -> None:
    prepare_dependency("lightgbm")
    prepare_dependency("seaborn")
```

`prepare_dependency()` 必须先读取版本，再只在对应谓词为真时调用适配器。`hscredit/__init__.py`
在其他包级导入前调用 `prepare_runtime_compatibility()`。

- [ ] **Step 7: 运行并通过兼容层测试**

Run: `pytest tests/test_dependency_compat.py -v`

Expected: PASS，且测试源码不存在 `try`、`except`、`inspect.signature` 兼容分支依赖。

- [ ] **Step 8: 提交兼容注册中心**

```powershell
git add hscredit/_compat.py hscredit/__init__.py tests/test_dependency_compat.py
git commit -m "fix: add explicit dependency compatibility matrix"
```

### Task 2: LightGBM 统一公开接口与版本化 fit

**Files:**
- Modify: `hscredit/core/models/boosting/lightgbm_model.py`
- Modify: `hscredit/core/models/boosting/__init__.py`
- Modify: `hscredit/core/models/tuning/__init__.py`
- Modify: `hscredit/core/models/__init__.py`
- Modify: `tests/test_dependency_compat.py`
- Test: `tests/test_models/test_risk_models.py`
- Test: `tests/test_encoding/test_gbm_encoder_missing.py`
- Test: `tests/test_feature_selection/test_selector_model_compat.py`
- Test: `tests/test_models/test_tuning.py`

**Interfaces:**
- Consumes: Task 1 `prepare_dependency()`、`installed_version()`、`needs_lightgbm_sklearn_compat()`。
- Produces: `install_lightgbm_sklearn_compat(lightgbm_module, lightgbm_version, sklearn_version) -> None`。
- Preserves: `LightGBMRiskModel.fit(...) -> self` and all lazy public names.

- [ ] **Step 1: 写真实异常和版本分支失败测试**

```python
def test_boosting_lazy_loader_preserves_real_import_error(monkeypatch):
    import importlib
    import hscredit.core.models.boosting as boosting

    marker = RuntimeError("真实依赖错误")
    monkeypatch.setattr(importlib, "import_module", lambda *args, **kwargs: (_ for _ in ()).throw(marker))
    with pytest.raises(RuntimeError, match="真实依赖错误"):
        boosting.__getattr__("LightGBMRiskModel")
    assert "LightGBMRiskModel" not in boosting.__dict__


def test_lightgbm_fit_strategy_is_version_based():
    from hscredit.core.models.boosting.lightgbm_model import _lightgbm_fit_api
    assert _lightgbm_fit_api(Version("3.3.5")) == "legacy"
    assert _lightgbm_fit_api(Version("4.0.0")) == "callbacks"
```

- [ ] **Step 2: 运行测试确认当前异常被错误转换且策略函数不存在**

Run: `pytest tests/test_dependency_compat.py -v`

Expected: FAIL with `AttributeError` instead of the marker, and missing `_lightgbm_fit_api`.

- [ ] **Step 3: 实现版本化 LightGBM 导入和 sklearn 适配**

在导入 LightGBM 前调用 `prepare_dependency("lightgbm")`。使用 `find_spec("lightgbm")` 判断是否
安装；已安装则直接导入，导入过程的真实异常自然传播。加载后按明确版本调用
`install_lightgbm_sklearn_compat()`。

适配函数仅在 `lightgbm < 4.6.0` 且 `sklearn >= 1.8.0` 时包装
`lightgbm.compat` 和 `lightgbm.sklearn` 的 `_LGBMCheckXY`、`_LGBMCheckArray`，包装器只把
`force_all_finite` 重命名为 `ensure_all_finite`；版本不命中时原样返回。

- [ ] **Step 4: 删除异常驱动的懒加载降级**

```python
def __getattr__(name):
    module_name = _LAZY_SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
```

`tuning.__getattr__` 使用同一语义；`core.models.__getattr__` 将
`getattr(importlib.import_module(...), name, None)` 改为不带默认值的 `getattr(...)`。

- [ ] **Step 5: 用版本分支重写 fit 参数**

```python
def _lightgbm_fit_api(version: Version) -> str:
    return "legacy" if version < Version("4.0.0") else "callbacks"
```

- legacy：有验证集和早停时传 `early_stopping_rounds`、`verbose`；无早停时仍只传旧版支持的
  `verbose`。
- callbacks：有验证集和早停时传 `callbacks=[lgb.early_stopping(...), lgb.log_evaluation(...)]`；
  永不向 `fit()` 传 `verbose` 或 `early_stopping_rounds`。
- 删除整个 `try/except TypeError` 重试块。

- [ ] **Step 6: 运行当前 7 个 LightGBM 回归**

Run:

```powershell
pytest tests/test_encoding/test_gbm_encoder_missing.py::TestGBMEncoderMissing::test_lightgbm_with_missing tests/test_feature_selection/test_selector_model_compat.py::TestGetFeatureImportances::test_lgbm_classifier tests/test_feature_selection/test_selector_model_compat.py::TestFeatureImportanceSelectorCompat::test_with_lgbm tests/test_feature_selection/test_selector_model_compat.py::TestNullImportanceSelectorCompat::test_with_lgbm tests/test_models/test_risk_models.py::TestLightGBMRiskModel::test_fit_predict tests/test_models/test_risk_models.py::TestLightGBMRiskModel::test_early_stopping tests/test_models/test_tuning.py::test_lightgbm_tuner_samples_num_leaves_after_max_depth_constraint -v
```

Expected: 7 passed; `LightGBMRiskModel` is a class, not `None`.

- [ ] **Step 7: 提交 LightGBM 和懒加载修复**

```powershell
git add hscredit/core/models/boosting/lightgbm_model.py hscredit/core/models/boosting/__init__.py hscredit/core/models/tuning/__init__.py hscredit/core/models/__init__.py tests/test_dependency_compat.py
git commit -m "fix: adapt LightGBM by explicit versions"
```

### Task 3: Seaborn/Pandas 版本兼容与无穷值语义

**Files:**
- Modify: `hscredit/_lazy.py`
- Modify: `hscredit/core/viz/binning_plots.py`
- Modify: `tests/test_visualization/test_hist_plot_theme_color.py`

**Interfaces:**
- Consumes: Task 1 `prepare_dependency("seaborn")`、`needs_seaborn_pandas_compat()`。
- Produces: `normalize_seaborn_inf(values)` internal helper preserving Series index or ndarray order.

- [ ] **Step 1: 写无穷值和 LazyModule 失败测试**

```python
def test_hist_plot_treats_infinite_values_as_missing_for_old_seaborn():
    score = pd.Series([0.1, np.inf, 0.4, -np.inf, 0.8], index=[5, 4, 3, 2, 1])
    fig = hist_plot(score, bins=3, kde=False)
    heights = sum(float(p.get_height()) for p in fig.axes[0].patches)
    assert heights == pytest.approx(1.0)
    plt.close(fig)
```

在 `tests/test_dependency_compat.py` 中替换 `prepare_dependency`，断言
`LazyModule("seaborn")._load()` 在 `importlib.import_module("seaborn")` 前先调用准备函数。

- [ ] **Step 2: 运行测试确认当前 Pandas OptionError**

Run: `pytest tests/test_visualization/test_hist_plot_theme_color.py tests/test_dependency_compat.py -v`

Expected: existing hist test and new inf test fail at `mode.use_inf_as_null`，LazyModule 顺序测试失败。

- [ ] **Step 3: 接入 LazyModule 与版本化数据规范化**

`LazyModule._load()` 在 `importlib.import_module()` 前调用 `prepare_dependency(import_name)`。
`normalize_seaborn_inf()` 只在 Seaborn/Pandas 矩阵命中时执行：Series 用
`.replace([np.inf, -np.inf], np.nan)`，其他数组型输入使用 `np.asarray` 和 `np.where`，不排序、
不重置 Series index。`hist_plot` 在构建 `hist_kwargs` 前调用该函数。

- [ ] **Step 4: 运行并通过绘图测试**

Run: `pytest tests/test_visualization/test_hist_plot_theme_color.py tests/test_dependency_compat.py -v`

Expected: PASS，无 `OptionError`，输入顺序和 Series index 不变。

- [ ] **Step 5: 提交 Seaborn 适配**

```powershell
git add hscredit/_lazy.py hscredit/core/viz/binning_plots.py tests/test_visualization/test_hist_plot_theme_color.py tests/test_dependency_compat.py
git commit -m "fix: adapt seaborn to pandas by version"
```

### Task 4: Rule.save 使用当前 Excel 核心

**Files:**
- Modify: `hscredit/core/rules/rule.py`
- Create: `tests/test_rules/test_rule_save.py`

**Interfaces:**
- Consumes: `hscredit.excel.ExcelWriter`、`hscredit.excel.dataframe2excel`。
- Preserves: `Rule.save(...) -> ExcelWriter`。

- [ ] **Step 1: 写路径和 writer 输入失败测试**

```python
def test_rule_save_writes_path_and_returns_writer(tmp_path):
    report = pd.DataFrame({"规则名称": ["成年"], "命中率": [0.25]})
    output = tmp_path / "rule.xlsx"
    writer = Rule.save(report, output, sheet_name="规则报告", excel_params={"percent_cols": ["命中率"]})
    assert isinstance(writer, ExcelWriter)
    assert output.exists()
    workbook = load_workbook(output)
    assert "规则报告" in workbook.sheetnames
    assert workbook["规则报告"]["B3"].value == "成年"


def test_rule_save_reuses_writer_and_preserves_multiindex(tmp_path):
    columns = pd.MultiIndex.from_tuples([("样本", "数量"), ("指标", "命中率")])
    index = pd.MultiIndex.from_tuples([("规则A", "整体")], names=["规则", "分组"])
    report = pd.DataFrame([[10, 0.2]], columns=columns, index=index)
    writer = ExcelWriter()
    returned = Rule.save(report, writer, sheet_name="多层级", excel_params={"index": True})
    assert returned is writer
    output = tmp_path / "multi.xlsx"
    writer.save(str(output))
    loaded = load_workbook(output)
    assert loaded["多层级"].max_column == 4
```

- [ ] **Step 2: 运行测试确认旧导入路径失败**

Run: `pytest tests/test_rules/test_rule_save.py -v`

Expected: FAIL with `ModuleNotFoundError: hscredit.core.report`.

- [ ] **Step 3: 实现最小 Rule.save 修复**

```python
import os

from ...excel import ExcelWriter, dataframe2excel

params = dict(excel_params or {})
params.pop("data", None)
params.pop("excel_writer", None)
params.pop("sheet_name", None)

if isinstance(excel_writer, ExcelWriter):
    writer = excel_writer
    output_path = None
elif isinstance(excel_writer, (str, os.PathLike)):
    writer_params = dict(params.get("writer_params") or {})
    writer_params.setdefault("theme_color", params.get("theme_color", "2639E9"))
    writer_params.setdefault("mode", params.get("mode", "replace"))
    writer = ExcelWriter(**writer_params)
    output_path = os.fspath(excel_writer)
else:
    raise TypeError("excel_writer 必须是路径或 ExcelWriter 对象")

dataframe2excel(report, writer, sheet_name=sheet_name, **params)
if output_path is not None:
    writer.save(output_path)
return writer
```

同步把 `Rule.save` 的类型标注扩展到 `os.PathLike`，并把文档中的“pandas to_excel 参数”改为
“dataframe2excel 参数”。测试还需断言调用前后的 MultiIndex、列顺序和行顺序完全相同。

- [ ] **Step 4: 增加冲突参数和非法类型测试**

验证 `excel_params` 中伪造的 `sheet_name`/`excel_writer` 不覆盖显式参数，`Rule.save(report, 1)`
抛出中文 `TypeError`。运行：`pytest tests/test_rules/test_rule_save.py -v`，Expected: PASS。

- [ ] **Step 5: 运行规则与 Excel 相关回归**

Run: `pytest tests/test_rules tests/test_report/test_excel_writer.py -v`

Expected: PASS；现有报告与多层级写入行为不回归。

- [ ] **Step 6: 提交 Rule.save 修复**

```powershell
git add hscredit/core/rules/rule.py tests/test_rules/test_rule_save.py
git commit -m "fix: repair Rule.save Excel export"
```

### Task 5: 集成验证与兼容规则审计

**Files:**
- Modify only if verification exposes a scoped regression in files already listed above.

**Interfaces:**
- Consumes all prior tasks.
- Produces a verified current-environment compatibility baseline.

- [ ] **Step 1: 搜索禁止的兼容模式**

Run:

```powershell
rg -n "inspect\.signature|except TypeError|error_msg|unexpected keyword|getattr\(.*None\)" hscredit/_compat.py hscredit/core/models/boosting/lightgbm_model.py hscredit/core/models/boosting/__init__.py hscredit/core/models/tuning/__init__.py hscredit/core/models/__init__.py
```

Expected: 新实现中没有异常驱动或签名驱动的依赖兼容分支；与业务无关的既有代码需人工确认。

- [ ] **Step 2: 运行定向完整集合**

Run:

```powershell
pytest tests/test_dependency_compat.py tests/test_rules/test_rule_save.py tests/test_visualization/test_hist_plot_theme_color.py tests/test_encoding/test_gbm_encoder_missing.py tests/test_feature_selection/test_selector_model_compat.py tests/test_models/test_risk_models.py tests/test_models/test_tuning.py -v
```

Expected: PASS。

- [ ] **Step 3: 运行快速项目回归**

Run: `pytest tests/ -m "not slow and not integration"`

Expected: 所有收集到的非慢速/非集成测试通过；若存在与工作树其他并行改动有关的失败，记录
文件和栈，不修改其代码。

- [ ] **Step 4: 运行静态门禁**

Run:

```powershell
flake8 hscredit/_compat.py hscredit/_lazy.py hscredit/core/models/boosting/lightgbm_model.py hscredit/core/rules/rule.py tests/test_dependency_compat.py tests/test_rules/test_rule_save.py
mypy hscredit/core/models/boosting/lightgbm_model.py hscredit/core/rules/rule.py
```

Expected: 修改范围内无新的 fatal lint/type 错误。

- [ ] **Step 5: 检查工作树归属**

Run: `git status --short` and `git diff --name-only HEAD~4..HEAD`。

Expected: 本任务提交只包含计划列出的文件；用户其他未提交 EDA、pandas extension、gitignore 和
既有设计文档改动仍保持原样。
