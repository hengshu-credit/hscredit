# Hsbin and Hsreport Skills Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Every production behavior follows a RED-GREEN-REFACTOR cycle.

**Goal:** Deliver independently installable `hsbin` and `hsreport` Agent Skills, backed by a shared hscredit runtime that executes the approved binning, visualization, and Excel-report operations from file or in-process object inputs.

**Architecture:** Each root `skills/<name>` directory is a self-contained discovery and bootstrap package. Thin standard-library launchers create or reuse an isolated environment, install the controlled hscredit extras, and invoke `hscredit.skills_runtime`; the runtime validates a versioned request, resolves data and object references, dispatches only registered operations, stages artifacts, and returns a compact JSON manifest.

**Tech Stack:** Python 3.9-3.14, hscredit, pandas, scikit-learn, matplotlib, openpyxl, joblib, jsonschema, venv, pytest.

**Spec:** `docs/skills-framework-design.md`

## Global Constraints

- Phase 1 fully implements only `skills/hsbin` and `skills/hsreport`. The other six planned names exist only as directories containing `.gitkeep` and remain roadmap entries.
- Preserve all existing unrelated working-tree changes.
- Do not run `git add`, `git commit`, `git push`, publish a Skill, or modify a remote service.
- All user-facing messages, error messages, table labels, and manifests use Chinese where a natural-language value is required.
- Explicit hscredit API parameters keep their existing priority over inferred and default values.
- Do not expose arbitrary imports, Python evaluation, shell commands, package names, package indexes, or Git URLs from a request.
- Default dependency mode is an isolated venv. Current-environment installation requires `mode="current"` and `install_missing=true`.
- `object_ref` never crosses into a newly created interpreter; missing dependencies in that case produce `OBJECT_REF_REQUIRES_CURRENT_ENV`.
- CLI input is file-based. In-process Python calls may provide an object registry.
- Existing report, binning, visualization, serialization, and parallel-execution APIs remain the calculation source of truth.
- No production function or method is written before its failing test has been run and its expected failure confirmed.

---

## File Structure

### Root Skill packages

- Create `skills/CATALOG.md`: implemented Skill and operation catalog.
- Create `skills/ROADMAP.md`: planned `hscredit`, `hsmodel`, `hsrule`, `hsselect`, `hsviz`, and `hsexcel` Skills.
- Create `skills/install.py`: repository fallback installer for the two implemented Skills.
- Create `skills/hscredit/.gitkeep`.
- Create `skills/hsbin/SKILL.md`.
- Create `skills/hsbin/runtime.json`.
- Create `skills/hsbin/agents/openai.yaml`.
- Create `skills/hsbin/references/operations.md`.
- Create `skills/hsbin/schemas/request.schema.json`.
- Create `skills/hsbin/scripts/run.py`.
- Create the corresponding six files under `skills/hsreport/`.
- Create `skills/hsmodel/.gitkeep`.
- Create `skills/hsrule/.gitkeep`.
- Create `skills/hsselect/.gitkeep`.
- Create `skills/hsviz/.gitkeep`.
- Create `skills/hsexcel/.gitkeep`.

### Packaged runtime

- Create `hscredit/skills_runtime/__init__.py`: public `execute_skill` and object-registry API.
- Create `hscredit/skills_runtime/__main__.py`: JSON-file CLI.
- Create `hscredit/skills_runtime/contracts.py`: request, context, artifact, and result data structures.
- Create `hscredit/skills_runtime/errors.py`: stable error codes and `SkillExecutionError`.
- Create `hscredit/skills_runtime/registry.py`: operation specs and allowlisted dispatch.
- Create `hscredit/skills_runtime/io.py`: file and trusted-artifact resolution.
- Create `hscredit/skills_runtime/objects.py`: in-process object registry.
- Create `hscredit/skills_runtime/artifacts.py`: staging, publish, preview, and cleanup.
- Create `hscredit/skills_runtime/dependencies.py`: controlled extras and environment-key logic.
- Create `hscredit/skills_runtime/bootstrap.py`: venv creation, installation, and re-execution.
- Create `hscredit/skills_runtime/operations/__init__.py`.
- Create `hscredit/skills_runtime/operations/binning.py`.
- Create `hscredit/skills_runtime/operations/visualization.py`.
- Create `hscredit/skills_runtime/operations/reports.py`.

### Tests and packaging

- Create `tests/test_skills/__init__.py`.
- Create `tests/test_skills/conftest.py`.
- Create `tests/test_skills/test_contracts.py`.
- Create `tests/test_skills/test_inputs_and_artifacts.py`.
- Create `tests/test_skills/test_dependency_bootstrap.py`.
- Create `tests/test_skills/test_hsbin_tables.py`.
- Create `tests/test_skills/test_hsbin_binners.py`.
- Create `tests/test_skills/test_hsbin_visualization.py`.
- Create `tests/test_skills/test_hsreport.py`.
- Create `tests/test_skills/test_skill_installation.py`.
- Modify `pyproject.toml`: add the `skills` optional dependency group, include it in `all`, and add `pyarrow` to the base dependencies.
- Modify `docs/skills-framework-design.md`: mark Phase 1 implementation status after verification.

---

### Task 1: Versioned request contracts and allowlisted operation registry

**Files:**

- Create: `hscredit/skills_runtime/__init__.py`
- Create: `hscredit/skills_runtime/contracts.py`
- Create: `hscredit/skills_runtime/errors.py`
- Create: `hscredit/skills_runtime/registry.py`
- Create: `tests/test_skills/__init__.py`
- Create: `tests/test_skills/test_contracts.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Produces `SkillExecutionError(code, message, field=None, cause=None)`.
- Produces immutable `OperationSpec(skill, name, handler, extras=())`.
- Produces `OperationRegistry.register(spec)`, `get(skill, operation)`, and `list_operations(skill)`.
- Produces `validate_request(skill, request, registry) -> SkillRequest` using the common envelope and the selected operation's internal Schema.
- Produces `execute_skill(skill, request, objects=None) -> dict` as the future public entrypoint; at this task it validates and dispatches a registered test operation.

- [ ] **Step 1: Write failing contract tests**

```python
import pytest

from hscredit.skills_runtime import execute_skill
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.skills_runtime.registry import OperationRegistry, OperationSpec


def test_registry_rejects_an_operation_from_another_skill():
    registry = OperationRegistry()
    registry.register(OperationSpec("hsbin", "feature_bin_stats", lambda context: {"status": "success"}))

    with pytest.raises(SkillExecutionError) as exc_info:
        registry.get("hsreport", "feature_bin_stats")

    assert exc_info.value.code == "OPERATION_NOT_ALLOWED"
    assert "feature_bin_stats" in str(exc_info.value)


def test_execute_skill_rejects_an_unknown_top_level_field(tmp_path):
    request = {
        "version": "1",
        "operation": "feature_bin_stats",
        "inputs": {},
        "parameters": {},
        "output": {"directory": str(tmp_path), "name": "stats", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
        "unexpected": True,
    }

    with pytest.raises(SkillExecutionError) as exc_info:
        execute_skill("hsbin", request)

    assert exc_info.value.code == "SCHEMA_INVALID"
    assert exc_info.value.field == "unexpected"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
pytest tests/test_skills/test_contracts.py -v
```

Expected: collection fails because `hscredit.skills_runtime` does not exist.

- [ ] **Step 3: Implement the minimal contracts, errors, and registry**

Use dataclasses with JSON-compatible fields. `SkillExecutionError.__str__` must render the Chinese message while preserving `__cause__` when raised from another exception. Registry lookup keys are exact `(skill, operation)` tuples; no import-path fallback is permitted.

Add the Parquet engine to the base dependencies and add the Skills optional dependency:

```toml
dependencies = [
    # Existing base dependencies...
    "pyarrow>=12",
]

[project.optional-dependencies]
skills = ["jsonschema>=4.18"]
```

Add `skills` to the existing `all` extra.

- [ ] **Step 4: Implement schema validation and the public execution seam**

The request dataclass must normalize these defaults:

```python
output = {"directory": ".", "name": operation, "overwrite": False}
environment = {"mode": "isolated", "reuse": True, "install_missing": True}
```

Reject versions other than `"1"`, unknown top-level fields, unsupported environment modes, and unknown operations before resolving any inputs.

- [ ] **Step 5: Run the focused tests and verify GREEN**

```powershell
pytest tests/test_skills/test_contracts.py -v
git diff --check -- pyproject.toml hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 6: Review checkpoint without committing**

Inspect `git diff -- pyproject.toml hscredit/skills_runtime tests/test_skills/test_contracts.py`. Confirm no existing source file other than `pyproject.toml` changed.

---

### Task 2: File inputs, object references, and transactional artifacts

**Files:**

- Create: `hscredit/skills_runtime/io.py`
- Create: `hscredit/skills_runtime/objects.py`
- Create: `hscredit/skills_runtime/artifacts.py`
- Create: `tests/test_skills/conftest.py`
- Create: `tests/test_skills/test_inputs_and_artifacts.py`
- Modify: `hscredit/skills_runtime/contracts.py`
- Modify: `hscredit/skills_runtime/__init__.py`

**Interfaces:**

- Produces `ObjectRegistry.register(ref, value)`, `resolve(ref)`, and `contains(ref)`.
- Produces `InputResolver.resolve(spec)` for `file` and `object_ref` sources.
- Produces `ArtifactTransaction(output_spec)` with `stage_path`, `publish`, and context-manager cleanup.
- Produces `summarize_dataframe(frame, preview_rows=10)`.
- Consumes `SkillRequest` and `SkillExecutionError` from Task 1.

- [ ] **Step 1: Write failing real-I/O tests**

```python
import pandas as pd
import pytest

from hscredit.skills_runtime.artifacts import ArtifactTransaction
from hscredit.skills_runtime.errors import SkillExecutionError
from hscredit.skills_runtime.io import InputResolver
from hscredit.skills_runtime.objects import ObjectRegistry


def test_input_resolver_reads_the_requested_excel_sheet(tmp_path):
    path = tmp_path / "sample.xlsx"
    expected = pd.DataFrame({"年龄": [22, 35], "target": [0, 1]})
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"忽略": [1]}).to_excel(writer, sheet_name="其他", index=False)
        expected.to_excel(writer, sheet_name="建模样本", index=False)

    result = InputResolver().resolve({"kind": "file", "path": str(path), "sheet_name": "建模样本"})

    pd.testing.assert_frame_equal(result, expected)


def test_untrusted_joblib_input_is_rejected(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not-loaded")

    with pytest.raises(SkillExecutionError) as exc_info:
        InputResolver().resolve({"kind": "file", "path": str(path)})

    assert exc_info.value.code == "ARTIFACT_UNTRUSTED"


def test_failed_artifact_transaction_does_not_publish_a_partial_file(tmp_path):
    final_path = tmp_path / "report.xlsx"

    with pytest.raises(RuntimeError):
        with ArtifactTransaction({"directory": str(tmp_path), "name": "report", "overwrite": False}) as transaction:
            transaction.stage_path("report.xlsx").write_bytes(b"partial")
            raise RuntimeError("render failed")

    assert not final_path.exists()
```

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_inputs_and_artifacts.py -v
```

Expected: imports fail for the three missing runtime modules.

- [ ] **Step 3: Implement file and object resolution**

Support `.csv`, `.xlsx`, and `.parquet`. Normalize paths with `Path.resolve()`. Map missing files to `INPUT_NOT_FOUND`, unsupported suffixes to `INPUT_FORMAT_UNSUPPORTED`, missing Excel sheets to `INPUT_NOT_FOUND`, and unknown object refs to `OBJECT_REF_NOT_FOUND`.

Artifact loads through `hscredit.utils.load_pickle` only when `trusted` is exactly `true`. Object refs require a caller-provided registry.

- [ ] **Step 4: Implement transactional artifacts and compact previews**

Create a temporary directory under the resolved output directory. On `publish`, use `Path.replace` only for exact staged files after checking `overwrite`; otherwise raise `ARTIFACT_EXISTS`. On exception, remove only the transaction's validated temporary directory.

DataFrame summaries contain literal `rows`, stringified `columns`, and at most ten records. MultiIndex columns are represented as lists of strings.

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
pytest tests/test_skills/test_inputs_and_artifacts.py -v
git diff --check -- hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 6: Review checkpoint without committing**

Confirm the cleanup target is always a child of the output directory and no recursive deletion can reach the workspace root.

---

### Task 3: Isolated dependency bootstrap and self-contained Skill launchers

**Files:**

- Create: `hscredit/skills_runtime/dependencies.py`
- Create: `hscredit/skills_runtime/bootstrap.py`
- Create: `hscredit/skills_runtime/__main__.py`
- Create: `tests/test_skills/test_dependency_bootstrap.py`
- Create: `skills/hsbin/runtime.json`
- Create: `skills/hsbin/scripts/run.py`
- Create: `skills/hsreport/runtime.json`
- Create: `skills/hsreport/scripts/run.py`

**Interfaces:**

- Produces `resolve_required_extras(skill, operation, request)`, returning an immutable tuple of allowed extra names.
- Produces `resolve_cache_root()` and `environment_key(source, extras, python_version)`.
- Produces `ensure_environment(runtime_config, request) -> Path`.
- Produces `run_in_environment(python, skill, request_path) -> int`.
- CLI: `python -m hscredit.skills_runtime --skill hsbin --request request.json`.

- [ ] **Step 1: Write failing authorization and environment tests**

```python
import pytest

from hscredit.skills_runtime.bootstrap import plan_environment
from hscredit.skills_runtime.errors import SkillExecutionError


def test_current_environment_install_requires_explicit_permission():
    request = {"environment": {"mode": "current", "install_missing": False}}

    with pytest.raises(SkillExecutionError) as exc_info:
        plan_environment("hsbin", "feature_bin_stats", request, missing_extras=("skills",))

    assert exc_info.value.code == "DEPENDENCY_MISSING"


def test_object_ref_cannot_be_moved_to_an_isolated_interpreter():
    request = {
        "inputs": {"binner": {"kind": "object_ref", "ref": "binner:fitted"}},
        "environment": {"mode": "isolated", "install_missing": True},
    }

    with pytest.raises(SkillExecutionError) as exc_info:
        plan_environment("hsbin", "optimal_binning_transform", request, missing_extras=("skills",))

    assert exc_info.value.code == "OBJECT_REF_REQUIRES_CURRENT_ENV"
```

Add an integration test that creates a tiny local Python package named `hscredit_skill_test_dep`, installs it into a temporary venv through the real bootstrap installer, and imports it with the venv interpreter. The fixture uses a local filesystem path, never an external package index.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_dependency_bootstrap.py -v
```

Expected: bootstrap and dependency modules are missing.

- [ ] **Step 3: Implement environment planning**

The allowlist maps Phase 1 operations to `skills`; Parquet support comes from the base installation. Request data cannot add extras. The environment hash includes Python major/minor, normalized hscredit source, and sorted extras.

Use a user cache path, not the repository. Use the current interpreter's `venv` module and invoke pip as argument arrays with `shell=False`. Sanitize captured output and map nonzero exits to `DEPENDENCY_INSTALL_FAILED`.

- [ ] **Step 4: Implement the standard-library launchers**

Before creating the launchers, initialize `hsbin` and `hsreport` with the Skill Creator initializer so each folder has `SKILL.md` and `agents/openai.yaml`; request only the `references` and `scripts` resource directories. Do not run the initializer again after the folders exist.

Each `run.py` accepts exactly one request JSON path plus an optional `--debug` flag. It reads only enough JSON to choose the environment, then executes the runtime CLI. `runtime.json` declares:

```json
{
  "protocol_version": "1",
  "skill": "hsbin",
  "hscredit": {
    "package": "hscredit",
    "repository": "https://github.com/hengshu-credit/hscredit.git",
    "ref": "main"
  },
  "extras": ["skills"]
}
```

Use `"skill": "hsreport"` in the report file. When the launcher detects the complete source checkout, install the resolved repository root editable; an independently installed Skill uses the controlled repository/ref from `runtime.json`.

- [ ] **Step 5: Run focused tests and both launchers**

```powershell
pytest tests/test_skills/test_dependency_bootstrap.py -v
python skills/hsbin/scripts/run.py --help
python skills/hsreport/scripts/run.py --help
git diff --check -- skills/hsbin skills/hsreport hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 6: Review checkpoint without committing**

Inspect subprocess calls and confirm no request value can become a package name, index URL, Git URL, interpreter path, or shell fragment.

---

### Task 4: hsbin table-analysis operations

**Files:**

- Create: `hscredit/skills_runtime/operations/__init__.py`
- Create: `hscredit/skills_runtime/operations/binning.py`
- Create: `tests/test_skills/test_hsbin_tables.py`
- Modify: `hscredit/skills_runtime/registry.py`
- Modify: `hscredit/skills_runtime/artifacts.py`

**Interfaces:**

- Registers `feature_bin_stats`.
- Registers `benchmark_binning_methods`.
- Registers `feature_binning_summary`.
- Registers `feature_group_binning_summary`.
- Every handler consumes a resolved DataFrame named `data`, passes validated parameters unchanged to the matching hscredit function, and returns styled Excel plus compact manifest data.

- [ ] **Step 1: Add a deterministic real-data fixture**

Create `credit_frame` in `tests/test_skills/conftest.py` with 120 literal deterministic rows derived from `numpy.random.default_rng(42)`. Columns are `score`, `age`, `amount`, `apply_date`, `segment`, `MOB1`, and `target`; assert in the fixture that both target classes exist.

- [ ] **Step 2: Write failing table-operation tests**

```python
from openpyxl import load_workbook

from hscredit.skills_runtime import execute_skill


def test_feature_bin_stats_executes_real_hscredit_and_writes_excel(tmp_path, credit_frame):
    request = {
        "version": "1",
        "operation": "feature_bin_stats",
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": {"feature": "score", "target": "target", "method": "quantile", "max_n_bins": 4, "n_jobs": 1},
        "output": {"directory": str(tmp_path), "name": "score_stats", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})

    assert result["status"] == "success"
    assert result["summary"]["rows"] >= 2
    workbook = load_workbook(result["artifacts"][0]["path"], read_only=True)
    assert workbook.sheetnames == ["分箱统计"]
```

Add table-driven tests for the other three operations. Assert literal artifact sheet names:

- `benchmark_binning_methods`: `方法对比`.
- `feature_binning_summary`: `分箱摘要` plus sanitized feature-method sheets.
- `feature_group_binning_summary`: `分组摘要` plus sanitized feature-method-group sheets.

- [ ] **Step 3: Run and verify RED**

```powershell
pytest tests/test_skills/test_hsbin_tables.py -v
```

Expected: `OPERATION_NOT_ALLOWED` for each unregistered operation.

- [ ] **Step 4: Implement direct adapters and nested-table workbook writing**

Use explicit adapter functions, not reflection. Call the exact existing hscredit functions. Flatten nested result dictionaries only for artifact sheet naming; retain input dictionary order. Sanitize Excel sheet names to 31 characters and de-duplicate collisions deterministically.

Use hscredit `ExcelWriter` and `dataframe2excel` for output. Preserve MultiIndex columns, index meaning, percentage columns, and existing conditional-format defaults.

- [ ] **Step 5: Run focused and existing report tests**

```powershell
pytest tests/test_skills/test_hsbin_tables.py -v
pytest tests/test_report/test_feature_binning_summary.py tests/test_report/test_bin_stats_formatting.py -q
git diff --check -- hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 6: Review checkpoint without committing**

Confirm the adapter does not reimplement bin statistics, target construction, binning precedence, or metric aggregation.

---

### Task 5: OptimalBinning and OptimalBinning2D lifecycle operations

**Files:**

- Create: `tests/test_skills/test_hsbin_binners.py`
- Modify: `hscredit/skills_runtime/operations/binning.py`
- Modify: `hscredit/skills_runtime/io.py`
- Modify: `hscredit/skills_runtime/artifacts.py`

**Interfaces:**

- Registers `optimal_binning_fit`, `optimal_binning_transform`, and `optimal_binning_fit_transform`.
- Registers `optimal_binning_2d_fit` and `optimal_binning_2d_transform`.
- Fit operations return a trusted hscredit artifact plus transformed data and bin-table workbook.
- Transform operations accept `object_ref` or `trusted: true` artifact input.

- [ ] **Step 1: Write failing lifecycle tests**

```python
from hscredit.core.binning import OptimalBinning
from hscredit.skills_runtime import execute_skill


def test_optimal_binning_fit_publishes_a_loadable_artifact(tmp_path, credit_frame):
    request = {
        "version": "1",
        "operation": "optimal_binning_fit",
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": {"features": ["score", "age"], "target": "target", "method": "quantile", "max_n_bins": 4},
        "output": {"directory": str(tmp_path), "name": "binner", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    artifact_path = next(item["path"] for item in result["artifacts"] if item["type"] == "hscredit-artifact")
    loaded = OptimalBinning.load_artifact(artifact_path)

    assert list(loaded.feature_names_in_) == ["score", "age"]
```

Add a two-dimensional test using `features=["score", "age"]`, and a transform test that loads the produced artifact with `trusted: true` and verifies the output row count equals the input row count.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_hsbin_binners.py -v
```

Expected: lifecycle operations are not registered.

- [ ] **Step 3: Implement binner construction and target extraction**

For `target` as a column name, copy the selected feature frame and separate `y`; explicit `y` object input takes priority. Do not infer WOE versus bin-index semantics. Expose the transform `metric` parameter directly and preserve hscredit defaults when omitted.

Save binners with `save_artifact`. Save transformed DataFrames and bin tables as Excel artifacts. For OptimalBinning2D, call its current `fit`, `transform`, and `get_bin_table` APIs without adding a nonexistent `fit_transform` shortcut.

- [ ] **Step 4: Run focused and existing binner tests**

```powershell
pytest tests/test_skills/test_hsbin_binners.py -v
pytest tests/test_binning/test_binning_contracts.py tests/test_binning/test_optimal_binning_2d.py -q
git diff --check -- hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 5: Review checkpoint without committing**

Verify artifacts retain real bin metadata and labels, and that no adapter trains on bin indices while labeling them as WOE.

---

### Task 6: hsbin efficiency analysis and visualization operations

**Files:**

- Create: `hscredit/skills_runtime/operations/visualization.py`
- Create: `tests/test_skills/test_hsbin_visualization.py`
- Modify: `hscredit/skills_runtime/registry.py`
- Modify: `hscredit/skills_runtime/operations/binning.py`
- Modify: `hscredit/skills_runtime/artifacts.py`

**Interfaces:**

- Registers `feature_efficiency_analysis`.
- Registers `bin_plot`, `bin_trend_plot`, `bin_overdues_plot`, and `bin_2d_plot` for `hsbin`.
- Figure handlers save PNG by default and SVG when requested.
- `feature_efficiency_analysis` publishes manual and automatic tables, rules JSON, comparison figure, and zero or more trend figures.

- [ ] **Step 1: Write failing real-render tests**

```python
from PIL import Image

from hscredit.skills_runtime import execute_skill


def test_bin_plot_renders_a_decodable_nonempty_png(tmp_path, credit_frame):
    request = {
        "version": "1",
        "operation": "bin_plot",
        "inputs": {"data": {"kind": "object_ref", "ref": "data:credit"}},
        "parameters": {"feature": "score", "target": "target", "method": "quantile", "n_bins": 4, "anchor": 0.17},
        "output": {"directory": str(tmp_path), "name": "score_bin_plot", "format": "png", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill("hsbin", request, objects={"data:credit": credit_frame})
    image_path = result["artifacts"][0]["path"]

    with Image.open(image_path) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width >= 600
        assert rendered.height >= 350
```

Add real render cases for:

- `bin_trend_plot` with `date_col="apply_date"`.
- `bin_overdues_plot` with `overdue=["MOB1"]` and `dpds=[0]`.
- `bin_2d_plot` with `features=["score", "age"]`.
- `feature_efficiency_analysis` with `date_col="apply_date"`, asserting comparison and trend images plus two table sheets.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_hsbin_visualization.py -v
```

Expected: visualization operations are not registered.

- [ ] **Step 3: Implement explicit figure adapters**

Resolve DataFrame, bin-table, or binner inputs by operation. Pass all allowed plotting parameters unchanged, including explicit `anchor`. Override only output `save` with the transaction's staged path. Save returned figures when the underlying API does not save them. Close only figures created or returned by the operation.

For `feature_efficiency_analysis`, set its `output_dir` and `save` to staged paths, then convert its returned tables and rules to named artifacts. Do not serialize matplotlib figure objects into the JSON response.

- [ ] **Step 4: Run focused visualization tests and inspect rendered images**

```powershell
pytest tests/test_skills/test_hsbin_visualization.py -v
pytest tests/test_visualization/test_adaptive_legend_anchor.py tests/test_visualization/test_bin_plot_layout.py tests/test_visualization/test_bin_plot_modes.py -q
git diff --check -- hscredit/skills_runtime tests/test_skills
```

Open at least one small and one large rendered PNG and verify title, legend, axes, labels, and annotations do not overlap or clip.

- [ ] **Step 5: Review checkpoint without committing**

Confirm `hsbin` and future `hsviz` can reuse the same visualization functions without copying behavior.

---

### Task 7: hsreport complete Excel operations

**Files:**

- Create: `hscredit/skills_runtime/operations/reports.py`
- Create: `tests/test_skills/test_hsreport.py`
- Modify: `hscredit/skills_runtime/registry.py`
- Modify: `hscredit/skills_runtime/artifacts.py`

**Interfaces:**

- Registers `auto_feature_analysis` for `hsreport`.
- Registers `auto_model_report` for `hsreport`.
- Registers `swap_out_report` for `hsreport`.
- All three operations require an `.xlsx` artifact target and execute against staged paths.

- [ ] **Step 1: Write failing real-report tests**

```python
from openpyxl import load_workbook
from sklearn.linear_model import LogisticRegression

from hscredit.skills_runtime import execute_skill


def test_auto_model_report_writes_a_real_workbook(tmp_path, credit_frame):
    X = credit_frame[["score", "age"]]
    y = credit_frame["target"]
    model = LogisticRegression(max_iter=200).fit(X, y)
    request = {
        "version": "1",
        "operation": "auto_model_report",
        "inputs": {
            "model": {"kind": "object_ref", "ref": "model:lr"},
            "train": {"kind": "object_ref", "ref": "data:credit"},
        },
        "parameters": {"datasets": {"训练集": "train"}, "target": "target", "feature_names": ["score", "age"], "with_plots": False, "verbose": False, "n_jobs": 1},
        "output": {"directory": str(tmp_path), "name": "model_report", "overwrite": False},
        "environment": {"mode": "current", "install_missing": False},
    }

    result = execute_skill(
        "hsreport",
        request,
        objects={"model:lr": model, "data:credit": credit_frame},
    )
    workbook = load_workbook(result["artifacts"][0]["path"], read_only=True)

    assert "1-基本信息" in workbook.sheetnames
    assert "2-模型性能" in workbook.sheetnames
```

Add:

- `auto_feature_analysis`: assert the workbook contains `分析报告`, has nonempty dimensions, and reports the final `(end_row, end_col)` in summary metadata.
- `swap_out_report`: pass literal rule strings such as `score < 560` and assert `策略迭代` and `变量分箱` sheets exist.
- A failure case with a missing target column that asserts `HSCREDIT_EXECUTION_FAILED`, preserves `ValueError` as the cause, and publishes no workbook.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_hsreport.py -v
```

Expected: report operations are not registered.

- [ ] **Step 3: Implement report-specific input assembly**

`auto_feature_analysis` resolves one DataFrame and maps the staged path to `excel_writer`. `auto_model_report` resolves a model plus named datasets; the request's dataset mapping refers to keys in `inputs`, not filesystem paths hidden in parameters. `swap_out_report` resolves a DataFrame and passes strings or resolved Rule objects to `rules` while mapping the staged path to `save`.

Reject attempts to pass `excel_writer`, `excel_path`, `save`, or `output_dir` inside `parameters`; output paths come only from the normalized output contract.

- [ ] **Step 4: Preserve report defaults and publish atomically**

Do not change `condition_color`, `theme_color`, `anchor`, parallel settings, or feature-selection defaults. Only fill output-path arguments and values explicitly required by the request Schema. Publish the staged workbook after the hscredit function returns successfully and the file can be opened by openpyxl.

- [ ] **Step 5: Run focused and existing report tests**

```powershell
pytest tests/test_skills/test_hsreport.py -v
pytest tests/test_report/test_feature_report_layout.py tests/test_report/test_model_report.py tests/test_report/test_rule_strategy.py -q
git diff --check -- hscredit/skills_runtime tests/test_skills
```

- [ ] **Step 6: Review checkpoint without committing**

Confirm each handler calls exactly one top-level report operation and never retries a failed report.

---

### Task 8: Skill instructions, Schemas, catalogs, and fallback installer

**Files:**

- Modify: `skills/hsbin/SKILL.md`
- Modify: `skills/hsbin/agents/openai.yaml`
- Create: `skills/hsbin/references/operations.md`
- Create: `skills/hsbin/schemas/request.schema.json`
- Modify: `skills/hsreport/SKILL.md`
- Modify: `skills/hsreport/agents/openai.yaml`
- Create: `skills/hsreport/references/operations.md`
- Create: `skills/hsreport/schemas/request.schema.json`
- Create: `skills/CATALOG.md`
- Create: `skills/ROADMAP.md`
- Create: `skills/install.py`
- Create: `skills/hscredit/.gitkeep`
- Create: `skills/hsmodel/.gitkeep`
- Create: `skills/hsrule/.gitkeep`
- Create: `skills/hsselect/.gitkeep`
- Create: `skills/hsviz/.gitkeep`
- Create: `skills/hsexcel/.gitkeep`
- Create: `tests/test_skills/test_skill_installation.py`

**Interfaces:**

- `$hsbin` discovers only the eleven approved binning/statistics/plot capabilities.
- `$hsreport` discovers only the three approved complete-report capabilities.
- Installer supports `--suite hscredit`, `--skills`, and `--target-dir`.
- Existing destination directories fail without deletion or overwrite.

- [ ] **Step 1: Write failing installer behavior tests**

```python
import subprocess
import sys


def test_repository_installer_installs_only_the_requested_skill(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "skills/install.py",
            "--skills",
            "hsbin",
            "--target-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "hsbin" / "SKILL.md").is_file()
    assert not (tmp_path / "hsreport").exists()
```

Add tests that `--suite hscredit` installs exactly `hsbin` and `hsreport`, and a second install fails while preserving a marker file placed inside the existing target.

Add a structure test asserting that the six planned directories exist, contain exactly `.gitkeep`, have no `SKILL.md`, and are absent from the installer's allowed Skill names.

- [ ] **Step 2: Run and verify RED**

```powershell
pytest tests/test_skills/test_skill_installation.py -v
```

Expected: `skills/install.py` does not exist.

- [ ] **Step 3: Initialize and write the two Skills**

Replace every initializer placeholder. Keep `SKILL.md` concise and route operation detail to `references/operations.md`. Include the dual input contract, isolated-environment default, trusted-artifact warning, output manifest, and one complete request example per operation family.

`agents/openai.yaml` values:

```yaml
interface:
  display_name: "HSCredit 分箱分析"
  short_description: "调用 hscredit 完成分箱统计、评估和分箱可视化"
  default_prompt: "Use $hsbin to analyze this credit-risk feature and save the requested artifacts."
```

Use corresponding report copy:

```yaml
interface:
  display_name: "HSCredit 分析报告"
  short_description: "生成 hscredit 特征、模型和策略 Excel 分析报告"
  default_prompt: "Use $hsreport to assemble the data and generate the requested Excel report."
```

- [ ] **Step 4: Write operation-specific JSON Schemas**

Use `oneOf` branches keyed by a literal `operation`. Set `additionalProperties: false` at the envelope and operation-parameter level except explicitly documented hscredit parameter maps. Reserve output-path parameters so an operation cannot bypass artifact staging.

- [ ] **Step 5: Implement the fallback installer**

Resolve the repository `skills` directory from `__file__`; accept only the literal implemented names. Copy with `shutil.copytree` after resolving and verifying source and destination. Do not remove or merge an existing directory. Print installed absolute paths and state that the Skill is available on the Agent's next turn.

- [ ] **Step 6: Write catalogs without overstating availability**

`CATALOG.md` lists only `hsbin` and `hsreport` as implemented. `ROADMAP.md` lists the remaining six names and the deferred APIs, with status `规划中`. Create their directories with only `.gitkeep`; do not add `SKILL.md`, metadata, Schema, or scripts.

- [ ] **Step 7: Run installer and Skill structure validation**

```powershell
pytest tests/test_skills/test_skill_installation.py -v
python C:/Users/18306/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hsbin
python C:/Users/18306/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/hsreport
git diff --check -- skills tests/test_skills
```

- [ ] **Step 8: Review checkpoint without committing**

Install each Skill separately into two temporary target directories and run its launcher with a controlled current-environment request. Confirm neither installed directory reads a sibling Skill path.

---

### Task 9: End-to-end verification and design status update

**Files:**

- Modify: `docs/skills-framework-design.md`
- Verify all files from Tasks 1-8.

**Interfaces:**

- Produces a verified Phase 1 status statement with exact commands and results.
- Does not change the public hscredit top-level import surface unless a failing integration test proves it necessary.

- [ ] **Step 1: Run both Skills through their real script entrypoints**

Create request JSON files under the system temporary directory, not in the repository. Execute them from the deterministic smoke directory:

```powershell
$skillSmokeDir = Join-Path ([System.IO.Path]::GetTempPath()) "hscredit-skills-smoke"
python skills/hsbin/scripts/run.py (Join-Path $skillSmokeDir "hsbin-request.json")
python skills/hsreport/scripts/run.py (Join-Path $skillSmokeDir "hsreport-request.json")
```

Use `environment.mode="current"` for the focused smoke run so it does not mutate the existing environment. Separately run the local-package isolated bootstrap test from Task 3.

- [ ] **Step 2: Run the complete Phase 1 Skill suite**

```powershell
pytest tests/test_skills -q
```

Expected: all Phase 1 tests pass with no unexpected warnings.

- [ ] **Step 3: Run related existing suites**

```powershell
pytest tests/test_report tests/test_binning tests/test_visualization -q
```

Record the actual pass, skip, warning, and duration summary. Do not report this as the repository's full test suite.

- [ ] **Step 4: Verify packaging and repository hygiene**

```powershell
python -m build
python -m compileall hscredit/skills_runtime skills/hsbin/scripts skills/hsreport/scripts
git diff --check
git status --short
```

Verify the built wheel contains `hscredit/skills_runtime` and imports it in a temporary clean venv. Root Skill directories are distributed through GitHub/Skill installation, not treated as a top-level Python package.

- [ ] **Step 5: Inspect generated artifacts**

Open the three report workbooks with openpyxl and inspect at least one workbook visually. Decode all generated images and visually inspect representative `bin_plot`, `bin_trend_plot`, and `bin_2d_plot` outputs for clipping or overlap.

- [ ] **Step 6: Update the design status**

Only after all required verification passes, change the design-document status to state that Phase 1 `hsbin` and `hsreport` are implemented and list the verification evidence. If an environment gate remains, record it explicitly instead of marking the phase complete.

- [ ] **Step 7: Final diff review without committing**

Review `git diff -- skills hscredit/skills_runtime tests/test_skills pyproject.toml docs/skills-framework-design.md`. Confirm unrelated pre-existing changes remain untouched and no generated report, image, wheel, venv, cache, or credential file is added.

---

## Execution Handoff

This plan deliberately excludes commits because the working tree already contains unrelated user changes and no commit authorization was given.

Execution choices:

1. **Inline execution:** implement Tasks 1-9 in this session with review checkpoints.
2. **Subagent-driven execution:** dispatch isolated implementation tasks with two-stage review; this requires an explicit user request for subagents.
