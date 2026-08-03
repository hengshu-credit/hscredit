# hscredit Setuptools Install Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hscredit build and install with setuptools 77 through the latest release without relying on `pkg_resources`.

**Architecture:** Keep the PEP 517 backend limited to setuptools, declare `packaging` as the runtime dependency used by `hscredit._compat`, and validate the metadata contract with fast repository tests. Add CI and local build checks for a setuptools version that still contains `pkg_resources` and versions where it has been removed.

**Tech Stack:** Python 3.9–3.14, setuptools build backend, PEP 517, pytest, GitHub Actions, wheel metadata.

## Global Constraints

- Support setuptools 77 through the current latest release.
- Python 3.9 uses the latest setuptools compatible with Python 3.9; Python 3.10+ may use the current latest release.
- hscredit source, `setup.py`, and build configuration must not import or require `pkg_resources`.
- Runtime compatibility uses explicit `packaging.version` comparisons, not `try/except` version branching.
- Do not add setuptools, wheel, build, or pkg_resources to runtime dependencies.
- Do not set a global `setuptools<82` upper bound to work around third-party source packages.
- Do not change the Pandas or Pandas-related dependency policy in this plan.
- Preserve all unrelated EDA parallel-summary working-tree changes.

---

### Task 1: Lock and update the build metadata contract

**Files:**
- Create: `tests/test_build_metadata.py`
- Modify: `pyproject.toml:1-11`
- Modify: `pyproject.toml:46-110`
- Modify: `requirements.txt:8-21`

**Interfaces:**
- Consumes: PEP 621 metadata in `pyproject.toml` and the direct `packaging.version.Version` import in `hscredit._compat`.
- Produces: A build backend requiring only `setuptools>=77`, SPDX license metadata, and an unversioned direct runtime requirement named `packaging`.

- [ ] **Step 1: Write failing metadata contract tests**

```python
"""构建和安装元数据契约测试。"""

import sys
from pathlib import Path

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_build_backend_has_no_pkg_resources_era_helper_dependencies():
    build_system = _pyproject()["build-system"]
    assert build_system == {
        "requires": ["setuptools>=77"],
        "build-backend": "setuptools.build_meta",
    }


def test_project_uses_spdx_license_string():
    assert _pyproject()["project"]["license"] == "MIT"


def test_packaging_is_an_unversioned_direct_runtime_dependency():
    requirements = [Requirement(item) for item in _pyproject()["project"]["dependencies"]]
    packaging = [item for item in requirements if item.name.lower() == "packaging"]
    assert len(packaging) == 1
    assert str(packaging[0].specifier) == ""


def test_build_tools_are_not_runtime_dependencies():
    names = {
        Requirement(item).name.lower()
        for item in _pyproject()["project"]["dependencies"]
    }
    assert names.isdisjoint({"setuptools", "wheel", "build", "pkg-resources"})


def test_source_does_not_reference_pkg_resources():
    checked = [PROJECT_ROOT / "setup.py", PROJECT_ROOT / "hscredit"]
    matches = []
    for path in checked:
        files = [path] if path.is_file() else path.rglob("*.py")
        for file in files:
            if "pkg_resources" in file.read_text(encoding="utf-8"):
                matches.append(str(file.relative_to(PROJECT_ROOT)))
    assert matches == []
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```powershell
pytest tests/test_build_metadata.py -v
```

Expected: the build-system test fails on `cython`, `wheel`, and `build`; the license test receives a table; and the direct `packaging` dependency test finds no entry.

- [ ] **Step 3: Apply the minimal metadata changes**

Update `pyproject.toml` to contain:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
license = "MIT"
dependencies = [
    # existing runtime requirements remain in their current order
    "packaging",
]
```

Remove `packaging>=20.0` from the `boost` extra because the package is now a core direct dependency. Add an unversioned `packaging` line to the core section of `requirements.txt`.
Because the metadata test must run on Python 3.9 and 3.10, add
`tomli; python_version < "3.11"` to the `dev` extra and the development section of
`requirements.txt`. Python 3.11+ continues to use the standard-library `tomllib`.

- [ ] **Step 4: Run metadata and dependency compatibility tests**

Run:

```powershell
pytest tests/test_build_metadata.py tests/test_dependency_compat.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the metadata contract**

```powershell
git add pyproject.toml requirements.txt tests/test_build_metadata.py
git commit -m "build: support setuptools without pkg_resources"
```

---

### Task 2: Add installation validation, documentation, and CI coverage

**Files:**
- Modify: `tests/test_build_metadata.py`
- Modify: `scripts/validate_environment.py:12-24`
- Modify: `docs/installation.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: The build metadata contract from Task 1.
- Produces: Environment validation for `packaging`, user-facing setuptools guidance, and CI jobs covering setuptools 77.0.3, 82.0.1, and the latest release.

- [ ] **Step 1: Add failing validation and documentation contract tests**

Append to `tests/test_build_metadata.py`:

```python
import runpy


def test_environment_validator_checks_packaging():
    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "validate_environment.py"))
    assert "packaging" in namespace["REQUIRED_MODULES"]


def test_installation_docs_explain_setuptools_compatibility():
    text = (PROJECT_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    assert "setuptools 77" in text
    assert "pkg_resources" in text
    assert "setuptools<82" in text


def test_ci_covers_setuptools_with_and_without_pkg_resources():
    text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'setuptools-version: "77.0.3"' in text
    assert 'setuptools-version: "82.0.1"' in text
    assert 'setuptools-version: "latest"' in text
```

- [ ] **Step 2: Run the new tests and verify they fail for missing coverage**

Run:

```powershell
pytest tests/test_build_metadata.py -v
```

Expected: the three new tests fail because the validator, installation guide, and CI matrix do not yet contain the required entries.

- [ ] **Step 3: Extend environment validation and installation documentation**

Add `"packaging"` to `REQUIRED_MODULES` in `scripts/validate_environment.py`.

Add an “安装工具兼容” section to `docs/installation.md` stating:

- Standard users should use PEP 517 installation and do not need `pkg_resources`.
- hscredit supports setuptools 77 through the latest release.
- setuptools 82+ removes `pkg_resources`, but hscredit does not depend on it.
- Do not globally pin `setuptools<82`; use the traceback to identify a third-party source package and prefer a compatible wheel or release.

- [ ] **Step 4: Add the setuptools build compatibility CI matrix**

Add this `build-compat` job to `.github/workflows/ci.yml`:

```yaml
build-compat:
  name: 构建兼容 (Python ${{ matrix.python-version }}, setuptools ${{ matrix.setuptools-version }})
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      include:
        - python-version: "3.9"
          setuptools-version: "77.0.3"
          pkg_resources: "present"
        - python-version: "3.9"
          setuptools-version: "82.0.1"
          pkg_resources: "absent"
        - python-version: "3.14"
          setuptools-version: "latest"
          pkg_resources: "absent"
  steps:
    - uses: actions/checkout@v4

    - name: 安装 Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: pip

    - name: 安装指定构建后端
      shell: bash
      run: |
        python -m pip install --upgrade pip build
        if [ "${{ matrix.setuptools-version }}" = "latest" ]; then
          python -m pip install --upgrade setuptools
        else
          python -m pip install "setuptools==${{ matrix.setuptools-version }}"
        fi

    - name: 验证 pkg_resources 状态
      env:
        EXPECTED_PKG_RESOURCES: ${{ matrix.pkg_resources }}
      run: |
        python - <<'PY'
        import importlib.util
        import os
        import setuptools

        found = importlib.util.find_spec("pkg_resources") is not None
        expected = os.environ["EXPECTED_PKG_RESOURCES"] == "present"
        print(f"setuptools={setuptools.__version__}, pkg_resources={found}")
        if found != expected:
            raise SystemExit("pkg_resources 状态与测试矩阵不一致")
        PY

    - name: 验证无隔离和 PEP 517 构建
      run: |
        python -m build --wheel --no-isolation --outdir dist-build-compat
        python -m pip wheel --no-deps --use-pep517 --wheel-dir dist-pep517 .
```

- [ ] **Step 5: Run the metadata tests and environment validator**

Run:

```powershell
pytest tests/test_build_metadata.py -v
python scripts/validate_environment.py
```

Expected: tests pass and the validator reports a successful environment.

- [ ] **Step 6: Commit installation validation and CI coverage**

```powershell
git add tests/test_build_metadata.py scripts/validate_environment.py docs/installation.md .github/workflows/ci.yml
git commit -m "ci: verify supported setuptools versions"
```

---

### Task 3: Verify real isolated builds and regressions

**Files:**
- Verify only; modify Task 1 or Task 2 files only if a failing check identifies a defect in the implementation.

**Interfaces:**
- Consumes: The metadata, validation, documentation, and CI changes from Tasks 1–2.
- Produces: Build evidence for setuptools 77/82/latest and a regression-tested worktree.

- [ ] **Step 1: Build with setuptools 77.0.3 on Python 3.9**

Create a temporary Python 3.9 virtual environment, install `build` and `setuptools==77.0.3`, confirm `pkg_resources` exists, and run `python -m build --wheel --no-isolation`.

Expected: one `hscredit-*.whl` is produced without a `project.license` deprecation warning.

- [ ] **Step 2: Build with setuptools 82.0.1 on Python 3.9**

Create a second temporary Python 3.9 environment, install `build` and `setuptools==82.0.1`, confirm `pkg_resources` is absent, and run both no-isolation and PEP 517 wheel builds.

Expected: both builds produce wheels without importing `pkg_resources`.

- [ ] **Step 3: Build with the latest setuptools on Python 3.14**

Create a Python 3.14 temporary environment, upgrade setuptools, confirm the installed version and absence of `pkg_resources`, and run both build paths.

Expected: both builds succeed and the wheel is tagged `py3-none-any`.

- [ ] **Step 4: Inspect built wheel metadata**

Open the wheel `METADATA` file and assert:

```text
Requires-Dist: packaging
```

is present, while `Requires-Dist: setuptools`, `Requires-Dist: wheel`, `Requires-Dist: build`, and `Requires-Dist: pkg_resources` are absent.

- [ ] **Step 5: Run focused and full regression suites**

Run:

```powershell
pytest tests/test_build_metadata.py tests/test_dependency_compat.py -v
pytest tests -q --disable-warnings
```

Expected: focused and full suites pass. Any failures in pre-existing unrelated EDA working-tree files must be reported separately and must not be overwritten.

- [ ] **Step 6: Review the final diff and working-tree scope**

Run:

```powershell
git diff --check
git status --short
git log -3 --oneline
```

Expected: no whitespace errors; setuptools work is committed separately; unrelated EDA files remain exactly as they were before implementation.
