"""示例执行器的契约测试。"""

import subprocess
import sys
from pathlib import Path

import nbformat

from scripts.validate_examples import discover_examples, execute_notebook, execute_python, main


def _write_notebook(path: Path, *sources: str) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source) for source in sources],
    )
    nbformat.write(notebook, path)


def test_discover_examples_returns_sorted_notebooks_and_python_files(tmp_path: Path) -> None:
    """纳入生成产物、遗漏真实嵌套示例或返回不稳定顺序时应失败。"""
    (tmp_path / "nested").mkdir()
    (tmp_path / "20_script.py").write_text("print('script')", encoding="utf-8")
    _write_notebook(tmp_path / "10_notebook.ipynb", "print('notebook')")
    _write_notebook(tmp_path / "nested" / "30_nested.ipynb", "print('nested')")
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")
    generated_files = [
        tmp_path / "model_report" / "scorecard_deploy.py",
        tmp_path / "model_report_demo" / "report.ipynb",
        tmp_path / "tree_viz_output" / "tree.py",
        tmp_path / "__pycache__" / "cached.py",
        tmp_path / ".ipynb_checkpoints" / "draft.ipynb",
    ]
    for path in generated_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".ipynb":
            _write_notebook(path, "print('generated')")
        else:
            path.write_text("print('generated')", encoding="utf-8")

    examples = discover_examples(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in examples] == [
        "10_notebook.ipynb",
        "20_script.py",
        "nested/30_nested.ipynb",
    ]


def test_execute_notebook_succeeds_in_its_own_directory_without_rewriting_source(tmp_path: Path) -> None:
    """错误的 notebook 工作目录或写回源文件时应失败。"""
    path = tmp_path / "success.ipynb"
    _write_notebook(path, "from pathlib import Path\nPath('created.txt').write_text('ok')")
    source_before = path.read_text(encoding="utf-8")

    result = execute_notebook(path, timeout=10)

    assert result.success is True
    assert result.error is None
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "ok"
    assert path.read_text(encoding="utf-8") == source_before


def test_main_executes_nested_notebook_from_examples_root(tmp_path: Path) -> None:
    """嵌套 notebook 在自身子目录而非 examples 根目录执行时应失败。"""
    examples_dir = tmp_path / "examples"
    notebook_dir = examples_dir / "nested"
    notebook_dir.mkdir(parents=True)
    path = notebook_dir / "run.ipynb"
    _write_notebook(path, "from pathlib import Path\nPath('root-marker.txt').write_text('ok')")

    exit_code = main(
        ["--examples-dir", str(examples_dir), "--pattern", "nested/*.ipynb", "--timeout", "10"]
    )

    assert exit_code == 0
    assert (examples_dir / "root-marker.txt").read_text(encoding="utf-8") == "ok"
    assert not (notebook_dir / "root-marker.txt").exists()


def test_execute_notebook_derives_examples_root_from_nested_path(tmp_path: Path) -> None:
    """直接调用嵌套 notebook 时未推导 examples 根目录应失败。"""
    examples_dir = tmp_path / "examples"
    notebook_dir = examples_dir / "nested"
    notebook_dir.mkdir(parents=True)
    path = notebook_dir / "run.ipynb"
    _write_notebook(path, "from pathlib import Path\nPath('derived-root-marker.txt').write_text('ok')")

    result = execute_notebook(path, timeout=10)

    assert result.success is True
    assert (examples_dir / "derived-root-marker.txt").read_text(encoding="utf-8") == "ok"
    assert not (notebook_dir / "derived-root-marker.txt").exists()


def test_execute_notebook_reports_failing_cell_number_and_original_exception(tmp_path: Path) -> None:
    """吞掉 notebook 原始异常或错误定位到错误单元格时应失败。"""
    path = tmp_path / "failure.ipynb"
    _write_notebook(path, "answer = 42", "raise ValueError('notebook boom')")
    source_before = path.read_text(encoding="utf-8")

    result = execute_notebook(path, timeout=10)

    assert result.success is False
    assert "单元格 2" in result.error
    assert "ValueError: notebook boom" in result.error
    assert path.read_text(encoding="utf-8") == source_before


def test_execute_notebook_returns_failure_for_invalid_notebook(tmp_path: Path) -> None:
    """读取损坏 notebook 时执行器抛异常而不是返回失败结果时应失败。"""
    path = tmp_path / "invalid.ipynb"
    path.write_text("{invalid json", encoding="utf-8")

    result = execute_notebook(path, timeout=10)

    assert result.success is False
    assert "Notebook 执行失败" in result.error


def test_execute_python_returns_output_and_uses_script_directory(tmp_path: Path) -> None:
    """Python 示例未在自身目录执行或输出丢失时应失败。"""
    path = tmp_path / "success.py"
    path.write_text(
        "from pathlib import Path\nPath('created.txt').write_text('ok')\nprint('中文输出')\n",
        encoding="utf-8",
    )

    result = execute_python(path, timeout=10)

    assert result.success is True
    assert result.error is None
    assert result.output == "中文输出\n"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "ok"


def test_main_executes_python_from_relative_examples_directory(tmp_path: Path, monkeypatch) -> None:
    """相对 examples 目录使脚本路径重复拼接时应失败。"""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    script = examples_dir / "run.py"
    script.write_text("from pathlib import Path\nPath('marker.txt').write_text('ok')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--examples-dir", "examples", "--pattern", "*.py", "--timeout", "10"])

    assert exit_code == 0
    assert (examples_dir / "marker.txt").read_text(encoding="utf-8") == "ok"


def test_execute_python_reports_failure_and_timeout(tmp_path: Path) -> None:
    """Python 错误或超时未转换为失败结果时应失败。"""
    failure = tmp_path / "failure.py"
    failure.write_text("raise RuntimeError('python boom')", encoding="utf-8")
    timeout = tmp_path / "timeout.py"
    timeout.write_text("import time\ntime.sleep(2)", encoding="utf-8")

    failed_result = execute_python(failure, timeout=10)
    timeout_result = execute_python(timeout, timeout=1)

    assert failed_result.success is False
    assert "RuntimeError: python boom" in failed_result.error
    assert timeout_result.success is False
    assert "超时" in timeout_result.error


def test_main_filters_examples_summarizes_failures_and_returns_nonzero(tmp_path: Path, capsys, monkeypatch) -> None:
    """过滤遗漏、失败未汇总或失败退出码为零时应失败。"""
    success = tmp_path / "keep.py"
    success.write_text("print('ok')", encoding="utf-8")
    failure = tmp_path / "keep_failure.py"
    failure.write_text("raise RuntimeError('cli boom')", encoding="utf-8")
    skipped = tmp_path / "skip.py"
    skipped.write_text("raise RuntimeError('should not run')", encoding="utf-8")
    monkeypatch.setattr("scripts.validate_examples.DEFAULT_EXAMPLES_DIR", tmp_path)

    exit_code = main(["--pattern", "keep*.py", "--timeout", "10"])

    captured = capsys.readouterr().out
    assert exit_code == 1
    assert "共执行 2 个示例，成功 1 个，失败 1 个" in captured
    assert "keep_failure.py" in captured
    assert "skip.py" not in captured


def test_cli_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    """fail-fast 仍继续执行后续示例时应失败。"""
    first = tmp_path / "01_failure.py"
    first.write_text("raise RuntimeError('first failure')", encoding="utf-8")
    second = tmp_path / "02_should_not_run.py"
    second.write_text("from pathlib import Path\nPath('ran.txt').write_text('ran')", encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_examples.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--examples-dir",
            str(tmp_path),
            "--pattern",
            "*.py",
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    assert completed.stdout is not None, completed.stderr
    assert "01_failure.py" in completed.stdout
    assert "02_should_not_run.py" not in completed.stdout
    assert not (tmp_path / "ran.txt").exists()
