"""可重复地执行 examples 目录中的 notebook 与 Python 示例。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES_DIR = PROJECT_ROOT / "examples"
EXAMPLE_SUFFIXES = {".ipynb", ".py"}


@dataclass
class ExampleResult:
    """单个示例的执行结果。"""

    path: Path
    success: bool
    error: Optional[str] = None
    output: str = ""
    duration: float = 0.0


def discover_examples(examples_dir: Path) -> list[Path]:
    """按稳定顺序发现目录下的 notebook 与 Python 示例。"""
    return sorted(
        (path for path in examples_dir.rglob("*") if path.is_file() and path.suffix.lower() in EXAMPLE_SUFFIXES),
        key=lambda path: path.relative_to(examples_dir).as_posix(),
    )


def _notebook_output(notebook: nbformat.NotebookNode) -> str:
    """收集 notebook 已执行单元格的文本输出。"""
    outputs: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                outputs.append(output.get("text", ""))
    return "".join(outputs)


def _notebook_error(notebook: nbformat.NotebookNode, exception: Exception) -> str:
    """返回包含失败单元格编号的 notebook 错误说明。"""
    for index, cell in enumerate(notebook.cells, start=1):
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                return "单元格 {} 执行失败: {}: {}".format(
                    index,
                    output.get("ename", type(exception).__name__),
                    output.get("evalue", str(exception)),
                )
    return "notebook 执行失败: {}".format(exception)


def execute_notebook(path: Path, timeout: int) -> ExampleResult:
    """在 notebook 所在目录隔离执行 notebook，且不写回源文件。"""
    started_at = time.monotonic()
    notebook: Optional[nbformat.NotebookNode] = None
    try:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=timeout,
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
    except CellTimeoutError as exception:
        return ExampleResult(
            path=path,
            success=False,
            error="Notebook 执行超时: {}".format(exception),
            output=_notebook_output(notebook),
            duration=time.monotonic() - started_at,
        )
    except CellExecutionError as exception:
        return ExampleResult(
            path=path,
            success=False,
            error=_notebook_error(notebook, exception),
            output=_notebook_output(notebook),
            duration=time.monotonic() - started_at,
        )
    except Exception as exception:
        return ExampleResult(
            path=path,
            success=False,
            error="Notebook 执行失败: {}".format(exception),
            output=_notebook_output(notebook) if notebook is not None else "",
            duration=time.monotonic() - started_at,
        )
    return ExampleResult(
        path=path,
        success=True,
        output=_notebook_output(notebook),
        duration=time.monotonic() - started_at,
    )


def _text_output(value: object) -> str:
    """将 subprocess 输出统一为文本。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def execute_python(path: Path, timeout: int) -> ExampleResult:
    """在脚本所在目录隔离执行 Python 示例。"""
    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=path.parent,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exception:
        output = _text_output(exception.stdout)
        error_output = _text_output(exception.stderr)
        return ExampleResult(
            path=path,
            success=False,
            error="Python 示例执行超时（{} 秒）{}".format(
                timeout,
                ":\n" + error_output if error_output else "",
            ),
            output=output,
            duration=time.monotonic() - started_at,
        )

    output = completed.stdout
    if completed.returncode == 0:
        return ExampleResult(
            path=path,
            success=True,
            output=output,
            duration=time.monotonic() - started_at,
        )
    return ExampleResult(
        path=path,
        success=False,
        error="Python 示例退出码 {}:\n{}".format(completed.returncode, completed.stderr),
        output=output,
        duration=time.monotonic() - started_at,
    )


def _filter_examples(paths: Iterable[Path], pattern: str) -> list[Path]:
    """按 glob 模式过滤已发现的示例。"""
    return [path for path in paths if path.match(pattern)]


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 HSCredit 示例并汇总结果")
    parser.add_argument("--pattern", default="*", help="要执行的示例 glob 模式")
    parser.add_argument("--timeout", type=int, default=300, help="单个示例的超时时间（秒）")
    parser.add_argument("--fail-fast", action="store_true", help="首个失败后停止执行")
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.timeout <= 0:
        parser.error("--timeout 必须为正整数")
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
    """运行示例执行器命令行入口，并返回进程退出码。"""
    arguments = _parse_args(argv)
    examples_dir = arguments.examples_dir
    if not examples_dir.is_dir():
        print("示例目录不存在: {}".format(examples_dir))
        return 2

    paths = _filter_examples(discover_examples(examples_dir), arguments.pattern)
    if not paths:
        print("未找到匹配的示例: {}".format(arguments.pattern))
        return 0

    results: list[ExampleResult] = []
    for path in paths:
        print("执行: {}".format(path.relative_to(examples_dir)))
        if path.suffix.lower() == ".ipynb":
            result = execute_notebook(path, arguments.timeout)
        else:
            result = execute_python(path, arguments.timeout)
        results.append(result)

        if result.success:
            print("  成功 ({:.1f} 秒)".format(result.duration))
        else:
            print("  失败: {}".format(result.error))
            if arguments.fail_fast:
                break

    failures = [result for result in results if not result.success]
    print("共执行 {} 个示例，成功 {} 个，失败 {} 个".format(len(results), len(results) - len(failures), len(failures)))
    if failures:
        print("失败示例汇总:")
        for result in failures:
            print("- {}: {}".format(result.path.relative_to(examples_dir), result.error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
