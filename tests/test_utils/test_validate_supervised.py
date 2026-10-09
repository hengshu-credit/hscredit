"""监督运行器的退出/崩溃/超时证据不依赖本机可选原生库。"""

import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location(
    "validate_supervised", Path(__file__).resolve().parents[2] / "scripts" / "validate_supervised.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _command(body):
    return [sys.executable, "-c", body]


def test_success_requires_valid_junit_and_logs_survive(tmp_path):
    junit = tmp_path / "ok.junit.xml"
    body = f'from pathlib import Path; print(\'child stdout\'); Path({str(junit)!r}).write_text(\'<testsuite tests="1" failures="0" errors="0" skipped="0"/>\')'
    record = runner.run_group("ok", _command(body), output_dir=tmp_path)
    assert record["status"] == "passed"
    assert record["returncode"] == 0
    assert record["junit"]["tests"] == 1
    assert "child stdout" in Path(record["stdout"]).read_text()


@pytest.mark.parametrize("exit_code", [1, 7])
def test_abrupt_exit_has_evidence_without_junit(tmp_path, exit_code):
    body = f"import os,sys; print('before exit', flush=True); print('native failure', file=sys.stderr, flush=True); os._exit({exit_code})"
    record = runner.run_group("failed", _command(body), output_dir=tmp_path)
    assert record["status"] == "failed"
    assert record["returncode"] == exit_code
    assert record["junit"]["available"] is False
    assert "before exit" in Path(record["stdout"]).read_text()
    assert "native failure" in Path(record["stderr"]).read_text()


def test_zero_exit_without_junit_is_not_green(tmp_path):
    record = runner.run_group("nojunit", _command("pass"), output_dir=tmp_path)
    assert record["status"] == "junit_missing_or_invalid"


def test_timeout_is_recorded_and_process_is_stopped(tmp_path):
    record = runner.run_group("timeout", _command("import time; time.sleep(30)"), output_dir=tmp_path, timeout=0.1)
    assert record["status"] == "timeout"
    assert record["returncode"] is not None
    assert record["duration_seconds"] < 10


def test_missing_interpreter_does_not_prevent_summary(tmp_path):
    record = runner.run_group("launch", [str(tmp_path / "no-python")], output_dir=tmp_path)
    assert record["status"] == "launch_error"
    summary = runner.write_summary(tmp_path / "summary.json", [record], ["launch"])
    assert summary["status"] == "failed"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["groups"][0]["error"]


def test_skips_are_explicit_not_plain_pass(tmp_path):
    junit = tmp_path / "skip.junit.xml"
    body = f'from pathlib import Path; Path({str(junit)!r}).write_text(\'<testsuite tests="1" skipped="1"/>\')'
    record = runner.run_group("skip", _command(body), output_dir=tmp_path)
    assert record["status"] == "passed_with_skips"
    assert runner.write_summary(tmp_path / "summary.json", [record], ["skip"])["status"] == "passed_with_skips"


def test_default_groups_cover_tests_without_duplicate_dangerous_files():
    assert runner.GROUPS["core"][0] == "tests"
    for path in runner.NOTEBOOK_TESTS + runner.NATIVE_TESTS:
        assert f"--ignore={path}" in runner.GROUPS["core"]
        assert (runner.PROJECT_ROOT / path).is_file()


def test_summary_remains_failed_when_later_group_passes(tmp_path):
    records = [
        {"group": "native", "status": "crashed", "returncode": -11},
        {"group": "core", "status": "passed", "returncode": 0},
    ]
    summary = runner.write_summary(tmp_path / "summary.json", records, ["native", "core"])
    assert summary["complete"] is True
    assert summary["status"] == "failed"


def test_notebook_modules_are_separate_supervised_processes():
    stages = runner.execution_groups(["core", "notebooks", "native"])
    notebooks = [stage for stage in stages if stage[2] == "notebooks"]
    assert len(notebooks) == len(runner.NOTEBOOK_TESTS)
    assert all(len(stage[1]) == 1 for stage in notebooks)
    assert len({stage[0] for stage in stages}) == len(stages)


def test_crashing_parent_does_not_leave_child_process(tmp_path):
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "child.pid"
    body = (
        "import os,subprocess,sys; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).write_text(str(child.pid)); os._exit(7)"
    )
    record = runner.run_group("orphan", _command(body), output_dir=tmp_path)
    assert record["status"] == "failed"
    pid = int(marker.read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.02)
    else:
        assert not psutil.pid_exists(pid)
