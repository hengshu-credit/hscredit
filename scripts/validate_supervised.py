"""监督隔离验证：原生崩溃也保留退出码、日志、JUnit 状态及汇总。

默认运行全部三组（仅排除 slow）；例如：
    python scripts/validate_supervised.py --groups core notebooks native
    python scripts/validate_supervised.py --groups native --include-slow

依赖解释器通过 --python 显式选择；不安装依赖、不修改 pytest 配置。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Optional, Sequence
from uuid import uuid4
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_TESTS = (
    "tests/test_models/test_interpretability_example.py",
    "tests/test_models/test_model_workflow_notebook.py",
    "tests/test_utils/test_validate_examples.py",
    "tests/test_utils/test_pandas_parallel_interrupt.py",
)
NATIVE_TESTS = ("tests/test_models/test_loss_adapter_derivatives.py",)
GROUPS = {
    "core": ("tests",) + tuple(f"--ignore={path}" for path in NOTEBOOK_TESTS + NATIVE_TESTS),
    "notebooks": NOTEBOOK_TESTS,
    "native": NATIVE_TESTS,
}


def execution_groups(requested):
    """Notebook 模块继续隔离，后续 ZeroMQ 崩溃不能吞掉前一模块 JUnit。"""
    stages = []
    for name in requested:
        if name == "notebooks":
            stages.extend(
                (f"notebooks_{Path(path).stem.removeprefix('test_')}", (path,), name) for path in GROUPS[name]
            )
        else:
            stages.append((name, GROUPS[name], name))
    return stages


class _WindowsJob:
    """在恢复新进程主线程前纳入 kill-on-close Job，避免崩溃留下孤儿。

    Win32 契约: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
    """

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel = self.kernel

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("ProcessTime", ctypes.c_longlong),
                ("JobTime", ctypes.c_longlong),
                ("Flags", wintypes.DWORD),
                ("MinimumWorkingSet", ctypes.c_size_t),
                ("MaximumWorkingSet", ctypes.c_size_t),
                ("ActiveProcesses", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("Priority", wintypes.DWORD),
                ("Scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("Basic", BasicLimits),
                ("IoCounters", ctypes.c_ulonglong * 6),
                ("ProcessMemory", ctypes.c_size_t),
                ("JobMemory", ctypes.c_size_t),
                ("PeakProcessMemory", ctypes.c_size_t),
                ("PeakJobMemory", ctypes.c_size_t),
            ]

        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE；不允许 breakaway。
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def attach_and_resume(self, process):
        from ctypes import wintypes

        ctypes, kernel = self.ctypes, self.kernel
        if not kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

        class ThreadEntry(ctypes.Structure):
            _fields_ = [
                ("Size", wintypes.DWORD),
                ("Usage", wintypes.DWORD),
                ("ThreadId", wintypes.DWORD),
                ("ProcessId", wintypes.DWORD),
                ("BasePriority", wintypes.LONG),
                ("DeltaPriority", wintypes.LONG),
                ("Flags", wintypes.DWORD),
            ]

        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Thread32First.argtypes = kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        kernel.Thread32First.restype = kernel.Thread32Next.restype = wintypes.BOOL
        kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenThread.restype = wintypes.HANDLE
        kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel.ResumeThread.restype = wintypes.DWORD
        snapshot = kernel.CreateToolhelp32Snapshot(0x4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.Size = ctypes.sizeof(entry)
            available = kernel.Thread32First(snapshot, ctypes.byref(entry))
            while available:
                if entry.ProcessId == process.pid:
                    thread = kernel.OpenThread(0x2, False, entry.ThreadId)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        previous_count = kernel.ResumeThread(thread)
                        if previous_count == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        kernel.CloseHandle(thread)
                    if previous_count:
                        return
                available = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("无法找到待恢复的验证进程主线程")
        finally:
            kernel.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _stop_process_tree(process):
    """只终止本次创建的进程组，避免超时后遗留 kernel/worker。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def inspect_junit(path: Path):
    """区分未生成、损坏和可读取的 JUnit；不伪造崩溃后的成功 XML。"""
    if not path.is_file():
        return {"available": False, "valid": False, "reason": "子进程未生成 JUnit"}
    try:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        if not suites:
            raise ValueError("缺少 testsuite")
        return {
            "available": True,
            "valid": True,
            **{
                name: sum(int(suite.get(name, "0")) for suite in suites)
                for name in ("tests", "failures", "errors", "skipped")
            },
        }
    except (ET.ParseError, OSError, ValueError) as exc:
        return {"available": True, "valid": False, "reason": str(exc)}


def run_group(name, command, *, output_dir, cwd=PROJECT_ROOT, timeout=1800, environment=None):
    """执行一个隔离组；Python 异常、非零退出、原生崩溃和超时均返回结构化记录。"""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("验证组名只能包含小写字母、数字和下划线")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = directory / f"{name}.stdout.log", directory / f"{name}.stderr.log"
    junit_path = directory / f"{name}.junit.xml"
    started = time.perf_counter()
    record = {
        "group": name,
        "command": list(command),
        "cwd": str(cwd),
        "timeout_seconds": timeout,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "junit_path": str(junit_path),
        "returncode": None,
        "status": "launch_error",
    }
    process = None
    job = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            if os.name == "nt":
                job = _WindowsJob()
            options = (
                {
                    "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | 0x4  # CREATE_SUSPENDED：先加入 Job，再执行用户代码。
                }
                if os.name == "nt"
                else {"start_new_session": True}
            )
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                **options,
            )
            record["pid"] = process.pid
            if job is not None:
                job.attach_and_resume(process)
            try:
                code = process.wait(timeout=timeout)
                record["returncode"] = code
                record["status"] = (
                    "passed" if code == 0 else ("crashed" if code < 0 or code >= 0x40000000 else "failed")
                )
            except subprocess.TimeoutExpired:
                _stop_process_tree(process)
                record.update(status="timeout", returncode=process.returncode)
            except KeyboardInterrupt:
                _stop_process_tree(process)
                record.update(status="interrupted", returncode=process.returncode)
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if process is not None and process.poll() is None:
                _stop_process_tree(process)
        finally:
            if job is not None:
                job.close()  # 包括正常退出、native崩溃、启动失败和超时。
            elif process is not None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    record["duration_seconds"] = time.perf_counter() - started
    record["junit"] = inspect_junit(junit_path)
    # 即使退出码为零也核对 XML，缺失/损坏及记录失败不得伪装绿色。
    if record["status"] == "passed":
        junit = record["junit"]
        if not junit["valid"]:
            record["status"] = "junit_missing_or_invalid"
        elif junit["failures"] or junit["errors"]:
            record["status"] = "failed"
        elif junit["skipped"]:
            record["status"] = "passed_with_skips"
    return record


def write_summary(path, records, requested_groups):
    """每组之后原子写快照，即使后续 native 崩溃也有先前证据。"""
    complete = len(records) == len(requested_groups)
    failed = any(record["status"] not in {"passed", "passed_with_skips"} for record in records)
    skipped = any(record["status"] == "passed_with_skips" for record in records)
    status = "failed" if failed else "running" if not complete else "passed_with_skips" if skipped else "passed"
    summary = {
        "format": "hscredit-supervised-validation",
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "requested_groups": list(requested_groups),
        "complete": complete,
        "status": status,
        "groups": records,
    }
    path = Path(path)
    temporary = path.with_name(f".{uuid4().hex}.{path.name}")
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), help="core notebooks native；也支持逗号分隔")
    parser.add_argument("--include-slow", action="store_true", help="包含 slow 用例；默认其余测试均包含")
    parser.add_argument("--timeout", type=float, default=1800, help="每组最大秒数")
    parser.add_argument("--python", default=sys.executable, help="待验证的 Python 解释器")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tmp_outputs" / "validation")
    options = parser.parse_args(argv)
    requested = list(dict.fromkeys(name for item in options.groups for name in item.split(",")))
    unknown = set(requested) - set(GROUPS)
    if unknown:
        parser.error(f"未知验证组: {sorted(unknown)}")
    if options.timeout <= 0 or not float(options.timeout) < float("inf"):
        parser.error("--timeout 必须为有限正数")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = options.output_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    summary_path = directory / "summary.json"
    records = []
    stages = execution_groups(requested)
    stage_names = [name for name, _, _ in stages]
    write_summary(summary_path, records, stage_names)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONFAULTHANDLER="1", MPLBACKEND="Agg")
    for name, targets, requested_group in stages:
        command = [
            options.python,
            "-m",
            "pytest",
            *targets,
            "--tb=short",
            "-q",
            "--junitxml",
            str(directory / f"{name}.junit.xml"),
        ]
        if not options.include_slow:
            command += ["-m", "not slow"]
        print(f"开始隔离验证组 {name}，日志目录: {directory}", flush=True)
        record = run_group(name, command, output_dir=directory, timeout=options.timeout, environment=environment)
        record["requested_group"] = requested_group
        records.append(record)
        summary = write_summary(summary_path, records, stage_names)
        print(
            f"{name}: {record['status']}，退出码 {record['returncode']}，JUnit有效 {record['junit']['valid']}",
            flush=True,
        )
        if record["status"] == "interrupted":
            print(f"验证已中断，未执行组保留在 requested_groups；汇总: {summary_path}")
            return 130
    print(f"监督验证结果: {summary['status']}；汇总: {summary_path}")
    return 0 if summary["status"] in {"passed", "passed_with_skips"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
