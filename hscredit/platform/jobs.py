"""任务队列与可终止的独立计算进程，主进程独占任务状态的发布权。"""

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import psutil

from .storage import now, write_json

ACTIVE_STATUSES = {"排队中", "运行中", "停止中"}


def terminate_process_tree(process):
    """终止当前任务及其派生进程，等待进程实际退出后才返回。"""
    try:
        root = psutil.Process(process.pid)
        if getattr(process, "_hscredit_created", root.create_time()) != root.create_time():
            process.wait()
            return
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        process.wait()
        return
    # 先暂停计算进程，防止退出期间继续启动子任务或发布结果。
    for item in [root, *children]:
        try:
            item.suspend()
        except psutil.NoSuchProcess:
            pass
    for item in children[::-1] + [root]:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children + [root], timeout=5)
    if any(item.is_running() and item.status() != psutil.STATUS_ZOMBIE for item in alive):
        raise RuntimeError("计算进程尚未退出，请稍后再次停止")
    process.wait(timeout=5)


class JobManager:
    """串行调度独立任务进程，支持取消排队任务及中止运行任务。

    **参数**
    :param workspace: 当前平台工作区。

    **属性**
    :ivar thread: 负责调度与状态发布的后台线程。

    **参考样例**
    >>> manager = JobManager(workspace)  # doctest: +SKIP
    >>> manager.start()  # doctest: +SKIP
    """

    def __init__(self, workspace):
        self.workspace = workspace
        self.condition = threading.Condition(workspace.lock)
        self.queue = deque()
        self.stop_requested = set()
        self.closed = False
        self.current = None
        self.process = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="hscredit-job-manager", daemon=True)
        self.thread.start()

    def enqueue(self, identifier):
        with self.condition:
            if self.closed:
                raise RuntimeError("服务正在关闭，不能提交新任务")
            self.queue.append(identifier)
            self.condition.notify_all()

    def stop(self, identifier):
        with self.condition:
            metadata = self.workspace.get("jobs", identifier)
            if metadata["status"] not in ACTIVE_STATUSES:
                return metadata
            self.stop_requested.add(identifier)
            if metadata["status"] == "排队中":
                self.queue = deque(item for item in self.queue if item != identifier)
                self.stop_requested.discard(identifier)
                return self.workspace.update("jobs", identifier, status="已停止", message="已取消排队，可重新执行", artifacts=[], finished_at=now())
            self.condition.notify_all()
            return self.workspace.update("jobs", identifier, status="停止中", message="正在终止计算进程，请稍候", artifacts=[])

    def close(self):
        with self.condition:
            self.closed = True
            for identifier in list(self.queue):
                self.stop(identifier)
            if self.current:
                self.stop(self.current)
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=20)
            if self.thread.is_alive():
                raise RuntimeError("后台计算仍在关闭，请稍后检查任务状态")

    def _command(self, identifier):
        return [sys.executable, "-m", "hscredit.platform._worker", str(self.workspace.root), identifier]

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or bool(self.queue))
                if self.closed:
                    return
                identifier = self.queue.popleft()
                if self.workspace.get("jobs", identifier)["status"] != "排队中":
                    continue
                self.current = identifier
            try:
                self._run(identifier)
            except Exception as exc:
                if self.process is not None and self.process.poll() is None:
                    terminate_process_tree(self.process)
                with self.condition:
                    self.workspace.update("jobs", identifier, status="失败", message=f"任务执行异常：{exc}", artifacts=[], finished_at=now())
            finally:
                with self.condition:
                    if self.process is not None and self.process.stdin is not None:
                        self.process.stdin.close()
                    self.current = None
                    self.process = None
                    self.stop_requested.discard(identifier)

    def _run(self, identifier):
        directory = self.workspace.path("jobs", identifier)
        environment = os.environ.copy()
        environment.update(HSCREDIT_LIGHT_IMPORT="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        environment["HSCREDIT_PLATFORM_PARENT_PID"] = str(os.getpid())
        environment["HSCREDIT_PLATFORM_PARENT_CREATED"] = str(psutil.Process().create_time())
        package_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        with (directory / "worker.log").open("wb") as log:
            with self.condition:
                if identifier in self.stop_requested or self.closed or self.workspace.get("jobs", identifier)["status"] != "排队中":
                    return
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
                self.process = subprocess.Popen(self._command(identifier), env=environment, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **options)
                process = self.process
                try:
                    process._hscredit_created = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    pass
                self.workspace.update("jobs", identifier, status="运行中", started_at=now(), worker_pid=process.pid, message="正在启动计算")
            last_progress = None
            while process.poll() is None:
                with self.condition:
                    stopping = identifier in self.stop_requested or self.closed
                if stopping:
                    try:
                        terminate_process_tree(process)
                    except (psutil.Error, RuntimeError, subprocess.TimeoutExpired) as exc:
                        self.workspace.update("jobs", identifier, status="停止中", message=f"仍在停止计算：{exc}")
                        with self.condition:
                            self.condition.wait(timeout=0.5)
                        continue
                    break
                progress_path = directory / "progress.json"
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    progress = None
                if progress and progress != last_progress:
                    with self.condition:
                        if identifier not in self.stop_requested:
                            self.workspace.update("jobs", identifier, progress=progress["progress"], message=progress["message"])
                    last_progress = progress
                with self.condition:
                    self.condition.wait(timeout=0.1)
            with self.condition:
                if identifier in self.stop_requested or self.closed:
                    # 结果发布与停止共享同一锁；停止后永远不再发布成功状态。
                    (directory / "result.json").unlink(missing_ok=True)
                    (directory / "outcome.json").unlink(missing_ok=True)
                    self.workspace.update("jobs", identifier, status="已停止", message="计算已停止，可按原配置重新执行", artifacts=[], finished_at=now())
                    return
                outcome_path = directory / "outcome.json"
                if process.returncode != 0 or not outcome_path.exists():
                    raise RuntimeError(f"计算进程异常退出（退出码 {process.returncode}），请检查任务日志")
                outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
                if outcome["ok"]:
                    result = outcome["result"]
                    write_json(directory / "result.json", result)
                    self.workspace.update("jobs", identifier, status="成功", progress=100, message="任务完成", finished_at=now(), artifacts=result["artifacts"])
                else:
                    if outcome.get("result") is not None:
                        write_json(directory / "result.json", outcome["result"])
                    self.workspace.update("jobs", identifier, status="失败", message=outcome["message"], artifacts=[], finished_at=now())
