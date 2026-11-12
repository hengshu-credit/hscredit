"""独立进程内执行平台计算；只写结果信封，不直接改写任务状态。"""

import os
import sys
import threading
import time

import psutil

from .storage import Workspace, records, write_json


def run(root, identifier):
    workspace = Workspace(root)
    metadata = workspace.get("jobs", identifier)
    config = metadata["config"]
    directory = workspace.path("jobs", identifier)

    def progress(percent, message):
        try:
            write_json(directory / "progress.json", {"progress": percent, "message": message})
        except PermissionError:
            # Windows 在读取进度文件的极短窗口内可能拒绝替换；下次进度更新重试。
            pass

    try:
        from .analysis import strategy, third_party, train
        from .exploration import explore_dataset
        from hscredit.core.viz.utils import _threaded_agg_rendering

        with _threaded_agg_rendering():
            if config["kind"] == "explore":
                progress(10, "正在计算数据探索结果")
                keys = ("feature", "other", "bins", "targets", "positive_labels", "freq", "label_mode", "target", "target_order", "overdue", "dpds", "overdue_operator", "del_grey", "handle_missing_target", "psi_dataset", "library_params")
                axes = config.get("features") if config.get("workflow") else None
                eda = explore_dataset(workspace, config["dataset_id"], revision=config.get("dataset_revision"), axes=axes, group=config.get("group"), **{k: config[k] for k in keys if k in config})
                combined = eda.pop("interaction", None)
                result = {"artifacts": [], "tables": {"字段概况": eda["summary"]}, "exploration": eda}
                if combined is not None:
                    result["interaction"] = combined
            elif config["kind"] == "interaction":
                from .interactions import interactions

                data = workspace.read_dataset(config["dataset_id"], config["dataset_revision"])
                kinds = {c["name"]: c["kind"] for c in workspace.dataset_version(config["dataset_id"], config["dataset_revision"])["columns"]}
                result = {"artifacts": [], "interaction": interactions(data, config["features"], config["target"], config["bins"], config["freq"], kinds)}
            elif config.get("workflow") and config["kind"] in {"strategy", "ruleset", "explain"}:
                from .workflows import strategy_workflow, ruleset_workflow
                from .experiments import sample_explanation

                result = {"strategy": strategy_workflow, "ruleset": ruleset_workflow, "explain": sample_explanation}[config["kind"]](workspace, config, directory, progress)
            else:
                data = workspace.read_dataset(config["dataset_id"], config.get("dataset_revision"))
                if config.get("workflow") and config["kind"] == "model":
                    from .experiments import experiment

                    result = experiment(workspace, config, directory, progress)
                else:
                    result = {"strategy": strategy, "model": train, "third_party": third_party}[config["kind"]](data, config, directory, progress)
        if not any(file.endswith(".xlsx") for file in result["artifacts"]):
            from .job_reports import write_result_workbook

            report = write_result_workbook(directory, result)
            result["artifacts"].append(report.name)
        outcome = {"ok": True, "result": result}
    except Exception as exc:
        report = getattr(exc, "result", None)
        partial = {"tables": {"章节状态": records(report.status_table())}, "artifacts": [], "complete": False} if report is not None else None
        outcome = {"ok": False, "message": f"任务失败：{exc}", "result": partial}
    write_json(directory / "outcome.json", outcome)


if __name__ == "__main__":

    def watch_parent():
        # 不阻塞读取标准输入，避免 Windows 数值库加载期间的 CRT 锁冲突。
        pid = int(os.environ["HSCREDIT_PLATFORM_PARENT_PID"])
        created = float(os.environ["HSCREDIT_PLATFORM_PARENT_CREATED"])
        try:
            parent = psutil.Process(pid)
            while parent.is_running() and parent.create_time() == created and parent.status() != psutil.STATUS_ZOMBIE:
                time.sleep(0.5)
        except psutil.NoSuchProcess:
            pass
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        os._exit(75)

    if "HSCREDIT_PLATFORM_PARENT_PID" in os.environ:
        threading.Thread(target=watch_parent, daemon=True).start()
    run(*sys.argv[1:])
