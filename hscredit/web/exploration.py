"""按数据版本与标签配置持久化字段概况，供接口和独立任务进程复用。"""

import hashlib
import json
from contextlib import suppress
from uuid import uuid4

from .parameters import validate_library_params
from .analysis import explore_charts, explore_summary, numeric_correlation
from .explore_time import exploration_offset
from .targets import iter_labels, preview_counts, source_columns


def bind_psi_dataset(workspace, source, binding, excluded):
    """锁定对照表版本，仅投影两个样本共有的分析字段。"""
    if binding["id"] == source["id"]:
        raise ValueError("请选择与分析样本不同的 PSI 对照样本")
    comparison = workspace.dataset_version(binding["id"], binding.get("revision"), include_examples=False)
    features = {c["name"] for c in source["columns"]}.difference(excluded)
    shared = [c["name"] for c in comparison["columns"] if c["name"] in features]
    if not shared:
        raise ValueError(f"PSI 对照样本「{comparison['name']}」与分析样本没有可比较的同名特征")
    return {"id": comparison["id"], "revision": comparison.get("revision", 1)}, shared


def explore_dataset(workspace, identifier, feature=None, other=None, bins=20, targets=None, positive_labels=None, *, revision=None, freq="M", label_mode="target", target=None, target_order=None, overdue=None, dpds=None, overdue_operator=">", del_grey=False, handle_missing_target="drop", psi_dataset=None, summary_only=False, axes=None, group=None, library_params=None):
    """图表配置不参与摘要缓存键；版本固定后读取，避免新旧数据混用。"""
    exploration_offset(freq)
    metadata = workspace.dataset_version(identifier, revision, include_examples=False)
    version = metadata.get("revision", 1)
    config = {"format": 1, "revision": version, "targets": targets or [], "positive_labels": positive_labels or {}}
    options = {} if handle_missing_target == "drop" else {"handle_missing_target": handle_missing_target}
    if label_mode == "overdue":
        options.update(label_mode=label_mode, overdue=overdue or [], dpds=dpds or [], overdue_operator=overdue_operator, del_grey=del_grey)
        config.update(targets=[], positive_labels={})
    elif isinstance(target, dict):
        options.update(target=target, target_order=target_order or list(target), del_grey=del_grey)
    if library_params:
        validate_library_params({"kind": "explore", "library_params": library_params})
        options["summary_params"] = library_params.get("summary", {})
    config.update(options)
    if psi_dataset:
        excluded = source_columns(dict(kind="explore", **config))
        config["psi_dataset"], shared = bind_psi_dataset(workspace, metadata, psi_dataset, excluded)
    key = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if metadata.get("storage") == "indices":
        directory = workspace.subsets.directory(metadata["root_id"], metadata["generation"]) / "summary-cache" / identifier
    else:
        directory = workspace.path("datasets", identifier) / "summary-cache"
    path = directory / f"{key}.json"
    data = None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        data = workspace.read_dataset(identifier, version)
        if psi_dataset:
            bound = config["psi_dataset"]
            options["val_df"] = workspace.read_dataset(bound["id"], bound["revision"], columns=shared)
        summary = explore_summary(data, targets, positive_labels, **options)
        directory.mkdir(parents=True, exist_ok=True)
        # 每次写入使用独立临时文件，完成后才发布，停止任务不会留下半份缓存。
        temporary = directory / f"{key}-{uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            # Windows 可能拒绝替换其他进程正在读取的缓存，本次任务仍使用已算出的摘要。
            with suppress(PermissionError):
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if summary_only:
        return summary
    if data is None:
        data = workspace.read_dataset(identifier, version)
    if axes:
        from .interactions import interactions

        kinds = {c["name"]: c["kind"] for c in metadata["columns"]}
        charts = interactions(data, axes, group, bins, freq, kinds)
        charts["target_counts"] = list(preview_counts(data, dict(config, kind="explore")))
        return {**summary, "other": other, **charts["distributions"][0], "interaction": charts}
    kind = next((c["kind"] for c in metadata["columns"] if c["name"] == feature), None)
    return {**summary, **explore_charts(data, feature, other, bins, freq=freq, feature_kind=kind), "correlation": numeric_correlation(data)}


def target_sample(data, config, name, population):
    """按已运行任务的目标口径筛选，保留该标签自己的有效样本。"""
    label = next(iter_labels(data, config, name), None)
    if label is None:
        raise ValueError("请选择本次分析已配置的目标标签")
    _, values = label
    mask = values.notna() if population == "valid" else values.eq({"good": 0, "bad": 1}[population])
    return data.loc[mask]


def explore_target_view(workspace, job, name=None, population="valid"):
    """历史任务也从绑定的数据版本读取，切换标签无需重新提交任务。"""
    from .interactions import interactions

    config = job["config"]
    chart = (job.get("result") or {}).get("interaction")
    if config.get("kind") != "explore" or not chart:
        raise ValueError("请选择已完成的数据探索任务")
    fields = list(source_columns(config))
    axes = [axis["name"] for axis in chart["axes"]]
    group = chart.get("target")
    if name is None and "target_counts" in chart:
        return chart["target_counts"]
    # 保留一个分析轴也能为只由标签向量定义的旧任务恢复原样本索引。
    if name is not None:
        fields.extend(axes)
        if group:
            fields.append(group)
    data = workspace.read_dataset(job["dataset_id"], job["dataset_revision"], columns=list(dict.fromkeys(fields)) or axes[:1])
    if name is None:
        return list(preview_counts(data, config))
    sample = target_sample(data, config, name, population)
    return interactions(sample, axes, group, config["bins"], config.get("freq", "M"), axis_definitions=chart["axes"])
