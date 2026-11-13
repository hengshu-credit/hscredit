"""组合划分条件树：顺序匹配、嵌套动作、样本归属追踪与完整性校验。"""

import math
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from .splitting import SplitRequest, split_frame, split_preview
from .storage import records

MAX_DEPTH = 8
MAX_NODES = 100
MAX_LEAVES = 30
OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "between", "contains", "is_null", "not_null"}


class SplitTreeRequest(BaseModel):
    class Config:
        extra = "forbid"

    expected_revision: int = Field(ge=1)
    tree: Dict[str, Any]


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("条件树节点字段缺失或包含不支持的配置")


def validate_tree(tree, revision):
    """先约束深度与节点总数，再解析每个切分配置，空分支也完整校验。"""
    stack = [(tree, 1, False)]
    count, names = 0, []
    while stack:
        node, depth, is_condition = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError("条件树最多支持 100 个节点、8 层嵌套")
        if not isinstance(node, dict):
            raise ValueError("条件树节点必须为对象")
        kind = node.get("type")
        if is_condition:
            if kind == "group":
                _object(node, ("type", "op", "children"))
                if not isinstance(node["op"], str) or node["op"] not in {"AND", "OR"} or not isinstance(node["children"], list) or not 1 <= len(node["children"]) <= 30:
                    raise ValueError("条件组需要选择且／或，并包含 1 到 30 个条件")
                stack.extend((child, depth + 1, True) for child in node["children"])
            elif kind == "leaf":
                _object(node, ("type", "field", "operator"), ("value",))
                if not isinstance(node["field"], str) or not node["field"] or not isinstance(node["operator"], str) or node["operator"] not in OPERATORS:
                    raise ValueError("请选择条件字段和有效的比较操作")
                op, value = node["operator"], node.get("value")
                if op not in {"is_null", "not_null"}:
                    values = value if op in {"in", "not_in", "between"} else [value]
                    if not isinstance(values, list) or not 1 <= len(values) <= 200 or (op == "between" and len(values) != 2):
                        raise ValueError("集合条件需要 1 到 200 个取值，区间需要两个端点")
                    if any(not isinstance(v, (str, int, float, bool)) or (isinstance(v, float) and not math.isfinite(v)) or (isinstance(v, str) and len(v) > 1000) for v in values):
                        raise ValueError("条件值需要为有效文本、数字或布尔值；缺失值请用为空条件")
            else:
                raise ValueError("不支持的条件节点类型")
            continue
        if "label" in node and (not isinstance(node["label"], str) or not 1 <= len(node["label"].strip()) <= 150):
            raise ValueError("分组名称长度必须为 1 到 150 个字符")
        if kind == "dataset":
            _object(node, ("type", "name"))
            if not isinstance(node["name"], str) or not 1 <= len(node["name"].strip()) <= 150:
                raise ValueError("结果数据集名称长度必须为 1 到 150 个字符")
            names.append(node["name"].strip())
        elif kind == "exclude":
            _object(node, ("type", "reason"))
            if not isinstance(node["reason"], str) or not 1 <= len(node["reason"].strip()) <= 150:
                raise ValueError("排除样本时请填写原因，最多 150 个字符")
        elif kind == "condition":
            _object(node, ("type", "rules", "otherwise"), ("label",))
            if not isinstance(node["rules"], list) or not 1 <= len(node["rules"]) <= 20:
                raise ValueError("每个条件节点需要 1 到 20 条规则及其他分支")
            stack.append((node["otherwise"], depth + 1, False))
            for rule in node["rules"]:
                _object(rule, ("condition", "action"))
                stack.extend([(rule["condition"], depth + 1, True), (rule["action"], depth + 1, False)])
        elif kind == "split":
            _object(node, ("type", "config", "children"), ("label",))
            if not isinstance(node["config"], dict) or "expected_revision" in node["config"]:
                raise ValueError("嵌套划分配置不能单独指定源数据版本")
            try:
                config = SplitRequest(expected_revision=revision, **node["config"])
            except (ValidationError, TypeError) as exc:
                raise ValueError("嵌套划分参数格式或范围不正确") from exc
            if not isinstance(node["children"], list) or len(node["children"]) != len(config.parts):
                raise ValueError("划分结果数量必须与后续动作数量一致")
            if config.method == "oot" and (len(config.parts) != 2 or not config.oot_start):
                raise ValueError("时间外划分需要起始时间和两个后续动作")
            if config.stratify and config.method != "random":
                raise ValueError("分层字段仅适用于随机划分")
            labels = [p.name.strip() for p in config.parts]
            if any(not label for label in labels) or len(set(labels)) != len(labels):
                raise ValueError("同一划分节点的分支名称需要非空且唯一")
            if config.method == "oot":
                try:
                    start = pd.Timestamp(config.oot_start)
                    end = pd.Timestamp(config.oot_end) if config.oot_end else None
                    if pd.isna(start) or (end is not None and (pd.isna(end) or end < start)):
                        raise ValueError()
                except (ValueError, TypeError) as exc:
                    raise ValueError("时间外起止时间不合法") from exc
            if config.method == "time" and config.time_mode == "cutoff":
                try:
                    boundaries = [pd.Timestamp(v) for v in config.cutoffs]
                    if len(boundaries) != len(config.parts) - 1 or any(pd.isna(v) for v in boundaries) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
                        raise ValueError()
                except (ValueError, TypeError) as exc:
                    raise ValueError("时间边界需要按从早到晚填写，数量比划分份数少一") from exc
            elif config.method != "oot":
                ratios = [p.ratio for p in config.parts]
                if any(v is None or not math.isfinite(v) for v in ratios) or not math.isclose(sum(ratios), 1, rel_tol=0, abs_tol=1e-8):
                    raise ValueError("嵌套划分的每份比例必须大于 0，合计为 100%")
            stack.extend((child, depth + 1, False) for child in node["children"])
        else:
            raise ValueError("不支持的划分动作类型")
    if not 1 <= len(names) <= MAX_LEAVES or len(set(names)) != len(names):
        raise ValueError("条件树需要 1 到 30 个结果数据集，名称必须互不重复")
    return names


def condition_mask(data, node, kinds):
    """只执行类型明确的向量比较；缺失值仅命中显式的为空条件。"""
    if node["type"] == "group":
        result = pd.Series(node["op"] == "AND", index=data.index)
        for child in node["children"]:
            mask = condition_mask(data, child, kinds)
            result = result & mask if node["op"] == "AND" else result | mask
        return result
    field, op = node["field"], node["operator"]
    if field not in data:
        raise ValueError(f"条件字段不存在：{field}")
    values = data[field]
    if op in {"is_null", "not_null"}:
        return values.isna() if op == "is_null" else values.notna()
    raw = node.get("value")
    literals = raw if op in {"in", "not_in", "between"} else [raw]
    is_date = kinds.get(field) == "date" or pd.api.types.is_datetime64_any_dtype(values)
    is_bool = pd.api.types.is_bool_dtype(values)
    is_numeric = pd.api.types.is_numeric_dtype(values) and not is_bool
    try:
        if is_date:
            if any(not isinstance(v, str) for v in literals):
                raise ValueError()
            values = pd.to_datetime(values, errors="raise")
            literals = [pd.Timestamp(v) for v in literals]
            if any(pd.isna(v) for v in literals):
                raise ValueError()
        elif is_bool:
            if any(not isinstance(v, (bool, str)) or v not in (True, False, "true", "false") for v in literals):
                raise ValueError()
            literals = [v is True or v == "true" for v in literals]
        elif is_numeric:
            if any(isinstance(v, bool) for v in literals):
                raise ValueError()
            literals = [int(v) if pd.api.types.is_integer_dtype(values) and (isinstance(v, int) or isinstance(v, str) and v.lstrip("+-").isdigit()) else float(v) for v in literals]
            if any(not math.isfinite(v) for v in literals):
                raise ValueError()
        if op == "contains":
            if is_date or is_bool or is_numeric or not isinstance(literals[0], str):
                raise ValueError()
            mask = values.astype("string").str.contains(literals[0], regex=False, na=False)
        elif op in {"in", "not_in"}:
            mask = values.isin(literals)
            if op == "not_in":
                mask = ~mask
        elif op == "between":
            if literals[0] > literals[1]:
                raise ValueError()
            mask = (values >= literals[0]) & (values <= literals[1])
        else:
            method = {"eq": "eq", "ne": "ne", "gt": "gt", "ge": "ge", "lt": "lt", "le": "le"}[op]
            mask = getattr(values, method)(literals[0])
        return (mask & values.notna()).fillna(False).astype(bool)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"条件字段「{field}」的比较值或操作不适用于该字段类型") from exc


def replace_group_tree(group, subtree, members):
    """只替换选中分支；保留其父边界、同级配置及已重命名的叶子。"""
    from copy import deepcopy

    tree = deepcopy(group["splitting"]["config"]["tree"])
    leaves = [d for d in members if "group_path" not in d]
    by_output = {d["splitting"].get("output_name", d["name"]): d["name"] for d in leaves}
    by_path = {d["splitting"]["path"]: d["name"] for d in leaves if "path" in d["splitting"]}
    target = tuple(group["group_path"])

    def visit(node, slot, path):
        if slot == target:
            result = deepcopy(subtree)
            if result["type"] in {"condition", "split"} and "label" in node:
                result["label"] = node["label"]
            return result
        if node["type"] == "dataset":
            node["name"] = by_path.get(path, by_output.get(node["name"].strip(), node["name"]))
        elif node["type"] == "condition":
            for i, rule in enumerate(node["rules"]):
                rule["action"] = visit(rule["action"], (*slot, i), f"{path} / 条件{i+1} / 动作")
            node["otherwise"] = visit(node["otherwise"], (*slot, "otherwise"), path + " / 其他")
        elif node["type"] == "split":
            for i, child in enumerate(node["children"]):
                part = node["config"]["parts"][i]
                follows_leaf = child["type"] == "dataset" and part["name"] == child["name"]
                updated = visit(child, (*slot, i), f"{path} / 划分{i+1}（{part['name'].strip()}）")
                if follows_leaf and updated["type"] == "dataset":
                    part["name"] = updated["name"]
                node["children"][i] = updated
        return node

    return visit(tree, (), "全部样本")


def execute_tree(data, request, columns, groups=None):
    """执行树并产生叶子及节点轨迹，每一行只能保存或排除一次。"""
    validate_tree(request.tree, request.expected_revision)
    kinds = {c["name"]: c["kind"] for c in columns}
    # 内部位置索引与业务字段分离，重复业务值和重复索引都保留。
    source = data.reset_index(drop=True)
    parts, summaries, trace = [], [], []
    ownership = np.zeros(len(source), dtype=np.uint8)
    excluded_rows = 0

    def claim(frame):
        positions = frame.index.to_numpy()
        if not frame.index.is_unique or ownership[positions].any():
            raise ValueError("划分结果存在重复的样本，请检查条件树")
        ownership[positions] = 1

    def exclude(frame, path, reason):
        nonlocal excluded_rows
        claim(frame)
        excluded_rows += len(frame)
        trace.append({"path": path, "action": "排除样本", "rows": len(frame), "reason": reason})

    def walk(node, frame, path, slot=(), label="", split_config=None):
        kind = node["type"]
        if groups is not None and slot and kind in {"condition", "split"} and len(frame):
            groups.append((slot, node.get("label") or label, frame))
        if kind == "dataset":
            claim(frame)
            name = node["name"].strip()
            summary = {"name": name, "path": path, "rows": len(frame), "fraction": len(frame) / len(source), "preview": records(frame.head(5))}
            if split_config is not None and len(frame):
                summary.update(split_preview(source, [(name, frame)], split_config)["parts"][0])
            summaries.append(summary)
            trace.append({"path": path, "action": "保存数据集", "name": name, "rows": len(frame)})
            if len(frame):
                parts.append((name, frame))
        elif kind == "exclude":
            exclude(frame, path, node["reason"].strip())
        elif kind == "condition":
            remaining = frame
            for index, rule in enumerate(node["rules"]):
                branch = f"{path} / 条件{index + 1}"
                mask = condition_mask(remaining, rule["condition"], kinds)
                selected = remaining.loc[mask]
                trace.append({"path": branch, "action": "条件匹配", "input_rows": len(remaining), "rows": len(selected)})
                walk(rule["action"], selected, branch + " / 动作", (*slot, index), f"条件 {index + 1} 命中")
                remaining = remaining.loc[~mask]
            walk(node["otherwise"], remaining, path + " / 其他", (*slot, "otherwise"), "其余样本")
        else:
            config = SplitRequest(expected_revision=request.expected_revision, **node["config"])
            for field in (config.stratify if config.method == "random" else None, config.group_col if config.method == "group" else None, config.time_col if config.method in {"time", "oot"} else None):
                if field is not None and field not in source:
                    raise ValueError(f"划分字段不存在：{field}")
            if (config.method == "group" and not config.group_col) or (config.method in {"time", "oot"} and not config.time_col):
                raise ValueError("请选择嵌套划分所需的分组或时间字段")
            if config.method in {"time", "oot"} and pd.api.types.is_numeric_dtype(source[config.time_col]):
                raise ValueError("时间划分需要日期或日期文本字段，不能直接使用数值字段")
            if len(frame):
                try:
                    outputs = split_frame(frame, config)
                except ValueError as exc:
                    raise ValueError(f"{path}：{exc}") from exc
                assigned = np.concatenate([part.index.to_numpy() for _, part in outputs])
                dropped = frame.loc[~frame.index.isin(assigned)]
                if len(dropped):
                    exclude(dropped, path, "时间间隔或时间外结束边界")
            else:
                outputs = [(p.name, frame) for p in config.parts]
            for index, ((label, output), child) in enumerate(zip(outputs, node["children"])):
                branch = f"{path} / 划分{index + 1}（{label}）"
                trace.append({"path": branch, "action": {"random": "随机划分", "time": "时间划分", "group": "分组划分", "oot": "时间外划分"}[config.method], "input_rows": len(frame), "rows": len(output)})
                walk(child, output, branch, (*slot, index), label, config)

    walk(request.tree, source, "全部样本")
    if not ownership.all():
        raise ValueError("划分结果存在重复或遗漏的样本，请检查条件树")
    if not parts:
        raise ValueError("条件树没有产生非空数据集，请调整条件或排除动作")
    preview = {"revision": request.expected_revision, "source_rows": len(source), "excluded_rows": excluded_rows, "parts": summaries, "trace": trace, "empty_datasets": [p["name"] for p in summaries if p["rows"] == 0]}
    return parts, preview
