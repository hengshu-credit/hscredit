"""划分方案与子集索引。一次切换清单发布整批结果，版本文件供历史任务按需读取。"""

import json
import shutil
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np

from .processing import DatasetConflict
from .storage import now, records, write_json


def output_slots(config):
    """叶子地址与显示名称分离，调整比例或只改名称时沿用原有子集标识。"""
    slots = {}

    def walk(node, slot):
        if node["type"] == "dataset":
            slots[node["name"].strip()] = slot
        elif node["type"] == "condition":
            for index, rule in enumerate(node["rules"]):
                walk(rule["action"], (*slot, ("rule", index)))
            walk(node["otherwise"], (*slot, ("else", 0)))
        elif node["type"] == "split":
            for index, child in enumerate(node["children"]):
                walk(child, (*slot, ("part", index)))

    if "tree" in config:
        walk(config["tree"], ())
    else:
        slots = {p["name"].strip(): (("part", i),) for i, p in enumerate(config["parts"])}
    return slots


class SubsetStore:
    def __init__(self, workspace):
        self.workspace = workspace
        self.catalog = workspace.root / "split-catalog.json"
        self.examples = lru_cache(maxsize=16)(self._examples)

    def load(self):
        if self.catalog.exists():
            return json.loads(self.catalog.read_text(encoding="utf-8"))
        return {"plans": {}, "members": {}}

    def directory(self, root_id, generation):
        return self.workspace.path("datasets", root_id) / "splits" / generation

    def metadata(self, catalog, identifier, revision=None, *, include_examples=True):
        """没有索引版本时交还旧版 Parquet 读取流程。"""
        member = catalog["members"].get(identifier)
        if member is None:
            return None
        if revision is None and not member["active"]:
            raise FileNotFoundError("该子集已从当前划分方案移除")
        number = member["revision"] if revision is None else revision
        generation = member["versions"].get(str(number))
        if generation is None:
            if 0 < number <= member.get("legacy_revision", 0):
                return None
            raise ValueError("子集版本不存在")
        path = self.directory(member["root_id"], generation) / (identifier + ".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        if number == member["revision"]:
            data["name"] = member["name"]
            data["splitting"]["plan_revision"] = catalog["plans"][member["plan_id"]]["revision"]
        if include_examples:
            samples = self.examples(identifier, number)
            for field in data["columns"]:
                field.pop("example_indices", None)
                field["values"] = samples[field["name"]]
        return data

    def _examples(self, identifier, revision):
        metadata = self.metadata(self.load(), identifier, revision, include_examples=False)
        fields = metadata["columns"]
        rows = np.unique([i for field in fields for i in field["example_indices"]])
        frame = self.read(metadata, None, positions=rows)
        frame.index = rows
        return {f["name"]: records(frame.loc[f["example_indices"], f["name"]].to_frame("value")) for f in fields}

    def visible(self, physical, catalog, *, include_examples=True):
        items = []
        physical_ids = {d["id"] for d in physical}
        for row in physical:
            if row["id"] in catalog["members"]:
                continue
            info = row.get("splitting", {})
            # 旧版同批副本共享保存时间。原文件保留，首次重新划分时接入索引版本。
            if info.get("source_dataset_id") in physical_ids and info.get("at") and info.get("config") and row.get("revision", 1) == 1:
                plan = uuid5(NAMESPACE_URL, info["source_dataset_id"] + "/" + info["at"]).hex
                row = {**row, "parent_id": info["source_dataset_id"], "splitting": {**info, "plan_id": plan, "plan_revision": 1}}
            items.append(row)
        for identifier, member in catalog["members"].items():
            if member["active"]:
                items.append(self.metadata(catalog, identifier, include_examples=include_examples))
        return items

    def plan_members(self, source_id, plan_id, plan_revision, datasets, catalog):
        if plan_id is None and plan_revision is None:
            return []
        if not plan_id or not plan_revision:
            raise ValueError("重新划分需要方案标识和方案版本")
        children = [d for d in datasets if d.get("parent_id") == source_id and d.get("splitting", {}).get("plan_id") == plan_id]
        plan = catalog["plans"].get(plan_id)
        if not children or (plan and plan["source_id"] != source_id):
            raise DatasetConflict("划分方案不存在或已移除，请刷新数据资产")
        if children[0]["splitting"]["plan_revision"] != plan_revision:
            raise DatasetConflict("划分方案已更新，请重新打开后调整")
        return children

    def remove_plan(self, source_id, plan_id, revision):
        """从当前列表移除方案；历史任务按已绑定的版本仍可读取子集。"""
        with self.workspace.lock:
            catalog = self.load()
            datasets = self.workspace.list("datasets", include_examples=False)
            previous = self.plan_members(source_id, plan_id, revision, datasets, catalog)
            ids = {item["id"] for item in previous}
            if any(d.get("parent_id") in ids for d in datasets):
                raise DatasetConflict("该方案还有下级划分，请先删除下级方案")
            if any(self.workspace.dataset_in_use(i) for i in ids):
                raise DatasetConflict("子集仍被待完成的任务使用，请稍后删除方案")
            for item in previous:
                member = catalog["members"].setdefault(item["id"], {"plan_id": plan_id, "root_id": source_id, "revision": item["revision"], "legacy_revision": item["revision"], "versions": {}, "name": item["name"]})
                member["active"] = False
            catalog["plans"][plan_id] = {"source_id": source_id, "revision": revision + 1}
            write_json(self.catalog, catalog)

    def save(self, parts, source, config, excluded_rows, paths, plan_id, plan_revision, groups=None, pinned=False):
        ws = self.workspace
        with ws.lock:
            if not pinned and ws.dataset_version(source["id"], include_examples=False).get("revision", 1) != config["expected_revision"]:
                raise DatasetConflict("源数据集已被更新，请刷新后重新预览划分")
            catalog = self.load()
            datasets = ws.list("datasets", include_examples=False)
            previous = self.plan_members(source["id"], plan_id, plan_revision, datasets, catalog)
            previous_ids = {item["id"] for item in previous}
            occupied = {item["name"] for item in datasets if item["id"] not in previous_ids and "group_path" not in item}
            names = {name for name, _ in parts}
            if len(names) != len(parts) or occupied.intersection(names):
                raise ValueError("结果名称与已有数据集重复，请使用其他名称")
            previous_groups = {tuple(item["group_path"]): item for item in previous if "group_path" in item}
            leaves = [item for item in previous if "group_path" not in item]
            previous_by_name = {item["name"]: item for item in leaves}
            slots = output_slots(config)
            available_slots = {}
            for item in leaves:
                if item["name"] not in names:
                    info = item["splitting"]
                    old_slot = output_slots(info["config"]).get(info.get("output_name", item["name"]))
                    if old_slot is not None:
                        available_slots[old_slot] = item
            matches = {name: previous_by_name.get(name) or available_slots.get(slots[name]) for name in names}
            retained = {item["id"] for item in matches.values() if item}
            retained.update(previous_groups[slot]["id"] for slot, _, _ in groups or () if slot in previous_groups)
            removed = previous_ids - retained
            if any(d.get("parent_id") in removed for d in datasets):
                raise DatasetConflict("待移除的子集还有下级划分，请先调整下级方案")
            root_id = catalog["members"].get(source["id"], {}).get("root_id", source["id"])
            # 旧版实体副本升级为子集后，它之前创建的下级方案仍沿用原存储位置。
            if plan_id in catalog["plans"]:
                root_id = catalog["plans"][plan_id].get("root_id", root_id)
            generation = uuid4().hex
            folder = self.directory(root_id, generation)
            plan_id = plan_id or uuid4().hex
            version = (plan_revision or 0) + 1
            at = now()
            lineage = {"source_dataset_id": source["id"], "source_name": source["name"], "source_revision": config["expected_revision"], "config": config, "plan_id": plan_id, "plan_revision": version, "excluded_rows": excluded_rows, "at": at}
            result = []
            folder.mkdir(parents=True)
            try:
                for prior in previous:
                    member = catalog["members"].setdefault(prior["id"], {"plan_id": plan_id, "root_id": root_id, "legacy_revision": prior["revision"], "revision": prior["revision"], "versions": {}, "name": prior["name"]})
                    member["active"] = False
                for index, (name, frame) in enumerate(parts):
                    prior = matches[name]
                    info = {**lineage, "part_index": index, "output_name": name}
                    if paths:
                        info["path"] = paths[name]
                    result.append(self.write_member(catalog, folder, root_id, source, info, name, frame, prior))
                for slot, name, frame in groups or ():
                    self.write_member(catalog, folder, root_id, source, lineage, name, frame, previous_groups.get(slot), slot)
                catalog["plans"][plan_id] = {"source_id": source["id"], "root_id": root_id, "revision": version, "groups_indexed": groups is not None}
                write_json(self.catalog, catalog)
            except Exception:
                shutil.rmtree(folder)
                raise
            return result

    def write_member(self, catalog, folder, root_id, source, lineage, name, frame, prior=None, slot=None):
        """写入一个绑定源版本的样本索引，分组与叶子使用相同的分析读取接口。"""
        identifier = prior["id"] if prior else uuid4().hex
        number = prior["revision"] + 1 if prior else 1
        positions = frame.index.to_numpy()
        if not len(positions) or positions.dtype.kind not in "iu" or not frame.index.is_unique or positions.min() < 0 or positions.max() >= source["rows"]:
            raise ValueError("划分结果的行位置不在源数据范围内")
        np.save(folder / (identifier + ".npy"), positions.astype(np.int32), allow_pickle=False)
        normalized, summary = self.workspace.prepare_dataset(frame, index_examples=True)
        at = lineage["at"]
        item = {"id": identifier, "name": name, "parent_id": source["id"], "storage": "indices", "root_id": root_id, "generation": folder.name, "splitting": lineage, "revision": number, "created_at": prior["created_at"] if prior else at, "updated_at": at, "history": [], **summary, **self.workspace.dataset_defaults(source, frame)}
        if slot is not None:
            item["group_path"] = list(slot)
        write_json(folder / (identifier + ".json"), item)
        for field in item["columns"]:
            sample = normalized[field["name"]].iloc[field.pop("example_indices")]
            field["values"] = records(sample.to_frame("value"))
        member = catalog["members"].setdefault(identifier, {"plan_id": lineage["plan_id"], "root_id": root_id, "versions": {}})
        member.update(active=True, revision=number, name=name)
        member["versions"][str(number)] = folder.name
        return item

    def datasets_with_groups(self):
        """首次列出旧组合方案时，按其绑定的源版本补齐中间索引，不改写原有子集。"""
        from .split_tree import SplitTreeRequest, execute_tree

        ws = self.workspace
        with ws.lock:
            datasets = ws.list("datasets")
            catalog = self.load()
            seen = set()
            for item in list(datasets):
                info = item.get("splitting") or {}
                plan_id = info.get("plan_id")
                if not plan_id or plan_id in seen:
                    continue
                seen.add(plan_id)
                plan = catalog["plans"].get(plan_id, {})
                tree = info["config"].get("tree")
                if plan.get("groups_indexed") or not tree:
                    continue
                children = tree.get("children", []) if tree["type"] == "split" else [r["action"] for r in tree.get("rules", [])] + [tree.get("otherwise", {})]
                if not any(n.get("type") in {"split", "condition"} for n in children):
                    continue
                source = ws.dataset_version(info["source_dataset_id"], info["source_revision"], include_examples=False)
                groups = []
                execute_tree(ws.read_dataset(source["id"], info["source_revision"]), SplitTreeRequest(**info["config"]), source["columns"], groups)
                root_id = plan.get("root_id", item.get("root_id", source["id"]))
                folder = self.directory(root_id, uuid4().hex)
                lineage = {k: v for k, v in info.items() if k not in {"part_index", "path", "output_name"}}
                folder.mkdir(parents=True)
                try:
                    for slot, name, frame in groups:
                        datasets.append(self.write_member(catalog, folder, root_id, source, lineage, name, frame, slot=slot))
                    catalog["plans"][plan_id] = {**plan, "revision": info["plan_revision"], "source_id": source["id"], "root_id": root_id, "groups_indexed": True}
                    write_json(self.catalog, catalog)
                except Exception:
                    shutil.rmtree(folder)
                    raise
            return datasets

    def read(self, metadata, columns, *, positions=None):
        """先合成多级行索引，再从原始 Parquet 读取命中的行组和请求的列。"""
        import pyarrow as pa
        import pyarrow.parquet as pq

        while metadata.get("storage") == "indices":
            info = metadata["splitting"]
            parent = self.workspace.dataset_version(info["source_dataset_id"], info["source_revision"], include_examples=False)
            path = self.directory(metadata["root_id"], metadata["generation"]) / (metadata["id"] + ".npy")
            indices = np.load(path, allow_pickle=False)
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != metadata["rows"] or indices.min() < 0 or indices.max() >= parent["rows"]:
                raise ValueError("子集索引损坏或与源数据版本不匹配")
            positions = indices if positions is None else indices[positions]
            metadata = parent
        order = np.argsort(positions)
        ordered = positions[order]
        pieces = []
        with pq.ParquetFile(self.workspace.data_path(metadata)) as source:
            if columns is not None and any(col not in source.schema_arrow.names for col in columns):
                raise ValueError("所选字段不在绑定的源数据版本中")
            offset = start = 0
            for group in range(source.num_row_groups):
                end = offset + source.metadata.row_group(group).num_rows
                stop = ordered.searchsorted(end)
                if stop > start:
                    block = source.read_row_group(group, columns=columns)
                    pieces.append(block.take(ordered[start:stop] - offset))
                offset, start = end, stop
            if start != len(positions):
                raise ValueError("源文件行数与绑定的数据版本不一致")
        return pa.concat_tables(pieces).take(np.argsort(order)).to_pandas().reset_index(drop=True)
