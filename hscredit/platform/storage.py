"""本地数据集和任务清单的持久化，所有文件使用内部标识定位。"""

import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


def now():
    return datetime.now(timezone.utc).isoformat()


def records(frame):
    """将表格转换为标准 JSON，统一处理缺失值、时间与无穷大。"""
    result = json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))
    # 浏览器数值不能精确表达超出 53 位的客户/订单编号，仅在展示层转字符串。
    for row in result:
        for key, value in row.items():
            if isinstance(value, int) and abs(value) > 2**53 - 1:
                row[key] = str(value)
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Workspace:
    """保存本地工作台的数据和任务。

    **参数**

    :param root: 数据目录，默认 ~/.hscredit/platform。

    **属性**

    :ivar root: 解析后的数据目录。

    **参考样例**

    >>> workspace = Workspace('/tmp/hscredit-platform')  # doctest: +SKIP
    """

    def __init__(self, root=None):
        self.root = Path(root or Path.home() / ".hscredit" / "platform").expanduser().resolve()
        for name in ("datasets", "jobs", "services"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        from .subsets import SubsetStore

        self.subsets = SubsetStore(self)

    def path(self, kind, identifier):
        if kind not in {"datasets", "jobs", "services"} or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("资源标识不合法")
        return self.root / kind / identifier

    def list(self, kind, *, include_examples=True):
        with self.lock:
            items = [json.loads(p.read_text(encoding="utf-8")) for p in (self.root / kind).glob("*/metadata.json")]
            if kind == "datasets":
                items = self.subsets.visible(items, self.subsets.load(), include_examples=include_examples)
            return sorted(items, key=lambda x: x["created_at"], reverse=True)

    def get(self, kind, identifier):
        with self.lock:
            path = self.path(kind, identifier) / "metadata.json"
            if kind == "datasets":
                subset = self.subsets.metadata(self.subsets.load(), identifier)
                if subset is not None:
                    return subset
            if not path.exists():
                raise FileNotFoundError("数据集或任务不存在")
            return json.loads(path.read_text(encoding="utf-8"))

    def update(self, kind, identifier, **changes):
        with self.lock:
            value = self.get(kind, identifier)
            if kind == "datasets" and value.get("storage") == "indices":
                if "group_path" in value:
                    raise ValueError("请在所属划分方案中修改中间分组名称")
                if set(changes) != {"name"}:
                    raise ValueError("索引子集只支持更新名称或重新划分")
                name = changes["name"]
                if any(d["name"] == name and d["id"] != identifier for d in self.list("datasets", include_examples=False)):
                    raise ValueError("数据集名称重复")
                catalog = self.subsets.load()
                catalog["members"][identifier]["name"] = name
                catalog["plans"][value["splitting"]["plan_id"]]["revision"] += 1
                write_json(self.subsets.catalog, catalog)
                return self.get(kind, identifier)
            value.update(changes)
            write_json(self.path(kind, identifier) / "metadata.json", value)
            return value

    @staticmethod
    def prepare_dataset(frame, *, index_examples=False):
        """校验数据并统一生成字段统计，导入和加工使用同一口径。"""
        if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame.columns) == 0:
            raise ValueError("数据集必须是非空 DataFrame")
        if len(frame) > 2_000_000 or len(frame.columns) > 2000:
            raise ValueError("单个数据集最多支持 200 万行、2000 个字段")
        frame = frame.copy().reset_index(drop=True)
        frame.columns = frame.columns.map(str)
        if not frame.columns.is_unique or any(not x.strip() for x in frame.columns):
            raise ValueError("字段名称必须非空且唯一")
        columns = []
        for col in frame:
            series = frame[col]
            numeric = pd.api.types.is_numeric_dtype(series)
            dates = pd.api.types.is_datetime64_any_dtype(series)
            if not numeric and not dates and re.search(r"date|time|日期|时间", col, re.I):
                sample = series.dropna().head(100)
                dates = bool(len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9)
            examples = series.drop_duplicates().head(50)
            sample = {"example_indices": examples.index.tolist()} if index_examples else {"values": records(examples.to_frame("value"))}
            columns.append({"name": col, "dtype": str(series.dtype), "kind": "date" if dates else "numeric" if numeric else "category", "missing": int(series.isna().sum()), "unique": int(series.nunique()), **sample})
        return frame, {"rows": len(frame), "columns": columns, "missing_rate": float(frame.isna().mean().mean()), "duplicates": int(frame.duplicated().sum())}

    @staticmethod
    def dataset_defaults(template, frame):
        """只保留在新字段结构中仍然有效的分析默认选项。"""
        result = {}
        for key in ("default_target", "default_date", "default_amount", "default_features", "default_overdue"):
            value = template.get(key)
            if isinstance(value, list):
                result[key] = [col for col in value if col in frame]
            elif value in frame and (key != "default_target" or frame[value].nunique() == 2):
                result[key] = value
        return result

    def add_dataset(self, frame, name, *, processing=None, defaults=None, splitting=None):
        frame, summary = self.prepare_dataset(frame)
        summary.update(self.dataset_defaults(defaults or {}, frame))
        if processing:
            summary["processing"] = processing
        if splitting:
            summary["splitting"] = splitting
        return self._write_new_dataset(name, summary, lambda path: frame.to_parquet(path, index=False))

    def _write_new_dataset(self, name, summary, write_data):
        name = str(name).strip()
        if not name or len(name) > 150:
            raise ValueError("数据集名称长度必须为 1 到 150 个字符")
        identifier = uuid4().hex
        metadata = {"id": identifier, "name": name, **summary, "created_at": now(), "revision": 1, "data_file": "data.parquet", "history": []}
        with self.lock:
            directory = self.path("datasets", identifier)
            directory.mkdir()
            try:
                write_data(directory / "data.parquet")
                write_json(directory / "metadata.json", metadata)
            except Exception:
                shutil.rmtree(directory)
                raise
        return metadata

    def copy_dataset(self, identifier, expected_revision):
        """将当前版本保存为可独立加工和划分的新数据集。"""
        from .processing import DatasetConflict

        with self.lock:
            datasets = self.list("datasets", include_examples=False)
            if any(d["id"] == identifier and d.get("parent_id") for d in datasets):
                raise ValueError("仅原始数据集支持复制，请从所属源数据集复制")
            source = self.dataset_version(identifier, include_examples=False)
            if source.get("revision", 1) != expected_revision:
                raise DatasetConflict("数据集已更新，请刷新后复制")
            occupied = {d["name"] for d in datasets}
            number = 1
            while True:
                suffix = "（副本）" if number == 1 else f"（副本 {number}）"
                name = source["name"][: 150 - len(suffix)] + suffix
                if name not in occupied:
                    break
                number += 1
            fields = {k: v for k, v in source.items() if k in {"rows", "columns", "missing_rate", "duplicates", "note"} or k.startswith("default_")}

            return self._write_new_dataset(name, fields, lambda path: shutil.copyfile(self.data_path(source), path))

    def add_split_datasets(self, parts, *, source, config, excluded_rows, paths=None, plan_id=None, plan_revision=None, groups=None, pinned=False):
        """仅保存相对源版本的行位置，原子发布划分方案。"""
        return self.subsets.save(parts, source, config, excluded_rows, paths, plan_id, plan_revision, groups, pinned)

    def split_names_in_use(self, source_id, plan_id, plan_revision):
        with self.lock:
            datasets = self.list("datasets", include_examples=False)
            previous = self.subsets.plan_members(source_id, plan_id, plan_revision, datasets, self.subsets.load())
            owned = {child["id"] for child in previous}
            return {d["name"] for d in datasets if d["id"] not in owned and "group_path" not in d}

    def dataset_version(self, identifier, revision=None, *, include_examples=True):
        with self.lock:
            if revision is not None and (not isinstance(revision, int) or isinstance(revision, bool) or revision < 1):
                raise ValueError("数据版本不存在")
            physical = self.path("datasets", identifier) / "metadata.json"
            subset = self.subsets.metadata(self.subsets.load(), identifier, revision, include_examples=include_examples)
            if subset is not None:
                return subset
            if not physical.exists():
                raise FileNotFoundError("数据集不存在")
            current = json.loads(physical.read_text(encoding="utf-8"))
            number = current.get("revision", 1)
            if revision is None or revision == number:
                return current
            if revision > number:
                raise ValueError("数据版本不存在")
            snapshot = self.path("datasets", identifier) / "versions" / f"{revision}.json"
            if not snapshot.exists():
                raise ValueError("数据版本不存在")
            return json.loads(snapshot.read_text(encoding="utf-8"))

    def data_path(self, metadata):
        filename = metadata.get("data_file", "data.parquet")
        if not re.fullmatch(r"data(?:-[0-9a-f]{32})?\.parquet", filename):
            raise ValueError("数据文件标识不合法")
        return self.path("datasets", metadata["id"]) / filename

    def read_dataset(self, identifier, revision=None, *, columns=None):
        with self.lock:
            metadata = self.dataset_version(identifier, revision, include_examples=False)
            if metadata.get("storage") == "indices":
                return self.subsets.read(metadata, columns)
            return pd.read_parquet(self.data_path(metadata), columns=columns)

    def dataset_in_use(self, identifier):
        """同时保护训练、测试与其他评估集。"""
        for job in self.list("jobs"):
            if job["status"] in {"排队中", "运行中", "停止中"}:
                config = job["config"]
                if identifier in {job["dataset_id"], (config.get("test_dataset") or {}).get("id"), (config.get("psi_dataset") or {}).get("id"), *(b["id"] for b in config.get("evaluation_datasets", []))}:
                    return True
        return False

    def replace_dataset(self, identifier, frame, *, expected_revision, processing, template=None):
        """新数据文件写完后原子切换元数据指针，失败不会留下半完成数据集。"""
        from .processing import DatasetConflict

        frame, summary = self.prepare_dataset(frame)
        with self.lock:
            current = self.get("datasets", identifier)
            if current.get("storage") == "indices":
                raise DatasetConflict("索引子集引用源数据，请另存字段加工结果")
            revision = current.get("revision", 1)
            if revision != expected_revision:
                raise DatasetConflict("数据集已被更新，请刷新后重新预览")
            if self.dataset_in_use(identifier):
                raise DatasetConflict("数据集有正在运行或排队的任务，请完成后更新，或另存为新数据集")
            directory = self.path("datasets", identifier)
            versions = directory / "versions"
            versions.mkdir(exist_ok=True)
            baseline = versions / f"{revision}.json"
            if not baseline.exists():
                write_json(baseline, current)
            filename = f"data-{uuid4().hex}.parquet"
            updated = {**current, **summary, "data_file": filename, "revision": revision + 1, "updated_at": now(), "processing": processing}
            history = current.get("history") or [{"revision": revision, "at": current["created_at"], "description": "初始数据", "columns": len(current["columns"])}]
            updated["history"] = history + [{"revision": revision + 1, "at": updated["updated_at"], "description": processing["description"], "columns": len(frame.columns)}]
            for key in ("default_target", "default_date", "default_amount", "default_features", "default_overdue"):
                updated.pop(key, None)
            updated.update(self.dataset_defaults(template or current, frame))
            snapshot = versions / f"{revision + 1}.json"
            try:
                frame.to_parquet(directory / filename, index=False)
                write_json(snapshot, updated)
                write_json(directory / "metadata.json", updated)
            except Exception:
                (directory / filename).unlink(missing_ok=True)
                snapshot.unlink(missing_ok=True)
                raise
            return updated

    def delete_dataset(self, identifier):
        with self.lock:
            current = self.get("datasets", identifier)
            if current.get("storage") == "indices":
                raise ValueError("请在重新划分中移除子集，或删除整个划分方案")
            if any(d.get("parent_id") == identifier for d in self.list("datasets", include_examples=False)):
                raise ValueError("该数据集仍有划分方案，请先删除方案再删除源数据")
            catalog = self.subsets.load()
            dependent_ids = [key for key, member in catalog["members"].items() if member["root_id"] == identifier]
            if any(self.dataset_in_use(key) for key in [identifier, *dependent_ids]):
                raise ValueError("该数据集仍有待完成的任务，请任务结束后删除")
            shutil.rmtree(self.path("datasets", identifier))
