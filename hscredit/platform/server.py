"""风控平台 HTTP 接口、后台任务调度和静态前端资源。"""

import asyncio
import hmac
import io
import json
import mimetypes
import os
import shutil
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, root_validator

from .parameters import OptimalBinning, parameter_catalog, validate_library_params
from .analysis import MODEL_CATALOG
from .targets import label_map, overdue_options, preview_counts, source_columns, training_label, validate_columns
from .exploration import bind_psi_dataset, explore_dataset
from .explore_time import exploration_offset
from .storage import Workspace, now, write_json
from .processing import DatasetConflict, FUNCTIONS, OPERATORS, compile_rules, parse_expression_text, process_frame, preview_result
from .profiles import column_profile
from .correlation import dataset_correlation
from .preview import PreviewFilter, preview_rows, preview_values
from .pivot import PivotRequest, pivot_table
from .splitting import SplitRequest, split_frame, split_preview
from .split_tree import SplitTreeRequest, execute_tree, replace_group_tree, validate_tree
from .jobs import ACTIVE_STATUSES, JobManager


class LabelField(BaseModel):
    """页面中的字段引用或坏样本条件，在每份数据集上单独取值。"""

    class Config:
        extra = "forbid"

    field: str = Field(min_length=1)
    operator: Optional[Literal["==", "!=", ">", ">=", "<", "<="]] = None
    value: Any = None


LabelDictionary = Dict[str, Union[LabelField, List[Any]]]


class TargetOptions(BaseModel):
    """页面共用的标签来源、阈值与缺失处理。"""

    label_mode: Literal["target", "overdue"] = "target"
    overdue: List[str] = Field(default_factory=list)
    dpds: List[float] = Field(default_factory=list, max_items=20)
    overdue_operator: Literal[">", ">=", "<", "<="] = ">"
    del_grey: bool = False
    handle_missing_target: Literal["drop", "positive", "negative", "raise"] = "drop"
    target_order: List[str] = Field(default_factory=list)

    @root_validator(pre=True)
    def infer_label_mode(cls, values):
        target = values.get("target")
        if isinstance(target, dict):
            if not target or values.get("label_mode") == "overdue" or any(values.get(k) for k in ("overdue", "dpds", "targets", "evaluation_targets", "positive_labels")):
                raise ValueError("请单独传入非空的 target 标签字典")
            if values.get("kind") in {"interaction", "explain"}:
                raise ValueError("此任务不接受目标标签字典")
            for definition in target.values():
                if isinstance(definition, list) and any(isinstance(v, (dict, list)) for v in definition):
                    raise ValueError("字典中的标签向量必须是一维")
            order = values.get("target_order") or list(target)
            if not isinstance(order, list) or any(not isinstance(n, str) for n in order) or len(order) != len(target) or set(order) != set(target):
                raise ValueError("标签顺序必须完整且与字典键一致")
            return {**values, "target_order": order}
        # 历史页面同时提交两个模式的草稿，以显式模式为准。
        if "label_mode" not in values and (values.get("overdue") or values.get("dpds")):
            if values.get("target") or values.get("targets") or values.get("evaluation_targets"):
                raise ValueError("现有标签与逾期口径只能选择一种")
            return {**values, "label_mode": "overdue"}
        return values


class TargetPreviewRequest(TargetOptions):
    class Config:
        extra = "forbid"

    targets: List[str] = Field(default_factory=list)
    target: Optional[LabelDictionary] = None
    positive_labels: Dict[str, Any] = Field(default_factory=dict)


class DatasetBinding(BaseModel):
    """按标识和可选版本引用一个平台数据集。"""

    class Config:
        extra = "forbid"

    id: str
    revision: Optional[int] = Field(default=None, ge=1)


class ExploreRequest(TargetPreviewRequest):
    """字段探索配置，支持各标签独立指定坏样本取值。"""

    library_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    feature: Optional[str] = None
    psi_dataset: Optional[DatasetBinding] = None
    other: Optional[str] = None
    bins: int = Field(default=20, ge=2, le=100)
    freq: str = Field(default="M", min_length=1, max_length=32)


class JobRequest(TargetOptions):
    """来自页面的受约束任务参数，拒绝任意代码与未知配置。"""

    class Config:
        extra = "forbid"

    kind: Literal["strategy", "model", "third_party", "explore", "interaction", "ruleset", "explain"]
    primary_index: int = Field(default=0, ge=0)
    workflow: bool = False
    strategy_mode: Literal["single", "pair", "binary", "multiway"] = "single"
    model_mode: Literal["experiment", "compare"] = "experiment"
    evaluation_datasets: List[Dict[str, Any]] = Field(default_factory=list, max_items=20)
    psi_dataset: Optional[DatasetBinding] = None
    test_dataset: Optional[Dict[str, Any]] = None
    evaluation_targets: List[str] = Field(default_factory=list, max_items=10)
    use_holdout: bool = True
    selector: Literal["none", "null", "iv"] = "none"
    selector_threshold: float = Field(default=0.02, ge=0, le=1)
    scorecard: bool = False
    pdo: float = Field(default=20, gt=0)
    base_score: float = Field(default=600)
    explain: bool = False
    source_job: Optional[str] = None
    model_file: str = "model.joblib"
    sample_rows: List[int] = Field(default_factory=list, max_items=10)
    tree_depth: int = Field(default=3, ge=1, le=10)
    tree_min_samples: int = Field(default=10, ge=1)
    tree_edits: List[Dict[str, Any]] = Field(default_factory=list, max_items=30)
    selected_rules: List[Dict[str, str]] = Field(default_factory=list, max_items=200)
    rule_logic: Literal["or", "and"] = "or"
    rule_name: str = Field(default="规则集", max_length=100)
    group: Optional[str] = None
    dataset_id: str
    dataset_revision: Optional[int] = Field(default=None, ge=1)
    features: List[str] = Field(default_factory=list, max_items=200)
    feature: Optional[str] = None
    other: Optional[str] = None
    bins: int = Field(default=20, ge=2, le=100)
    targets: List[str] = Field(default_factory=list, max_items=30)
    positive_labels: Dict[str, Any] = Field(default_factory=dict)
    target: Optional[Union[str, LabelDictionary]] = None
    positive_label: Any = 1
    method: str = "quantile"
    library_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    max_n_bins: int = Field(default=5, ge=2, le=50)
    min_n_bins: int = Field(default=1, ge=1, le=50)
    min_bin_size: float = Field(default=1, gt=0, le=1000000)
    mining: bool = True
    min_lift: float = Field(default=1.1, ge=0, le=100)
    min_samples: int = Field(default=10, ge=1)
    top_n: int = Field(default=30, ge=1, le=500)
    models: List[str] = Field(default_factory=lambda: ["logistic", "random_forest"], min_items=1, max_items=50)
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    model_fit_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    search_spaces: Dict[str, Dict[str, List[Any]]] = Field(default_factory=dict)
    search: bool = False
    n_iter: int = Field(default=10, ge=1, le=100)
    cv: int = Field(default=3, ge=2, le=10)
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    random_state: int = Field(default=42, ge=0, le=2147483647)
    date: Optional[str] = None
    amount: Optional[str] = None
    freq: str = Field(default="M", min_length=1, max_length=32)
    pictures: List[Literal["ks", "hist", "bin"]] = Field(default_factory=lambda: ["bin", "ks"])
    corr: bool = False
    dropna: bool = False
    margins: bool = True
    long_format: bool = False


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)


class CopyDatasetRequest(BaseModel):
    expected_revision: int = Field(ge=1)


class ProcessingRequest(BaseModel):
    class Config:
        extra = "forbid"

    rules: List[Dict[str, Any]] = Field(min_items=1, max_items=30)
    expected_revision: int = Field(ge=1)
    save_as: bool = False
    name: Optional[str] = Field(default=None, max_length=150)


class RestoreRequest(BaseModel):
    class Config:
        extra = "forbid"

    revision: int = Field(ge=1)
    expected_revision: int = Field(ge=1)


class ExpressionParseRequest(BaseModel):
    class Config:
        extra = "forbid"

    text: str = Field(min_length=1, max_length=10000)
    expected_revision: int = Field(ge=1)
    prior_rules: List[Dict[str, Any]] = Field(default_factory=list, max_items=29)


class DatasetRunBinding(DatasetBinding):
    revision: int = Field(ge=1)


class BatchJobRequest(BaseModel):
    class Config:
        extra = "forbid"

    config: JobRequest
    datasets: List[DatasetRunBinding] = Field(min_items=1)


def validate_job(data, config):
    config["method"] = OptimalBinning.validate_method(config.get("method", "quantile"))
    validate_library_params(config)
    if config["kind"] == "interaction":
        validate_columns(data, config["features"])
        if len(config["features"]) > 3:
            raise ValueError("最多选择三个交互变量")
        if config["target"]:
            validate_columns(data, [config["target"]])
            if config["target"] in config["features"]:
                raise ValueError("标签不能同时作为交互变量")
        exploration_offset(config["freq"])
        return
    if config["kind"] == "explain":
        if not config["source_job"] or not config["sample_rows"] or any(i < 0 or i >= len(data) for i in config["sample_rows"]):
            raise ValueError("请选择模型任务及有效样本行号")
        return
    if config.get("workflow"):
        from .serving import validated_rules

        if config["kind"] == "model" and config["model_mode"] == "experiment" and len(config["models"]) != 1:
            raise ValueError("模型实验每次只能选择一个模型")
        if config["kind"] == "strategy":
            validate_columns(data, config["features"])
            count = {"single": 1, "pair": 2}.get(config["strategy_mode"])
            if count and len(config["features"]) != count:
                raise ValueError(f"当前挖掘模式需要选择 {count} 个变量")
            if config["strategy_mode"] == "binary" and any(not pd.api.types.is_numeric_dtype(data[c]) for c in config["features"]):
                raise ValueError("手工二叉树需要数值特征；类别特征可使用多叉树或先加工编码")
        if config["kind"] == "ruleset":
            rules = validated_rules(config["selected_rules"])
            for rule in rules:
                try:
                    rule.predict(data.head(5))
                except Exception as exc:
                    raise ValueError(f"规则“{rule.name}”不能在当前训练集上执行：{exc}") from exc
        if config.get("group"):
            validate_columns(data, [config["group"]])
    if config["kind"] == "explore":
        exploration_offset(config["freq"])
        if config.get("workflow") and config["features"]:
            fields = config["features"]
            validate_columns(data, fields)
            if len(fields) > 3:
                raise ValueError("最多选择三个探索字段")
            if config.get("group") in fields:
                raise ValueError("分组标签不能同时作为探索字段")
        for field in (config["feature"], config["other"]):
            if field:
                validate_columns(data, [field])
        if config["other"] and (not config["feature"] or any(not pd.api.types.is_numeric_dtype(data[c]) for c in (config["feature"], config["other"]))):
            raise ValueError("散点图需要两个数值字段")
        if config["label_mode"] == "overdue":
            overdue_options(data, config)
        else:
            label_map(data, config)
        return
    if config["freq"] not in {"D", "W", "M", "Q"}:
        raise ValueError("报告时间粒度只支持日、周、月和季度")
    if config["min_n_bins"] > config["max_n_bins"]:
        raise ValueError("最小分箱数不能大于最大分箱数")
    validate_columns(data, config["features"])
    if set(config["features"]).intersection(source_columns(config)):
        raise ValueError("标签字段不能同时作为特征")
    if config["kind"] == "third_party" and config["label_mode"] == "overdue":
        overdue_options(data, config)
    elif config["kind"] in {"third_party", "ruleset"}:
        label_map(data, config, evaluation=config["kind"] == "ruleset")
    else:
        training_label(label_map(data, config, evaluation=True))
    if len(config["models"]) != len(set(config["models"])):
        raise ValueError("模型列表不能重复")
    for key in config["models"]:
        if key not in MODEL_CATALOG:
            raise ValueError(f"未知模型：{key}")
        if config["kind"] == "model" and not MODEL_CATALOG[key]["available"]:
            raise ValueError(f"{MODEL_CATALOG[key]['name']}：{MODEL_CATALOG[key]['reason']}")
    for key in ("date", "amount", "group"):
        if config[key]:
            validate_columns(data, [config[key]])
    for model, params in config.get("model_fit_params", {}).items():
        if config["kind"] != "model" or model not in config["models"]:
            raise ValueError(f"拟合参数未绑定到所选模型：{model}")
        unknown = params.keys() - MODEL_CATALOG[model]["fit_params"].keys()
        if unknown:
            raise ValueError(f"{model} 不支持拟合参数：{'、'.join(sorted(unknown))}")
        for key in params.keys() & {"sample_weight", "base_margin", "init_score", "baseline"}:
            value = params[key]
            if isinstance(value, str):
                validate_columns(data, [value])
            elif value is not None and (not isinstance(value, list) or len(value) != len(data)):
                raise ValueError(f"{model} 的 {key} 需要字段名或与原样本等长的数组")
    for model, params in config["model_params"].items():
        if model not in config["models"] or params.keys() - (MODEL_CATALOG[model]["params"].keys() | MODEL_CATALOG[model]["extra_params"].keys()):
            raise ValueError(f"{model} 的模型参数不在支持范围内：{', '.join(params)}")
        for key, value in params.items():
            _validate_model_value(model, key, value)
    for model, space in config["search_spaces"].items():
        if model not in config["models"] or space.keys() - (MODEL_CATALOG[model]["params"].keys() | MODEL_CATALOG[model]["extra_params"].keys()):
            raise ValueError("搜索参数不在当前模型支持的配置范围内")
        for key, values in space.items():
            if not values or len(values) > 100:
                raise ValueError("每个搜索参数需要 1 到 100 个候选值")
            for value in values:
                _validate_model_value(model, key, value)


def _validate_model_value(model, key, value):
    entry = MODEL_CATALOG[model]
    if key in entry["extra_params"]:
        # 可选参数常有联合类型，具体值交给当前版本的底层估计器校验。
        if key == "class_weight" and isinstance(value, dict) and not set(value) <= {"0", "1"}:
            raise ValueError("类别权重的键应为 0 或 1")
        return
    if key == "max_depth" and (value is None or (model == "lightgbm" and value == -1)):
        return
    if key == "min_samples_leaf" and type(value) is float and 0 < value <= 1:
        return
    default = entry["params"][key]
    choices = MODEL_CATALOG[model]["choices"].get(key)
    if choices:
        if value not in choices:
            raise ValueError(f"模型参数 {key} 不在可选值中")
        return
    if isinstance(default, str):
        if not isinstance(value, str) or len(value) > 200:
            raise ValueError(f"模型参数 {key} 必须为有效字符串")
        return
    limits = {"C": (0.000001, 1000000), "max_iter": (10, 10000), "n_estimators": (1, 1000), "max_depth": (1, 30), "min_samples_leaf": (1, 100000), "learning_rate": (0.000001, 1), "max_epochs": (1, 1000), "batch_size": (2, 8192), "calib_ratio": (0.05, 0.5)}
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"模型参数 {key} 必须为布尔值")
        return
    low, high = limits.get(key, (-float("inf"), float("inf")))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"模型参数 {key} 必须在 {low} 到 {high} 之间")
    if isinstance(default, int) and not isinstance(value, int):
        raise ValueError(f"模型参数 {key} 必须为整数")


def create_app(data_dir=None, *, token=None):
    """创建本地工作台应用；同一数据目录同时只允许一个实例。"""
    # Windows 注册表可能将 SVG 登记为 image/svg，浏览器在 nosniff 下无法显示。
    mimetypes.add_type("image/svg+xml", ".svg")
    workspace = Workspace(data_dir)
    manager = JobManager(workspace)

    @asynccontextmanager
    async def lifespan(app):
        # OS 文件锁在进程退出时自动释放，避免多个实例改写同一任务清单。
        lock_file = (workspace.root / "server.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise RuntimeError("该数据目录已有运行中的平台，请使用其他 data_dir") from exc
        try:
            for job in workspace.list("jobs"):
                if job["status"] in ACTIVE_STATUSES:
                    workspace.update("jobs", job["id"], status="失败", message="上次服务退出，任务中断，请重新提交", finished_at=now())
            await asyncio.to_thread(calls.initialize)
            manager.start()
            cleanup = asyncio.create_task(calls.maintain())
            try:
                yield
            finally:
                cleanup.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup
        finally:
            try:
                await asyncio.to_thread(manager.close)
            finally:
                lock_file.close()

    app = FastAPI(title="hscredit 风控平台", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.workspace = workspace
    app.state.jobs = manager

    @app.middleware("http")
    async def protect(request, call_next):
        host = request.headers.get("host", "")
        hostname = urlsplit("http://" + host).hostname
        if not token and hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "仅允许本机地址访问"}, status_code=403)
        if request.url.path.startswith("/api/"):
            if token and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
                return JSONResponse({"detail": "访问令牌无效，请使用完整的启动链接"}, status_code=401)
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != host:
                return JSONResponse({"detail": "不允许跨站请求"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "不允许跨站请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(DatasetConflict)
    async def dataset_conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def not_found(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = ", ".join(".".join(str(i) for i in e["loc"][1:]) for e in exc.errors())
        return JSONResponse({"detail": f"参数格式或范围不正确：{fields}"}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"status": "正常"}

    @app.get("/api/catalog")
    def catalog():
        return {"models": MODEL_CATALOG, "parameters": parameter_catalog(), "max_upload_mb": 100, "processing": {"functions": FUNCTIONS, "operators": OPERATORS}}

    @app.get("/api/datasets")
    def datasets():
        return workspace.subsets.datasets_with_groups()

    @app.post("/api/datasets")
    def upload(file: UploadFile = File(...), name: str = Form(""), encoding: str = Form("utf-8-sig"), sheet: str = Form("")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".csv", ".xlsx", ".parquet"}:
            raise ValueError("仅支持 CSV、XLSX 和 Parquet 文件")
        if encoding not in {"utf-8-sig", "utf-8", "gb18030"}:
            raise ValueError("不支持该编码")
        content = file.file.read(100 * 1024 * 1024 + 1)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(413, "文件大小不能超过 100 MB")
        try:
            source = io.BytesIO(content)
            if suffix == ".csv":
                data = pd.read_csv(source, encoding=encoding)
            elif suffix == ".xlsx":
                data = pd.read_excel(source, sheet_name=sheet or 0)
            else:
                data = pd.read_parquet(source)
        except Exception as exc:
            raise ValueError(f"读取文件失败，请检查文件内容、编码或工作表名称：{exc}") from exc
        return workspace.add_dataset(data, name.strip() or Path(file.filename).stem)

    @app.post("/api/datasets/demo")
    def demo_dataset():
        from .demo import load_demo

        return load_demo(workspace)

    @app.get("/api/datasets/{identifier}")
    def dataset(identifier: str, revision: Optional[int] = Query(None, ge=1)):
        return workspace.dataset_version(identifier, revision) if revision else workspace.get("datasets", identifier)

    @app.patch("/api/datasets/{identifier}")
    def rename(identifier: str, request: RenameRequest):
        if not request.name.strip():
            raise ValueError("数据集名称不能为空")
        return workspace.update("datasets", identifier, name=request.name.strip())

    @app.post("/api/datasets/{identifier}/copy")
    def copy_dataset(identifier: str, request: CopyDatasetRequest):
        return workspace.copy_dataset(identifier, request.expected_revision)

    @app.delete("/api/datasets/{identifier}")
    def delete(identifier: str):
        workspace.delete_dataset(identifier)
        return {"message": "数据集已删除，历史报告仍保留"}

    @app.get("/api/datasets/{identifier}/preview")
    def preview(identifier: str, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), query: str = ""):
        return preview_rows(workspace.read_dataset(identifier), page, page_size, query, {})

    @app.post("/api/datasets/{identifier}/preview")
    def filtered_preview(identifier: str, request: PreviewFilter, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), revision: Optional[int] = Query(None, ge=1)):
        data = workspace.read_dataset(identifier, revision)
        return preview_rows(data, page, page_size, request.query, request.filters)

    @app.get("/api/datasets/{identifier}/preview-values")
    def dataset_preview_values(identifier: str, field: str, query: str = "", revision: Optional[int] = Query(None, ge=1)):
        metadata = workspace.dataset_version(identifier, revision, include_examples=False)
        if field not in {c["name"] for c in metadata["columns"]}:
            raise ValueError("请选择数据集中存在的字段")
        data = workspace.read_dataset(identifier, metadata.get("revision", 1), columns=[field])
        return preview_values(data[field], query)

    @app.get("/api/datasets/{identifier}/summary")
    def dataset_summary(identifier: str, revision: Optional[int] = Query(None, ge=1)):
        return explore_dataset(workspace, identifier, revision=revision, summary_only=True)

    @app.get("/api/datasets/{identifier}/correlation")
    def correlation(identifier: str, revision: Optional[int] = Query(None, ge=1), row_page: int = Query(1, ge=1), column_page: int = Query(1, ge=1), page_size: int = Query(16, ge=1, le=64)):
        return dataset_correlation(workspace, identifier, revision, row_page, column_page, page_size)

    @app.get("/api/datasets/{identifier}/explore")
    def exploration(identifier: str, feature: Optional[str] = None, other: Optional[str] = None, bins: int = Query(20, ge=2, le=100), freq: str = Query("M", min_length=1, max_length=32)):
        return explore_dataset(workspace, identifier, feature, other, bins, freq=freq)

    @app.post("/api/datasets/{identifier}/target-preview")
    def preview_targets(identifier: str, request: TargetPreviewRequest, revision: Optional[int] = Query(None, ge=1)):
        config = dict(request.dict(), kind="explore")
        fields = source_columns(config)
        metadata = workspace.dataset_version(identifier, revision)
        if fields or not isinstance(config.get("target"), dict):
            validate_columns({c["name"] for c in metadata["columns"]}, fields)
            frame = workspace.read_dataset(identifier, metadata["revision"], columns=fields)
        else:
            frame = pd.DataFrame(index=pd.RangeIndex(metadata["rows"]))
        return list(preview_counts(frame, config))

    @app.get("/api/datasets/{identifier}/column-profile")
    def preview_column_profile(identifier: str, column: str, revision: Optional[int] = Query(None, ge=1)):
        metadata = workspace.dataset_version(identifier, revision)
        field = next((item for item in metadata["columns"] if item["name"] == column), None)
        if field is None:
            raise ValueError("请选择数据集中存在的字段")
        version = metadata.get("revision", 1)
        data = workspace.read_dataset(identifier, version, columns=[column])
        return {**column_profile(data, column, field["kind"]), "revision": version}

    @app.post("/api/datasets/{identifier}/explore")
    def configured_exploration(identifier: str, request: ExploreRequest):
        return explore_dataset(workspace, identifier, **request.dict())

    def processing_input(identifier, request):
        with workspace.lock:
            metadata = workspace.get("datasets", identifier)
            revision = metadata.get("revision", 1)
            if revision != request.expected_revision:
                raise DatasetConflict("数据集已被更新，请刷新后重新预览")
            return metadata, workspace.read_dataset(identifier, revision)

    def splitting_input(identifier, request, plan_id, plan_revision):
        metadata, data = processing_input(identifier, request)
        if "group_path" in metadata:
            raise ValueError("请通过分组的划分入口更新所属方案")
        existing = workspace.split_names_in_use(identifier, plan_id, plan_revision)
        if any(part.name.strip() in existing for part in request.parts):
            raise ValueError("结果名称与已有数据集重复，请使用其他名称")
        parts = split_frame(data, request)
        return metadata, data, parts

    @app.post("/api/datasets/{identifier}/split/preview")
    def splitting_preview(identifier: str, request: SplitRequest, plan_id: Optional[str] = None, plan_revision: Optional[int] = Query(None, ge=1)):
        _, data, parts = splitting_input(identifier, request, plan_id, plan_revision)
        return split_preview(data, parts, request)

    def tree_input(identifier, request, plan_id, plan_revision, groups=None):
        with workspace.lock:
            metadata = workspace.dataset_version(identifier, include_examples=False)
            if metadata.get("revision", 1) != request.expected_revision:
                raise DatasetConflict("数据集已更新，请刷新后重新预览")
            branch = "group_path" in metadata
            if branch:
                info = metadata["splitting"]
                if (plan_id, plan_revision) != (info["plan_id"], info["plan_revision"]):
                    raise DatasetConflict("划分方案已更新，请重新打开后调整")
                validate_tree(request.tree, request.expected_revision)
                members = workspace.subsets.plan_members(metadata["parent_id"], plan_id, plan_revision, workspace.list("datasets", include_examples=False), workspace.subsets.load())
                tree = replace_group_tree(metadata, request.tree, members)
                request = SplitTreeRequest(expected_revision=info["source_revision"], tree=tree)
                metadata = workspace.dataset_version(info["source_dataset_id"], info["source_revision"], include_examples=False)
            names = validate_tree(request.tree, request.expected_revision)
            existing = workspace.split_names_in_use(metadata["id"], plan_id, plan_revision)
            data = workspace.read_dataset(metadata["id"], request.expected_revision)
        if existing.intersection(names):
            raise ValueError("结果名称与已有数据集重复，请使用其他名称")
        parts, result = execute_tree(data, request, metadata["columns"], groups)
        return metadata, parts, result, request, branch

    @app.post("/api/datasets/{identifier}/split-tree/preview")
    def tree_preview(identifier: str, request: SplitTreeRequest, plan_id: Optional[str] = None, plan_revision: Optional[int] = Query(None, ge=1)):
        return tree_input(identifier, request, plan_id, plan_revision)[2]

    @app.post("/api/datasets/{identifier}/split-tree/apply")
    def tree_apply(identifier: str, request: SplitTreeRequest, plan_id: Optional[str] = None, plan_revision: Optional[int] = Query(None, ge=1)):
        groups = []
        metadata, parts, result, request, branch = tree_input(identifier, request, plan_id, plan_revision, groups)
        config = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        result["datasets"] = workspace.add_split_datasets(parts, source=metadata, config=config, excluded_rows=result["excluded_rows"], paths={p["name"]: p["path"] for p in result["parts"]}, plan_id=plan_id, plan_revision=plan_revision, groups=groups, pinned=branch)
        return result

    @app.post("/api/datasets/{identifier}/split/apply")
    def splitting_apply(identifier: str, request: SplitRequest, plan_id: Optional[str] = None, plan_revision: Optional[int] = Query(None, ge=1)):
        metadata, data, parts = splitting_input(identifier, request, plan_id, plan_revision)
        result = split_preview(data, parts, request)
        config = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        result["datasets"] = workspace.add_split_datasets(parts, source=metadata, config=config, excluded_rows=result["excluded_rows"], plan_id=plan_id, plan_revision=plan_revision)
        return result

    @app.delete("/api/datasets/{identifier}/split-plans/{plan_id}")
    def remove_plan(identifier: str, plan_id: str, plan_revision: int = Query(..., ge=1)):
        workspace.subsets.remove_plan(identifier, plan_id, plan_revision)
        return {"message": "已移除划分方案及当前子集"}

    @app.post("/api/datasets/{identifier}/processing/preview")
    def processing_preview(identifier: str, request: ProcessingRequest):
        metadata, data = processing_input(identifier, request)
        result, plan, stats = process_frame(data, request.rules)
        return preview_result(data, result, plan, stats, metadata.get("revision", 1))

    @app.post("/api/datasets/{identifier}/processing/parse")
    def processing_parse(identifier: str, request: ExpressionParseRequest):
        metadata = workspace.get("datasets", identifier)
        if metadata.get("revision", 1) != request.expected_revision:
            raise DatasetConflict("数据集已更新，请重新打开字段加工后编辑")
        fields = {c["name"]: "boolean" if "bool" in c["dtype"].lower() else "numeric" if c["kind"] == "numeric" else "date" if c["kind"] == "date" else "string" for c in metadata["columns"]}
        fields, _, _, _ = compile_rules(fields, request.prior_rules)
        return parse_expression_text(request.text, fields)

    @app.post("/api/datasets/{identifier}/processing/apply")
    def processing_apply(identifier: str, request: ProcessingRequest):
        metadata, data = processing_input(identifier, request)
        result, plan, stats = process_frame(data, request.rules)
        processing = {"source_dataset_id": identifier, "source_revision": request.expected_revision, "rules": request.rules, "plan": plan, "statistics": stats, "description": f"字段加工：{len(request.rules)} 个步骤", "engine": "NumExprDerive", "at": now()}
        if request.save_as:
            if not request.name or not request.name.strip():
                raise ValueError("另存时请填写数据集名称")
            return workspace.add_dataset(result, request.name, processing=processing, defaults=metadata)
        return workspace.replace_dataset(identifier, result, expected_revision=request.expected_revision, processing=processing)

    @app.post("/api/datasets/{identifier}/processing/restore")
    def processing_restore(identifier: str, request: RestoreRequest):
        with workspace.lock:
            template = workspace.dataset_version(identifier, request.revision)
            data = workspace.read_dataset(identifier, request.revision)
        return workspace.replace_dataset(identifier, data, expected_revision=request.expected_revision, processing={"description": f"恢复到版本 {request.revision}", "restored_revision": request.revision, "rules": [], "at": now()}, template=template)

    def prepare_job(config, source):
        config["dataset_revision"] = source.get("revision", 1)
        data = workspace.read_dataset(config["dataset_id"], config["dataset_revision"])
        if config["kind"] == "ruleset":
            from .serving import validated_rules
            from hscredit.core.rules.rule import get_columns_from_query

            config["features"] = sorted({c for rule in validated_rules(config["selected_rules"]) for c in get_columns_from_query(rule.expr)})
        if config.get("workflow") and config["kind"] == "model" and not config["features"]:
            excluded = set(source_columns(config))
            config["features"] = [c for c in source.get("default_features", data.columns) if c not in excluded]
            if len(config["features"]) > 200:
                raise ValueError("默认特征超过 200 个，请添加字段选择参数")
        validate_job(data, config)
        if config.get("psi_dataset"):
            if config["kind"] != "explore":
                raise ValueError("PSI 对照样本仅用于字段概况")
            config["psi_dataset"], _ = bind_psi_dataset(workspace, source, config["psi_dataset"], source_columns(config))
        bound_ids = {config["dataset_id"]}
        for binding in [*config["evaluation_datasets"], *([config["test_dataset"]] if config["test_dataset"] else [])]:
            if set(binding) - {"id", "revision"} or not binding.get("id"):
                raise ValueError("数据集绑定格式不正确")
            if not isinstance(binding["id"], str) or binding["id"] in bound_ids:
                raise ValueError("训练、测试与评估不能重复绑定同一数据集")
            bound_ids.add(binding["id"])
            version = workspace.dataset_version(binding["id"], binding.get("revision"))
            binding["revision"] = version.get("revision", 1)
            evaluation = workspace.read_dataset(binding["id"], binding["revision"])
            required = list(dict.fromkeys([*config["features"], *source_columns(config), *[config[k] for k in ("date", "group", "amount") if config[k]]]))
            missing = [col for col in required if col not in evaluation]
            if missing:
                raise ValueError(f"数据集「{version['name']}」缺少字段：{'、'.join(missing)}")
            try:
                labels = label_map(evaluation, config, evaluation=True)
                if config["kind"] == "model" and binding is config["test_dataset"]:
                    training_label(labels)
            except ValueError as exc:
                raise ValueError(f"数据集「{version['name']}」：{exc}") from exc
        if config["kind"] == "explain":
            from .serving import job_model_path

            job_model_path(workspace, config["source_job"], config["model_file"])
        return {"id": uuid4().hex, "kind": config["kind"], "dataset_id": source["id"], "dataset_revision": config["dataset_revision"], "dataset_name": source["name"], "status": "排队中", "progress": 0, "message": "等待执行", "created_at": now(), "config": config, "artifacts": []}

    def create_jobs(request, bindings, *, batch=False):
        with workspace.lock:
            if manager.closed:
                raise HTTPException(503, "任务执行器正在关闭，请重启服务后重试")
            pending = sum(job["status"] in ACTIVE_STATUSES for job in workspace.list("jobs"))
            if pending + len(bindings) > 10:
                raise HTTPException(429, f"队列当前可接收 {max(0, 10-pending)} 个任务，本次需要 {len(bindings)} 个，请稍后提交或减少所选数据集")
            jobs = []
            batch_id = uuid4().hex if batch else None
            for index, (identifier, revision) in enumerate(bindings):
                config = request.model_dump() if hasattr(request, "model_dump") else request.dict()
                config["dataset_id"] = identifier
                source = workspace.dataset_version(identifier, revision)
                try:
                    metadata = prepare_job(config, source)
                except ValueError as exc:
                    if batch:
                        raise ValueError(f"数据集「{source['name']}」：{exc}") from exc
                    raise
                if batch:
                    metadata.update(batch_id=batch_id, batch_position=index)
                jobs.append(metadata)

            # 全部样本通过校验并写完任务清单后，再唤醒工作进程。
            directories = []
            try:
                for job in jobs:
                    directory = workspace.path("jobs", job["id"])
                    directory.mkdir()
                    directories.append(directory)
                    write_json(directory / "metadata.json", job)
            except Exception:
                for directory in directories:
                    shutil.rmtree(directory)
                raise
            for job in jobs:
                manager.enqueue(job["id"])
            return jobs

    @app.post("/api/jobs/batch", status_code=202)
    def submit_batch(request: BatchJobRequest):
        bindings = [(item.id, item.revision) for item in request.datasets]
        if len({identifier for identifier, _ in bindings}) != len(bindings):
            raise ValueError("同一数据集不能在一批任务中重复选择")
        if request.config.dataset_id != bindings[0][0]:
            raise ValueError("配置使用的数据集必须是第一个选中的数据集")
        return {"jobs": create_jobs(request.config, bindings, batch=True)}

    @app.post("/api/jobs", status_code=202)
    def submit(request: JobRequest):
        return create_jobs(request, [(request.dataset_id, request.dataset_revision)])[0]

    @app.get("/api/jobs")
    def jobs():
        return workspace.list("jobs")

    @app.post("/api/jobs/{identifier}/stop", status_code=202)
    def stop_job(identifier: str):
        return manager.stop(identifier)

    @app.get("/api/jobs/{identifier}")
    def job(identifier: str):
        metadata = workspace.get("jobs", identifier)
        result = workspace.path("jobs", identifier) / "result.json"
        if metadata["status"] in {"成功", "失败"} and result.exists():
            metadata["result"] = json.loads(result.read_text(encoding="utf-8"))
            if legacy_report(metadata):
                metadata["artifacts"] = [*metadata["artifacts"], "report.xlsx"]
        return metadata

    @app.post("/api/jobs/{identifier}/pivot")
    def job_pivot(identifier: str, request: PivotRequest):
        return pivot_table(workspace, job(identifier), request)

    report_lock = Lock()

    def legacy_report(metadata):
        return metadata["status"] == "成功" and not any(name.endswith(".xlsx") for name in metadata["artifacts"])

    def artifact_path(identifier, filename):
        metadata = workspace.get("jobs", identifier)
        directory = workspace.path("jobs", identifier)
        if filename == "report.xlsx" and legacy_report(metadata):
            from .job_reports import write_result_workbook

            path = directory / "history-report.xlsx"
            with report_lock:
                if not path.is_file():
                    write_result_workbook(directory, json.loads((directory / "result.json").read_text(encoding="utf-8")), path.name)
            return path
        elif metadata["status"] != "成功" or filename not in metadata["artifacts"]:
            raise HTTPException(404, "该任务没有可下载的文件")
        path = directory / filename
        if not path.is_file():
            raise HTTPException(404, "任务文件不存在")
        return path

    def workbook_path(identifier, filename):
        if not filename.endswith(".xlsx"):
            raise HTTPException(404, "请选择 Excel 报告")
        return artifact_path(identifier, filename)

    @app.get("/api/jobs/{identifier}/workbooks/{filename}")
    def preview_workbook(identifier: str, filename: str, sheet: Optional[str] = None, page: int = Query(1, ge=1), column_page: int = Query(1, ge=1), first_col: Optional[int] = Query(None, ge=1, le=16384), column_count: int = Query(30, ge=1, le=64), include_catalog: bool = True):
        from .job_reports import workbook_page

        return workbook_page(workbook_path(identifier, filename), sheet, page, column_page, first_col=first_col, column_count=column_count, include_catalog=include_catalog)

    @app.get("/api/jobs/{identifier}/workbooks/{filename}/images/{index}")
    def preview_workbook_image(identifier: str, filename: str, index: int, sheet: str):
        from .job_reports import workbook_image

        data, media_type = workbook_image(workbook_path(identifier, filename), sheet, index)
        return Response(data, media_type=media_type)

    @app.get("/api/jobs/{identifier}/files/{filename}")
    def download(identifier: str, filename: str, inline: bool = False):
        return FileResponse(artifact_path(identifier, filename), filename=filename, content_disposition_type="inline" if inline and filename.endswith(".png") else "attachment")

    from .serving import install_serving

    calls = install_serving(app, workspace)
    static = Path(__file__).resolve().parent.parent / "resources" / "platform"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="平台页面")
    else:

        @app.get("/")
        def missing_frontend():
            raise HTTPException(503, "缺少前端资源，请在 platform-web 中执行 npm ci && npm run build")

    return app
