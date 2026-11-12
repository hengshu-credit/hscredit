"""模型与 RuleSet 服务发布、独立令牌鉴权及接口说明。"""

import ast
import hashlib
import hmac
import secrets
import shutil
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, conint
from typing import Any, Dict, List, Literal, Optional

from .storage import now, write_json
from .service_calls import RecordPrediction, ServiceCalls


def validated_rules(items):
    """HTTP 输入只开放 Rule 的声明式条件子集，拒绝任意函数与属性调用。"""
    from hscredit.core.rules import Rule
    from hscredit.core.rules.rule import _replace_backtick_columns

    if not items or len(items) > 200:
        raise ValueError("规则集需要 1 到 200 条规则")
    rules, names = [], set()
    allowed = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.And, ast.Or, ast.Not, ast.BitAnd, ast.BitOr, ast.Invert, ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Call, ast.Attribute)
    for item in items:
        name, expression = item.get("name", ""), item.get("expression", "")
        if not name.strip() or name in names or len(name) > 100 or not expression or len(expression) > 20000:
            raise ValueError("规则名称不能为空或重复，表达式长度不能超过 20000")
        parsed, _ = _replace_backtick_columns(expression)
        try:
            tree = ast.parse(parsed, mode="eval")
        except SyntaxError as exc:
            raise ValueError("规则表达式语法不正确") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > 2000:
            raise ValueError("规则表达式过于复杂")
        for node in nodes:
            if not isinstance(node, allowed):
                raise ValueError("规则仅支持字段比较、集合、空值判断和布尔组合")
            if isinstance(node, ast.Attribute) and (node.attr not in {"isna", "notna", "isin", "between"} or not isinstance(node.value, ast.Name)):
                raise ValueError("规则不允许访问该属性")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Attribute) or node.keywords):
                raise ValueError("规则不允许调用函数")
        names.add(name)
        rules.append(Rule(expression, name=name, n_jobs=1))
    return rules


def job_model_path(workspace, identifier, filename):
    metadata = workspace.get("jobs", identifier)
    if metadata["status"] != "成功" or filename not in metadata["artifacts"] or not filename.startswith("model") or not filename.endswith(".joblib") or "/" in filename or "\\" in filename:
        raise ValueError("请选择成功任务中可下载的模型文件")
    return workspace.path("jobs", identifier) / filename


class ServiceRequest(BaseModel):
    class Config:
        extra = "forbid"
    name: str = Field(min_length=1, max_length=100)
    source_job: Optional[str] = None
    model_file: str = "model.joblib"
    rules: List[Dict[str, str]] = Field(default_factory=list, max_items=200)
    logic: Literal["or", "and"] = "or"


class PredictRequest(BaseModel):
    class Config:
        extra = "forbid"
    records: List[Dict[str, Any]] = Field(min_items=1, max_items=10000)


class ServiceState(BaseModel):
    class Config:
        extra = "forbid"
    enabled: bool


class CallRetention(BaseModel):
    class Config:
        extra = "forbid"

    retention_days: Optional[conint(strict=True, ge=1, le=36500)] = Field(...)


def install_serving(app, workspace):
    calls = ServiceCalls(workspace)

    def public(metadata):
        return {k: v for k, v in metadata.items() if k != "token_hash"}

    def publish(name, model, *, source=None, input_fields=None, threshold=None):
        if not callable(getattr(model, "predict", None)):
            raise ValueError("文件不包含可预测的模型")
        identifier, token = uuid4().hex, secrets.token_urlsafe(32)
        directory = workspace.path("services", identifier)
        directory.mkdir()
        try:
            joblib.dump(model, directory / "model.joblib")
            features = list(input_fields if input_fields is not None else getattr(model, "feature_names_in_", []))
            if not features and hasattr(model, "rules"):
                from hscredit.core.rules.rule import get_columns_from_query

                def dependencies(rule):
                    if hasattr(rule, "expr"):
                        return get_columns_from_query(rule.expr)
                    return [c for r in rule.rules for c in dependencies(r)]

                components = model.rules if isinstance(model.rules, list) else [model.rules]
                features = sorted(set(c for rule in components for c in dependencies(rule)))
            metadata = {"id": identifier, "name": name, "created_at": now(), "enabled": True, "features": [str(f) for f in features], "source": source, "threshold": threshold, "endpoint": f"/services/{identifier}/predict", "token_hash": hashlib.sha256(token.encode()).hexdigest()}
            write_json(directory / "metadata.json", metadata)
        except Exception:
            shutil.rmtree(directory)
            raise
        return {**public(metadata), "token": token}

    def authenticate(identifier, authorization):
        metadata = workspace.get("services", identifier)
        candidate = (authorization or "").removeprefix("Bearer ")
        if not authorization or not authorization.startswith("Bearer ") or not hmac.compare_digest(hashlib.sha256(candidate.encode()).hexdigest(), metadata["token_hash"]):
            raise HTTPException(401, "服务令牌无效", headers={"WWW-Authenticate": "Bearer"})
        if not metadata["enabled"]:
            raise HTTPException(503, "服务已下线")
        return metadata

    @app.get("/api/services")
    def services():
        return [public(item) for item in workspace.list("services")]

    @app.get("/api/services/{identifier}/calls")
    def list_calls(identifier: str, page: int = Query(1, ge=1, le=2147483647), page_size: int = Query(20, ge=1, le=100),
                   failed_only: bool = False, call_id: str = Query("", max_length=32)):
        workspace.get("services", identifier)
        return calls.page(identifier, page, page_size, failed_only, call_id)

    @app.put("/api/services/{identifier}/retention")
    def save_retention(identifier: str, request: CallRetention):
        workspace.get("services", identifier)
        calls.set_retention(identifier, request.retention_days)
        return {"retention_days": request.retention_days}

    @app.get("/api/services/{identifier}/calls/{call_id}")
    def call_detail(identifier: str, call_id: str):
        workspace.get("services", identifier)
        return calls.detail(identifier, call_id)

    @app.post("/api/services", status_code=201)
    def create_service(request: ServiceRequest):
        from hscredit.core.models import RulesClassifier, RuleSet

        if bool(request.source_job) == bool(request.rules):
            raise ValueError("请选择任务模型或填写 RuleSet，两种来源只能选择一种")
        if request.source_job:
            model = joblib.load(job_model_path(workspace, request.source_job, request.model_file))
        else:
            model = RulesClassifier(RuleSet(name=request.name, rules=validated_rules(request.rules), logic=request.logic, n_jobs=1), n_jobs=1)
        if request.source_job:
            job = workspace.get("jobs", request.source_job)
            return publish(request.name, model, source=request.source_job, input_fields=job["config"]["features"] if job["kind"] == "model" else None, threshold=job["config"]["threshold"] if job["kind"] == "model" else None)
        return publish(request.name, model)

    @app.post("/api/services/upload", status_code=201)
    def upload_model(file: UploadFile = File(...), name: str = Form(...), trusted: bool = Form(False)):
        if not trusted:
            raise ValueError("模型反序列化会执行文件中的代码，仅可导入您信任的 hscredit 模型")
        if not name.strip() or len(name) > 100:
            raise ValueError("请填写 1 到 100 字的服务名称")
        content = file.file.read(100 * 1024 * 1024 + 1)
        if len(content) > 100 * 1024 * 1024:
            raise ValueError("模型文件不能超过 100 MB")
        import io

        try:
            model = joblib.load(io.BytesIO(content))
        except Exception as exc:
            raise ValueError("无法读取模型，请上传 hscredit 导出的 pickle/joblib 推理对象") from exc
        return publish(name, model, source="手工上传")

    @app.patch("/api/services/{identifier}")
    def update_service(identifier: str, request: ServiceState):
        return public(workspace.update("services", identifier, enabled=request.enabled))

    @app.post("/api/services/{identifier}/rotate-token")
    def rotate_token(identifier: str):
        token = secrets.token_urlsafe(32)
        workspace.update("services", identifier, token_hash=hashlib.sha256(token.encode()).hexdigest())
        return {"token": token}

    @app.get("/api/services/{identifier}/schema")
    def service_schema(identifier: str):
        metadata = public(workspace.get("services", identifier))
        schema = PredictRequest.model_json_schema() if hasattr(PredictRequest, "model_json_schema") else PredictRequest.schema()
        schema["properties"]["records"]["items"].update(properties={f: {} for f in metadata["features"]}, required=metadata["features"])
        operation = {
            "summary": "批量预测", "security": [{"serviceToken": []}],
            "requestBody": {"required": True, "content": {"application/json": {"schema": schema, "example": {"records": [{f: 0 for f in metadata["features"]}]}}}},
            "responses": {
                "200": {"description": "与输入顺序相同；概率按 classes 顺序排列", "content": {"application/json": {"schema": {
                    "type": "object", "required": ["service_id", "classes", "predictions"],
                    "properties": {"service_id": {"type": "string"}, "classes": {"type": ["array", "null"], "items": {}},
                                   "predictions": {"type": "array", "items": {"type": "object", "required": ["预测"], "properties": {
                                       "预测": {}, "概率": {"type": "array", "items": {"type": "number"}}, "评分": {"type": "number"}
                                   }}}}
                }}}},
                "401": {"description": "服务令牌无效"}, "422": {"description": "字段或数据类型不符合模型要求"}, "503": {"description": "服务已下线"}
            }
        }
        for response in operation["responses"].values():
            response["headers"] = {"X-Request-ID": {"description": "在平台调用记录中查询本次请求的编号", "schema": {"type": "string"}}}
        return {"openapi": "3.1.0", "info": {"title": metadata["name"], "version": "1.0.0"}, "paths": {metadata["endpoint"]: {"post": operation}}, "components": {"securitySchemes": {"serviceToken": {"type": "http", "scheme": "bearer"}}}, "x-required-features": metadata["features"]}

    # 公开推理与文档共用启动端口；管理端不安装 CORS。
    from fastapi.middleware.cors import CORSMiddleware

    inference = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, exception_handlers=app.exception_handlers.copy())
    inference.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"], expose_headers=["X-Request-ID"])
    app.mount("/services", RecordPrediction(inference, workspace, calls))

    @inference.get("/{identifier}/openapi.json")
    def external_schema(identifier: str, authorization: Optional[str] = Header(None)):
        authenticate(identifier, authorization)
        return service_schema(identifier)

    @inference.post("/{identifier}/predict")
    def predict(identifier: str, request: PredictRequest, authorization: Optional[str] = Header(None)):
        metadata = authenticate(identifier, authorization)
        if any(set(metadata["features"]) - set(row) for row in request.records):
            raise HTTPException(422, "样本缺少模型所需字段，请查看接口文档")
        frame = pd.DataFrame(request.records)
        if metadata["features"]:
            frame = frame.loc[:, metadata["features"]]
        # 每次加载独立对象，避免 Rule 等有预测状态的模型被并发请求改写。
        model = joblib.load(workspace.path("services", identifier) / "model.joblib")
        try:
            prediction = np.asarray(model.predict(frame))
            result = pd.DataFrame({"预测": prediction})
            classes = getattr(model, "classes_", None)
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(frame))
                result["概率"] = probabilities.tolist()
                if classes is None and probabilities.shape[1] == 2 and metadata.get("threshold") is not None:
                    classes = [0, 1]
                if metadata.get("threshold") is not None:
                    result["预测"] = (probabilities[:, list(classes).index(1)] >= metadata["threshold"]).astype(int)
            if hasattr(model, "predict_score"):
                result["评分"] = np.asarray(model.predict_score(frame))
            return {"service_id": identifier, "classes": np.asarray(classes).tolist() if classes is not None else None, "predictions": result.to_dict(orient="records")}
        except Exception as exc:
            raise HTTPException(422, f"输入数据无法预测：{exc}") from exc

    return calls
