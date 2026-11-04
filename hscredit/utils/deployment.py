"""模型部署包的原子发布、字段清单、摘要校验和预测往返验收。"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from importlib.metadata import PackageNotFoundError
from .artifacts import ArtifactTransaction
from .serialization import load_verified_pickle, unpack_artifact
from .json import json_safe
from .environment import runtime_version
from ..exceptions import SerializationError


def _deployment_components(model):
    """只遍历组件关系，不扫描训练矩阵或捕获的业务数据。"""
    seen = set()
    components = []

    def visit(value):
        if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif hasattr(value, "get_params") or callable(getattr(value, "transform", None)) or isinstance(getattr(value, "_modules", None), dict) or type(value).__module__.split(".")[0] in {"torch", "lightgbm", "catboost", "xgboost", "tabm", "rtdl_revisiting_models"}:
            components.append(value)
            state = vars(value) if hasattr(value, "__dict__") else {}
            for name in (
                "_model",
                "model",
                "model_",
                "base_model",
                "calibrator_",
                "lr_model_",
                "encoder_",
                "binner",
                "binner_",
                "transformer_",
                "score_transformer_",
                "scorecard_",
                "steps",
                "_network",
                "network",
                "network_",
                "_modules",
                "preprocessor_",
                "_preprocessor",
                "pipeline",
                "pipeline_",
                "scaler_",
                "imputer_",
            ):
                if name in state:
                    visit(state[name])

    visit(model)
    return components


def _deployment_metadata(model, base):
    """记录使用到的后端依赖及前处理身份，不要求安装无关可选框架。"""
    import cloudpickle

    fingerprints = {}

    def fingerprint(component):
        if id(component) in fingerprints:
            return fingerprints[id(component)]
        digest = hashlib.sha256()

        class DigestWriter:
            def write(self, data):
                digest.update(data)
                return len(data)

        cloudpickle.dump(component, DigestWriter())
        fingerprints[id(component)] = digest.hexdigest()
        return fingerprints[id(component)]

    packages = set(base["dependencies"]) | {"cloudpickle", "joblib"}
    mapping = {
        "sklearn": "scikit-learn",
        "rtdl_revisiting_models": "rtdl-revisiting-models",
        "rtdl_num_embeddings": "rtdl-num-embeddings",
        "pytorch_tabnet": "pytorch-tabnet",
    }
    components = _deployment_components(model)
    for component in components:
        name = type(component).__module__.split(".")[0]
        if name not in ("builtins", "hscredit"):
            packages.add(mapping.get(name, name))
    dependencies = {}
    for package in sorted(packages):
        try:
            dependencies[package] = runtime_version(package)
        except PackageNotFoundError:
            continue
    identities = {}
    for name in ("binner_", "binner", "encoder_", "calibrator_", "transformer_", "score_transformer_"):
        component = getattr(model, name, None)
        if component is not None:
            identities[name] = {
                "class": f"{type(component).__module__}.{type(component).__qualname__}",
                "fingerprint": fingerprint(component),
            }
    for index, component in enumerate(components):
        if callable(getattr(component, "transform", None)):
            identities[f"transform[{index}]"] = {
                "class": f"{type(component).__module__}.{type(component).__qualname__}",
                "fingerprint": fingerprint(component),
            }
    return dependencies, identities


def save_deployment(model, directory, *, name="model", validation_data=None, prediction_method=None, overwrite=False):
    """发布推理制品和部署清单；可提供验证数据做保存前后的预测一致性检查。"""
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_is_fitted

    if isinstance(model, BaseEstimator) and callable(getattr(model, "fit", None)):
        check_is_fitted(model)
    method = prediction_method or ("predict_proba" if callable(getattr(model, "predict_proba", None)) else "predict" if callable(getattr(model, "predict", None)) else "transform")
    if method == "predict_prob":
        method = "predict_proba"
    if method not in ("predict", "predict_proba", "predict_score", "transform") or not callable(getattr(model, method, None)):
        raise ValueError("prediction_method 必须是模型支持的 predict、predict_proba、predict_score 或 transform")
    metadata = model.get_artifact_metadata()
    dependencies, component_identities = _deployment_metadata(model, metadata)
    features = metadata.get("features")
    schema = getattr(model, "feature_schema_", getattr(model, "_feature_schema", None))
    dtypes = getattr(schema, "dtypes", ())
    verification = {"已验证": False}
    with ArtifactTransaction({"directory": directory, "name": name, "overwrite": overwrite}) as transaction:
        path = transaction.stage_path("model.pkl")
        inference = callable(getattr(model, "save_inference", None))
        if inference:
            model.save_inference(path)
        else:
            model.save_artifact(path, engine="cloudpickle")
        loaded = unpack_artifact(load_verified_pickle(path), expected_type=type(model), metadata_policy="ignore")
        if validation_data is not None:
            expected = np.asarray(getattr(model, method)(validation_data))
            actual = np.asarray(getattr(loaded, method)(validation_data))
            if expected.shape != actual.shape:
                raise SerializationError("部署制品往返后的预测形状不同")
            if np.issubdtype(expected.dtype, np.number):
                equal = np.allclose(expected, actual, rtol=1e-10, atol=1e-12, equal_nan=True)
            else:
                equal = np.array_equal(expected, actual)
            if not equal:
                raise SerializationError("部署制品往返后的预测不一致，未发布")
            verification = {
                "已验证": True,
                "样本数": len(validation_data),
                "输出形状": list(actual.shape),
                "输出摘要": hashlib.sha256(json.dumps(json_safe(actual), ensure_ascii=False).encode()).hexdigest(),
            }
            if not dtypes and hasattr(validation_data, "dtypes"):
                dtypes = tuple(validation_data.loc[:, features].dtypes) if features else tuple(validation_data.dtypes)
        manifest = {
            "format": "hscredit-deployment",
            "version": 1,
            "class": metadata["class"],
            "prediction_method": method,
            "features": json_safe(features),
            "feature_types": [type(feature).__qualname__ for feature in (features or [])],
            "dtypes": [str(dtype) for dtype in dtypes],
            "classes": json_safe(getattr(model, "classes_", None)),
            "dependencies": dependencies,
            "component_identities": component_identities,
            "inference_only": inference,
            "verification": verification,
            "preprocessing": {key: type(getattr(model, key)).__name__ for key in ("binner_", "binner", "encoder_", "calibrator_", "scorecard_", "transformer_") if getattr(model, key, None) is not None},
        }
        item = transaction.publish(path, "model.pkl", artifact_type="model")
        item["relative_path"] = (Path(transaction.run_dir.name) / "model.pkl").as_posix()
        manifest_path = transaction.write_json("deployment.json", manifest)
        item = transaction.publish(manifest_path, "deployment.json", artifact_type="deployment")
        item["relative_path"] = (Path(transaction.run_dir.name) / "deployment.json").as_posix()
    return str(transaction.manifest_path)


def load_deployment(path, *, expected_type=None, trusted=True, metadata_policy="warn"):
    """按原子完成清单加载部署包；先校验所有文件摘要，再反序列化。"""
    if trusted is not True:
        raise SerializationError("部署包包含对象制品，只能加载可信来源")
    from .serialization import verify_artifact_digest

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("状态") != "完成" or payload.get("协议版本") != 1:
        raise SerializationError("部署包缺少受支持的完成清单")
    entries = payload.get("制品", [])
    files = {}
    for item in entries:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or not item.get("sha256"):
            raise SerializationError("部署制品路径或摘要无效")
        source = (manifest_path.parent / relative).resolve()
        if manifest_path.parent not in source.parents:
            raise SerializationError("部署制品路径超出完成清单所在目录")
        verify_artifact_digest(source, item.get("sha256"))
        if item["type"] in files:
            raise SerializationError("部署完成清单包含重复制品")
        files[item["type"]] = (source, item.get("sha256"))
    if not {"model", "deployment"} <= files.keys():
        raise SerializationError("部署包缺少模型或部署清单")
    config = json.loads(files["deployment"][0].read_text(encoding="utf-8"))
    if config.get("format") != "hscredit-deployment" or config.get("version") != 1:
        raise SerializationError("不支持的部署清单版本")
    obj = unpack_artifact(
        load_verified_pickle(files["model"][0], expected_sha256=files["model"][1]),
        expected_type=expected_type,
        metadata_policy="ignore",
    )
    actual_metadata = obj.get_artifact_metadata()
    if json_safe(actual_metadata.get("features")) != config.get("features"):
        raise SerializationError("部署清单字段与模型不一致")
    if json_safe(getattr(obj, "classes_", None)) != config.get("classes"):
        raise SerializationError("部署清单类别映射与模型不一致")
    if config.get("feature_types") is not None and config["feature_types"] != [type(feature).__qualname__ for feature in (actual_metadata.get("features") or [])]:
        raise SerializationError("部署清单字段类型与模型不一致")
    if not callable(getattr(obj, config.get("prediction_method", ""), None)):
        raise SerializationError("部署清单的预测函数不可用")
    envelope = {
        **obj.get_artifact_metadata(),
        "dependencies": config["dependencies"],
        "class": config["class"],
        "features": actual_metadata.get("features"),
        "object": obj,
    }
    return unpack_artifact(envelope, expected_type=expected_type, metadata_policy=metadata_policy)
