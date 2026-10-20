"""领域操作共用的输入装配、参数校验和制品交付。"""

import inspect
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ...core.models._contracts import extract_target
from ...utils.io import save_pickle
from ...utils.serialization import ARTIFACT_FORMAT, ARTIFACT_VERSION
from ..artifacts import summarize_dataframe
from ..errors import SkillExecutionError
from .binning import _write_workbook


def invalid(message, field=None):
    return SkillExecutionError(code="SCHEMA_INVALID", message=message, field=field)


def input_value(context, name, required=True):
    if name not in context.request.inputs:
        if required:
            raise invalid(f"缺少 inputs.{name}", f"inputs.{name}")
        return None
    return context.resolver.resolve(context.request.inputs[name])


def options(context, allowed):
    params = dict(context.request.parameters)
    unknown = set(params) - set(allowed)
    if unknown:
        raise invalid(f"不支持的操作参数：{sorted(unknown)}", "parameters")
    return params


def resolve(context, value):
    """递归解析显式输入引用和白名单组件；普通 JSON 对象保持原有结构。"""
    if isinstance(value, Mapping):
        if set(value) == {"input"}:
            return input_value(context, value["input"])
        if "component" in value:
            from .components import create_component

            if set(value) - {"component", "params"}:
                raise invalid("组件只接受 component 与 params")
            return create_component(context, value["component"], value.get("params", {}))
        return {key: resolve(context, item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(context, item) for item in value]
    return value


def bind(function, arguments):
    """在执行前检查命名参数，避免参数拼错后出现部分副作用。"""
    try:
        inspect.signature(function).bind(**arguments)
    except TypeError as exc:
        raise invalid(f"参数与 {function.__name__} 不匹配：{exc}", "parameters") from exc
    return arguments


def dataset(context, params, require_y=True, model=None):
    frame = input_value(context, "data")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise invalid("inputs.data 必须是非空 DataFrame", "inputs.data")
    target = params.get("target", getattr(model, "target", None) or "target")
    y = input_value(context, "y", required=False)
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    X, y = extract_target(frame, y, target, require_y=require_y)
    features = params.get("features")
    if features is None and model is not None:
        features = getattr(model, "feature_names_in_", None)
    if features is not None:
        if isinstance(features, str):
            features = [features]
        features = list(features)
        if target in features:
            raise invalid("目标字段不能加入 features", "parameters.features")
        missing = [name for name in features if name not in X]
        if missing:
            raise SkillExecutionError(code="COLUMN_MISSING", message=f"缺少特征：{missing}")
        X = X.loc[:, features]
    return X, y


def frame(value, index=None, label="结果"):
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame(name=value.name or label)
    values = np.asarray(value)
    if values.ndim == 1:
        return pd.DataFrame({label: values}, index=index)
    if values.ndim == 2:
        return pd.DataFrame(values, index=index, columns=[f"{label}{i + 1}" for i in range(values.shape[1])])
    raise invalid("结果必须为一维或二维表格")


def tables(context, values):
    _write_workbook(context, list(values.items()))


def table(context, value, label="数据"):
    value = frame(value, label=label)
    tables(context, {label: value})
    return {"summary": summarize_dataframe(value)}


def json_artifact(context, value, suffix=""):
    name = f"{context.request.output.name}{suffix}.json"
    context.artifacts.publish(context.artifacts.write_json(name, value), name, artifact_type="json")


def artifact(context, value, suffix=""):
    """无混入类的 Selector/Rule 同样使用可验证 hscredit 制品信封。"""
    name = f"{context.request.output.name}{suffix}.joblib"
    path = context.artifacts.stage_path(name)
    if hasattr(value, "save_artifact"):
        value.save_artifact(path)
    else:
        save_pickle(
            {
                "format": ARTIFACT_FORMAT,
                "version": ARTIFACT_VERSION,
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "object": value,
            },
            path,
            engine="joblib",
        )
    context.artifacts.publish(path, name, artifact_type="hscredit-artifact")
