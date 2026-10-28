"""记录训练过程和运行环境，训练失败时保留可诊断的状态。"""

import copy
import inspect
from datetime import datetime, timezone
from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
import numpy as np
import pandas as pd


def restore_model_defaults(model):
    """迁移旧制品缺失的新配置；不猜测或重建任何已学习模型参数。"""
    added_options = {
        "history_policy",
        "max_history",
        "statistics_level",
        "max_dense_bytes",
        "weight_type",
        "loss",
        "loss_params",
        "optimizer",
        "optimizer_params",
        "finetune",
        "finetune_params",
    }
    defaults = {"history_policy": "summary", "max_history": 20}
    for name, parameter in inspect.signature(type(model).__init__).parameters.items():
        if name in added_options and parameter.default is not inspect.Parameter.empty:
            defaults[name] = parameter.default
    for name, value in defaults.items():
        if name not in model.__dict__:
            setattr(model, name, copy.deepcopy(value))
    return model


def validate_history_policy(model):
    """历史保留策略是包装器配置，不能透传为底层框架参数。"""
    policy = getattr(model, "history_policy", "summary")
    limit = getattr(model, "max_history", 20)
    if policy not in ("summary", "diagnostics", "full"):
        raise ValueError("history_policy 必须为 summary、diagnostics 或 full")
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("max_history 必须为正整数")
    return policy, int(limit)


@lru_cache(maxsize=1)
def dependency_versions():
    """记录制品产生时的依赖版本，不导入可选框架。"""
    result = {}
    for package in (
        "hscredit",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "ngboost",
        "optuna",
        "torch",
        "rtdl-revisiting-models",
        "tabpfn",
        "tabm",
        "rtdl-num-embeddings",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def atomic_save_pickle(value, path, **kwargs):
    """完整写入临时文件后再替换目标，序列化失败时保留原文件。"""
    from pathlib import Path
    from uuid import uuid4
    from ...utils import save_pickle

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid4().hex}.{target.name}")
    try:
        save_pickle(value, temporary, **kwargs)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


def record_training(method=None, *, transactional=False, fitted=True):
    """保留每次训练配置、划分、曲线和异常；不复制原始训练数据。"""
    if method is None:
        return lambda wrapped: record_training(wrapped, transactional=transactional, fitted=fitted)

    @wraps(method)
    def fit(self, *args, **kwargs):
        restore_model_defaults(self)
        policy, limit = validate_history_policy(self)
        previous = dict(self.__dict__) if transactional else None
        bound = inspect.signature(method).bind(self, *args, **kwargs)
        first = next(name for name in inspect.signature(method).parameters if name != "self")
        data = bound.arguments.get(first)
        if hasattr(data, "predict_proba"):
            data = bound.arguments.get("X" if first == "model" else "y")  # 兼容校准器 fit(model, X, y)。
        if data is None:
            data = kwargs.get("proba", [])
        record = {
            "开始时间": datetime.now(timezone.utc).isoformat(),
            "样本数": data.shape[0] if hasattr(data, "shape") else len(data),
            "依赖版本": dict(dependency_versions()),
            "状态": "训练中",
            "历史策略": policy,
        }
        history = self.__dict__.setdefault("training_history_", [])
        history.append(record)
        del history[:-limit]
        self.training_summary_ = record
        self.fit_revision_ = getattr(self, "fit_revision_", 0) + 1
        record["训练修订"] = self.fit_revision_
        self._fit_was_fitted_ = bool(getattr(self, "_is_fitted", False))
        self._is_fitted = False
        self._feature_importances = None
        self._best_iteration = None
        self._best_score = None
        self._evals_result = {}
        for name in (
            "_eval_train_indices_",
            "_eval_val_indices_",
            "eval_names_",
            "loss_sample_sizes_",
            "native_params_",
            "eval_results_",
            "transformation_summary_",
        ):
            self.__dict__.pop(name, None)
        start = perf_counter()
        try:
            record["参数"] = _fit_parameter_snapshot(self.get_params(deep=False), policy=policy)
            record["训练参数"] = {name: _fit_parameter_snapshot(value, policy=policy) for name, value in kwargs.items()}
            result = method(self, *args, **kwargs)
            record["状态"] = "完成"
            self._is_fitted = fitted
            return result
        except BaseException as exc:
            self._is_fitted = False
            if previous is not None:
                # 规则模型原有契约：验证失败保留上次已学习状态，但保留本次失败记录。
                keep = {key: self.__dict__[key] for key in ("training_history_", "training_summary_", "fit_revision_")}
                self.__dict__.clear()
                self.__dict__.update(previous)
                self.__dict__.update(keep)
            record.update(
                状态="中断" if isinstance(exc, KeyboardInterrupt) else "失败",
                错误类型=type(exc).__name__,
                错误信息=str(exc),
            )
            raise
        finally:
            self.__dict__.pop("_fit_was_fitted_", None)
            record["耗时秒"] = perf_counter() - start
            # 曲线是已计算的数值诊断，与可能很大的原始训练参数分别保留。
            restored = previous is not None and record["状态"] != "完成"
            record["评估曲线"] = {} if restored else copy.deepcopy(getattr(self, "_evals_result", {}))
            record["实际参数"] = _fit_parameter_snapshot(getattr(self, "native_params_", {}), policy=policy)
            record["最终评估"] = getattr(self, "eval_results_", pd.DataFrame()).to_dict("records")
            if restored:
                record.update(实际参数={}, 最终评估=[], 状态说明="本次拟合失败，保留之前已学习状态")
            if hasattr(self, "transformation_summary_"):
                record["内部转换"] = copy.deepcopy(self.transformation_summary_)
            if hasattr(self, "loss_sample_sizes_"):
                record["金融参数切分"] = dict(self.loss_sample_sizes_)
            if getattr(self, "eval_names_", None):
                record["评估集名称"] = list(self.eval_names_)
            record["最佳迭代"] = getattr(self, "_best_iteration", None)
            record["最佳得分"] = _fit_parameter_snapshot(getattr(self, "_best_score", None), policy=policy)
            for name, label in (("_eval_train_indices_", "训练位置"), ("_eval_val_indices_", "验证位置")):
                if hasattr(self, name):
                    indices = getattr(self, name)
                    if policy == "full":
                        record[label] = indices.tolist()
                    elif policy == "diagnostics":
                        record[label] = {"数量": len(indices)}

    return fit


def _fit_parameter_snapshot(value, *, policy="summary", _depth=0):
    """默认仅记录有界元数据；只有显式 full 才复制逐行数据及任意对象。"""
    if type(value).__module__.startswith("catboost") and type(value).__name__ == "Pool":
        summary = {
            "容器类型": "CatBoost.Pool",
            "样本数": value.num_row(),
            "特征名": value.get_feature_names(),
            "类别特征位置": value.get_cat_feature_indices(),
        }
        if policy == "full":
            summary.update(标签=value.get_label(), 权重=value.get_weight(), 基线=value.get_baseline())
        return summary
    if policy != "full":
        if value is None or isinstance(value, (str, bool, int, float, np.number)):
            return value.item() if isinstance(value, np.generic) else value
        if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)) or hasattr(value, "tocsr"):
            summary = {"容器类型": type(value).__name__, "形状": list(value.shape)}
            if isinstance(value, pd.DataFrame):
                summary["字段数"] = len(value.columns)
                summary["字段"] = [str(column) for column in value.columns[:50]]
                summary["类型"] = [str(dtype) for dtype in value.dtypes.iloc[:50]]
            else:
                summary["类型"] = str(value.dtype)
            return summary
        if _depth >= 5:
            return {"容器类型": type(value).__name__, "摘要": "嵌套深度已截断"}
        capacity = 256 if policy == "diagnostics" else 32
        if isinstance(value, (list, tuple)) and len(value) > capacity:
            return {"容器类型": type(value).__name__, "数量": len(value), "摘要": "未保留逐项明细"}
        if isinstance(value, dict) and len(value) > 256:
            return {"容器类型": "dict", "数量": len(value), "摘要": "未保留逐项明细"}
        if not isinstance(value, (list, tuple, dict)):
            from .losses.base import BaseLoss, BaseMetric

            if isinstance(value, (BaseLoss, BaseMetric)):
                return {
                    "对象类型": type(value).__name__,
                    "配置": _fit_parameter_snapshot(vars(value), policy=policy, _depth=_depth + 1),
                }
            return {"对象类型": f"{type(value).__module__}.{type(value).__qualname__}"}
    if isinstance(value, list):
        return [_fit_parameter_snapshot(item, policy=policy, _depth=_depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_fit_parameter_snapshot(item, policy=policy, _depth=_depth + 1) for item in value)
    if isinstance(value, dict):
        return {name: _fit_parameter_snapshot(item, policy=policy, _depth=_depth + 1) for name, item in value.items()}
    return copy.deepcopy(value)


def training_history_table(model):
    """不在训练摘要表中重复复制逐轮数组。"""
    records = getattr(model, "training_history_", [])
    return pd.DataFrame(
        [
            {
                key: _fit_parameter_snapshot(value)
                for key, value in record.items()
                if key not in {"评估曲线", "最终评估", "训练位置", "验证位置"}
            }
            for record in records
        ]
    )


def learning_curve_table(curves, *, context=None, names=None):
    """将不同框架的曲线转为中文长表，保留原生数据集名以便核对。"""
    rows = []
    aliases = {
        "train": "训练集",
        "learn": "训练集",
        "training": "训练集",
        "val": "验证集",
        "validation": "验证集",
        "internal_validation": "内部验证集",
    }
    for i, name in enumerate(names or []):
        aliases.update({f"valid_{i}": name, f"validation_{i}": name})
    if names:
        aliases["val"] = names[-1]
        aliases["validation"] = names[0]
    for dataset, metrics in curves.items():
        for metric, values in metrics.items():
            if isinstance(values, dict):
                raise ValueError("旧训练记录仅保留曲线摘要，无法还原逐轮数据；请重新训练或使用原始制品")
            if np.isscalar(values):
                continue  # 普通分类器的最终评估值不是逐轮训练曲线。
            for iteration, value in enumerate(values, 1):
                rows.append(
                    {
                        **(context or {}),
                        "数据集": aliases.get(dataset, dataset),
                        "原生数据集": dataset,
                        "迭代": iteration,
                        "指标": metric,
                        "数值": float(value),
                    }
                )
    return pd.DataFrame(rows, columns=[*(context or {}), "数据集", "原生数据集", "迭代", "指标", "数值"])


def model_learning_curve(model, revision=None):
    history = getattr(model, "training_history_", [])
    if not history:
        return learning_curve_table({})
    current = getattr(model, "fit_revision_", len(history))
    revision = current if revision is None else revision
    for offset, record in enumerate(history, current - len(history) + 1):
        if record.get("训练修订", offset) == revision:
            curves = record.get("评估曲线", {})
            if revision == current and any(
                isinstance(values, dict) for dataset in curves.values() for values in dataset.values()
            ):
                curves = getattr(model, "_evals_result", curves)
            return learning_curve_table(curves, context={"训练修订": revision}, names=record.get("评估集名称"))
    raise ValueError(f"未保留训练修订 {revision}，请检查 max_history 或历史制品")
