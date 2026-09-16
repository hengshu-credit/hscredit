"""记录训练过程和运行环境，训练失败时保留可诊断的状态。"""

import copy
from datetime import datetime, timezone
from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter


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


def record_training(method):
    """保留每次训练配置、划分、曲线和异常；不复制原始训练数据。"""

    @wraps(method)
    def fit(self, X, *args, **kwargs):
        record = {
            "开始时间": datetime.now(timezone.utc).isoformat(),
            "样本数": X.shape[0] if hasattr(X, "shape") else len(X),
            "依赖版本": dict(dependency_versions()),
            "状态": "训练中",
        }
        history = self.__dict__.setdefault("training_history_", [])
        history.append(record)
        self.training_summary_ = record
        self._is_fitted = False
        self._feature_importances = None
        self._best_iteration = None
        self._best_score = None
        self._evals_result = {}
        for name in ("_eval_train_indices_", "_eval_val_indices_"):
            self.__dict__.pop(name, None)
        start = perf_counter()
        try:
            record["参数"] = copy.deepcopy(self.get_params(deep=False))
            record["训练参数"] = {name: _fit_parameter_snapshot(value) for name, value in kwargs.items()}
            result = method(self, X, *args, **kwargs)
            record["状态"] = "完成"
            self._is_fitted = True
            return result
        except BaseException as exc:
            self._is_fitted = False
            record.update(
                状态="中断" if isinstance(exc, KeyboardInterrupt) else "失败",
                错误类型=type(exc).__name__,
                错误信息=str(exc),
            )
            raise
        finally:
            record["耗时秒"] = perf_counter() - start
            record["评估曲线"] = copy.deepcopy(getattr(self, "_evals_result", {}))
            record["最佳迭代"] = getattr(self, "_best_iteration", None)
            record["最佳得分"] = copy.deepcopy(getattr(self, "_best_score", None))
            for name, label in (("_eval_train_indices_", "训练位置"), ("_eval_val_indices_", "验证位置")):
                if hasattr(self, name):
                    record[label] = getattr(self, name).tolist()

    return fit


def _fit_parameter_snapshot(value):
    """Pool 持有不可复制的原生句柄；记录其标签、权重和结构，不阻断模型训练。"""
    if type(value).__module__.startswith("catboost") and type(value).__name__ == "Pool":
        return {
            "容器类型": "CatBoost.Pool",
            "样本数": value.num_row(),
            "特征名": value.get_feature_names(),
            "类别特征位置": value.get_cat_feature_indices(),
            "标签": value.get_label(),
            "权重": value.get_weight(),
            "基线": value.get_baseline(),
        }
    if isinstance(value, list):
        return [_fit_parameter_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_fit_parameter_snapshot(item) for item in value)
    if isinstance(value, dict):
        return {name: _fit_parameter_snapshot(item) for name, item in value.items()}
    return copy.deepcopy(value)
