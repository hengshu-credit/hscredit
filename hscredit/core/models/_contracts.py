"""模型公共契约：字段、标签、概率、样本参数及轻量推理制品。"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted

SAMPLE_PARAMETER_NAMES = frozenset({"sample_weight", "base_margin", "init_score", "baseline", "groups"})


def extract_target(X, y=None, target=None, *, require_y=True):
    """显式 y 优先；目标列无论是否提供 y 都不能成为特征。"""
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("输入特征列名不能重复")
        if target is not None and target in X.columns:
            if y is None:
                y = X[target]
            X = X.drop(columns=[target])
    elif not issparse(X):
        X = np.asarray(X)
    if getattr(X, "ndim", None) != 2:
        raise ValueError("输入特征必须是二维数组或 DataFrame")
    if require_y and y is None:
        raise ValueError("请提供 y，或通过 target 指定数据中的目标列")
    if y is not None and (np.asarray(y).ndim != 1 or len(y) != X.shape[0]):
        raise ValueError("标签必须是一维且与特征样本等长")
    return X, y


def validate_labels(y, *, classes=(0, 1), require_both=True):
    """校验二分类标签域，不展平二维标签，也不把未知标签默认为负类。"""
    values = np.asarray(y)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("标签必须是一维非空数组")
    observed = set(np.unique(values))
    allowed = set(classes)
    if not observed.issubset(allowed) or (require_both and observed != allowed):
        if allowed == {0, 1}:
            raise ValueError("训练标签必须同时包含 0 和 1，且 1 表示坏样本")
        raise ValueError(f"标签必须{'同时包含' if require_both else '属于'} {list(classes)}，不能包含其他标签")
    return values


def validate_sample_weight(sample_weight, n_samples):
    """样本权重按位置对齐，必须有限、非负且总和为正。"""
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1 or len(weights) != n_samples:
        raise ValueError("sample_weight 必须是一维且与样本等长")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight 必须是有限非负数且总和大于0")
    return weights


def take_rows(values, indices):
    """按位置切分，保留 DataFrame 类别类型、索引及稀疏矩阵。"""
    if values is None:
        return None
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices] if issparse(values) else np.asarray(values)[indices]


def split_sample_params(params, indices, n_samples):
    """仅切分已知样本参数（含 Pipeline 前缀），不按长度猜测回调或字段配置。"""
    result = dict(params)
    for name, value in result.items():
        if name.rsplit("__", 1)[-1] not in SAMPLE_PARAMETER_NAMES or value is None:
            continue
        if np.ndim(value) == 0 or len(value) != n_samples:
            raise ValueError(f"训练参数 {name} 必须与完整训练数据等长")
        if indices is not None:
            result[name] = take_rows(value, indices)
    return result


@dataclass(frozen=True)
class FeatureSchema:
    """记录字段顺序与类型；仅在显式重建样本时恢复 dtype。"""

    names: tuple
    dtypes: tuple = ()
    named: bool = True

    @classmethod
    def from_data(cls, X):
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValueError("输入特征列名不能重复")
            return cls(tuple(X.columns), tuple(X.dtypes))
        if getattr(X, "ndim", None) != 2:
            raise ValueError("输入特征必须是二维数组")
        return cls(tuple(f"feature_{i}" for i in range(X.shape[1])), named=False)

    def align(self, X, *, restore_dtypes=False):
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValueError("输入特征列名不能重复")
            if self.named:
                missing = [name for name in self.names if name not in X.columns]
                if missing:
                    raise ValueError(f"输入数据缺少训练字段: {missing}")
                X = X.loc[:, list(self.names)]
            elif X.shape[1] != len(self.names):
                raise ValueError(f"输入特征数量不匹配：训练时为{len(self.names)}，当前为{X.shape[1]}")
            if restore_dtypes and self.dtypes:
                X = X.astype(dict(zip(self.names, self.dtypes)))
            return X
        if not issparse(X):
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("输入特征必须是二维数组")
        if X.shape[1] != len(self.names):
            raise ValueError(f"输入特征数量不匹配：训练时为{len(self.names)}，当前为{X.shape[1]}")
        return X


def record_feature_schema(model, X):
    """记录公共字段契约，并兼容既有 sklearn 风格属性。"""
    schema = FeatureSchema.from_data(X)
    model.feature_schema_ = schema
    model.feature_names_in_ = list(schema.names)
    model.n_features_in_ = len(schema.names)
    model._feature_names_known_ = schema.named
    return schema


def align_model_features(model, X):
    """按已记录字段对齐；旧制品没有 schema 时仍使用原字段元数据。"""
    schema = getattr(model, "feature_schema_", None)
    if schema is None:
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            return X
        schema = FeatureSchema(tuple(names), named=getattr(model, "_feature_names_known_", True))
    return schema.align(X)


def positive_class_index(classes, positive_class=None):
    """按真实 classes_ 查找正类；None 延续二分类第 2 个类别的约定。"""
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(classes) < 2:
        raise ValueError("模型必须提供至少两个类别的 classes_")
    label = classes[1] if positive_class is None else positive_class
    matches = np.flatnonzero(classes == label)
    if len(matches) != 1:
        raise ValueError(f"模型概率列中不存在唯一正类标签: {label!r}")
    return int(matches[0])


def positive_probability(proba, classes=None, positive_class=None):
    """提取指定正类的有限概率；二维概率须与类别数和归一化约束一致。"""
    values = np.asarray(proba, dtype=float)
    if values.ndim not in (1, 2) or values.shape[0] == 0:
        raise ValueError("概率必须是非空的一维数组或二维概率矩阵")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("概率必须是[0, 1]范围内的有限数")
    if values.ndim == 1:
        return values
    classes = np.asarray([0, 1] if classes is None else classes)
    if values.shape[1] != len(classes) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("概率列数必须与 classes_ 一致，且每行概率之和必须为1")
    return values[:, positive_class_index(classes, positive_class)]


def is_model_fitted(model):
    """显式失败状态优先于遗留 coef_/classes_ 等属性。"""
    if model is None:
        return False
    if hasattr(model, "_is_fitted"):
        return bool(model._is_fitted)
    try:
        check_is_fitted(model)
        return True
    except (TypeError, ValueError, AttributeError):
        return False


class ExtraParamsMixin:
    """让通过 **kwargs 接收的配置参与 clone/get_params/set_params 往返。"""

    _extra_params_attribute = "kwargs"

    def get_params(self, deep=True):
        params = dict(getattr(self, self._extra_params_attribute, {}))
        params.update(super().get_params(deep=False))
        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params") and not isinstance(value, type):
                    params.update((f"{name}__{key}", item) for key, item in value.get_params().items())
        return params

    def set_params(self, **params):
        explicit = set(self._get_param_names())
        extra = dict(getattr(self, self._extra_params_attribute, {}))
        nested = {}
        for name, value in params.items():
            root, separator, child = name.partition("__")
            if separator:
                nested.setdefault(root, {})[child] = value
            elif name in explicit:
                setattr(self, name, value)
                if name in extra:
                    extra[name] = value
            else:
                extra[name] = value
                if hasattr(self, name):
                    setattr(self, name, value)
        setattr(self, self._extra_params_attribute, extra)
        for name, values in nested.items():
            obj = self.get_params(deep=False).get(name)
            if not hasattr(obj, "set_params"):
                raise ValueError(f"参数 {name!r} 不支持嵌套参数设置")
            obj.set_params(**values)
        return self


class InferenceExportMixin:
    """显式导出不携带 tuner 和训练记录的推理对象，不改变当前实例。"""

    def save_inference(self, path, *, engine="cloudpickle", **kwargs):
        """保留预测/评分状态，移除已知训练历史；不清理自定义函数捕获的数据。"""
        if not is_model_fitted(self):
            raise ValueError("请先完成训练，再导出推理制品")
        if str(path).lower().endswith(".json"):
            raise ValueError("推理对象制品请使用 .pkl 或 .joblib 后缀，不支持 JSON 清单")
        from ._lifecycle import atomic_save_pickle

        memo = {}

        def without_training_state(model):
            if id(model) in memo:
                return memo[id(model)]
            slim = copy.copy(model)
            slim.__dict__ = dict(model.__dict__)
            memo[id(model)] = slim
            for name in ("training_history_", "training_summary_", "_eval_train_indices_", "_eval_val_indices_"):
                slim.__dict__.pop(name, None)
            if hasattr(slim, "tuner"):
                slim.tuner = None
            # 内置概率评分卡也有训练记录，不能让它把训练数组带回推理制品。
            for name in (
                "scorecard_",
                "model_",
                "model",
                "calibrator",
                "calibrator_",
                "base_model",
                "lr_model_",
                "lr_model",
            ):
                child = slim.__dict__.get(name)
                if type(child).__module__.startswith("hscredit.core.models"):
                    setattr(slim, name, without_training_state(child))
            return slim

        slim = without_training_state(self)
        return atomic_save_pickle(slim, path, engine=engine, **kwargs)
