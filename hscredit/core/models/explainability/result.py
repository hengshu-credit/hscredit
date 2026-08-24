"""结构化模型解释结果与输入规范化工具。"""

import copy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from hscredit.exceptions import ValidationError


def coerce_explanation_frame(data: Any, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """将解释输入转为保留样本身份和列顺序的 DataFrame。"""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if frame.empty or frame.shape[1] == 0:
            raise ValidationError("解释数据不能为空")
        if feature_names is not None and list(frame.columns) != list(feature_names):
            raise ValidationError("解释数据的特征名称或顺序与模型不一致")
        return frame
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValidationError("解释数据必须是二维表格")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValidationError("解释数据不能为空")
    names = list(feature_names) if feature_names is not None else [f"特征{i + 1}" for i in range(array.shape[1])]
    if len(names) != array.shape[1]:
        raise ValidationError("feature_names 数量与解释数据列数不一致")
    return pd.DataFrame(array, columns=names)


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """计算同时覆盖索引、模式、列顺序和数据值的稳定指纹。"""
    normalized = coerce_explanation_frame(frame)
    digest = sha256()
    digest.update(repr(tuple(normalized.columns)).encode("utf-8"))
    digest.update(repr(tuple(map(str, normalized.dtypes))).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(normalized.index, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _select_output(values: Any, output_index: Optional[int]) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        index = 0 if output_index is None else output_index
        if index < 0 or index >= len(values):
            raise ValidationError("目标类别对应的 SHAP 输出索引不存在")
        return np.asarray(values[index])
    array = np.asarray(values)
    if array.ndim == 3:
        index = 0 if output_index is None else output_index
        if index < 0 or index >= array.shape[2]:
            raise ValidationError("目标类别对应的 SHAP 输出索引不存在")
        return array[:, :, index]
    if array.ndim != 2:
        raise ValidationError("选定类别后的 SHAP 值必须是二维数组")
    return array


def _select_base_values(
    base_values: Any,
    output_index: Optional[int],
    n_samples: int,
    *,
    multi_output: bool = False,
) -> np.ndarray:
    base = np.asarray(base_values)
    if base.ndim == 0:
        return np.repeat(float(base), n_samples)
    if base.ndim == 1:
        if multi_output:
            index = 0 if output_index is None else output_index
            if index < 0 or index >= base.shape[0]:
                raise ValidationError("目标类别对应的 SHAP 基准值索引不存在")
            return np.repeat(float(base[index]), n_samples)
        if base.shape[0] == n_samples:
            return base.astype(float)
        index = 0 if output_index is None else output_index
        if 0 <= index < base.shape[0]:
            return np.repeat(float(base[index]), n_samples)
    if base.ndim >= 2:
        index = 0 if output_index is None else output_index
        selected = base[:, index]
        if selected.shape[0] == 1:
            selected = np.repeat(selected, n_samples)
        return np.asarray(selected, dtype=float).reshape(-1)
    raise ValidationError("无法识别 SHAP 基准值的输出形状")


def normalize_explanation_output(explanation: Any, output_index: Optional[int] = None) -> Any:
    """把旧式 list、三维多输出和现代 Explanation 统一到二维选定类别。"""
    values_source = getattr(explanation, "values", explanation)
    values = _select_output(values_source, output_index)
    base_source = getattr(explanation, "base_values", 0.0)
    source_array = None if isinstance(values_source, (list, tuple)) else np.asarray(values_source)
    multi_output = isinstance(values_source, (list, tuple)) or (source_array is not None and source_array.ndim == 3)
    base_values = _select_base_values(
        base_source,
        output_index,
        values.shape[0],
        multi_output=multi_output,
    )
    data = getattr(explanation, "data", None)
    feature_names = getattr(explanation, "feature_names", None)
    try:
        import shap

        return shap.Explanation(values=values, base_values=base_values, data=data, feature_names=feature_names)
    except (ImportError, AttributeError, TypeError) as exc:
        raise ValidationError(f"无法构造结构化 SHAP 解释结果: {exc}") from exc


@dataclass(frozen=True)
class ExplanationResult:
    """一次模型解释计算的只读、可审计结果。"""

    _explanation: Any
    _data: pd.DataFrame
    _sample_ids: pd.Index
    target_class: Any
    output_index: Optional[int]
    model_output: str
    explainer_type: str
    background_summary: Mapping[str, Any]
    dataset_fingerprint: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_explanation(
        cls,
        explanation: Any,
        *,
        data: Any,
        target_class: Any,
        output_index: Optional[int],
        model_output: str,
        explainer_type: str,
        background_summary: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> "ExplanationResult":
        """规范化 SHAP 输出并构造形状一致、外部不可原地修改的审计结果。"""
        names = getattr(explanation, "feature_names", None)
        frame = coerce_explanation_frame(data, feature_names=names)
        normalized = normalize_explanation_output(explanation, output_index=output_index)
        if normalized.values.shape != frame.shape:
            raise ValidationError("SHAP 值形状与解释数据不一致")
        base_values = np.asarray(normalized.base_values, dtype=float).reshape(-1)
        if base_values.shape != (len(frame),):
            raise ValidationError("SHAP 基准值数量与解释样本数不一致")
        return cls(
            _explanation=copy.deepcopy(normalized),
            _data=frame.copy(deep=True),
            _sample_ids=frame.index.copy(deep=True),
            target_class=target_class,
            output_index=output_index,
            model_output=model_output,
            explainer_type=explainer_type,
            background_summary=MappingProxyType(copy.deepcopy(dict(background_summary))),
            dataset_fingerprint=fingerprint_frame(frame),
            metadata=MappingProxyType(copy.deepcopy(dict(metadata))),
        )

    @property
    def explanation(self) -> Any:
        """返回独立的 SHAP Explanation 副本。"""
        return copy.deepcopy(self._explanation)

    @property
    def data(self) -> pd.DataFrame:
        """返回解释输入的深复制，避免外部修改审计结果。"""
        return self._data.copy(deep=True)

    @property
    def sample_ids(self) -> pd.Index:
        """返回样本索引副本。"""
        return self._sample_ids.copy(deep=True)

    @property
    def values(self) -> np.ndarray:
        """返回二维 SHAP 贡献值副本。"""
        return np.array(self._explanation.values, copy=True)

    @property
    def base_values(self) -> np.ndarray:
        """返回每个样本的 SHAP 基准值副本。"""
        return np.array(self._explanation.base_values, dtype=float, copy=True).reshape(-1)

    @property
    def feature_names(self) -> list:
        """返回固定顺序的特征名。"""
        return list(self._data.columns)

    def position_for(self, sample_id: Any) -> int:
        """按样本索引返回唯一的位置。"""
        positions = np.flatnonzero(self._sample_ids == sample_id)
        if len(positions) == 0:
            raise ValidationError(f"样本索引不存在: {sample_id}")
        if len(positions) > 1:
            raise ValidationError(f"样本索引不唯一: {sample_id}")
        return int(positions[0])
