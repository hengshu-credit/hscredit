"""跨组件的输入、目标与位置对齐契约。

本模块不依赖模型层。辅助业务字段由调用方显式选择，只有声明的目标列
会自动从学习特征中移除；不按名称猜测日期、金额或客户标识的用途。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import ValidationError


@dataclass(frozen=True)
class TargetSpec:
    """一次输入解析使用的目标与对齐口径。"""

    name: Optional[str] = None
    target_type: Optional[str] = None
    alignment: str = "position"
    allow_single_class: bool = True


@dataclass(frozen=True)
class PreparedData:
    """规范化的特征、标签和原始行位置；不隐式过滤任何样本。"""

    X: pd.DataFrame
    y: Optional[pd.Series]
    positions: np.ndarray
    target_spec: TargetSpec


def extract_target_column(X, y=None, target=None):
    """仅隔离目标列，保留 ndarray/稀疏矩阵和标签的原始表示。"""
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValidationError("输入特征列名不能重复")
        if target is not None and target in X.columns:
            if y is None:
                y = X[target]
            X = X.drop(columns=[target])
    return X, y


def validate_target(y, *, target_type=None, allow_single_class=True, allow_empty=False):
    """验证一维、非缺失有限目标；二分类在任何整数转换之前验证 0/1。"""
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValidationError("目标变量必须是一维数据")
    if not allow_empty and len(values) == 0:
        raise ValidationError("目标变量不能为空")
    if pd.isna(values).any():
        raise ValidationError("目标变量不能包含缺失值")
    if target_type not in (None, "binary", "continuous"):
        raise ValidationError("target_type 必须为 None、binary 或 continuous")
    if np.iscomplexobj(values):
        raise ValidationError("目标变量不能包含复数")
    if target_type in ("binary", "continuous") or pd.api.types.is_numeric_dtype(values.dtype):
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("目标变量必须为数值") from exc
        if not np.isfinite(numeric).all():
            raise ValidationError("目标变量不能包含无穷值或非有限数值")
    if target_type == "binary":
        if not np.isin(values, [0, 1]).all():
            raise ValidationError("目标变量必须是二分类，且只能包含 0 和 1")
        if not allow_single_class and len(np.unique(values)) != 2:
            raise ValidationError("目标变量必须是二分类，必须同时包含 0 和 1")
    elif not allow_single_class and len(pd.unique(values)) < 2:
        raise ValidationError("目标变量必须包含至少两个类别")
    return values


def prepare_xy(
    X, y=None, *, target=None, alignment="position", require_y=False,
    target_type=None, allow_single_class=True, allow_empty=False, allow_empty_features=False,
):
    """统一三种调用风格并隔离目标列，默认按位置对齐标签。

    ``alignment='index'`` 仅接受唯一且完全匹配的索引；不使用索引交集
    静默丢弃样本。返回的特征表不深复制，调用方不得原地修改输入。
    """
    if not isinstance(X, pd.DataFrame):
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValidationError("输入特征必须为二维数据")
        X = pd.DataFrame(values)
    if not X.columns.is_unique:
        raise ValidationError("输入特征列名不能重复")
    if not allow_empty and len(X) == 0:
        raise ValidationError("输入数据不能为空")
    if alignment not in ("position", "index"):
        raise ValidationError("alignment 必须为 position 或 index")
    X, y = extract_target_column(X, y, target)
    if X.shape[1] == 0 and not allow_empty_features:
        raise ValidationError("目标列之外至少需要一个特征")
    if y is None:
        if require_y:
            raise ValidationError("必须提供目标变量 y 或配置的目标列")
        normalized_y = None
    else:
        if alignment == "index":
            if not isinstance(y, pd.Series):
                raise ValidationError("按索引对齐时 y 必须为 Series")
            if not X.index.is_unique or not y.index.is_unique:
                raise ValidationError("按索引对齐时特征与目标索引必须唯一")
            if len(X) != len(y) or not X.index.isin(y.index).all():
                raise ValidationError("按索引对齐时特征与目标索引必须完全匹配")
            y = y.reindex(X.index)
        values = validate_target(
            y, target_type=target_type, allow_single_class=allow_single_class, allow_empty=allow_empty,
        )
        if len(values) != len(X):
            raise ValidationError(f"输入特征与目标变量的样本数量不一致: [{len(X)}, {len(values)}]")
        normalized_y = pd.Series(values, index=X.index, name=target or getattr(y, "name", None))
    return PreparedData(
        X=X, y=normalized_y, positions=np.arange(len(X)),
        target_spec=TargetSpec(target, target_type, alignment, allow_single_class),
    )
