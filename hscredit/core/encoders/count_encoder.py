"""Count Encoder (计数编码器).

基于类别出现频次进行编码。
"""

from typing import Optional, List, Dict, Union, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder
from ._category_protocol import CategoryToken, MISSING, UNKNOWN, OTHER


class CountEncoder(BaseEncoder):
    """计数编码器.

    用每个类别的出现次数（或频率）进行编码。
    适用于高基数类别特征，能有效捕捉类别的流行度信息。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有列（支持类别型和数值型）
    :param normalize: 是否返回频率而不是计数，默认为False
    :param min_group_size: 将频次低于此值的类别合并为"其他"，默认为None
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping_: 计数编码映射字典，格式为 {col: {category: count}}
    - total_count_: 总样本数

    **参考样例**

    >>> from hscredit.core.encoders import CountEncoder
    >>> encoder = CountEncoder(cols=['category'])
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 返回频率
    >>> encoder = CountEncoder(cols=['category'], normalize=True)
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 合并低频类别
    >>> encoder = CountEncoder(cols=['category'], min_group_size=10)
    >>> X_encoded = encoder.fit_transform(X)

    **注意**

    计数/频率编码为无监督方法，不使用标签 ``y``，仅以类别出现频次反映其流行度，对高基数
    类别尤其紧凑；但不同类别若频次相同会被编码为同一值（信息混淆），必要时与其他编码并用。

    **引用**

    频率/计数编码（frequency / count encoding）是类别特征工程的常用基线方法，参见
    category_encoders ``CountEncoder``：
    https://contrib.scikit-learn.org/category_encoders/count.html
    """

    # total_count_ 为训练样本总数，随映射一并序列化
    _EXTRA_STATE_ATTRS = ["total_count_", "infrequent_categories_", "_legacy_other_routing_"]

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        CountEncoder支持数值型和类别型列，因此返回所有列。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        normalize: bool = False,
        min_group_size: Optional[int] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        passthrough_target: bool = False,
    ):
        """初始化计数编码器。

        :param cols: 需要编码的列名列表
        :param normalize: 是否返回频率，默认为False
        :param min_group_size: 将频次低于此值的类别合并为"其他"，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param target: scorecardpipeline风格的目标列名。计数编码器不使用此参数，仅为API一致性保留
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            passthrough_target=passthrough_target,
        )
        self.normalize = normalize
        self.min_group_size = min_group_size

        self.total_count_: int = 0
        self.infrequent_categories_ = {}
        self._legacy_other_routing_ = False

    @classmethod
    def _canonicalize_nan_keys(cls, mapping: Dict) -> Dict:
        """按浮点标量类型规范化计数键，保留 pandas 的 typed-NaN 分组。"""
        return {
            cls._float_nan_representative(key): value
            for key, value in mapping.items()
        }

    def _serialize_mapping(self, mapping: Dict) -> Dict:
        """导出时保留 CountEncoder 的 typed-NaN 公开代表键。"""
        serialized = {}
        for key, value in mapping.items():
            key = self._float_nan_representative(key)
            if isinstance(key, CategoryToken):
                key = {MISSING: np.nan, UNKNOWN: "__UNKNOWN__", OTHER: "__OTHER__"}.get(key, key)
            if key in serialized:
                raise ValueError("传统映射存在保留键冲突，请使用默认类型化 export_mapping()")
            if isinstance(value, pd.Series):
                serialized[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_mapping(value)
            else:
                serialized[key] = value
        return serialized

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合计数编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），计数编码器不需要
        """
        total_count = len(X)
        self._fit_columns(X, y, state_attrs=("mapping_", "infrequent_categories_"), shared_state={"total_count_": total_count})
        self.total_count_ = total_count

    def _fit_column(self, column, values, y=None):
        counts = values.value_counts(dropna=False)
        small_categories = []

        if self.min_group_size is not None:
            small_categories = [value for value in counts[counts < self.min_group_size].index if not pd.isna(value)]
            if len(small_categories) > 0:
                other_count = counts[small_categories].sum()
                counts = counts.drop(index=small_categories)
                counts[OTHER] = other_count

        if self.normalize:
            counts = counts / self.total_count_

        mapping = {}
        for key, value in counts.items():
            bucket = self._float_nan_bucket(key)
            normalized = self._float_nan_representative(key)
            if bucket is None:
                mapping[normalized] = value
            else:
                mapping[normalized] = mapping.get(normalized, 0) + value

        if self.handle_missing == 'value':
            if not any(self._is_float_nan_key(key) for key in mapping):
                mapping[MISSING] = 0 if not self.normalize else 0.0
        elif self.handle_missing == 'return_nan':
            typed_nan_keys = [key for key in mapping if self._is_float_nan_key(key)]
            if typed_nan_keys:
                for key in typed_nan_keys:
                    mapping[key] = np.nan
            else:
                mapping[MISSING] = np.nan

        if self.handle_unknown == 'value':
            mapping[UNKNOWN] = 0 if not self.normalize else 0.0
        elif self.handle_unknown == 'return_nan':
            mapping[UNKNOWN] = np.nan

        return {"mapping_": mapping, "infrequent_categories_": list(small_categories)}

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），计数编码器不需要
        :return: 编码后的数据
        """
        return self._transform_columns(X, y)

    def _transform_column(self, column, values, y=None, context=None):
        mapping = self.mapping_[column]
        result = values.copy()

        if self.min_group_size is not None and OTHER in mapping:
            known_categories = set(mapping.keys())
            known_categories.discard(OTHER)
            known_categories.discard(UNKNOWN)

            if getattr(self, "_legacy_other_routing_", False):
                result = result.apply(lambda x: OTHER if x not in known_categories and pd.notna(x) else x)
            else:
                result = result.astype(object).mask(result.isin(self.infrequent_categories_.get(column, [])), OTHER)

        result = self._map_with_typed_float_nan(result, mapping)
        typed_missing = pd.Series(
            [self._is_float_nan_key(value) for value in values.array],
            index=values.index,
        )
        if self.handle_missing == 'value':
            missing_default = 0 if not self.normalize else 0.0
            result = result.mask(result.isna() & typed_missing, missing_default)
        unknown = result.isna() & ~typed_missing

        if self.handle_unknown == 'value':
            default_value = 0 if not self.normalize else 0.0
            result = result.mask(unknown, default_value)
        elif self.handle_unknown == 'error' and unknown.any():
            raise ValueError(f"列'{column}'包含未知类别")

        return result
