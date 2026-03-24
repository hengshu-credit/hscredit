"""Ordinal Encoder (序数编码器).

将类别特征转换为整数编码。
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder


class OrdinalEncoder(BaseEncoder):
    """序数编码器.

    将每个类别映射为一个整数，保留类别的顺序关系（如果存在）。
    适用于树模型和需要保留单一特征维度的场景。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param mapping: 自定义映射字典，如{'col': {'a': 1, 'b': 2}}，默认为None
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping_: 序数编码映射字典，格式为 {col: {category: integer}}

    **参考样例**

    基本使用::

        >>> encoder = OrdinalEncoder(cols=['education'])
        >>> X_encoded = encoder.fit_transform(X)

    自定义映射::

        >>> mapping = {'education': {'high': 3, 'medium': 2, 'low': 1}}
        >>> encoder = OrdinalEncoder(cols=['education'], mapping=mapping)
        >>> X_encoded = encoder.fit_transform(X)

    参考:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        mapping: Optional[Dict[str, Dict[Any, int]]] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
    ):
        """初始化序数编码器。

        :param cols: 需要编码的列名列表
        :param mapping: 自定义映射字典，如{'col': {'a': 1, 'b': 2}}，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param target: scorecardpipeline风格的目标列名。序数编码器不使用此参数，仅为API一致性保留
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.mapping = mapping or {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合序数编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），序数编码器不需要
        """
        for col in self.cols_:
            if col in self.mapping:
                self.mapping_[col] = self.mapping[col].copy()
            else:
                categories = X[col].dropna().unique()
                categories = sorted([c for c in categories if c is not np.nan])

                mapping = {cat: i for i, cat in enumerate(categories)}

                if self.handle_missing == 'value':
                    mapping[np.nan] = -1
                elif self.handle_missing == 'return_nan':
                    mapping[np.nan] = np.nan

                if self.handle_unknown == 'value':
                    mapping['__UNKNOWN__'] = -1
                elif self.handle_unknown == 'return_nan':
                    mapping['__UNKNOWN__'] = np.nan

                self.mapping_[col] = mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），序数编码器不需要
        :return: 编码后的数据
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]

            X[col] = X[col].map(mapping)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(-1)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

        return X

    def get_mapping(self, col: Optional[str] = None) -> Dict:
        """获取序数映射。

        :param col: 列名，如果为None则返回所有映射
        :return: 序数映射字典
        """
        if col is not None:
            return self.mapping_.get(col, {})
        return self.mapping_
