"""Count Encoder (计数编码器).

基于类别出现频次进行编码。
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from .base import BaseEncoder


class CountEncoder(BaseEncoder):
    """计数编码器.

    用每个类别的出现次数（或频率）进行编码。
    适用于高基数类别特征，能有效捕捉类别的流行度信息。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
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

    基本使用::

        >>> encoder = CountEncoder(cols=['category'])
        >>> X_encoded = encoder.fit_transform(X)

    返回频率::

        >>> encoder = CountEncoder(cols=['category'], normalize=True)
        >>> X_encoded = encoder.fit_transform(X)

    合并低频类别::

        >>> encoder = CountEncoder(cols=['category'], min_group_size=10)
        >>> X_encoded = encoder.fit_transform(X)

    参考:
        https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10928
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        normalize: bool = False,
        min_group_size: Optional[int] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
    ):
        """初始化计数编码器。

        :param cols: 需要编码的列名列表
        :param normalize: 是否返回频率，默认为False
        :param min_group_size: 将频次低于此值的类别合并为"其他"，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.normalize = normalize
        self.min_group_size = min_group_size

        self.total_count_: int = 0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合计数编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），计数编码器不需要
        """
        self.total_count_ = len(X)

        for col in self.cols_:
            counts = X[col].value_counts(dropna=False)

            if self.min_group_size is not None:
                small_categories = counts[counts < self.min_group_size].index
                if len(small_categories) > 0:
                    other_count = counts[small_categories].sum()
                    counts = counts[counts >= self.min_group_size]
                    counts['__OTHER__'] = other_count

            if self.normalize:
                counts = counts / self.total_count_

            mapping = counts.to_dict()

            if self.handle_missing == 'value':
                if np.nan not in mapping:
                    mapping[np.nan] = 0 if not self.normalize else 0.0
            elif self.handle_missing == 'return_nan':
                mapping[np.nan] = np.nan

            if self.handle_unknown == 'value':
                mapping['__UNKNOWN__'] = 0 if not self.normalize else 0.0
            elif self.handle_unknown == 'return_nan':
                mapping['__UNKNOWN__'] = np.nan

            self.mapping_[col] = mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），计数编码器不需要
        :return: 编码后的数据
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]

            if self.min_group_size is not None and '__OTHER__' in mapping:
                known_categories = set(mapping.keys())
                known_categories.discard('__OTHER__')
                known_categories.discard('__UNKNOWN__')

                X[col] = X[col].apply(
                    lambda x: '__OTHER__' if x not in known_categories and pd.notna(x) else x
                )

            X[col] = X[col].map(mapping)

            if self.handle_unknown == 'value':
                default_value = 0 if not self.normalize else 0.0
                X[col] = X[col].fillna(default_value)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

        return X
