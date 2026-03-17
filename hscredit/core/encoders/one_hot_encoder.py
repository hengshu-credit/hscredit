"""One-Hot Encoder (独热编码器).

将类别特征转换为独热编码形式。
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """独热编码器.

    将每个类别转换为一个二进制列，适用于类别数量不多的特征。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param drop: 是否删除某一列以避免多重共线性，默认为None
        - None: 保留所有列
        - 'first': 删除第一列
        - 'if_binary': 二值特征时删除一列
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - categories_: 各列的类别列表，格式为 {col: [category1, category2, ...]}
    - feature_names_: 编码后的特征名列表

    **参考样例**

    基本使用::

        >>> encoder = OneHotEncoder(cols=['color'])
        >>> X_encoded = encoder.fit_transform(X)

    删除第一列避免多重共线性::

        >>> encoder = OneHotEncoder(cols=['color'], drop='first')
        >>> X_encoded = encoder.fit_transform(X)

    参考:
        https://en.wikipedia.org/wiki/One-hot
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
    ):
        """初始化独热编码器。

        :param cols: 需要编码的列名列表
        :param drop: 是否删除某一列以避免多重共线性，默认为None
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
        self.drop = drop

        self.categories_: Dict[str, List] = {}
        self.feature_names_: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合独热编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），独热编码器不需要
        """
        for col in self.cols_:
            categories = X[col].dropna().unique().tolist()
            categories = sorted([c for c in categories if c is not np.nan])

            if self.drop == 'first' and len(categories) > 0:
                categories_to_use = categories[1:]
            elif self.drop == 'if_binary' and len(categories) == 2:
                categories_to_use = categories[:1]
            else:
                categories_to_use = categories

            self.categories_[col] = categories_to_use

            for cat in categories_to_use:
                safe_cat = str(cat).replace(' ', '_').replace('-', '_')
                self.feature_names_.append(f"{col}_{safe_cat}")

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），独热编码器不需要
        :return: 编码后的数据
        """
        result_dfs = []

        other_cols = [c for c in X.columns if c not in self.cols_]
        if other_cols:
            result_dfs.append(X[other_cols].copy())

        for col in self.cols_:
            if col not in self.categories_:
                continue

            categories = self.categories_[col]

            if self.handle_unknown == 'error':
                unknown = set(X[col].dropna().unique()) - set(categories)
                if unknown:
                    raise ValueError(f"列'{col}'包含未知类别: {unknown}")

            for cat in categories:
                col_name = f"{col}_{str(cat).replace(' ', '_').replace('-', '_')}"
                X[col_name] = (X[col] == cat).astype(int)

            X = X.drop(columns=[col])

        return X

    def get_feature_names(self) -> List[str]:
        """获取编码后的特征名。

        :return: 编码后的特征名列表
        """
        return self.feature_names_
