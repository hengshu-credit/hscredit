"""CatBoost Encoder.

基于CatBoost算法的有序目标编码器，使用排序来防止目标泄漏。
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from .base import BaseEncoder


class CatBoostEncoder(BaseEncoder):
    """CatBoost编码器.

    使用有序目标统计（Ordered Target Statistics）方法，
    通过随机排序和累积统计来防止过拟合和目标泄漏。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有列（支持类别型和数值型）
    :param sigma: 添加的高斯噪声标准差，默认为None
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True
    :param random_state: 随机种子，用于可复现性，默认为None

    **属性**

    - mapping_: 目标编码映射字典，格式为 {col: {category: encoded_value}}
    - global_mean_: 全局目标均值

    **参考样例**

    >>> from hscredit.core.encoders import CatBoostEncoder
    >>> encoder = CatBoostEncoder(cols=['category'])
    >>> X_encoded = encoder.fit_transform(X, y)
    >>>
    >>> # 添加噪声
    >>> encoder = CatBoostEncoder(cols=['category'], sigma=0.05, random_state=42)
    >>> X_encoded = encoder.fit_transform(X, y)
    """

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        CatBoostEncoder支持数值型和类别型列，因此返回所有列。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        random_state: Optional[int] = None,
        target: Optional[str] = None,
    ):
        """初始化CatBoost编码器。

        :param cols: 需要编码的列名列表
        :param sigma: 添加的高斯噪声标准差，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param random_state: 随机种子，默认为None
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.sigma = sigma
        self.random_state = random_state

        self.global_mean_: float = 0.0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合CatBoost编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量
        :raises ValueError: 当y为空时抛出
        """
        if y is None:
            raise ValueError("CatBoostEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y)
        self.global_mean_ = y.mean()

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')

        for col in self.cols_:
            df_temp = pd.DataFrame({col: X[col], 'target': y.values})
            category_stats = df_temp.groupby(col)['target'].agg(['mean', 'count'])

            mapping = category_stats['mean'].to_dict()

            if self.handle_missing == 'value':
                mapping[np.nan] = self.global_mean_
            elif self.handle_missing == 'return_nan':
                mapping[np.nan] = np.nan

            if self.handle_unknown == 'value':
                mapping['__UNKNOWN__'] = self.global_mean_
            elif self.handle_unknown == 'return_nan':
                mapping['__UNKNOWN__'] = np.nan

            self.mapping_[col] = mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），如果提供则使用有序统计
        :return: 编码后的数据
        """
        if y is not None and self.random_state is not None:
            np.random.seed(self.random_state)

        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]

            if y is not None:
                X[col] = self._transform_ordered(X[col], y, mapping)
            else:
                X[col] = X[col].map(mapping)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(self.global_mean_)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

            if self.sigma is not None and y is not None:
                X[col] = X[col] * (1 + np.random.normal(0, self.sigma, len(X)))

        return X

    def _transform_ordered(
        self, x: pd.Series, y: pd.Series, mapping: Dict
    ) -> pd.Series:
        """使用有序统计进行转换（防止目标泄漏）。

        :param x: 特征列
        :param y: 目标变量
        :param mapping: 编码映射
        :return: 编码后的序列
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=x.index)

        n = len(x)
        random_order = np.random.permutation(n)

        result = pd.Series(index=x.index, dtype=float)

        result[:] = self.global_mean_

        category_sums = {}
        category_counts = {}

        for idx in random_order:
            category = x.iloc[idx]

            if pd.isna(category):
                result.iloc[idx] = mapping.get(np.nan, self.global_mean_)
                continue

            if category in category_counts and category_counts[category] > 0:
                prior = self.global_mean_
                posterior = category_sums[category] / category_counts[category]
                count = category_counts[category]
                a = 1.0
                result.iloc[idx] = (count * posterior + a * prior) / (count + a)
            else:
                result.iloc[idx] = self.global_mean_

            if category not in category_sums:
                category_sums[category] = 0
                category_counts[category] = 0
            category_sums[category] += y.iloc[idx]
            category_counts[category] += 1

        return result
