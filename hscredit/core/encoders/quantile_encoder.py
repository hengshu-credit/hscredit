"""Quantile Encoder (分位数编码器).

基于目标变量分位数对类别特征进行编码。
"""

from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd

from .base import BaseEncoder


class QuantileEncoder(BaseEncoder):
    """分位数编码器.

    用目标变量的指定分位数（如中位数）对每个类别进行编码。
    适用于回归任务和存在异常值的场景。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有列（支持类别型和数值型）
    :param quantile: 分位数，范围[0, 1]，默认为0.5（中位数）
    :param smoothing: 平滑参数，默认为1.0
    :param m: 先验权重参数，默认为1.0
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping_: 分位数编码映射字典，格式为 {col: {category: quantile_value}}
    - global_quantile_: 全局分位数

    **参考样例**

    使用中位数编码::

        >>> encoder = QuantileEncoder(cols=['category'], quantile=0.5)
        >>> X_encoded = encoder.fit_transform(X, y)

    使用第90百分位数::

        >>> encoder = QuantileEncoder(cols=['category'], quantile=0.9)
        >>> X_encoded = encoder.fit_transform(X, y)

    参考:
        https://contrib.scikit-learn.org/category_encoders/quantile.html
    """

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        QuantileEncoder支持数值型和类别型列，因此返回所有列。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        quantile: float = 0.5,
        smoothing: float = 1.0,
        m: float = 1.0,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
    ):
        """初始化分位数编码器。

        :param cols: 需要编码的列名列表
        :param quantile: 分位数，范围[0, 1]，默认为0.5（中位数）
        :param smoothing: 平滑参数，默认为1.0
        :param m: 先验权重参数，默认为1.0
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
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
        self.quantile = quantile
        self.smoothing = smoothing
        self.m = m

        self.global_quantile_: float = 0.0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合分位数编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量
        :raises ValueError: 当y为空时抛出
        """
        if y is None:
            raise ValueError("QuantileEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y)
        self.global_quantile_ = y.quantile(self.quantile)

        for col in self.cols_:
            mapping = {}

            for category in X[col].dropna().unique():
                mask = X[col] == category
                category_y = y[mask]

                category_quantile = category_y.quantile(self.quantile)
                n = len(category_y)

                smoothed_quantile = (
                    n * category_quantile + self.m * self.global_quantile_
                ) / (n + self.m)

                mapping[category] = smoothed_quantile

            if self.handle_missing == 'value':
                mapping[np.nan] = self.global_quantile_
            elif self.handle_missing == 'return_nan':
                mapping[np.nan] = np.nan

            if self.handle_unknown == 'value':
                mapping['__UNKNOWN__'] = self.global_quantile_
            elif self.handle_unknown == 'return_nan':
                mapping['__UNKNOWN__'] = np.nan

            self.mapping_[col] = mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]
            X[col] = X[col].map(mapping)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(self.global_quantile_)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

        return X
