"""Target Encoder (目标编码器).

基于目标变量均值对类别特征进行编码。
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from .base import BaseEncoder


class TargetEncoder(BaseEncoder):
    """目标编码器.

    用目标变量的均值对每个类别进行编码：
    - 对于分类任务：该类别的正样本比例
    - 对于回归任务：该类别的目标变量均值

    使用平滑技术防止过拟合：
    encoded = (count * mean + smoothing * global_mean) / (count + smoothing)

    属性:
        cols: 需要编码的列名列表。
        smoothing: 平滑参数。
        min_samples_leaf: 每个类别的最小样本数。
        noise: 添加的高斯噪声标准差。
        handle_unknown: 处理未知类别的方式。
        handle_missing: 处理缺失值的方式。
        drop_invariant: 是否删除方差为0的列。
        return_df: 是否返回DataFrame。
        global_mean_: 全局目标均值。

    示例:
        >>> encoder = TargetEncoder(cols=['category'])
        >>> X_encoded = encoder.fit_transform(X, y)

    参考:
        https://dl.acm.org/doi/10.1145/507533.507538
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        noise: Optional[float] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
    ):
        """初始化目标编码器。

        :param cols: 需要编码的列名列表。
        :param smoothing: 平滑参数，值越大收缩到全局均值的程度越大，默认为1.0。
        :param min_samples_leaf: 每个类别的最小样本数，少于该值则使用全局均值，默认为1。
        :param noise: 添加的高斯噪声标准差，用于防止过拟合，默认为None。
        :param handle_unknown: 处理未知类别的方式，默认为'value'。
        :param handle_missing: 处理缺失值的方式，默认为'value'。
        :param drop_invariant: 是否删除方差为0的列，默认为False。
        :param return_df: 是否返回DataFrame，默认为True。
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise

        self.global_mean_: float = 0.0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合目标编码器。

        :param X: 输入数据。
        :param y: 目标变量。
        :raises ValueError: 当y为空时抛出。
        """
        if y is None:
            raise ValueError("TargetEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y)
        self.global_mean_ = y.mean()

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')

        for col in self.cols_:
            mapping = {}

            df_temp = pd.DataFrame({col: X[col], 'target': y.values})
            stats = df_temp.groupby(col)['target'].agg(['mean', 'count'])

            smoothed_means = (
                stats['count'] * stats['mean'] + self.smoothing * self.global_mean_
            ) / (stats['count'] + self.smoothing)

            small_sample_mask = stats['count'] < self.min_samples_leaf
            smoothed_means[small_sample_mask] = self.global_mean_

            mapping = smoothed_means.to_dict()

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

        :param X: 输入数据。
        :param y: 目标变量（可选）。
        :return: 编码后的数据。
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]

            original_values = X[col].copy()

            X[col] = original_values.map(mapping)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(self.global_mean_)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

            if self.noise is not None and y is not None:
                X[col] = X[col] * (1 + np.random.normal(0, self.noise, len(X)))

        return X
