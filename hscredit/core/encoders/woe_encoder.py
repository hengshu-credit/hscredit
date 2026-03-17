"""WOE (Weight of Evidence) 编码器.

提供直接计算WOE的编码功能，不依赖分箱模块。
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from .base import BaseEncoder


class WOEEncoder(BaseEncoder):
    """WOE (证据权重) 编码器.

    直接对类别特征计算WOE值，不依赖分箱功能。

    WOE计算公式：
    WOE = ln(P(好样本|类别) / P(坏样本|类别)) = ln(好样本占比/坏样本占比)

    属性:
        cols: 需要编码的列名列表。
        bins: 分箱数（已废弃，保留参数兼容性）。
        binning_method: 分箱方法（已废弃）。
        min_bin_size: 每箱最小样本占比（已废弃）。
        regularization: 正则化参数，防止除零。
        handle_unknown: 处理未知类别的方式。
        handle_missing: 处理缺失值的方式。
        drop_invariant: 是否删除方差为0的列。
        return_df: 是否返回DataFrame。
        mapping_: WOE编码映射字典。
        iv_: 各特征的IV值。

    示例:
        >>> from hscredit.core.encoders import WOEEncoder
        >>> encoder = WOEEncoder(cols=['category', 'score'])
        >>> X_encoded = encoder.fit_transform(X, y)
        >>> print(encoder.iv_)

    参考:
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        bins: Optional[int] = None,
        binning_method: Optional[str] = None,
        min_bin_size: Optional[float] = None,
        regularization: float = 1.0,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
    ):
        """初始化WOE编码器。

        :param cols: 需要编码的列名列表。
        :param bins: 已废弃参数，保留兼容性。
        :param binning_method: 已废弃参数，保留兼容性。
        :param min_bin_size: 已废弃参数，保留兼容性。
        :param regularization: 正则化参数，防止除零。默认为1.0。
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
        self.bins = bins
        self.binning_method = binning_method
        self.min_bin_size = min_bin_size
        self.regularization = regularization

        self.iv_: Dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合WOE编码器。

        :param X: 输入数据。
        :param y: 目标变量。
        :raises ValueError: 当y为空或目标变量不是二元时抛出。
        """
        if y is None:
            raise ValueError("WOEEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y).astype(int)

        unique = y.unique()
        if len(unique) != 2:
            raise ValueError(f"目标变量必须是二元的，当前有{len(unique)}个唯一值")
        if not set(unique).issubset({0, 1}):
            raise ValueError("目标变量必须是0和1")

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for col in self.cols_:
            woe_map, iv = self._fit_categorical(X[col], y, total_good, total_bad)
            self.mapping_[col] = woe_map
            self.iv_[col] = iv

    def _fit_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> tuple:
        """拟合类别特征的WOE。

        :param x: 特征列。
        :param y: 目标变量。
        :param total_good: 好样本总数。
        :param total_bad: 坏样本总数。
        :return: WOE映射和IV值的元组。
        """
        woe_map = {}

        for category in x.unique():
            if pd.isna(category):
                continue

            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            woe = self._compute_woe(good_count, bad_count, total_good, total_bad)
            woe_map[category] = woe

        if self.handle_missing == 'value':
            woe_map[np.nan] = 0.0
        elif self.handle_missing == 'return_nan':
            woe_map[np.nan] = np.nan

        if self.handle_unknown == 'value':
            woe_map['__UNKNOWN__'] = 0.0
        elif self.handle_unknown == 'return_nan':
            woe_map['__UNKNOWN__'] = np.nan

        iv = self._compute_iv_categorical(x, y, total_good, total_bad)

        return woe_map, iv

    def _compute_woe(
        self, good_count: int, bad_count: int, total_good: int, total_bad: int
    ) -> float:
        """计算WOE值（带正则化）。

        :param good_count: 好样本数量。
        :param bad_count: 坏样本数量。
        :param total_good: 好样本总数。
        :param total_bad: 坏样本总数。
        :return: WOE值。
        """
        good_rate = (good_count + self.regularization) / (total_good + 2 * self.regularization)
        bad_rate = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

        woe = np.log(good_rate / bad_rate)
        return woe

    def _compute_iv_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> float:
        """计算类别特征的IV。

        :param x: 特征列。
        :param y: 目标变量。
        :param total_good: 好样本总数。
        :param total_bad: 坏样本总数。
        :return: IV值。
        """
        iv = 0.0
        for category in x.dropna().unique():
            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            good_dist = (good_count + self.regularization) / (total_good + 2 * self.regularization)
            bad_dist = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

            iv += (good_dist - bad_dist) * np.log(good_dist / bad_dist)

        return iv

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据为WOE编码。

        :param X: 输入数据。
        :param y: 目标变量（可选）。
        :return: 编码后的数据。
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            woe_map = self.mapping_[col]
            X[col] = X[col].map(woe_map)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(0.0)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

        return X

    def get_iv(self) -> Dict[str, float]:
        """获取各特征的IV值。

        :return: 特征名到IV值的映射字典。
        """
        return self.iv_

    def summary(self) -> pd.DataFrame:
        """获取WOE编码摘要。

        :return: 包含各特征IV值和预测能力的摘要表。
        """
        if not self.iv_:
            return pd.DataFrame()

        summary = []
        for col, iv in self.iv_.items():
            if iv < 0.02:
                power = '无预测力'
            elif iv < 0.1:
                power = '弱预测力'
            elif iv < 0.3:
                power = '中等预测力'
            elif iv < 0.5:
                power = '强预测力'
            else:
                power = '超强预测力(需检查)'

            summary.append({
                '特征': col,
                'IV值': round(iv, 4),
                '预测能力': power,
            })

        return pd.DataFrame(summary).sort_values('IV值', ascending=False)
