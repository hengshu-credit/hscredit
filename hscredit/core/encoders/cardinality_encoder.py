"""Cardinality Reducer (高基数降维编码器).

将高基数类别特征合并为低基数，保留频次最高的 top-N 类别，
其余合并为统一的 other 标签。支持缺失值和特殊值单独成类，支持逆编码。
"""

from typing import Optional, List, Union, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder


class CardinalityEncoder(BaseEncoder):
    """高基数降维编码器.

    将类别型特征从高基数（如100个枚举值）降维为低基数（如最多10类）。
    按训练集中各类别的频次排序，保留前 ``max_categories - 1`` 个类别，
    其余合并为统一的 ``other_label`` 标签。

    缺失值和特殊值不占 ``max_categories`` 配额，始终独立成类。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param max_categories: 每列最大类别数（含 other 标签，不含缺失及特殊值），默认为10
    :param other_label: 合并后的标签名称，默认为 'other'，支持自定义
    :param special_values: 需要独立成类的特殊值列表（如 [-999, 'unknown']），默认为None
    :param handle_unknown: 处理未知类别的方式，默认为 'other'
        - 'other': 映射为 other_label
        - 'error': 抛出错误
        - 'return_nan': 返回 NaN
    :param handle_missing: 处理缺失值的方式，默认为 'value'
        - 'value': 缺失值保持原样（NaN），独立成类
        - 'error': 抛出错误
        - 'return_nan': 返回 NaN
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping\_: 编码映射字典，格式为 ``{col: {原始类别: 编码后类别}}``
    - top_categories\_: 各列保留的高频类别列表，格式为 ``{col: [类别列表]}``
    - category_counts\_: 各列训练集类别频次统计

    **参考样例**

    >>> from hscredit.core.encoders import CardinalityEncoder
    >>> encoder = CardinalityEncoder(cols=['city'], max_categories=10)
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 自定义标签
    >>> encoder = CardinalityEncoder(
    ...     cols=['city'], max_categories=5, other_label='其他城市'
    ... )
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 特殊值独立成类
    >>> encoder = CardinalityEncoder(
    ...     cols=['score_level'],
    ...     max_categories=8,
    ...     special_values=[-999, 'missing'],
    ... )
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 逆编码
    >>> X_original = encoder.inverse_transform(X_encoded)
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        max_categories: int = 10,
        other_label: Any = 'other',
        special_values: Optional[List[Any]] = None,
        handle_unknown: str = 'other',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
    ):
        """初始化高基数降维编码器。

        :param cols: 需要编码的列名列表
        :param max_categories: 每列最大类别数（含 other，不含缺失及特殊值），默认为10
        :param other_label: 合并后的标签名称，默认为 'other'
        :param special_values: 需要独立成类的特殊值列表，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为 'other'
        :param handle_missing: 处理缺失值的方式，默认为 'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param target: scorecardpipeline风格的目标列名。本编码器不使用此参数，仅为API一致性保留
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.max_categories = max_categories
        self.other_label = other_label
        self.special_values = special_values or []

        self.top_categories_: Dict[str, list] = {}
        self.category_counts_: Dict[str, pd.Series] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合高基数降维编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），本编码器不需要
        """
        special_set = set(self.special_values)

        for col in self.cols_:
            series = X[col]

            # 排除缺失值和特殊值后统计频次
            mask_special = series.isin(special_set) if special_set else pd.Series(False, index=series.index)
            mask_valid = series.notna() & ~mask_special
            counts = series[mask_valid].value_counts()

            self.category_counts_[col] = counts

            # 保留前 max_categories - 1 个（给 other 留一个位置）
            keep_n = max(self.max_categories - 1, 0)
            top_cats = counts.head(keep_n).index.tolist()
            self.top_categories_[col] = top_cats

            # 构建映射：top 类别 → 自身，其余 → other_label
            mapping = {}
            for cat in counts.index:
                mapping[cat] = cat if cat in top_cats else self.other_label

            # 特殊值 → 自身（不受 max_categories 限制）
            for sv in special_set:
                mapping[sv] = sv

            # 未知类别处理
            if self.handle_unknown == 'other':
                mapping['__UNKNOWN__'] = self.other_label
            elif self.handle_unknown == 'return_nan':
                mapping['__UNKNOWN__'] = np.nan

            self.mapping_[col] = mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），本编码器不需要
        :return: 编码后的数据
        """
        special_set = set(self.special_values)

        for col in self.cols_:
            if col not in self.mapping_:
                continue

            mapping = self.mapping_[col]
            original = X[col]

            # 标记缺失
            is_na = original.isna()

            # 映射：先用 map，未命中的为 NaN
            mapped = original.map(mapping)

            # 对未命中且非缺失的值，按 handle_unknown 策略处理
            unmapped = mapped.isna() & ~is_na
            if unmapped.any():
                if self.handle_unknown == 'other':
                    mapped[unmapped] = self.other_label
                elif self.handle_unknown == 'error':
                    bad = original[unmapped].unique().tolist()
                    raise ValueError(f"列 '{col}' 包含未知类别: {bad}")
                # 'return_nan' 不需要额外处理，已经是 NaN

            # 缺失值保持
            if self.handle_missing == 'error' and is_na.any():
                raise ValueError(f"列 '{col}' 包含缺失值")

            X[col] = mapped

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆编码。

        将编码后的数据还原为原始类别值。
        **注意**: 被合并为 other_label 的类别无法恢复为原始值，逆编码后仍为 other_label。

        :param X: 编码后的数据
        :return: 逆编码后的数据
        """
        X = self._check_input(X).copy()

        for col in self.cols_:
            if col not in self.mapping_ or col not in X.columns:
                continue

            mapping = self.mapping_[col]

            # 构建逆映射：只对保留类别和特殊值有效
            inverse_mapping = {}
            for orig, encoded in mapping.items():
                if orig == '__UNKNOWN__':
                    continue
                # 保留类别和特殊值映射回自身；other_label 保持不变
                if encoded != self.other_label:
                    inverse_mapping[encoded] = orig

            X[col] = X[col].map(lambda v: inverse_mapping.get(v, v))

        return X

    def get_mapping(self, col: Optional[str] = None) -> Dict:
        """获取类别映射关系。

        :param col: 列名，如果为None则返回所有映射
        :return: 映射字典
        """
        if col is not None:
            return self.mapping_.get(col, {})
        return self.mapping_

    def get_top_categories(self, col: Optional[str] = None) -> Union[list, Dict[str, list]]:
        """获取各列保留的高频类别。

        :param col: 列名，如果为None则返回所有列
        :return: 类别列表或字典
        """
        if col is not None:
            return self.top_categories_.get(col, [])
        return self.top_categories_

    def get_summary(self) -> pd.DataFrame:
        """获取各列的降维摘要。

        :return: DataFrame，含列名/原始类别数/保留类别数/合并类别数/特殊值数
        """
        rows = []
        for col in self.cols_:
            counts = self.category_counts_.get(col, pd.Series(dtype=int))
            top_cats = self.top_categories_.get(col, [])
            special_set = set(self.special_values)
            n_special = sum(1 for sv in special_set if sv in set(counts.index) | special_set)
            rows.append({
                '列名': col,
                '原始类别数': len(counts),
                '保留类别数': len(top_cats),
                '合并为other': max(len(counts) - len(top_cats), 0),
                '特殊值数': n_special,
                'other标签': self.other_label,
            })
        return pd.DataFrame(rows)
