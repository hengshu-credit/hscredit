"""等距分箱算法.

基于数值范围等距切分的分箱方法，适用于均匀分布的数据。
"""

from typing import Union, List, Dict, Optional, Any
import numpy as np
import pandas as pd

from .base import BaseBinning


class UniformBinning(BaseBinning):
    """等距分箱.

    将特征值的范围等分为指定数量的区间，每个区间宽度相同。
    适用于数据分布相对均匀的场景。

    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param max_bin_size: 每箱最大样本占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求单调性，默认为False
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None，如[-999, -98]
    :param left_clip: 左侧截断分位数，默认为None，如0.01表示截断1%分位数以下的值
    :param right_clip: 右侧截断分位数，默认为None，如0.99表示截断99%分位数以上的值
    :param force_numerical: 是否强制作为数值型处理，默认为True
        - True: 将所有特征视为数值型进行等距分箱（默认，因为等距分箱适用于数值型）
        - False: 自动检测特征类型（根据dtype判断）
    :param random_state: 随机种子，默认为None

    **示例**

    >>> from hscredit.core.binning import UniformBinning
    >>> # 基础用法
    >>> binner = UniformBinning(max_n_bins=5)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> 
    >>> # 使用截断处理异常值
    >>> binner = UniformBinning(max_n_bins=5, left_clip=0.01, right_clip=0.99)
    >>> binner.fit(X_train, y_train)
    >>>
    >>> # 指定特殊值
    >>> binner = UniformBinning(max_n_bins=5, special_codes=[-999, -98])
    >>> binner.fit(X_train, y_train)

    **注意**

    等距分箱的特点:
    1. 每个分箱的区间宽度相同
    2. 分箱边界由 (max - min) / n_bins 计算得出
    3. 支持通过left_clip/right_clip截断异常值
    4. 支持通过special_codes处理特殊值（如-999表示缺失）
    5. 默认force_numerical=True，确保进行数值型等距分箱
    6. 计算速度快，实现简单
    """

    def __init__(
        self,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        left_clip: Optional[float] = None,
        right_clip: Optional[float] = None,
        force_numerical: bool = True,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            missing_separate=missing_separate,
            special_codes=special_codes,
            random_state=random_state,
            **kwargs
        )
        self.left_clip = left_clip
        self.right_clip = right_clip
        self.force_numerical = force_numerical

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'UniformBinning':
        """拟合等距分箱.

        :param X: 训练数据
        :param y: 目标变量
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        for feature in X.columns:
            self._fit_feature(feature, X[feature], y)

        self._is_fitted = True
        return self

    def _fit_feature(
        self,
        feature: str,
        X: pd.Series,
        y: pd.Series
    ) -> None:
        """对单个特征进行分箱.

        :param feature: 特征名
        :param X: 特征数据
        :param y: 目标变量
        """
        # 检测特征类型
        if self.force_numerical:
            feature_type = 'numerical'
        else:
            feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        # 处理缺失值和特殊值
        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        # 获取有效数据（非缺失、非特殊值）
        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if feature_type == 'categorical':
            # 类别型变量：每个类别作为一个箱
            unique_values = X_valid.unique()
            self.splits_[feature] = np.array([])  # 类别型没有数值切分点
            self.n_bins_[feature] = len(unique_values)
        else:
            # 数值型变量：等距分箱
            # 转换为数值型，确保正确处理
            X_numeric = pd.to_numeric(X_valid, errors='coerce')
            X_numeric = X_numeric.dropna()
            
            if len(X_numeric) == 0:
                # 没有有效数值数据
                self.splits_[feature] = np.array([])
                self.n_bins_[feature] = 1
            else:
                # 应用截断（如果指定）
                min_val = X_numeric.min()
                max_val = X_numeric.max()
                
                # 保存原始边界用于后续处理
                clip_lower = None
                clip_upper = None
                
                if self.left_clip is not None and 0 <= self.left_clip < 1:
                    clip_lower = X_numeric.quantile(self.left_clip)
                    min_val = clip_lower
                
                if self.right_clip is not None and 0 < self.right_clip <= 1:
                    clip_upper = X_numeric.quantile(self.right_clip)
                    max_val = clip_upper
                
                # 保存截断边界
                self.clip_bounds_ = getattr(self, 'clip_bounds_', {})
                self.clip_bounds_[feature] = (clip_lower, clip_upper)
                
                # 计算切分点
                n_bins = max(self.min_n_bins, min(self.max_n_bins, 10))
                
                # 处理边界相同的情况（所有值相等）
                if max_val == min_val:
                    self.splits_[feature] = np.array([])
                    self.n_bins_[feature] = 1
                else:
                    bin_width = (max_val - min_val) / n_bins

                    # 生成切分点（不包括边界）
                    splits = []
                    for i in range(1, n_bins):
                        split_point = min_val + i * bin_width
                        splits.append(split_point)

                    self.splits_[feature] = self._round_splits(splits)
                    self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _assign_bins(
        self,
        X: pd.Series,
        feature: str
    ) -> np.ndarray:
        """为数据分配分箱索引.

        :param X: 特征数据
        :param feature: 特征名
        :return: 分箱索引数组
        """
        if self.feature_types_[feature] == 'categorical':
            # 类别型：使用类别编码
            return pd.Categorical(X).codes
        else:
            # 数值型：使用切分点
            splits = self.splits_[feature]
            
            # 获取截断边界
            clip_lower, clip_upper = None, None
            if hasattr(self, 'clip_bounds_') and feature in self.clip_bounds_:
                clip_lower, clip_upper = self.clip_bounds_[feature]

            # 处理缺失值和特殊值
            bins = np.zeros(len(X), dtype=int)

            for i, val in enumerate(X):
                if pd.isna(val):
                    bins[i] = -1  # 缺失值
                elif self.special_codes and val in self.special_codes:
                    bins[i] = -2  # 特殊值
                else:
                    # 尝试转换为数值
                    try:
                        val_numeric = float(val)
                    except (ValueError, TypeError):
                        # 无法转换为数值，使用哈希分配
                        val_numeric = hash(val) % (2**31)
                    
                    # 应用截断
                    if clip_lower is not None and val_numeric < clip_lower:
                        val_numeric = clip_lower
                    if clip_upper is not None and val_numeric > clip_upper:
                        val_numeric = clip_upper
                    
                    # 找到对应的分箱
                    if len(splits) == 0:
                        bin_idx = 0
                    else:
                        bin_idx = np.searchsorted(splits, val_numeric, side='right')
                    bins[i] = bin_idx

            return bins

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.
        
        将原始特征值转换为分箱索引、分箱标签或WOE值。
        
        :param X: 待转换数据, DataFrame或数组格式
        :param metric: 转换类型, 可选值:
            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理
            - 'bins': 返回分箱标签字符串, 用于可视化或报告
            - 'woe': 返回WOE值, 用于逻辑回归建模
        :param kwargs: 其他参数(保留兼容性)
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
        >>> binner = UniformBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature in self.splits_:
                bins = self._assign_bins(X[feature], feature)

                if metric == 'indices':
                    result[feature] = bins
                elif metric == 'bins':
                    # 转换为分箱标签
                    labels = self._get_bin_labels(self.splits_[feature], bins)
                    result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special')
                                      for b in bins]
                elif metric == 'woe':
                    # 转换为WOE值，优先使用_woe_maps_（从export/load导入）
                    if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                        woe_map = self._woe_maps_[feature]
                    elif feature in self.bin_tables_:
                        bin_table = self.bin_tables_[feature]
                        woe_map = {}
                        for idx, row in bin_table.iterrows():
                            bin_idx = idx
                            woe_map[bin_idx] = row['分档WOE值']
                        self._enrich_woe_map(woe_map, bin_table)
                    else:
                        raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                    result[feature] = [woe_map.get(b, 0) for b in bins]
                else:
                    raise ValueError(f"不支持的metric: {metric}")
            else:
                result[feature] = X[feature]

        return result
