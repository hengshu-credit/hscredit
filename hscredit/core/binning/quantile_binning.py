"""等频分箱算法.

基于分位数切分的分箱方法，确保每个箱的样本数大致相等。
适用于数据分布不均匀或存在异常值的场景。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .base import BaseBinning


class QuantileBinning(BaseBinning):
    """等频分箱算法.

    将特征值按照分位数切分成多个区间，确保每个区间的样本数大致相等。
    适用于数据分布不均匀或存在异常值的场景。

    :param min_n_bins: 最小分箱数，默认为2
    :param max_n_bins: 最大分箱数，默认为10
    :param quantiles: 自定义分位点列表，如[0, 0.2, 0.5, 0.8, 1.0]，默认为None
        - 如果提供，将直接使用这些分位点进行分箱
    :param force_numerical: 是否强制作为数值型处理，默认为True
        - True: 将所有特征视为数值型进行等频分箱
        - False: 自动检测特征类型
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求坏样本率单调，默认为False
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表

    **示例**

    >>> from hscredit.core.binning import QuantileBinning
    >>> # 基础用法
    >>> binner = QuantileBinning(max_n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    >>>
    >>> # 使用自定义分位点
    >>> binner = QuantileBinning(quantiles=[0, 0.1, 0.3, 0.7, 0.9, 1.0])
    >>> binner.fit(X, y)
    """

    def __init__(
        self,
        target: str = 'target',
        min_n_bins: int = 2,
        max_n_bins: int = 10,
        quantiles: Optional[List[float]] = None,
        force_numerical: bool = True,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ):
        super().__init__(
            target=target,
            min_n_bins=min_n_bins,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            random_state=random_state,
            verbose=verbose,
        )
        self.quantiles = quantiles
        self.force_numerical = force_numerical
        
        # 验证quantiles参数
        if quantiles is not None:
            if not isinstance(quantiles, (list, tuple, np.ndarray)):
                raise ValueError("quantiles必须是列表或数组")
            if len(quantiles) < 2:
                raise ValueError("quantiles至少需要2个元素")
            if quantiles[0] != 0 or quantiles[-1] != 1:
                raise ValueError("quantiles第一个元素必须是0，最后一个必须是1")
            if not all(0 <= q <= 1 for q in quantiles):
                raise ValueError("quantiles所有元素必须在[0, 1]范围内")
            if not all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1)):
                raise ValueError("quantiles必须是非递减的")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'QuantileBinning':
        """拟合等频分箱.

        :param X: 训练数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1)
        :param kwargs: 其他参数
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            # 检测特征类型
            if self.force_numerical:
                feature_type = 'numerical'
            else:
                feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                # 类别型特征：每个类别作为一个箱
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                # 数值型特征：等频分箱
                splits = self._fit_numerical(X[feature], y)
                splits = self._round_splits(splits)
                if self.monotonic not in [False, None, 'none'] and len(splits) > 0:
                    from .monotonic_binning import MonotonicBinning
                    mono = MonotonicBinning(
                        monotonic=self.monotonic,
                        max_n_bins=self.max_n_bins,
                        min_n_bins=self.min_n_bins,
                        min_bin_size=self.min_bin_size,
                        special_codes=self.special_codes,
                        missing_separate=self.missing_separate,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    splits = mono._ensure_monotonic(X[feature].dropna(), y.loc[X[feature].dropna().index], splits, mono._detect_monotonic_mode(X[feature].dropna(), y.loc[X[feature].dropna().index], splits))
                    splits = self._round_splits(splits)
                self.splits_[feature] = splits
            self.n_bins_[feature] = len(splits) + 1

            # 计算分箱统计信息
            bins = self._apply_bins(X[feature], self.splits_[feature])
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

        self._is_fitted = True
        return self

    def _fit_numerical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> np.ndarray:
        """对数值型特征进行等频分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组
        """
        # 处理缺失值和特殊值
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            return np.array([])

        # 使用自定义分位点或基于max_n_bins计算
        if self.quantiles is not None:
            # 使用自定义分位点
            quantiles_to_use = self.quantiles
            target_n_bins = len(self.quantiles) - 1
        else:
            # 基于max_n_bins计算分位点
            target_n_bins = self.max_n_bins
            quantiles_to_use = np.linspace(0, 1, target_n_bins + 1)

        # 确保目标分箱数在约束范围内
        target_n_bins = max(self.min_n_bins, min(target_n_bins, self.max_n_bins))
        
        # 重新计算分位点（如果需要调整分箱数）
        if self.quantiles is None:
            quantiles_to_use = np.linspace(0, 1, target_n_bins + 1)

        # 获取唯一值
        unique_values = np.unique(x_valid)

        # 如果唯一值数量小于等于目标分箱数，直接使用唯一值作为切分点
        if len(unique_values) <= target_n_bins:
            return unique_values[:-1]  # 最后一个值不需要作为切分点

        # 计算分位数对应的切分点（排除0和1）
        split_quantiles = quantiles_to_use[1:-1]
        if len(split_quantiles) == 0:
            return np.array([])
            
        splits = np.percentile(x_valid, np.array(split_quantiles) * 100)

        # 处理重复值边界问题
        splits = self._handle_duplicate_boundaries(splits, x_valid)

        # 根据约束调整分箱数
        splits = self._adjust_bins(x_valid, y_valid, splits)

        return splits

    def _handle_duplicate_boundaries(
        self,
        splits: np.ndarray,
        x: pd.Series
    ) -> np.ndarray:
        """处理重复值边界问题.

        当分位数切分点与数据中的重复值重合时，调整切分点以避免空箱。

        :param splits: 初始切分点
        :param x: 特征数据
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits
            
        x_values = x.values
        unique_splits = []
        min_samples = self._get_min_samples(len(x))

        for i, split in enumerate(splits):
            # 如果当前切分点与已有切分点相同或更小，需要调整
            if i > 0 and len(unique_splits) > 0 and split <= unique_splits[-1]:
                # 找到该值的下一个不同值
                larger_values = x_values[x_values > unique_splits[-1]]
                if len(larger_values) > 0:
                    next_value = np.min(larger_values)
                    # 使用中间值，但要确保大于上一个切分点
                    split = max(split, (unique_splits[-1] + next_value) / 2)
                    # 确保不会等于上一个切分点
                    if split <= unique_splits[-1]:
                        split = unique_splits[-1] + 1e-10
                else:
                    # 如果没有更大的值，跳过这个切分点
                    continue

            # 检查该切分点是否会导致空箱或样本数过少
            if len(unique_splits) == 0:
                count = np.sum(x_values <= split)
            else:
                count = np.sum((x_values > unique_splits[-1]) & (x_values <= split))

            if count >= min_samples:
                unique_splits.append(split)

        return np.array(unique_splits)

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> np.ndarray:
        """对类别型特征进行分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组（类别列表）
        """
        # 类别型特征：按坏样本率排序
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        # 计算每个类别的坏样本率
        cat_stats = pd.DataFrame({
            'category': x_valid,
            'target': y_valid
        }).groupby('category')['target'].agg(['mean', 'count'])

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(x_valid))
        cat_stats = cat_stats[cat_stats['count'] >= min_samples]

        # 按坏样本率排序
        cat_stats = cat_stats.sort_values('mean')

        # 返回排序后的类别列表
        return cat_stats.index.tolist()

    def _adjust_bins(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray
    ) -> np.ndarray:
        """根据约束条件调整分箱.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits
            
        min_samples = self._get_min_samples(len(x))

        # 迭代调整直到满足约束
        max_iter = 20
        for iteration in range(max_iter):
            bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)

            # 检查每箱样本数
            bin_counts = bins.value_counts().sort_index()

            # 合并样本数过少的箱
            new_splits = []
            skip_next = False

            for i in range(len(splits)):
                if skip_next:
                    skip_next = False
                    continue

                # 当前箱的样本数
                count = bin_counts.get(i, 0)

                # 如果样本数过少且不是最后一个箱，尝试与下一个箱合并
                if count < min_samples and i < len(splits) - 1:
                    skip_next = True
                else:
                    new_splits.append(splits[i])

            new_splits = np.array(new_splits)

            # 如果切分点没有变化，退出循环
            if len(new_splits) == len(splits):
                break

            splits = new_splits
            
            # 检查分箱数约束
            n_bins = len(splits) + 1
            if n_bins < self.min_n_bins:
                # 需要增加分箱数，重新计算
                if self.quantiles is None and iteration < max_iter - 1:
                    quantiles = np.linspace(0, 1, self.min_n_bins + 1)
                    splits = np.percentile(x, quantiles[1:-1] * 100)
                    splits = self._handle_duplicate_boundaries(splits, x)
                break

        # 最终检查分箱数约束
        n_bins = len(splits) + 1
        
        # 如果分箱数超过最大值，减少分箱数
        if n_bins > self.max_n_bins:
            # 自定义分位点模式下，减少分箱数需要重新计算
            if self.quantiles is not None:
                # 保留首尾，均匀选择中间的分位点
                n_splits_needed = self.max_n_bins - 1
                if n_splits_needed > 0:
                    indices = np.linspace(0, len(splits) - 1, n_splits_needed).astype(int)
                    splits = splits[indices]
            else:
                # 非自定义模式下，重新计算分位数
                quantiles = np.linspace(0, 1, self.max_n_bins + 1)
                splits = np.percentile(x, quantiles[1:-1] * 100)
                splits = self._handle_duplicate_boundaries(splits, x)
        
        # 如果分箱数少于最小值，增加分箱数
        elif n_bins < self.min_n_bins:
            if self.quantiles is None:
                quantiles = np.linspace(0, 1, self.min_n_bins + 1)
                splits = np.percentile(x, quantiles[1:-1] * 100)
                splits = self._handle_duplicate_boundaries(splits, x)

        return splits

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数.

        :param n_total: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size < 1:
            return int(n_total * self.min_bin_size)
        return int(self.min_bin_size)

    def _apply_bins(
        self,
        x: pd.Series,
        splits: Union[np.ndarray, List]
    ) -> np.ndarray:
        """应用分箱.

        :param x: 特征数据
        :param splits: 切分点
        :return: 分箱索引
        """
        if isinstance(splits, list):
            # 类别型特征
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
                bins[x == cat] = i
            bins[x.isna()] = -1  # 缺失值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2  # 特殊值
            return bins
        else:
            # 数值型特征
            bins = np.zeros(len(x), dtype=int)

            # 处理缺失值
            if self.missing_separate:
                bins[x.isna()] = -1

            # 处理特殊值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2

            # 正常值分箱
            mask = x.notna()
            if self.special_codes:
                for code in self.special_codes:
                    mask = mask & (x != code)

            if len(splits) > 0:
                bins[mask] = np.digitize(x[mask], splits)
            else:
                bins[mask] = 0

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
        >>> binner = QuantileBinning()
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
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=['feature'])
                else:
                    X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            splits = self.splits_[feature]
            bins = self._apply_bins(X[feature], splits)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels(splits, bins)
                result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special') for b in bins]
            elif metric == 'woe':
                # 根据WOE映射，优先使用_export的_woe_maps_
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息，请先fit或加载包含WOE信息的规则")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 生成测试数据（偏态分布）
    X = pd.DataFrame({
        'feature1': np.random.exponential(2, n_samples),  # 指数分布
        'feature2': np.random.beta(2, 5, n_samples) * 100,  # Beta分布
        'feature3': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    })
    y = pd.Series(np.random.binomial(1, 0.3, n_samples))

    # 添加一些缺失值和重复值
    X.loc[np.random.choice(n_samples, 50, replace=False), 'feature1'] = np.nan
    # 添加大量重复值测试边界处理
    X.loc[np.random.choice(n_samples, 200, replace=False), 'feature2'] = 50

    print("=" * 50)
    print("等频分箱测试")
    print("=" * 50)

    # 创建分箱器
    binner = QuantileBinning(max_n_bins=5, verbose=True)

    # 拟合
    binner.fit(X, y)

    # 转换
    X_binned = binner.transform(X, metric='indices')
    print("\n分箱索引:")
    print(X_binned.head())

    X_woe = binner.transform(X, metric='woe')
    print("\nWOE值:")
    print(X_woe.head())

    # 查看分箱统计
    print("\n分箱统计表 (feature1):")
    print(binner.get_bin_table('feature1'))

    print("\n分箱统计表 (feature2):")
    print(binner.get_bin_table('feature2'))

    print("\n分箱统计表 (feature3):")
    print(binner.get_bin_table('feature3'))

    print("\n切分点:")
    for feature, splits in binner.splits_.items():
        print(f"  {feature}: {splits}")

    # 验证等频特性
    print("\n各箱样本数统计 (feature2):")
    print(X_binned['feature2'].value_counts().sort_index())
