"""K-Means聚类分箱算法.

使用K-Means聚类算法将连续变量划分为K个簇，每个簇作为一个分箱。
这是一种无监督分箱方法，适用于发现数据中的自然分组。
"""

import warnings
from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from .base import BaseBinning


class KMeansBinning(BaseBinning):
    """K-Means聚类分箱算法.

    使用K-Means算法将特征值聚类成K个簇，每个簇作为一个分箱。
    根据聚类中心排序确定分箱边界（相邻聚类中心的中点）。

    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求坏样本率单调，默认为False
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，用于K-Means初始化，默认为None
    :param force_numerical: 是否强制作为数值型处理，默认为True
        - True: 将所有特征视为数值型进行K-Means聚类分箱
        - False: 自动检测特征类型
    :param n_init: K-Means初始化次数，默认为10
    :param max_iter: K-Means最大迭代次数，默认为300
    :param verbose: 是否输出详细信息，默认为False

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表

    **示例**

    >>> from hscredit.core.binning import KMeansBinning
    >>> # 基础用法
    >>> binner = KMeansBinning(max_n_bins=5, random_state=42)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    >>>
    >>> # 查看分箱统计
    >>> bin_table = binner.get_bin_table('feature_name')
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        force_numerical: bool = True,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
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
            decimal=decimal,
        )
        self.n_init = n_init
        self.max_iter = max_iter
        self.force_numerical = force_numerical

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'KMeansBinning':
        """拟合K-Means分箱.

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
                # 数值型特征：K-Means聚类分箱
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
        """对数值型特征进行K-Means聚类分箱.

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

        # 获取唯一值数量
        unique_values = np.unique(x_valid)
        n_unique = len(unique_values)

        # 如果唯一值数量小于等于最小分箱数，直接使用唯一值作为切分点
        if n_unique <= self.min_n_bins:
            return unique_values[:-1] if n_unique > 1 else np.array([])

        # 确定目标分箱数
        target_n_bins = min(self.max_n_bins, n_unique)
        target_n_bins = max(self.min_n_bins, target_n_bins)

        # 执行K-Means聚类
        splits = self._kmeans_clustering(x_valid, y_valid, target_n_bins)

        # 根据约束调整分箱
        splits = self._adjust_bins(x_valid, y_valid, splits)

        return splits

    def _kmeans_clustering(
        self,
        x: pd.Series,
        y: pd.Series,
        n_bins: int
    ) -> np.ndarray:
        """执行K-Means聚类并确定切分点.

        :param x: 特征数据（已清洗）
        :param y: 目标变量
        :param n_bins: 目标分箱数
        :return: 切分点数组
        """
        # 准备数据
        X_reshaped = x.values.reshape(-1, 1).astype(np.float64)

        # 使用 scipy 的 kmeans2 进行聚类
        # 设置 random_state 为可复现性
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 执行多次聚类取最佳结果
        best_centers = None
        best_labels = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            try:
                centers, labels = kmeans2(
                    X_reshaped,
                    n_bins,
                    iter=self.max_iter,
                    minit='points',  # 从数据点中选择初始中心
                    missing='warn'
                )
                
                # 计算惯性（簇内平方和）
                inertia = 0
                for i in range(n_bins):
                    cluster_points = X_reshaped[labels == i]
                    if len(cluster_points) > 0:
                        inertia += np.sum((cluster_points - centers[i]) ** 2)
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centers = centers
                    best_labels = labels
            except Exception:
                continue

        if best_centers is None:
            # 如果聚类失败，使用等频分箱作为备选
            return np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])

        # 获取聚类中心并排序
        centers = best_centers.flatten()
        sorted_indices = np.argsort(centers)
        sorted_centers = centers[sorted_indices]

        # 确定切分点（相邻聚类中心的中点）
        if len(sorted_centers) > 1:
            splits = (sorted_centers[:-1] + sorted_centers[1:]) / 2
        else:
            splits = np.array([])

        return splits

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
                # 需要重新聚类
                if iteration < max_iter - 1:
                    splits = self._kmeans_clustering(x, y, self.min_n_bins)
                break

        # 最终检查分箱数约束
        n_bins = len(splits) + 1

        # 如果分箱数超过最大值，需要减少
        if n_bins > self.max_n_bins:
            # 重新聚类，使用更少的簇数
            splits = self._kmeans_clustering(x, y, self.max_n_bins)

        # 如果分箱数少于最小值，需要增加
        elif n_bins < self.min_n_bins:
            unique_values = np.unique(x)
            if len(unique_values) >= self.min_n_bins:
                splits = self._kmeans_clustering(x, y, self.min_n_bins)

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
        :param kwargs: 其他参数
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
        >>> binner = KMeansBinning()
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
                # 根据WOE映射，优先使用_woe_maps_（从export/load导入）
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 生成测试数据（多峰分布）
    # 创建三个高斯分布的混合
    cluster1 = np.random.normal(20, 3, n_samples // 3)
    cluster2 = np.random.normal(50, 5, n_samples // 3)
    cluster3 = np.random.normal(80, 4, n_samples // 3)
    feature_values = np.concatenate([cluster1, cluster2, cluster3])
    np.random.shuffle(feature_values)

    X = pd.DataFrame({
        'feature1': feature_values,
        'feature2': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    })
    y = pd.Series(np.random.binomial(1, 0.3, n_samples))

    # 添加一些缺失值
    X.loc[np.random.choice(n_samples, 50, replace=False), 'feature1'] = np.nan

    print("=" * 50)
    print("K-Means聚类分箱测试")
    print("=" * 50)

    # 创建分箱器
    binner = KMeansBinning(max_n_bins=5, random_state=42, verbose=True)

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

    print("\n切分点:")
    for feature, splits in binner.splits_.items():
        print(f"  {feature}: {splits}")

    # 验证聚类效果
    print("\n各箱样本数统计 (feature1):")
    print(X_binned['feature1'].value_counts().sort_index())
