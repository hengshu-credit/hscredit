"""Best IV 分箱算法.

基于最大化IV（Information Value）的分箱方法，寻找能够最大化预测能力的分箱点。
IV是衡量特征预测能力的重要指标。

算法流程：
1. 预分割：将数据分成足够细的初始箱（默认50个）
2. 合并优化：使用贪心算法，在单调性约束下选择最优分割点
3. 目标：最大化IV值
"""

from typing import Union, List, Dict, Optional, Any
import numpy as np
import pandas as pd

from .base import BaseBinning


class BestIVBinning(BaseBinning):
    """Best IV 分箱.

    基于最大化IV的分箱方法，寻找能够最大化预测能力的分箱点。
    IV是衡量特征预测能力的重要指标。

    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 坏样本率单调性约束，默认为False
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param random_state: 随机种子，默认为None

    **参考样例**

    >>> from hscredit.core.binning import BestIVBinning
    >>> binner = BestIVBinning(max_n_bins=5)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    **注意**

    Best IV 分箱的特点:
    1. 最大化IV值
    2. IV > 0.02 表示特征有预测能力
    3. IV > 0.1 表示特征有较强的预测能力
    4. IV > 0.3 表示特征预测能力过强（可能有问题）
    5. 使用贪心算法逐步优化
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
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            target=target,
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'BestIVBinning':
        """拟合 Best IV 分箱.

        :param X: 训练数据
        :param y: 目标变量
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        for feature in X.columns:
            self._fit_feature(feature, X[feature], y)

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
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
        feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        # 处理缺失值和特殊值
        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        # 获取有效数据
        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if feature_type == 'categorical':
            # 类别型变量：按IV排序后分箱
            splits = self._best_iv_categorical(X_valid, y_valid)
            self.splits_[feature] = np.array(splits)
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            # 数值型变量：Best IV分箱
            # 使用全量样本规模对齐后处理约束（避免 valid-only 与 full-data 约束不一致）
            self._n_total_samples = len(X)
            splits = self._best_iv_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _best_iv_numerical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对数值型变量进行Best IV分箱 (优化版本).

        使用排序后数据的累积统计信息快速计算IV值。

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表
        """
        # 转换为 numpy 数组加速计算
        x_vals = X.values
        y_vals = y.values

        # 获取唯一值
        unique_values = np.unique(x_vals)

        if len(unique_values) <= self.max_n_bins:
            # 唯一值较少时，直接使用唯一值边界（避免退化为单箱）
            return unique_values[:-1].tolist()

        # 限制候选分割点数量
        max_candidates = min(len(unique_values) - 1, 100)
        if len(unique_values) > max_candidates + 1:
            # 使用样本分位点（按频次加权），而非唯一值分位点
            quantiles = np.linspace(0, 1, max_candidates + 1)
            candidates = np.quantile(x_vals, quantiles[1:-1])
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            candidates = np.unique(candidates)
            candidates = candidates[(candidates > x_min) & (candidates < x_max)]
            # 去重并确保在开区间内
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            candidates = np.unique(candidates)
            candidates = candidates[(candidates > x_min) & (candidates < x_max)]
        else:
            candidates = (unique_values[:-1] + unique_values[1:]) / 2

        # 预计算排序后的数据
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]

        # 计算总体统计
        total_good = np.sum(y_vals == 0)
        total_bad = np.sum(y_vals == 1)
        n_total_samples = int(getattr(self, '_n_total_samples', len(x_vals)))

        if total_good == 0 or total_bad == 0:
            return []

        # 预计算累积统计
        cum_bad = np.cumsum(y_sorted)
        cum_good = np.arange(1, len(y_sorted) + 1) - cum_bad

        # 使用贪心算法选择最优分割点
        selected_splits = []
        enforce_monotonic = self.monotonic in [
            True, 'auto', 'auto_asc_desc', 'auto_heuristic',
            'ascending', 'descending', 'peak', 'valley',
            'peak_heuristic', 'valley_heuristic'
        ]

        while len(selected_splits) < self.max_n_bins - 1 and len(candidates) > 0:
            best_iv = -1.0
            best_violation = np.inf
            best_split_idx = -1
            best_split = None

            for i, candidate in enumerate(candidates):
                test_splits = sorted(selected_splits + [candidate])
                iv = self._calc_iv_fast(
                    x_sorted, y_sorted, cum_good, cum_bad,
                    total_good, total_bad, test_splits,
                    n_total_samples=n_total_samples
                )

                if iv < 0:
                    continue

                violation = 0
                if enforce_monotonic and len(test_splits) > 0:
                    bad_rates = self._calc_bad_rates_fast(
                        x_sorted, cum_good, cum_bad, test_splits
                    )
                    target_mode = self._resolve_monotonic_target_mode(
                        bad_rates, self.monotonic
                    )
                    violation = self._count_monotonic_violations(
                        bad_rates, target_mode
                    )

                # 优先减少单调违例，其次最大化 IV
                if (violation < best_violation) or (
                    violation == best_violation and iv > best_iv + 1e-12
                ):
                    best_iv = iv
                    best_violation = violation
                    best_split_idx = i
                    best_split = candidate

            # 若新增分割点会引入单调违例，且已满足最小分箱数，则停止扩展
            min_splits_required = max(1, self.min_n_bins - 1)
            if enforce_monotonic and best_split is not None and best_violation > 0 and len(selected_splits) >= min_splits_required:
                break

            if best_split is not None:
                selected_splits.append(best_split)
                candidates = np.delete(candidates, best_split_idx)
            else:
                break

        return sorted(selected_splits)

    def _calc_iv_fast(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        cum_good: np.ndarray,
        cum_bad: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float],
        n_total_samples: Optional[int] = None
    ) -> float:
        """快速计算IV值.

        使用预计算的累积统计信息。

        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param cum_good: 累积好样本数
        :param cum_bad: 累积坏样本数
        :param total_good: 总好样本数
        :param total_bad: 总坏样本数
        :param splits: 分割点列表
        :return: IV值
        """
        if not splits:
            return 0.0

        # 找到所有分割点的位置
        split_positions = [np.searchsorted(x_sorted, s, side='right') for s in sorted(splits)]
        split_positions = [0] + split_positions + [len(x_sorted)]

        iv = 0.0
        eps = 1e-10
        base_n_samples = int(len(x_sorted) if n_total_samples is None else n_total_samples)
        min_samples = self._get_min_samples(base_n_samples)

        for i in range(len(split_positions) - 1):
            start = split_positions[i]
            end = split_positions[i + 1]

            if start >= end:
                continue

            # 约束：每箱最小样本数
            if (end - start) < min_samples:
                return -1.0

            # 使用累积统计计算该箱的好/坏样本数
            good_in_bin = cum_good[end - 1] - (cum_good[start - 1] if start > 0 else 0)
            bad_in_bin = cum_bad[end - 1] - (cum_bad[start - 1] if start > 0 else 0)

            good_dist = good_in_bin / total_good
            bad_dist = bad_in_bin / total_bad

            # 避免除零和对零取对数
            if good_dist > eps and bad_dist > eps:
                iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)

        return iv

    def _calc_bad_rates_fast(
        self,
        x_sorted: np.ndarray,
        cum_good: np.ndarray,
        cum_bad: np.ndarray,
        splits: List[float]
    ) -> np.ndarray:
        """基于累积统计快速计算各箱坏样本率。"""
        if not splits:
            return np.array([], dtype=float)

        split_positions = [np.searchsorted(x_sorted, s, side='right') for s in sorted(splits)]
        split_positions = [0] + split_positions + [len(x_sorted)]

        bad_rates: List[float] = []
        for i in range(len(split_positions) - 1):
            start = split_positions[i]
            end = split_positions[i + 1]
            if start >= end:
                continue
            good_in_bin = cum_good[end - 1] - (cum_good[start - 1] if start > 0 else 0)
            bad_in_bin = cum_bad[end - 1] - (cum_bad[start - 1] if start > 0 else 0)
            count = float(good_in_bin + bad_in_bin)
            bad_rates.append(float(bad_in_bin / count) if count > 0 else 0.0)

        return np.asarray(bad_rates, dtype=float)

    def _best_iv_categorical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对类别型变量进行Best IV分箱 (优化版本).

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表
        """
        # 计算总体统计
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        if total_good == 0 or total_bad == 0:
            return []

        # 使用向量化操作计算类别统计
        df = pd.DataFrame({'X': X, 'y': y})
        category_stats = df.groupby('X')['y'].agg(['sum', 'count']).reset_index()
        category_stats.columns = ['category', 'bad_count', 'count']
        category_stats['good_count'] = category_stats['count'] - category_stats['bad_count']

        # 计算WOE
        eps = 1e-10
        category_stats['good_dist'] = category_stats['good_count'] / total_good
        category_stats['bad_dist'] = category_stats['bad_count'] / total_bad
        category_stats['woe'] = np.log(
            (category_stats['bad_dist'] + eps) / (category_stats['good_dist'] + eps)
        )

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(X))
        category_stats = category_stats[category_stats['count'] >= min_samples]

        if len(category_stats) <= self.max_n_bins:
            return []

        # 按WOE排序
        category_stats = category_stats.sort_values('woe')

        # 返回编码边界
        n_categories = len(category_stats)
        return [i - 0.5 for i in range(1, min(n_categories, self.max_n_bins))]

    def _calc_iv(
        self,
        X: pd.Series,
        y: pd.Series,
        splits: List[float]
    ) -> float:
        """计算IV值 (兼容旧代码).

        :param X: 特征数据
        :param y: 目标变量
        :param splits: 分割点列表
        :return: IV值
        """
        x_vals = X.values if isinstance(X, pd.Series) else X
        y_vals = y.values if isinstance(y, pd.Series) else y

        # 根据分割点分箱
        bins = np.searchsorted(splits, x_vals, side='right')

        # 计算总体统计
        total_good = np.sum(y_vals == 0)
        total_bad = np.sum(y_vals == 1)

        if total_good == 0 or total_bad == 0:
            return 0.0

        # 使用 bincount 快速计算每箱统计
        n_bins = len(splits) + 1
        bin_good = np.bincount(bins[y_vals == 0], minlength=n_bins).astype(float)
        bin_bad = np.bincount(bins[y_vals == 1], minlength=n_bins).astype(float)

        # 计算IV（使用平滑处理避免log(0)和除零错误）
        eps = 1e-10
        # 平滑处理：将0替换为eps
        bin_good_smooth = np.where(bin_good == 0, eps, bin_good)
        bin_bad_smooth = np.where(bin_bad == 0, eps, bin_bad)
        
        # 重新计算平滑后的总数
        total_good_smooth = bin_good_smooth.sum()
        total_bad_smooth = bin_bad_smooth.sum()
        
        # 计算分布（保持归一化）
        good_dist = bin_good_smooth / total_good_smooth
        bad_dist = bin_bad_smooth / total_bad_smooth

        # 计算IV
        iv = np.sum((bad_dist - good_dist) * np.log(bad_dist / good_dist))

        return iv

    def _assign_bins(
        self,
        X: pd.Series,
        feature: str
    ) -> np.ndarray:
        """为数据分配分箱索引 (优化版本).

        :param X: 特征数据
        :param feature: 特征名
        :return: 分箱索引数组
        """
        x_vals = X.values

        if self.feature_types_[feature] == 'categorical':
            codes = pd.Categorical(X).codes
            return np.where(X.isna(), -1, codes)
        else:
            splits = self.splits_[feature]
            n = len(x_vals)
            bins = np.zeros(n, dtype=int)

            # 处理缺失值
            missing_mask = X.isna()
            bins[missing_mask] = -1

            # 处理特殊值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x_vals == code] = -2

            # 正常值
            valid_mask = ~missing_mask
            if self.special_codes:
                for code in self.special_codes:
                    valid_mask = valid_mask & (x_vals != code)

            if valid_mask.any() and len(splits) > 0:
                bins[valid_mask] = np.searchsorted(
                    splits, x_vals[valid_mask], side='right'
                )

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
        >>> binner = BestIVBinning()
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
                    labels = self._get_bin_labels(self.splits_[feature], bins)
                    result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special')
                                      for b in bins]
                elif metric == 'woe':
                    # 优先使用_woe_maps_（从export/load导入）
                    if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                        woe_map = self._woe_maps_[feature]
                    elif feature in self.bin_tables_:
                        woe_map = dict(zip(
                            range(len(self.bin_tables_[feature])),
                            self.bin_tables_[feature]['分档WOE值']
                        ))
                        self._enrich_woe_map(woe_map, self.bin_tables_[feature])
                    else:
                        raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                    result[feature] = [woe_map.get(b, 0) for b in bins]
                else:
                    raise ValueError(f"不支持的metric: {metric}")
            else:
                result[feature] = X[feature]

        return result
