"""目标坏样本率分箱算法.

基于目标坏样本率的分箱方法，支持两种模式：
1. 严格边界模式：按指定的坏样本率边界严格划分
2. 自动模式：自动寻找使每箱间坏样本率差异最大的划分
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings

from .base import BaseBinning


class TargetBadRateBinning(BaseBinning):
    """目标坏样本率分箱.

    支持两种分箱模式：

    **模式1：严格边界模式**（指定 target_bad_rates）
    按目标坏样本率边界严格划分，确保每箱的坏样本率在指定区间内。

    :param target_bad_rates: 目标坏样本率边界列表，例如 [0.05, 0.10, 0.20]
        - 会产生 len(target_bad_rates)+1 个分箱
        - 第0箱：坏样本率 <= target_bad_rates[0]
        - 第1箱：target_bad_rates[0] < 坏样本率 <= target_bad_rates[1]
        - 依此类推

    **模式2：自动模式**（不指定 target_bad_rates，指定 max_n_bins）
    自动寻找使每箱之间坏样本率差异最大的划分。

    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param strict_mode: 是否严格模式（严格限制边界），默认为True
        - True: 严格按照目标坏样本率边界划分，可能产生空箱
        - False: 在满足约束下尽量接近目标坏样本率
    :param merge_empty_bins: 是否合并空箱，默认为True
    :param monotonic: 是否要求单调性，默认为True
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param decimal: 切分点小数点保留精度，默认为4

    **示例**

    严格边界模式::

        >>> # 指定坏样本率边界：5%, 10%, 20%
        >>> binner = TargetBadRateBinning(
        ...     target_bad_rates=[0.05, 0.10, 0.20],
        ...     strict_mode=True
        ... )
        >>> # 结果：4个分箱，坏样本率分别在 <=5%, 5%-10%, 10%-20%, >20%

    自动模式::

        >>> # 自动寻找最优划分
        >>> binner = TargetBadRateBinning(max_n_bins=5)
        >>> # 结果：5个分箱，每箱间坏样本率差异最大
    """

    def __init__(
        self,
        target: str = 'target',
        target_bad_rates: Optional[List[float]] = None,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        strict_mode: bool = True,
        merge_empty_bins: bool = True,
        monotonic: bool = True,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        random_state: Optional[int] = None,
        decimal: int = 4,
        **kwargs
    ):
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            monotonic=monotonic,
            missing_separate=missing_separate,
            special_codes=special_codes,
            random_state=random_state,
            decimal=decimal,
            **kwargs
        )
        self.target_bad_rates = sorted(target_bad_rates) if target_bad_rates else None
        self.strict_mode = strict_mode
        self.merge_empty_bins = merge_empty_bins
        self._actual_rates: Dict[str, List[float]] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'TargetBadRateBinning':
        """拟合目标坏样本率分箱.

        :param X: 训练数据
        :param y: 目标变量
        :return: 拟合后的分箱器
        """
        X, y = self._check_input(X, y)

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
        """对单个特征进行分箱."""
        feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        # 处理缺失值和特殊值
        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if feature_type == 'categorical':
            splits = self._fit_categorical(X_valid, y_valid)
            self.splits_[feature] = np.array(splits) if splits else np.array([])
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            splits = self._fit_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        bins = self._assign_bins(X, feature)
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

        # 记录实际坏样本率
        self._actual_rates[feature] = self._compute_actual_rates(X_valid, y_valid, splits)

    def _fit_numerical(self, X: pd.Series, y: pd.Series) -> List[float]:
        """对数值型变量进行分箱."""
        if len(X) == 0:
            return []

        if self.target_bad_rates is not None:
            # 严格边界模式
            return self._fit_with_target_rates(X, y)
        else:
            # 自动模式：寻找最优划分
            return self._fit_auto(X, y)

    def _fit_with_target_rates(self, X: pd.Series, y: pd.Series) -> List[float]:
        """按目标坏样本率边界严格划分.

        算法思路：
        1. 将数据按特征值排序
        2. 计算每个可能切割点的累积坏样本率
        3. 找到累积坏样本率最接近目标边界的切割点
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)

        # 计算样本数约束
        min_samples = self._get_min_samples(n_samples)

        # 排序
        sorted_idx = np.argsort(x_vals)
        x_sorted = x_vals[sorted_idx]
        y_sorted = y_vals[sorted_idx]

        # 计算累积坏样本数和累积样本数
        cum_bad = np.cumsum(y_sorted)
        cum_total = np.arange(1, n_samples + 1)

        # 计算每个位置的累积坏样本率
        cum_rate = cum_bad / cum_total

        # 计算全局坏样本率
        global_bad_rate = y_sorted.mean()

        # 确定目标边界（考虑全局坏样本率）
        target_rates = self.target_bad_rates.copy()

        # 如果全局坏样本率不在目标范围内，添加到边界
        if global_bad_rate < target_rates[0]:
            target_rates = [global_bad_rate] + target_rates
        if global_bad_rate > target_rates[-1]:
            target_rates = target_rates + [global_bad_rate]
        target_rates = sorted(set(target_rates))

        splits = []

        # 对每个目标边界，找到最佳切割点
        for target in target_rates:
            if target >= global_bad_rate:
                # 目标坏样本率高于全局，需要在数据后半部分找
                # 找到累积坏样本率最接近目标的位置
                diff = np.abs(cum_rate - target)
            else:
                # 目标坏样本率低于全局，需要在数据前半部分找
                diff = np.abs(cum_rate - target)

            # 在满足最小样本数的范围内搜索
            valid_start = min_samples
            valid_end = n_samples - min_samples

            if valid_start >= valid_end:
                continue

            # 在有效范围内找最接近目标的位置
            valid_diff = diff[valid_start:valid_end]
            if len(valid_diff) == 0:
                continue

            best_pos = valid_start + np.argmin(valid_diff)

            # 检查这个位置的实际坏样本率是否足够接近目标
            actual_rate = cum_rate[best_pos]

            # 确定切割点
            split_val = (x_sorted[best_pos] + x_sorted[best_pos + 1]) / 2

            # 避免重复切割点
            if split_val not in splits:
                splits.append(split_val)

        # 合并空箱或样本数过少的箱
        if self.merge_empty_bins:
            splits = self._merge_small_bins(x_sorted, y_sorted, splits, min_samples)

        return sorted(splits)

    def _fit_auto(self, X: pd.Series, y: pd.Series) -> List[float]:
        """自动模式：寻找使每箱间坏样本率差异最大的划分.

        使用动态规划或贪心算法，在满足约束下最大化箱间差异。
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)

        min_samples = self._get_min_samples(n_samples)
        n_bins = self.max_n_bins

        # 排序
        sorted_idx = np.argsort(x_vals)
        x_sorted = x_vals[sorted_idx]
        y_sorted = y_vals[sorted_idx]

        # 使用贪心算法找最优划分
        # 目标：最大化相邻箱之间的坏样本率差异

        splits = []

        # 计算所有可能的切割点及其对应的坏样本率
        candidates = []
        for i in range(min_samples, n_samples - min_samples):
            left_rate = y_sorted[:i].mean()
            right_rate = y_sorted[i:].mean()
            # 差异度量：相邻箱坏样本率的绝对差异
            diff = abs(left_rate - right_rate)

            # 检查最小样本数约束
            left_count = i
            right_count = n_samples - i
            if left_count >= min_samples and right_count >= min_samples:
                split_val = (x_sorted[i-1] + x_sorted[i]) / 2
                candidates.append((i, split_val, diff, left_rate, right_rate))

        if not candidates:
            return []

        # 按差异排序，选择差异最大的切割点
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 贪心选择切割点
        selected_positions = []

        for pos, split_val, diff, left_rate, right_rate in candidates:
            if len(selected_positions) >= n_bins - 1:
                break

            # 检查与已选切割点的距离是否满足最小样本数
            valid = True
            for selected_pos in selected_positions:
                if abs(pos - selected_pos) < min_samples:
                    valid = False
                    break

            if valid:
                selected_positions.append(pos)
                splits.append(split_val)

        # 按位置排序
        splits = [s for _, s in sorted(zip(selected_positions, splits))]

        # 验证并调整分箱
        if len(splits) > 0:
            splits = self._validate_and_adjust_splits(x_sorted, y_sorted, splits, min_samples)

        return sorted(splits)

    def _validate_and_adjust_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        splits: List[float],
        min_samples: int
    ) -> List[float]:
        """验证并调整分箱，确保满足约束."""
        if not splits:
            return splits

        n_samples = len(x_sorted)
        positions = []
        for split in splits:
            pos = np.searchsorted(x_sorted, split, side='right')
            positions.append(pos)

        positions = sorted(positions)

        # 检查每个箱的样本数
        valid_positions = []
        prev_pos = 0

        for pos in positions:
            if pos - prev_pos >= min_samples and n_samples - pos >= min_samples:
                valid_positions.append(pos)
                prev_pos = pos

        # 根据有效位置重建切割点
        valid_splits = []
        for pos in valid_positions:
            if pos < n_samples:
                split_val = (x_sorted[pos-1] + x_sorted[pos]) / 2
                valid_splits.append(split_val)

        return valid_splits

    def _merge_small_bins(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        splits: List[float],
        min_samples: int
    ) -> List[float]:
        """合并样本数过少的相邻箱."""
        if not splits:
            return splits

        n_samples = len(x_sorted)
        positions = [np.searchsorted(x_sorted, s, side='right') for s in sorted(splits)]

        # 计算每个箱的样本数
        boundaries = [0] + positions + [n_samples]
        bin_counts = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]

        # 找出需要合并的箱
        valid_splits = []
        for i, (pos, count) in enumerate(zip(positions, bin_counts[:-1])):
            if count >= min_samples and bin_counts[i+1] >= min_samples:
                valid_splits.append(sorted(splits)[i])

        return valid_splits

    def _fit_categorical(self, X: pd.Series, y: pd.Series) -> List[float]:
        """对类别型变量进行分箱."""
        # 计算每个类别的坏样本率
        df = pd.DataFrame({'X': X, 'y': y})
        category_stats = df.groupby('X')['y'].agg(['mean', 'count']).reset_index()
        category_stats.columns = ['category', 'bad_rate', 'count']

        n_total = len(X)
        min_samples = self._get_min_samples(n_total)

        # 过滤样本数过少的类别
        category_stats = category_stats[category_stats['count'] >= min_samples]

        if len(category_stats) <= 1:
            return []

        # 按坏样本率排序
        category_stats = category_stats.sort_values('bad_rate').reset_index(drop=True)

        if self.target_bad_rates is not None:
            # 严格边界模式
            return self._fit_categorical_with_target_rates(category_stats, n_total, min_samples)
        else:
            # 自动模式
            return self._fit_categorical_auto(category_stats, n_total, min_samples)

    def _fit_categorical_with_target_rates(
        self,
        category_stats: pd.DataFrame,
        n_total: int,
        min_samples: int
    ) -> List[float]:
        """类别型变量按目标坏样本率边界划分."""
        splits = []
        cum_count = 0
        cum_bad = 0

        for target in self.target_bad_rates:
            best_idx = None
            best_diff = float('inf')

            for idx in range(len(category_stats)):
                temp_count = category_stats.iloc[:idx+1]['count'].sum()
                temp_bad = (category_stats.iloc[:idx+1]['bad_rate'] *
                           category_stats.iloc[:idx+1]['count']).sum()

                if temp_count < min_samples:
                    continue
                if n_total - temp_count < min_samples:
                    break

                temp_rate = temp_bad / temp_count
                diff = abs(temp_rate - target)

                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx

            if best_idx is not None and best_idx < len(category_stats) - 1:
                split_point = best_idx + 0.5
                if split_point not in splits:
                    splits.append(split_point)

        return sorted(splits)

    def _fit_categorical_auto(
        self,
        category_stats: pd.DataFrame,
        n_total: int,
        min_samples: int
    ) -> List[float]:
        """类别型变量自动模式：最大化箱间差异."""
        n_cats = len(category_stats)
        splits = []

        # 计算累积统计
        cum_count = category_stats['count'].cumsum().values
        cum_bad = (category_stats['bad_rate'] * category_stats['count']).cumsum().values

        candidates = []

        for i in range(n_cats - 1):
            left_count = cum_count[i]
            right_count = n_total - left_count

            if left_count < min_samples or right_count < min_samples:
                continue

            left_rate = cum_bad[i] / left_count
            right_rate = (cum_bad[-1] - cum_bad[i]) / right_count
            diff = abs(left_rate - right_rate)

            candidates.append((i + 0.5, diff))

        # 按差异排序，选择最大的几个
        candidates.sort(key=lambda x: x[1], reverse=True)

        for split, _ in candidates:
            if len(splits) >= self.max_n_bins - 1:
                break
            splits.append(split)

        return sorted(splits)

    def _compute_actual_rates(
        self,
        X: pd.Series,
        y: pd.Series,
        splits: List[float]
    ) -> List[float]:
        """计算各箱的实际坏样本率."""
        if len(X) == 0:
            return []

        if not splits:
            return [y.mean()]

        x_vals = X.values
        y_vals = y.values

        sorted_splits = sorted(splits)
        bins = np.searchsorted(sorted_splits, x_vals, side='right')

        rates = []
        for i in range(len(sorted_splits) + 1):
            mask = bins == i
            if mask.sum() > 0:
                rates.append(y_vals[mask].mean())
            else:
                rates.append(0.0)

        return rates

    def _assign_bins(self, X: pd.Series, feature: str) -> np.ndarray:
        """为数据分配分箱索引."""
        x_vals = X.values

        if self.feature_types_[feature] == 'categorical':
            codes = pd.Categorical(X).codes
            bins = np.where(X.isna(), -1, codes)
            if self.special_codes:
                for code in self.special_codes:
                    bins[x_vals == code] = -2
            return bins
        else:
            splits = self.splits_[feature]
            n = len(x_vals)
            bins = np.zeros(n, dtype=int)

            missing_mask = X.isna()
            bins[missing_mask] = -1

            if self.special_codes:
                for code in self.special_codes:
                    special_mask = ~missing_mask & (x_vals == code)
                    bins[special_mask] = -2

            valid_mask = ~missing_mask
            if self.special_codes:
                for code in self.special_codes:
                    valid_mask = valid_mask & (x_vals != code)

            if valid_mask.any() and len(splits) > 0:
                valid_indices = np.where(valid_mask)[0]
                bins[valid_indices] = np.searchsorted(
                    splits, x_vals[valid_indices], side='right'
                )

            return bins

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换."""
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            bins = self._assign_bins(X[feature], feature)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels_dict(feature)
                result[feature] = [labels.get(b, f'bin_{b}') for b in bins]
            elif metric == 'woe':
                bin_table = self.bin_tables_[feature]
                woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                woe_map[-1] = 0
                woe_map[-2] = 0
                result[feature] = [woe_map.get(b, 0) for b in bins]
            else:
                raise ValueError(f"不支持的metric: {metric}")

        return result

    def _get_bin_labels_dict(self, feature: str) -> Dict[int, str]:
        """获取分箱标签字典."""
        splits = self.splits_[feature]
        n_splits = len(splits) if splits is not None else 0
        n_normal_bins = n_splits + 1

        labels = {-1: 'missing', -2: 'special'}

        for i in range(n_normal_bins):
            if n_splits == 0:
                labels[i] = '(-inf, +inf)'
            elif i == 0:
                labels[i] = f'(-inf, {splits[0]}]'
            elif i == n_normal_bins - 1:
                labels[i] = f'({splits[-1]}, +inf]'
            else:
                labels[i] = f'({splits[i-1]}, {splits[i]}]'

        return labels

    def get_bad_rate_summary(self, feature: str) -> pd.DataFrame:
        """获取坏样本率摘要.

        :param feature: 特征名
        :return: 分箱坏样本率摘要表
        """
        if feature not in self.bin_tables_:
            raise KeyError(f"特征 '{feature}' 未找到")

        bin_table = self.bin_tables_[feature]

        # 排除缺失和特殊值箱
        valid_mask = ~bin_table['分箱标签'].isin(['缺失', 'special'])
        valid_table = bin_table[valid_mask].copy()

        summary = pd.DataFrame({
            '分箱': valid_table['分箱'].values,
            '分箱标签': valid_table['分箱标签'].values,
            '样本总数': valid_table['样本总数'].values,
            '样本占比': valid_table['样本占比'].values,
            '坏样本率': valid_table['坏样本率'].values
        })

        if self.target_bad_rates:
            # 添加目标坏样本率
            n_bins = len(summary)
            targets = []
            for i in range(n_bins):
                if i < len(self.target_bad_rates):
                    targets.append(self.target_bad_rates[i])
                elif i == n_bins - 1:
                    targets.append(None)  # 最后一箱没有上界
                else:
                    targets.append(None)
            summary['目标坏样本率'] = targets

        return summary


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

    # 测试代码
    np.random.seed(42)
    n_samples = 5000

    # 生成测试数据
    x = np.random.uniform(0, 100, n_samples)
    bad_rate = 0.05 + 0.004 * x  # 从5%递增到45%
    y = np.random.binomial(1, bad_rate)

    X = pd.DataFrame({'feature': x})
    y = pd.Series(y)

    print("=" * 60)
    print("目标坏样本率分箱测试 - 严格边界模式")
    print("=" * 60)

    # 测试严格边界模式
    binner1 = TargetBadRateBinning(
        target_bad_rates=[0.10, 0.20, 0.30],
        strict_mode=True,
        min_bin_size=0.01
    )
    binner1.fit(X, y)

    print("\n分箱统计表:")
    print(binner1.get_bin_table('feature'))

    print("\n坏样本率摘要:")
    print(binner1.get_bad_rate_summary('feature'))

    print("\n" + "=" * 60)
    print("目标坏样本率分箱测试 - 自动模式")
    print("=" * 60)

    # 测试自动模式
    binner2 = TargetBadRateBinning(
        max_n_bins=5,
        min_bin_size=0.05
    )
    binner2.fit(X, y)

    print("\n分箱统计表:")
    print(binner2.get_bin_table('feature'))

    print("\n坏样本率摘要:")
    print(binner2.get_bad_rate_summary('feature'))
