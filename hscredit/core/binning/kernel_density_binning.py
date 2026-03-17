"""核密度分箱.

基于核密度估计（KDE）识别数据分布的模态和谷值，进行自适应分箱。
适用于多峰分布和需要反映数据自然结构的场景。

优化版本：
- 使用scipy.stats.gaussian_kde进行高效核密度估计
- 改进峰谷检测算法，支持自适应参数
- 结合目标变量y优化切分点选择
- 支持单调性约束
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from .base import BaseBinning


class KernelDensityBinning(BaseBinning):
    """核密度分箱.

    使用核密度估计识别数据分布的局部极大值（峰）和极小值（谷），
    以谷值作为分箱边界，使分箱反映数据的自然分布结构。

    优化版本特点：
    - 使用scipy.stats.gaussian_kde进行高效核密度估计
    - 改进峰谷检测算法，支持自适应参数
    - 结合目标变量y优化切分点选择
    - 支持单调性约束

    :param kernel: 核函数类型，默认为'gaussian'，可选'gaussian', 'epanechnikov', 'tophat'
    :param bandwidth: 带宽，默认为'scott'，可选'scott', 'silverman'或具体数值
    :param min_peak_height: 最小峰高（相对于最大密度），默认为0.05
    :param min_peak_distance: 峰之间的最小距离（相对于数据范围），默认为0.1
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param monotonic: 单调性约束，默认为None，可选'ascending', 'descending', 'peak', 'valley'或None
    :param use_target: 是否结合目标变量优化切分点，默认为True
    :param n_grid_points: 核密度估计的网格点数，默认为500
    :param smooth_density: 是否对密度曲线进行平滑处理，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **示例**

    >>> from hscredit.core.binning import KernelDensityBinning
    >>> binner = KernelDensityBinning(kernel='gaussian', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """

    def __init__(
        self,
        kernel: str = 'gaussian',
        bandwidth: Union[str, float] = 'scott',
        min_peak_height: float = 0.05,
        min_peak_distance: float = 0.1,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        monotonic: Optional[str] = None,
        use_target: bool = True,
        n_grid_points: int = 500,
        smooth_density: bool = True,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ):
        super().__init__(
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            random_state=random_state,
            verbose=verbose,
        )
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.use_target = use_target
        self.n_grid_points = n_grid_points
        self.smooth_density = smooth_density

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'KernelDensityBinning':
        """拟合核密度分箱."""
        X, y = self._check_input(X, y)

        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1 if isinstance(splits, np.ndarray) else len(splits)

            bins = self._apply_bins(X[feature], splits, feature_type)
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
        """对数值型特征进行核密度分箱."""
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask].values
        y_valid = y[mask].values if self.use_target else None

        if len(x_valid) == 0:
            return np.array([])

        # 计算核密度估计
        kde_x, kde_density = self._compute_kde_optimized(x_valid)

        # 寻找峰和谷
        peaks, valleys = self._find_peaks_and_valleys_optimized(kde_x, kde_density, x_valid)

        # 选择作为切分点的谷值
        splits = self._select_valleys_as_splits_optimized(kde_x, kde_density, valleys, x_valid, y_valid)

        return splits

    def _compute_kde_optimized(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """优化的核密度估计 - 使用scipy.stats.gaussian_kde."""
        # 创建评估网格
        x_min, x_max = x.min(), x.max()
        pad = (x_max - x_min) * 0.1
        kde_x = np.linspace(x_min - pad, x_max + pad, self.n_grid_points)

        # 使用scipy的gaussian_kde
        if self.kernel == 'gaussian':
            # 使用scipy.stats.gaussian_kde
            kde = stats.gaussian_kde(x)

            # 设置带宽
            if isinstance(self.bandwidth, str):
                # scipy的带宽是协方差因子，需要调整
                if self.bandwidth == 'scott':
                    bw_factor = len(x) ** (-1/5)
                elif self.bandwidth == 'silverman':
                    bw_factor = (len(x) * (len(x[0].shape if hasattr(x[0], 'shape') else 1) + 2) / 4) ** (-1/5)
                else:
                    bw_factor = len(x) ** (-1/5)
                kde.set_bandwidth(bw_method=bw_factor)
            else:
                # 直接设置带宽
                kde.set_bandwidth(bw_method=self.bandwidth / x.std())

            density = kde(kde_x)

        else:
            # 对于其他核函数，使用手动实现但向量化
            density = self._vectorized_kde(x, kde_x)

        # 平滑处理
        if self.smooth_density:
            density = gaussian_filter1d(density, sigma=3)

        # 归一化
        density = density / density.max()

        return kde_x, density

    def _vectorized_kde(self, x: np.ndarray, kde_x: np.ndarray) -> np.ndarray:
        """向量化的核密度估计."""
        if self.bandwidth == 'scott':
            bw = len(x) ** (-1/5) * np.std(x)
        elif self.bandwidth == 'silverman':
            bw = (4/3) ** (1/5) * len(x) ** (-1/5) * np.std(x)
        else:
            bw = float(self.bandwidth)

        if bw == 0:
            bw = 1e-6

        # 向量化计算
        # 使用广播机制，避免循环
        x_reshaped = x.reshape(-1, 1)  # (n, 1)
        kde_x_reshaped = kde_x.reshape(1, -1)  # (1, m)

        u = (kde_x_reshaped - x_reshaped) / bw  # (n, m)

        if self.kernel == 'epanechnikov':
            mask = np.abs(u) <= 1
            density = np.where(mask, 0.75 * (1 - u ** 2), 0)
            density = density.sum(axis=0) / (len(x) * bw)
        elif self.kernel == 'tophat':
            mask = np.abs(u) <= 1
            density = np.where(mask, 0.5, 0)
            density = density.sum(axis=0) / (len(x) * bw)
        else:  # 默认高斯
            density = np.exp(-0.5 * u ** 2)
            density = density.sum(axis=0) / (len(x) * bw * np.sqrt(2 * np.pi))

        return density

    def _find_peaks_and_valleys_optimized(
        self,
        kde_x: np.ndarray,
        density: np.ndarray,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """优化的峰谷检测."""
        x_range = x.max() - x.min()

        # 计算最小峰距离（网格点数）
        min_dist = int(len(kde_x) * self.min_peak_distance)
        min_dist = max(min_dist, 10)

        # 寻找峰
        peaks, peak_props = find_peaks(
            density,
            height=self.min_peak_height,
            distance=min_dist,
            prominence=0.02  # 添加显著度要求
        )

        # 寻找谷（在峰之间找）
        valleys = []
        if len(peaks) > 1:
            for i in range(len(peaks) - 1):
                # 在两个峰之间找最小值
                start_idx = peaks[i]
                end_idx = peaks[i + 1]
                valley_region = density[start_idx:end_idx + 1]

                if len(valley_region) > 0:
                    valley_local_idx = np.argmin(valley_region)
                    valleys.append(start_idx + valley_local_idx)

        # 如果峰太少，尝试用更宽松的条件找谷
        if len(peaks) <= 1 and len(valleys) == 0:
            # 降低峰高要求
            peaks_relaxed, _ = find_peaks(density, distance=min_dist // 2)

            if len(peaks_relaxed) > 1:
                for i in range(len(peaks_relaxed) - 1):
                    start_idx = peaks_relaxed[i]
                    end_idx = peaks_relaxed[i + 1]
                    valley_region = density[start_idx:end_idx + 1]
                    if len(valley_region) > 0:
                        valley_local_idx = np.argmin(valley_region)
                        valleys.append(start_idx + valley_local_idx)

        valleys = np.array(valleys, dtype=int)

        return peaks, valleys

    def _select_valleys_as_splits_optimized(
        self,
        kde_x: np.ndarray,
        kde_density: np.ndarray,
        valleys: np.ndarray,
        x: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """优化的谷值选择作为切分点 - 结合目标变量和单调性."""
        min_samples = self._get_min_samples(len(x))

        if len(valleys) == 0:
            # 如果没有谷，使用基于目标变量的等频分箱或纯等频分箱
            if self.use_target and y is not None:
                return self._get_target_based_splits(x, y)
            else:
                n_splits = min(self.max_n_bins - 1, self.min_n_bins - 1)
                if n_splits <= 0:
                    return np.array([])
                quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
                return np.percentile(x, quantiles * 100)

        # 获取谷值位置
        valley_positions = kde_x[valleys]

        # 限制在数据范围内
        valid_valley_mask = (valley_positions > x.min()) & (valley_positions < x.max())
        valley_positions = valley_positions[valid_valley_mask]
        valleys = valleys[valid_valley_mask]

        if len(valley_positions) == 0:
            if self.use_target and y is not None:
                return self._get_target_based_splits(x, y)
            return np.array([])

        # 如果谷值太多，需要选择最优的
        if len(valley_positions) > self.max_n_bins - 1:
            if self.use_target and y is not None:
                valley_positions = self._select_best_valleys_by_iv(valley_positions, x, y)
            else:
                # 选择密度最低的谷（最显著的谷）
                valley_densities = kde_density[valleys]
                sorted_indices = np.argsort(valley_densities)[:self.max_n_bins - 1]
                valley_positions = valley_positions[sorted_indices]

        # 检查单调性和样本数
        valley_positions = self._validate_and_adjust_splits(
            valley_positions, x, y, min_samples
        )

        # 确保满足最小分箱数
        if len(valley_positions) < self.min_n_bins - 1:
            additional = self._get_additional_splits_optimized(x, valley_positions, y)
            valley_positions = np.sort(np.concatenate([valley_positions, additional]))

        return np.sort(valley_positions)

    def _get_target_based_splits(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """基于目标变量的分箱切分点."""
        # 使用决策树的思想找切分点
        n_splits = min(self.max_n_bins - 1, self.min_n_bins - 1)
        if n_splits <= 0:
            return np.array([])

        # 按x排序
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # 使用累积坏样本率找切分点
        cum_bad = np.cumsum(y_sorted)
        total_bad = cum_bad[-1]
        total_good = len(y) - total_bad

        if total_bad == 0 or total_good == 0:
            # 使用等频分箱
            quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
            return np.percentile(x, quantiles * 100)

        # 找最优切分点
        best_splits = []
        for i in range(n_splits):
            best_iv = -np.inf
            best_split = None

            # 候选切分点
            n_candidates = min(50, len(x) // 10)
            candidate_positions = np.linspace(0, len(x) - 1, n_candidates, dtype=int)

            for pos in candidate_positions:
                if pos == 0 or pos == len(x) - 1:
                    continue

                split = x_sorted[pos]

                # 计算IV
                left_mask = x_sorted <= split
                right_mask = ~left_mask

                if left_mask.sum() < self._get_min_samples(len(x)) or \
                   right_mask.sum() < self._get_min_samples(len(x)):
                    continue

                left_bad = y_sorted[left_mask].sum()
                left_good = left_mask.sum() - left_bad
                right_bad = y_sorted[right_mask].sum()
                right_good = right_mask.sum() - right_bad

                iv = self._calculate_bin_iv(left_bad, left_good, right_bad, right_good, total_bad, total_good)

                if iv > best_iv:
                    best_iv = iv
                    best_split = split

            if best_split is not None:
                best_splits.append(best_split)

        return np.array(sorted(best_splits[:n_splits]))

    def _select_best_valleys_by_iv(
        self,
        valley_positions: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """根据IV值选择最优的谷值组合."""
        n_select = min(self.max_n_bins - 1, len(valley_positions))

        if n_select == len(valley_positions):
            return valley_positions

        # 贪心选择：每次选择能最大化IV增益的谷值
        selected = []
        remaining = list(range(len(valley_positions)))

        total_bad = y.sum()
        total_good = len(y) - total_bad

        while len(selected) < n_select and remaining:
            best_gain = -np.inf
            best_idx = None

            for idx in remaining:
                # 尝试添加这个谷值
                test_splits = sorted([valley_positions[i] for i in selected] + [valley_positions[idx]])

                # 计算IV
                bins = np.digitize(x, test_splits)
                iv = self._calculate_total_iv(bins, y)

                if iv > best_gain:
                    best_gain = iv
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return valley_positions[sorted(selected)]

    def _validate_and_adjust_splits(
        self,
        splits: np.ndarray,
        x: np.ndarray,
        y: Optional[np.ndarray],
        min_samples: int
    ) -> np.ndarray:
        """验证并调整切分点，确保满足样本数和单调性要求."""
        if len(splits) == 0:
            return splits

        valid_splits = []

        for i, split in enumerate(splits):
            # 检查样本数
            if i == 0:
                left_count = (x <= split).sum()
            else:
                left_count = ((x > splits[i - 1]) & (x <= split)).sum()

            if i == len(splits) - 1:
                right_count = (x > split).sum()
            else:
                right_count = ((x > split) & (x <= splits[i + 1])).sum()

            if left_count >= min_samples and right_count >= min_samples:
                valid_splits.append(split)

        # 检查单调性
        if self.monotonic and y is not None and len(valid_splits) > 0:
            bins = np.digitize(x, valid_splits)
            bad_rates = []

            for b in range(len(valid_splits) + 1):
                mask = bins == b
                if mask.sum() > 0:
                    bad_rates.append(y[mask].mean())
                else:
                    bad_rates.append(0)

            if not self._check_monotonicity(np.array(bad_rates)):
                # 尝试移除违反单调性的切分点
                valid_splits = self._adjust_for_monotonicity(valid_splits, x, y)

        return np.array(valid_splits)

    def _adjust_for_monotonicity(
        self,
        splits: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """调整切分点以满足单调性."""
        # 简化处理：移除导致单调性违反的切分点
        splits = list(splits)

        for _ in range(len(splits)):
            if len(splits) == 0:
                break

            bins = np.digitize(x, splits)
            bad_rates = []

            for b in range(len(splits) + 1):
                mask = bins == b
                if mask.sum() > 0:
                    bad_rates.append(y[mask].mean())
                else:
                    bad_rates.append(0)

            if self._check_monotonicity(np.array(bad_rates)):
                break

            # 找到违反单调性的位置并移除切分点
            remove_idx = None
            for i in range(len(bad_rates) - 1):
                if self.monotonic == 'ascending' and bad_rates[i] > bad_rates[i + 1]:
                    # 移除这个切分点
                    if i < len(splits):
                        remove_idx = i
                        break
                elif self.monotonic == 'descending' and bad_rates[i] < bad_rates[i + 1]:
                    if i < len(splits):
                        remove_idx = i
                        break

            if remove_idx is not None:
                splits.pop(remove_idx)
            else:
                break  # 无法找到需要移除的切分点，退出循环

        return np.array(splits)

    def _check_monotonicity(self, rates: np.ndarray) -> bool:
        """检查单调性."""
        if len(rates) < 2:
            return True

        if self.monotonic == 'ascending':
            return all(rates[i] <= rates[i + 1] + 1e-6 for i in range(len(rates) - 1))
        elif self.monotonic == 'descending':
            return all(rates[i] >= rates[i + 1] - 1e-6 for i in range(len(rates) - 1))
        elif self.monotonic == 'peak':
            peak_idx = np.argmax(rates)
            left_mono = all(rates[i] <= rates[i + 1] + 1e-6 for i in range(peak_idx))
            right_mono = all(rates[i] >= rates[i + 1] - 1e-6 for i in range(peak_idx, len(rates) - 1))
            return left_mono and right_mono
        elif self.monotonic == 'valley':
            valley_idx = np.argmin(rates)
            left_mono = all(rates[i] >= rates[i + 1] - 1e-6 for i in range(valley_idx))
            right_mono = all(rates[i] <= rates[i + 1] + 1e-6 for i in range(valley_idx, len(rates) - 1))
            return left_mono and right_mono

        return True

    def _calculate_bin_iv(
        self,
        left_bad: int,
        left_good: int,
        right_bad: int,
        right_good: int,
        total_bad: int,
        total_good: int
    ) -> float:
        """计算单个切分点的IV值."""
        epsilon = 1e-10

        left_bad_rate = left_bad / total_bad if total_bad > 0 else epsilon
        left_good_rate = left_good / total_good if total_good > 0 else epsilon
        right_bad_rate = right_bad / total_bad if total_bad > 0 else epsilon
        right_good_rate = right_good / total_good if total_good > 0 else epsilon

        left_woe = np.log((left_good_rate + epsilon) / (left_bad_rate + epsilon))
        right_woe = np.log((right_good_rate + epsilon) / (right_bad_rate + epsilon))

        left_iv = (left_good_rate - left_bad_rate) * left_woe
        right_iv = (right_good_rate - right_bad_rate) * right_woe

        return left_iv + right_iv

    def _calculate_total_iv(self, bins: np.ndarray, y: np.ndarray) -> float:
        """计算总IV值."""
        n_bins = bins.max() + 1
        total_bad = y.sum()
        total_good = len(y) - total_bad

        if total_bad == 0 or total_good == 0:
            return 0.0

        iv = 0.0
        epsilon = 1e-10

        for b in range(n_bins):
            mask = bins == b
            bad = y[mask].sum()
            good = mask.sum() - bad

            bad_rate = bad / total_bad
            good_rate = good / total_good

            bad_rate = max(bad_rate, epsilon)
            good_rate = max(good_rate, epsilon)

            iv += (good_rate - bad_rate) * np.log(good_rate / bad_rate)

        return max(iv, 0.0)

    def _get_additional_splits_optimized(
        self,
        x: np.ndarray,
        existing_splits: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """获取额外的切分点 - 优化版."""
        n_needed = self.min_n_bins - 1 - len(existing_splits)
        if n_needed <= 0:
            return np.array([])

        if y is not None and self.use_target:
            # 使用基于目标变量的方法
            # 在现有切分点分割的区域内找额外的切分点
            boundaries = np.concatenate([[x.min()], existing_splits, [x.max()]])

            additional = []
            for i in range(len(boundaries) - 1):
                if len(additional) >= n_needed:
                    break

                # 在这个区域内找最优切分点
                region_mask = (x > boundaries[i]) & (x <= boundaries[i + 1])
                x_region = x[region_mask]
                y_region = y[region_mask]

                if len(x_region) > 0:
                    # 简单地在区域中点添加切分点
                    mid_point = (boundaries[i] + boundaries[i + 1]) / 2
                    additional.append(mid_point)

            return np.array(additional[:n_needed])
        else:
            # 原有的等分方法
            x_min, x_max = x.min(), x.max()
            existing_with_bounds = np.concatenate([[x_min], existing_splits, [x_max]])

            additional = []
            for i in range(len(existing_with_bounds) - 1):
                if len(additional) >= n_needed:
                    break
                mid = (existing_with_bounds[i] + existing_with_bounds[i + 1]) / 2
                additional.append(mid)

            return np.array(additional[:n_needed])

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行分箱."""
        cat_stats = pd.DataFrame({
            'category': x,
            'target': y
        }).groupby('category')['target'].agg(['mean', 'count'])

        min_samples = self._get_min_samples(len(x))
        cat_stats = cat_stats[cat_stats['count'] >= min_samples]
        cat_stats = cat_stats.sort_values('mean')

        if len(cat_stats) > self.max_n_bins:
            categories = self._merge_categories(cat_stats)
        else:
            categories = cat_stats.index.tolist()

        return categories

    def _merge_categories(self, cat_stats: pd.DataFrame) -> List:
        """合并类别."""
        categories = cat_stats.index.tolist()

        while len(categories) > self.max_n_bins:
            bad_rates = cat_stats['mean'].values
            min_diff = float('inf')
            merge_idx = 0

            for i in range(len(bad_rates) - 1):
                diff = abs(bad_rates[i] - bad_rates[i + 1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            cat1 = categories[merge_idx]
            cat2 = categories[merge_idx + 1]
            merged_cat = f"{cat1},{cat2}"

            categories.pop(merge_idx + 1)
            categories[merge_idx] = merged_cat

            merged_count = cat_stats.iloc[merge_idx]['count'] + cat_stats.iloc[merge_idx + 1]['count']
            merged_bad = (cat_stats.iloc[merge_idx]['mean'] * cat_stats.iloc[merge_idx]['count'] +
                         cat_stats.iloc[merge_idx + 1]['mean'] * cat_stats.iloc[merge_idx + 1]['count'])
            merged_rate = merged_bad / merged_count

            cat_stats = cat_stats.drop([cat1, cat2])
            cat_stats.loc[merged_cat] = {'mean': merged_rate, 'count': merged_count}
            cat_stats = cat_stats.sort_values('mean')

        return categories

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数."""
        if self.min_bin_size < 1:
            return int(n_total * self.min_bin_size)
        return int(self.min_bin_size)

    def _apply_bins(
        self,
        x: pd.Series,
        splits: Union[np.ndarray, List],
        feature_type: str
    ) -> np.ndarray:
        """应用分箱."""
        if feature_type == 'categorical':
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
                if ',' in str(cat):
                    cats = str(cat).split(',')
                    for c in cats:
                        bins[x == c] = i
                else:
                    bins[x == cat] = i
            bins[x.isna()] = -1
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2
            return bins
        else:
            bins = np.zeros(len(x), dtype=int)
            if self.missing_separate:
                bins[x.isna()] = -1
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2
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
        >>> binner = KernelDensityBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                raise KeyError(f"特征 '{feature}' 未在训练数据中找到")

            splits = self.splits_[feature]
            feature_type = self.feature_types_[feature]
            bins = self._apply_bins(X[feature], splits, feature_type)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels(splits, bins)
                result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special') for b in bins]
            elif metric == 'woe':
                bin_table = self.bin_tables_[feature]
                woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                woe_map[-1] = 0
                woe_map[-2] = 0
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result
