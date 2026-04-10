"""核密度分箱.

基于核密度估计（KDE）识别数据分布的模态和谷值，进行自适应分箱。
适用于多峰分布和需要反映数据自然结构的场景。

优化版本V3（针对平滑单峰分布）：
- 当峰谷检测失败时，使用基于IV的备选切分策略
- 改进带宽选择，支持更灵活的参数
- 增加平滑分布检测和自动切换机制
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d, median_filter
from .base import BaseBinning


class KernelDensityBinning(BaseBinning):
    """核密度分箱.

    使用核密度估计识别数据分布的局部极大值（峰）和极小值（谷），
    以谷值作为分箱边界，使分箱反映数据的自然分布结构。

    优化版本V3特点（针对平滑分布）：
    - 当KDE峰谷检测失败时，自动切换到基于IV的切分策略
    - 支持检测数据分布的平滑程度
    - 改进的带宽自适应

    :param kernel: 核函数类型，默认为'gaussian'
    :param bandwidth: 带宽，默认为'isj'，可选'isj', 'scott', 'silverman', 'normal_reference'或具体数值
    :param min_peak_height: 最小峰高（相对于最大密度），默认为0.05
    :param min_peak_distance: 峰之间的最小距离（相对于数据范围），默认为0.05
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param monotonic: 单调性约束，默认为None
    :param use_target: 是否结合目标变量优化切分点，默认为True
    :param n_grid_points: 核密度估计的网格点数，默认为1000
    :param smooth_density: 是否对密度曲线进行平滑处理，默认为True
    :param iv_weight: IV值在切分点选择中的权重，默认为0.7
    :param fallback_to_iv: 当KDE失败时是否回退到IV策略，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False
    """

    def __init__(
        self,
        target: str = 'target',
        kernel: str = 'gaussian',
        bandwidth: Union[str, float] = 'isj',
        min_peak_height: float = 0.05,
        min_peak_distance: float = 0.05,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        monotonic: Optional[str] = None,
        use_target: bool = True,
        n_grid_points: int = 1000,
        smooth_density: bool = True,
        iv_weight: float = 0.7,
        fallback_to_iv: bool = True,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ):
        super().__init__(
            target=target,
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
        self.iv_weight = iv_weight
        self.fallback_to_iv = fallback_to_iv

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
        kde_x, kde_density = self._compute_kde_v3(x_valid)

        # 寻找峰和谷
        peaks, valleys = self._find_peaks_and_valleys_v3(kde_x, kde_density, x_valid)

        # 检查是否是平滑单峰分布
        is_smooth_single_peak = len(peaks) <= 1 and len(valleys) == 0

        if is_smooth_single_peak and self.fallback_to_iv and y_valid is not None:
            # 对于平滑单峰分布，使用基于IV的切分策略
            if self.verbose:
                print(f"  检测到平滑单峰分布，切换到IV策略")
            splits = self._get_iv_based_splits(x_valid, y_valid)
        else:
            # 选择作为切分点的谷值
            splits = self._select_valleys_as_splits_v3(kde_x, kde_density, valleys, x_valid, y_valid)

        return splits

    def _compute_kde_v3(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """优化的核密度估计 - V3版本."""
        x_min, x_max = x.min(), x.max()
        
        if x_max == x_min:
            return np.array([x_min]), np.array([1.0])
        
        # 扩展范围
        pad = (x_max - x_min) * 0.15
        kde_x = np.linspace(x_min - pad, x_max + pad, self.n_grid_points)

        # 自动选择带宽
        bw = self._select_bandwidth_v3(x)

        # 使用scipy.stats.gaussian_kde
        if self.kernel == 'gaussian':
            try:
                kde = stats.gaussian_kde(x, bw_method=bw if isinstance(bw, str) else bw / x.std())
                density = kde(kde_x)
            except Exception:
                density = self._manual_kde(x, kde_x, bw)
        else:
            density = self._manual_kde(x, kde_x, bw)

        # 平滑处理
        if self.smooth_density:
            density = median_filter(density, size=5)
            density = gaussian_filter1d(density, sigma=2)

        if density.max() > 0:
            density = density / density.max()

        return kde_x, density

    def _select_bandwidth_v3(self, x: np.ndarray) -> float:
        """自适应带宽选择 - V3版本."""
        n = len(x)
        std = np.std(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        
        dispersion = min(std, iqr / 1.34) if iqr > 0 else std
        if dispersion == 0:
            dispersion = 1.0

        if self.bandwidth == 'scott':
            return 1.06 * dispersion * n ** (-1/5)
        elif self.bandwidth == 'silverman':
            return 0.9 * dispersion * n ** (-1/5)
        elif self.bandwidth == 'isj':
            return self._isj_bandwidth_v3(x, dispersion)
        elif self.bandwidth == 'normal_reference':
            return 1.059 * dispersion * n ** (-1/5)
        else:
            return float(self.bandwidth)

    def _isj_bandwidth_v3(self, x: np.ndarray, dispersion: float) -> float:
        """Improved Sheather-Jones带宽选择."""
        n = len(x)
        h0 = 1.06 * dispersion * n ** (-1/5)
        
        # 基于数据偏度调整
        skewness = stats.skew(x)
        kurtosis = stats.kurtosis(x)
        
        adjustment = 1 + 0.1 * abs(skewness) + 0.05 * max(0, kurtosis)
        
        return h0 * adjustment

    def _manual_kde(self, x: np.ndarray, kde_x: np.ndarray, bw: float) -> np.ndarray:
        """手动实现核密度估计."""
        n = len(x)
        
        if self.kernel == 'epanechnikov':
            u = (kde_x[:, None] - x[None, :]) / bw
            mask = np.abs(u) <= 1
            kernels = np.where(mask, 0.75 * (1 - u ** 2), 0)
            density = kernels.sum(axis=1) / (n * bw)
        elif self.kernel == 'tophat':
            u = (kde_x[:, None] - x[None, :]) / bw
            mask = np.abs(u) <= 1
            kernels = np.where(mask, 0.5, 0)
            density = kernels.sum(axis=1) / (n * bw)
        else:
            u = (kde_x[:, None] - x[None, :]) / bw
            kernels = np.exp(-0.5 * u ** 2)
            density = kernels.sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
        
        return density

    def _find_peaks_and_valleys_v3(
        self,
        kde_x: np.ndarray,
        density: np.ndarray,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """优化的峰谷检测 - V3版本."""
        x_range = x.max() - x.min()
        
        min_dist = int(len(kde_x) * self.min_peak_distance)
        min_dist = max(min_dist, 5)

        # 寻找峰
        peaks, _ = find_peaks(
            density,
            height=self.min_peak_height,
            distance=min_dist,
            prominence=0.01,  # 降低显著度要求
            width=1
        )

        # 寻找谷
        valleys = self._find_valleys_comprehensive_v3(density, peaks, min_dist)

        return peaks, valleys

    def _find_valleys_comprehensive_v3(
        self,
        density: np.ndarray,
        peaks: np.ndarray,
        min_dist: int
    ) -> np.ndarray:
        """综合方法寻找谷值 - V3版本."""
        valleys = []
        
        # 方法1：峰之间找最小值
        if len(peaks) > 1:
            for i in range(len(peaks) - 1):
                start_idx = peaks[i]
                end_idx = peaks[i + 1]
                
                valley_region = density[start_idx:end_idx + 1]
                if len(valley_region) > 0:
                    valley_local_idx = np.argmin(valley_region)
                    valley_idx = start_idx + valley_local_idx
                    
                    if 0 < valley_idx < len(density) - 1:
                        valleys.append(valley_idx)

        # 方法2：使用find_peaks的逆
        inverted_density = -density
        valleys_from_inverted, _ = find_peaks(
            inverted_density,
            distance=min_dist,
            prominence=0.005  # 降低要求
        )
        valleys.extend(valleys_from_inverted)

        # 方法3：局部极小值
        local_minima = argrelextrema(density, np.less, order=max(3, min_dist // 3))[0]
        valleys.extend(local_minima)

        valleys = np.unique(np.array(valleys, dtype=int))
        valleys = np.sort(valleys)

        return valleys

    def _get_iv_based_splits(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """基于IV值选择切分点（针对平滑分布的备选策略）- V3改进版."""
        n = len(x)
        # 修复：目标是max_n_bins个分箱，需要max_n_bins-1个切分点
        n_splits = self.max_n_bins - 1
        
        if n_splits <= 0:
            return np.array([])

        # 排序
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        total_bad = y.sum()
        total_good = len(y) - total_bad

        if total_bad == 0 or total_good == 0:
            quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
            return np.percentile(x, quantiles * 100)

        # 首先在所有可能位置找到top候选点
        min_samples = self._get_min_samples(n)
        all_candidates = []
        
        # 在数据范围内等间距采样候选点
        n_candidates_search = min(200, n // 20)  # 增加搜索点
        candidate_positions = np.linspace(min_samples, n - min_samples, n_candidates_search, dtype=int)
        
        for idx in candidate_positions:
            left_y = y_sorted[:idx]
            right_y = y_sorted[idx:]
            
            if len(left_y) < min_samples or len(right_y) < min_samples:
                continue
                
            iv = self._calculate_iv_gain(left_y, right_y, total_bad, total_good)
            split = (x_sorted[idx-1] + x_sorted[idx]) / 2
            all_candidates.append((split, idx, iv))
        
        # 按IV排序，选择top候选
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = all_candidates[:min(50, len(all_candidates))]
        
        if len(top_candidates) == 0:
            return np.array([])
        
        # 贪心选择：每次选择能最大化总IV的切分点
        selected = []
        
        for _ in range(n_splits):
            best_total_iv = -np.inf
            best_split = None
            
            for split, idx, single_iv in top_candidates:
                if split in selected:
                    continue
                    
                # 测试添加这个切分点后的总IV
                test_splits = sorted(selected + [split])
                bins = np.digitize(x_sorted, test_splits)
                total_iv = self._calculate_total_iv(bins, y_sorted, total_bad, total_good)
                
                if total_iv > best_total_iv:
                    best_total_iv = total_iv
                    best_split = split
            
            if best_split is not None:
                selected.append(best_split)
            else:
                break
        
        return np.array(sorted(selected))

    def _calculate_iv_gain(self, y_left: np.ndarray, y_right: np.ndarray,
                          total_bad: int, total_good: int) -> float:
        """计算IV增益."""
        epsilon = 1e-10
        
        left_bad, left_n = y_left.sum(), len(y_left)
        right_bad, right_n = y_right.sum(), len(y_right)
        left_good = left_n - left_bad
        right_good = right_n - right_bad
        
        if left_bad == 0 or left_good == 0 or right_bad == 0 or right_good == 0:
            return 0.0
        
        left_woe = np.log((left_good/total_good + epsilon) / (left_bad/total_bad + epsilon))
        right_woe = np.log((right_good/total_good + epsilon) / (right_bad/total_bad + epsilon))
        
        left_iv = (left_good/total_good - left_bad/total_bad) * left_woe
        right_iv = (right_good/total_good - right_bad/total_bad) * right_woe
        
        return left_iv + right_iv

    def _select_valleys_as_splits_v3(
        self,
        kde_x: np.ndarray,
        kde_density: np.ndarray,
        valleys: np.ndarray,
        x: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """优化的谷值选择作为切分点 - V3改进版."""
        min_samples = self._get_min_samples(len(x))
        x_min, x_max = x.min(), x.max()

        if len(valleys) == 0:
            if self.use_target and y is not None:
                return self._get_iv_based_splits(x, y)
            else:
                n_splits = min(self.max_n_bins - 1, self.min_n_bins - 1)
                if n_splits <= 0:
                    return np.array([])
                quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
                return np.percentile(x, quantiles * 100)

        valley_positions = kde_x[valleys]
        valley_densities = kde_density[valleys]

        # 过滤条件：
        # 1. 必须在数据范围内
        # 2. 密度必须足够低（是真正的谷）
        # 3. 不能太靠近边界（避免产生太小的边箱）
        data_range = x_max - x_min
        valid_valley_mask = (
            (valley_positions > x_min + 0.1 * data_range) &  # 不能太靠近左边界
            (valley_positions < x_max - 0.1 * data_range) &  # 不能太靠近右边界
            (valley_densities < kde_density.max() * 0.3)      # 密度要足够低
        )
        
        valley_positions = valley_positions[valid_valley_mask]
        valleys = valleys[valid_valley_mask]

        # 如果过滤后没有足够的有效谷，使用IV策略
        if len(valley_positions) < self.min_n_bins - 1:
            if self.use_target and y is not None:
                if self.verbose:
                    print(f"  有效谷不足({len(valley_positions)}个)，切换到IV策略")
                return self._get_iv_based_splits(x, y)
            else:
                n_splits = min(self.max_n_bins - 1, self.min_n_bins - 1)
                quantiles = np.linspace(0, 1, n_splits + 2)[1:-1]
                return np.percentile(x, quantiles * 100)

        if len(valley_positions) > self.max_n_bins - 1:
            if self.use_target and y is not None:
                valley_positions = self._select_best_valleys_by_iv_v3(valley_positions, x, y)
            else:
                valley_densities = kde_density[valleys]
                sorted_indices = np.argsort(valley_densities)[:self.max_n_bins - 1]
                valley_positions = valley_positions[sorted_indices]

        valley_positions = self._validate_and_adjust_splits_v3(valley_positions, x, y, min_samples)

        # 再次检查分箱数
        if len(valley_positions) < self.min_n_bins - 1 and self.use_target and y is not None:
            if self.verbose:
                print(f"  验证后谷不足，使用IV策略补充")
            return self._get_iv_based_splits(x, y)

        return np.sort(valley_positions)

    def _select_best_valleys_by_iv_v3(
        self,
        valley_positions: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """根据IV值选择最优的谷值组合 - V3版本."""
        n_select = min(self.max_n_bins - 1, len(valley_positions))

        if n_select == len(valley_positions):
            return valley_positions

        selected = []
        remaining = list(range(len(valley_positions)))

        total_bad = y.sum()
        total_good = len(y) - total_bad

        while len(selected) < n_select and remaining:
            best_gain = -np.inf
            best_idx = None

            for idx in remaining:
                test_splits = sorted([valley_positions[i] for i in selected] + [valley_positions[idx]])
                bins = np.digitize(x, test_splits)
                iv = self._calculate_total_iv(bins, y, total_bad, total_good)

                if iv > best_gain:
                    best_gain = iv
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return valley_positions[sorted(selected)]

    def _calculate_total_iv(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        total_bad: Optional[int] = None,
        total_good: Optional[int] = None
    ) -> float:
        """计算总IV值."""
        if total_bad is None:
            total_bad = y.sum()
        if total_good is None:
            total_good = len(y) - total_bad

        if total_bad == 0 or total_good == 0:
            return 0.0

        n_bins = bins.max() + 1
        iv = 0.0
        epsilon = 1e-10

        for b in range(n_bins):
            mask = bins == b
            if mask.sum() == 0:
                continue
            bad = y[mask].sum()
            good = mask.sum() - bad

            bad_rate = bad / total_bad
            good_rate = good / total_good

            bad_rate = max(bad_rate, epsilon)
            good_rate = max(good_rate, epsilon)

            iv += (good_rate - bad_rate) * np.log(good_rate / bad_rate)

        return max(iv, 0.0)

    def _validate_and_adjust_splits_v3(
        self,
        splits: np.ndarray,
        x: np.ndarray,
        y: Optional[np.ndarray],
        min_samples: int
    ) -> np.ndarray:
        """验证并调整切分点 - V3版本."""
        if len(splits) == 0:
            return splits

        valid_splits = []

        for i, split in enumerate(splits):
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

        if self.monotonic and y is not None and len(valid_splits) > 0:
            valid_splits = self._adjust_for_monotonicity_v3(valid_splits, x, y)

        return np.array(valid_splits)

    def _adjust_for_monotonicity_v3(
        self,
        splits: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """调整切分点以满足单调性 - V3版本."""
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

            remove_idx = None
            for i in range(len(bad_rates) - 1):
                if self.monotonic == 'ascending' and bad_rates[i] > bad_rates[i + 1] + 0.001:
                    remove_idx = i
                    break
                elif self.monotonic == 'descending' and bad_rates[i] < bad_rates[i + 1] - 0.001:
                    remove_idx = i
                    break

            if remove_idx is not None and remove_idx < len(splits):
                splits.pop(remove_idx)
            else:
                break

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

    def _get_additional_splits_v3(
        self,
        x: np.ndarray,
        existing_splits: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """获取额外的切分点 - V3版本."""
        n_needed = self.min_n_bins - 1 - len(existing_splits)
        if n_needed <= 0:
            return np.array([])

        boundaries = np.concatenate([[x.min()], existing_splits, [x.max()]])
        additional = []

        interval_sizes = np.diff(boundaries)
        sorted_intervals = np.argsort(interval_sizes)[::-1]

        for i in sorted_intervals:
            if len(additional) >= n_needed:
                break
            
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            
            left_count = ((x > boundaries[i]) & (x <= mid)).sum() if i > 0 else (x <= mid).sum()
            right_count = (x > mid).sum() if i == len(boundaries) - 2 else ((x > mid) & (x <= boundaries[i + 1])).sum()
            
            min_samples = self._get_min_samples(len(x))
            if left_count >= min_samples and right_count >= min_samples:
                additional.append(mid)

        return np.array(additional)

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数."""
        if self.min_bin_size < 1:
            return int(n_total * self.min_bin_size)
        return int(self.min_bin_size)

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
        """应用分箱转换."""
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
