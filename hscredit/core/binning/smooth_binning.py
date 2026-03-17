"""平滑/正则化分箱.

使用平滑技术和正则化防止过拟合，提高泛化能力。
适用于样本量较小或噪声较大的场景。

优化版本：
- 改进初始分箱策略，使用更合理的预分箱数量
- 优化合并逻辑，基于IV值增益和单调性
- 向量化计算提升效率
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .base import BaseBinning


class SmoothBinning(BaseBinning):
    """平滑/正则化分箱.

    通过平滑技术和正则化约束防止分箱过拟合，提高模型泛化能力。
    支持多种平滑方法：Laplace平滑、贝叶斯平滑、正则化优化等。

    优化版本特点：
    - 使用预分箱策略生成更合理的初始切分点
    - 基于IV值增益和单调性进行合并优化
    - 向量化计算提升效率

    :param method: 平滑方法，默认为'laplace'，可选'laplace', 'bayesian', 'iv_optimized'
    :param smoothing_param: 平滑参数，默认为1.0
        - laplace: 伪计数
        - bayesian: 先验强度
        - iv_optimized: IV增益阈值
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param prior_bad_rate: 先验坏样本率，默认为None（使用全局坏样本率）
    :param monotonic: 单调性约束，默认为None，可选'ascending', 'descending', 'peak', 'valley'或None
    :param n_prebins: 预分箱数量，默认为20
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **示例**

    >>> from hscredit.core.binning import SmoothBinning
    >>> binner = SmoothBinning(method='laplace', smoothing_param=1.0, max_n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """

    def __init__(
        self,
        method: str = 'laplace',
        smoothing_param: float = 1.0,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        prior_bad_rate: Optional[float] = None,
        monotonic: Optional[str] = None,
        n_prebins: int = 20,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ):
        super().__init__(
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            random_state=random_state,
            verbose=verbose,
        )
        self.method = method
        self.smoothing_param = smoothing_param
        self.prior_bad_rate = prior_bad_rate
        self.n_prebins = n_prebins

        if method not in ['laplace', 'bayesian', 'iv_optimized']:
            raise ValueError("method必须是'laplace', 'bayesian'或'iv_optimized'")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'SmoothBinning':
        """拟合平滑分箱."""
        X, y = self._check_input(X, y)

        # 计算先验坏样本率
        if self.prior_bad_rate is None:
            self.prior_bad_rate_ = y.mean()
        else:
            self.prior_bad_rate_ = self.prior_bad_rate

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
        """对数值型特征进行平滑分箱."""
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask].values
        y_valid = y[mask].values

        if len(x_valid) == 0:
            return np.array([])

        # 使用预分箱策略生成初始切分点
        initial_splits = self._get_initial_splits(x_valid)

        if len(initial_splits) == 0:
            return np.array([])

        # 根据平滑方法优化分箱
        splits = self._smooth_split_optimization(x_valid, y_valid, initial_splits)

        return splits

    def _get_initial_splits(self, x: np.ndarray) -> np.ndarray:
        """获取初始切分点 - 使用等频预分箱."""
        # 预分箱数量：确保有足够的候选切分点
        n_prebins = min(self.n_prebins, max(self.max_n_bins * 3, 10))
        n_prebins = min(n_prebins, len(x) // 5)  # 每箱至少5个样本
        n_prebins = max(n_prebins, self.min_n_bins)

        if n_prebins <= 1:
            return np.array([])

        # 等频分箱
        quantiles = np.linspace(0, 1, n_prebins + 1)[1:-1]
        splits = np.percentile(x, quantiles * 100)
        return np.unique(splits)

    def _smooth_split_optimization(
        self,
        x: np.ndarray,
        y: np.ndarray,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """使用平滑优化切分点 - 基于IV增益的贪心合并."""
        splits = list(initial_splits)
        min_samples = self._get_min_samples(len(x))
        prior = self.prior_bad_rate_

        max_iter = 100
        for iteration in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算当前分箱的统计信息
            bins = np.digitize(x, splits)
            bin_stats = self._compute_smoothed_stats_vectorized(bins, y, prior)

            # 检查是否满足单调性
            if self.monotonic and len(splits) >= self.min_n_bins - 1:
                if self._check_monotonicity(bin_stats['smoothed_rate'].values):
                    # 满足单调性，检查是否可以停止
                    if len(splits) <= self.max_n_bins - 1:
                        break

            # 找到最佳合并候选
            merge_idx = self._find_best_merge_candidate(bins, y, bin_stats, min_samples)

            if merge_idx is None:
                break

            # 执行合并
            splits.pop(merge_idx)

        return np.array(splits)

    def _compute_smoothed_stats_vectorized(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        prior: float
    ) -> pd.DataFrame:
        """向量化计算平滑后的统计信息."""
        n_bins = bins.max() + 1
        stats_data = []

        for b in range(n_bins):
            mask = bins == b
            bad = y[mask].sum()
            count = mask.sum()
            good = count - bad
            stats_data.append({'bin': b, 'bad': bad, 'count': count, 'good': good})

        stats = pd.DataFrame(stats_data)

        if self.method == 'laplace':
            # Laplace平滑: (bad + α) / (count + 2α)
            alpha = self.smoothing_param
            stats['smoothed_rate'] = (stats['bad'] + alpha) / (stats['count'] + 2 * alpha)
        elif self.method == 'bayesian':
            # 贝叶斯平滑: (bad + α*prior) / (count + α)
            alpha = self.smoothing_param
            stats['smoothed_rate'] = (stats['bad'] + alpha * prior) / (stats['count'] + alpha)
        else:  # iv_optimized
            # 使用更激进的平滑，减少噪声影响
            alpha = self.smoothing_param
            stats['smoothed_rate'] = (stats['bad'] + alpha * prior * stats['count']) / (stats['count'] + alpha)

        return stats

    def _find_best_merge_candidate(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        bin_stats: pd.DataFrame,
        min_samples: int
    ) -> Optional[int]:
        """找到最佳的合并候选 - 基于IV损失最小."""
        n_bins = len(bin_stats)
        if n_bins <= self.min_n_bins:
            return None

        counts = bin_stats['count'].values
        rates = bin_stats['smoothed_rate'].values

        # 优先处理样本数不足的箱
        for i in range(n_bins - 1):
            if counts[i] < min_samples or counts[i + 1] < min_samples:
                return i

        # 计算合并每对相邻箱的IV损失
        min_iv_loss = float('inf')
        merge_idx = None

        for i in range(n_bins - 1):
            # 计算合并前后的IV差异
            iv_loss = self._calculate_merge_iv_loss(bins, y, i)

            # 检查单调性约束
            if self.monotonic:
                # 模拟合并后的坏样本率序列
                merged_rate = (bin_stats.iloc[i]['bad'] + bin_stats.iloc[i + 1]['bad']) / \
                              (bin_stats.iloc[i]['count'] + bin_stats.iloc[i + 1]['count'])
                new_rates = np.concatenate([rates[:i], [merged_rate], rates[i + 2:]])

                # 如果合并后违反单调性，增加惩罚
                if not self._check_monotonicity(new_rates):
                    iv_loss += 1.0  # 惩罚项

            if iv_loss < min_iv_loss:
                min_iv_loss = iv_loss
                merge_idx = i

        # 如果最小IV损失太大，说明已经无法进一步优化
        if self.method == 'iv_optimized' and min_iv_loss > self.smoothing_param * 0.01:
            if len(bin_stats) <= self.max_n_bins:
                return None

        return merge_idx

    def _calculate_merge_iv_loss(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        merge_idx: int
    ) -> float:
        """计算合并两个箱的IV损失."""
        # 计算当前IV
        n_bins = bins.max() + 1

        # 计算合并前的IV
        iv_before = self._calculate_total_iv(bins, y)

        # 模拟合并
        bins_merged = bins.copy()
        bins_merged[bins == merge_idx + 1] = merge_idx
        bins_merged[bins > merge_idx + 1] -= 1

        # 计算合并后的IV
        iv_after = self._calculate_total_iv(bins_merged, y)

        return iv_before - iv_after

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

            # 避免log(0)
            bad_rate = max(bad_rate, epsilon)
            good_rate = max(good_rate, epsilon)

            iv += (good_rate - bad_rate) * np.log(good_rate / bad_rate)

        return max(iv, 0.0)

    def _check_monotonicity(self, rates: np.ndarray) -> bool:
        """检查单调性."""
        if len(rates) < 2:
            return True

        if self.monotonic == 'ascending':
            return all(rates[i] <= rates[i + 1] + 1e-6 for i in range(len(rates) - 1))
        elif self.monotonic == 'descending':
            return all(rates[i] >= rates[i + 1] - 1e-6 for i in range(len(rates) - 1))
        elif self.monotonic == 'peak':
            # 先升后降
            peak_idx = np.argmax(rates)
            left_mono = all(rates[i] <= rates[i + 1] + 1e-6 for i in range(peak_idx))
            right_mono = all(rates[i] >= rates[i + 1] - 1e-6 for i in range(peak_idx, len(rates) - 1))
            return left_mono and right_mono
        elif self.monotonic == 'valley':
            # 先降后升
            valley_idx = np.argmin(rates)
            left_mono = all(rates[i] >= rates[i + 1] - 1e-6 for i in range(valley_idx))
            right_mono = all(rates[i] <= rates[i + 1] + 1e-6 for i in range(valley_idx, len(rates) - 1))
            return left_mono and right_mono

        return True

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行平滑分箱."""
        # 计算每个类别的统计信息
        temp_df = pd.DataFrame({'category': x.values, 'target': y.values})
        cat_stats = temp_df.groupby('category')['target'].agg(['sum', 'count'])
        cat_stats.columns = ['bad', 'count']
        cat_stats['good'] = cat_stats['count'] - cat_stats['bad']

        # 应用平滑
        prior = self.prior_bad_rate_

        if self.method == 'laplace':
            alpha = self.smoothing_param
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + alpha) / (cat_stats['count'] + 2 * alpha)
        elif self.method == 'bayesian':
            alpha = self.smoothing_param
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + alpha * prior) / (cat_stats['count'] + alpha)
        else:  # iv_optimized
            alpha = self.smoothing_param
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + alpha * prior * cat_stats['count']) / (cat_stats['count'] + alpha)

        # 按平滑后的坏样本率排序
        cat_stats = cat_stats.sort_values('smoothed_rate')

        # 合并类别
        categories = self._merge_categories_optimized(cat_stats, y)

        return categories

    def _merge_categories_optimized(
        self,
        cat_stats: pd.DataFrame,
        y: pd.Series
    ) -> List:
        """优化的类别合并 - 基于IV增益."""
        categories = cat_stats.index.tolist()
        min_samples = self._get_min_samples(len(y))

        # 合并样本数不足的类别到相邻类别
        i = 0
        while i < len(categories):
            if cat_stats.iloc[i]['count'] < min_samples:
                # 找到最近的类别进行合并
                if i > 0:
                    merge_target = i - 1
                elif i < len(categories) - 1:
                    merge_target = i + 1
                else:
                    i += 1
                    continue

                # 执行合并
                cat1 = categories[i]
                cat2 = categories[merge_target]
                merged_cat = f"{cat1},{cat2}"

                # 更新统计信息
                idx1 = i if i < merge_target else merge_target
                idx2 = merge_target if i < merge_target else i

                merged_bad = cat_stats.iloc[idx1]['bad'] + cat_stats.iloc[idx2]['bad']
                merged_count = cat_stats.iloc[idx1]['count'] + cat_stats.iloc[idx2]['count']

                alpha = self.smoothing_param
                prior = self.prior_bad_rate_
                if self.method == 'laplace':
                    merged_rate = (merged_bad + alpha) / (merged_count + 2 * alpha)
                elif self.method == 'bayesian':
                    merged_rate = (merged_bad + alpha * prior) / (merged_count + alpha)
                else:
                    merged_rate = (merged_bad + alpha * prior * merged_count) / (merged_count + alpha)

                # 更新DataFrame
                cat_stats = cat_stats.drop([cat1, cat2])
                cat_stats.loc[merged_cat] = {
                    'bad': merged_bad,
                    'count': merged_count,
                    'good': merged_count - merged_bad,
                    'smoothed_rate': merged_rate
                }
                cat_stats = cat_stats.sort_values('smoothed_rate')
                categories = cat_stats.index.tolist()
            else:
                i += 1

        # 如果类别数超过最大分箱数，继续合并
        while len(categories) > self.max_n_bins:
            rates = cat_stats['smoothed_rate'].values
            counts = cat_stats['count'].values

            # 找到坏样本率最接近的相邻类别
            min_diff = float('inf')
            merge_idx = 0

            for i in range(len(rates) - 1):
                diff = abs(rates[i] - rates[i + 1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            # 合并
            cat1 = categories[merge_idx]
            cat2 = categories[merge_idx + 1]
            merged_cat = f"{cat1},{cat2}"

            # 更新统计信息
            merged_bad = cat_stats.iloc[merge_idx]['bad'] + cat_stats.iloc[merge_idx + 1]['bad']
            merged_count = cat_stats.iloc[merge_idx]['count'] + cat_stats.iloc[merge_idx + 1]['count']

            alpha = self.smoothing_param
            prior = self.prior_bad_rate_
            if self.method == 'laplace':
                merged_rate = (merged_bad + alpha) / (merged_count + 2 * alpha)
            elif self.method == 'bayesian':
                merged_rate = (merged_bad + alpha * prior) / (merged_count + alpha)
            else:
                merged_rate = (merged_bad + alpha * prior * merged_count) / (merged_count + alpha)

            cat_stats = cat_stats.drop([cat1, cat2])
            cat_stats.loc[merged_cat] = {
                'bad': merged_bad,
                'count': merged_count,
                'good': merged_count - merged_bad,
                'smoothed_rate': merged_rate
            }
            cat_stats = cat_stats.sort_values('smoothed_rate')
            categories = cat_stats.index.tolist()

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
        >>> binner = SmoothBinning()
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
