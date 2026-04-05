"""平滑/正则化分箱.

使用平滑技术和正则化防止过拟合，提高泛化能力。
适用于样本量较小或噪声较大的场景。

优化版本V3（针对平滑分布）：
- 改进预分箱策略：使用混合策略（等频+决策树预分箱）
- 智能合并逻辑：根据IV变化和样本分布决定是否合并
- 支持渐进式平滑：根据样本量自适应调整平滑强度
- 新增保守合并模式，避免过度合并
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from .base import BaseBinning


class SmoothBinning(BaseBinning):
    """平滑/正则化分箱.

    通过平滑技术和正则化约束防止分箱过拟合，提高模型泛化能力。
    支持多种平滑方法：Laplace平滑、贝叶斯平滑、Beta平滑等。

    优化版本V3特点（针对平滑分布）：
    - 改进合并逻辑，避免过度合并
    - 支持IV变化率检测，保留有价值的切分点
    - 自适应平滑强度

    :param method: 平滑方法，默认为'adaptive'，可选'laplace', 'bayesian', 'beta', 'adaptive'
    :param smoothing_param: 平滑参数，默认为0.5（降低默认值）
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param prior_bad_rate: 先验坏样本率，默认为None（使用全局坏样本率）
    :param monotonic: 单调性约束，默认为None，可选'ascending', 'descending', 'peak', 'valley'或None
    :param n_prebins: 预分箱数量，默认为100（提高预分箱数以获得更多候选点）
    :param merge_criterion: 合并准则，默认为'iv_chi2'，可选'iv', 'chi2', 'iv_chi2'
    :param chi2_threshold: 卡方检验阈值，默认为3.84（对应p=0.05）
    :param min_iv_improvement: 最小IV改进阈值，默认为0.001
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **示例**

    >>> from hscredit.core.binning import SmoothBinning
    >>> binner = SmoothBinning(method='adaptive', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """

    def __init__(
        self,
        target: str = 'target',
        method: str = 'adaptive',
        smoothing_param: float = 0.5,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        prior_bad_rate: Optional[float] = None,
        monotonic: Optional[str] = None,
        n_prebins: int = 100,
        merge_criterion: str = 'iv_chi2',
        chi2_threshold: float = 3.84,
        min_iv_improvement: float = 0.001,
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
        self.merge_criterion = merge_criterion
        self.chi2_threshold = chi2_threshold
        self.min_iv_improvement = min_iv_improvement

        if method not in ['laplace', 'bayesian', 'beta', 'adaptive']:
            raise ValueError("method必须是'laplace', 'bayesian', 'beta'或'adaptive'")

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

        # 计算自适应平滑参数
        if self.method == 'adaptive':
            self.adaptive_alpha_ = self._compute_adaptive_alpha(len(y), self.prior_bad_rate_)
        else:
            self.adaptive_alpha_ = self.smoothing_param

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

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._is_fitted = True
        return self

    def _compute_adaptive_alpha(self, n_samples: int, prior_rate: float) -> float:
        """计算自适应平滑参数.
        
        根据样本量和先验率动态调整平滑强度：
        - 样本量小 -> 增加平滑
        - 先验率极端 -> 增加平滑
        """
        base_alpha = self.smoothing_param
        
        # 样本量调整：小样本增加平滑
        if n_samples < 100:
            sample_factor = 2.0
        elif n_samples < 1000:
            sample_factor = 1.5
        else:
            sample_factor = 1.0
        
        # 先验率调整：极端值增加平滑
        if prior_rate < 0.05 or prior_rate > 0.95:
            rate_factor = 2.0
        elif prior_rate < 0.1 or prior_rate > 0.9:
            rate_factor = 1.5
        else:
            rate_factor = 1.0
        
        return base_alpha * sample_factor * rate_factor

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

        # 使用混合预分箱策略
        initial_splits = self._get_initial_splits_hybrid(x_valid, y_valid)

        if len(initial_splits) == 0:
            return np.array([])

        # 根据平滑方法优化分箱
        splits = self._smooth_split_optimization_v3(x_valid, y_valid, initial_splits)

        return splits

    def _get_initial_splits_hybrid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """混合预分箱策略：等频+基于目标变量的分箱点."""
        n = len(x)
        
        # 1. 等频预分箱（增加预分箱数以获得更多候选点）
        n_prebins = min(self.n_prebins, max(self.max_n_bins * 5, 50))
        n_prebins = min(n_prebins, n // 5)  # 每箱至少5个样本
        n_prebins = max(n_prebins, self.min_n_bins * 3)

        quantiles = np.linspace(0, 1, n_prebins + 1)[1:-1]
        quantile_splits = np.percentile(x, quantiles * 100)
        quantile_splits = np.unique(quantile_splits)
        
        # 2. 基于目标变量的候选切分点（决策树思想）
        tree_splits = self._get_tree_based_splits(x, y, max_splits=n_prebins // 3)
        
        # 3. 合并切分点并去重
        all_splits = np.sort(np.unique(np.concatenate([quantile_splits, tree_splits])))
        
        return all_splits

    def _get_tree_based_splits(self, x: np.ndarray, y: np.ndarray, max_splits: int = 10) -> np.ndarray:
        """基于目标变量获取候选切分点（决策树分裂思想）."""
        n = len(x)
        if n < 20:
            return np.array([])
        
        # 排序
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        
        splits = []
        n_candidates = min(max_splits * 3, n // 10)
        
        # 在类别变化处寻找候选切分点
        candidate_positions = []
        for i in range(1, n):
            if y_sorted[i] != y_sorted[i-1]:
                if i > 10 and i < n - 10:  # 确保有足够样本
                    candidate_positions.append(i)
        
        # 如果没有类别变化，使用等间距
        if len(candidate_positions) == 0:
            candidate_positions = np.linspace(10, n-10, min(n_candidates, 20), dtype=int)
        
        # 选择候选点并计算IV
        candidate_splits = []
        total_bad = y_sorted.sum()
        total_good = len(y_sorted) - total_bad
        
        for pos in candidate_positions[:n_candidates]:
            split = (x_sorted[pos-1] + x_sorted[pos]) / 2
            
            left_y = y_sorted[:pos]
            right_y = y_sorted[pos:]
            
            iv = self._calculate_iv_for_split(left_y, right_y, total_bad, total_good)
            candidate_splits.append((split, iv))
        
        # 按IV排序，选择前max_splits个
        candidate_splits.sort(key=lambda x: x[1], reverse=True)
        selected_splits = [s[0] for s in candidate_splits[:max_splits]]
        
        return np.array(selected_splits)

    def _calculate_iv_for_split(self, y_left: np.ndarray, y_right: np.ndarray, 
                                total_bad: int, total_good: int) -> float:
        """计算切分点的IV值."""
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

    def _smooth_split_optimization_v3(
        self,
        x: np.ndarray,
        y: np.ndarray,
        initial_splits: np.ndarray
    ) -> np.ndarray:
        """优化的平滑切分点选择 - V3版本（针对平滑分布优化）."""
        splits = list(initial_splits)
        min_samples = self._get_min_samples(len(x))
        prior = self.prior_bad_rate_

        # 第一阶段：从少到多，找到满足min_n_bins的切分点
        # 如果初始切分点太多，先进行粗略合并
        while len(splits) > self.max_n_bins * 2:
            bins = np.digitize(x, splits)
            bin_stats = self._compute_smoothed_stats(bins, y, prior)
            merge_idx = self._find_merge_candidate_conservative(bins, y, bin_stats, min_samples)
            if merge_idx is None:
                break
            splits.pop(merge_idx)

        # 第二阶段：精细合并，使用保守策略
        max_iter = 300
        for iteration in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算当前分箱的统计信息
            bins = np.digitize(x, splits)
            current_iv = self._calculate_total_iv(bins, y)
            bin_stats = self._compute_smoothed_stats(bins, y, prior)

            # 如果已经满足单调性且分箱数合适，尝试停止
            if self.monotonic and len(splits) >= self.min_n_bins - 1:
                if self._check_monotonicity(bin_stats['smoothed_rate'].values):
                    if len(splits) <= self.max_n_bins - 1:
                        # 检查IV是否足够高
                        if current_iv > self.min_iv_improvement * 10:
                            break

            # 找到最佳合并候选（使用保守策略）
            merge_idx = self._find_merge_candidate_conservative(bins, y, bin_stats, min_samples)

            if merge_idx is None:
                break

            # 模拟合并，检查IV损失
            bins_merged = bins.copy()
            bins_merged[bins == merge_idx + 1] = merge_idx
            bins_merged[bins > merge_idx + 1] -= 1
            new_iv = self._calculate_total_iv(bins_merged, y)
            iv_loss = current_iv - new_iv

            # 保守策略：如果IV损失太大，不合并
            if iv_loss > self.min_iv_improvement and len(splits) <= self.max_n_bins:
                break

            # 执行合并
            splits.pop(merge_idx)

        # 第三阶段：如果分箱数仍然过多，强制合并
        while len(splits) > self.max_n_bins - 1:
            bins = np.digitize(x, splits)
            bin_stats = self._compute_smoothed_stats(bins, y, prior)
            merge_idx = self._find_merge_candidate_conservative(bins, y, bin_stats, min_samples, force=True)
            if merge_idx is None:
                break
            splits.pop(merge_idx)

        return np.array(splits)

    def _compute_smoothed_stats(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        prior: float
    ) -> pd.DataFrame:
        """计算平滑后的统计信息."""
        n_bins = bins.max() + 1
        stats_data = []

        for b in range(n_bins):
            mask = bins == b
            bad = y[mask].sum()
            count = mask.sum()
            good = count - bad
            stats_data.append({'bin': b, 'bad': bad, 'count': count, 'good': good})

        stats = pd.DataFrame(stats_data)
        alpha = self.adaptive_alpha_

        if self.method == 'laplace':
            stats['smoothed_rate'] = (stats['bad'] + alpha) / (stats['count'] + 2 * alpha)
        elif self.method == 'bayesian':
            stats['smoothed_rate'] = (stats['bad'] + alpha * prior) / (stats['count'] + alpha)
        elif self.method == 'beta':
            effective_alpha = alpha * (1 + 1 / np.sqrt(stats['count'] + 1))
            stats['smoothed_rate'] = (stats['bad'] + effective_alpha * prior) / (stats['count'] + effective_alpha)
        else:  # adaptive
            weight = stats['count'] / (stats['count'] + alpha * 10)
            empirical_rate = stats['bad'] / stats['count'].clip(lower=1)
            stats['smoothed_rate'] = weight * empirical_rate + (1 - weight) * prior

        return stats

    def _find_merge_candidate_conservative(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        bin_stats: pd.DataFrame,
        min_samples: int,
        force: bool = False
    ) -> Optional[int]:
        """保守的合并候选选择."""
        n_bins = len(bin_stats)
        if n_bins <= 2:  # 至少保留2箱
            return None

        counts = bin_stats['count'].values
        rates = bin_stats['smoothed_rate'].values

        # 优先处理样本数不足的箱
        for i in range(n_bins - 1):
            if counts[i] < min_samples or counts[i + 1] < min_samples:
                return i

        if force:
            # 强制合并：选择差异最小的
            min_diff = float('inf')
            merge_idx = 0
            for i in range(n_bins - 1):
                diff = abs(rates[i] - rates[i + 1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i
            return merge_idx

        # 保守策略：基于IV损失和卡方检验
        candidates = []
        for i in range(n_bins - 1):
            iv_loss = self._calculate_merge_iv_loss(bins, y, i)
            chi2 = self._calculate_chi2(bins, y, i, i + 1)
            
            # 检查单调性
            monotonic_violation = False
            if self.monotonic:
                merged_rate = (bin_stats.iloc[i]['bad'] + bin_stats.iloc[i + 1]['bad']) / \
                              (bin_stats.iloc[i]['count'] + bin_stats.iloc[i + 1]['count'])
                new_rates = np.concatenate([rates[:i], [merged_rate], rates[i + 2:]])
                if not self._check_monotonicity(new_rates):
                    monotonic_violation = True

            # 评分：IV损失越小越好，卡方越小越好
            score = iv_loss
            if chi2 < self.chi2_threshold:
                score *= 0.7  # 卡方检验通过的给予优惠
            if monotonic_violation:
                score += 10  # 违反单调性的给予大惩罚
            
            candidates.append((i, score, iv_loss, chi2))

        # 选择评分最低的候选
        candidates.sort(key=lambda x: x[1])
        
        # 检查最优候选的IV损失是否可接受
        for idx, score, iv_loss, chi2 in candidates:
            if iv_loss < self.min_iv_improvement * 5:  # 放宽阈值
                return idx
        
        # 如果没有好的候选，返回None（停止合并）
        return None

    def _calculate_chi2(self, bins: np.ndarray, y: np.ndarray, bin1: int, bin2: int) -> float:
        """计算两个箱的卡方统计量."""
        mask1 = bins == bin1
        mask2 = bins == bin2
        
        bad1, count1 = y[mask1].sum(), mask1.sum()
        good1 = count1 - bad1
        
        bad2, count2 = y[mask2].sum(), mask2.sum()
        good2 = count2 - bad2
        
        total = count1 + count2
        total_bad = bad1 + bad2
        total_good = good1 + good2
        
        if total == 0 or total_bad == 0 or total_good == 0:
            return float('inf')
        
        expected_bad1 = count1 * total_bad / total
        expected_good1 = count1 * total_good / total
        expected_bad2 = count2 * total_bad / total
        expected_good2 = count2 * total_good / total
        
        chi2 = 0
        for obs, exp in [(bad1, expected_bad1), (good1, expected_good1),
                         (bad2, expected_bad2), (good2, expected_good2)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        
        return chi2

    def _calculate_merge_iv_loss(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        merge_idx: int
    ) -> float:
        """计算合并两个箱的IV损失."""
        iv_before = self._calculate_total_iv(bins, y)

        bins_merged = bins.copy()
        bins_merged[bins == merge_idx + 1] = merge_idx
        bins_merged[bins > merge_idx + 1] -= 1

        iv_after = self._calculate_total_iv(bins_merged, y)

        return max(0, iv_before - iv_after)

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

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行平滑分箱."""
        temp_df = pd.DataFrame({'category': x.values, 'target': y.values})
        cat_stats = temp_df.groupby('category')['target'].agg(['sum', 'count'])
        cat_stats.columns = ['bad', 'count']
        cat_stats['good'] = cat_stats['count'] - cat_stats['bad']

        prior = self.prior_bad_rate_
        alpha = self.adaptive_alpha_

        if self.method == 'laplace':
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + alpha) / (cat_stats['count'] + 2 * alpha)
        elif self.method == 'bayesian':
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + alpha * prior) / (cat_stats['count'] + alpha)
        elif self.method == 'beta':
            effective_alpha = alpha * (1 + 1 / np.sqrt(cat_stats['count'] + 1))
            cat_stats['smoothed_rate'] = (cat_stats['bad'] + effective_alpha * prior) / (cat_stats['count'] + effective_alpha)
        else:  # adaptive
            weight = cat_stats['count'] / (cat_stats['count'] + alpha * 10)
            empirical_rate = cat_stats['bad'] / cat_stats['count'].clip(lower=1)
            cat_stats['smoothed_rate'] = weight * empirical_rate + (1 - weight) * prior

        cat_stats = cat_stats.sort_values('smoothed_rate')
        categories = self._merge_categories_conservative(cat_stats, y)

        return categories

    def _merge_categories_conservative(
        self,
        cat_stats: pd.DataFrame,
        y: pd.Series
    ) -> List:
        """保守的类别合并."""
        categories = cat_stats.index.tolist()
        min_samples = self._get_min_samples(len(y))

        # 合并样本数不足的类别
        i = 0
        while i < len(categories):
            if cat_stats.iloc[i]['count'] < min_samples:
                if i > 0:
                    merge_target = i - 1
                elif i < len(categories) - 1:
                    merge_target = i + 1
                else:
                    i += 1
                    continue

                cat1 = categories[i]
                cat2 = categories[merge_target]
                merged_cat = f"{cat1},{cat2}"

                idx1 = i if i < merge_target else merge_target
                idx2 = merge_target if i < merge_target else i

                merged_bad = cat_stats.iloc[idx1]['bad'] + cat_stats.iloc[idx2]['bad']
                merged_count = cat_stats.iloc[idx1]['count'] + cat_stats.iloc[idx2]['count']

                prior = self.prior_bad_rate_
                alpha = self.adaptive_alpha_
                
                if self.method == 'laplace':
                    merged_rate = (merged_bad + alpha) / (merged_count + 2 * alpha)
                elif self.method == 'bayesian':
                    merged_rate = (merged_bad + alpha * prior) / (merged_count + alpha)
                elif self.method == 'beta':
                    effective_alpha = alpha * (1 + 1 / np.sqrt(merged_count + 1))
                    merged_rate = (merged_bad + effective_alpha * prior) / (merged_count + effective_alpha)
                else:
                    weight = merged_count / (merged_count + alpha * 10)
                    empirical_rate = merged_bad / merged_count if merged_count > 0 else prior
                    merged_rate = weight * empirical_rate + (1 - weight) * prior

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

            min_diff = float('inf')
            merge_idx = 0

            for i in range(len(rates) - 1):
                rate_diff = abs(rates[i] - rates[i + 1])
                size_penalty = abs(counts[i] - counts[i + 1]) / max(counts[i], counts[i + 1], 1)
                diff = rate_diff + 0.1 * size_penalty
                
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            cat1 = categories[merge_idx]
            cat2 = categories[merge_idx + 1]
            merged_cat = f"{cat1},{cat2}"

            merged_bad = cat_stats.iloc[merge_idx]['bad'] + cat_stats.iloc[merge_idx + 1]['bad']
            merged_count = cat_stats.iloc[merge_idx]['count'] + cat_stats.iloc[merge_idx + 1]['count']

            prior = self.prior_bad_rate_
            alpha = self.adaptive_alpha_
            
            if self.method == 'laplace':
                merged_rate = (merged_bad + alpha) / (merged_count + 2 * alpha)
            elif self.method == 'bayesian':
                merged_rate = (merged_bad + alpha * prior) / (merged_count + alpha)
            elif self.method == 'beta':
                effective_alpha = alpha * (1 + 1 / np.sqrt(merged_count + 1))
                merged_rate = (merged_bad + effective_alpha * prior) / (merged_count + effective_alpha)
            else:
                weight = merged_count / (merged_count + alpha * 10)
                empirical_rate = merged_bad / merged_count if merged_count > 0 else prior
                merged_rate = weight * empirical_rate + (1 - weight) * prior

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
                    woe_map[-1] = 0
                    woe_map[-2] = 0
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result
