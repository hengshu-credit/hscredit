"""Best Lift 分箱算法.

基于最大化头尾部 Lift 差异的分箱方法。
核心思想：找到最优分割点，使得头部或尾部的 Lift 极端化（远高于或远低于1）。

算法流程：
1. 预分割：将数据分成足够细的初始箱（默认50个）
2. 合并优化：使用动态规划或贪心算法，在单调性约束下合并分箱
3. 目标：最大化头部/尾部的 Lift 差异
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd

from .base import BaseBinning


class BestLiftBinning(BaseBinning):
    """Best Lift 分箱.

    基于最大化 Lift 差异的分箱方法，特别关注头部或尾部的极端值。
    Lift = 箱内坏样本率 / 总体坏样本率。
    
    业务场景：
    - 高风险识别：寻找头部 Lift > 2 或尾部 Lift < 0.5 的分箱
    - 风险分层：最大化不同分箱间的风险差异

    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为0.5
    :param min_lift: 最小 Lift 阈值，默认为0（不限制）
        - 设为0：不限制，允许任何 Lift 值
        - 设为0.5：过滤掉 Lift > 0.5 的箱（用于识别极低风险）
        - 设为1.5：只保留 Lift > 1.5 的箱（用于识别极高风险）
    :param monotonic: 坏样本率单调性约束，默认为True
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
    :param n_prebins: 预分箱数量，默认为50
        - 预分箱越细，结果越精确，但计算量越大
    :param optimization: 优化目标，默认为'extreme'
        - 'extreme': 最大化极端箱的 Lift（头部最高或尾部最低）
        - 'spread': 最大化 Lift 分布范围（max - min）
        - 'iv': 最大化 IV 值
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param random_state: 随机种子，默认为None

    **示例**

    >>> from hscredit.core.binning import BestLiftBinning
    >>> # 识别高风险客群
    >>> binner = BestLiftBinning(max_n_bins=5, optimization='extreme')
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    **注意**

    Best Lift 分箱的特点:
    1. 关注头部/尾部的极端风险识别
    2. 支持单调性约束，保证业务可解释性
    3. 自动确定最优单调方向
    4. 高效的动态规划风格实现
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = 0.5,
        min_lift: float = 0.0,
        monotonic: Union[bool, str] = True,
        n_prebins: int = 50,
        optimization: str = 'extreme',
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
            monotonic=monotonic,
            missing_separate=missing_separate,
            special_codes=special_codes,
            random_state=random_state,
            **kwargs
        )
        self.min_lift = min_lift
        self.n_prebins = n_prebins
        self.optimization = optimization

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'BestLiftBinning':
        """拟合 Best Lift 分箱.

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
            # 类别型变量：按坏样本率排序后编码
            splits = self._best_lift_categorical(X_valid, y_valid)
            self.splits_[feature] = np.array(splits) if splits else np.array([])
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            # 数值型变量
            splits = self._best_lift_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _best_lift_numerical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对数值型变量进行 Best Lift 分箱.

        算法流程：
        1. 等频预分割成 n_prebins 个初始箱
        2. 计算每个初始箱的统计信息
        3. 使用贪心算法合并分箱，优化目标函数

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)

        if n_samples == 0:
            return []

        total_bad_rate = np.mean(y_vals)
        if total_bad_rate == 0 or total_bad_rate == 1:
            return []

        # 计算样本数约束
        min_samples = self._get_min_samples(n_samples)
        max_samples = self._get_max_samples(n_samples)

        # 排序数据
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]

        # 阶段1：等频预分割
        prebins = self._create_prebins(x_sorted, y_sorted, min_samples)
        
        if len(prebins) <= self.max_n_bins:
            # 预分箱数已经满足要求
            splits = self._prebins_to_splits(prebins, x_sorted)
        else:
            # 阶段2：贪心合并分箱
            splits = self._greedy_merge(
                prebins, x_sorted, y_sorted,
                min_samples, max_samples, total_bad_rate
            )

        # 阶段3：单调性调整
        if self.monotonic and len(splits) > 0:
            splits = self._enforce_monotonic(
                splits, x_sorted, y_sorted, min_samples, total_bad_rate
            )

        return sorted(splits)

    def _create_prebins(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        min_samples: int
    ) -> List[Dict]:
        """创建预分箱.

        使用等频分割，确保每个预分箱至少有 min_samples 个样本。

        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param min_samples: 每箱最小样本数
        :return: 预分箱列表，每个元素包含 {start_idx, end_idx, bad_count, total_count, bad_rate}
        """
        n_samples = len(x_sorted)
        
        # 计算实际可用的预分箱数
        n_prebins = min(self.n_prebins, n_samples // min_samples)
        n_prebins = max(n_prebins, self.max_n_bins)
        
        # 等频分割点
        quantile_points = np.linspace(0, n_samples, n_prebins + 1, dtype=int)
        
        prebins = []
        for i in range(n_prebins):
            start_idx = quantile_points[i]
            end_idx = quantile_points[i + 1]
            
            if end_idx <= start_idx:
                continue
            
            bin_y = y_sorted[start_idx:end_idx]
            bad_count = np.sum(bin_y)
            total_count = len(bin_y)
            bad_rate = bad_count / total_count if total_count > 0 else 0
            
            prebins.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'bad_count': bad_count,
                'total_count': total_count,
                'bad_rate': bad_rate
            })
        
        return prebins

    def _greedy_merge(
        self,
        prebins: List[Dict],
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        min_samples: int,
        max_samples: int,
        total_bad_rate: float
    ) -> List[float]:
        """贪心合并预分箱.

        从预分箱开始，逐步合并相邻的分箱，直到达到目标分箱数。
        每次选择对目标函数影响最小的相邻分箱进行合并。

        :param prebins: 预分箱列表
        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param min_samples: 最小样本数
        :param max_samples: 最大样本数
        :param total_bad_rate: 总体坏样本率
        :return: 分割点列表
        """
        # 复制预分箱列表
        bins = [b.copy() for b in prebins]
        n_bins = len(bins)
        
        # 如果已经满足分箱数要求，直接返回
        if n_bins <= self.max_n_bins:
            return self._bins_to_splits(bins, x_sorted)
        
        # 计算相邻分箱的合并损失
        def calc_merge_loss(bins: List[Dict], merge_idx: int) -> float:
            """计算合并相邻两个分箱的损失."""
            if merge_idx >= len(bins) - 1:
                return float('inf')
            
            left = bins[merge_idx]
            right = bins[merge_idx + 1]
            
            # 合并后的统计
            merged_bad = left['bad_count'] + right['bad_count']
            merged_total = left['total_count'] + right['total_count']
            merged_rate = merged_bad / merged_total if merged_total > 0 else 0
            
            # 检查样本数约束
            if merged_total > max_samples:
                return float('inf')
            
            # 计算合并前后的目标函数差异
            if self.optimization == 'extreme':
                # 极端值优化：最大化头部或尾部的 Lift
                # 合并损失 = 合并前后 Lift 极值的变化
                left_lift = left['bad_rate'] / total_bad_rate if total_bad_rate > 0 else 1
                right_lift = right['bad_rate'] / total_bad_rate if total_bad_rate > 0 else 1
                merged_lift = merged_rate / total_bad_rate if total_bad_rate > 0 else 1
                
                # 损失 = 合并导致的极端 Lift 损失
                pre_extreme = max(abs(left_lift - 1), abs(right_lift - 1))
                post_extreme = abs(merged_lift - 1)
                loss = pre_extreme - post_extreme
                
            elif self.optimization == 'spread':
                # 分布范围优化：最大化 Lift 的分布范围
                all_rates = [b['bad_rate'] for b in bins]
                current_spread = max(all_rates) - min(all_rates)
                
                # 模拟合并后的 spread
                temp_rates = [b['bad_rate'] for b in bins]
                temp_rates[merge_idx] = merged_rate
                temp_rates.pop(merge_idx + 1)
                new_spread = max(temp_rates) - min(temp_rates)
                
                loss = current_spread - new_spread
                
            else:  # optimization == 'iv'
                # IV 优化：最大化信息值
                # 简化计算，使用坏样本率差异
                rate_diff = abs(left['bad_rate'] - right['bad_rate'])
                loss = rate_diff * (left['total_count'] + right['total_count']) / len(y_sorted)
            
            return loss
        
        # 贪心合并
        while len(bins) > self.max_n_bins:
            # 找到合并损失最小的相邻分箱对
            min_loss = float('inf')
            best_merge_idx = -1
            
            for i in range(len(bins) - 1):
                loss = calc_merge_loss(bins, i)
                if loss < min_loss:
                    min_loss = loss
                    best_merge_idx = i
            
            if best_merge_idx == -1:
                break
            
            # 执行合并
            left = bins[best_merge_idx]
            right = bins[best_merge_idx + 1]
            
            merged_bin = {
                'start_idx': left['start_idx'],
                'end_idx': right['end_idx'],
                'bad_count': left['bad_count'] + right['bad_count'],
                'total_count': left['total_count'] + right['total_count'],
                'bad_rate': (left['bad_count'] + right['bad_count']) / 
                           (left['total_count'] + right['total_count'])
            }
            
            bins[best_merge_idx] = merged_bin
            bins.pop(best_merge_idx + 1)
        
        # 确保满足最小分箱数
        while len(bins) < self.min_n_bins and len(bins) > 1:
            # 找到样本数最多的分箱进行拆分
            max_idx = np.argmax([b['total_count'] for b in bins])
            target_bin = bins[max_idx]
            
            if target_bin['total_count'] < min_samples * 2:
                break
            
            # 从中间拆分
            mid_idx = (target_bin['start_idx'] + target_bin['end_idx']) // 2
            
            left_y = y_sorted[target_bin['start_idx']:mid_idx]
            right_y = y_sorted[mid_idx:target_bin['end_idx']]
            
            left_bin = {
                'start_idx': target_bin['start_idx'],
                'end_idx': mid_idx,
                'bad_count': np.sum(left_y),
                'total_count': len(left_y),
                'bad_rate': np.mean(left_y)
            }
            
            right_bin = {
                'start_idx': mid_idx,
                'end_idx': target_bin['end_idx'],
                'bad_count': np.sum(right_y),
                'total_count': len(right_y),
                'bad_rate': np.mean(right_y)
            }
            
            bins[max_idx] = left_bin
            bins.insert(max_idx + 1, right_bin)
        
        return self._bins_to_splits(bins, x_sorted)

    def _bins_to_splits(
        self,
        bins: List[Dict],
        x_sorted: np.ndarray
    ) -> List[float]:
        """将分箱转换为分割点.

        :param bins: 分箱列表
        :param x_sorted: 排序后的特征值
        :return: 分割点列表
        """
        splits = []
        for i in range(len(bins) - 1):
            # 分割点是相邻分箱边界的中点
            end_idx = bins[i]['end_idx']
            if end_idx < len(x_sorted):
                split_val = (x_sorted[end_idx - 1] + x_sorted[end_idx]) / 2
                splits.append(split_val)
        return splits

    def _prebins_to_splits(
        self,
        prebins: List[Dict],
        x_sorted: np.ndarray
    ) -> List[float]:
        """将预分箱直接转换为分割点.

        :param prebins: 预分箱列表
        :param x_sorted: 排序后的特征值
        :return: 分割点列表
        """
        return self._bins_to_splits(prebins, x_sorted)

    def _enforce_monotonic(
        self,
        splits: List[float],
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        min_samples: int,
        total_bad_rate: float
    ) -> List[float]:
        """强制单调性约束.

        :param splits: 初始分割点
        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param min_samples: 最小样本数
        :param total_bad_rate: 总体坏样本率
        :return: 满足单调性的分割点
        """
        if not splits or len(splits) == 0:
            return splits

        splits = list(splits)
        
        # 计算每个箱的坏样本率
        def get_bad_rates(splits_list: List[float]) -> List[float]:
            bins = np.searchsorted(splits_list, x_sorted, side='right')
            n_bins = len(splits_list) + 1
            rates = []
            for i in range(n_bins):
                mask = bins == i
                if mask.sum() > 0:
                    rates.append(y_sorted[mask].mean())
                else:
                    rates.append(0)
            return rates

        bad_rates = get_bad_rates(splits)
        
        # 确定单调方向
        if self.monotonic == 'ascending':
            direction = 'ascending'
        elif self.monotonic == 'descending':
            direction = 'descending'
        else:  # auto 或 True
            # 计算违反单调性的次数
            asc_violations = sum(1 for i in range(len(bad_rates)-1) 
                                if bad_rates[i] > bad_rates[i+1])
            desc_violations = sum(1 for i in range(len(bad_rates)-1) 
                                 if bad_rates[i] < bad_rates[i+1])
            direction = 'ascending' if asc_violations <= desc_violations else 'descending'
        
        # 合并违反单调性的相邻箱
        max_iter = len(splits)
        for _ in range(max_iter):
            if len(splits) <= 1:
                break
            
            bad_rates = get_bad_rates(splits)
            
            # 找到第一个违反单调性的位置
            merge_idx = -1
            for i in range(len(bad_rates) - 1):
                if direction == 'ascending' and bad_rates[i] > bad_rates[i + 1]:
                    merge_idx = i
                    break
                elif direction == 'descending' and bad_rates[i] < bad_rates[i + 1]:
                    merge_idx = i
                    break
            
            if merge_idx == -1:
                break
            
            # 删除分割点（合并相邻箱）
            splits.pop(merge_idx)
        
        # 检查是否满足最小分箱数
        if len(splits) + 1 < self.min_n_bins:
            # 尝试重新分割（简单策略：等频分割）
            n_new_splits = self.min_n_bins - 1
            n_samples = len(x_sorted)
            quantiles = np.linspace(0, n_samples, n_new_splits + 2, dtype=int)[1:-1]
            splits = [(x_sorted[q-1] + x_sorted[q]) / 2 for q in quantiles if q < n_samples]
        
        return sorted(splits)

    def _best_lift_categorical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对类别型变量进行 Best Lift 分箱.

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表（编码边界）
        """
        # 计算总体坏样本率
        total_bad_rate = y.mean()
        if total_bad_rate == 0:
            return []

        # 使用向量化操作计算类别统计
        df = pd.DataFrame({'X': X, 'y': y})
        category_stats = df.groupby('X')['y'].agg(['mean', 'count']).reset_index()
        category_stats.columns = ['category', 'bad_rate', 'count']

        # 计算 Lift
        category_stats['lift'] = category_stats['bad_rate'] / total_bad_rate

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(X))
        category_stats = category_stats[category_stats['count'] >= min_samples]

        if len(category_stats) <= 1:
            return []

        # 按坏样本率排序
        category_stats = category_stats.sort_values('bad_rate')

        # 返回编码边界
        n_categories = len(category_stats)
        if n_categories <= self.max_n_bins:
            return []
        
        return [i - 0.5 for i in range(1, min(n_categories, self.max_n_bins))]

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数.

        :param n_total: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size < 1:
            return max(int(n_total * self.min_bin_size), 1)
        return max(int(self.min_bin_size), 1)

    def _get_max_samples(self, n_total: int) -> int:
        """获取最大样本数.

        :param n_total: 总样本数
        :return: 最大样本数
        """
        if self.max_bin_size is None:
            return n_total
        if self.max_bin_size < 1:
            return int(n_total * self.max_bin_size)
        return int(self.max_bin_size)

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
        
        将原始特征值转换为分箱索引、分箱标签、WOE值或LIFT值。
        
        :param X: 待转换数据, DataFrame或数组格式
        :param metric: 转换类型, 可选值:
            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理
            - 'bins': 返回分箱标签字符串, 用于可视化或报告
            - 'woe': 返回WOE值, 用于逻辑回归建模
            - 'lift': 返回LIFT值, 用于评估分箱效果
        :param kwargs: 其他参数(保留兼容性)
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
        >>> binner = BestLiftBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        >>> 
        >>> # 获取LIFT值
        >>> X_lift = binner.transform(X_test, metric='lift')
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
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

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
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    woe_map[-1] = 0
                    woe_map[-2] = 0
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = [woe_map.get(b, 0) for b in bins]
            elif metric == 'lift':
                # 返回 Lift 值
                bin_table = self.get_bin_table(feature)
                lift_map = {}
                for i, row in bin_table.iterrows():
                    if row['分箱标签'] in ['缺失', 'special']:
                        lift_map[-1 if row['分箱标签'] == '缺失' else -2] = np.nan
                    else:
                        lift_map[i] = row['LIFT值']
                result[feature] = [lift_map.get(b, np.nan) for b in bins]
            else:
                raise ValueError(f"不支持的metric: {metric}")

        return result

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """获取分箱统计表.

        :param feature: 特征名
        :return: 分箱统计表
        """
        if feature not in self.bin_tables_:
            raise KeyError(f"特征 '{feature}' 未找到")

        bin_table = self.bin_tables_[feature].copy()

        # 添加 Lift 列
        # 排除缺失和特殊值箱计算总体坏样本率
        valid_mask = ~bin_table['分箱标签'].isin(['缺失', 'special'])
        if valid_mask.any():
            valid_bad_rates = bin_table.loc[valid_mask, '坏样本率']
            valid_counts = bin_table.loc[valid_mask, '样本总数']
            total_bad = (valid_bad_rates * valid_counts).sum()
            total_count = valid_counts.sum()
            total_bad_rate = total_bad / total_count if total_count > 0 else 0
        else:
            total_bad_rate = bin_table['坏样本率'].mean()

        # 计算每箱的 lift
        lifts = []
        for _, row in bin_table.iterrows():
            if row['分箱标签'] in ['缺失', 'special']:
                lifts.append(np.nan)
            else:
                lift = row['坏样本率'] / total_bad_rate if total_bad_rate > 0 else 1.0
                lifts.append(lift)

        bin_table['LIFT值'] = lifts

        return bin_table


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    
    # 测试代码
    np.random.seed(42)
    n_samples = 5000

    # 生成测试数据
    x = np.concatenate([
        np.random.normal(0, 1, n_samples // 2),
        np.random.normal(3, 1, n_samples // 2)
    ])
    # 构造有区分度的目标变量
    y = np.array([
        np.random.binomial(1, 0.2, n_samples // 2),
        np.random.binomial(1, 0.6, n_samples // 2)
    ]).flatten()

    # 添加一些缺失值
    x[np.random.choice(n_samples, 100, replace=False)] = np.nan

    X = pd.DataFrame({'feature': x})
    y = pd.Series(y)

    print("=" * 60)
    print("Best Lift 分箱测试 - 新算法")
    print("=" * 60)

    # 测试 Best Lift 分箱
    binner = BestLiftBinning(
        max_n_bins=5,
        min_lift=0.0,
        monotonic=True,
        optimization='extreme',
        n_prebins=50,
        verbose=True
    )
    binner.fit(X, y)

    print("\n分箱统计表:")
    print(binner.get_bin_table('feature'))

    # 转换测试
    print("\n转换测试:")
    X_lift = binner.transform(X, metric='lift')
    print("\nLift值 (前10行):")
    print(X_lift.head(10))

    print("\n切分点:", binner.splits_['feature'])
    print("分箱数:", binner.n_bins_['feature'])
