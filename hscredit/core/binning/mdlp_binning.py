"""MDLP (Minimum Description Length Principle) 分箱.

基于信息论的递归分箱算法，由 Fayyad 和 Irani 于 1993 年提出。
使用 MDLP 准则作为停止条件，自动确定最优分箱数。

优化版本V3（针对平滑分布）：
- 放宽终止条件，避免过早停止
- 支持强制最小分箱数
- 改进候选切分点选择，优先选择IV增益大的点
- 针对平滑分布数据优化
"""

from typing import Union, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy import special

from .base import BaseBinning


class MDLPBinning(BaseBinning):
    """MDLP 分箱算法.

    基于最小描述长度原理的递归分箱方法，自动确定最优分箱数。
    使用信息增益和 MDLP 准则决定是否继续分割。

    优化版本V3特点（针对平滑分布）：
    - 放宽终止条件，支持force_min_bins强制最小分箱数
    - 改进候选切分点选择，结合IV值评估
    - 更好的平滑分布处理

    :param target: 目标变量列名，默认为'target'
    :param max_n_bins: 最大分箱数，默认为10
    :param min_n_bins: 最小分箱数，默认为2
    :param min_samples_split: 分割内部节点所需的最小样本数，默认为2
    :param min_samples_leaf: 叶子节点所需的最小样本数，默认为2
    :param max_candidates: 每次评估的最大候选切分点数，默认为32
    :param min_iv_gain: 最小IV增益阈值，默认为0.0001（降低以允许更多分箱）
    :param force_min_bins: 是否强制满足最小分箱数，默认为True
    :param mdlp_weight: MDLP准则权重（0-1之间），默认为0.7（降低以放宽终止条件）
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **参考样例**

    >>> from hscredit.core.binning import MDLPBinning
    >>> binner = MDLPBinning(max_n_bins=5, min_n_bins=2)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 10,
        min_n_bins: int = 2,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        max_candidates: int = 32,
        min_iv_gain: float = 0.0001,
        force_min_bins: bool = True,
        mdlp_weight: float = 0.7,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        **kwargs
    ):
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            random_state=random_state,
            verbose=verbose,
        )
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.min_iv_gain = min_iv_gain
        self.force_min_bins = force_min_bins
        self.mdlp_weight = mdlp_weight

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'MDLPBinning':
        """拟合 MDLP 分箱.

        :param X: 特征数据
        :param y: 目标变量
        :return: self
        """
        X, y = self._check_input(X, y)

        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            # 检测特征类型
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                # 类别型特征：按坏样本率排序分组
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
                self.n_bins_[feature] = len(splits) + 1
                bins = self._apply_splits(X[feature], splits, 'categorical')
            else:
                # 数值型特征：使用 MDLP 算法
                x_numeric = pd.to_numeric(X[feature], errors='coerce')
                x_clean = x_numeric.dropna()
                y_clean = y[x_numeric.notna()]

                if len(x_clean) >= self.min_samples_split:
                    splits = self._mdlp_split_v3(x_clean.values, y_clean.values)
                    self.splits_[feature] = self._round_splits(np.sort(splits))
                    self.n_bins_[feature] = len(splits) + 1
                else:
                    self.splits_[feature] = np.array([])
                    self.n_bins_[feature] = 1
                bins = self._apply_splits(X[feature], self.splits_[feature], 'numerical')

            # 计算分箱统计
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._is_fitted = True
        return self

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> list:
        """对类别型特征进行分箱（按坏样本率排序）.

        MDLP 算法仅适用于数值型特征，对于类别型特征回退到按坏样本率排序的方式。

        :param x: 特征数据
        :param y: 目标变量
        :return: 类别列表（按坏样本率排序）
        """
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            return []

        # 计算每个类别的坏样本率并按其排序
        cat_stats = pd.DataFrame({
            'category': x_valid,
            'target': y_valid
        }).groupby('category')['target'].agg(['mean', 'count'])

        cat_stats = cat_stats.sort_values('mean')

        return cat_stats.index.tolist()

    def _mdlp_split_v3(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """MDLP 递归分箱算法 - V3版本（针对平滑分布优化）.

        :param x: 特征值数组
        :param y: 目标变量数组
        :return: 切分点列表
        """
        # 按特征值排序
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]

        splits = []
        
        # 首先尝试找到所有候选分割点
        all_candidates = self._find_all_candidates_v3(x_sorted, y_sorted)
        
        # 使用递归分割
        self._recurse_v3(x_sorted, y_sorted, splits, all_candidates, 0)
        
        # 如果分箱数不足且force_min_bins为True，只补足到 min_n_bins
        if self.force_min_bins and len(splits) < self.min_n_bins - 1:
            splits = self._force_additional_splits_v3(x_sorted, y_sorted, splits, all_candidates)
        
        return sorted(list(set(splits)))

    def _find_all_candidates_v3(self, x: np.ndarray, y: np.ndarray) -> List[int]:
        """找到所有可能的候选切分点位置 - V3版本."""
        n = len(x)
        candidates = []
        
        # 找到所有类别变化的位置
        for i in range(1, n):
            if y[i] != y[i-1]:
                candidates.append(i)
        
        # 如果没有足够的类别变化点，使用动态策略
        if len(candidates) < self.min_n_bins * 2:
            if n <= 100:
                step = 1
            elif n <= 1000:
                step = max(1, n // 100)
            else:
                step = max(1, n // 200)
            
            additional = list(range(self.min_samples_leaf, n - self.min_samples_leaf, step))
            candidates = sorted(list(set(candidates + additional)))
        
        return candidates

    def _recurse_v3(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: List[float],
        all_candidates: List[int],
        depth: int = 0,
        start_idx: int = 0
    ) -> None:
        """递归分割 - V3版本.

        :param x: 特征值数组（已排序）
        :param y: 目标变量数组
        :param splits: 切分点列表（原地修改）
        :param all_candidates: 全局候选切分点位置
        :param depth: 当前递归深度
        :param start_idx: 当前区间在全局数组中的起始索引
        """
        # 检查最大分箱数限制
        if len(splits) >= self.max_n_bins - 1:
            return

        # 获取唯一特征值数量和目标类别数
        u_x = np.unique(x)
        n_x = len(u_x)
        n_y = len(np.bincount(y))

        # 基本可分割性检查
        if n_x < self.min_samples_split or n_y < 2 or len(x) < self.min_samples_split:
            return

        # 在当前区间内筛选候选切分点
        local_candidates = [i - start_idx for i in all_candidates 
                           if start_idx + self.min_samples_leaf <= i < start_idx + len(x) - self.min_samples_leaf]
        
        local_candidates = sorted(list(set(local_candidates)))
        
        # 限制候选切分点数量
        if len(local_candidates) > self.max_candidates:
            indices = np.linspace(0, len(local_candidates) - 1, self.max_candidates, dtype=int)
            local_candidates = [local_candidates[i] for i in indices]

        # 找到最优切分点
        split, split_idx = self._find_best_split_v3(x, y, local_candidates)

        if split is not None and split_idx is not None:
            y_left, y_right = y[:split_idx], y[split_idx:]

            # 检查终止条件（V3：放宽条件）
            if not self._terminate_v3(len(x), y, y_left, y_right):
                splits.append(split)

                if len(splits) >= self.max_n_bins - 1:
                    return

                # 递归处理左右子区间
                x_left, x_right = x[:split_idx], x[split_idx:]
                if len(x_left) >= self.min_samples_split:
                    self._recurse_v3(x_left, y_left, splits, all_candidates, depth + 1, start_idx)
                if len(x_right) >= self.min_samples_split:
                    self._recurse_v3(x_right, y_right, splits, all_candidates, depth + 1, start_idx + split_idx)

    def _find_best_split_v3(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        candidates: List[int]
    ) -> tuple:
        """找到最优切分点 - V3版本.

        结合IV增益和MDLP准则。

        :param x: 特征值数组（已排序）
        :param y: 目标变量数组
        :param candidates: 候选切分点位置列表
        :return: (最优切分点值, 切分点位置) 或 (None, None)
        """
        if not candidates:
            return None, None

        n = len(x)
        max_score = -np.inf
        best_split = None
        best_idx = None

        total_bad = y.sum()
        total_good = n - total_bad

        for idx in candidates:
            if idx < self.min_samples_leaf or n - idx < self.min_samples_leaf:
                continue

            split = (x[idx - 1] + x[idx]) / 2

            y_left, y_right = y[:idx], y[idx:]

            if len(np.unique(y_left)) < 1 or len(np.unique(y_right)) < 1:
                continue

            # 计算IV增益
            iv_gain = self._calculate_iv_gain_v3(y_left, y_right, total_bad, total_good)
            
            # 如果IV增益太小，跳过
            if iv_gain < self.min_iv_gain:
                continue

            if iv_gain > max_score:
                max_score = iv_gain
                best_split = split
                best_idx = idx

        return best_split, best_idx

    def _calculate_iv_gain_v3(
        self,
        y_left: np.ndarray,
        y_right: np.ndarray,
        total_bad: int,
        total_good: int
    ) -> float:
        """计算IV增益 - V3版本."""
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

    def _terminate_v3(
        self,
        n: int,
        y: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> bool:
        """MDLP 终止条件 - V3版本（放宽条件）.

        :param n: 当前节点样本数
        :param y: 分割前的目标变量
        :param y_left: 左子区间的目标变量
        :param y_right: 右子区间的目标变量
        :return: 是否应该终止分割
        """
        n_left = len(y_left)
        n_right = len(y_right)

        # 计算信息增益
        gain = self._entropy_gain(y, y_left, y_right)

        # 获取类别数量
        k = len(np.bincount(y))
        k_left = len(np.bincount(y_left))
        k_right = len(np.bincount(y_right))

        # 计算熵
        ent_y = self._entropy(y)
        ent_left = self._entropy(y_left)
        ent_right = self._entropy(y_right)

        # MDLP 准则计算
        delta = np.log(3**k - 2) - (k * ent_y - k_left * ent_left - k_right * ent_right)
        threshold = (np.log(n - 1) + delta) / n

        # V3：使用权重调整终止条件
        adjusted_threshold = threshold * self.mdlp_weight

        return gain <= adjusted_threshold

    def _force_additional_splits_v3(
        self,
        x: np.ndarray,
        y: np.ndarray,
        existing_splits: List[float],
        all_candidates: List[int]
    ) -> List[float]:
        """强制添加额外的切分点以满足分箱数要求 - V3改进版（目标是max_n_bins）."""
        splits = list(existing_splits)
        n = len(x)
        
        total_bad = y.sum()
        total_good = n - total_bad
        min_samples = self._get_min_samples(n)
        
        # 目标是达到max_n_bins个分箱（而不仅仅是min_n_bins）
        target_n_bins = self.max_n_bins - 1
        
        # 首先找到所有可能的候选切分点及其IV
        candidates_with_iv = []
        for idx in all_candidates:
            if idx < min_samples or n - idx < min_samples:
                continue
            
            y_left = y[:idx]
            y_right = y[idx:]
            
            if len(np.unique(y_left)) < 1 or len(np.unique(y_right)) < 1:
                continue
            
            iv = self._calculate_iv_gain_v3(y_left, y_right, total_bad, total_good)
            split = (x[idx-1] + x[idx]) / 2
            candidates_with_iv.append((split, idx, iv))
        
        # 按IV排序
        candidates_with_iv.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心选择：每次选择能最大化总IV的切分点
        while len(splits) < target_n_bins:
            best_total_iv = -np.inf
            best_split = None
            
            for split, idx, single_iv in candidates_with_iv:
                if split in splits:
                    continue
                
                # 测试添加这个切分点后的总IV
                test_splits = sorted(splits + [split])
                bins = np.digitize(x, test_splits)
                total_iv = 0
                epsilon = 1e-10
                
                for b in range(len(test_splits) + 1):
                    mask = bins == b
                    if mask.sum() == 0:
                        continue
                    bad = y[mask].sum()
                    good = mask.sum() - bad
                    if bad > 0 and good > 0:
                        bad_rate = bad / total_bad
                        good_rate = good / total_good
                        woe = np.log((good_rate + epsilon) / (bad_rate + epsilon))
                        total_iv += (good_rate - bad_rate) * woe
                
                if total_iv > best_total_iv:
                    best_total_iv = total_iv
                    best_split = split
            
            if best_split is not None:
                splits.append(best_split)
            else:
                # 如果没有找到好的切分点，使用等间距
                if len(splits) < target_n_bins:
                    x_min, x_max = x.min(), x.max()
                    additional = np.linspace(x_min, x_max, self.max_n_bins + 1)[1:-1]
                    for s in additional:
                        if s not in splits:
                            splits.append(s)
                            if len(splits) >= target_n_bins:
                                break
                break
        
        return sorted(list(set(splits)))

    def _entropy(self, y: np.ndarray) -> float:
        """计算熵.

        :param y: 目标变量数组
        :return: 熵值
        """
        n = len(y)
        if n == 0:
            return 0.0

        n_pos = np.sum(y)
        n_neg = n - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.0

        p = np.array([n_neg, n_pos]) / n
        return -np.sum(special.xlogy(p, p))

    def _entropy_gain(
        self,
        y: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """计算信息增益.

        :param y: 分割前的目标变量
        :param y_left: 左子区间的目标变量
        :param y_right: 右子区间的目标变量
        :return: 信息增益
        """
        n = len(y)
        n_left = len(y_left)
        n_right = n - n_left

        if n_left == 0 or n_right == 0:
            return 0.0

        ent_y = self._entropy(y)
        ent_left = self._entropy(y_left)
        ent_right = self._entropy(y_right)

        return ent_y - (n_left * ent_left + n_right * ent_right) / n

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数."""
        # 使用 min_samples_leaf 或根据数据量计算
        return max(self.min_samples_leaf, int(n_total * 0.01))

    def _apply_splits(
        self,
        x: pd.Series,
        splits,
        feature_type: str
    ) -> np.ndarray:
        """应用切分点.

        :param x: 特征数据
        :param splits: 切分点数组（数值型为np.ndarray, 类别型为list）
        :param feature_type: 特征类型
        :return: 分箱标签
        """
        if feature_type == 'categorical':
            # 基于 splits 列表中的顺序映射类别到分箱索引
            x_str = x.astype(str).where(x.notna(), other=np.nan)
            cat_to_bin = {str(cat): i for i, cat in enumerate(splits)}
            bins = np.full(len(x), 0, dtype=int)
            bins[x.isna()] = -1
            for cat_str, bin_idx in cat_to_bin.items():
                bins[x_str == cat_str] = bin_idx
            if self.special_codes:
                for code in self.special_codes:
                    bins[(x == code) | (x_str == str(code))] = -2
        else:
            if len(splits) == 0:
                bins = np.zeros(len(x), dtype=int)
            else:
                bins = np.digitize(x.values, splits, right=False)
                bins = np.where(x.isna(), -1, bins)

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
        >>> binner = MDLPBinning()
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
                result[feature] = X[feature]
                continue
            splits = self.splits_[feature]
            feature_type = self.feature_types_[feature]
            bins = self._apply_splits(X[feature], splits, feature_type)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels(splits, bins)
                result[feature] = [labels[b] if b >= 0 else 'missing' for b in bins]
            elif metric == 'woe':
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = {i: bin_table.iloc[i]['分档WOE值'] for i in range(len(bin_table))}
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = [woe_map.get(b, 0) for b in bins]
            else:
                raise ValueError(f"不支持的metric: {metric}")

        return result if isinstance(X, pd.DataFrame) else result.values
