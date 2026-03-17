"""MDLP (Minimum Description Length Principle) 分箱.

基于信息论的递归分箱算法，由 Fayyad 和 Irani 于 1993 年提出。
使用 MDLP 准则作为停止条件，自动确定最优分箱数。

参考 optbinning 实现修复。
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

    参考 optbinning 实现，修复分箱数过少的问题。

    :param target: 目标变量列名，默认为'target'
    :param max_n_bins: 最大分箱数，默认为10
    :param min_samples_split: 分割内部节点所需的最小样本数，默认为2
    :param min_samples_leaf: 叶子节点所需的最小样本数，默认为2
    :param max_candidates: 每次评估的最大候选切分点数，默认为32

    示例:
        >>> from hscredit.core.binning import MDLPBinning
        >>> binner = MDLPBinning(max_n_bins=5)
        >>> binner.fit(X_train, y_train)
        >>> X_binned = binner.transform(X_test)
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        max_candidates: int = 32,
        **kwargs
    ):
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            **kwargs
        )
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates

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
            # MDLP 只适用于数值型特征，强制转换为数值型处理
            x_numeric = pd.to_numeric(X[feature], errors='coerce')
            self.feature_types_[feature] = 'numerical'

            # 数值型特征：使用 MDLP 算法
            x_clean = x_numeric.dropna()
            y_clean = y[x_numeric.notna()]

            if len(x_clean) >= self.min_samples_split:
                splits = self._mdlp_split(x_clean.values, y_clean.values)
                self.splits_[feature] = self._round_splits(np.sort(splits))
                self.n_bins_[feature] = len(splits) + 1
            else:
                self.splits_[feature] = np.array([])
                self.n_bins_[feature] = 1

            # 计算分箱统计
            bins = self._apply_splits(X[feature], self.splits_[feature], 'numerical')
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

        self._is_fitted = True
        return self

    def _mdlp_split(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """MDLP 递归分箱算法.

        参考 optbinning 实现，先检查终止条件再添加分割点。

        :param x: 特征值数组
        :param y: 目标变量数组
        :return: 切分点列表
        """
        # 按特征值排序
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]

        splits = []
        self._recurse(x_sorted, y_sorted, splits)
        return splits

    def _recurse(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: List[float],
        depth: int = 0
    ) -> None:
        """递归分割.

        关键修复：先找到分割点，检查终止条件，通过后才添加并递归。

        :param x: 特征值数组
        :param y: 目标变量数组
        :param splits: 切分点列表（原地修改）
        :param depth: 当前递归深度
        """
        # 检查最大分箱数限制
        if len(splits) >= self.max_n_bins - 1:
            return

        # 获取唯一特征值数量和目标类别数
        u_x = np.unique(x)
        n_x = len(u_x)
        n_y = len(np.bincount(y))

        # 基本可分割性检查
        if n_x < self.min_samples_split or n_y < 2:
            return

        # 找到最优切分点
        split = self._find_split(x, y)

        if split is not None:
            # 计算分割位置
            t = np.searchsorted(x, split, side='right')
            y_left, y_right = y[:t], y[t:]

            # 检查终止条件 - 关键修复：在添加分割点前检查
            if not self._terminate(n_x, n_y, y, y_left, y_right):
                # 通过终止条件检查后才添加分割点
                splits.append(split)
                
                # 立即检查是否达到最大分箱数限制
                if len(splits) >= self.max_n_bins - 1:
                    return

                # 递归处理左右子区间
                x_left, x_right = x[:t], x[t:]
                if len(x_left) >= self.min_samples_split:
                    self._recurse(x_left, y_left, splits, depth + 1)
                if len(x_right) >= self.min_samples_split:
                    self._recurse(x_right, y_right, splits, depth + 1)

    def _find_split(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """找到最优切分点.

        改进：在所有类别变化的位置考虑切分，不仅限于相邻不同类别。

        :param x: 特征值数组（已排序）
        :param y: 目标变量数组
        :return: 最优切分点或 None
        """
        n = len(x)

        # 获取唯一值（已排序）
        u_x = np.unique(x)

        # 如果唯一值太少，无法分割
        if len(u_x) < 2:
            return None

        # 候选切分点：相邻唯一值的中点
        # 这样比遍历所有样本点更高效
        candidates = 0.5 * (u_x[1:] + u_x[:-1])

        # 限制候选切分点数量
        if len(candidates) > self.max_candidates:
            indices = np.linspace(0, len(candidates) - 1, self.max_candidates, dtype=int)
            candidates = candidates[indices]

        max_gain = -np.inf
        best_split = None

        # 评估每个候选切分点
        for split in candidates:
            t = np.searchsorted(x, split, side='right')

            # 检查最小叶子样本数
            if t < self.min_samples_leaf or n - t < self.min_samples_leaf:
                continue

            y_left, y_right = y[:t], y[t:]

            # 检查叶子节点是否有足够的类别多样性
            if len(np.unique(y_left)) < 1 or len(np.unique(y_right)) < 1:
                continue

            gain = self._entropy_gain(y, y_left, y_right)

            if gain > max_gain:
                max_gain = gain
                best_split = split

        # 只有当增益为正时才返回分割点
        if max_gain > 0:
            return best_split
        return None

    def _entropy(self, y: np.ndarray) -> float:
        """计算熵.

        :param y: 目标变量数组
        :return: 熵值
        """
        n = len(y)
        if n == 0:
            return 0.0

        # 计算正负样本数
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

    def _terminate(
        self,
        n_x: int,
        n_y: int,
        y: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> bool:
        """MDLP 终止条件.

        参考 optbinning 实现，确保正确的终止条件判断。

        :param n_x: 唯一特征值数量
        :param n_y: 唯一目标值数量
        :param y: 分割前的目标变量
        :param y_left: 左子区间的目标变量
        :param y_right: 右子区间的目标变量
        :return: 是否应该终止分割
        """
        n = len(y)
        n_left = len(y_left)
        n_right = n - n_left

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
        # delta = log(3^k - 2) - [k*Ent(S) - k1*Ent(S1) - k2*Ent(S2)]
        delta = np.log(3**k - 2) - (k * ent_y - k_left * ent_left - k_right * ent_right)

        # MDLP 阈值
        threshold = (np.log(n - 1) + delta) / n

        # 终止条件：信息增益小于阈值
        return gain <= threshold

    def _apply_splits(
        self,
        x: pd.Series,
        splits: np.ndarray,
        feature_type: str
    ) -> np.ndarray:
        """应用切分点.

        :param x: 特征数据
        :param splits: 切分点数组
        :param feature_type: 特征类型
        :return: 分箱标签
        """
        if feature_type == 'categorical':
            # 类别型特征：每个类别一箱
            cat_to_bin = {cat: i for i, cat in enumerate(x.dropna().unique())}
            bins = x.map(lambda v: cat_to_bin.get(v, -1)).values
        else:
            # 数值型特征：使用切分点
            if len(splits) == 0:
                bins = np.zeros(len(x), dtype=int)
            else:
                bins = np.digitize(x.values, splits, right=False)
                # 处理缺失值
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
        :param kwargs: 其他参数(保留兼容性)
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
            splits = self.splits_[feature]
            feature_type = self.feature_types_[feature]
            bins = self._apply_splits(X[feature], splits, feature_type)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels(splits, bins)
                result[feature] = [labels[b] if b >= 0 else 'missing' for b in bins]
            elif metric == 'woe':
                bin_table = self.bin_tables_[feature]
                woe_map = {i: bin_table.iloc[i]['分档WOE值'] for i in range(len(bin_table))}
                result[feature] = [woe_map.get(b, 0) for b in bins]
            else:
                raise ValueError(f"不支持的metric: {metric}")

        return result if isinstance(X, pd.DataFrame) else result.values
