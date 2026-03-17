"""决策树分箱算法.

使用 sklearn DecisionTreeClassifier 提取最优分割点的分箱方法。
基于信息增益选择切分点，支持单调性约束。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .base import BaseBinning


class TreeBinning(BaseBinning):
    """决策树分箱算法.

    使用决策树提取最优分割点进行分箱，基于信息增益选择切分点。
    支持最大深度、叶子节点数限制和单调性约束。

    :param max_depth: 决策树最大深度，默认为5
    :param max_leaf_nodes: 最大叶子节点数，默认为None
    :param min_samples_leaf: 叶子节点最小样本数，默认为0.05
    :param min_n_bins: 最小分箱数，默认为2
    :param max_n_bins: 最大分箱数，默认为10
    :param force_numerical: 是否强制作为数值型处理，默认为False
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求坏样本率单调，默认为False
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表
    - tree_models_: 每个特征的决策树模型

    **示例**

    >>> from hscredit.core.binning import TreeBinning
    >>> binner = TreeBinning(max_depth=5, monotonic=True)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """

    def __init__(
        self,
        max_depth: int = 5,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: Union[float, int] = 0.05,
        min_n_bins: int = 2,
        max_n_bins: int = 10,
        force_numerical: bool = False,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ):
        super().__init__(
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
        )
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.force_numerical = force_numerical
        self.tree_models_ = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'TreeBinning':
        """拟合决策树分箱.

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
                # 类别型特征：按坏样本率排序
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                # 数值型特征：决策树分箱
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

            # 计算分箱统计信息
            bins = self._apply_bins(X[feature], splits)
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
        """对数值型特征进行决策树分箱.

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

        # 计算最小样本数
        if self.min_samples_leaf < 1:
            min_samples_leaf = int(len(x_valid) * self.min_samples_leaf)
        else:
            min_samples_leaf = int(self.min_samples_leaf)

        # 计算最大叶子节点数
        max_leaf_nodes = self.max_leaf_nodes
        if max_leaf_nodes is None:
            max_leaf_nodes = self.max_n_bins

        # 训练决策树
        X_reshaped = x_valid.values.reshape(-1, 1)

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )
        tree.fit(X_reshaped, y_valid)

        # 保存模型
        self.tree_models_[x.name if hasattr(x, 'name') else 'feature'] = tree

        # 提取切分点
        splits = self._extract_splits_from_tree(tree, x_valid)

        # 应用单调性约束
        if self.monotonic:
            splits = self._apply_monotonic_constraint(x_valid, y_valid, splits)

        # 根据约束调整分箱数
        splits = self._adjust_bins(x_valid, y_valid, splits)

        return splits

    def _extract_splits_from_tree(
        self,
        tree: DecisionTreeClassifier,
        x: pd.Series
    ) -> np.ndarray:
        """从决策树中提取切分点.

        :param tree: 决策树模型
        :param x: 特征数据
        :return: 切分点数组
        """
        tree_ = tree.tree_

        # 收集所有内部节点的切分点
        splits = []

        def traverse(node_id=0):
            # 使用 _tree.TREE_UNDEFINED 常量
            from sklearn.tree._tree import TREE_UNDEFINED
            if tree_.feature[node_id] != TREE_UNDEFINED:
                # 内部节点
                threshold = tree_.threshold[node_id]
                splits.append(threshold)
                # 递归遍历左右子树
                traverse(tree_.children_left[node_id])
                traverse(tree_.children_right[node_id])

        traverse()

        # 排序并去重
        splits = np.unique(splits)

        return splits

    def _apply_monotonic_constraint(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray
    ) -> np.ndarray:
        """应用单调性约束.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        # 计算每个箱的坏样本率
        bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)

        bin_stats = pd.DataFrame({
            'bin': bins,
            'target': y
        }).groupby('bin')['target'].mean()
        
        # 确保所有分箱都在bin_stats中
        bin_stats = self._ensure_all_bins_in_series(bin_stats, len(splits) + 1)

        # 检查单调性
        is_monotonic_increasing = all(
            bin_stats.iloc[i] <= bin_stats.iloc[i + 1]
            for i in range(len(bin_stats) - 1)
        )
        is_monotonic_decreasing = all(
            bin_stats.iloc[i] >= bin_stats.iloc[i + 1]
            for i in range(len(bin_stats) - 1)
        )

        if self.monotonic == 'ascending' and not is_monotonic_increasing:
            # 强制递增：合并违反单调性的相邻箱
            splits = self._merge_for_monotonicity(
                x, y, splits, increasing=True
            )
        elif self.monotonic == 'descending' and not is_monotonic_decreasing:
            # 强制递减：合并违反单调性的相邻箱
            splits = self._merge_for_monotonicity(
                x, y, splits, increasing=False
            )
        elif self.monotonic is True or self.monotonic == 'auto':
            # 自动判断方向
            if not is_monotonic_increasing and not is_monotonic_decreasing:
                # 既不递增也不递减，选择更接近的方向
                inc_violations = sum(
                    1 for i in range(len(bin_stats) - 1)
                    if bin_stats.iloc[i] > bin_stats.iloc[i + 1]
                )
                dec_violations = sum(
                    1 for i in range(len(bin_stats) - 1)
                    if bin_stats.iloc[i] < bin_stats.iloc[i + 1]
                )

                if inc_violations <= dec_violations:
                    splits = self._merge_for_monotonicity(
                        x, y, splits, increasing=True
                    )
                else:
                    splits = self._merge_for_monotonicity(
                        x, y, splits, increasing=False
                    )

        return splits

    def _merge_for_monotonicity(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        increasing: bool
    ) -> np.ndarray:
        """合并箱以满足单调性约束.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :param increasing: 是否要求递增
        :return: 调整后的切分点
        """
        if len(splits) <= 1:
            return splits

        max_iter = len(splits)
        for _ in range(max_iter):
            bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)

            bin_stats = pd.DataFrame({
                'bin': bins,
                'target': y
            }).groupby('bin')['target'].mean()
            
            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_series(bin_stats, len(splits) + 1)

            # 找到违反单调性的位置
            violations = []
            for i in range(len(bin_stats) - 1):
                if increasing:
                    if bin_stats.iloc[i] > bin_stats.iloc[i + 1]:
                        violations.append(i)
                else:
                    if bin_stats.iloc[i] < bin_stats.iloc[i + 1]:
                        violations.append(i)

            if not violations:
                break

            # 合并第一个违反单调性的箱
            merge_idx = violations[0]
            new_splits = np.delete(splits, merge_idx)

            if len(new_splits) < self.min_n_bins - 1:
                break

            splits = new_splits

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

        # 如果要求单调性，类别型特征已经按坏样本率排序
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
        min_samples = self._get_min_samples(len(x))

        # 迭代调整直到满足约束
        max_iter = 20
        for _ in range(max_iter):
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

                count = bin_counts.get(i, 0)

                if count < min_samples and i < len(splits) - 1:
                    skip_next = True
                else:
                    new_splits.append(splits[i])

            new_splits = np.array(new_splits)

            if len(new_splits) == len(splits):
                break

            splits = new_splits

            n_bins = len(splits) + 1
            if n_bins < self.min_n_bins:
                break

        # 最终检查分箱数约束
        n_bins = len(splits) + 1
        if n_bins > self.max_n_bins:
            # 减少分箱数
            n_remove = n_bins - self.max_n_bins
            splits = splits[n_remove:]
        elif n_bins < self.min_n_bins:
            # 增加分箱数：使用等频分箱
            quantiles = np.linspace(0, 1, self.min_n_bins + 1)
            splits = np.percentile(x, quantiles[1:-1] * 100)

        return splits

    def _ensure_all_bins_in_series(
        self,
        bin_stats: pd.Series,
        n_bins: int
    ) -> pd.Series:
        """确保bin_stats包含所有分箱（即使某些分箱为空）.
        
        :param bin_stats: 分箱统计Series (索引为bin标签)
        :param n_bins: 分箱数量
        :return: 补全后的分箱统计Series
        """
        expected_bins = list(range(n_bins))
        for bin_idx in expected_bins:
            if bin_idx not in bin_stats.index:
                bin_stats[bin_idx] = 0.0
        
        return bin_stats.sort_index()

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
            bins[x.isna()] = -1
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2
            return bins
        else:
            # 数值型特征
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

            bins[mask] = np.digitize(x[mask], splits)

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
        >>> binner = TreeBinning()
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
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=['feature'])
                else:
                    X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                raise KeyError(f"特征 '{feature}' 未在训练数据中找到")

            splits = self.splits_[feature]
            bins = self._apply_bins(X[feature], splits)

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


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 生成测试数据（具有单调关系）
    x1 = np.random.randn(n_samples)
    x2 = np.random.uniform(0, 100, n_samples)
    # 创建与目标变量有单调关系的特征
    y_prob = 1 / (1 + np.exp(-(x1 * 0.5 + x2 * 0.02 - 2)))
    y = pd.Series(np.random.binomial(1, y_prob, n_samples))

    X = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    })

    # 添加一些缺失值
    X.loc[np.random.choice(n_samples, 50, replace=False), 'feature1'] = np.nan

    print("=" * 50)
    print("决策树分箱测试")
    print("=" * 50)

    # 测试无单调性约束
    print("\n1. 无单调性约束:")
    binner1 = TreeBinning(max_depth=5, verbose=True)
    binner1.fit(X, y)
    print("\n分箱统计表 (feature1):")
    print(binner1.get_bin_table('feature1'))

    # 测试单调递增
    print("\n2. 单调递增约束:")
    binner2 = TreeBinning(max_depth=5, monotonic='ascending', verbose=True)
    binner2.fit(X, y)
    print("\n分箱统计表 (feature1):")
    print(binner2.get_bin_table('feature1'))

    # 测试单调递减
    print("\n3. 单调递减约束:")
    binner3 = TreeBinning(max_depth=5, monotonic='descending', verbose=True)
    binner3.fit(X, y)
    print("\n分箱统计表 (feature1):")
    print(binner3.get_bin_table('feature1'))

    # 转换测试
    print("\n4. 转换测试:")
    X_binned = binner2.transform(X, metric='indices')
    print("\n分箱索引:")
    print(X_binned.head())

    X_woe = binner2.transform(X, metric='woe')
    print("\nWOE值:")
    print(X_woe.head())

    print("\n切分点:")
    for feature, splits in binner2.splits_.items():
        print(f"  {feature}: {splits}")
