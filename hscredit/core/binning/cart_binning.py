"""CART 分箱算法.

参考 optbinning 的 CART 预分箱方法实现，使用决策树提取最优分割点。
支持分类和回归目标变量，提供更灵活的约束控制。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree

from .base import BaseBinning


class CartBinning(BaseBinning):
    """CART 分箱算法.

    基于决策树的分箱方法，参考 optbinning 的 CART 预分箱实现。
    使用 sklearn 的 DecisionTreeClassifier/Regressor 提取最优分割点。

    :param max_n_bins: 最大分箱数，默认为10
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_event_rate_diff: 相邻箱最小坏样本率差异，默认为0
        - 如果设置，会合并坏样本率差异过小的相邻箱
    :param max_pvalue: 最大 p-value 阈值，默认为None
        - 如果设置，会进行统计检验，合并差异不显著的相邻箱
    :param max_pvalue_policy: p-value 检验策略，默认为"consecutive"
        - "all": 检验所有箱对
        - "consecutive": 只检验相邻箱
    :param monotonic: 是否要求单调性，默认为False
        - False: 不要求
        - True 或 'auto': 自动判断单调方向
        - 'ascending': 强制递增
        - 'descending': 强制递减
    :param class_weight: 类别权重，默认为None
        - None: 不调整
        - "balanced": 自动根据类别频率调整
        - dict: 自定义权重，如 {0: 1, 1: 2}
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param random_state: 随机种子，默认为None

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表
    - tree_models_: 每个特征的决策树模型

    **示例**

    >>> from hscredit.core.binning import CartBinning
    >>> binner = CartBinning(max_n_bins=5, monotonic=True)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    **注意**

    CART 分箱的特点:
    1. 基于决策树的信息增益选择切分点
    2. 支持分类（二分类/多分类）和回归目标变量
    3. 可设置类别权重处理不平衡数据
    4. 支持 p-value 检验确保分箱统计显著性
    5. 支持单调性约束
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 10,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_event_rate_diff: float = 0.0,
        max_pvalue: Optional[float] = None,
        max_pvalue_policy: str = "consecutive",
        monotonic: Union[bool, str] = False,
        class_weight: Optional[Union[str, Dict]] = None,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
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
            **kwargs
        )
        self.min_event_rate_diff = min_event_rate_diff
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.class_weight = class_weight
        self.tree_models_ = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'CartBinning':
        """拟合 CART 分箱.

        :param X: 训练数据，shape (n_samples, n_features)
        :param y: 目标变量，可以是二分类、多分类或连续型
        :param kwargs: 其他参数
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 判断问题类型
        self.problem_type_ = self._detect_problem_type(y)

        # 对每个特征进行分箱
        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            # 检测特征类型
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                # 类别型特征
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                # 数值型特征
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

            # 计算分箱统计信息
            bins = self._assign_bins(X[feature], feature)
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._is_fitted = True
        return self

    def _detect_problem_type(self, y: pd.Series) -> str:
        """检测问题类型.

        :param y: 目标变量
        :return: "binary", "multiclass", 或 "regression"
        """
        unique_values = y.nunique()

        if unique_values == 2:
            return "binary"
        elif y.dtype in ['int64', 'int32'] and unique_values <= 10:
            return "multiclass"
        elif y.dtype in ['float64', 'float32'] and unique_values > 10:
            return "regression"
        else:
            # 默认二分类
            return "binary"

    def _fit_numerical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> np.ndarray:
        """对数值型特征进行 CART 分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组
        """
        # 处理缺失值和特殊值
        mask = x.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x != code)

        x_valid = x[mask].values.reshape(-1, 1)
        y_valid = y[mask].values

        if len(x_valid) == 0:
            return np.array([])

        n_samples = len(x_valid)

        # 计算最小样本数
        if self.min_bin_size < 1:
            min_samples_leaf = max(int(n_samples * self.min_bin_size), 1)
        else:
            min_samples_leaf = int(self.min_bin_size)

        # 训练决策树：先做更细的内部预分箱，再通过约束合并到目标箱数
        tree_leaf_nodes = max(self.max_n_bins * 4, 20)
        if self.problem_type_ == "regression":
            tree = DecisionTreeRegressor(
                max_leaf_nodes=tree_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state
            )
        else:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=tree_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                class_weight=self.class_weight,
                random_state=self.random_state
            )

        tree.fit(x_valid, y_valid)

        # 保存模型
        feature_name = x.name if hasattr(x, 'name') else 'feature'
        self.tree_models_[feature_name] = tree

        # 提取切分点
        splits = self._extract_splits_from_tree(tree)

        # 应用额外约束
        if len(splits) > 0:
            # 单调性约束
            if self.monotonic:
                splits = self._apply_monotonic_constraint(
                    x_valid.flatten(), y_valid, splits
                )

            # 最小事件率差异约束
            if self.min_event_rate_diff > 0:
                splits = self._apply_event_rate_constraint(
                    x_valid.flatten(), y_valid, splits
                )

            # p-value 约束
            if self.max_pvalue is not None:
                splits = self._apply_pvalue_constraint(
                    x_valid.flatten(), y_valid, splits
                )

            # 样本数约束
            splits = self._apply_bin_size_constraint(
                x_valid.flatten(), y_valid, splits
            )

        return splits

    def _extract_splits_from_tree(
        self,
        tree: Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ) -> np.ndarray:
        """从决策树中提取切分点.

        参考 optbinning 的实现，提取所有内部节点的 threshold。

        :param tree: 决策树模型
        :return: 切分点数组
        """
        tree_ = tree.tree_

        # 收集所有内部节点的切分点
        splits = []

        def traverse(node_id=0):
            if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
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

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行 CART 分箱.

        按坏样本率排序，然后进行分组。

        :param x: 特征数据
        :param y: 目标变量
        :return: 类别列表（按坏样本率排序）
        """
        # 处理缺失值和特殊值
        mask = x.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x != code)

        x_valid = x[mask]
        y_valid = y[mask]

        # 计算每个类别的目标均值（坏样本率或回归目标均值）
        df = pd.DataFrame({'category': x_valid, 'target': y_valid})

        if self.problem_type_ == "regression":
            cat_stats = df.groupby('category')['target'].agg(['mean', 'count'])
        else:
            cat_stats = df.groupby('category')['target'].agg(['mean', 'count'])

        # 过滤掉样本数过少的类别
        n_samples = len(x_valid)
        if self.min_bin_size < 1:
            min_samples = max(int(n_samples * self.min_bin_size), 1)
        else:
            min_samples = int(self.min_bin_size)

        cat_stats = cat_stats[cat_stats['count'] >= min_samples]

        if len(cat_stats) <= self.min_n_bins:
            return cat_stats.index.tolist()

        # 按目标均值排序
        cat_stats = cat_stats.sort_values('mean')

        # 如果类别数超过最大分箱数，需要合并
        if len(cat_stats) > self.max_n_bins:
            # 按目标均值进行分组
            n_groups = self.max_n_bins
            cat_stats['group'] = pd.qcut(
                cat_stats['mean'], 
                q=n_groups, 
                labels=False, 
                duplicates='drop'
            )

            # 返回排序后的类别列表
            return cat_stats.index.tolist()

        return cat_stats.index.tolist()

    def _apply_monotonic_constraint(
        self,
        x: np.ndarray,
        y: np.ndarray,
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

        splits = list(splits)
        max_iter = len(splits)

        for _ in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算每箱的均值
            bins = np.searchsorted(splits, x, side='right')

            df = pd.DataFrame({'bin': bins, 'target': y})
            bin_means = df.groupby('bin')['target'].mean().values

            if len(bin_means) < 2:
                break

            # 检查单调性
            is_ascending = all(
                bin_means[i] <= bin_means[i + 1]
                for i in range(len(bin_means) - 1)
            )
            is_descending = all(
                bin_means[i] >= bin_means[i + 1]
                for i in range(len(bin_means) - 1)
            )

            # 确定期望的单调方向
            if self.monotonic == 'ascending':
                target_ascending = True
            elif self.monotonic == 'descending':
                target_ascending = False
            elif self.monotonic == 'auto_asc_desc':
                corr = pd.Series(x).corr(pd.Series(y), method='spearman')
                if pd.notna(corr) and abs(corr) >= 0.02:
                    target_ascending = bool(corr > 0)
                else:
                    target_ascending = bool(bin_means[-1] >= bin_means[0])
            else:  # auto / True
                target_ascending = is_ascending or not is_descending

            # 检查是否满足单调性
            if target_ascending and is_ascending:
                break
            if not target_ascending and is_descending:
                break

            # 找到违反单调性的位置并合并
            violations = []
            for i in range(len(bin_means) - 1):
                if target_ascending:
                    if bin_means[i] > bin_means[i + 1]:
                        violations.append(i)
                else:
                    if bin_means[i] < bin_means[i + 1]:
                        violations.append(i)

            if not violations:
                break

            # 合并第一个违反单调性的箱
            merge_idx = violations[0]
            if merge_idx < len(splits):
                splits.pop(merge_idx)
            else:
                splits.pop()

        return np.array(splits)

    def _apply_event_rate_constraint(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: np.ndarray
    ) -> np.ndarray:
        """应用最小事件率差异约束.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0 or self.min_event_rate_diff <= 0:
            return splits

        splits = list(splits)
        max_iter = len(splits)

        for _ in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算每箱的事件率
            bins = np.searchsorted(splits, x, side='right')

            df = pd.DataFrame({'bin': bins, 'target': y})
            bin_stats = df.groupby('bin')['target'].agg(['mean', 'count'])

            if len(bin_stats) < 2:
                break

            # 检查相邻箱的事件率差异
            rates = bin_stats['mean'].values
            counts = bin_stats['count'].values

            min_diff = float('inf')
            merge_idx = -1

            for i in range(len(rates) - 1):
                diff = abs(rates[i] - rates[i + 1])
                if diff < min_diff and diff < self.min_event_rate_diff:
                    min_diff = diff
                    # 选择样本数较少的箱进行合并
                    if counts[i] < counts[i + 1]:
                        merge_idx = i
                    else:
                        merge_idx = i + 1

            if merge_idx == -1:
                break

            # 合并箱
            if merge_idx < len(splits):
                splits.pop(merge_idx)
            else:
                splits.pop()

        return np.array(splits)

    def _apply_pvalue_constraint(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: np.ndarray
    ) -> np.ndarray:
        """应用 p-value 约束.

        使用统计检验确保分箱之间的差异是显著的。

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0 or self.max_pvalue is None:
            return splits

        splits = list(splits)
        max_iter = len(splits)

        for _ in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算每箱的统计信息
            bins = np.searchsorted(splits, x, side='right')

            df = pd.DataFrame({'bin': bins, 'target': y})

            if self.problem_type_ == "regression":
                # 回归问题使用 t 检验
                bin_groups = df.groupby('bin')['target'].apply(list).values
            else:
                # 分类问题使用卡方检验
                bin_crosstab = pd.crosstab(df['bin'], df['target'])

            # 检查相邻箱的显著性
            n_bins = len(np.unique(bins))

            if self.max_pvalue_policy == "consecutive":
                # 只检查相邻箱
                check_pairs = [(i, i + 1) for i in range(n_bins - 1)]
            else:
                # 检查所有箱对
                check_pairs = [(i, j) for i in range(n_bins) for j in range(i + 1, n_bins)]

            max_p = 0
            merge_idx = -1

            for i, j in check_pairs:
                if self.problem_type_ == "regression":
                    if i < len(bin_groups) and j < len(bin_groups):
                        _, p_value = stats.ttest_ind(
                            bin_groups[i], bin_groups[j], equal_var=False
                        )
                else:
                    if i < len(bin_crosstab) and j < len(bin_crosstab):
                        # 构造 2x2 列联表
                        contingency = bin_crosstab.iloc[[i, j]].values
                        if contingency.sum() > 0:
                            _, p_value, _, _ = stats.chi2_contingency(contingency)
                        else:
                            p_value = 1.0

                if p_value > max_p and p_value > self.max_pvalue:
                    max_p = p_value
                    merge_idx = min(i, j)

            if merge_idx == -1:
                break

            # 合并箱
            if merge_idx < len(splits):
                splits.pop(merge_idx)
            else:
                splits.pop()

        return np.array(splits)

    def _apply_bin_size_constraint(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: np.ndarray
    ) -> np.ndarray:
        """应用样本数约束.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        n_samples = len(x)

        # 计算样本数约束
        if self.min_bin_size < 1:
            min_samples = max(int(n_samples * self.min_bin_size), 1)
        else:
            min_samples = int(self.min_bin_size)

        if self.max_bin_size is None:
            max_samples = n_samples
        elif self.max_bin_size < 1:
            max_samples = int(n_samples * self.max_bin_size)
        else:
            max_samples = int(self.max_bin_size)

        splits = list(splits)
        max_iter = len(splits)

        for _ in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            # 计算每箱的样本数
            bins = np.searchsorted(splits, x, side='right')
            bin_counts = np.bincount(bins, minlength=len(splits) + 1)

            # 检查是否有违反约束的箱
            violations = []
            for i, count in enumerate(bin_counts):
                if count < min_samples or count > max_samples:
                    violations.append((i, count))

            if not violations:
                break

            # 找到违反约束最严重的箱
            # 优先合并样本数过少的箱
            min_count_idx = min(violations, key=lambda x: x[1])
            merge_idx = min_count_idx[0]

            # 合并到相邻箱
            if merge_idx == 0 and len(splits) > 0:
                splits.pop(0)
            elif merge_idx >= len(splits) and len(splits) > 0:
                splits.pop()
            elif len(splits) > 0:
                # 选择相邻箱中样本数较少的进行合并
                if merge_idx < len(bin_counts) - 1:
                    if bin_counts[merge_idx] < bin_counts[merge_idx + 1]:
                        if merge_idx < len(splits):
                            splits.pop(merge_idx)
                    else:
                        if merge_idx > 0:
                            splits.pop(merge_idx - 1)
                else:
                    if merge_idx > 0:
                        splits.pop(merge_idx - 1)

        return np.array(splits)

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
            # 类别型特征
            splits = self.splits_[feature]
            if isinstance(splits, list):
                # 类别列表，按顺序分配索引
                bins = np.zeros(len(X), dtype=int)
                for i, cat in enumerate(splits):
                    bins[x_vals == cat] = i
                bins[X.isna()] = -1
                if self.special_codes:
                    for code in self.special_codes:
                        bins[x_vals == code] = -2
                return bins
            else:
                # 数值型切分点（编码后的）
                codes = pd.Categorical(X).codes
                return np.where(X.isna(), -1, codes)
        else:
            # 数值型特征
            splits = self.splits_[feature]
            n = len(x_vals)
            bins = np.zeros(n, dtype=int)

            # 处理缺失值 - 使用 pd.isna 更准确地检测
            missing_mask = pd.isna(X)
            bins[missing_mask] = -1

            # 处理特殊值
            if self.special_codes:
                for code in self.special_codes:
                    special_mask = ~missing_mask & (x_vals == code)
                    bins[special_mask] = -2

            # 正常值（非缺失且非特殊值）
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

    def _get_bin_labels_for_transform(
        self,
        feature: str,
        bins: np.ndarray
    ) -> Dict[int, str]:
        """生成分箱标签映射.

        返回字典，key 是 bin 索引，value 是标签。

        :param feature: 特征名
        :param bins: 分箱索引数组
        :return: 标签字典
        """
        splits = self.splits_[feature]
        labels = {}

        # 处理缺失值和特殊值
        labels[-1] = 'missing'
        labels[-2] = 'special'

        # 处理正常分箱
        if isinstance(splits, list):
            # 类别型特征
            for i, cat in enumerate(splits):
                if i == 0:
                    labels[i] = f'(-inf, {cat}]'
                elif i == len(splits) - 1:
                    labels[i] = f'({splits[i-1]}, {cat}]'
                else:
                    labels[i] = f'({splits[i-1]}, {cat}]'
            # 最后一个箱
            labels[len(splits)] = f'({splits[-1] if splits else "-inf"}, +inf)'
        else:
            # 数值型特征
            n_splits = len(splits) if splits is not None else 0
            for i in range(n_splits + 1):
                if n_splits == 0:
                    labels[i] = '(-inf, +inf)'
                elif i == 0:
                    labels[i] = f'(-inf, {splits[i]}]'
                elif i == n_splits:
                    labels[i] = f'({splits[i-1]}, +inf)'
                else:
                    labels[i] = f'({splits[i-1]}, {splits[i]}]'

        return labels

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
        >>> binner = CARTBinning()
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
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            bins = self._assign_bins(X[feature], feature)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels_for_transform(feature, bins)
                result[feature] = [labels.get(b, f'bin_{b}') for b in bins]
            elif metric == 'woe':
                # 优先使用_woe_maps_（从export/load导入）
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = [woe_map.get(b, 0) for b in bins]
            else:
                raise ValueError(f"不支持的metric: {metric}")

        return result


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    n_samples = 5000

    # 生成测试数据
    x1 = np.random.uniform(0, 100, n_samples)
    bad_rate = 0.05 + 0.004 * x1  # 从5%递增到45%
    y_binary = np.random.binomial(1, bad_rate)

    X = pd.DataFrame({'feature': x1})
    y = pd.Series(y_binary)

    print("=" * 60)
    print("CART 分箱测试")
    print("=" * 60)

    # 测试 CART 分箱
    binner = CartBinning(
        max_n_bins=5,
        min_bin_size=0.05,
        monotonic=True,
        verbose=True
    )
    binner.fit(X, y)

    print("\n分箱统计表:")
    print(binner.get_bin_table('feature'))

    print("\n切分点:", binner.splits_['feature'])
    print("分箱数:", binner.n_bins_['feature'])

    # 测试 p-value 约束
    print("\n" + "=" * 60)
    print("带 p-value 约束的 CART 分箱")
    print("=" * 60)

    binner2 = CartBinning(
        max_n_bins=10,
        min_bin_size=0.01,
        max_pvalue=0.05,
        max_pvalue_policy="consecutive",
        monotonic=True
    )
    binner2.fit(X, y)

    print("\n分箱统计表:")
    print(binner2.get_bin_table('feature'))
    print("\n切分点:", binner2.splits_['feature'])
    print("分箱数:", binner2.n_bins_['feature'])

    # 转换测试
    print("\n转换测试:")
    X_test = pd.DataFrame({'feature': [10, 30, 50, 70, 90, np.nan]})
    X_binned = binner.transform(X_test, metric='bins')
    print(X_binned)
