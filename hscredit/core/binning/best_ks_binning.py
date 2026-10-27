"""Best KS 分箱算法.

基于最大化KS统计量的分箱方法，使用贪心算法寻找最优分箱点。
KS统计量衡量好样本和坏样本的累积分布差异。

算法流程：
1. 预分割：将数据分成足够细的初始箱（默认50个）
2. 合并优化：使用贪心算法，在单调性约束下选择最优分割点
3. 目标：最大化KS统计量
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd

from ...exceptions import NotFittedError
from .base import BaseBinning


class BestKSBinning(BaseBinning):
    """Best KS 分箱.

    基于最大化KS统计量的分箱方法，寻找能够最大化区分能力的分箱点。
    使用贪心算法逐步优化。

    **参数**

    :param constraint_policy: 约束冲突策略，默认 ``"raise"`` 报错；``"legacy"`` 在约束不可行时
        保留原生尽力满足的规则（优化器无可行分区时可回退单箱）。不放宽输入校验或吞掉回调错误。
        实际未满足项记录于 ``constraint_violations_``，不增加业务统计表列。
    :type constraint_policy: str

    :param kwargs: 传递给 BaseBinning 的附加构造参数，例如 ``decimal``、``woe_clip``；
        与显式参数重名或不存在的参数会被构造函数拒绝。

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。

    :type target: str
    :param cat_cutoff: 数值字段转为类别字段的识别阈值，默认 ``None`` （数值字段保持数值型）。
        ``0 < cat_cutoff < 1`` 时，非缺失唯一值数 / 非缺失样本数不超过该比例即按类别处理；
        ``cat_cutoff >= 1`` 时，唯一值数不超过 ``int(cat_cutoff)`` 即按类别处理。
        字符串、category 和布尔字段本身按类别处理；该参数不用于合并低频类别。

    :type cat_cutoff: Optional[Union[float, int]]
    :param category_order: 类别顺序，默认 ``None`` 按训练集坏样本率升序排列，同率按首次出现顺序。
        可传 ``{字段名: 完整类别序列}`` 或 ``callable(feature, x, y)`` ；
        顺序必须覆盖该字段全部非缺失训练类别，不能重复或包含未知类别。

    :param handle_unknown: 预测期未知类别策略，默认 ``-3`` ； ``"value"`` 等价于 ``-3`` ，
        返回未知箱（标签 ``"unknown"`` ，WOE 为 ``0.0`` ）； ``"raise"`` 抛出错误。
        也可传入已记录的整数箱号，将未知类别归入该箱；不接受布尔值。

    :type handle_unknown: Union[int, str]
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Union[int, float]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
    :param user_splits: 自定义规则字典或返回规则的可调用对象，默认 ``None`` 。数值规则为严格递增的
        有限内部切点（不含正负无穷）；类别规则为完整覆盖训练类别的嵌套列表。
        缺失标记最多出现一次，可独立成箱或加入指定箱。

    :type user_splits: Optional[Dict[str, List]]
    :param user_splits_fixed: 固定用户节点，默认 ``None`` 表示不固定；可传全局布尔值或
        ``{字段名: 布尔值或布尔序列}`` 。数值序列长度须等于有限切点数；
        类别序列长度须等于用户分组数。 ``True`` 保留节点， ``False`` 允许合并。

    :type user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, List[bool]]]]]

    :param max_n_bins: 最大分箱数，默认为5
    :type max_n_bins: int
    :param min_n_bins: 最小分箱数，默认为2
    :type min_n_bins: int
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01

        - 如果 < 1, 表示占比 (如 0.01 表示 1%)

        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)

    :type min_bin_size: Union[float, int]
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :type max_bin_size: Optional[Union[float, int]]
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :type min_bad_rate: float
    :param monotonic: 坏样本率单调性约束，默认为False

        - False: 不要求单调性

        - True 或 'auto': 自动检测并应用最佳单调方向

        - 'ascending': 强制坏样本率递增

        - 'descending': 强制坏样本率递减

    :type monotonic: Union[bool, str]
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :type missing_separate: bool
    :param special_codes: 特殊值列表，默认为None
    :type special_codes: Optional[List]
    :param random_state: 随机种子，默认为None
    :type random_state: Optional[int]

    **参考样例**

    >>> from hscredit.core.binning import BestKSBinning
    >>> binner = BestKSBinning(max_n_bins=5)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    **注意**

    Best KS 分箱的特点:

    1. 以最大化 KS（Kolmogorov–Smirnov）统计量为目标选择切分点

    2. KS = max(abs(累计好样本占比 - 累计坏样本占比))，衡量好坏样本累积分布的最大差异

    3. 使用贪心算法逐步优化，支持单调性约束

    4. 相比等频/等距计算复杂度较高，但区分度更优

    **引用**

    Kolmogorov–Smirnov 检验：https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test ；
    KS 统计量在信用风险中的应用见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley.
    """

    def __init__(
        self,
        target: str = "target",
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        category_order=None,
        handle_unknown: Union[int, str] = -3,
        random_state: Optional[int] = None,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        user_splits: Optional[Dict[str, List]] = None,
        user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, List[bool]]]]] = None,
        constraint_policy: str = "raise",
        **kwargs,
    ):
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            missing_separate=missing_separate,
            special_codes=special_codes,
            cat_cutoff=cat_cutoff,
            user_splits=user_splits,
            user_splits_fixed=user_splits_fixed,
            category_order=category_order,
            handle_unknown=handle_unknown,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            **kwargs,
            constraint_policy=constraint_policy,
        )

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "BestKSBinning":
        """拟合 Best KS 分箱。

        对每个特征预分割为细箱后，在单调性约束下贪心选择切分点以最大化 KS 统计量。
        支持 sklearn 风格 ``fit(X, y)`` 与 scorecardpipeline 风格 ``fit(df)`` ，
        详见  :meth:`BaseBinning.fit` 。

        :param X: 训练数据，shape ``(n_samples, n_features)`` ，DataFrame 或 ndarray
        :type X: Union[pd.DataFrame, np.ndarray]
        :param y: 默认值为 ``None`` 。二分类目标变量（0=好/1=坏）；scorecardpipeline 风格下可省略
        :type y: Optional[Union[pd.Series, np.ndarray]]
        :param kwargs: 透传给基类的其他参数
        :return: 拟合后的分箱器自身（便于链式调用）
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        self._fit_features(X, y, "_fit_feature")

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._finalize_categorical_fit()
        self._finalize_reserved_bins(X, y)
        self._is_fitted = True
        return self

    def _fit_feature(self, feature: str, X: pd.Series, y: pd.Series) -> None:
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

        if feature_type == "categorical":
            # 类别型变量：按KS排序后分箱
            splits = self._best_ks_categorical(X_valid, y_valid)
            self.splits_[feature] = np.array(splits)
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            # 数值型变量：Best KS分箱
            splits = self._best_ks_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _best_ks_numerical(self, X: pd.Series, y: pd.Series) -> List[float]:
        """对数值型变量进行Best KS分箱 (优化版本).

        使用排序后数据的累积统计信息快速计算KS值。

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表
        """
        # 转换为 numpy 数组加速计算
        x_vals = X.values
        y_vals = y.values

        # 获取排序后的唯一值
        unique_values = np.unique(x_vals)

        if len(unique_values) <= self.max_n_bins:
            # 唯一值较少时，直接使用唯一值边界（避免退化为单箱）
            return (unique_values[:-1] / 2.0 + unique_values[1:] / 2.0).tolist()

        # 限制候选分割点数量，避免过多唯一值导致性能问题
        max_candidates = min(len(unique_values) - 1, 100)
        if len(unique_values) > max_candidates + 1:
            # 使用分位数选择候选点
            quantiles = np.linspace(0, 1, max_candidates + 1)
            candidates = np.quantile(x_vals, quantiles[1:-1])
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            candidates = np.unique(candidates)
            candidates = candidates[(candidates > x_min) & (candidates < x_max)]
        else:
            # 相邻唯一值的中点
            candidates = unique_values[:-1] / 2.0 + unique_values[1:] / 2.0

        # 预计算排序后的数据
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]

        # 计算总体统计
        total_good = np.sum(y_vals == 0)
        total_bad = np.sum(y_vals == 1)

        if total_good == 0 or total_bad == 0:
            return []

        # 预计算累积统计
        cum_bad = np.cumsum(y_sorted)
        cum_good = np.arange(1, len(y_sorted) + 1) - cum_bad

        # 使用贪心算法选择最优分割点
        selected_splits = []
        min_samples = self._get_min_samples(len(x_sorted))

        while len(selected_splits) < self.max_n_bins - 1 and len(candidates) > 0:
            best_ks = -1
            best_iv = -1.0
            best_split_idx = -1
            best_split = None

            for i, candidate in enumerate(candidates):
                # 找到候选点在排序数据中的位置
                split_pos = np.searchsorted(x_sorted, candidate, side="left")

                if split_pos == 0 or split_pos >= len(x_sorted):
                    continue

                # 计算该分割点与已选分割点组合的 KS
                test_splits = sorted(selected_splits + [candidate])
                split_positions = [np.searchsorted(x_sorted, s, side="left") for s in test_splits]
                split_positions = [0] + split_positions + [len(x_sorted)]
                if any(
                    (split_positions[j + 1] - split_positions[j]) < min_samples for j in range(len(split_positions) - 1)
                ):
                    continue
                ks = self._calc_ks_fast(x_sorted, y_sorted, cum_good, cum_bad, total_good, total_bad, test_splits)

                from ..metrics._binning import iv_for_splits

                iv = iv_for_splits(x_sorted, y_sorted, test_splits)
                if ks > best_ks + 1e-12 or (abs(ks - best_ks) <= 1e-12 and iv > best_iv + 1e-12):
                    best_ks = ks
                    best_iv = iv
                    best_split_idx = i
                    best_split = candidate

            if best_split is not None:
                selected_splits.append(best_split)
                candidates = np.delete(candidates, best_split_idx)
            else:
                break

        return sorted(selected_splits)

    def _calc_ks_fast(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        cum_good: np.ndarray,
        cum_bad: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float],
    ) -> float:
        """快速计算KS统计量.

        使用预计算的累积统计信息。

        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param cum_good: 累积好样本数
        :param cum_bad: 累积坏样本数
        :param total_good: 总好样本数
        :param total_bad: 总坏样本数
        :param splits: 分割点列表
        :return: KS值
        """
        if not splits:
            return 0.0

        # 找到所有分割点的位置
        split_positions = [np.searchsorted(x_sorted, s, side="left") for s in sorted(splits)]
        split_positions = [0] + split_positions + [len(x_sorted)]

        max_ks = 0

        for i in range(len(split_positions) - 1):
            start = split_positions[i]
            end = split_positions[i + 1]

            if start >= end:
                continue

            cum_good_rate = cum_good[end - 1] / total_good
            cum_bad_rate = cum_bad[end - 1] / total_bad

            ks = abs(cum_good_rate - cum_bad_rate)
            max_ks = max(max_ks, ks)

        return max_ks

    def _best_ks_categorical(self, X: pd.Series, y: pd.Series) -> List[float]:
        """对类别型变量进行Best KS分箱 (优化版本).

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表
        """
        # 使用向量化操作计算类别统计
        df = pd.DataFrame({"X": X, "y": y})
        category_stats = df.groupby("X")["y"].agg(["mean", "count"]).reset_index()
        category_stats.columns = ["category", "bad_rate", "count"]

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(X))
        category_stats = category_stats[category_stats["count"] >= min_samples]

        if len(category_stats) <= self.max_n_bins:
            return []

        # 按坏样本率排序
        category_stats = category_stats.sort_values("bad_rate")

        # 返回编码边界
        n_categories = len(category_stats)
        return [i - 0.5 for i in range(1, min(n_categories, self.max_n_bins))]

    def _calc_ks(self, X: pd.Series, y: pd.Series, splits: List[float]) -> float:
        """计算KS统计量 (兼容旧代码).

        :param X: 特征数据
        :param y: 目标变量
        :param splits: 分割点列表
        :return: KS值
        """
        x_vals = X.values if isinstance(X, pd.Series) else X
        y_vals = y.values if isinstance(y, pd.Series) else y

        # 根据分割点分箱
        bins = np.searchsorted(splits, x_vals, side="right")

        # 计算每个箱的统计
        total_good = np.sum(y_vals == 0)
        total_bad = np.sum(y_vals == 1)

        if total_good == 0 or total_bad == 0:
            return 0.0

        # 使用 bincount 快速计算累积统计
        n_bins = len(splits) + 1
        bin_good = np.bincount(bins[y_vals == 0], minlength=n_bins)
        bin_bad = np.bincount(bins[y_vals == 1], minlength=n_bins)

        # 计算累积分布和 KS
        cum_good = np.cumsum(bin_good)
        cum_bad = np.cumsum(bin_bad)

        cum_good_rate = cum_good / total_good
        cum_bad_rate = cum_bad / total_bad

        ks_values = np.abs(cum_good_rate - cum_bad_rate)

        return np.max(ks_values)

    def _assign_bins(self, X: pd.Series, feature: str) -> np.ndarray:
        """为数据分配分箱索引 (优化版本).

        :param X: 特征数据
        :param feature: 特征名
        :return: 分箱索引数组
        """
        x_vals = X.values

        if self.feature_types_[feature] == "categorical" and feature in self._cat_bins_:
            return self._assign_categorical_bins(feature, X)
        if self.feature_types_[feature] == "categorical":
            # 使用 pd.Categorical 的 codes
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
                bins[valid_mask] = np.searchsorted(splits, x_vals[valid_mask], side="right")

            return bins

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], metric: str = "indices", **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.

        将原始特征值转换为分箱索引、分箱标签或WOE值。

        :param X: 待转换数据, DataFrame或数组格式
        :type X: Union[pd.DataFrame, np.ndarray]
        :param metric: 默认值为 ``'indices'`` 。转换类型, 可选值:

            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理

            - 'bins': 返回分箱标签字符串, 用于可视化或报告

            - 'woe': 返回WOE值, 用于逻辑回归建模

        :type metric: str
        :param kwargs: 其他参数
        :return: 转换后的数据, 格式与输入X相同

        **参考样例**

        >>> binner = BestKSBinning()
        >>> binner.fit(X_train, y_train)
        >>>
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>>
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        return self._transform_binning_features(
            X,
            metric,
            lambda feature: self._assign_bins(X[feature], feature),
            woe_default=0.0,
        )
