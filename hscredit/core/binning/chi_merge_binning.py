"""卡方分箱算法 (ChiMerge).

基于卡方统计量合并相邻箱的分箱方法。
通过迭代合并卡方值最小的相邻箱，直到满足停止条件。
"""

import logging
from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from ...exceptions import NotFittedError
from .base import BaseBinning

logger = logging.getLogger(__name__)


class ChiMergeBinning(BaseBinning):
    """卡方分箱算法 (ChiMerge)。

    一种自底向上（bottom-up）的有监督分箱方法。其核心假设是：若相邻两箱的好坏样本
    分布无显著差异（卡方值小），则可以合并。算法流程：

    1. **初始化**：将每个唯一值（或预分位点）各作为一个箱；
    2. **迭代合并**：对所有相邻箱对计算 2×2 列联表的卡方统计量，合并卡方值最小的一对；
    3. **停止**：当最小卡方值超过阈值 ``min_chi2_threshold``，或箱数降至 ``max_n_bins``、
       ``min_n_bins`` 限制时停止。

    卡方值越大表示相邻两箱坏样本率差异越显著、越不应合并。继承 :class:`BaseBinning`，
    完整的分箱通用参数、属性与转换语义见基类。

    **参数（本算法特有及关键项）**

    :param target: 目标列名（scorecardpipeline 风格），默认为 ``'target'``
    :param max_n_bins: 最大分箱数，默认为 ``10``
    :param min_n_bins: 最小分箱数，默认为 ``2``
    :param min_chi2_threshold: 卡方合并阈值（停止合并的下限），默认为 ``None``。
        为 ``None`` 时取自由度 1、显著性水平 ``significance_level`` 的卡方临界值
        （如 0.05 对应约 3.841）。当相邻箱的最小卡方值大于该阈值即停止合并
    :param min_chi2: **已废弃**，请改用 ``min_chi2_threshold``；传入非 ``None`` 将报错
    :param significance_level: 显著性水平，用于在 ``min_chi2_threshold`` 为 ``None`` 时
        换算卡方临界值，默认为 ``0.05``（值越小阈值越大、分箱越粗）
    :param min_bin_size: 每箱最小样本数（``>=1``）或占比（``<1``），默认为 ``0.01``
    :param max_bin_size: 每箱最大样本数或占比，默认为 ``None``
    :param min_bad_rate: 每箱最小坏样本率，默认为 ``0.0``
    :param monotonic: 坏样本率单调性约束，取值见 :class:`BaseBinning`，默认为 ``False``
    :param special_codes: 特殊值列表，单独成箱，默认为 ``None``
    :param missing_separate: 是否将缺失值单独成箱，默认为 ``True``
    :param random_state: 随机种子，默认为 ``None``
    :param verbose: 是否输出合并过程日志，默认为 ``False``
    :param decimal: 数值切分点保留小数位，默认为 ``4``

    **属性**

    - ``splits_``: 每个特征的分箱切分点
    - ``n_bins_``: 每个特征的实际分箱数
    - ``bin_tables_``: 每个特征的分箱统计表（中文列名，见 :class:`BaseBinning`）

    **参考样例**

    >>> from hscredit.core.binning import ChiMergeBinning
    >>> binner = ChiMergeBinning(max_n_bins=5, significance_level=0.05)
    >>> binner.fit(X, y)                       # sklearn 风格
    >>> X_woe = binner.transform(X, metric='woe')
    >>> binner.get_bin_table('age')           # 查看某特征分箱明细

    指定卡方阈值并要求坏样本率单调递减::

        >>> binner = ChiMergeBinning(min_chi2_threshold=6.635, monotonic='descending')
        >>> binner.fit(X, y)

    **引用**

    Kerber, R. (1992). *ChiMerge: Discretization of Numeric Attributes.*
    Proceedings of AAAI-92. https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
    """

    def __init__(
        self,
        target: str = "target",
        max_n_bins: int = 10,
        min_n_bins: int = 2,
        min_chi2_threshold: Optional[float] = None,
        min_chi2: Optional[float] = None,
        significance_level: float = 0.05,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        cat_cutoff: Optional[Union[float, int]] = None,
        category_order=None,
        handle_unknown: str = "value",
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target=target,
            min_n_bins=min_n_bins,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            cat_cutoff=cat_cutoff,
            category_order=category_order,
            handle_unknown=handle_unknown,
            random_state=random_state,
            verbose=verbose,
            decimal=decimal,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.min_chi2 = min_chi2
        if min_chi2 is not None:
            raise ValueError("min_chi2 参数已移除，请使用 min_chi2_threshold")
        self.min_chi2_threshold = min_chi2_threshold
        self.significance_level = significance_level

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "ChiMergeBinning":
        """拟合卡方分箱。

        对每个特征执行 ChiMerge 合并，得到切分点与分箱统计表。支持 sklearn 风格
        ``fit(X, y)`` 与 scorecardpipeline 风格 ``fit(df)``（目标列由 ``target`` 指定），
        详见 :meth:`BaseBinning.fit`。

        :param X: 训练数据，shape ``(n_samples, n_features)``；可为 DataFrame 或 ndarray
        :param y: 目标变量，二分类（0=好/1=坏）；scorecardpipeline 风格下可省略
        :param kwargs: 透传给基类的其他参数
        :return: 拟合后的分箱器自身（便于链式调用）

        **参考样例**

        >>> ChiMergeBinning(max_n_bins=5).fit(X, y).transform(X, metric='woe')
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 确定卡方阈值
        if self.min_chi2_threshold is None:
            # 自由度为1，显著性水平为significance_level的卡方临界值
            self.min_chi2_threshold = chi2.ppf(1 - self.significance_level, df=1)

        # 对每个特征进行分箱
        def _fit_one(feature):
            if self.verbose:
                logger.info(f"处理特征: {feature}")

            # 检测特征类型
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == "categorical":
                # 类别型特征
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                # 数值型特征：卡方分箱
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

            # 计算分箱统计信息
            bins = self._apply_bins(X[feature], splits)
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)

        self._fit_features(X.columns, _fit_one)

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._finalize_categorical_fit()
        self._is_fitted = True
        return self

    def _fit_numerical(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """对数值型特征进行卡方分箱 (优化版本).

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组
        """
        # 处理缺失值和特殊值
        mask = x.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x != code)

        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            return np.array([])

        # 获取唯一值并排序 (使用 numpy 加速)
        unique_values = np.sort(x_valid.unique())

        if len(unique_values) <= self.min_n_bins:
            # 如果唯一值数量小于等于最小分箱数，直接使用唯一值作为切分点
            return unique_values[:-1].astype(float)

        # 限制初始分箱数量，避免过多的唯一值导致性能问题
        max_initial_bins = min(len(unique_values) - 1, 100)
        if len(unique_values) > max_initial_bins + 1:
            # 使用样本分位数而非唯一值分位数，效果更接近 optbinning/toad
            quantiles = np.linspace(0, 1, max_initial_bins + 1)
            splits = np.quantile(x_valid.astype(float), quantiles[1:-1])
            x_min, x_max = x_valid.min(), x_valid.max()
            splits = np.unique(splits)
            splits = splits[(splits > x_min) & (splits < x_max)].astype(float)
        else:
            # 初始分箱：每个唯一值作为一个箱
            splits = unique_values[:-1].astype(float)

        # 迭代合并
        splits = self._chi_merge(x_valid, y_valid, splits)

        # 应用单调性约束
        if self.monotonic:
            splits = self._apply_monotonic_constraint(x_valid, y_valid, splits)

        # 根据约束调整分箱数
        splits = self._adjust_bins(x_valid, y_valid, splits)

        return splits

    def _chi_merge(self, x: pd.Series, y: pd.Series, splits: np.ndarray) -> np.ndarray:
        """执行卡方合并算法 (优化版本).

        使用向量化操作和增量更新提高性能。

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 合并后的切分点
        """
        splits = splits.copy()
        min_samples = self._get_min_samples(len(x))

        if len(splits) == 0:
            return splits

        # 转换为 numpy 数组加速计算
        x_vals = x.values
        y_vals = y.values

        # 预计算每个样本的分箱索引
        bins = np.searchsorted(splits, x_vals, side="right")

        # 构建分箱统计信息 (good_count, bad_count)
        n_bins = len(splits) + 1
        bin_stats = np.zeros((n_bins, 2), dtype=np.int64)

        for i in range(n_bins):
            mask = bins == i
            bin_stats[i, 0] = np.sum(y_vals[mask] == 0)  # good
            bin_stats[i, 1] = np.sum(y_vals[mask] == 1)  # bad

        max_iter = 1000
        for iteration in range(max_iter):
            if len(splits) < self.min_n_bins - 1:
                break

            if len(splits) == 0:
                break

            # 向量化计算所有相邻箱的卡方值
            chi2_values = self._compute_chi2_vectorized(bin_stats)

            if len(chi2_values) == 0:
                break

            # 找到卡方值最小的相邻箱
            min_chi2_idx = np.argmin(chi2_values)
            min_chi2_val = chi2_values[min_chi2_idx]

            # 检查停止条件
            if min_chi2_val > self.min_chi2_threshold:
                if self.verbose:
                    logger.info(f"  迭代 {iteration}: 最小卡方值 {min_chi2_val:.4f} > 阈值 {self.min_chi2_threshold:.4f}，停止合并")
                break

            if len(splits) + 1 <= self.max_n_bins:
                if self.verbose:
                    logger.info(f"  迭代 {iteration}: 分箱数 {len(splits) + 1} 达到上限 {self.max_n_bins}，停止合并")
                break

            if len(splits) <= self.min_n_bins - 1:
                if self.verbose:
                    logger.info(f"  迭代 {iteration}: 分箱数达到最小值 {self.min_n_bins}，停止合并")
                break

            # 合并卡方值最小的相邻箱
            if self.verbose:
                logger.info(f"  迭代 {iteration}: 合并箱 {min_chi2_idx} 和 {min_chi2_idx + 1}，卡方值={min_chi2_val:.4f}")

            # 更新分箱统计：合并相邻两个箱
            bin_stats[min_chi2_idx] += bin_stats[min_chi2_idx + 1]
            bin_stats = np.delete(bin_stats, min_chi2_idx + 1, axis=0)

            # 更新切分点
            splits = np.delete(splits, min_chi2_idx)

            # 检查样本数约束
            if any(bin_stats[:, 0] + bin_stats[:, 1] < min_samples):
                continue

        return splits

    def _compute_chi2_vectorized(self, bin_stats: np.ndarray) -> np.ndarray:
        """向量化计算所有相邻箱的卡方值.

        :param bin_stats: 分箱统计数组，shape (n_bins, 2)，列分别为 good_count, bad_count
        :return: 相邻箱之间的卡方值数组
        """
        n_bins = len(bin_stats)
        if n_bins < 2:
            return np.array([])

        # 相邻箱的统计
        bin1_good = bin_stats[:-1, 0]
        bin1_bad = bin_stats[:-1, 1]
        bin2_good = bin_stats[1:, 0]
        bin2_bad = bin_stats[1:, 1]

        # 计算边际和
        total_good = bin1_good + bin2_good
        total_bad = bin1_bad + bin2_bad
        total = total_good + total_bad

        # 检查有效样本
        valid_mask = total > 0

        # 计算期望频数
        row1_total = bin1_good + bin1_bad
        row2_total = bin2_good + bin2_bad

        e1_good = np.divide(
            row1_total * total_good,
            total,
            out=np.zeros_like(total, dtype=float),
            where=valid_mask,
        )
        e1_bad = np.divide(
            row1_total * total_bad,
            total,
            out=np.zeros_like(total, dtype=float),
            where=valid_mask,
        )
        e2_good = np.divide(
            row2_total * total_good,
            total,
            out=np.zeros_like(total, dtype=float),
            where=valid_mask,
        )
        e2_bad = np.divide(
            row2_total * total_bad,
            total,
            out=np.zeros_like(total, dtype=float),
            where=valid_mask,
        )

        # 计算卡方值 (向量化)
        eps = 1e-10
        chi2_vals = np.zeros(n_bins - 1)

        for observed, expected in (
            (bin1_good, e1_good),
            (bin1_bad, e1_bad),
            (bin2_good, e2_good),
            (bin2_bad, e2_bad),
        ):
            chi2_vals += np.divide(
                (observed - expected) ** 2,
                expected,
                out=np.zeros_like(expected, dtype=float),
                where=expected > eps,
            )

        # 无效样本设为无穷大
        chi2_vals[~valid_mask] = np.inf

        return chi2_vals

    def _compute_chi2(self, y1: pd.Series, y2: pd.Series) -> float:
        """计算两个箱之间的卡方值 (兼容旧代码).

        :param y1: 第一个箱的目标变量
        :param y2: 第二个箱的目标变量
        :return: 卡方值
        """
        n1_good = (y1 == 0).sum()
        n1_bad = (y1 == 1).sum()
        n2_good = (y2 == 0).sum()
        n2_bad = (y2 == 1).sum()

        if n1_good + n1_bad == 0 or n2_good + n2_bad == 0:
            return np.inf

        total_good = n1_good + n2_good
        total_bad = n1_bad + n2_bad
        total = total_good + total_bad

        n1_total = n1_good + n1_bad
        n2_total = n2_good + n2_bad

        e1_good = n1_total * total_good / total
        e1_bad = n1_total * total_bad / total
        e2_good = n2_total * total_good / total
        e2_bad = n2_total * total_bad / total

        eps = 1e-10

        chi2_val = 0
        chi2_val += (n1_good - e1_good) ** 2 / e1_good if e1_good > eps else 0
        chi2_val += (n1_bad - e1_bad) ** 2 / e1_bad if e1_bad > eps else 0
        chi2_val += (n2_good - e2_good) ** 2 / e2_good if e2_good > eps else 0
        chi2_val += (n2_bad - e2_bad) ** 2 / e2_bad if e2_bad > eps else 0

        return chi2_val

    def _apply_monotonic_constraint(self, x: pd.Series, y: pd.Series, splits: np.ndarray) -> np.ndarray:
        """应用单调性约束.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)

        bin_stats = pd.DataFrame({"bin": bins, "target": y}).groupby("bin")["target"].mean()

        # 确保所有分箱都在bin_stats中
        bin_stats = self._ensure_all_bins_in_series(bin_stats, len(splits) + 1)

        is_monotonic_increasing = all(bin_stats.iloc[i] <= bin_stats.iloc[i + 1] for i in range(len(bin_stats) - 1))
        is_monotonic_decreasing = all(bin_stats.iloc[i] >= bin_stats.iloc[i + 1] for i in range(len(bin_stats) - 1))

        if self.monotonic == "ascending" and not is_monotonic_increasing:
            splits = self._merge_for_monotonicity(x, y, splits, increasing=True)
        elif self.monotonic == "descending" and not is_monotonic_decreasing:
            splits = self._merge_for_monotonicity(x, y, splits, increasing=False)
        elif self.monotonic is True or self.monotonic == "auto":
            if not is_monotonic_increasing and not is_monotonic_decreasing:
                inc_violations = sum(1 for i in range(len(bin_stats) - 1) if bin_stats.iloc[i] > bin_stats.iloc[i + 1])
                dec_violations = sum(1 for i in range(len(bin_stats) - 1) if bin_stats.iloc[i] < bin_stats.iloc[i + 1])

                if inc_violations <= dec_violations:
                    splits = self._merge_for_monotonicity(x, y, splits, increasing=True)
                else:
                    splits = self._merge_for_monotonicity(x, y, splits, increasing=False)

        return splits

    def _merge_for_monotonicity(self, x: pd.Series, y: pd.Series, splits: np.ndarray, increasing: bool) -> np.ndarray:
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

            bin_stats = pd.DataFrame({"bin": bins, "target": y}).groupby("bin")["target"].mean()

            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_series(bin_stats, len(splits) + 1)

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

            merge_idx = violations[0]
            new_splits = np.delete(splits, merge_idx)

            if len(new_splits) < self.min_n_bins - 1:
                break

            splits = new_splits

        return splits

    def _fit_categorical(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """对类别型特征进行分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组（类别列表）
        """
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        # 计算每个类别的坏样本率
        cat_stats = (
            pd.DataFrame({"category": x_valid, "target": y_valid}).groupby("category")["target"].agg(["mean", "count"])
        )

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(x_valid))
        cat_stats = cat_stats[cat_stats["count"] >= min_samples]

        # 按坏样本率排序
        cat_stats = cat_stats.sort_values("mean")

        return cat_stats.index.tolist()

    def _adjust_bins(self, x: pd.Series, y: pd.Series, splits: np.ndarray) -> np.ndarray:
        """根据约束条件调整分箱 (优化版本).

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        x_vals = x.values
        min_samples = self._get_min_samples(len(x))

        max_iter = 20
        for _ in range(max_iter):
            # 使用 numpy 的 searchsorted 替代 pd.cut
            bins = np.searchsorted(splits, x_vals, side="right")

            # 使用 numpy 的 bincount 计算每箱样本数
            n_bins = len(splits) + 1
            bin_counts = np.bincount(bins, minlength=n_bins)

            # 找出样本数过少的箱并合并
            new_splits = []
            skip_next = False

            for i in range(len(splits)):
                if skip_next:
                    skip_next = False
                    continue

                count = bin_counts[i] if i < len(bin_counts) else 0

                if count < min_samples and i < len(splits) - 1:
                    skip_next = True
                else:
                    new_splits.append(splits[i])

            new_splits = np.array(new_splits)

            if len(new_splits) == len(splits):
                break

            splits = new_splits

            if len(splits) + 1 < self.min_n_bins:
                break

        n_bins = len(splits) + 1
        if n_bins > self.max_n_bins:
            n_remove = n_bins - self.max_n_bins
            # 确保至少保留 min_n_bins - 1 个切分点
            n_remove = min(n_remove, len(splits) - (self.min_n_bins - 1))
            if n_remove > 0:
                splits = splits[n_remove:]
        elif n_bins < self.min_n_bins:
            quantiles = np.linspace(0, 1, self.min_n_bins + 1)
            splits = np.percentile(x_vals, quantiles[1:-1] * 100)

        return splits

    def _ensure_all_bins_in_series(self, bin_stats: pd.Series, n_bins: int) -> pd.Series:
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

    def _apply_bins(self, x: pd.Series, splits: Union[np.ndarray, List]) -> np.ndarray:
        """应用分箱.

        :param x: 特征数据
        :param splits: 切分点
        :return: 分箱索引
        """
        feature = x.name
        if feature in self._cat_bins_ and self.feature_types_.get(feature) == "categorical":
            return self._assign_categorical_bins(feature, x)
        if isinstance(splits, list):
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
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

            bins[mask] = np.digitize(x[mask], splits)

            return bins

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], metric: str = "indices", **kwargs
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
        >>> binner = ChiMergeBinning()
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
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=["feature"])
                else:
                    X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            splits = self.splits_[feature]
            bins = self._apply_bins(X[feature], splits)

            if metric == "indices":
                result[feature] = bins
            elif metric == "bins":
                result[feature] = self._assign_bin_labels(feature, bins)
            elif metric == "woe":
                # 优先使用_woe_maps_（从export/load导入）
                if hasattr(self, "_woe_maps_") and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table["分档WOE值"].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 生成测试数据
    x1 = np.random.randn(n_samples)
    x2 = np.random.uniform(0, 100, n_samples)
    y_prob = 1 / (1 + np.exp(-(x1 * 0.5 + x2 * 0.02 - 2)))
    y = pd.Series(np.random.binomial(1, y_prob, n_samples))

    X = pd.DataFrame({"feature1": x1, "feature2": x2, "feature3": np.random.choice(["A", "B", "C", "D"], n_samples)})

    # 添加一些缺失值
    X.loc[np.random.choice(n_samples, 50, replace=False), "feature1"] = np.nan

    print("=" * 50)
    print("卡方分箱测试 (ChiMerge)")
    print("=" * 50)

    # 测试卡方分箱
    binner = ChiMergeBinning(max_n_bins=5, significance_level=0.05, verbose=True)
    binner.fit(X, y)

    print("\n分箱统计表 (feature1):")
    print(binner.get_bin_table("feature1"))

    print("\n分箱统计表 (feature2):")
    print(binner.get_bin_table("feature2"))

    print("\n分箱统计表 (feature3):")
    print(binner.get_bin_table("feature3"))

    # 转换测试
    print("\n转换测试:")
    X_binned = binner.transform(X, metric="indices")
    print("\n分箱索引:")
    print(X_binned.head())

    X_woe = binner.transform(X, metric="woe")
    print("\nWOE值:")
    print(X_woe.head())

    print("\n切分点:")
    for feature, splits in binner.splits_.items():
        print(f"  {feature}: {splits}")

    print(f"\n卡方阈值: {binner.min_chi2_threshold:.4f}")
