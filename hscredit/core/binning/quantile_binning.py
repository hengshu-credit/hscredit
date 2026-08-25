"""等频分箱算法.

基于分位数切分的分箱方法，确保每个箱的样本数大致相等。
适用于数据分布不均匀或存在异常值的场景。
"""

import logging
from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from ...exceptions import NotFittedError
from .base import BaseBinning

logger = logging.getLogger(__name__)


class QuantileBinning(BaseBinning):
    """等频分箱算法.

    将特征值按照分位数切分成多个区间，确保每个区间的样本数大致相等。
    适用于数据分布不均匀或存在异常值的场景。

    :param min_n_bins: 最小分箱数，默认为2
    :param max_n_bins: 最大分箱数，默认为10
    :param quantiles: 自定义分位点列表，如[0, 0.2, 0.5, 0.8, 1.0]，默认为None
        - 如果提供，将直接使用这些分位点进行分箱
        - 首尾的 0 与 1 支持自动补齐：可传入完整的 [0, ..., 1]，
          也可只传中间分位点（如 [0.2, 0.5, 0.8]），缺失的 0 / 1 会自动补齐
    :param force_numerical: 是否强制作为数值型处理，默认为False（自动识别类别型）
        - True: 将所有特征视为数值型进行等频分箱
        - False: 自动检测特征类型
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01；None表示不限制
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求坏样本率单调，默认为False
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False
    :param decimal: 数值切点保留的小数位数，默认为4
    :param woe_clip: WOE 绝对值截断上限，默认为None

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表

    **参考样例**

    >>> from hscredit.core.binning import QuantileBinning
    >>> # 基础用法
    >>> binner = QuantileBinning(max_n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    >>>
    >>> # 使用自定义分位点
    >>> binner = QuantileBinning(quantiles=[0, 0.1, 0.3, 0.7, 0.9, 1.0])
    >>> binner.fit(X, y)

    **注意**

    等频分箱为无监督方法，仅依据特征自身分布切分、不使用标签 ``y``（``y`` 仅用于
    生成分箱统计表），因此对异常值稳健、各箱样本量均衡，常用作有监督分箱的预分箱。
    当多个分位点落在同一个重复值上时会合并重复边界，因此实际箱数可能少于
    ``max_n_bins``，但不会生成空箱。

    **引用**

    等频（equal-frequency）离散化综述见 Dougherty, J., Kohavi, R., & Sahami, M.
    (1995). *Supervised and Unsupervised Discretization of Continuous Features.*
    ICML-95. https://ai.stanford.edu/~ronnyk/disc.pdf
    """

    def __init__(
        self,
        target: str = "target",
        min_n_bins: int = 2,
        max_n_bins: int = 10,
        quantiles: Optional[List[float]] = None,
        force_numerical: bool = False,
        min_bin_size: Optional[Union[float, int]] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        cat_cutoff: Optional[Union[float, int]] = None,
        category_order=None,
        handle_unknown: Union[int, str] = -3,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        woe_clip: Optional[float] = None,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        user_splits: Optional[Dict[str, List]] = None,
        user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, List[bool]]]]] = None,
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
            user_splits=user_splits,
            user_splits_fixed=user_splits_fixed,
            category_order=category_order,
            handle_unknown=handle_unknown,
            random_state=random_state,
            verbose=verbose,
            decimal=decimal,
            woe_clip=woe_clip,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.force_numerical = force_numerical

        # 保留构造参数原值以兼容 sklearn.clone；标准化结果在 fit 时保存到 quantiles_。
        self.quantiles = quantiles

    @staticmethod
    def _normalize_quantiles(quantiles: Optional[List[float]]) -> Optional[List[float]]:
        """校验并标准化自定义分位点。

        首尾的 0 与 1 支持自动补齐：既可显式传入完整的 ``[0, ..., 1]``，
        也可只传入中间分位点（如 ``[0.01, 0.5, 0.99]``），缺失的 0 / 1 会自动补齐。

        :param quantiles: 自定义分位点，None 表示不使用
        :return: 标准化后的分位点列表（已补齐首尾 0 / 1），None 时原样返回
        """
        if quantiles is None:
            return None

        if not isinstance(quantiles, (list, tuple, np.ndarray)):
            raise ValueError("quantiles必须是列表或数组")

        quantiles = [float(q) for q in quantiles]
        if len(quantiles) == 0:
            raise ValueError("quantiles不能为空")
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("quantiles所有元素必须在[0, 1]范围内")
        if not all(quantiles[i] <= quantiles[i + 1] for i in range(len(quantiles) - 1)):
            raise ValueError("quantiles必须是非递减的")

        # 自动补齐首尾的 0 / 1
        if quantiles[0] != 0:
            quantiles = [0.0] + quantiles
        if quantiles[-1] != 1:
            quantiles = quantiles + [1.0]

        if len(quantiles) < 2:
            raise ValueError("quantiles至少需要2个元素")
        return quantiles

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "QuantileBinning":
        """拟合等频分箱.

        :param X: 训练数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1)
        :param kwargs: 其他参数
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)
        self.quantiles_ = self._normalize_quantiles(self.quantiles)

        self._fit_features(X, y, "_fit_feature")

        # 分箱表、reserved-bin 和 WOE 统一在最终收口中一次生成，避免先按
        # 中间状态统计后又被相同最终状态覆盖。
        self._finalize_categorical_fit(build_stats=False)
        self._finalize_reserved_bins(X, y)
        for feature, feature_type in self.feature_types_.items():
            if feature_type == "categorical":
                self._validate_categorical_constraints(feature, y)
        self._is_fitted = True
        return self

    def _fit_feature(self, feature: str, X: pd.Series, y: pd.Series) -> None:
        """拟合单个特征。"""
        if self.verbose:
            logger.info(f"处理特征: {feature}")
        feature_type = "numerical" if self.force_numerical else self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type
        if feature_type == "categorical":
            splits = self._fit_categorical(X, y)
            self.splits_[feature] = splits
        else:
            splits = self._round_splits(self._fit_numerical(X, y))
            if self.quantiles is not None and len(splits) > 0:
                splits = np.unique(splits)
            if self.monotonic not in [False, None, "none"] and len(splits) > 0:
                from .monotonic_binning import MonotonicBinning

                mono = MonotonicBinning(
                    monotonic=self.monotonic,
                    max_n_bins=self.max_n_bins,
                    min_n_bins=self.min_n_bins,
                    min_bin_size=self.min_bin_size,
                    special_codes=self.special_codes,
                    missing_separate=self.missing_separate,
                    random_state=self.random_state,
                    verbose=False,
                )
                clean = X.dropna()
                splits = mono._ensure_monotonic(
                    clean, y.loc[clean.index], splits, mono._detect_monotonic_mode(clean, y.loc[clean.index], splits)
                )
                splits = self._round_splits(splits)
            self.splits_[feature] = splits
        self.n_bins_[feature] = len(splits) + 1

    def _fit_numerical(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """对数值型特征进行等频分箱.

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

        # 使用自定义分位点或基于max_n_bins计算
        if self.quantiles is not None:
            # 使用自定义分位点：直接按用户指定的分位点切分，
            # 不受 min_n_bins / max_n_bins / min_bin_size 的二次约束（见类文档）
            quantiles_to_use = np.asarray(self.quantiles_, dtype=float)
            target_n_bins = len(quantiles_to_use) - 1
        else:
            # 基于max_n_bins计算分位点，并约束在 [min_n_bins, max_n_bins] 范围内
            target_n_bins = max(self.min_n_bins, min(self.max_n_bins, self.max_n_bins))
            quantiles_to_use = np.linspace(0, 1, target_n_bins + 1)

        # 获取唯一值
        unique_values = np.unique(x_valid.to_numpy(dtype=float))

        # 默认分位数下，唯一值不多于目标箱数时，每个唯一值各占一箱。
        # 数值分箱统一使用左闭右开区间，因此切点必须落在相邻取值之间；
        # 直接使用 unique_values[:-1] 会让最小值切点产生一个空的首箱。
        if self.quantiles is None and len(unique_values) <= target_n_bins:
            return unique_values[:-1] + (unique_values[1:] - unique_values[:-1]) / 2

        # 先计算包含最小值/最大值的完整分位边界，再整体去重并移除两端。
        # 这与 qcut(duplicates="drop") 的边界语义一致：当多个分位点都落在
        # 最小值等重复值上时，不能把最小值本身保留为左闭切点。
        quantile_edges = np.percentile(x_valid, np.asarray(quantiles_to_use) * 100)
        unique_edges = np.unique(quantile_edges)
        if len(unique_edges) <= 2:
            return np.array([])
        splits = unique_edges[1:-1]

        if self.quantiles is not None:
            # 自定义分位点：完整边界已去除离散重复值导致的相同切分点，
            # 不进行 max_n_bins / min_bin_size 的合并裁剪，确保严格按指定分位点切分
            return splits
        else:
            # 根据约束调整分箱数
            splits = self._adjust_bins(x_valid, y_valid, splits)

        return splits

    def _handle_duplicate_boundaries(self, splits: np.ndarray, x: pd.Series) -> np.ndarray:
        """处理重复值边界问题.

        当分位数切分点与数据中的重复值重合时，调整切分点以避免空箱。

        :param splits: 初始切分点
        :param x: 特征数据
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        x_values = x.values
        unique_splits = []
        min_samples = self._get_min_samples(len(x))

        for i, split in enumerate(splits):
            # 如果当前切分点与已有切分点相同或更小，需要调整
            if i > 0 and len(unique_splits) > 0 and split <= unique_splits[-1]:
                # 找到该值的下一个不同值
                larger_values = x_values[x_values > unique_splits[-1]]
                if len(larger_values) > 0:
                    next_value = np.min(larger_values)
                    # 使用中间值，但要确保大于上一个切分点
                    split = max(split, (unique_splits[-1] + next_value) / 2)
                    # 确保不会等于上一个切分点
                    if split <= unique_splits[-1]:
                        split = unique_splits[-1] + 1e-10
                else:
                    # 如果没有更大的值，跳过这个切分点
                    continue

            # 检查该切分点是否会导致空箱或样本数过少
            if len(unique_splits) == 0:
                count = np.sum(x_values <= split)
            else:
                count = np.sum((x_values > unique_splits[-1]) & (x_values <= split))

            if count >= min_samples:
                unique_splits.append(split)

        return np.array(unique_splits)

    def _fit_categorical(self, x: pd.Series, y: pd.Series) -> np.ndarray:
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
        cat_stats = (
            pd.DataFrame({"category": x_valid, "target": y_valid}).groupby("category")["target"].agg(["mean", "count"])
        )

        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(x_valid))
        cat_stats = cat_stats[cat_stats["count"] >= min_samples]

        # 按坏样本率排序
        cat_stats = cat_stats.sort_values("mean")

        # 返回排序后的类别列表
        return cat_stats.index.tolist()

    def _adjust_bins(self, x: pd.Series, y: pd.Series, splits: np.ndarray) -> np.ndarray:
        """根据约束条件调整分箱.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        min_samples = self._get_min_samples(len(x))
        splits = np.unique(np.asarray(splits, dtype=float))
        values = x.to_numpy(dtype=float)

        # 按 transform 的左闭右开语义检查真实箱计数。发现小箱时仅删除其相邻
        # 边界，每轮合并一个箱，避免旧实现隔一个切点删除一次并最终塌缩为两箱。
        while len(splits) > 0:
            bin_indices = np.searchsorted(splits, values, side="right")
            counts = np.bincount(bin_indices, minlength=len(splits) + 1)
            small_bins = np.flatnonzero(counts < min_samples)
            if len(small_bins) == 0:
                break

            small_bin = int(small_bins[np.argmin(counts[small_bins])])
            if small_bin == 0:
                split_to_remove = 0
            elif small_bin == len(counts) - 1:
                split_to_remove = len(splits) - 1
            elif counts[small_bin - 1] <= counts[small_bin + 1]:
                split_to_remove = small_bin - 1
            else:
                split_to_remove = small_bin
            splits = np.delete(splits, split_to_remove)

        return splits

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数.

        :param n_total: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size is None:
            return 0
        if self.min_bin_size < 1:
            return max(1, int(n_total * self.min_bin_size))
        return max(1, int(self.min_bin_size))

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
            # 类别型特征
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
                bins[x == cat] = i
            bins[x.isna()] = -1  # 缺失值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2  # 特殊值
            return bins
        else:
            # 数值型特征
            bins = np.zeros(len(x), dtype=int)

            # 处理缺失值
            if self.missing_separate:
                bins[x.isna()] = -1

            # 处理特殊值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2

            # 正常值分箱
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
        >>> binner = QuantileBinning()
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

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=["feature"])
                else:
                    X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        return self._transform_binning_features(
            X,
            metric,
            lambda feature: self._apply_bins(X[feature], self.splits_[feature]),
        )


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_samples = 1000

    # 生成测试数据（偏态分布）
    X = pd.DataFrame(
        {
            "feature1": np.random.exponential(2, n_samples),  # 指数分布
            "feature2": np.random.beta(2, 5, n_samples) * 100,  # Beta分布
            "feature3": np.random.choice(["A", "B", "C", "D"], n_samples),
        }
    )
    y = pd.Series(np.random.binomial(1, 0.3, n_samples))

    # 添加一些缺失值和重复值
    X.loc[np.random.choice(n_samples, 50, replace=False), "feature1"] = np.nan
    # 添加大量重复值测试边界处理
    X.loc[np.random.choice(n_samples, 200, replace=False), "feature2"] = 50

    print("=" * 50)
    print("等频分箱测试")
    print("=" * 50)

    # 创建分箱器
    binner = QuantileBinning(max_n_bins=5, verbose=True)

    # 拟合
    binner.fit(X, y)

    # 转换
    X_binned = binner.transform(X, metric="indices")
    print("\n分箱索引:")
    print(X_binned.head())

    X_woe = binner.transform(X, metric="woe")
    print("\nWOE值:")
    print(X_woe.head())

    # 查看分箱统计
    print("\n分箱统计表 (feature1):")
    print(binner.get_bin_table("feature1"))

    print("\n分箱统计表 (feature2):")
    print(binner.get_bin_table("feature2"))

    print("\n分箱统计表 (feature3):")
    print(binner.get_bin_table("feature3"))

    print("\n切分点:")
    for feature, splits in binner.splits_.items():
        print(f"  {feature}: {splits}")

    # 验证等频特性
    print("\n各箱样本数统计 (feature2):")
    print(X_binned["feature2"].value_counts().sort_index())
