"""等距分箱算法.

基于数值范围等距切分的分箱方法，适用于均匀分布的数据。
"""

from typing import Union, List, Dict, Optional, Any, Sequence
import numpy as np
import pandas as pd

from ...exceptions import NotFittedError
from .base import BaseBinning


class UniformBinning(BaseBinning):
    """等距分箱.

    将特征值的范围等分为指定数量的区间，每个区间宽度相同。
    适用于数据分布相对均匀的场景。

    **参数**

    :param constraint_policy: 约束冲突策略，默认 ``"raise"`` 报错；``"legacy"`` 在约束不可行时
        保留原生尽力满足的规则（优化器无可行分区时可回退单箱）。不放宽输入校验或吞掉回调错误。
        实际未满足项记录于 ``constraint_violations_``，不增加业务统计表列。
    :type constraint_policy: str

    :param kwargs: 传递给 BaseBinning 的附加构造参数，例如 ``decimal``、``woe_clip``；
        与显式参数重名或不存在的参数会被构造函数拒绝。

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

    :type user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, Sequence[bool]]]]]

    :param max_n_bins: 最大分箱数，默认为5
    :type max_n_bins: int
    :param min_n_bins: 最小分箱数，默认为2
    :type min_n_bins: int
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01；None表示不限制
    :type min_bin_size: Optional[Union[float, int]]
    :param max_bin_size: 每箱最大样本占比，默认为None
    :type max_bin_size: Optional[Union[float, int]]
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :type min_bad_rate: float
    :param monotonic: 是否要求单调性，默认为False
    :type monotonic: Union[bool, str]
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :type missing_separate: bool
    :param special_codes: 特殊值列表，默认为None，如[-999, -98]
    :type special_codes: Optional[List]
    :param left_clip: 左侧截断分位数，默认为None，如0.01表示截断1%分位数以下的值
    :type left_clip: Optional[float]
    :param right_clip: 右侧截断分位数，默认为None，如0.99表示截断99%分位数以上的值
    :type right_clip: Optional[float]
    :param force_numerical: 是否强制作为数值型处理，默认为False（自动识别类别型）

        - True: 将所有特征视为数值型进行等距分箱（默认，因为等距分箱适用于数值型）

        - False: 自动检测特征类型（根据dtype判断）

    :type force_numerical: bool
    :param random_state: 随机种子，默认为None
    :type random_state: Optional[int]
    :param target: 使用完整数据框拟合时的目标列名，默认为 ``"target"``
    :type target: str
    :param verbose: 是否输出详细信息，默认为False
    :type verbose: Union[bool, int]
    :param decimal: 数值切点保留的小数位数，默认为4
    :type decimal: int
    :param woe_clip: WOE 绝对值截断上限，默认为None
    :type woe_clip: Optional[float]

    **参考样例**

    >>> from hscredit.core.binning import UniformBinning
    >>> # 基础用法
    >>> binner = UniformBinning(max_n_bins=5)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>>
    >>> # 使用截断处理异常值
    >>> binner = UniformBinning(max_n_bins=5, left_clip=0.01, right_clip=0.99)
    >>> binner.fit(X_train, y_train)
    >>>
    >>> # 指定特殊值
    >>> binner = UniformBinning(max_n_bins=5, special_codes=[-999, -98])
    >>> binner.fit(X_train, y_train)

    **注意**

    等距分箱的特点:

    1. 每个分箱的区间宽度相同

    2. 分箱边界由 (max - min) / n_bins 计算得出

    3. 支持通过left_clip/right_clip截断异常值

    4. 支持通过special_codes处理特殊值（如-999表示缺失）

    5. 默认force_numerical=False，自动识别类别型；如需强制数值等距分箱可显式设为 True

    6. 计算速度快，实现简单；为无监督方法，不使用标签 ``y`` 决定切分

    7. 对偏态分布或含极端值的特征不友好（可能某些箱样本极少），此时优先用等频分箱

    **引用**

    等距（equal-width）离散化综述见 Dougherty, J., Kohavi, R., & Sahami, M. (1995).
    *Supervised and Unsupervised Discretization of Continuous Features.* ICML-95.
    https://ai.stanford.edu/~ronnyk/disc.pdf
    """

    def __init__(
        self,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Optional[Union[float, int]] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        category_order=None,
        handle_unknown: Union[int, str] = -3,
        left_clip: Optional[float] = None,
        right_clip: Optional[float] = None,
        force_numerical: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        user_splits: Optional[Dict[str, List]] = None,
        user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, Sequence[bool]]]]] = None,
        target: str = "target",
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        woe_clip: Optional[float] = None,
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
            verbose=verbose,
            decimal=decimal,
            woe_clip=woe_clip,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            **kwargs,
            constraint_policy=constraint_policy,
        )
        self.left_clip = left_clip
        self.right_clip = right_clip
        self.force_numerical = force_numerical

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "UniformBinning":
        """拟合等距分箱.

        :param kwargs: 其他参数，传递给具体的分箱算法

        :param X: 训练数据
        :type X: Union[pd.DataFrame, np.ndarray]
        :param y: 默认值为 ``None`` 。目标变量
        :type y: Optional[Union[pd.Series, np.ndarray]]
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        self._fit_features(X, y, "_fit_feature")

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
        if self.force_numerical:
            feature_type = "numerical"
        else:
            feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        # 处理缺失值和特殊值
        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        # 获取有效数据（非缺失、非特殊值）
        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]

        if feature_type == "categorical":
            # 类别型变量：每个类别作为一个箱
            unique_values = X_valid.unique()
            self.splits_[feature] = np.array([])  # 类别型没有数值切分点
            self.n_bins_[feature] = len(unique_values)
        else:
            # 数值型变量：等距分箱
            # 转换为数值型，确保正确处理
            X_numeric = pd.to_numeric(X_valid, errors="coerce")
            X_numeric = X_numeric.dropna()

            if len(X_numeric) == 0:
                # 没有有效数值数据
                self.splits_[feature] = np.array([])
                self.n_bins_[feature] = 1
            else:
                # 应用截断（如果指定）
                min_val = X_numeric.min()
                max_val = X_numeric.max()

                # 保存原始边界用于后续处理
                clip_lower = None
                clip_upper = None

                if self.left_clip is not None and 0 <= self.left_clip < 1:
                    clip_lower = X_numeric.quantile(self.left_clip)
                    min_val = clip_lower

                if self.right_clip is not None and 0 < self.right_clip <= 1:
                    clip_upper = X_numeric.quantile(self.right_clip)
                    max_val = clip_upper

                # 保存截断边界
                self.clip_bounds_ = getattr(self, "clip_bounds_", {})
                self.clip_bounds_[feature] = (clip_lower, clip_upper)

                # 处理边界相同的情况（所有值相等）
                if max_val == min_val:
                    self.splits_[feature] = np.array([])
                    self.n_bins_[feature] = 1
                else:
                    bin_width = (max_val - min_val) / self.max_n_bins
                    splits = [min_val + i * bin_width for i in range(1, self.max_n_bins)]
                    self.splits_[feature] = self._round_splits(splits)
                    self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _assign_bins(self, X: pd.Series, feature: str) -> np.ndarray:
        """为数据分配分箱索引.

        :param X: 特征数据
        :param feature: 特征名
        :return: 分箱索引数组
        """
        if self.feature_types_[feature] == "categorical" and feature in self._cat_bins_:
            return self._assign_categorical_bins(feature, X)
        if self.feature_types_[feature] == "categorical":
            # 类别型：使用类别编码
            return pd.Categorical(X).codes
        else:
            # 数值型：使用切分点
            splits = self.splits_[feature]

            # 获取截断边界
            clip_lower, clip_upper = None, None
            if hasattr(self, "clip_bounds_") and feature in self.clip_bounds_:
                clip_lower, clip_upper = self.clip_bounds_[feature]

            # 处理缺失值和特殊值
            bins = np.zeros(len(X), dtype=int)

            for i, val in enumerate(X):
                if pd.isna(val):
                    bins[i] = -1  # 缺失值
                elif self.special_codes and val in self.special_codes:
                    bins[i] = -2  # 特殊值
                else:
                    # 尝试转换为数值
                    try:
                        val_numeric = float(val)
                    except (ValueError, TypeError):
                        # 无法转换为数值，使用哈希分配
                        val_numeric = hash(val) % (2**31)

                    # 应用截断
                    if clip_lower is not None and val_numeric < clip_lower:
                        val_numeric = clip_lower
                    if clip_upper is not None and val_numeric > clip_upper:
                        val_numeric = clip_upper

                    # 找到对应的分箱
                    if len(splits) == 0:
                        bin_idx = 0
                    else:
                        bin_idx = np.searchsorted(splits, val_numeric, side="right")
                    bins[i] = bin_idx

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

        >>> binner = UniformBinning()
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
            if isinstance(X, np.ndarray) and X.ndim == 1:
                X = pd.DataFrame(X, columns=list(self.feature_names_in_))
            else:
                X = pd.DataFrame(X, columns=list(self.feature_names_in_))
        else:
            unknown = [feature for feature in X.columns if feature not in self.splits_]
            if unknown:
                raise KeyError(f"转换数据包含未拟合特征: {unknown}")

        return self._transform_binning_features(
            X, metric, lambda feature: self._assign_bins(X[feature], feature), woe_default=0.0
        )
