"""分箱算法基类.

定义分箱算法的统一接口和通用方法。
所有分箱算法都继承此类，确保API的一致性。

设计原则:
1. 参数命名统一，与其他库保持一致
2. 支持高度自定义，但提供合理默认值
3. 遵循sklearn API风格
4. 使用 core.metrics 中的指标计算方法
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils.misc import round_float

# 从 metrics 导入指标计算方法
from ..metrics.binning_metrics import (
    compute_bin_stats,
    woe_iv_vectorized,
    iv_for_splits,
    ks_for_splits,
    compare_splits_iv,
    compare_splits_ks,
)


class BaseBinning(BaseEstimator, TransformerMixin, ABC):
    """分箱算法基类.

    所有分箱算法都继承此类，实现统一的fit/transform接口。
    支持16种分箱方法，适用于风控评分卡开发场景。

    **参数**

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param min_n_bins: 最小分箱数，默认为2
    :param max_n_bins: 最大分箱数，默认为5
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，用于避免极端情况，默认为0.0
    :param monotonic: 坏样本率单调性约束，默认为False
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向(递增或递减)
        - 'ascending': 强制坏样本率递增(分箱索引增大时坏样本率增大)
        - 'descending': 强制坏样本率递减(分箱索引增大时坏样本率减小)
        - 'auto_heuristic': 使用启发式方法自动确定单调方向
        - 'peak' 或 'valley': 允许单峰或单谷形态,即先升后降或先降后升
    :param special_codes: 特殊值列表，这些值会被单独分箱，例如[-99, -98, 'missing']
    :param split_points: 自定义分箱切分点，例如{'age': [25, 35, 45, 55]}
    :param cat_cutoff: 类别型变量处理阈值，默认为None
        - 如果 < 1, 表示保留占比超过该值的类别
        - 如果 >= 1, 表示保留频率最高的N个类别
    :param user_splits: 用户自定义分箱规则，例如{'feature': [0, 10, 20, 30]}
    :param random_state: 随机种子，用于可复现性，默认为None
    :param n_jobs: 并行计算的任务数，默认为1
        - 1: 单进程
        - -1: 使用所有CPU
        - n: 使用n个进程
    :param verbose: 是否输出详细信息，默认为False
    :param decimal: 数值型切分点小数点保留精度，默认为4

    **属性**

    - splits_: 每个特征的分箱切分点，数值型特征为numpy数组，类别型特征为列表
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表，包含中文列名:
        - 分箱: 分箱索引
        - 分箱标签: 分箱区间标签
        - 样本总数: 样本数
        - 样本占比: 样本占比
        - 好样本数: 好样本数
        - 坏样本数: 坏样本数
        - 坏样本率: 坏样本率
        - 分档WOE值: WOE值
        - 分档IV值: IV值
        - 指标IV值: 总IV值
        - LIFT值: Lift值
        - 坏账改善: 坏账改善
        - 累积LIFT值: 累积Lift值
        - 累积坏账改善: 累积坏账改善
        - 累积好样本数: 累积好样本数
        - 累积坏样本数: 累积坏样本数
        - 分档KS值: KS值
    - feature_types_: 每个特征的类型 ('numerical' 或 'categorical')

    **支持的分箱方法**

    | 方法 | 类名 | 说明 |
    |------|------|------|
    | uniform | UniformBinning | 等宽分箱，将数值范围等分 |
    | quantile | QuantileBinning | 等频分箱，每箱样本数相等 |
    | tree | TreeBinning | 决策树分箱，基于信息增益 |
    | chi | ChiMergeBinning | 卡方分箱，基于卡方统计量合并 |
    | best_ks | BestKSBinning | 最优KS分箱，最大化KS统计量 |
    | best_iv | BestIVBinning | 最优IV分箱，最大化IV值(推荐) |
    | mdlp | MDLPBinning | MDLP分箱，信息论方法 |
    | or_tools | ORBinning | 运筹规划分箱（基于Google OR-Tools） |
    | cart | CartBinning | CART分箱，参考optbinning实现 |
    | monotonic | MonotonicBinning | 单调性约束分箱，支持U型/倒U型 |
    | genetic | GeneticBinning | 遗传算法分箱，全局优化 |
    | smooth | SmoothBinning | 平滑分箱，正则化方法 |
    | kernel_density | KernelDensityBinning | 核密度分箱，密度估计 |
    | best_lift | BestLiftBinning | Best Lift分箱，提升度优化 |
    | target_bad_rate | TargetBadRateBinning | 目标坏样本率分箱 |
    | kmeans | KMeansBinning | K-Means聚类分箱 |
    | optimal | OptimalBinning | 统一接口，支持上述所有方法 |

    **参考样例**

    基本使用 (sklearn风格)::

        >>> from hscredit.core.binning import OptimalBinning
        >>> binner = OptimalBinning(method='best_iv', max_n_bins=5)
        >>> binner.fit(X, y)  # X是特征矩阵，y是目标变量
        >>> X_binned = binner.transform(X)
        >>> bin_table = binner.get_bin_table('feature_name')

    scorecardpipeline风格 (目标列在DataFrame中)::

        >>> from hscredit.core.binning import OptimalBinning
        >>> # 初始化时指定目标列名，fit时传入完整DataFrame
        >>> binner = OptimalBinning(target='target', method='best_iv', max_n_bins=5)
        >>> binner.fit(df)  # df包含特征列和目标列'target'
        >>> X_binned = binner.transform(df.drop(columns=['target']))
        >>> bin_table = binner.get_bin_table('feature_name')

    混合风格 (y参数优先)::

        >>> # 即使初始化时指定了target，fit时传入y会优先使用y
        >>> binner = OptimalBinning(target='target', method='best_iv')
        >>> binner.fit(df, y=external_y)  # 使用external_y，忽略df中的'target'列

    设置切分点精度::

        >>> # 默认4位小数
        >>> binner = OptimalBinning(method='best_iv', decimal=4)
        >>> # 设置为2位小数
        >>> binner = OptimalBinning(method='best_iv', decimal=2)

    单调性约束::

        >>> binner = OptimalBinning(method='best_iv', monotonic='descending')
        >>> binner.fit(X, y)

    使用独立分箱类::

        >>> from hscredit.core.binning import ChiMergeBinning, BestIVBinning
        >>> chi_binner = ChiMergeBinning(max_n_bins=5)
        >>> chi_binner.fit(X, y)

    **注意**

    分箱算法的一般流程:
    1. fit(): 训练分箱模型
       - 数据预处理 (缺失值处理、特殊值处理)
       - 检测特征类型 (数值型/类别型)
       - 计算最优分箱切分点
       - 对数值型切分点进行四舍五入(精度由decimal参数控制)
       - 生成分箱统计表

    2. transform(): 应用分箱
       - 根据切分点对数据进行分箱
       - 支持多种输出格式: 'indices'(分箱索引), 'labels'(分箱标签),
         'woe'(WOE值), 'bin_code'(分箱编码)
    """

    def __init__(
        self,
        target: str = 'target',
        missing_separate: bool = True,
        min_n_bins: int = 2,
        max_n_bins: int = 5,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        split_points: Optional[Dict[str, List]] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        user_splits: Optional[Dict[str, List]] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
    ):
        self.target = target
        self.missing_separate = missing_separate
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bad_rate = min_bad_rate
        self.monotonic = monotonic
        self.special_codes = special_codes
        self.split_points = split_points
        self.cat_cutoff = cat_cutoff
        self.user_splits = user_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.decimal = decimal

        # 拟合后的属性
        self.splits_ = {}
        self.n_bins_ = {}
        self.bin_tables_ = {}
        self.feature_types_ = {}
        self._cat_bins_ = {}  # 类别型变量的分组信息，格式: {'feature': [['A', 'B'], ['C'], [np.nan]]}
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'BaseBinning':
        """拟合分箱。

        支持两种API风格：
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级规则：如果y不是None，直接使用y（优先）；否则从X中提取target列。

        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)，可以是数值型或类别型特征
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
            - 支持DataFrame或numpy数组
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量，必须是二分类 (0/1 或 False/True)
            - scorecardpipeline风格: 不传，从X中提取
            - 如果传入y，优先使用y而忽略X中的target列
        :param kwargs: 其他参数，传递给具体的分箱算法
        :return: 拟合后的分箱器

        **注意**

        fit方法会进行以下操作:
        1. 数据验证和预处理（通过_check_input方法）
        2. 识别特征类型 (数值型/类别型)
        3. 处理缺失值和特殊值
        4. 计算最优分箱切分点
        5. 生成分箱统计表

        **使用示例**

        sklearn风格::

            >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [5000, 6000, 7000]})
            >>> y = pd.Series([0, 1, 0])
            >>> binner.fit(X, y)

        scorecardpipeline风格::

            >>> df = pd.DataFrame({
            ...     'age': [25, 30, 35],
            ...     'income': [5000, 6000, 7000],
            ...     'target': [0, 1, 0]
            ... })
            >>> binner = OptimalBinning(target='target')
            >>> binner.fit(df)  # 自动从df中提取'target'列

        混合风格（y参数优先）::

            >>> binner = OptimalBinning(target='target')
            >>> binner.fit(df, y=external_y)  # 使用external_y，忽略df中的'target'列
        """
        pass

    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.
        
        将原始特征值转换为分箱索引、分箱标签或WOE值。
        这是分箱器的核心方法，用于将新数据应用到已训练的分箱规则。

        :param X: 待转换的数据，shape (n_samples, n_features)
            - 支持DataFrame或numpy数组
            - 列名必须与fit时的特征名一致
        :param metric: 转换类型，默认为'indices'
            - 'indices': 返回分箱索引 (0, 1, 2, ...)
                * 用途: 后续处理、特征工程
                * 示例: [0, 1, 2, 0, 1, ...]
            - 'bins': 返回分箱标签字符串
                * 用途: 可视化、报告展示
                * 示例: ['(-inf, 25]', '(25, 35]', '(35, 45]', ...]
                * 类别型: ['北京,上海', '广州,深圳', ...]
            - 'woe': 返回WOE值
                * 用途: 逻辑回归建模
                * 示例: [0.234, -0.456, 0.123, ...]
        :param kwargs: 其他参数(保留兼容性)
        :return: 转换后的数据，返回类型与输入类型一致
        
        **重要说明**
        
        1. metric参数是枚举值，只能使用以下3个值之一:
           - 'indices' (不是'分箱'、'索引'等中文)
           - 'bins' (不是'分箱标签'等中文)
           - 'woe' (不是'分档WOE值'等中文)
        
        2. 中文列名出现在分箱结果表中:
           - 使用 binner.get_bin_table(feature) 查看
           - 列名: '分箱', '样本总数', '坏样本率', '分档WOE值'等
        
        **示例**
        
        >>> binner = OptimalBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> print(X_binned.head())
        >>> 
        >>> # 获取分箱标签
        >>> X_labels = binner.transform(X_test, metric='bins')
        >>> print(X_labels.head())
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        >>> print(X_woe.head())
        >>> 
        >>> # 错误示例 - 不要使用中文
        >>> # X_error = binner.transform(X_test, metric='分档WOE值')  # ❌ ValueError!
        
        **处理特殊值**
        
        transform方法会自动处理:
        1. 缺失值: 如果missing_separate=True，分配到专门的缺失箱 (索引=-1)
        2. 特殊值: 如果指定了special_codes，分配到专门的特殊值箱 (索引=-2)
        3. 超出范围的值: 分配到最近的分箱
        """
        pass

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并应用分箱。

        支持两种API风格：
        1. sklearn风格: fit_transform(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit_transform(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取
        :param metric: 返回值的类型，默认为'indices'
            - 'indices': 返回分箱索引
            - 'bins': 返回分箱标签
            - 'woe': 返回WOE值
        :return: 分箱后的数据

        **使用示例**

        sklearn风格::

            >>> X_binned = binner.fit_transform(X, y, metric='woe')

        scorecardpipeline风格::

            >>> binner = OptimalBinning(target='target')
            >>> X_binned = binner.fit_transform(df, metric='woe')  # 自动提取target列
        """
        return self.fit(X, y, **kwargs).transform(X, metric=metric, **kwargs)

    def _check_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """检查并准备输入数据，支持sklearn和scorecardpipeline两种API风格。

        该方法统一处理两种风格的输入：
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级规则：
        - 如果y不是None，直接使用y（优先）
        - 如果y是None且X是DataFrame，从X中提取target列

        :param X: 输入特征或完整数据框
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征和目标列
            - 支持DataFrame或numpy数组
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取
            - 支持pd.Series或numpy数组
        :return: (特征DataFrame, 目标Series)
            - 特征DataFrame: 纯特征数据，不含目标列
            - 目标Series: 二分类目标变量
        :raises ValueError: 如果输入数据格式不正确或目标变量不符合要求

        **使用示例**

        sklearn风格::

            >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [5000, 6000, 7000]})
            >>> y = pd.Series([0, 1, 0])
            >>> X_processed, y_processed = binner._check_input(X, y)

        scorecardpipeline风格::

            >>> df = pd.DataFrame({
            ...     'age': [25, 30, 35],
            ...     'income': [5000, 6000, 7000],
            ...     'target': [0, 1, 0]
            ... })
            >>> binner = OptimalBinning(target='target')
            >>> X_processed, y_processed = binner._check_input(df)  # y=None

        numpy数组输入::

            >>> X = np.array([[25, 5000], [30, 6000], [35, 7000]])
            >>> y = np.array([0, 1, 0])
            >>> X_processed, y_processed = binner._check_input(X, y)
        """
        # 保存原始索引，用于后续对齐
        original_index = None

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=['feature'])
                else:
                    # 为numpy数组生成默认列名 feature_0, feature_1, ...
                    n_cols = X.shape[1]
                    columns = [f'feature_{i}' for i in range(n_cols)]
                    X = pd.DataFrame(X, columns=columns)
            else:
                X = pd.DataFrame(X)

        original_index = X.index

        # 获取目标变量
        if y is not None:
            # sklearn风格: 使用传入的y（优先）
            if isinstance(y, np.ndarray):
                if y.ndim != 1:
                    raise ValueError(f"目标变量y必须是一维数组，但得到 {y.ndim} 维")
                y = pd.Series(y, index=original_index, name=self.target)
            elif isinstance(y, pd.Series):
                # 确保索引对齐
                if not y.index.equals(original_index):
                    y = y.reset_index(drop=True)
                    y.index = original_index
                y.name = self.target
            else:
                # 其他可迭代类型
                y = pd.Series(y, index=original_index, name=self.target)
        else:
            # scorecardpipeline风格: 从X中提取target列
            if self.target in X.columns:
                y = X[self.target].copy()
                y.name = self.target
                X = X.drop(columns=[self.target])
            else:
                raise ValueError(
                    f"目标变量 '{self.target}' 未在数据中找到。"
                    f"请提供y参数（sklearn风格）或在数据中包含 '{self.target}' 列（scorecardpipeline风格）。"
                    f"可用列: {list(X.columns)}"
                )

        # 验证数据长度
        if len(X) != len(y):
            raise ValueError(
                f"特征和标签数量不匹配: {len(X)} != {len(y)}"
            )

        # 验证目标变量
        unique_values = y.dropna().unique()
        if len(unique_values) != 2:
            raise ValueError(
                f"目标变量必须是二分类，但发现 {len(unique_values)} 个唯一值: {unique_values}"
            )

        if not set(unique_values).issubset({0, 1, False, True}):
            raise ValueError(
                f"目标变量必须是 0/1 或 False/True，但发现 {unique_values}"
            )

        return X, y

    def _get_min_samples(self, n_samples: int) -> int:
        """计算最小样本数.

        根据min_bin_size参数计算每个分箱的最小样本数。

        :param n_samples: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size < 1:
            # 比例值，如0.05表示5%
            return max(1, int(n_samples * self.min_bin_size))
        else:
            # 绝对数量
            return max(1, int(self.min_bin_size))

    def _round_splits(self, splits: Union[np.ndarray, list]) -> np.ndarray:
        """对数值型切分点进行四舍五入.

        使用 round_float 函数保留指定位数的小数进行四舍五入。

        :param splits: 切分点数组
        :return: 四舍五入后的切分点数组
        """
        if splits is None or len(splits) == 0:
            return np.array([]) if not isinstance(splits, np.ndarray) else splits

        # 转换为 numpy 数组
        if not isinstance(splits, np.ndarray):
            splits = np.array(splits)

        # 对每个切分点进行四舍五入，使用 self.decimal 指定精度
        rounded_splits = np.array([round_float(s, decimal=self.decimal) for s in splits])
        return rounded_splits

    def _detect_feature_type(self, data: Union[pd.Series, np.ndarray]) -> str:
        """检测特征类型.

        参考 scorecardpipeline (scp) 的 Combiner 实现:
        根据 dtype 判断特征类型，优先使用 str(dtype) 判断

        :param data: 特征数据，支持 pd.Series 或 np.ndarray
        :return: 特征类型，'numerical' 或 'categorical'

        **判断逻辑**

        1. 将输入转换为 pd.Series（如果是 np.ndarray）
        2. 如果 str(dtype) 是 "object", "string", "category"，认为是类别型
        3. 如果 dtype 是数值型，先判断是否为布尔型（bool 视为类别型）
        4. 数值型特征只有在明确设置 cat_cutoff 且满足条件时，才视为类别型
        5. 否则认为是数值型
        """
        # 统一转换为 pd.Series 处理
        if isinstance(data, np.ndarray):
            series = pd.Series(data)
        elif isinstance(data, pd.Series):
            series = data
        else:
            # 其他类型（如 list），转换为 pd.Series
            series = pd.Series(data)

        # 缺失值不计入
        series_valid = series.dropna()

        # 获取 dtype 字符串表示
        dtype_str = str(series.dtype)

        # 参考 scp: 如果 dtype 是 object/string/category，直接认为是类别型
        if dtype_str in ["object", "string", "category"]:
            return 'categorical'

        # 布尔型视为类别型
        if dtype_str == 'bool':
            return 'categorical'

        # 如果是数值型
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series_valid.nunique()
            n_valid = len(series_valid)
            unique_ratio = n_unique / n_valid if n_valid > 0 else 0

            # 只有在明确设置 cat_cutoff 时，才根据唯一值判断类型
            if self.cat_cutoff is not None:
                if self.cat_cutoff < 1:
                    # 如果唯一值比例小于等于阈值，认为是类别型
                    if unique_ratio <= self.cat_cutoff:
                        return 'categorical'
                else:
                    # 如果唯一值数量小于等于阈值，认为是类别型
                    if n_unique <= int(self.cat_cutoff):
                        return 'categorical'

            # 默认：数值型保持为数值型
            return 'numerical'

        # 默认认为是类别型
        return 'categorical'

    def _compute_bin_stats(
        self,
        feature: str,
        X: pd.Series,
        y: pd.Series,
        bins: np.ndarray
    ) -> pd.DataFrame:
        """计算分箱统计信息（使用metrics模块优化版本，中文列名）.

        :param feature: 特征名
        :param X: 特征数据
        :param y: 目标变量
        :param bins: 分箱索引
        :return: 分箱统计表，包含中文列名
        """
        # 生成分箱标签
        bin_labels = None
        if feature in self.splits_:
            splits = self.splits_[feature]
            feature_type = self.feature_types_.get(feature, 'numerical')
            
            # 检查是否为类别型变量的List[List]格式
            if feature_type == 'categorical' and isinstance(splits, list) and len(splits) > 0 and isinstance(splits[0], list):
                # List[List]格式：先获取unique bins，再生成标签
                unique_bins = np.unique(bins)
                bin_labels = []
                for bin_idx in unique_bins:
                    if bin_idx == -1:
                        bin_labels.append('missing')
                    elif bin_idx == -2:
                        bin_labels.append('special')
                    elif 0 <= bin_idx < len(splits):
                        group = splits[bin_idx]
                        if isinstance(group, list):
                            # 将np.nan转换为字符串"nan"
                            group_str = [str(v) if not (isinstance(v, float) and np.isnan(v)) else 'nan' 
                                        for v in group]
                            bin_labels.append(','.join(group_str))
                        else:
                            bin_labels.append(str(group))
                    else:
                        bin_labels.append(f'bin_{bin_idx}')
            else:
                # 数值型或字符串格式的类别型
                bin_labels = self._get_bin_labels(splits, bins)
        
        # 使用 metrics 模块的向量化计算，传入分箱标签
        bin_stats = compute_bin_stats(bins, y.values, bin_labels=bin_labels)
        
        # 如果没有分箱标签，生成默认标签
        if '分箱标签' not in bin_stats.columns:
            bin_stats['分箱标签'] = bin_stats['分箱'].apply(
                lambda x: f'bin_{x}' if x >= 0 else ('缺失' if x == -1 else 'special')
            )

        # 调整列顺序（将分箱标签放在分箱后面）
        chinese_columns = [
            '分箱', '分箱标签', '样本总数', '好样本数', '坏样本数',
            '样本占比', '好样本占比', '坏样本占比', '坏样本率',
            '分档WOE值', '分档IV值', '指标IV值',
            'LIFT值', '坏账改善', '累积LIFT值', '累积坏账改善',
            '累积好样本数', '累积坏样本数', '分档KS值'
        ]

        # 确保所有列都存在
        available_cols = [c for c in chinese_columns if c in bin_stats.columns]
        bin_stats = bin_stats[available_cols]

        return bin_stats

    def _get_bin_labels(
        self,
        splits: np.ndarray,
        bins: Optional[np.ndarray] = None
    ) -> List[str]:
        """根据切分点生成分箱标签.

        :param splits: 切分点数组
        :param bins: 分箱索引，用于处理缺失值和特殊值
        :return: 分箱标签列表
        """
        labels = []
        n_splits = len(splits) if splits is not None else 0

        # 如果有分箱索引，需要处理缺失值等特殊箱
        if bins is not None:
            unique_bins = np.unique(bins)
            n_bins = n_splits + 1

            for i in unique_bins:
                if i == -1:  # 缺失值箱
                    labels.append('missing')
                elif i == -2:  # 特殊值箱
                    labels.append('special')
                elif n_splits == 0:
                    # 没有切分点时，所有正常值在一个箱
                    labels.append('(-inf, +inf)')
                elif i < n_bins:
                    if i == 0:
                        labels.append(f'(-inf, {splits[i]}]')
                    elif i == n_bins - 1:
                        labels.append(f'({splits[i-1]}, +inf]')
                    else:
                        labels.append(f'({splits[i-1]}, {splits[i]}]')
                else:
                    labels.append(f'bin_{i}')
        else:
            # 只根据切分点生成标签
            if n_splits == 0:
                # 没有切分点时，只有一个箱
                labels.append('(-inf, +inf)')
            else:
                for i in range(n_splits + 1):
                    if i == 0:
                        labels.append(f'(-inf, {splits[i]}]')
                    elif i == n_splits:
                        labels.append(f'({splits[i-1]}, +inf]')
                    else:
                        labels.append(f'({splits[i-1]}, {splits[i]}]')

        return labels

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """获取指定特征的分箱表.

        :param feature: 特征名
        :return: 分箱统计表
        :raises ValueError: 如果分箱器尚未拟合
        :raises KeyError: 如果特征不存在
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.bin_tables_:
            raise KeyError(f"特征 '{feature}' 未找到")

        return self.bin_tables_[feature]

    def get_splits(self, feature: str) -> np.ndarray:
        """获取指定特征的切分点.

        :param feature: 特征名
        :return: 切分点数组
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.splits_:
            raise KeyError(f"特征 '{feature}' 未找到")

        return self.splits_[feature]

    def export_rules(self) -> Dict[str, Union[List, List[List]]]:
        """导出分箱规则.

        数值型变量返回切分点列表，类别型变量返回分组列表。

        :return: 分箱规则字典
            - 数值型: key为特征名，value为切分点列表，如 [25, 35, 45, 55]
            - 类别型: key为特征名，value为分组列表，如 [['A', 'B'], ['C'], [np.nan]]

        **参考样例**

        >>> binner = OptimalBinning()
        >>> binner.fit(X, y)
        >>> rules = binner.export_rules()
        >>> 
        >>> # 数值型变量
        >>> print(rules['age'])  # [25, 35, 45, 55]
        >>> 
        >>> # 类别型变量
        >>> print(rules['city'])  # [['北京', '上海'], ['广州', '深圳'], [np.nan]]
        >>> 
        >>> # 保存规则
        >>> import json
        >>> import numpy as np
        >>> 
        >>> # 处理np.nan以便JSON序列化
        >>> def convert_nan(obj):
        ...     if isinstance(obj, dict):
        ...         return {k: convert_nan(v) for k, v in obj.items()}
        ...     elif isinstance(obj, list):
        ...         return [convert_nan(item) for item in obj]
        ...     elif isinstance(obj, float) and np.isnan(obj):
        ...         return "NaN"
        ...     return obj
        >>> 
        >>> with open('binning_rules.json', 'w') as f:
        ...     json.dump(convert_nan(rules), f, indent=2)
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        rules = {}
        for feature in self.splits_:
            if self.feature_types_.get(feature) == 'categorical':
                # 类别型变量：返回分组列表
                if feature in self._cat_bins_:
                    # 将numpy数组转换为列表
                    bins = self._cat_bins_[feature]
                    rules[feature] = [
                        [item if not (isinstance(item, float) and np.isnan(item)) else np.nan 
                         for item in group] if isinstance(group, (list, np.ndarray)) else group
                        for group in bins
                    ]
                else:
                    # 如果没有分组信息，返回空列表
                    rules[feature] = []
            else:
                # 数值型变量：返回切分点列表
                splits = self.splits_[feature]
                rules[feature] = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
        
        return rules

    def import_rules(self, rules: Dict[str, Union[List, List[List]]]):
        """导入分箱规则.

        支持数值型切分点和类别型分组列表。

        :param rules: 分箱规则字典
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}

        **参考样例**

        >>> # 导入数值型规则
        >>> rules = {'age': [25, 35, 45, 55]}
        >>> binner.import_rules(rules)
        >>> 
        >>> # 导入类别型规则
        >>> rules = {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}
        >>> binner.import_rules(rules)
        >>> 
        >>> # 从JSON文件导入
        >>> import json
        >>> import numpy as np
        >>> 
        >>> def convert_nan_back(obj):
        ...     if isinstance(obj, dict):
        ...         return {k: convert_nan_back(v) for k, v in obj.items()}
        ...     elif isinstance(obj, list):
        ...         return [convert_nan_back(item) for item in obj]
        ...     elif obj == "NaN":
        ...         return np.nan
        ...     return obj
        >>> 
        >>> with open('binning_rules.json', 'r') as f:
        ...     rules = json.load(f)
        >>>     rules = convert_nan_back(rules)
        >>> binner.import_rules(rules)
        """
        import numpy as np
        
        for feature, rule in rules.items():
            # 判断是否为类别型变量
            if isinstance(rule, list) and len(rule) > 0 and isinstance(rule[0], list):
                # 类别型变量：List[List]格式
                self._cat_bins_[feature] = rule
                self.feature_types_[feature] = 'categorical'
                self.splits_[feature] = rule  # 保持List[List]格式
                self.n_bins_[feature] = len(rule)
            else:
                # 数值型变量：切分点列表
                self.splits_[feature] = np.array(rule)
                self.feature_types_[feature] = 'numerical'
                self.n_bins_[feature] = len(rule) + 1
        
        self._is_fitted = True

    def plot(
        self,
        feature: str,
        save: Optional[str] = None,
        **kwargs
    ):
        """绘制分箱图.

        :param feature: 特征名
        :param save: 图片保存路径，默认为None
        :param kwargs: 其他绘图参数

        **注意**

        绘制内容包括:
        1. 各分箱的样本数分布
        2. 各分箱的坏样本率
        3. 各分箱的WOE值
        """
        # TODO: 实现可视化功能
        raise NotImplementedError("可视化功能将在后续版本实现")

    def export(self, to_json: Optional[str] = None) -> Dict[str, Union[List, List[List]]]:
        """导出分箱规则，兼容 toad/scorecardpipeline 格式.

        数值型变量返回切分点列表，类别型变量返回分组列表。
        数据格式与 toad.Combiner.export() 和 scorecardpipeline.Combiner.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :return: 分箱规则字典
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}

        **示例**

        >>> binner = OptimalBinning()
        >>> binner.fit(X, y)
        >>> 
        >>> # 导出为字典
        >>> rules = binner.export()
        >>> 
        >>> # 导出并保存到 JSON 文件
        >>> rules = binner.export(to_json='binning_rules.json')
        
        **与 toad/scorecardpipeline 的兼容性**

        导出的规则可以直接被 toad 和 scorecardpipeline 加载:
        
        >>> # toad 加载
        >>> import toad
        >>> combiner = toad.transform.Combiner()
        >>> combiner.load(rules)
        >>> 
        >>> # scorecardpipeline 加载
        >>> from scorecardpipeline import Combiner
        >>> combiner = Combiner()
        >>> combiner.load(rules)
        """
        import json
        
        rules = self.export_rules()
        
        # 处理 numpy 类型和 np.nan，使其可 JSON 序列化
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_for_json(obj.tolist())
            elif isinstance(obj, float) and np.isnan(obj):
                return None  # JSON 中 null
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        rules_json = convert_for_json(rules)
        
        if to_json is not None:
            # 确保目录存在
            import os
            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(to_json, 'w', encoding='utf-8') as f:
                json.dump(rules_json, f, ensure_ascii=False, indent=2)
        
        return rules

    def load(self, from_json: Union[str, Dict], update: bool = False) -> 'BaseBinning':
        """加载分箱规则，兼容 toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载分箱规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: 分箱规则字典或 JSON 文件路径
            - 字典: {'age': [25, 35, 45, 55]}
            - 文件路径: 'binning_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :return: self，支持链式调用

        **示例**

        >>> binner = OptimalBinning()
        >>> 
        >>> # 从字典加载
        >>> rules = {'age': [25, 35, 45, 55], 'gender': [['M'], ['F']]}
        >>> binner.load(rules)
        >>> 
        >>> # 从 JSON 文件加载
        >>> binner.load('binning_rules.json')
        >>> 
        >>> # 更新现有规则
        >>> binner.load({'new_feature': [1, 2, 3]}, update=True)
        
        **与 toad/scorecardpipeline 的兼容性**

        可以直接加载 toad 和 scorecardpipeline 导出的规则:
        
        >>> # toad 导出
        >>> import toad
        >>> toad_combiner = toad.transform.Combiner()
        >>> toad_combiner.fit(df, y)
        >>> rules = toad_combiner.export()
        >>> 
        >>> # hscredit 加载
        >>> from hscredit.core.binning import OptimalBinning
        >>> binner = OptimalBinning()
        >>> binner.load(rules)
        """
        import json
        
        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        else:
            # 直接使用字典
            rules = from_json
        
        # 处理 JSON 中的 null 转换为 np.nan
        def convert_from_json(obj):
            if isinstance(obj, dict):
                return {k: convert_from_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_from_json(item) for item in obj]
            elif obj is None:
                return np.nan
            return obj
        
        rules = convert_from_json(rules)
        
        if update:
            # 更新模式：逐个特征更新
            for feature, rule in rules.items():
                if isinstance(rule, list) and len(rule) > 0 and isinstance(rule[0], list):
                    # 类别型
                    self._cat_bins_[feature] = rule
                    self.feature_types_[feature] = 'categorical'
                    self.splits_[feature] = rule
                    self.n_bins_[feature] = len(rule)
                else:
                    # 数值型
                    self.splits_[feature] = np.array(rule)
                    self.feature_types_[feature] = 'numerical'
                    self.n_bins_[feature] = len(rule) + 1
            self._is_fitted = True
        else:
            # 替换模式：使用 import_rules
            self.import_rules(rules)
        
        return self

    def __repr__(self):
        if self._is_fitted:
            n_features = len(self.splits_)
            return f"{self.__class__.__name__}(fitted=True, n_features={n_features})"
        else:
            return f"{self.__class__.__name__}(fitted=False)"
