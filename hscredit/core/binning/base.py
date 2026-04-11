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

from ...exceptions import FeatureNotFoundError, NotFittedError
from ...utils.misc import round_float

# 从 metrics 导入指标计算方法
from ..metrics._binning import (
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
        - True 或 'auto': 自动检测最佳趋势（允许单增、单减、正U、倒U）
        - 'auto_asc_desc': 自动检测，但只允许单增或单减（不允许U型）
        - 'auto_heuristic': 使用启发式方法自动确定单调方向
        - 'ascending': 强制坏样本率递增(分箱索引增大时坏样本率增大)
        - 'descending': 强制坏样本率递减(分箱索引增大时坏样本率减小)
        - 'peak': 倒U型/峰值（先增后降）
        - 'valley': U型/谷值（先降后增）
        - 'peak_heuristic': 使用启发式方法检测峰值
        - 'valley_heuristic': 使用启发式方法检测谷值
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
        if isinstance(decimal, (bool, np.bool_)) or not isinstance(decimal, (int, np.integer)) or int(decimal) < 0:
            raise ValueError("decimal 必须是大于等于 0 的整数")
        self.decimal = int(decimal)

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
        :param kwargs: 其他参数
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

    @staticmethod
    def _enrich_woe_map(woe_map: dict, bin_table) -> None:
        """为 woe_map 补充缺失值/特殊值箱的 WOE 映射.

        ``_apply_splits`` 对缺失值返回 -1、特殊值返回 -2，而 ``woe_map``
        默认只包含 0..n-1 的映射。本方法从 bin_table 的 missing/special
        行中提取真实 WOE 值写入 woe_map[-1] / woe_map[-2]。
        """
        if '分箱标签' not in bin_table.columns:
            woe_map.setdefault(-1, 0.0)
            woe_map.setdefault(-2, 0.0)
            return
        for idx in range(len(bin_table)):
            lbl = str(bin_table.iloc[idx].get('分箱标签', '')).lower()
            if lbl in ('missing', '缺失值', '缺失'):
                woe_map[-1] = float(bin_table.iloc[idx]['分档WOE值'])
            elif lbl in ('special', '特殊值', '特殊'):
                woe_map[-2] = float(bin_table.iloc[idx]['分档WOE值'])
        woe_map.setdefault(-1, 0.0)
        woe_map.setdefault(-2, 0.0)

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
        if isinstance(X, pd.Series):
            # 将Series转换为DataFrame，保留Series的名称作为列名
            col_name = X.name if X.name is not None else 'feature'
            X = X.to_frame(name=col_name)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = pd.DataFrame(X, columns=['feature'])
            else:
                # 为numpy数组生成默认列名 feature_0, feature_1, ...
                n_cols = X.shape[1]
                columns = [f'feature_{i}' for i in range(n_cols)]
                X = pd.DataFrame(X, columns=columns)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        original_index = X.index

        # 获取目标变量
        if y is not None:
            # sklearn风格: 使用传入的y（优先）
            if isinstance(y, np.ndarray):
                if y.ndim != 1:
                    raise ValueError(f"目标变量y必须是一维数组，但得到 {y.ndim} 维")
                if len(y) != len(original_index):
                    raise ValueError(
                        f"特征和标签数量不匹配: {len(original_index)} != {len(y)}"
                    )
                y = pd.Series(y, index=original_index, name=self.target)
            elif isinstance(y, pd.Series):
                y = y.copy()
                if y.index.equals(original_index):
                    pass
                elif len(y) == len(original_index):
                    # 长度一致但索引不同：按位置对齐
                    y = y.reset_index(drop=True)
                    y.index = original_index
                else:
                    # 长度不一致：尝试按索引交集对齐（常见于调用方先对y做过滤）
                    common_index = original_index.intersection(y.index)
                    if len(common_index) == 0:
                        raise ValueError(
                            f"特征和标签数量不匹配且无公共索引: {len(original_index)} != {len(y)}"
                        )
                    X = X.loc[common_index].copy()
                    original_index = X.index
                    y = y.loc[common_index].copy()
                y.name = self.target
            else:
                # 其他可迭代类型
                if len(y) != len(original_index):
                    raise ValueError(
                        f"特征和标签数量不匹配: {len(original_index)} != {len(y)}"
                    )
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

    def _get_max_samples(self, n_samples: int) -> Optional[int]:
        """计算最大样本数。"""
        if self.max_bin_size is None:
            return None
        if self.max_bin_size < 1:
            return max(1, int(np.ceil(n_samples * self.max_bin_size)))
        return max(1, int(self.max_bin_size))

    def _choose_merge_split_index(
        self,
        counts: np.ndarray,
        bad_counts: np.ndarray,
        bin_idx: int
    ) -> Optional[int]:
        """为样本量不足的分箱选择要删除的切分点索引。"""
        n_bins = len(counts)
        if n_bins <= max(1, self.min_n_bins):
            return None
        if bin_idx <= 0:
            return 0
        if bin_idx >= n_bins - 1:
            return n_bins - 2

        bad_rates = bad_counts / np.maximum(counts, 1.0)
        left_score = abs(bad_rates[bin_idx] - bad_rates[bin_idx - 1])
        right_score = abs(bad_rates[bin_idx] - bad_rates[bin_idx + 1])
        return bin_idx - 1 if left_score <= right_score else bin_idx

    def _choose_split_point_within_bin(
        self,
        x: pd.Series,
        bins: np.ndarray,
        bin_idx: int,
        min_samples: int
    ) -> Optional[float]:
        """为样本量过大的分箱选择新的切分点。"""
        values = np.sort(pd.to_numeric(x[bins == bin_idx], errors='coerce').dropna().to_numpy(dtype=float))
        if len(values) < max(2, min_samples * 2):
            return None

        center = len(values) // 2
        candidate_positions = sorted(
            range(min_samples, len(values) - min_samples + 1),
            key=lambda pos: abs(pos - center)
        )

        for pos in candidate_positions:
            left_value = values[pos - 1]
            right_value = values[pos]
            if np.isclose(left_value, right_value, atol=1e-12, rtol=0):
                continue
            return float((left_value + right_value) / 2.0)
        return None

    def _adjust_splits_for_bin_size_constraints(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: Union[np.ndarray, list],
        min_samples: int,
        max_samples: Optional[int]
    ) -> np.ndarray:
        """调整切分点以满足最小/最大样本量约束。"""
        if splits is None or len(splits) == 0:
            return np.array([])

        current = np.unique(np.sort(np.asarray(splits, dtype=float)))
        x_numeric = pd.to_numeric(x, errors='coerce')
        valid_mask = x_numeric.notna()
        if self.special_codes:
            for code in self.special_codes:
                valid_mask &= (x_numeric != code)
        x_valid = x_numeric[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) == 0:
            return current

        max_splits_allowed = max(0, self.max_n_bins - 1)
        min_splits_allowed = max(0, self.min_n_bins - 1)

        for _ in range(200):
            bins = np.digitize(x_valid, current) if len(current) > 0 else np.zeros(len(x_valid), dtype=int)
            counts = np.bincount(bins, minlength=len(current) + 1).astype(int)
            bad_counts = np.bincount(bins, weights=y_valid, minlength=len(current) + 1).astype(float)

            small_bins = np.where(counts < min_samples)[0]
            large_bins = np.array([], dtype=int) if max_samples is None else np.where(counts > max_samples)[0]

            changed = False

            if len(small_bins) > 0 and len(current) > min_splits_allowed:
                merge_bin = int(small_bins[np.argmin(counts[small_bins])])
                split_idx = self._choose_merge_split_index(counts, bad_counts, merge_bin)
                if split_idx is not None and 0 <= split_idx < len(current):
                    current = np.delete(current, split_idx)
                    changed = True

            if changed:
                continue

            if len(large_bins) > 0 and len(current) < max_splits_allowed:
                split_bin = int(large_bins[np.argmax(counts[large_bins])])
                new_split = self._choose_split_point_within_bin(x_valid, bins, split_bin, min_samples)
                if new_split is not None and np.isfinite(new_split):
                    current = np.unique(np.sort(np.append(current, new_split)))
                    changed = True

            if not changed:
                break

        return current

    def _enforce_bin_size_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """统一收口最小/最大分箱样本量约束。"""
        for feature in list(self.splits_.keys()):
            if self.feature_types_.get(feature) != 'numerical':
                continue

            splits = self.splits_.get(feature)
            if splits is None:
                continue

            min_samples = self._get_min_samples(len(y))
            max_samples = self._get_max_samples(len(y))
            adjusted = self._adjust_splits_for_bin_size_constraints(X[feature], y, splits, min_samples, max_samples)
            adjusted = self._round_splits(adjusted)

            old_splits = np.asarray(splits, dtype=float) if len(splits) > 0 else np.array([])
            if np.array_equal(adjusted, old_splits):
                continue

            self.splits_[feature] = adjusted
            self.n_bins_[feature] = len(adjusted) + 1

            apply_bins = getattr(self, '_apply_bins', None)
            if callable(apply_bins):
                try:
                    bins = apply_bins(X[feature], adjusted, 'numerical', feature)
                except TypeError:
                    try:
                        bins = apply_bins(X[feature], adjusted, feature)
                    except TypeError:
                        bins = apply_bins(X[feature], adjusted)
            else:
                values = X[feature]
                bins = np.zeros(len(values), dtype=int)
                if self.missing_separate:
                    bins[pd.isna(values)] = -1
                mask = pd.notna(values)
                if self.special_codes:
                    for code in self.special_codes:
                        bins[values == code] = -2
                        mask &= (values != code)
                if len(adjusted) > 0:
                    bins[mask] = np.digitize(pd.to_numeric(values[mask], errors='coerce'), adjusted)
                else:
                    bins[mask] = 0
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)

    def _enforce_max_n_bins_hard_cap(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """硬性限制最大分箱数，不超过 max_n_bins。

        当其他约束调整后切分点仍超出限制时，按相邻箱坏样本率差异最小的
        优先合并策略进行截断。
        """
        max_splits = max(0, self.max_n_bins - 1)
        for feature in list(self.splits_.keys()):
            if self.feature_types_.get(feature) != 'numerical':
                continue
            splits = self.splits_[feature]
            if splits is None or len(splits) <= max_splits:
                continue

            current = np.unique(np.sort(np.asarray(splits, dtype=float)))
            x = X[feature]
            while len(current) > max_splits:
                bins = self._get_feature_bins(feature, x, current)
                bin_table = self._compute_bin_stats(feature, x, y, bins)
                valid = bin_table[bin_table['分箱'] >= 0].reset_index(drop=True)
                bad_rates = valid['坏样本率'].to_numpy(dtype=float)
                if len(bad_rates) <= 2:
                    current = current[:max_splits]
                    break
                diffs = np.abs(np.diff(bad_rates))
                merge_idx = int(np.argmin(diffs))
                if merge_idx < 0 or merge_idx >= len(current):
                    current = current[:max_splits]
                    break
                current = np.delete(current, merge_idx)

            self.splits_[feature] = self._round_splits(current)
            self.n_bins_[feature] = len(current) + 1
            bins = self._get_feature_bins(feature, x, current)
            self.bin_tables_[feature] = self._compute_bin_stats(feature, x, y, bins)

    def _apply_post_fit_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        enforce_monotonic: bool = True
    ) -> None:
        """拟合后统一收口约束。"""
        self._enforce_bin_size_constraints(X, y)

        monotonic_adjuster = getattr(self, '_apply_monotonic_adjustment', None)
        if enforce_monotonic and self.monotonic and callable(monotonic_adjuster):
            monotonic_adjuster(X, y)
            self._enforce_bin_size_constraints(X, y)

        # 最终硬性限制：确保不超过 max_n_bins
        self._enforce_max_n_bins_hard_cap(X, y)

    def _get_feature_bins(
        self,
        feature: str,
        x: pd.Series,
        splits: Union[np.ndarray, list]
    ) -> np.ndarray:
        """获取指定特征切分点对应的分箱索引。"""
        apply_bins = getattr(self, '_apply_bins', None)
        if callable(apply_bins):
            try:
                return apply_bins(x, splits, 'numerical', feature)
            except TypeError:
                try:
                    return apply_bins(x, splits, feature)
                except TypeError:
                    return apply_bins(x, splits)

        bins = np.zeros(len(x), dtype=int)
        if self.missing_separate:
            bins[pd.isna(x)] = -1

        mask = pd.notna(x)
        if self.special_codes:
            for code in self.special_codes:
                bins[x == code] = -2
                mask &= (x != code)

        if len(splits) > 0:
            bins[mask] = np.digitize(pd.to_numeric(x[mask], errors='coerce'), splits)
        else:
            bins[mask] = 0
        return bins

    def _resolve_monotonic_target_mode(self, bad_rates: np.ndarray, target_mode: Union[bool, str]) -> str:
        """为自动单调模式选择目标趋势。"""
        if target_mode in ['ascending', 'descending', 'peak', 'valley']:
            return target_mode

        if target_mode in ['auto_asc_desc']:
            candidates = ['ascending', 'descending']
        else:
            candidates = ['ascending', 'descending', 'peak', 'valley']

        best_mode = candidates[0]
        best_score = None
        for mode in candidates:
            violations = self._count_monotonic_violations(bad_rates, mode)
            score = (violations, 0 if mode in ['ascending', 'descending'] else 1)
            if best_score is None or score < best_score:
                best_score = score
                best_mode = mode
        return best_mode

    def _count_monotonic_violations(self, bad_rates: np.ndarray, mode: str) -> int:
        """计算指定趋势下的违例数量。"""
        tol = 1e-10
        diffs = np.diff(bad_rates)
        if len(diffs) == 0:
            return 0
        if mode == 'ascending':
            return int(np.sum(diffs < -tol))
        if mode == 'descending':
            return int(np.sum(diffs > tol))
        if mode == 'peak':
            if len(bad_rates) < 3:
                return self._count_monotonic_violations(bad_rates, 'descending')
            return min(
                self._count_monotonic_violations(bad_rates[:pivot + 1], 'ascending') +
                self._count_monotonic_violations(bad_rates[pivot:], 'descending')
                for pivot in range(1, len(bad_rates) - 1)
            )
        if mode == 'valley':
            if len(bad_rates) < 3:
                return self._count_monotonic_violations(bad_rates, 'ascending')
            return min(
                self._count_monotonic_violations(bad_rates[:pivot + 1], 'descending') +
                self._count_monotonic_violations(bad_rates[pivot:], 'ascending')
                for pivot in range(1, len(bad_rates) - 1)
            )
        return 0

    def _choose_monotonic_merge_index(self, bad_rates: np.ndarray, mode: str) -> int:
        """选择需要移除的切分点索引。"""
        tol = 1e-10
        diffs = np.diff(bad_rates)
        if len(diffs) == 0:
            return 0
        if mode == 'ascending':
            violations = np.where(diffs < -tol)[0]
            return int(violations[np.argmin(diffs[violations])]) if len(violations) > 0 else 0
        if mode == 'descending':
            violations = np.where(diffs > tol)[0]
            return int(violations[np.argmax(diffs[violations])]) if len(violations) > 0 else 0
        if mode == 'peak':
            best_pivot = min(
                range(1, len(bad_rates) - 1),
                key=lambda pivot: self._count_monotonic_violations(bad_rates[:pivot + 1], 'ascending') +
                                  self._count_monotonic_violations(bad_rates[pivot:], 'descending')
            )
            left_diffs = diffs[:best_pivot]
            right_diffs = diffs[best_pivot:]
            left_idx = np.where(left_diffs < -tol)[0]
            right_idx = np.where(right_diffs > tol)[0]
            left_choice = None if len(left_idx) == 0 else int(left_idx[np.argmin(left_diffs[left_idx])])
            right_choice = None if len(right_idx) == 0 else int(best_pivot + right_idx[np.argmax(right_diffs[right_idx])])
            if left_choice is None:
                return right_choice if right_choice is not None else 0
            if right_choice is None:
                return left_choice
            return left_choice if abs(diffs[left_choice]) >= abs(diffs[right_choice]) else right_choice
        if mode == 'valley':
            best_pivot = min(
                range(1, len(bad_rates) - 1),
                key=lambda pivot: self._count_monotonic_violations(bad_rates[:pivot + 1], 'descending') +
                                  self._count_monotonic_violations(bad_rates[pivot:], 'ascending')
            )
            left_diffs = diffs[:best_pivot]
            right_diffs = diffs[best_pivot:]
            left_idx = np.where(left_diffs > tol)[0]
            right_idx = np.where(right_diffs < -tol)[0]
            left_choice = None if len(left_idx) == 0 else int(left_idx[np.argmax(left_diffs[left_idx])])
            right_choice = None if len(right_idx) == 0 else int(best_pivot + right_idx[np.argmin(right_diffs[right_idx])])
            if left_choice is None:
                return right_choice if right_choice is not None else 0
            if right_choice is None:
                return left_choice
            return left_choice if abs(diffs[left_choice]) >= abs(diffs[right_choice]) else right_choice
        return 0

    def _merge_splits_for_monotonicity(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        splits: Union[np.ndarray, list],
        target_mode: Union[bool, str]
    ) -> tuple[np.ndarray, str]:
        """基于当前切分点，通过相邻合并满足单调约束。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        if len(current) == 0:
            return current, 'unknown'

        min_splits_allowed = max(0, self.min_n_bins - 1)
        final_mode = str(target_mode)

        for _ in range(200):
            bins = self._get_feature_bins(feature, x, current)
            bin_table = self._compute_bin_stats(feature, x, y, bins)
            valid_bins = bin_table[bin_table['分箱'] >= 0]
            bad_rates = valid_bins['坏样本率'].to_numpy(dtype=float)
            if len(bad_rates) <= 1:
                return current, final_mode

            final_mode = self._resolve_monotonic_target_mode(bad_rates, target_mode)
            violations = self._count_monotonic_violations(bad_rates, final_mode)
            max_splits_allowed = max(0, self.max_n_bins - 1)
            if violations == 0 and len(current) <= max_splits_allowed:
                return current, final_mode
            if len(current) <= min_splits_allowed:
                return current, final_mode

            if violations == 0:
                diffs = np.abs(np.diff(bad_rates))
                merge_idx = int(np.argmin(diffs)) if len(diffs) > 0 else len(current) - 1
            else:
                merge_idx = self._choose_monotonic_merge_index(bad_rates, final_mode)
            if merge_idx < 0 or merge_idx >= len(current):
                break
            current = np.delete(current, merge_idx)

        return current, final_mode

    def _count_adjacent_zero_bad_rate_pairs(
        self,
        bad_rates: np.ndarray
    ) -> int:
        """统计相邻全零坏样本率箱对数。"""
        arr = np.asarray(bad_rates, dtype=float)
        if len(arr) <= 1:
            return 0
        zero_mask = arr <= 1e-12
        return int(np.sum(zero_mask[:-1] & zero_mask[1:]))

    def _merge_adjacent_zero_bad_rate_bins(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        splits: Union[np.ndarray, list]
    ) -> np.ndarray:
        """合并相邻坏样本率全为 0 的分箱。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        if len(current) == 0:
            return current

        for _ in range(200):
            bins = self._get_feature_bins(feature, x, current)
            bin_table = self._compute_bin_stats(feature, x, y, bins)
            valid = bin_table[bin_table['分箱'] >= 0].reset_index(drop=True)
            bad_rates = valid['坏样本率'].to_numpy(dtype=float)
            zero_pairs = np.where((bad_rates[:-1] <= 1e-12) & (bad_rates[1:] <= 1e-12))[0] if len(bad_rates) > 1 else np.array([])
            if len(zero_pairs) == 0:
                break
            if len(current) == 0:
                break
            merge_idx = int(zero_pairs[0])
            if merge_idx < 0 or merge_idx >= len(current):
                break
            current = np.delete(current, merge_idx)

        return current

    def _apply_monotonic_adjustment(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """基于当前方法的切分点执行单调性收口。"""
        needs_adjustment_modes = [
            'auto', 'auto_asc_desc', 'auto_heuristic',
            'ascending', 'descending',
            'peak', 'valley',
            'peak_heuristic', 'valley_heuristic', True
        ]
        if self.monotonic not in needs_adjustment_modes:
            return

        monotonic_trend = getattr(self, 'monotonic_trend_', None)
        if monotonic_trend is None:
            self.monotonic_trend_ = {}

        for feature in list(self.splits_.keys()):
            if self.feature_types_.get(feature) != 'numerical':
                continue
            splits = self.splits_.get(feature)
            if splits is None or len(splits) == 0:
                continue

            bins = self._get_feature_bins(feature, X[feature], splits)
            bin_table = self._compute_bin_stats(feature, X[feature], y, bins)
            valid_bins = bin_table[bin_table['分箱'] >= 0]
            bad_rates = valid_bins['坏样本率'].to_numpy(dtype=float)
            if len(bad_rates) <= 1:
                continue

            target_mode = self._resolve_monotonic_target_mode(bad_rates, self.monotonic)
            if self._count_monotonic_violations(bad_rates, target_mode) == 0:
                adjusted_splits = np.unique(np.sort(np.asarray(splits, dtype=float)))
                final_mode = target_mode
            else:
                adjusted_splits, final_mode = self._merge_splits_for_monotonicity(
                    feature, X[feature], y, splits, target_mode
                )

            adjusted_splits = self._expand_splits_with_monotonicity(
                feature, X[feature], y, adjusted_splits, final_mode
            )
            adjusted_splits = self._merge_adjacent_zero_bad_rate_bins(
                feature, X[feature], y, adjusted_splits
            )
            self.splits_[feature] = self._round_splits(adjusted_splits)
            self.n_bins_[feature] = len(self.splits_[feature]) + 1
            bins = self._get_feature_bins(feature, X[feature], self.splits_[feature])
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)
            self.monotonic_trend_[feature] = final_mode

    def _quadratic_curve_coefficient(
        self,
        values: np.ndarray
    ) -> float:
        """计算曲线二次拟合系数。"""
        arr = np.asarray(values, dtype=float)
        if len(arr) < 3 or np.allclose(arr, arr[0], atol=1e-12, rtol=0):
            return 0.0
        x = np.linspace(-1.0, 1.0, len(arr), dtype=float)
        coeffs = np.polyfit(x, arr, 2)
        return float(coeffs[0])

    def _quadratic_curve_score(
        self,
        values: np.ndarray,
        mode: str
    ) -> float:
        """根据目标趋势解释二次拟合系数方向。"""
        coef = self._quadratic_curve_coefficient(values)
        if mode == 'peak':
            return -coef
        return coef

    def _evaluate_split_scheme(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        mode: str
    ) -> tuple[bool, float, np.ndarray, np.ndarray]:
        """评估切分方案是否满足单调与样本约束，并给出 lift 导向评分。"""
        bins = self._get_feature_bins(feature, x, splits)
        bin_table = self._compute_bin_stats(feature, x, y, bins)
        valid = bin_table[bin_table['分箱'] >= 0].reset_index(drop=True)
        bad_rates = valid['坏样本率'].to_numpy(dtype=float)
        counts = valid['样本总数'].to_numpy(dtype=int)
        min_samples = self._get_min_samples(len(y))
        max_samples = self._get_max_samples(len(y))

        is_valid = len(valid) >= self.min_n_bins
        if len(bad_rates) > 1:
            is_valid = is_valid and self._count_monotonic_violations(bad_rates, mode) == 0
        if len(bad_rates) > 2:
            is_valid = is_valid and self._count_adjacent_zero_bad_rate_pairs(bad_rates) == 0
        is_valid = is_valid and np.all(counts >= min_samples)
        if max_samples is not None:
            is_valid = is_valid and np.all(counts <= max_samples)

        curve_values = valid['LIFT值'].to_numpy(dtype=float) if 'LIFT值' in valid.columns else bad_rates
        curve_spread = float(np.max(curve_values) - np.min(curve_values)) if len(curve_values) > 0 else 0.0
        curve_step_sum = float(np.sum(np.abs(np.diff(curve_values)))) if len(curve_values) > 1 else 0.0
        quad_score = self._quadratic_curve_score(curve_values, mode)
        flat_penalty = float(np.sum(np.abs(np.diff(curve_values)) < 1e-8)) if len(curve_values) > 1 else 0.0
        iv_value = float(valid['分档IV值'].sum()) if '分档IV值' in valid.columns else 0.0
        score = (
            quad_score * 1000.0
            + curve_spread * 100.0
            + curve_step_sum * 10.0
            + len(valid) * 80.0
            - flat_penalty * 200.0
            + iv_value * 1e-3
        )
        return is_valid, score, bad_rates, counts

    def _expand_splits_with_monotonicity(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        splits: Union[np.ndarray, list],
        mode: str
    ) -> np.ndarray:
        """在保持单调的前提下，尽量补足到允许的分箱预算。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        max_splits_allowed = max(0, self.max_n_bins - 1)
        if len(current) >= max_splits_allowed:
            return current

        x_numeric = pd.to_numeric(x, errors='coerce')
        valid_mask = x_numeric.notna()
        if self.special_codes:
            for code in self.special_codes:
                valid_mask &= (x_numeric != code)
        x_valid = x_numeric[valid_mask]
        y_valid = y[valid_mask]
        if len(x_valid) == 0:
            return current

        base_ok, base_score, _, _ = self._evaluate_split_scheme(feature, x, y, current, mode)
        if not base_ok:
            return current

        min_samples = self._get_min_samples(len(y_valid))
        quantiles = [0.25, 0.5, 0.75]

        for _ in range(max_splits_allowed - len(current)):
            bins = np.digitize(x_valid, current) if len(current) > 0 else np.zeros(len(x_valid), dtype=int)
            best_candidate = None
            best_score = None

            for bin_idx in range(len(current) + 1):
                bin_values = np.sort(x_valid[bins == bin_idx].to_numpy(dtype=float))
                if len(bin_values) < max(2 * min_samples, 8):
                    continue

                candidate_positions = set()
                for q in quantiles:
                    pos = int(round(q * (len(bin_values) - 1)))
                    pos = min(max(pos, min_samples - 1), len(bin_values) - min_samples - 1)
                    if 0 <= pos < len(bin_values) - 1:
                        candidate_positions.add(pos)

                for pos in sorted(candidate_positions):
                    left_value = bin_values[pos]
                    right_value = bin_values[pos + 1]
                    if np.isclose(left_value, right_value, atol=1e-12, rtol=0):
                        continue

                    candidate = float((left_value + right_value) / 2.0)
                    if len(current) > 0 and np.any(np.isclose(current, candidate, atol=1e-12, rtol=0)):
                        continue

                    trial = np.unique(np.sort(np.append(current, candidate)))
                    ok, score, _, _ = self._evaluate_split_scheme(feature, x, y, trial, mode)
                    if not ok:
                        continue
                    if best_score is None or score > best_score + 1e-9:
                        best_score = score
                        best_candidate = candidate

            if best_candidate is None:
                break

            current = np.unique(np.sort(np.append(current, best_candidate)))
            base_score = best_score

        return current

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
        rounded_splits = np.array([round_float(s, decimal=self.decimal) for s in splits], dtype=float)
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
            elif feature_type == 'categorical' and isinstance(splits, list) and len(splits) > 0 and not isinstance(splits[0], list):
                # 扁平列表格式：每个元素对应一个分箱的类别名
                unique_bins = np.unique(bins)
                bin_labels = []
                for bin_idx in unique_bins:
                    if bin_idx == -1:
                        bin_labels.append('missing')
                    elif bin_idx == -2:
                        bin_labels.append('special')
                    elif 0 <= bin_idx < len(splits):
                        bin_labels.append(str(splits[bin_idx]))
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
        """根据切分点生成分箱标签（左闭右开 [a, b) 风格，与 toad/scorecardpipeline 一致）.

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
                    labels.append('[-inf, +inf)')
                elif i < n_bins:
                    if i == 0:
                        labels.append(f'[-inf, {splits[i]})')
                    elif i == n_bins - 1:
                        labels.append(f'[{splits[i-1]}, +inf)')
                    else:
                        labels.append(f'[{splits[i-1]}, {splits[i]})')
                else:
                    labels.append(f'bin_{i}')
        else:
            # 只根据切分点生成标签
            if n_splits == 0:
                # 没有切分点时，只有一个箱
                labels.append('[-inf, +inf)')
            else:
                for i in range(n_splits + 1):
                    if i == 0:
                        labels.append(f'[-inf, {splits[i]})')
                    elif i == n_splits:
                        labels.append(f'[{splits[i-1]}, +inf)')
                    else:
                        labels.append(f'[{splits[i-1]}, {splits[i]})')

        return labels

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """获取指定特征的分箱表.

        :param feature: 特征名
        :return: 分箱统计表（返回副本，修改不会影响分箱器内部数据）
        :raises NotFittedError: 如果分箱器尚未拟合
        :raises FeatureNotFoundError: 如果特征不存在
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.bin_tables_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self.bin_tables_[feature].copy()

    def _splits_with_nan(self, feature: str) -> Union[np.ndarray, list]:
        """返回包含缺失值标记的切分点（scorecardpipeline 格式）.

        数值型特征：如果 missing_separate=True，在切分点末尾追加 np.nan。
        类别型特征：直接返回 _cat_bins_（已包含 np.nan）。
        """
        if self.feature_types_.get(feature) == 'categorical':
            if feature in self._cat_bins_:
                return self._cat_bins_[feature]
            return self.splits_.get(feature, [])

        splits = self.splits_.get(feature, np.array([]))
        if self.missing_separate:
            arr = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
            arr.append(np.nan)
            return arr
        return splits

    def __getitem__(self, feature: str):
        """通过 `binner['feature']` 获取分箱规则（toad/scorecardpipeline风格）.

        数值型特征返回切分点列表，末尾 np.nan 表示缺失值单独一箱。
        类别型特征返回 List[List] 分组列表。
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.splits_ and feature not in self._cat_bins_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self._splits_with_nan(feature)

    def get_splits(self, feature: str) -> Union[np.ndarray, list]:
        """获取指定特征的切分点（scorecardpipeline 格式）.

        数值型特征：切分点末尾 np.nan 表示缺失值单独一箱。
        类别型特征：返回 List[List] 分组列表。

        :param feature: 特征名
        :return: 切分点列表
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.splits_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self._splits_with_nan(feature)

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
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

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
                # 数值型变量：返回切分点列表（scorecardpipeline 格式，末尾 nan 表示缺失箱）
                splits = self.splits_[feature]
                arr = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
                if self.missing_separate:
                    arr.append(np.nan)
                rules[feature] = arr
        
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
                # 数值型变量：切分点列表（兼容 scorecardpipeline 格式，自动剥离末尾 nan）
                numeric_splits = pd.to_numeric(pd.Series(list(rule)), errors='coerce')
                clean = numeric_splits[numeric_splits.notna()].to_numpy(dtype=float)
                self.splits_[feature] = self._round_splits(np.unique(np.sort(clean)))
                self.feature_types_[feature] = 'numerical'
                self.n_bins_[feature] = len(self.splits_[feature]) + 1
        
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
        同时导出WOE映射信息，支持加载后直接进行WOE转换。
        数据格式与 toad.Combiner.export() 和 scorecardpipeline.Combiner.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :return: 分箱规则字典
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}
            - WOE映射: {'_woe_maps_': {'age': {0: 0.5, 1: -0.3, ...}}}

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
        
        **WOE转换支持**
        
        导出的规则包含WOE映射信息，加载后可直接进行WOE转换:
        
        >>> binner = OptimalBinning()
        >>> binner.load('binning_rules.json')
        >>> X_woe = binner.transform(X_test, metric='woe')  # 直接使用，无需重新fit
        """
        import json
        
        rules = self.export_rules()
        
        # 导出WOE映射信息，支持加载后直接进行WOE转换
        woe_maps = {}
        for feature in self.splits_:
            if feature in self.bin_tables_:
                bin_table = self.bin_tables_[feature]
                woe_map = {}
                for idx, row in bin_table.iterrows():
                    woe_map[int(idx)] = float(row['分档WOE值'])
                # 添加缺失值和特殊值的WOE
                self._enrich_woe_map(woe_map, bin_table)
                woe_maps[feature] = woe_map
        
        if woe_maps:
            rules['_woe_maps_'] = woe_maps
        
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
        同时加载WOE映射信息，支持加载后直接进行WOE转换。

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
        
        **WOE转换支持**
        
        加载包含WOE映射信息的规则后，可直接进行WOE转换:
        
        >>> binner.load('binning_rules.json')
        >>> X_woe = binner.transform(X_test, metric='woe')  # 直接使用，无需重新fit
        """
        import json
        
        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        else:
            # 直接使用字典
            rules = from_json
        
        # 提取WOE映射信息（如果存在）
        woe_maps = rules.pop('_woe_maps_', None)
        
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
                    # 数值型：兼容 scorecardpipeline 格式，自动剥离末尾 nan
                    numeric_splits = pd.to_numeric(pd.Series(list(rule)), errors='coerce')
                    clean = numeric_splits[numeric_splits.notna()].to_numpy(dtype=float)
                    self.splits_[feature] = self._round_splits(np.unique(np.sort(clean)))
                    self.feature_types_[feature] = 'numerical'
                    self.n_bins_[feature] = len(self.splits_[feature]) + 1
            self._is_fitted = True
        else:
            # 替换模式：使用 import_rules
            self.import_rules(rules)
        
        # 恢复WOE映射信息，支持直接WOE转换
        if woe_maps is not None:
            self._woe_maps_ = {}
            for feature, woe_map in woe_maps.items():
                # 将字符串键转换为整数键
                self._woe_maps_[feature] = {int(k): float(v) for k, v in woe_map.items()}
        
        return self

    def __repr__(self):
        if self._is_fitted:
            n_features = len(self.splits_)
            return f"{self.__class__.__name__}(fitted=True, n_features={n_features})"
        else:
            return f"{self.__class__.__name__}(fitted=False)"
