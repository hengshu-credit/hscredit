# -*- coding: utf-8 -*-
"""二维分箱模块.

对两个特征进行交叉分箱分析，生成二维分箱矩阵，用于分析特征间的交互效应。
接口设计参考 optbinning 的 OptimalBinning2D，支持为两个特征分别传入独立参数。

**核心功能**

- fit: 对两个特征分别进行分箱，计算交叉分箱统计
- transform: 将新数据映射到二维分箱矩阵
- get_cross_table: 获取二维交叉分箱统计表
- plot: 绘制交互分箱热力图

**参考样例**

>>> from hscredit.core.binning import OptimalBinning2D
>>>
>>> # 基本使用（全局参数）
>>> binner = OptimalBinning2D(max_n_bins=5)
>>> binner.fit(df, y=df['target'], features=['age', 'income'])
>>> binner.plot()
>>>
>>> # 两个特征使用不同参数
>>> binner = OptimalBinning2D(
...     max_n_bins_x=5, max_n_bins_y=3,
...     min_bin_size_x=0.05, min_bin_size_y=0.1
... )
>>> binner.fit(df, y=df['target'], features=['age', 'income'])
>>>
>>> # 通过 x_params / y_params 传入扩展参数
>>> binner = OptimalBinning2D(
...     max_n_bins=5,
...     x_params={'method': 'best_iv', 'monotonic': 'descending'},
...     y_params={'method': 'quantile', 'max_n_bins': 3}
... )
>>> binner.fit(df, y=df['target'], features=['age', 'income'])
>>>
>>> # 自定义分箱规则
>>> binner = OptimalBinning2D(user_splits_x=[25, 35, 45], user_splits_y=[5000, 10000])
>>> binner.fit(df, y=df['target'], features=['age', 'income'])
"""

from __future__ import annotations

import re
import warnings
from typing import Union, List, Dict, Optional, Any, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

from .optimal_binning import OptimalBinning
from ..metrics._binning import compute_bin_stats
from ...exceptions import HSCreditError, NotFittedError
from ..._lazy import LazyModule
from ...utils.serialization import ArtifactSerializableMixin

# 延迟加载 seaborn：仅在首次实际绘图（访问 sns 属性）时才导入，
# 避免 import hscredit 时即触发 seaborn（及其 ipywidgets/IPython 依赖）的加载。
sns = LazyModule("seaborn")


class OptimalBinning2D(ArtifactSerializableMixin, BaseEstimator, TransformerMixin):
    """二维分箱器.

    对两个特征进行交叉分箱分析，生成二维分箱矩阵，用于揭示特征间的交互效应。
    每个单元格包含该交叉区间的样本数、坏样本率、WOE、IV等统计指标。

    接口与 OptimalBinning 保持一致，支持 sklearn 和 scorecardpipeline 两种调用风格。

    **参数**

    目标与分箱控制
        :param target: 目标变量列名，默认为 'target'
        :param max_n_bins: 两个特征的最大分箱数，默认 5
        :param min_bin_size: 两个特征的每箱最小样本占比，默认 0.02
        :param method: 分箱方法，默认 'quantile'（等频）
        :param monotonic: 单调性约束，默认 False。设为 'ascending'/'descending'/True 时，
            作为**硬约束**作用于二维合并：最终各轴向相邻分箱的坏样本率保证满足单调趋势
            （通过持续合并违例相邻分箱实现，可能使二维分箱数低于 max_n_bins_2d）
            ``ascending`` 表示特征值越大坏样本率越高（越大越差），``descending`` 表示
            特征值越大坏样本率越低（越大越好）；自动模式复用内部一维分箱器识别的方向。
        :param max_n_bins_2d: 相邻格子合并后的最大二维分箱数，默认使用 max_n_bins

    特征1 专用参数（以 _x 后缀区分）
        :param max_n_bins_x: 特征1的最大分箱数
        :param min_bin_size_x: 特征1的每箱最小样本占比
        :param method_x: 特征1的分箱方法
        :param monotonic_x: 特征1的单调性约束
        :param user_splits_x: 特征1的自定义切分点，包含 np.nan/None 时显式预留缺失箱
        :param special_codes_x: 特征1的特殊值列表
        :param dtype_x: 特征1的数据类型

    特征2 专用参数（以 _y 后缀区分）
        :param max_n_bins_y: 特征2的最大分箱数
        :param min_bin_size_y: 特征2的每箱最小样本占比
        :param method_y: 特征2的分箱方法
        :param monotonic_y: 特征2的单调性约束
        :param user_splits_y: 特征2的自定义切分点，包含 np.nan/None 时显式预留缺失箱
        :param special_codes_y: 特征2的特殊值列表
        :param dtype_y: 特征2的数据类型

    扩展参数
        :param x_params: 额外参数仅传递给特征1的内部 OptimalBinning
        :param y_params: 额外参数仅传递给特征2的内部 OptimalBinning
            参数优先级统一为：显式 ``_x``/``_y`` 参数 > ``x_params``/``y_params`` > 全局参数。
            所有参数会在构造内部 ``OptimalBinning`` 前完成合并和校验。

    其他参数
        :param missing_separate: 是否将缺失值单独分箱，默认 True
        :param missing_separate_x: 是否将特征1缺失值单独分箱，None 时继承 missing_separate
        :param missing_separate_y: 是否将特征2缺失值单独分箱，None 时继承 missing_separate
        :param random_state: 随机种子
        :param decimal: 数值精度
        :param woe_clip: WOE截断阈值
        :param verbose: 是否输出详细信息

    **属性**

    - binner_x_: 特征1的分箱器
    - binner_y_: 特征2的分箱器
    - splits_x_: 特征1的切分点
    - splits_y_: 特征2的切分点
    - n_bins_x_: 特征1的分箱数
    - n_bins_y_: 特征2的分箱数
    - solution_: 预分箱网格到最终二维分箱索引的映射矩阵；存在缺失值时追加缺失行/列
    - n_bins_2d_: 合并后的最终二维分箱数
    - binning_table_: 合并后的二维分箱统计表
    - cross_table_: 二维交叉分箱统计表
    - iv_interaction_: 交互IV值

    **参考样例**

    sklearn 风格::

        >>> binner = OptimalBinning2D(max_n_bins=5)
        >>> binner.fit(X_2d, y_array)  # X_2d shape=(n, 2)

    scorecardpipeline 风格::

        >>> binner = OptimalBinning2D(max_n_bins=5)
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])

    获取统计::

        >>> cross = binner.get_cross_table()      # 交叉分箱表
        >>> table_x = binner.get_bin_table('age')  # 特征1独立分箱表
        >>> stats = binner.get_stats()          # 两特征统计
        >>> splits = binner.get_splits()        # 两特征切分点
        >>> rules = binner.export_rules()       # 分箱规则（末尾 np.nan 表示缺失箱）

    **引用**

    二维交互最优分箱接口与求解思路参考 optbinning 的 ``OptimalBinning2D``：
    Navas-Palencia, G. (2020). *Optimal binning: mathematical programming
    formulation.* arXiv:2001.08025. https://arxiv.org/abs/2001.08025 ；
    交互效应（feature interaction）背景见
    https://gnpalencia.org/optbinning/binning_2d.html
    """

    artifact_kind = "分箱器"

    def __init__(
        self,
        target: str = 'target',
        # 全局分箱参数
        max_n_bins: int = 5,
        min_bin_size: Union[float, int] = 0.02,
        method: str = 'quantile',
        monotonic: Union[bool, str] = False,
        # 二维合并分箱参数
        max_n_bins_2d: Optional[int] = None,
        # 特征1专用参数
        max_n_bins_x: Optional[int] = None,
        min_bin_size_x: Optional[Union[float, int]] = None,
        method_x: Optional[str] = None,
        monotonic_x: Optional[Union[bool, str]] = None,
        user_splits_x: Optional[List] = None,
        special_codes_x: Optional[List] = None,
        dtype_x: str = 'numerical',
        # 特征2专用参数
        max_n_bins_y: Optional[int] = None,
        min_bin_size_y: Optional[Union[float, int]] = None,
        method_y: Optional[str] = None,
        monotonic_y: Optional[Union[bool, str]] = None,
        user_splits_y: Optional[List] = None,
        special_codes_y: Optional[List] = None,
        dtype_y: str = 'numerical',
        # 扩展参数
        x_params: Optional[Dict] = None,
        y_params: Optional[Dict] = None,
        # 其他参数
        missing_separate: bool = True,
        missing_separate_x: Optional[bool] = None,
        missing_separate_y: Optional[bool] = None,
        random_state: Optional[int] = None,
        decimal: int = 4,
        woe_clip: Optional[float] = None,
        verbose: Union[bool, int] = False,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        # 目标变量
        self.target = target

        # 全局分箱参数
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.method = method
        self.monotonic = monotonic
        # 二维合并分箱参数
        self.max_n_bins_2d = max_n_bins_2d

        # 特征1专用参数
        self.max_n_bins_x = max_n_bins_x
        self.min_bin_size_x = min_bin_size_x
        self.method_x = method_x
        self.monotonic_x = monotonic_x
        self.user_splits_x = user_splits_x
        self.special_codes_x = special_codes_x
        self.dtype_x = dtype_x

        # 特征2专用参数
        self.max_n_bins_y = max_n_bins_y
        self.min_bin_size_y = min_bin_size_y
        self.method_y = method_y
        self.monotonic_y = monotonic_y
        self.user_splits_y = user_splits_y
        self.special_codes_y = special_codes_y
        self.dtype_y = dtype_y

        # 扩展参数
        self.x_params = x_params
        self.y_params = y_params

        # 其他参数
        self.missing_separate = missing_separate
        self.missing_separate_x = missing_separate_x
        self.missing_separate_y = missing_separate_y
        self.random_state = random_state
        self.decimal = decimal
        self.woe_clip = woe_clip
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config

        # 内部属性
        self.binner_x_: Optional[OptimalBinning] = None
        self.binner_y_: Optional[OptimalBinning] = None
        self.splits_x_: Optional[np.ndarray] = None
        self.splits_y_: Optional[np.ndarray] = None
        self.n_bins_x_: int = 0
        self.n_bins_y_: int = 0
        self.feature_x_: str = ''
        self.feature_y_: str = ''
        self.feature_name_: str = ''
        self._is_fitted: bool = False
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self.cross_table_: Optional[pd.DataFrame] = None
        self.iv_interaction_: float = 0.0

        # 二维合并分箱结果
        self.solution_: Optional[np.ndarray] = None      # (n_bins_x_, n_bins_y_) 网格格子 -> 二维分箱编号
        self.n_bins_2d_: int = 0                          # 合并后的二维分箱数
        self.binning_table_: Optional[pd.DataFrame] = None  # 合并后的二维分箱表
        self.iv_2d_: float = 0.0                          # 合并后二维分箱总IV
        self._grid_event_: Optional[np.ndarray] = None    # 网格坏样本数矩阵
        self._grid_nonevent_: Optional[np.ndarray] = None  # 网格好样本数矩阵
        self._woe_2d_: Optional[np.ndarray] = None        # 每个二维分箱的WOE（transform查表用）
        self._event_rate_2d_: Optional[np.ndarray] = None  # 每个二维分箱的坏样本率
        self._woe_map_2d_: Optional[Dict[int, float]] = None  # 含缺失箱/特殊箱的WOE映射
        self._event_rate_map_2d_: Optional[Dict[int, float]] = None  # 含缺失箱/特殊箱的坏样本率映射
        self._bin_labels_2d_: Optional[List[str]] = None  # 每个二维分箱的标签
        self._bin_cells_2d_: Optional[List[List[Tuple[int, int]]]] = None

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        features: Optional[List[str]] = None,
    ) -> 'OptimalBinning2D':
        """拟合二维分箱器.

        支持两种调用风格：

        sklearn 风格 — X 为二维特征数组，y 为目标数组，features 不传::

            >>> binner = OptimalBinning2D(max_n_bins=5)
            >>> binner.fit(X_2d, y_array)  # X_2d shape=(n, 2)
            >>> # 通过 get_bin_table() 访问各特征的分箱表

        scorecardpipeline 风格 — X 为 DataFrame，通过 features 指定两个特征名::

            >>> binner = OptimalBinning2D(max_n_bins=5)
            >>> binner.fit(df, y=df['target'], features=['age', 'income'])
            >>> # 通过 get_cross_table() 访问交叉分箱表

        :param X: 训练数据（DataFrame 或二维数组）
        :param y: 目标变量（可选，默认从 X 中提取 target 列）
        :param features: 两个特征名列表，如 ['age', 'income']（sklearn 风格不传）
        :return: self
        """
        # sklearn 风格检测：X 是二维数组且 y 有值，features 未传
        is_sklearn_style = (
            features is None
            and y is not None
            and isinstance(X, (pd.DataFrame, np.ndarray))
        )
        if is_sklearn_style and isinstance(X, np.ndarray):
            # 二维数组直接作为两个特征的数值矩阵
            if X.ndim != 2 or X.shape[1] != 2:
                raise HSCreditError(
                    f"sklearn 风格需要 X 为二维数组且恰好有2列，"
                    f"但得到 shape={X.shape}，请使用 features=['col1','col2'] 指定列名"
                )
            col_x = 'feature_0'
            col_y = 'feature_1'
            X_df = pd.DataFrame(X, columns=[col_x, col_y])
            self.feature_x_ = col_x
            self.feature_y_ = col_y
            if isinstance(y, np.ndarray):
                y = pd.Series(y, name=self.target)
            # 不 drop target 列（因为不存在），直接拟合
            X = X_df
        elif is_sklearn_style and isinstance(X, pd.DataFrame):
            # DataFrame：取前两列作为特征，不传 features
            if len(X.columns) < 2:
                raise HSCreditError(
                    f"sklearn 风格需要 X 至少有2列，但只有 {len(X.columns)} 列，"
                    f"请使用 features=['col1','col2'] 指定列名"
                )
            col_x, col_y = X.columns[0], X.columns[1]
            self.feature_x_ = col_x
            self.feature_y_ = col_y
            X = X.copy()
            # 从 X 提取 y
            if isinstance(y, np.ndarray):
                y = pd.Series(y, index=X.index, name=self.target)
        else:
            # scorecardpipeline 风格
            X, y = self._check_input(X, y)
            if features is None or len(features) != 2:
                raise ValueError("必须提供包含两个特征名的列表，如 features=['age', 'income']")
            self.feature_x_ = features[0]
            self.feature_y_ = features[1]
            for feat in features:
                if feat not in X.columns:
                    raise HSCreditError(f"特征 '{feat}' 不在数据中，可用列: {list(X.columns)}")

        self._X = X
        self._y = y
        self.feature_name_ = f'{self.feature_x_}X{self.feature_y_}'

        # 创建并拟合特征1的分箱器
        self.binner_x_ = self._create_binner(is_x=True)
        self.binner_x_.fit(X[[self.feature_x_]], y)
        self.splits_x_ = self.binner_x_.splits_.get(self.feature_x_, np.array([]))
        self.n_bins_x_ = self.binner_x_.n_bins_.get(self.feature_x_, 0)

        # 创建并拟合特征2的分箱器
        self.binner_y_ = self._create_binner(is_x=False)
        self.binner_y_.fit(X[[self.feature_y_]], y)
        self.splits_y_ = self.binner_y_.splits_.get(self.feature_y_, np.array([]))
        self.n_bins_y_ = self.binner_y_.n_bins_.get(self.feature_y_, 0)

        # 计算交叉分箱统计（保留完整网格，供热力图/边缘视图使用）
        self._compute_cross_table()

        # 对二维网格相邻格子进行合并分箱，得到最终二维分箱
        self.solution_ = self._merge_2d_bins(self._grid_event_, self._grid_nonevent_)
        self.n_bins_2d_ = int(self.solution_.max()) + 1 if self.solution_.size else 0

        # 基于合并结果生成二维分箱表（复用 compute_bin_stats 计算指标）
        self._compute_binning_table()

        self.n_features_in_ = 2
        self.feature_names_in_ = np.asarray(
            [self.feature_x_, self.feature_y_],
            dtype=object,
        )
        self._is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: Literal['indices', 'bins', 'woe', 'event_rate'] = 'indices'
    ) -> pd.DataFrame:
        """将新数据映射到二维分箱.

        :param X: 待转换的数据
        :param metric: 转换类型，可选值:
            - 'indices': 返回合并后的二维分箱索引 (0, 1, 2, ...)，默认
            - 'bins': 返回合并后的二维分箱标签
            - 'woe': 返回合并后的 WOE 值
            - 'event_rate': 返回合并后的坏样本率
        :return: 转换后的数据

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>>
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>>
        >>> # 获取分箱标签
        >>> X_labels = binner.transform(X_test, metric='bins')
        >>>
        >>> # 获取 WOE 编码
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != 2:
                raise ValueError(f"X 必须是恰好包含 2 列的二维数组，当前 shape={X.shape}")
            X = pd.DataFrame(X, columns=[self.feature_x_, self.feature_y_])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        bins_x = self.binner_x_.transform(
            X[[self.feature_x_]], metric='indices')[self.feature_x_].values
        bins_y = self.binner_y_.transform(
            X[[self.feature_y_]], metric='indices')[self.feature_y_].values

        merged = self._map_grid_to_2d_bins(bins_x, bins_y)
        if metric == 'indices':
            values = merged
        elif metric == 'bins':
            values = np.array([
                self._bin_labels_2d_[b] if b >= 0 else ('特殊值' if b == -2 else '缺失值')
                for b in merged
            ], dtype=object)
        elif metric == 'woe':
            values = np.array([self._woe_map_2d_.get(int(b), np.nan) for b in merged], dtype=float)
        elif metric == 'event_rate':
            values = np.array([self._event_rate_map_2d_.get(int(b), np.nan) for b in merged], dtype=float)
        else:
            raise ValueError(f"不支持的 metric: {metric}，可选 'indices'/'bins'/'woe'/'event_rate'")

        result = pd.DataFrame({self.feature_name_: values}, index=X.index)
        if metric == 'woe':
            result.attrs['hscredit_encoding'] = 'woe'
            result.attrs['hscredit_source'] = 'OptimalBinning2D'
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """返回二维交叉分箱转换后的特征名."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError(
                    f"input_features 必须与拟合时特征一致，期望 "
                    f"{self.feature_names_in_.tolist()}，实际 {input_features.tolist()}"
                )
        return np.asarray([self.feature_name_], dtype=object)

    def get_cross_table(self) -> pd.DataFrame:
        """获取二维交叉分箱统计表.

        :return: 交叉分箱统计表，包含以下列：
            - 分箱, 分箱标签: 最终二维分箱索引和标签
            - 特征1名称, 特征2名称: 特征名
            - 特征1分箱, 特征2分箱: 单特征分箱索引
            - 特征1标签, 特征2标签: 单特征分箱标签
            - 样本总数, 好样本数, 坏样本数: 样本统计
            - 坏样本率: 坏样本占比
            - 样本占比: 交叉区间样本占全量样本的比例
            - 分档WOE值, 分档IV值: WOE/IV统计
            - LIFT值: 提升度
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.copy()

    def get_bad_rate_matrix(self) -> pd.DataFrame:
        """获取坏样本率矩阵（用于热力图）."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return self._merged_metric_matrix('坏样本率')

    def get_count_matrix(self) -> pd.DataFrame:
        """获取样本数矩阵（用于热力图标注）."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        row_labels = [self._get_bin_label(self.feature_x_, i, self.binner_x_) for i in range(self.n_bins_x_)]
        col_labels = [self._get_bin_label(self.feature_y_, j, self.binner_y_) for j in range(self.n_bins_y_)]
        return pd.DataFrame(
            (self._grid_event_ + self._grid_nonevent_)[:self.n_bins_x_, :self.n_bins_y_],
            index=row_labels,
            columns=col_labels,
        ).astype(int)

    def get_woe_matrix(self) -> pd.DataFrame:
        """获取 WOE 值矩阵."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return self._merged_metric_matrix('分档WOE值')

    def get_iv_matrix(self) -> pd.DataFrame:
        """获取 IV 值矩阵."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return self._merged_metric_matrix('分档IV值')

    def get_lift_matrix(self) -> pd.DataFrame:
        """获取 LIFT 值矩阵."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return self._merged_metric_matrix('LIFT值')

    def get_marginal_stats(self) -> Dict[str, pd.DataFrame]:
        """获取边缘分箱统计（各特征独立分箱的统计）."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        return {
            self.feature_x_: self.binner_x_.get_bin_table(self.feature_x_),
            self.feature_y_: self.binner_y_.get_bin_table(self.feature_y_)
        }

    def get_bin_table(self, feature: Optional[str] = None) -> pd.DataFrame:
        """获取分箱统计表.

        与 OptimalBinning 的 get_bin_table 保持一致。
        - 若指定特征名，返回该特征的独立分箱表（等同于 OptimalBinning.get_bin_table）
        - 若不指定（默认 None），返回合并后的二维分箱表

        :param feature: 特征名，None 时返回交叉分箱表
        :return: 分箱统计表

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>>
        >>> # 获取交叉分箱表（默认）
        >>> cross = binner.get_bin_table()
        >>>
        >>> # 获取特征1的分箱表
        >>> table_x = binner.get_bin_table(binner.feature_x_)
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if feature is None:
            return self.binning_table_.copy()

        if feature == self.feature_x_:
            return self.binner_x_.get_bin_table(feature)
        elif feature == self.feature_y_:
            return self.binner_y_.get_bin_table(feature)
        else:
            raise HSCreditError(f"特征 '{feature}' 未找到，可用: ['{self.feature_x_}', '{self.feature_y_}']")

    def get_stats(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """获取分箱统计信息.

        与 OptimalBinning 的 get_stats 保持一致。

        :param feature: 特征名，None 时返回两个特征的统计字典
        :return: 统计信息字典
            - 'n_bins': 分箱数
            - 'bin_table': 分箱统计表
            - 'iv': IV值
            - 'ks': KS值

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>>
        >>> # 获取两个特征的统计
        >>> stats = binner.get_stats()
        >>> for feat, s in stats.items():
        ...     print(f"{feat}: IV={s.get('iv', 'N/A'):.4f}")
        >>>
        >>> # 获取单个特征统计
        >>> s = binner.get_stats(binner.feature_x_)
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        def _stats_for(binner_inst: OptimalBinning, feat: str) -> Dict[str, Any]:
            stats = binner_inst.get_stats(feat)
            # 额外补充交叉分箱相关的属性
            if feat == self.feature_x_:
                stats['iv_interaction'] = self.iv_interaction_
                stats['cross_table'] = self.cross_table_
            elif feat == self.feature_y_:
                stats['iv_interaction'] = self.iv_interaction_
                stats['cross_table'] = self.cross_table_
            return stats

        if feature is not None:
            if feature == self.feature_name_:
                normal = self.binning_table_[self.binning_table_['分箱'] >= 0]
                return {
                    'n_bins': self.n_bins_2d_,
                    'bin_table': self.binning_table_.copy(),
                    'iv': self.iv_2d_,
                    'ks': float(normal['分档KS值'].max()) if len(normal) else 0.0,
                    'solution': self.solution_.copy(),
                }
            if feature not in (self.feature_x_, self.feature_y_):
                raise HSCreditError(
                    f"特征 '{feature}' 未找到，可用: "
                    f"['{self.feature_x_}', '{self.feature_y_}', '{self.feature_name_}']"
                )
            binner_inst = self.binner_x_ if feature == self.feature_x_ else self.binner_y_
            return _stats_for(binner_inst, feature)
        else:
            return {
                self.feature_x_: _stats_for(self.binner_x_, self.feature_x_),
                self.feature_y_: _stats_for(self.binner_y_, self.feature_y_),
            }

    def get_splits(self, feature: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """获取切分点.

        与 OptimalBinning 的 get_splits 保持一致。

        :param feature: 特征名，None 时返回两个特征的切分点字典
        :return: 切分点数组或字典

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>>
        >>> # 获取两个特征的切分点
        >>> splits = binner.get_splits()
        >>> print(splits['age'])
        >>> print(splits['income'])
        >>>
        >>> # 获取单个特征切分点
        >>> splits_x = binner.get_splits(binner.feature_x_)
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if feature is not None:
            if feature not in (self.feature_x_, self.feature_y_):
                raise HSCreditError(f"特征 '{feature}' 未找到，可用: ['{self.feature_x_}', '{self.feature_y_}']")
            if feature == self.feature_x_:
                return self.binner_x_.get_splits(feature)
            return self.binner_y_.get_splits(feature)
        else:
            return {
                self.feature_x_: self.binner_x_.get_splits(self.feature_x_),
                self.feature_y_: self.binner_y_.get_splits(self.feature_y_),
            }

    def export_rules(self) -> Dict[str, List]:
        """导出分箱规则.

        与 OptimalBinning.export_rules 保持一致。启用缺失值独立分箱的数值型特征，
        末尾追加 np.nan 表示缺失值单独一箱。

        :return: 分箱规则字典，格式同 OptimalBinning.export_rules

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>> rules = binner.export_rules()
        >>> print(rules['age'])      # [25, 35, 45, np.nan]
        >>> print(rules['income'])   # [5000, 10000, 20000, np.nan]
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        def _export_one(splits: np.ndarray, missing_separate: bool) -> List:
            arr = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
            if missing_separate:
                arr.append(np.nan)
            return arr

        return {
            self.feature_x_: _export_one(self.splits_x_, bool(self.binner_x_.missing_separate)),
            self.feature_y_: _export_one(self.splits_y_, bool(self.binner_y_.missing_separate)),
        }

    def import_rules(self, rules: Dict[str, List]) -> 'OptimalBinning2D':
        """导入分箱规则.

        :param rules: 分箱规则字典，格式同 export_rules
        :return: self

        **参考样例**

        >>> rules = {'age': [25, 35, 45], 'income': [5000, 10000]}
        >>> binner = OptimalBinning2D(user_splits_x=[25, 35, 45],
        ...                          user_splits_y=[5000, 10000])
        >>> # 先 fit（用于初始化结构），再 import_rules 覆盖切分点
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>> binner.import_rules(rules)
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if self.feature_x_ in rules:
            vals = rules[self.feature_x_]
            self.user_splits_x = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if self.feature_y_ in rules:
            vals = rules[self.feature_y_]
            self.user_splits_y = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return self

    def plot(
        self,
        metric: Literal['bad_rate', 'woe', 'iv', 'lift', 'count'] = 'bad_rate',
        figsize: Optional[tuple] = None,
        cmap: Optional[str] = None,
        annot: bool = True,
        fmt: str = '.2%',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save: Optional[str] = None,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """绘制二维分箱热力图.

        :param metric: 显示指标，可选 'bad_rate'（默认）、'woe'、'iv'、'lift'、'count'
        :param figsize: 图像尺寸
        :param cmap: 配色方案
        :param annot: 是否在热力图中显示数值
        :param fmt: 数值格式
        :param title: 图表标题
        :param xlabel: X轴标签
        :param ylabel: Y轴标签
        :param save: 保存路径
        :param ax: 可选的 matplotlib Axes 对象
        :param kwargs: 其他参数传递给 seaborn.heatmap
        :return: matplotlib Figure 或 Axes
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if metric == 'bad_rate':
            matrix = self.get_bad_rate_matrix()
            default_cmap = 'RdYlGn_r'
            if fmt == '.2%':
                fmt = '.1%'
        elif metric == 'woe':
            matrix = self.get_woe_matrix()
            default_cmap = 'RdBu'
        elif metric == 'iv':
            matrix = self.get_iv_matrix()
            default_cmap = 'YlOrRd'
        elif metric == 'lift':
            matrix = self.get_lift_matrix()
            default_cmap = 'RdYlGn_r'
        elif metric == 'count':
            matrix = self.get_count_matrix()
            default_cmap = 'Blues'
            if fmt == '.2%':
                fmt = 'd'
        else:
            raise ValueError(f"不支持的指标类型: {metric}")

        cmap = cmap or default_cmap

        if figsize is None:
            n_cols = matrix.shape[1]
            n_rows = matrix.shape[0]
            figsize = (max(6, n_cols * 1.5), max(5, n_rows * 1.2))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            return_fig = True
        else:
            fig = ax.figure
            return_fig = False

        sns.heatmap(
            matrix, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': self._get_metric_label(metric)},
            **kwargs)

        # 用粗线标出最终二维分箱边界，同一边界内的格子属于同一个箱。
        for i in range(self.n_bins_x_):
            for j in range(self.n_bins_y_ - 1):
                if self.solution_[i, j] != self.solution_[i, j + 1]:
                    ax.plot([j + 1, j + 1], [i, i + 1], color='black', linewidth=2.0)
        for i in range(self.n_bins_x_ - 1):
            for j in range(self.n_bins_y_):
                if self.solution_[i, j] != self.solution_[i + 1, j]:
                    ax.plot([j, j + 1], [i + 1, i + 1], color='black', linewidth=2.0)
        for bin_id, cells in enumerate(self._bin_cells_2d_):
            visible_cells = [
                cell for cell in cells
                if cell[0] < self.n_bins_x_ and cell[1] < self.n_bins_y_
            ]
            if not visible_cells:
                continue
            center = np.asarray(visible_cells, dtype=float).mean(axis=0)
            ax.text(
                center[1] + 0.5, center[0] + 0.18, f'#{bin_id}',
                ha='center', va='center', fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'),
            )

        ax.set_xlabel(xlabel or self.feature_y_, fontsize=12)
        ax.set_ylabel(ylabel or self.feature_x_, fontsize=12)

        if title is None:
            title = f'{self.feature_x_} × {self.feature_y_} 二维分箱分析'
        # 标题上移，为其下方的摘要框预留空间，避免摘要遮挡横坐标刻度和轴标题。
        ax.set_title(title, fontsize=14, fontweight='bold', pad=42)

        total_samples = self.cross_table_['样本总数'].sum()
        total_bad = self.cross_table_['坏样本数'].sum()
        overall_bad_rate = total_bad / total_samples if total_samples > 0 else 0.0
        stats_text = (
            f'总样本: {total_samples:,} | '
            f'坏样本: {total_bad:,} ({overall_bad_rate:.1%}) | '
            f'二维分箱: {self.n_bins_2d_} | '
            f'交互IV: {self.iv_interaction_:.4f}'
        )
        ax.text(0.5, 1.015, stats_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        return fig if return_fig else ax

    def plot_3d(
        self,
        metric: Literal['bad_rate', 'woe', 'lift'] = 'bad_rate',
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        save: Optional[str] = None,
        **kwargs
    ) -> Any:
        """绘制三维表面图展示交互效应."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        if metric == 'bad_rate':
            matrix = self.get_bad_rate_matrix()
        elif metric == 'woe':
            matrix = self.get_woe_matrix()
        elif metric == 'lift':
            matrix = self.get_lift_matrix()
        else:
            raise ValueError(f"不支持的指标类型: {metric}")

        if figsize is None:
            figsize = (12, 8)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        X_data, Y_data = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
        Z_data = matrix.values

        surf = ax.plot_surface(
            X_data, Y_data, Z_data,
            cmap='RdYlGn_r', edgecolor='none', alpha=0.8, **kwargs)

        ax.set_xlabel(f'\n{self.feature_y_}', fontsize=11)
        ax.set_ylabel(f'\n{self.feature_x_}', fontsize=11)
        ax.set_zlabel(f'\n{self._get_metric_label(metric)}', fontsize=11)

        if title is None:
            title = f'{self.feature_x_} × {self.feature_y_} 交互效应 (3D)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        fig.colorbar(surf, ax=ax, shrink=0.5, label=self._get_metric_label(metric))

        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)

        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        return fig

    def plot2d(
        self,
        figsize: Optional[tuple] = None,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        annot: bool = True,
        fontsize: int = 10,
        save: Optional[str] = None,
    ) -> Any:
        """快捷绘制二维分箱联合分析图.

        该方法等价于 ``hscredit.core.viz.bin_2d_plot(self, ...)``。

        :param figsize: 图像尺寸
        :param colors: 配色方案
        :param title: 图表总标题
        :param annot: 热力图是否标注数值
        :param fontsize: 单元格字体大小
        :param save: 保存路径
        :return: matplotlib Figure
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        from ..viz import bin_2d_plot

        return bin_2d_plot(
            self,
            figsize=figsize,
            colors=colors,
            title=title,
            annot=annot,
            fontsize=fontsize,
            save=save,
        )

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------

    def _get_missing_separate(self, is_x: bool) -> bool:
        """获取指定特征轴最终生效的缺失值分箱配置."""
        axis_value = self.missing_separate_x if is_x else self.missing_separate_y
        return bool(self.missing_separate if axis_value is None else axis_value)

    def _resolve_axis_params(self, is_x: bool) -> Dict[str, Any]:
        """按显式轴向参数、轴向 params、全局参数的顺序解析有效配置."""
        params = {
            'target': self.target,
            'max_n_bins': self.max_n_bins,
            'min_bin_size': self.min_bin_size,
            'method': self.method,
            'monotonic': self.monotonic,
            'missing_separate': self.missing_separate,
            'random_state': self.random_state,
            'decimal': self.decimal,
            'woe_clip': self.woe_clip,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
            'parallel_backend': self.parallel_backend,
            'parallel_config': self.parallel_config,
        }
        extra_params = self.x_params if is_x else self.y_params
        valid_names = set(OptimalBinning().get_params(deep=False))
        for key, value in dict(extra_params or {}).items():
            if key not in valid_names:
                warnings.warn(f"OptimalBinning 无此参数 '{key}'，将忽略")
            else:
                params[key] = value

        axis_missing_separate = self.missing_separate_x if is_x else self.missing_separate_y
        explicit_params = {
            'max_n_bins': self.max_n_bins_x if is_x else self.max_n_bins_y,
            'min_bin_size': self.min_bin_size_x if is_x else self.min_bin_size_y,
            'method': self.method_x if is_x else self.method_y,
            'monotonic': self.monotonic_x if is_x else self.monotonic_y,
            'missing_separate': axis_missing_separate,
            'special_codes': self.special_codes_x if is_x else self.special_codes_y,
        }
        params.update({key: value for key, value in explicit_params.items() if value is not None})

        user_splits = self.user_splits_x if is_x else self.user_splits_y
        if user_splits is not None:
            feature = self.feature_x_ if is_x else self.feature_y_
            params['user_splits'] = {feature: user_splits}
        return params

    def _create_binner(self, is_x: bool) -> OptimalBinning:
        """创建内部 OptimalBinning 实例.

        :param is_x: True 表示创建特征1的分箱器，False 表示特征2
        :return: 配置好的 OptimalBinning 实例
        """
        return OptimalBinning(**self._resolve_axis_params(is_x))

    def _check_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """检查并准备输入数据."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.Series(y, index=X.index, name=self.target)
            elif not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index, name=self.target)
        else:
            if self.target in X.columns:
                y = X[self.target].copy()
                X = X.drop(columns=[self.target])
            else:
                raise ValueError(
                    f"目标变量 '{self.target}' 未在数据中找到。"
                    f"请提供 y 参数或确保数据中包含 '{self.target}' 列。"
                )

        unique_values = y.dropna().unique()
        if len(unique_values) != 2:
            raise ValueError(
                f"目标变量必须是二分类，但发现 {len(unique_values)} 个唯一值: {unique_values}"
            )

        return X, y

    def _user_splits_include_missing(self, is_x: bool) -> bool:
        """判断用户分箱规则是否显式声明缺失箱."""
        splits = self.user_splits_x if is_x else self.user_splits_y
        if splits is None:
            return False

        for value in splits:
            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                values = value
            else:
                values = [value]
            for item in values:
                if item is None or (np.isscalar(item) and pd.isna(item)):
                    return True
        return False

    def _normalize_missing_bins(self, bins: np.ndarray, is_x: bool) -> np.ndarray:
        """按轴级配置统一缺失值索引.

        部分底层分箱器即使 ``missing_separate=False`` 仍会为缺失值返回 -1。
        二维分箱中不单独处理缺失值时，将其归入首个普通箱，避免样本被网格过滤。
        """
        result = np.asarray(bins, dtype=int).copy()
        binner = self.binner_x_ if is_x else self.binner_y_
        if not bool(binner.missing_separate):
            result[result == -1] = 0
        return result

    def _compute_cross_table(self) -> None:
        """计算二维交叉分箱统计表."""
        X, y = self._X, self._y

        bins_x = self.binner_x_.transform(
            X[[self.feature_x_]], metric='indices')[self.feature_x_].values
        bins_y = self.binner_y_.transform(
            X[[self.feature_y_]], metric='indices')[self.feature_y_].values
        bins_x = self._normalize_missing_bins(bins_x, is_x=True)
        bins_y = self._normalize_missing_bins(bins_y, is_x=False)

        # 缺失箱扩展为独立行/列，与另一特征的所有普通箱组成笛卡尔积。
        # 用户规则中的 np.nan/None 表示显式预留缺失箱，即使训练集暂时没有缺失值也要展示，
        # 并保证后续 transform 遇到缺失值时能映射到该二维箱。
        missing_separate_x = bool(self.binner_x_.missing_separate)
        missing_separate_y = bool(self.binner_y_.missing_separate)
        self._has_missing_x_ = bool(
            missing_separate_x
            and (np.any(bins_x == -1) or self._user_splits_include_missing(is_x=True))
        )
        self._has_missing_y_ = bool(
            missing_separate_y
            and (np.any(bins_y == -1) or self._user_splits_include_missing(is_x=False))
        )
        grid_n_x = self.n_bins_x_ + int(self._has_missing_x_)
        grid_n_y = self.n_bins_y_ + int(self._has_missing_y_)

        grid_x = bins_x.copy()
        grid_y = bins_y.copy()
        if self._has_missing_x_:
            grid_x[grid_x == -1] = self.n_bins_x_
        if self._has_missing_y_:
            grid_y[grid_y == -1] = self.n_bins_y_

        # 特殊值仍沿用二维特殊箱；笛卡尔积统计包含普通值和缺失值组合。
        valid_mask = (
            (grid_x >= 0) & (grid_x < grid_n_x)
            & (grid_y >= 0) & (grid_y < grid_n_y)
        )
        bins_x_valid = grid_x[valid_mask]
        bins_y_valid = grid_y[valid_mask]
        y_valid = y.to_numpy()[valid_mask]

        # 网格坏/好样本数矩阵（供二维合并分箱使用）
        grid_event = np.zeros((grid_n_x, grid_n_y), dtype=float)
        grid_nonevent = np.zeros((grid_n_x, grid_n_y), dtype=float)

        # 临时分箱索引严格采用最终展示顺序，确保累计指标与表格行顺序一致。
        cells = [
            (i, j) for i in range(self.n_bins_x_) for j in range(self.n_bins_y_)
        ]
        if self._has_missing_y_:
            cells.extend((i, self.n_bins_y_) for i in range(self.n_bins_x_))
        if self._has_missing_x_:
            cells.extend((self.n_bins_x_, j) for j in range(self.n_bins_y_))
        if self._has_missing_x_ and self._has_missing_y_:
            cells.append((self.n_bins_x_, self.n_bins_y_))

        cell_id_grid = np.full((grid_n_x, grid_n_y), -1, dtype=int)
        for cell_id, (i, j) in enumerate(cells):
            cell_id_grid[i, j] = cell_id

        cell_indices = cell_id_grid[bins_x_valid, bins_y_valid]
        stats = self._compute_bin_stats(
            cell_indices,
            y_valid,
        ).drop(columns='分箱标签', errors='ignore')
        stats = stats.set_index('分箱').reindex(range(len(cells)))

        # compute_bin_stats 不生成无样本箱；笛卡尔积要求保留这些组合。
        cumulative_columns = [
            '累积LIFT值', '累积坏账改善', '累计风险拒绝比',
            '累积好样本数', '累积坏样本数', '分档KS值',
        ]
        total_iv = float(stats['指标IV值'].dropna().iloc[0]) if stats['指标IV值'].notna().any() else 0.0
        for column in stats.columns:
            if column == '指标IV值':
                stats[column] = stats[column].fillna(total_iv)
            elif column in cumulative_columns:
                stats[column] = stats[column].ffill().fillna(0)
            else:
                stats[column] = stats[column].fillna(0)
        for column in ['样本总数', '好样本数', '坏样本数', '累积好样本数', '累积坏样本数']:
            stats[column] = stats[column].astype(int)
        stats = stats.reset_index()

        descriptions = []
        for i, j in cells:
            mask = (bins_x_valid == i) & (bins_y_valid == j)
            bad = int(y_valid[mask].sum())
            count = int(mask.sum())
            grid_event[i, j] = bad
            grid_nonevent[i, j] = count - bad
            descriptions.append({
                '特征1名称': self.feature_x_,
                '特征1分箱': -1 if i == self.n_bins_x_ else i,
                '特征1标签': self._get_grid_bin_label(i, is_x=True),
                '特征2名称': self.feature_y_,
                '特征2分箱': -1 if j == self.n_bins_y_ else j,
                '特征2标签': self._get_grid_bin_label(j, is_x=False),
            })

        self.cross_table_ = pd.concat(
            [pd.DataFrame(descriptions), stats.drop(columns='分箱')],
            axis=1,
        )
        self.iv_grid_ = float(self.cross_table_['分档IV值'].sum())
        self.iv_interaction_ = self.iv_grid_
        # 保存展平的WOE数组供transform查表使用
        self.woe_flat_ = self.cross_table_['分档WOE值'].to_numpy()
        # 保存网格样本数矩阵供二维合并分箱使用
        self._grid_event_ = grid_event
        self._grid_nonevent_ = grid_nonevent

    def _merge_2d_bins(self, event: np.ndarray, nonevent: np.ndarray) -> np.ndarray:
        """贪心合并相邻网格生成二维分箱（两阶段）.

        阶段一（样本量 + 分箱数）：在满足 ``min_bin_size``、``max_n_bins_2d`` 约束下，
        优先合并过小分箱并尽量减少 IV 损失（单调违例数仅作轻量 tiebreak）。

        阶段二（单调性硬约束）：当设置了单调趋势时，持续合并违反趋势的相邻分箱，
        直到任意轴向相邻分箱坏样本率均满足单调性。合并只会增大分箱，不会破坏
        样本量与分箱数约束，因此最终结果同时满足三者，且**保证零单调违例**。
        """
        if event.shape != nonevent.shape or event.ndim != 2:
            raise ValueError("二维好坏样本矩阵形状不一致")

        n_cells = event.size
        if n_cells == 0:
            return np.empty(event.shape, dtype=int)

        max_bins = self.max_n_bins_2d if self.max_n_bins_2d is not None else self.max_n_bins
        if isinstance(max_bins, (bool, np.bool_)) or not isinstance(max_bins, (int, np.integer)) or max_bins < 1:
            raise ValueError("max_n_bins_2d 必须是正整数")
        normal_cells = self.n_bins_x_ * self.n_bins_y_
        max_bins = min(int(max_bins), normal_cells)

        total = float((event + nonevent)[:self.n_bins_x_, :self.n_bins_y_].sum())
        if self.min_bin_size is None:
            min_count = 1
        elif self.min_bin_size < 1:
            min_count = max(1, int(np.ceil(total * float(self.min_bin_size))))
        else:
            min_count = max(1, int(self.min_bin_size))

        solution = np.arange(n_cells, dtype=int).reshape(event.shape)
        trend_x = self._resolve_axis_monotonic_trend(is_x=True)
        trend_y = self._resolve_axis_monotonic_trend(is_x=False)

        total_event = max(float(event.sum()), 1.0)
        total_nonevent = max(float(nonevent.sum()), 1.0)

        def region_counts(matrix: np.ndarray) -> Dict[int, Tuple[float, float]]:
            counts = {}
            for bin_id in np.unique(matrix):
                mask = matrix == bin_id
                counts[int(bin_id)] = (float(event[mask].sum()), float(nonevent[mask].sum()))
            return counts

        def cell_group(i: int, j: int) -> int:
            if i < self.n_bins_x_ and j < self.n_bins_y_:
                return 0
            if i < self.n_bins_x_:
                return 1  # 特征1正常，特征2缺失
            if j < self.n_bins_y_:
                return 2  # 特征1缺失，特征2正常
            return 3

        def adjacent_pairs(matrix: np.ndarray, normal_only: bool = False) -> List[Tuple[int, int]]:
            pairs = set()
            for i in range(matrix.shape[0] - 1):
                for j in range(matrix.shape[1]):
                    if cell_group(i, j) != cell_group(i + 1, j):
                        continue
                    if normal_only and cell_group(i, j) != 0:
                        continue
                    a, b = matrix[i, j], matrix[i + 1, j]
                    if a != b:
                        pairs.add(tuple(sorted((int(a), int(b)))))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1] - 1):
                    if cell_group(i, j) != cell_group(i, j + 1):
                        continue
                    if normal_only and cell_group(i, j) != 0:
                        continue
                    a, b = matrix[i, j], matrix[i, j + 1]
                    if a != b:
                        pairs.add(tuple(sorted((int(a), int(b)))))
            return sorted(pairs)

        def normal_ids(matrix: np.ndarray) -> set:
            return set(np.unique(matrix[:self.n_bins_x_, :self.n_bins_y_]).astype(int))

        def iv_part(ev: float, nev: float) -> float:
            p_event = max(ev / total_event, 1e-10)
            p_nonevent = max(nev / total_nonevent, 1e-10)
            return (p_event - p_nonevent) * np.log(p_event / p_nonevent)

        # ---------------- 阶段一：样本量与分箱数约束 ----------------
        while True:
            counts = region_counts(solution)
            current_normal_ids = normal_ids(solution)
            small = {
                k for k in current_normal_ids
                if sum(counts[k]) < min_count
            }
            must_reduce = len(current_normal_ids) > max_bins
            if not must_reduce and not small:
                break
            candidates = adjacent_pairs(solution, normal_only=True)
            if not candidates or len(current_normal_ids) == 1:
                break

            ranked = []
            for left, right in candidates:
                ev_l, nev_l = counts[left]
                ev_r, nev_r = counts[right]
                trial = solution.copy()
                trial[trial == right] = left
                trial_counts = region_counts(trial)
                trial_small = sum(
                    sum(trial_counts[k]) < min_count for k in normal_ids(trial)
                )
                trial_violations = len(self._monotonic_violations(trial, trial_counts, trend_x, trend_y))
                iv_loss = (
                    iv_part(ev_l, nev_l) + iv_part(ev_r, nev_r)
                    - iv_part(ev_l + ev_r, nev_l + nev_r)
                )
                involves_small = left in small or right in small
                ranked.append((
                    0 if (not small or involves_small) else 1,
                    trial_small,
                    trial_violations,
                    iv_loss,
                    left,
                    right,
                ))
            _, _, _, _, keep, remove = min(ranked)
            solution[solution == remove] = keep

        # ---------------- 阶段二：单调性硬约束（合并违例相邻分箱至零违例） ----------------
        if trend_x is not None or trend_y is not None:
            while True:
                counts = region_counts(solution)
                violations = self._monotonic_violations(solution, counts, trend_x, trend_y)
                if not violations or len(counts) == 1:
                    break
                candidates = adjacent_pairs(solution)
                if not candidates:
                    break

                ranked = []
                for left, right in candidates:
                    ev_l, nev_l = counts[left]
                    ev_r, nev_r = counts[right]
                    trial = solution.copy()
                    trial[trial == right] = left
                    trial_counts = region_counts(trial)
                    trial_violations = len(self._monotonic_violations(trial, trial_counts, trend_x, trend_y))
                    iv_loss = (
                        iv_part(ev_l, nev_l) + iv_part(ev_r, nev_r)
                        - iv_part(ev_l + ev_r, nev_l + nev_r)
                    )
                    ranked.append((
                        trial_violations,
                        0 if (left, right) in violations else 1,
                        iv_loss,
                        left,
                        right,
                    ))
                _, _, _, keep, remove = min(ranked)
                solution[solution == remove] = keep

            if self.verbose and len(np.unique(solution)) <= 1:
                warnings.warn(
                    "单调性硬约束将二维分箱合并为单一分箱，"
                    "请考虑放宽单调性约束（monotonic_x/monotonic_y）或调整预分箱粒度"
                )

        def order_key(bin_id: int) -> Tuple[int, int, int]:
            cells = np.argwhere(solution == bin_id)
            i, j = cells.min(axis=0)
            return cell_group(int(i), int(j)), int(i), int(j)

        ordered_ids = sorted(np.unique(solution), key=order_key)
        remap = {int(old): new for new, old in enumerate(ordered_ids)}
        return np.vectorize(remap.get, otypes=[int])(solution)

    def _resolve_axis_monotonic_trend(self, is_x: bool) -> Optional[str]:
        """复用内部一维分箱器的单调方向，转换为二维单向硬约束."""
        binner = self.binner_x_ if is_x else self.binner_y_
        feature = self.feature_x_ if is_x else self.feature_y_
        if binner is None:
            return None

        value = binner.monotonic
        if value in (False, None, '', 'none'):
            return None
        if isinstance(value, str):
            value = value.lower()
        if value in ('ascending', 'descending'):
            return value

        fitted_trend = getattr(binner, 'monotonic_trend_', {}).get(feature)
        if fitted_trend in ('ascending', 'descending'):
            return fitted_trend
        if fitted_trend is not None:
            return None

        table = getattr(binner, 'bin_tables_', {}).get(feature, pd.DataFrame())
        if table.empty or '坏样本率' not in table.columns:
            return None
        ordinary = table
        if '分箱' in ordinary.columns:
            ordinary = ordinary.loc[pd.to_numeric(ordinary['分箱'], errors='coerce') >= 0]
        bad_rates = pd.to_numeric(ordinary['坏样本率'], errors='coerce').dropna().to_numpy(dtype=float)
        if len(bad_rates) < 2:
            return None

        resolved = binner._resolve_monotonic_target_mode(bad_rates, value)
        return resolved if resolved in ('ascending', 'descending') else None

    def _monotonic_violations(
        self,
        solution: np.ndarray,
        counts: Dict[int, Tuple[float, float]],
        trend_x: Optional[str],
        trend_y: Optional[str],
    ) -> set:
        """返回轴向坏样本率违反单调性的相邻分箱对."""
        rates = {
            bin_id: ev / (ev + nev) if ev + nev > 0 else 0.0
            for bin_id, (ev, nev) in counts.items()
        }
        violations = set()

        def check(a: int, b: int, trend: Optional[str]) -> None:
            if trend is None or a == b:
                return
            invalid = rates[a] > rates[b] + 1e-12 if trend == 'ascending' else rates[a] < rates[b] - 1e-12
            if invalid:
                violations.add(tuple(sorted((int(a), int(b)))))

        # 缺失行/列只沿另一个非缺失特征的轴向受单调性约束。
        for i in range(solution.shape[0] - 1):
            if i + 1 >= getattr(self, 'n_bins_x_', solution.shape[0]):
                continue
            for j in range(solution.shape[1]):
                check(int(solution[i, j]), int(solution[i + 1, j]), trend_x)
        for j in range(solution.shape[1] - 1):
            if j + 1 >= getattr(self, 'n_bins_y_', solution.shape[1]):
                continue
            for i in range(solution.shape[0]):
                check(int(solution[i, j]), int(solution[i, j + 1]), trend_y)
        return violations

    def _map_grid_to_2d_bins(self, bins_x: np.ndarray, bins_y: np.ndarray) -> np.ndarray:
        """将两个一维分箱索引映射到最终二维分箱."""
        bins_x = self._normalize_missing_bins(bins_x, is_x=True)
        bins_y = self._normalize_missing_bins(bins_y, is_x=False)
        result = np.full(len(bins_x), -1, dtype=int)
        special = (bins_x == -2) | (bins_y == -2)
        grid_x = bins_x.copy()
        grid_y = bins_y.copy()
        if getattr(self, '_has_missing_x_', False):
            grid_x[grid_x == -1] = self.n_bins_x_
        if getattr(self, '_has_missing_y_', False):
            grid_y[grid_y == -1] = self.n_bins_y_
        valid = (
            (grid_x >= 0) & (grid_x < self.solution_.shape[0])
            & (grid_y >= 0) & (grid_y < self.solution_.shape[1])
        )
        result[special] = -2
        result[valid] = self.solution_[grid_x[valid], grid_y[valid]]
        return result

    def _compute_binning_table(self) -> None:
        """根据合并后的二维索引生成 hscredit 标准中文分箱表."""
        bins_x = self.binner_x_.transform(
            self._X[[self.feature_x_]], metric='indices')[self.feature_x_].to_numpy()
        bins_y = self.binner_y_.transform(
            self._X[[self.feature_y_]], metric='indices')[self.feature_y_].to_numpy()
        merged = self._map_grid_to_2d_bins(bins_x, bins_y)

        cells_by_bin = []
        x_proj, y_proj, labels = [], [], []
        for bin_id in range(self.n_bins_2d_):
            cells = [tuple(cell) for cell in np.argwhere(self.solution_ == bin_id)]
            cells_by_bin.append(cells)
            # 将连通区域在各特征轴上投影为紧凑区间，作为可读分箱标签（参考 optbinning 的 Bin x / Bin y）
            xp = self._axis_projection_label({i for i, _ in cells}, is_x=True)
            yp = self._axis_projection_label({j for _, j in cells}, is_x=False)
            x_proj.append(xp)
            y_proj.append(yp)
            labels.append(f'{xp} × {yp}')
        self._bin_cells_2d_ = cells_by_bin
        self._bin_labels_2d_ = labels

        unique_bins = np.unique(merged)
        special = {-1: '缺失值', -2: '特殊值'}
        label_by_id = {**{i: labels[i] for i in range(self.n_bins_2d_)}, **special}

        bin_labels = [label_by_id[int(bin_id)] for bin_id in unique_bins]
        table = self._compute_bin_stats(
            merged,
            self._y.to_numpy(),
            bin_labels=bin_labels,
        )

        # 显式预留的缺失箱可能在训练集中没有样本。统计函数默认只返回已观测箱，
        # 此处补齐 solution_ 中的全部二维箱，确保汇总表和指标映射保留空缺失箱。
        observed_special_bins = sorted(int(bin_id) for bin_id in unique_bins if bin_id < 0)
        expected_bins = list(range(self.n_bins_2d_)) + observed_special_bins
        if len(table) != len(expected_bins):
            table = table.set_index('分箱').reindex(expected_bins)
            table.index.name = '分箱'
            table['分箱标签'] = [label_by_id[bin_id] for bin_id in expected_bins]

            cumulative_columns = [
                '累积LIFT值', '累积坏账改善', '累计风险拒绝比',
                '累积好样本数', '累积坏样本数', '分档KS值',
            ]
            total_iv = (
                float(table['指标IV值'].dropna().iloc[0])
                if table['指标IV值'].notna().any()
                else 0.0
            )
            for column in table.columns:
                if column == '分箱标签':
                    continue
                if column == '指标IV值':
                    table[column] = table[column].fillna(total_iv)
                elif column in cumulative_columns:
                    table[column] = table[column].ffill().fillna(0)
                else:
                    table[column] = table[column].fillna(0)
            for column in ['样本总数', '好样本数', '坏样本数', '累积好样本数', '累积坏样本数']:
                if column in table.columns:
                    table[column] = table[column].astype(int)
            table = table.reset_index()

        table.insert(0, '指标名称', self.feature_name_)
        table.insert(1, '指标含义', None)
        table = table[[
            '指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比',
            *[
                column for column in table.columns
                if column not in {'指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比'}
            ],
        ]]
        self.binning_table_ = table

        normal = table[table['分箱'] >= 0].set_index('分箱')
        self._woe_2d_ = normal['分档WOE值'].reindex(range(self.n_bins_2d_)).to_numpy(dtype=float)
        self._event_rate_2d_ = normal['坏样本率'].reindex(range(self.n_bins_2d_)).to_numpy(dtype=float)
        indexed = table.set_index('分箱')
        self._woe_map_2d_ = indexed['分档WOE值'].astype(float).to_dict()
        self._event_rate_map_2d_ = indexed['坏样本率'].astype(float).to_dict()
        self.iv_2d_ = float(normal['分档IV值'].sum())
        self.iv_interaction_ = self.iv_2d_

        lookup = normal['分箱标签']

        def grid_index(value: int, is_x: bool) -> int:
            normal_count = self.n_bins_x_ if is_x else self.n_bins_y_
            return normal_count if value == -1 else value

        self.cross_table_['分箱'] = [
            int(self.solution_[grid_index(i, True), grid_index(j, False)])
            for i, j in zip(self.cross_table_['特征1分箱'], self.cross_table_['特征2分箱'])
        ]
        self.cross_table_['分箱标签'] = self.cross_table_['分箱'].map(lookup)
        leading = [
            '分箱', '分箱标签',
            '特征1名称', '特征1分箱', '特征1标签',
            '特征2名称', '特征2分箱', '特征2标签',
            '样本总数', '样本占比',
        ]
        self.cross_table_ = self.cross_table_[
            leading + [column for column in self.cross_table_.columns if column not in leading]
        ]
        category = (
            (self.cross_table_['特征1分箱'] == -1).astype(int) * 2
            + (self.cross_table_['特征2分箱'] == -1).astype(int)
        )
        self.cross_table_ = (
            self.cross_table_.assign(_组合顺序=category)
            .sort_values(['_组合顺序', '特征1分箱', '特征2分箱'], kind='stable')
            .drop(columns='_组合顺序')
            .reset_index(drop=True)
        )

    def _compute_bin_stats(
        self,
        bins: np.ndarray,
        y: np.ndarray,
        bin_labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """统一计算二维汇总箱和笛卡尔明细箱的完整指标."""
        return compute_bin_stats(
            bins,
            y,
            bin_labels=bin_labels,
            round_digits=True,
            woe_clip=self.woe_clip,
        )

    def _merged_metric_matrix(self, metric: str) -> pd.DataFrame:
        """将最终二维分箱指标回填到预分箱网格."""
        values = self.binning_table_[self.binning_table_['分箱'] >= 0].set_index('分箱')[metric]
        normal_solution = self.solution_[:self.n_bins_x_, :self.n_bins_y_]
        matrix = np.vectorize(lambda bin_id: values.loc[int(bin_id)], otypes=[float])(normal_solution)
        row_labels = [self._get_bin_label(self.feature_x_, i, self.binner_x_) for i in range(self.n_bins_x_)]
        col_labels = [self._get_bin_label(self.feature_y_, j, self.binner_y_) for j in range(self.n_bins_y_)]
        return pd.DataFrame(matrix, index=row_labels, columns=col_labels)

    def _axis_projection_label(self, prebin_indices, is_x: bool) -> str:
        """将二维分箱在单个特征轴上覆盖的预分箱投影为紧凑区间标签.

        数值型分箱合并连续区间（如 {0,1,2} -> '[-inf, s2)'），非区间型标签去重后用 ∪ 连接。
        注意：对于非矩形的连通区域，该投影是各轴覆盖范围的并集（边界外延），
        精确归属以 solution_ 矩阵与分箱图为准。
        """
        feat = self.feature_x_ if is_x else self.feature_y_
        binner = self.binner_x_ if is_x else self.binner_y_
        normal_count = self.n_bins_x_ if is_x else self.n_bins_y_
        idxs = sorted({int(i) for i in prebin_indices})
        if not idxs:
            return ''
        if idxs == [normal_count]:
            return '缺失值'
        label_of = [self._get_bin_label(feat, i, binner) for i in idxs]
        interval_like = all(re.match(r'^[\[(].*,.*[\])]$', str(lab)) for lab in label_of)
        if not interval_like:
            return ' ∪ '.join(dict.fromkeys(label_of))
        # 合并连续预分箱区间
        groups: List[List[int]] = []
        for i in idxs:
            if groups and i == groups[-1][-1] + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        parts = []
        for g in groups:
            first = self._get_bin_label(feat, g[0], binner)
            if g[0] == g[-1]:
                parts.append(first)
            else:
                last = self._get_bin_label(feat, g[-1], binner)
                left = first.split(',')[0].strip()
                right = last.split(',')[-1].strip()
                parts.append(f'{left}, {right}')
        return ' ∪ '.join(parts)

    def _get_grid_bin_label(self, bin_idx: int, is_x: bool) -> str:
        """获取扩展网格标签，缺失行/列使用统一中文标签."""
        normal_count = self.n_bins_x_ if is_x else self.n_bins_y_
        if bin_idx == normal_count:
            return '缺失值'
        feature = self.feature_x_ if is_x else self.feature_y_
        binner = self.binner_x_ if is_x else self.binner_y_
        return self._get_bin_label(feature, bin_idx, binner)

    def _get_bin_label(self, feature: str, bin_idx: int, binner: OptimalBinning) -> str:
        """获取指定特征和分箱索引的标签."""
        if feature in binner.bin_tables_:
            bin_table = binner.bin_tables_[feature]
            if '分箱标签' in bin_table.columns:
                # 分箱表可能因报表或指标后处理而调整行序，不能假设 iloc 与分箱索引一致。
                # 优先按“分箱”列精确查找，保证 -inf 到 +inf 始终按 bin 0..n-1 排列。
                if '分箱' in bin_table.columns:
                    bin_ids = pd.to_numeric(bin_table['分箱'], errors='coerce')
                    matched = bin_table.loc[bin_ids == bin_idx, '分箱标签']
                    if not matched.empty and pd.notna(matched.iloc[0]):
                        return str(matched.iloc[0])
                if 0 <= bin_idx < len(bin_table):
                    label = bin_table['分箱标签'].iloc[bin_idx]
                else:
                    label = None
                if pd.notna(label):
                    return str(label)
        return f'Bin_{bin_idx}'

    def _get_metric_label(self, metric: str) -> str:
        """获取指标的中文标签."""
        labels = {
            'bad_rate': '坏样本率',
            'woe': 'WOE值',
            'iv': 'IV值',
            'lift': 'LIFT值',
            'count': '样本数'
        }
        return labels.get(metric, metric)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"OptimalBinning2D("
                f"fitted=True, "
                f"features=['{self.feature_x_}', '{self.feature_y_}'], "
                f"prebins=[{self.n_bins_x_}, {self.n_bins_y_}], "
                f"n_bins_2d={self.n_bins_2d_}, "
                f"iv={self.iv_interaction_:.4f})"
            )
        else:
            return "OptimalBinning2D(fitted=False)"


__all__ = ['OptimalBinning2D']
