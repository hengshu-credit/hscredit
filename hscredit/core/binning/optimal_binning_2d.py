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

import warnings
from typing import Union, List, Dict, Optional, Any, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .optimal_binning import OptimalBinning
from ...exceptions import HSCreditError


class OptimalBinning2D:
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
        :param monotonic: 单调性约束，默认 False

    特征1 专用参数（以 _x 后缀区分）
        :param max_n_bins_x: 特征1的最大分箱数
        :param min_bin_size_x: 特征1的每箱最小样本占比
        :param method_x: 特征1的分箱方法
        :param monotonic_x: 特征1的单调性约束
        :param user_splits_x: 特征1的自定义切分点
        :param special_codes_x: 特征1的特殊值列表
        :param dtype_x: 特征1的数据类型

    特征2 专用参数（以 _y 后缀区分）
        :param max_n_bins_y: 特征2的最大分箱数
        :param min_bin_size_y: 特征2的每箱最小样本占比
        :param method_y: 特征2的分箱方法
        :param monotonic_y: 特征2的单调性约束
        :param user_splits_y: 特征2的自定义切分点
        :param special_codes_y: 特征2的特殊值列表
        :param dtype_y: 特征2的数据类型

    扩展参数
        :param x_params: 额外参数仅传递给特征1的内部 OptimalBinning
        :param y_params: 额外参数仅传递给特征2的内部 OptimalBinning

    其他参数
        :param missing_separate: 是否将缺失值单独分箱，默认 True
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
    """

    def __init__(
        self,
        target: str = 'target',
        # 全局分箱参数
        max_n_bins: int = 5,
        min_bin_size: Union[float, int] = 0.02,
        method: str = 'quantile',
        monotonic: Union[bool, str] = False,
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
        random_state: Optional[int] = None,
        decimal: int = 4,
        woe_clip: Optional[float] = None,
        verbose: Union[bool, int] = False,
    ):
        # 目标变量
        self.target = target

        # 全局分箱参数
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.method = method
        self.monotonic = monotonic

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
        self.x_params = x_params or {}
        self.y_params = y_params or {}

        # 其他参数
        self.missing_separate = missing_separate
        self.random_state = random_state
        self.decimal = decimal
        self.woe_clip = woe_clip
        self.verbose = verbose

        # 内部属性
        self.binner_x_: Optional[OptimalBinning] = None
        self.binner_y_: Optional[OptimalBinning] = None
        self.splits_x_: Optional[np.ndarray] = None
        self.splits_y_: Optional[np.ndarray] = None
        self.n_bins_x_: int = 0
        self.n_bins_y_: int = 0
        self.feature_x_: str = ''
        self.feature_y_: str = ''
        self._is_fitted: bool = False
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self.cross_table_: Optional[pd.DataFrame] = None
        self.iv_interaction_: float = 0.0

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
            col_x = f'feature_0'
            col_y = f'feature_1'
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

        # 创建并拟合特征1的分箱器
        self.binner_x_ = self._create_binner(is_x=True)
        user_splits_x = {self.feature_x_: self.user_splits_x} if self.user_splits_x is not None else None
        if user_splits_x:
            self.binner_x_.user_splits = user_splits_x
        self.binner_x_.fit(X[[self.feature_x_]], y)
        self.splits_x_ = self.binner_x_.splits_.get(self.feature_x_, np.array([]))
        self.n_bins_x_ = self.binner_x_.n_bins_.get(self.feature_x_, 0)

        # 创建并拟合特征2的分箱器
        self.binner_y_ = self._create_binner(is_x=False)
        user_splits_y = {self.feature_y_: self.user_splits_y} if self.user_splits_y is not None else None
        if user_splits_y:
            self.binner_y_.user_splits = user_splits_y
        self.binner_y_.fit(X[[self.feature_y_]], y)
        self.splits_y_ = self.binner_y_.splits_.get(self.feature_y_, np.array([]))
        self.n_bins_y_ = self.binner_y_.n_bins_.get(self.feature_y_, 0)

        # 计算交叉分箱统计
        self._compute_cross_table()

        self._is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: Literal['indices', 'bins', 'woe'] = 'indices'
    ) -> Union[pd.DataFrame, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """将新数据映射到二维分箱.

        :param X: 待转换的数据
        :param metric: 转换类型，可选值:
            - 'indices': 返回分箱索引 (0, 1, 2, ...)，默认
            - 'bins': 返回分箱标签字符串
            - 'woe': 返回交叉分箱对应的 WOE 值
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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        bins_x = self.binner_x_.transform(
            X[[self.feature_x_]], metric='indices')[self.feature_x_].values
        bins_y = self.binner_y_.transform(
            X[[self.feature_y_]], metric='indices')[self.feature_y_].values

        if metric == 'indices':
            return pd.DataFrame({
                self.feature_x_: bins_x,
                self.feature_y_: bins_y
            }, index=X.index)
        elif metric == 'bins':
            labels_x = [self._get_bin_label(self.feature_x_, int(b), self.binner_x_) for b in bins_x]
            labels_y = [self._get_bin_label(self.feature_y_, int(b), self.binner_y_) for b in bins_y]
            return pd.DataFrame({
                self.feature_x_: labels_x,
                self.feature_y_: labels_y
            }, index=X.index)
        elif metric == 'woe':
            # 通过查表获取每个样本的交叉分箱WOE值
            woe_flat = self.woe_flat_
            woe_vals = np.array([
                woe_flat[i * self.n_bins_y_ + j] if i >= 0 and j >= 0 else np.nan
                for i, j in zip(bins_x, bins_y)
            ])
            return pd.DataFrame({
                self.feature_x_: woe_vals,
                self.feature_y_: woe_vals
            }, index=X.index)
        else:
            raise ValueError(f"不支持的 metric: {metric}，可选 'indices'/'bins'/'woe'")

    def get_cross_table(self) -> pd.DataFrame:
        """获取二维交叉分箱统计表.

        :return: 交叉分箱统计表，包含以下列：
            - 特征1, 特征2: 特征名
            - 分箱1, 分箱2: 分箱索引
            - 分箱1标签, 分箱2标签: 分箱区间标签
            - 样本总数, 好样本数, 坏样本数: 样本统计
            - 坏样本率: 坏样本占比
            - 样本占比: 交叉区间样本占全量样本的比例
            - 分档WOE值, 分档IV值: WOE/IV统计
            - LIFT值: 提升度
        """
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.copy()

    def get_bad_rate_matrix(self) -> pd.DataFrame:
        """获取坏样本率矩阵（用于热力图）."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.pivot_table(
            index='分箱1标签', columns='分箱2标签', values='坏样本率', aggfunc='first')

    def get_count_matrix(self) -> pd.DataFrame:
        """获取样本数矩阵（用于热力图标注）."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.pivot_table(
            index='分箱1标签', columns='分箱2标签', values='样本总数', aggfunc='sum')

    def get_woe_matrix(self) -> pd.DataFrame:
        """获取 WOE 值矩阵."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.pivot_table(
            index='分箱1标签', columns='分箱2标签', values='分档WOE值', aggfunc='first')

    def get_iv_matrix(self) -> pd.DataFrame:
        """获取 IV 值矩阵."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.pivot_table(
            index='分箱1标签', columns='分箱2标签', values='分档IV值', aggfunc='sum')

    def get_lift_matrix(self) -> pd.DataFrame:
        """获取 LIFT 值矩阵."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return self.cross_table_.pivot_table(
            index='分箱1标签', columns='分箱2标签', values='LIFT值', aggfunc='first')

    def get_marginal_stats(self) -> Dict[str, pd.DataFrame]:
        """获取边缘分箱统计（各特征独立分箱的统计）."""
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")
        return {
            self.feature_x_: self.binner_x_.get_bin_table(self.feature_x_),
            self.feature_y_: self.binner_y_.get_bin_table(self.feature_y_)
        }

    def get_bin_table(self, feature: Optional[str] = None) -> pd.DataFrame:
        """获取分箱统计表.

        与 OptimalBinning 的 get_bin_table 保持一致。
        - 若指定特征名，返回该特征的独立分箱表（等同于 OptimalBinning.get_bin_table）
        - 若不指定（默认 None），返回二维交叉分箱表（等同于 get_cross_table）

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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

        if feature is None:
            return self.cross_table_.copy()

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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

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
            if feature not in (self.feature_x_, self.feature_y_):
                raise HSCreditError(f"特征 '{feature}' 未找到，可用: ['{self.feature_x_}', '{self.feature_y_}']")
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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

        if feature is not None:
            if feature not in (self.feature_x_, self.feature_y_):
                raise HSCreditError(f"特征 '{feature}' 未找到，可用: ['{self.feature_x_}', '{self.feature_y_}']")
            return self.binner_x_.get_splits(feature) if feature == self.feature_x_ else self.binner_y_.get_splits(feature)
        else:
            return {
                self.feature_x_: self.binner_x_.get_splits(self.feature_x_),
                self.feature_y_: self.binner_y_.get_splits(self.feature_y_),
            }

    def export_rules(self) -> Dict[str, List]:
        """导出分箱规则.

        与 OptimalBinning.export_rules 保持一致。数值型特征末尾追加 np.nan 表示缺失值单独一箱。

        :return: 分箱规则字典，格式同 OptimalBinning.export_rules

        **参考样例**

        >>> binner = OptimalBinning2D()
        >>> binner.fit(df, y=df['target'], features=['age', 'income'])
        >>> rules = binner.export_rules()
        >>> print(rules['age'])      # [25, 35, 45, np.nan]
        >>> print(rules['income'])   # [5000, 10000, 20000, np.nan]
        """
        if not self._is_fitted:
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

        def _export_one(splits: np.ndarray) -> List:
            arr = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
            if self.missing_separate:
                arr.append(np.nan)
            return arr

        return {
            self.feature_x_: _export_one(self.splits_x_),
            self.feature_y_: _export_one(self.splits_y_),
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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

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

        ax.set_xlabel(xlabel or self.feature_y_, fontsize=12)
        ax.set_ylabel(ylabel or self.feature_x_, fontsize=12)

        if title is None:
            title = f'{self.feature_x_} × {self.feature_y_} 二维分箱分析'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        total_samples = self.cross_table_['样本总数'].sum()
        total_bad = self.cross_table_['坏样本数'].sum()
        overall_bad_rate = total_bad / total_samples if total_samples > 0 else 0.0
        stats_text = (
            f'总样本: {total_samples:,} | '
            f'坏样本: {total_bad:,} ({overall_bad_rate:.1%}) | '
            f'交互IV: {self.iv_interaction_:.4f}'
        )
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=10,
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
            raise HSCreditError("分箱器尚未拟合，请先调用 fit 方法")

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

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------

    def _create_binner(self, is_x: bool) -> OptimalBinning:
        """创建内部 OptimalBinning 实例.

        :param is_x: True 表示创建特征1的分箱器，False 表示特征2
        :return: 配置好的 OptimalBinning 实例
        """
        if is_x:
            max_n_bins = self.max_n_bins_x if self.max_n_bins_x is not None else self.max_n_bins
            min_bin_size = self.min_bin_size_x if self.min_bin_size_x is not None else self.min_bin_size
            method = self.method_x if self.method_x is not None else self.method
            monotonic = self.monotonic_x if self.monotonic_x is not None else self.monotonic
            special_codes = self.special_codes_x
            extra_params = self.x_params
        else:
            max_n_bins = self.max_n_bins_y if self.max_n_bins_y is not None else self.max_n_bins
            min_bin_size = self.min_bin_size_y if self.min_bin_size_y is not None else self.min_bin_size
            method = self.method_y if self.method_y is not None else self.method
            monotonic = self.monotonic_y if self.monotonic_y is not None else self.monotonic
            special_codes = self.special_codes_y
            extra_params = self.y_params

        binner = OptimalBinning(
            target=self.target,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            method=method,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=self.missing_separate,
            random_state=self.random_state,
            decimal=self.decimal,
            woe_clip=self.woe_clip,
            verbose=self.verbose,
        )

        # 合并扩展参数（extra_params 优先级高于构造函数参数，
        # 但构造函数参数已设置好，extra_params 仅用于传递额外参数）
        if extra_params:
            for key, value in extra_params.items():
                if not hasattr(binner, key):
                    warnings.warn(f"OptimalBinning 无此参数 '{key}'，将忽略")
                else:
                    setattr(binner, key, value)

        return binner

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

    def _compute_cross_table(self) -> None:
        """计算二维交叉分箱统计表."""
        X, y = self._X, self._y

        bins_x = self.binner_x_.transform(
            X[[self.feature_x_]], metric='indices')[self.feature_x_].values
        bins_y = self.binner_y_.transform(
            X[[self.feature_y_]], metric='indices')[self.feature_y_].values

        # 只保留两个特征均有效的样本
        valid_mask = (bins_x >= 0) & (bins_y >= 0)
        bins_x_valid = bins_x[valid_mask]
        bins_y_valid = bins_y[valid_mask]
        y_valid = y.values[valid_mask]

        total_samples = len(y_valid)
        total_bad = int(y_valid.sum())
        total_good = total_samples - total_bad
        overall_bad_rate = total_bad / total_samples if total_samples > 0 else 0.0

        rows = []
        for i in range(self.n_bins_x_):
            for j in range(self.n_bins_y_):
                mask = (bins_x_valid == i) & (bins_y_valid == j)
                count = int(mask.sum())
                bad = int(y_valid[mask].sum())
                good = count - bad
                bad_rate = bad / count if count > 0 else 0.0

                good_ratio = good / total_good if total_good > 0 else 0
                bad_ratio = bad / total_bad if total_bad > 0 else 0
                woe = np.log(bad_ratio / good_ratio) if (bad_ratio > 0 and good_ratio > 0) else 0.0
                woe = np.clip(woe, -self.woe_clip, self.woe_clip) if self.woe_clip else woe
                iv = (bad_ratio - good_ratio) * woe if (bad_ratio > 0 and good_ratio > 0) else 0.0
                lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0

                label_x = self._get_bin_label(self.feature_x_, i, self.binner_x_)
                label_y = self._get_bin_label(self.feature_y_, j, self.binner_y_)

                rows.append({
                    '特征1': self.feature_x_,
                    '特征2': self.feature_y_,
                    '分箱1': i,
                    '分箱2': j,
                    '分箱1标签': label_x,
                    '分箱2标签': label_y,
                    '样本总数': count,
                    '好样本数': good,
                    '坏样本数': bad,
                    '坏样本率': bad_rate,
                    '样本占比': count / total_samples if total_samples > 0 else 0.0,
                    '分档WOE值': woe,
                    '分档IV值': iv,
                    'LIFT值': lift,
                })

        self.cross_table_ = pd.DataFrame(rows)
        self.iv_interaction_ = self.cross_table_['分档IV值'].sum()
        # 保存展平的WOE数组供transform查表使用
        self.woe_flat_ = self.cross_table_['分档WOE值'].to_numpy()

    def _get_bin_label(self, feature: str, bin_idx: int, binner: OptimalBinning) -> str:
        """获取指定特征和分箱索引的标签."""
        if feature in binner.bin_tables_:
            bin_table = binner.bin_tables_[feature]
            if '分箱标签' in bin_table.columns and bin_idx < len(bin_table):
                label = bin_table['分箱标签'].iloc[bin_idx]
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
                f"n_bins=[{self.n_bins_x_}, {self.n_bins_y_}], "
                f"iv={self.iv_interaction_:.4f})"
            )
        else:
            return "OptimalBinning2D(fitted=False)"


__all__ = ['OptimalBinning2D']