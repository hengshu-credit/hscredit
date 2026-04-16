"""逾期数据预估模块.

基于特征分箱统计（feature_bin_stats）计算分箱逾期率，
并利用加权逾期率对无标签样本进行逾期风险预估。

核心功能:
- 支持从原始数据（含target或逾期天数）自动分箱并提取逾期率
- 支持从现成分箱表（feature_bin_stats输出或人工提供）直接提取逾期率
- 将分箱逾期率应用于无标签样本，计算加权逾期率
- 提供系数设置功能，调整预测偏差
- 输出包含逾期率预估结果的报告数据（DataFrame）

设计原则:
1. 遵循sklearn的fit/transform API风格
2. 与feature_bin_stats保持一致的数据格式
3. 支持多逾期标签（overdue+dpds）场景
"""

import warnings
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin

from .feature_analyzer import feature_bin_stats, _create_bin_table
from ..core.binning.base import BaseBinning


class OverduePredictor(BaseEstimator, TransformerMixin):
    """逾期率预测器.

    基于特征分箱对应的逾期率，对无标签样本进行加权逾期率预测。
    支持从原始数据或现成分箱表两种方式获取分箱逾期率。

    **两种拟合模式**

    模式一：从原始数据拟合
    - 传入含target或逾期天数的DataFrame，自动分箱并计算各箱逾期率
    - 支持overdue+dpds多标签场景

    模式二：从分箱表拟合
    - 传入已有的分箱统计表（feature_bin_stats输出或人工构建的DataFrame）
    - 直接从分箱表中提取各箱逾期率

    **参数**

    :param feature: 特征名称，用于分箱和预估
    :param target: 目标变量名称，默认为'target'
    :param overdue: 逾期天数字段名称或列表，如 'MOB1' 或 ['MOB1', 'MOB3']
    :param dpds: 逾期定义天数或列表，如 7 或 [0, 7, 30]
        - 逾期天数 > dpds 为坏样本(1)，其他为好样本(0)
    :param method: 分箱方法，默认'mdlp'
    :param max_n_bins: 最大分箱数，默认5
    :param min_bin_size: 每箱最小样本占比，默认0.05
    :param missing_separate: 是否将缺失值单独分箱，默认True
    :param coefficients: 逾期率调整系数，支持以下格式：
        - None: 不调整（默认）
        - float: 对所有分箱统一乘以该系数
        - dict: 按分箱标签指定系数，如 {'(-inf, 300]': 1.2, '(300, 500]': 0.9}
        - 'auto': 自动基于整体逾期率偏差校正
    :param bad_rate_col: 分箱表中逾期率列名，默认为'坏样本率'
        - 单标签时直接使用该列名
        - 多标签时需包含目标名称，函数会自动匹配
    :param bin_label_col: 分箱表中分箱标签列名，默认为'分箱标签'
    :param rules: 自定义分箱切分点列表，如 [300, 500, 700]
    :param desc: 特征描述，用于报告展示
    :param bin_params: 传递给feature_bin_stats的额外参数

    **属性**

    - bin_table_: 拟合后的分箱统计表
    - bin_rates_: 各分箱的逾期率字典 {分箱标签: 逾期率}
    - splits_: 分箱切分点
    - feature_names_in_: 输入特征名称
    - target_names_: 目标标签名称列表
    - coefficients_: 实际使用的调整系数

    **参考样例**

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hscredit.report.overdue_estimator import OverduePredictor
    >>>
    >>> # 准备有标签的训练数据
    >>> train_df = pd.DataFrame({
    ...     'score': np.random.randn(1000) * 100 + 500,
    ...     'target': np.random.randint(0, 2, 1000)
    ... })
    >>>
    >>> # 方式一：从原始数据拟合（自动分箱计算各箱逾期率）
    >>> predictor = OverduePredictor(feature='score', target='target', max_n_bins=5)
    >>> predictor.fit(train_df)
    >>>
    >>> # 对无标签数据预测（根据样本所在分箱加权计算逾期率）
    >>> test_df = pd.DataFrame({'score': np.random.randn(200) * 100 + 500})
    >>> result = predictor.transform(test_df)
    >>> print(result.head())
    >>>
    >>> # 设置调整系数（对逾期率进行整体缩放校正）
    >>> predictor.set_coefficients(1.1)
    >>> result_adjusted = predictor.transform(test_df)
    >>>
    >>> # 方式二：从分箱表拟合（直接使用现成分箱逾期率，无需原始数据）
    >>> bin_table = pd.DataFrame({
    ...     '分箱标签': ['(-inf, 400]', '(400, 500]', '(500, 600]', '(600, +inf)'],
    ...     '坏样本率': [0.15, 0.08, 0.04, 0.02]
    ... })
    >>> predictor2 = OverduePredictor(feature='score')
    >>> predictor2.fit(bin_table)
    >>> result2 = predictor2.transform(test_df)
    """

    def __init__(
        self,
        feature: str,
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        method: str = 'mdlp',
        max_n_bins: int = 5,
        min_bin_size: float = 0.05,
        missing_separate: bool = True,
        coefficients: Optional[Union[float, Dict[str, float], str]] = None,
        bad_rate_col: str = '坏样本率',
        bin_label_col: str = '分箱标签',
        rules: Optional[List] = None,
        desc: Optional[str] = None,
        bin_params: Optional[Dict] = None,
    ):
        self.feature = feature
        self.target = target
        self.overdue = overdue
        self.dpds = dpds
        self.method = method
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.missing_separate = missing_separate
        self.coefficients = coefficients
        self.bad_rate_col = bad_rate_col
        self.bin_label_col = bin_label_col
        self.rules = rules
        self.desc = desc
        self.bin_params = bin_params or {}

    def fit(self, X: Union[pd.DataFrame, pd.Series], y=None) -> "OverduePredictor":
        """拟合预估器.

        支持两种输入模式:
        1. 传入DataFrame（含target或逾期天数列）: 自动分箱并计算逾期率
        2. 传入分箱表DataFrame（含分箱标签和逾期率列）: 直接提取逾期率

        :param X: 训练数据DataFrame或分箱表DataFrame
        :param y: sklearn兼容参数，此处不使用（目标列从X中提取）
        :return: self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 pd.DataFrame，支持原始数据或分箱表")

        self.feature_names_in_ = [self.feature]

        if self._is_bin_table(X):
            self._fit_from_bin_table(X)
        else:
            self._fit_from_raw_data(X)

        return self

    def _is_bin_table(self, df: pd.DataFrame) -> bool:
        """判断输入是否为分箱表.

        分箱表的特征：
        - 包含 bin_label_col 指定的分箱标签列
        - 包含逾期率相关的列（坏样本率 或 多级表头下包含坏样本率）

        :param df: 输入DataFrame
        :return: 是否为分箱表
        """
        if df.empty:
            return False

        cols = df.columns.tolist()

        # 单层表头检查
        if self.bin_label_col in cols:
            if self.bad_rate_col in cols:
                return True
            # 检查是否有含"坏样本率"的列（多标签时）
            for c in cols:
                if isinstance(c, str) and '坏样本率' in c:
                    return True
            return True  # 有分箱标签列即认为是分箱表

        # 多级表头检查
        if isinstance(df.columns, pd.MultiIndex):
            for c in df.columns:
                col_name = c[-1] if isinstance(c, (tuple, list)) else c
                if col_name == self.bin_label_col:
                    return True

        return False

    def _fit_from_raw_data(self, df: pd.DataFrame) -> None:
        """从原始数据拟合，自动分箱并计算逾期率.

        :param df: 含target或逾期天数的DataFrame
        """
        if self.overdue is not None and self.target is None:
            # 检查逾期天数字段是否存在
            overdue_cols = [self.overdue] if isinstance(self.overdue, str) else self.overdue
            missing_cols = [c for c in overdue_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"数据集缺少逾期天数字段: {missing_cols}")
        elif self.target is not None:
            if self.target not in df.columns:
                raise ValueError(f"数据集缺少目标变量字段: {self.target}")
        else:
            raise ValueError("必须传入 target 或 overdue+dpds 参数")

        # 构建目标变量名称列表
        self.target_names_ = self._build_target_names()

        # 调用 feature_bin_stats 生成分箱表
        bin_kwargs = {
            'method': self.method,
            'max_n_bins': self.max_n_bins,
            'min_bin_size': self.min_bin_size,
            'missing_separate': self.missing_separate,
            'margins': True,
        }
        if self.rules is not None:
            bin_kwargs['rules'] = self.rules
        bin_kwargs.update(self.bin_params)

        result = feature_bin_stats(
            df,
            feature=self.feature,
            target=self.target,
            overdue=self.overdue,
            dpds=self.dpds,
            desc=self.desc or self.feature,
            **bin_kwargs,
        )

        self.bin_table_ = result
        self._extract_bin_rates_from_table(result)

        # 提取分箱切分点（尝试从bin_table中获取）
        self._extract_splits_from_raw_data(df)

    def _fit_from_bin_table(self, df: pd.DataFrame) -> None:
        """从现成分箱表拟合，直接提取逾期率.

        :param df: 分箱统计表DataFrame
        """
        self.bin_table_ = df.copy()

        # 解析目标名称（从多级表头或列名推断）
        self.target_names_ = self._infer_target_names(df)

        self._extract_bin_rates_from_table(df)
        self._extract_splits_from_bin_table(df)

    def _build_target_names(self) -> List[str]:
        """构建目标变量名称列表.

        :return: 目标名称列表
        """
        if self.overdue is not None and self.dpds is not None:
            overdue_list = [self.overdue] if isinstance(self.overdue, str) else self.overdue
            dpd_list = [self.dpds] if isinstance(self.dpds, int) else self.dpds
            names = []
            for mob in overdue_list:
                for d in dpd_list:
                    names.append(f"{mob}_{d}+")
            return names
        elif self.target is not None:
            return [self.target]
        return []

    def _infer_target_names(self, df: pd.DataFrame) -> List[str]:
        """从分箱表推断目标名称.

        :param df: 分箱表
        :return: 目标名称列表
        """
        if isinstance(df.columns, pd.MultiIndex):
            # 多级表头：收集所有非"分箱详情"的顶级列名
            names = []
            for col in df.columns:
                top_level = col[0] if isinstance(col, tuple) else col
                if top_level != '分箱详情' and top_level not in names:
                    names.append(top_level)
            return names if names else ['target']
        else:
            return ['target']

    def _extract_bin_rates_from_table(self, table: pd.DataFrame) -> None:
        """从分箱表中提取各分箱逾期率.

        :param table: 分箱统计表
        """
        self.bin_rates_ = {}

        if isinstance(table.columns, pd.MultiIndex):
            # 多级表头：每个目标标签对应一组逾期率
            for target_name in self.target_names_:
                rates = self._extract_single_target_rates(table, target_name)
                self.bin_rates_[target_name] = rates

            # 单标签时提供便捷访问
            if len(self.target_names_) == 1:
                self.bin_rates_['_default'] = self.bin_rates_[self.target_names_[0]]
        else:
            # 单层表头
            rates = self._extract_single_target_rates_flat(table)
            self.bin_rates_['_default'] = rates
            if self.target_names_:
                for name in self.target_names_:
                    self.bin_rates_[name] = rates

    def _extract_single_target_rates(self, table: pd.DataFrame, target_name: str) -> Dict[str, float]:
        """从多级表头表中提取指定目标的逾期率.

        :param table: 多级表头分箱表
        :param target_name: 目标名称
        :return: {分箱标签: 逾期率}
        """
        rates = {}

        # 查找分箱标签列
        bin_label_col = None
        for col in table.columns:
            col_name = col[-1] if isinstance(col, tuple) else col
            if col_name == self.bin_label_col:
                bin_label_col = col
                break

        if bin_label_col is None:
            raise ValueError(f"分箱表中未找到 '{self.bin_label_col}' 列")

        # 查找该目标下的坏样本率列
        bad_rate_col = None
        for col in table.columns:
            if isinstance(col, tuple):
                if col[0] == target_name and col[1] == self.bad_rate_col:
                    bad_rate_col = col
                    break
            elif col == self.bad_rate_col:
                bad_rate_col = col
                break

        if bad_rate_col is None:
            # 尝试模糊匹配：找目标下含"坏样本率"的列
            for col in table.columns:
                if isinstance(col, tuple) and col[0] == target_name:
                    col_sub = col[-1] if isinstance(col, (tuple, list)) else col
                    if '坏样本率' in str(col_sub):
                        bad_rate_col = col
                        break

        if bad_rate_col is None:
            raise ValueError(f"分箱表中未找到目标 '{target_name}' 对应的 '{self.bad_rate_col}' 列")

        # 提取逾期率（排除合计行）
        for idx, row in table.iterrows():
            label = row[bin_label_col]
            if label == '合计':
                continue
            rate = row[bad_rate_col]
            rates[str(label)] = float(rate) if pd.notna(rate) else 0.0

        return rates

    def _extract_single_target_rates_flat(self, table: pd.DataFrame) -> Dict[str, float]:
        """从单层表头表中提取逾期率.

        :param table: 单层表头分箱表
        :return: {分箱标签: 逾期率}
        """
        rates = {}

        if self.bin_label_col not in table.columns:
            raise ValueError(f"分箱表中未找到 '{self.bin_label_col}' 列")

        if self.bad_rate_col not in table.columns:
            # 尝试模糊匹配
            rate_col = None
            for c in table.columns:
                if '坏样本率' in str(c):
                    rate_col = c
                    break
            if rate_col is None:
                raise ValueError(f"分箱表中未找到 '{self.bad_rate_col}' 列")
        else:
            rate_col = self.bad_rate_col

        for idx, row in table.iterrows():
            label = row[self.bin_label_col]
            if label == '合计':
                continue
            rate = row[rate_col]
            rates[str(label)] = float(rate) if pd.notna(rate) else 0.0

        return rates

    def _extract_splits_from_raw_data(self, df: pd.DataFrame) -> None:
        """从原始数据中提取分箱切分点.

        :param df: 原始数据
        """
        if self.rules is not None:
            self.splits_ = np.array(self.rules)
        else:
            self.splits_ = None

    def _extract_splits_from_bin_table(self, df: pd.DataFrame) -> None:
        """从分箱表中解析分箱切分点.

        :param df: 分箱表
        """
        if self.bin_label_col not in df.columns:
            self.splits_ = None
            return

        labels = df[self.bin_label_col].tolist()
        splits = self._parse_bin_labels_to_splits(labels)
        self.splits_ = splits

    def _parse_bin_labels_to_splits(self, labels: List[str]) -> Optional[np.ndarray]:
        """从分箱标签解析分箱切分点.

        支持的标签格式:
        - '(-inf, 300]', '(300, 500]', '(500, +inf)'
        - '缺失', '特殊'

        :param labels: 分箱标签列表
        :return: 切分点数组
        """
        import re

        splits = []
        for label in labels:
            label = str(label).strip()
            if label in ('缺失', '特殊', '合计', ''):
                continue

            # 匹配区间格式: (left, right] 或 [left, right)
            pattern = r'[\(\[]\s*([^\s,]+)\s*,\s*([^\s\)]+)\s*[\)\]]'
            match = re.match(pattern, label)
            if match:
                right = match.group(2)
                if right not in ('+inf', 'inf'):
                    try:
                        val = float(right)
                        if val not in splits:
                            splits.append(val)
                    except ValueError:
                        pass

        if splits:
            return np.array(sorted(splits))
        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """对无标签样本进行逾期率预测.

        根据每个样本的特征值映射到对应分箱，然后赋予该分箱的逾期率，
        并应用调整系数，最终输出加权逾期率。

        :param X: 待预测数据，必须包含feature指定的列
        :return: 包含逾期率预测结果的DataFrame，新增列：
            - '{feature}_分箱': 样本所在分箱标签
            - '{feature}_基础逾期率': 分箱原始逾期率
            - '{feature}_预测逾期率': 调整后的逾期率
            - 多标签时每个目标分别输出上述三列
        """
        if not hasattr(self, 'bin_rates_'):
            raise ValueError("预估器尚未拟合，请先调用fit方法")

        X = X.copy()

        if self.feature not in X.columns:
            raise ValueError(f"待预估数据中缺少特征列: {self.feature}")

        if len(self.target_names_) <= 1:
            result = self._transform_single_target(X)
        else:
            result = self._transform_multi_target(X)

        return result

    def _transform_single_target(self, X: pd.DataFrame) -> pd.DataFrame:
        """单标签逾期率预估.

        :param X: 待预估数据
        :return: 包含预估结果的数据
        """
        rates = self.bin_rates_.get('_default', self.bin_rates_.get(self.target_names_[0] if self.target_names_ else 'target', {}))

        # 为每个样本分配分箱和逾期率
        bin_labels, base_rates = self._assign_bins_and_rates(X[self.feature], rates)

        # 应用调整系数
        coefficients_map = self._resolve_coefficients(rates)
        adjusted_rates = self._apply_coefficients(bin_labels, base_rates, coefficients_map)

        # 写入结果
        X[f'{self.feature}_分箱'] = bin_labels
        X[f'{self.feature}_基础逾期率'] = base_rates
        X[f'{self.feature}_预测逾期率'] = adjusted_rates

        return X

    def _transform_multi_target(self, X: pd.DataFrame) -> pd.DataFrame:
        """多标签逾期率预测.

        :param X: 待预测数据
        :return: 包含多标签预测结果的数据
        """
        # 分箱只需要做一次
        first_target = self.target_names_[0]
        first_rates = self.bin_rates_.get(first_target, {})
        bin_labels, _ = self._assign_bins_and_rates(X[self.feature], first_rates)
        X[f'{self.feature}_分箱'] = bin_labels

        for target_name in self.target_names_:
            rates = self.bin_rates_.get(target_name, {})
            _, base_rates = self._assign_bins_and_rates(X[self.feature], rates)

            coefficients_map = self._resolve_coefficients(rates, target_name)
            adjusted_rates = self._apply_coefficients(bin_labels, base_rates, coefficients_map)

            X[f'{self.feature}_{target_name}_基础逾期率'] = base_rates
            X[f'{self.feature}_{target_name}_预测逾期率'] = adjusted_rates

        return X

    def _assign_bins_and_rates(
        self,
        values: pd.Series,
        rates: Dict[str, float],
    ) -> Tuple[List[str], List[float]]:
        """将样本值映射到分箱，并分配对应的逾期率.

        :param values: 特征值序列
        :param rates: {分箱标签: 逾期率}
        :return: (分箱标签列表, 逾期率列表)
        """
        import re

        # 解析分箱区间
        bin_intervals = self._parse_rate_intervals(rates)

        bin_labels = []
        base_rates = []

        for val in values:
            if pd.isna(val):
                # 缺失值处理
                if '缺失' in rates:
                    bin_labels.append('缺失')
                    base_rates.append(rates['缺失'])
                else:
                    bin_labels.append('缺失')
                    base_rates.append(0.0)
                continue

            assigned = False
            for label, (left, right, left_inc, right_inc) in bin_intervals.items():
                if label in ('缺失', '特殊', '合计'):
                    continue

                # 检查是否在区间内
                in_range = True
                if left is not None:
                    if left_inc:
                        in_range = in_range and (val >= left)
                    else:
                        in_range = in_range and (val > left)
                if right is not None:
                    if right_inc:
                        in_range = in_range and (val <= right)
                    else:
                        in_range = in_range and (val < right)

                if in_range:
                    bin_labels.append(label)
                    base_rates.append(rates.get(label, 0.0))
                    assigned = True
                    break

            if not assigned:
                # 尝试特殊值匹配
                if '特殊' in rates:
                    bin_labels.append('特殊')
                    base_rates.append(rates['特殊'])
                else:
                    # 使用最近的分箱逾期率
                    bin_labels.append('未知')
                    base_rates.append(self._get_overall_rate(rates))

        return bin_labels, base_rates

    def _parse_rate_intervals(self, rates: Dict[str, float]) -> Dict[str, Tuple]:
        """解析分箱标签为区间元组.

        :param rates: {分箱标签: 逾期率}
        :return: {标签: (left, right, left_inclusive, right_inclusive)}
        """
        import re

        intervals = {}
        for label in rates:
            label_str = str(label).strip()
            if label_str in ('缺失', '特殊', '合计', '未知'):
                intervals[label_str] = (None, None, False, False)
                continue

            # 匹配 (left, right] 或 [left, right)
            pattern = r'([\(\[])\s*([^\s,]+)\s*,\s*([^\s\)]+)\s*([\)\]])'
            match = re.match(pattern, label_str)
            if match:
                left_bracket, left_str, right_str, right_bracket = match.groups()
                left_inc = left_bracket == '['
                right_inc = right_bracket == ']'

                left = None if left_str in ('-inf', 'inf') else float(left_str)
                right = None if right_str in ('+inf', 'inf', '-inf') else float(right_str)

                intervals[label_str] = (left, right, left_inc, right_inc)
            else:
                intervals[label_str] = (None, None, False, False)

        return intervals

    def _resolve_coefficients(
        self,
        rates: Dict[str, float],
        target_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """解析并确定实际使用的调整系数.

        :param rates: 各分箱逾期率
        :param target_name: 目标名称（多标签时区分）
        :return: {分箱标签: 系数}
        """
        if self.coefficients is None:
            return {label: 1.0 for label in rates}

        if isinstance(self.coefficients, (int, float)):
            return {label: float(self.coefficients) for label in rates}

        if isinstance(self.coefficients, dict):
            result = {label: 1.0 for label in rates}
            for label, coef in self.coefficients.items():
                if label in rates:
                    result[label] = float(coef)
            return result

        if self.coefficients == 'auto':
            overall_rate = self._get_overall_rate(rates)
            if overall_rate <= 0:
                return {label: 1.0 for label in rates}
            # 基于加权平均逾期率与整体逾期率的比值调整
            # 目标是使预估的整体逾期率接近实际
            weighted_sum = sum(r * self._estimate_weight(l, rates) for l, r in rates.items())
            if weighted_sum > 0:
                factor = overall_rate / weighted_sum
            else:
                factor = 1.0
            return {label: factor for label in rates}

        return {label: 1.0 for label in rates}

    def _estimate_weight(self, label: str, rates: Dict[str, float]) -> float:
        """估算分箱权重（用于auto系数模式）.

        简单实现：均匀权重。可在子类中覆盖。

        :param label: 分箱标签
        :param rates: 逾期率字典
        :return: 权重
        """
        return 1.0

    def _get_overall_rate(self, rates: Dict[str, float]) -> float:
        """获取整体逾期率（从分箱率加权计算或从合计行获取）.

        :param rates: 逾期率字典
        :return: 整体逾期率
        """
        if '合计' in rates:
            return rates['合计']
        # 简单平均
        valid_rates = [r for l, r in rates.items() if l not in ('合计', '缺失', '特殊')]
        return np.mean(valid_rates) if valid_rates else 0.0

    def _apply_coefficients(
        self,
        bin_labels: List[str],
        base_rates: List[float],
        coefficients_map: Dict[str, float],
    ) -> List[float]:
        """应用调整系数到基础逾期率.

        :param bin_labels: 分箱标签列表
        :param base_rates: 基础逾期率列表
        :param coefficients_map: 系数映射
        :return: 调整后的逾期率列表
        """
        adjusted = []
        for label, rate in zip(bin_labels, base_rates):
            coef = coefficients_map.get(label, 1.0)
            adjusted_rate = rate * coef
            adjusted.append(min(adjusted_rate, 1.0))  # 逾期率不超过1

        return adjusted

    def set_coefficients(self, coefficients: Union[float, Dict[str, float], str]) -> "OverduePredictor":
        """设置或更新逾期率调整系数.

        :param coefficients: 调整系数，支持：
            - float: 统一系数
            - dict: 按分箱标签指定系数
            - 'auto': 自动校正
            - None: 取消调整
        :return: self
        """
        self.coefficients = coefficients
        self.coefficients_ = coefficients
        return self

    def get_report(self, metric: str = 'count') -> pd.DataFrame:
        """获取逾期率预估报告.

        输出各分箱的逾期率、样本分布及预估信息。

        :param metric: 统计口径，'count'（订单口径）或 'amount'（金额口径）
        :return: 报告DataFrame

        **参考样例**

        >>> predictor = OverduePredictor(feature='score', target='target')  # 初始化逾期率预估器
        >>> predictor.fit(train_df)  # 拟合分箱表并计算各箱逾期率
        >>> report = predictor.get_report()  # 获取分箱统计报告（含逾期率、样本数等）
        >>> print(report)
        """
        if not hasattr(self, 'bin_table_'):
            raise ValueError("预估器尚未拟合，请先调用fit方法")

        table = self.bin_table_.copy()

        # 添加系数信息
        if self.coefficients is not None:
            coefficients_map = self._resolve_coefficients(
                self.bin_rates_.get('_default', list(self.bin_rates_.values())[0] if self.bin_rates_ else {})
            )

            if isinstance(table.columns, pd.MultiIndex):
                # 多级表头：在第一个目标下添加系数列
                first_target = self.target_names_[0] if self.target_names_ else table.columns[1][0]
                coef_col = (first_target, '调整系数')
                adjusted_col = (first_target, '调整后逾期率')

                rates = self.bin_rates_.get(first_target, {})
                bin_col = None
                for c in table.columns:
                    c_name = c[-1] if isinstance(c, tuple) else c
                    if c_name == self.bin_label_col:
                        bin_col = c
                        break

                if bin_col is not None:
                    coefs = []
                    adjusted = []
                    for _, row in table.iterrows():
                        label = str(row[bin_col])
                        coef = coefficients_map.get(label, 1.0)
                        coefs.append(coef)
                        rate = rates.get(label, 0.0)
                        adjusted.append(min(rate * coef, 1.0))

                    table[coef_col] = coefs
                    table[adjusted_col] = adjusted
            else:
                # 单层表头
                rates = self.bin_rates_.get('_default', {})
                coefs = []
                adjusted = []
                for _, row in table.iterrows():
                    label = str(row.get(self.bin_label_col, ''))
                    coef = coefficients_map.get(label, 1.0)
                    coefs.append(coef)
                    rate = rates.get(label, 0.0)
                    adjusted.append(min(rate * coef, 1.0))

                table['调整系数'] = coefs
                table['调整后逾期率'] = adjusted

        return table

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
        """预测逾期率（简化接口）.

        直接返回逾期率预测结果，不含分箱和基础逾期率列。

        :param X: 待预测数据
        :return: 逾期率Series（单标签）或 {目标名: 逾期率Series}（多标签）

        **参考样例**

        >>> predictor = OverduePredictor(feature='score', target='target')
        >>> predictor.fit(train_df)
        >>> predicted_rates = predictor.predict(test_df)  # 直接返回逾期率Series（单标签）或字典（多标签）
        """
        result = self.transform(X)

        if len(self.target_names_) <= 1:
            rate_col = f'{self.feature}_预测逾期率'
            return result[rate_col]
        else:
            output = {}
            for target_name in self.target_names_:
                rate_col = f'{self.feature}_{target_name}_预测逾期率'
                output[target_name] = result[rate_col]
            return output


def overdue_prediction_report(
    data: pd.DataFrame,
    feature: str,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    predict_data: Optional[pd.DataFrame] = None,
    coefficients: Optional[Union[float, Dict[str, float], str]] = None,
    method: str = 'mdlp',
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    bin_table: Optional[pd.DataFrame] = None,
    rules: Optional[List] = None,
    desc: Optional[str] = None,
    excel_writer=None,
    sheet: str = "逾期率预估报告",
    **kwargs,
) -> pd.DataFrame:
    """逾期率预估报告便捷函数.

    统一的入口函数，支持从原始数据或分箱表进行逾期率预估，
    并可输出包含预估结果的报告。

    :param data: 有标签数据集（含target或逾期天数），或分箱表
    :param feature: 特征名称
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称或列表
    :param dpds: 逾期定义天数或列表
    :param predict_data: 待预估的无标签数据（可选），
        传入后会计算各样本的预估逾期率
    :param coefficients: 逾期率调整系数
    :param method: 分箱方法，默认'mdlp'
    :param max_n_bins: 最大分箱数，默认5
    :param min_bin_size: 每箱最小样本占比，默认0.05
    :param missing_separate: 是否将缺失值单独分箱，默认True
    :param bin_table: 现成分箱表（可选），传入后直接使用而不从data计算
    :param rules: 自定义分箱切分点列表
    :param desc: 特征描述
    :param excel_writer: Excel文件路径或ExcelWriter对象（可选），用于输出报告
    :param sheet: Excel工作表名称
    :return: 包含预估结果的DataFrame

    **参考样例**

    >>> from hscredit.report.overdue_estimator import overdue_estimation_report
    >>>
    >>> # 方式一：从原始数据生成报告（自动拟合+预估）
    >>> report = overdue_prediction_report(
    ...     train_df, feature='score', target='target',
    ...     predict_data=test_df, coefficients=1.1
    ... )
    >>>
    >>> # 方式二：从分箱表生成报告（复用现成分箱逾期率）
    >>> report = overdue_estimation_report(
    ...     bin_table, feature='score',
    ...     predict_data=test_df
    ... )
    >>>
    >>> # 方式三：多标签场景（同时预估MOB1/MOB3等多个时间窗口的逾期率）
    >>> report = overdue_estimation_report(
    ...     train_df, feature='score',
    ...     overdue='MOB1', dpds=[7, 15, 30],
    ...     predict_data=test_df,
    ...     excel_writer='overdue_report.xlsx'
    ... )
    """
    # 确定拟合数据来源
    fit_data = bin_table if bin_table is not None else data

    predictor = OverduePredictor(
        feature=feature,
        target=target,
        overdue=overdue,
        dpds=dpds,
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        missing_separate=missing_separate,
        coefficients=coefficients,
        rules=rules,
        desc=desc,
    )

    predictor.fit(fit_data)

    # 获取分箱报告
    report = predictor.get_report()

    # 对无标签数据预测
    if predict_data is not None and isinstance(predict_data, pd.DataFrame):
        predict_result = predictor.transform(predict_data)
        # 将预测结果附加到报告
        predict_summary = _build_prediction_summary(predict_result, predictor)
        report = pd.concat([report, pd.DataFrame([{'_section': '分隔'}]), predict_summary], ignore_index=True)

    # 输出到Excel
    if excel_writer is not None:
        try:
            from ..excel import ExcelWriter, dataframe2excel
            from ..utils import init_setting

            init_setting()

            if isinstance(excel_writer, str):
                writer = ExcelWriter()
                worksheet = writer.get_sheet_by_name(sheet)

                end_row, end_col = dataframe2excel(
                    report, writer, worksheet,
                    sheet_name=sheet if isinstance(excel_writer, str) else None,
                    percent_cols=['坏样本率', '样本占比', '好样本占比', '坏样本占比',
                                  '调整后逾期率', '预估逾期率'],
                    condition_cols=['坏样本率', '调整后逾期率'],
                    start_row=2,
                )

                writer.save(excel_writer)
        except Exception:
            warnings.warn("Excel输出失败，请确保已安装openpyxl")

    return report


def _build_prediction_summary(predict_result: pd.DataFrame, predictor: OverduePredictor) -> pd.DataFrame:
    """构建预测汇总统计.

    :param predict_result: transform输出结果
    :param predictor: 预测器实例
    :return: 汇总统计DataFrame
    """
    rows = []
    feature = predictor.feature

    if len(predictor.target_names_) <= 1:
        rate_col = f'{feature}_预测逾期率'
        bin_col = f'{feature}_分箱'

        if rate_col in predict_result.columns:
            rows.append({
                '指标名称': feature,
                '指标含义': '预测汇总',
                '分箱标签': '预测样本总数',
                '预测逾期率': len(predict_result),
            })
            rows.append({
                '指标名称': feature,
                '指标含义': '预测汇总',
                '分箱标签': '平均预测逾期率',
                '预测逾期率': predict_result[rate_col].mean(),
            })

            if bin_col in predict_result.columns:
                bin_dist = predict_result[bin_col].value_counts()
                for bin_label, count in bin_dist.items():
                    subset = predict_result[predict_result[bin_col] == bin_label]
                    avg_rate = subset[rate_col].mean()
                    rows.append({
                        '指标名称': feature,
                        '指标含义': '分箱预测',
                        '分箱标签': str(bin_label),
                        '样本总数': count,
                        '预测逾期率': avg_rate,
                    })
    else:
        for target_name in predictor.target_names_:
            rate_col = f'{feature}_{target_name}_预测逾期率'
            if rate_col in predict_result.columns:
                rows.append({
                    '指标名称': feature,
                    '指标含义': f'{target_name} 预测汇总',
                    '分箱标签': '平均预测逾期率',
                    '预测逾期率': predict_result[rate_col].mean(),
                })

    return pd.DataFrame(rows)
