"""规则挖掘基础类和工具函数."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.base import BaseEstimator


class BaseRuleMiner(BaseEstimator, ABC):
    """规则挖掘器基类.
    
    所有规则挖掘器的基类，遵循sklearn API规范，
    fit方法兼容scorecardpipeline风格。
    
    :param target: 目标变量列名，默认为'target'
    :param exclude_cols: 需要排除的列名列表
    """
    
    def __init__(
        self,
        target: str = 'target',
        exclude_cols: Optional[List[str]] = None
    ):
        self.target = target
        self.exclude_cols = exclude_cols or []
        self._is_fitted = False
    
    def _check_input_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """检查并处理输入数据.
        
        支持三种模式:
        1. sklearn风格: fit(X, y)
        2. scorecardpipeline风格: fit(df) - df包含特征和目标列
        3. 仅特征: fit(X) - 用于无监督规则挖掘
        
        :param X: 输入数据
        :param y: 目标变量（可选）
        :return: (处理后的X, y)
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X = X.copy()
        
        # 如果y为None且target列存在于X中，提取target
        if y is None and self.target in X.columns:
            y = X[self.target].copy()
            X = X.drop(columns=[self.target])
        
        # 排除指定列
        for col in self.exclude_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        return X, y
    
    def _get_numeric_features(self, X: pd.DataFrame) -> List[str]:
        """获取数值型特征列表."""
        return [col for col in X.columns 
                if pd.api.types.is_numeric_dtype(X[col])]
    
    def _get_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """获取类别型特征列表."""
        return [col for col in X.columns 
                if not pd.api.types.is_numeric_dtype(X[col])]
    
    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        """拟合挖掘器."""
        pass
    
    @abstractmethod
    def get_rules(self) -> List[Dict[str, Any]]:
        """获取挖掘的规则."""
        pass


def check_features_valid(X: pd.DataFrame, required_features: List[str]) -> None:
    """检查必需的特征是否都存在.
    
    :param X: 输入数据
    :param required_features: 必需的特征列表
    :raises ValueError: 当必需特征不存在时
    """
    missing = set(required_features) - set(X.columns)
    if missing:
        raise ValueError(f"缺少必需的特征: {missing}")


def format_rule_expression(
    feature: str,
    threshold: float,
    operator: str = '>=',
    precision: int = 4
) -> str:
    """格式化规则表达式.
    
    :param feature: 特征名
    :param threshold: 阈值
    :param operator: 操作符
    :param precision: 数值精度
    :return: 规则表达式字符串
    """
    return f"{feature} {operator} {threshold:.{precision}f}"


def calculate_lift(
    subset_badrate: float,
    overall_badrate: float
) -> float:
    """计算lift值.
    
    :param subset_badrate: 子集坏账率
    :param overall_badrate: 整体坏账率
    :return: lift值
    """
    if overall_badrate <= 0:
        return 0.0
    return subset_badrate / overall_badrate


def calculate_ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算KS统计量.
    
    :param y_true: 真实标签
    :param y_score: 预测分数
    :return: KS值
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.max(tpr - fpr)


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """合并重叠的区间.
    
    :param intervals: 区间列表，每个元素为(start, end)
    :return: 合并后的区间列表
    """
    if not intervals:
        return []
    
    # 按起始点排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # 有重叠
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


class RuleCondition:
    """规则条件类.
    
    表示单个规则条件，如 "age >= 18"
    
    :param feature: 特征名
    :param threshold: 阈值
    :param operator: 操作符，支持 '>=', '<=', '>', '<', '==', '!='
    """
    
    VALID_OPERATORS = {'>=', '<=', '>', '<', '==', '!='}
    
    def __init__(
        self,
        feature: str,
        threshold: Union[float, str],
        operator: str = '>='
    ):
        if operator not in self.VALID_OPERATORS:
            raise ValueError(f"不支持的操作符: {operator}，可选: {self.VALID_OPERATORS}")
        
        self.feature = feature
        self.threshold = threshold
        self.operator = operator
    
    def evaluate(self, X: pd.DataFrame) -> pd.Series:
        """评估条件在数据上的匹配情况.
        
        :param X: 输入数据
        :return: 匹配结果Series
        """
        if self.feature not in X.columns:
            raise ValueError(f"特征 '{self.feature}' 不存在")
        
        values = X[self.feature]
        
        if self.operator == '>=':
            return values >= self.threshold
        elif self.operator == '<=':
            return values <= self.threshold
        elif self.operator == '>':
            return values > self.threshold
        elif self.operator == '<':
            return values < self.threshold
        elif self.operator == '==':
            return values == self.threshold
        elif self.operator == '!=':
            return values != self.threshold
    
    def __repr__(self) -> str:
        return f"RuleCondition({self.feature} {self.operator} {self.threshold})"
    
    def to_expression(self) -> str:
        """转换为表达式字符串."""
        return f"{self.feature} {self.operator} {self.threshold}"


class MinedRule:
    """挖掘出的规则类.
    
    封装一条完整的规则，包括条件、指标和元信息。
    
    :param conditions: 规则条件列表
    :param metric_score: 规则评估分数
    :param description: 规则描述
    :param metadata: 额外的元数据
    """
    
    def __init__(
        self,
        conditions: List[RuleCondition],
        metric_score: float = 0.0,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.conditions = conditions
        self.metric_score = metric_score
        self.description = description
        self.metadata = metadata or {}
    
    def evaluate(self, X: pd.DataFrame) -> pd.Series:
        """评估规则在数据上的匹配情况.
        
        :param X: 输入数据
        :return: 匹配结果Series
        """
        if not self.conditions:
            return pd.Series(True, index=X.index)
        
        result = self.conditions[0].evaluate(X)
        for condition in self.conditions[1:]:
            result &= condition.evaluate(X)
        
        return result
    
    def to_expression(self) -> str:
        """转换为表达式字符串."""
        if not self.conditions:
            return "True"
        
        return " AND ".join([c.to_expression() for c in self.conditions])
    
    def to_rule_object(self):
        """转换为hscredit Rule对象."""
        from ..rule import Rule
        expr = self.to_expression()
        return Rule(
            expr=expr,
            name=f"MinedRule_{id(self)}",
            description=self.description,
            weight=1.0
        )
    
    def __repr__(self) -> str:
        return f"MinedRule({self.to_expression()}, score={self.metric_score:.4f})"
