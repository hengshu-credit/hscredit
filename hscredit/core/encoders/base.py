"""编码器基类.

提供统一的编码器接口和通用功能。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseEncoder(BaseEstimator, TransformerMixin, ABC):
    """编码器基类.

    所有编码器的抽象基类，提供统一的接口和通用功能。
    遵循sklearn Transformer接口规范。

    属性:
        cols: 需要编码的列名列表。
        drop_invariant: 是否删除方差为0的列。
        return_df: 是否返回DataFrame。
        handle_unknown: 处理未知类别的方式。
        handle_missing: 处理缺失值的方式。
        mapping_: 编码映射字典。
        cols_: 实际进行编码的列名列表。
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
    ):
        """初始化编码器基类。

        :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列。
        :param drop_invariant: 是否删除方差为0的列，默认为False。
        :param return_df: 是否返回DataFrame，默认为True。
        :param handle_unknown: 处理未知类别的方式，默认为'value'。
            - 'value': 使用默认值（通常是0）
            - 'error': 抛出错误
            - 'return_nan': 返回NaN
        :param handle_missing: 处理缺失值的方式，默认为'value'。
            - 'value': 使用默认值（通常是0）
            - 'error': 抛出错误
            - 'return_nan': 返回NaN
        """
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

        self.mapping_: Dict = {}
        self.cols_: Optional[List[str]] = None
        self._dropped_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseEncoder':
        """拟合编码器。

        :param X: 训练数据。
        :param y: 目标变量。对于有监督编码器（如WOE、Target）必须提供。
        :return: 编码器自身。
        """
        X = self._check_input(X)

        if self.cols is None:
            self.cols_ = self._get_category_cols(X)
        else:
            self.cols_ = [c for c in self.cols if c in X.columns]

        if len(self.cols_) == 0:
            return self

        if self.drop_invariant:
            self._dropped_cols = self._find_invariant_cols(X)
            self.cols_ = [c for c in self.cols_ if c not in self._dropped_cols]

        self._fit(X, y)

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, np.ndarray]:
        """转换数据。

        :param X: 需要转换的数据。
        :param y: 目标变量。某些编码器需要。
        :return: 编码后的数据，类型由return_df参数决定。
        """
        X = self._check_input(X)

        if not hasattr(self, 'cols_') or self.cols_ is None:
            raise ValueError("编码器尚未拟合，请先调用fit()")

        X_transformed = X.copy()

        X_transformed = self._transform(X_transformed, y)

        if not self.return_df:
            return X_transformed.values
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        :param X: 训练数据。
        :param y: 目标变量。
        :return: 编码后的数据。
        """
        return self.fit(X, y).transform(X, y)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """子类实现的具体拟合逻辑。"""
        pass

    @abstractmethod
    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """子类实现的具体转换逻辑。"""
        pass

    def _check_input(self, X) -> pd.DataFrame:
        """检查并转换输入数据。

        :param X: 输入数据。
        :return: 转换后的DataFrame。
        :raises TypeError: 当输入类型不正确时抛出。
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"输入必须是DataFrame或ndarray，got {type(X)}")
        return X

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别类别型列。

        :param X: 输入数据。
        :return: 类别型列名列表。
        """
        return X.select_dtypes(include=['object', 'category']).columns.tolist()

    def _find_invariant_cols(self, X: pd.DataFrame) -> List[str]:
        """查找方差为0的列。

        :param X: 输入数据。
        :return: 不变列名列表。
        """
        invariant_cols = []
        for col in self.cols_:
            if X[col].nunique() <= 1:
                invariant_cols.append(col)
        return invariant_cols

    def get_mapping(self) -> Dict[str, Any]:
        """获取编码映射。

        :return: 编码映射字典。
        """
        return self.mapping_

    def export_mapping(self) -> Dict[str, Any]:
        """导出编码映射（可序列化）。

        :return: 可序列化的编码映射字典。
        """
        return {
            'encoder_type': self.__class__.__name__,
            'cols': self.cols,
            'cols_': self.cols_,
            'mapping_': self._serialize_mapping(self.mapping_),
            'drop_invariant': self.drop_invariant,
            'handle_unknown': self.handle_unknown,
            'handle_missing': self.handle_missing,
        }

    def import_mapping(self, mapping: Dict[str, Any]):
        """导入编码映射。

        :param mapping: 编码映射字典。
        """
        self.cols = mapping.get('cols')
        self.cols_ = mapping.get('cols_')
        self.mapping_ = self._deserialize_mapping(mapping.get('mapping_', {}))
        self.drop_invariant = mapping.get('drop_invariant', False)
        self.handle_unknown = mapping.get('handle_unknown', 'value')
        self.handle_missing = mapping.get('handle_missing', 'value')

    def _serialize_mapping(self, mapping: Dict) -> Dict:
        """序列化映射（处理特殊类型）。

        :param mapping: 映射字典。
        :return: 序列化后的字典。
        """
        serialized = {}
        for key, value in mapping.items():
            if isinstance(value, pd.Series):
                serialized[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_mapping(value)
            else:
                serialized[key] = value
        return serialized

    def _deserialize_mapping(self, mapping: Dict) -> Dict:
        """反序列化映射。

        :param mapping: 序列化后的字典。
        :return: 反序列化后的字典。
        """
        deserialized = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                deserialized[key] = pd.Series(value)
            else:
                deserialized[key] = value
        return deserialized
