"""编码器基类.

提供统一的编码器接口和通用功能。
所有编码器都继承此类，确保API的一致性。

设计原则:
1. 参数命名统一，与其他库保持一致
2. 支持高度自定义，但提供合理默认值
3. 遵循sklearn API风格，同时支持scorecardpipeline风格

API风格说明:
- sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
- scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...exceptions import FeatureNotFoundError, NotFittedError


class BaseEncoder(BaseEstimator, TransformerMixin, ABC):
    """编码器基类.

    所有编码器的抽象基类，提供统一的接口和通用功能。
    遵循sklearn Transformer接口规范，同时支持scorecardpipeline风格。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True
    :param handle_unknown: 处理未知类别的方式，默认为'value'
        - 'value': 使用默认值（通常是0或全局均值）
        - 'error': 抛出错误
        - 'return_nan': 返回NaN
    :param handle_missing: 处理缺失值的方式，默认为'value'
        - 'value': 使用默认值（通常是0或全局均值）
        - 'error': 抛出错误
        - 'return_nan': 返回NaN
    :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y

    **属性**

    - mapping\_: 编码映射字典，格式为 {col: {category: encoded_value}}
    - cols_: 实际进行编码的列名列表（经过自动识别或过滤后）
    - _dropped_cols: 被删除的方差为0的列
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        target: Optional[str] = None,
    ):
        """初始化编码器基类。

        :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.target = target

        self.mapping_: Dict = {}
        self.cols_: Optional[List[str]] = None
        self._dropped_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseEncoder':
        """拟合编码器。

        支持两种API风格:
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级: fit时传入的y > 从X中提取target列

        :param X: 训练数据，shape (n_samples, n_features) 或包含目标列的完整数据框
        :param y: 目标变量，对于有监督编码器（如WOE、Target）必须提供。
                  如果为None且初始化时提供了target参数，则从X中提取target列
        :return: 拟合后的编码器自身

        **注意**

        fit方法会进行以下操作:
        1. 数据验证和预处理
        2. 自动识别类别型列（如果cols为None）
        3. 删除方差为0的列（如果drop_invariant=True）
        4. 计算编码映射
        """
        X = self._check_input(X)

        # 处理两种API风格：如果y为None且提供了target参数，从X中提取目标列
        X, y = self._extract_target(X, y)

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

        将原始类别特征值转换为编码后的数值。
        这是编码器的核心方法，用于将新数据应用到已训练的编码规则。

        :param X: 需要转换的数据，shape (n_samples, n_features)
            - 支持DataFrame
            - 列名必须与fit时的特征名一致
        :param y: 目标变量，某些编码器需要，默认为None
        :return: 编码后的数据，类型由return_df参数决定
        :raises ValueError: 当编码器尚未拟合时抛出

        **注意**

        transform方法会自动处理:
        1. 缺失值: 根据handle_missing参数处理
        2. 未知类别: 根据handle_unknown参数处理
        """
        X = self._check_input(X)

        if not hasattr(self, 'cols_') or self.cols_ is None:
            raise ValueError("编码器尚未拟合，请先调用fit()")

        X_transformed = X.copy()

        X_transformed = self._transform(X_transformed, y)

        if not self.return_df:
            # 处理稀疏矩阵的情况
            if hasattr(X_transformed, 'toarray'):
                # 已经是稀疏矩阵，直接返回
                return X_transformed
            return X_transformed.values
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        支持两种API风格:
        1. sklearn风格: fit_transform(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit_transform(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        :param X: 训练数据，shape (n_samples, n_features) 或包含目标列的完整数据框
        :param y: 目标变量，对于某些编码器是必需的。
                  如果为None且初始化时提供了target参数，则从X中提取target列
        :return: 编码后的数据
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

        :param X: 输入数据，支持DataFrame、ndarray或Series
        :return: 转换后的DataFrame
        :raises TypeError: 当输入类型不正确时抛出
        """
        if isinstance(X, pd.Series):
            # 将Series转换为DataFrame
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                # 一维数组转换为单列DataFrame
                X = pd.DataFrame(X, columns=['feature'])
            else:
                X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"输入必须是DataFrame、ndarray或Series，got {type(X)}")
        return X

    def _extract_target(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """提取目标变量，支持两种API风格。

        优先级: fit时传入的y > 从X中提取target列

        :param X: 输入数据框
        :param y: 目标变量（可能为None）
        :return: (X_features, y_target) 元组
                  X_features: 不包含目标列的特征数据框
                  y_target: 目标变量Series或None
        """
        # 如果y不为None，直接使用sklearn风格
        if y is not None:
            return X, y

        # 如果y为None且提供了target参数，从X中提取目标列（scorecardpipeline风格）
        if self.target is not None:
            if self.target not in X.columns:
                raise ValueError(f"目标列'{self.target}'不在数据框中。可用的列: {list(X.columns)}")

            y_extracted = X[self.target].copy()
            X_features = X.drop(columns=[self.target])
            return X_features, y_extracted

        # y为None且没有提供target参数，返回原数据
        return X, None

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别类别型列。

        :param X: 输入数据
        :return: 类别型列名列表
        """
        return X.select_dtypes(include=['object', 'category']).columns.tolist()

    def _find_invariant_cols(self, X: pd.DataFrame) -> List[str]:
        """查找方差为0的列。

        :param X: 输入数据
        :return: 不变列名列表
        """
        invariant_cols = []
        for col in self.cols_:
            if X[col].nunique() <= 1:
                invariant_cols.append(col)
        return invariant_cols

    def get_mapping(self) -> Dict[str, Any]:
        """获取编码映射。

        :return: 编码映射字典，格式为 {col: {category: encoded_value}}
        """
        return self.mapping_

    def __getitem__(self, feature: str):
        """通过 `encoder['feature']` 获取该特征的编码映射（toad/scorecardpipeline风格）."""
        if not hasattr(self, 'mapping_') or self.mapping_ is None or len(self.mapping_) == 0:
            raise NotFittedError("编码器尚未拟合，请先调用fit()")

        if feature not in self.mapping_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self.mapping_[feature]

    def export_mapping(self) -> Dict[str, Any]:
        """导出编码映射（可序列化）。

        :return: 可序列化的编码映射字典

        **参考样例**

        >>> encoder.fit(X, y)
        >>> mapping = encoder.export_mapping()
        >>> import json
        >>> with open('encoder_mapping.json', 'w') as f:
        ...     json.dump(mapping, f)
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

        :param mapping: 编码映射字典

        **参考样例**

        >>> import json
        >>> with open('encoder_mapping.json', 'r') as f:
        ...     mapping = json.load(f)
        >>> encoder.import_mapping(mapping)
        """
        self.cols = mapping.get('cols')
        self.cols_ = mapping.get('cols_')
        self.mapping_ = self._deserialize_mapping(mapping.get('mapping_', {}))
        self.drop_invariant = mapping.get('drop_invariant', False)
        self.handle_unknown = mapping.get('handle_unknown', 'value')
        self.handle_missing = mapping.get('handle_missing', 'value')

    def _serialize_mapping(self, mapping: Dict) -> Dict:
        """序列化映射（处理特殊类型）。

        :param mapping: 映射字典
        :return: 序列化后的字典
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

        :param mapping: 序列化后的字典
        :return: 反序列化后的字典
        """
        deserialized = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                deserialized[key] = pd.Series(value)
            else:
                deserialized[key] = value
        return deserialized
