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

    **参考样例**

    所有子类（WOE/Target/Count/OneHot/Ordinal/Quantile/CatBoost/GBM/Cardinality）
    共享下述两种调用风格::

        >>> from hscredit.core.encoders import WOEEncoder
        >>> # sklearn 风格：X、y 分开传
        >>> enc = WOEEncoder(cols=['city'])
        >>> X_enc = enc.fit_transform(X, y)
        >>> # scorecardpipeline 风格：目标列在 df 中，初始化指定 target
        >>> enc = WOEEncoder(target='target', cols=['city'])
        >>> X_enc = enc.fit_transform(df)
        >>> enc.get_mapping('city')        # 查看某列的类别→编码映射
        >>> enc.export_mapping()           # 导出可序列化映射

    **引用**

    接口约定（``cols`` / ``drop_invariant`` / ``handle_unknown`` / ``handle_missing`` /
    ``return_df``）对齐 category_encoders 库：
    https://contrib.scikit-learn.org/category_encoders/
    """

    #: 子类声明的“额外拟合状态”属性名。这些状态是 transform 正确性所必需，
    #: 但不在 mapping_ 中（如均值/类别清单），需随 export_mapping/import_mapping 一并往返。
    _EXTRA_STATE_ATTRS: List[str] = []

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = "value",
        handle_missing: str = "value",
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
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseEncoder":
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
            self._is_fitted = True
            return self

        if self.drop_invariant:
            self._dropped_cols = self._find_invariant_cols(X)
            self.cols_ = [c for c in self.cols_ if c not in self._dropped_cols]

        # 统一缺失值策略校验：handle_missing='error' 时，fit 阶段任一编码列含缺失即报错
        # （与 transform 阶段保持一致）
        if self.handle_missing == "error":
            for col in self.cols_:
                if col in X.columns and X[col].isna().any():
                    raise ValueError(f"列'{col}'包含缺失值，但 handle_missing='error'")

        self._fit(X, y)

        self._is_fitted = True
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

        if not hasattr(self, "cols_") or self.cols_ is None:
            raise NotFittedError("编码器尚未拟合，请先调用fit()")

        # 统一缺失值策略校验：handle_missing='error' 时，任一编码列含缺失即报错
        if self.handle_missing == "error":
            for col in self.cols_:
                if col in X.columns and X[col].isna().any():
                    raise ValueError(f"列'{col}'包含缺失值，但 handle_missing='error'")

        X_transformed = X.copy()
        X_transformed = self._transform(X_transformed, y)

        # scorecardpipeline 风格: 如果输入的 X 中包含 target 列，透传到输出
        target_col = getattr(self, "target", None)
        if (
            target_col is not None
            and isinstance(X_transformed, pd.DataFrame)
            and target_col not in X_transformed.columns
            and target_col in X.columns
        ):
            # 按位置赋值，避免非默认索引下 concat 对齐错位
            X_transformed[target_col] = np.asarray(X[target_col])

        if not self.return_df:
            # 处理稀疏矩阵的情况
            if hasattr(X_transformed, "toarray"):
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
                X = pd.DataFrame(X, columns=["feature"])
            else:
                X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"输入必须是DataFrame、ndarray或Series，got {type(X)}")
        return X

    def _extract_target(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
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
        return X.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def _sort_categories(categories: List[Any]) -> List[Any]:
        """对类别取值做类型安全的稳定排序。

        优先按原生类型排序（同质数值/字符串保持自然顺序）；当类别为混合类型
        （如同时含 int 与 str）导致原生比较抛 ``TypeError`` 时，回退到按字符串排序，
        避免崩溃并保证结果确定。

        :param categories: 待排序的类别列表
        :return: 排序后的类别列表
        """
        try:
            return sorted(categories)
        except TypeError:
            return sorted(categories, key=str)

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

    def get_mapping(self, col: Optional[str] = None) -> Dict[str, Any]:
        """获取编码映射。

        :param col: 列名。如果提供，返回该列的映射 {category: encoded_value}；
            如果为None，返回所有列的映射 {col: {category: encoded_value}}
        :return: 编码映射字典
        :raises NotFittedError: 当编码器尚未拟合时抛出
        :raises FeatureNotFoundError: 当指定的 col 不在编码器中时抛出
        """
        fitted = getattr(self, "_is_fitted", False) or bool(getattr(self, "mapping_", None))
        if not fitted:
            raise NotFittedError("编码器尚未拟合，请先调用fit()")
        if col is None:
            return self.mapping_
        if col not in self.mapping_:
            raise FeatureNotFoundError(f"特征 '{col}' 未找到")
        return self.mapping_[col]

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆编码（将编码值还原为原始类别）。

        编码器基类的默认实现：多数有监督/有损编码器（WOE/Target/Count/Quantile/CatBoost/GBM）
        无法唯一还原原始类别，调用时抛出 :class:`NotImplementedError`。
        支持逆编码的子类（OneHot/Ordinal/Cardinality）会覆盖本方法。

        :param X: 编码后的数据
        :return: 逆编码后的数据
        :raises NotImplementedError: 当该编码器不支持逆编码时抛出
        """
        raise NotImplementedError(f"{self.__class__.__name__} 不支持逆编码（inverse_transform）")

    def __contains__(self, feature: str) -> bool:
        """检查特征是否在编码器中（支持 `feature in encoder` 语法）."""
        if not hasattr(self, "mapping_") or self.mapping_ is None:
            return False
        return feature in self.mapping_

    def __getitem__(self, feature: str):
        """通过 `encoder['feature']` 获取该特征的编码映射（toad/scorecardpipeline风格）."""
        if not hasattr(self, "mapping_") or self.mapping_ is None or len(self.mapping_) == 0:
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
            "encoder_type": self.__class__.__name__,
            "cols": self.cols,
            "cols_": self.cols_,
            "mapping_": self._serialize_mapping(self.mapping_),
            "drop_invariant": self.drop_invariant,
            "handle_unknown": self.handle_unknown,
            "handle_missing": self.handle_missing,
            # 子类声明的额外拟合状态（如 global_mean_、categories_），
            # 否则 import_mapping 后这些 transform 必需的状态丢失，导致编码结果错误
            "extra_state": self._export_extra_state(),
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
        self.cols = mapping.get("cols")
        self.cols_ = mapping.get("cols_")
        self.mapping_ = self._deserialize_mapping(mapping.get("mapping_", {}))
        self.drop_invariant = mapping.get("drop_invariant", False)
        self.handle_unknown = mapping.get("handle_unknown", "value")
        self.handle_missing = mapping.get("handle_missing", "value")
        self._import_extra_state(mapping.get("extra_state", {}))
        self._is_fitted = True
        return self

    def _export_extra_state(self) -> Dict[str, Any]:
        """导出子类声明的额外拟合状态。

        遍历 :attr:`_EXTRA_STATE_ATTRS`，将 ``pd.Series`` 转为 ``dict``（便于序列化），
        其余按原样导出。

        :return: 额外状态字典 {属性名: 值}
        """
        state: Dict[str, Any] = {}
        for attr in self._EXTRA_STATE_ATTRS:
            value = getattr(self, attr, None)
            state[attr] = value.to_dict() if isinstance(value, pd.Series) else value
        return state

    def _import_extra_state(self, state: Dict[str, Any]) -> None:
        """还原子类声明的额外拟合状态。

        :param state: 由 :meth:`_export_extra_state` 导出的状态字典
        """
        for attr in self._EXTRA_STATE_ATTRS:
            if attr in state:
                setattr(self, attr, state[attr])

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

        ``mapping_`` 的内层值约定为 ``dict`` （``{类别: 编码值}``），所有编码器子类均如此存储。
        因此反序列化时保持内层 ``dict`` 不变，避免被误转为 ``pd.Series`` 而破坏
        ``get_mapping`` / ``__getitem__`` 的 dict 返回契约。

        :param mapping: 序列化后的字典
        :return: 反序列化后的字典
        """
        deserialized = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                deserialized[key] = dict(value)
            else:
                deserialized[key] = value
        return deserialized
