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
from dataclasses import dataclass
from copy import deepcopy
import inspect
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from ...exceptions import FeatureNotFoundError, NotFittedError
from ...utils.parallel import ParallelizableMixin, ParallelWorkload
from ...utils.serialization import ArtifactSerializableMixin
from ...utils.data_contracts import prepare_xy
from ._category_protocol import CategoryToken, MISSING, UNKNOWN, OTHER, pack, unpack


@dataclass(frozen=True)
class _FloatNaNBucket:
    """浮点 NaN 的稳定类型桶；仅用于内部查找，不作为用户类别键。"""

    family: str
    dtype: Optional[str] = None


_PYTHON_FLOAT_NAN_BUCKET = _FloatNaNBucket("python")
_FLOAT_NAN_REPRESENTATIVES = {_PYTHON_FLOAT_NAN_BUCKET: float("nan")}
for _float_dtype in (np.float16, np.float32, np.float64, np.longdouble):
    _bucket = _FloatNaNBucket("numpy", np.dtype(_float_dtype).str)
    _FLOAT_NAN_REPRESENTATIVES.setdefault(_bucket, _float_dtype("nan"))


def _fit_encoder_column_worker(task):
    """在隔离编码器候选对象上拟合一个列任务。"""
    ordinal, column, candidate, values, y = task
    payload = candidate._fit_column(column, values, y)
    return ordinal, column, payload


def _transform_encoder_column_worker(task):
    """只读转换一个编码列并保留提交序号。"""
    ordinal, column, encoder, values, y, context = task
    transformed = encoder._transform_column(column, values, y, context)
    return ordinal, column, transformed


class BaseEncoder(ParallelizableMixin, ArtifactSerializableMixin, BaseEstimator, TransformerMixin, ABC):
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
    :param passthrough_target: 默认False，转换仅输出特征，可安全连接原生sklearn模型；
        True显式恢复旧表格流程的目标列透传，不能用于无需标签预测的Pipeline
    :param n_jobs: 并行工作数，默认为-1；None沿用旧串行行为
    :param parallel_backend: joblib并行后端，默认为None
    :param parallel_config: joblib扩展配置，默认为None

    **属性**

    - mapping\\_: 编码映射字典，格式为 {col: {category: encoded_value}}
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

    artifact_kind = "编码器"

    #: 子类声明的“额外拟合状态”属性名。这些状态是 transform 正确性所必需，
    #: 但不在 mapping_ 中（如均值/类别清单），需随 export_mapping/import_mapping 一并往返。
    _EXTRA_STATE_ATTRS: List[str] = []
    _TARGET_TYPE = None
    _OUTPUT_ATTRS = {}

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = "value",
        handle_missing: str = "value",
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        passthrough_target: bool = False,
    ):
        """初始化编码器基类。

        :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        :param n_jobs: 并行工作数，默认为-1；None沿用旧串行行为
        :param parallel_backend: joblib并行后端，默认为None
        :param parallel_config: joblib扩展配置，默认为None
        """
        self.cols = cols
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.target = target
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config
        self.passthrough_target = passthrough_target

        self.mapping_: Dict = {}
        self.cols_: Optional[List[str]] = None
        self._dropped_cols: List[str] = []
        self._is_fitted: bool = False

    def _set_input_feature_attributes(self, X: pd.DataFrame) -> None:
        """记录 sklearn 兼容的输入特征元数据。"""
        feature_names = list(X.columns)
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

    def _adopt_fitted_state(self, candidate):
        public_params = {name: getattr(self, name) for name in self.get_params(deep=False)}
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
        self.__dict__.update(public_params)

    def _validate_public_policies(self):
        if not isinstance(self.passthrough_target, (bool, np.bool_)):
            raise ValueError("passthrough_target 必须为布尔值")
        if not isinstance(self.return_df, (bool, np.bool_)):
            raise ValueError("return_df 必须为布尔值")
        if self.handle_missing not in ("value", "return_nan", "error"):
            raise ValueError("handle_missing 必须为 value、return_nan 或 error")
        if self.handle_unknown not in ("value", "return_nan", "error", "ignore", "other"):
            raise ValueError("handle_unknown 配置无效")

    def __setstate__(self, state):
        """补齐新增配置，保留旧完整制品的既有变换语义和已学习状态。"""
        legacy_target_mode = "passthrough_target" not in state
        super().__setstate__(state)
        for name, parameter in inspect.signature(type(self).__init__).parameters.items():
            if name != "self" and parameter.default is not inspect.Parameter.empty and not hasattr(self, name):
                setattr(self, name, deepcopy(parameter.default))
        self.__dict__.setdefault("_dropped_cols", [])
        self.__dict__.setdefault("_target_in_fit_", bool(getattr(self, "target", None)))
        if legacy_target_mode and getattr(self, "target", None) is not None:
            self.passthrough_target = True
            warnings.warn("旧编码器制品沿用目标透传；用于纯特征Pipeline前请设置 passthrough_target=False 并重新训练下游模型", FutureWarning)
        if type(self).__name__ == "WOEEncoder":
            self.__dict__.setdefault("_legacy_string_keys_", True)
        if type(self).__name__ == "CountEncoder":
            self.__dict__.setdefault("infrequent_categories_", {})
            self.__dict__.setdefault("_legacy_other_routing_", any("__OTHER__" in mapping for mapping in self.mapping_.values()))
        for mapping in getattr(self, "mapping_", {}).values():
            if not isinstance(mapping, dict):
                continue
            if legacy_target_mode and "__UNKNOWN__" in mapping:
                mapping[UNKNOWN] = mapping.pop("__UNKNOWN__")
            if type(self).__name__ == "CountEncoder":
                if legacy_target_mode and "__OTHER__" in mapping:
                    mapping[OTHER] = mapping.pop("__OTHER__")
            elif legacy_target_mode:
                for key in list(mapping):
                    if self._is_float_nan_key(key):
                        mapping[MISSING] = mapping.pop(key)

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
        if not getattr(self, "_fit_transaction_active", False):
            candidate = clone(self)
            candidate._fit_transaction_active = True
            candidate.fit(X, y)
            candidate.__dict__.pop("_fit_transaction_active", None)

            # 拟合不得替换调用方传入的可变构造参数；仅提交候选对象的学习状态。
            self._adopt_fitted_state(candidate)
            return self

        X = self._check_input(X)
        self._validate_public_policies()
        self._target_in_fit_ = self.target is not None and self.target in X.columns

        # 处理两种API风格：如果y为None且提供了target参数，从X中提取目标列
        X, y = self._extract_target(X, y)
        if getattr(self, "training_mode", "in_sample") not in ("in_sample", "oof"):
            raise ValueError("training_mode 必须为 in_sample 或 oof")
        self._set_input_feature_attributes(X)

        if self.cols is None:
            self.cols_ = self._get_category_cols(X)
        else:
            self.cols_ = [c for c in self.cols if c in X.columns]

        if self.drop_invariant:
            self._dropped_cols = self._find_invariant_cols(X)
            self.cols_ = [c for c in self.cols_ if c not in self._dropped_cols]

        # 统一缺失值策略校验：handle_missing='error' 时，fit 阶段任一编码列含缺失即报错
        # （与 transform 阶段保持一致）
        if self.handle_missing == "error":
            for col in self.cols_:
                if col in X.columns and X[col].isna().any():
                    raise ValueError(f"列'{col}'包含缺失值，但 handle_missing='error'")

        self._fit(X.drop(columns=self._dropped_cols, errors="ignore"), y)

        self._is_fitted = True
        return self

    def _fit_columns(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        state_attrs: Tuple[str, ...] = ("mapping_",),
        shared_state: Optional[Dict[str, Any]] = None,
        has_parallel_children: bool = False,
    ) -> None:
        """隔离拟合各列，并在全部成功后按学习列顺序一次提交状态。"""
        shared_state = shared_state or {}
        columns = list(self.cols_ or [])

        def iter_tasks():
            for ordinal, column in enumerate(columns):
                candidate = clone(self)
                candidate.cols_ = [column]
                for attr, value in shared_state.items():
                    setattr(candidate, attr, value)
                yield ordinal, column, candidate, X[column].copy(), y

        results = self._parallel_execute(
            _fit_encoder_column_worker,
            iter_tasks(),
            task_labels=columns,
            default_backend="loky",
            has_parallel_children=has_parallel_children,
            workload=ParallelWorkload(
                task_count=len(columns),
                rows=len(X),
                columns=len(columns),
                data_bytes=int(X.loc[:, self.cols_ or []].memory_usage(deep=True).sum()),
                cost_per_item=(10.0 if self.__class__.__name__ in {"WOEEncoder", "TargetEncoder", "CatBoostEncoder", "QuantileEncoder", "GBMEncoder"} else 3.0),
                capability="process_safe",
                has_parallel_children=has_parallel_children,
                auto_max_workers=8,
                operation=f"{self.__class__.__name__}列拟合",
            ),
        )

        staged = {attr: {} for attr in state_attrs}
        for _, column, payload in results:
            for attr in state_attrs:
                if attr not in payload:
                    raise ValueError(f"列'{column}'拟合结果缺少状态'{attr}'")
                value = payload[attr]
                if attr == "mapping_" and isinstance(value, dict):
                    value = self._canonicalize_nan_keys(value)
                staged[attr][column] = value

        for attr, value in staged.items():
            setattr(self, attr, value)

    def _transform_columns(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        contexts: Optional[Dict[str, Any]] = None,
        passthrough: bool = False,
    ) -> pd.DataFrame:
        """只读并行转换各列，并按学习列顺序恢复索引、dtype 与列布局。"""
        contexts = contexts or {}
        columns = list(self.cols_ or [])
        tasks = ((ordinal, column, self, X[column].copy(), y, contexts.get(column)) for ordinal, column in enumerate(columns))
        results = self._parallel_execute(
            _transform_encoder_column_worker,
            tasks,
            task_labels=columns,
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=len(columns),
                rows=len(X),
                columns=len(columns),
                data_bytes=int(X.loc[:, self.cols_ or []].memory_usage(deep=True).sum()),
                cost_per_item=1.0,
                capability="thread_safe",
                releases_gil=True,
                auto_max_workers=8,
                operation=f"{self.__class__.__name__}列转换",
            ),
        )

        if not results:
            return X.copy()

        ordered = [result for _, _, result in sorted(results, key=lambda item: item[0])]
        if all(isinstance(result, pd.Series) for result in ordered):
            output = X.copy()
            for (_, column, _), result in zip(sorted(results, key=lambda item: item[0]), ordered):
                result = result.copy()
                result.index = X.index
                output[column] = result
            return output

        if not all(isinstance(result, pd.DataFrame) for result in ordered):
            raise TypeError("编码列转换结果必须全部为Series或全部为DataFrame")

        blocks = []
        if passthrough:
            other_cols = [column for column in X.columns if column not in (self.cols_ or [])]
            if other_cols:
                blocks.append(X[other_cols].copy())
        for block in ordered:
            block = block.copy()
            block.index = X.index
            blocks.append(block)
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=X.index)

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
        original = X
        X = self._prepare_transform_input(X)

        # 统一缺失值策略校验：handle_missing='error' 时，任一编码列含缺失即报错
        if self.handle_missing == "error":
            for col in self.cols_:
                if col in X.columns and X[col].isna().any():
                    raise ValueError(f"列'{col}'包含缺失值，但 handle_missing='error'")

        X_transformed = X.copy()
        X_transformed = self._transform(X_transformed, y)

        return self._finalize_output(X_transformed, original)

    def _prepare_transform_input(self, X):
        """默认纯特征输出；目标和已删除列永远不参与后续转换。"""
        self._validate_public_policies()
        removed = list(getattr(self, "_dropped_cols", []))
        if self.target is not None:
            removed.append(self.target)
        X = X.drop(columns=removed, errors="ignore")
        fitted = getattr(self, "feature_names_in_", None)
        if fitted is not None:
            columns = [column for column in fitted if column not in removed]
            # 显式cols调用保留历史的“仅转换已选择列”用法；待编码列必须完整。
            missing = [column for column in self.cols_ or [] if column not in X.columns]
            if missing:
                raise FeatureNotFoundError(f"转换输入缺少拟合特征: {missing}")
            X = X.loc[:, [column for column in columns if column in X.columns]]
        return X

    def _finalize_output(self, X_transformed, original):
        target_col = getattr(self, "target", None)
        if getattr(self, "passthrough_target", False) and target_col is not None and target_col in original.columns:
            if isinstance(X_transformed, pd.DataFrame):
                X_transformed[target_col] = np.asarray(original[target_col])
            else:
                from scipy import sparse
                if sparse.issparse(X_transformed):
                    try:
                        labels = np.asarray(original[target_col], dtype=float).reshape(-1, 1)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("稀疏输出的兼容目标透传要求目标为数值") from exc
                    X_transformed = sparse.hstack([X_transformed, sparse.csr_matrix(labels)], format="csr")
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed.attrs.update(self._OUTPUT_ATTRS)
        if not self.return_df:
            # 处理稀疏矩阵的情况
            if hasattr(X_transformed, "toarray"):
                # 已经是稀疏矩阵，直接返回
                return X_transformed
            return X_transformed.values
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """返回固定学习schema；兼容透传模式仅在拟合表含目标时附加其列名。"""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("编码器尚未拟合，请先调用fit()")
        columns = list(getattr(self, "feature_names_in_", self.cols_ or []))
        columns = [column for column in columns if column not in self._dropped_cols and column != self.target]
        if getattr(self, "passthrough_target", False) and getattr(self, "_target_in_fit_", False):
            columns.append(self.target)
        return np.asarray(columns, dtype=object)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, *, groups=None) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        支持两种API风格:
        1. sklearn风格: fit_transform(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit_transform(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        :param X: 训练数据，shape (n_samples, n_features) 或包含目标列的完整数据框
        :param y: 目标变量，对于某些编码器是必需的。
                  如果为None且初始化时提供了target参数，则从X中提取target列
        :return: 编码后的数据
        """
        original = self._check_input(X)
        features, target = self._extract_target(original, y)
        if getattr(self, "training_mode", "in_sample") == "oof":
            return self._fit_transform_oof(original, features, target, groups=groups)
        return self.fit(original, y).transform(original, target)

    def _fit_transform_oof(self, original, features, target, groups=None):
        """折外训练编码；最终映射在全训练集拟合，未被验证折覆盖的行保留 NaN。"""
        from sklearn.model_selection import KFold

        if target is None:
            raise ValueError("折外编码必须提供目标变量")
        cv = getattr(self, "cv", 5)
        if isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)):
            cv = KFold(n_splits=int(cv), shuffle=True, random_state=getattr(self, "random_state", 42))
        if not hasattr(cv, "split"):
            raise ValueError("cv 必须是至少两折的整数或具有 split 方法的切分器")
        # 全训练集只决定最终推理状态与无监督列schema；每行训练编码仍只来自其训练折。
        fitted = clone(self).fit(original, target)
        features = fitted._prepare_transform_input(features)
        columns = list(fitted.cols_)
        output = features.copy()
        if not columns:
            self._adopt_fitted_state(fitted)
            self.oof_coverage_ = np.ones(len(features), dtype=bool)
            return self._finalize_output(output, original)
        for column in columns:
            output[column] = np.nan
        covered = np.zeros(len(features), dtype=bool)
        for train, valid in cv.split(features, target, groups):
            train, valid = np.asarray(train), np.asarray(valid)
            for indices in (train, valid):
                if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
                    raise ValueError("折外编码的训练/验证位置必须为非空一维整数数组")
                if (indices < 0).any() or (indices >= len(features)).any() or len(np.unique(indices)) != len(indices):
                    raise ValueError("折外编码切分位置越界或重复")
            if np.intersect1d(train, valid).size or covered[valid].any():
                raise ValueError("折外编码训练/验证位置重叠或验证位置重复")
            fold = clone(self).set_params(training_mode="in_sample", return_df=True, drop_invariant=False, cols=columns, passthrough_target=False)
            fold.fit(features.iloc[train], target.iloc[train])
            transformed = fold.transform(features.iloc[valid])
            output.iloc[valid, output.columns.get_indexer(columns)] = transformed[columns].to_numpy()
            covered[valid] = True
        if not covered.any():
            raise ValueError("折外编码切分器未提供任何验证样本")
        self._adopt_fitted_state(fitted)
        self.oof_coverage_ = covered
        if not covered.all():
            import warnings
            warnings.warn("部分行未被折外验证集覆盖（例如时间切分的起始窗口），编码结果保留 NaN", UserWarning)
        return self._finalize_output(output, original)

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
        prepared = prepare_xy(X, y, target=self.target, target_type=self._TARGET_TYPE)
        return prepared.X, prepared.y

    def _apply_missing_unknown(self, column, values, result, default):
        """以原始缺失掩码区分缺失与未知；两项策略互不覆盖。"""
        missing = values.isna()
        unknown = result.isna() & ~missing
        if self.handle_unknown == "error" and unknown.any():
            raise ValueError(f"列'{column}'包含未知类别")
        if self.handle_unknown == "value":
            result = result.mask(unknown, default)
        if self.handle_missing == "error" and missing.any():
            raise ValueError(f"列'{column}'包含缺失值")
        if self.handle_missing == "return_nan":
            result = result.mask(missing, np.nan)
        elif self.handle_missing == "value":
            result = result.mask(missing & result.isna(), default)
        return result

    @staticmethod
    def _map_values(values, mapping):
        """映射真实类别并单独应用类型化缺失桶，不占用业务类别键。"""
        mapped = values.map(mapping)
        if isinstance(mapped.dtype, pd.CategoricalDtype):
            mapped = pd.Series(np.asarray(mapped), index=values.index, name=values.name)
        if MISSING in mapping:
            mapped = mapped.mask(values.isna(), mapping[MISSING])
        return mapped

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
            if X[col].nunique(dropna=False) <= 1:
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

    def export_mapping(self, *, legacy: bool = False) -> Dict[str, Any]:
        """导出严格JSON兼容的v2类型记录映射。

        :param legacy: True返回旧Python字典结构（可能含NaN/非字符串键，不承诺JSON
            保真）；遇业务类别与旧保留键冲突时拒绝。默认v2保留类别类型、schema及
            推理参数。自定义cv对象只记录省略说明，完整重训状态应使用save_artifact。

        :return: 可序列化的编码映射字典

        **参考样例**

        >>> encoder.fit(X, y)
        >>> mapping = encoder.export_mapping()
        >>> import json
        >>> with open('encoder_mapping.json', 'w') as f:
        ...     json.dump(mapping, f)
        """
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("编码器尚未拟合，不能导出推理映射")
        if not legacy:
            parameters = {}
            omitted = []
            for name, value in self.get_params(deep=False).items():
                if name in {"n_jobs", "parallel_backend", "parallel_config"}:
                    continue
                try:
                    parameters[name] = pack(value)
                except ValueError:
                    if name == "cv":
                        omitted.append(name)
                    else:
                        raise
            return {
                "format": "hscredit-encoder-mapping", "version": 2,
                "encoder_type": type(self).__name__, "parameters": parameters,
                "mapping_": pack(self.mapping_), "cols_": pack(self.cols_),
                "extra_state": pack(self._export_extra_state()),
                "schema": pack({"feature_names_in_": getattr(self, "feature_names_in_", None),
                                "_dropped_cols": self._dropped_cols,
                                "_target_in_fit_": getattr(self, "_target_in_fit_", False)}),
                "omitted_refit_parameters": omitted,
            }
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
        if mapping.get("format") == "hscredit-encoder-mapping":
            if type(mapping.get("version")) is not int or mapping["version"] != 2:
                raise ValueError("不支持的编码映射协议版本")
            if mapping.get("encoder_type") != type(self).__name__:
                raise ValueError("编码映射类型与当前编码器不一致")
            # 完整解析到候选对象后提交；损坏制品不得破坏现有映射。
            candidate = clone(self)
            parameters = {name: unpack(value) for name, value in mapping["parameters"].items()}
            candidate.set_params(**parameters)
            candidate._validate_public_policies()
            candidate.mapping_ = unpack(mapping["mapping_"])
            candidate.cols_ = unpack(mapping["cols_"])
            if not isinstance(candidate.mapping_, dict) or not isinstance(candidate.cols_, list):
                raise ValueError("编码映射状态结构无效")
            if any(column not in candidate.mapping_ or not isinstance(candidate.mapping_[column], dict) for column in candidate.cols_):
                raise ValueError("编码映射缺少已学习字段状态")
            candidate._import_extra_state(unpack(mapping["extra_state"]))
            schema = unpack(mapping["schema"])
            candidate._dropped_cols = schema["_dropped_cols"]
            candidate._target_in_fit_ = schema["_target_in_fit_"]
            if schema["feature_names_in_"] is not None:
                candidate.feature_names_in_ = schema["feature_names_in_"]
                candidate.n_features_in_ = len(candidate.feature_names_in_)
            candidate._is_fitted = True
            if mapping.get("omitted_refit_parameters"):
                warnings.warn("此推理映射未保存自定义cv切分器；重拟合前请显式配置cv，完整训练对象请使用save_artifact", UserWarning)
            self.__dict__.clear()
            self.__dict__.update(candidate.__dict__)
            return self
        if "version" in mapping and mapping.get("format") is not None:
            raise ValueError("不支持的编码映射格式或版本")
        self.cols = mapping.get("cols")
        self.cols_ = mapping.get("cols_")
        self.mapping_ = self._deserialize_mapping(mapping.get("mapping_", {}))
        # 旧格式无法区分同名业务类别和保留键；仅在旧协议入口按旧含义迁移。
        for column_mapping in self.mapping_.values():
            if isinstance(column_mapping, dict):
                if "__UNKNOWN__" in column_mapping:
                    column_mapping[UNKNOWN] = column_mapping.pop("__UNKNOWN__")
                if type(self).__name__ == "CountEncoder" and "__OTHER__" in column_mapping:
                    column_mapping[OTHER] = column_mapping.pop("__OTHER__")
                    self._legacy_other_routing_ = True
                if type(self).__name__ != "CountEncoder":
                    for key in list(column_mapping):
                        if self._is_float_nan_key(key):
                            column_mapping[MISSING] = column_mapping.pop(key)
        self.drop_invariant = mapping.get("drop_invariant", False)
        self.handle_unknown = mapping.get("handle_unknown", "value")
        self.handle_missing = mapping.get("handle_missing", "value")
        self._import_extra_state(mapping.get("extra_state", {}))
        if type(self).__name__ == "WOEEncoder" and "_legacy_string_keys_" not in mapping.get("extra_state", {}):
            self._legacy_string_keys_ = True
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
            if isinstance(key, CategoryToken):
                key = {MISSING: np.nan, UNKNOWN: "__UNKNOWN__", OTHER: "__OTHER__"}.get(key, key)
            if key in serialized:
                raise ValueError("传统映射存在保留键冲突，请使用默认类型化 export_mapping()")
            if self._is_float_nan_key(key):
                key = np.nan
            if isinstance(value, pd.Series):
                serialized[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_mapping(value)
            else:
                serialized[key] = value
        return serialized

    @classmethod
    def _canonicalize_nan_keys(cls, mapping: Dict) -> Dict:
        """将进程往返产生的不同 NaN 键归一为同一稳定键。"""
        canonical = {}
        for key, value in mapping.items():
            normalized = np.nan if cls._is_float_nan_key(key) else key
            canonical[normalized] = value
        return canonical

    @classmethod
    def _float_nan_bucket(cls, key: Any) -> Optional[_FloatNaNBucket]:
        """按 Python float 或 NumPy 浮点 dtype 区分 NaN 桶。"""
        if type(key) is float and np.isnan(key):
            return _PYTHON_FLOAT_NAN_BUCKET
        if isinstance(key, np.floating) and np.isnan(key):
            return _FloatNaNBucket("numpy", np.asarray(key).dtype.str)
        return None

    @classmethod
    def _float_nan_representative(cls, key: Any) -> Any:
        """返回 typed NaN 桶在当前进程中的稳定公开代表键。"""
        bucket = cls._float_nan_bucket(key)
        if bucket is None:
            return key
        representative = _FLOAT_NAN_REPRESENTATIVES.get(bucket)
        if representative is None:
            representative = np.asarray(np.nan, dtype=np.dtype(bucket.dtype))[()]
            _FLOAT_NAN_REPRESENTATIVES[bucket] = representative
        return representative

    @classmethod
    def _is_float_nan_key(cls, key: Any) -> bool:
        """仅识别 Python/NumPy 浮点 NaN，不合并其他 missing-like 标量。"""
        return cls._float_nan_bucket(key) is not None

    @classmethod
    def _map_with_typed_float_nan(cls, values: pd.Series, mapping: Dict) -> pd.Series:
        """按 typed NaN 桶修正 pandas map，避免依赖 NaN 对象身份。"""
        result = values.map(mapping)
        typed_lookup = {bucket: value for key, value in mapping.items() if (bucket := cls._float_nan_bucket(key)) is not None}
        if not typed_lookup:
            return result

        for position, original in enumerate(values.array):
            bucket = cls._float_nan_bucket(original)
            if bucket in typed_lookup:
                result.iloc[position] = typed_lookup[bucket]
        return result

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
