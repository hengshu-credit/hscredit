"""GBM Encoder (梯度提升树编码器).

基于 XGBoost、LightGBM 或 CatBoost 树模型，
将原始特征转换为叶子节点索引或 embedding，
用于后续 LR 等模型的训练。

典型应用场景:
1. GBM + LR: 使用 GBM 提取特征，输入逻辑回归
2. 特征增强: 将树模型的叶子节点作为新的特征
3. Embedding 提取: 获取树模型的中间表示

**依赖**
- XGBoost: pip install xgboost
- LightGBM: pip install lightgbm
- CatBoost: pip install catboost
"""

from typing import Optional, List, Dict, Union, Any, Literal
import numpy as np
import pandas as pd
import warnings

from .base import BaseEncoder

# 从 hscredit.core.models 统一导入模型
from ..models import (
    XGBoostRiskModel,
    LightGBMRiskModel,
    CatBoostRiskModel,
)


class GBMEncoder(BaseEncoder):
    """梯度提升树编码器.

    使用 XGBoost、LightGBM 或 CatBoost 训练树模型，
    将样本在树中的位置（叶子节点）转换为特征。

    支持多种输出格式:
    - 'leaves': 叶子节点索引
    - 'onehot': 叶子节点独热编码
    - 'probability': 预测概率
    - 'embedding': 树路径 embedding

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则使用所有特征列
    :param model_type: GBM模型类型，可选 'xgboost'、'lightgbm'、'catboost'，默认为'xgboost'
    :param n_estimators: 树的数量，默认为100
    :param max_depth: 树的最大深度，默认为5
    :param learning_rate: 学习率，默认为0.1
    :param subsample: 样本采样比例，默认为0.8
    :param colsample_bytree: 特征采样比例，默认为0.8
    :param min_child_samples: 叶子节点最小样本数，默认为20
    :param random_state: 随机种子，用于可复现性，默认为None
    :param output_type: 输出特征类型，默认为'leaves'
        - 'leaves': 返回每棵树上的叶子节点索引
        - 'onehot': 对叶子节点进行独热编码
        - 'probability': 返回预测概率（仅分类任务）
        - 'embedding': 返回树路径的embedding表示
    :param drop_origin: 是否删除原始特征列，默认为True（推荐，避免原始类别特征影响下游模型）
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True
    :param model_params: 额外的模型参数，用于覆盖默认参数，默认为None
    :param task: 任务类型，'classification' 或 'regression'，默认为'classification'
    
    **说明**
    
    默认设置 drop_origin=True，即只保留GBM生成的特征（叶子节点、概率等），
    删除原始特征。这样可以避免原始类别特征对下游模型（如LR）造成影响。

    **属性**

    - model_: 训练好的GBM模型
    - n_trees_: 树的数量
    - n_features_: 原始特征数量
    - leaf_indices_: 每棵树的叶子节点索引映射
    - feature_names_: 编码后的特征名列表
    - classes_: 类别标签（分类任务）

    **缺失值支持**

    XGBoost、LightGBM 和 CatBoost 都原生支持缺失值处理：
    - XGBoost: 自动学习缺失值的最优分裂方向
    - LightGBM: 自动处理缺失值，无需填充
    - CatBoost: 将缺失值作为特殊类别处理

    对于类别特征中的缺失值，在编码为数值时会保留np.nan格式
    对于数值特征中的缺失值，直接传递给GBM模型处理

    **参考样例**

    基本使用（XGBoost + 叶子节点特征）::

        >>> from hscredit.core.encoders import GBMEncoder
        >>> encoder = GBMEncoder(
        ...     model_type='xgboost',
        ...     n_estimators=50,
        ...     max_depth=4,
        ...     output_type='leaves'
        ... )
        >>> X_encoded = encoder.fit_transform(X, y)

    LightGBM + 独热编码::

        >>> encoder = GBMEncoder(
        ...     model_type='lightgbm',
        ...     output_type='onehot',
        ...     n_estimators=30,
        ...     max_depth=3
        ... )
        >>> X_encoded = encoder.fit_transform(X, y)

    CatBoost + 概率输出::

        >>> encoder = GBMEncoder(
        ...     model_type='catboost',
        ...     output_type='probability',
        ...     n_estimators=100
        ... )
        >>> X_encoded = encoder.fit_transform(X, y)

    GBM + LR 组合训练::

        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.pipeline import Pipeline
        >>> 
        >>> # 创建GBM编码器
        >>> gbm_encoder = GBMEncoder(
        ...     model_type='xgboost',
        ...     output_type='leaves',
        ...     n_estimators=50,
        ...     max_depth=3
        ... )
        >>> 
        >>> # 与LR组合
        >>> pipeline = Pipeline([
        ...     ('gbm', gbm_encoder),
        ...     ('lr', LogisticRegression(max_iter=1000))
        ... ])
        >>> pipeline.fit(X_train, y_train)
        >>> y_pred = pipeline.predict(X_test)

    参考:
        - Facebook GBDT + LR: https://dl.acm.org/doi/10.1145/2648584.2648589
        - XGBoost: https://xgboost.readthedocs.io/
        - LightGBM: https://lightgbm.readthedocs.io/
        - CatBoost: https://catboost.ai/
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        model_type: Literal['xgboost', 'lightgbm', 'catboost'] = 'xgboost',
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        random_state: Optional[int] = None,
        output_type: Literal['leaves', 'onehot', 'probability', 'embedding'] = 'leaves',
        drop_origin: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
        task: Literal['classification', 'regression'] = 'classification',
        target: Optional[str] = None,
    ):
        """初始化GBM编码器。

        :param cols: 需要编码的列名列表
        :param model_type: GBM模型类型，默认为'xgboost'
        :param n_estimators: 树的数量，默认为100
        :param max_depth: 树的最大深度，默认为5
        :param learning_rate: 学习率，默认为0.1
        :param subsample: 样本采样比例，默认为0.8
        :param colsample_bytree: 特征采样比例，默认为0.8
        :param min_child_samples: 叶子节点最小样本数，默认为20
        :param random_state: 随机种子，默认为None
        :param output_type: 输出特征类型，默认为'leaves'
        :param drop_origin: 是否删除原始特征列，默认为True
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param model_params: 额外的模型参数，默认为None
        :param task: 任务类型，默认为'classification'
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.output_type = output_type
        self.drop_origin = drop_origin
        self.model_params = model_params or {}
        self.task = task

        # 拟合后的属性
        self.model_: Optional[Any] = None
        self.n_trees_: int = 0
        self.n_features_: int = 0
        self.leaf_indices_: Dict[int, Dict[int, int]] = {}
        self.feature_names_: List[str] = []
        self.classes_: Optional[np.ndarray] = None
        self.missing_stats_: Dict[str, Dict[str, Any]] = {}

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """获取所有列名（GBMEncoder使用所有特征，不限于类别特征）。

        :param X: 输入数据
        :return: 所有列名列表
        """
        return X.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合GBM编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1) 或多分类
        :raises ValueError: 当y为空时抛出
        :raises ImportError: 当所需的GBM库未安装时抛出
        """
        if y is None:
            raise ValueError("GBMEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y)
        self.classes_ = np.unique(y)

        # 确定使用的列
        if self.cols is None:
            self.cols_ = X.columns.tolist()
        else:
            self.cols_ = [c for c in self.cols if c in X.columns]

        self.n_features_ = len(self.cols_)

        # 准备训练数据
        X_train = X[self.cols_].copy()

        # 统计缺失值信息
        self._compute_missing_stats(X_train)

        # 处理类别特征（CatBoost自动处理，其他需要编码）
        if self.model_type in ['xgboost', 'lightgbm']:
            X_train = self._preprocess_categorical(X_train, fit=True)

        # 根据模型类型训练
        if self.model_type == 'xgboost':
            self._fit_xgboost(X_train, y)
        elif self.model_type == 'lightgbm':
            self._fit_lightgbm(X_train, y)
        elif self.model_type == 'catboost':
            self._fit_catboost(X_train, y)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 生成特征名
        self._generate_feature_names()

    def _compute_missing_stats(self, X: pd.DataFrame):
        """计算缺失值统计信息。

        :param X: 输入数据
        """
        self.missing_stats_ = {}
        total_samples = len(X)

        for col in X.columns:
            missing_count = X[col].isna().sum()
            missing_ratio = missing_count / total_samples if total_samples > 0 else 0

            if missing_count > 0:
                self.missing_stats_[col] = {
                    'missing_count': int(missing_count),
                    'missing_ratio': float(missing_ratio),
                    'total_samples': total_samples
                }

        if self.missing_stats_:
            total_features_with_missing = len(self.missing_stats_)
            print(f"警告: {total_features_with_missing} 个特征存在缺失值，GBM将自动处理")

    def _preprocess_categorical(
        self, X: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """预处理类别特征。

        对于非CatBoost模型，需要将类别特征转换为数值。
        保留数值特征的缺失值(np.nan)，让GBM模型自动处理。

        :param X: 输入数据
        :param fit: 是否处于拟合阶段
        :return: 处理后的数据
        """
        X = X.copy()

        # 识别类别特征
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) == 0:
            return X

        # 使用序数编码处理类别特征
        for col in cat_cols:
            if fit:
                # 创建映射（只包含非缺失值）
                categories = X[col].dropna().unique()
                mapping = {cat: i + 1 for i, cat in enumerate(categories)}  # 从1开始编码，0留给缺失值
                # 缺失值保持为np.nan，会在转换为数值类型时变成0或保持nan
                self.mapping_[col] = mapping

            # 应用映射
            if col in self.mapping_:
                # 映射类别值，缺失值保持为np.nan
                X[col] = X[col].map(self.mapping_[col])
                # 转换为数值类型，缺失值会变成nan
                X[col] = pd.to_numeric(X[col], errors='coerce')

        return X

    def _preprocess_catboost_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """预处理CatBoost的缺失值。

        CatBoost要求类别特征中的缺失值必须是字符串。

        :param X: 输入数据
        :return: 处理后的数据
        """
        X = X.copy()

        # 识别类别特征
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            # 将缺失值转换为特殊字符串
            X[col] = X[col].fillna('__MISSING__')
            # 确保列为字符串类型
            X[col] = X[col].astype(str)

        return X

    def _fit_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """拟合XGBoost模型。

        :param X: 训练特征
        :param y: 目标变量
        :raises ImportError: 当xgboost未安装时抛出
        """
        # 基础参数
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_samples,
            'random_state': self.random_state,
            'n_jobs': -1,
        }

        # 添加任务相关参数
        if self.task == 'classification':
            if len(self.classes_) == 2:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
            else:
                params['objective'] = 'multi:softprob'
                params['num_class'] = len(self.classes_)
                params['eval_metric'] = 'mlogloss'
        else:
            params['objective'] = 'reg:squarederror'

        # 合并用户自定义参数
        params.update(self.model_params)

        # 使用 hscredit 的 XGBoostRiskModel
        self.model_ = XGBoostRiskModel(**params)
        self.model_.fit(X, y)
        self.n_trees_ = self.n_estimators

    def _fit_lightgbm(self, X: pd.DataFrame, y: pd.Series):
        """拟合LightGBM模型。

        :param X: 训练特征
        :param y: 目标变量
        :raises ImportError: 当lightgbm未安装时抛出
        """
        # 基础参数
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_samples': self.min_child_samples,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': False,
        }

        # 添加任务相关参数
        if self.task == 'classification':
            if len(self.classes_) == 2:
                params['objective'] = 'binary'
            else:
                params['objective'] = 'multiclass'
                params['num_class'] = len(self.classes_)
        else:
            params['objective'] = 'regression'

        # 合并用户自定义参数
        params.update(self.model_params)

        # 使用 hscredit 的 LightGBMRiskModel
        self.model_ = LightGBMRiskModel(**params)
        self.model_.fit(X, y)
        self.n_trees_ = self.n_estimators

    def _fit_catboost(self, X: pd.DataFrame, y: pd.Series):
        """拟合CatBoost模型。

        :param X: 训练特征
        :param y: 目标变量
        :raises ImportError: 当catboost未安装时抛出
        """
        # 复制数据，避免修改原始数据
        X_cb = X.copy()

        # 识别类别特征
        cat_features = X_cb.select_dtypes(include=['object', 'category']).columns.tolist()

        # CatBoost要求类别特征中的缺失值必须是字符串
        for col in cat_features:
            # 将缺失值转换为字符串 "missing"
            X_cb[col] = X_cb[col].fillna('__MISSING__')
            # 确保列为字符串类型
            X_cb[col] = X_cb[col].astype(str)

        # 基础参数 (CatBoostRiskModel使用不同的参数名)
        params = {
            'iterations': self.n_estimators,
            'depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'min_data_in_leaf': self.min_child_samples,
            'random_state': self.random_state,
            'verbose': False,
        }

        # 添加任务相关参数
        if self.task == 'classification':
            if len(self.classes_) == 2:
                params['objective'] = 'Logloss'
            else:
                params['objective'] = 'MultiClass'
        else:
            params['objective'] = 'RMSE'

        # 合并用户自定义参数
        params.update(self.model_params)

        # 使用 hscredit 的 CatBoostRiskModel
        self.model_ = CatBoostRiskModel(**params)

        self.model_.fit(X_cb, y, cat_features=cat_features if cat_features else None)
        self.n_trees_ = self.n_estimators

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据为GBM特征。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），GBM编码器不需要
        :return: 编码后的数据
        """
        X_transformed = X.copy()

        # 提取用于编码的特征
        X_encode = X_transformed[self.cols_].copy()

        # 预处理类别特征
        if self.model_type in ['xgboost', 'lightgbm']:
            X_encode = self._preprocess_categorical(X_encode, fit=False)
        elif self.model_type == 'catboost':
            # CatBoost需要将类别特征的缺失值转换为字符串
            X_encode = self._preprocess_catboost_missing(X_encode)

        # 根据输出类型转换
        if self.output_type == 'leaves':
            features = self._transform_to_leaves(X_encode)
        elif self.output_type == 'onehot':
            features = self._transform_to_onehot(X_encode)
        elif self.output_type == 'probability':
            features = self._transform_to_probability(X_encode)
        elif self.output_type == 'embedding':
            features = self._transform_to_embedding(X_encode)
        else:
            raise ValueError(f"不支持的输出类型: {self.output_type}")

        # 删除原始特征（如果指定）
        if self.drop_origin:
            X_transformed = X_transformed.drop(columns=self.cols_)

        # 合并新特征
        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                X_transformed[col] = features[col].values
        else:
            # 如果是numpy数组，创建DataFrame
            feature_cols = self.feature_names_
            for i, col in enumerate(feature_cols):
                if i < features.shape[1]:
                    X_transformed[col] = features[:, i]

        return X_transformed

    def _transform_to_leaves(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换为叶子节点索引特征。

        :param X: 输入数据
        :return: 叶子节点索引DataFrame
        """
        if self.model_type == 'xgboost':
            return self._get_xgboost_leaves(X)
        elif self.model_type == 'lightgbm':
            return self._get_lightgbm_leaves(X)
        elif self.model_type == 'catboost':
            return self._get_catboost_leaves(X)

    def _get_xgboost_leaves(self, X: pd.DataFrame) -> pd.DataFrame:
        """获取XGBoost叶子节点索引。

        :param X: 输入数据
        :return: 叶子节点索引DataFrame
        """
        # 使用 hscredit 模型类的 get_leaf_indices 方法
        leaf_indices = self.model_.get_leaf_indices(X)

        # 转换为DataFrame
        columns = [f'gbm_tree_{i}' for i in range(leaf_indices.shape[1])]
        return pd.DataFrame(leaf_indices, index=X.index, columns=columns)

    def _get_lightgbm_leaves(self, X: pd.DataFrame) -> pd.DataFrame:
        """获取LightGBM叶子节点索引。

        :param X: 输入数据
        :return: 叶子节点索引DataFrame
        """
        # 使用 hscredit 模型类的 get_leaf_indices 方法
        leaf_indices = self.model_.get_leaf_indices(X)

        # 转换为DataFrame
        columns = [f'gbm_tree_{i}' for i in range(leaf_indices.shape[1])]
        return pd.DataFrame(leaf_indices, index=X.index, columns=columns)

    def _get_catboost_leaves(self, X: pd.DataFrame) -> pd.DataFrame:
        """获取CatBoost叶子节点索引。

        :param X: 输入数据
        :return: 叶子节点索引DataFrame
        """
        # 使用 hscredit 模型类的 get_leaf_indices 方法
        leaf_indices = self.model_.get_leaf_indices(X)

        # 转换为DataFrame
        columns = [f'gbm_tree_{i}' for i in range(leaf_indices.shape[1])]
        return pd.DataFrame(leaf_indices, index=X.index, columns=columns)

    def _transform_to_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换为叶子节点独热编码。

        :param X: 输入数据
        :return: 独热编码DataFrame
        """
        # 首先获取叶子索引
        leaf_df = self._transform_to_leaves(X)

        # 对每个树的叶子节点进行独热编码
        onehot_dfs = []

        for tree_idx in range(leaf_df.shape[1]):
            tree_col = leaf_df.iloc[:, tree_idx]
            unique_leaves = np.unique(tree_col)

            for leaf in unique_leaves:
                col_name = f'gbm_tree{tree_idx}_leaf{leaf}'
                onehot_dfs.append(pd.DataFrame(
                    {col_name: (tree_col == leaf).astype(int)},
                    index=X.index
                ))

        if onehot_dfs:
            return pd.concat(onehot_dfs, axis=1)
        else:
            return pd.DataFrame(index=X.index)

    def _transform_to_probability(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换为预测概率。

        :param X: 输入数据
        :return: 概率DataFrame
        """
        if self.task == 'classification':
            proba = self.model_.predict_proba(X)

            if proba.shape[1] == 2:
                # 二分类，只返回正类概率
                return pd.DataFrame(
                    {'gbm_proba': proba[:, 1]},
                    index=X.index
                )
            else:
                # 多分类
                columns = [f'gbm_proba_class_{i}' for i in range(proba.shape[1])]
                return pd.DataFrame(proba, index=X.index, columns=columns)
        else:
            # 回归任务
            pred = self.model_.predict(X)
            return pd.DataFrame({'gbm_prediction': pred}, index=X.index)

    def _transform_to_embedding(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换为树路径embedding。

        使用每棵树的输出作为embedding的一个维度。

        :param X: 输入数据
        :return: embedding DataFrame
        """
        # 统一使用叶子索引作为embedding（跨模型通用方法）
        return self._transform_to_leaves(X)

    def _generate_feature_names(self):
        """生成编码后的特征名列表。"""
        if self.output_type == 'leaves':
            self.feature_names_ = [f'gbm_tree_{i}' for i in range(self.n_trees_)]
        elif self.output_type == 'probability':
            if self.task == 'classification':
                if len(self.classes_) == 2:
                    self.feature_names_ = ['gbm_proba']
                else:
                    self.feature_names_ = [f'gbm_proba_class_{i}' for i in range(len(self.classes_))]
            else:
                self.feature_names_ = ['gbm_prediction']
        elif self.output_type == 'embedding':
            self.feature_names_ = [f'gbm_emb_{i}' for i in range(self.n_trees_)]
        # onehot的特征名在转换时动态生成

    def get_model(self) -> Any:
        """获取训练好的GBM模型。

        :return: 训练好的GBM模型对象
        """
        return self.model_

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性。

        :return: 特征重要性DataFrame，包含'feature'和'importance'两列
        """
        if self.model_ is None:
            raise ValueError("模型尚未拟合")

        # 使用 hscredit 模型类的 get_feature_importances 方法
        importance_series = self.model_.get_feature_importances()

        return pd.DataFrame({
            'feature': importance_series.index,
            'importance': importance_series.values
        }).sort_values('importance', ascending=False)

    def get_missing_stats(self) -> pd.DataFrame:
        """获取缺失值统计信息。

        :return: 缺失值统计DataFrame，包含'feature'、'missing_count'、'missing_ratio'三列
        """
        if not self.missing_stats_:
            return pd.DataFrame(columns=['feature', 'missing_count', 'missing_ratio'])

        stats = []
        for col, info in self.missing_stats_.items():
            stats.append({
                'feature': col,
                'missing_count': info['missing_count'],
                'missing_ratio': info['missing_ratio']
            })

        return pd.DataFrame(stats).sort_values('missing_ratio', ascending=False)

    def plot_tree(self, tree_idx: int = 0, **kwargs):
        """绘制树结构。

        :param tree_idx: 树的索引，默认为0（第一棵树）
        :param kwargs: 传递给plot_tree的其他参数
        :raises NotImplementedError: 当模型类型不支持可视化时抛出
        """
        if self.model_ is None:
            raise ValueError("模型尚未拟合")

        try:
            import matplotlib.pyplot as plt
            # 使用 hscredit 模型类的 plot_tree 方法
            self.model_.plot_tree(tree_idx, **kwargs)
            plt.show()
        except ImportError:
            raise ImportError("绘制树需要安装matplotlib: pip install matplotlib")

    def __repr__(self) -> str:
        return (
            f"GBMEncoder(model_type='{self.model_type}', "
            f"n_estimators={self.n_estimators}, "
            f"output_type='{self.output_type}')"
        )
