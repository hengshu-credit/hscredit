"""CatBoost风控模型.

基于CatBoost实现的风控模型，对类别特征处理更友好。

**依赖**
pip install catboost

**示例**
>>> from hscredit.core.models import CatBoostRiskModel
>>> model = CatBoostRiskModel(
...     depth=6,
...     learning_rate=0.1,
...     iterations=100,
...     eval_metric='AUC'
... )
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_test)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

# 检测CatBoost版本
def _get_catboost_version():
    """获取CatBoost版本号."""
    if not CATBOOST_AVAILABLE:
        return None
    try:
        from packaging import version
        return version.parse(cb.__version__)
    except Exception:
        return None

CATBOOST_VERSION = _get_catboost_version()

from ..base import BaseRiskModel


class CatBoostRiskModel(BaseRiskModel):
    """CatBoost风控模型.

    基于CatBoost的二分类模型，针对风控场景优化。
    CatBoost对类别特征有原生支持，无需编码。

    **参数**

    :param depth: 树深度，默认6
    :param learning_rate: 学习率，默认0.1
    :param iterations: 迭代次数，默认100
    :param l2_leaf_reg: L2正则化系数，默认3.0
    :param border_count: 边界分割数，默认254
    :param random_strength: 随机强度，默认1
    :param bagging_temperature: 采样温度，默认1
    :param scale_pos_weight: 正负样本权重比，默认1
    :param min_data_in_leaf: 叶子节点最小样本数，默认1
    :param grow_policy: 生长策略，默认'SymmetricTree'
        - 'SymmetricTree': 对称树
        - 'Depthwise': 逐层生长
        - 'Lossguide': 按损失导向生长
    :param objective: 目标函数，默认'Logloss'
    :param eval_metric: 评估指标，默认'AUC'
        - 支持字符串或列表（多个评估指标）
    :param early_stopping_rounds: 早停轮数，默认None
        - 当验证集指标连续N轮没有提升时停止训练
        - CatBoost仍支持此参数（与XGBoost/LightGBM新版不同）
    :param early_stopping_metric: 用于早停的评估指标名称，默认None（使用eval_metric）
        - 当eval_metric有多个时，指定用哪个指标进行早停判断
    :param validation_fraction: 验证集比例，默认0.2
    :param random_state: 随机种子，默认None
    :param verbose: 是否输出详细信息，默认False
    :param params: CatBoost原生参数字典，默认None
        - 如果传入，将覆盖其他参数设置
        - 可直接使用CatBoost原生参数名
    :param kwargs: 其他CatBoost参数

    **属性**

    :ivar feature_importances_: 特征重要性
    :ivar evals_result_: 训练过程评估结果
    :ivar best_iteration_: 最佳迭代次数
    :ivar best_score_: 最佳得分
    
    **示例**
    
    >>> # 基础使用
    >>> model = CatBoostRiskModel(depth=6, learning_rate=0.1)
    >>> model.fit(X_train, y_train)
    
    >>> # 使用原生CatBoost参数
    >>> params = {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0}
    >>> model = CatBoostRiskModel(params=params)
    >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 100,
        l2_leaf_reg: float = 3.0,
        border_count: int = 254,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        scale_pos_weight: float = 1.0,
        min_data_in_leaf: int = 1,
        grow_policy: str = 'SymmetricTree',
        objective: str = 'Logloss',
        eval_metric: Union[str, List[str], None] = 'AUC',
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        verbose: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost未安装，请使用 pip install catboost 安装"
            )

        # 保存原生params参数
        self.params = params  # 用于sklearn get_params兼容性
        self._native_params = params or {}

        # 从params中提取参数（如果提供了原生参数）
        depth = self._native_params.get('depth', depth)
        learning_rate = self._native_params.get('learning_rate', learning_rate)
        iterations = self._native_params.get('iterations', iterations)
        l2_leaf_reg = self._native_params.get('l2_leaf_reg', l2_leaf_reg)
        border_count = self._native_params.get('border_count', border_count)
        random_strength = self._native_params.get('random_strength', random_strength)
        bagging_temperature = self._native_params.get('bagging_temperature', bagging_temperature)
        scale_pos_weight = self._native_params.get('scale_pos_weight', scale_pos_weight)
        min_data_in_leaf = self._native_params.get('min_data_in_leaf', min_data_in_leaf)
        grow_policy = self._native_params.get('grow_policy', grow_policy)
        objective = self._native_params.get('loss_function', objective)
        random_state = self._native_params.get('random_seed', random_state)

        super().__init__(
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            random_state=random_state,
            n_jobs=None,  # CatBoost使用thread_count
            verbose=verbose,
            **kwargs
        )

        # CatBoost特有参数
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.scale_pos_weight = scale_pos_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.grow_policy = grow_policy

        # 早停相关参数
        self.early_stopping_metric = early_stopping_metric

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        cat_features: Optional[List[int]] = None,
        **fit_params
    ) -> 'CatBoostRiskModel':
        """训练CatBoost模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: fit(X) 在init中指定target

        :param X: 特征矩阵
        :param y: 目标变量，可选
        :param sample_weight: 样本权重
        :param eval_set: 验证集列表
        :param cat_features: 类别特征索引列表
        :param fit_params: 其他fit参数
        :return: self
        """
        # CatBoost 在 numpy 矩阵上要求 cat_features 为列下标；若传入列名则先映射
        if cat_features is not None and isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            cat_features = [
                cols.index(c) if isinstance(c, str) else int(c)
                for c in cat_features
            ]

        # 准备数据（支持从X中提取target）
        X, y, sample_weight = self._prepare_data(X, y, sample_weight, extract_target=True)

        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 创建验证集
        if eval_set is None and self.validation_fraction > 0:
            X_train, X_val, y_train, y_val, sw_train, sw_val = self._create_eval_set(
                X, y, sample_weight
            )
            eval_set = [(X_val, y_val)]
            sample_weight = sw_train
        else:
            X_train, y_train = X, y

        # 构建参数
        params = {
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'iterations': self.iterations,
            'l2_leaf_reg': self.l2_leaf_reg,
            'border_count': self.border_count,
            'random_strength': self.random_strength,
            'bagging_temperature': self.bagging_temperature,
            'scale_pos_weight': self.scale_pos_weight,
            'min_data_in_leaf': self.min_data_in_leaf,
            'grow_policy': self.grow_policy,
            'loss_function': self.objective,
            'random_seed': self.random_state,
            'verbose': self.verbose,
            'thread_count': -1,  # 使用所有CPU
        }

        # 处理评估指标
        if self.eval_metric is not None:
            # 转换评估指标格式
            eval_metric_converted = self._convert_metrics(self.eval_metric)
            params['eval_metric'] = eval_metric_converted

        # 处理早停 - CatBoost仍支持early_stopping_rounds参数
        if self.early_stopping_rounds is not None:
            params['early_stopping_rounds'] = self.early_stopping_rounds

            # 如果指定了专门的早停指标，覆盖eval_metric
            if self.early_stopping_metric is not None:
                params['eval_metric'] = self.early_stopping_metric
            # 如果有多个评估指标且没有指定早停指标，使用第一个
            elif isinstance(self.eval_metric, list) and len(self.eval_metric) > 0:
                params['eval_metric'] = self._convert_metrics(self.eval_metric[0])

        # 更新kwargs参数
        params.update(self.kwargs)

        # 最后更新原生params（优先级最高）
        params.update(self._native_params)

        # 创建模型
        self._model = cb.CatBoostClassifier(**params)

        # 准备训练参数
        fit_kwargs = {}
        if eval_set:
            fit_kwargs['eval_set'] = eval_set
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        if cat_features is not None:
            fit_kwargs['cat_features'] = cat_features

        # 训练
        self._model.fit(X_train, y_train, **fit_kwargs)

        # 保存结果
        self._best_iteration = self._model.get_best_iteration()
        self._best_score = self._model.get_best_score()
        self._evals_result = self._model.get_evals_result()
        self._is_fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict(X).flatten()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict_proba(X)

    def get_feature_importances(self, importance_type: str = 'PredictionValuesChange') -> pd.Series:
        """获取特征重要性.

        :param importance_type: 重要性类型，可选:
            - 'PredictionValuesChange': 预测值变化 (默认)
            - 'LossFunctionChange': 损失函数变化
            - 'FeatureImportance': 分裂次数
        :return: 特征重要性Series
        """
        check_is_fitted(self, '_is_fitted')

        importances = self._model.get_feature_importance(type=importance_type)

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=self.feature_names_in_,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    def plot_tree(self, tree_index: int = 0, **kwargs):
        """绘制树结构.

        :param tree_index: 树的索引
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, '_is_fitted')
        return self._model.plot_tree(tree_idx=tree_index, **kwargs)

    def get_leaf_indices(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """获取叶子节点索引.

        返回每棵树上的叶子节点索引，用于GBDT+LR等场景。

        :param X: 特征矩阵
        :return: 叶子节点索引，形状 (n_samples, n_trees)

        **示例**

        >>> model = CatBoostRiskModel(iterations=50)
        >>> model.fit(X, y)
        >>> leaf_indices = model.get_leaf_indices(X)
        >>> print(leaf_indices.shape)  # (n_samples, 50)
        """
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.calc_leaf_indexes(X)

    def save_model(self, path: str):
        """保存模型.

        :param path: 保存路径
        """
        check_is_fitted(self, '_is_fitted')
        self._model.save_model(path)

    def load_model(self, path: str) -> 'CatBoostRiskModel':
        """加载模型.

        :param path: 模型路径
        :return: self
        """
        self._model = cb.CatBoostClassifier()
        self._model.load_model(path)
        self._is_fitted = True
        return self

    def _convert_metrics(self, metrics: Union[str, List[str]]) -> Union[str, List[str]]:
        """转换评估指标名称.

        :param metrics: 指标名称或列表
        :return: CatBoost格式的指标名称
        """
        metric_map = {
            'auc': 'AUC',
            'logloss': 'Logloss',
            'error': 'Accuracy',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mse': 'MSE',
            'msle': 'MSLE',
            'poisson': 'Poisson',
            'quantile': 'Quantile',
            'mape': 'MAPE',
            'r2': 'R2',
            'ndcg': 'NDCG',
            'map': 'MAP',
            'recall': 'Recall',
            'precision': 'Precision',
            'f1': 'F1',
            'balanced_accuracy': 'BalancedAccuracy',
            'balanced_error_rate': 'BalancedErrorRate',
            'kappa': 'Kappa',
            'wkappa': 'WKappa',
            'total_f1': 'TotalF1',
            'mcc': 'MCC',
            'brier_score': 'BrierScore',
            'hinge_loss': 'HingeLoss',
            'hamming_loss': 'HammingLoss',
            'zero_one_loss': 'ZeroOneLoss',
            'kappa:use_weights': 'Kappa:use_weights',
            'wkappa:use_weights': 'WKappa:use_weights',
        }

        if isinstance(metrics, str):
            return metric_map.get(metrics.lower(), metrics)

        return [metric_map.get(m.lower(), m) for m in metrics]
