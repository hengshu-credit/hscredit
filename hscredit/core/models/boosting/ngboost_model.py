"""NGBoost风控模型.

基于NGBoost实现的风控模型，支持概率预测和不确定性估计。
NGBoost(Natural Gradient Boosting)使用自然梯度来提升概率预测的质量，
能够同时输出预测值和预测不确定性。

**依赖**
pip install ngboost

**参考样例**
>>> from hscredit.core.models import NGBoostRiskModel
>>> model = NGBoostRiskModel(
...     n_estimators=500,
...     learning_rate=0.01,
...     eval_metric=['auc', 'ks']
... )
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_test)
>>> # 获取概率分布对象
>>> dist = model.pred_dist(X_test)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:
    import ngboost
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from ngboost.scores import LogScore
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    ngboost = None
    NGBClassifier = None
    Bernoulli = None
    LogScore = None

from ..base import BaseRiskModel


class NGBoostRiskModel(BaseRiskModel):
    """NGBoost风控模型.

    基于NGBoost的二分类模型，针对风控场景优化。
    NGBoost通过自然梯度提升进行概率预测，能够输出预测不确定性，
    适合风控中需要概率校准和不确定性量化的场景。

    **参数**

    :param n_estimators: 提升迭代次数（基学习器数量），默认500
    :param learning_rate: 学习率，默认0.01
        - NGBoost对学习率较敏感，通常使用较小的值
    :param minibatch_frac: 每次迭代使用的样本比例，默认1.0
        - 设为<1.0可进行随机梯度提升，加速训练
    :param col_sample: 每次迭代使用的特征比例，默认1.0
        - 类似XGBoost的colsample_bytree
    :param base_max_depth: 基学习器（决策树）最大深度，默认3
        - NGBoost使用回归树作为基学习器
    :param base_criterion: 基学习器分裂准则，默认'friedman_mse'
    :param base_min_samples_split: 基学习器节点分裂最小样本数，默认2
    :param base_min_samples_leaf: 基学习器叶子节点最小样本数，默认1
    :param natural_gradient: 是否使用自然梯度，默认True
        - True: 使用自然梯度（推荐）
        - False: 使用普通梯度
    :param objective: 目标函数，默认'binary'
    :param eval_metric: 评估指标，可选列表
    :param early_stopping_rounds: 早停轮数，默认None
        - 连续N轮验证集loss没有改善则停止训练
    :param validation_fraction: 验证集比例，默认0.2
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1（NGBoost基学习器内部使用）
    :param verbose: 是否输出详细信息，默认False
    :param params: NGBoost原生参数字典，默认None
        - 如果传入，将覆盖其他参数设置
    :param kwargs: 其他NGBoost参数

    **属性**

    :ivar feature_importances_: 特征重要性（基于loc参数的基学习器）
    :ivar evals_result_: 训练过程评估结果
    :ivar best_iteration_: 最佳迭代次数（验证集最优时的迭代轮数）
    :ivar best_score_: 最佳得分

    **参考样例**

    >>> # 基础使用
    >>> model = NGBoostRiskModel(n_estimators=500, learning_rate=0.01)
    >>> model.fit(X_train, y_train)

    >>> # 使用验证集早停
    >>> model = NGBoostRiskModel(
    ...     n_estimators=1000,
    ...     learning_rate=0.01,
    ...     early_stopping_rounds=50
    ... )
    >>> model.fit(X_train, y_train)

    >>> # 使用原生NGBoost参数
    >>> params = {'n_estimators': 500, 'learning_rate': 0.01, 'minibatch_frac': 0.8}
    >>> model = NGBoostRiskModel(params=params)
    >>> model.fit(X_train, y_train)

    >>> # 获取概率分布（不确定性估计）
    >>> dist = model.pred_dist(X_test)
    >>> print(dist.params)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        col_sample: float = 1.0,
        base_max_depth: int = 3,
        base_criterion: str = 'friedman_mse',
        base_min_samples_split: int = 2,
        base_min_samples_leaf: int = 1,
        natural_gradient: bool = True,
        objective: str = 'binary',
        eval_metric: Union[str, List[str], None] = None,
        early_stopping_rounds: Optional[int] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not NGBOOST_AVAILABLE:
            raise ImportError(
                "NGBoost未安装，请使用 pip install ngboost 安装"
            )

        # 保存原生params参数
        self.params = params  # 用于sklearn get_params兼容性
        self._native_params = params or {}

        # 从params中提取参数（如果提供了原生参数）
        n_estimators = self._native_params.get('n_estimators', n_estimators)
        learning_rate = self._native_params.get('learning_rate', learning_rate)
        minibatch_frac = self._native_params.get('minibatch_frac', minibatch_frac)
        col_sample = self._native_params.get('col_sample', col_sample)
        base_max_depth = self._native_params.get('base_max_depth', base_max_depth)
        base_criterion = self._native_params.get('base_criterion', base_criterion)
        base_min_samples_split = self._native_params.get('base_min_samples_split', base_min_samples_split)
        base_min_samples_leaf = self._native_params.get('base_min_samples_leaf', base_min_samples_leaf)
        natural_gradient = self._native_params.get('natural_gradient', natural_gradient)
        objective = self._native_params.get('objective', objective)
        random_state = self._native_params.get('random_state', random_state)

        super().__init__(
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs
        )

        # NGBoost特有参数
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.base_max_depth = base_max_depth
        self.base_criterion = base_criterion
        self.base_min_samples_split = base_min_samples_split
        self.base_min_samples_leaf = base_min_samples_leaf
        self.natural_gradient = natural_gradient

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'NGBoostRiskModel':
        """训练NGBoost模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: fit(X) 在init中指定target

        :param X: 特征矩阵
        :param y: 目标变量，可选
        :param sample_weight: 样本权重
        :param eval_set: 验证集列表 [(X_val, y_val)]
        :param fit_params: 其他fit参数
        :return: self
        """
        from sklearn.tree import DecisionTreeRegressor

        # 准备数据（支持从X中提取target）
        X, y, sample_weight = self._prepare_data(X, y, sample_weight, extract_target=True)

        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 创建验证集
        X_val, y_val = None, None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            X_train, y_train = X, y
        elif self.validation_fraction > 0 and (self.early_stopping_rounds is not None):
            X_train, X_val, y_train, y_val, sw_train, _ = self._create_eval_set(
                X, y, sample_weight
            )
            sample_weight = sw_train
        else:
            X_train, y_train = X, y

        # 构建基学习器
        base_learner = DecisionTreeRegressor(
            criterion=self.base_criterion,
            max_depth=self.base_max_depth,
            min_samples_split=self.base_min_samples_split,
            min_samples_leaf=self.base_min_samples_leaf,
        )

        # 构建NGBoost参数
        ngb_params = {
            'Dist': Bernoulli,
            'Score': LogScore,
            'Base': base_learner,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'minibatch_frac': self.minibatch_frac,
            'col_sample': self.col_sample,
            'natural_gradient': self.natural_gradient,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

        # 更新kwargs参数
        ngb_params.update(self.kwargs)

        # 最后更新原生params（优先级最高，但排除已处理的base参数）
        for k, v in self._native_params.items():
            if not k.startswith('base_'):
                ngb_params[k] = v

        # 创建模型
        self._model = NGBClassifier(**ngb_params)

        # 训练
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        if X_val is not None and y_val is not None:
            fit_kwargs['X_val'] = X_val
            fit_kwargs['Y_val'] = y_val
            if self.early_stopping_rounds is not None:
                fit_kwargs['early_stopping_rounds'] = self.early_stopping_rounds

        self._model.fit(X_train, y_train, **fit_kwargs)

        # 保存结果
        self._best_iteration = getattr(self._model, 'best_val_loss_itr', None)
        self._best_score = None
        self._evals_result = {}
        self._is_fitted = True

        return self

    @property
    def best_iteration_(self):
        """最佳迭代次数."""
        return self._best_iteration

    @property
    def best_score_(self):
        """最佳得分."""
        return self._best_score

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签.

        支持传入包含target列的数据框（scorecardpipeline风格）。
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率.

        支持传入包含target列的数据框（scorecardpipeline风格）。

        :param X: 特征矩阵
        :return: 预测概率，形状 (n_samples, 2)
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.predict_proba(X)

    def pred_dist(self, X: Union[np.ndarray, pd.DataFrame]):
        """预测概率分布.

        返回NGBoost分布对象，可用于获取预测不确定性。

        :param X: 特征矩阵
        :return: NGBoost分布对象

        **参考样例**

        >>> model = NGBoostRiskModel()
        >>> model.fit(X_train, y_train)
        >>> dist = model.pred_dist(X_test)
        >>> print(dist.params)
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.pred_dist(X)

    def get_feature_importances(self, importance_type: str = 'gain') -> pd.Series:
        """获取特征重要性.

        NGBoost为每个分布参数维护独立的基学习器序列。
        对于二分类(Bernoulli)，使用第一个参数(p)的特征重要性。

        :param importance_type: 重要性类型（保留参数，NGBoost使用基学习器默认重要性）
        :return: 特征重要性Series
        """
        check_is_fitted(self, '_is_fitted')

        # NGBoost feature_importances_ 返回 (n_params, n_features) 数组
        # 对于Bernoulli二分类，取第一个参数(p)的重要性
        raw_importances = self._model.feature_importances_
        if isinstance(raw_importances, np.ndarray) and raw_importances.ndim == 2:
            importances = raw_importances[0]
        elif isinstance(raw_importances, list):
            importances = np.asarray(raw_importances[0])
        else:
            importances = np.asarray(raw_importances).ravel()

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=self.feature_names_in_,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性属性 (兼容sklearn风格).

        返回一维numpy数组，与其他RiskModel保持一致。
        """
        check_is_fitted(self, '_is_fitted')
        raw_importances = self._model.feature_importances_
        if isinstance(raw_importances, np.ndarray) and raw_importances.ndim == 2:
            return raw_importances[0].astype(float)
        elif isinstance(raw_importances, list):
            return np.asarray(raw_importances[0], dtype=float)
        return np.asarray(raw_importances, dtype=float).ravel()

    def staged_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> List[np.ndarray]:
        """阶段预测.

        返回每个迭代阶段的预测结果列表。

        :param X: 特征矩阵
        :return: 预测结果列表，长度为n_estimators
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.staged_predict(X)

    def staged_pred_dist(self, X: Union[np.ndarray, pd.DataFrame]) -> list:
        """阶段分布预测.

        返回每个迭代阶段的完整分布预测。

        :param X: 特征矩阵
        :return: 分布对象列表
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.staged_pred_dist(X)

    def plot_importance(self, max_num_features: int = 10, figsize: Tuple = (10, 6), **kwargs):
        """绘制特征重要性.

        :param max_num_features: 显示的最大特征数
        :param figsize: 图像大小
        :param kwargs: 其他绘图参数
        """
        import matplotlib.pyplot as plt

        check_is_fitted(self, '_is_fitted')
        importances = self.get_feature_importances()
        top_features = importances.head(max_num_features)

        fig, ax = plt.subplots(figsize=figsize)
        top_features.iloc[::-1].plot.barh(ax=ax, **kwargs)
        ax.set_title('NGBoost Feature Importance')
        ax.set_xlabel('Importance')
        return fig

    def save_model(self, path: str):
        """保存模型.

        :param path: 保存路径（.pkl格式）
        """
        import pickle
        check_is_fitted(self, '_is_fitted')
        with open(path, 'wb') as f:
            pickle.dump(self._model, f)

    def load_model(self, path: str) -> 'NGBoostRiskModel':
        """加载模型.

        :param path: 模型路径（.pkl格式）
        :return: self
        """
        import pickle
        with open(path, 'rb') as f:
            self._model = pickle.load(f)
        self._is_fitted = True
        return self

    def _convert_metrics(self, metrics: Union[str, List[str]]) -> Union[str, List[str]]:
        """转换评估指标名称.

        :param metrics: 指标名称或列表
        :return: 转换后的指标名称
        """
        # NGBoost使用NLL作为内部损失，评估指标通过evaluate方法处理
        metric_map = {
            'auc': 'auc',
            'logloss': 'logloss',
            'nll': 'nll',
        }

        if isinstance(metrics, str):
            return metric_map.get(metrics.lower(), metrics)

        return [metric_map.get(m.lower(), m) for m in metrics]
