"""XGBoost风控模型.

基于XGBoost实现的风控模型，支持自定义损失函数和评估指标。

**依赖**
pip install xgboost

**示例**
>>> from hscredit.core.models import XGBoostRiskModel
>>> model = XGBoostRiskModel(
...     max_depth=5,
...     learning_rate=0.1,
...     n_estimators=100,
...     eval_metric=['auc', 'ks']
... )
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_test)
>>> report = model.generate_report(X_train, y_train, X_test, y_test)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from ..base import BaseRiskModel


class XGBoostRiskModel(BaseRiskModel):
    """XGBoost风控模型 - 基于内部建模经验优化.

    基于XGBoost的二分类模型，针对风控场景优化。
    参考内部建模经验，支持自动scale_pos_weight计算、KS评估指标等。

    **参数**

    :param max_depth: 树最大深度，默认6
        - 内部经验: 特征>1000时建议5-13，否则3-9
    :param learning_rate: 学习率，默认0.1
        - 内部经验: 一般0.01-0.3，常用0.1
    :param n_estimators: 树的数量，默认100
        - 内部经验: 样本>10000时50-300，否则20-100
    :param min_child_weight: 叶子节点最小样本权重和，默认1
        - 内部经验: 样本>10000时10-2000，否则10-300
    :param subsample: 样本采样比例，默认0.8
        - 内部经验: 常用0.6-0.9防止过拟合
    :param colsample_bytree: 特征采样比例，默认0.8
        - 内部经验: 常用0.6-0.9防止过拟合
    :param colsample_bylevel: 每层的特征采样比例，默认1.0
    :param reg_alpha: L1正则化系数，默认0
        - 内部经验: 常用0, 0.01, 0.1, 1, 10, 100
    :param reg_lambda: L2正则化系数，默认1
    :param scale_pos_weight: 正负样本权重比，默认'auto'
        - 'auto': 自动计算 (当bad_rate<0.05时)
        - float: 自定义权重
        - 内部经验: 当bad_rate<0.05时设置0.05 * n_samples / n_positive
    :param gamma: 节点分裂所需的最小损失减少，默认0
    :param max_delta_step: 每棵树权重改变的最大步长，默认0
    :param tree_method: 树构建算法，默认'hist'
        - 'hist': 直方图算法（推荐，速度快）
        - 'exact': 精确贪心算法
        - 'approx': 近似算法
        - 'gpu_hist': GPU直方图算法
    :param objective: 目标函数，默认'binary:logistic'
    :param eval_metric: 评估指标，可选列表
        - 支持'ks'作为自定义评估指标（风控常用）
        - 多个指标时，默认使用第一个指标进行早停
    :param early_stopping_rounds: 早停轮数，默认None
        - 连续N轮没有改善则停止训练
        - XGBoost 2.0+ 需要在构造函数中传入此参数
    :param early_stopping_metric: 用于早停的评估指标名称，默认None（使用第一个指标）
        - 当eval_metric有多个时，指定用哪个指标进行早停判断
        - XGBoost 2.0+ 使用方式: 传入metric名称，如 'auc', 'logloss', 'error'
        - 注意: 指标名称必须是eval_metric中指定的名称之一
        - 例如: eval_metric=['auc','logloss']时，可指定'auc'或'logloss'作为早停指标
    :param early_stopping_data: 用于早停的验证集名称，默认None（使用第一个验证集）
        - XGBoost内部会自动命名为'validation_0', 'validation_1'等
        - 通常不需要指定，除非有多个验证集
    :param validation_fraction: 验证集比例，默认0.2
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1
    :param verbose: 是否输出详细信息，默认False
    :param params: XGBoost原生参数字典，默认None
        - 如果传入，将覆盖其他参数设置
        - 可直接使用XGBoost原生参数名
    :param kwargs: 其他XGBoost参数

    **属性**

    :ivar feature_importances_: 特征重要性
    :ivar evals_result_: 训练过程评估结果
    :ivar best_iteration_: 最佳迭代次数
    :ivar best_score_: 最佳得分
    :ivar scale_pos_weight_: 实际使用的scale_pos_weight值
    :ivar booster_: 底层XGBoost模型

    **示例**

    >>> # 基础使用
    >>> model = XGBoostRiskModel(max_depth=5, learning_rate=0.1)
    >>> model.fit(X_train, y_train)
    
    >>> # 自动处理不平衡数据
    >>> model = XGBoostRiskModel(scale_pos_weight='auto')
    >>> model.fit(X_train, y_train)  # 自动计算权重
    
    >>> # 使用KS作为评估指标
    >>> model = XGBoostRiskModel(eval_metric='ks')
    >>> model.fit(X_train, y_train)
    
    >>> # 使用原生XGBoost参数
    >>> params = {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}
    >>> model = XGBoostRiskModel(params=params)
    >>> model.fit(X_train, y_train)
    
    >>> # 早停设置 - 使用多个评估指标，指定logloss作为早停指标（越小越好）
    >>> model = XGBoostRiskModel(
    ...     n_estimators=1000,
    ...     eval_metric=['auc', 'logloss'],  # 同时监控AUC和logloss
    ...     early_stopping_rounds=10,       # 10轮没有改善则停止
    ...     early_stopping_metric='logloss' # 使用logloss作为早停判断标准
    ... )
    >>> model.fit(X_train, y_train)
    >>> print(f'最佳迭代次数: {model.best_iteration_}')
    
    >>> # 早停设置 - 使用AUC作为早停指标（越大越好）
    >>> model = XGBoostRiskModel(
    ...     n_estimators=1000,
    ...     eval_metric=['auc', 'logloss'],
    ...     early_stopping_rounds=10,
    ...     early_stopping_metric='auc'  # 使用AUC作为早停判断标准（默认）
    ... )
    >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_weight: float = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bylevel: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        scale_pos_weight: Union[str, float] = 'auto',
        gamma: float = 0,
        max_delta_step: float = 0,
        tree_method: str = 'hist',
        objective: str = 'binary:logistic',
        eval_metric: Union[str, List[str], None] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        early_stopping_data: Optional[str] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost未安装，请使用 pip install xgboost 安装"
            )

        # 保存原生params参数
        self.params = params  # 用于sklearn get_params兼容性
        self._native_params = params or {}

        # 从params中提取参数（如果提供了原生参数）
        max_depth = self._native_params.get('max_depth', max_depth)
        learning_rate = self._native_params.get('learning_rate', learning_rate)
        n_estimators = self._native_params.get('n_estimators', n_estimators)
        min_child_weight = self._native_params.get('min_child_weight', min_child_weight)
        subsample = self._native_params.get('subsample', subsample)
        colsample_bytree = self._native_params.get('colsample_bytree', colsample_bytree)
        colsample_bylevel = self._native_params.get('colsample_bylevel', colsample_bylevel)
        reg_alpha = self._native_params.get('reg_alpha', reg_alpha)
        reg_lambda = self._native_params.get('reg_lambda', reg_lambda)
        scale_pos_weight = self._native_params.get('scale_pos_weight', scale_pos_weight)
        gamma = self._native_params.get('gamma', gamma)
        max_delta_step = self._native_params.get('max_delta_step', max_delta_step)
        tree_method = self._native_params.get('tree_method', tree_method)
        objective = self._native_params.get('objective', objective)
        random_state = self._native_params.get('random_state', random_state)
        n_jobs = self._native_params.get('n_jobs', n_jobs)
        # 从params中提取early_stopping_rounds（优先级最高）
        early_stopping_rounds = self._native_params.get('early_stopping_rounds', early_stopping_rounds)

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

        # 早停相关参数
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_data = early_stopping_data

        # XGBoost特有参数
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self._scale_pos_weight_input = scale_pos_weight  # 保存原始输入
        self.scale_pos_weight = scale_pos_weight if scale_pos_weight != 'auto' else 1.0
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.tree_method = tree_method
        self.scale_pos_weight_ = None  # 实际使用的值

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'XGBoostRiskModel':
        """训练XGBoost模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: fit(X) 在init中指定target

        :param X: 特征矩阵
        :param y: 目标变量，可选
        :param sample_weight: 样本权重
        :param eval_set: 验证集列表
        :param fit_params: 其他fit参数
        :return: self
        """
        # 准备数据（支持从X中提取target）
        X, y, sample_weight = self._prepare_data(X, y, sample_weight, extract_target=True)

        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 自动计算scale_pos_weight（内部建模经验：当bad_rate<0.05时）
        if self._scale_pos_weight_input == 'auto':
            pos_ratio = np.mean(y == 1)
            if pos_ratio < 0.05:
                # 内部经验: 0.05 * n_samples / n_positive
                self.scale_pos_weight = 0.05 * len(y) / np.sum(y == 1)
                if self.verbose:
                    print(f"自动计算scale_pos_weight: {self.scale_pos_weight:.2f} (bad_rate={pos_ratio:.4f})")
            else:
                self.scale_pos_weight = 1.0
        else:
            self.scale_pos_weight = self._scale_pos_weight_input
        
        self.scale_pos_weight_ = self.scale_pos_weight

        # 创建验证集
        if eval_set is None and self.validation_fraction > 0:
            X_train, X_val, y_train, y_val, sw_train, sw_val = self._create_eval_set(
                X, y, sample_weight
            )
            eval_set = [(X_val, y_val)]
            sample_weight = sw_train
        else:
            X_train, y_train = X, y
            # 处理用户传入的 eval_set - 确保与训练数据格式一致（numpy数组）
            if eval_set is not None:
                processed_eval_set = []
                for eval_X, eval_y in eval_set:
                    # 将验证集转换为 numpy 数组（与训练数据保持一致）
                    if isinstance(eval_X, pd.DataFrame):
                        eval_X = eval_X.values
                    if isinstance(eval_y, pd.Series):
                        eval_y = eval_y.values
                    processed_eval_set.append((eval_X, eval_y))
                eval_set = processed_eval_set

        # 构建参数 - 在构造函数中传入所有参数
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'scale_pos_weight': self.scale_pos_weight,
            'gamma': self.gamma,
            'max_delta_step': self.max_delta_step,
            'tree_method': self.tree_method,
            'objective': self.objective,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbosity': 2 if self.verbose else 0,
        }

        # 处理评估指标
        if self.eval_metric is not None:
            params['eval_metric'] = self._convert_metrics(self.eval_metric)

        # 处理早停参数（XGBoost 2.0+ 在构造函数中传入）
        if self.early_stopping_rounds is not None and eval_set:
            # 如果指定了早停指标，使用EarlyStopping回调
            if self.early_stopping_metric is not None:
                try:
                    from xgboost.callback import EarlyStopping
                    callbacks = [EarlyStopping(
                        rounds=self.early_stopping_rounds,
                        metric_name=self.early_stopping_metric,
                        data_name=self.early_stopping_data,
                        save_best=True
                    )]
                    params['callbacks'] = callbacks
                    if self.verbose:
                        print(f"使用早停: rounds={self.early_stopping_rounds}, "
                              f"metric='{self.early_stopping_metric}'")
                except ImportError:
                    # 回退到旧方式
                    params['early_stopping_rounds'] = self.early_stopping_rounds
                    if self.verbose:
                        print(f"使用早停: rounds={self.early_stopping_rounds} (默认指标)")
            else:
                # 未指定早停指标，使用默认方式（第一个eval_metric）
                params['early_stopping_rounds'] = self.early_stopping_rounds
                if self.verbose:
                    print(f"使用早停: rounds={self.early_stopping_rounds} (默认使用第一个指标)")

        # 更新kwargs参数
        params.update(self.kwargs)
        
        # 最后更新原生params（优先级最高）
        params.update(self._native_params)

        # 创建模型
        self._model = xgb.XGBClassifier(**params)

        # 训练 - fit时不传早停参数（已在构造函数中传入）
        fit_kwargs = {'eval_set': eval_set} if eval_set else {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        fit_kwargs['verbose'] = self.verbose

        # 执行训练
        self._model.fit(X_train, y_train, **fit_kwargs)

        # 保存结果
        self._best_iteration = getattr(self._model, 'best_iteration', None)
        self._best_score = getattr(self._model, 'best_score', None)
        self._evals_result = getattr(self._model, 'evals_result_', {})
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

    def _ks_metric(self, y_pred, dtrain):
        """KS评估指标（用于XGBoost内部评估）.

        参考内部建模常用KS作为评估指标。
        """
        from sklearn.metrics import roc_curve
        y_true = dtrain.get_label()
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        ks = abs(tpr - fpr).max()
        return 'KS', ks

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
        """
        check_is_fitted(self, '_is_fitted')
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._model.predict_proba(X)

    def get_feature_importances(self, importance_type: str = 'gain') -> pd.Series:
        """获取特征重要性.

        :param importance_type: 重要性类型，可选:
            - 'gain': 平均增益 (默认)
            - 'weight': 分裂次数
            - 'cover': 平均覆盖度
            - 'total_gain': 总增益
            - 'total_cover': 总覆盖度
        :return: 特征重要性Series
        """
        check_is_fitted(self, '_is_fitted')

        importances = self._model.feature_importances_

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=self.feature_names_in_,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    def get_booster(self) -> 'xgb.Booster':
        """获取底层XGBoost booster对象.

        :return: XGBoost Booster对象
        """
        check_is_fitted(self, '_is_fitted')
        return self._model.get_booster()

    def plot_tree(self, num_trees: int = 0, **kwargs):
        """绘制树结构.

        :param num_trees: 树的索引
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, '_is_fitted')
        return xgb.plot_tree(self._model, num_trees=num_trees, **kwargs)

    def plot_importance(self, max_num_features: int = 10, **kwargs):
        """绘制特征重要性.

        :param max_num_features: 显示的最大特征数
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, '_is_fitted')
        return xgb.plot_importance(
            self._model,
            max_num_features=max_num_features,
            **kwargs
        )

    def get_leaf_indices(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """获取叶子节点索引.

        返回每棵树上的叶子节点索引，用于GBDT+LR等场景。

        :param X: 特征矩阵
        :return: 叶子节点索引，形状 (n_samples, n_trees)

        **示例**

        >>> model = XGBoostRiskModel(n_estimators=50)
        >>> model.fit(X, y)
        >>> leaf_indices = model.get_leaf_indices(X)
        >>> print(leaf_indices.shape)  # (n_samples, 50)
        """
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.apply(X)

    def _convert_metrics(self, metrics: Union[str, List[str]]) -> Union[str, List[str]]:
        """转换评估指标名称.

        :param metrics: 指标名称或列表
        :return: XGBoost格式的指标名称
        """
        metric_map = {
            'auc': 'auc',
            'logloss': 'logloss',
            'error': 'error',
            'rmse': 'rmse',
            'mae': 'mae',
            'map': 'map',
            'merror': 'merror',
            'mlogloss': 'mlogloss',
        }

        if isinstance(metrics, str):
            return metric_map.get(metrics.lower(), metrics)

        return [metric_map.get(m.lower(), m) for m in metrics]

    def save_model(self, path: str):
        """保存模型.

        :param path: 保存路径
        """
        check_is_fitted(self, '_is_fitted')
        self._model.save_model(path)

    def load_model(self, path: str) -> 'XGBoostRiskModel':
        """加载模型.

        :param path: 模型路径
        :return: self
        """
        self._model = xgb.XGBClassifier()
        self._model.load_model(path)
        self._is_fitted = True
        return self
