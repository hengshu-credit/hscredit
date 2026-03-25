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

# 检测XGBoost版本和API支持情况
def _check_xgboost_api_support():
    """检测XGBoost API支持情况.
    
    返回: (支持callbacks, 支持_early_stopping_rounds)
    """
    if not XGBOOST_AVAILABLE:
        return False, False

    supports_callbacks = False
    supports_early_stopping = False

    # 方法1: 检查XGBClassifier.fit签名
    try:
        import inspect
        fit_signature = inspect.signature(xgb.XGBClassifier.fit)
        params = list(fit_signature.parameters.keys())
        supports_callbacks = 'callbacks' in params
        supports_early_stopping = 'early_stopping_rounds' in params
    except Exception:
        pass

    # 方法2: 通过版本号检测（如果方法1失败）
    if not supports_callbacks and not supports_early_stopping:
        try:
            from packaging import version
            xgb_version = xgb.__version__
            ver = version.parse(xgb_version)
            if ver >= version.parse("2.0.0"):
                supports_callbacks = True
            elif ver >= version.parse("0.90"):
                supports_early_stopping = True
        except Exception:
            pass

    return supports_callbacks, supports_early_stopping

XGBOOST_CALLBACKS_AVAILABLE, XGBOOST_EARLY_STOPPING_AVAILABLE = _check_xgboost_api_support()

# 导入回调（如果可用）
if XGBOOST_CALLBACKS_AVAILABLE:
    try:
        from xgboost.callback import EarlyStopping, EvaluationMonitor
    except ImportError:
        EarlyStopping = None
        EvaluationMonitor = None
        XGBOOST_CALLBACKS_AVAILABLE = False
else:
    EarlyStopping = None
    EvaluationMonitor = None

from .base import BaseRiskModel


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
    :param early_stopping_metric: 用于早停的评估指标名称，默认None（使用第一个指标）
        - 当eval_metric有多个时，指定用哪个指标进行早停判断
        - 例如: 'auc', 'logloss', 'validation_0-auc'
    :param early_stopping_data: 用于早停的验证集名称，默认None（使用第一个验证集）
        - 例如: 'validation_0', 'validation_1'
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

        # 构建参数
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

        # 更新kwargs参数
        params.update(self.kwargs)
        
        # 最后更新原生params（优先级最高）
        params.update(self._native_params)

        # 创建模型
        self._model = xgb.XGBClassifier(**params)

        # 训练
        fit_kwargs = {'eval_set': eval_set} if eval_set else {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        # 处理早停 - 适配新旧版本XGBoost
        use_callbacks = False
        early_stopping_enabled = False
        if self.early_stopping_rounds is not None and eval_set:
            # 检测XGBoost版本和可用的早停方式
            callbacks = self._build_early_stopping_callbacks()
            if callbacks is not None:
                # XGBoost 2.0+ 使用 callbacks
                fit_kwargs['callbacks'] = callbacks
                use_callbacks = True
                early_stopping_enabled = True
            elif XGBOOST_EARLY_STOPPING_AVAILABLE:
                # 旧版本API (< 2.0, >= 0.90)
                fit_kwargs['early_stopping_rounds'] = self.early_stopping_rounds
                fit_kwargs['verbose'] = self.verbose
                early_stopping_enabled = True
            else:
                # 非常旧的版本(< 0.90)不支持早停，给出警告
                import warnings
                warnings.warn(
                    f"当前XGBoost版本({xgb.__version__})不支持early_stopping_rounds参数，"
                    f"将忽略早停设置。建议升级XGBoost: pip install -U xgboost",
                    UserWarning
                )

            # 支持KS作为自定义评估指标
            if self.eval_metric == 'ks' or (isinstance(self.eval_metric, list) and 'ks' in self.eval_metric):
                fit_kwargs['eval_metric'] = self._ks_metric
        
        # 传递verbose参数（如果没有使用早停或者旧版本）
        if not early_stopping_enabled or not use_callbacks:
            fit_kwargs['verbose'] = self.verbose

        # 尝试训练，如果失败则自动回退
        try:
            self._model.fit(X_train, y_train, **fit_kwargs)
        except TypeError as e:
            error_msg = str(e).lower()
            if 'callbacks' in error_msg and use_callbacks:
                # callbacks参数不被支持，回退到旧API
                if self.verbose:
                    print(f"callbacks参数不被支持，回退到旧版early_stopping_rounds API")
                del fit_kwargs['callbacks']
                fit_kwargs['early_stopping_rounds'] = self.early_stopping_rounds
                fit_kwargs['verbose'] = self.verbose
                self._model.fit(X_train, y_train, **fit_kwargs)
            elif 'early_stopping_rounds' in error_msg:
                # 旧版本不支持early_stopping_rounds，忽略早停
                import warnings
                warnings.warn(
                    f"当前XGBoost版本不支持early_stopping_rounds参数，"
                    f"将忽略早停设置。建议升级XGBoost: pip install -U xgboost",
                    UserWarning
                )
                if 'early_stopping_rounds' in fit_kwargs:
                    del fit_kwargs['early_stopping_rounds']
                fit_kwargs['verbose'] = self.verbose
                self._model.fit(X_train, y_train, **fit_kwargs)
            else:
                raise

        # 保存结果
        self._best_iteration = getattr(self._model, 'best_iteration', None)
        self._best_score = getattr(self._model, 'best_score', None)
        self._evals_result = getattr(self._model, 'evals_result_', {})
        self._is_fitted = True

        return self

    def _ks_metric(self, y_pred, dtrain):
        """KS评估指标（用于XGBoost内部评估）.

        参考内部建模常用KS作为评估指标。
        """
        from sklearn.metrics import roc_curve
        y_true = dtrain.get_label()
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        ks = abs(tpr - fpr).max()
        return 'KS', ks

    def _build_early_stopping_callbacks(self) -> Optional[List[Any]]:
        """构建早停回调函数列表。

        适配XGBoost 2.0+版本的callbacks机制，同时兼容旧版本。

        :return: 回调函数列表，如果无法使用新机制则返回None（使用旧API）
        """
        if not XGBOOST_CALLBACKS_AVAILABLE:
            # XGBoost < 2.0，返回None使用旧API
            return None

        try:
            callbacks = []

            # 构建早停回调参数
            es_kwargs = {
                'rounds': self.early_stopping_rounds,
                'save_best': True,
            }

            # 指定用于早停的评估指标
            if self.early_stopping_metric is not None:
                es_kwargs['metric_name'] = self.early_stopping_metric

            # 指定用于早停的验证集
            if self.early_stopping_data is not None:
                es_kwargs['data_name'] = self.early_stopping_data

            # 创建早停回调
            callbacks.append(EarlyStopping(**es_kwargs))

            # 添加日志监控回调（如果需要verbose）
            if self.verbose and EvaluationMonitor is not None:
                try:
                    callbacks.append(EvaluationMonitor(period=1))
                except (TypeError, ValueError):
                    # 某些版本可能不支持
                    pass

            return callbacks

        except Exception as e:
            # 其他异常，降级到旧API
            if self.verbose:
                print(f"无法创建早停回调: {e}，降级到旧API")
            return None

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率."""
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
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
