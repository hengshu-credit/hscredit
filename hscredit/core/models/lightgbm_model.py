"""LightGBM风控模型.

基于LightGBM实现的风控模型，支持自定义损失函数和评估指标。

**依赖**
pip install lightgbm

**示例**
>>> from hscredit.core.models import LightGBMRiskModel
>>> model = LightGBMRiskModel(
...     num_leaves=31,
...     learning_rate=0.1,
...     n_estimators=100,
...     eval_metric=['auc', 'ks']
... )
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_test)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# 检测LightGBM版本和API支持情况
def _check_lightgbm_api_support():
    """检测LightGBM API支持情况.
    
    返回: (支持callbacks, 支持_early_stopping_rounds)
    """
    if not LIGHTGBM_AVAILABLE:
        return False, False

    supports_callbacks = False
    supports_early_stopping = False

    # 方法1: 检查LGBMClassifier.fit签名
    try:
        import inspect
        fit_signature = inspect.signature(lgb.LGBMClassifier.fit)
        params = list(fit_signature.parameters.keys())
        supports_callbacks = 'callbacks' in params
        supports_early_stopping = 'early_stopping_rounds' in params
    except Exception:
        pass

    # 方法2: 通过版本号检测（如果方法1失败）
    if not supports_callbacks and not supports_early_stopping:
        try:
            from packaging import version
            lgb_version = lgb.__version__
            ver = version.parse(lgb_version)
            if ver >= version.parse("4.0.0"):
                supports_callbacks = True
            elif ver >= version.parse("2.0.0"):
                supports_early_stopping = True
        except Exception:
            pass

    return supports_callbacks, supports_early_stopping

LIGHTGBM_CALLBACKS_AVAILABLE, LIGHTGBM_EARLY_STOPPING_AVAILABLE = _check_lightgbm_api_support()

# 导入回调（如果可用）
if LIGHTGBM_CALLBACKS_AVAILABLE:
    try:
        from lightgbm import early_stopping, log_evaluation
    except ImportError:
        early_stopping = None
        log_evaluation = None
else:
    early_stopping = None
    log_evaluation = None

from .base import BaseRiskModel


class LightGBMRiskModel(BaseRiskModel):
    """LightGBM风控模型.

    基于LightGBM的二分类模型，针对风控场景优化。
    LightGBM相比XGBoost训练更快，内存占用更少。

    **参数**

    :param num_leaves: 叶子节点数，默认31
    :param max_depth: 树最大深度，默认-1(无限制)
    :param learning_rate: 学习率，默认0.1
    :param n_estimators: 树的数量，默认100
    :param min_child_samples: 叶子节点最小样本数，默认20
    :param min_child_weight: 叶子节点最小权重和，默认1e-3
    :param subsample: 样本采样比例，默认1.0
    :param colsample_bytree: 特征采样比例，默认1.0
    :param reg_alpha: L1正则化系数，默认0
    :param reg_lambda: L2正则化系数，默认0
    :param scale_pos_weight: 正负样本权重比，默认1
    :param min_split_gain: 节点分裂所需的最小增益，默认0
    :param boosting_type: 提升类型，默认'gbdt'
        - 'gbdt': 传统梯度提升树
        - 'dart': Dropouts meet Multiple Additive Regression Trees
        - 'goss': Gradient-based One-Side Sampling
        - 'rf': 随机森林
    :param objective: 目标函数，默认'binary'
    :param eval_metric: 评估指标，可选列表
        - 多个指标时，默认使用第一个指标进行早停
    :param early_stopping_rounds: 早停轮数，默认None
    :param early_stopping_metric: 用于早停的评估指标名称，默认None（使用第一个指标）
        - 当eval_metric有多个时，可通过此参数指定用哪个指标进行早停判断
    :param first_metric_only: 是否只用第一个评估指标进行早停，默认True
        - 当eval_metric为列表时，True表示只用第一个指标早停，False表示监控所有指标
    :param validation_fraction: 验证集比例，默认0.2
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1
    :param verbose: 是否输出详细信息，默认False
    :param params: LightGBM原生参数字典，默认None
        - 如果传入，将覆盖其他参数设置
        - 可直接使用LightGBM原生参数名
    :param kwargs: 其他LightGBM参数

    **属性**

    :ivar feature_importances_: 特征重要性
    :ivar evals_result_: 训练过程评估结果
    :ivar best_iteration_: 最佳迭代次数
    :ivar best_score_: 最佳得分
    :ivar booster_: 底层LightGBM模型
    
    **示例**
    
    >>> # 基础使用
    >>> model = LightGBMRiskModel(num_leaves=31, learning_rate=0.1)
    >>> model.fit(X_train, y_train)
    
    >>> # 使用原生LightGBM参数
    >>> params = {'num_leaves': 31, 'learning_rate': 0.05, 'subsample': 0.8}
    >>> model = LightGBMRiskModel(params=params)
    >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        scale_pos_weight: float = 1,
        min_split_gain: float = 0,
        boosting_type: str = 'gbdt',
        objective: str = 'binary',
        eval_metric: Union[str, List[str], None] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        first_metric_only: bool = True,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM未安装，请使用 pip install lightgbm 安装"
            )

        # 保存原生params参数
        self.params = params  # 用于sklearn get_params兼容性
        self._native_params = params or {}

        # 从params中提取参数（如果提供了原生参数）
        num_leaves = self._native_params.get('num_leaves', num_leaves)
        max_depth = self._native_params.get('max_depth', max_depth)
        learning_rate = self._native_params.get('learning_rate', learning_rate)
        n_estimators = self._native_params.get('n_estimators', n_estimators)
        min_child_samples = self._native_params.get('min_child_samples', min_child_samples)
        min_child_weight = self._native_params.get('min_child_weight', min_child_weight)
        subsample = self._native_params.get('subsample', subsample)
        colsample_bytree = self._native_params.get('colsample_bytree', colsample_bytree)
        reg_alpha = self._native_params.get('reg_alpha', reg_alpha)
        reg_lambda = self._native_params.get('reg_lambda', reg_lambda)
        scale_pos_weight = self._native_params.get('scale_pos_weight', scale_pos_weight)
        min_split_gain = self._native_params.get('min_split_gain', min_split_gain)
        boosting_type = self._native_params.get('boosting_type', boosting_type)
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
        self.first_metric_only = first_metric_only

        # LightGBM特有参数
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.min_split_gain = min_split_gain
        self.boosting_type = boosting_type

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'LightGBMRiskModel':
        """训练LightGBM模型.

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
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'scale_pos_weight': self.scale_pos_weight,
            'min_split_gain': self.min_split_gain,
            'boosting_type': self.boosting_type,
            'objective': self.objective,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': -1 if not self.verbose else 1,
        }

        # 处理评估指标
        if self.eval_metric is not None:
            params['metric'] = self._convert_metrics(self.eval_metric)

        # 更新kwargs参数
        params.update(self.kwargs)
        
        # 最后更新原生params（优先级最高）
        params.update(self._native_params)

        # 创建模型
        self._model = lgb.LGBMClassifier(**params)

        # 训练
        fit_kwargs = {'eval_set': eval_set} if eval_set else {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        # 处理早停 - 适配新旧版本LightGBM
        use_callbacks = False
        early_stopping_enabled = False
        if self.early_stopping_rounds is not None and eval_set:
            callbacks = self._build_early_stopping_callbacks()
            if callbacks is not None:
                # LightGBM 4.0+ 使用 callbacks
                fit_kwargs['callbacks'] = callbacks
                use_callbacks = True
                early_stopping_enabled = True
            elif LIGHTGBM_EARLY_STOPPING_AVAILABLE:
                # 旧版本API (< 4.0, >= 2.0)
                fit_kwargs['early_stopping_rounds'] = self.early_stopping_rounds
                fit_kwargs['verbose'] = self.verbose
                early_stopping_enabled = True
            else:
                # 非常旧的版本(< 2.0)不支持早停，给出警告
                import warnings
                warnings.warn(
                    f"当前LightGBM版本({lgb.__version__})不支持early_stopping_rounds参数，"
                    f"将忽略早停设置。建议升级LightGBM: pip install -U lightgbm",
                    UserWarning
                )
        
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
                    f"当前LightGBM版本不支持early_stopping_rounds参数，"
                    f"将忽略早停设置。建议升级LightGBM: pip install -U lightgbm",
                    UserWarning
                )
                if 'early_stopping_rounds' in fit_kwargs:
                    del fit_kwargs['early_stopping_rounds']
                fit_kwargs['verbose'] = self.verbose
                self._model.fit(X_train, y_train, **fit_kwargs)
            else:
                raise

        # 保存结果
        self._best_iteration = getattr(self._model, 'best_iteration_', None)
        self._best_score = getattr(self._model, 'best_score_', None)
        self._evals_result = getattr(self._model, 'evals_result_', {})
        self._is_fitted = True

        return self

    def _build_early_stopping_callbacks(self) -> Optional[List[Any]]:
        """构建早停回调函数列表。

        适配LightGBM 4.0+版本的callbacks机制，同时兼容旧版本。

        :return: 回调函数列表，如果无法使用新机制则返回None（使用旧API）
        """
        if not LIGHTGBM_CALLBACKS_AVAILABLE:
            # LightGBM < 4.0，返回None使用旧API
            return None

        try:
            callbacks = []

            # 构建早停回调参数
            es_kwargs = {
                'stopping_rounds': self.early_stopping_rounds,
                'first_metric_only': self.first_metric_only,
                'verbose': self.verbose,
            }

            # 创建早停回调
            callbacks.append(early_stopping(**es_kwargs))

            # 添加日志监控回调（如果需要verbose）
            if self.verbose and log_evaluation is not None:
                try:
                    callbacks.append(log_evaluation(period=1))
                except (TypeError, ValueError):
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
            - 'split': 分裂次数
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

    def get_booster(self) -> 'lgb.Booster':
        """获取底层LightGBM booster对象.

        :return: LightGBM Booster对象
        """
        check_is_fitted(self, '_is_fitted')
        return self._model.booster_

    def plot_tree(self, tree_index: int = 0, **kwargs):
        """绘制树结构.

        :param tree_index: 树的索引
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, '_is_fitted')
        return lgb.plot_tree(self._model, tree_index=tree_index, **kwargs)

    def plot_importance(self, max_num_features: int = 10, **kwargs):
        """绘制特征重要性.

        :param max_num_features: 显示的最大特征数
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, '_is_fitted')
        return lgb.plot_importance(
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

        >>> model = LightGBMRiskModel(n_estimators=50)
        >>> model.fit(X, y)
        >>> leaf_indices = model.get_leaf_indices(X)
        >>> print(leaf_indices.shape)  # (n_samples, 50)
        """
        check_is_fitted(self, '_is_fitted')
        X = self._prepare_data(X)[0]
        return self._model.predict(X, pred_leaf=True)

    def _convert_metrics(self, metrics: Union[str, List[str]]) -> Union[str, List[str]]:
        """转换评估指标名称.

        :param metrics: 指标名称或列表
        :return: LightGBM格式的指标名称
        """
        metric_map = {
            'auc': 'auc',
            'logloss': 'binary_logloss',
            'error': 'binary_error',
            'rmse': 'rmse',
            'mae': 'mae',
            'map': 'map',
        }

        if isinstance(metrics, str):
            return metric_map.get(metrics.lower(), metrics)

        return [metric_map.get(m.lower(), m) for m in metrics]

    def save_model(self, path: str):
        """保存模型.

        :param path: 保存路径
        """
        check_is_fitted(self, '_is_fitted')
        self._model.booster_.save_model(path)

    def load_model(self, path: str) -> 'LightGBMRiskModel':
        """加载模型.

        :param path: 模型路径
        :return: self
        """
        self._model = lgb.LGBMClassifier()
        self._model.booster_ = lgb.Booster(model_file=path)
        self._is_fitted = True
        return self
