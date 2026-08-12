"""CatBoost风控模型.

基于CatBoost实现的风控模型，对类别特征处理更友好。

**依赖**
pip install catboost

**参考样例**
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

    **参考样例**

    >>> # 基础使用
    >>> model = CatBoostRiskModel(depth=6, learning_rate=0.1)
    >>> model.fit(X_train, y_train)

    >>> # 使用原生CatBoost参数
    >>> params = {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0}
    >>> model = CatBoostRiskModel(params=params)
    >>> model.fit(X_train, y_train)

    **引用**

    基于 CatBoost 梯度提升框架（有序提升 + 类别特征原生处理），见
    Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with
    categorical features.* NeurIPS；文档 https://catboost.ai/docs/ 。
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
        grow_policy: str = "SymmetricTree",
        objective: str = "Logloss",
        eval_metric: Union[str, List[str], None] = "AUC",
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost未安装，请使用 pip install catboost 安装")

        # 保存原生params参数
        self.params = params  # 用于sklearn get_params兼容性
        self._native_params = params or {}

        # 从params中提取参数（如果提供了原生参数）
        depth = self._native_params.get("depth", depth)
        learning_rate = self._native_params.get("learning_rate", learning_rate)
        iterations = self._native_params.get("iterations", iterations)
        # n_estimators / num_boost_round / num_trees 是 CatBoost iterations 的常见别名，
        # 统一映射到 iterations 并从 kwargs/native_params 中移除，避免与 iterations 同时
        # 传入 CatBoost 触发 "only one of the parameters ... should be initialized" 错误，
        # 同时保持与其它 boosting 模型（均接受 n_estimators）的接口一致性
        for _alias in ("n_estimators", "num_boost_round", "num_trees"):
            if _alias in self._native_params:
                iterations = self._native_params.pop(_alias)
            if _alias in kwargs:
                iterations = kwargs.pop(_alias)
        l2_leaf_reg = self._native_params.get("l2_leaf_reg", l2_leaf_reg)
        border_count = self._native_params.get("border_count", border_count)
        random_strength = self._native_params.get("random_strength", random_strength)
        bagging_temperature = self._native_params.get("bagging_temperature", bagging_temperature)
        scale_pos_weight = self._native_params.get("scale_pos_weight", scale_pos_weight)
        min_data_in_leaf = self._native_params.get("min_data_in_leaf", min_data_in_leaf)
        grow_policy = self._native_params.get("grow_policy", grow_policy)
        objective = self._native_params.get("loss_function", objective)
        random_state = self._native_params.get("random_seed", random_state)

        super().__init__(objective=objective, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, validation_fraction=validation_fraction, random_state=random_state, n_jobs=n_jobs, verbose=verbose, **kwargs)

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

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None, sample_weight: Optional[np.ndarray] = None, eval_set: Optional[List[Tuple]] = None, cat_features: Optional[List[int]] = None, **fit_params) -> "CatBoostRiskModel":
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
            cat_features = [cols.index(c) if isinstance(c, str) else int(c) for c in cat_features]

        # 准备数据（支持从X中提取target）
        X, y, sample_weight = self._prepare_data(X, y, sample_weight, extract_target=True)

        # 保存特征信息
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 创建验证集
        if eval_set is None and self.validation_fraction > 0:
            X_train, X_val, y_train, y_val, sw_train, sw_val = self._create_eval_set(X, y, sample_weight)
            eval_set = [(X_val, y_val)]
            sample_weight = sw_train
        else:
            X_train, y_train = X, y

        # 构建参数
        params = {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_leaf_reg": self.l2_leaf_reg,
            "border_count": self.border_count,
            "random_strength": self.random_strength,
            "bagging_temperature": self.bagging_temperature,
            "scale_pos_weight": self.scale_pos_weight,
            "min_data_in_leaf": self.min_data_in_leaf,
            "grow_policy": self.grow_policy,
            "loss_function": self.objective,
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "thread_count": -1,  # 使用所有CPU
        }

        # 处理评估指标
        if self.eval_metric is not None:
            # 转换评估指标格式
            eval_metric_converted = self._convert_metrics(self.eval_metric)
            params["eval_metric"] = eval_metric_converted

        # 处理早停 - CatBoost仍支持early_stopping_rounds参数
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

            # 如果指定了专门的早停指标，覆盖eval_metric
            if self.early_stopping_metric is not None:
                params["eval_metric"] = self.early_stopping_metric
            # 如果有多个评估指标且没有指定早停指标，使用第一个
            elif isinstance(self.eval_metric, list) and len(self.eval_metric) > 0:
                params["eval_metric"] = self._convert_metrics(self.eval_metric[0])

        # 更新kwargs参数
        params.update(self.kwargs)

        # 最后更新原生params（优先级最高）
        params.update(self._native_params)
        # 公共 n_jobs 是 HSCredit 的统一总预算，优先于历史 thread_count=-1
        # 和 params/kwargs 中可能造成嵌套超额并发的设置。
        params["thread_count"] = max(1, int(self.n_jobs or 1))

        # 解析自定义损失（BaseLoss 实例 -> CatBoost 可用的损失对象）
        resolved_loss = self._resolve_catboost_loss(params.get("loss_function"))
        params["loss_function"] = resolved_loss

        # CatBoost 自定义损失（非内置字符串）不支持 scale_pos_weight，需移除以避免报错
        if not isinstance(resolved_loss, str) and "scale_pos_weight" in params:
            params.pop("scale_pos_weight", None)

        # 创建模型
        self._model = cb.CatBoostClassifier(**params)

        # 准备训练参数
        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if cat_features is not None:
            fit_kwargs["cat_features"] = cat_features

        # 训练
        self._model.fit(X_train, y_train, **fit_kwargs)

        # 保存结果
        self._best_iteration = self._model.get_best_iteration()
        self._best_score = self._model.get_best_score()
        self._evals_result = self._model.get_evals_result()
        self._is_fitted = True

        return self

    @staticmethod
    def _resolve_catboost_loss(loss_function):
        """将自定义损失对象解析为 CatBoost 可用的损失对象.

        当传入 :class:`~hscredit.core.models.losses.BaseLoss` 实例时，通过
        :class:`~hscredit.core.models.losses.CatBoostLossAdapter` 转换为带有
        ``calc_ders_range`` 接口的 CatBoost 自定义损失对象；其他对象原样返回。
        """
        try:
            from ..losses.base import BaseLoss
            from ..losses.adapters import CatBoostLossAdapter
        except Exception:
            return loss_function

        if isinstance(loss_function, BaseLoss):
            return CatBoostLossAdapter(loss_function).objective()
        return loss_function

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签.

        基于 predict_proba 取阈值，确保自定义损失（原始分数输出）下也能返回正确类别。
        """
        check_is_fitted(self, "_is_fitted")
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.asarray(self.classes_)[indices]

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率.

        当使用自定义损失函数（loss_function 为可调用对象）时，CatBoost 返回的是
        未经过链接函数转换的原始分数（raw margin，一维数组），此处自动应用
        sigmoid 转换为概率并补齐为二维 (n_samples, 2) 输出，与内置目标保持一致。
        """
        check_is_fitted(self, "_is_fitted")
        X = self._prepare_data(X)[0]
        proba = np.asarray(self._model.predict_proba(X))

        # 自定义损失返回一维原始分数，应用 sigmoid 并补齐为两列概率
        if proba.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-proba))
            proba = np.column_stack([1.0 - p1, p1])

        return proba

    def get_feature_importances(self, importance_type: str = "PredictionValuesChange") -> pd.Series:
        """获取特征重要性.

        :param importance_type: 重要性类型，可选:
            - 'PredictionValuesChange': 预测值变化 (默认)
            - 'LossFunctionChange': 损失函数变化
            - 'FeatureImportance': 分裂次数
        :return: 特征重要性Series
        """
        check_is_fitted(self, "_is_fitted")

        importances = self._model.get_feature_importance(type=importance_type)

        # 创建Series
        importance_series = pd.Series(importances, index=self.feature_names_in_, name="importance").sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性属性 (兼容sklearn风格).

        直接在包装类上暴露重要性，兼容sklearn RFE/SFS等组件的 importance_getter。
        """
        check_is_fitted(self, "_is_fitted")
        if self._feature_importances is None:
            self._feature_importances = self.get_feature_importances()
        return self._feature_importances.values

    def plot_tree(self, tree_index: int = 0, **kwargs):
        """绘制树结构.

        :param tree_index: 树的索引
        :param kwargs: 其他绘图参数
        """
        check_is_fitted(self, "_is_fitted")
        return self._model.plot_tree(tree_idx=tree_index, **kwargs)

    def get_leaf_indices(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """获取叶子节点索引.

        返回每棵树上的叶子节点索引，用于GBDT+LR等场景。

        :param X: 特征矩阵
        :return: 叶子节点索引，形状 (n_samples, n_trees)

        **参考样例**

        >>> model = CatBoostRiskModel(iterations=50)
        >>> model.fit(X, y)
        >>> leaf_indices = model.get_leaf_indices(X)
        >>> print(leaf_indices.shape)
        """
        check_is_fitted(self, "_is_fitted")
        X = self._prepare_data(X)[0]
        return self._model.calc_leaf_indexes(X)

    def save_model(self, path: str):
        """保存底层CatBoost模型（原生格式）.

        :param path: 保存路径（.cbm/.json 格式）
        """
        check_is_fitted(self, "_is_fitted")
        self._model.save_model(path)

    def load_model(self, path: str) -> "CatBoostRiskModel":
        """加载底层CatBoost模型（原生格式）.

        :param path: 模型路径
        :return: self
        """
        self._model = cb.CatBoostClassifier()
        self._model.load_model(path)
        self._is_fitted = True
        self.classes_ = getattr(self, "classes_", np.array([0, 1]))
        if not hasattr(self, "feature_names_in_"):
            n_feat = self._model.feature_count_ if hasattr(self._model, "feature_count_") else 0
            self.feature_names_in_ = [f"feature_{i}" for i in range(n_feat)]
            self.n_features_in_ = n_feat
        return self

    def _convert_metrics(self, metrics: Union[str, List[str]]) -> Union[str, List[str]]:
        """转换评估指标名称.

        :param metrics: 指标名称或列表
        :return: CatBoost格式的指标名称
        """
        metric_map = {
            "auc": "AUC",
            "logloss": "Logloss",
            "error": "Accuracy",
            "rmse": "RMSE",
            "mae": "MAE",
            "mse": "MSE",
            "msle": "MSLE",
            "poisson": "Poisson",
            "quantile": "Quantile",
            "mape": "MAPE",
            "r2": "R2",
            "ndcg": "NDCG",
            "map": "MAP",
            "recall": "Recall",
            "precision": "Precision",
            "f1": "F1",
            "balanced_accuracy": "BalancedAccuracy",
            "balanced_error_rate": "BalancedErrorRate",
            "kappa": "Kappa",
            "wkappa": "WKappa",
            "total_f1": "TotalF1",
            "mcc": "MCC",
            "brier_score": "BrierScore",
            "hinge_loss": "HingeLoss",
            "hamming_loss": "HammingLoss",
            "zero_one_loss": "ZeroOneLoss",
            "kappa:use_weights": "Kappa:use_weights",
            "wkappa:use_weights": "WKappa:use_weights",
        }

        if isinstance(metrics, str):
            return metric_map.get(metrics.lower(), metrics)

        return [metric_map.get(m.lower(), m) for m in metrics]
