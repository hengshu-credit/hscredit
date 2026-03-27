"""风控模型基类.

提供统一的风控模型接口，支持:
- 统一fit/predict接口
- 特征重要性获取
- 模型评估报告
- 自定义loss和评估目标
- Optuna超参数调优

设计原则:
1. 所有风控模型继承BaseRiskModel
2. 统一的API风格，参考sklearn和scorecardpipeline
3. 支持自定义损失函数和评估指标
4. 内置风控常用评估指标(KS、AUC、Gini、PSI等)
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split

from ..metrics.classification import KS, AUC, Gini
from ..metrics.stability import PSI
from ..metrics.finance import lift_at, lift_monotonicity_check


def _lift_score(y_true, y_proba, top_ratio=0.1):
    """计算Lift值（内部辅助函数）."""
    n = len(y_true)
    n_top = int(n * top_ratio)
    
    # 按概率降序排序
    sorted_indices = np.argsort(-y_proba)
    y_sorted = y_true[sorted_indices]
    
    # 计算整体坏样本率和top_ratio的坏样本率
    overall_bad_rate = y_true.mean()
    top_bad_rate = y_sorted[:n_top].mean()
    
    if overall_bad_rate == 0:
        return 1.0
    
    return top_bad_rate / overall_bad_rate


class BaseRiskModel(BaseEstimator, ClassifierMixin, ABC):
    """风控模型基类.

    所有风控模型的抽象基类，定义统一接口。
    继承sklearn的BaseEstimator和ClassifierMixin。
    支持scorecardpipeline风格的fit（可在init中指定target列）。

    **参数**

    :param objective: 目标函数，可选:
        - 'binary': 二分类(默认)
        - 'binary:logistic': 二分类逻辑回归
        - 'regression': 回归
        - 自定义可调用对象
    :param eval_metric: 评估指标，可选列表或单个指标:
        - 'auc': AUC
        - 'ks': KS统计量
        - 'gini': Gini系数
        - 'lift': Lift值
        - 'logloss': 对数损失
        - 自定义可调用对象
    :param target: 目标列名，默认None
        - 如果指定，fit时只需传入X，会自动从X中提取target列作为y
        - 支持scorecardpipeline风格的有监督fit
    :param early_stopping_rounds: 早停轮数，默认None
    :param validation_fraction: 验证集比例，默认0.2
    :param random_state: 随机种子，默认None
    :param n_jobs: 并行任务数，默认-1
    :param verbose: 是否输出详细信息，默认False
    :param kwargs: 模型特定参数

    **属性**

    :ivar classes_: 类别标签
    :ivar n_features_in_: 特征数量
    :ivar feature_names_in_: 特征名称
    :ivar feature_importances_: 特征重要性
    :ivar evals_result_: 训练过程评估结果
    :ivar best_iteration_: 最佳迭代次数
    :ivar best_score_: 最佳得分
    """

    # 支持的评估指标
    SUPPORTED_METRICS = [
        'auc', 'ks', 'gini', 'lift',
        'lift@1%', 'lift@3%', 'lift@5%', 'lift@10%',
        'logloss', 'accuracy', 'precision', 'recall', 'f1',
    ]
    # 默认评估指标（evaluate() 不传 metrics 时使用）
    DEFAULT_METRICS = ['auc', 'ks', 'gini', 'lift@1%', 'lift@3%', 'lift@5%', 'lift@10%']

    def __init__(
        self,
        objective: Union[str, Callable] = 'binary',
        eval_metric: Union[str, List[str], Callable, None] = None,
        target: Optional[str] = None,
        early_stopping_rounds: Optional[int] = None,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.objective = objective
        self.eval_metric = eval_metric
        self.target = target
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.verbose = verbose
        self.kwargs = kwargs

        # 内部属性
        self._model = None
        self._evals_result = {}
        self._best_iteration = None
        self._best_score = None
        self._feature_importances = None
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        **fit_params
    ) -> 'BaseRiskModel':
        """训练模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: 在__init__中指定target，然后fit(X)

        :param X: 特征矩阵，支持numpy数组或pandas DataFrame
        :param y: 目标变量，可选。如果未提供且init中指定了target，则从X中提取
        :param sample_weight: 样本权重，可选
        :param eval_set: 验证集列表 [(X_val1, y_val1), ...]，可选
        :param fit_params: 其他fit参数
        :return: self
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签.

        :param X: 特征矩阵
        :return: 预测类别
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率.

        :param X: 特征矩阵
        :return: 预测概率，形状 (n_samples, n_classes)
        """
        pass

    def predict_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测风险评分 (概率转换).

        :param X: 特征矩阵
        :return: 风险评分 (0-1000)
        """
        proba = self.predict_proba(X)
        # 转换为评分 (0-1000)
        scores = (1 - proba[:, 1]) * 1000
        return scores

    @abstractmethod
    def get_feature_importances(self, importance_type: str = 'gain') -> pd.Series:
        """获取特征重要性.

        :param importance_type: 重要性类型，可选:
            - 'gain': 增益 (默认)
            - 'split': 分裂次数
            - 'weight': 权重
            - 'cover': 覆盖度
        :return: 特征重要性Series
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息.

        :return: 包含模型信息的字典
        """
        check_is_fitted(self, '_is_fitted')

        info = {
            'model_type': self.__class__.__name__,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'n_features': self.n_features_in_,
            'n_classes': len(self.classes_),
            'best_iteration': self._best_iteration,
            'best_score': self._best_score,
            'params': self.get_params(),
        }

        # 添加特征重要性统计
        if self._feature_importances is not None:
            importances = self._feature_importances
            info['feature_importance_stats'] = {
                'top_feature': importances.index[0] if len(importances) > 0 else None,
                'top_importance': importances.iloc[0] if len(importances) > 0 else None,
                'mean_importance': importances.mean(),
                'std_importance': importances.std(),
            }

        return info

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估模型性能.

        :param X: 特征矩阵
        :param y: 真实标签
        :param sample_weight: 样本权重
        :param metrics: 评估指标列表，默认全部
        :return: 评估结果字典
        """
        check_is_fitted(self, '_is_fitted')

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        if metrics is None:
            metrics = self.DEFAULT_METRICS

        results = {}

        for metric in metrics:
            metric_lower = metric.lower()
            try:
                if metric_lower == 'auc':
                    results['AUC'] = AUC(y, y_proba)
                elif metric_lower == 'ks':
                    results['KS'] = KS(y, y_proba)
                elif metric_lower == 'gini':
                    results['Gini'] = Gini(y, y_proba)
                elif metric_lower == 'lift':
                    results['Lift@10%'] = _lift_score(y, y_proba, top_ratio=0.1)
                elif metric_lower in ('lift@1%', 'lift_1'):
                    results['LIFT@1%'] = _lift_score(y, y_proba, top_ratio=0.01)
                elif metric_lower in ('lift@3%', 'lift_3'):
                    results['LIFT@3%'] = _lift_score(y, y_proba, top_ratio=0.03)
                elif metric_lower in ('lift@5%', 'lift_5'):
                    results['LIFT@5%'] = _lift_score(y, y_proba, top_ratio=0.05)
                elif metric_lower in ('lift@10%', 'lift_10'):
                    results['LIFT@10%'] = _lift_score(y, y_proba, top_ratio=0.10)
                elif metric_lower == 'logloss':
                    from sklearn.metrics import log_loss
                    results['LogLoss'] = log_loss(y, y_proba, sample_weight=sample_weight)
                elif metric_lower == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    results['Accuracy'] = accuracy_score(y, y_pred, sample_weight=sample_weight)
                elif metric_lower == 'precision':
                    from sklearn.metrics import precision_score
                    results['Precision'] = precision_score(y, y_pred, sample_weight=sample_weight)
                elif metric_lower == 'recall':
                    from sklearn.metrics import recall_score
                    results['Recall'] = recall_score(y, y_pred, sample_weight=sample_weight)
                elif metric_lower == 'f1':
                    from sklearn.metrics import f1_score
                    results['F1'] = f1_score(y, y_pred, sample_weight=sample_weight)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"计算指标 {metric} 时出错: {e}")
                continue

        # 头部单调性检验（始终计算，不依赖 metrics 参数）
        try:
            mono = lift_monotonicity_check(y, y_proba, n_bins=10, direction='both')
            results['头部LIFT单调'] = mono['head_monotonic']
            results['头部违反单调比例'] = mono['head_violation_ratio']
            results['尾部LIFT单调'] = mono['tail_monotonic']
        except Exception:
            pass

        return results

    def generate_report(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'ModelReport':
        """生成模型评估报告.

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :param X_test: 测试集特征，可选
        :param y_test: 测试集标签，可选
        :param feature_names: 特征名称列表，可选
        :return: ModelReport对象
        """
        from .report import ModelReport

        return ModelReport(
            model=self,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names
        )

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        extract_target: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """准备数据.

        支持从X中提取target列（scorecardpipeline风格）。

        :param X: 特征矩阵
        :param y: 目标变量
        :param sample_weight: 样本权重
        :param extract_target: 是否从X中提取target列
        :return: 处理后的X, y, sample_weight
        """
        # 处理DataFrame
        if isinstance(X, pd.DataFrame):
            if not hasattr(self, 'feature_names_in_'):
                self.feature_names_in_ = X.columns.tolist()

            # scorecardpipeline风格：从X中提取target列
            if extract_target and self.target is not None and self.target in X.columns:
                y = X[self.target].values
                X = X.drop(columns=[self.target])
                # 更新特征名列表
                self.feature_names_in_ = X.columns.tolist()

            X = X.values
        else:
            if not hasattr(self, 'feature_names_in_'):
                self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]

        # 处理y
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values

        # 处理样本权重
        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                sample_weight = sample_weight.values

        return X, y, sample_weight

    def _create_eval_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """创建验证集.

        :param X: 特征矩阵
        :param y: 目标变量
        :param sample_weight: 样本权重
        :return: X_train, X_val, y_train, y_val, sw_train, sw_val
        """
        if self.validation_fraction > 0 and self.validation_fraction < 1:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y
                )
                return X_train, X_val, y_train, y_val, sw_train, sw_val
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y
                )
                return X_train, X_val, y_train, y_val, None, None
        else:
            return X, None, y, None, sample_weight, None

    def get_native_model(self) -> Any:
        """获取底层原生模型对象.

        用于需要访问底层模型特定功能的场景，如:
        - 获取叶子节点索引
        - 绘制树结构
        - 访问底层模型特有的方法

        :return: 底层模型对象（如xgboost.Booster、lgb.Booster等）

        **示例**

        >>> model = XGBoostRiskModel()
        >>> model.fit(X, y)
        >>> native_model = model.get_native_model()
        >>> # 使用底层模型方法
        >>> leaf_indices = native_model.apply(X)
        """
        check_is_fitted(self, '_is_fitted')
        return self._model

    def _get_metric_func(self, metric: str) -> Callable:
        """获取评估指标函数.

        :param metric: 指标名称
        :return: 评估函数
        """
        metric_map = {
            'auc': lambda y, p: AUC(y, p),
            'ks': lambda y, p: KS(y, p),
            'gini': lambda y, p: Gini(y, p),
        }
        return metric_map.get(metric.lower())

    def __sklearn_is_fitted__(self):
        """用于sklearn的check_is_fitted检查."""
        return hasattr(self, '_is_fitted') and self._is_fitted

    def plot_feature_importance(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        top_n: int = 20,
        importance_type: str = 'gain',
        method: str = 'traditional',
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> 'matplotlib.figure.Figure':
        """绘制特征重要性图.

        支持传统特征重要性和SHAP值两种方法。

        :param X: 特征矩阵，SHAP方法必需
        :param y: 目标变量，可选
        :param top_n: 显示前N个特征，默认20
        :param importance_type: 重要性类型（传统方法），默认'gain'
        :param method: 计算方法，默认'traditional'
            - 'traditional': 传统特征重要性
            - 'shap': SHAP值重要性
            - 'combined': 两者对比
        :param figsize: 图表大小，默认(10, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :param kwargs: 其他绘图参数
        :return: matplotlib Figure对象

        **示例**

        >>> # 传统特征重要性
        >>> fig = model.plot_feature_importance(top_n=15)
        >>> fig.savefig('importance.png')

        >>> # SHAP特征重要性
        >>> fig = model.plot_feature_importance(X_test, method='shap', top_n=15)

        >>> # 组合对比图
        >>> fig = model.plot_feature_importance(X_test, method='combined', top_n=10)
        """
        from .interpretability import (
            plot_feature_importance,
            plot_shap_importance,
            plot_importance_comparison
        )

        if method == 'traditional':
            return plot_feature_importance(
                self, X=X, top_n=top_n,
                importance_type=importance_type,
                figsize=figsize, title=title,
                show=show, **kwargs
            )
        elif method == 'shap':
            if X is None:
                raise ValueError("SHAP方法需要提供X参数")
            return plot_shap_importance(
                self, X, top_n=top_n,
                figsize=figsize, title=title,
                show=show
            )
        elif method == 'combined':
            if X is None:
                raise ValueError("组合方法需要提供X参数")
            return plot_importance_comparison(
                self, X, top_n=top_n,
                importance_type=importance_type,
                figsize=figsize, title=title,
                show=show
            )
        else:
            raise ValueError(f"不支持的method: {method}，可选: 'traditional', 'shap', 'combined'")

    def get_shap_explainer(self, **kwargs) -> 'ModelExplainer':
        """获取SHAP解释器.

        :param kwargs: ModelExplainer的初始化参数
        :return: ModelExplainer对象

        **示例**

        >>> explainer = model.get_shap_explainer()
        >>> shap_values = explainer.compute_shap_values(X_test)
        >>> explainer.plot_shap_summary(X_test)
        """
        from .interpretability import ModelExplainer
        return ModelExplainer(self, **kwargs)
