"""net 模型的分类、回归、字段和表示学习公共契约。"""

import inspect

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from ....exceptions import ValidationError
from ..base import BaseRiskModel
from .._contracts import extract_target, validate_labels, validate_sample_weight


class BaseNetModel(BaseRiskModel):
    """复用风控模型生命周期，并按 objective 选择分类或回归语义。"""

    _preserve_dataframe = True
    _supports_native_feature_importance = False
    _parameter_dicts = ("loss_params", "optimizer_params", "num_embedding_params", "finetune_params")

    def get_params(self, deep=True):
        """为配置字典暴露 sklearn 风格的双下划线参数，供 tune 与 GridSearchCV 使用。"""
        params = super().get_params(deep=deep)
        if deep:
            for name in self._parameter_dicts:
                if isinstance(params.get(name), dict):
                    params.update((f"{name}__{key}", value) for key, value in params[name].items())
        return params

    def set_params(self, **params):
        """原生可扩展模型保留透传参数，其余模型严格校验公开参数名。"""
        params = dict(params)
        updates = {}
        for key in list(params):
            name, separator, child = key.partition("__")
            if separator and name in self._parameter_dicts and hasattr(self, name):
                updates.setdefault(name, {})[child] = params.pop(key)
        for name, update in updates.items():
            params[name] = {**(params.get(name, getattr(self, name)) or {}), **update}
        extensible = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(self.__init__).parameters.values()
        )
        if not extensible:
            unknown = {key.split("__", 1)[0] for key in params} - set(self.get_params(deep=False))
            if unknown:
                raise ValidationError(f"未知模型参数: {sorted(unknown)}")
        return super().set_params(**params)

    @property
    def _estimator_type(self):
        return "regressor" if self.objective == "regression" else "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = self._estimator_type
        tags.input_tags.allow_nan = True
        if self._estimator_type == "regressor":
            from sklearn.utils import RegressorTags

            tags.classifier_tags = None
            tags.regressor_tags = RegressorTags()
        return tags

    def _validate_task(self):
        if self.objective not in ("binary", "regression"):
            raise ValidationError("objective 必须为 binary（二分类）或 regression（回归）")
        self.objective_ = self.objective
        for name in ("classes_", "scorecard_", "score_transformer_", "bad_rate_", "base_odds_"):
            self.__dict__.pop(name, None)

    def _validate_y(self, y, *, training=False):
        if self.objective_ == "binary":
            values = validate_labels(y, require_both=training)
            if training:
                self.classes_ = np.array([0, 1])
            return np.asarray(values, dtype=np.float64)
        try:
            values = np.asarray(y, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValidationError("回归标签必须是有限数值") from exc
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValidationError("回归标签必须是非空、有限的一维数值数组")
        return values

    def _require_fitted(self):
        super()._require_fitted()
        if self.objective != self.objective_:
            raise ValidationError("objective 已改变，请重新训练模型")

    def _require_binary(self):
        self._require_fitted()
        if self.objective_ != "binary":
            raise ValidationError("回归模型不提供分类概率或风险评分，请使用 predict 获取回归结果")

    def transform(self, X):
        """提取每行的二维 embedding，等价于 get_embeddings(X)。"""
        return self.get_embeddings(X)

    def _require_representation_fitted(self):
        """仅预训练的编码器可输出表示，但不能冒充已拟合的监督预测器。"""
        if getattr(self, "_pretrain_only_", False) and hasattr(self, "_pretrained_state_"):
            return
        self._require_fitted()

    def get_native_model(self):
        """返回已预训练或已监督拟合的底层模型；仅预训练时预测接口仍不可用。"""
        self._require_representation_fitted()
        return self._model

    def save(self, path, engine="auto", **kwargs):
        """保存监督模型或仅完成 pretrain 的编码器及其预处理状态。"""
        if getattr(self, "_pretrain_only_", False):
            self._require_representation_fitted()
            if str(path).lower().endswith(".json"):
                self._save_json(path)
            from .._lifecycle import atomic_save_pickle

            return atomic_save_pickle(self, path, engine="cloudpickle" if engine == "auto" else engine, **kwargs)
        return super().save(path, engine=engine, **kwargs)

    def fit_transform(self, X, y=None, **fit_params):
        """拟合后提取输入行的表示；监督训练表示不等同于折外特征。"""
        return self.fit(X, y, **fit_params).transform(X)

    def score(self, X, y, sample_weight=None):
        """分类返回准确率，回归返回 R²，支持 sklearn 交叉验证。"""
        self._require_fitted()
        metric = r2_score if self.objective_ == "regression" else accuracy_score
        return metric(y, self.predict(X), sample_weight=sample_weight)

    def tune(self, X, y=None, **kwargs):
        """搜索网络配置并返回最佳模型；回归默认使用 RMSE。

        :param X: 特征或包含 target 的 DataFrame。
        :param y: 一维目标，None 时按 target 提取。
        :param kwargs: 与 BaseRiskModel.tune 相同；支持 loss、search_space、fixed_params、metric、cv 等。
            优化器参数可搜索 optimizer_params__momentum 等键；loss_params 同理。
            最终 full_data 重训关闭内部早停，使用搜索各折最优轮数的中位数。
        :return: 已拟合的最佳模型，其 tuner 属性保留搜索历史。

        >>> best = model.tune(X_train, y_train, n_trials=3, metric='rmse')
        """
        kwargs.setdefault("metric", "rmse" if self.objective == "regression" else "ks")
        return super().tune(X, y, **kwargs)

    def predict_score(self, X):
        self._require_binary()
        return super().predict_score(X)

    def evaluate(self, X, y=None, sample_weight=None, metrics=None, positive_class=None, score_direction="higher_risk"):
        """分类沿用统一风控指标；回归输出中文指标名。

        :param X: 二维特征或含 target 的 DataFrame，按训练字段对齐。
        :param y: 真实标签；None 时从 target 提取，显式 y 优先。
        :param sample_weight: 非负有限评估权重；None 表示等权，不继承训练权重。
        :param metrics: 分类沿用统一指标；回归接受 mse、rmse、mae、r2 或其列表。
        :param positive_class: 分类正类，默认 classes_[1]；回归不使用。
        :param score_direction: 分类 AUC/Gini 方向，默认 higher_risk；回归不使用。
        :return: 指标名称到数值的字典。

        >>> values = model.evaluate(X_test, y_test, metrics=['rmse', 'mae'])
        """
        self._require_fitted()
        if self.objective_ == "binary":
            return super().evaluate(X, y, sample_weight, metrics, positive_class, score_direction)
        X, y = extract_target(X, y, self.target)
        y = self._validate_y(y)
        weights = validate_sample_weight(sample_weight, len(y))
        prediction = self.predict(X)
        requested = (
            ["rmse", "mae", "r2"] if metrics is None else [metrics] if isinstance(metrics, str) else list(metrics)
        )
        result = {}
        for metric in requested:
            if metric == "mse":
                result["均方误差"] = float(mean_squared_error(y, prediction, sample_weight=weights))
            elif metric == "rmse":
                result["均方根误差"] = float(np.sqrt(mean_squared_error(y, prediction, sample_weight=weights)))
            elif metric == "mae":
                result["平均绝对误差"] = float(mean_absolute_error(y, prediction, sample_weight=weights))
            elif metric == "r2":
                result["决定系数"] = float(r2_score(y, prediction, sample_weight=weights))
            else:
                raise ValidationError(f"不支持的回归评估指标: {metric!r}，请使用 mse、rmse、mae、r2")
        return result

    def _record_net_evaluation(self, eval_set, metrics=None):
        rows = []
        for name, (X, y) in zip(self.eval_names_, eval_set or []):
            values = self.evaluate(X, y, metrics=metrics)
            rows.extend({"数据集": name, "指标": key, "数值": value} for key, value in values.items())
        self.eval_results_ = pd.DataFrame(rows, columns=["数据集", "指标", "数值"])

    def get_feature_importances(self, importance_type="gain"):
        """这些模型没有树增益重要性；请使用 sklearn.inspection.permutation_importance。"""
        self._require_fitted()
        raise ValidationError("net 模型不提供树增益特征重要性，请使用置换重要性分析")

    def get_model_info(self):
        """返回当前任务与已拟合模型的中文元信息。"""
        self._require_representation_fitted()
        return {
            "模型类型": type(self).__name__,
            "任务类型": (
                "特征重建预训练"
                if getattr(self, "_pretrain_only_", False)
                else "二分类" if self.objective_ == "binary" else "回归"
            ),
            "特征数量": self.n_features_in_,
            "嵌入维度": getattr(self, "embedding_dim_", None),
            "最佳迭代": self.best_iteration_,
            "参数": self.get_params(),
        }

    def _save_json(self, path):
        raise ValidationError("net 模型请保存为 .pkl 或 .joblib，以同时保留网络、预处理和训练上下文")
