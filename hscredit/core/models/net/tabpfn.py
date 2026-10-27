"""官方 TabPFN 分类器/回归器的统一模型适配。"""

import inspect
from numbers import Integral

import numpy as np

from ....exceptions import DependencyError, ValidationError
from .._contracts import positive_probability
from .._lifecycle import record_training
from .base import BaseNetModel


def _load_tabpfn():
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:
        raise DependencyError('TabPFN 需要 tabpfn>=2.0.9，请安装 pip install "hscredit[net]"') from exc
    return TabPFNClassifier, TabPFNRegressor


class TabPFN(BaseNetModel):
    """使用预训练 TabPFN 对表格进行二分类、回归和表示提取。

    **参数**

    :param objective: binary（标签 0/1）或 regression。
    :param n_estimators: 原生集成成员数量，默认 4。
    :param device: auto、cpu 或 cuda[:设备编号]，由官方实现选择。
    :param model_path: 官方模型权重路径；auto 使用安装版本的默认权重，首次使用可能下载。
    :param target: 可选目标列，支持 fit(df)；显式 y 优先且目标列始终剔除。
    :param eval_metric: 训练后评估集的指标，评估数据不加入训练上下文。
    :param random_state: 随机种子，默认 0。
    :param n_jobs: 原生预处理并行度，默认 1；新版后端自动映射到 n_preprocessing_jobs。
    :param scorecard_params: 仅用于二分类的评分卡配置。
    :param history_policy: 训练历史策略，默认 summary。
    :param max_history: 训练历史上限，默认 20。
    :param weight_type: cost 或 frequency。
    :param kwargs: 透传原生构造器参数，如 categorical_features_indices、fit_mode、inference_config。

    **属性**

    :ivar classes_: 二分类时为 [0, 1]，回归时不存在。
    :ivar embedding_dim_: 首次提取 embedding 后记录其维度。

    **参考样例**

    >>> model = TabPFN(target='FPD', n_estimators=4)
    >>> model.fit(train_df)
    >>> probability = model.predict_proba(test_df)
    >>> embedding = model.get_embeddings(test_df)

    使用官方本地推理，无需上传数据。权重版本、可用规模及许可由所选官方权重决定。
    默认平均多个集成成员的表示；aggregate=None 保留成员轴。
    transform 始终提取传入行的 test-token 表示，不返回上下文训练行的缓存表示。
    训练行表示仍属于监督训练产物；下游训练及评估应使用折外划分防止泄漏。
    """

    def __init__(
        self,
        objective="binary",
        n_estimators=4,
        device="auto",
        model_path="auto",
        target=None,
        eval_metric=None,
        random_state=0,
        n_jobs=1,
        scorecard_params=None,
        history_policy="summary",
        max_history=20,
        weight_type="cost",
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            target=target,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
            scorecard_params=scorecard_params,
            history_policy=history_policy,
            max_history=max_history,
            weight_type=weight_type,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.device = device
        self.model_path = model_path

    @record_training
    def fit(self, X, y=None, sample_weight=None, eval_set=None, *, eval_metric=None):
        """拟合官方模型；不支持 sample_weight 的后端会明确拒绝该参数。

        :param X: 二维特征或 DataFrame；目标列始终从输入中剔除。
        :param y: 一维 0/1 标签或连续目标；None 时按 target 提取，显式 y 优先。
        :param sample_weight: 样本权重，仅在所安装的原生后端明确支持时传入。
        :param eval_set: 单个 (X, y)、列表、具名字典或含 target 的 DataFrame；仅训练后评估。
        :param eval_metric: 覆盖构造器的最终评估指标，不参与训练上下文拟合。
        :return: 已拟合模型。

        >>> model.fit(train_df, eval_set={'验证': valid_df})
        >>> values = model.get_eval_results()
        """
        self._validate_task()
        training_options = {
            "loss",
            "loss_params",
            "optimizer",
            "optimizer_params",
            "learning_rate",
            "weight_decay",
            "max_epochs",
        }
        unsupported = training_options.intersection(self.kwargs)
        if unsupported:
            raise ValidationError(
                f"当前 TabPFN 封装使用预训练推理，不支持梯度训练参数 {sorted(unsupported)}；可搜索原生推理参数"
            )
        self.__dict__.pop("embedding_dim_", None)
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValidationError("n_estimators 必须为正整数")
        X, y, weight = self._prepare_data(X, y, sample_weight, extract_target=True, training=True)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValidationError("训练特征必须至少有一行和一列")
        y = self._validate_y(y, training=True)
        sets = self._prepare_eval_set(eval_set)
        for _, labels in sets or []:
            self._validate_y(labels)
        classifier, regressor = _load_tabpfn()
        backend = classifier if self.objective_ == "binary" else regressor
        if not callable(getattr(backend, "get_embeddings", None)):
            raise DependencyError("当前 tabpfn 不支持 get_embeddings，请安装 tabpfn>=2.0.9")
        options = {}
        if weight is not None:
            if "sample_weight" not in inspect.signature(backend.fit).parameters:
                raise ValidationError("当前 TabPFN 后端不支持 sample_weight，请移除权重参数")
            options["sample_weight"] = weight
        self.native_params_ = {
            **self.kwargs,
            "n_estimators": self.n_estimators,
            "device": self.device,
            "model_path": self.model_path,
            "random_state": self.random_state,
        }
        jobs_key = (
            "n_preprocessing_jobs" if "n_preprocessing_jobs" in inspect.signature(backend).parameters else "n_jobs"
        )
        self.native_params_.setdefault(jobs_key, self.n_jobs)
        self._model = backend(**self.native_params_)
        self._model.fit(X, y, **options)
        self._is_fitted = True
        if self.objective_ == "binary":
            self.classes_ = np.asarray(self._model.classes_)
            self._fit_probability_scorecard(X, y, sample_weight=weight)
        self._record_net_evaluation(sets, metrics=self.eval_metric if eval_metric is None else eval_metric)
        return self

    def _inference_data(self, X):
        self._require_fitted()
        X, _, _ = self._prepare_data(X, extract_target=True)
        return X

    def predict_proba(self, X):
        """返回两列概率，列顺序由 classes_ 标记。"""
        self._require_binary()
        values = np.asarray(self._model.predict_proba(self._inference_data(X)), dtype=float)
        positive_probability(values, self.classes_, 1)
        return values

    def predict(self, X):
        """分类返回类别；回归返回与原始目标相同单位的预测值。"""
        self._require_fitted()
        if self.objective_ == "binary":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        X = self._inference_data(X)
        values = np.asarray(self._model.predict(X), dtype=float)
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValidationError("TabPFN 回归结果必须是与输入等长的有限一维数组")
        return values

    def get_embeddings(self, X, aggregate="mean"):
        """返回 (样本数, 嵌入维度)；aggregate=None 返回 (成员数, 样本数, 嵌入维度)。"""
        if aggregate not in ("mean", None):
            raise ValidationError("aggregate 必须为 mean 或 None")
        X = self._inference_data(X)
        values = self._model.get_embeddings(X, data_source="test")
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        # TabPFN 2.0.x 对单行调用 squeeze，会压掉样本轴。
        if values.ndim == 1 and len(X) == 1:
            values = values[np.newaxis, np.newaxis, :]
        elif values.ndim == 2:
            values = values[:, np.newaxis, :] if len(X) == 1 else values[np.newaxis, ...]
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1] != len(X)
            or values.shape[2] == 0
            or not np.isfinite(values).all()
        ):
            raise ValidationError("TabPFN embedding 必须为有限的 (成员数, 样本数, 嵌入维度) 数组")
        self.embedding_dim_ = values.shape[2]
        return values.mean(axis=0) if aggregate == "mean" else values
