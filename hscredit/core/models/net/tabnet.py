"""具有稀疏注意力的表格 TabNet，复用 net 预训练、训练和报告契约。"""

from numbers import Integral, Real

import numpy as np
import pandas as pd

from ....exceptions import DependencyError, ValidationError
from .._lifecycle import record_training
from ._torch_model import TorchTabularModel


class TabNet(TorchTabularModel):
    """官方 pytorch-tabnet 网络的二分类与回归封装。

    **参数**

    :param objective: binary（二分类 0/1）或 regression，默认 binary。
    :param n_d: 各决策步及输出 embedding 的宽度，默认 8。
    :param n_a: 注意力表示宽度，默认 8。
    :param n_steps: 决策步数量，默认 3。
    :param gamma: 注意力先验的更新系数，至少 1，默认 1.3。
    :param n_independent: 每步独立 GLU 层数，默认 2；与 n_shared 不能同时为 0。
    :param n_shared: 共享 GLU 层数，默认 2。
    :param cat_embedding_dim: 每个类别字段的 embedding 宽度，默认 1。
    :param categorical_features: 类别字段名或位置列表；None 自动识别非数值/布尔字段。
    :param virtual_batch_size: Ghost BatchNorm 的虚拟批次大小，至少 2，默认 128。
    :param momentum: BatchNorm 动量，(0, 1) 内，默认 0.02。
    :param mask_type: sparsemax 或 entmax，默认 sparsemax。
    :param lambda_sparse: 监督训练的注意力熵正则系数，默认 0.001；0 禁用。
    :param epsilon: 注意力对数稳定项，(0, 1) 内，默认 1e-15。
    :param learning_rate: 优化器学习率，默认 0.02。
    :param weight_decay: 优化器权重衰减，默认 0.00001。
    :param max_epochs: 最大训练轮数，默认 100。
    :param batch_size: 训练和推理批量，至少 2，默认 256；末尾单行并入前一批。
    :param early_stopping_rounds: 按第一份验证集的任务损失早停，默认 10；None 关闭。
    :param validation_fraction: 自动留出验证比例，默认 0.2；0 关闭自动留出。
    :param device: auto、cpu 或 cuda[:设备编号]，默认 auto。
    :param target: 可选目标字段名，支持 fit(df)；显式 y 优先且目标列始终移除。
    :param eval_metric: 最终评估集指标；不改变按任务损失早停的规则。
    :param random_state: 随机种子，默认 None。
    :param verbose: 是否输出中文逐轮日志，默认 False。
    :param scorecard_params: 二分类概率评分卡配置，回归不使用。
    :param history_policy: 训练历史策略，默认 summary。
    :param max_history: 最大训练历史数量，默认 20。
    :param weight_type: cost 或 frequency，沿用统一模型的权重语义。
    :param loss: None 使用默认损失；支持名称、BaseLoss、PyTorch 损失或逐行张量函数。
    :param loss_params: 损失参数字典，训练层统一归约逐行损失。
    :param optimizer: adamw（默认）、adam、sgd、rmsprop、adagrad 或 PyTorch 优化器类。
    :param optimizer_params: 优化器额外参数；lr/weight_decay 使用模型同名配置。

    **属性**

    :ivar embedding_dim_: 等于 n_d；各决策步表示求和，位于预测头之前。
    :ivar feature_importances_: 按原始字段顺序的注意力重要性；使用实际训练行及样本权重归一化。
    :ivar n_iter_: 实际训练轮数；best_iteration_ 为最优轮（从 1 开始）。
    :ivar preprocessor_: 仅在实际训练行拟合的填补、标准化和类别编码。
    :ivar pretrainer_: 已完成预训练的官方 TabNetPretrainer，含解码器及训练历史。

    **参考样例**

    >>> model = TabNet(target='FPD', random_state=42)
    >>> model.pretrain(unlabeled_df)  # 官方 TabNetPretrainer 无监督阶段
    >>> model.fit(train_df)          # 自动复制本实例的预训练快照
    >>> probability = model.predict_proba(test_df)[:, 1]
    >>> embedding = model.get_embeddings(test_df)

    pretrain 不读取业务标签，调用官方 TabNetPretrainer 的掩码、解码器及无监督重建损失。
    fit 的训练曲线包含任务损失和稀疏正则；验证曲线及早停只使用任务损失。
    """

    _network_kind = "TabNet"
    _supports_native_feature_importance = True

    def __init__(
        self,
        objective="binary",
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        cat_embedding_dim=1,
        categorical_features=None,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        lambda_sparse=0.001,
        epsilon=1e-15,
        learning_rate=0.02,
        weight_decay=0.00001,
        max_epochs=100,
        batch_size=256,
        early_stopping_rounds=10,
        validation_fraction=0.2,
        device="auto",
        target=None,
        eval_metric=None,
        random_state=None,
        verbose=False,
        scorecard_params=None,
        history_policy="summary",
        max_history=20,
        weight_type="cost",
        loss=None,
        loss_params=None,
        optimizer="adamw",
        optimizer_params=None,
    ):
        super().__init__(
            objective=objective,
            target=target,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=1,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            verbose=verbose,
            scorecard_params=scorecard_params,
            history_policy=history_policy,
            max_history=max_history,
            weight_type=weight_type,
        )
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.cat_embedding_dim = cat_embedding_dim
        self.categorical_features = categorical_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.lambda_sparse = lambda_sparse
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.loss_params = loss_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    @record_training(fitted=False)
    def pretrain(
        self,
        X,
        eval_set=None,
        sample_weight=None,
        *,
        pretraining_ratio=0.5,
        max_epochs=None,
        batch_size=None,
        early_stopping_rounds=None,
        validation_fraction=None,
    ):
        """官方 TabNetPretrainer 的无标签预训练，返回当前模型。

        :param X: 无标签特征；配置的 target 字段只移除，不读取。
        :param eval_set: 无标签矩阵、列表或具名字典；第一项用于早停，不接受 (X, y)。
        :param sample_weight: 有限非负权重，映射为官方 weights 采样权重；零权重行不参与预处理。
        :param pretraining_ratio: (0, 1] 内的官方随机掩码比例，默认 0.5。
        :param max_epochs: 本次预训练轮数；None 沿用构造器。
        :param batch_size: 本次预训练批量，至少 2；None 沿用构造器。
        :param early_stopping_rounds: None 沿用构造器；构造器设为 None 时关闭早停。
        :param validation_fraction: None 沿用构造器；0 关闭自动留出。
        :return: self。可提取 embedding、保存及通过 from_pretrained 迁移，fit 后才可预测。

        使用官方在嵌入空间的方差归一化重建损失；缺失值先由统一预处理填补，类别重建其 embedding。
        ``pretrainer_`` 保留实际官方预训练器及解码器；fit 复制其编码器快照，不修改预训练来源。

        >>> model.pretrain(X_unlabeled, pretraining_ratio=0.5, max_epochs=20)
        >>> model.fit(X_train, y_train)
        """
        try:
            from ._tabnet_pretraining import pretrain_tabnet
        except ImportError as exc:
            raise DependencyError(
                'TabNet 需要 torch 和 pytorch-tabnet>=4.1，请安装 pip install "hscredit[net]"'
            ) from exc
        return pretrain_tabnet(
            self,
            X,
            eval_set,
            sample_weight,
            pretraining_ratio,
            dict(
                max_epochs=max_epochs,
                batch_size=batch_size,
                early_stopping_rounds=early_stopping_rounds,
                validation_fraction=validation_fraction,
            ),
        )

    def _validate_options(self):
        super()._validate_options()
        for name in ("n_d", "n_a", "n_steps", "cat_embedding_dim", "virtual_batch_size", "n_independent", "n_shared"):
            value = getattr(self, name)
            minimum = 0 if name in ("n_independent", "n_shared") else 2 if name == "virtual_batch_size" else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValidationError(f"{name} 必须是至少为 {minimum} 的整数")
        if self.n_independent + self.n_shared == 0:
            raise ValidationError("n_independent 和 n_shared 不能同时为 0")
        if self.batch_size < 2:
            raise ValidationError("TabNet 的 batch_size 必须至少为 2")
        for name in ("gamma", "momentum", "epsilon", "lambda_sparse"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValidationError(f"{name} 必须为有限数值")
        if self.gamma < 1 or self.lambda_sparse < 0:
            raise ValidationError("gamma 必须至少为 1，lambda_sparse 必须非负")
        if not 0 < self.momentum < 1 or not 0 < self.epsilon < 1:
            raise ValidationError("momentum 和 epsilon 必须在 (0, 1) 内")
        if self.mask_type not in ("sparsemax", "entmax"):
            raise ValidationError("mask_type 必须为 sparsemax 或 entmax")

    def _initialize_network(self):
        try:
            from importlib.metadata import version
            from packaging.version import Version

            import torch

            if Version(version("pytorch-tabnet")) < Version("4.1"):
                raise DependencyError('TabNet 需要 pytorch-tabnet>=4.1，请安装 pip install "hscredit[net]"')
            from ._tabnet_network import TabNetNetwork
        except ImportError as exc:
            raise DependencyError(
                'TabNet 需要 torch 和 pytorch-tabnet>=4.1，请安装 pip install "hscredit[net]"'
            ) from exc
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        try:
            self.device_ = torch.device("cpu" if device == "auto" else device)
            torch.empty(0, device=self.device_)
        except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
            raise ValidationError(f"无法使用训练设备 {device!r}") from exc
        self.native_params_ = {
            name: getattr(self, name)
            for name in (
                "n_d",
                "n_a",
                "n_steps",
                "gamma",
                "n_independent",
                "n_shared",
                "cat_embedding_dim",
                "virtual_batch_size",
                "momentum",
                "mask_type",
                "epsilon",
            )
        }
        self.native_params_.update(
            n_numeric=len(self.preprocessor_.num_indices_), cardinalities=self.preprocessor_.cardinalities_
        )
        self.embedding_dim_, self.n_members_ = self.n_d, 1
        self._model = TabNetNetwork(**self.native_params_).to(self.device_)

    def _training_regularization(self):
        return -self.lambda_sparse * self._model.__dict__.pop("mask_loss")

    def _train(self, tensors, eval_set):
        import torch

        super()._train(tensors, eval_set)
        self._model.eval()
        totals = np.zeros(self.n_features_in_, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(tensors[0]), self.batch_size):
                numeric, categorical, _, weights = (
                    v[start : start + self.batch_size].to(self.device_) for v in tensors
                )
                attention = self._model.feature_attention(numeric, categorical)
                totals += (attention * weights[:, None]).sum(dim=0).cpu().numpy()
        values = np.zeros_like(totals)
        positions = self.preprocessor_.num_indices_ + self.preprocessor_.cat_indices_
        values[positions] = totals / totals.sum() if totals.sum() else totals
        self._feature_importances = pd.Series(values, index=self.feature_names_in_, name="importance")

    def get_feature_importances(self, importance_type="attention"):
        """返回按重要性降序排列的原始字段注意力；gain 仅作为统一报告默认值的兼容别名。"""
        self._require_fitted()
        if importance_type not in ("attention", "gain"):
            raise ValidationError("TabNet 仅支持 attention 特征重要性（gain 为兼容别名）")
        return self._feature_importances.sort_values(ascending=False).copy()

    @property
    def feature_importances_(self):
        """按训练字段顺序的注意力重要性，兼容 sklearn 特征筛选和统一模型报告。"""
        self._require_fitted()
        return self._feature_importances.reindex(self.feature_names_in_).to_numpy(copy=True)
