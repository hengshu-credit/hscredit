"""ResNet、FTTransformer 与 TabM 共用的训练、早停和分批推理。"""

from numbers import Integral, Real

import numpy as np
from sklearn.model_selection import train_test_split

from ....exceptions import DependencyError, ValidationError
from .._contracts import take_rows
from .._lifecycle import record_training
from .base import BaseNetModel
from ._preprocessing import TabularPreprocessor


class TorchTabularModel(BaseNetModel):
    """训练预处理与目标缩放只使用实际训练行，不使用验证集。"""

    _supports_gradient_training = True

    def _validate_options(self):
        for name in ("d_block", "n_blocks", "max_epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValidationError(f"{name} 必须为正整数")
        if self._network_kind == "ResNet":
            if self.batch_size < 2:
                raise ValidationError("ResNet 的 batch_size 必须至少为 2，以支持批归一化")
            if (
                isinstance(self.cat_embedding_dim, bool)
                or not isinstance(self.cat_embedding_dim, Integral)
                or self.cat_embedding_dim < 1
            ):
                raise ValidationError("cat_embedding_dim 必须为正整数")
        elif self._network_kind == "FTTransformer":
            if (
                isinstance(self.n_heads, bool)
                or not isinstance(self.n_heads, Integral)
                or self.n_heads < 1
                or self.d_block % self.n_heads
            ):
                raise ValidationError("n_heads 必须是正整数，且 d_block 必须能被 n_heads 整除")
        elif self._network_kind == "TabM":
            if isinstance(self.k, bool) or not isinstance(self.k, Integral) or self.k < 1:
                raise ValidationError("k 必须为正整数")
            if self.arch_type not in ("tabm", "tabm-mini", "tabm-packed"):
                raise ValidationError("arch_type 必须为 tabm、tabm-mini 或 tabm-packed")
            if self.num_embeddings not in (None, "linear_relu", "periodic"):
                raise ValidationError("num_embeddings 必须为 None、linear_relu 或 periodic")
            if self.num_embedding_params is not None and not isinstance(self.num_embedding_params, dict):
                raise ValidationError("num_embedding_params 必须是字典或 None")
            if self.num_embeddings is None and self.num_embedding_params:
                raise ValidationError("num_embedding_params 需要同时指定 num_embeddings")
        for name, lower, upper in (("dropout", 0, 1), ("learning_rate", 0, np.inf), ("weight_decay", 0, np.inf)):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value) or not lower <= value < upper:
                raise ValidationError(f"{name} 必须是 [{lower}, {upper}) 内的有限数值")
        if self.learning_rate == 0:
            raise ValidationError("learning_rate 必须大于 0")
        if self.early_stopping_rounds is not None and (
            isinstance(self.early_stopping_rounds, bool)
            or not isinstance(self.early_stopping_rounds, Integral)
            or self.early_stopping_rounds < 1
        ):
            raise ValidationError("early_stopping_rounds 必须为正整数或 None")

    @record_training
    def fit(self, X, y=None, sample_weight=None, eval_set=None, *, eval_metric=None, from_pretrained=None):
        """训练网络；eval_set 第一项用于早停，其余只评估。

        无显式 eval_set 且开启早停时，按 validation_fraction 留出验证行。
        eval_set=[] 或 early_stopping_rounds=None 可关闭自动留出。
        sample_weight 参与训练损失；零权重行不进入预处理及网络训练。

        :param X: 二维数组或 DataFrame；含 target 时自动剔除该字段。
        :param y: 一维标签；None 时从 target 提取，显式 y 优先。
        :param sample_weight: 与样本等长的非负有限训练权重，总和必须大于 0。
        :param eval_set: 单个 (X, y)、列表、具名字典或含 target 的 DataFrame。
        :param eval_metric: 覆盖构造器的最终评估指标；不改变按损失早停的规则。
        :param from_pretrained: None 自动使用本实例的预训练快照；False 从头训练；
            True 要求本实例已 pretrain；也可传入已 pretrain 的同类模型。冻结其预处理，复制权重后监督训练。
        :return: 已拟合模型。

        >>> model.fit(X_train, y_train, eval_set={'验证': (X_valid, y_valid)})
        >>> values = model.get_eval_results()
        """
        import copy
        from ._pretraining import resolve_pretrained
        from .._contracts import extract_target

        pretrained = resolve_pretrained(self, from_pretrained)
        self._pretrain_only_ = False
        if pretrained is not None:
            X, y = extract_target(X, y, self.target)
            X = pretrained.schema.align(X)
        self._validate_task()
        self._validate_options()
        X, y, weight = self._prepare_data(X, y, sample_weight, extract_target=True, training=True)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValidationError("训练特征必须至少有一行和一列")
        y = self._validate_y(y, training=True)
        sets = self._prepare_eval_set(eval_set)
        for _, labels in sets or []:
            self._validate_y(labels)
        indices = np.arange(len(y))
        if weight is not None:
            indices = indices[weight > 0]
        if eval_set is None and self.early_stopping_rounds is not None and self.validation_fraction > 0:
            try:
                indices, valid_indices = train_test_split(
                    indices,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=y[indices] if self.objective_ == "binary" else None,
                )
            except ValueError as exc:
                raise ValidationError("样本不足以自动划分验证集，请提供 eval_set 或关闭早停") from exc
            self._eval_train_indices_, self._eval_val_indices_ = indices, valid_indices
            sets = [(take_rows(X, valid_indices), y[valid_indices])]
            self.eval_names_ = ["验证集"]
        train_X, train_y = take_rows(X, indices), y[indices]
        self._validate_y(train_y, training=True)
        if self._network_kind == "ResNet" and len(indices) < 2:
            raise ValidationError("ResNet 至少需要两个正权重训练样本")
        train_weight = np.ones(len(indices)) if weight is None else weight[indices]
        train_weight = train_weight / train_weight.mean()
        self.preprocessor_ = (
            copy.deepcopy(pretrained.preprocessor)
            if pretrained is not None
            else TabularPreprocessor().fit(train_X, self.categorical_features)
        )
        numeric, categorical = self.preprocessor_.transform(train_X)
        self.target_mean_ = float(np.average(train_y, weights=train_weight)) if self.objective_ == "regression" else 0.0
        self.target_scale_ = (
            float(np.sqrt(np.average((train_y - self.target_mean_) ** 2, weights=train_weight)))
            if self.objective_ == "regression"
            else 1.0
        ) or 1.0
        target = (train_y - self.target_mean_) / self.target_scale_
        try:
            import torch
        except ImportError as exc:
            raise DependencyError('请安装 pip install "hscredit[net]"') from exc
        # 局部种子不会覆盖调用方的 CPU/CUDA 随机数状态。
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            seed = self.random_state if self.random_state is not None else int(np.random.default_rng().integers(2**32))
            torch.manual_seed(seed)
            self._initialize_network()
            if pretrained is not None:
                required = {k: v for k, v in self.native_params_.items() if k != "dropout"}
                original = {k: v for k, v in pretrained.params.items() if k != "dropout"}
                if required != original:
                    raise ValidationError("网络结构与预训练快照不一致，请沿用预训练的结构参数")
                self._model.load_state_dict(pretrained.weights)
            tensors = tuple(
                torch.as_tensor(value)
                for value in (numeric, categorical, target.astype(np.float32), train_weight.astype(np.float32))
            )
            self._train(tensors, sets or [])
        self._model.eval()
        self._is_fitted = True
        if self.objective_ == "binary":
            self._fit_probability_scorecard(X, y, sample_weight=weight)
        self._record_net_evaluation(sets, metrics=self.eval_metric if eval_metric is None else eval_metric)
        return self

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
        """只用 X 做掩码重建预训练，返回可提取 embedding 的当前模型。

        :param X: 无标签特征；若包含配置的 target 列，仅移除该列，完全不读取标签。
        :param eval_set: 无标签验证矩阵、列表或具名字典；不接受 (X, y)。
        :param sample_weight: 有限非负的样本权重。
        :param pretraining_ratio: (0, 1] 内随机遮盖比例，仅对真实观测单元格计算重建损失。
        :param max_epochs: 本次预训练轮数，None 沿用构造器配置。
        :param batch_size: 本次预训练批量，None 沿用构造器配置。
        :param early_stopping_rounds: 早停轮数，None 沿用构造器；以验证重建损失选优。
        :param validation_fraction: 内部验证比例，None 沿用构造器；0 禁用自动划分。
        :return: self。pretrain 后只可提取表示；fit 后才可进行业务预测。

        >>> model.pretrain(X_unlabeled, max_epochs=20)
        >>> representation = model.get_embeddings(X_test)
        >>> model.fit(X_train, y_train)  # 自动复制预训练权重并监督训练
        """
        from ._pretraining import pretrain_torch

        return pretrain_torch(
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

    def _initialize_network(self):
        """按已拟合预处理器构建网络，不在这里改动随机种子。"""
        try:
            import torch

            if self._network_kind == "TabM":
                from ._tabm_network import TabMNetwork as Network
            else:
                from ._networks import TabularNetwork as Network
        except ImportError as exc:
            raise DependencyError(
                '网络训练需要 torch 及对应的 rtdl/tabm 依赖，请安装 pip install "hscredit[net]"'
            ) from exc
        device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )
        try:
            self.device_ = torch.device(device)
            torch.empty(0, device=self.device_)
        except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
            raise ValidationError(f"无法使用训练设备 {device!r}") from exc
        self.embedding_dim_ = self.d_block
        self.n_members_ = self.k if self._network_kind == "TabM" else 1
        self.native_params_ = {
            "kind": self._network_kind,
            "n_numeric": len(self.preprocessor_.num_indices_),
            "cardinalities": self.preprocessor_.cardinalities_,
            "d_block": self.d_block,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
            "cat_embedding_dim": getattr(self, "cat_embedding_dim", 16),
            "n_heads": getattr(self, "n_heads", 8),
        }
        if self._network_kind == "TabM":
            for name in ("kind", "cat_embedding_dim", "n_heads"):
                self.native_params_.pop(name)
            self.native_params_.update(
                k=self.k,
                arch_type=self.arch_type,
                num_embeddings=self.num_embeddings,
                num_embedding_params=self.num_embedding_params,
            )
        self._model = Network(**self.native_params_).to(self.device_)

    def _train(self, tensors, eval_set):
        import torch
        from scipy.special import expit, logit
        from ._training import make_loss, make_optimizer, loss_values

        self.loss_fn_ = make_loss(self, self.device_)
        self.optimizer_ = make_optimizer(self)
        self._evals_result = {"train": {"损失": []}}
        validation = []
        for index, (X, y) in enumerate(eval_set):
            name = f"validation_{index}"
            numeric, categorical = self.preprocessor_.transform(X)
            labels = (self._validate_y(y) - self.target_mean_) / self.target_scale_
            validation.append((name, numeric, categorical, labels))
            self._evals_result[name] = {"损失": []}
        best_loss, best_state, stale = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self._model.train()
            batches = list(torch.randperm(len(tensors[0])).split(self.batch_size))
            # 合并末尾单行，避免 BatchNorm 失败且不丢弃样本。
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat((batches[-2], batches[-1]))
                batches.pop()
            total_loss = 0.0
            for indices in batches:
                numeric, categorical, target, weight = (value[indices].to(self.device_) for value in tensors)
                self.optimizer_.zero_grad()
                loss = (loss_values(self, self._model(numeric, categorical), target) * weight).mean()
                if not torch.isfinite(loss):
                    raise ValidationError("训练损失出现非有限值，请检查数据尺度或降低 learning_rate")
                loss.backward()
                self.optimizer_.step()
                total_loss += loss.item() * len(indices)
            self._evals_result["train"]["损失"].append(total_loss / len(tensors[0]))
            self._model.eval()
            valid_losses = []
            for name, numeric, categorical, labels in validation:
                raw = self._infer_arrays(numeric, categorical)
                if raw.ndim == 2:
                    # 官方 TabM：验证集评估集成预测，训练时则独立优化每个成员。
                    raw = (
                        logit(np.clip(expit(raw.astype(float)).mean(axis=1), 1e-12, 1 - 1e-12))
                        if self.objective_ == "binary"
                        else raw.mean(axis=1)
                    )
                with torch.no_grad():
                    value = float(
                        loss_values(
                            self,
                            torch.as_tensor(raw, dtype=torch.float32, device=self.device_),
                            torch.as_tensor(labels, dtype=torch.float32, device=self.device_),
                        ).mean()
                    )
                if not np.isfinite(value):
                    raise ValidationError("验证损失出现非有限值，请检查验证数据")
                self._evals_result[name]["损失"].append(value)
                valid_losses.append(value)
            self.n_iter_ = epoch + 1
            if self.verbose:
                print(f"第 {self.n_iter_} 轮，训练损失: {total_loss / len(tensors[0]):.6f}")
            if valid_losses and self.early_stopping_rounds is not None:
                if valid_losses[0] < best_loss:
                    best_loss, stale = valid_losses[0], 0
                    best_state = {
                        name: value.detach().cpu().clone() for name, value in self._model.state_dict().items()
                    }
                    self._best_iteration, self._best_score = epoch + 1, best_loss
                else:
                    stale += 1
                    if stale >= self.early_stopping_rounds:
                        break
        if best_state is not None:
            self._model.load_state_dict(best_state)

    def _infer_arrays(self, numeric, categorical, *, embeddings=False):
        import torch

        result = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(numeric), self.batch_size):
                numbers = torch.as_tensor(numeric[start : start + self.batch_size], device=self.device_)
                categories = torch.as_tensor(categorical[start : start + self.batch_size], device=self.device_)
                value = self._model.embeddings(numbers, categories) if embeddings else self._model(numbers, categories)
                result.append(value.cpu().numpy())
        shape = (0, self.embedding_dim_) if embeddings else (0,)
        if self._network_kind == "TabM":
            shape = (0, self.n_members_, self.embedding_dim_) if embeddings else (0, self.n_members_)
        values = np.concatenate(result) if result else np.empty(shape, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValidationError("网络输出包含非有限值，请检查数据或重新训练")
        return values

    def _inference(self, X, *, embeddings=False):
        self._require_representation_fitted() if embeddings else self._require_fitted()
        X, _, _ = self._prepare_data(X, extract_target=True)
        return self._infer_arrays(*self.preprocessor_.transform(X), embeddings=embeddings)

    def predict_proba(self, X):
        """二分类两列概率，列顺序为类别 0、1，与 ``classes_`` 一致。"""
        from scipy.special import expit

        self._require_binary()
        positive = expit(self._inference(X).astype(np.float64))
        if positive.ndim == 2:
            positive = positive.mean(axis=1)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        """二分类返回 0/1；回归返回恢复至原始目标单位的连续预测值。"""
        self._require_fitted()
        if self.objective_ == "binary":
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        prediction = self._inference(X).astype(np.float64)
        if prediction.ndim == 2:
            prediction = prediction.mean(axis=1)
        return prediction * self.target_scale_ + self.target_mean_

    def get_embeddings(self, X, aggregate="mean"):
        """返回预测头之前的表示；默认 (样本数, d_block)，None 保留 (成员数, 样本数, d_block)。"""
        if aggregate not in ("mean", None):
            raise ValidationError("aggregate 必须为 mean 或 None")
        values = self._inference(X, embeddings=True)
        if values.ndim == 2:
            return values if aggregate == "mean" else values[np.newaxis, ...]
        return values.mean(axis=1) if aggregate == "mean" else values.transpose(1, 0, 2)
