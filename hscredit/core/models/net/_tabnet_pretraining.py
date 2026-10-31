"""官方 TabNetPretrainer 的生命周期、字段及预训练权重适配。"""

import copy
from numbers import Integral, Real

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.utils import PredictDataset
from torch.utils.data import DataLoader

from ....exceptions import ValidationError
from .._contracts import record_feature_schema, validate_sample_weight, take_rows
from ._preprocessing import TabularPreprocessor
from ._pretraining import PretrainedState, unlabeled_input
from ._tabnet_network import _safe_batches
from ._training import optimizer_config


class _MergedBatches:
    """官方 DataLoader 的轻量视图：末尾单行并入前批，保留采样器语义。"""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        pending = None
        for batch in self.loader:
            if pending is not None:
                if len(batch) == 1:
                    batch = torch.cat((pending, batch))
                else:
                    yield pending
            pending = batch
        if pending is not None:
            yield pending


class _FiniteHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if any(not np.isfinite(value) for value in (logs or {}).values()):
            raise ValidationError("TabNet 预训练损失出现非有限值，请检查数据或降低 learning_rate")


class NativeTabNetPretrainer(TabNetPretrainer):
    """保留官方掩码、编码器、解码器、无监督损失和训练循环。"""

    def _set_network(self):
        super()._set_network()
        _safe_batches(self.network)

    def _construct_loaders(self, X_train, eval_set):
        train, _ = super()._construct_loaders(X_train, [])
        # 官方 4.1 将训练 sampler 复用于验证集，长度不同时会越界。
        # 验证集固定按原始行遍历，不采样、不丢行，也不使用训练权重。
        valid = [
            DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=self.pin_memory,
            )
            for X in eval_set
        ]
        return _MergedBatches(train), valid

    def _set_metrics(self, eval_names):
        super()._set_metrics(eval_names)
        # hscredit 所有 net 模型由第一份验证集控制早停。
        self.early_stopping_metric = self._metrics_names[0] if self._metrics_names else None

    def compute_loss(self, output, embedded_x, obf_vars):
        loss = super().compute_loss(output, embedded_x, obf_vars)
        if not torch.isfinite(loss):
            raise ValidationError("TabNet 预训练损失出现非有限值，请检查数据或降低 learning_rate")
        return loss


def pretrain_tabnet(model, X, eval_set, sample_weight, ratio, options):
    """只传 X 给官方预训练器，再精确复制 embedder/encoder 到统一预测网络。"""
    model._pretrain_only_ = False
    model.__dict__.pop("_pretrained_state_", None)
    model.__dict__.pop("pretrainer_", None)
    if (
        isinstance(ratio, (bool, np.bool_))
        or not isinstance(ratio, Real)
        or not np.isfinite(ratio)
        or not 0 < ratio <= 1
    ):
        raise ValidationError("pretraining_ratio 必须在 (0, 1] 内")
    model._validate_task()
    model._validate_options()
    config = {name: getattr(model, name) if value is None else value for name, value in options.items()}
    for name in ("max_epochs", "batch_size"):
        value = config[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < (2 if name == "batch_size" else 1)
        ):
            raise ValidationError("max_epochs 必须为正整数，TabNet 的 batch_size 必须至少为 2")
    patience, fraction = config["early_stopping_rounds"], config["validation_fraction"]
    if patience is not None and (
        isinstance(patience, (bool, np.bool_)) or not isinstance(patience, Integral) or patience < 1
    ):
        raise ValidationError("early_stopping_rounds 必须为正整数或 None")
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real) or not 0 <= fraction < 1:
        raise ValidationError("validation_fraction 必须在 [0, 1) 内")
    X = unlabeled_input(X, model.target)
    record_feature_schema(model, X)
    weight = validate_sample_weight(sample_weight, len(X))
    indices = np.arange(len(X)) if weight is None else np.flatnonzero(weight > 0)
    validation = []
    if eval_set is None and fraction > 0 and patience is not None:
        try:
            indices, valid = train_test_split(indices, test_size=fraction, random_state=model.random_state)
        except ValueError as exc:
            raise ValidationError("样本不足以自动划分预训练验证集，请提供 eval_set 或关闭早停") from exc
        model._eval_train_indices_, model._eval_val_indices_ = indices, valid
        validation = [("预训练验证集", take_rows(X, valid))]
    elif eval_set is not None:
        if isinstance(eval_set, dict):
            validation = list(eval_set.items())
        elif isinstance(eval_set, (list, tuple)):
            validation = [(f"预训练验证集{i+1}", value) for i, value in enumerate(eval_set)]
        else:
            validation = [("预训练验证集", eval_set)]
    train = take_rows(X, indices)
    if len(train) < 2:
        raise ValidationError("至少需要两个正权重的预训练样本")
    model.preprocessor_ = TabularPreprocessor().fit(train, model.categorical_features)

    def transform(frame):
        n, c = model.preprocessor_.transform(frame)
        return np.column_stack((n, c)).astype(np.float32)

    arrays = []
    for _, value in validation:
        frame = model.feature_schema_.align(unlabeled_input(value, model.target))
        if len(frame) < 2:
            raise ValidationError("TabNet 预训练验证集至少需要两行，以计算官方重建损失的样本方差")
        arrays.append(transform(frame))
    optimizer, optimizer_params = optimizer_config(model)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed = model.random_state if model.random_state is not None else int(np.random.default_rng().integers(2**32))
        torch.manual_seed(seed)
        model._initialize_network()
        params = dict(model.native_params_)
        n_numeric, cardinalities = params.pop("n_numeric"), params.pop("cardinalities")
        params.pop("virtual_batch_size")
        params["cat_emb_dim"] = params.pop("cat_embedding_dim")
        backend = NativeTabNetPretrainer(
            **params,
            cat_idxs=list(range(n_numeric, n_numeric + len(cardinalities))),
            cat_dims=cardinalities,
            optimizer_fn=optimizer,
            optimizer_params=optimizer_params,
            seed=seed,
            device_name=str(model.device_),
            verbose=int(model.verbose),
        )
        names = [f"validation_{i}" for i in range(len(arrays))]
        backend.fit(
            transform(train),
            eval_set=arrays,
            eval_name=names,
            pretraining_ratio=ratio,
            weights=0 if weight is None else weight[indices],
            max_epochs=config["max_epochs"],
            patience=0 if patience is None else patience,
            batch_size=config["batch_size"],
            virtual_batch_size=model.virtual_batch_size,
            drop_last=False,
            callbacks=[_FiniteHistory()],
        )
        # 官方 from_unsupervised 同样迁移 embedder 与 encoder；预测头保持独立初始化。
        for target, source in (
            (model._model.backbone.embedder, backend.network.embedder),
            (model._model.backbone.tabnet.encoder, backend.network.encoder),
        ):
            state = source.state_dict()
            for name in ("embedding_group_matrix", "group_attention_matrix"):
                if hasattr(source, name):
                    state[name] = getattr(source, name).detach().clone()
            target.load_state_dict(state, strict=True)
    model.pretrainer_ = backend
    model.optimizer_ = backend._optimizer
    history = backend.history.history
    model._evals_result = {"train": {"重建损失": list(history["loss"])}}
    model.eval_names_ = [str(name) for name, _ in validation]
    for name in names:
        model._evals_result[name] = {"重建损失": list(history[f"{name}_unsup_loss_numpy"])}
    model.n_iter_ = len(history["loss"])
    if arrays and patience is not None:
        model._best_iteration, model._best_score = int(backend.best_epoch) + 1, float(backend.best_cost)
    model._model.eval()
    model._pretrained_state_ = PretrainedState(
        model._network_kind,
        copy.deepcopy(model.feature_schema_),
        copy.deepcopy(model.preprocessor_),
        copy.deepcopy(model.native_params_),
        {key: value.detach().cpu().clone() for key, value in model._model.state_dict().items()},
    )
    model._pretrain_only_ = True
    model.pretraining_summary_ = model.training_summary_
    model.pretraining_summary_.update(
        训练阶段="TabNet 半监督预训练", 遮盖比例=ratio, 后端="pytorch_tabnet.TabNetPretrainer"
    )
    return model
