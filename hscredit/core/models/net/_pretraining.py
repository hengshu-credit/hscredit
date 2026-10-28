"""仅使用 X 的掩码重建、预训练快照和监督训练迁移契约。"""

import copy
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ....exceptions import DependencyError, ValidationError
from .._contracts import record_feature_schema, validate_sample_weight, take_rows
from ._preprocessing import TabularPreprocessor


@dataclass
class PretrainedState:
    """只保存编码器和字段处理状态，不保留原始预训练样本。"""

    kind: str
    schema: object
    preprocessor: object
    params: dict
    weights: Any


def unlabeled_input(X, target):
    """已配置目标列只删除不读取；不从 X 中构造业务监督标签。"""
    if isinstance(X, pd.DataFrame):
        X = X.drop(columns=[target]) if target is not None and target in X else X
    else:
        X = np.asarray(X)
    if getattr(X, "ndim", None) != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValidationError("预训练需要非空的稠密二维 X")
    return X


def resolve_pretrained(model, source):
    """None 自动沿用本实例预训练；False 从头训练；True 要求已有快照。"""
    if source is False:
        return None
    owner = model if source is None or source is True else source
    state = getattr(owner, "_pretrained_state_", None)
    if state is None and source is not None:
        raise ValidationError("from_pretrained 需要已完成 pretrain 的同类模型；True 表示本实例")
    if state is not None and state.kind != model._network_kind:
        raise ValidationError("预训练来源与当前网络类型不一致")
    return state


def observed_mask(preprocessor, X):
    frame = preprocessor._frame(X)
    numeric = ~np.isnan(preprocessor._numeric(frame))
    _, categorical = preprocessor.transform(frame)
    return np.column_stack((numeric, categorical > 0))


def random_mask(observed, ratio, rng):
    """只遮盖真实观测值，每个有观测的样本至少遮盖一个字段。"""
    mask = (rng.random(observed.shape) < ratio) & observed
    for row in np.flatnonzero(observed.any(axis=1) & ~mask.any(axis=1)):
        mask[row, rng.choice(np.flatnonzero(observed[row]))] = True
    return mask


def pretrain_torch(model, X, eval_set, sample_weight, ratio, options):
    try:
        import torch
    except ImportError as exc:
        raise DependencyError('网络预训练需要 torch，请安装 pip install "hscredit[net]"') from exc
    from torch import nn
    from torch.nn import functional as F
    from ._training import make_optimizer

    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValidationError("pretraining_ratio 必须在 (0, 1] 内")
    model._pretrain_only_ = False
    model.__dict__.pop("_pretrained_state_", None)
    model._validate_task()
    model._validate_options()
    X = unlabeled_input(X, model.target)
    record_feature_schema(model, X)
    weight = validate_sample_weight(sample_weight, len(X))
    epochs = model.max_epochs if options["max_epochs"] is None else options["max_epochs"]
    batch_size = model.batch_size if options["batch_size"] is None else options["batch_size"]
    patience = (
        model.early_stopping_rounds if options["early_stopping_rounds"] is None else options["early_stopping_rounds"]
    )
    fraction = model.validation_fraction if options["validation_fraction"] is None else options["validation_fraction"]
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (epochs, batch_size)):
        raise ValidationError("max_epochs、batch_size 必须为正整数")
    if model._network_kind == "ResNet" and batch_size < 2:
        raise ValidationError("ResNet 的 batch_size 必须至少为 2")
    if (
        not isinstance(fraction, Real)
        or not 0 <= fraction < 1
        or (
            patience is not None
            and (isinstance(patience, (bool, np.bool_)) or not isinstance(patience, (int, np.integer)) or patience < 1)
        )
    ):
        raise ValidationError("validation_fraction 必须在 [0, 1)，early_stopping_rounds 必须为正整数或 None")
    indices = np.arange(len(X)) if weight is None else np.flatnonzero(weight > 0)
    validation = []
    if eval_set is None and fraction > 0 and patience is not None:
        indices, valid = train_test_split(indices, test_size=fraction, random_state=model.random_state)
        model._eval_train_indices_, model._eval_val_indices_ = indices, valid
        validation = [("预训练验证集", take_rows(X, valid))]
    elif eval_set is not None:
        if isinstance(eval_set, dict):
            validation = list(eval_set.items())
        elif isinstance(eval_set, (list, tuple)):
            validation = [(f"预训练验证集{i+1}", x) for i, x in enumerate(eval_set)]
        else:
            validation = [("预训练验证集", eval_set)]
    train = take_rows(X, indices)
    if len(train) < 2:
        raise ValidationError("至少需要两个正权重的预训练样本")
    model.preprocessor_ = TabularPreprocessor().fit(train, model.categorical_features)
    numeric, categorical = model.preprocessor_.transform(train)
    observed = observed_mask(model.preprocessor_, train)
    if not observed.any():
        raise ValidationError("X 没有可用于重建的观测值")
    weights = np.ones(len(train)) if weight is None else weight[indices]
    n_numeric = numeric.shape[1]
    rng = np.random.default_rng(model.random_state)

    def row_loss(numbers, categories, flags, numeric_head, category_heads):
        flags = torch.as_tensor(flags, dtype=torch.bool, device=model.device_)
        n = torch.as_tensor(numbers, device=model.device_)
        c = torch.as_tensor(categories, device=model.device_)
        hidden = model._model.embeddings(n.masked_fill(flags[:, :n_numeric], 0), c.masked_fill(flags[:, n_numeric:], 0))
        member = hidden.ndim == 3
        losses = []
        if numeric_head is not None:
            prediction = numeric_head(hidden)
            value = (prediction - (n[:, None] if member else n)).square()
            losses.extend((value.mean(dim=1) if member else value).unbind(dim=-1))
        for j, head in enumerate(category_heads):
            if head is None:
                losses.append(hidden.sum(dim=tuple(range(1, hidden.ndim))) * 0)
                continue
            logits = head(hidden)
            labels = (c[:, j] - 1).clamp_min(0)
            if member:
                labels = labels[:, None].expand(logits.shape[:2])
            value = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
            losses.append(value.reshape(logits.shape[:-1]).mean(dim=1) if member else value)
        matrix = torch.stack(losses, dim=1)
        counts = flags.sum(dim=1)
        return (matrix * flags).sum(dim=1) / counts.clamp_min(1), counts > 0

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(model.random_state if model.random_state is not None else int(rng.integers(2**32)))
        model._initialize_network()
        numeric_head = nn.Linear(model.d_block, n_numeric).to(model.device_) if n_numeric else None
        category_heads = nn.ModuleList(
            [nn.Linear(model.d_block, size - 1) if size > 1 else None for size in model.preprocessor_.cardinalities_]
        ).to(model.device_)
        decoder = nn.ModuleDict({"categories": category_heads})
        if numeric_head is not None:
            decoder["numeric"] = numeric_head
        model._pretrain_decoder_ = decoder
        parameters = list(model._model.parameters()) + list(decoder.parameters())
        model.optimizer_ = make_optimizer(model, parameters)
        prepared = []
        for name, value in validation:
            value = model.feature_schema_.align(unlabeled_input(value, model.target))
            n, c = model.preprocessor_.transform(value)
            prepared.append((str(name), n, c, random_mask(observed_mask(model.preprocessor_, value), ratio, rng)))
        model.eval_names_ = [p[0] for p in prepared]
        model._evals_result = {
            "train": {"重建损失": []},
            **{f"validation_{i}": {"重建损失": []} for i in range(len(prepared))},
        }
        best, best_state, stale = np.inf, None, 0
        for epoch in range(epochs):
            model._model.train()
            order = rng.permutation(len(train))
            batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
            if model._network_kind == "ResNet" and len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = np.concatenate((batches[-2], batches[-1]))
                batches.pop()
            total, denominator = 0.0, 0.0
            for batch in batches:
                flags = random_mask(observed[batch], ratio, rng)
                model.optimizer_.zero_grad()
                values, valid = row_loss(numeric[batch], categorical[batch], flags, numeric_head, category_heads)
                w = torch.as_tensor(weights[batch], device=model.device_, dtype=values.dtype) * valid
                if w.sum() == 0:
                    continue
                loss = (values * w).sum() / w.sum()
                if not torch.isfinite(loss):
                    raise ValidationError("预训练重建损失出现非有限值")
                loss.backward()
                model.optimizer_.step()
                total += float((values.detach() * w).sum())
                denominator += float(w.sum())
            model._evals_result["train"]["重建损失"].append(total / denominator)
            model._model.eval()
            validation_losses = []
            with torch.no_grad():
                for i, (name, n, c, flags) in enumerate(prepared):
                    loss_sum, count = 0.0, 0
                    for start in range(0, len(n), batch_size):
                        stop = start + batch_size
                        values, valid = row_loss(
                            n[start:stop], c[start:stop], flags[start:stop], numeric_head, category_heads
                        )
                        loss_sum += float(values[valid].sum())
                        count += int(valid.sum())
                    if count == 0:
                        raise ValidationError("预训练验证集没有可重建的观测值")
                    value = loss_sum / count
                    if not np.isfinite(value):
                        raise ValidationError("预训练验证损失不是有限数")
                    model._evals_result[f"validation_{i}"]["重建损失"].append(value)
                    validation_losses.append(value)
            model.n_iter_ = epoch + 1
            if validation_losses and patience is not None:
                if validation_losses[0] < best:
                    best, stale = validation_losses[0], 0
                    best_state = copy.deepcopy(model._model.state_dict()), copy.deepcopy(decoder.state_dict())
                    model._best_iteration, model._best_score = epoch + 1, best
                else:
                    stale += 1
                    if stale >= patience:
                        break
        if best_state is not None:
            model._model.load_state_dict(best_state[0])
            decoder.load_state_dict(best_state[1])
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
    model.pretraining_summary_.update(训练阶段="掩码预训练", 遮盖比例=ratio, 重建字段数=int(observed.any(axis=0).sum()))
    return model
