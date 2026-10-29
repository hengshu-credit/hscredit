"""TabPFN 官方微调适配及仅用 X 构造的掩码字段预测任务。"""

import copy
from importlib.metadata import version
from numbers import Real

import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn.model_selection import train_test_split

from ....exceptions import DependencyError, ValidationError
from .._contracts import record_feature_schema
from ._pretraining import PretrainedState, unlabeled_input, random_mask


def load_finetuning():
    try:
        if Version(version("tabpfn")) < Version("9.2"):
            raise ImportError("需要 tabpfn>=9.2")
        from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor
        from tabpfn.base import ModelSpecs
    except ImportError as exc:
        raise DependencyError(
            "TabPFN 微调和 X-only 预训练需要 Python 3.10+ 及 tabpfn>=9.2；普通推理仍支持旧版后端"
        ) from exc
    return FinetunedTabPFNClassifier, FinetunedTabPFNRegressor, ModelSpecs


class EpochLogger:
    """收集官方逐轮数值，并可转发给调用方日志器。"""

    def __init__(self, delegate=None):
        self.delegate = delegate
        self.epochs = []

    def setup(self, config):
        if self.delegate is not None:
            self.delegate.setup(config)

    def log_step(self, metrics, step):
        if self.delegate is not None:
            self.delegate.log_step(metrics, step)

    def log_epoch(self, metrics, step):
        self.epochs.append(dict(metrics))
        if self.delegate is not None:
            self.delegate.log_epoch(metrics, step)

    def finish(self):
        if self.delegate is not None:
            self.delegate.finish()


def model_spec(estimator):
    """沿用官方克隆方式，复制已更新网络及配套架构/分布配置。"""
    _, _, specs = load_finetuning()
    return specs(
        model=copy.deepcopy(estimator.models_[0]).cpu(),
        architecture_config=copy.deepcopy(estimator.configs_[0]),
        inference_config=copy.deepcopy(estimator.inference_config_),
        norm_criterion=copy.deepcopy(getattr(estimator, "znorm_space_bardist_", None)),
    )


def fine_tune(model, X, y, params, native_params, validation=None):
    """仅返回官方微调后的推理估计器，不使用未更新权重的原始模型回退。"""
    classifier, regressor, _ = load_finetuning()
    if params is not None and not isinstance(params, dict):
        raise ValidationError("finetune_params 必须为字典或 None")
    options = copy.deepcopy(params or {})
    mixed_precision = options.pop("mixed_precision", False)
    if not isinstance(mixed_precision, (bool, np.bool_)):
        raise ValidationError("mixed_precision 必须为布尔值")
    aliases = {
        "max_epochs": "epochs",
        "batch_size": "n_finetune_ctx_plus_query_samples",
        "validation_fraction": "validation_split_ratio",
    }
    for name, native in aliases.items():
        if name in options:
            if native in options:
                raise ValidationError(f"微调参数 {name} 与 {native} 不能同时提供")
            options[native] = options.pop(name)
    if "epochs" in options and (
        isinstance(options["epochs"], (bool, np.bool_))
        or not isinstance(options["epochs"], (int, np.integer))
        or options["epochs"] < 1
    ):
        raise ValidationError("max_epochs/epochs 必须为正整数")
    if "learning_rate" in options and (
        not isinstance(options["learning_rate"], Real)
        or not np.isfinite(options["learning_rate"])
        or options["learning_rate"] <= 0
    ):
        raise ValidationError("learning_rate 必须为有限正数")
    if "early_stopping_rounds" in options:
        rounds = options.pop("early_stopping_rounds")
        if "early_stopping" in options or "early_stopping_patience" in options:
            raise ValidationError("early_stopping_rounds 与原生早停参数不能同时提供")
        options["early_stopping"] = rounds is not None
        if rounds is not None:
            options["early_stopping_patience"] = rounds
    native_params = copy.deepcopy(native_params)
    if native_params.get("model_path") == "auto":
        native_params.pop("model_path")  # 允许官方 model_version 选择其默认权重。
    native_params.pop("n_estimators", None)
    options.setdefault("device", native_params.pop("device"))
    if options["device"] == "auto":
        import torch

        options["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if not mixed_precision:
        import torch

        if native_params.get("inference_precision", "auto") not in ("auto", torch.float32):
            raise ValidationError("mixed_precision=False 需要 inference_precision 为 auto 或 torch.float32")
        native_params["inference_precision"] = torch.float32
    options.setdefault("random_state", model.random_state if model.random_state is not None else 0)
    options.setdefault("n_estimators_final_inference", model.n_estimators)
    options.setdefault("n_estimators_finetune", model.n_estimators)
    options.setdefault("n_estimators_validation", model.n_estimators)
    key = "extra_classifier_kwargs" if model.objective_ == "binary" else "extra_regressor_kwargs"
    if key in options:
        raise ValidationError(f"{key} 请通过 TabPFN 的普通原生参数配置，避免两处配置互相覆盖")
    options[key] = native_params
    logger = EpochLogger(options.pop("experiment_logger", None))
    options["experiment_logger"] = logger
    constructor = classifier if model.objective_ == "binary" else regressor
    if not mixed_precision:
        if not callable(getattr(constructor, "_forward_with_loss", None)):
            raise DependencyError("当前 TabPFN 微调后端缺少全精度损失入口，请使用兼容的 tabpfn>=9.2")

        class FullPrecisionTrainer(constructor):
            def _forward_with_loss(self, batch):
                # 官方训练循环仍可使用 GradScaler；关闭内外层 autocast 避免大额回归的 FP16 梯度溢出。
                device_type = "cuda" if self.device.startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    return super()._forward_with_loss(batch)

        constructor = FullPrecisionTrainer
    try:
        trainer = constructor(**options)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"TabPFN 微调参数无效: {exc}") from exc
    fit_options = {} if validation is None else {"X_val": validation[0], "y_val": validation[1]}
    trainer.fit(X, y, **fit_options)
    losses = [entry["train/mean_loss"] for entry in logger.epochs if "train/mean_loss" in entry]
    if not losses:
        raise ValidationError("TabPFN 微调未产生有效训练批次，请检查样本量、类别分布和 batch_size")
    if not np.isfinite(losses).all():
        raise ValidationError("TabPFN 微调损失不是有限数，请检查数据尺度、学习率和 mixed_precision 配置")
    attr = "finetuned_inference_classifier_" if model.objective_ == "binary" else "finetuned_inference_regressor_"
    estimator = getattr(trainer, attr, None)
    if estimator is None:
        raise ValidationError("当前 TabPFN 微调后端未提供已微调的推理模型，不能回退到未微调权重")
    return estimator, logger.epochs


def _field_tasks(train, valid, objective):
    """回归重建数值；分类量化数值，类别用二进制位重建，避免序号距离损失。"""
    from ...binning import QuantileBinning

    for index, column in enumerate(train.columns):
        raw = train[column]
        observed = raw.notna().to_numpy()
        if raw[observed].nunique() < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(raw.dtype) and not pd.api.types.is_bool_dtype(raw.dtype)
        if numeric and objective == "regression":
            values = raw.to_numpy(dtype=float, na_value=np.nan)
            valid_values = None if valid is None else valid[column].to_numpy(dtype=float, na_value=np.nan)
            yield index, str(column), values, observed, valid_values, (
                None if valid is None else np.isfinite(valid_values)
            )
            continue
        if numeric:
            name = "重建目标"
            # 等频切点不依赖 y；统一分箱统计接口要求双类，使用本字段最小值指示作辅助统计标签。
            # 关闭单调约束，实际预训练目标只取等频箱编号，不读取任何业务标签或分箱指标。
            clean = raw[observed].to_numpy()
            binner = QuantileBinning(max_n_bins=8, force_numerical=True, monotonic=False, n_jobs=1).fit(
                pd.DataFrame({name: clean}), (clean > clean.min()).astype(int)
            )
            codes = binner.transform(pd.DataFrame({name: raw.to_numpy()}), metric="indices")[name].to_numpy(dtype=int)
            valid_codes = (
                None
                if valid is None
                else binner.transform(pd.DataFrame({name: valid[column].to_numpy()}), metric="indices")[name].to_numpy(
                    dtype=int
                )
            )
            valid_observed = None if valid is None else valid[column].notna().to_numpy()
        else:
            categories = pd.Index(raw[observed].unique())
            codes = pd.Categorical(raw, categories=categories).codes.astype(int)
            valid_codes = (
                None if valid is None else pd.Categorical(valid[column], categories=categories).codes.astype(int)
            )
            valid_observed = None if valid is None else valid_codes >= 0
        for bit in range(max(1, int(np.ceil(np.log2(int(codes[observed].max()) + 1))))):
            values = ((codes >> bit) & 1).astype(float)
            if np.unique(values[observed]).size < 2:
                continue
            yield index, f"{column}：位{bit+1}", values, observed, (
                None if valid is None else ((valid_codes >> bit) & 1).astype(float)
            ), valid_observed


def pretrain_pfn(model, X, eval_set, ratio, max_epochs, batch_size, finetune_params):
    load_finetuning()
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValidationError("pretraining_ratio 必须在 (0, 1] 内")
    model._pretrain_only_ = False
    model.__dict__.pop("_pretrained_state_", None)
    model._validate_task()
    X = unlabeled_input(X, model.target)
    record_feature_schema(model, X)
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if frame.shape[1] < 2:
        raise ValidationError("TabPFN 字段预测预训练至少需要两个特征")
    if np.isinf(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValidationError("预训练数值特征不允许正负无穷")
    if any(value is not None and not isinstance(value, dict) for value in (model.finetune_params, finetune_params)):
        raise ValidationError("finetune_params 必须是字典或 None")
    options = copy.deepcopy(model.finetune_params or {})
    options.update(finetune_params or {})
    if max_epochs is not None:
        options["max_epochs"] = max_epochs
    if batch_size is not None:
        options["batch_size"] = batch_size
    if "validation_fraction" in options and "validation_split_ratio" in options:
        raise ValidationError("微调参数 validation_fraction 与 validation_split_ratio 不能同时提供")
    fraction = options.pop("validation_fraction", options.pop("validation_split_ratio", 0.1))
    options["validation_fraction"] = 0
    if not isinstance(fraction, Real) or not 0 <= fraction < 1:
        raise ValidationError("validation_fraction 必须在 [0, 1) 内")
    valid = None
    indices = np.arange(len(frame))
    if eval_set is not None:
        values = (
            list(eval_set.values())
            if isinstance(eval_set, dict)
            else list(eval_set) if isinstance(eval_set, (list, tuple)) else [eval_set]
        )
        if len(values) != 1:
            raise ValidationError("TabPFN 预训练请提供一个无标签验证集")
        value = model.feature_schema_.align(unlabeled_input(values[0], model.target))
        valid = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value, columns=frame.columns)
    elif fraction > 0:
        indices, valid_idx = train_test_split(indices, test_size=fraction, random_state=model.random_state)
        valid = frame.iloc[valid_idx]
        model._eval_train_indices_, model._eval_val_indices_ = indices, valid_idx
    train = frame.iloc[indices]
    rng = np.random.default_rng(model.random_state)
    observed = train.notna().to_numpy()
    mask = random_mask(observed, ratio, rng)
    corrupted = train.copy()
    for j, column in enumerate(train.columns):
        corrupted[column] = train[column].where(~mask[:, j], np.nan)
    valid_mask = None if valid is None else random_mask(valid.notna().to_numpy(), ratio, rng)
    native = model._native_parameters()
    state, count, cells = None, 0, 0
    model._evals_result = {}
    for index, name, target, present, valid_target, valid_present in _field_tasks(train, valid, model.objective_):
        selected = present & mask[:, index]
        if selected.sum() < 4 or np.unique(target[selected]).size < 2:
            continue
        if model.objective_ == "binary" and np.bincount(target[selected].astype(int), minlength=2).min() < 2:
            continue
        task_X = corrupted.loc[selected].copy()
        task_X[train.columns[index]] = np.nan  # 上下文标签在 y 中；禁止目标值留在 X 里泄漏答案。
        validation = None
        if valid is not None:
            usable = valid_present & valid_mask[:, index]
            if usable.sum() >= 2 and (model.objective_ != "binary" or np.unique(valid_target[usable]).size == 2):
                task_valid = valid.loc[usable].copy()
                for j, column in enumerate(valid.columns):
                    task_valid[column] = task_valid[column].where(~valid_mask[usable, j], np.nan)
                task_valid[valid.columns[index]] = np.nan
                validation = task_valid, valid_target[usable]
        if state is not None:
            native["model_path"] = state
        model.native_params_ = dict(native)
        estimator, history = fine_tune(model, task_X, target[selected], options, native, validation)
        state = model_spec(estimator)
        model._model = estimator
        count += 1
        cells += int(selected.sum())
        losses = [entry["train/mean_loss"] for entry in history if "train/mean_loss" in entry]
        model._evals_result[name] = {"字段重建损失": losses}
    if state is None:
        raise ValidationError("X 中没有足够的非恒定观测字段用于 TabPFN 掩码预训练")
    model._pretrained_state_ = PretrainedState(
        "TabPFN", copy.deepcopy(model.feature_schema_), None, {"objective": model.objective_}, state
    )
    model._pretrain_only_ = True
    model.pretraining_summary_ = model.training_summary_
    model.pretraining_summary_.update(
        训练阶段="掩码字段预测预训练", 字段任务数=count, 自监督目标数=cells, 遮盖比例=ratio
    )
    return model
