"""提升树继续训练的来源、字段与原生参数适配。"""

import copy
import hashlib
from pathlib import Path

from ....exceptions import ValidationError
from ..base import BaseRiskModel
from .._contracts import align_model_features, extract_target


def prepare_continuation(model, X, y, init_model, continue_training, fit_params, native_key):
    """在覆盖目标模型前取得独立的来源；同实例续训使用生命周期捕获的旧状态。"""
    if not isinstance(continue_training, bool):
        raise ValidationError("continue_training 必须为布尔值")
    native_source = fit_params.pop(native_key, None) if native_key != "init_model" else None
    sources = [source for source in (init_model, native_source) if source is not None]
    if len(sources) > 1 or (sources and continue_training):
        raise ValidationError("init_model、原生续训参数和 continue_training 只能选择一种")
    source = model if continue_training else sources[0] if sources else None
    model.__dict__.pop("_continuation_objective_", None)
    if source is None:
        model.__dict__.pop("continuation_summary_", None)
        return X, y, None
    if isinstance(source, BaseRiskModel):
        fitted = getattr(source, "_fit_was_fitted_", False) if source is model else source.__sklearn_is_fitted__()
        if not fitted or getattr(source, "_model", None) is None:
            raise ValidationError("继续训练需要已拟合的来源模型")
        if type(source) is not type(model):
            raise ValidationError("继续训练的来源必须是相同类型的提升树模型")
        X, y = extract_target(X, y, model.target)
        X = align_model_features(source, X)
        if hasattr(source, "_boost_objective_"):
            model._continuation_objective_ = copy.deepcopy(source._boost_objective_)
        source = source._model
    elif isinstance(source, (str, Path)):
        source = str(source)
    source = copy.deepcopy(source)
    if not isinstance(source, str):
        X, y = extract_target(X, y, model.target)
        count = getattr(source, "n_features_in_", getattr(source, "n_features", None))
        if not count and getattr(source, "feature_names_", None):
            # CatBoost 的 deepcopy/load 可能把 n_features_in_ 置 0，模型内部字段名仍完整。
            count = len(source.feature_names_)
        if count is not None and X.shape[1] != count:
            raise ValidationError("继续训练的特征数量必须与来源模型一致")
    model.continuation_summary_ = {"方式": "继续训练", "原生参数": native_key, "来源类型": type(source).__name__}
    return X, y, source


def validate_ngboost_objective(model, objective):
    import cloudpickle

    def signature(value):
        if value is None or isinstance(value, str) and value in ("binary", "binary:logistic"):
            return "binary"
        return hashlib.sha256(
            cloudpickle.dumps((type(value).__module__, type(value).__qualname__, vars(value)))
        ).hexdigest()

    if hasattr(model, "_continuation_objective_") and signature(objective) != signature(model._continuation_objective_):
        raise ValidationError("NGBoost 续训不能改变训练损失或分布，请沿用来源模型配置")


def continue_ngboost(source, params):
    """NGBoost 在保留旧树、分布和学习率的条件下追加轮数。"""
    from ngboost import NGBClassifier

    if not isinstance(source, NGBClassifier) or not getattr(source, "base_models", None):
        raise ValidationError("NGBoost 续训需要已拟合的 NGBoost/NGBClassifier 对象；文件请先通过 load 加载")
    if params["learning_rate"] != source.learning_rate:
        raise ValidationError("NGBoost 续训必须保持 learning_rate，修改它会改变已有树的预测贡献")
    for name in ("natural_gradient", "minibatch_frac", "col_sample", "tol"):
        if name in params and params[name] != getattr(source, name):
            raise ValidationError(f"NGBoost 续训不能更改 {name}，请沿用来源模型配置")
    if "Base" in params and source.Base.get_params() != params["Base"].get_params():
        raise ValidationError("NGBoost 续训必须保持基学习器配置")
    import inspect

    for key in ("Score", "Dist"):
        expected = params.get(key, inspect.signature(NGBClassifier).parameters[key].default)
        actual = getattr(source, key)
        if (actual.__module__, actual.__qualname__) != (expected.__module__, expected.__qualname__):
            raise ValidationError(f"NGBoost 续训必须保持 {key} 配置")
    source.n_estimators = params["n_estimators"]
    source.verbose = params.get("verbose", source.verbose)
    for name in ("early_stopping_rounds", "validation_fraction"):
        if name in params:
            setattr(source, name, params[name])
    return source


def fit_ngboost(model, X, y, params, continuing):
    """原生 partial_fit 会重估初始分布；续训时固定旧分布，保持旧树前缀预测。"""
    if not continuing:
        return model.fit(X, y, **params)
    if params.get("X_val") is None:
        model.best_val_loss_itr = None
    previous = model.__dict__.get("fit_init_params_to_marginal")
    model.fit_init_params_to_marginal = lambda *args, **kwargs: None
    try:
        return model.partial_fit(X, y, **params)
    finally:
        if previous is None:
            model.__dict__.pop("fit_init_params_to_marginal", None)
        else:
            model.fit_init_params_to_marginal = previous


def reset_xgboost_early_stopping():
    """在新阶段开始前清除检查点的旧早停标记，新回调随后可记录本阶段最优轮数。"""
    from xgboost.callback import TrainingCallback

    class ResetMetadata(TrainingCallback):
        def before_training(self, model):
            model.set_attr(best_iteration=None, best_score=None, best_ntree_limit=None)
            return model

    return ResetMetadata()
