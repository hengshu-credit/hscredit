"""模型训练、推理、调参、校准、解释及组件制品操作。"""

import inspect

import pandas as pd

from ...core.models._contracts import positive_probability
from ..registry import OperationSpec
from .common import artifact, dataset, frame, input_value, invalid, json_artifact, options, resolve, table, tables
from .components import COMPONENTS, MODELS, component_class, create_component

DATA = ("target", "features")
PREDICTIONS = ("predict", "predict_proba", "predict_score")


def _prediction(model, X, method, params):
    if method not in PREDICTIONS or not hasattr(model, method):
        raise invalid(f"模型不支持预测方法：{method}")
    value = getattr(model, method)(X, **params)
    if method == "predict_proba":
        value = positive_probability(value, getattr(model, "classes_", None), positive_class=1)
    return frame(
        value, index=X.index, label={"predict": "预测", "predict_proba": "坏样本概率", "predict_score": "评分"}[method]
    )


def model_fit(context):
    p = options(
        context,
        (
            *DATA,
            "model",
            "model_params",
            "fit_params",
            "save_model",
            "predict",
            "predict_proba",
            "prediction_params",
            "n_jobs",
        ),
    )
    name = p.get("model", "LogisticRegression")
    if name not in MODELS:
        raise invalid(f"模型名称未登记：{name}")
    ctor = dict(p.get("model_params", {}))
    if "n_jobs" in p:
        ctor.setdefault("n_jobs", p["n_jobs"])
    model = create_component(context, name, ctor)
    X, y = dataset(context, p)
    fit_params = resolve(context, p.get("fit_params", {}))
    if name in {"ScoreCard", "RoundScoreCard"} and "input_type" not in fit_params:
        raise invalid("ScoreCard 拟合须显式选择 fit_params.input_type：raw 或 woe，避免输入含义混淆")
    model.fit(X, y, **fit_params)
    prediction_params = resolve(context, p.get("prediction_params", {}))
    if name in {"ScoreCard", "RoundScoreCard"}:
        prediction_params.setdefault("input_type", fit_params["input_type"])
    outputs = []
    for method in ("predict", "predict_proba"):
        if p.get(method, True) and hasattr(model, method):
            outputs.append(_prediction(model, X, method, prediction_params))
    if outputs:
        tables(context, {"预测结果": pd.concat(outputs, axis=1)})
    if p.get("save_model", True):
        artifact(context, model)
    return {"summary": {"model": name, "rows": len(X), "features": list(X.columns)}}


def model_predict(context):
    p = options(context, (*DATA, "method", "prediction_params"))
    model = input_value(context, "model")
    X, _ = dataset(context, p, require_y=False, model=model)
    return table(
        context,
        _prediction(model, X, p.get("method", "predict_proba"), resolve(context, p.get("prediction_params", {}))),
        "预测结果",
    )


def model_evaluate(context):
    from ...report import ModelReport

    p = options(context, (*DATA, "report_params"))
    model = input_value(context, "model")
    X, y = dataset(context, p, model=model)
    report = ModelReport(model, X_train=X, y_train=y, **resolve(context, p.get("report_params", {})))
    return table(context, report.summary(), "模型评估")


def compare_models(context):
    from ...report import compare_models as compare

    p = options(context, (*DATA, "models", "n_jobs", "parallel_backend", "parallel_config"))
    models = resolve(context, p.pop("models"))
    X, y = dataset(context, p)
    test = input_value(context, "test", required=False)
    kwargs = {k: p[k] for k in ("n_jobs", "parallel_backend", "parallel_config") if k in p}
    if test is not None:
        from ...core.models._contracts import extract_target

        X_test, y_test = extract_target(test, target=p.get("target", "target"))
        kwargs.update(X_test=X_test.loc[:, X.columns], y_test=y_test)
    return table(context, compare(models, X, y, **kwargs), "模型对比")


def model_tune(context):
    from ...core.models import AutoTuner, ModelTuner

    p = dict(context.request.parameters)
    wrapper = {"model_type", "model", "tuner", "tuner_params", "n_trials", "timeout", "features", "target"}
    allowed = wrapper | set(inspect.signature(ModelTuner).parameters)
    if set(p) - allowed:
        raise invalid(f"未知调参参数：{sorted(set(p) - allowed)}")
    X, y = dataset(context, p)
    ctor = {key: value for key, value in p.items() if key not in wrapper}
    ctor.update(p.get("tuner_params", {}))
    ctor = resolve(context, ctor)
    # 远端存储和磁盘保留需要独立管理生命周期；本入口使用内存 Study。
    if ctor.get("storage") is not None or ctor.get("artifact_dir") is not None or ctor.get("retention") == "disk":
        raise invalid("本操作支持内存 Study；disk/storage 请在 Python 创建 tuner 后通过 inputs.tuner 交接")
    if p.get("tuner", "AutoTuner") == "ModelTuner":
        tuner = ModelTuner(component_class(p.get("model", "LogisticRegression")), **ctor)
    else:
        tuner = AutoTuner.create(p.get("model_type", "lr"), **ctor)
    tuner.fit(X, y, n_trials=p.get("n_trials", 10), timeout=p.get("timeout"), show_progress_bar=False)
    best = tuner.get_best_model()
    best.tuner = tuner
    artifact(context, tuner, "-tuner")
    artifact(context, best, "-model")
    history = pd.DataFrame(tuner.optimization_history_)
    tables(context, {"调参历史": history})
    return {"summary": {"best_params": tuner.best_params_, "best_score": tuner.best_score_, "trials": len(history)}}


def model_calibrate(context):
    from ...core.models import calibrate_model

    p = options(context, (*DATA, "method", "calibrator_params"))
    model = input_value(context, "model")
    X, y = dataset(context, p, model=model)
    calibrated = calibrate_model(
        model,
        X,
        y,
        method=p.get("method", "platt"),
        target=p.get("target", "target"),
        **resolve(context, p.get("calibrator_params", {})),
    )
    artifact(context, calibrated)
    return table(context, _prediction(calibrated, X, "predict_proba", {}), "校准概率")


def model_explain(context):
    from ...core.models import model_explain_report

    p = options(context, (*DATA, "importance_type", "top_n", "normalize"))
    model = input_value(context, "model")
    X, _ = dataset(context, p, require_y=False, model=model)
    return table(context, model_explain_report(model, X, **{k: v for k, v in p.items() if k not in DATA}), "模型解释")


def component_create(context):
    p = options(context, ("component", "params", "loss"))
    name = p.get("component", p.get("loss", "FocalLoss"))
    if name not in COMPONENTS:
        raise invalid(f"组件未登记：{name}")
    component = create_component(context, name, p.get("params", {}))
    artifact(context, component)
    summary = {"name": name, "class": f"{type(component).__module__}.{type(component).__name__}"}
    json_artifact(context, summary, "-component")
    return {"summary": summary}


def register_model_operations(registry):
    for name, handler in {
        "model_fit": model_fit,
        "model_predict": model_predict,
        "model_evaluate": model_evaluate,
        "compare_models": compare_models,
        "model_tune": model_tune,
        "model_calibrate": model_calibrate,
        "model_explain": model_explain,
        "component_create": component_create,
        "loss_info": component_create,
    }.items():
        registry.register(
            OperationSpec("hsmodel", name, handler, extras=("skills", "tune") if name == "model_tune" else ("skills",))
        )
