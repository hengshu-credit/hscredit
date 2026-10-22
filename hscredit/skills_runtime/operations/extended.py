"""hscredit 其余 Skills 的统一、白名单化操作适配器。"""


from __future__ import annotations

from .binning import _target_parameters, _overdue_summary

import inspect
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import pandas as pd

from ...excel import ExcelWriter, dataframe2excel
from ..artifacts import summarize_dataframe
from ..errors import SkillExecutionError
from ..registry import OperationSpec
from .binning import _data, _parameters, _safe_sheet_name
from .common import resolve


def _input(context, name: str, required: bool = True) -> Any:
    spec = context.request.inputs.get(name)
    if spec is None:
        if required:
            raise SkillExecutionError(code="SCHEMA_INVALID", message=f"缺少 inputs.{name}", field=f"inputs.{name}")
        return None
    return context.resolver.resolve(spec)


def _data_xy(context, parameters: Mapping[str, Any]):
    frame = _data(context)
    target = parameters.get("target", "target")
    y = _input(context, "y", required=False)
    if y is None:
        if target not in frame.columns:
            raise SkillExecutionError(
                code="COLUMN_MISSING", message=f"数据缺少目标字段“{target}”", field="parameters.target"
            )
        y = frame[target]
        X = frame.drop(columns=[target])
    else:
        X = frame
    features = parameters.get("features")
    if features is not None:
        X = X.loc[:, list(features)]
    return X, y


def _write_table(context, table: pd.DataFrame, sheet: str = "数据") -> None:
    staged = context.artifacts.stage_path(f"{context.request.output.name}.xlsx")
    writer = ExcelWriter()
    dataframe2excel(table, writer, sheet_name=sheet, index=True, auto_width=True)
    writer.save(str(staged))
    context.artifacts.publish(staged, f"{context.request.output.name}.xlsx", artifact_type="excel")


def _write_tables(context, tables: Mapping[str, pd.DataFrame]) -> None:
    staged = context.artifacts.stage_path(f"{context.request.output.name}.xlsx")
    writer = ExcelWriter()
    used = set()
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            dataframe2excel(table, writer, sheet_name=_safe_sheet_name(name, used), index=True, auto_width=True)
    writer.save(str(staged))
    context.artifacts.publish(staged, f"{context.request.output.name}.xlsx", artifact_type="excel")


def _artifact(context, value: Any, name: str, artifact_type: str = "hscredit-artifact") -> None:
    staged = context.artifacts.stage_path(name)
    if not hasattr(value, "save_artifact"):
        raise SkillExecutionError(
            code="ARTIFACT_WRITE_FAILED", message=f"对象不支持 hscredit 制品保存：{type(value).__name__}"
        )
    value.save_artifact(staged)
    context.artifacts.publish(staged, name, artifact_type=artifact_type)


_MODEL_NAMES = {
    "RandomForest": "RandomForest",
    "ExtraTrees": "ExtraTrees",
    "GradientBoosting": "GradientBoosting",
    "SVM": "SVM",
    "DecisionTreeClassifier": "DecisionTreeClassifier",
    "LogisticRegression": "LogisticRegression",
    "ScoreCard": "ScoreCard",
    "RoundScoreCard": "RoundScoreCard",
    "ProbabilityScoreCard": "ProbabilityScoreCard",
    "RuleSet": "RuleSet",
    "RulesClassifier": "RulesClassifier",
}


def _model_class(name: str):
    if name not in _MODEL_NAMES:
        raise SkillExecutionError(code="SCHEMA_INVALID", message=f"不允许的模型名称“{name}”", field="parameters.model")
    import hscredit.core.models as models

    return getattr(models, _MODEL_NAMES[name])


def _model_fit(context) -> dict:
    params = dict(context.request.parameters)
    model_name = params.pop("model", "LogisticRegression")
    model_params = params.pop("model_params", {})
    fit_params = params.pop("fit_params", {})
    X, y = _data_xy(context, params)
    unknown = set(params) - {"target", "features", "save_model", "predict", "predict_proba", "n_jobs"}
    if unknown:
        raise SkillExecutionError(code="SCHEMA_INVALID", message=f"模型操作不支持参数：{'、'.join(sorted(unknown))}")
    model = _model_class(model_name)(**model_params)
    model.fit(X, y, **fit_params)
    result = pd.DataFrame(index=X.index)
    if params.get("predict", True) and hasattr(model, "predict"):
        result["预测"] = model.predict(X)
    if params.get("predict_proba", True) and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["预测概率"] = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
    _write_table(context, result, "预测结果")
    if params.get("save_model", True):
        _artifact(context, model, f"{context.request.output.name}.joblib")
    return {"summary": {"model": model_name, "rows": len(X), "features": list(X.columns)}}


def _model_tune(context) -> dict:
    params = dict(context.request.parameters)
    model_type = params.pop("model_type", "lr")
    n_trials = int(params.pop("n_trials", 10))
    X, y = _data_xy(context, params)
    from hscredit.core.models import AutoTuner

    tuner = AutoTuner.create(model_type, **params)
    tuner.fit(X, y, n_trials=n_trials)
    best = tuner.get_best_model()
    _artifact(context, tuner, f"{context.request.output.name}-tuner.joblib")
    _artifact(context, best, f"{context.request.output.name}-model.joblib")
    return {"summary": {"model_type": model_type, "n_trials": n_trials, "best_params": tuner.best_params_}}


def _model_calibrate(context) -> dict:
    params = dict(context.request.parameters)
    model = _input(context, "model")
    X, y = _data_xy(context, params)
    from hscredit.core.models import calibrate_model

    calibrated = calibrate_model(
        model, X, y, method=params.get("method", "platt"), target=params.get("target", "target")
    )
    _artifact(context, calibrated, f"{context.request.output.name}.joblib")
    return {"summary": {"method": params.get("method", "platt"), "rows": len(X)}}


def _model_explain(context) -> dict:
    model = _input(context, "model")
    X, _ = _data_xy(context, context.request.parameters)
    from hscredit.core.models import model_explain_report

    table = model_explain_report(
        model,
        X,
        importance_type=context.request.parameters.get("importance_type", "gain"),
        top_n=context.request.parameters.get("top_n"),
    )
    _write_table(context, table, "模型解释")
    return {"summary": summarize_dataframe(table)}


def _loss_info(context) -> dict:
    name = context.request.parameters.get("loss", "FocalLoss")
    import hscredit.core.models as models

    if not hasattr(models, name) or not name.endswith(("Loss", "Metric")):
        raise SkillExecutionError(
            code="SCHEMA_INVALID", message=f"不允许的损失或指标名称“{name}”", field="parameters.loss"
        )
    cls = getattr(models, name)
    obj = cls(**context.request.parameters.get("params", {}))
    return {"summary": {"name": name, "class": f"{cls.__module__}.{cls.__name__}", "repr": repr(obj)}}


def register_model_operations(registry) -> None:
    for name, handler, extras in [
        ("model_fit", _model_fit, ("skills",)),
        ("model_tune", _model_tune, ("skills", "tune")),
        ("model_calibrate", _model_calibrate, ("skills",)),
        ("model_explain", _model_explain, ("skills",)),
        ("loss_info", _loss_info, ("skills",)),
    ]:
        registry.register(OperationSpec("hsmodel", name, handler, extras=extras))


def _rule_objects(value):
    from hscredit.core.rules import Rule

    if isinstance(value, str):
        return [Rule(value)]
    if isinstance(value, list):
        return [item if isinstance(item, Rule) else Rule(item) for item in value]
    raise SkillExecutionError(
        code="SCHEMA_INVALID", message="rules 必须是表达式字符串或字符串列表", field="parameters.rules"
    )


def _rule_report(context) -> dict:
    data = _data(context)
    params = dict(context.request.parameters)
    params = _target_parameters(context, params)
    rules = _rule_objects(params.pop("rules"))
    from ...utils.targets import bind_targets

    data, targets = bind_targets(
        data, params.get("target"), params.pop("overdue", None), params.pop("dpds", None),
        params.get("del_grey", False), params.get("overdue_operator", ">"), default_target="target",
    )
    params["target"] = targets
    reports = {rule.name or f"规则{i + 1}": rule.report(data, **params) for i, rule in enumerate(rules)}
    _write_tables(context, reports)
    return {"summary": {"rules": len(rules), "rows": len(data)}}


def _ruleset_analysis(context) -> dict:
    from hscredit.report import ruleset_analysis

    params = dict(context.request.parameters)
    params = _target_parameters(context, params)
    rules = _rule_objects(params.pop("rules"))
    table = ruleset_analysis(_data(context), rules, **params)
    _write_table(context, table, "规则集分析")
    return {"summary": summarize_dataframe(table)}


def _rule_swap_analysis(context) -> dict:
    from hscredit.report import rule_swap_analysis

    params = dict(context.request.parameters)
    params = _target_parameters(context, params)
    for key in ("rules_in", "rules_out", "rules_base"):
        if key in params:
            params[key] = _rule_objects(params[key])
    result = rule_swap_analysis(_data(context), **params)
    _write_tables(context, result)
    return {"summary": {"tables": list(result)}}


def _rule_flow_report(context) -> dict:
    from hscredit.core.rules import RuleFlow

    params = dict(context.request.parameters)
    rules = _rule_objects(params.pop("rules"))
    flow = RuleFlow(rules, mode=params.pop("mode", "serial"), name=params.pop("name", None))
    table = flow.report(_data(context), **params)
    _write_table(context, table, "规则流报告")
    return {"summary": summarize_dataframe(table)}


def register_rule_operations(registry) -> None:
    for name, handler in [
        ("rule_report", _rule_report),
        ("ruleset_analysis", _ruleset_analysis),
        ("rule_swap_analysis", _rule_swap_analysis),
        ("rule_flow_report", _rule_flow_report),
    ]:
        registry.register(OperationSpec("hsrule", name, handler, extras=("skills",)))


_SELECTOR_NAMES = {
    "TypeSelector",
    "RegexSelector",
    "NullSelector",
    "ModeSelector",
    "CardinalitySelector",
    "VarianceSelector",
    "CorrSelector",
    "VIFSelector",
    "IVSelector",
    "KSSelector",
    "LiftSelector",
    "PSISelector",
    "FeatureImportanceSelector",
    "NullImportanceSelector",
    "RFESelector",
    "SequentialFeatureSelector",
    "StepwiseSelector",
    "BorutaSelector",
    "MutualInfoSelector",
    "Chi2Selector",
    "FTestSelector",
    "StabilityAwareSelector",
    "ScorecardFeatureSelection",
    "CompositeFeatureSelector",
}


def _selector_fit(context) -> dict:
    params = dict(context.request.parameters)
    name = params.pop("selector", "IVSelector")
    if name not in _SELECTOR_NAMES:
        raise SkillExecutionError(
            code="SCHEMA_INVALID", message=f"不允许的筛选器名称“{name}”", field="parameters.selector"
        )
    import hscredit.core.selectors as selectors

    selector_params = params.pop("selector_params", {})
    selector = getattr(selectors, name)(**selector_params)
    X, y = _data_xy(context, params)
    selected = selector.fit_transform(X, y)
    selected = selected if isinstance(selected, pd.DataFrame) else pd.DataFrame(selected)
    _write_table(context, selected, "筛选结果")
    report = selector.get_selection_report() if hasattr(selector, "get_selection_report") else {}
    report_path = context.artifacts.write_json(f"{context.request.output.name}-report.json", report)
    context.artifacts.publish(report_path, f"{context.request.output.name}-report.json", artifact_type="json")
    return {"summary": {"selector": name, "selected": list(selected.columns), "report": report}}


def register_selector_operations(registry) -> None:
    registry.register(OperationSpec("hsselect", "selector_fit", _selector_fit, extras=("skills",)))


_VIZ_NAMES = {
    "bin_plot",
    "bin_2d_plot",
    "corr_plot",
    "ks_plot",
    "hist_plot",
    "psi_plot",
    "dataframe_plot",
    "distribution_plot",
    "bin_trend_plot",
    "batch_bin_trend_plot",
    "bin_overdues_plot",
    "plot_weights",
    "plot_model_feature_importance",
    "plot_model_sample_shap",
    "roc_plot",
    "pr_plot",
    "lift_plot",
    "gain_plot",
    "confusion_matrix_plot",
    "calibration_plot",
    "score_dist_plot",
    "score_bin_plot",
    "threshold_analysis_plot",
    "strategy_compare_plot",
    "vintage_plot",
    "feature_importance_plot",
    "approval_rate_trend_plot",
    "bad_rate_trend_plot",
    "metric_comparison_plot",
    "variable_iv_plot",
    "variable_woe_trend_plot",
    "variable_psi_heatmap",
    "variable_importance_grouped_plot",
    "variable_missing_badrate_plot",
    "score_ks_plot",
    "score_distribution_comparison_plot",
    "score_badrate_bin_plot",
    "score_lift_plot",
    "score_approval_badrate_curve",
    "rule_swap_plot",
    "strategy_simulation_plot",
    "feature_trend_by_time",
    "feature_drift_comparison",
    "feature_effectiveness_by_segment",
    "feature_cross_heatmap",
    "population_drift_monitor",
    "segment_scorecard_comparison",
    "plot_tree",
    "plot_tree_matplotlib",
    "plot_tree_pyecharts",
    "plot_tree_graphviz",
    "tree_leaf_comparison_plot",
}


def _viz_plot(context) -> dict:
    params = dict(context.request.parameters)
    plot_name = params.pop("plot")
    if plot_name not in _VIZ_NAMES:
        raise SkillExecutionError(
            code="SCHEMA_INVALID", message=f"不允许的可视化名称“{plot_name}”", field="parameters.plot"
        )
    import hscredit.core.viz as viz

    function = getattr(viz, plot_name)
    kwargs = dict(params.pop("kwargs", {}))
    signature = inspect.signature(function)

    def target_references(value):
        if isinstance(value, Mapping):
            if set(value) == {"input"}:
                return {value["input"]}
            return set().union(*(target_references(item) for item in value.values()))
        return set()

    references = set().union(*(target_references(kwargs.get(name)) for name in ("target", "date", "group", "amount")))
    for key, spec in context.request.inputs.items():
        if key in ("target", "date", "group", "amount") or (key in references and key not in signature.parameters):
            continue
        kwargs[key] = context.resolver.resolve(spec)
    kwargs = _target_parameters(context, kwargs)
    fmt = (context.request.output.format or "png").lower()
    if fmt not in {"png", "svg"}:
        raise SkillExecutionError(code="SCHEMA_INVALID", message="图片格式只支持 png 或 svg", field="output.format")
    staged = context.artifacts.stage_path(f"{context.request.output.name}.{fmt}")
    if "save" in signature.parameters:
        kwargs["save"] = str(staged)
    result = function(**kwargs)
    figures = []
    if hasattr(result, "savefig"):
        figures = [(context.request.output.name, result)]
    elif isinstance(result, dict):
        figures = [(str(key), value) for key, value in result.items() if hasattr(value, "savefig")]
    if not staged.is_file() and figures:
        figures[0][1].savefig(staged, format=fmt, bbox_inches="tight")
    if staged.is_file():
        context.artifacts.publish(staged, f"{context.request.output.name}.{fmt}", artifact_type="image")
    for key, figure in figures:
        if key != context.request.output.name:
            child = context.artifacts.stage_path(f"{context.request.output.name}-{key}.{fmt}")
            figure.savefig(child, format=fmt, bbox_inches="tight")
            context.artifacts.publish(child, f"{context.request.output.name}-{key}.{fmt}", artifact_type="image")
        plt.close(figure)
    operator_parameter = signature.parameters.get("overdue_operator")
    label_summary = {} if kwargs.get("bin_table") is not None else _overdue_summary(
        kwargs, default_operator=operator_parameter.default if operator_parameter else ">",
        paired=plot_name in {"bin_overdues_plot", "distribution_plot"},
    )
    return {"summary": {"plot": plot_name, "format": fmt, "figures": len(figures), **label_summary}}


def register_viz_operations(registry) -> None:
    registry.register(OperationSpec("hsviz", "plot", _viz_plot, extras=("skills",)))


def _excel_export(context) -> dict:
    table = _data(context)
    params = dict(context.request.parameters)
    sheet_name = params.pop("sheet_name", "数据")
    params.pop("save", None)
    params.pop("excel_writer", None)
    staged = context.artifacts.stage_path(f"{context.request.output.name}.xlsx")
    writer = ExcelWriter(**resolve(context, params.pop("writer_params", {})))
    dataframe2excel(table, writer, sheet_name=sheet_name, index=True, **resolve(context, params))
    writer.save(str(staged))
    context.artifacts.publish(staged, f"{context.request.output.name}.xlsx", artifact_type="excel")
    return {"summary": summarize_dataframe(table)}


def _excel_workbook(context) -> dict:
    tables = {}
    for name, spec in context.request.inputs.items():
        value = context.resolver.resolve(spec)
        if not isinstance(value, pd.DataFrame):
            raise SkillExecutionError(
                code="SCHEMA_INVALID", message=f"inputs.{name} 必须是 DataFrame", field=f"inputs.{name}"
            )
        tables[name] = value
    params = resolve(context, dict(context.request.parameters))
    writer_params = params.pop("writer_params", {})
    staged = context.artifacts.stage_path(f"{context.request.output.name}.xlsx")
    writer = ExcelWriter(**writer_params)
    used = set()
    for name, table in tables.items():
        dataframe2excel(table, writer, sheet_name=_safe_sheet_name(name, used), index=True, **params)
    writer.save(str(staged))
    context.artifacts.publish(staged, f"{context.request.output.name}.xlsx", artifact_type="excel")
    return {"summary": {"sheets": list(tables)}}


def register_excel_operations(registry) -> None:
    registry.register(OperationSpec("hsexcel", "dataframe_to_excel", _excel_export, extras=("skills",)))
    registry.register(OperationSpec("hsexcel", "workbook", _excel_workbook, extras=("skills",)))


def _capabilities(context) -> dict:
    from ..registry import ensure_default_operations

    registry = ensure_default_operations()
    requested = context.request.parameters.get("skill")
    skills = [requested] if requested else ["hsbin", "hsreport", "hsmodel", "hsrule", "hsselect", "hsviz", "hsexcel"]
    return {"summary": {"skills": {skill: list(registry.list_operations(skill)) for skill in skills}}}


def register_hscredit_operations(registry) -> None:
    registry.register(OperationSpec("hscredit", "capabilities", _capabilities, extras=("skills",)))
