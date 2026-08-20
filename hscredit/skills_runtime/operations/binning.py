"""hsbin 分箱统计与分箱器操作适配器。"""

import inspect
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd

from ...core.binning import OptimalBinning, OptimalBinning2D
from ...excel import ExcelWriter, dataframe2excel
from ...report.feature_analyzer import (
    benchmark_binning_methods,
    feature_efficiency_analysis,
    feature_bin_stats,
    feature_binning_summary,
    feature_group_binning_summary,
)
from ..artifacts import summarize_dataframe
from ..errors import SkillExecutionError
from ..registry import OperationSpec


def _parameters(function, parameters: Mapping[str, Any], *, reserved: Iterable[str] = ("data",)) -> Dict[str, Any]:
    """只接受目标函数显式声明的参数。"""
    signature = inspect.signature(function)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    allowed.difference_update(reserved)
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"操作“{function.__name__}”不支持参数：{'、'.join(unknown)}",
            field=f"parameters.{unknown[0]}",
        )
    return dict(parameters)


def _data(context) -> pd.DataFrame:
    spec = context.request.inputs.get("data")
    if spec is None:
        raise SkillExecutionError(code="SCHEMA_INVALID", message="缺少 inputs.data", field="inputs.data")
    value = context.resolver.resolve(spec)
    if not isinstance(value, pd.DataFrame):
        raise SkillExecutionError(code="SCHEMA_INVALID", message="inputs.data 必须解析为 DataFrame", field="inputs.data")
    if value.empty:
        raise SkillExecutionError(code="SCHEMA_INVALID", message="inputs.data 不能为空", field="inputs.data")
    return value


def _as_parameter_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _label_combinations(parameters: Mapping[str, Any]) -> list:
    overdue_fields = _as_parameter_list(parameters.get("overdue"))
    dpds = _as_parameter_list(parameters.get("dpds"))
    return [{"overdue": overdue, "dpd": dpd} for overdue in overdue_fields for dpd in dpds]


def _summarize_binning_table(table: pd.DataFrame, parameters: Mapping[str, Any]) -> dict:
    summary = summarize_dataframe(table)
    combinations = _label_combinations(parameters)
    if combinations:
        summary["label_combinations"] = combinations
    return summary


def _safe_sheet_name(name: str, used: set) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "-", str(name)).strip(" '") or "数据"
    base = cleaned[:31]
    candidate = base
    counter = 2
    while candidate in used:
        suffix = f"-{counter}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _write_workbook(context, tables: Iterable[Tuple[str, pd.DataFrame]]) -> None:
    staged = context.artifacts.stage_path(f"{context.request.output.name}.xlsx")
    writer = ExcelWriter()
    used = set()
    try:
        for raw_name, table in tables:
            if not isinstance(table, pd.DataFrame):
                continue
            sheet_name = _safe_sheet_name(raw_name, used)
            dataframe2excel(table, writer, sheet_name=sheet_name, index=True, auto_width=True)
        writer.save(str(staged))
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        raise
    context.artifacts.publish(
        staged,
        f"{context.request.output.name}.xlsx",
        artifact_type="excel",
    )


def _feature_bin_stats(context) -> dict:
    params = _parameters(feature_bin_stats, context.request.parameters)
    result = feature_bin_stats(_data(context), **params)
    table = result[0] if isinstance(result, tuple) else result
    _write_workbook(context, [("分箱统计", table)])
    return {"summary": _summarize_binning_table(table, params)}


def _benchmark_binning_methods(context) -> dict:
    params = _parameters(benchmark_binning_methods, context.request.parameters)
    table = benchmark_binning_methods(_data(context), **params)
    _write_workbook(context, [("方法对比", table)])
    return {"summary": summarize_dataframe(table)}


def _flatten_tables(tables: Mapping[str, Any]) -> Iterable[Tuple[str, pd.DataFrame]]:
    for feature, methods in tables.items():
        if isinstance(methods, pd.DataFrame):
            yield str(feature), methods
            continue
        for method, value in methods.items():
            if isinstance(value, pd.DataFrame):
                yield f"{feature}-{method}", value
                continue
            for group, table in value.items():
                if isinstance(table, pd.DataFrame):
                    yield f"{feature}-{method}-{group}", table


def _feature_binning_summary(context) -> dict:
    params = _parameters(feature_binning_summary, context.request.parameters)
    tables, summary = feature_binning_summary(_data(context), **params)
    workbook_tables = [("分箱摘要", summary), *_flatten_tables(tables)]
    _write_workbook(context, workbook_tables)
    return {"summary": _summarize_binning_table(summary, params)}


def _feature_group_binning_summary(context) -> dict:
    params = _parameters(feature_group_binning_summary, context.request.parameters)
    tables, summary = feature_group_binning_summary(_data(context), **params)
    workbook_tables = [("分组摘要", summary), *_flatten_tables(tables)]
    _write_workbook(context, workbook_tables)
    return {"summary": _summarize_binning_table(summary, params)}


def _feature_efficiency_analysis(context) -> dict:
    params = _parameters(
        feature_efficiency_analysis,
        context.request.parameters,
        reserved=("data", "save", "output_dir"),
    )
    comparison = context.artifacts.stage_path(f"{context.request.output.name}-comparison.png")
    trend_dir = context.artifacts.stage_path("efficiency-trends/.anchor").parent
    result = feature_efficiency_analysis(
        _data(context),
        save=str(comparison),
        output_dir=str(trend_dir),
        **params,
    )
    manual_table = result["manual_table"]
    auto_table = result["auto_table"]
    _write_workbook(context, [("手工分箱", manual_table), ("自动分箱", auto_table)])

    rules_path = context.artifacts.write_json(
        f"{context.request.output.name}-rules.json",
        {"manual_rules": result["manual_rules"], "auto_rules": result["auto_rules"]},
    )
    context.artifacts.publish(
        rules_path,
        f"{context.request.output.name}-rules.json",
        artifact_type="json",
    )

    published = set()
    image_paths = [comparison]
    image_paths.extend(result.get("saved_paths", {}).values())
    for path in image_paths:
        source_path = Path(path).resolve()
        if context.artifacts.staging_dir not in source_path.parents:
            raise SkillExecutionError(
                code="ARTIFACT_WRITE_FAILED",
                message=f"效率分析图片越过临时目录：{source_path}",
            )
        if source_path.is_file() and source_path not in published:
            published.add(source_path)
            context.artifacts.publish(source_path, source_path.name, artifact_type="image")

    import matplotlib.pyplot as plt

    figures = [result.get("comparison_figure"), *result.get("trend_figures", {}).values()]
    for figure in figures:
        if figure is not None:
            plt.close(figure)
    return {
        "summary": {
            "feature": result["feature"],
            "target": result["target"],
            "manual_rows": int(len(manual_table)),
            "auto_rows": int(len(auto_table)),
        }
    }


def _fit_inputs(context, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, list, str]:
    data = _data(context)
    features = parameters.pop("features", None)
    if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message="parameters.features 必须是非空字段名列表",
            field="parameters.features",
        )
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise SkillExecutionError(
            code="COLUMN_MISSING",
            message=f"数据缺少分箱字段：{'、'.join(missing)}",
            field="parameters.features",
        )
    target = parameters.pop("target", "target")
    if "y" in context.request.inputs:
        y = context.resolver.resolve(context.request.inputs["y"])
        y = y if isinstance(y, pd.Series) else pd.Series(y, index=data.index, name=str(target))
    else:
        if target not in data.columns:
            raise SkillExecutionError(
                code="COLUMN_MISSING",
                message=f"数据缺少目标字段“{target}”",
                field="parameters.target",
            )
        y = data[target]
    return data[features].copy(), y, features, str(target)


def _constructor_parameters(cls, parameters: Mapping[str, Any], reserved: Iterable[str]) -> Dict[str, Any]:
    signature = inspect.signature(cls)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    params = dict(parameters)
    for name in reserved:
        params.pop(name, None)
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"{cls.__name__} 不支持构造参数：{'、'.join(unknown)}",
            field=f"parameters.{unknown[0]}",
        )
    return params


def _save_binner(context, binner: Any) -> None:
    staged = context.artifacts.stage_path(f"{context.request.output.name}.joblib")
    binner.save_artifact(staged)
    context.artifacts.publish(
        staged,
        f"{context.request.output.name}.joblib",
        artifact_type="hscredit-artifact",
    )


def _binner_tables(binner: Any, transformed: pd.DataFrame) -> list:
    tables = [("转换结果", transformed)]
    if isinstance(binner, OptimalBinning2D):
        tables.append(("二维分箱表", binner.get_bin_table()))
        return tables
    for feature in binner.feature_names_in_:
        tables.append((f"{feature}-分箱表", binner.get_bin_table(feature)))
    return tables


def _optimal_binning_fit(context) -> dict:
    params = dict(context.request.parameters)
    metric = params.pop("metric", "indices")
    X, y, features, target = _fit_inputs(context, params)
    ctor = _constructor_parameters(OptimalBinning, params, {"features", "target"})
    binner = OptimalBinning(target=target, **ctor)
    binner.fit(X, y)
    transformed = binner.transform(X, metric=metric)
    transformed = transformed if isinstance(transformed, pd.DataFrame) else pd.DataFrame(transformed, columns=features)
    _save_binner(context, binner)
    _write_workbook(context, _binner_tables(binner, transformed))
    return {"summary": summarize_dataframe(transformed)}


def _optimal_binning_fit_transform(context) -> dict:
    return _optimal_binning_fit(context)


def _resolve_binner(context, expected_type):
    spec = context.request.inputs.get("binner")
    if spec is None:
        raise SkillExecutionError(code="SCHEMA_INVALID", message="缺少 inputs.binner", field="inputs.binner")
    binner = context.resolver.resolve(spec)
    if not isinstance(binner, expected_type):
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"inputs.binner 必须解析为 {expected_type.__name__}",
            field="inputs.binner",
        )
    return binner


def _optimal_binning_transform(context) -> dict:
    binner = _resolve_binner(context, OptimalBinning)
    data = _data(context)
    params = dict(context.request.parameters)
    features = params.pop("features", list(binner.feature_names_in_))
    metric = params.pop("metric", "indices")
    if params:
        unknown = sorted(params)
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"optimal_binning_transform 不支持参数：{'、'.join(unknown)}",
            field=f"parameters.{unknown[0]}",
        )
    transformed = binner.transform(data[features], metric=metric)
    transformed = transformed if isinstance(transformed, pd.DataFrame) else pd.DataFrame(transformed, columns=features)
    _write_workbook(context, _binner_tables(binner, transformed))
    return {"summary": summarize_dataframe(transformed)}


def _optimal_binning_2d_fit(context) -> dict:
    params = dict(context.request.parameters)
    metric = params.pop("metric", "indices")
    X, y, features, target = _fit_inputs(context, params)
    if len(features) != 2:
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message="OptimalBinning2D 必须且只能接收两个特征",
            field="parameters.features",
        )
    ctor = _constructor_parameters(OptimalBinning2D, params, {"features", "target"})
    binner = OptimalBinning2D(target=target, **ctor)
    binner.fit(X, y, features=features)
    transformed = binner.transform(X, metric=metric)
    _save_binner(context, binner)
    _write_workbook(context, _binner_tables(binner, transformed))
    return {"summary": summarize_dataframe(transformed)}


def _optimal_binning_2d_transform(context) -> dict:
    binner = _resolve_binner(context, OptimalBinning2D)
    data = _data(context)
    params = dict(context.request.parameters)
    features = params.pop("features", [binner.feature_x_, binner.feature_y_])
    metric = params.pop("metric", "indices")
    if params:
        unknown = sorted(params)
        raise SkillExecutionError(
            code="SCHEMA_INVALID",
            message=f"optimal_binning_2d_transform 不支持参数：{'、'.join(unknown)}",
            field=f"parameters.{unknown[0]}",
        )
    transformed = binner.transform(data[features], metric=metric)
    _write_workbook(context, _binner_tables(binner, transformed))
    return {"summary": summarize_dataframe(transformed)}


def register_binning_operations(registry) -> None:
    """登记 hsbin 首批表格操作。"""
    handlers = OrderedDict(
        [
            ("feature_bin_stats", _feature_bin_stats),
            ("benchmark_binning_methods", _benchmark_binning_methods),
            ("feature_binning_summary", _feature_binning_summary),
            ("feature_group_binning_summary", _feature_group_binning_summary),
            ("feature_efficiency_analysis", _feature_efficiency_analysis),
            ("optimal_binning_fit", _optimal_binning_fit),
            ("optimal_binning_fit_transform", _optimal_binning_fit_transform),
            ("optimal_binning_transform", _optimal_binning_transform),
            ("optimal_binning_2d_fit", _optimal_binning_2d_fit),
            ("optimal_binning_2d_transform", _optimal_binning_2d_transform),
        ]
    )
    for name, handler in handlers.items():
        registry.register(OperationSpec("hsbin", name, handler, extras=("skills",)))
