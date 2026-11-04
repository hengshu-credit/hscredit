"""分析接口的字段选择、逐行数据绑定及公共参数校验。"""

import inspect
import copy
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps

import numpy as np
import pandas as pd
from .dates import normalize_freq


def _query_source(data):
    """读取查询完成标记，不与编码器的 hscredit_source 字符串标记混用。"""
    if not isinstance(data, pd.DataFrame):
        return None
    source = data.attrs.get("hscredit_query_source")
    if not isinstance(source, Mapping):
        source = data.attrs.get("hscredit_source")
    if not isinstance(source, Mapping) and {"completed", "state", "rows_read"}.issubset(data.attrs):
        source = data.attrs
    return source if isinstance(source, Mapping) else None


def copy_query_source(source, result):
    """在 DataFrame 编码/分箱后保留查询完整性，不覆盖变换本身的标记。"""
    metadata = _query_source(source)
    if metadata is not None and isinstance(result, pd.DataFrame):
        result.attrs["hscredit_query_source"] = {
            key: metadata[key] for key in ("completed", "state", "rows_read", "total_rows") if key in metadata
        }
    return result


def validate_source_completion(data):
    """拒绝将明确标记为未完成的查询结果用于完整报告。"""
    source = _query_source(data)
    if source is not None and source.get("completed") is False:
        from ..exceptions import StateError

        raise StateError("数据来源读取未完成，不能作为完整报告输入；请完成查询或显式确认部分数据口径")


def normalize_features(features=None, *, feature=None, x1=None, x2=None, size=None, required=False):
    """将单字段或一维字段集合规范化；二维字段允许使用成对的 x1/x2。"""
    if feature is not None:
        if features is not None or x1 is not None or x2 is not None:
            raise ValueError("feature、features 和 x1/x2 不能同时指定")
        features = feature
    if x1 is not None or x2 is not None:
        if features is not None:
            raise ValueError("features 与 x1/x2 不能同时指定")
        if x1 is None or x2 is None:
            raise ValueError("x1 和 x2 必须同时指定")
        features = [x1, x2]
    if features is None:
        if required:
            raise ValueError("请提供 features 或 feature")
        return None
    if isinstance(features, (str, int, np.integer)):
        result = [features]
    elif isinstance(features, (list, tuple, np.ndarray, pd.Index)):
        if np.ndim(features) != 1:
            raise ValueError("features 必须是一维字段集合")
        result = list(features)
    else:
        raise TypeError("features 必须是字段名或一维字段集合")
    if (not result and required) or any(
        not isinstance(value, (str, int, np.integer)) or (isinstance(value, str) and not value) for value in result
    ):
        raise ValueError("features 必须包含非空字段名")
    if len(set(result)) != len(result):
        if size == 2:
            raise ValueError("两个特征必须不同，features 不能包含重复字段")
        raise ValueError("features 不能包含重复字段")
    if size is not None and len(result) != size:
        if size == 2:
            raise ValueError("features 必须包含两个特征名（恰好 2 个字段）")
        raise ValueError(f"features 必须恰好包含 {size} 个字段")
    return result


def bind_column(data, value, role):
    """绑定列名或逐行数据，保留索引和扩展类型，不修改输入 DataFrame。"""
    if value is None:
        return data, None
    if isinstance(value, str):
        if value not in data.columns:
            raise KeyError(f"{role} 指定的字段不存在: {value!r}")
        return data, value
    if not isinstance(value, (pd.Series, np.ndarray, list, tuple, pd.Index)):
        raise TypeError(f"{role} 必须是列名或一维 Series、ndarray、list、tuple")
    if isinstance(value, pd.Series):
        if not value.index.equals(data.index):
            if not value.index.is_unique or not data.index.is_unique:
                raise ValueError(f"{role} 含重复索引，必须与作用数据完全同序")
            if len(value) != len(data) or not data.index.isin(value.index).all():
                raise ValueError(f"{role} 索引必须与作用数据完全匹配")
            value = value.reindex(data.index)
        values = value.array
        preferred = value.name if isinstance(value.name, str) and value.name else role
    else:
        values = np.asarray(value)
        preferred = role
    if np.ndim(values) != 1 or len(values) != len(data):
        raise ValueError(f"{role} 必须是一维且与作用数据等长（数据行数: {len(data)}）")
    # 相同名称的数据也不覆盖已有字段；允许标签 lambda 继续读取原始字段。
    column = preferred
    counter = 1
    while column in data.columns:
        candidate = pd.Series(values, index=data.index)
        if candidate.equals(data[column]):
            return data, column
        column = f"{preferred}_{counter}"
        counter += 1
    result = data.copy(deep=False)
    result[column] = pd.Series(values, index=data.index)
    return result, column


def validate_amount_column(data, amount):
    """金额是实际订单字段，明确请求后必须存在且逐行有效。"""
    if amount is None:
        return
    if amount not in data.columns:
        raise ValueError(f"amount 指定的金额字段不存在: {amount!r}")
    series = data[amount]
    if not pd.api.types.is_numeric_dtype(series.dtype):
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
            or pd.api.types.is_timedelta64_dtype(series.dtype)
            or any(isinstance(value, (str, bytes)) for value in series.array)
        ):
            raise ValueError("amount 金额必须为数值，文本、日期或分类字段请先显式转换")
    from ..core.metrics._semantics import validate_amounts

    validate_amounts(data[amount], len(data))


def bind_groups(data, value):
    """多维分组支持字段集合、单组逐行数据或按组名定义的数据字典。"""
    if value is None:
        return data, None
    if isinstance(value, str):
        return data, [value]
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)) and np.ndim(value) == 1:
        if all(isinstance(item, str) and item in data.columns for item in value):
            return data, list(value)
    if isinstance(value, Mapping):
        columns = []
        for name, values in value.items():
            data, column = bind_column(data, values, str(name))
            columns.append(column)
        return data, columns
    data, column = bind_column(data, value, "group")
    return data, [column]


def analysis_inputs(
    data_parameter,
    *,
    roles=(),
    feature_alias=False,
    dimensions=None,
    shared_frames=(),
    multi_group=False,
    normalize_frequency=True,
):
    """在任何投影、排序和筛选之前解析公开字段参数，保留原函数签名。"""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if normalize_frequency and "freq" in signature.parameters and bound.arguments["freq"] is not None:
                bound.arguments["freq"] = normalize_freq(bound.arguments["freq"])
            if "features" in signature.parameters:
                selected = normalize_features(
                    bound.arguments.get("features"),
                    feature=bound.arguments.get("feature") if feature_alias else None,
                    x1=bound.arguments.get("x1") if dimensions == 2 else None,
                    x2=bound.arguments.get("x2") if dimensions == 2 else None,
                    size=dimensions,
                    required=feature_alias,
                )
                if feature_alias:
                    bound.arguments["feature"] = selected
                    bound.arguments["features"] = None
                else:
                    bound.arguments["features"] = selected
                if dimensions == 2:
                    bound.arguments["x1"] = bound.arguments["x2"] = None
            frame = bound.arguments.get(data_parameter)
            if isinstance(frame, pd.DataFrame):
                validate_source_completion(frame)
                for role in roles:
                    value = bound.arguments.get(role)
                    if role == "group" and multi_group:
                        frame, columns = bind_groups(frame, value)
                        bound.arguments[role] = columns
                        continue
                    if isinstance(value, str):
                        if role == "amount":
                            validate_amount_column(frame, value)
                            for other_name in shared_frames:
                                other = bound.arguments.get(other_name)
                                if isinstance(other, pd.DataFrame):
                                    validate_amount_column(other, value)
                        continue
                    source = value.get(data_parameter) if isinstance(value, Mapping) else value
                    if isinstance(value, Mapping) and data_parameter not in value:
                        raise ValueError(f"{role} 缺少 {data_parameter!r} 的数据")
                    frame, column = bind_column(frame, source, role)
                    if role == "amount":
                        validate_amount_column(frame, column)
                    for other_name in shared_frames:
                        other = bound.arguments.get(other_name)
                        if value is None or not isinstance(other, pd.DataFrame):
                            continue
                        if isinstance(value, Mapping):
                            if other_name not in value:
                                raise ValueError(f"{role} 缺少 {other_name!r} 的数据")
                            other_value = value[other_name]
                        else:
                            other_value = value
                        other, other_column = bind_column(other, other_value, role)
                        if role == "amount":
                            validate_amount_column(other, other_column)
                        if other_column != column:
                            if column in other.columns:
                                raise ValueError(f"{role} 在不同数据集中的字段名冲突，请使用相同列名或无名数组")
                            other = other.copy(deep=False)
                            other[column] = other[other_column]
                        bound.arguments[other_name] = other
                    bound.arguments[role] = column
                bound.arguments[data_parameter] = frame
            return function(*bound.args, **bound.kwargs)

        return wrapped

    return decorate


def report_inputs(function):
    """为模型报告绑定辅助数据；临时数据与缓存不污染原报告实例。"""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        original = bound.arguments["self"]
        amount = bound.arguments.get("amount")
        if isinstance(amount, str):
            keys = (
                [original._resolve_dataset_key(bound.arguments["dataset"])]
                if "dataset" in bound.arguments
                else original._datasets
            )
            for key in keys:
                validate_amount_column(original._datasets[key].X, amount)
        roles = {
            role: bound.arguments[role]
            for role in ("date", "group", "amount")
            if role in bound.arguments
            and bound.arguments[role] is not None
            and not isinstance(bound.arguments[role], str)
        }
        if not roles:
            return function(*bound.args, **bound.kwargs)
        original = bound.arguments["self"]
        working = copy.copy(original)
        if hasattr(working, "_init_resource_state"):
            working._init_resource_state()
        working._datasets = dict(original._datasets)
        keys = list(working._datasets)
        if "dataset" in bound.arguments:
            keys = [working._resolve_dataset_key(bound.arguments["dataset"])]
        for role, value in roles.items():
            column = role
            suffix = 1
            while any(column in working._datasets[key].X.columns for key in keys):
                column = f"{role}_{suffix}"
                suffix += 1
            for key in keys:
                dataset = working._datasets[key]
                source = value
                if isinstance(value, Mapping):
                    if key in value:
                        source = value[key]
                    elif dataset.label in value:
                        source = value[dataset.label]
                    else:
                        raise ValueError(f"{role} 缺少数据集 {key!r} 的数据")
                frame, temporary = bind_column(dataset.X, source, role)
                if role == "amount":
                    validate_amount_column(frame, temporary)
                frame = frame.copy(deep=False)
                frame[column] = frame[temporary]
                working._datasets[key] = replace(dataset, X=frame)
            bound.arguments[role] = column
        # 新的辅助数据可能改变金额指标和分组统计，不能复用旧缓存。
        for name, value in vars(working).items():
            if name.endswith("_cache"):
                setattr(working, name, {} if isinstance(value, dict) else None)
        bound.arguments["self"] = working
        try:
            return function(*bound.args, **bound.kwargs)
        finally:
            if hasattr(working, "last_report_result_"):
                original.last_report_result_ = working.last_report_result_

    return wrapped


def binning_options(function=None, *, keep_config=False):
    """合并分箱配置：显式参数优先于 bin_params，后者优先于函数默认值。"""
    if function is None:
        return lambda wrapped: binning_options(wrapped, keep_config=keep_config)
    signature = inspect.signature(function)
    extra_name = next(
        (name for name, parameter in signature.parameters.items() if parameter.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        options = bound.arguments.get("bin_params")
        if options is not None:
            if not isinstance(options, Mapping):
                raise TypeError("bin_params 必须是参数字典")
            for name, value in options.items():
                if name in ("data", "df", "target", "overdue", "dpds", "feature", "features", "bin_params"):
                    raise ValueError(f"bin_params 不允许指定 {name}，请使用顶层参数")
                if name in signature.parameters:
                    bound.arguments.setdefault(name, value)
                elif extra_name is not None:
                    bound.arguments.setdefault(extra_name, {}).setdefault(name, value)
                elif not keep_config:
                    raise TypeError(f"不支持的分箱参数: {name}")
            if not keep_config:
                bound.arguments["bin_params"] = None
        extra = bound.arguments.get(extra_name, {})
        aliases = {
            "n_bins": "max_n_bins",
            "bin_method": "method",
            "binning_method": "method",
            "return_frame": "result",
            "figure_size": "figsize",
            "font_size": "fontsize",
            "date_col": "date",
            "group_col": "group",
            "amount_col": "amount",
            "date_freq": "freq",
            "feature_names": "features",
            "binning_params": "bin_params",
            "binner_kwargs": "bin_params",
            "auto_kwargs": "bin_params",
            "auto_method": "method",
            "hscredit_methods": "methods",
            "manual_rules": "rules",
            "custom_bins": "rules",
        }
        for old, new in aliases.items():
            if (old in extra or (keep_config and old in (options or {}))) and new in signature.parameters:
                raise TypeError(f"{old} 参数已移除，请使用 {new}")
        return function(*bound.args, **bound.kwargs)

    return wrapped
