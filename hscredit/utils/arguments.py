"""分析接口的字段选择、逐行数据绑定及公共参数校验。"""

import inspect
import copy
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps

import numpy as np
import pandas as pd


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
    data_parameter, *, roles=(), feature_alias=False, dimensions=None, shared_frames=(), multi_group=False
):
    """在任何投影、排序和筛选之前解析公开字段参数，保留原函数签名。"""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
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
                for role in roles:
                    value = bound.arguments.get(role)
                    if role == "group" and multi_group:
                        frame, columns = bind_groups(frame, value)
                        bound.arguments[role] = columns
                        continue
                    if isinstance(value, str):
                        continue
                    source = value.get(data_parameter) if isinstance(value, Mapping) else value
                    if isinstance(value, Mapping) and data_parameter not in value:
                        raise ValueError(f"{role} 缺少 {data_parameter!r} 的数据")
                    frame, column = bind_column(frame, source, role)
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


def binning_options(function):
    """合并分箱配置：显式参数优先于 bin_params，后者优先于函数默认值。"""
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
                else:
                    raise TypeError(f"不支持的分箱参数: {name}")
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
        }
        for old, new in aliases.items():
            if old in extra and new in signature.parameters:
                raise TypeError(f"{old} 参数已移除，请使用 {new}")
        return function(*bound.args, **bound.kwargs)

    return wrapped
