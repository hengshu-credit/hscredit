"""将平台字段选择转换为 hscredit 的显式标签或逾期口径。"""

import operator
from math import isfinite

import pandas as pd

from hscredit.utils.input_utils import normalize_dpd_values
from hscredit.utils.overdue import apply_missing_target, overdue_label
from hscredit.utils.targets import bind_targets


def validate_columns(data, columns):
    if not columns or len(set(columns)) != len(columns) or any(c not in data for c in columns):
        raise ValueError("请选择数据集中存在且不重复的字段")


def source_columns(config):
    if config.get("label_mode") == "overdue":
        return config.get("overdue", [])
    if isinstance(config.get("target"), dict):
        return list(dict.fromkeys(item["field"] for item in config["target"].values() if isinstance(item, dict)))
    if config.get("kind") == "explore":
        return config.get("targets", [])
    return list(dict.fromkeys([config.get("target"), *config.get("evaluation_targets", [])]))


def overdue_options(data, config):
    columns = source_columns(config)
    validate_columns(data, columns)
    if any(not pd.api.types.is_numeric_dtype(data[c]) for c in columns):
        raise ValueError("逾期字段必须为数值类型")
    if config.get("handle_missing_target") == "raise" and any(data[c].isna().any() for c in columns):
        raise ValueError("目标标签包含缺失值，请调整缺失标签处理")
    if not config.get("dpds"):
        raise ValueError("请至少填写一个逾期阈值")
    thresholds = normalize_dpd_values(config["dpds"])
    if any(value < 0 for value in thresholds):
        raise ValueError("逾期阈值必须为非负数")
    return dict(overdue=columns, dpds=thresholds, overdue_operator=config.get("overdue_operator", ">"), del_grey=config.get("del_grey", False), handle_missing_target=config.get("handle_missing_target", "drop"))


def _dictionary_value(data, name, definition):
    if not isinstance(definition, dict):
        return definition
    field = definition["field"]
    validate_columns(data, [field])
    series = data[field]
    op = definition.get("operator")
    if op is None:
        return series
    value = definition.get("value")
    if not isinstance(value, (bool, str, int, float)) or (isinstance(value, float) and not isfinite(value)):
        raise ValueError(f"标签“{name}”需要有效的比较值")
    comparisons = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
    if op not in ("==", "!=") and (not pd.api.types.is_numeric_dtype(series) or not isinstance(value, (int, float))):
        raise ValueError(f"标签“{name}”的大小比较需要数值字段和数值阈值")
    return comparisons[op](series, value).astype(float).where(series.notna())


def label_map(data, config, *, evaluation=False):
    """保留完整索引。每个口径独立应用缺失和灰样本规则。"""
    dictionary = config.get("target")
    if config.get("label_mode") == "overdue" or isinstance(dictionary, dict):
        if isinstance(dictionary, dict):
            order = config.get("target_order") or list(dictionary)
            definitions = {name: _dictionary_value(data, name, dictionary[name]) for name in order}
            frame, targets = bind_targets(data, target=definitions, del_grey=config.get("del_grey", False), handle_missing_target=config.get("handle_missing_target", "drop"))
        else:
            frame, targets = bind_targets(data, **overdue_options(data, config))
        labels = targets.y_map(frame)
        primary = config.get("primary_index", 0)
        if not 0 <= primary < len(labels):
            raise ValueError("主标签不在已配置的标签中")
        if primary:
            name = targets.names()[primary]
            return {name: labels[name], **labels}
        return labels
    names = source_columns(config)
    if names:
        validate_columns(data, names)
    positives = config.get("positive_labels", {})
    if set(positives).difference(names):
        raise ValueError("坏样本取值只能配置给已选择的标签")
    labels = {}
    for name in names:
        series = data[name]
        values = series.dropna().unique()
        positive = positives.get(name, config.get("positive_label", 1) if name == config.get("target") else 1)
        if pd.api.types.is_integer_dtype(series) and isinstance(positive, str) and positive.lstrip("-").isdigit():
            positive = int(positive)
        if len(values) > 2 or (not evaluation and len(values) != 2):
            raise ValueError(f"标签“{name}”必须恰好包含两个非空类别")
        if not isinstance(positive, (str, int, float, bool)) or (len(values) == 2 and positive not in values):
            raise ValueError(f"坏样本取值不在标签类别中：{name}")
        labels[name] = apply_missing_target(series.eq(positive).fillna(False).astype(float), series.isna(), config.get("handle_missing_target", "drop"))
    return labels


def training_label(labels):
    """训练主标签需同时有好坏样本，其他口径仍可用于单类评估。"""
    name, values = next(iter(labels.items()))
    if values.nunique() != 2:
        raise ValueError("主标签必须恰好包含两个非空类别，请检查坏样本取值和过滤条件")
    return name, values


def iter_labels(data, config, name=None):
    """逐个生成汇总或筛选所需的标签，沿用同一套灰样本和缺失规则。"""
    if config.get("label_mode") == "overdue":
        options = overdue_options(data, config)
        columns = options.pop("overdue")
        thresholds = options.pop("dpds")
        for col in columns:
            projected = data[[col]]
            for dpd in thresholds:
                if name is not None and overdue_label(col, dpd, options["overdue_operator"]) != name:
                    continue
                frame, targets = bind_targets(projected, overdue=col, dpds=dpd, **options)
                yield targets.names()[0], frame[targets.columns[0]]
    else:
        if name is not None:
            if isinstance(config.get("target"), dict) and name in config["target"]:
                config = dict(config, target={name: config["target"][name]}, target_order=[name])
            elif name in config.get("targets", []):
                positives = config.get("positive_labels", {})
                config = dict(config, targets=[name], positive_labels={name: positives[name]} if name in positives else {})
            else:
                return
        yield from label_map(data, config).items()


def preview_counts(data, config):
    for name, values in iter_labels(data, config):
        yield _counts(name, values)


def _counts(name, values):
    valid, bad = int(values.count()), int(values.sum())
    return {"标签": name, "有效样本": valid, "好样本": valid - bad, "坏样本": bad, "剔除样本": len(values) - valid, "坏样本率": bad / valid if valid else None}
