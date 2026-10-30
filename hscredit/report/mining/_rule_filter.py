"""手工二叉树与多叉树共用的训练集规则筛选。"""

import operator
from numbers import Integral, Real

import numpy as np
import pandas as pd

_OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def normalize_rule_filter(config):
    """None 不筛选；显式字典启用叶节点、LIFT >= 1 的默认条件。"""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("rule_filter 必须为字典或 None")
    defaults = dict(leaf_only=True, metric="lift", threshold=1.0, operator=">=", min_samples=None, max_samples=None)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"rule_filter 不支持的配置：{sorted(unknown)}")
    result = {**defaults, **config}
    if not isinstance(result["leaf_only"], (bool, np.bool_)):
        raise ValueError("rule_filter.leaf_only 必须为布尔值")
    metric = result["metric"]
    if not isinstance(metric, str) or metric not in {"lift", "badrate"}:
        raise ValueError("rule_filter.metric 必须为 lift 或 badrate")
    if not isinstance(result["operator"], str) or result["operator"] not in _OPERATORS:
        raise ValueError("rule_filter.operator 必须为 >、>=、< 或 <=")
    value = result["threshold"]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("rule_filter.threshold 必须为有限数值")
    for key in ("min_samples", "max_samples"):
        value = result[key]
        if value is None:
            continue
        valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and np.isfinite(value)
        valid = valid and (value >= 0 if isinstance(value, Integral) else 0 <= value <= 1)
        if not valid:
            raise ValueError(f"rule_filter.{key} 必须为非负整数（个数）、[0, 1] 浮点数（比例）或 None")
    return result


def select_rule_nodes(table, config):
    """使用未舍入的训练订单指标选节点，根节点只作总体基准。"""
    config = normalize_rule_filter(config)
    multi = isinstance(table.columns, pd.MultiIndex)
    meta = lambda name: ("分箱详情", name) if multi else name
    if config is None:
        return table[meta("节点编号")].tolist()
    if multi:
        raise ValueError("规则筛选需要单一 0/1 训练目标；报告仍可评估多个标签")
    counts = table["样本总数"]
    total = table.loc[table["节点编号"].eq(0), "样本总数"].iloc[0]

    def limit(value, fallback):
        return fallback if value is None else (value if isinstance(value, Integral) else value * total)

    minimum = limit(config["min_samples"], 0)
    maximum = limit(config["max_samples"], np.inf)
    if minimum > maximum:
        raise ValueError("rule_filter.min_samples 不能大于 max_samples（按训练样本总数换算）")
    mask = table["节点编号"].ne(0) & counts.ge(minimum) & counts.le(maximum)
    if config["leaf_only"]:
        mask &= table["是否叶子"].eq("是")
    metric_column = {"lift": "LIFT值", "badrate": "坏样本率"}[config["metric"]]
    if config["metric"] == "badrate" and config["threshold"] > 1:
        metric_column = "坏样本数"
    mask &= _OPERATORS[config["operator"]](table[metric_column], config["threshold"])
    return table.loc[mask, "节点编号"].tolist()


def filter_rule_tables(tree, result, enabled=True, training=None):
    """各数据集沿用训练筛选的节点；不更改树结构、预测路由或评估指标。"""
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("apply_rule_filter 必须为布尔值")
    config = normalize_rule_filter(getattr(tree, "rule_filter", None)) if enabled else None
    if config is None:
        return result
    if tree._data is None:
        raise ValueError("规则筛选需要已保存的训练数据，请先 fit")
    if training is None:
        training = tree.get_rule_table(long_format=False, round_digits=False, apply_rule_filter=False)
    selected = select_rule_nodes(training, config)

    def subset(table):
        if isinstance(table, dict):
            return {name: subset(value) for name, value in table.items()}
        if isinstance(table, (list, tuple)):
            return [subset(value) for value in table]
        key = ("分箱详情", "节点编号") if isinstance(table.columns, pd.MultiIndex) else "节点编号"
        return table.loc[table[key].isin(selected)].reset_index(drop=True)

    return subset(result)
