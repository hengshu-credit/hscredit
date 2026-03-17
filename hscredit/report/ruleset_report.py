"""规则集报告模块.

提供规则集综合评估报告功能。
"""

from functools import reduce
from typing import List, Union, Optional
import pandas as pd

from ..core.rules import Rule


def ruleset_report(
    datasets: pd.DataFrame,
    rules: List[Rule],
    target: str = "target",
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    filter_cols: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """规则集报告，评估多个规则的综合效果。

    分析规则集在数据集上的应用效果，展示：
    1. 原始样本的风险数据
    2. 每条规则命中和剩余样本的风险数据
    3. 所有规则合计命中效果

    流程: 原始样本 -> 规则1命中效果 -> 规则1剩余样本 -> 规则2命中效果 -> 规则2剩余样本 ... -> 所有规则合计命中

    :param datasets: 数据集
    :param rules: 规则列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称（支持多标签，传入列表）
    :param dpds: 逾期定义方式（支持多标签，传入列表）
    :param filter_cols: 指定返回的字段列表
    :return: pd.DataFrame，规则集效果评估表。
             单标签时返回单层列结构，多标签时返回多层列结构（MultiIndex）
    """
    datasets = datasets.copy()

    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if len(feature_names_missing) > 0:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()

    # 计算所有规则的并集
    all_rules = reduce(lambda r1, r2: r1 | r2, rules)

    # 获取汇总报告（所有规则的合计）
    table_total = all_rules.report(datasets, target=target, overdue=overdue, dpds=dpds,
                                   filter_cols=filter_cols, margins=True, **kwargs)

    # 重命名分箱列
    if isinstance(table_total.columns, pd.MultiIndex):
        # 多标签情况：找到"规则详情"层级下的"分箱"列
        merge_columns = ['规则分类', '指标名称', '分箱']
        table_total[("规则详情", "分箱")] = ["所有规则", "剩余样本", "原始样本"]
        # 删除不需要的列
        cols_to_drop = [("规则详情", "规则分类"), ("规则详情", "指标名称")]
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])
    else:
        # 单标签情况
        merge_columns = ['规则分类', '指标名称', '分箱']
        table_total["分箱"] = ["所有规则", "剩余样本", "原始样本"]
        # 删除不需要的列
        cols_to_drop = ['规则分类', '指标名称']
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])

    # 添加原始样本行
    if isinstance(table_total.columns, pd.MultiIndex):
        original_row = table_total.loc[table_total[("规则详情", "分箱")] == "原始样本", :]
    else:
        original_row = table_total.loc[table_total["分箱"] == "原始样本", :]
    report = pd.concat([report, original_row])

    # 遍历每个规则
    for rule in rules:
        # 获取当前规则在当前数据集上的报告
        table = rule.report(datasets, target=target, overdue=overdue, dpds=dpds,
                           filter_cols=filter_cols, margins=False, **kwargs)

        # 重命名分箱列为规则表达式和"剩余样本"
        if isinstance(table.columns, pd.MultiIndex):
            table[("规则详情", "分箱")] = [rule.expr, "剩余样本"]
            # 删除不需要的列
            cols_to_drop = [("规则详情", "规则分类"), ("规则详情", "指标名称")]
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])
        else:
            table["分箱"] = [rule.expr, "剩余样本"]
            # 删除不需要的列
            cols_to_drop = ['规则分类', '指标名称']
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])

        report = pd.concat([report, table])

        # 更新数据集为剩余样本
        datasets = datasets[~rule.predict(datasets)]

    # 添加所有规则合计行
    if isinstance(table_total.columns, pd.MultiIndex):
        summary_row = table_total.loc[table_total[("规则详情", "分箱")] == "所有规则", :]
    else:
        summary_row = table_total.loc[table_total["分箱"] == "所有规则", :]
    report = pd.concat([report, summary_row]).reset_index(drop=True)

    return report
