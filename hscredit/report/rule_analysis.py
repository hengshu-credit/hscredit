"""规则分析模块.

提供规则集综合评估与多标签规则分析功能。
"""

from functools import reduce
from typing import Dict, List, Optional, Union

import pandas as pd

from ..core.rules import Rule
from .mining.multi_label import MultiLabelRuleMiner


def _get_detail_group_name(table: pd.DataFrame) -> str:
    """兼容旧版 `规则详情` 和新版 `分箱详情` 顶层分组名。"""
    if not isinstance(table.columns, pd.MultiIndex):
        return ""

    level0_names = set(table.columns.get_level_values(0))
    if "分箱详情" in level0_names:
        return "分箱详情"
    if "规则详情" in level0_names:
        return "规则详情"
    raise KeyError("未找到多层表头中的详情分组列")


def ruleset_analysis(
    datasets: pd.DataFrame,
    rules: List[Rule],
    target: str = "target",
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    filter_cols: Optional[List[str]] = None,
    amount: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """规则集综合分析.

    分析规则集在数据集上的应用效果，展示原始样本、每条规则命中效果、
    各规则剩余样本以及所有规则合计命中效果。

    :param datasets: 数据集
    :param rules: 规则列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称（支持多标签，传入列表）
    :param dpds: 逾期定义方式（支持多标签，传入列表）
    :param filter_cols: 指定返回的字段列表
    :param amount: 金额字段名称，用于金额口径分析
    :return: 规则集效果评估表。单标签时返回单层列结构，多标签时返回多层列结构（MultiIndex）
    """
    datasets = datasets.copy()

    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if len(feature_names_missing) > 0:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()
    all_rules = reduce(lambda r1, r2: r1 | r2, rules)

    table_total = all_rules.report(
        datasets,
        target=target,
        overdue=overdue,
        dpds=dpds,
        filter_cols=filter_cols,
        margins=True,
        amount=amount,
        **kwargs,
    )

    if isinstance(table_total.columns, pd.MultiIndex):
        detail_group = _get_detail_group_name(table_total)
        table_total[(detail_group, "分箱")] = ["所有规则", "剩余样本", "原始样本"]
        cols_to_drop = [(detail_group, "规则分类"), (detail_group, "指标名称")]
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])
        original_row = table_total.loc[table_total[(detail_group, "分箱")] == "原始样本", :]
    else:
        table_total["分箱"] = ["所有规则", "剩余样本", "原始样本"]
        cols_to_drop = ["规则分类", "指标名称"]
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])
        original_row = table_total.loc[table_total["分箱"] == "原始样本", :]
    report = pd.concat([report, original_row])

    for rule in rules:
        table = rule.report(
            datasets,
            target=target,
            overdue=overdue,
            dpds=dpds,
            filter_cols=filter_cols,
            margins=False,
            amount=amount,
            **kwargs,
        )

        if isinstance(table.columns, pd.MultiIndex):
            detail_group = _get_detail_group_name(table)
            table[(detail_group, "分箱")] = [rule.expr, "剩余样本"]
            cols_to_drop = [(detail_group, "规则分类"), (detail_group, "指标名称")]
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])
        else:
            table["分箱"] = [rule.expr, "剩余样本"]
            cols_to_drop = ["规则分类", "指标名称"]
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])

        report = pd.concat([report, table])
        datasets = datasets[~rule.predict(datasets)]

    if isinstance(table_total.columns, pd.MultiIndex):
        detail_group = _get_detail_group_name(table_total)
        summary_row = table_total.loc[table_total[(detail_group, "分箱")] == "所有规则", :]
    else:
        summary_row = table_total.loc[table_total["分箱"] == "所有规则", :]

    report = pd.concat([report, summary_row]).reset_index(drop=True)
    return report


def multi_label_rule_analysis(
    df: pd.DataFrame,
    features: List[str],
    labels: Dict[str, str],
    miner_params: Optional[dict] = None,
    output_path: str = 'rule_analysis.xlsx',
) -> str:
    """多标签规则分析（Excel 输出）.

    报告包含：
    - 规则汇总：各规则在每个标签下的覆盖率/坏率/LIFT/有效性分类
    - 有效性矩阵：行=规则，列=标签，格=LIFT值
    - 规则分类统计：按规则类型分组的汇总统计

    :param df: 输入数据 DataFrame
    :param features: 参与挖掘的特征列表
    :param labels: 标签映射 {中文名: 列名}
    :param miner_params: 传递给 MultiLabelRuleMiner 的额外参数
    :param output_path: 输出 Excel 文件路径
    :return: 输出文件路径
    """
    label_cols = list(labels.values())
    label_names = list(labels.keys())

    params = dict(
        labels=label_cols,
        label_names=label_names,
        min_support=0.02,
        min_lift=1.5,
    )
    if miner_params:
        params.update(miner_params)

    miner = MultiLabelRuleMiner(**params)
    miner.fit(df, features=features)

    all_rules = miner.get_report()
    matrix = miner.get_effectiveness_matrix()

    if len(all_rules) > 0:
        category_stats = all_rules.groupby('规则类型').agg(
            规则条数=('规则', 'count'),
            平均覆盖率=('覆盖率', 'mean'),
        ).reset_index()
    else:
        category_stats = pd.DataFrame(columns=['规则类型', '规则条数', '平均覆盖率'])

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        all_rules.to_excel(writer, sheet_name='规则汇总', index=False)
        matrix.to_excel(writer, sheet_name='有效性矩阵', index=False)
        category_stats.to_excel(writer, sheet_name='规则分类统计', index=False)

    return output_path