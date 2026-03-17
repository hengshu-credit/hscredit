"""规则引擎.

提供规则定义、评估和规则挖掘功能。
"""

import ast
import re
from enum import Enum
from functools import reduce
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

from ..analysis.feature_analyzer import feature_bin_stats
from ..core.binning import OptimalBinning
from .expr_optimizer import optimize_expr, beautify_expr


def get_columns_from_query(query_str: str) -> List[str]:
    """获取 pandas query 语句使用的列。

    :param query_str: pandas query 支持的查询语句
    :return: query 语句使用的列名列表

    示例:
        >>> get_columns_from_query("age > 18 and income < 5000")
        ['age', 'income']
    """
    tree = ast.parse(query_str, mode='eval')
    columns = set()

    def visit_node(node):
        if isinstance(node, ast.Attribute):
            visit_node(node.value)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            pass
        elif isinstance(node, ast.Name) and node.id not in {'and', 'or', 'not'}:
            columns.add(node.id)
        elif isinstance(node, ast.Call):
            visit_node(node.func)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(columns)


class RuleState(str, Enum):
    """规则状态枚举."""
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(RuntimeError):
    """规则状态错误."""
    pass


class RuleUnAppliedError(RuleStateError):
    """规则未应用错误."""
    pass


class Rule:
    """规则类。

    支持使用 pandas eval 语法的规则定义和评估。

    :param expr: 规则表达式字符串

    示例:
        >>> from hscredit.rules import Rule
        >>> rule1 = Rule("age > 18")
        >>> rule2 = Rule("income > 5000")
        >>> # 规则组合
        >>> combined = rule1 & rule2
        >>> # 应用规则
        >>> result = combined.predict(df)
    """

    def __init__(self, expr: str):
        """
        :param expr: 规则表达式字符串，支持 pandas eval 语法
        """
        self._state = RuleState.INITIALIZED
        self.expr = expr
        self.feature_names_in_ = get_columns_from_query(self.expr)
        self.result_ = None

    def __str__(self):
        return f"Rule({repr(self.expr)})"

    def __repr__(self):
        return f"Rule({repr(self.expr)})"

    def __and__(self, other):
        """规则与操作。"""
        if not isinstance(other, Rule):
            raise TypeError(f"unsupported operand type(s) for &: 'Rule' and '{type(other).__name__}'")
        combined_expr = f"({self.expr}) & ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        return Rule(optimized)

    def __or__(self, other):
        """规则或操作。"""
        if not isinstance(other, Rule):
            raise TypeError(f"unsupported operand type(s) for |: 'Rule' and '{type(other).__name__}'")
        combined_expr = f"({self.expr}) | ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        return Rule(optimized)

    def __invert__(self):
        """规则非操作。"""
        combined_expr = f"~({self.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        return Rule(optimized)

    def __xor__(self, other):
        """规则异或操作。"""
        if not isinstance(other, Rule):
            raise TypeError(f"unsupported operand type(s) for ^: 'Rule' and '{type(other).__name__}'")
        combined_expr = f"({self.expr}) ^ ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        return Rule(optimized)

    def __eq__(self, other):
        """规则相等比较。"""
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        return self.expr == other.expr

    def predict(self, X: DataFrame) -> pd.Series:
        """应用规则进行预测。

        :param X: 输入数据 DataFrame
        :return: 规则匹配结果 Series (bool 类型)
        """
        if not isinstance(X, DataFrame):
            raise ValueError("Rule can only predict on DataFrame.")

        check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")

        # 检查必需的列是否存在
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        result = X.eval(self.expr)
        self.result_ = result
        self._state = RuleState.APPLIED

        return result

    def result(self):
        """获取规则预测结果。"""
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("Invoke `predict` to make a rule applied.")
        return self.result_

    def filter(self, X: DataFrame) -> DataFrame:
        """根据规则过滤数据。

        :param X: 输入数据 DataFrame
        :return: 满足规则的数据子集
        """
        prediction = self.predict(X)
        return X[prediction]

    def report(
        self,
        datasets: pd.DataFrame,
        target: str = "target",
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        del_grey: bool = False,
        desc: str = "",
        filter_cols: Optional[List[str]] = None,
        prior_rules: Optional["Rule"] = None,
        amount: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """规则效果报告表格输出。

        :param datasets: 数据集，需要包含目标变量或逾期天数
        :param target: 目标变量名称，默认 target
        :param desc: 规则相关的描述
        :param filter_cols: 指定返回的字段列表
        :param prior_rules: 先验规则
        :param overdue: 逾期天数字段名称
        :param dpds: 逾期定义方式，逾期天数 > DPD 为 1
        :param del_grey: 是否删除逾期天数 (0, DPD] 的数据
        :param amount: 金额字段名称
        :return: pd.DataFrame，规则效果评估表
        """
        return_cols = ['指标名称', '指标含义', '分箱', '样本总数', '样本占比', '好样本数',
                       '好样本占比', '坏样本数', '坏样本占比', '坏样本率', 'LIFT值', '坏账改善']
        if not desc:
            if '指标含义' in return_cols:
                return_cols.remove('指标含义')

        rule_expr = self.expr

        def _report_one_rule(data, target, desc='', prior_rules=None):
            if prior_rules:
                prior_tables = prior_rules.report(data, target=target, desc=desc, prior_rules=None)
                prior_tables["规则分类"] = "先验规则"
                temp = data[~prior_rules.predict(data)]
                if amount:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(temp), "命中", "未命中"),
                        amount: temp[amount],
                        "target": temp[target].tolist()
                    })
                else:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(temp), "命中", "未命中"),
                        "target": temp[target].tolist()
                    })
            else:
                prior_tables = pd.DataFrame(columns=return_cols)
                if amount:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(data), "命中", "未命中"),
                        amount: data[amount],
                        "target": data[target].tolist()
                    })
                else:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(data), "命中", "未命中"),
                        "target": data[target].tolist()
                    })

            # 使用 OptimalBinning 进行分箱
            try:
                binner = OptimalBinning(name=rule_expr, solver="cp")
                binner.fit(rule_result[rule_expr], rule_result["target"])
                bin_table = binner.binning_table.build()
            except Exception:
                # 如果分箱失败，手动计算统计
                pass

            # 构建分箱结果
            bin_result = pd.DataFrame({
                '指标名称': [rule_expr] * 2,
                '指标含义': [desc] * 2,
                '分箱': ['命中', '未命中'],
            })

            # 计算统计数据
            total = len(rule_result)
            for idx, row in bin_result.iterrows():
                bin_name = row['分箱']
                if bin_name == "命中":
                    matched = rule_result[rule_result[rule_expr] == "命中"]
                else:
                    matched = rule_result[rule_result[rule_expr] == "未命中"]

                bin_total = len(matched)
                bin_bad = matched["target"].sum() if len(matched) > 0 else 0

                bin_result.loc[idx, '样本总数'] = bin_total
                bin_result.loc[idx, '样本占比'] = bin_total / total if total > 0 else 0
                bin_result.loc[idx, '好样本数'] = bin_total - bin_bad
                bin_result.loc[idx, '好样本占比'] = (bin_total - bin_bad) / total if total > 0 else 0
                bin_result.loc[idx, '坏样本数'] = bin_bad
                bin_result.loc[idx, '坏样本占比'] = bin_bad / total if total > 0 else 0
                bin_result.loc[idx, '坏样本率'] = bin_bad / bin_total if bin_total > 0 else 0

            # 计算 LIFT 和坏账改善
            overall_bad_rate = rule_result["target"].mean()
            for idx in bin_result.index:
                bad_rate = bin_result.loc[idx, '坏样本率']
                bin_result.loc[idx, 'LIFT值'] = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            bin_result['坏账改善'] = bin_result['LIFT值'] * bin_result['样本占比'] - bin_result['样本占比']
            bin_result['风险拒绝比'] = bin_result['坏账改善'] / bin_result['样本占比']

            # 计算准确率、精确率、召回率、F1分数
            y_true = rule_result["target"].map({"命中": 1, "未命中": 0}).fillna(0)
            y_pred = rule_result[rule_expr].map({"命中": 1, "未命中": 0}).fillna(0)

            # 过滤掉 NaN 值
            valid_mask = ~(y_true.isna() | y_pred.isna())
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

            metrics = pd.DataFrame({
                "分箱": ["命中", "未命中"],
                "准确率": [
                    accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0,
                    accuracy_score(y_true, 1 - y_pred) if len(y_true) > 0 else 0
                ],
                "精确率": [
                    precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    precision_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0
                ],
                "召回率": [
                    recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    recall_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0
                ],
                "F1分数": [
                    f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    f1_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0
                ],
            })

            table = bin_result.merge(metrics, on="分箱", how="left")

            if prior_rules:
                table.insert(0, "规则分类", ["验证规则"] * len(table))
                table = pd.concat([prior_tables, table])
            else:
                table.insert(0, "规则分类", ["验证规则"] * len(table))

            return table

        # 处理 overdue 参数
        if overdue is not None:
            if not isinstance(overdue, list):
                overdue = [overdue]
            if not isinstance(dpds, list):
                dpds = [dpds] if dpds is not None else [0]

            for col in overdue:
                for d in dpds:
                    _datasets = datasets.copy()
                    _datasets[f"{col}_{d}"] = (_datasets[col] > d).astype(int)

                    if del_grey:
                        _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)").reset_index(drop=True)

                    table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules)
        else:
            table = _report_one_rule(datasets, target, desc=desc, prior_rules=prior_rules)

        if filter_cols:
            if not isinstance(filter_cols, list):
                filter_cols = [filter_cols]
            cols_to_keep = [c for c in table.columns if c in filter_cols or c in ['规则分类', '指标名称', '指标含义', '分箱']]
            if cols_to_keep:
                table = table[cols_to_keep]

        return table

    @staticmethod
    def save(report: pd.DataFrame, excel_writer: Union[str, "ExcelWriter"], sheet_name: str = None,
             merge_column: Optional[List[str]] = None, percent_cols: Optional[List[str]] = None,
             condition_cols: Optional[List[str]] = None, custom_cols: Optional[List[str]] = None,
             custom_format: str = "#,##0", color_cols: Optional[List[str]] = None,
             start_col: int = 2, start_row: int = 2, **kwargs):
        """保存规则结果至 Excel。

        :param report: 规则报告 DataFrame
        :param excel_writer: Excel 文件路径或 ExcelWriter 对象
        :param sheet_name: 工作表名称
        :param merge_column: 合并列
        :param percent_cols: 百分比列
        :param condition_cols: 条件格式列
        :param custom_cols: 自定义列
        :param custom_format: 自定义格式
        :param color_cols: 颜色列
        :param start_col: 起始列
        :param start_row: 起始行
        :return: (end_row, end_col)
        """
        from ..report.excel import ExcelWriter as HWExcelWriter, dataframe2excel

        if isinstance(excel_writer, str):
            writer = HWExcelWriter(excel_writer)
        else:
            writer = excel_writer

        # 处理列名（支持多级索引）
        if merge_column:
            merge_column = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in merge_column) or (not isinstance(c, tuple) and c in merge_column)]

        if percent_cols:
            percent_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in percent_cols) or (not isinstance(c, tuple) and c in percent_cols)]

        if condition_cols:
            condition_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in condition_cols) or (not isinstance(c, tuple) and c in condition_cols)]

        if custom_cols:
            custom_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in custom_cols) or (not isinstance(c, tuple) and c in custom_cols)]

        if color_cols:
            color_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in color_cols) or (not isinstance(c, tuple) and c in color_cols)]

        end_row, end_col = dataframe2excel(
            report, writer, sheet_name=sheet_name,
            merge_column=merge_column, percent_cols=percent_cols,
            condition_cols=condition_cols, custom_cols=custom_cols,
            custom_format=custom_format, color_cols=color_cols,
            start_col=start_col, start_row=start_row, **kwargs
        )

        if isinstance(excel_writer, str):
            writer.save(excel_writer)

        return end_row, end_col


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
    2. 每条规则命中后剩余样本的风险数据
    3. 最终规则集命中效果合计

    流程: 原始样本 -> 规则1命中效果 -> 剩余样本 -> 规则2命中效果 -> 剩余样本 ... -> 规则集命中效果合计

    :param datasets: 数据集
    :param rules: 规则列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称
    :param dpds: 逾期定义方式
    :param filter_cols: 指定返回的字段列表
    :return: pd.DataFrame，规则集效果评估表
    """
    datasets = datasets.copy()

    # 检查必需的列
    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if feature_names_missing:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()

    # 复制原始数据
    original_data = datasets.copy()
    remaining_data = datasets.copy()

    # 定义计算统计信息的函数
    def calc_stats(data, base_data):
        """计算数据集的统计信息"""
        total = len(data)
        base_total = len(base_data)
        stats = {
            '样本总数': total,
            '样本占比': total / base_total if base_total > 0 else 0,
        }
        if target in data.columns:
            bad_count = data[target].sum()
            stats['坏样本数'] = bad_count
            stats['坏样本率'] = bad_count / total if total > 0 else 0
            stats['好样本数'] = total - bad_count
            stats['好样本占比'] = (total - bad_count) / total if total > 0 else 0
        return stats

    # 1. 添加原始样本统计
    original_stats = calc_stats(original_data, original_data)
    original_row = pd.DataFrame([{
        '规则顺序': '原始样本',
        '分箱': '全部',
        **original_stats
    }])
    report = pd.concat([report, original_row], ignore_index=True)

    # 2. 遍历每个规则，计算命中效果和剩余样本
    for i, rule in enumerate(rules):
        # 在当前剩余样本上计算规则命中效果
        current_data = remaining_data.copy()
        rule_pred = rule.predict(current_data)
        hit_data = current_data[rule_pred]  # 命中的样本
        new_remaining_data = current_data[~rule_pred]  # 未命中的样本（新的剩余样本）

        # 计算命中样本的统计
        if len(hit_data) > 0:
            hit_stats = calc_stats(hit_data, original_data)
            hit_row = pd.DataFrame([{
                '规则顺序': f'规则{i+1}: {rule.expr}',
                '分箱': '命中',
                **hit_stats
            }])
            report = pd.concat([report, hit_row], ignore_index=True)

        # 计算剩余样本的统计
        if len(new_remaining_data) > 0:
            remaining_stats = calc_stats(new_remaining_data, original_data)
            remaining_row = pd.DataFrame([{
                '规则顺序': f'规则{i+1}: {rule.expr}',
                '分箱': '剩余样本',
                **remaining_stats
            }])
            report = pd.concat([report, remaining_row], ignore_index=True)

        # 更新剩余数据
        remaining_data = new_remaining_data

    # 3. 添加汇总行（所有规则合计命中效果）
    if len(rules) > 0:
        # 计算规则集整体命中的样本
        all_rules = reduce(lambda r1, r2: r1 | r2, rules)
        all_pred = all_rules.predict(original_data)
        total_hit_data = original_data[all_pred]

        if len(total_hit_data) > 0:
            total_hit_stats = calc_stats(total_hit_data, original_data)
            total_hit_row = pd.DataFrame([{
                '规则顺序': '规则集汇总',
                '分箱': '合计命中',
                **total_hit_stats
            }])
            report = pd.concat([report, total_hit_row], ignore_index=True)

    # 处理 filter_cols
    if filter_cols:
        if not isinstance(filter_cols, list):
            filter_cols = [filter_cols]
        cols_to_keep = [c for c in ['规则顺序', '分箱'] if c in report.columns]
        cols_to_keep += [c for c in filter_cols if c in report.columns]
        report = report[cols_to_keep]

    return report
