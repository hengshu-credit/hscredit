"""规则引擎.

提供基于 pandas eval/query 语法的规则定义、评估、组合与效果评估能力。核心类
:class:`Rule` 支持用 ``&``/``|``/``~``/``^`` 组合多条规则，并通过 :meth:`Rule.report`
输出风控口径的命中率、坏账率、LIFT、风险拒绝比等指标；辅助函数
:func:`get_columns_from_query` / :func:`get_rule_columns` 从表达式解析所引用的列名。

**参考样例**

>>> from hscredit.core.rules import Rule, get_columns_from_query, get_rule_columns
>>> rule1 = Rule("age > 18", name="成年规则")
>>> rule2 = Rule("income > 5000", name="高收入规则")
>>> combined = rule1 & rule2          # 复合规则
>>> get_columns_from_query("age > 18 and income < 5000")
['age', 'income']

**引用**

- pandas eval/query 表达式语法：
  https://pandas.pydata.org/docs/user_guide/enhancingperf.html#expression-evaluation-via-eval
"""

import ast
import re
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

from ..binning import OptimalBinning
from .expr_optimizer import optimize_expr, beautify_expr
from ...exceptions import FeatureNotFoundError, InputTypeError, StateError

if TYPE_CHECKING:
    from ...excel import ExcelWriter


def _replace_backtick_columns(query_str: str) -> Tuple[str, Dict[str, str]]:
    """将 pandas 反引号列名替换为 Python AST 可解析的占位符。"""
    replacements = {}

    def replace(match):
        placeholder = f"__hscredit_backtick_col_{len(replacements)}"
        replacements[placeholder] = match.group(1)
        return placeholder

    return re.sub(r"`([^`]+)`", replace, query_str), replacements


def get_columns_from_query(query_str: str) -> List[str]:
    """获取 pandas query 语句使用的列。

    解析 query 语法树，提取其中涉及的全部列名，返回去重排序后的列表。

    **参数**

    :param query_str: pandas query 支持的查询语句，如 "age > 18 and income < 5000"
    :return: query 语句使用的列名列表（去重后按字母排序）

    **参考样例**

    >>> from hscredit.core.rules import get_columns_from_query
    >>> get_columns_from_query("age > 18 and income < 5000")
    ['age', 'income']
    >>> get_columns_from_query("salary >= 3000 & age.between(20, 60)")
    ['age', 'salary']
    >>> get_columns_from_query("`衡枢鉴真分老客版` < 600 & `逾期(天)` > 7")
    ['衡枢鉴真分老客版', '逾期(天)']
    """

    parsed_query, backtick_columns = _replace_backtick_columns(query_str)
    tree = ast.parse(parsed_query, mode='eval')

    columns = set()
    reserved_names = {'and', 'or', 'not', 'True', 'False', 'None'}

    # 反引号包裹的列名（pandas query/eval 语法，用于含空格/特殊字符/非标识符的列名），
    # ast.parse 无法解析反引号，需先用正则提取并替换为占位标识符
    backtick_names = re.findall(r'`([^`]+)`', query_str)
    columns.update(name.strip() for name in backtick_names)
    cleaned = re.sub(r'`[^`]+`', ' _hscredit_bt_ ', query_str)

    def visit_node(node):
        if isinstance(node, ast.Attribute):
            visit_node(node.value)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            pass
        elif isinstance(node, ast.Name) and node.id not in reserved_names:
            columns.add(backtick_columns.get(node.id, node.id))
        elif isinstance(node, ast.Call):
            visit_node(node.func)

    try:
        tree = ast.parse(cleaned, mode='eval')
        for node in ast.walk(tree):
            visit_node(node)
    except SyntaxError:
        # 极端情况下（如复杂的非标识符表达式）退化为仅返回反引号列名
        pass

    columns.discard('_hscredit_bt_')
    return sorted(columns)


class RuleState(str, Enum):
    """规则生命周期状态枚举。

    继承 ``str``，可直接与字符串比较。:class:`Rule` 用它标记是否已执行过 predict，
    从而约束 :meth:`Rule.result` 等依赖结果的方法。

    **枚举值**

    - ``INITIALIZED`` (``"initialized"``)：规则已创建但尚未调用 :meth:`Rule.predict`，
      此时无可用结果
    - ``APPLIED`` (``"applied"``)：规则已对某数据集执行过 :meth:`Rule.predict`，
      ``result_`` 中存有最近一次命中结果
    """
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(StateError):
    """规则状态异常基类。

    当在规则不允许的状态下调用方法时抛出，继承自
    :class:`~hscredit.exceptions.StateError`。
    """
    pass


class RuleUnAppliedError(RuleStateError):
    """规则尚未应用异常。

    在未先调用 :meth:`Rule.predict` 的情况下访问 :meth:`Rule.result` 等依赖预测
    结果的方法时抛出。
    """
    pass


class Rule:
    """规则类。

    支持使用 pandas eval 语法的规则定义和评估，支持 &（与）、|（或）、
    ~（非）、^（异或）等运算符组合多个规则为复合规则。

    **属性**

    :param expr: 规则表达式字符串
    :param name: 规则名称，用于标识和展示，默认为None（使用表达式作为名称）
    :param description: 规则描述，默认为空字符串
    :param weight: 规则权重，用于规则集分类器，默认为1.0
    :ivar feature_names_in_: 从表达式中解析出的特征名列表
    :ivar result_: 最近一次 predict 的结果 Series
    :ivar _state: 当前规则状态（initialized/applied）

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [20, 30, 40], 'income': [3000, 8000, 12000]})
    >>> rule1 = Rule("age > 18", name="成年规则", description="判断用户是否成年")
    >>> rule2 = Rule("income > 5000", name="高收入规则")
    >>> # 规则组合
    >>> combined = rule1 & rule2
    >>> # 应用规则
    >>> result = combined.predict(df)
    >>> print(result)
    """

    def __init__(
        self,
        expr: str,
        name: Optional[str] = None,
        description: str = "",
        weight: float = 1.0
    ):
        """初始化规则。

        **参数**

        :param expr: 规则表达式字符串，支持 pandas eval 语法，
            如 "age > 18 and income < 5000"
        :param name: 规则名称，用于标识和展示，默认为None（使用表达式作为名称）
        :param description: 规则描述，默认为空字符串
        :param weight: 规则权重，用于规则集分类器，默认为1.0
        """
        self._state = RuleState.INITIALIZED
        self.expr = expr
        self.name = name or expr
        self.description = description
        self.weight = weight
        self.feature_names_in_ = get_columns_from_query(self.expr)
        self.result_ = None

    def __str__(self):
        return f"Rule({repr(self.expr)})"

    def __repr__(self):
        return f"Rule({repr(self.expr)})"

    def __and__(self, other):
        """规则"与"（AND）组合，对应 Python ``&`` 运算符。

        将两条规则用 ``&`` 拼接为复合规则，命中条件为两者同时成立；表达式会经
        :func:`beautify_expr` 美化与 :func:`optimize_expr` 化简，名称拼为
        ``(A)_AND_(B)``，权重取两者较大值。

        :param other: 另一条 :class:`Rule` 规则
        :return: 组合后的新 :class:`Rule`（不修改原规则）
        :raises InputTypeError: ``other`` 不是 :class:`Rule` 时

        **参考样例**

        >>> r = Rule("age > 18") & Rule("income > 5000")
        >>> r.expr
        'age > 18 & income > 5000'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"& 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        combined_expr = f"({self.expr}) & ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, 'name', None) or self.expr
        other_name = getattr(other, 'name', None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_AND_({other_name})",
            description=f"{self.description} 且 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight)
        )

    def __or__(self, other):
        """规则"或"（OR）组合，对应 Python ``|`` 运算符。

        将两条规则用 ``|`` 拼接为复合规则，命中条件为两者任一成立；表达式经美化与
        化简，名称拼为 ``(A)_OR_(B)``，权重取两者较大值。

        :param other: 另一条 :class:`Rule` 规则
        :return: 组合后的新 :class:`Rule`（不修改原规则）
        :raises InputTypeError: ``other`` 不是 :class:`Rule` 时

        **参考样例**

        >>> r = Rule("age > 18") | Rule("income > 5000")
        >>> r.expr
        'age > 18 | income > 5000'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"| 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        combined_expr = f"({self.expr}) | ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, 'name', None) or self.expr
        other_name = getattr(other, 'name', None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_OR_({other_name})",
            description=f"{self.description} 或 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight)
        )

    def __invert__(self):
        """规则"非"（NOT）取反，对应 Python ``~`` 运算符。

        对规则整体取反得到新规则，命中条件为原规则不成立；名称拼为 ``NOT_(A)``，
        权重保持不变。

        :return: 取反后的新 :class:`Rule`（不修改原规则）

        **参考样例**

        >>> r = ~Rule("age > 18")
        >>> r.expr
        '~(age > 18)'
        """
        combined_expr = f"~({self.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        return Rule(
            optimized,
            name=f"NOT_({self.name})",
            description=f"非: {self.description}" if self.description else "",
            weight=self.weight
        )

    def __xor__(self, other):
        """规则"异或"（XOR）组合，对应 Python ``^`` 运算符。

        命中条件为两条规则恰有一条成立。由于 pandas eval 不支持布尔 ``^``，内部以等价
        表达式 ``(A & ~B) | (~A & B)`` 实现；名称拼为 ``(A)_XOR_(B)``，权重取较大值。

        :param other: 另一条 :class:`Rule` 规则
        :return: 组合后的新 :class:`Rule`（不修改原规则）
        :raises InputTypeError: ``other`` 不是 :class:`Rule` 时

        **参考样例**

        >>> r = Rule("age > 18") ^ Rule("income > 5000")
        >>> r.expr
        '(age > 18 & ~(income > 5000)) | (~(age > 18) & income > 5000)'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"^ 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        # pandas eval 不支持布尔 ^（BitXor），用等价的 (a & ~b) | (~a & b) 表达异或
        combined_expr = (
            f"(({self.expr}) & ~({other.expr})) | (~({self.expr}) & ({other.expr}))"
        )
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, 'name', None) or self.expr
        other_name = getattr(other, 'name', None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_XOR_({other_name})",
            description=f"{self.description} 异或 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight)
        )

    def __eq__(self, other):
        """规则相等比较，对应 Python ``==`` 运算符。

        两条规则当且仅当 ``expr`` 表达式字符串完全相同时视为相等（与 :meth:`__hash__`
        保持一致，因此 :class:`Rule` 可放入 set / 作为 dict 键）。

        :param other: 另一条 :class:`Rule` 规则
        :return: 两者表达式是否相同（bool）
        :raises InputTypeError: ``other`` 不是 :class:`Rule` 时
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"输入必须是 Rule 类型，实际类型为 {type(other).__name__}")
        return self.expr == other.expr

    def predict(self, X: DataFrame) -> pd.Series:
        """应用规则进行预测。

        使用 pandas eval 对 DataFrame 执行规则表达式，返回命中的布尔 Series。

        **参数**

        :param X: 输入数据 DataFrame（必须包含规则表达式中引用的全部列）
        :return: 规则匹配结果 Series（布尔类型，True表示命中）
        :raises InputTypeError: X 不是 DataFrame 时
        :raises FeatureNotFoundError: X 缺少规则表达式所需的列时

        **参考样例**

        >>> from hscredit.core.rules import Rule
        >>> import pandas as pd
        >>> df = pd.DataFrame({'age': [20, 30, 40], 'income': [3000, 8000, 12000]})
        >>> rule = Rule("age > 25 and income > 5000")
        >>> rule.predict(df)
        """
        if not isinstance(X, DataFrame):
            raise InputTypeError("Rule 只能对 DataFrame 执行预测")

        # 检查必需的列是否存在（规则集基于pandas eval实现，支持各种数据类型）
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise FeatureNotFoundError(f"输入数据缺少列: {missing_cols}")

        result = X.eval(self.expr)
        self.result_ = result
        self._state = RuleState.APPLIED

        return result

    def result(self):
        """获取规则预测结果。

        返回最近一次调用 predict() 的结果。必须先调用 predict() 才能使用此方法。

        :return: 最近一次预测的布尔 Series
        :raises RuleUnAppliedError: 尚未调用 predict() 时

        **参考样例**

        >>> from hscredit.core.rules import Rule
        >>> import pandas as pd
        >>> df = pd.DataFrame({'age': [20, 30, 40]})
        >>> rule = Rule("age > 25")
        >>> rule.predict(df)
        >>> rule.result()
        """
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("规则尚未应用，请先调用 predict()")
        return self.result_

    def filter(self, X: DataFrame) -> DataFrame:
        """根据规则过滤数据。

        应用规则后返回满足条件（命中）的数据子集。

        **参数**

        :param X: 输入数据 DataFrame
        :return: 满足规则的数据子集 DataFrame
        :raises InputTypeError: X 不是 DataFrame 时
        :raises FeatureNotFoundError: X 缺少规则表达式所需的列时

        **参考样例**

        >>> from hscredit.core.rules import Rule
        >>> import pandas as pd
        >>> df = pd.DataFrame({'age': [20, 30, 40], 'name': ['A', 'B', 'C']})
        >>> rule = Rule("age > 25")
        >>> rule.filter(df)
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
        margins: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """规则效果报告表格输出。

        将规则命中与否作为二分类，对数据集计算统计指标，
        包括样本数、坏账率、LIFT值、风险拒绝比、精确率、召回率、F1分数等。
        支持金额口径分析与多标签（不同逾期天数定义）联合输出。

        **参数**

        :param datasets: 数据集 DataFrame，需要包含目标变量列或逾期天数列
        :param target: 目标变量列名，默认为"target"，0=好样本，1=坏样本
        :param overdue: 逾期天数字段名（可选，传入时以逾期天数>DPD定义坏样本，
            支持多标签多DPD联合分析）
        :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本，默认为0；
            传入列表时支持多DPD联合分析
        :param del_grey: 是否删除逾期天数在(0, DPD]区间内的灰度样本，默认为False
        :param desc: 规则描述，用于报告的"指标含义"列，默认为空字符串
        :param filter_cols: 指定返回的字段列表（可选）
        :param prior_rules: 先验规则（可选），先对数据应用先验规则排除部分样本，
            再对当前规则进行评估
        :param amount: 金额字段名（可选），传入时以金额口径而非样本数口径进行统计
        :param margins: 是否在报告末尾添加合计行，默认为False
        :return: 规则效果评估表DataFrame。
            单标签时返回单层列结构，多标签时返回多层列结构（MultiIndex）；
            列包括：规则分类、指标名称、指标含义、分箱、样本总数、样本占比、
            好样本数、好样本占比、坏样本数、坏样本占比、坏账率、LIFT值、
            坏账改善、风险拒绝比、准确率、精确率、召回率、F1分数
        :raises FeatureNotFoundError: 数据集缺少规则表达式所需的列时
        :raises KeyError: overdue字段在数据集中不存在时

        **参考样例**

        >>> from hscredit.core.rules import Rule
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [20, 30, 40, 50],
        ...     'income': [3000, 8000, 12000, 5000],
        ...     'target': [0, 1, 1, 0]
        ... })
        >>> rule = Rule("age > 25 and income > 5000")
        >>> report = rule.report(df, target='target')
        >>> print(report)
        """
        detail_group_name = "分箱详情"
        return_cols = ['指标名称', '指标含义', '分箱', '样本总数', '样本占比', '好样本数',
                       '好样本占比', '坏样本数', '坏样本占比', '坏样本率', 'LIFT值', '坏账改善']
        if not desc:
            if '指标含义' in return_cols:
                return_cols.remove('指标含义')

        rule_expr = self.expr

        def _report_one_rule(data, target, desc='', prior_rules=None):
            """生成单标签的规则报告。
            
            直接手动计算统计指标，规则只有命中/未命中两种状态，不需要分箱器。
            支持金额口径分析，当传入 amount 参数时会计算金额相关指标。
            列名与 feature_bin_stats 保持一致，便于统一处理。
            """
            if prior_rules:
                prior_tables = prior_rules.report(data, target=target, desc=desc, prior_rules=None, 
                                                  margins=margins, amount=amount)
                prior_tables["规则分类"] = "先验规则"
                temp = data[~prior_rules.predict(data)]
                if amount is not None and amount in temp.columns:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(temp), "命中", "未命中"),
                        amount: temp[amount].values,
                        "target": temp[target].tolist()
                    })
                else:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(temp), "命中", "未命中"),
                        "target": temp[target].tolist()
                    })
            else:
                prior_tables = pd.DataFrame(columns=return_cols)
                if amount is not None and amount in data.columns:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(data), "命中", "未命中"),
                        amount: data[amount].values,
                        "target": data[target].tolist()
                    })
                else:
                    rule_result = pd.DataFrame({
                        rule_expr: np.where(self.predict(data), "命中", "未命中"),
                        "target": data[target].tolist()
                    })

            # 判断是否使用金额口径
            has_amount = amount is not None and amount in rule_result.columns
            
            if has_amount:
                # 金额口径：使用金额替代样本数，但列名保持统一
                total_amount = rule_result[amount].sum()
                total_good_amount = rule_result[rule_result["target"] == 0][amount].sum()
                total_bad_amount = rule_result[rule_result["target"] == 1][amount].sum()
                overall_bad_rate = total_bad_amount / total_amount if total_amount > 0 else 0
                
                rows = []
                for bin_name in ["命中", "未命中"]:
                    matched = rule_result[rule_result[rule_expr] == bin_name]
                    bin_amount = matched[amount].sum()
                    bin_bad_amount = matched[matched["target"] == 1][amount].sum()
                    bin_good_amount = bin_amount - bin_bad_amount
                    
                    # 使用统一的列名，但存储金额数据
                    sample_ratio = bin_amount / total_amount if total_amount > 0 else 0
                    good_ratio = bin_good_amount / total_good_amount if total_good_amount > 0 else 0
                    bad_ratio = bin_bad_amount / total_bad_amount if total_bad_amount > 0 else 0
                    bad_rate = bin_bad_amount / bin_amount if bin_amount > 0 else 0
                    lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
                    # 拒绝后剩余样本坏样本率 = (全量坏金额 - 当前箱坏金额) / (全量金额 - 当前箱金额)
                    other_bad = total_bad_amount - bin_bad_amount
                    other_total = total_amount - bin_amount
                    remaining_bad_rate = other_bad / other_total if other_total > 0 else 0
                    bad_decrease = (overall_bad_rate - remaining_bad_rate) / overall_bad_rate if overall_bad_rate > 0 else 0
                    
                    row = {
                        '指标名称': rule_expr,
                        '指标含义': desc if desc else '',
                        '分箱': bin_name,
                        '样本总数': bin_amount,  # 金额口径：存储金额
                        '样本占比': sample_ratio,  # 金额占比
                        '好样本数': bin_good_amount,  # 好金额
                        '好样本占比': good_ratio,  # 好金额占比
                        '坏样本数': bin_bad_amount,  # 坏金额
                        '坏样本占比': bad_ratio,  # 坏金额占比
                        '坏样本率': bad_rate,  # 金额口径坏账率
                        'LIFT值': lift,
                        '坏账改善': bad_decrease,
                    }
                    rows.append(row)
                
                table = pd.DataFrame(rows)
                
                # 计算合计行
                total_row_data = {
                    '指标名称': '合计',
                    '指标含义': '',
                    '分箱': '合计',
                    '样本总数': total_amount,
                    '样本占比': 1.0,
                    '好样本数': total_good_amount,
                    '好样本占比': 1.0,
                    '坏样本数': total_bad_amount,
                    '坏样本占比': 1.0,
                    '坏样本率': overall_bad_rate,
                    'LIFT值': 1.0,
                    '坏账改善': 0.0,
                }
            else:
                # 样本数口径（原有逻辑）
                total = len(rule_result)
                total_good = rule_result["target"].eq(0).sum()
                total_bad = rule_result["target"].eq(1).sum()
                overall_bad_rate = rule_result["target"].mean() if total > 0 else 0
                
                rows = []
                for bin_name in ["命中", "未命中"]:
                    matched = rule_result[rule_result[rule_expr] == bin_name]
                    bin_total = len(matched)
                    bin_bad = matched["target"].sum() if bin_total > 0 else 0
                    bin_good = bin_total - bin_bad
                    
                    sample_ratio = bin_total / total if total > 0 else 0
                    good_ratio = bin_good / total_good if total_good > 0 else 0
                    bad_ratio = bin_bad / total_bad if total_bad > 0 else 0
                    bad_rate = bin_bad / bin_total if bin_total > 0 else 0
                    lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
                    # 拒绝后剩余样本坏样本率 = (全量坏样本数 - 当前箱坏样本数) / (全量样本数 - 当前箱样本数)
                    other_bad = total_bad - bin_bad
                    other_total = total - bin_total
                    remaining_bad_rate = other_bad / other_total if other_total > 0 else 0
                    bad_decrease = (overall_bad_rate - remaining_bad_rate) / overall_bad_rate if overall_bad_rate > 0 else 0
                    
                    row = {
                        '指标名称': rule_expr,
                        '指标含义': desc if desc else '',
                        '分箱': bin_name,
                        '样本总数': bin_total,
                        '样本占比': sample_ratio,
                        '好样本数': bin_good,
                        '好样本占比': good_ratio,
                        '坏样本数': bin_bad,
                        '坏样本占比': bad_ratio,
                        '坏样本率': bad_rate,
                        'LIFT值': lift,
                        '坏账改善': bad_decrease,
                    }
                    rows.append(row)
                
                table = pd.DataFrame(rows)
                
                # 计算合计行
                y_true = rule_result["target"]
                y_pred = rule_result[rule_expr].map({"命中": 1, "未命中": 0})
                
                total_row_data = {
                    '指标名称': '合计',
                    '指标含义': '',
                    '分箱': '合计',
                    '样本总数': total,
                    '样本占比': 1.0,
                    '好样本数': total_good,
                    '好样本占比': 1.0 if total_good > 0 else 0.0,
                    '坏样本数': total_bad,
                    '坏样本占比': 1.0 if total_bad > 0 else 0.0,
                    '坏样本率': rule_result["target"].mean(),
                    'LIFT值': 1.0,
                    '坏账改善': 0.0,
                }
            
            # 添加风险拒绝比 = 坏账改善 / 样本占比
            table["风险拒绝比"] = np.divide(
                table["坏账改善"],
                table["样本占比"],
                out=np.zeros(len(table), dtype=float),
                where=table["样本占比"].to_numpy() != 0,
            )

            # 计算准确率、精确率、召回率、F1分数
            y_true = rule_result["target"]
            y_pred = rule_result[rule_expr].map({"命中": 1, "未命中": 0})

            metrics_data = {
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
            }
            metrics = pd.DataFrame(metrics_data)

            # 合并指标
            table = table.merge(metrics, on="分箱", how="left")

            # 如果需要合计行，添加合计
            if margins:
                total_row_data.update({
                    '风险拒绝比': 0.0,
                    '准确率': accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0,
                    '精确率': precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    '召回率': recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    'F1分数': f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                })
                table = pd.concat([table, pd.DataFrame([total_row_data])], ignore_index=True)

            if not desc and "指标含义" in table.columns:
                table = table.drop(columns=["指标含义"])

            if prior_rules:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))
                table = pd.concat([prior_tables, table], ignore_index=True)
            else:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))

            return table

        # 处理 overdue 参数 - 构建多层级列结构
        if overdue is not None:
            if not isinstance(overdue, list):
                overdue = [overdue]
            if not isinstance(dpds, list):
                dpds = [dpds] if dpds is not None else [0]
            # 处理dpds中的None值并转换为整数
            dpds = [0 if d is None else int(d) for d in dpds]
            # 去重，保留顺序，保留第一个出现的
            seen = set()
            dpds_unique = []
            for d in dpds:
                if d not in seen:
                    seen.add(d)
                    dpds_unique.append(d)
            dpds = dpds_unique

            # 确定merge_columns（多标签时需要的列）
            if isinstance(del_grey, bool) and del_grey:
                merge_columns = ["规则分类", "指标名称", "分箱"]
            else:
                merge_columns = ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]
            if "指标含义" in return_cols and "指标含义" not in merge_columns:
                merge_columns = ["指标含义"] + merge_columns

            # 遍历所有逾期标签组合
            for i, col in enumerate(overdue):
                for j, d in enumerate(dpds):
                    _datasets = datasets.copy()
                    _datasets[f"{col}_{d}"] = (_datasets[col] > d).astype(int)

                    if isinstance(del_grey, bool) and del_grey:
                        _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)").reset_index(drop=True)

                    # 生成当前标签的报告
                    _table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules)

                    if i == 0 and j == 0:
                        # 第一个标签，作为基础表
                        table = _table
                        # 转换为多层级列结构
                        table.columns = pd.MultiIndex.from_tuples(
                            [(detail_group_name, c) if c in merge_columns else (f"{col} {d}+", c) for c in table.columns]
                        )
                    else:
                        # 后续标签，合并到基础表
                        _table.columns = pd.MultiIndex.from_tuples(
                            [(detail_group_name, c) if c in merge_columns else (f"{col} {d}+", c) for c in _table.columns]
                        )
                        # 合并表
                        table = table.merge(_table, on=[(detail_group_name, c) for c in merge_columns])
        else:
            # 单标签情况
            table = _report_one_rule(datasets, target, desc=desc, prior_rules=prior_rules)

        # 处理 filter_cols
        if filter_cols:
            if not isinstance(filter_cols, list):
                filter_cols = [filter_cols]

            if isinstance(table.columns, pd.MultiIndex):
                # 多层级列结构
                cols_to_keep = []
                for col in table.columns:
                    # 保留 merge_columns 中的列
                    if col[1] in ['规则分类', '指标名称', '指标含义', '分箱'] or col[1] in filter_cols:
                        cols_to_keep.append(col)
                if cols_to_keep:
                    table = table[cols_to_keep]
            else:
                # 单层级列结构
                cols_to_keep = [c for c in table.columns if c in filter_cols or c in ['规则分类', '指标名称', '指标含义', '分箱']]
                if cols_to_keep:
                    table = table[cols_to_keep]

        return table

    @staticmethod
    def save(report: pd.DataFrame, excel_writer: Union[str, "ExcelWriter"], sheet_name: str = None,
             excel_params: dict = None) -> "ExcelWriter":
        """保存规则报告到 Excel。

        **参数**

        :param report: 规则报告 DataFrame（由 report() 方法生成）
        :param excel_writer: Excel 文件路径（字符串）或 ExcelWriter 对象；
            传入路径时会自动创建并写入后关闭，传入对象时追加写入不关闭
        :param sheet_name: 工作表名称，默认为None（使用默认名称Sheet1）
        :param excel_params: 额外的 pandas to_excel 写入参数（可选）
        :return: ExcelWriter 对象
        :raises TypeError: excel_writer 类型不正确时

        **参考样例**

        >>> from hscredit.core.rules import Rule
        >>> import pandas as pd
        >>> df = pd.DataFrame({'age': [20, 30], 'income': [3000, 8000], 'target': [0, 1]})
        >>> rule = Rule("age > 25")
        >>> report = rule.report(df)
        >>> writer = Rule.save(report, "rule_report.xlsx", sheet_name="规则报告")
        """
        from ..report import ExcelWriter

        if isinstance(excel_writer, str):
            writer = ExcelWriter(excel_writer)
        elif isinstance(excel_writer, ExcelWriter):
            writer = excel_writer
        else:
            raise TypeError(f"excel_writer 必须是 str 或 ExcelWriter 对象")

        writer.add_dataframe(report, sheet_name=sheet_name)

        if isinstance(excel_writer, str):
            writer.close()

        return writer

    def __hash__(self):
        """使 Rule 可哈希，基于表达式。"""
        return hash(self.expr)


def get_rule_columns(rule_expr: str) -> List[str]:
    """从规则表达式中提取列名。

    封装 get_columns_from_query，提供更直观的函数名称。

    **参数**

    :param rule_expr: 规则表达式字符串，如 "age > 18 and income < 5000"
    :return: 规则表达式中引用的列名列表（去重后按字母排序）

    **参考样例**

    >>> from hscredit.core.rules import get_rule_columns
    >>> get_rule_columns("age > 18 and income < 5000")
    ['age', 'income']
    """
    return get_columns_from_query(rule_expr)
