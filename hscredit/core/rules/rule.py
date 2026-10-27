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

from ...utils.targets import TargetInput, bind_targets, reject_legacy_target_keywords
from ...utils.arguments import analysis_inputs
from ..metrics._semantics import metric_metadata, rejection_statistics, validate_amounts

import ast
import copy
import os
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

from ...utils.overdue import validate_overdue_operator
from .expr_optimizer import optimize_expr, beautify_expr
from ...exceptions import FeatureNotFoundError, InputTypeError, StateError
from ...utils.parallel import ParallelizableMixin, ParallelWorkload

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
    :type query_str: str
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
    tree = ast.parse(parsed_query, mode="eval")

    columns = set()
    reserved_names = {"and", "or", "not", "True", "False", "None"}

    # 反引号包裹的列名（pandas query/eval 语法，用于含空格/特殊字符/非标识符的列名），
    # ast.parse 无法解析反引号，需先用正则提取并替换为占位标识符
    backtick_names = re.findall(r"`([^`]+)`", query_str)
    columns.update(name.strip() for name in backtick_names)
    cleaned = re.sub(r"`[^`]+`", " _hscredit_bt_ ", query_str)

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
        tree = ast.parse(cleaned, mode="eval")
        for node in ast.walk(tree):
            visit_node(node)
    except SyntaxError:
        # 极端情况下（如复杂的非标识符表达式）退化为仅返回反引号列名
        pass

    columns.discard("_hscredit_bt_")
    return sorted(columns)


class RuleState(str, Enum):
    """规则生命周期状态枚举。

    继承 ``str`` ，可直接与字符串比较。 :class:`Rule`  用它标记是否已执行过 predict，
    从而约束  :meth:`Rule.result`  等依赖结果的方法。

    **枚举值**

    - ``INITIALIZED`` ( ``"initialized"`` )：规则已创建但尚未调用  :meth:`Rule.predict` ，
      此时无可用结果

    - ``APPLIED`` ( ``"applied"`` )：规则已对某数据集执行过  :meth:`Rule.predict` ，
      ``result_`` 中存有最近一次命中结果
    """

    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(StateError):
    """规则状态异常基类。

    当在规则不允许的状态下调用方法时抛出，继承自
    :class:`~hscredit.exceptions.StateError` 。
    """

    pass


class RuleUnAppliedError(RuleStateError):
    """规则尚未应用异常。

    在未先调用  :meth:`Rule.predict`  的情况下访问  :meth:`Rule.result`  等依赖预测
    结果的方法时抛出。
    """

    pass


def _rule_report_target_worker(task: Tuple[Any, ...]) -> pd.DataFrame:
    """执行一个独立目标标签的规则报告任务。"""
    rule, data, column, desc, prior_rules, amount, margins = task
    data = data.loc[data[column].notna()]
    return rule.report(data, target=column, desc=desc, prior_rules=prior_rules, amount=amount, margins=margins)


class Rule(ParallelizableMixin):
    """规则类。

    支持使用 pandas eval 语法的规则定义和评估，支持 ``&`` （与）、 ``|`` （或）、
    ``~`` （非）、 ``^`` （异或）等运算符组合多个规则为复合规则。

    **属性**

    **参数**

    :param expr: 规则表达式字符串
    :type expr: str
    :param name: 规则名称，用于标识和展示，默认为None（使用表达式作为名称）
    :type name: Optional[str]
    :param description: 规则描述，默认为空字符串
    :type description: str
    :param weight: 规则权重，用于规则集分类器，默认为1.0
    :type weight: float
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Union[int, float]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
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
        weight: float = 1.0,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        """初始化规则。

        **参数**

        :param expr: 规则表达式字符串，支持 pandas eval 语法，
            如 "age > 18 and income < 5000"

        :param name: 规则名称，用于标识和展示，默认为None（使用表达式作为名称）
        :param description: 规则描述，默认为空字符串
        :param weight: 规则权重，用于规则集分类器，默认为1.0
        :param n_jobs: 并行任务数，默认为-1
        :param parallel_backend: joblib并行后端，默认为None
        :param parallel_config: joblib扩展配置，默认为None
        """
        self._state = RuleState.INITIALIZED
        self.expr = expr
        self.name = name or expr
        self.description = description
        self.weight = weight
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config
        self.feature_names_in_ = get_columns_from_query(self.expr)
        self.result_ = None

    def __str__(self):
        return f"Rule({repr(self.expr)})"

    def __repr__(self):
        return f"Rule({repr(self.expr)})"

    def __and__(self, other):
        """规则"与"（AND）组合，对应 Python ``&`` 运算符。

        将两条规则用 ``&`` 拼接为复合规则，命中条件为两者同时成立；表达式会经
        :func:`beautify_expr`  美化与  :func:`optimize_expr`  化简，名称拼为
        ``(A)_AND_(B)`` ，权重取两者较大值。

        :param other: 另一条  :class:`Rule`  规则
        :return: 组合后的新  :class:`Rule` （不修改原规则）
        :raises InputTypeError: ``other`` 不是  :class:`Rule`  时

        **参考样例**

        >>> r = Rule("age > 18") & Rule("income > 5000")
        >>> r.expr
        'age > 18 & income > 5000'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"& 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        combined_expr = f"({self.expr}) & ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, "name", None) or self.expr
        other_name = getattr(other, "name", None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_AND_({other_name})",
            description=f"{self.description} 且 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight),
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
        )

    def __or__(self, other):
        """规则"或"（OR）组合，对应 Python ``|`` 运算符。

        将两条规则用 ``|`` 拼接为复合规则，命中条件为两者任一成立；表达式经美化与
        化简，名称拼为 ``(A)_OR_(B)`` ，权重取两者较大值。

        :param other: 另一条  :class:`Rule`  规则
        :return: 组合后的新  :class:`Rule` （不修改原规则）
        :raises InputTypeError: ``other`` 不是  :class:`Rule`  时

        **参考样例**

        >>> r = Rule("age > 18") | Rule("income > 5000")
        >>> r.expr
        'age > 18 | income > 5000'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"| 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        combined_expr = f"({self.expr}) | ({other.expr})"
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, "name", None) or self.expr
        other_name = getattr(other, "name", None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_OR_({other_name})",
            description=f"{self.description} 或 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight),
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
        )

    def __invert__(self):
        """规则"非"（NOT）取反，对应 Python ``~`` 运算符。

        对规则整体取反得到新规则，命中条件为原规则不成立；名称拼为 ``NOT_(A)`` ，
        权重保持不变。

        :return: 取反后的新  :class:`Rule` （不修改原规则）

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
            weight=self.weight,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
        )

    def __xor__(self, other):
        """规则"异或"（XOR）组合，对应 Python ``^`` 运算符。

        命中条件为两条规则恰有一条成立。由于 pandas eval 不支持布尔 ``^`` ，内部以等价
        表达式 ``(A & ~B) | (~A & B)`` 实现；名称拼为 ``(A)_XOR_(B)`` ，权重取较大值。

        :param other: 另一条  :class:`Rule`  规则
        :return: 组合后的新  :class:`Rule` （不修改原规则）
        :raises InputTypeError: ``other`` 不是  :class:`Rule`  时

        **参考样例**

        >>> r = Rule("age > 18") ^ Rule("income > 5000")
        >>> r.expr
        '(age > 18 & ~(income > 5000)) | (~(age > 18) & income > 5000)'
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"^ 运算两侧必须是 Rule，实际右侧类型为 {type(other).__name__}")
        # pandas eval 不支持布尔 ^（BitXor），用等价的 (a & ~b) | (~a & b) 表达异或
        combined_expr = f"(({self.expr}) & ~({other.expr})) | (~({self.expr}) & ({other.expr}))"
        optimized = optimize_expr(beautify_expr(combined_expr))
        self_name = getattr(self, "name", None) or self.expr
        other_name = getattr(other, "name", None) or other.expr
        return Rule(
            optimized,
            name=f"({self_name})_XOR_({other_name})",
            description=f"{self.description} 异或 {other.description}" if self.description or other.description else "",
            weight=max(self.weight, other.weight),
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
        )

    def __eq__(self, other):
        """规则相等比较，对应 Python ``==`` 运算符。

        两条规则当且仅当 ``expr`` 表达式字符串完全相同时视为相等（与  :meth:`__hash__`
        保持一致，因此  :class:`Rule`  可放入 set / 作为 dict 键）。

        :param other: 另一条  :class:`Rule`  规则
        :return: 两者表达式是否相同（bool）
        :raises InputTypeError: ``other`` 不是  :class:`Rule`  时
        """
        if not isinstance(other, Rule):
            raise InputTypeError(f"输入必须是 Rule 类型，实际类型为 {type(other).__name__}")
        return self.expr == other.expr

    def predict(self, X: DataFrame) -> pd.Series:
        """应用规则进行预测。

        使用 pandas eval 对 DataFrame 执行规则表达式，返回命中的布尔 Series。

        **参数**

        :param X: 输入数据 DataFrame（必须包含规则表达式中引用的全部列）
        :type X: DataFrame
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
        if np.isscalar(result):
            if not isinstance(result, (bool, np.bool_)):
                raise ValueError("规则表达式必须返回布尔命中结果")
            result = pd.Series(bool(result), index=X.index)
        if not isinstance(result, pd.Series) or len(result) != len(X):
            raise ValueError("规则表达式必须为每个样本返回一个命中结果")
        result = result.fillna(False).astype(bool)
        self.result_ = result
        self._state = RuleState.APPLIED

        return result

    def predict_batches(self, data: DataFrame, batch_size: int = 10000):
        """逐块返回命中结果，不拼接全量明细；保留原始索引与行顺序。"""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size 必须为正整数")
        if not isinstance(data, DataFrame):
            raise InputTypeError("Rule 只能对 DataFrame 执行预测")
        for start in range(0, len(data), batch_size):
            yield self.predict(data.iloc[start : start + batch_size])

    def aggregate(self, batches, target=None):
        """只保留固定规则的可合并命中计数，不保存逐行结果。"""
        from .accumulator import RuleAccumulator

        accumulator = RuleAccumulator(self, target=target)
        for batch in batches:
            accumulator.update(batch)
        return accumulator.finalize()

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
        :type X: DataFrame
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

    @analysis_inputs("datasets", roles=("amount",))
    def report(
        self,
        datasets: pd.DataFrame,
        target: Optional[TargetInput] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, float, List[Union[int, float]]]] = None,
        del_grey: bool = False,
        desc: str = "",
        filter_cols: Optional[List[str]] = None,
        prior_rules: Optional["Rule"] = None,
        amount: Optional[str] = None,
        margins: bool = False,
        *,
        overdue_operator: str = ">",
        **kwargs,
    ) -> pd.DataFrame:
        """规则效果报告表格输出。

        将规则命中与否作为二分类，对数据集计算统计指标，
        包括样本数、坏账率、LIFT值、风险拒绝比、精确率、召回率、F1分数等。
        支持金额口径分析与多标签（不同逾期天数定义）联合输出。

        **参数**

        :param datasets: 数据集 DataFrame，需要包含目标变量列或逾期天数列
        :type datasets: pd.DataFrame
        :param target: 默认值为 ``None`` 。支持列名、一维 Series/ndarray/list、callable，
            或 ``{标签名: 标签数据或 callable}``；与 ``overdue+dpds`` 互斥。
            callable 接收当前完整 DataFrame；Series 按索引对齐，数组和列表按行位置对应。
            None 且未指定逾期模式时使用数据中的 ``target`` 列。
        :type target: Optional[TargetInput]
        :param overdue: 默认值为 ``None`` 。逾期天数字段名或字段名序列；与 ``dpds`` 必须同时传入，
            与 ``target`` 互斥，标签方向由 ``overdue_operator`` 控制。
        :type overdue: Optional[Union[str, List[str]]]
        :param dpds: 默认值为 ``None`` 。逾期阈值标量或非空序列，支持整数或小数；
            必须与 ``overdue`` 同时传入。按 ``overdue × dpds`` 组合生成标签。
        :type dpds: Optional[Union[int, float, List[Union[int, float]]]]
        :param del_grey: 是否逐标签剔除灰客户。``target`` 模式剔除非 0/1 值（含缺失）；
            逾期模式下 ``>`` 剔除 ``(0, DPD]``，``>=`` 剔除 ``(0, DPD)``，
            ``<``、``<=`` 不设置灰区。各标签独立计算有效样本分母。
            默认 False；显式标签中非 0/1 值按 0 计算，逾期标签按比较结果计算。
        :type del_grey: bool
        :param desc: 规则描述，用于报告的"指标含义"列，默认为空字符串
        :type desc: str
        :param filter_cols: 默认值为 ``None`` 。指定返回的字段列表（可选）
        :type filter_cols: Optional[List[str]]
        :param prior_rules: 默认值为 ``None`` 。先验规则（可选），先对数据应用先验规则排除部分样本
            再对当前规则进行评估

        :type prior_rules: Optional['Rule']
        :param amount: 默认值为 ``None`` 。金额字段名（可选），有效标签人群的金额必须为有限非负数值；
            传入时业务指标按金额统计，准确率/精确率/召回率/F1仍按样本数统计
        :type amount: Optional[str]
        :param margins: 是否在报告末尾添加合计行，默认为False
        :type margins: bool
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

        :param overdue_operator: 逾期标签比较符，支持 ``>`` 、 ``>=`` 、 ``<`` 、 ``<=`` ，默认 ``>`` 。
            满足比较条件记为坏样本(1)，否则为好样本(0)。
            ``del_grey=True`` 时， ``>`` 剔除 ``(0, dpd]`` ， ``>=`` 剔除 ``(0, dpd)`` ；
            ``<`` 和 ``<=`` 暂无灰客户，保留参数但不剔除样本。

        :type overdue_operator: str
        """
        validate_overdue_operator(overdue_operator)
        reject_legacy_target_keywords(kwargs)
        datasets, targets = bind_targets(
            datasets, target, overdue, dpds, del_grey, overdue_operator, default_target="target"
        )
        del_grey = targets.needs_filtering(datasets) or (
            del_grey and (targets.mode == "target" or targets.operator in (">", ">="))
        )
        detail_group_name = "分箱详情"
        return_cols = [
            "指标名称",
            "指标含义",
            "分箱",
            "样本总数",
            "样本占比",
            "好样本数",
            "好样本占比",
            "坏样本数",
            "坏样本占比",
            "坏样本率",
            "LIFT值",
            "坏账改善",
        ]
        if not desc:
            if "指标含义" in return_cols:
                return_cols.remove("指标含义")

        rule_expr = self.expr

        def _report_one_rule(data, target, desc="", prior_rules=None):
            """生成单标签的规则报告。

            直接手动计算统计指标，规则只有命中/未命中两种状态，不需要分箱器。
            支持金额口径分析，当传入 amount 参数时会计算金额相关指标。
            列名与 feature_bin_stats 保持一致，便于统一处理。
            """
            if prior_rules:
                prior_tables = prior_rules.report(
                    data, target=target, desc=desc, prior_rules=None, margins=margins, amount=amount
                )
                prior_tables["规则分类"] = "先验规则"
                data = data.loc[~prior_rules.predict(data)]
            else:
                prior_tables = pd.DataFrame(columns=return_cols)

            # 标签、命中掩码和金额分开保存，避免用户字段名与表达式覆盖内部列。
            hits = np.asarray(self.predict(data), dtype=bool)
            labels = data[target].to_numpy()
            weights = (
                validate_amounts(data[amount], len(data)) if amount is not None else np.ones(len(data), dtype=np.int64)
            )
            total = weights.sum()
            total_bad = weights[labels == 1].sum()
            total_good = total - total_bad
            overall_bad_rate = total_bad / total if total > 0 else 0.0
            rows = []
            for bin_name, selected in [("命中", hits), ("未命中", ~hits)]:
                count = weights[selected].sum()
                bad = weights[selected & (labels == 1)].sum()
                good = count - bad
                rate = bad / count if count > 0 else 0.0
                _, improvement, efficiency = rejection_statistics(total, total_bad, count, bad)
                rows.append(
                    {
                        "指标名称": rule_expr,
                        "指标含义": desc if desc else "",
                        "分箱": bin_name,
                        "样本总数": count,
                        "样本占比": count / total if total > 0 else 0.0,
                        "好样本数": good,
                        "好样本占比": good / total_good if total_good > 0 else 0.0,
                        "坏样本数": bad,
                        "坏样本占比": bad / total_bad if total_bad > 0 else 0.0,
                        "坏样本率": rate,
                        "LIFT值": rate / overall_bad_rate if overall_bad_rate > 0 else 0.0,
                        "坏账改善": float(improvement),
                        "风险拒绝比": float(efficiency),
                    }
                )
            table = pd.DataFrame(rows)
            total_row_data = {
                "指标名称": "合计",
                "指标含义": "",
                "分箱": "合计",
                "样本总数": total,
                "样本占比": float(total > 0),
                "好样本数": total_good,
                "好样本占比": float(total_good > 0),
                "坏样本数": total_bad,
                "坏样本占比": float(total_bad > 0),
                "坏样本率": overall_bad_rate,
                "LIFT值": 1.0 if overall_bad_rate > 0 else 0.0,
                "坏账改善": 0.0,
            }
            y_true = pd.Series(labels)
            y_pred = pd.Series(hits.astype(int))

            metrics_data = {
                "分箱": ["命中", "未命中"],
                "准确率": [
                    accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0,
                    accuracy_score(y_true, 1 - y_pred) if len(y_true) > 0 else 0,
                ],
                "精确率": [
                    precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    precision_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0,
                ],
                "召回率": [
                    recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    recall_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0,
                ],
                "F1分数": [
                    f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    f1_score(y_true, 1 - y_pred, zero_division=0) if len(y_true) > 0 else 0,
                ],
            }
            metrics = pd.DataFrame(metrics_data)

            # 合并指标
            table = table.merge(metrics, on="分箱", how="left")

            # 如果需要合计行，添加合计
            if margins:
                total_row_data.update(
                    {
                        "风险拒绝比": 0.0,
                        "准确率": accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0,
                        "精确率": precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                        "召回率": recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                        "F1分数": f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0,
                    }
                )
                table = pd.concat([table, pd.DataFrame([total_row_data])], ignore_index=True)

            if not desc and "指标含义" in table.columns:
                table = table.drop(columns=["指标含义"])

            if prior_rules:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))
                table = pd.concat([prior_tables, table], ignore_index=True)
            else:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))

            return table

        if len(targets) > 1 or targets.mode == "overdue":
            merge_columns = ["规则分类", "指标名称", "分箱"]
            if not del_grey:
                merge_columns += ["样本总数", "样本占比"]
            if "指标含义" in return_cols:
                merge_columns.append("指标含义")
            labels = targets.names("space")
            task_data = datasets
            expression_rules = [self] + ([prior_rules] if prior_rules is not None else [])
            if all(
                type(rule) is Rule and getattr(rule.predict, "__func__", None) is Rule.predict
                for rule in expression_rules
            ):
                # 标签已绑定；标准表达式规则只携带所需字段，避免每个标签复制整张宽表。
                # 自定义 predict 的依赖不可由表达式推断，保持其完整输入。
                required = set(targets.columns)
                if amount is not None:
                    required.add(amount)
                try:
                    for rule in expression_rules:
                        required.update(rule.feature_names_in_)
                        required.update(get_columns_from_query(rule.expr))
                except (ValueError, SyntaxError):
                    required = set(datasets.columns)
                if datasets.columns.is_unique and required.issubset(datasets.columns):
                    task_data = datasets.loc[:, [column for column in datasets.columns if column in required]]
            tasks = (
                (copy.deepcopy(self), task_data, definition.column, desc, copy.deepcopy(prior_rules), amount, margins)
                for definition in targets.labels
            )
            tables = self._parallel_execute(
                _rule_report_target_worker,
                tasks,
                task_labels=labels,
                default_backend="threading",
                workload=ParallelWorkload(
                    task_count=len(targets),
                    rows=len(task_data),
                    columns=len(task_data.columns),
                    data_bytes=int(task_data.memory_usage(deep=True).sum()),
                    cost_per_item=6.0,
                    capability="thread_safe",
                    releases_gil=True,
                    operation="规则多标签报告",
                ),
            )
            merge_keys = pd.concat([item[merge_columns] for item in tables], ignore_index=True).drop_duplicates()
            merge_keys.columns = pd.MultiIndex.from_tuples([(detail_group_name, column) for column in merge_columns])
            table = pd.DataFrame()
            for position, (label, current) in enumerate(zip(labels, tables)):
                current.columns = pd.MultiIndex.from_tuples(
                    [(detail_group_name if column in merge_columns else label, column) for column in current.columns]
                )
                table = (
                    current
                    if position == 0
                    else table.merge(
                        current, on=[(detail_group_name, column) for column in merge_columns], how="outer", sort=False
                    )
                )
            table = merge_keys.merge(table, on=list(merge_keys.columns), how="left", sort=False)
        else:
            column = targets.columns[0]
            valid = datasets[column].notna()
            table = _report_one_rule(datasets.loc[valid], column, desc=desc, prior_rules=prior_rules)
        table.attrs["target_mode"] = targets.mode
        table.attrs["target_names"] = targets.names("space")
        table.attrs["统计口径"] = {
            **metric_metadata("amount_weighted" if amount is not None else "binary"),
            "分类指标单位": "准确率、精确率、召回率、F1按样本计数",
            "逾期比较符": targets.operator if targets.mode == "overdue" else None,
            "剔灰": bool(del_grey),
        }

        # 处理 filter_cols
        if filter_cols:
            if not isinstance(filter_cols, list):
                filter_cols = [filter_cols]

            if isinstance(table.columns, pd.MultiIndex):
                # 多层级列结构
                cols_to_keep = []
                for col in table.columns:
                    # 保留 merge_columns 中的列
                    if col[1] in ["规则分类", "指标名称", "指标含义", "分箱"] or col[1] in filter_cols:
                        cols_to_keep.append(col)
                if cols_to_keep:
                    table = table[cols_to_keep]
            else:
                # 单层级列结构
                cols_to_keep = [
                    c for c in table.columns if c in filter_cols or c in ["规则分类", "指标名称", "指标含义", "分箱"]
                ]
                if cols_to_keep:
                    table = table[cols_to_keep]

        return table

    @staticmethod
    def save(
        report: pd.DataFrame,
        excel_writer: Union[str, os.PathLike, "ExcelWriter"],
        sheet_name: Optional[str] = None,
        excel_params: Optional[Dict] = None,
    ) -> "ExcelWriter":
        """保存规则报告到 Excel。

        **参数**

        :param report: 规则报告 DataFrame（由 report() 方法生成）
        :type report: pd.DataFrame
        :param excel_writer: Excel 文件路径（字符串或 PathLike）或 ExcelWriter 对象；
            传入路径时会自动创建并写入后关闭，传入对象时追加写入不关闭

        :type excel_writer: Union[str, os.PathLike, 'ExcelWriter']
        :param sheet_name: 工作表名称，默认为None（使用默认名称Sheet1）
        :type sheet_name: Optional[str]
        :param excel_params: 默认值为 ``None`` 。额外的 dataframe2excel 写入参数（可选）
        :type excel_params: Optional[Dict]
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
        from ...excel import ExcelWriter, dataframe2excel

        params = dict(excel_params or {})
        params.pop("data", None)
        params.pop("excel_writer", None)
        params.pop("sheet_name", None)
        writer_params = dict(params.pop("writer_params", {}) or {})

        if isinstance(excel_writer, ExcelWriter):
            writer = excel_writer
            output_path = None
        elif isinstance(excel_writer, (str, os.PathLike)):
            writer_params.setdefault("theme_color", params.get("theme_color", "2639E9"))
            writer_params.setdefault("mode", params.get("mode", "replace"))
            writer = ExcelWriter(**writer_params)
            output_path = os.fspath(excel_writer)
        else:
            raise TypeError("excel_writer 必须是路径或 ExcelWriter 对象")

        dataframe2excel(report, writer, sheet_name=sheet_name, **params)

        if output_path is not None:
            writer.save(output_path)

        return writer

    def __hash__(self):
        """使 Rule 可哈希，基于表达式。"""
        return hash(self.expr)


def get_rule_columns(rule_expr: str) -> List[str]:
    """从规则表达式中提取列名。

    封装 get_columns_from_query，提供更直观的函数名称。

    **参数**

    :param rule_expr: 规则表达式字符串，如 "age > 18 and income < 5000"
    :type rule_expr: str
    :return: 规则表达式中引用的列名列表（去重后按字母排序）

    **参考样例**

    >>> from hscredit.core.rules import get_rule_columns
    >>> get_rule_columns("age > 18 and income < 5000")
    ['age', 'income']
    """
    return get_columns_from_query(rule_expr)
