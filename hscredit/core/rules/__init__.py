"""规则引擎模块.

提供规则定义、评估和报告功能，支持使用 pandas eval 语法的规则表达式编写与组合。

子模块:
    - rule: 规则定义、评估（Rule类）和报告生成
    - expr_optimizer: 规则表达式优化与美化

**参考样例**

>>> from hscredit.core.rules import Rule, get_columns_from_query
>>> rule = Rule("age > 18 and income > 5000", name="优质客群规则")
>>> cols = get_columns_from_query("age > 18 and income < 5000")
>>> print(cols)
"""

from .rule import Rule, get_columns_from_query, RuleState, RuleStateError, RuleUnAppliedError
from .expr_optimizer import optimize_expr, beautify_expr, get_expr_variables

__all__ = [
    'Rule',
    'get_columns_from_query',
    'optimize_expr',
    'beautify_expr',
    'get_expr_variables',
    'RuleState',
    'RuleStateError',
    'RuleUnAppliedError',
]
