"""规则引擎模块.

提供规则定义、评估、报告和规则挖掘功能。

子模块:
    - mining: 规则挖掘功能，包括单特征、多特征交叉、决策树规则提取
    - rule: 规则定义和评估
    - expr_optimizer: 表达式优化

示例:
    >>> from hscredit.core.rules import Rule, mining
    >>> 
    >>> # 使用规则
    >>> rule = Rule("age > 18", name="成年规则")
    >>> result = rule.predict(df)
    >>> 
    >>> # 使用规则挖掘
    >>> from hscredit.core.rules.mining import SingleFeatureRuleMiner
    >>> miner = SingleFeatureRuleMiner(target='ISBAD')
    >>> miner.fit(df)
    >>> rules = miner.get_top_rules(top_n=10)
"""

from .rule import Rule, get_columns_from_query, RuleState, RuleStateError, RuleUnAppliedError
from .expr_optimizer import optimize_expr, beautify_expr, get_expr_variables

# 导出mining子模块
from . import mining

__all__ = [
    'Rule',
    'get_columns_from_query',
    'optimize_expr',
    'beautify_expr',
    'get_expr_variables',
    'RuleState',
    'RuleStateError',
    'RuleUnAppliedError',
    'mining',
]
