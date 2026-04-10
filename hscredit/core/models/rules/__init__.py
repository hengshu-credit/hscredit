"""规则分类子包.

包含基于规则的分类器:
- RuleSet: 规则集
- RulesClassifier: 规则分类器
- RuleSetClassifier: 规则集分类器
"""

from .rule_classifier import (
    RuleSet,
    RulesClassifier,
    RuleSetClassifier,
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules,
)

__all__ = [
    "RuleSet",
    "RulesClassifier",
    "RuleSetClassifier",
    "LogicOperator",
    "RuleResult",
    "create_and_ruleset",
    "create_or_ruleset",
    "combine_rules",
]
