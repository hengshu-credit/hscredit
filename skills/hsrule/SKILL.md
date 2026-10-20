---
name: hsrule
description: Build and evaluate hscredit Rule expressions and RuleFlow objects, then generate rule, ruleset, and rule swap analysis tables. Use for strategy rule work; use hsmodel for rules-as-model classifiers.
---

# HSCredit 规则

登记操作：`rule_report`、`ruleset_analysis`、`rule_swap_analysis`、`rule_flow_report`。

规则输入使用表达式字符串，例如 `score < 560`。Skill 会交给 hscredit 的 Rule 解析器，不执行任意 Python `eval`。规则分析输出 Excel；RuleFlow 和多规则报告保留中文指标与逾期 operator 配置。
