# hsrule 操作

| operation | 必要参数 | 产物 |
|---|---|---|
| `rule_report` | `rules`, `target` | 每条规则 Excel |
| `ruleset_analysis` | `rules`, `target` | 规则集效果 Excel |
| `rule_swap_analysis` | `score`, `rules_in/out/base` | Swap 多表 Excel |
| `rule_flow_report` | `rules`, `mode` | RuleFlow 报告 Excel |

`overdue_operator` 支持当前 hscredit 的 `>`、`>=`、`<`、`<=` 语义，并会记录在摘要中。
