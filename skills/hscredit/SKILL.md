---
name: hscredit
description: Inspect the currently implemented hscredit Agent Skills and route a request to the correct domain Skill. Use as the entry point when the user has not chosen hsbin, hsreport, hsmodel, hsrule, hsselect, hsviz, or hsexcel.
---

# hscredit Skills 总入口

先调用 `capabilities` 查看当前操作目录，再把任务交给最匹配的领域 Skill：

- 分箱、分箱统计和分箱器：`hsbin`
- 完整特征/模型/策略 Excel 报告：`hsreport`
- 模型训练、调参、Loss、评分卡、校准和解释：`hsmodel`
- 表达式规则、RuleFlow、规则报告和 Swap：`hsrule`
- 特征筛选：`hsselect`
- 模型、评分、策略和稳定性图表：`hsviz`
- Excel 表格和多 Sheet 工作簿：`hsexcel`

每个领域 Skill 使用同一版本化请求信封、文件/object_ref 输入和原子制品 manifest。需要具体参数时读取目标 Skill 的 `references/operations.md`。
