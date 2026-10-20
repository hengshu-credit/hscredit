---
name: hsselect
description: Fit an allowlisted hscredit feature selector, transform a dataset, and export selected features plus the selector's Chinese report. Use for feature selection workflows.
---

# HSCredit 特征筛选

使用 `selector_fit`，通过 `parameters.selector` 选择公开筛选器，例如 `IVSelector`、`KSSelector`、`CorrSelector`、`VIFSelector`、`VarianceSelector`、`StabilityAwareSelector` 或 `CompositeFeatureSelector`；构造参数放在 `selector_params`。当前筛选器白名单与 `hscredit.core.selectors.__all__` 同步。

输入支持 DataFrame 文件或同进程对象。输出包含筛选结果 Excel 和 JSON 筛选报告。显式 `include`、`exclude`、`force_drop`、`target_rm` 和并行参数按 hscredit 当前契约传递。
