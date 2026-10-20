---
name: hsexcel
description: Export hscredit DataFrames and named tables to styled Excel workbooks with the current ExcelWriter and dataframe2excel contracts. Use for workbook delivery after analysis.
---

# HSCredit Excel

使用 `dataframe_to_excel` 导出 `inputs.data`，或使用 `workbook` 把多个 named inputs 写入多 Sheet 工作簿。`sheet_name`、`title`、数字格式、条件格式和 `speed` 参数按 hscredit Excel API 传递。

输出使用原子制品 manifest，默认不覆盖已有清单。条件格式沿用显式调用值、Writer 默认值和主题默认值的优先级。
