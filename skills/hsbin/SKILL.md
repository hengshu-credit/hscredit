---
name: hsbin
description: Use hscredit to bin credit-risk variables, calculate bin statistics and efficiency, compare methods, fit reusable one- or two-dimensional binners, and render binning plots. Use for variable binning work; use hsreport instead when the requested deliverable is a complete feature, model, or strategy Excel report.
---

# HSCredit 分箱分析

调用 hscredit 的真实分箱、统计和绘图 API，并将完整结果保存为 Excel、图片、JSON 或 hscredit artifact。

## 选择操作

先根据任务选择一个 `operation`。需要了解操作输入、参数和产物时读取 [operations.md](references/operations.md)。不要把未登记的 hscredit 函数名或 Python import 路径作为操作。

常用选择：

- 单个或多个变量的分箱统计：`feature_bin_stats`
- 手工与自动分箱效率对比：`feature_efficiency_analysis`
- 多方法或跨分组摘要：`benchmark_binning_methods`、`feature_binning_summary`、`feature_group_binning_summary`
- 可复用分箱器：`optimal_binning_fit`、`optimal_binning_transform`、`optimal_binning_2d_fit`
- 分箱图片：`bin_plot`、`bin_trend_plot`、`bin_overdues_plot`、`bin_2d_plot`

## 组装请求

请求必须符合 [request.schema.json](schemas/request.schema.json)。数据使用以下一种来源：

- `{"kind":"file","path":"..."}`：CSV、XLSX 或 Parquet。
- `{"kind":"object_ref","ref":"..."}`：仅限调用方提供同进程对象注册表的 Python/Notebook 调用。

默认使用隔离环境并自动安装受控依赖。实时 `object_ref` 不能跨解释器；使用它时明确设置 `environment.mode="current"`。只有用户明确要求修改当前代码环境时，才设置 `install_missing=true`。

输出路径只放在 `output` 中。不要在 `parameters` 中传 `save`、`output_dir` 或任意 shell、包名、索引 URL、Git URL。

## 执行

把请求保存为 UTF-8 JSON，然后运行：

```powershell
python scripts/run.py request.json
```

脚本输出一个 JSON 信封。向用户交付 `artifacts` 中的绝对路径，并简要说明 `summary` 和 `warnings`。不要把完整明细表复制进 Chat。

## 约束

- 支持组合式逾期标签的表格分析把所有逾期字段放入同一个 `overdue` 列表、所有阈值放入同一个 `dpds` 列表，只提交一个请求并生成一个 Excel；字段与阈值的全部组合由 hscredit 展开。
- 这些表格分析的运行摘要保留原有行列和受限预览，并追加实际标签组合。
- 显式参数优先于 hscredit 默认值；不得自行改写 `anchor`、分箱规则、WOE/indices 指标或并行参数。
- Binner artifact 只有在输入显式声明 `trusted=true` 时才反序列化。
- 失败不自动重试训练、分箱或绘图；先报告稳定错误码和中文原因。
- 不重新实现 bin statistics；以 hscredit 当前 API 为唯一计算口径。
