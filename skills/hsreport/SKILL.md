---
name: hsreport
description: Generate complete hscredit Excel deliverables for feature analysis, trained-model evaluation, or rule-strategy iteration. Use when the user wants a polished multi-section Excel report; use hsbin for standalone bin statistics, reusable binners, or individual binning plots.
---

# HSCredit 分析报告

组装数据、模型和规则，调用 hscredit 的完整报告入口，并交付经过重新打开校验的 Excel 工作簿。

## 选择报告

需要参数和示例时读取 [operations.md](references/operations.md)。只允许以下操作：

- `auto_feature_analysis`：变量有效性、分箱、分布和时间分析报告。
- `auto_model_report`：已训练模型的完整评估与部署信息报告。
- `swap_out_report`：规则策略迭代、业务影响、稳定性和变量分箱报告。

## 输入与执行

请求必须符合 [request.schema.json](schemas/request.schema.json)。文件输入支持 CSV、XLSX 和 Parquet；模型可使用同进程 `object_ref` 或显式可信的 hscredit artifact。

默认在隔离环境安装受控依赖。`object_ref` 需要当前进程，因此使用它时设置 `environment.mode="current"`；只有用户明确同意修改当前环境时才启用当前环境自动安装。

把请求保存为 UTF-8 JSON，然后运行：

```powershell
python scripts/run.py request.json
```

向用户返回 `artifacts` 中的 Excel 绝对路径以及必要的摘要。不要把完整工作簿内容粘贴进 Chat。

## 报告约束

- 输出路径只从 `output` 注入，拒绝 `parameters.excel_path`、`excel_writer`、`save` 和 `output_dir`。
- 不覆盖 hscredit 的显式参数优先级、条件格式颜色、主题色、绘图 anchor、字段默认选择和并行配置。
- 一个请求只调用一次对应顶层报告函数，失败不自动重试。
- 发布前必须重新打开 Excel；失败时不留下被标记为成功的半成品。
- 不自动上传报告，不提交代码，不发布远端 Skill。
