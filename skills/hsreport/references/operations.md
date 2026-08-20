# hsreport 操作参考

## auto_feature_analysis

需要 `inputs.data`。常用参数包括 `features`、`target`、`overdue`、`dpds`、`date`、`amount`、`feature_map`、`pictures`、`bin_params` 和并行配置。

```json
{
  "version": "1",
  "operation": "auto_feature_analysis",
  "inputs": {
    "data": {"kind": "file", "path": "data/sample.xlsx", "sheet_name": "建模样本"}
  },
  "parameters": {
    "features": ["score", "age"],
    "target": "target",
    "date": "apply_date",
    "amount": "amount",
    "n_jobs": 1
  },
  "output": {"directory": "outputs", "name": "特征分析报告", "overwrite": false},
  "environment": {"mode": "isolated", "reuse": true}
}
```

产物是 Excel；报告生成的独立 PNG/SVG 也进入 manifest。

## auto_model_report

需要 `inputs.model`，并在 `inputs` 中提供一个或多个数据集。`parameters.datasets` 的值是 `inputs` 键，不是隐藏的文件路径。

```json
{
  "version": "1",
  "operation": "auto_model_report",
  "inputs": {
    "model": {"kind": "file", "path": "artifacts/model.joblib", "trusted": true},
    "train": {"kind": "file", "path": "data/train.parquet"},
    "test": {"kind": "file", "path": "data/test.parquet"}
  },
  "parameters": {
    "datasets": {"训练集": "train", "测试集": "test"},
    "target": "target",
    "with_plots": true,
    "verbose": false,
    "n_jobs": 1
  },
  "output": {"directory": "outputs", "name": "模型分析报告", "overwrite": false},
  "environment": {"mode": "isolated", "reuse": true}
}
```

显式 `y`、`target`、`overdue+dpds` 的优先级继续由 hscredit 决定。不要在 Skill 层重新生成预测或标签。

## swap_out_report

需要 `inputs.data` 和 `parameters.rules`。规则可以是表达式字符串；同进程调用也可通过运行时扩展使用 Rule 对象。

```json
{
  "version": "1",
  "operation": "swap_out_report",
  "inputs": {
    "data": {"kind": "file", "path": "data/strategy.csv"}
  },
  "parameters": {
    "rules": ["score < 560", "age < 25"],
    "target": "target",
    "features": ["score", "age"],
    "amount": "amount",
    "date_col": "apply_date",
    "methods": "quantile",
    "n_jobs": 1
  },
  "output": {"directory": "outputs", "name": "策略迭代报告", "overwrite": false},
  "environment": {"mode": "isolated", "reuse": true}
}
```

产物必须同时包含“策略迭代”和“变量分箱”工作表。

## 安全边界

- Pickle/joblib 模型必须 `trusted=true`。
- 输出参数不允许出现在 `parameters`。
- 请求不能指定安装包、包索引、Git URL、Python 文件或 shell 命令。
- 大表只返回摘要和受限预览，完整结果留在 Excel。
