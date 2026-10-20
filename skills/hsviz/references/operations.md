# hsviz 操作

```json
{
  "operation": "plot",
  "inputs": {"y_true": {"kind":"file","path":"y.csv"}, "y_score": {"kind":"file","path":"score.csv"}},
  "parameters": {"plot":"roc_plot", "kwargs":{"show_auc":true}},
  "output": {"directory":"outputs", "name":"roc", "format":"png"}
}
```

分箱、KS、ROC/PR/Lift/Gain、评分、策略、漂移、树图和通用特征图都通过同一 `plot` 白名单适配器调用。
