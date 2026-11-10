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

## 模型解释图

- `plot_model_feature_importance`：提供 `inputs.model`、`inputs.X`，需要分标签展示时提供 `inputs.y`；`importance_source`、`background_size`、`right_top_n` 等绘图选项放入 `parameters.kwargs`。
- `plot_model_sample_shap`：提供 `inputs.model`、`inputs.sample`、`inputs.background_data`；样本必须符合方法要求的单样本结构。`prediction_method` 和 `class_index` 指定解释的模型输出。

模型文件必须显式声明 `trusted=true`；同进程对象用 `object_ref` 并选择 `environment.mode="current"`。输出图片使用 `output` 管理的路径，不另传 `kwargs.save`。

区分 SHAP 正负贡献、好坏样本标签和特征值高低三种语义。解释坏样本概率时，正贡献提高该概率；解释其他类别或评分时，应按实际输出说明方向。
