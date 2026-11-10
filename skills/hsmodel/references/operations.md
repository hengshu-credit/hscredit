# hsmodel 操作

| operation | 主要参数 | 产物 |
|---|---|---|
| `model_fit` | `model`, `model_params`, `target`, `features` | 预测 Excel、模型 artifact |
| `model_predict` | `inputs.model`, `method`, `features` | 预测 Excel |
| `model_evaluate` | `inputs.model`, `target`, `features` | 模型评估 Excel |
| `compare_models` | `models` 输入映射、训练/测试数据 | 模型对比 Excel |
| `model_tune` | `model_type`, `n_trials`, `metric`, `search_space` | tuner 和最佳模型 artifact |
| `model_calibrate` | `inputs.model`, `method`, `target` | 校准模型 artifact |
| `model_explain` | `inputs.model`, `importance_type`, `top_n` | 解释 Excel |
| `component_create` | `component`, `params` | 通用组件 artifact |
| `loss_info` | `loss`, `params` | Loss/Metric 元数据摘要 |

## 调参与资源保留

`model_tune` 可通过 `tuner_params` 传递 `ModelTuner` 支持的参数，例如 `cv`、`random_state` 和内存保留策略。默认完整保留；`predictions`、`summary`、`best` 的功能差异以运行时的 ModelTuner 契约为准，不承诺已释放的逐折模型仍可读取。

当前 Skill 的 `model_tune` 只创建内存 Study，不支持 `storage`、`artifact_dir` 或 `retention="disk"`，也没有恢复 `inputs.tuner` 的操作。用户需要持久化调参或浏览看板时，应在 Python/Notebook 中创建带 `storage` 的 tuner，并通过独立命令启动：

```bash
hscredit dashboard sqlite:///hscredit.db
```

这个命令使用 hscredit 配色与包内字体；它不是 Skill JSON operation，不应把启动长期服务混入一次性制品请求。

`model_explain` 用于解释表。全局 SHAP 图和单样本力图/瀑布图使用 `hsviz.plot`，分别选择 `plot_model_feature_importance` 与 `plot_model_sample_shap`，输入名应与绘图方法参数一致。
