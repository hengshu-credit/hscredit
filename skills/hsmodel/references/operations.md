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
