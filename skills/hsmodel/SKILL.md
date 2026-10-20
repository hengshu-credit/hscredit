---
name: hsmodel
description: Train and serialize allowlisted hscredit models, run AutoTuner, inspect losses, calibrate probabilities, and generate model explanation tables. Use when the task is model lifecycle work rather than a complete Excel report.
---

# HSCredit 模型

登记操作：`model_fit`、`model_predict`、`model_evaluate`、`compare_models`、`model_tune`、`model_calibrate`、`model_explain`、`component_create`、`loss_info`。

`model_fit` 使用 `parameters.model` 和 `model_params` 选择当前公开模型；训练数据来自 `inputs.data`，目标字段由 `target` 指定。`model_tune` 使用 `AutoTuner.create`，需要 `n_trials`，缺少 Optuna 时使用隔离 `tune` extra。`model_calibrate` 和 `model_explain` 使用 `inputs.model` 的可信制品或同进程对象。

模型制品只在用户明确确认来源后用 `trusted=true` 载入。大表结果保存为 Excel，模型和 tuner 保存为 hscredit artifact。
