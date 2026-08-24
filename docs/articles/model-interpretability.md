# 模型可解释性完整工作流

HSCredit 基础安装已经包含 SHAP，无需安装额外的 `explain` extra。解释 API 将模型贡献、业务原因和行动建议明确分层：SHAP 解释模型为何产生当前输出，原因码仅筛选风险方向一致的不利贡献，反事实则是在模型与约束下搜索的非因果候选变化。

```python
from hscredit.core.models.explainability import CounterfactualExplainer, ModelExplainer

explainer = ModelExplainer(
    model,
    background_data=X_train,
    model_output="probability",
    target_class=1,
    random_state=42,
)
result = explainer.explain(X_test, max_samples=200)

global_table = explainer.get_global_report(result)
sample_table = explainer.get_sample_report(result, sample_id=result.sample_ids[0])
reasons = explainer.get_reason_codes(result, keep=3)
stability = explainer.get_stability_report(result, mode="sample", n_bootstrap=100)
```

`ExplanationResult` 固定保存样本索引、特征顺序、目标类别、实际输出尺度、背景数据摘要和数据指纹。二分类默认解释标签 `1`；多分类应显式传入 `target_class`。`model_output="probability"` 表示贡献和基准值可加回目标类别概率，`raw` 表示模型原始输出。

稳定性支持两种模式：`sample` 对固定解释结果进行样本 Bootstrap，不重训模型；`refit` 每轮 clone 并重训模型，再在固定验证集上解释。两者的含义不可混用。

## 原因码与反事实

`get_reason_codes()` 默认采用 `higher_output_higher_risk`，只输出正向推高风险的贡献；如果模型输出越高风险越低，应改为 `higher_output_lower_risk`。没有不利贡献时返回“无不利贡献”审计状态，不用有利因素补足数量。

```python
counter = CounterfactualExplainer(
    model,
    reference_data=X_train,
    constraints={
        "年龄": {"mutable": False},
        "收入": {"min": 0, "direction": "increase_only", "weight": 2.0},
    },
)
plans = counter.generate(X_test.iloc[[0]], target_probability=0.20, max_changes=2)
```

反事实支持不可变字段、上下界、允许类别、只增/只减、成本权重和最大变更特征数。结果是“模型条件下的非因果建议”，不代表真实因果效果、审批依据或授信承诺。

## Excel 报告

高成本解释默认关闭。启用后，`ModelReport` 在原有页面之后追加 `7-模型解释`：

```python
report = ModelReport(
    model,
    X_train=X_train,
    y_train=y_train,
    explain_config={
        "enabled": True,
        "data": X_test,
        "background_data": X_train,
        "max_samples": 200,
        "n_bootstrap": 50,
    },
)
report.to_excel("模型解释报告.xlsx")
```

完整教程见 `examples/27_model_interpretability.ipynb`；命令行生成报告可运行
`examples/27_model_interpretability.py --input <数据.xlsx> --output <报告.xlsx>`。
