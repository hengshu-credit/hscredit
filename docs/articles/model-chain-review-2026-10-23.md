# 建模链路复核与函数文档修复

复核日期：2026-10-23。范围是全部公开模型的超参数搜索、最终重训、效果评估、模型报告，
同时检查 Python `help()` / IDE 中实际可见的参数说明，而不只检查外部教程。

## 本次发现并修复的问题

| 问题 | 原来的实际影响 | 修复后的行为 |
| --- | --- | --- |
| 三个类的说明不是有效 docstring | LR、RulesClassifier、ProbabilityScoreCard 把长说明写在 `artifact_kind` 之后，Python 不把它们识别为类说明 | 将说明移回第一条语句，参数、属性及示例能通过 `help()`/IDE 获取 |
| 搜索入口丢失目标列 | `ModelTuner(RandomForest(target='FPD')).fit(train_df)` 仍按默认 `target` 取标签而失败 | 未显式配置时继承实例/末端模型目标列；显式 target 传到最终 HSCredit 模型 |
| 评估入口不衔接 DataFrame 风格 | `fit(df)` 和搜索可用，但 LR/其他分类器 `evaluate(df)` 强制要求单独 y | 支持按 target 提取评估标签，显式 y 仍优先，目标列从特征中剔除 |
| 自动概率评分卡带入旧基础模型 | 搜索及最终重训可能复用传入模型的旧系数，`model__C` 等参数未真正参与重训 | 搜索中 prefit=None/False 克隆并重训基础模型；True 明确冻结且拒绝搜索其参数 |
| 其他预训练组件参数可被静默忽略 | 独立概率校准、已训练 LR 评分卡及其 Pipeline 嵌套参数看似被修改，实际仍使用旧模型 | 对冻结组件的参数搜索明确报错；需要基础训练时启用对应训练模式或直接搜索基础模型 |
| 严格报告误判正常能力限制 | LR `calculate_stats=False` 被记为统计失败；RBF SVM 的重要性属性在 `hasattr` 时抛错 | 显式关闭/不适用诊断和缺少原生重要性记为“不适用”并说明；真正异常仍保留失败状态 |
| 包装模型报告缺少可用字段/重要性 | 校准包装器未携带特征元数据，报告可能把标签也列为输入；只检查一层模型取不到重要性 | 校准包装器保留基础字段契约，报告沿 Pipeline/组合模型查找已有的重要性 |
| 包装器 tune 选项路由不同 | `catch`、`gc_after_trial`、`optimize_kwargs` 在部分模型上无法使用 | 统一传给优化阶段，训练配置仍通过 fit_params 传递 |
| 查曲线会加载整批磁盘模型 | retention='disk' 时仅查看曲线也会反序列化折模型 | 曲线直接读取保留的摘要数据，不加载折模型 |
| 参数含义和返回值不完整 | 过程方法缺参数/示例；to_excel 的返回类型与 return_result 不一致；base_odds 被混称为坏率 | 补齐静态 docstring、返回类型、示例、原生输出转换说明及评分刻度语义 |

函数文档覆盖 `fit`、`tune`、`ModelTuner` 构造/搜索/取模型/各点曲线和预测查询、`evaluate`、
`model_report`、`get_metrics`、`get_training_process`、`to_excel`，以及评分转换与校准入口。
API 文档页也显式展示继承的过程查询方法与构造器说明。

## 连续调用的推荐方式

在 Jupyter 中顺序执行 [完整示例 Notebook](../../examples/model_training_chain.ipynb)，
可在配置单元格修改输入和输出路径；也可通过统一验证器运行：

```console
python scripts/validate_examples.py --pattern model_training_chain.ipynb
```

示例按放款时间切分 970 行真实数据：训练582行、选参验证194行、独立测试194行。
训练仅使用约定的三项特征和 FPD；缺失填补只学习训练集统计量。
随机森林仍使用原生训练目标，成本敏感损失的 `metric(kind='business')` 用于选模、独立测试和报告。
金额分别显式传给训练权重、搜索验证权重、测试评估权重与报告权重。

主要调用顺序：

```python
metric = get_loss('cost_sensitive', fn_cost=5, fp_cost=1).metric(kind='business')
tuner = ModelTuner(model, search_space=space, metric=metric, cv=None)
params = tuner.fit(train_df, eval_set={'验证': valid_df}, n_trials=4)
best = tuner.get_best_model()
values = best.evaluate(test_df, metrics=[metric])
report = best.model_report(datasets={'独立测试': test_df}, metrics=[metric], verbose=False)
process = report.get_training_process()
status = report.to_excel('模型报告.xlsx', mode='strict', return_result=True)
assert status.complete
```

`tuner.fit()` 返回参数字典；`model.tune()` 返回最终模型；`model_report()` 默认返回报告对象；
`to_excel()` 默认返回路径，`return_result=True` 返回执行状态。不要把这四种返回值混用。
搜索验证集用于选参，不能当成独立测试集；`get_best_model()` 不自动把验证/测试数据并入训练。
最终重训和搜索试验是不同模型，应分别查看 `best.get_learning_curve()` 和 `tuner.get_learning_curve()`。

示例同时保存最终模型、完整调参制品和验收 JSON，并检查：

- 模型与报告的独立测试金融指标逐项相同。
- 最终重训样本数仍为582，外部验证/测试没有混入。
- 保存加载后的概率逐行一致。
- 严格模式报告完整，Excel 包含执行状态、模型口径与调参过程。

示例的测试 AUC 约0.3993，按原概率方向如实保留，没有翻转成大于0.5；
这说明接口核对通过，不代表演示参数具有可投产的模型效果。

## 仍需明确配置的边界

1. **Pipeline 的外部评估集**：目前用于外层搜索评分，不自动穿过每折预处理并传给末端提升器作早停。
   末端模型仍使用训练折内部验证。各折预处理应放在 Pipeline 内，不能先用全量数据学习 WOE 后再交叉验证。
2. **跨折的逐行金融数组**：绑定在 loss/metric 中的金额、LGD/EAD 等数组仍需显式切分；
   普通金额加权可使用 sample_weight/evaluation_weight/validation_weight，复杂情况使用 trial_objective。
3. **算法固有能力**：原生不支持自定义梯度的模型不能训练任意 BaseLoss，但可用同一 metric 评估、选模及出报告。
   没有逐轮接口的算法提供最终评估与训练记录；不生成虚构曲线。
4. **报告和解释**：完整报告默认不生成可选 SHAP 解释，需显式 explain_config。
   非线性 SVM 没有原生逐字段重要性，可选解释须独立配置；执行状态会说明缺少的是哪种能力。
5. **多目标选择**：默认按 metric 列表顺序在 Pareto 前沿选择最终点，并非自动执行某种未声明的业务折中。

## 验证入口

- `tests/test_models/test_model_workflow_docstrings.py`：检查公开参数和方法内示例可见。
- `tests/test_models/test_model_chain_contracts.py`：目标列贯穿、冻结组件、Pipeline/校准包装和真实执行 docstring 示例。
- `tests/test_models/test_all_model_workflow.py`：15类模型在真实数据上从 DataFrame 搜索一直到严格 Excel 导出。
- `artifacts/models_chain_20261023/real_data/链路验收.json`：真实数据的样本量、测试指标、保存恢复与报告一致性。

验证可选后端使用项目内兼容依赖，未修改全局 Anaconda 环境。

本轮结果：1863项模型/报告相关测试通过；其中包含78项公开入口文档契约检查，以及15类模型的严格 Excel
链路检查。排除了两个使用旧 ZeroMQ 内核的 Notebook 执行模块，另有1项按 slow/integration 标记未运行；
函数内完整示例通过直接 doctest 执行，真实数据流程完整运行。API HTML 已生成并核对关键参数、方法与示例；
全站其他模块仍有既有 RST 排版告警，未将站点整体宣称为零告警。

提交前补充验证：使用项目内兼容的 PyZMQ 27.2.0 后，原先两个 Notebook 内核执行模块已实际通过，
同时示例验证器和并行报告相关测试共131项通过；全局 Anaconda 环境保持不变。
