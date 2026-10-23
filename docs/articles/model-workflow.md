# 模型训练、调参与保存

## 常用路径：单模型、单次验证

真实放款样例：[Notebook](../../examples/single_validation_workflow.ipynb)。
在 Jupyter 中按单元格顺序执行，或运行 `python scripts/validate_examples.py --pattern single_validation_workflow.ipynb`。

`fit(X_train, y_train)` 始终训练一个模型，不自动做交叉验证。
`tune()` / `ModelTuner` / `AutoTuner.create()` 的 `cv` 默认均为 `None`：
优先使用显式 `eval_set`；未提供时只按 `validation_fraction=0.2` 留出一次。
所有参数点复用同一次划分。只有显式 `cv=5`、分割器或索引序列才启用多折。

评估数据统一使用 XGBoost / LightGBM sklearn 接口的参数名 **`eval_set`**。
其容器形式参考 `ModelReport(datasets=...)`，训练和调参均可使用：

| 输入模式 | 示例 | 数据集名称 |
| --- | --- | --- |
| 原生列表 | `[(X_train, y_train), (X_valid, y_valid)]` | 数据集1、数据集2 |
| 具名字典 | `{'训练监控': (X_train, y_train), '验证集': (X_valid, y_valid)}` | 保留字典名称和顺序 |
| DataFrame 列表 | `[train_df, valid_df]` | 数据集1、数据集2；通过模型 target 提取标签 |
| DataFrame 字典 | `{'训练监控': train_df, '验证集': valid_df}` | 保留名称；通过 target 提取标签 |
| 混合形式 | `{'训练监控': train_df, '验证集': (X_valid, y_valid)}` | 每集独立解析标签 |
| 单个数据集 | `(X_valid, y_valid)` 或 `valid_df` | 数据集1 |

显式 `y` 优先于 DataFrame 中的目标列，目标列始终从特征中剔除；每个评估集沿用训练字段顺序。
含标签的 DataFrame 需要在模型或调参器上配置 `target='FPD'`。原生矩阵也可使用 ndarray/稀疏矩阵。
调参以**最后一个数据集**计算搜索指标，`validation_weight` 也对应这一集；所有输入均不会自动并入最终重训。
所有可训练特征模型收到完整评估集列表。NGBoost 原生早停使用最后一集，其余集合也保留最终指标与逐轮预测评估。
最终测试/OOT留在模型选定后单独评估，不作为选参数据。

名称可在模型的 `eval_names_`、`training_summary_['评估集名称']`，调参器的
`validation_summary_`、每次试验记录及 `get_validation_predictions()['数据集']` 中查看。
名称元数据描述传入的评估集，原生训练曲线保持框架自身的键名与能力；Pipeline的内部早停集见下文说明。

```python
from hscredit import LightGBM, get_loss, get_metric, list_losses, list_metrics

list_losses()                         # 中文用途、类名与全部构造参数
list_metrics()                        # 中文用途、参数和优化方向
loss = get_loss("cost_sensitive", fn_cost=5, fp_cost=1)
monitor = loss.metric()                # 训练代理损失，默认 kind="loss"
metric = loss.metric(kind="business")  # 实际误判成本；同一方法选择评估含义

model = LightGBM(
    objective=loss,
    eval_metric=[monitor, metric, get_metric("auc")],
    early_stopping_metric=monitor.name,
    early_stopping_rounds=20,
    n_estimators=300,
    random_state=42,
)
# 已有参数：直接单次训练。
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
# 需要选参：每个 point 只训练一次，用同一验证集比较。
best = model.tune(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    metric=metric,
    search_space={"max_depth": [2, 3, 4], "num_leaves": [4, 8]},
    n_trials=10,
)
best.tuner.validation_summary_         # 验证方式、划分数、最终训练样本数
best.tuner.get_optimization_history()  # 每个 point 的效果
best.tuner.get_validation_predictions()
best.tuner.get_trial_result(0)["各折"][0]["评估曲线"]
best.evaluate(X_test, y_test, metrics=[metric, get_metric("auc")])
```

验证集用于早停和选参，独立测试集/OOT用于最后验收。直接提升器复用留出验证集早停，
不额外随机拆一次；Pipeline 保留预处理契约，其末端模型若启用早停会在训练部分内部划分，
不会将未经预处理的验证特征直接送入末端模型。需要指定其早停集时仍可通过原生 fit 参数配置。

`validation_weight` 只对最后一个评估集的指标加权；训练权重仍为 `sample_weight`。
逐轮监控的权重使用各框架原生 `eval_sample_weight` 等配置，不把指标权重静默变成训练权重。
`eval_set` 不能与显式多折 `cv` 同时使用。按客户组留出时可传 `groups`；
需要时间验证时建议提供按时间划好的验证集，默认单次留出仍是分层随机划分。

选参后的 `best` 是一个重新训练的模型，不自动平均各折预测。提升器默认依据验证最佳轮数
在传给 `tune` 的完整训练输入上重训，独立 `eval_set` 不会自动并入。
若希望纳入验证数据，应显式组合数据后再训练，并保留独立测试集。

多折仅在需要时启用：`model.tune(X_train, y_train, cv=5, ...)`。
`get_validation_predictions()` 统一查看验证结果；多折时仍支持原有 `get_oof_predictions()`。
`get_fold_models()` 显式取得各折模型，需 `retention="full"/"disk"` 或保留完整最佳点；
它不会替换 `get_best_model()` 返回的单模型。单次留出只返回参加验证的样本，独立验证集不称为训练集OOF。

`get_metric(loss, kind="business")` 与 `loss.metric(kind="business")` 共用实现；
也可直接 `get_metric("topk_capture", top_ratio=.05)` 或传自定义概率函数与 `greater_is_better`。
旧 `business_metric()` 保留兼容并给出弃用提示，新代码统一使用 `metric(kind=...)`。
原有 `loss.metric()` 的损失值语义保持不变，避免把原来最小化的目标静默改成最大化。

**默认值迁移**：以前省略 `cv` 会运行五折，现在只做一次留出验证。要复现旧实验请显式设置 `cv=5`。
已有持久化 Study 不应切换验证方式续跑；请为新的验证口径建立新 Study。

## 结果核对、训练过程与金融指标报告

函数内也提供完整参数说明和示例，可在 IDE 中查看或调用 `help(ModelTuner)`、
`help(model.model_report)`、`help(report.to_excel)`。新增的
[完整链路 Notebook](../../examples/model_training_chain.ipynb) 使用真实数据从搜索运行到模型保存和 Excel 导出；
[链路复核记录](model-chain-review-2026-10-23.md) 说明已修复断点与剩余能力边界。

### 全模型适配范围

`eval_set` 同时支持 `(X, y)`、`[(X, y), ...]`、`{"时间验证": (X, y)}`、单个含目标列的
DataFrame、DataFrame 列表、具名 DataFrame 字典和混合形式。含目标列时配置 `target`；显式 y 优先，
目标列始终从特征剔除。以下模型均支持 `fit(..., eval_set=..., eval_metric=...)`、
`evaluate(..., metrics=...)`、`get_eval_results()`、`get_training_history()`、`get_learning_curve()`：

| 模型 | 验证过程 | 训练损失 |
| --- | --- | --- |
| LogisticRegression、RandomForest、ExtraTrees、SVM、DecisionTreeClassifier | 完成训练后评估各集，不因 eval_set 拆分训练样本 | 使用算法原生目标；统一金融指标可用于离线评估和选模 |
| GradientBoosting | 保留原生训练损失，并用 staged_predict_proba 记录各评估集逐轮指标 | sklearn 原生 loss；完整数据重训关闭 n_iter_no_change |
| XGBoost、LightGBM | 原生训练曲线、早停及最终模型评估 | objective 接收 BaseLoss，处理 raw margin 与概率的链接转换 |
| CatBoost | 原生曲线、早停及最终模型评估，保留原生 Pool 兼容 | objective 接收可独立逐样本计算的 BaseLoss |
| NGBoost | 最后一集用于原生早停；所有评估集保留最终指标及逐轮预测评估 | objective 接收可独立逐样本计算的 BaseLoss，自动配套 Score/Dist |
| ScoreCard、RoundScoreCard | 最终概率指标；记录 WOE 输入、评分公式、方向与取整配置 | 由底层 LR 训练；取整只影响评分，不替换底层概率 |
| ProbabilityScoreCard | 最终概率指标，报告保留底层模型与评分转换记录 | 由 model 的算法决定；prefit 模式复用已有模型 |
| RulesClassifier | 最终规则评估；分类指标使用真实 AND/OR 逻辑及 threshold | 验证规则和字段，不进行梯度训练 |
| ProbabilityCalibrator | 最终校准概率指标、校准前后指标、基础模型是否重训 | 使用所选校准算法；calib_ratio 控制基础训练和校准划分 |

上述 15 类模型均可通过 `ModelTuner(model, metric=..., cv=None)` 做单次验证搜索；显式 `cv=2`
等配置启用多折。评分卡、规则模型和校准器需要显式提供适合其业务的 `search_space`。
`CalibratedModel` 组合已训练对象，支持统一评估和模型报告；需要搜索时配置可训练的基础模型或
`ProbabilityCalibrator`。预训练评分卡和校准包装器不会为了统一 API 而偷偷重新训练基础模型。

纯变换组件也保留过程：`Standard/Linear/Quantile/BoxCoxScoreTransformer` 和统一 `ScoreTransformer`
记录每次拟合；`Platt/Isotonic/Beta/HistogramCalibrator` 支持同一金融 metric 评估校准后的概率；
`ScoreDriftCalibrator` 的 linear/quantile/binning 方法记录拟合。
它们保留概率或评分输入语义，不能把特征矩阵训练参数直接传给纯变换。

没有逐轮接口的算法返回空曲线表，最终结果在 `get_eval_results()`，不会把最终值伪装成一轮训练。
NGBoost 和 GradientBoosting 的额外指标使用 staged 预测计算，不重新训练；会增加与评估集大小、轮数相应的预测开销。
Boosting 的 `get_eval_results()` 默认计算 AUC、KS、LogLoss；传入 BaseMetric 时计算这些对象。
框架专用的字符串指标及原生回调值保留在 `evals_result_` / `get_learning_curve()`。
NGBoost 原生早停要求训练和验证权重同时存在。只给其中一方时，包装器用全1表示另一方的等权语义，
并在“内部转换 / 原生权重适配”中记录；不会拿训练金额代替验证金额。

```python
from hscredit import LogisticRegression, get_loss, get_metric

cost = get_loss("cost_sensitive", fn_cost=7, fp_cost=1).metric(kind="business")
model = LogisticRegression(target="FPD", calculate_stats=False, max_iter=1000)
model.fit(train_df, eval_set={"时间验证": valid_df}, eval_metric=[cost, get_metric("auc")])
model.get_eval_results()             # 最终模型，各评估集的指标
model.get_training_history()         # 包含内部转换与训练配置
model.get_learning_curve()           # LR 无公开逐轮曲线，返回空表
report = model.model_report(datasets={"验证": valid_df}, metrics=[cost], verbose=False)
report.get_training_process()        # 报告持有的训练/调参/底层变换快照
```

所有特征模型的完整报告统一用 `model.model_report(...)` 或 `ModelReport(model, datasets=...)`。
原有 `model.report(...)` 保持兼容：ProbabilityScoreCard.report 仍为评分分箱表，
ProbabilityCalibrator.report 仍为校准对照表。新代码使用 model_report 可避开这些历史差异。

`loss` 的**训练能力**取决于底层算法；其 `metric()` 的**评估能力**与模型类型无关。
例如随机森林不能接受任意梯度目标，但可以用同一个成本指标评价、搜索和出报告。
不能训练的 objective 会明确报错。CatBoost/NGBoost 不接受依赖全量排序或绑定逐行数组的训练损失，
可用原生 sample_weight 表达金额权重，或训练后使用绑定相应评估样本的 metric。
CatBoost 指标回调也无法可靠标识外部金融数组的行身份，包装器拒绝这类绑定；不会按长度猜数据集。
框架协议参见 [CatBoost 自定义损失](https://catboost.ai/docs/en/concepts/python-usages-examples#user-defined-loss-function)
和 [NGBoost Score/Dist](https://stanfordmlgroup.github.io/ngboost/5-dev.html)。

训练、选参、离线评估和报告可以使用同一个 `get_metric(...)` 或 `loss.metric(...)` 对象：

```python
metric = loss.metric(kind="business")
offline = best.evaluate(X_test, y_test, metrics=[metric])
report = best.report(
    datasets={"测试集": (X_test, y_test)}, metrics=[metric], verbose=False,
)
report.get_metrics()                   # 包含同名、同数值的业务指标
best.get_training_history()            # 历次训练状态、配置、实际参数和耗时
best.get_learning_curve()              # 最新训练逐轮长表
best.get_learning_curve(revision=1)    # 旧训练曲线，受 max_history 保留次数限制
best.tuner.get_learning_curve(0)        # 第0个参数点各折的逐轮记录
report.get_training_process()          # 报告绑定时的训练与调参过程快照
report.to_excel("模型报告.xlsx", mode="strict", return_result=True)
```

Excel新增“8-训练与调参过程”：模型口径、历史训练、曲线、逐点参数与得分、各折指标、选中试验曲线和重训配置。
最终重训没有配置评估集时，本身没有验证曲线；对应搜索点的曲线仍单独保留，不能把二者混为同一次训练。
默认训练历史摘要继续限制原始矩阵/参数对象的保存量，但逐轮曲线独立保存，不再按32或256项截断。
`get_training_history()` 返回有界摘要，显式full历史的原始参数仍在 `training_history_` 中。

报告绑定训练版本与配置快照。模型重新fit后，旧报告会明确报错，需重新创建；不会展示旧预测和新模型参数。
调参器继续搜索时，已经创建的报告保留其当时的过程快照。过程缺少制品时会在过程表中写明原因。

指标口径现在固定如下：

- 模型概率评估的AUC/Gini、调参字符串`auc`与指标对象保持坏样本概率方向；反向预测不会被自动美化。
  通用 `hscredit.core.metrics.auc` 仍保留既有auto默认，处理未知方向评分时可显式使用。
- `logloss` 与 `log_loss` 都是正损失、最小化；需要负损失时使用 `neg_log_loss`。
  旧Study与新指标协议不可混用，迁移后创建新Study。
- 报告 `prediction_method="predict_score"` 根据模型评分刻度确定方向；也可显式传 `score_direction="higher_safe"/"higher_risk"`。
  原始评分保持不变，风险排序单独处理。取整/截断造成的同分可能使评分与原始概率的指标略有差别。
- TOP表、模型评估和业务指标共用“人数向上取整、边界同分等比例计入”；无有效坏样本时报告显示空值。
  分档LIFT图则使用统一等频分箱器和 `compute_bin_stats`，同分不会按行序随意拆开；分档LIFT与累计TOP-LIFT有不同含义。
- KS、AUC、金融指标与TOP表支持显式权重。报告 `sample_weight="放款金额"` 或 `{数据集名称: 权重}`
  用于指标统计；样本数、字段描述和基础分箱样本表仍保留人数口径，金额分箱另用 `amount` 指定。
  统一金融指标需要概率输入，不能把0–1000信用分直接当作概率计算利润或交叉熵。

自定义训练损失不能再静默忽略 `scale_pos_weight/is_unbalance`；无效组合在fit前报错。
应通过损失自身参数或 `sample_weight` 控制成本。XGBoost的自定义损失下，auto类别权重的实际值为1。

XGBoost/LightGBM单次自动留出现在会同步切分 `AmountWeightedLoss` 的金额，及 `ExpectedValueLoss` 的逐行金融参数，
不修改调用者的原始loss；切分样本数记录在训练摘要中。外部验证需显式配置其自身的配套指标，
例如 `eval_metric=loss.metric(amounts=valid_amounts)`。多个验证集不能共用一份绑定数组。
复杂逐行参数的CV自动路由仍未开放；现阶段金额加权可用 `sample_weight` / `validation_weight` / `evaluation_weight`，
复杂场景使用明确的自定义trial流程。此限制是为了防止同长度、不同样本的数组被错误复用。

## 1. 最短训练路径

```python
import pandas as pd
from hscredit import LightGBM

features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
df = pd.read_excel("examples/hscredit_yyp.xlsx")
model = LightGBM(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=4,
    target="FPD",
    eval_metric=["auc", "ks"],
    early_stopping_rounds=20,
)
model.fit(df[features + ["FPD"]])
probability = model.predict_proba(df[features])[:, 1]
score = model.predict_score(df[features])
```

这段代码演示接口，效果评估应另留测试集。标签必须同时包含 `0` 和 `1`，`1` 表示坏样本。
也可以使用 `fit(X, y, sample_weight=weights)`；显式 `y` 优先，并从特征中移除 `target` 列。
DataFrame 预测和验证集会按训练字段重新排序；缺字段、重名列、标签或权重长度不符会直接报中文错误。

这一输入契约也适用于概率校准器、概率评分卡、传统评分卡和调参器：显式 `y` 优先，
但 `target` 列始终剔除；样本参数按位置而不是 pandas 索引标签对齐。
正类概率按真实 `classes_` 查找，`ProbabilityScoreCard(positive_class=1)` 默认选择坏样本类别 1，
不能以“最后一列”代替正类。失败的重复训练不可继续使用旧的预测状态。

`ScoreCard.fit` 会重新训练自身创建的 LR，并按当前 PDO 等配置刷新刻度；显式提供的预训练 LR 仍可复用。
`ScoreCard(pipeline=...)` 要求 LR 前的输入确实是相同的 WOE；若 Pipeline 使用分箱索引、缩放或其他变换，
请用 `ProbabilityScoreCard(model=pipeline)` 保留完整预处理，不能把它静默替换为 WOE。

早停需要验证数据。包装模型未收到 `eval_set` 时，会按 `validation_fraction` 从训练输入划出验证集；
不启用早停时使用全部输入。训练权重、XGBoost `base_margin`、LightGBM `init_score`、
CatBoost `baseline` 会跟随自动划分同步切分。

类别字段保留原来的 DataFrame 类型：LightGBM 可使用 pandas `category`，XGBoost 使用其原生
`enable_categorical=True` 配置；CatBoost 可传 `cat_features=["商品类别"]`。
包装模型以 DataFrame、二维数组或框架支持的稀疏矩阵为输入；框架专属容器需按该框架支持情况使用。

## 2. 参数如何复用

```python
from sklearn.base import clone
from xgboost import XGBClassifier
from hscredit import XGBoost

native_params = XGBClassifier(
    n_estimators=50, max_depth=3, max_bin=64, n_jobs=2, random_state=42,
).get_params()
source = XGBoost(**native_params, target="FPD")
model = clone(source)                   # 保留 target、原生扩展参数和自定义配置
model.set_params(max_bin=128)           # 可用于 Pipeline / GridSearchCV
model.fit(df[features], df["FPD"])
actual_params = model.get_native_params()
native_model = model.get_native_model()
```

- `get_params()` 返回构造配置，适合克隆、搜索和重新配置。
- `get_native_params()` 返回实际训练使用的原生构造参数；自动类别权重等计算值在这里检查。
- `get_native_model()` 返回已训练的底层对象，可以直接使用原生方法。
- `params={...}` 继续支持。初始化时保留历史规则：该字典优先于同名关键字；后续 `set_params()`
  的显式更新优先于旧字典，不会修改调用者传入的字典。
- 常用别名会被规范化，例如 CatBoost 的 `n_estimators`、`max_depth`、`reg_lambda`，
  LightGBM 的 `num_iterations`、`feature_fraction`，XGBoost 的 `eta`、`lambda`。
- 各框架的专属参数、损失函数和评估函数保留原生语义。XGBoost 的自定义评估函数返回数值，
  LightGBM 返回 `(名称, 数值, 是否越大越好)`，CatBoost 接收原生评估对象。
  `predict()` / `predict_proba()` 的额外预测参数也会传递给原生模型。

特征重要性有两个用途：`feature_importances_` 始终与输入字段顺序一致，供 sklearn 筛选器使用；
`get_feature_importances(importance_type=...)` 返回排序后的具名 Series，供查看和报告使用。

## 3. 保存完整模型与训练过程

```python
model.save("artifacts/model.joblib")
restored = type(model).load("artifacts/model.joblib")

restored.training_summary_              # 最近一次训练的状态、参数、用时、依赖版本等
restored.training_history_              # 历次训练，包括失败和中断
restored.evals_result_                  # 框架实际产生的逐轮指标
restored.best_iteration_
restored.predict_score(df[features])
```

默认保存使用 cloudpickle，支持 Notebook 中定义的函数、损失和回调，原有 joblib/pickle/dill
引擎选项与旧文件加载方式保留。完整模型文件写入成功后才替换目标，序列化失败不会破坏原文件。
未实际计算的指标不会伪造为训练曲线；查看逐轮验证曲线时，应配置 `eval_set` 或启用自动早停。

| 保存方式 | 内容与用途 |
| --- | --- |
| `save("model.joblib")` | 完整 Python 对象：模型、评分转换器、训练历史及关联调参器 |
| `save_artifact(...)` | 既有统一制品接口，可显式指定 `engine="cloudpickle"` |
| `save("model.json")` | 支持框架的 JSON 清单，配套原生模型、评分转换器和 `.state.pkl` 完整状态文件；需一起移动 |
| `save_model(...)` | 原生模型文件与评分转换器附属文件，保留已有接口 |
| `get_native_model()` | 直接调用底层框架的导出、树结构、叶节点、预测或其他专属能力 |
| `save_inference("model.joblib")` | 风险模型/LR 的轻量推理对象：移除 tuner、训练历史和划分索引，保留预测、评分和字段契约；不修改原对象 |

输入预处理也是复用的一部分：推荐把编码、标准化与模型放进 sklearn Pipeline 一起保存；
如果预处理在外部完成，必须同时保存已拟合的预处理器和字段顺序。
训练记录保存配置和模型过程，不自动复制作为位置参数传入的原始训练矩阵。
CatBoost 验证集 Pool 的原生句柄不能深拷贝，训练记录保留其标签、权重、基线与字段结构。

Python 完整对象不承诺任意跨版本加载。版本迁移时，先在新环境运行验证脚本；
若原框架的对象结构变化，使用该框架支持的原生导出和加载方式。记录中的 `依赖版本` 用于定位来源环境。

## 4. 在 tune 中直接使用 Optuna

`tune()` 返回最佳模型；原始模型和返回模型的 `.tuner` 指向同一调参器。
原始模型本身不会被替换为最佳模型。

```python
import optuna
from pathlib import Path
from hscredit import RandomForest

Path("artifacts").mkdir(exist_ok=True)
study = optuna.create_study(
    direction="maximize",
    study_name="FPD随机森林",
    storage="sqlite:///artifacts/tuning.db",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)

def search(trial):
    depth = trial.suggest_int("max_depth", 2, 8)
    return {
        "max_depth": depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        # 可继续调用 suggest_float、suggest_categorical，或用 if 定义条件空间。
    }

def on_trial(study, trial):
    print(trial.number, trial.state, trial.values)

model = RandomForest(n_estimators=100, target="FPD", random_state=42, n_jobs=4)
best = model.tune(
    df[features + ["FPD"]],
    search_space=search,
    study=study,
    callbacks=[on_trial],
    metric="auc",
    cv=3,
    n_trials=20,
    catch=(ValueError,),
    gc_after_trial=True,
    artifact_dir="artifacts/trials",
)
```

先创建数据库父目录。传入已有 Study 时，采样器、剪枝器与持久化连接由该 Study 管理；
不要在包装器上重复配置另一套 Study 设置。`direction` 必须与已有 Study 一致。
已有字典搜索空间、Optuna 分布对象、多目标、手工点、Hyperopt/skopt 风格声明和各类可视化保持可用。

| 需求 | 入口 |
| --- | --- |
| 原生 Trial 条件采样 | `search_space=lambda trial: {...}` |
| 已有 Study、采样器、剪枝器 | `study=study`；自行创建 Study 时也可用 `sampler=`、`pruner=` |
| Trial 完成后的回调 | `callbacks=[fn]`，签名与 `Study.optimize` 一致 |
| 异常策略、垃圾回收 | `catch=(...)`、`gc_after_trial=True` |
| 底层模型 fit 参数 | `fit_params={"callbacks": [...], ...}` |
| 每折创建原生训练回调 | `fit_params_factory(trial, fold_index) -> dict` |
| 分组、时间、重复验证 | `cv=GroupKFold(...)`、`cv=TimeSeriesSplit(...)` 或索引迭代器；必要时 `groups=` |
| 原生完整目标函数 | `trial_objective=fn`，`fn(trial)` 完全接管试验；与既有 `objective(y, p)` 评分函数区分 |
| 原生高级操作 | `model.tuner.study_.enqueue_trial(...)`、`stop()`、`ask()`、`tell()` 等原生 Study 方法 |

`fit_params_factory` 每折调用一次；最终重训会以 `(None, None)` 调用，应在此分支返回最终训练参数，
避免把绑定某个 Trial 的剪枝回调复用到最终模型上。已知样本级参数如 `sample_weight`、`base_margin`、
`init_score`、`baseline` 及 Pipeline 中对应的 `步骤名__参数` 会按折同步切分。
外部固定 `eval_set` 会按原生语义传递，应由调用者确保它不包含外层验证或最终测试样本。

自动评分采用外层交叉验证；包装模型内部早停和原生提升器的内部分割均只使用训练折。
单目标试验每完成一折执行一次 `trial.report()` / `should_prune()`。
多目标继续返回各目标结果与 Pareto 前沿，不调用 Optuna 不支持的多目标 `report()`。

为了维持现有 CPU 预算和可中断行为，Trial 依旧顺序执行，`n_jobs` 分配给当前模型。
这不是将 `n_jobs` 直接当成 `Study.optimize` 的并行试验数。

## 5. 保留每折内容、继续搜索、复用最佳模型

```python
tuner = best.tuner
tuner.get_trial_result(0)                # 参数、每折模型、训练/验证预测、指标、曲线和异常
tuner.get_oof_predictions()              # 样本位置、真实标签、折外预测概率、验证次数
tuner.get_optimization_history()         # 保留原有 Optuna 风格历史表
tuner.get_best_model() is best           # True：不会再次训练

tuner.save("artifacts/tuner.joblib")
restored = type(tuner).load("artifacts/tuner.joblib")
restored.enqueue_trial({"max_depth": 4})
restored.fit(df[features + ["FPD"]], n_trials=10)
new_best = restored.get_best_model()
```

完整调参制品包括输入数据、标签和权重，适合继续原任务；换数据集或改变验证口径时，应创建新的 Study。
SQLite 保留 Optuna 试验状态；`artifact_dir` 额外逐折写入模型过程，可在新调参器中通过 Study 记录的路径恢复。
只使用 SQLite、不保存完整对象或配置制品目录，不能恢复已丢失的 Python 模型和预测数组。
同样，只调用原生 `trial_objective` 时，内部训练过程由用户函数管理，自动逐折记录不适用。

通过 `ModelTuner(..., retention=...)` 或 `model.tune(..., retention=...)` 选择结果保留策略。
所有策略都保留完整 Optuna Study、各试验指标、搜索参数、状态与最佳模型缓存，不改变训练和搜索计算。

| retention | 常驻结果 | OOF 与折模型 |
| --- | --- | --- |
| `full` | 所有折模型、训练/验证预测、完整训练记录 | 全部可用，默认行为 |
| `predictions` | 训练/验证预测、指标、精简训练记录 | OOF 可用，不保留折模型 |
| `summary` | 指标、曲线、样本数、早停轮数、异常 | 不保留逐样本数组；请求 OOF 会明确报错 |
| `best` | 当前最佳试验完整结果；其余仅摘要 | 最佳试验可用；其他试验的 OOF 不再保留 |
| `disk` | 逐折制品索引与摘要 | 必须设置 artifact_dir，逐折写盘、按需读取 |

未指定 `retention` 时继续兼容 `store_models=True/False`，分别对应 `full/predictions`。
显式 `retention` 优先。`disk` 与“full 加 artifact_dir”不同：每折写入成功后就释放其常驻模型和数组，
`get_trial_result(n, load=False)` 只返回摘要；默认 `load=True` 返回完整结果但不把折模型放回常驻缓存。
`get_oof_predictions()` 逐折读取并聚合，重复验证仍取均值，未参加验证的样本保留缺失值。
磁盘档的完整制品依赖其折文件，须同时保留 `artifact_dir`；缺失文件会明确报错，不重新训练补造结果。
将整个折制品目录复制到其他位置后，可用 `ModelTuner.load(path, artifact_dir="新目录")` 恢复读取。
若使用 SQLite 等外部 Study 存储，还需独立维护对应数据库的可访问性。

输入数据默认仍保留以支持最终重训与续跑。取得最佳模型后可调用 `tuner.release_training_data()`，
释放调参器持有的输入、标签、权重、CV 索引和已知样本级 fit 参数；该操作不删除折结果或磁盘制品。
缓存最佳模型仍可预测；重训及完整标签的 OOF 需要重新 `fit`，续跑时也必须重新提供所需样本参数。
`save_inference` 不携带整个搜索历史，适合部署；完整 `save` 继续满足复盘与续跑用途。
上述清理仅针对库管理的字段，不会任意删除用户函数闭包捕获的数据或底层模型推理所需的状态。
自定义函数返回不能放入 Optuna JSON 属性的模型对象或损失对象时，完整配置保存在调参制品中。

最佳模型只训练一次并缓存。Boosting 最终默认使用各折最佳轮数的中位数关闭内部早停，在全部输入上训练；
可用 `get_best_model(refit=True, full_data=False)` 显式重训并保留内部验证。
中断会继续抛出 `KeyboardInterrupt`，不会悄悄开始最终重训，但已经完成的试验和历史仍可检查、保存。

## 6. 两种调参入口都能查看完整搜索过程

两种方式均保留同一个原生 Optuna Study，不只保留最佳参数。训练完成后，
既可以通过 `visualization` 自动绑定 Study，也可以把 `get_study()` 的返回值直接交给 Optuna。

```python
from hscredit import RandomForest, ModelTuner

# 方式一：model.tune。原模型和最佳模型均关联同一个搜索器。
model = RandomForest(n_estimators=30, random_state=42, n_jobs=2)
best = model.tune(
    df[features], df["FPD"], search_space={"max_depth": [2, 3, 4, 5]},
    n_trials=8, cv=3, n_jobs=2,
)
assert model.tuner is best.tuner
best.tuner.visualization.plot_optimization_history().show()
best.tuner.visualization.plot_intermediate_values().show()

# 方式二：直接使用 ModelTuner，fit 后立即可分析，不需要先调用 get_best_model。
tuner = ModelTuner(
    RandomForest(n_estimators=30, random_state=42, n_jobs=2),
    search_space={"max_depth": [2, 3, 4, 5]}, metric="auc", cv=3, n_jobs=2,
)
tuner.fit(df[features], df["FPD"], n_trials=8)
tuner.visualization.plot_timeline().show()
tuner.visualization.plot_rank(params=["max_depth"]).show()

# 与原生 Optuna 完全互通。
import optuna
study = tuner.get_study()                 # 与 tuner.study_ 是同一对象
optuna.visualization.plot_param_importances(study).show()
study.trials_dataframe()                  # 全部试验的参数、状态、得分、时间及属性
study.trials[0].intermediate_values        # 每折结束时的累计平均目标值
study.trials[0].user_attrs["各折指标"]     # 每折、每个目标的实际分数
study.sampler                            # 本次搜索的原生采样器
study.pruner                             # 本次搜索的原生剪枝器
```

`visualization` 根据当前安装的 Optuna 动态提供全部公开入口，包括其后续新增的绘图函数。
用 `dir(tuner.visualization)` 或 `dir(tuner.visualization.matplotlib)` 可查看当前版本提供的函数。
原生参数和返回对象不变，只自动传入当前 Study；显式传 `study=[study1, study2]` 可比较多个搜索。

| 可查看的内容 | 原生函数 |
| --- | --- |
| 搜索得分历史、中间值、时间轴 | `plot_optimization_history`、`plot_intermediate_values`、`plot_timeline` |
| 参数重要性、切片、等高线、排名、平行坐标 | `plot_param_importances`、`plot_slice`、`plot_contour`、`plot_rank`、`plot_parallel_coordinate` |
| 得分分布 | `plot_edf` |
| 多目标 Pareto 前沿与超体积历史 | `plot_pareto_front`、`plot_hypervolume_history` |
| 搜索终止改进诊断 | `plot_terminator_improvement` |

Matplotlib 后端也完整开放，返回原生 Axes 或 Axes 数组：

```python
import matplotlib.pyplot as plt

ax = tuner.visualization.matplotlib.plot_optimization_history()
ax.figure.savefig("search-history.png", dpi=150, bbox_inches="tight")
plt.show()
```

这些图仍遵循 [Optuna 原生绘图条件](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html)：
多目标单曲线图的 `target` 是函数，例如 `target=lambda trial: trial.values[1]`；
超体积图需要多目标 Study 和显式 `reference_point`。
终止改进图按 Optuna 版本和 evaluator 要求使用相应依赖与足够的试验；
Optuna 低于 4.9 时，自动单目标交叉验证会通过官方接口记录专用 CV 分数。
Optuna 4.9 起弃用了 `terminator`，默认不再为此调用弃用接口；常规各折指标、
中间值和曲线仍完整保留。若确需旧终止改进图，可显式配置
`ModelTuner(..., record_terminator_scores=True)`，此时保留上游弃用提示。
只有一折或用户完全接管的 `trial_objective`，需按原生要求提供适用的误差估计或记录。

`plot_intermediate_values` 默认显示**各交叉验证折结束时的累计平均目标值**。
每个模型的逐轮学习曲线在 `tuner.get_trial_result(编号)["各折"][折编号]["评估曲线"]` 中；
失败、剪枝的 Trial、时间和已上报的中间值也保留在 Study，不会为绘图而过滤。
多目标的逐折分数保存在 `各折指标`，不伪造 Optuna 不支持的多目标 intermediate_values。

原有 `tuner.plot_*` 便捷方法继续支持整数 `target` 与 HSCredit 搜索空间参数名转换；
`visualization` 则使用 Optuna 原生 `params` / `target` 语义。
保存最佳模型或调参器后，重新加载仍可通过相同入口画图；访问绘图入口本身不会重新训练模型。

## 7. 可执行 Notebook 演示

```console
python scripts/validate_examples.py --pattern 28_model_workflow.ipynb
```

在 Jupyter 中打开 [模型工作流 Notebook](../../examples/28_model_workflow.ipynb)，按单元格顺序执行即可。
它优先使用 `examples/hscredit_yyp.xlsx` 的指定三项特征和 `FPD`；没有工作簿时使用固定种子的模拟数据。
覆盖双 API 防泄漏、字段顺序、概率校准、五档结果保留的数值等价、按需读取、
完整与推理制品保存恢复、续跑和显式资源释放。缺失值只使用训练集统计量处理。
输出默认位于独立临时目录，可在配置单元格修改路径。示例不再依赖独立 `.py` 演示脚本。

[模型家族验证 Notebook](../../examples/model_family_validation.ipynb) 补充十种基础分类器的训练、
概率/评分保存恢复、类别特征与 SQLite 调参续跑验证；需要兼容的 boosting 可选依赖。
