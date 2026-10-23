# models 模块可用性与结果一致性研究

研究日期：2026-10-23。范围：当前工作区 `hscredit/core/models/`，以及相连的指标、报告、文档与测试。

后续落实：已实现“默认单次验证、显式启用多折、最终单模型重训”和统一
`get_loss/get_metric`、`loss.metric(kind=...)` 入口。以下清单为改造进度，正文保留原始研究快照与复现值，
不代表当前仍有相同缺陷。最新用法及完整模型范围见 [模型工作流](model-workflow.md)。

- 已修复：信用分报告 TOP 风险方向、概率 AUC 自动翻转、TOP 同分和人数口径。
- 已修复：模型重训后的旧报告缓存混用；包装模型还检查底层模型和校准/转换器状态。
- 已修复：曲线超过 32 点被截断；最终重训和选中试验的曲线分开保留。
- 已接通：统一金融指标到模型评估、搜索和 ModelReport；金额权重显式配置。
- 已接通：训练/调参/最终评估/内部转换记录到报告，Excel 包含执行状态和缺失过程说明。
- 已覆盖：10 种普通分类器、3 种评分卡、规则分类器、概率校准包装器；单次验证和显式 CV 均验证。
- 已补齐：NGBoost 自定义训练目标与指标监控、GradientBoosting 逐轮验证指标、LR 的 eval_set，
  以及纯评分转换、四种概率校准算法、三种漂移校准方法的过程记录。
- 已修复：XGBoost/LightGBM 自动留出时逐行金额/LGD/EAD 参数同步切分，拒绝静默无效的原生类权重。
- 仍需显式处理：跨多折/多个独立数据集的绑定金融数组不能自动猜行身份；用 sample_weight /
  evaluation_weight / validation_weight 或 trial_objective 明确切分。
- 算法边界：非可加损失不能交给分批导数回调；无公开逐轮接口的算法提供最终评估和训练记录。

## 结论

目前主要问题是：**数据、预测含义、指标口径和训练版本没有作为同一份结果贯穿训练、调参、解释与报告。**
模型、损失、指标、调参、报告分别有不少能力，但使用者必须自己记住它们之间的约定。
因此只补参数文档不够，直接再加一个“一键建模”入口也会继续隐藏问题。

研究阶段没有修改库实现或升级 Python 环境；当时新增的是复现脚本、实测证据和完整调用示例。
以下明确区分已运行复现、静态确认和改造建议。旧设计复盘、AGENTS 中的历史测试数字不作为当前验证结果。

## 1. 已复现的结果问题

### 1.1 高优先级：信用分报告取错风险尾部

位置：`hscredit/report/model_report.py:2187`，`hscredit/core/models/scorecard_support.py:15`。

报告支持 `prediction_method='predict_score'`，但 TOP 表固定执行 `np.argsort(-prediction)`。
默认信用分为 `direction='descending'`，即概率越大、评分越低。高分组是低风险组。

真实数据：970 行，按 FPD 分层随机划分为训练 776 行、测试 194 行，固定种子 42。
同一 LightGBM 模型、同一测试集的 TOP 10% 结果：

| 计算方式 | LIFT |
| --- | ---: |
| 报告使用坏样本概率 | 1.134503 |
| 报告使用默认信用分 | 0.378168 |
| 信用分按低分高风险排序 | 1.134503 |

修复方向：显式保存预测类型与风险方向，让 TOP、捕获率、坏账率、图表使用同一风险排序。
不能通过把全部分值取反来修复展示：报告仍应保留用户选择的原始分值。

### 1.2 高优先级：字符串 AUC 与指标对象不是同一口径

位置：`hscredit/core/models/tuning/_metrics.py:96`、`hscredit/core/models/losses/business_metrics.py:46`、
`hscredit/core/metrics/classification.py:122`、`hscredit/report/model_report.py:268`。

`metric='auc'` 走自动方向；`AUCMetric()` 保留坏样本概率的固定方向。
报告的 AUC 也调用默认自动方向。对 `y=[0,0,1,1]`、`p=[.9,.8,.2,.1]`：

| 入口 | AUC |
| --- | ---: |
| sklearn 原始 AUC | 0 |
| 调参字符串 `auc` | 1 |
| `AUCMetric()` | 0 |

自动方向适合方向未知的业务评分，但不适合已声明为 `P(y=1)` 的预测概率。
它可能奖励反向预测，也使信用分 TOP 排序问题被总览 AUC 掩盖。
当前可用规避：调参用 `AUCMetric()`；模型离线评估可指定 `score_direction='higher_risk'`。
报告尚无相同的公开方向参数，不能只靠换一个调用参数完整解决。

### 1.3 高优先级：报告和调参对 TOP 人数、同分边界的处理不同

位置：`hscredit/report/model_report.py:2194`，`hscredit/core/models/losses/business_metrics.py:134`，
`hscredit/core/models/tuning/_metrics.py:112`。

报告按 `int(n * pct)` 向下取整，并直接切排序数组；调参的 `TopKLiftMetric` 向上取整，边界同分按比例计入。
10 人、2 个坏样本、全部预测为 0.5，TOP 10%：只改变样本行序，报告 LIFT 可从 **5 变为 0**；
指标对象两次均为 **1**。这不是模型性能变化。

修复方向：共用名单选择实现；报告披露覆盖人数、同分处理、分母、人数/金额口径。

### 1.4 高优先级：同一报告可能混用旧预测与新模型参数

位置：`hscredit/report/model_report.py:624`、`:895`、`:1236`、`:4594`。

报告缓存创建时的预测与指标，却继续持有可变模型对象；参数页在导出时读取 `model.get_params()`。
模型从 3 轮重训到 45 轮后，旧报告仍保留 AUC **0.503216**，模型当前 AUC 为 **0.506321**；
报告所引用模型的轮数已是 45，缓存预测与当前预测最大差 **0.295465**。

修复方向：报告绑定模型训练修订号，并快照实际参数；模型重训后拒绝混用，或要求显式刷新全部关联结果。
“报告是快照”和“报告随模型更新”均可设计，但必须统一。

### 1.5 高优先级：自定义损失下的原生类别权重在直接训练和调参中行为不一致

位置：`hscredit/core/models/boosting/lightgbm_model.py:406`、`hscredit/core/models/losses/base.py:67`、
`hscredit/core/models/tuning/tuning.py:730`。CatBoost 在 `catboost_model.py:387` 直接移除此配置。

实测 `LightGBM(objective=WeightedBCELoss(), scale_pos_weight=1/9)`，两个模型的预测最大差为 **0**。
同样的 `scale_pos_weight=9` 配置传给调参器会被明确拒绝。
这意味着直接训练时参数看似接收了，却没有进入自定义导数的权重计算。

修复方向：将调参器的校验下沉至模型训练入口。不要静默移除有业务含义的参数；
使用 `WeightedBCELoss(pos_weight=...)`、`CostSensitiveLoss(...)` 或显式样本权重表达目的。
CatBoost 本次因环境问题无法动态验证，移除行为是静态确认。

### 1.6 中优先级：重复训练会失去旧轮次曲线

位置：`hscredit/core/models/_lifecycle.py:76`、`:137`。

默认 `history_policy='summary'` 对长度超过 32 的列表只保留数量；`diagnostics` 的阈值是 256。
45 轮训练的 `evals_result_` 保留全部点，但 `training_history_` 只保存“45 项，未保留逐项明细”。
再训练 3 轮后，`evals_result_` 被替换为新曲线，第一次训练的 45 个点无法从历史摘要恢复。

这是有资源预算的保留策略，不是每次训练都会丢曲线；但曲线被当作普通参数列表裁剪，
与“保留训练过程”的直觉不符。建议训练曲线独立保留、压缩或落盘，原始矩阵继续按预算处理。
不建议仅为保存曲线而一律开启 `history_policy='full'`，因为它还会复制训练参数中的数组和对象。

### 1.7 中优先级：金额损失与自动划分不能自然组合

位置：`hscredit/core/models/losses/amount_weighted_loss.py:95`、`hscredit/core/models/tuning/tuning.py:638`。

真实训练集绑定 `AmountWeightedLoss(amounts=776个金额)`，启用自动早停后，训练折只有 620 行，
报错“金额数组长度 (776) 与样本数 (620) 不一致”。模型会切样本权重，但不会切损失对象内部数组。
调参器已经主动拒绝这种数组绑定，避免错位，这是正确防护；使用体验仍缺少统一的逐样本参数路由。

简单金额加权可通过 `fit(sample_weight=金额, evaluation_weight=金额)` 分别指定训练和验证统计权重。
此处并不承诺它与绑定金额损失完全等价：归一化、权重下限和树正则化可能改变效果。
逐行 LGD/EAD/rate/cost 等复杂参数当前需要自行划分，或使用 `trial_objective`。

## 2. 哪些“被换掉了”是当前有意设计

| 行为 | 当前实现与影响 | 当前查看/控制方式 |
| --- | --- | --- |
| 原始分数转概率 | 自定义目标的 margin 经过 sigmoid；是接口适配，不是概率校准 | LightGBM `predict_proba(X, raw_score=True)` 对照默认概率 |
| 概率转信用分 | 默认 PDO=50、基准分600、范围0–1000、取整、截断、高分低风险 | `scorecard_params`、`score_transformer_`、`base_odds_`、`score_prior_source_` |
| WOE 系数正向化 | LR 默认 auto 在 DataFrame 带 WOE 元数据时触发；输入列和系数同时翻向 | `positive_woe_coef=False`；检查 `raw_coef_`、`woe_coef_signs_` |
| XGBoost 自动权重 | 坏率小于5%时使用 `0.05 / 坏率`，否则1；不是通常的好坏样本数比 | `scale_pos_weight=1` 明确关闭；`scale_pos_weight_` 查看实际值 |
| 自动早停划分 | 未给 eval_set 且开启早停时，从输入训练集分层随机切出默认20% | 显式 `eval_set`、`validation_fraction`；完整记录可看划分位置 |
| 调参自动早停 | 模型 early_stopping_rounds=None 可能被调参器默认20补上 | 在 tuner 的 `fixed_params` 中明确固定 None，或设调参器 early_stopping_rounds=None |
| 最优模型重新训练 | `get_best_model()` 默认取各折最佳轮数中位数，关闭内部验证，在完整输入重训 | `refit_params_`、`refit_model_`；`full_data=False` 保留原早停设置 |
| 概率校准时重训 | `ProbabilityCalibrator` 默认 calib_ratio=.2，会克隆并重训基础模型 | 预训练模型配独立校准集时显式 `calib_ratio=None` |
| LightGBM 搜索约束 | 搜索参数会将 num_leaves 限制到 `2**max_depth` | 区分 Optuna 采样值、约束后参数和原生实际参数 |
| 多目标默认最优 | 帕累托集合中按目标列表顺序进行字典序优选 | `get_pareto_front()`；不要默认认为做了加权折中 |

WOE 坐标变换验证使用真实样例派生的一列，并显式设置 WOE 元数据来触发该路径；不是一次正式 WOE 建模。
原系数 -1.998005 被公开为 +1.998005，输入方向为 -1，两种配置的预测完全一致；
直接拿公开系数乘未变换输入则出现最大 **0.131474** 的概率差。应在系数表同时展示原始坐标和模型坐标。

自定义损失实测中，margin 和概率最大差 1.991745，但 `sigmoid(margin)` 与包装器概率最大差为0。
这种适配与 XGBoost 的 [自定义目标及链接函数协议](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html) 一致。
使用类别权重也不等于输出已经校准的违约概率；[LightGBM 文档](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)明确说明权重会影响概率估计。

## 3. 调参中每个 point 到底保存了什么

核心位置：`tuning/tuning.py:1382`、`:1555`、`:1583`、`:1939`、`:2460`。

```text
一个 Trial（参数点）
  └─ 多个外层 CV 折
       ├─ 训练折内划出早停集（若启用）
       ├─ 内部早停集逐轮指标 → 最佳轮数
       └─ 外层验证折预测 → 该折业务指标
  └─ 各折指标平均 → 该 Trial 的搜索目标
选定 Trial
  └─ 汇总最佳轮数 → 完整训练输入重训 → 最终模型
独立测试/OOT
  └─ 最终模型预测 → 最终泛化指标
```

实测一次 AUC 搜索：最佳 Trial 的三折 AUC 为 0.592987、0.572924、0.618306，平均 **0.594739**。
三折最佳轮数 20、10、20，最终重训20轮；重训训练集 AUC **0.722453**，测试集 **0.565757**。
这三个分数来自不同模型/样本口径，不能要求相等。汇总 OOF AUC 也不必等于各折 AUC 平均。

此外，早停后曲线可能包含最佳轮数之后的耐心轮次，预测使用最佳轮数时不能拿曲线最后一点直接对账。

当前真实入口：

```python
tuner = best.tuner
history = tuner.get_optimization_history()       # 各 point 的参数、状态、得分、耗时
detail = tuner.get_trial_result(trial_number)    # 每折预测、指标、曲线及可选模型
fold = detail['各折'][0]
fold['指标']
fold['评估曲线']                                # 内部早停集曲线，不是外层验证折逐轮曲线
fold['最佳迭代']
oof = tuner.get_oof_predictions(trial_number)
tuner.refit_params_                             # 最终重训实际配置
tuner.study_.trials[trial_number].intermediate_values
```

`trial.report(..., step=fold_index)` 当前在每折结束后记录累计平均。因此 Optuna 中间步骤是**折**，不是 boosting 轮次。
[Optuna Trial.report](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) 的 step 由调用者定义。
若想在一折训练中实时剪枝，需要相应框架回调；现有折后汇报不能替代这一能力。

| 保留策略 | 能看到什么 | 关键限制 |
| --- | --- | --- |
| full，默认 | 各折模型、预测、曲线 | 大规模搜索占用内存 |
| predictions | 预测、指标、曲线 | 不保留折模型，训练摘要被压缩 |
| summary | 指标、曲线、最佳轮数及数量 | 无逐行 OOF、无折模型 |
| best | 最佳候选保留完整内容，其余压缩 | 非最佳点不能按完整模型追溯 |
| disk | 完整折制品落盘，按需加载 | 必须一起保留 artifact_dir |

SQLite 仅保存 Study，不替代逐折预测/模型制品。`save_inference` 则有意移除训练历史和 tuner，不能用来续跑研究。

另外 `evaluate_trials()` 是“比较若干配置的 CV 汇总表”，没有像 `fit()` 那样传入 Trial 并保存各折记录。
如果每个 point 都需要复盘，使用 `trial_points`/`enqueue_trials()` 后运行 `fit()`，不要把两个入口视为完全等价。

## 4. 金融损失为何难以使用

至少有四个独立位置，名称又高度相似：

| 目的 | 入口 | 控制什么 |
| --- | --- | --- |
| 怎么学习 | 模型 `objective=loss`；调参便利入口 `loss=loss` | 梯度和二阶导 |
| 每轮看什么 | 模型 `eval_metric=[...]` | 学习曲线 |
| 何时早停 | `early_stopping_metric` 等框架参数 | 一次训练的最佳轮数 |
| 选哪个参数点 | tuner `metric=...` | Trial 的优劣 |

尤其 `ModelTuner(objective=...)` 是历史兼容的**搜索评分目标**，不是模型训练损失。
`loss.metric()` 是代理损失值，`loss.business_metric()` 才尝试映射到真实业务指标；两者最优点和方向可能不同。
后者对 FocalLoss、AmountWeightedLoss 等没有专用业务映射的类型仍返回损失指标。
指标生成时会复制相关配置；修改损失参数后需要重新生成配套指标。

按需求查找的现有入口：

| 需求 | 损失/参数 | 业务验收指标 |
| --- | --- | --- |
| 提高坏样本学习权重 | `WeightedBCELoss(pos_weight, neg_weight)` | `AUCMetric()`、头部捕获、校准质量 |
| 漏判与误杀代价不同 | `CostSensitiveLoss(fn_cost, fp_cost)` | `ClassificationCostMetric(..., threshold)` |
| 聚焦难样本/不平衡 | FocalLoss、BalancedFocalLoss、AsymmetricFocalLoss | 按实际目标另选；不能把 focal loss 当坏账率 |
| 前5%名单抓坏人 | `TopKBadCaptureLoss(top_ratio, miss_penalty, fa_penalty, temperature, bce_weight)` | `TopKCaptureMetric(top_ratio)` |
| 头部提升 | `LiftFocusedLoss(top_ratio, ...)` | `TopKLiftMetric(top_ratio)` |
| 排序能力 | RankingAUCProxyLoss、OrdinalRankLoss | `AUCMetric()` |
| KS区分 | `KSFocusedLoss(ks_weight, bce_weight, bandwidth, ...)` | `KSMetric()` |
| 固定通过率压坏账 | `BadDebtLoss(target_approval_rate, ...)` | `BadDebtMetric(approval_rate)`，最小化 |
| 坏账约束下放量 | `ApprovalRateLoss(target_bad_debt_rate)` | `ApprovalRateMetric(target_bad_debt_rate)`，最大化 |
| 收益与违约成本 | `ExpectedProfitLoss(revenue, default_cost, cutoff, temperature, bce_weight)` | `ProfitMetric(...)`，最大化 |
| 金额/敞口口径 | AmountWeightedLoss、ExpectedValueLoss | 明确人数/金额、归一化与样本参数对齐；不能只看“利润”名字 |

这些是实现映射，不是已证明对实际业务最优的参数推荐。代理目标和真实业务效果应分别验证。
训练适配器还会把概率导数转为 margin 导数，并将树模型曲率下限设为 `1e-6`；
这是优化近似，应记录在训练说明里，不能宣称直接精确优化不可微的真实 TOP/KS 指标。

框架能力也不完全一致：

- XGBoost、LightGBM、CatBoost 支持当前 `BaseLoss` 便利接入；具体评估回调协议不同。
- tuner 的 `loss=` 不直接支持 Pipeline、NGBoost；Pipeline 要先配置末端模型，NGBoost 走专用适配。
- LR、RandomForest 等不能用相同参数随意替换成这些梯度损失。
- 损失对象没有完整 sklearn 参数协议，不能假定 `objective__alpha` 可直接搜索。
  当前可用 callable `search_space(trial)` 为每点构造损失和配套监控指标；不要再用固定 `loss=` 覆盖它。

## 5. 报告“缺一块”的结构性原因

1. **报告未消费训练与调参结果。** `ModelReport.get_metrics()` 核心仍为 KS、AUC、样本数、坏率及跨集 PSI，
   不是 `model.evaluate(metrics=...)` 的结果镜像。没有正式的统一金融指标参数，逐轮曲线和 Trial 明细也未自动纳入。
2. **一个预测数组承担多种语义。** `ReportDataset.__post_init__` 把 `prediction` 同时赋给 `score` 和 `y_proba`。
   默认传概率时，所谓评分字段实际上也是概率；选信用分时，所谓 y_proba 实际又是信用分。
   应保留一次调用的结果，但增加类型/方向元数据，不能仅靠别名解释。
3. **参数页显示构造参数。** 当前读 `get_params()`，没有同时展示 `get_native_params()`、训练覆盖参数、
   自动权重与重训参数。参数看起来不变不代表实际过程不变。
4. **解释表采用不同呈现。** LightGBM 报告优先调用默认 gain 重要性，而原生 `feature_importances_` 可为 split；
   报告还取绝对值并计算占比。这不是必然错误，但应标明方法、单位和归一化规则。
5. **完整性检查已有，但不是默认完整研究报告。** `to_excel(mode='strict'/'best_effort', return_result=True)`
   提供 `ReportResult.status_table()`、失败原因和状态页；本次实际验证可用。
   默认未指定 mode/return_result 时不创建这份结构化结果。一些重要性探测异常仍会静默换后备路径。
   `complete=True` 只代表本次计划章节已完成，不代表金融指标、调参过程、SHAP等所有能力都被包含。

## 6. 当前就能使用的完整示例

脚本：`artifacts/models_audit_20261023/workflow_reference.py`。

它显式区分：成本敏感训练损失、逐轮代理损失监控、真实误判成本选模、独立测试集验收。
使用3个参数点×3折，保留到磁盘；逐轮数据整理成“试验/折/数据集/迭代/指标/数值”的长表。
同时生成当前报告，并将尚未被报告接收的金融指标单独保存。

```powershell
python artifacts/models_audit_20261023/reproduce.py
python artifacts/models_audit_20261023/workflow_reference.py
```

本次实测得到9条折记录、456条逐轮指标、776行OOF、8个工作表。
模型保存恢复后概率和信用分分别逐项一致。Excel 用于表结构/数值核验，没有生成图形，也未做版式视觉验收。
示例的成本数值、随机划分仅用于演示接线；正式实验应明确业务成本，并按时间留独立 OOT。

## 7. 推荐改造顺序及验收标准

### 第一阶段：先保证同一结果能对账

- 统一概率、margin、信用分、类别标签的类型与方向；禁止靠 AUC 结果猜坏样本概率方向。
- TOP/LIFT/捕获率共用名单选择与权重协议；修复报告排序、取整、同分问题。
- 报告绑定训练修订，参数页快照构造/实际/重训配置；拒绝旧预测混入新模型。
- 自定义损失的无效类别权重在所有 fit/tune 入口一致报错。

验收：同一预测、标签、权重、方向、TOP规则，在离线评估、Trial和报告中的指标一致；
概率转分后只允许量化/截断带来的已解释差异；模型重训后报告行为明确。

### 第二阶段：把过程变成可查询的结果

- 在现有 training_history_、trial_results_ 基础上提供公共表接口，避免用户遍历多层字典。
- 曲线独立保留；区分训练修订、Trial、CV折、内部早停集、迭代和最终重训。
- 每个 point 同时展示采样参数、约束后参数、实际原生参数、失败/剪枝原因、均值与离散程度。
- 形成明确的重训记录：来源 Trial、轮数推导、数据范围、模型修订和预测来源。

验收：任意一个已保存 point 都可查配置/状态/各折/曲线；磁盘加载和保存恢复后可对账；
不把“折后剪枝”展示为“逐轮剪枝”，不把“未产生曲线”展示为“曲线丢失”。

### 第三阶段：把金融需求连接到训练和报告

- 增加可查询的损失/指标能力目录，列出用途、参数、方向、支持框架、输入域、样本元数据要求。
- 使用同一指标配置对象连接训练监控、调参评分、离线评估和报告；允许它们显式不同，但展示关系。
- 给金额/LGD/EAD等增加统一切分与验证入口，不再要求用户手动修改损失内部数组。
- 报告包含逐轮曲线、选模依据、金融指标、校准、评分刻度、数据口径与章节状态。

验收：用户只定义一次“前5%捕获率”或“人均误判成本”及业务参数，便可在各阶段复用；
修改业务参数后不会沿用旧指标快照；不支持的模型/加权方式在训练前明确报错。

建议保留原生模型逃生口、sklearn Pipeline兼容与底层专属参数；先补共享结果协议，
再提供场景化短路径。不要重写已通过的全部模型和损失，也不要把原生能力再藏进更大的黑盒。

## 8. 验证范围与材料

- 选取模型契约、训练历史、WOE、调参可用性与保留策略、损失API/集成/业务指标、报告预测方法共345项测试：
  **329通过、16失败**。15项失败涉及 CatBoost 导入的 NumPy ABI 不兼容；1项为旧NGBoost的线性求解维度错误。
- 当前环境：Python3.11.5，NumPy2.4.6，pandas3.0.6，sklearn1.6.1，LightGBM4.1.0，XGBoost2.0.2，
  CatBoost1.2.2，NGBoost0.3.12，Optuna4.9.0。仓库 boost 依赖已要求 ngboost>=0.5.11，本机安装尚未满足。
- 环境失败与已复现的业务逻辑问题分别记录；没有把失败算作通过，没有运行全库测试，也未实测GPU和其他版本组合。
- 静态研究覆盖公共契约/生命周期、经典模型、boosting适配、评分/校准、损失/指标、调参和报告衔接；
  本次没有声称穷尽证明每个损失公式、全部规则分类器与SHAP实现的正确性。

材料目录：`artifacts/models_audit_20261023/`：

- `reproduce.py`：11组行为/缺陷检查；`evidence.json`、`run.log`：数值和异常证据。
- `trial_history.json`：AUC搜索的逐次试验记录。
- `environment_failure.log`：环境导入失败的单项复核日志。
- `workflow_reference.py`、`workflow.log`：完整金融工作流及运行记录。
- `workflow/逐折指标.json`、`workflow/逐轮曲线.json`、`workflow/工作流结果.json`：过程与验收结果。
- `workflow/模型报告核验.xlsx`、`workflow/报告章节状态.json`：当前报告结构与状态。

工作流的模型、SQLite和折制品仅供本地复核；报告未导出订单样例，折制品按当前保留协议包含逐样本数据。
