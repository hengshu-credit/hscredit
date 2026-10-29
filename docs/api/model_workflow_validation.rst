全模型建模链路验收
==================

可复现入口为 :download:`all_model_workflow_validation.ipynb <../../examples/all_model_workflow_validation.ipynb>`。
Notebook 使用真实放款数据，依次执行数据划分、特征筛选、基线对比、超参数搜索、最终模型、完整报告及保存恢复。
测试目标是验证接口、数值和产物的一致性；模型效果另用独立时间外样本及固定训练基准评估。

数据与阶段边界
--------------

使用 ``examples/hscredit_yyp.xlsx`` 的 970 行。输入为衡枢鉴真分老客版、近六个月非银多头机构数、青云24。
分类目标为 FPD，回归目标为放款金额；日期、商品类别、客户编号及金额只按各报告章节的用途传入。
按完整放款日期划分 706 行训练、118 行选参验证、146 行独立测试，同一天不会跨集合。

分类 Pipeline 使用 ScorecardFeatureSelection，回归使用 VarianceSelector；填补和必要的标准化也在 Pipeline 中。
筛选及预处理只拟合训练数据。模型对比、参数搜索只查看训练和选参验证，独立测试不参与选型。
每个案例使用两个不同参数点的网格搜索；最终模型仅重训训练集，不把验证或测试并入。
树模型使用 100 轮量级配置，常规 net 最多 30 轮并按内部验证早停，TabPFN 微调验收使用 1 轮。

覆盖范围
--------

.. list-table::
   :header-rows: 1

   * - 类别
     - 覆盖模型
     - 输出及适用流程
   * - 经典模型
     - LogisticRegression、RandomForest、ExtraTrees、GradientBoosting、SVM、DecisionTreeClassifier
     - 二分类、概率、搜索、完整报告
   * - 提升模型
     - XGBoost、LightGBM、CatBoost、NGBoost
     - 二分类、概率、搜索、完整报告
   * - net
     - ResNet、FTTransformer、TabM、TabPFN
     - 二分类与回归、可用 embedding、搜索、完整报告；另验 TabPFN 两种任务的微调
   * - 评分卡与规则
     - ScoreCard、RoundScoreCard、ProbabilityScoreCard、RulesClassifier
     - 评分/类别和概率语义、真实参数搜索、评分公式与适用分值表
   * - 校准模型
     - ProbabilityCalibrator 的 platt、isotonic、beta、histogram
     - 训练及校准分开，四种方法分别搜索并报告

上述共 28 个模型/任务案例。另有 26 个组件检查：四种独立校准算法、四种评分转换算法及工厂入口的
升序/降序、三种评分漂移校准、RuleSet 的 AND/OR，以及 CalibratedModel。
CalibratedModel 是已拟合对象的推理组合，可置于 Pipeline 中预测；重新训练或调参使用 ProbabilityCalibrator。
抽象基类、损失/指标适配器、结果容器和枚举不当作独立预测模型重复计数。
另对 CatBoost 和四种 net 加入商品类别，完成 5 个类别输入与完整报告检查；这组检查不参与前述选型。
相关性表及图只使用数值字段，类别变量保留众数、频率和分箱统计。

预测与报告核对
--------------

* 概率检查两列、有限值、范围及行和；回归检查连续值及行数。
* 类别输出按模型约定核对。评分卡的 predict 是评分，按 A/B、odds、截断和取整边界独立核算。
  RulesClassifier 的组合决策与其加权命中概率分开，accuracy/precision/recall/f1 使用真实决策类别。
* AUC 使用正负样本逐对比较，KS 使用同分组累计分布，概率均方误差、交叉熵、MSE/RMSE/MAE/R² 使用原始行核算。
  这些结果与模型对比表、报告及实际预测逐项对照。
* 调换特征顺序、添加无关字段或改变输入中的目标列，不改变模型预测。
* 保存并恢复整条 Pipeline，重新比较概率、预测值及评分；不只保存最后一个学习器。
* 每份完整报告使用 strict 模式，开启图表、金额、日期、商品类别及生产样例；保存后重新打开核验章节、图片及错误单元格。
* 参数、筛选、预处理、逐轮记录、参数搜索和最终重训保留在报告中。参数按路径展开，长文本换行，模板列宽调整不扩散到所有空白列。
* 订单定位字段按文本写入生产样例，保留读取到的长整数标识，避免再次经过 Excel 数值精度截断；模型输入数值类型不变。

回归报告约定
------------

ModelReport、auto_model_report 及模型的 model_report 按估计器任务协议选择报告类型。
prediction_method 默认 None：分类使用 predict_proba，回归使用 predict。
回归保留连续目标和小数，支持显式 y、target 列名/一维数据/callable，Series 按索引对齐。

回归报告提供 MSE/RMSE/MAE/R²、目标/预测均值、平均残差、固定训练均值基准、预测分布 PSI、
连续目标分箱、分期/分组误差、变量描述、参数、训练过程和实际值/残差图。
基准取 ``train`` 键的数据集，否则取首个数据集；在其余集合保持该常数，不使用测试集均值拟合基准。
残差定义为预测值减实际值。评估权重只作用于误差指标；分箱表使用明确的件数口径。
等频切点由训练数据确定，连续目标统计使用 compute_bin_stats，不计算二分类坏率或 WOE/IV。

回归分箱当前支持 quantile；不支持逾期二分类标签、剔灰或完整的二分类解释流程。
SHAP 全局重要性使用 explain_config 的 scope='importance'，解释连续预测值。
没有原生逐字段重要性的模型保留变量描述与分箱，重要性不编造为零。
ProbabilityScoreCard 保留概率转换公式和刻度；没有逐变量加分能力时不生成空的分值表。

.. code-block:: python

   model = ResNet(objective="regression", target="放款金额").fit(train)
   report = model.model_report(
       datasets={"训练集": train, "选参验证集": valid, "独立测试集": test},
       metrics=["rmse", "mae", "r2"],
       verbose=False,
   )
   result = report.to_excel("回归报告.xlsx", date="放款时间", group="商品类别",
                            mode="strict", return_result=True)
   assert result.complete

本轮观察（2026-10-29）
----------------------

28 个模型/任务案例、26 个组件检查与 5 个类别输入检查完成，生成 34 份完整模型报告及分类/回归基线对比表。
报告的生成和数值一致性通过，不表示每个模型都有足够的时间外预测能力。

.. list-table:: 数据分布
   :header-rows: 1

   * - 数据集
     - 样本数
     - 坏样本数
     - 坏样本率
     - 放款金额均值
   * - 训练集
     - 706
     - 105
     - 14.8725%
     - 4971.5907
   * - 选参验证集
     - 118
     - 21
     - 17.7966%
     - 2403.8051
   * - 独立测试集
     - 146
     - 10
     - 6.8493%
     - 2011.0342

按验证集选择的分类模型为 FTTransformer：验证 AUC 0.6406，独立测试 AUC 0.4537。
回归同样选中 FTTransformer：验证 RMSE 2940.8471，独立测试 RMSE 3308.4098、R² -6.4599；
固定训练金额均值在独立测试的 RMSE 为 3198.7731。当前配置下预测效果需要继续改进，
没有根据独立测试结果反向选模型或改预测方向以制造更好的指标。

运行产物写入 ``artifacts/all_model_workflow``，包含逐模型工作簿、预测核验 CSV、模型制品、
搜索过程及验收 JSON。真实数据、权重和逐行产物不随源码提交。
34 份报告的 306 个工作表已渲染，1046 个嵌入图片完成可读性/非空检查；
合并单元格完整边界内检查首屏，另复核各模型效果图和参数、部署、训练过程等关键页。
实际解释器为仓库验收 Python 3.11.5；TabPFN 权重验收使用独立目录中的 9.2.0 和官方 v2 分类/回归权重。
本地全库回归为 7112 通过、9 个未配置数据库服务的跳过；后续报告排版和决策指标改动另做专项复验。
远程交付以最终提交 SHA 的全部 CI 工作流和矩阵为准。

SHAP 与训练页复核（2026-10-29）
--------------------------------

在相同日期划分上重新导出 34 份报告，使用选参验证集中的 32 行解释、训练集中的 16 行背景，
随机种子 42，每行预算为 ``2 * 特征数 + 1``。本次原始字段经 Pipeline 或通用模型预测，
采用排列式 SHAP；预算较小，适用于链路验收，不能据此声称重要性排名已稳定。
逐份核对贡献加总、平均绝对贡献和报告表，并保留可用原生重要性。
输出在 ``artifacts/shap_training_report_20261029``，每个模型包括工作簿和重要性 CSV。

TabPFN 类别输入的原混合精度模型未通过严格加总校验（最大残差约 0.0003775）。
对单独副本显式设置 ``inference_precision=torch.float32``，以相同训练行重新 fit 建立上下文后，
残差降为约 0.000000313；相对原模型的最大概率差约 0.01336。原制品保留，
新报告和解释均对应明确标注的全精度副本；其他模型沿用原制品，不放宽加总校验阈值。

训练页按实际章节默认编号 7，标题从 B 列开始，使用 7、7.1、7.2 等层级序号。
标题合并范围在最终列宽确定后计算，覆盖本节表格、合并表头及图片。
原始训练记录与逐轮曲线由 ``get_training_process()`` 保留，Excel 展示摘要、试验对比及曲线。
专项检查覆盖自定义编号、目录链接、真实 SHAP 加性公式、类别类型和图表之后的连续标题序号。

指标定义参考 `scikit-learn 回归指标 <https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics>`_
及 `ROC AUC <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`_。
