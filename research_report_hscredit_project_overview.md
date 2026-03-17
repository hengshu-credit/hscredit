# hscredit项目全景研究报告

## 执行摘要

hscredit是一个功能完备的金融信贷风险建模工具包,已成功实现对第三方风控库(toad、optbinning、scorecardpy)的完全解耦。项目包含16种分箱算法、8种特征编码器、20+种特征筛选方法,以及完整的评分卡建模、策略分析和报告生成功能。通过对第三方库和生产代码的分析,hscredit在保持API简洁性的同时,自主实现了核心算法,具备成为公司级开源项目的潜力。主要改进方向包括增强可视化效果、丰富报告类型、优化建模流程集成度。

## 项目背景

hscredit是从scorecardpipeline迁移而来的金融信贷风险建模工具包,旨在成为公司级开源项目。项目核心目标包括:去除对第三方风控库的依赖并自主实现核心功能、继承scorecardpipeline的API简洁性和优秀可视化风格、提供快捷的Excel写入器和精美报告生成能力。针对建模人员,hscredit提供集成的预处理、特征工程和模型调参功能;针对策略人员,hscredit支持特征有效性分析、规则挖掘和报告产出。项目在金融信贷场景中遵循"分数越大风险越低"的原则(值域非[0,1])或"分数越小风险越低"的原则(值域在[0,1])。

## hscredit项目架构分析

### 目录结构

hscredit采用清晰的分层架构,分为核心算法模块、报告生成模块和工具模块三层。

核心算法模块位于core目录,包含binning(分箱算法)、encoders(特征编码器)、selectors(特征筛选器)、models(模型)、metrics(指标计算)、viz(可视化)、rules(规则引擎)、financial(金融计算)、feature_engineering(特征工程)九个子模块。报告生成模块位于report目录,包含excel(Excel写入器)、feature_analyzer(特征分箱统计)、feature_report(特征分析报告)、ruleset_report(规则集评估)、swap_analysis_report(Swap置换分析)五个子模块。工具模块位于utils目录,提供随机种子、数据IO、特征描述、数据集加载、日志管理、杂项工具等功能支持。

### 核心功能模块

#### 分箱算法模块

hscredit实现了16种分箱算法,全面覆盖金融风控场景需求。等宽分箱(UniformBinning)和等频分箱(QuantileBinning)提供基础分箱能力。决策树分箱(TreeBinning)和卡方合并分箱(ChiMergeBinning)基于统计方法优化分箱边界。最优KS分箱(OptimalKSBinning)和最优IV分箱(OptimalIVBinning)针对风控场景最大化区分度。MDLP分箱(MDLPBinning)采用信息论方法,CART分箱(CartBinning)基于分类回归树。K-Means聚类分箱(KMeansBinning)适用于无明显单调性特征。单调性约束分箱(MonotonicBinning)支持递增、递减、U型、倒U型四种单调性模式。遗传算法分箱(GeneticBinning)和平滑分箱(SmoothBinning)提供高级优化能力。核密度分箱(KernelDensityBinning)和BestLift分箱(BestLiftBinning)适用于特定场景。目标坏样本率分箱(TargetBadRateBinning)直接优化业务目标。OptimalBinning作为统一接口整合所有方法。

所有分箱算法遵循sklearn Transformer接口,支持单调性约束、预分箱、自定义切分点,自动识别数值型和类别型特征,并生成完整的分箱统计表(WOE、IV、KS、LIFT等指标)。

#### 特征编码器模块

hscredit提供8种特征编码器,满足不同建模场景需求。WOEEncoder实现证据权重编码,是评分卡建模的标准方法。TargetEncoder提供目标编码,适用于高基数类别特征。CountEncoder和OneHotEncoder分别实现计数编码和独热编码。OrdinalEncoder实现序数编码,QuantileEncoder实现分位数编码。CatBoostEncoder提供CatBoost算法的编码方法。GBMEncoder实现梯度提升树编码器,支持XGBoost、LightGBM、CatBoost等框架与逻辑回归的结合。

所有编码器遵循sklearn Transformer接口,自动处理未知类别和缺失值,支持正则化防止过拟合。

#### 特征筛选器模块

hscredit实现了20+种特征筛选方法,分为过滤法、包装法和嵌入法三大类。

过滤法包括方差筛选(VarianceSelector)、缺失率筛选(NullSelector)、众数占比筛选(ModeSelector)、基数筛选(CardinalitySelector)、相关性筛选(CorrSelector)、VIF筛选(VIFSelector)、IV值筛选(IVSelector)、LIFT值筛选(LiftSelector)、PSI稳定性筛选(PSISelector)、类型筛选(TypeSelector)、正则筛选(RegexSelector)。包装法包括递归特征消除(RFESelector)、序贯特征选择(SequentialFeatureSelector)、逐步回归(StepwiseSelector、StepwiseFeatureSelector)、Boruta算法(BorutaSelector)。嵌入法包括特征重要性筛选(FeatureImportanceSelector)、零重要性筛选(NullImportanceSelector)、互信息筛选(MutualInfoSelector)、卡方检验筛选(Chi2Selector)、F检验筛选(FTestSelector)。组合器(CompositeFeatureSelector)支持多种筛选方法的灵活组合,筛选报告收集器(SelectionReportCollector)汇总筛选过程和结果。

#### 模型模块

hscredit的模型模块包含逻辑回归、评分卡和自定义损失函数三大核心组件。

LogisticRegression扩展了sklearn的逻辑回归,自动计算统计信息(标准误差、z值、p值)、VIF计算,提供summary方法输出回归结果表。ScoreCard实现评分卡模型,支持PDO和基础分数配置、评分卡输出和导出、PMML导出支持。

自定义损失函数覆盖多种风控场景需求。FocalLoss处理不平衡数据,WeightedBCELoss实现加权二元交叉熵,CostSensitiveLoss实现成本敏感损失,BadDebtLoss优化坏账率,ApprovalRateLoss优化通过率,ProfitMaxLoss实现利润最大化。自定义评估指标包括KS统计量(KSMetric)、Gini系数(GiniMetric)、PSI稳定性指标(PSIMetric)。框架适配器支持XGBoost、LightGBM、CatBoost、TabNet等主流框架。

#### 指标计算模块

hscredit提供丰富的指标计算功能,覆盖分类指标、稳定性指标、特征重要性指标、回归指标和分箱指标。

分类指标包括KS、AUC、Gini(模型区分度)、KS_bucket(KS分桶统计)、ROC_curve(ROC曲线数据)、confusion_matrix(混淆矩阵)、classification_report(分类报告)。稳定性指标包括PSI、CSI(特征和模型稳定性)、PSI_table、CSI_table(稳定性统计表)。特征重要性包括IV、IV_table(信息值计算)、gini_importance、entropy_importance(基于不纯度的重要性)。回归指标包括MSE、MAE、RMSE、R2。分箱指标包括woe_iv_vectorized(向量化WOE/IV计算)、compute_bin_stats(分箱统计计算)、ks_by_bin、chi2_by_bin(分箱级别指标)、batch_iv、compare_splits_iv(批量IV计算和比较)。

#### 可视化模块

hscredit提供专业风控场景的可视化功能。bin_plot绘制特征分箱图,corr_plot绘制特征相关性热力图,ks_plot绘制KS/ROC曲线图,hist_plot绘制特征分布直方图,psi_plot绘制PSI稳定性分析图,dataframe_plot实现DataFrame表格可视化,distribution_plot绘制时间分布图,plot_weights绘制逻辑回归系数误差图。

#### 规则引擎模块

hscredit的规则引擎提供规则定义、评估和优化功能。Rule类支持pandas eval语法的规则定义,支持规则组合(&、|、~、^)和自动优化规则表达式。get_columns_from_query函数获取query语句使用的列,optimize_expr和beautify_expr函数实现规则表达式优化。

#### 金融计算模块

hscredit提供常用金融计算函数,参考numpy_financial实现。基础计算包括fv(未来值)、pv(现值)、pmt(每期付款)、nper(期数)、ipmt/ppmt(利息/本金部分)、rate(利率)。高级计算包括npv(净现值)、irr(内部收益率)、mirr(修正内部收益率)。

### 报告生成模块

#### Excel写入器

ExcelWriter类提供专业的Excel报告生成功能。支持DataFrame写入(多层索引/多层列名)、图片插入、超链接插入、条件格式设置(数据条、颜色刻度)、自定义样式和主题色、中文字体和列宽自适应。

#### 特征分析器

feature_bin_stats函数支持单特征/多特征分析、多逾期标签+逾期天数组合分析、自定义分箱规则、金额口径分析、灰样本剔除,生成多级表头报告。FeatureAnalyzer类提供批量特征分析、多维度对比分析、IV值汇总功能。

#### Swap置换分析报告

Swap分析模块基于参考数据集计算评分区间逾期率,对无标签swap数据进行风险预估,out-in样本支持风险上浮因子,提供完整的swap四象限风险分析报告、通过率变化分析、风险拒绝率分析。核心类包括SwapType(四象限类型枚举)、SwapRiskConfig(分析配置)、ReferenceDataProvider(参考数据提供者)、SwapAnalyzer(Swap分析器)、SwapAnalysisResult(分析结果)。

### 与第三方库的解耦情况

hscredit已实现对toad、optbinning、scorecardpy的完全解耦,对scorecardpipeline采用参考实现策略。所有分箱算法独立实现不依赖optbinning,WOE/IV计算使用metrics模块向量化计算,逻辑回归扩展sklearn并增加统计功能,Excel写入器迁移自scorecardpipeline并优化。

## 第三方风控库功能分析

### toad库

toad是一个完整的Python风控工具库,提供数据预处理、特征工程、模型训练和评估的完整功能链。核心模块包括merge(分箱合并)、transform(WOE转换)、selection(特征筛选)、scorecard(评分卡)、plot(可视化)、metrics(指标计算)、stats(统计信息)。

merge模块提供ChiMerge、DTMerge、KMeansMerge等多种分箱方法,支持单调性约束和自定义切分点。transform模块实现WOE转换和分箱转换。selection模块提供IV、PSI、相关性、VIF等多种特征筛选方法。scorecard模块实现完整的评分卡建模流程,包括分箱、WOE转换、逻辑回归、评分卡生成。plot模块提供KS曲线、ROC曲线、分箱图、PSI图等可视化功能。metrics模块计算KS、AUC、PSI等指标。stats模块提供数据质量统计、特征分布统计等功能。

### optbinning库

optbinning是最优分箱算法的专业实现,提供数学优化方法求解最优分箱问题。核心模块包括binning(最优分箱)、scorecard(评分卡)、plots(可视化)。

binning模块实现OptimalBinning和OptimalBinningProcess类,支持单调性约束、预分箱、缺失值处理、大样本优化。OptimalBinning针对单变量优化分箱方案,OptimalBinningProcess处理多变量批量分箱。scorecard模块提供Scorecard类,整合分箱、WOE编码、逻辑回归、评分转换。plots模块提供分箱边界可视化、事件率曲线、WOE曲线等功能。

### scorecardpy库

scorecardpy是一个轻量级评分卡开发工具,提供传统评分卡建模的核心功能。核心模块包括woebin(分箱)、scorecard(评分卡)、perf(性能评估)。

woebin模块实现自动分箱和手动分箱,支持决策树、卡方、等频等分箱方法,生成WOE表和分箱统计表。scorecard模块实现评分卡生成和评分计算,支持PDO和基础分数配置。perf模块提供模型性能评估,包括KS曲线、ROC曲线、提升图、混淆矩阵等。

### skorecard库

skorecard是基于sklearn的评分卡库,强调Pipeline集成和模块化设计。核心模块包括bucketers(分箱器)、skorecard(评分卡)、reporting(报告和可视化)。

bucketers模块提供多种分箱器,包括DecisionTreeBucketer、OptimalBucketer、OrdinalCategoricalBucketer等,遵循sklearn Transformer接口。skorecard模块实现Skorecard类,整合分箱、逻辑回归、评分转换,支持Pipeline集成。reporting模块提供分箱报告、PSI报告、特征重要性报告等可视化功能。

### 功能对比总结

| 功能领域 | toad | optbinning | scorecardpy | skorecard |
|---------|------|------------|-------------|-----------|
| 分箱算法 | ChiMerge, DTMerge, KMeans | 最优分箱(数学优化) | 决策树, 卡方, 等频 | 决策树, 最优, 类别 |
| 特征筛选 | IV, PSI, 相关性, VIF | 集成在BinningProcess | 基础筛选 | 集成在Pipeline |
| 评分卡 | 完整流程 | 完整流程 | 完整流程 | 完整流程 |
| 可视化 | KS, ROC, 分箱图, PSI | 分箱边界, 事件率 | KS, ROC, 提升图 | 分箱报告, PSI报告 |
| Excel报告 | 无 | 无 | 无 | 无 |
| Pipeline | 无 | BinningProcess | 无 | sklearn Pipeline |
| API设计 | 简洁 | 专业 | 简洁 | sklearn风格 |

hscredit在分箱算法多样性、特征筛选方法丰富度、Excel报告生成能力方面具有优势。optbinning在数学优化方法上有优势,skorecard在Pipeline集成方面有优势。

## 生产建模代码分析

### 联合建模代码

联合建模目录包含scorecard_functions_py3_V2.py、Model_Docu.py、IV_Calculation.py、PSI_Calculation.py等核心文件。

scorecard_functions_py3_V2.py提供完整的评分卡建模函数,包括数据预处理、特征分箱、WOE转换、特征筛选、模型训练、评分卡生成、模型评估等功能。函数设计注重实用性和可配置性,支持自定义分箱规则、筛选阈值、模型参数等。

Model_Docu.py提供建模文档生成功能,自动生成分箱报告、模型报告、特征报告等,支持Word格式输出。报告包含详细的统计信息、可视化图表、模型说明等内容。

IV_Calculation.py和PSI_Calculation.py分别实现IV值和PSI值的批量计算,支持特征级和样本级分析,输出详细的分析报告和可视化图表。

### 同盾联合建模代码

同盾联合建模目录包含funcs.py等核心文件,实现了针对同盾数据的特定建模功能。funcs.py包含数据预处理、特征工程、模型训练、模型评估等函数,针对同盾数据特点进行了优化。特别关注时间序列特征处理、缺失值处理、异常值处理等实际建模中的常见问题。

### 驻场代码工具包

驻场代码工具包是一个综合性的建模工具集,包含model_main.py、model_tool.py等核心文件。

model_main.py提供端到端的建模流程,从数据加载到模型部署的完整pipeline。流程包括数据质量检查、特征工程、特征筛选、分箱、WOE转换、模型训练、模型评估、评分卡生成、报告输出等环节。代码结构清晰,支持配置化运行,适合快速产出建模结果。

model_tool.py提供大量实用工具函数,包括数据质量检查函数、特征工程函数、可视化函数、报告生成函数、Excel写入函数等。这些函数在实际建模中反复使用,经过充分测试和优化,具有良好的稳定性和可扩展性。

### scorecardpipeline原项目

scorecardpipeline是hscredit的前身项目,提供processing.py、feature_selection.py、model.py、auto_report.py等核心模块。

processing.py实现数据预处理和特征工程功能,包括缺失值处理、异常值检测、特征编码、特征变换等。API设计简洁,支持链式调用和Pipeline集成。

feature_selection.py实现多种特征筛选方法,包括过滤法、包装法、嵌入法。提供统一的接口,支持灵活组合和定制。

model.py实现模型训练和评估功能,支持逻辑回归、决策树、随机森林、XGBoost、LightGBM等多种模型。提供模型比较、参数调优、交叉验证等功能。

auto_report.py实现自动报告生成功能,支持一键生成完整的建模报告,包括数据质量报告、特征分析报告、模型评估报告、评分卡报告等。报告格式精美,包含丰富的可视化图表和统计表格。

## 最佳实践总结

### 建模流程

生产代码中体现的标准建模流程包括以下步骤。数据质量检查环节检查缺失率、异常值、数据类型、样本分布等,生成数据质量报告。特征工程环节创建衍生特征、时间特征、交叉特征、统计特征等,进行特征编码和变换。特征筛选环节使用IV、PSI、相关性、VIF等方法筛选特征,逐步回归优化特征组合。分箱处理环节选择合适的分箱方法(最优IV分箱、卡方分箱等),确保单调性和业务合理性。WOE转换环节对训练集和测试集进行WOE转换,处理缺失值和异常值。模型训练环节训练逻辑回归模型,计算统计信息(标准误差、z值、p值、VIF),调整模型参数。模型评估环节计算KS、AUC、Gini等指标,绘制KS曲线、ROC曲线,生成混淆矩阵和分类报告。评分卡生成环节配置PDO和基础分数,生成评分卡表,导出PMML格式。报告输出环节生成完整的建模报告,包括数据质量报告、特征分析报告、模型评估报告、评分卡报告,输出Excel格式便于业务使用。

### 工具函数设计

生产代码中的工具函数设计遵循以下原则。功能单一性原则要求每个函数只做一件事,降低复杂度提高可维护性。参数灵活性原则支持默认参数和自定义参数,满足不同场景需求。错误处理原则检查输入数据有效性,提供清晰的错误提示。日志记录原则记录关键步骤和计算结果,便于问题排查。可视化集成原则函数输出包含可视化选项,支持快速查看结果。报告生成原则支持生成格式化报告,便于业务汇报和存档。

### Excel报告最佳实践

Excel报告生成需要关注以下要点。样式统一性使用统一的主题色和字体,设置合适的列宽和行高,应用条件格式突出关键信息。内容层次性采用多级表头组织数据,使用超链接跳转相关工作表,添加注释说明业务含义。可视化集成将图表插入Excel工作表,调整图表大小和位置,保持图表风格一致。性能优化避免重复写入相同数据,批量操作减少IO次数,合理使用公式减少计算量。

## 功能优化建议

### 核心算法增强

基于第三方库和生产代码的分析,建议hscredit在核心算法方面进行以下增强。分箱算法优化参考optbinning的数学优化方法,实现更精确的最优分箱求解,支持大规模数据的高效处理。特征筛选增强增加更多嵌入式筛选方法,支持基于模型的特征重要性分析,提供特征筛选的可视化报告。模型扩展支持更多模型类型(决策树、随机森林、XGBoost等),提供模型比较和选择功能,集成自动调参框架。时间序列建模支持时间序列特征工程,实现时间序列模型,提供时间维度的模型评估。

### 可视化增强

建议在可视化方面进行以下增强。交互式可视化引入plotly等交互式可视化库,支持图表缩放、悬停提示、数据筛选,提供更友好的用户交互体验。报告模板化设计多套可视化主题,支持自定义主题配置,提供多种报告模板选择。图表丰富性增加更多图表类型(箱线图、小提琴图、桑基图等),支持图表组合和布局定制,提供图表导出功能。可视化性能优化支持大规模数据的可视化,实现数据采样和聚合,提高渲染速度。

### 报告生成增强

建议在报告生成方面进行以下增强。Word报告支持生成Word格式报告,支持报告模板定制,提供报告合并和导出功能。PPT报告支持生成PPT格式报告,支持幻灯片布局设计,提供演示模式。报告自动化实现一键生成完整报告,支持报告版本管理,提供报告更新和对比功能。多维度报告支持分样本、分时间、分产品等多维度分析,提供交叉分析和对比分析,生成定制化业务报告。

### API设计优化

建议在API设计方面进行以下优化。Pipeline集成参考skorecard的设计,提供sklearn Pipeline集成,支持管道式建模流程,提高代码复用性。配置化管理支持YAML或JSON配置文件,实现建模流程的配置化运行,降低使用门槛。易用性提升提供更简洁的API接口,支持快速模式和专家模式,提供丰富的使用示例和文档。扩展性设计支持自定义算法和指标,提供插件机制,便于功能扩展和定制。

### 性能优化

建议在性能方面进行以下优化。大数据支持优化算法以支持大规模数据,实现数据分块处理,提供内存优化选项。并行计算支持多进程和多线程计算,实现任务并行和流水线并行,提供分布式计算支持。缓存机制缓存中间计算结果,避免重复计算,提供缓存管理和清理功能。性能监控记录各步骤的计算时间,识别性能瓶颈,提供性能优化建议。

## 结论

hscredit是一个功能完备、架构清晰的金融信贷风险建模工具包,已成功实现对第三方风控库的完全解耦。项目在分箱算法多样性、特征筛选方法丰富度、报告生成能力方面具有明显优势,具备成为公司级开源项目的潜力。通过对第三方库和生产代码的深入分析,识别出在可视化增强、报告类型扩展、Pipeline集成、性能优化等方面的改进方向。建议优先实现交互式可视化、Word/PPT报告支持、sklearn Pipeline集成等核心功能,进一步提升hscredit的实用性和竞争力。

## 局限性

本研究基于代码静态分析,未进行实际运行测试。部分功能细节和性能表现需要进一步验证。生产代码的具体使用场景和效果需要实际建模案例验证。第三方库的高级功能和边缘情况可能未被充分覆盖。建议在后续工作中进行实际建模测试和用户调研,进一步完善功能优化建议。

## 参考文献

1. [hscredit项目README](/Users/xiaoxi/CodeBuddy/hscredit/hscredit/README.md)
2. [toad库GitHub仓库](https://github.com/amphibian-dev/toad)
3. [optbinning库GitHub仓库](https://github.com/guillermo-navas-palencia/optbinning)
4. [scorecardpy库GitHub仓库](https://github.com/ShichenXie/scorecardpy)
5. [skorecard库GitHub仓库](https://github.com/ing-bank/skorecard)
