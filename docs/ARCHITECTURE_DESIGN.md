# hscredit 整体代码架构设计方案

## 一、项目概述

### 1.1 项目定位
hscredit 是一个完整的金融信贷风险建模工具包，旨在为公司策略分析人员和模型开发人员提供一站式解决方案。

### 1.2 目标用户
- **建模人员**：需要预处理、特征工程、模型训练、调参、自定义loss等全链路功能
- **策略人员**：需要快速特征有效性分析、规则提取、精美报告生成

### 1.3 核心设计理念
1. **API简单易用**：继承scorecardpipeline的简洁API设计
2. **优秀的可视化**：所有分析都有精美的可视化输出
3. **快捷报告生成**：Excel报告快速产出
4. **自主可控**：去除对toad/optbinning/scorecardpy的第三方依赖
5. **业务导向**：深度适配金融信贷场景

---

## 二、整体架构设计

### 2.1 架构分层图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户接口层                                 │
├─────────────────────────────────────────────────────────────────────┤
│  hscredit/                                                         │
│    ├── info()          # 包信息打印                                 │
│    ├── get_version()   # 版本获取                                   │
│    └── __all__         # 统一API导出                                │
├─────────────────────────────────────────────────────────────────────┤
│                           报告层 (Report)                            │
├─────────────────────────────────────────────────────────────────────┤
│  report/                                                           │
│    ├── excel/          # Excel报告生成                             │
│    │   └── writer.py   # 样式化Excel写入器                          │
│    ├── feature_analyzer.py     # 特征分箱统计分析                    │
│    ├── feature_report.py       # 特征分析报告生成                    │
│    ├── ruleset_report.py       # 规则集综合评估报告                  │
│    └── swap_analysis_report.py # 规则置换风险分析                    │
├─────────────────────────────────────────────────────────────────────┤
│                           核心层 (Core)                              │
├─────────────────────────────────────────────────────────────────────┤
│  core/                                                             │
│    ├── binning/          # 分箱算法 (15+种)                          │
│    ├── encoders/         # 特征编码 (WOE/Target/Count等)             │
│    ├── selectors/        # 特征筛选 (20+种方法)                      │
│    ├── models/           # 模型/损失函数                            │
│    │   ├── losses/       # 自定义损失函数                            │
│    │   ├── logistic_regression.py  # 逻辑回归                       │
│    │   └── scorecard.py  # 评分卡模型                               │
│    ├── metrics/          # 评估指标 (KS/AUC/PSI/IV/Gini)            │
│    ├── viz/              # 可视化 (分箱图/KS图/相关性图等)           │
│    ├── feature_engineering/  # 特征工程                             │
│    ├── rules/            # 规则引擎                                 │
│    └── financial/        # 金融计算 (FV/PV/PMT等)                   │
├─────────────────────────────────────────────────────────────────────┤
│                           工具层 (Utils)                             │
├─────────────────────────────────────────────────────────────────────┤
│  utils/                                                            │
│    ├── datasets.py       # 数据集工具                               │
│    ├── io.py            # IO工具 (pickle等)                         │
│    ├── logger.py        # 日志工具                                  │
│    ├── describe.py      # 数据描述                                  │
│    └── misc.py          # 杂项工具                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块详细设计

#### 2.2.1 分箱模块 (core/binning)

**设计原则：**
- 所有分箱器继承 `BaseBinning` 基类
- 统一的 `fit/transform/fit_transform` 接口
- 支持 `user_splits` 用户指定切分点
- 支持 `get_bin_table` 获取分箱统计表

**分箱方法矩阵：**

| 类别 | 方法 | 类名 | 特点 | 适用场景 |
|------|------|------|------|----------|
| 基础 | 等宽分箱 | UniformBinning | 等宽区间 | 均匀分布数据 |
| 基础 | 等频分箱 | QuantileBinning | 等数量样本 | 偏态分布数据 |
| 基础 | 决策树分箱 | TreeBinning | 基于树结构 | 快速分箱 |
| 基础 | 卡方分箱 | ChiMergeBinning | 基于卡方检验 | 类别合并 |
| 优化 | 最优KS分箱 | BestKSBinning | 最大化KS | 评分卡建模 |
| 优化 | 最优IV分箱 | BestIVBinning | 最大化IV | 评分卡建模 |
| 优化 | MDLP分箱 | MDLPBinning | 基于信息论 | 有监督分箱 |
| 高级 | CART分箱 | CartBinning | 参考optbinning | 精确控制 |
| 高级 | 单调性分箱 | MonotonicBinning | 支持U型/倒U型 | 业务单调性要求 |
| 高级 | 遗传算法分箱 | GeneticBinning | 全局最优 | 复杂场景 |
| 高级 | 平滑分箱 | SmoothBinning | 正则化 | 小样本 |
| 高级 | 核密度分箱 | KernelDensityBinning | 基于密度 | 连续变量 |
| 高级 | Best Lift分箱 | BestLiftBinning | 最大化Lift | 营销场景 |
| 高级 | 目标坏样本率分箱 | TargetBadRateBinning | 指定坏样本率 | 策略场景 |

**统一接口类：**
```python
class OptimalBinning:
    """统一分箱接口，自动选择最优方法或指定方法"""
    def __init__(self, method='optimal_iv', max_n_bins=5, ...)
    def fit(X, y)
    def transform(X)
    def fit_transform(X, y)
    def get_bin_table(feature_name)
    @staticmethod
    def auto_select_method(X, y, feature_name)
```

#### 2.2.2 特征编码模块 (core/encoders)

**设计原则：**
- 继承sklearn的BaseEstimator和TransformerMixin
- 统一的 `fit/transform` 接口
- 支持分类变量的多种编码方式

**编码器矩阵：**

| 编码器 | 类名 | 特点 | 适用场景 |
|--------|------|------|----------|
| WOE编码 | WOEEncoder | 证据权重编码 | 评分卡建模 |
| 目标编码 | TargetEncoder | 基于目标变量均值 | 高基数类别 |
| 计数编码 | CountEncoder | 基于频次 | 快速编码 |
| One-Hot | OneHotEncoder | 独热编码 | 低基数类别 |
| 有序编码 | OrdinalEncoder | 整数编码 | 树模型 |
| 分位数编码 | QuantileEncoder | 基于分位数 | 非线性关系 |
| CatBoost编码 | CatBoostEncoder | 有序目标编码 | 防止过拟合 |
| GBM编码 | GBMEncoder | 基于梯度提升 | 复杂关系 |

#### 2.2.3 特征筛选模块 (core/selectors)

**设计原则：**
- 所有筛选器继承 `BaseFeatureSelector` 基类
- 支持fit/transform接口
- 生成中文筛选报告
- 支持Pipeline集成

**筛选方法分类：**

| 类别 | 筛选器 | 类名 | 原理 |
|------|--------|------|------|
| 过滤法-基础 | 方差筛选 | VarianceSelector | 基于方差阈值 |
| 过滤法-基础 | 缺失率筛选 | NullSelector | 基于缺失率 |
| 过滤法-基础 | 众数率筛选 | ModeSelector | 基于单一值比例 |
| 过滤法-基础 | 基数筛选 | CardinalitySelector | 基于唯一值数量 |
| 过滤法-基础 | 类型筛选 | TypeSelector | 基于数据类型 |
| 过滤法-基础 | 正则筛选 | RegexSelector | 基于列名匹配 |
| 过滤法-相关 | 相关性筛选 | CorrSelector | 基于Pearson/Spearman |
| 过滤法-相关 | VIF筛选 | VIFSelector | 基于方差膨胀因子 |
| 过滤法-目标 | IV筛选 | IVSelector | 基于信息价值 |
| 过滤法-目标 | Lift筛选 | LiftSelector | 基于提升度 |
| 过滤法-目标 | PSI筛选 | PSISelector | 基于群体稳定性 |
| 嵌入法 | 特征重要性 | FeatureImportanceSelector | 基于模型重要性 |
| 嵌入法 | Null重要性 | NullImportanceSelector | 基于Null Importance |
| 嵌入法 | RFE | RFESelector | 递归特征消除 |
| 嵌入法 | 逐步选择 | SequentialFeatureSelector | 前向/后向选择 |
| 嵌入法 | 逐步回归 | StepwiseSelector | 统计显著性检验 |
| 高级 | Boruta | BorutaSelector | 影子特征比较 |
| 高级 | 互信息 | MutualInfoSelector | 基于互信息 |
| 高级 | 卡方检验 | Chi2Selector | 基于卡方检验 |
| 高级 | F检验 | FTestSelector | 基于ANOVA F值 |

#### 2.2.4 模型与损失函数模块 (core/models)

**损失函数设计：**

| 损失函数 | 类名 | 用途 |
|----------|------|------|
| Focal Loss | FocalLoss | 处理类别不平衡 |
| 加权BCE | WeightedBCELoss | 样本权重调整 |
| 成本敏感 | CostSensitiveLoss | 错误分类成本不同 |
| 坏账损失 | BadDebtLoss | 风控业务导向 |
| 通过率损失 | ApprovalRateLoss | 通过率约束优化 |
| 利润最大化 | ProfitMaxLoss | 利润导向优化 |

**评估指标：**

| 指标 | 类名 | 说明 |
|------|------|------|
| KS | KSMetric | Kolmogorov-Smirnov |
| Gini | GiniMetric | Gini系数 |
| PSI | PSIMetric | Population Stability Index |

**框架适配器：**

| 适配器 | 类名 | 说明 |
|--------|------|------|
| XGBoost适配 | XGBoostLossAdapter | 自定义loss用于XGBoost |
| LightGBM适配 | LightGBMLossAdapter | 自定义loss用于LightGBM |
| CatBoost适配 | CatBoostLossAdapter | 自定义loss用于CatBoost |
| TabNet适配 | TabNetLossAdapter | 自定义loss用于TabNet |

#### 2.2.5 可视化模块 (core/viz)

| 函数 | 说明 | 适用场景 |
|------|------|----------|
| bin_plot | 分箱图 | 展示分箱效果 |
| corr_plot | 相关性图 | 特征相关性分析 |
| ks_plot | KS图 | 模型区分能力 |
| hist_plot | 分布图 | 数据分布查看 |
| psi_plot | PSI图 | 稳定性分析 |
| dataframe_plot | DataFrame可视化 | 数据预览 |
| distribution_plot | 分布对比图 | 多数据集对比 |
| plot_weights | 权重图 | 模型权重展示 |

#### 2.2.6 报告模块 (report)

| 功能 | 类/函数 | 说明 |
|------|---------|------|
| Excel写入 | ExcelWriter | 样式化Excel写入器 |
| 特征分析 | FeatureAnalyzer | 特征分箱统计分析 |
| 特征报告 | auto_feature_analysis_report | 自动生成特征分析报告 |
| 规则报告 | ruleset_report | 规则集综合评估 |
| 置换分析 | SwapAnalyzer | 规则置换风险分析 |

---

## 三、待完善功能清单

### 3.1 高优先级 (核心功能缺失)

#### 3.1.1 模型调参模块
```
core/tuning/
    ├── __init__.py
    ├── optuna_tuner.py      # Optuna超参数搜索
    ├── bayesian_tuner.py    # 贝叶斯优化
    ├── grid_search.py       # 网格搜索
    ├── param_space.py       # 参数空间定义
    └── visualization.py     # 调参过程可视化
```

**功能点：**
- Optuna集成支持LGBM/XGB/CatBoost
- 自定义搜索空间
- 早停机制
- 并行搜索
- 调参过程可视化 (重要性图/等高线图/历史图)
- 最佳参数自动保存

#### 3.1.2 SHAP模型可解释性模块
```
core/explainability/
    ├── __init__.py
    ├── shap_explainer.py    # SHAP解释器封装
    ├── feature_importance.py # 特征重要性汇总
    └── visualizer.py        # SHAP可视化
```

**功能点：**
- TreeSHAP (LGBM/XGB/CatBoost)
- KernelSHAP (任意模型)
- 全局特征重要性
- 局部样本解释
- SHAP值交互分析
- 力图/瀑布图/散点图/摘要图

#### 3.1.3 完整的评分卡转换模块
```
core/models/scorecard.py 需要完善:
    - 概率到分数的线性转换
    - PDO/基准分配置
    - 分数校准
    - 评分卡表生成
    - 评分卡部署代码生成 (Python/Java/SQL)
```

### 3.2 中优先级 (功能增强)

#### 3.2.1 特征工程扩展
```
core/feature_engineering/
    ├── __init__.py
    ├── expression.py        # 数学表达式衍生 (已有)
    ├── auto_features.py     # 自动特征工程
    ├── time_features.py     # 时序特征
    ├── cross_features.py    # 交叉特征
    └── polynomial.py        # 多项式特征
```

#### 3.2.2 PPT报告生成
```
report/
    ├── ppt/
    │   ├── __init__.py
    │   ├── generator.py     # PPT生成器
    │   └── templates/       # 报告模板
```

#### 3.2.3 模型持久化
```
core/persistence/
    ├── __init__.py
    ├── serializer.py        # 模型序列化
    ├── pmml_export.py       # PMML导出
    └── onnx_export.py       # ONNX导出
```

### 3.3 低优先级 (优化提升)

#### 3.3.1 更多机器学习模型
```
core/models/
    ├── ensemble/            # 集成学习
    ├── neural_networks/     # 神经网络
    └── svm/                 # 支持向量机
```

#### 3.3.2 AutoML功能
```
core/automl/
    ├── __init__.py
    ├── auto_binning.py      # 自动分箱选择
    ├── auto_encoding.py     # 自动编码选择
    ├── auto_selection.py    # 自动特征筛选
    └── auto_modeling.py     # 自动建模流程
```

---

## 四、参考库借鉴点

### 4.1 toad 借鉴

**优秀设计：**
- `Combiner` 统一分箱接口
- `WOETransformer` WOE编码转换
- `quality` 数据质量检测
- `KS/KS_bucket` 简洁的KS计算
- 数据预处理流水线设计

**整合建议：**
- 保留 `Combiner` 的简洁接口设计
- 学习 `quality` 的数据质量报告风格
- 参考 `KS_bucket` 的实现优化

### 4.2 optbinning 借鉴

**优秀设计：**
- 严格的分箱优化算法实现
- 丰富的约束条件支持 (单调性/连续性/单峰性)
- 完整的分箱统计报告
- 可视化分箱效果
- 数值稳定性处理

**整合建议：**
- 学习其数学优化方法
- 借鉴约束条件的处理方式
- 参考分箱质量评估指标

### 4.3 scorecardpy 借鉴

**优秀设计：**
- `woebin/woebin_ply/woebin_plot` 简洁API
- `scorecard/scorecard_ply` 评分卡转换
- `perf_eva/perf_psi` 模型评估
- 完整的评分卡建模流程

**整合建议：**
- 保持相似的API命名
- 学习评分卡转换逻辑
- 参考模型评估指标体系

### 4.4 skorecard 借鉴

**优秀设计：**
- sklearn风格的API设计
- Pipeline友好
- 详细的文档和示例
- 模块化设计

**整合建议：**
- 保持sklearn兼容性
- Pipeline集成测试
- 文档风格参考

### 4.5 驻场代码工具包 借鉴

**优秀实践：**
- `ModelTuner` 类实现参数调优可视化
- `calc_ks` 多方法实现
- `model_set` 统一模型创建接口
- 贝叶斯调参与结果可视化
- 建模文档自动生成

**整合建议：**
- 集成 `ModelTuner` 的调参功能
- 参考多模型统一接口设计
- 整合建模报告生成逻辑
- 学习循环删变量的实现

---

## 五、代码质量改进建议

### 5.1 类型注解
- 全模块添加类型注解
- 使用 `typing` 模块
- 复杂类型使用 `TypeVar`

### 5.2 文档规范
- 统一Google风格docstring
- 添加使用示例
- 参数/返回值详细说明

### 5.3 测试覆盖
- 单元测试覆盖所有公共API
- 集成测试覆盖主要流程
- 边界条件测试

### 5.4 错误处理
- 自定义异常类
- 详细的错误信息
- 合理的异常层次

### 5.5 性能优化
- 关键路径使用Numba加速
- 大数据集分块处理
- 并行计算支持

---

## 六、API设计规范

### 6.1 命名规范
- 类名: PascalCase (如 `OptimalBinning`)
- 函数/方法: snake_case (如 `get_bin_table`)
- 常量: UPPER_SNAKE_CASE
- 私有: _leading_underscore

### 6.2 接口一致性
- 所有transformer实现 `fit/transform/fit_transform`
- 所有selector实现 `get_support/get_feature_names_out`
- 所有binner实现 `get_bin_table/plot_bin`

### 6.3 参数规范
- 数据参数: `X, y` (sklearn标准)
- 样本权重: `sample_weight`
- 随机种子: `random_state`
- 并行数: `n_jobs`

---

## 七、目录结构最终形态

```
hscredit/
├── hscredit/                      # 主包
│   ├── __init__.py               # 统一API导出
│   ├── core/                     # 核心模块
│   │   ├── binning/              # 分箱算法
│   │   ├── encoders/             # 特征编码
│   │   ├── selectors/            # 特征筛选
│   │   ├── models/               # 模型与损失函数
│   │   │   └── losses/           # 自定义损失函数
│   │   ├── metrics/              # 评估指标
│   │   ├── viz/                  # 可视化
│   │   ├── feature_engineering/  # 特征工程
│   │   ├── rules/                # 规则引擎
│   │   ├── financial/            # 金融计算
│   │   ├── tuning/               # 【新增】超参数调优
│   │   └── explainability/       # 【新增】模型解释性
│   ├── report/                   # 报告模块
│   │   ├── excel/                # Excel报告
│   │   └── ppt/                  # 【新增】PPT报告
│   └── utils/                    # 工具模块
├── tests/                        # 测试
│   ├── unit/                     # 单元测试
│   └── integration/              # 集成测试
├── examples/                     # 示例
│   ├── binning/                  # 分箱示例
│   ├── modeling/                 # 建模示例
│   └── reporting/                # 报告示例
├── docs/                         # 文档
│   ├── api/                      # API文档
│   ├── tutorials/                # 教程
│   └── examples/                 # 示例文档
├── benchmarks/                   # 【新增】性能基准测试
├── scripts/                      # 辅助脚本
├── pyproject.toml               # 项目配置
├── setup.py                     # 安装配置
├── requirements.txt             # 依赖
└── README.md                    # 项目说明
```

---

## 八、实施路线图

### Phase 1: 核心功能补全 (4-6周)
1. **模型调参模块** (2周)
   - Optuna集成
   - 可视化功能
   - 示例和文档

2. **SHAP可解释性** (2周)
   - SHAP封装
   - 可视化集成
   - 报告生成

3. **评分卡完善** (2周)
   - 分数转换
   - 部署代码生成
   - 评分卡表

### Phase 2: 功能增强 (3-4周)
1. **特征工程扩展** (1周)
2. **PPT报告生成** (1周)
3. **模型持久化** (1周)
4. **测试覆盖提升** (1周)

### Phase 3: 优化完善 (2-3周)
1. **性能优化** (1周)
2. **文档完善** (1周)
3. **示例丰富** (1周)

### Phase 4: 开源准备 (2周)
1. **CI/CD配置**
2. **发布流程**
3. **社区文档**

---

## 九、总结

本设计方案在继承scorecardpipeline优势的基础上，整合toad/optbinning/scorecardpy/skorecard等优秀开源库的设计思想，结合驻场代码工具包的实践经验，打造一款自主可控、功能完整、API简洁的金融风控建模工具包。

核心设计原则：
1. **简单性**：API简单易用，降低学习成本
2. **完整性**：覆盖建模全流程，无需切换工具
3. **美观性**：可视化精美，报告专业
4. **扩展性**：模块化设计，易于扩展
5. **性能**：关键路径优化，大数据集友好
