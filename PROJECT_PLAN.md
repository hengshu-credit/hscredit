# hscredit 项目规划文档

**版本**: v0.1.0  
**最后更新**: 2026-03-15  
**状态**: 开发中

---

## 一、项目概述

### 1.1 项目背景

hscredit 是一个专业的金融信贷风险策略和模型开发库，从 scorecardpipeline (scp) 迁移而来。核心目标是成为公司级开源项目，去除对第三方风控库(toad、optbinning、scorecardpy)的依赖，自主实现核心功能，提供更易用的API和完善的文档。

### 1.2 核心价值

- **独立性**: 不依赖第三方风控库，核心功能完全自主实现
- **易用性**: sklearn风格的统一API，降低学习成本
- **专业性**: 针对金融风控场景深度优化
- **完整性**: 覆盖评分卡建模全流程
- **可扩展性**: 模块化设计，易于扩展新功能

### 1.3 主要功能模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 分箱算法 | 🟡 框架完成 | 9种分箱方法 |
| 编码转换 | 🟡 框架完成 | WOE、Target编码等 |
| 特征筛选 | 🟡 框架完成 | 10种筛选方法 |
| 指标计算 | 🟡 框架完成 | KS、AUC、PSI、IV、CSI等 |
| 自定义损失函数 | ✅ 已完成 | 6种损失函数+3种评估指标 |
| 评分卡建模 | 🟡 待开发 | 评分卡生成和转换 |
| 策略分析 | 🟡 待开发 | 规则挖掘和效果评估 |
| 报告输出 | 🟡 待开发 | Excel报告和可视化 |
| PMML导出 | 🟡 待开发 | 模型导出标准格式 |

**图例**: ✅ 已完成 | 🟡 进行中 | ⚪ 待开始

---

## 二、技术架构

### 2.1 整体架构

```
hscredit/
├── core/                      # 核心算法层
│   ├── binning/              # 分箱算法
│   │   ├── base.py           # 分箱基类
│   │   ├── uniform_binning.py    # 等距分箱
│   │   ├── quantile_binning.py   # 等频分箱
│   │   ├── tree_binning.py       # 决策树分箱
│   │   ├── chi_merge_binning.py  # 卡方分箱
│   │   ├── mdlp_binning.py       # MDLP分箱
│   │   ├── optimal_ks_binning.py # 最优KS分箱
│   │   ├── optimal_auc_binning.py # 最优AUC分箱
│   │   ├── kmeans_binning.py     # K-means分箱
│   │   └── optimal_binning.py    # 统一接口
│   │
│   ├── encoding/             # 编码转换
│   │   ├── base.py
│   │   ├── woe_encoder.py    # WOE编码
│   │   └── target_encoder.py # 目标编码
│   │
│   ├── selection/            # 特征筛选
│   │   ├── iv_selector.py         # IV筛选
│   │   ├── correlation_selector.py # 相关性筛选
│   │   ├── vif_selector.py        # VIF筛选
│   │   ├── missing_rate_selector.py # 缺失率筛选
│   │   ├── unique_rate_selector.py  # 单一值率筛选
│   │   ├── variance_selector.py     # 方差筛选
│   │   ├── feature_importance_selector.py # 特征重要性筛选
│   │   ├── rfe_selector.py          # 递归特征消除
│   │   ├── stepwise_selector.py     # 逐步回归
│   │   └── feature_selector.py      # 统一接口
│   │
│   └── metrics/              # 指标计算
│       ├── classification.py # 分类指标(KS、AUC等)
│       ├── stability.py      # 稳定性指标(PSI、CSI)
│       └── importance.py     # 特征重要性(IV)
│
├── model/                    # 模型层
│   ├── linear/               # 线性模型
│   │   └── logistic_regression.py
│   │
│   ├── scorecard/            # 评分卡
│   │   ├── scorecard.py      # 评分卡生成
│   │   └── validator.py      # 模型验证
│   │
│   ├── losses/               # 自定义损失函数 ✅
│   │   ├── base.py           # 基类定义
│   │   ├── focal_loss.py     # Focal Loss
│   │   ├── weighted_loss.py  # 加权损失
│   │   ├── risk_loss.py      # 风控业务损失
│   │   ├── custom_metrics.py # 自定义评估指标
│   │   └── adapters.py       # 框架适配器
│   │
│   └── ensemble/             # 集成模型
│       └── (未来扩展)
│
├── analysis/                 # 分析层
│   ├── strategy/             # 策略分析
│   │   ├── strategy_evaluator.py
│   │   └── rule_optimizer.py
│   │
│   ├── rules/                # 规则挖掘
│   │   ├── rule_extractor.py
│   │   └── rule_evaluator.py
│   │
│   └── stability/            # 稳定性分析
│       ├── psi_monitor.py
│       └── drift_detector.py
│
├── report/                   # 报告层
│   ├── excel/                # Excel报告
│   │   ├── report_generator.py
│   │   └── templates/
│   │
│   ├── plot/                 # 可视化
│   │   ├── ks_plot.py
│   │   ├── roc_plot.py
│   │   ├── distribution_plot.py
│   │   └── feature_importance_plot.py
│   │
│   └── template/             # 模板文件
│       └── excel_templates/
│
├── utils/                    # 工具层
│   ├── data/                 # 数据处理
│   │   ├── data_cleaner.py
│   │   ├── data_splitter.py
│   │   └── data_sampler.py
│   │
│   ├── validation/           # 数据验证
│   │   └── data_validator.py
│   │
│   └── logging/              # 日志系统
│       └── logger.py
│
├── examples/                 # 示例代码 ✅
│   ├── basic_usage.py        # 基础用法
│   ├── custom_loss_usage.py  # 自定义损失函数示例 ✅
│   ├── advanced_pipeline.py  # 高级Pipeline
│   └── datasets/             # 示例数据集
│
├── tests/                    # 测试代码 ✅
│   ├── test_binning.py       # 分箱测试
│   ├── test_losses.py        # 损失函数测试 ✅
│   ├── test_encoding.py      # 编码测试
│   └── test_metrics.py       # 指标测试
│
└── docs/                     # 文档 ✅
    ├── api/                  # API文档
    ├── tutorials/            # 教程
    ├── loss_functions.md     # 损失函数文档 ✅
    ├── CUSTOM_LOSS_IMPLEMENTATION.md ✅
    └── migration.md          # 迁移指南
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────┐
│           应用层 (Application)           │
│  examples / tests / docs                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           报告层 (Report)                │
│  report (Excel报告、可视化)              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           分析层 (Analysis)              │
│  analysis (策略分析、规则挖掘)           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           模型层 (Model)                 │
│  model (线性模型、评分卡、损失函数)      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           核心层 (Core)                  │
│  core (分箱、编码、筛选、指标)           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           工具层 (Utils)                 │
│  utils (数据处理、验证、日志)            │
└─────────────────────────────────────────┘
```

### 2.3 技术栈

#### 核心依赖
- **Python**: >= 3.7
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Scikit-learn**: 机器学习基础框架

#### 可选依赖
- **XGBoost**: 梯度提升框架
- **LightGBM**: 轻量级梯度提升框架
- **CatBoost**: CatBoost梯度提升框架
- **PyTorch**: TabNet支持
- **OR-Tools**: 最优分箱求解器
- **OpenPyXL**: Excel报告生成

---

## 三、核心模块详细设计

### 3.1 分箱模块 (core/binning) 🟡

#### 实现状态
- [x] 基类设计 (BaseBinning)
- [ ] 等距分箱 (UniformBinning)
- [ ] 等频分箱 (QuantileBinning)
- [ ] 决策树分箱 (TreeBinning)
- [ ] 卡方分箱 (ChiMergeBinning)
- [ ] MDLP分箱 (MDLPBinning)
- [ ] 最优KS分箱 (OptimalKSBinning)
- [ ] 最优AUC分箱 (OptimalAUCBinning)
- [ ] K-Means分箱 (KMeansBinning)
- [ ] 统一接口 (OptimalBinning)

#### 设计原则

**统一的参数命名**:
```python
# 所有分箱方法使用统一的参数
binner = TreeBinning(
    target='target',           # 目标变量名
    max_n_bins=5,             # 最大分箱数
    min_n_bins=2,             # 最小分箱数
    min_bin_size=0.05,        # 最小箱样本占比
    max_bin_size=0.5,         # 最大箱样本占比
    monotonic=True,           # 单调性约束
    missing_separate=True,    # 缺失值单独分箱
    n_jobs=-1,               # 并行计算
    random_state=42          # 随机种子
)
```

**关键特性**:
1. **单调性约束**: 支持ascending、descending、auto
2. **缺失值处理**: 单独分箱或合并到最近箱
3. **特殊值处理**: 支持自定义特殊值编码
4. **类别型变量**: 自动识别和处理
5. **Pipeline支持**: 与sklearn Pipeline无缝集成

#### 实现示例

```python
from hscredit.core.binning import OptimalBinning

# 方法1: 自动选择最优方法
binner = OptimalBinning(method='auto')
binner.fit(X, y)

# 方法2: 指定分箱方法
binner = OptimalBinning(
    method='chi_merge',
    max_n_bins=5,
    monotonic_trend='ascending',
    min_bin_size=0.05
)

# 拟合和转换
binner.fit(X_train['age'], y_train)
X_test_binned = binner.transform(X_test['age'])

# 获取分箱信息
bin_table = binner.bin_table_
splits = binner.splits_
iv_value = binner.iv_
```

#### 算法实现细节

**决策树分箱**:
- 基于sklearn DecisionTreeClassifier
- 提取决策树的分割点作为分箱边界
- 支持限制树的深度和叶子节点数
- 自动处理类别型变量

**卡方分箱**:
- 自主实现Python版本（不依赖Rust）
- 基于卡方统计量合并相邻箱
- 支持最小卡方值阈值和最大分箱数
- 迭代合并直到满足停止条件

**最优分箱**:
- 参考 optbinning 的实现思路
- 使用约束优化求解最优分箱
- 支持多种目标函数（KS、AUC、IV）
- 使用 OR-Tools CP-SAT 求解器

### 3.2 编码模块 (core/encoding) 🟡

#### 实现状态
- [x] 基类设计 (BaseEncoder)
- [ ] WOE编码 (WOEEncoder)
- [ ] 目标编码 (TargetEncoder)
- [ ] 计数编码 (CountEncoder)
- [ ] 频率编码 (FrequencyEncoder)

#### WOE编码实现

**数学公式**:
```
WOE = ln(P(Good|Bin) / P(Bad|Bin))
    = ln((Good_i / Good_total) / (Bad_i / Bad_total))
```

**平滑处理**:
```python
# 添加平滑因子避免零频问题
WOE = ln((Good_i + alpha) / (Good_total + alpha * n_bins) / 
         (Bad_i + alpha) / (Bad_total + alpha * n_bins))
```

**API设计**:
```python
from hscredit.core.encoding import WOEEncoder

encoder = WOEEncoder(
    smooth=True,              # 是否平滑
    smooth_factor=1.0,        # 平滑因子
    handle_unknown='value',   # 未知类别处理
    handle_missing='value'    # 缺失值处理
)

encoder.fit(X_binned, y)
X_woe = encoder.transform(X_binned)

# 获取WOE映射表
woe_mapping = encoder.woe_mapping_
```

### 3.3 特征筛选模块 (core/selection) 🟡

#### 实现状态
- [x] 模块框架设计
- [ ] IV筛选器 (IVSelector)
- [ ] 相关性筛选器 (CorrelationSelector)
- [ ] VIF筛选器 (VIFSelector)
- [ ] 缺失率筛选器 (MissingRateSelector)
- [ ] 单一值率筛选器 (UniqueRateSelector)
- [ ] 方差筛选器 (VarianceSelector)
- [ ] 特征重要性筛选器 (FeatureImportanceSelector)
- [ ] RFE筛选器 (RFESelector)
- [ ] 逐步回归 (StepwiseSelector)
- [ ] 统一筛选器 (FeatureSelector)

#### 统一筛选器设计

```python
from hscredit.core.selection import FeatureSelector

selector = FeatureSelector(
    # IV筛选
    iv_threshold=0.02,
    
    # 相关性筛选
    corr_threshold=0.7,
    corr_method='pearson',
    
    # VIF筛选
    vif_threshold=10,
    
    # 缺失率筛选
    missing_threshold=0.95,
    
    # 单一值率筛选
    unique_threshold=0.95,
    
    # 方差筛选
    variance_threshold=0.01,
    
    # 执行顺序
    selection_order=['missing', 'unique', 'variance', 'iv', 'corr', 'vif']
)

selector.fit(X, y)
X_selected = selector.transform(X)

# 查看筛选结果
print(selector.selected_features_)
print(selector.removed_features_)
print(selector.selection_report_)
```

#### 逐步回归实现

```python
from hscredit.core.selection import StepwiseSelector

stepwise = StepwiseSelector(
    direction='both',         # 'forward', 'backward', 'both'
    criterion='aic',          # 'aic', 'bic', 'pvalue'
    p_enter=0.05,            # 进入阈值
    p_remove=0.10,           # 移除阈值
    max_steps=100,           # 最大步数
    verbose=True             # 显示过程
)

stepwise.fit(X_woe, y)
X_final = stepwise.transform(X_woe)

# 查看逐步回归过程
print(stepwise.steps_)
```

### 3.4 指标计算模块 (core/metrics) 🟡

#### 实现状态
- [x] 模块框架设计
- [ ] KS计算和分桶
- [ ] AUC计算
- [ ] PSI计算
- [ ] CSI计算
- [ ] IV计算

#### API设计

```python
from hscredit.core.metrics import KS, AUC, PSI, CSI, IV

# KS计算
ks_value = KS(y_true, y_pred)
ks_table = KS_bucket(y_true, y_pred, bucket=10)

# AUC计算
auc_value = AUC(y_true, y_pred)

# PSI计算
psi_value = PSI(train_score, test_score, n_bins=10)

# CSI计算（特征稳定性）
csi_value = CSI(train_feature, test_feature, n_bins=10)

# IV计算
iv_df = IV(X, y, return_dataframe=True)
```

### 3.5 自定义损失函数模块 (model/losses) ✅

#### 实现状态
- [x] 基类设计 (BaseLoss, BaseMetric)
- [x] Focal Loss
- [x] Weighted BCE Loss
- [x] Cost Sensitive Loss
- [x] Bad Debt Loss
- [x] Approval Rate Loss
- [x] Profit Max Loss
- [x] KS Metric
- [x] Gini Metric
- [x] PSI Metric
- [x] XGBoost适配器
- [x] LightGBM适配器
- [x] CatBoost适配器
- [x] TabNet适配器

#### 完整文档
详见: `docs/loss_functions.md` 和 `docs/CUSTOM_LOSS_IMPLEMENTATION.md`

#### 快速示例

```python
from hscredit.core.models import (
    FocalLoss, CostSensitiveLoss, BadDebtLoss,
    KSMetric, XGBoostLossAdapter
)

# 示例1: 不平衡数据处理
loss = FocalLoss(alpha=0.75, gamma=2.0)
adapter = XGBoostLossAdapter(loss)

# 示例2: 成本敏感学习
loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)

# 示例3: 业务目标优化
loss = BadDebtLoss(target_approval_rate=0.3)

# 示例4: 自定义评估指标
ks_metric = KSMetric()
```

---

## 四、开发计划

### 4.1 总体时间线

```
Phase 1: 核心算法层 (Week 1-5)
    ├── Week 1-2: 分箱算法实现
    ├── Week 3: 编码转换实现
    ├── Week 4: 特征筛选实现
    └── Week 5: 指标计算实现

Phase 2: 模型层 (Week 6-7)
    ├── Week 6: 评分卡建模
    └── Week 7: 模型验证和导出

Phase 3: 分析层 (Week 8-9)
    ├── Week 8: 策略分析和规则挖掘
    └── Week 9: 稳定性分析

Phase 4: 报告层 (Week 10-11)
    ├── Week 10: Excel报告生成
    └── Week 11: 可视化

Phase 5: 完善和发布 (Week 12)
    ├── 文档完善
    ├── 测试覆盖
    └── 发布准备
```

### 4.2 详细开发计划

#### Week 1-2: 分箱算法实现

**Week 1: 基础分箱**
- [ ] 实现UniformBinning（等距分箱）
- [ ] 实现QuantileBinning（等频分箱）
- [ ] 实现KMeansBinning（K-means分箱）
- [ ] 编写单元测试
- [ ] 编写使用示例

**Week 2: 高级分箱**
- [ ] 实现TreeBinning（决策树分箱）
- [ ] 实现ChiMergeBinning（卡方分箱）
- [ ] 实现MDLPBinning（MDLP分箱）
- [ ] 实现OptimalKSBinning（最优KS分箱）
- [ ] 实现OptimalAUCBinning（最优AUC分箱）
- [ ] 实现OptimalBinning（统一接口）
- [ ] 编写完整测试和文档

**验收标准**:
- 所有分箱方法通过单元测试
- 分箱结果与toad/optbinning对比验证
- 完整的API文档和使用示例

#### Week 3: 编码转换实现

- [ ] 实现WOEEncoder
- [ ] 实现TargetEncoder
- [ ] 实现CountEncoder
- [ ] 实现FrequencyEncoder
- [ ] 添加平滑处理
- [ ] 支持Pipeline集成
- [ ] 编写测试和文档

**验收标准**:
- 编码结果正确性验证
- 与scorecardpy结果对比
- Pipeline兼容性测试

#### Week 4: 特征筛选实现

- [ ] 实现IVSelector
- [ ] 实现CorrelationSelector
- [ ] 实现VIFSelector
- [ ] 实现MissingRateSelector
- [ ] 实现UniqueRateSelector
- [ ] 实现VarianceSelector
- [ ] 实现FeatureImportanceSelector
- [ ] 实现RFESelector
- [ ] 实现StepwiseSelector
- [ ] 实现FeatureSelector（统一接口）

**验收标准**:
- 筛选逻辑正确性验证
- 与toad.selection对比
- 性能优化（大数据集测试）

#### Week 5: 指标计算实现

- [ ] 实现KS计算和分桶
- [ ] 实现AUC计算
- [ ] 实现PSI计算
- [ ] 实现CSI计算
- [ ] 实现IV计算
- [ ] 添加可视化函数
- [ ] 性能优化

**验收标准**:
- 计算结果与toad/sklearn对比
- 性能测试（百万级数据）
- 可视化效果验证

#### Week 6: 评分卡建模

- [ ] 实现ScoreCard类
- [ ] 支持评分转换（PDO、基准分）
- [ ] 实现评分卡表生成
- [ ] 支持PMML导出
- [ ] 实现评分卡验证
- [ ] 编写完整示例

**验收标准**:
- 评分转换正确性验证
- PMML导出可用性测试
- 与scorecardpy对比验证

#### Week 7: 模型验证和导出

- [ ] 实现模型验证器
- [ ] 支持多种验证指标
- [ ] 实现交叉验证
- [ ] 实现模型持久化
- [ ] 完善PMML导出
- [ ] 编写验证报告模板

#### Week 8: 策略分析和规则挖掘

- [ ] 实现策略效果评估
- [ ] 实现规则提取器
- [ ] 实现规则评估器
- [ ] 实现策略优化
- [ ] 支持组合规则

#### Week 9: 稳定性分析

- [ ] 实现PSI监控
- [ ] 实现特征漂移检测
- [ ] 实现模型性能监控
- [ ] 生成监控报告

#### Week 10: Excel报告生成

- [ ] 设计报告模板
- [ ] 实现报告生成器
- [ ] 支持自定义模板
- [ ] 生成模型摘要
- [ ] 生成特征分析报告

#### Week 11: 可视化

- [ ] 实现KS曲线图
- [ ] 实现ROC曲线图
- [ ] 实现分布图
- [ ] 实现特征重要性图
- [ ] 实现评分分布图

#### Week 12: 完善和发布

- [ ] 完善API文档
- [ ] 补充使用教程
- [ ] 提高测试覆盖率
- [ ] 性能优化
- [ ] 发布v0.1.0版本

---

## 五、质量保证

### 5.1 代码规范

- **风格**: 遵循PEP 8规范
- **格式化**: 使用black自动格式化
- **import排序**: 使用isort
- **类型注解**: 关键函数添加类型注解
- **文档字符串**: 所有公共API必须有docstring

### 5.2 测试策略

**单元测试**:
- 每个模块对应一个测试文件
- 测试覆盖率 >= 80%
- 使用pytest框架

**集成测试**:
- Pipeline集成测试
- 端到端流程测试
- 框架兼容性测试

**性能测试**:
- 大数据集性能测试
- 内存使用测试
- 并行计算测试

**对比验证**:
- 与toad结果对比
- 与optbinning结果对比
- 与scorecardpy结果对比

### 5.3 文档规范

**API文档**:
- 每个类和函数都有详细说明
- 包含参数说明、返回值、示例
- 使用NumPy风格的docstring

**使用教程**:
- 快速开始指南
- 完整示例代码
- 最佳实践建议

**迁移指南**:
- 从scorecardpipeline迁移
- 从toad/optbinning迁移
- API对比表

---

## 六、性能优化

### 6.1 算法优化

- **分箱算法**: 使用Cython加速关键计算
- **特征筛选**: 支持并行计算
- **指标计算**: 向量化实现
- **大数据集**: 分块处理

### 6.2 内存优化

- 使用稀疏矩阵
- 及时释放中间结果
- 支持流式处理

### 6.3 并行计算

- 使用joblib并行化
- 支持多进程
- 支持GPU加速（可选）

---

## 七、发布计划

### 7.1 版本规划

**v0.1.0 (Week 12)**:
- 核心算法层完成
- 基础模型功能
- 初步文档

**v0.2.0 (Month 2)**:
- 完整分析功能
- Excel报告生成
- 可视化功能

**v0.3.0 (Month 3)**:
- 性能优化
- 高级功能
- 完善文档

**v1.0.0 (Month 4)**:
- 生产就绪
- 完整测试
- 详细文档

### 7.2 发布渠道

- **PyPI**: 官方发布
- **GitHub**: 源码和文档
- **ReadTheDocs**: 在线文档

---

## 八、风险管理

### 8.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 算法实现难度大 | 高 | 充分研究参考实现，寻求专家支持 |
| 性能不达标 | 中 | 使用Cython优化，支持并行计算 |
| 兼容性问题 | 中 | 充分测试，支持多版本Python |

### 8.2 进度风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 开发时间不足 | 高 | 合理规划优先级，迭代发布 |
| 文档编写耗时 | 中 | 边开发边写文档，使用模板 |
| 测试覆盖不足 | 中 | 自动化测试，持续集成 |

---

## 九、成功标准

### 9.1 功能完整性

- [ ] 所有核心模块实现完成
- [ ] API设计一致性好
- [ ] 与参考库功能对等

### 9.2 质量标准

- [ ] 测试覆盖率 >= 80%
- [ ] 文档覆盖率 >= 90%
- [ ] 代码规范检查通过

### 9.3 性能标准

- [ ] 大数据集处理性能可接受
- [ ] 内存使用合理
- [ ] 并行计算有效

### 9.4 易用性标准

- [ ] API学习曲线平缓
- [ ] 错误提示清晰
- [ ] 文档完善易懂

---

## 十、参考资源

### 10.1 参考项目

- **toad**: https://github.com/amphibian-dev/toad
- **optbinning**: https://github.com/guillermo-navas-palencia/optbinning
- **scorecardpy**: https://github.com/ShichenXie/scorecardpy
- **scikit-learn**: https://scikit-learn.org/

### 10.2 学术资源

- Focal Loss论文: "Focal Loss for Dense Object Detection" (ICCV 2017)
- 卡方分箱: "ChiMerge: Discretization of Numeric Attributes" (1992)
- MDLP: "Multi-Interval Discretization of Continuous-Valued Attributes" (1993)

---

## 附录：API速查表

### 分箱模块

```python
from hscredit.core.binning import (
    OptimalBinning,     # 统一接口
    TreeBinning,        # 决策树分箱
    ChiMergeBinning,    # 卡方分箱
    QuantileBinning,    # 等频分箱
    UniformBinning,     # 等距分箱
)
```

### 编码模块

```python
from hscredit.core.encoding import (
    WOEEncoder,         # WOE编码
    TargetEncoder,      # 目标编码
)
```

### 特征筛选

```python
from hscredit.core.selection import (
    FeatureSelector,    # 统一筛选器
    StepwiseSelector,   # 逐步回归
    IVSelector,         # IV筛选
)
```

### 指标计算

```python
from hscredit.core.metrics import (
    KS, AUC, PSI, CSI, IV
)
```

### 损失函数

```python
from hscredit.core.models import (
    FocalLoss,          # Focal Loss
    WeightedBCELoss,    # 加权BCE
    CostSensitiveLoss,  # 成本敏感
    BadDebtLoss,        # 坏账率优化
    ProfitMaxLoss,      # 利润最大化
    KSMetric,           # KS指标
    XGBoostLossAdapter, # XGBoost适配器
)
```

---

**最后更新**: 2026-03-15  
**文档版本**: v1.0  
**状态**: 已更新
