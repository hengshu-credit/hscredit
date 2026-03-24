# hscredit 示例代码

本目录包含 hscredit 库各个模块的功能演示代码，使用真实信贷数据（hscredit.xlsx）。

## 数据说明

**数据集**: `hscredit.xlsx`
- **样本数**: 12,448 条信贷记录
- **目标变量**: 
  - FPD15: 首逾标签（0=未逾期，1=逾期），逾期率 6.64%
  - SFPD15: 首二逾标签
- **特征**: 82个信贷相关特征，包括：
  - 贷款行为特征（loan_behavior_score等）
  - 机构查询特征（lender_count_12m等）
  - 逾期记录特征（overdue_loan_m1_count_6m等）
  - 网贷特征（network_loan_lender_count等）
  - 消费金融特征
  - 履约记录特征
- **时间范围**: 放款时间从 2024-11 到 2025-08
- **MOB字段**: 
  - MOB1: 首期逾期天数
  - MOB2: 前两期逾期天数

## 目录结构

```
examples/
├── 01_binning.ipynb          # 分箱模块演示
├── 02_encoders.ipynb         # 编码器模块演示
├── 03_selectors.ipynb        # 特征筛选模块演示
├── 04_models.ipynb           # 模型模块演示
├── 05_rules.ipynb            # 规则引擎演示
├── 06_viz.ipynb              # 可视化模块演示
├── 07_metrics.ipynb          # 指标计算演示
├── 08_complete_workflow.ipynb # 完整工作流程演示
├── hscredit.xlsx             # 真实信贷数据
├── output/                   # 输出结果目录
│   ├── *.png                 # 可视化图表
│   └── *.pkl                 # 模型和组件文件
└── README.md                 # 本文件
```

## 快速开始

1. 确保已安装 hscredit 库
```bash
pip install -e ..
```

2. 启动 Jupyter Notebook
```bash
jupyter notebook
```

3. 按顺序运行各个 Notebook 学习 hscredit 的功能

## 各 Notebook 说明

### 01_binning.ipynb - 分箱模块
演示 hscredit 的各种分箱算法，使用真实信贷数据：
- **基础分箱**: 等宽分箱、等频分箱
- **目标导向分箱**: 决策树分箱、卡方分箱
- **优化分箱**: 最优KS分箱、最优IV分箱、MDLP分箱
- **高级分箱**: 单调性约束分箱
- **统一接口**: OptimalBinning

**重点特征**:
- loan_behavior_score (贷款行为评分)
- lender_count_12m (12个月机构查询数)
- overdue_loan_m1_count_6m (6个月M1逾期次数)

### 02_encoders.ipynb - 编码器模块
演示各种特征编码方法：
- **WOE编码**: 评分卡建模最常用
- **Target编码**: 目标均值编码
- **Count编码**: 频次编码
- **GBM编码**: 梯度提升树编码器

**应用场景**:
- 评分卡建模推荐 WOEEncoder
- 组合模型推荐 GBMEncoder

### 03_selectors.ipynb - 特征筛选模块
演示各种特征筛选方法：
- **基础筛选**: 方差、缺失率、单一值
- **相关性筛选**: 相关系数、VIF
- **目标导向筛选**: IV、PSI
- **模型-based筛选**: 特征重要性
- **组合筛选**: 多步骤筛选流程

**评分卡推荐筛选流程**:
1. VarianceSelector (方差筛选)
2. VIFSelector (共线性筛选)
3. CorrSelector (相关性筛选)
4. IVSelector (IV>0.02)
5. PSISelector (PSI<0.2)

### 04_models.ipynb - 模型模块
演示各种风控模型：
- **XGBoost模型**: 高精度
- **LightGBM模型**: 训练快速
- **CatBoost模型**: 类别特征友好
- **逻辑回归**: 评分卡基础
- **评分卡模型**: 业务可解释

### 05_rules.ipynb - 规则引擎
演示规则引擎功能：
- 规则定义和使用
- 规则评估指标（支持度、置信度、Lift）
- 单特征规则挖掘
- 决策树规则提取

**业务规则示例**:
- 历史逾期规则: `overdue_loan_m1_count_6m > 0`
- 高风险客户规则: `(loan_behavior_score < 500) & (lender_count_12m > 5)`

### 06_viz.ipynb - 可视化模块
演示各种可视化功能：
- 分箱图 (bin_plot)
- 相关性图 (corr_plot)
- KS曲线 (ks_plot)
- 分布图 (hist_plot)
- PSI图 (psi_plot)

### 07_metrics.ipynb - 指标计算
演示各种评估指标：
- **分类指标**: KS、AUC、Gini
- **特征重要性**: IV
- **稳定性指标**: PSI

**IV值解读**:
- IV < 0.02: 预测能力弱
- 0.02 <= IV < 0.1: 预测能力中等
- 0.1 <= IV < 0.3: 预测能力强
- IV >= 0.3: 预测能力过强（需检查）

**PSI值解读**:
- PSI < 0.1: 稳定性好
- 0.1 <= PSI < 0.25: 稳定性一般
- PSI >= 0.25: 稳定性差

### 08_complete_workflow.ipynb - 完整工作流程
演示完整的信贷风险建模流程：
1. 数据准备（加载真实信贷数据）
2. 特征工程（IV筛选、VIF筛选、分箱、WOE编码）
3. 模型训练（XGBoost、逻辑回归）
4. 模型评估（KS、AUC、Gini）
5. 评分卡构建（基础分600，PDO=50）
6. 规则挖掘（单特征规则）
7. 保存模型和组件

## 输出文件

运行 Notebook 后，会在 `output/` 目录下生成：

### 可视化图表
- `01_*.png`: 分箱相关图表
- `02_*.png`: 编码器相关图表
- `03_*.png`: 特征筛选相关图表
- `04_*.png`: 模型相关图表
- `05_*.png`: 规则引擎相关图表
- `06_*.png`: 可视化相关图表
- `07_*.png`: 指标计算相关图表
- `08_*.png`: 完整工作流程相关图表

### 模型文件
- `workflow_xgb_model.pkl`: XGBoost模型
- `workflow_lr_model.pkl`: 逻辑回归模型
- `workflow_scorecard.pkl`: 评分卡模型
- `workflow_binner.pkl`: 分箱器
- `workflow_iv_selector.pkl`: IV筛选器
- `woe_encoder.pkl`: WOE编码器
- `composite_selector.pkl`: 组合筛选器

## 依赖项

- Python >= 3.8
- hscredit
- jupyter
- matplotlib
- pandas
- numpy
- scikit-learn
- openpyxl (读取Excel)

## 安装依赖

```bash
pip install jupyter matplotlib pandas numpy scikit-learn openpyxl
```

## 注意事项

1. 所有 Notebook 都使用真实信贷数据（hscredit.xlsx）进行演示
2. 数据已脱敏处理，仅用于演示目的
3. 部分功能可能需要额外安装依赖（如 XGBoost、LightGBM、CatBoost）
4. 运行前请确保 hscredit 库已正确安装
5. 建议按编号顺序运行 Notebook

## 信贷建模最佳实践

### 1. 数据准备
- 检查目标变量分布（逾期率）
- 处理缺失值
- 划分训练集/测试集（注意时间顺序）

### 2. 特征工程
- IV筛选（阈值0.02）剔除弱特征
- VIF筛选（阈值10）剔除共线性特征
- 分箱（最优IV或最优KS）
- WOE编码（评分卡必备）

### 3. 模型选择
- **评分卡**: LogisticRegression + WOE编码
- **高精度**: XGBoost / LightGBM / CatBoost
- **可解释性**: 逻辑回归 + 评分卡

### 4. 模型评估
- KS > 0.3 可用
- AUC > 0.7 良好
- PSI < 0.1 稳定

### 5. 生产部署
- 保存分箱器、编码器、模型
- 监控模型稳定性（PSI）
- 定期更新模型

## 更多资源

- [hscredit 文档](../docs/)
- [API 参考](../docs/api.md)
- [使用指南](../docs/guide.md)
