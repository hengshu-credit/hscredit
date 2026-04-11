<p align="center">
  <img src="https://hengshucredit.com/images/hengshucredit_animated.svg" alt="衡枢真信" width="200">
</p>

<h1 align="center">HSCredit - 金融信贷风险建模工具包</h1>

<p align="center">
  <b>鉴真伪，斟信用，衡风险，枢定策 —— 智能风控，一站掌控</b>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/hscredit/hscredit"><img src="https://img.shields.io/badge/version-0.1.0-orange" alt="Version"></a>
</p>

---

## 为什么选择 HSCredit

**HSCredit** 是一个专为金融风控场景设计的评分卡建模工具包，提供从数据探索、特征工程、分箱、编码、筛选到建模和报告生成的**全流程解决方案**。

### 核心优势

| 优势 | 说明 |
|------|------|
| **完整性** | 覆盖评分卡建模全流程，一站式解决风控建模需求 |
| **统一性** | 所有模块遵循统一API设计，完美支持 sklearn Pipeline 集成 |
| **灵活性** | 支持双API风格、自定义参数、多种算法选择 |
| **专业性** | 16种分箱算法、20+种筛选器、8种编码器、7种风控专用损失函数 |
| **易用性** | 中文输出、Pandas扩展、25+可视化图表、详细文档 |
| **可扩展性** | 基类抽象、插件化设计，便于扩展新功能 |
| **兼容性** | 与 toad、scorecardpipeline 等主流风控库无缝兼容 |

### 支持的算法一览

- **16种分箱算法**: 等宽、等频、决策树、卡方、最优IV、最优KS、MDLP、遗传算法、单调约束等
- **20+种特征筛选**: IV、VIF、相关性、Boruta、Null Importance、逐步回归等
- **8种编码器**: WOE、Target、CatBoost、GBM、Count、OneHot等
- **7种损失函数**: Focal Loss、成本敏感、坏账损失、利润最大化等风控专用损失

---

## 安装

```bash
# 基础安装
pip install hscredit

# 开发模式安装
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e .

# 安装可选依赖
pip install hscredit[xgboost]      # XGBoost支持
pip install hscredit[lightgbm]     # LightGBM支持
pip install hscredit[catboost]     # CatBoost支持
pip install hscredit[deep-learning] # 深度学习支持
pip install hscredit[pmml]         # PMML导出支持
pip install hscredit[dev]          # 开发工具
pip install hscredit[docs]         # 文档工具
```

---

## 快速开始

### 1. 分箱 (Binning)

```python
from hscredit import OptimalBinning

# 使用最优IV分箱
binner = OptimalBinning(method='best_iv', max_n_bins=5)
binner.fit(X_train, y_train)

# 应用分箱
X_binned = binner.transform(X_test, metric='woe')

# 查看分箱统计表
bin_table = binner.get_bin_table('age')
print(bin_table)
```

### 2. 特征筛选 (Feature Selection)

```python
from hscredit import IVSelector, VIFSelector, CompositeFeatureSelector

# IV筛选
iv_selector = IVSelector(threshold=0.02)
iv_selector.fit(X, y)
X_selected = iv_selector.transform(X)

# 组合多个筛选器
selector = CompositeFeatureSelector([
    ('iv', IVSelector(threshold=0.02)),
    ('vif', VIFSelector(threshold=10)),
])
X_selected = selector.fit_transform(X, y)

# 获取筛选报告
report = selector.get_selection_report()
print(report)
```

### 3. 编码 (Encoding)

```python
from hscredit import WOEEncoder, TargetEncoder

# WOE编码
woe_encoder = WOEEncoder()
X_woe = woe_encoder.fit_transform(X, y)

# 目标编码
target_encoder = TargetEncoder()
X_target = target_encoder.fit_transform(X, y)
```

### 4. 建模 (Modeling)

```python
from hscredit import XGBoostRiskModel, LogisticRegression, ScoreCard

# XGBoost模型
xgb_model = XGBoostRiskModel(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    eval_metric=['auc', 'ks']
)
xgb_model.fit(X_train, y_train)
proba = xgb_model.predict_proba(X_test)

# 逻辑回归
lr_model = LogisticRegression(calculate_stats=True)
lr_model.fit(X_train, y_train)
summary = lr_model.summary()

# 评分卡
scorecard = ScoreCard(
    lr_model,
    pdo=20,
    base_score=600,
    base_odds=1/19
)
scores = scorecard.transform(X_test)
```

### 5. 可视化 (Visualization)

```python
from hscredit import bin_plot, ks_plot, roc_plot, score_dist_plot

# 分箱图
bin_plot(df, target='target', feature='age', save='bin_plot.png')

# KS曲线
ks_plot(score=y_proba, target=y_test, save='ks_plot.png')

# ROC曲线
roc_plot(y_test, y_proba, save='roc_plot.png')

# 评分分布
score_dist_plot(df, score_col='score', target_col='target')
```

### 6. 探索性数据分析 (EDA)

```python
import hscredit.core.eda as eda

# 数据概览
info = eda.data_info(df)

# IV分析
iv_result = eda.batch_iv_analysis(df, features=['age', 'income'], target='target')

# 逾期率趋势
trend = eda.bad_rate_trend(df, target_col='target', date_col='apply_date')

# Vintage分析
vintage = eda.vintage_analysis(df, vintage_col='issue_month', mob_col='mob', target_col='ever_dpd30')

# 综合报告
summary = eda.eda_summary(df, target='target')
```

### 7. Pandas 扩展

```python
import pandas as pd
import hscredit  # 自动注册扩展方法

# 数据摘要
summary = df.summary(y='target')

# 保存到Excel
df.save('report.xlsx', sheet_name='数据', title='统计表')

# 美化展示分箱表
bin_table.show(compact=True)
```

---

## 项目结构

### 核心模块 (Core)

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| `core.binning` | 分箱算法 | `OptimalBinning`, `BestIVBinning`, `MDLPBinning`, `MonotonicBinning`, 等16+种分箱方法 |
| `core.selectors` | 特征筛选 | `IVSelector`, `VIFSelector`, `CorrSelector`, `BorutaSelector`, 等20+种筛选器 |
| `core.encoders` | 特征编码 | `WOEEncoder`, `TargetEncoder`, `CatBoostEncoder`, `GBMEncoder`, 等8种编码器 |
| `core.models` | 模型 | `XGBoostRiskModel`, `LightGBMRiskModel`, `LogisticRegression`, `ScoreCard`, `RuleSet` |
| `core.metrics` | 评估指标 | `ks`, `auc`, `gini`, `iv`, `psi`, `lift`, `badrate` |
| `core.viz` | 可视化 | `bin_plot`, `ks_plot`, `roc_plot`, `score_dist_plot`, `vintage_plot`, 等25+图表 |
| `core.eda` | 探索性分析 | `data_info`, `iv_analysis`, `bad_rate_trend`, `vintage_analysis`, `eda_summary` |
| `core.rules` | 规则引擎 | `Rule`, `SingleFeatureRuleMiner`, `MultiFeatureRuleMiner` |
| `core.financial` | 金融计算 | `fv`, `pv`, `pmt`, `irr`, `npv` |

### 报告模块 (Report)

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| `excel` | Excel报告 | `ExcelWriter`, `dataframe2excel` |
| `report.feature_analyzer` | 特征分析 | `feature_bin_stats`, `auto_feature_analysis` |
| `report.swap_analysis` | 置换风险分析 | `SwapAnalyzer`, `swap_analysis` |
| `report.overdue_predictor` | 逾期预测 | `OverduePredictor` |

### 工具模块 (Utils)

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| `utils` | 工具函数 | `seed_everything`, `germancredit`, `feature_describe` |
| `utils.pandas_extensions` | Pandas扩展 | `df.summary()`, `df.save()`, `df.show()` |

---

## 详细功能说明

### 分箱算法 (16种)

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| `uniform` | 等宽分箱 | 均匀分布数据 |
| `quantile` | 等频分箱 | 偏态分布数据 |
| `tree` | 决策树分箱 | 非线性关系 |
| `chi` | 卡方分箱 | 类别型特征 |
| `best_ks` | 最优KS分箱 | 最大化KS统计量 |
| `best_iv` | 最优IV分箱 | 最大化IV值（推荐） |
| `mdlp` | MDLP分箱 | 信息论方法（默认） |
| `or_tools` | OR-Tools分箱 | 运筹规划优化 |
| `cart` | CART分箱 | 基于CART树 |
| `kmeans` | K-Means分箱 | 聚类方法 |
| `monotonic` | 单调性约束分箱 | 支持U型/倒U型 |
| `genetic` | 遗传算法分箱 | 全局优化 |
| `smooth` | 平滑分箱 | 正则化方法 |
| `kernel_density` | 核密度分箱 | 密度估计 |
| `best_lift` | Best Lift分箱 | 提升度优化 |
| `target_bad_rate` | 目标坏样本率分箱 | 指定坏样本率 |

### 特征筛选器 (20+种)

**过滤法**
- `VarianceSelector` - 方差筛选
- `NullSelector` - 缺失率筛选
- `ModeSelector` - 单一值率筛选
- `CardinalitySelector` - 基数筛选
- `CorrSelector` - 相关性筛选
- `VIFSelector` - VIF多重共线性筛选
- `IVSelector` - IV值筛选
- `LiftSelector` - Lift值筛选
- `PSISelector` - PSI稳定性筛选

**嵌入法/包装法**
- `FeatureImportanceSelector` - 特征重要性筛选
- `NullImportanceSelector` - Null Importance筛选
- `RFESelector` - 递归特征消除
- `SequentialFeatureSelector` - 顺序特征选择
- `StepwiseSelector` - 逐步回归选择
- `BorutaSelector` - Boruta算法

### 编码器 (8种)

- `WOEEncoder` - 证据权重编码（风控核心）
- `TargetEncoder` - 目标编码
- `CountEncoder` - 计数编码
- `OneHotEncoder` - 独热编码
- `OrdinalEncoder` - 序数编码
- `QuantileEncoder` - 分位数编码
- `CatBoostEncoder` - CatBoost编码
- `GBMEncoder` - 梯度提升树编码器

### 损失函数 (7种)

- `FocalLoss` - Focal Loss（处理不平衡数据）
- `WeightedBCELoss` - 加权BCE损失
- `CostSensitiveLoss` - 成本敏感损失
- `BadDebtLoss` - 坏账损失
- `ApprovalRateLoss` - 审批率损失
- `ProfitMaxLoss` - 利润最大化损失

---

## 设计特点

### 1. 统一的基类设计

所有核心模块都遵循抽象基类模式，确保API的一致性：

```python
# 分箱基类
class BaseBinning(BaseEstimator, TransformerMixin, ABC)

# 筛选器基类
class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC)

# 编码器基类
class BaseEncoder(BaseEstimator, TransformerMixin, ABC)

# 模型基类
class BaseRiskModel(BaseEstimator, ClassifierMixin, ABC)
```

### 2. 双API风格支持

所有有监督组件都支持两种API风格：

```python
# sklearn风格
binner = OptimalBinning(method='best_iv', max_n_bins=5)
binner.fit(X_train, y_train)

# scorecardpipeline风格
binner = OptimalBinning(target='target', method='best_iv', max_n_bins=5)
binner.fit(df)  # 自动从df中提取'target'列
```

### 3. 工厂模式

`OptimalBinning` 作为统一入口，支持所有分箱方法：

```python
class OptimalBinning(BaseBinning):
    VALID_METHODS = [
        'uniform', 'quantile', 'tree', 'chi',
        'best_ks', 'best_iv', 'mdlp', 'or_tools',
        'cart', 'kmeans', 'monotonic', 'genetic',
        'smooth', 'kernel_density', 'best_lift', 'target_bad_rate'
    ]
```

### 4. 报告收集器模式

`SelectionReportCollector` 自动收集 Pipeline 中所有筛选器的结果：

```python
collector = SelectionReportCollector()
collector.add_report(selector1, stage_name='粗筛')
collector.add_report(selector2, stage_name='精筛')
summary = collector.get_summary()
```

### 5. 中文输出设计

所有报告和统计表都使用中文列名，便于理解和展示：

```python
# 分箱统计表列名
chinese_columns = [
    '分箱', '分箱标签', '样本总数', '好样本数', '坏样本数',
    '样本占比', '好样本占比', '坏样本占比', '坏样本率',
    '分档WOE值', '分档IV值', '指标IV值',
    'LIFT值', '坏账改善', '累积LIFT值', '累积坏账改善'
]
```

### 6. Pandas扩展机制

自动注册 DataFrame 扩展方法：

```python
import hscredit  # 自动注册 df.summary(), df.save(), df.show() 等扩展方法

# 使用扩展方法
df.summary(y='target')  # 综合特征描述统计
df.save("report.xlsx")  # 保存到Excel
table.show(compact=True)  # 美化展示分箱表
```

### 7. 兼容性设计

与 toad 和 scorecardpipeline 的兼容性：

```python
# 导出规则兼容 toad/scorecardpipeline 格式
rules = binner.export(to_json='binning_rules.json')

# 加载 toad/scorecardpipeline 导出的规则
binner.load('binning_rules.json')
```

---

## 示例代码

详见 `examples/` 目录：

- `01_binning.ipynb` - 分箱算法演示
- `02_feature_selection.ipynb` - 特征筛选演示
- `03_encoding.ipynb` - 编码器演示
- `04_modeling.ipynb` - 建模演示
- `05_eda.ipynb` - 探索性数据分析演示
- `06_viz.ipynb` - 可视化演示

---

## 文档

- [API文档](https://hscredit.readthedocs.io/)
- [用户指南](https://hscredit.readthedocs.io/user_guide/)
- [示例教程](https://hscredit.readthedocs.io/tutorials/)

---

## 贡献

欢迎提交Issue和PR！

```bash
# 克隆仓库
git clone https://github.com/hscredit/hscredit.git

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black hscredit/
isort hscredit/

# 类型检查
mypy hscredit/
```

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 联系我们

- GitHub: [https://github.com/hscredit/hscredit](https://github.com/hscredit/hscredit)
- Email: hscredit@hengshucredit.com

---

## 致谢

本项目参考了以下优秀的开源项目：

- [optbinning](https://github.com/guillermo-navas-palencia/optbinning) - 最优分箱
- [toad](https://github.com/amphibian-dev/toad) - 风控特征工程
- [scorecardpy](https://github.com/ShichenXie/scorecardpy) - 评分卡建模
- [scorecardpipeline](https://github.com/) - 评分卡流程

---

<p align="center">
  <b>HSCredit</b> - 鉴真伪，斟信用，衡风险，枢定策 —— 智能风控，一站掌控
</p>
