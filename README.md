# hscredit

<p align="center">
  <img src="https://hengshucredit.com/images/hengshucredit_animated.svg" alt="衡枢真信" width="180">
</p>

<p align="center">
  <a href="https://pypi.org/project/hscredit/"><img src="https://img.shields.io/pypi/v/hscredit?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

`hscredit` 面向金融信贷风控策略分析与评分模型研发场景，提供符合 `sklearn Pipeline` 范式的量化建模工具箱，覆盖数据探索、特征筛选、分箱编码、有效性评估、模型训练、参数调优与报告生成，支持从策略验证到模型落地的端到端风险决策流程。

鉴真伪，斟信用，衡风险，枢定策。

## 定位

信贷风控建模通常不是单一算法问题，而是由数据质量、变量有效性、分箱稳定性、模型排序能力、审批策略、监控报表和上线交付共同决定的工程体系。`hscredit` 的目标是在一个统一的 Python 包中沉淀这些高频工作流，减少策略人员和模型人员在 `toad`、`optbinning`、`scorecardpy`、`scorecardpipeline`、Excel 脚本和自研工具之间反复切换的成本。

`hscredit` 更适合以下场景：

| 场景 | 典型问题 | hscredit 对应能力 |
|:---|:---|:---|
| 贷前信用评分 | 如何从原始申请数据构建评分卡 | EDA、分箱、WOE、IV/VIF/PSI 筛选、逻辑回归、ScoreCard |
| 策略规则分析 | 如何量化规则覆盖率、坏率、Lift 和跨规则效果 | Rule、规则挖掘、规则指标、Swap 分析 |
| 机器学习风控模型 | 如何训练并比较 LR、随机森林、GBDT、XGBoost、LightGBM、CatBoost 等模型 | 统一模型接口、Boosting 模型、调参、模型评估 |
| 变量有效性评估 | 如何判断变量是否有区分度、稳定性和业务可解释性 | IV、KS、Lift、PSI、CSI、单变量 AUC、稳定性分析 |
| 贷后表现分析 | 如何分析 Vintage、Roll Rate、逾期率趋势和 MOB 表现 | EDA strategy/vintage、OverduePredictor、趋势图表 |
| 模型监控 | 如何监控客群迁移、分数漂移、变量漂移和策略衰减 | PSI/CSI、population drift、score drift、稳定性图表 |
| 业务交付 | 如何将建模结果沉淀为中文报表和 Excel 交付物 | ExcelWriter、dataframe2excel、auto_model_report、pandas 扩展 |

## 核心特性

- 面向中文信贷风控业务，输出列名、错误提示和报告结构贴近国内策略与模型团队。
- 核心建模组件遵循 `fit` / `transform` / `predict` 风格，便于与 `sklearn Pipeline`、调参和模型评估流程集成。
- 支持 `X, y` 建模风格，也支持 DataFrame 内通过 `target` 指定目标列的风控建模习惯。
- 分箱、编码、筛选、评分卡、树模型、规则挖掘、稳定性分析和 Excel 报告在同一包内协同工作。
- 可选依赖按能力分组，基础安装保持轻量，Boosting、深度学习、调参、解释和 PMML 能力按需安装。

## 安装

```bash
pip install hscredit
```

可选依赖：

| 安装命令 | 适用场景 |
|:---|:---|
| `pip install hscredit[boost]` | 安装 XGBoost、LightGBM、CatBoost、NGBoost 等 Boosting 模型依赖 |
| `pip install hscredit[net]` | 安装 PyTorch 与 TabNet 相关依赖 |
| `pip install hscredit[tune]` | 安装 Optuna 调参与可视化面板依赖 |
| `pip install hscredit[explain]` | 安装 SHAP 模型解释依赖 |
| `pip install hscredit[pmml]` | 安装 PMML 相关依赖 |
| `pip install hscredit[dev]` | 安装测试、格式化、打包和发布校验工具 |
| `pip install hscredit[docs]` | 安装文档构建依赖 |
| `pip install hscredit[all]` | 安装全部可选能力 |

源码开发：

```bash
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e ".[dev]"
```

构建与发布前检查：

```bash
python -m build
python -m twine check dist/*
```

项目元数据、运行依赖和可选依赖统一维护在 `pyproject.toml`；`setup.py` 仅保留为旧版安装工具兼容入口。

## 快速开始

### 数据探索

```python
import hscredit
import hscredit.core.eda as eda

summary = eda.data_info(df)
iv_result = eda.batch_iv_analysis(df, features=["age", "income"], target="fpd30")
trend = eda.bad_rate_trend(df, target_col="fpd30", date_col="apply_month")

# 导入 hscredit 后注册 pandas 扩展
df.summary(y="fpd30")
```

### 分箱与 WOE 编码

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder

binner = OptimalBinning(method="best_iv", max_n_bins=5, target="fpd30")
binner.fit(train_df)
train_bins = binner.transform(train_df)

encoder = WOEEncoder(target="fpd30")
encoder.fit(train_bins)
train_woe = encoder.transform(train_bins)
```

### 特征筛选

```python
from hscredit.core.selectors import IVSelector, VIFSelector, CompositeFeatureSelector

selector = CompositeFeatureSelector([
    ("iv", IVSelector(threshold=0.02)),
    ("vif", VIFSelector(threshold=10.0)),
])

selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)
report = selector.get_selection_report()
```

### 评分卡建模

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard

binner = OptimalBinning(method="best_iv", max_n_bins=5)
binner.fit(X_train, y_train)
X_train_woe = binner.transform(X_train, metric="woe")

scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750, binner=binner)
scorecard.fit(X_train_woe, y_train)
scores = scorecard.predict(X_test)
```

### Boosting 风控模型

```python
from hscredit.core.models import XGBoostRiskModel, LightGBMRiskModel, CatBoostRiskModel

models = {
    "xgboost": XGBoostRiskModel(max_depth=4, n_estimators=200),
    "lightgbm": LightGBMRiskModel(num_leaves=31, n_estimators=200),
    "catboost": CatBoostRiskModel(depth=5, iterations=200),
}

for name, model in models.items():
    if model is None:
        continue
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(name, metrics)
```

### 规则挖掘

```python
from hscredit.report.mining import SingleFeatureRuleMiner, MultiFeatureRuleMiner, TreeRuleExtractor

single_miner = SingleFeatureRuleMiner(target="fpd30", method="optimal_iv", max_n_bins=5)
single_miner.fit(train_df)
single_rules = single_miner.get_top_rules(top_n=10, metric="lift")

cross_miner = MultiFeatureRuleMiner(target="fpd30", method="chi2", max_n_bins=4)
cross_miner.fit(train_df)
cross_rules = cross_miner.get_cross_rules("age", "income", top_n=10)

extractor = TreeRuleExtractor(algorithm="rf", max_depth=5)
extractor.fit(X_train, y_train)
tree_rules = extractor.extract_rules(top_n=20, metric="confidence")
```

### 模型报告与 Excel 输出

```python
from hscredit.report import auto_model_report

report_path = auto_model_report(
    model,
    X_test,
    y_test,
    save_path="模型评估报告.xlsx",
)
```

```python
import hscredit

bin_table.save("分箱结果.xlsx", title="年龄分箱")
bin_table.show()
```

## 功能地图

### 数据探索与业务分析

| 模块 | 能力 |
|:---|:---|
| `hscredit.core.eda.overview` | 数据概览、缺失分析、字段摘要、数据质量报告、客群稳定性监控 |
| `hscredit.core.eda.target` | 目标分布、整体坏率、分维度坏率、坏率趋势、样本分布 |
| `hscredit.core.eda.feature` | 类型推断、数值/类别分布、异常值、稀有类别、集中度分析 |
| `hscredit.core.eda.relationship` | IV、WOE、分箱坏率、单调性、单变量 AUC、特征重要性排序 |
| `hscredit.core.eda.stability` | PSI、CSI、跨期 PSI、特征漂移、分数漂移 |
| `hscredit.core.eda.population` | 客群画像、客群迁移、分群漂移、跨客群变量有效性 |
| `hscredit.core.eda.strategy` | 审批率/坏率权衡、策略仿真、Vintage 汇总、Roll Rate、标签泄漏检查 |
| `hscredit.core.eda.vintage` | Vintage 分析、账龄表现汇总、迁徙率分析 |

### 分箱、编码与筛选

| 模块 | 已实现能力 |
|:---|:---|
| `hscredit.core.binning` | 等宽、等频、决策树、CART、卡方、Best IV、Best KS、Best Lift、MDLP、OR-Tools、CP-SAT、KMeans、单调约束、遗传算法、平滑、核密度、目标坏率、二维最优分箱等 |
| `hscredit.core.encoders` | WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、Cardinality、GBM 编码 |
| `hscredit.core.selectors` | 缺失率、众数率、方差、相关性、VIF、IV、Lift、PSI、基数、类型、正则、模型重要性、零重要性、RFE、序列选择、逐步回归、Boruta、互信息、卡方、F 检验、稳定性感知、评分卡组合筛选 |

### 模型、指标与报告

| 模块 | 已实现能力 |
|:---|:---|
| `hscredit.core.models.classical` | LogisticRegression、RandomForest、ExtraTrees、GradientBoosting 风控模型 |
| `hscredit.core.models.boosting` | XGBoost、LightGBM、CatBoost、NGBoost 风控模型，可选依赖安装 |
| `hscredit.core.models.scorecard` | ScoreCard、RoundScoreCard、评分转换、分数漂移校准 |
| `hscredit.core.models.losses` | Focal、非对称 Focal、加权 BCE、成本敏感、坏账、审批率、利润最大化、排序、KS 聚焦、Top-K 坏样本捕获、金额加权等风控损失 |
| `hscredit.core.models.tuning` | 基于 Optuna 的模型调参能力 |
| `hscredit.core.models.evaluation` | 模型评估、概率校准、解释性分析 |
| `hscredit.core.metrics` | KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、回归指标、分箱统计等 |
| `hscredit.report` | 特征分析、规则分析、Swap 分析、逾期预测、模型报告、人群漂移、Excel 输出 |

### 规则、策略与可视化

| 模块 | 已实现能力 |
|:---|:---|
| `hscredit.core.rules` | Rule 表达式、变量解析、表达式优化与美化 |
| `hscredit.core.models.rules` | RuleSet、RulesClassifier、规则组合分类器 |
| `hscredit.report.mining` | 单特征规则、多特征交叉规则、多标签规则、树规则提取、手工树分析、规则指标、树可视化 |
| `hscredit.report.swap_analysis` | 新旧策略置换、通过/拒绝交叉矩阵、风险迁移分析 |
| `hscredit.core.viz` | 分箱趋势、模型评估、评分分布、策略曲线、变量稳定性、客群漂移、树图等图表函数 |
| `hscredit.excel` | `ExcelWriter`、`dataframe2excel`、格式化 Excel 输出 |

## 与同类库的关系

`hscredit` 不试图用单个算法替代所有成熟库，而是面向国内信贷风控交付链路做统一封装与场景增强。

| 库 | 优势 | 局限 | hscredit 的定位 |
|:---|:---|:---|:---|
| `toad` | API 简洁，评分卡主流程成熟 | 报告、策略分析和高级分箱能力有限 | 学习其易用性，补充更完整的中文报告和业务分析 |
| `optbinning` | 数学规划分箱专业，支持高级分箱与 XAI | 更聚焦分箱与评分卡，不是完整策略分析工具链 | 借鉴其分箱质量和解释能力，结合风控全流程使用 |
| `scorecardpipeline` | Pipeline 集成和报表交付能力强 | 底层依赖较多，长期维护方向已转向 hscredit | 继承 Pipeline 和报告思想，减少多库依赖并扩展原生能力 |
| `scorecardpy` | R/Python 评分卡基础流程易上手 | 工程化、可扩展性和中文风控报告较弱 | 提供更贴近 Python 工程体系的建模组件 |

## 项目结构

```text
hscredit/
├── core/
│   ├── binning/              # 分箱算法与统一分箱接口
│   ├── encoders/             # WOE、Target、Count、OneHot 等编码器
│   ├── selectors/            # 多维度特征筛选器与组合筛选报告
│   ├── models/               # 风控模型、评分卡、损失函数、调参、评估
│   ├── metrics/              # 分类、回归、特征、稳定性、金融风控指标
│   ├── eda/                  # 数据探索、客群、策略、Vintage、稳定性分析
│   ├── rules/                # 规则表达式与规则工具
│   ├── financial/            # FV、PV、PMT、NPV、IRR 等金融计算
│   ├── feature_engineering/  # 表达式特征衍生
│   └── viz/                  # 风控建模图表
├── report/
│   ├── mining/               # 规则挖掘与树规则提取
│   ├── model_report.py       # 模型报告
│   ├── swap_analysis.py      # 策略置换分析
│   ├── overdue_predictor.py  # 逾期预测
│   └── population_drift.py   # 客群漂移监控
├── excel/                    # Excel 写入与格式化
└── utils/                    # pandas 扩展、IO、日志、随机种子、数据集
```

## 适用人群

- 银行、消费金融、互联网信贷机构的风控策略人员。
- 负责评分卡、机器学习风控模型和贷后监控的模型研发人员。
- 金融科技、三方数据和咨询团队中的风险建模工程师。
- 金融工程、信用风险和数据科学方向的研究与教学人员。

## 开发命令

```bash
# 格式化
make format

# 静态检查
make lint

# 类型检查
make type-check

# 测试
make test

# 全量检查
make check

# 构建发布包
make build
```

也可以直接使用：

```bash
pytest tests/ -v
python -m build
python -m twine check dist/*
```

## 文档与规划

- 打包说明：`docs/PACKAGING.md`
- 迭代规划：`docs/ROADMAP.md`
- 示例数据和 Notebook：`examples/`

## 许可证

MIT License。可按许可证条款用于商业和非商业场景。

## 社区

微信公众号：衡枢风控

公众号 ID：`hengshucredit-com`

关注公众号，回复 `入群` 可加入 hscredit 技术交流群。
