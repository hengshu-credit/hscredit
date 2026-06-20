# hscredit

<p align="center">
  <img src="https://hengshucredit.com/images/hengshucredit_animated.svg" alt="衡枢真信" width="180">
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Version](https://img.shields.io/badge/Version-0.1.0-orange?style=flat-square)
<a href="https://pypi.org/project/hscredit/"><img src="https://img.shields.io/pypi/v/hscredit?style=flat-square" alt="PyPI"></a>
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

<h3 align="center">

🔍 鉴真伪 · 📊 斟信用 · ⚖️ 衡风险 · 🎯 枢定策

</h3>

<br>

`hscredit` 是面向金融信贷行业的量化分析工具箱，面向风控策略分析人员与模型人员提供全流程的信贷风控能力，包括数据探索、变量评估、分箱编码、特征筛选、评分卡建模、机器学习风控模型、策略规则分析、模型监控和报告交付。

这些能力覆盖模型研发与策略分析两条核心业务链路：

<p align="center">
  <img src="docs/assets/models.png" alt="hscredit 风控建模链路" width="100%">
</p>

<p align="center">
  <img src="docs/assets/celue.png" alt="hscredit 风控策略分析链路" width="100%">
</p>

## 为什么选择 hscredit？

信贷风控工作通常不是单一建模问题，而是策略、数据、变量、模型、规则、监控和报告共同构成的决策流程。`hscredit` 希望把这些工作沉淀到一个工具箱中，减少在多个建模库、分析脚本和 Excel 模板之间反复切换的成本。

| 风控工作 | 能力规模 | hscredit 能做什么 | 对策略和模型人员的价值 |
|:---|:---:|:---|:---|
| 数据分析 | **57 种 EDA** | 数据质量、目标分布、坏率趋势、客群画像、客群迁移、Vintage、Roll Rate、策略仿真 | 快速判断样本是否可建模、标签是否合理、客群是否稳定 |
| 变量分箱 | **18 种分箱器** | 等频、等宽、卡方、树分箱、CART、Best IV、Best KS、Best Lift、MDLP、单调分箱、遗传算法、二维分箱等 | 支持评分卡建模中的变量离散化、坏率趋势分析和单调性控制 |
| 特征编码 | **9 种编码器** | WOE、目标编码、频数编码、独热编码、序数编码、分位数编码、CatBoost/GBM 编码 | 同时服务评分卡、树模型和高基数类别变量处理 |
| 特征筛选 | **23 种筛选器** | 缺失率、众数率、基数、方差、相关性、VIF、IV、Lift、PSI、模型重要性、逐步回归、Boruta、组合筛选 | 从区分度、稳定性、共线性、模型贡献和业务解释多个角度筛选变量 |
| 风控指标 | **43 种指标** | KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、分箱统计、分类和回归指标 | 建模评估、变量评估、策略评估和模型监控使用同一套指标口径 |
| 模型训练 | **36 个建模组件** | 逻辑回归、评分卡、随机森林、GBDT、XGBoost、LightGBM、CatBoost、NGBoost、规则模型、风控损失函数、调参器 | 覆盖传统评分卡、机器学习风控模型和业务目标导向建模 |
| 可视化分析 | **46 种图表** | 分箱趋势、KS/ROC/PR/Lift/Gain、评分分布、策略阈值、Vintage、变量稳定性、客群漂移、树图 | 把分析结论转换为模型评审、策略沟通和业务复盘材料 |
| 报告交付 | **28 种报告工具** | 特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比、Excel 输出 | 减少手工整理报表，将建模和策略结果沉淀为中文交付物 |
| Excel 报表 | **10+ 种 Excel 操作** | 写入数据表、插入图片、插入超链接、设置条件格式、单元格样式、数字格式、冻结窗格、列宽调整、Sheet 复制、模板化输出 | 直接生成可用于评审、汇报和归档的风控分析报告 |
| 规则挖掘 | **8 种挖掘工具** | 单变量规则、多变量交叉规则、多标签规则、树规则提取、手工树分析、规则指标、树可视化 | 从数据和模型中发现可解释的风险模式，辅助策略规则设计 |
| Rule 规则体系 | **8 类规则能力** | Rule 表达式、任意层级嵌套、与/或/非逻辑运算、规则变量解析、规则美化、规则命中评估、规则分析报表、规则集分析、SWAP 置换分析 | 将零散策略条件沉淀为可组合、可评估、可追踪、可复用的规则资产 |

## 适用场景

| 场景 | 你通常关心的问题 | hscredit 对应能力 |
|:---|:---|:---|
| 贷前评分卡建模 | 哪些变量有效、如何分箱、评分卡是否稳定 | IV/WOE、Best IV/KS 分箱、单调分箱、VIF/PSI 筛选、逻辑回归、ScoreCard、模型报告 |
| 风控策略规则分析 | 哪些规则能拦截坏客户、规则覆盖是否重叠、新旧策略替换风险如何 | 单变量规则、多变量规则、树规则提取、规则指标、规则集分析、Swap 分析 |
| 机器学习风控模型 | 如何训练树模型、比较模型效果、优化坏账和审批目标 | 随机森林、GBDT、XGBoost、LightGBM、CatBoost、NGBoost、风控损失函数、调参、模型对比 |
| 变量有效性评估 | 变量是否有区分度、是否稳定、是否共线、是否能解释业务 | IV、KS、Lift、PSI、CSI、VIF、相关性、单变量 AUC、稳定性分析 |
| 贷后表现分析 | 逾期表现如何爬坡、不同账龄风险如何变化 | Vintage、Roll Rate、MOB 逾期预测、坏率趋势、账龄表现汇总 |
| 模型监控 | 客群是否漂移、分数是否漂移、变量是否衰减 | PSI/CSI、客群迁移、分数漂移、变量漂移、稳定性图表 |
| 建模报告交付 | 如何快速形成模型评审和业务汇报材料 | ExcelWriter、模型报告、特征报告、规则报告、图表输出、pandas 扩展 |

## 核心优势

- 面向金融信贷业务，而不是通用机器学习演示场景。
- 覆盖策略分析与模型研发的完整流程，减少多库拼装和重复造轮子。
- 输出列名、报告结构和图表表达更贴近中文风控团队的日常交付习惯。
- 同时支持评分卡、机器学习模型、规则模型和策略分析。
- 内置 IV、WOE、KS、Lift、PSI、CSI、Vintage、Roll Rate、Swap 等高频风控方法。
- 基础安装保持轻量，Boosting、深度学习、调参、解释、PMML 等能力按需安装。

## 安装

```bash
pip install hscredit
```

可选能力按需安装：

| 安装命令 | 适用场景 |
|:---|:---|
| `pip install hscredit[boost]` | XGBoost、LightGBM、CatBoost、NGBoost 等 Boosting 模型 |
| `pip install hscredit[net]` | PyTorch 与 TabNet 深度学习模型 |
| `pip install hscredit[tune]` | Optuna 参数调优和调参看板 |
| `pip install hscredit[explain]` | SHAP 模型解释 |
| `pip install hscredit[pmml]` | PMML 相关能力 |
| `pip install hscredit[all]` | 安装全部可选能力 |

## 典型工作流

### 1. 数据和变量初筛

```python
import hscredit
import hscredit.core.eda as eda

# 数据质量、缺失、字段类型、样本分布
summary = eda.data_info(df)

# 批量评估变量 IV 和坏率趋势
iv_result = eda.batch_iv_analysis(df, features=["age", "income"], target="fpd30")
trend = eda.bad_rate_trend(df, target_col="fpd30", date_col="apply_month")

# pandas 扩展方法，适合快速查看数据摘要
df.summary(y="fpd30")
```

### 2. 分箱、编码和变量筛选

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.selectors import IVSelector, VIFSelector, CompositeFeatureSelector

binner = OptimalBinning(method="best_iv", max_n_bins=5, target="fpd30")
binner.fit(train_df)
train_bins = binner.transform(train_df)

encoder = WOEEncoder(target="fpd30")
encoder.fit(train_bins)
train_woe = encoder.transform(train_bins)

selector = CompositeFeatureSelector([
    ("iv", IVSelector(threshold=0.02)),
    ("vif", VIFSelector(threshold=10.0)),
])
selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)
```

### 3. 评分卡建模

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

### 4. 机器学习风控模型

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
    print(name, model.evaluate(X_test, y_test))
```

### 5. 策略规则挖掘

```python
from hscredit.report.mining import SingleFeatureRuleMiner, MultiFeatureRuleMiner, TreeRuleExtractor

single_miner = SingleFeatureRuleMiner(target="fpd30", method="best_iv", max_n_bins=5)
single_miner.fit(train_df)
single_rules = single_miner.get_top_rules(top_n=10, metric="lift")

cross_miner = MultiFeatureRuleMiner(target="fpd30", method="chi", max_n_bins=4)
cross_miner.fit(train_df)
cross_rules = cross_miner.get_cross_rules("age", "income", top_n=10)

extractor = TreeRuleExtractor(algorithm="rf", max_depth=5)
extractor.fit(X_train, y_train)
tree_rules = extractor.extract_rules(top_n=20, metric="confidence")
```

### 6. 模型报告和 Excel 交付

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

## 功能总览

### 数据探索与策略分析

| 能力 | 说明 |
|:---|:---|
| 数据质量分析 | 数据概览、缺失分析、字段摘要、数据质量报告 |
| 目标变量分析 | 目标分布、整体坏率、分维度坏率、坏率趋势、样本分布 |
| 变量探索 | 数值/类别分布、异常值、稀有类别、集中度分析 |
| 变量有效性 | IV、WOE、分箱坏率、单调性、单变量 AUC、特征重要性排序 |
| 稳定性分析 | PSI、CSI、跨期 PSI、特征漂移、分数漂移 |
| 客群分析 | 客群画像、客群迁移、分群漂移、跨客群变量有效性 |
| 策略分析 | 审批率/坏率权衡、策略仿真、Vintage 汇总、Roll Rate、标签泄漏检查 |

### 分箱、编码与筛选

| 能力 | 说明 |
|:---|:---|
| 分箱方法 | 等宽、等频、树分箱、CART、卡方、Best IV、Best KS、Best Lift、MDLP、OR-Tools、CP-SAT、KMeans、单调约束、遗传算法、平滑、核密度、目标坏率、二维最优分箱 |
| 编码方法 | WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、Cardinality、GBM 编码 |
| 特征筛选 | 缺失率、众数率、方差、相关性、VIF、IV、Lift、PSI、基数、类型、正则、模型重要性、零重要性、RFE、序列选择、逐步回归、Boruta、互信息、卡方、F 检验、稳定性感知、评分卡组合筛选 |
| 特征衍生 | NumExprDerive 表达式引擎（基于 numexpr），支持 where/sin/cos/abs 等函数与条件逻辑的批量特征衍生 |

### 建模、评估与监控

| 能力 | 说明 |
|:---|:---|
| 评分卡模型 | 逻辑回归、ScoreCard、RoundScoreCard、评分转换、分数漂移校准 |
| 机器学习模型 | RandomForest、ExtraTrees、GradientBoosting、XGBoost、LightGBM、CatBoost、NGBoost |
| 风控损失函数 | Focal、非对称 Focal、加权 BCE、成本敏感、坏账、审批率、利润最大化、排序、KS 聚焦、Top-K 坏样本捕获、金额加权等 |
| 调参与解释 | Optuna 调参、模型评估、概率校准、解释性分析、SHAP 可选支持 |
| 风控指标 | KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、回归指标、分箱统计 |
| 金融计算 | FV/PV/PMT/NPER/IPMT/PPMT/RATE 现值终值年金计算，NPV/IRR/MIRR 净现值与收益率计算 |

### 规则、报告与交付

| 能力 | 说明 |
|:---|:---|
| 规则分析 | Rule 表达式、规则变量解析、规则美化、规则组合、规则集分类器、多标签规则分析 |
| 规则挖掘 | 单特征规则、多特征交叉规则、多标签规则、树规则提取、手工树分析、规则指标 |
| 策略置换 | 新旧策略置换、通过/拒绝交叉矩阵、风险迁移分析 |
| 可视化 | 分箱趋势、模型评估、评分分布、策略曲线、变量稳定性、客群漂移、树图 |
| Excel 报告 | 特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比、格式化 Excel 输出 |

## 与同类工具的关系

`hscredit` 的目标是逐步替代风控策略分析和模型研发中对多套工具库的拼装依赖，将评分卡建模、最优分箱、特征筛选、机器学习风控模型、规则挖掘、稳定性监控和报告交付沉淀到统一的 Python 工具箱中。

| 工具 | 常见用途 | hscredit 的替代方向 |
|:---|:---|:---|
| `toad` | 快速完成评分卡分箱、WOE、筛选和评分卡建模 | 覆盖评分卡主流程，并补充中文报告、策略分析、监控和更多分箱筛选方法 |
| `optbinning` | 专业最优分箱和评分卡相关分析 | 吸收最优分箱、质量评估和解释性思想，并纳入完整风控工作流 |
| `scorecardpipeline` | Pipeline 风格评分卡流程、报告和部署交付 | 承接其评分卡流程和报告经验，减少多库依赖并扩展原生能力 |
| `scorecardpy` | 快速评分卡建模 | 提供更贴近 Python 建模工作流和中文风控交付的评分卡能力 |
| Excel 脚本和内部模板 | 手工统计变量、规则和模型报告 | 用统一的数据分析、图表和 Excel 输出能力替代重复脚本 |

## 适用人群

- 银行、消费金融、互联网信贷机构的风控策略人员。
- 负责评分卡、机器学习风控模型和贷后监控的模型人员。
- 金融科技、三方数据和咨询团队中的风险分析与模型交付人员。
- 金融工程、信用风险和数据科学方向的研究与教学人员。

## 文档与规划

- 迭代规划：`docs/ROADMAP.md`
- 打包说明：`docs/PACKAGING.md`
- 示例数据和 Notebook：`examples/`

## 维护与开发

如果需要参与开发或本地验证，可以使用以下命令：

```bash
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e ".[dev]"
pytest tests/ -v
```

构建发布包：

```bash
python -m build
python -m twine check dist/*
```

项目依赖和可选能力统一维护在 `pyproject.toml`；`setup.py` 仅保留为旧版安装工具兼容入口。

## 许可证

MIT License。可按许可证条款用于商业和非商业场景。

## 社区

微信公众号：衡枢风控

公众号 ID：`hengshucredit-com`

关注公众号，回复 `入群` 可加入 hscredit 技术交流群。
