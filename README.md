<p align="center">
  <img src="https://hengshucredit.com/images/hengshucredit_animated.svg" alt="衡枢真信" width="200">
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-0.1.0-orange?style=flat-square)
[![Downloads](https://img.shields.io/pypi/dm/hscredit?style=flat-square)](https://pypi.org/project/hscredit/)

</div>

<h1 align="center">
  <span style="color:#6366f1">▌</span> hscredit <span style="color:#6366f1">▌</span>
</h1>

<h3 align="center">

🔍 鉴真伪 · 📊 斟信用 · ⚖️ 衡风险 · 🎯 枢定策

</h3>

<p align="center">
  专为风控策略师、建模工程师打造的<strong>一站式信用评分建模平台</strong><br>
  分箱 · 编码 · 筛选 · 建模 · 评估 · 报告 —— 全流程覆盖
</p>

---

## ✨ 为什么选择 hscredit？

| 你在做的事情 | hscredit 帮你做到 |
|:---|:---|
| 分箱调参调到怀疑人生 | 19 种分箱算法，遗传算法自动寻优 |
| 特征筛选没有标准答案 | 20+ 筛选器，组合筛选 + 阶段化报告 |
| 评分卡上线流程冗长 | sklearn Pipeline 全链路串联 |
| 规则挖掘靠人工经验 | 4 大规则挖掘器，自动生成可解释规则 |
| 报表格式手动调一整天 | 一键生成专业 Excel 报告 |
| 催收策略缺乏量化依据 | 加权逾期预测，精准预估各期逾期率 |
| python版本兼容很痛苦 | 支持最新python3.14，环境不折腾，小白好上手 |

> 💡 **一个库，覆盖风控建模全生命周期。告别 toad / scorecardpy / optbinning / scorecardpipeline 混用的时代——hscredit 全搞定。**

---

## 🚀 3 分钟安装

```bash
# 标准安装
pip install hscredit

# 完整安装（含全部依赖）
pip install hscredit[all]
```

---

## 💡 一行代码上手

### 最优分箱 —— 遗传算法自动寻优

```python
from hscredit.core.binning import GeneticBinning

# 全局最优分箱，支持单调性约束
binner = GeneticBinning(max_n_bins=5, monotonic=True, target='y')
binner.fit(df)
print(binner.binning_table)
```

### 全流程 Pipeline —— sklearn 风格无缝串联

```python
from sklearn.pipeline import Pipeline
from hscredit.core.selectors import IVSelector, VIFSelector
from hscredit.core.encoders import WOEEncoder
from hscredit.core.models.classical import LogisticRegression
from hscredit.core.models.scorecard import ScoreCard

pipeline = Pipeline([
    ('iv_filter', IVSelector(threshold=0.02)),   # 粗筛：IV > 0.02
    ('vif_filter', VIFSelector(threshold=10)),   # 精筛：VIF < 10
    ('woe', WOEEncoder()),                        # WOE 编码
    ('lr', LogisticRegression()),                 # 逻辑回归
    ('scorecard', ScoreCard(pdo=20, base=600)),   # 转换为标准评分卡
])

pipeline.fit(df, y)
scores = pipeline.transform(new_applicants)
```

### 规则挖掘 —— 从数据中发现可解释规则

```python
from hscredit.report import SingleFeatureRuleMiner, MultiFeatureRuleMiner, TreeRuleExtractor

# 单特征规则：找出每个变量的最优分箱规则
miner = SingleFeatureRuleMiner(target='ISBAD', method='optimal_iv', max_n_bins=5)
rules = miner.get_top_rules(top_n=10, metric='lift')

# 双特征交叉规则：发现特征组合的联合效应
cross_miner = MultiFeatureRuleMiner(target='ISBAD', method='chi2', max_n_bins=4)
rules = cross_miner.get_cross_rules('age', 'income', top_n=10)

# 树模型规则提取：从 XGBoost/LightGBM 中提取可解释 IF-THEN 规则
extractor = TreeRuleExtractor(algorithm='dt', max_depth=5)
extractor.fit(X_train, y_train)
rules = extractor.extract_rules(top_n=20, metric='confidence')
```

### 逾期预测 —— 加权估算各账龄逾期率

```python
from hscredit.report import OverduePredictor

predictor = OverduePredictor(feature='score', target='IS_OVERDUE')
predictor.fit(train_df)  # 支持有无逾期的两种拟合模式

# 按账龄分段预估逾期率
report = predictor.get_report()
print(report)
```

### Pandas 魔法扩展 —— 导入即生效

```python
import hscredit  # 导入即注册 DataFrame 扩展方法

df.summary(y='target')            # 全维度数据摘要
df.save('分箱结果.xlsx', title='年龄分箱')  # 带格式导出 Excel
bin_table.show()                 # 格式化表格输出
```

### 一键模型报告

```python
from hscredit.report import auto_model_report

auto_model_report(model, X_test, y_test, save_path='模型报告.xlsx')
# 报告包含：KS / ROC / LIFT / PSI / 分箱表 / 变量系数 / 特征重要性
```

---

## 🧮 算法全家福

<details>
<summary><b>🧮 分箱算法 (19 种)</b></summary>

| 算法 | 说明 |
|------|------|
| **GeneticBinning** | 遗传算法全局最优分箱，支持单调性约束 |
| **OptimalBinning** | 运筹优化（OR-Tools）分箱，IV/KS 最优 |
| **BestIVBinning / BestKSBinning / BestLiftBinning** | 最优 IV / KS / LIFT 分箱 |
| **MDLPBinning** | 基于最小描述长度原理的信息论分箱 |
| **ChiMergeBinning** | 卡方分箱，相近坏率箱自动合并 |
| **CartBinning** | CART 决策树分箱 |
| **MonotonicBinning** | 强制单调性约束分箱 |
| **TreeBinning** | 决策树引导分箱 |
| **QuantileBinning** | 等频分箱 |
| **UniformBinning** | 等宽分箱 |
| **KMeansBinning** | KMeans 聚类分箱 |
| **KernelDensityBinning** | 核密度估计分箱 |
| **SmoothBinning** | 平滑分箱（减少稀疏箱波动） |
| **TargetBadRateBinning** | 目标坏样本率分箱 |
| **OrBinning** | OR-Tools 运筹优化分箱 |

</details>

<details>
<summary><b>🔍 特征筛选器 (21 种)</b></summary>

| 筛选器 | 说明 |
|------|------|
| **IVSelector** | IV 值筛选，最常用的风控特征筛选指标 |
| **PSISelector** | PSI 稳定性筛选，监控特征跨时间稳定性 |
| **CorrSelector** | 相关系数筛选，避免多重共线性 |
| **VIFSelector** | 方差膨胀因子筛选 |
| **Chi2Selector** | 卡方独立性检验筛选 |
| **LiftSelector** | LIFT 提升度筛选 |
| **VarianceSelector** | 方差筛选，剔除无区分度特征 |
| **NullSelector** | 缺失率筛选 |
| **ModeSelector** | 众数频率筛选，剔除单一值主导特征 |
| **CardinalitySelector** | 基数筛选，高基数类别特征处理 |
| **ImportanceSelector** | 模型特征重要性筛选 |
| **NullImportanceSelector** | 零重要性筛选（Permutation） |
| **RFESelector** | 递归特征消除（RFE） |
| **BorutaSelector** | Boruta 全相关特征选择 |
| **MutualInfoSelector** | 互信息筛选 |
| **FTestSelector** | F 统计量筛选 |
| **StepwiseSelector** | 逐步回归筛选 |
| **SequentialFeatureSelector** | 序列前向/后向选择 |
| **StabilityAwareSelector** | 稳定性感知筛选 |
| **TypeSelector / RegexSelector** | 按类型/正则表达式筛选 |

</details>

<details>
<summary><b>🔤 特征编码器 (9 种)</b></summary>

| 编码器 | 说明 |
|------|------|
| **WOEEncoder** | WOE 证据权重编码，评分卡标准编码 |
| **TargetEncoder** | 目标编码，带贝叶斯平滑 |
| **CountEncoder** | 频数编码，高基数友好 |
| **OneHotEncoder** | 独热编码 |
| **OrdinalEncoder** | 有序编码 |
| **QuantileEncoder** | 分位数编码 |
| **GBMEncoder** | GBM 编码 |
| **CatBoostEncoder** | CatBoost 目标编码 |
| **CardinalityEncoder** | 基数编码 |

</details>

<details>
<summary><b>⚡ 自定义损失函数 (14 种)</b></summary>

| 损失函数 | 说明 |
|------|------|
| **FocalLoss** | Focal Loss，处理类别不平衡 |
| **BalancedFocalLoss** | 平衡版 Focal Loss |
| **AsymmetricFocalLoss** | 非对称 Focal Loss，误判成本差异化 |
| **WeightedLoss** | 样本权重损失 |
| **KSFocusedLoss** | KS 聚焦损失，直接优化排序能力 |
| **RankingLoss** | 排序损失 |
| **RankingAUCLoss** | AUC-proxy 排序损失 |
| **RiskLoss** | 风险损失，支持坏账率约束 |
| **AmountWeightedLoss** | 金额加权损失 |
| **TopKBadCaptureLoss** | Top-K 坏账捕获损失 |
| **ExpectedProfitLoss** | 期望利润损失 |

</details>

---

## 📊 可视化 (45+ 种图表)

| 类别 | 图表 |
|:---|:---|
| **分箱分析** | `bin_plot`, `bin_trend_plot`, `batch_bin_trend_plot`, `bin_overdues_plot` |
| **模型评估** | `roc_plot`, `pr_plot`, `lift_plot`, `gain_plot`, `ks_plot`, `calibration_plot` |
| **评分分析** | `score_ks_plot`, `score_dist_plot`, `score_bin_plot`, `score_lift_plot` |
| **风控策略** | `vintage_plot`, `threshold_analysis_plot`, `strategy_compare_plot` |
| **特征分析** | `variable_iv_plot`, `variable_woe_trend_plot`, `variable_psi_heatmap` |
| **稳定性监控** | `psi_plot`, `csi_plot`, `population_drift_monitor`, `feature_drift_comparison` |
| **辅助工具** | `corr_plot`, `hist_plot`, `confusion_matrix_plot`, `feature_importance_plot` |

```python
import hscredit.core.viz as viz

viz.bin_plot(df, 'age', 'y', show_iv=True)    # 分箱 WOE 趋势图
viz.roc_plot(y_true, y_prob)                    # ROC 曲线
viz.vintage_plot(df, '账龄', '逾期金额')          # Vintage 账龄分析
```

---

## 🏗️ 架构概览

```
hscredit/
├── core/
│   ├── binning/       ← 19 种分箱算法
│   ├── encoders/      ← 9 种特征编码器
│   ├── selectors/     ← 21 种特征筛选器
│   ├── models/
│   │   ├── classical/     # LogisticRegression
│   │   ├── boosting/     # XGBoost / LightGBM / CatBoost / NGBoost
│   │   ├── scorecard/     # ScoreCard, ScoreTransformer
│   │   ├── losses/        # 14 种自定义损失函数
│   │   ├── evaluation/    # 校准、可解释性
│   │   └── tuning/        # 超参调优
│   ├── metrics/       ← 分类 / 回归 / 特征 / 稳定性 / 金融指标
│   ├── viz/           ← 45+ 种可视化图表
│   ├── eda/           ← 数据探索分析
│   ├── rules/         ← 规则引擎 (Rule, RuleClassifier)
│   ├── financial/     ← FV / PV / PMT / NPER / IRR / NPV
│   └── feature_engineering/  # NumExprDerive
├── report/
│   ├── mining/        ← 规则挖掘器
│   ├── overdue_predictor.py  # 加权逾期预测
│   ├── feature_analyzer.py   # 特征分箱统计
│   ├── swap_analysis.py      # 置换风险分析
│   ├── model_report.py       # 模型报告
│   └── population_drift.py   # 群体稳定性监控
├── excel/             ← ExcelWriter 上下文管理器
└── utils/             ← Pandas 扩展 / IO / 日志 / 随机种子
```

---

## 🎓 谁在使用？

- 🏦 银行风控部策略师与建模工程师
- 💳 消费金融与互联网贷款团队
- 🔬 金融科技创业公司
- 📚 高校金融工程与风险管理研究者
- 📊 第三方数据与咨询机构

---

## 🤝 参与贡献

```bash
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

---

## 💬 交流群 & 公众号

<div align="center">

**微信公众号：衡枢风控**
![WeChat Official Account](https://img.shields.io/badge/微信公众号-衡枢风控-07C160?style=flat-square&logo=wechat&logoColor=white)

<p>

关注公众号，回复 <strong>"入群"</strong> 即可加入 hscredit 技术交流群，与业界风控从业者一起交流建模经验。

</p>

**公众号 ID**: `hengshucredit-com`

</div>

---

## 📜 许可证

MIT License — 可商用，无任何限制。

---

<p align="center">
  <strong>🏦 hscredit</strong><br>
  鉴真伪 · 斟信用 · 衡风险 · 枢定策<br>
  <sub>让每个风控人都能用上专业的建模工具</sub>
</p>

<p align="center">
  如果这个项目对你有帮助，欢迎点个 ⭐
</p>
