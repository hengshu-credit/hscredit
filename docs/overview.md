# 功能总览

`hscredit` 把信贷风控中策略、数据、变量、模型、规则、监控和报告等环节沉淀到一个
工具箱中，减少在多个建模库、分析脚本与 Excel 模板之间反复切换的成本。

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

## 数据探索与策略分析

| 能力 | 说明 |
|:---|:---|
| 数据质量分析 | 数据概览、缺失分析、字段摘要、数据质量报告 |
| 目标变量分析 | 目标分布、整体坏率、分维度坏率、坏率趋势、样本分布 |
| 变量探索 | 数值/类别分布、异常值、稀有类别、集中度分析 |
| 变量有效性 | IV、WOE、分箱坏率、单调性、单变量 AUC、特征重要性排序 |
| 稳定性分析 | PSI、CSI、跨期 PSI、特征漂移、分数漂移 |
| 客群分析 | 客群画像、客群迁移、分群漂移、跨客群变量有效性 |
| 策略分析 | 审批率/坏率权衡、策略仿真、Vintage 汇总、Roll Rate、标签泄漏检查 |

## 分箱、编码与筛选

| 能力 | 说明 |
|:---|:---|
| 分箱方法 | 等宽、等频、树分箱、CART、卡方、Best IV、Best KS、Best Lift、MDLP、OR-Tools、CP-SAT、KMeans、单调约束、遗传算法、平滑、核密度、目标坏率、二维最优分箱 |
| 编码方法 | WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、Cardinality、GBM 编码 |
| 特征筛选 | 缺失率、众数率、方差、相关性、VIF、IV、Lift、PSI、基数、类型、正则、模型重要性、零重要性、RFE、序列选择、逐步回归、Boruta、互信息、卡方、F 检验、稳定性感知、评分卡组合筛选 |
| 特征衍生 | NumExprDerive 表达式引擎（基于 numexpr），支持 where/sin/cos/abs 等函数与条件逻辑的批量特征衍生 |

## 建模、评估与监控

| 能力 | 说明 |
|:---|:---|
| 评分卡模型 | 逻辑回归、ScoreCard、RoundScoreCard、评分转换、分数漂移校准 |
| 机器学习模型 | RandomForest、ExtraTrees、GradientBoosting、XGBoost、LightGBM、CatBoost、NGBoost |
| 风控损失函数 | Focal、非对称 Focal、加权 BCE、成本敏感、坏账、审批率、利润最大化、排序、KS 聚焦、Top-K 坏样本捕获、金额加权等 |
| 调参与解释 | Optuna 调参、模型评估、概率校准、解释性分析、SHAP 可选支持 |
| 风控指标 | KS、AUC、Gini、Lift、坏率、IV、PSI、CSI、回归指标、分箱统计 |
| 金融计算 | FV/PV/PMT/NPER/IPMT/PPMT/RATE 现值终值年金计算，NPV/IRR/MIRR 净现值与收益率计算 |

## 规则、报告与交付

| 能力 | 说明 |
|:---|:---|
| 规则分析 | Rule 表达式、规则变量解析、规则美化、规则组合、规则集分类器、多标签规则分析 |
| 规则挖掘 | 单特征规则、多特征交叉规则、多标签规则、树规则提取、手工树分析、规则指标 |
| 策略置换 | 新旧策略置换、通过/拒绝交叉矩阵、风险迁移分析 |
| 可视化 | 分箱趋势、模型评估、评分分布、策略曲线、变量稳定性、客群漂移、树图 |
| Excel 报告 | 特征分析、规则分析、Swap 分析、逾期预测、模型报告、模型对比、格式化 Excel 输出 |

## 架构设计

所有核心组件均继承自 sklearn `BaseEstimator` + `TransformerMixin`/`ClassifierMixin`，
兼容 sklearn Pipeline，并统一支持两种调用风格：

```python
# sklearn 风格 —— X 和 y 分别传入
binner.fit(X_train, y_train)

# scorecardpipeline 风格 —— 通过 target 参数指定目标列名
binner = OptimalBinning(target="target")
binner.fit(df)  # 自动从 df 中提取目标列
```

模块组织：

```text
hscredit/
├── core/
│   ├── binning/              # 分箱算法 + BaseBinning + OptimalBinning 工厂
│   ├── encoders/             # 编码器（WOE/Target/Count/OneHot/...）
│   ├── selectors/            # 特征筛选器
│   ├── models/               # boosting / classical / scorecard / losses / rules / tuning
│   ├── metrics/              # 分类 / 稳定性 / 特征 / 金融指标
│   ├── viz/                  # 可视化图表
│   ├── eda/                  # 数据探索
│   ├── rules/                # 规则引擎（Rule 表达式）
│   ├── financial/            # 金融计算（FV/PV/PMT/NPER/IRR/NPV）
│   └── feature_engineering/  # NumExprDerive 表达式衍生
├── report/                   # 报告生成器 + mining 规则挖掘器
├── excel/                    # ExcelWriter 上下文管理器
└── utils/                    # Pandas 扩展 / IO / 日志 / 随机种子
```
