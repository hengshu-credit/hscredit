# 金融风控建模全流程数据分析与可视化方法调研报告

## 执行摘要

本报告系统梳理了信贷风控建模全流程中各阶段的标准分析方法和可视化内容，涵盖数据质量探索、目标变量分析、单变量特征分析、特征与标签关系分析、特征间关系分析、稳定性分析六大维度，共总结25个核心分析方法。报告按建模阶段分类整理，明确标注核心方法与增强方法，并提供中文化指标命名建议，为风控建模自动化工具开发提供方法论支撑。

---

## 一、数据质量探索阶段

### 1.1 数据完整性检查

#### 方法1：缺失率分析
- **函数名**：`calculate_missing_rate`
- **输入参数**：`df: pd.DataFrame`, `columns: List[str] = None`
- **返回值**：`pd.DataFrame`（列名、缺失数、缺失率、数据类型）
- **适用场景**：数据接入后的首要检查，识别需要填充或删除的字段
- **计算公式**：`缺失率 = 缺失值数量 / 总记录数 × 100%`
- **输出形式**：DataFrame包含：特征名、缺失数、缺失率、数据类型、建议处理方式
- **可视化方式**：缺失率热力图、缺失模式矩阵图
- **方法类型**：核心方法

#### 方法2：空值分布分析
- **函数名**：`analyze_null_pattern`
- **输入参数**：`df: pd.DataFrame`, `target_col: str = None`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析缺失值是否与目标变量相关（MNAR/MAR/MCAR判断）
- **计算公式**：分组统计各特征缺失情况下的目标变量分布
- **输出形式**：DataFrame包含：特征名、缺失时坏样本率、非缺失时坏样本率、差异显著性
- **可视化方式**：缺失值与目标变量关系条形图
- **方法类型**：增强方法

### 1.2 数据一致性检查

#### 方法3：异常值检测（数值型）
- **函数名**：`detect_outliers_numeric`
- **输入参数**：`df: pd.DataFrame`, `column: str`, `method: str = 'iqr'`
- **返回值**：`pd.DataFrame`
- **适用场景**：识别数值特征的异常值，判断是否为数据错误或真实极端值
- **计算公式**：
  - IQR方法：`Q1 - 1.5×IQR` 至 `Q3 + 1.5×IQR` 之外为异常值
  - Z-score方法：`|Z| > 3` 为异常值
- **输出形式**：DataFrame包含：特征名、异常值数量、异常值比例、异常值边界
- **可视化方式**：箱线图、散点图
- **方法类型**：核心方法

#### 方法4：异常值检测（类别型）
- **函数名**：`detect_rare_categories`
- **输入参数**：`df: pd.DataFrame`, `column: str`, `threshold: float = 0.01`
- **返回值**：`pd.DataFrame`
- **适用场景**：识别类别特征中的稀有类别，考虑合并或单独处理
- **计算公式**：类别频次占比 < threshold 视为稀有类别
- **输出形式**：DataFrame包含：类别值、频次、占比、是否稀有
- **可视化方式**：频次分布条形图
- **方法类型**：核心方法

### 1.3 数据时效性分析

#### 方法5：数据时间跨度检查
- **函数名**：`check_data_timeliness`
- **输入参数**：`df: pd.DataFrame`, `date_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：检查数据时间范围是否符合预期，识别时间断层
- **计算公式**：统计最小日期、最大日期、时间跨度、各月记录数
- **输出形式**：DataFrame包含：时间维度、最早日期、最晚日期、跨度天数、月均记录数
- **可视化方式**：时间序列记录数分布图
- **方法类型**：核心方法

---

## 二、目标变量分析

### 2.1 样本分布分析

#### 方法6：样本分布统计
- **函数名**：`analyze_target_distribution`
- **输入参数**：`df: pd.DataFrame`, `target_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：了解好坏样本比例，判断样本不平衡程度
- **计算公式**：各类别样本数及占比
- **输出形式**：DataFrame包含：样本类别、样本数、占比、建议采样策略
- **可视化方式**：饼图、条形图
- **方法类型**：核心方法

#### 方法7：逾期率分析（总体）
- **函数名**：`calculate_overall_default_rate`
- **输入参数**：`df: pd.DataFrame`, `target_col: str`
- **返回值**：`float`
- **适用场景**：获取整体逾期率基准值
- **计算公式**：`逾期率 = 坏样本数 / 总样本数 × 100%`
- **输出形式**：标量值
- **可视化方式**：仪表盘图
- **方法类型**：核心方法

#### 方法8：逾期率分析（分维度）
- **函数名**：`analyze_default_rate_by_dimension`
- **输入参数**：`df: pd.DataFrame`, `target_col: str`, `dimension_cols: List[str]`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析不同维度下的逾期率差异，识别高风险群体
- **计算公式**：按维度分组计算逾期率
- **输出形式**：DataFrame包含：维度名称、维度值、样本数、逾期率、与总体差异
- **可视化方式**：分组条形图、热力图
- **方法类型**：核心方法

### 2.2 Vintage分析

#### 方法9：Vintage cohort分析
- **函数名**：`vintage_analysis`
- **输入参数**：`df: pd.DataFrame`, `vintage_col: str`, `mob_col: str`, `target_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：追踪不同放款批次（Vintage）随账龄（MOB）的风险表现变化
- **计算公式**：`累积坏账率(MOB=n) = 截至MOB=n时该队列中坏账户数 / 该队列开户总账户数`
- **输出形式**：DataFrame包含：Vintage队列、各MOB点的累积坏账率
- **可视化方式**：Vintage曲线图（X轴MOB，Y轴累积坏账率，多条线代表不同Vintage）
- **方法类型**：核心方法

#### 方法10：滚动率分析（Roll Rate）
- **函数名**：`roll_rate_analysis`
- **输入参数**：`df: pd.DataFrame`, `state_col: str`, `period_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析客户在不同逾期状态间的迁移规律，确定不良客户定义阈值
- **计算公式**：`滚动率 = 第二时间点某状态客户数 / 第一时间点基准状态客户总数`
- **输出形式**：DataFrame包含：起始状态、目标状态、迁移数量、迁移率
- **可视化方式**：状态迁移矩阵热力图、桑基图
- **方法类型**：核心方法

---

## 三、单变量特征分析

### 3.1 数值型特征分析

#### 方法11：数值特征统计描述
- **函数名**：`numeric_feature_statistics`
- **输入参数**：`df: pd.DataFrame`, `column: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：全面了解数值特征的分布特征
- **计算公式**：均值、中位数、标准差、最小值、最大值、偏度、峰度、分位数
- **输出形式**：DataFrame包含：统计指标名称、指标值
- **可视化方式**：直方图、箱线图、Q-Q图
- **方法类型**：核心方法

#### 方法12：分布正态性检验
- **函数名**：`normality_test`
- **输入参数**：`df: pd.DataFrame`, `column: str`, `method: str = 'shapiro'`
- **返回值**：`pd.DataFrame`
- **适用场景**：判断特征是否符合正态分布，指导后续变换方法选择
- **计算公式**：Shapiro-Wilk检验或Kolmogorov-Smirnov检验
- **输出形式**：DataFrame包含：检验统计量、p值、是否正态分布
- **可视化方式**：Q-Q图、概率密度图
- **方法类型**：增强方法

### 3.2 类别型特征分析

#### 方法13：类别特征频次分析
- **函数名**：`categorical_frequency_analysis`
- **输入参数**：`df: pd.DataFrame`, `column: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：了解类别特征的分布情况
- **计算公式**：各类别频次、占比、累计占比
- **输出形式**：DataFrame包含：类别值、频次、占比、累计占比
- **可视化方式**：条形图、帕累托图
- **方法类型**：核心方法

#### 方法14：类别集中度分析（Gini系数）
- **函数名**：`calculate_category_gini`
- **输入参数**：`df: pd.DataFrame`, `column: str`
- **返回值**：`float`
- **适用场景**：衡量类别分布的集中程度，识别过于集中的特征
- **计算公式**：`Gini = 1 - Σ(Pi²)`，其中Pi为第i个类别的占比
- **输出形式**：标量值（0-1，越接近1表示集中度越高）
- **可视化方式**：洛伦兹曲线
- **方法类型**：增强方法

### 3.3 时间型特征分析

#### 方法15：时间趋势分析
- **函数名**：`time_trend_analysis`
- **输入参数**：`df: pd.DataFrame`, `date_col: str`, `value_col: str`, `freq: str = 'M'`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析特征随时间的变化趋势
- **计算公式**：按时间周期聚合统计（均值、总和、计数等）
- **输出形式**：DataFrame包含：时间周期、聚合值、环比、同比
- **可视化方式**：时间序列折线图
- **方法类型**：核心方法

---

## 四、特征与标签关系分析

### 4.1 WOE和IV分析

#### 方法16：最优分箱
- **函数名**：`optimal_binning`
- **输入参数**：`df: pd.DataFrame`, `feature_col: str`, `target_col: str`, `method: str = 'chi2'`
- **返回值**：`pd.DataFrame`
- **适用场景**：将连续特征离散化为最优分箱，满足单调性要求
- **计算公式**：
  - 等频分箱：每箱样本数相等
  - 等距分箱：每箱取值范围相等
  - 卡方分箱：基于卡方统计量合并相邻区间
  - 决策树分箱：基于决策树分裂点分箱
- **输出形式**：DataFrame包含：分箱编号、分箱边界、样本数、坏样本数、好样本数
- **可视化方式**：分箱边界图、各箱样本分布图
- **方法类型**：核心方法

#### 方法17：WOE（证据权重）计算
- **函数名**：`calculate_woe`
- **输入参数**：`df: pd.DataFrame`, `feature_col: str`, `target_col: str`, `bins: pd.DataFrame`
- **返回值**：`pd.DataFrame`
- **适用场景**：将特征转换为WOE值，用于逻辑回归建模
- **计算公式**：`WOE = ln(好客户占比 / 坏客户占比) = ln(%Good / %Bad)`
- **输出形式**：DataFrame包含：分箱编号、分箱边界、WOE值、样本数、坏样本率
- **可视化方式**：WOE趋势图（检查单调性）
- **方法类型**：核心方法

#### 方法18：IV（信息值）计算
- **函数名**：`calculate_iv`
- **输入参数**：`df: pd.DataFrame`, `feature_col: str`, `target_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：评估特征的预测能力，用于特征筛选
- **计算公式**：`IV = Σ(%Good - %Bad) × WOE`
- **输出形式**：DataFrame包含：特征名、IV值、预测能力评级
- **可视化方式**：IV值条形图
- **方法类型**：核心方法

**IV值解释标准**：
- IV < 0.02：预测能力弱，建议排除
- 0.02 ≤ IV < 0.1：预测能力较弱
- 0.1 ≤ IV < 0.3：预测能力中等
- 0.3 ≤ IV < 0.5：预测能力强
- IV ≥ 0.5：预测能力极强（需警惕数据泄露）

### 4.2 单调性检验

#### 方法19：WOE单调性检验
- **函数名**：`check_woe_monotonicity`
- **输入参数**：`df: pd.DataFrame`, `feature_col: str`, `target_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：检验特征WOE是否单调，满足评分卡业务解释性要求
- **计算公式**：检查WOE序列是否单调递增或单调递减
- **输出形式**：DataFrame包含：特征名、是否单调、单调方向、违反单调性的分箱
- **可视化方式**：WOE趋势图
- **方法类型**：核心方法

### 4.3 坏账率趋势分析

#### 方法20：分箱坏账率分析
- **函数名**：`bin_default_rate_analysis`
- **输入参数**：`df: pd.DataFrame`, `feature_col: str`, `target_col: str`, `bins: pd.DataFrame`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析特征各分箱的坏账率分布，验证特征区分能力
- **计算公式**：`分箱坏账率 = 分箱内坏样本数 / 分箱总样本数`
- **输出形式**：DataFrame包含：分箱编号、分箱边界、样本数、坏账率、与总体坏账率差异
- **可视化方式**：坏账率趋势图
- **方法类型**：核心方法

---

## 五、特征间关系分析

### 5.1 相关性分析

#### 方法21：相关性矩阵计算
- **函数名**：`correlation_matrix`
- **输入参数**：`df: pd.DataFrame`, `columns: List[str] = None`, `method: str = 'pearson'`
- **返回值**：`pd.DataFrame`
- **适用场景**：识别特征间的线性相关关系
- **计算公式**：
  - Pearson相关系数：衡量线性相关
  - Spearman相关系数：衡量单调相关
- **输出形式**：DataFrame（相关系数矩阵）
- **可视化方式**：相关性热力图
- **方法类型**：核心方法

### 5.2 多重共线性检验

#### 方法22：VIF（方差膨胀因子）计算
- **函数名**：`calculate_vif`
- **输入参数**：`df: pd.DataFrame`, `columns: List[str]`
- **返回值**：`pd.DataFrame`
- **适用场景**：检测特征间的多重共线性，避免回归模型系数不稳定
- **计算公式**：`VIF = 1 / (1 - R²)`，其中R²为该特征对其他特征回归的决定系数
- **输出形式**：DataFrame包含：特征名、VIF值、共线性程度判断
- **可视化方式**：VIF值条形图
- **方法类型**：核心方法

**VIF值解释标准**：
- VIF < 5：无明显共线性
- 5 ≤ VIF < 10：中度共线性
- VIF ≥ 10：严重共线性，建议剔除或合并

### 5.3 特征交叉分析

#### 方法23：特征交叉分析
- **函数名**：`feature_cross_analysis`
- **输入参数**：`df: pd.DataFrame`, `col1: str`, `col2: str`, `target_col: str`
- **返回值**：`pd.DataFrame`
- **适用场景**：分析两个特征的联合分布与目标变量的关系
- **计算公式**：构建交叉表，计算各组合的坏账率
- **输出形式**：DataFrame包含：特征1取值、特征2取值、样本数、坏账率
- **可视化方式**：交叉热力图、分组条形图
- **方法类型**：增强方法

---

## 六、稳定性分析

### 6.1 群体稳定性分析

#### 方法24：PSI（群体稳定性指数）计算
- **函数名**：`calculate_psi`
- **输入参数**：`df_base: pd.DataFrame`, `df_current: pd.DataFrame`, `column: str`, `bins: int = 10`
- **返回值**：`pd.DataFrame`
- **适用场景**：比较两个数据集（如训练集与测试集、不同时间窗口）的分布稳定性
- **计算公式**：`PSI = Σ(%Current - %Base) × ln(%Current / %Base)`
- **输出形式**：DataFrame包含：分箱编号、基准占比、当前占比、PSI贡献值、总PSI
- **可视化方式**：分布对比图、PSI贡献条形图
- **方法类型**：核心方法

**PSI值解释标准**：
- PSI < 0.1：分布基本无变化，模型稳定
- 0.1 ≤ PSI < 0.2：分布有轻微变化，需关注
- PSI ≥ 0.2：分布显著变化，建议重建模型

### 6.2 特征稳定性分析

#### 方法25：CSI（特征稳定性指数）计算
- **函数名**：`calculate_csi`
- **输入参数**：`df_base: pd.DataFrame`, `df_current: pd.DataFrame`, `column: str`, `bins: int = 10`
- **返回值**：`pd.DataFrame`
- **适用场景**：监控单个特征的分布稳定性，定位PSI异常的原因
- **计算公式**：`CSI = Σ(%Current - %Base) × ln(%Current / %Base)`（与PSI公式相同，应用于特征而非预测概率）
- **输出形式**：DataFrame包含：特征名、CSI值、稳定性评级、各分箱占比变化
- **可视化方式**：特征分布对比图
- **方法类型**：核心方法

**CSI值解释标准**：
- CSI < 0.1：特征分布非常稳定
- 0.1 ≤ CSI < 0.25：特征分布有轻微变化，需关注
- CSI ≥ 0.25：特征分布发生显著偏移，需深入排查

---

## 七、可视化图表类型汇总

### 7.1 风控领域特有图表

| 图表名称 | 适用方法 | 用途 |
|---------|---------|------|
| ROC曲线 | 模型评估 | 展示不同阈值下的TPR和FPR关系 |
| KS曲线 | 模型评估 | 展示好坏样本累积分布差异 |
| Lift曲线 | 模型评估 | 展示模型相对随机选择的提升效果 |
| Gini系数图 | 模型评估 | 展示模型区分能力 |
| Vintage曲线 | 目标变量分析 | 展示不同队列随账龄的风险表现 |
| 滚动率矩阵 | 目标变量分析 | 展示逾期状态迁移规律 |
| WOE趋势图 | 特征分析 | 展示分箱WOE值的单调性 |
| IV值条形图 | 特征筛选 | 展示各特征的预测能力排序 |

### 7.2 通用分析图表

| 图表名称 | 适用阶段 | 用途 |
|---------|---------|------|
| 缺失率热力图 | 数据质量 | 直观展示各特征缺失情况 |
| 箱线图 | 数据质量 | 展示数值特征分布和异常值 |
| 直方图 | 单变量分析 | 展示数值特征分布形态 |
| 条形图 | 单变量分析 | 展示类别特征频次分布 |
| 相关性热力图 | 特征关系 | 展示特征间相关关系 |
| 时间序列图 | 时效性分析 | 展示指标随时间的变化趋势 |
| 分布对比图 | 稳定性分析 | 对比两个数据集的分布差异 |

---

## 八、方法分类总结

### 8.1 核心方法（必须实现）

共15个核心方法，覆盖建模全流程的必要分析：

1. **数据质量阶段**：缺失率分析、异常值检测（数值型）、异常值检测（类别型）、数据时间跨度检查
2. **目标变量分析**：样本分布统计、逾期率分析（总体）、逾期率分析（分维度）、Vintage cohort分析、滚动率分析
3. **单变量分析**：数值特征统计描述、类别特征频次分析、时间趋势分析
4. **特征标签关系**：最优分箱、WOE计算、IV计算
5. **特征间关系**：相关性矩阵、VIF计算
6. **稳定性分析**：PSI计算、CSI计算

### 8.2 增强方法（可选实现）

共10个增强方法，用于深度分析：

1. **数据质量阶段**：空值分布分析
2. **单变量分析**：分布正态性检验、类别集中度分析
3. **特征标签关系**：WOE单调性检验、分箱坏账率分析
4. **特征间关系**：特征交叉分析

---

## 九、中文化指标命名建议

| 英文名称 | 中文名称 | 缩写 | 适用场景 |
|---------|---------|------|---------|
| Missing Rate | 缺失率 | MR | 数据质量检查 |
| Outlier Detection | 异常值检测 | OD | 数据质量检查 |
| Default Rate | 逾期率/坏账率 | DR | 目标变量分析 |
| Vintage Analysis | 账龄分析 | VA | 目标变量分析 |
| Roll Rate | 滚动率 | RR | 目标变量分析 |
| Weight of Evidence | 证据权重 | WOE | 特征工程 |
| Information Value | 信息值 | IV | 特征筛选 |
| Population Stability Index | 群体稳定性指数 | PSI | 模型监控 |
| Characteristic Stability Index | 特征稳定性指数 | CSI | 特征监控 |
| Variance Inflation Factor | 方差膨胀因子 | VIF | 共线性检测 |
| Kolmogorov-Smirnov Statistic | KS统计量 | KS | 模型评估 |
| Receiver Operating Characteristic | 受试者工作特征曲线 | ROC | 模型评估 |
| Area Under Curve | 曲线下面积 | AUC | 模型评估 |
| Gini Coefficient | Gini系数 | Gini | 模型评估 |
| Lift | 提升度 | Lift | 模型评估 |

---

## 十、参考资料

1. [Credit Risk : Vintage Analysis - ListenData](https://www.listendata.com/2019/09/credit-risk-vintage-analysis.html)
2. [Roll Rate Analysis - ListenData](https://www.listendata.com/2019/09/roll-rate-analysis.html)
3. [Population Stability Index - ListenData](https://www.listendata.com/2015/05/population-stability-index.html)
4. [Optimal Binning and Weight of Evidence Framework](https://evandeilton.github.io/OptimalBinningWoE/)
5. [Credit Risk Modeling Framework - GitHub](https://github.com/rudrathorat/Credit-Risk-Modeling-Framework)
6. [风控模型—特征稳定性指标 (CSI)深入理解应用 - 知乎](https://zhuanlan.zhihu.com/p/86559671)
7. [如何向门外汉讲解ks值（风控模型术语）？- 知乎](https://www.zhihu.com/question/34820996)

---

*报告生成时间：2026年3月26日*
*研究员：researcher-finance-e*
