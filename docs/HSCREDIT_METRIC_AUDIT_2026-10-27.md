# HSCredit 当前设计、分析流程与结果表指标审计

> 本文保留第一轮审计与验证快照。后续逐项修复、确定的边界口径及额外复核结果见 [指标逐项修复与逻辑复核](HSCREDIT_METRIC_RECHECK_2026-10-27.md)。

审查日期：2026-10-27（Asia/Shanghai）。起始版本：`0.1.3`，分支 `main`，HEAD `40d04400663ed9015c79902445ed3dd6d15cd3b4`。

## 1. 审查方法与结论边界

本次先研究公开接口、输入处理、核心公式和报告调用链，再建立可手算样本与真实数据独立核算，最后才修改已证实的问题。新测试的预期值来自原始行计数、金额求和、数学恒等式或第三方公开指标，不以“两个入口调用相同实现”作为正确性的唯一证据。

HSCredit 已是覆盖数据接入、EDA、分箱编码、筛选、建模、调参、解释、规则、监控和报告的工具库。主要风险集中于**结果表的派生公式、口径传递和边界状态**。已有数千个测试并不意味着所有表格指标都经过独立数学核算：例如旧测试曾明确要求金额“累计风险拒绝比”等于累计金额占比，能够通过测试，却违反同项目订单口径与规则报告的定义。

本次进行了全库结构盘点和重点计算链路审查，并不声称逐行证明全部算法、完成所有操作系统/依赖组合验收或验证未配置的真实数据库。后文区分已复现错误、需要业务选择的统计口径和工程完善项。

**本次已确认并修复 M01–M08 共 8 组问题，新增 62 个回归用例。修复后常规测试 6,544 通过、0 失败、9 项真实数据库测试跳过；27 项 slow/integration 标记测试未运行。** 原始放款数据、分箱算法选择和未决业务口径均未修改。

## 2. 当前规模与文档漂移

起始源码共有 **263 个 Python 文件、151,278 行**（含注释和文档字符串），原有 **273 个测试文件、39 个根级示例 Notebook**。`examples/` 没有根级 `.py` 演示。

| 旧记录 | 当前源码证据 | 处理建议 |
|---|---|---|
| 18 种统一分箱方法 | `OptimalBinning.VALID_METHODS` 实际是 17 种；二维分箱单独提供 | 以注册表为准，不混合一维算法数和二维组件数 |
| 版本 0.1.1 | 顶层版本为 0.1.3，构建版本由同一字面量提取 | 不再把早期版本不一致直接认定为当前缺陷 |
| NumPy 2 窄整数排序溢出 | 已改为元组排序与索引映射 | 历史问题已修复 |
| Black 无法解析目标版本 | 当前 `black>=24.10`、`target-version=['py39']`，有只读格式门禁 | 重新执行检查，不能沿用旧结论 |
| import 时无条件屏蔽警告 | `init_setting(suppress_warnings=False)` 默认不屏蔽 | 不再列为当前默认行为 |
| 682 通过 / 12 失败 | 已严重过期；本次重新执行并记录环境及失败原因 | 固定引用带日期的运行结果 |
| 190 处广义异常、84 处 pass | 起始 AST 统计为 270 处广义捕获，其中 35 处直接 pass | 数量只是定位线索；结构化错误收集和连接回滚不等于吞错 |

数据接入 `database/`、`skills_runtime/`、模型生命周期/过程报告、冻结监控基准、折外编码、受约束反事实等能力已经存在；旧 ROADMAP 中一些“计划新增”项目也需要重新对照实现。

## 3. 模块结构与实际实现

| 模块 | 主要实现与处理方式 | 需要关注的约束 |
|---|---|---|
| `utils/data_contracts.py`、`targets.py`、`arguments.py` | 学习接口解析 `X/y/target`，分析接口绑定列名、数组、Series、callable 和多标签；辅助字段在筛选前绑定 | 学习接口默认按位置；分析接口的 Series 要完整索引对齐，不能混用两种假设 |
| `database/` | Database 门面、适配器注册、连接池、流式读取、类型推断、批量写入和写入结果 | 流式读取不代表后续模型/Excel 全链路流式；真实服务集成需单独验证 |
| `core/eda/` | 画像、标签、关联、漂移、客群、阈值策略、Vintage、Roll Rate | 样本数、账户数、金额、成熟度和缺失处理必须各自说明 |
| `core/binning/` | BaseBinning 统一数据处理、规则保存、类别协议、缺失/特殊/未知箱；OptimalBinning 分派算法，支持预分箱与后续约束 | “最优”受候选切点、目标函数、求解时限与约束影响，不代表任意数据上全局最优 |
| `core/metrics/` | 分箱充分统计、WOE/IV、排序曲线、稳定性、金融与回归指标；BinStatsAccumulator 合并分块计数 | 二分类和金额模式应共用派生公式；合计不能再次累加前缀和或复制首行 |
| `core/encoders/` | WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost、GBM、Cardinality 共 9 类 | WOE/Target 有 `training_mode='oof'`；默认仍是 in_sample，使用者需明确选择 |
| `core/selectors/` | 过滤、包装、嵌入式方法；基类管理输入、拟合状态、中文报告；Composite 支持顺序与交集 | 选择器应只在训练折拟合；统计意义、模型重要性与业务可用性不是同一目标 |
| `core/models/` | BaseRiskModel 与 `_contracts/_lifecycle/_workflow` 统一特征 schema、类别概率、验证集、权重、拟合事务、训练历史和报告 | 训练权重、评估权重、逐行金融参数应分别管理；失败重拟合不能留下半成品状态 |
| `models/classical`、`boosting` | LR、RF、ExtraTrees、GBDT 与 XGBoost/LightGBM/CatBoost/NGBoost 包装 | 可选依赖按需加载；原生框架参数与 sklearn clone/set_params 要保持一致 |
| `models/scorecard/` | ScoreCard/RoundScoreCard、分值转换、概率转换、分项分数、部署导出 | 区分分数与类别预测；输出代码、内部转换和线上预测需做一致性验证 |
| `models/losses/`、`tuning/` | 损失/指标注册、框架适配、自定义业务目标、搜索空间、CV、Optuna 与试验留存 | 代理损失不等于最终业务收益；跨折逐行金额不能绑定错数据 |
| `models/calibration/`、`explainability/` | 概率校准、SHAP、原因码、受约束反事实和解释报告 | 校准需独立验证数据；反事实是模型条件建议，不是因果证明 |
| `core/rules/`、`report/mining/` | Rule 表达式、规则组合与流、单/多特征、多标签和树规则；Rule.report 是业务指标来源 | 命中人数与金额应分别计算；正类、规则方向、先验过滤必须一致 |
| `report/` | 特征/模型/规则/置换/漂移报告、OverduePredictor；ReportResult 保留章节状态与失败原因 | strict/best_effort 的完整性语义比“生成了文件”更重要 |
| `core/viz/`、`excel/` | 共用统计数据生成图形，ExcelWriter 管理格式、图片和导出 | 展示格式不能提前损失计算精度；图、内存表、Excel 数值需一致 |
| `skills_runtime/` | 结构化输入、同进程对象引用、操作注册、制品暂存与发布 | 制品成功应依赖实际验证，不能将空文件或失败章节报为成功 |
| `core/financial/` | PV/FV/PMT/NPER/IPMT/PPMT/RATE、NPV/IRR/MIRR | 符号、期初/期末、零利率、IRR 多根与不收敛要显式区分 |
| `core/feature_engineering/` | 当前以 NumExprDerive 表达式衍生为主 | 通用时间/实体聚合、拒绝推断等不能仅凭 ROADMAP 当作已有完整能力 |

### 分箱算法的设计分组

- 无监督：uniform、quantile、kmeans、kernel_density，主要依据特征分布。
- 监督与约束：tree、cart、chi、mdlp、best_ks、best_iv、best_lift、target_bad_rate、monotonic、genetic、smooth，目标与搜索方式不同。
- 运筹：or_tools、cp_sat，在候选分区及约束下优化。
- 二维：OptimalBinning2D 单独建模交互关系，统计仍应回到统一内核。

算法比较必须固定样本、标签、缺失/特殊值、最小箱占比、最大箱数及训练/验证划分。训练集 IV 最大不等于泛化最优。

## 4. 推荐的完整分析链路

流程（Mermaid 源码）：

```text
flowchart TD
    A[数据接入与字段类型核验] --> B[目标定义、成熟度、金额与日期绑定]
    B --> C[训练、验证、OOT 或时间/群组切分]
    C --> D[训练集 EDA、分箱、编码与特征筛选]
    D --> E[模型训练与调参]
    E --> F[固定规则应用到验证与 OOT]
    F --> G[区分度、校准、稳定性与策略评估]
    G --> H[结果表独立核算和报告完整性检查]
    H --> I[模型与规则导出、预测一致性验收]
    I --> J[冻结基准监控与跨期追踪]
```

1. **先确定统计对象。** 一行是申请、贷款、客户还是账户×MOB；重复记录不能默认视为新增账户。
2. **再确定目标。** `FPD` 的二值定义与 `MOB1 > 7` 不是同一标签。指定 `overdue_operator`；判断灰客户和未成熟样本是否应纳入。
3. **在学习前切分。** 有监督分箱、Target/WOE 编码、特征筛选和调参放在训练/CV 流程内；验证/OOT 复用已学规则。
4. **保留方向和基准。** 模型概率按正类风险方向评估；评分方向显式指定。PSI 监控保存固定基准，而不是每期重新挑分箱。
5. **评估多种口径。** AUC/KS 衡量排序，校准衡量概率含义，件数/金额坏率描述业务结果，策略仿真描述阈值选择；不能互相代替。
6. **交付前复核。** 合计数对原始数据、累计比例对各类别总量、规则命中对原始布尔掩码、金额对原始求和。最后再格式化和导出。

## 5. 结果表必须明确的指标定义

记全量件数或金额为 `T`，坏件数或坏金额为 `B`，某箱为 `t_i/b_i`，总体坏率 `r=B/T`。

| 指标 | 本项目采用的定义 | 最容易混淆的地方 |
|---|---|---|
| 样本占比 | `t_i/T` | 金额模式表示金额占比，不能仍按人数作分母 |
| 坏样本率 | `b_i/t_i` | 不是该箱占全量坏客户的比例 |
| 坏样本占比 | `b_i/B` | 与坏样本率分母不同 |
| 好样本占比 | `(t_i-b_i)/(T-B)` | KS 的好/坏累计比例应各自使用全量类别分母 |
| WOE | `ln(坏样本占比/好样本占比)` | 此处是坏/好方向，不能套用相反符号而不改模型解释 |
| IV | `Σ(坏占比-好占比)×WOE` | 某一类完全不存在时沿用二分类内核的 0 哨兵口径，并非证明特征有效 |
| LIFT | `(b_i/t_i)/(B/T)` | 累计 LIFT 用累计命中人群重新算，不是分箱 LIFT 求和 |
| 坏账改善 | `[r-(B-b_i)/(T-t_i)]/r` | 表示拒绝该箱后剩余人群坏率的相对变化 |
| 风险拒绝比 | `坏账改善/(t_i/T)` | 不是拒绝占比本身；负值表示拒绝后坏率反而变差 |
| KS | `max(abs(累计坏数/B-累计好数/(T-B)))` | 原始分数 KS 与分箱后 KS 可能不同，不能强求相等 |
| 连续目标合计均值 | `Σ目标值/Σ样本数` | 不能复制首箱均值或简单平均箱均值 |
| 连续目标合计方差 | `Σn_i[σ_i²+(μ_i-μ)²]/Σn_i` | 需要箱内与箱间两部分；只平均箱标准差不正确 |

分母为零、全部拒绝、单类标签等边界还涉及业务展示约定，见第 7 节。本次没有借修复公式之机统一改写这些约定。

## 6. 已独立复现的问题与修复范围

### M01 · P1 · 金额风险拒绝比被写成金额占比

位置：[`_binning.py`](../hscredit/core/metrics/_binning.py) 的金额分箱分支。

手算样本：三箱金额各 1,000，坏金额分别 100/500/900。总体坏率 50%，拒绝第一箱后坏率 70%，改善为 -40%，风险拒绝比应为 **-1.2**；旧实现输出 **0.3333**。累计指标也错误。

真实数据的首个评分箱金额 1,017,776、坏金额 83,896，旧表在“坏账改善”为负时仍输出正的风险拒绝比 0.249；修复后为 **-0.5589**，与拒绝该低风险箱后总体坏率上升一致。

修复依据是同项目二分类内核和 `Rule.report` 的明确公式，且“所有金额为 1 时应与件数结果一致”构成独立恒等关系。二分类和金额模式改为复用派生指标内核，避免再次分叉。

### M02 · P1 · 全部好客户仍产生金额 IV

位置同 M01。标签 `[0,0,0]`、分箱 `[0,0,1]`、金额 `[1,1,8]`，旧金额分支输出 IV **0.415888**，而件数内核对无好或无坏样本返回 0。原因是金额分支绕过共用 WOE 函数，对不存在的坏金额分布自行平滑，构造出了虚假的差异。

修复后复用已有 WOE/IV 内核，覆盖全好、全坏、某类金额全零、全零金额。

### M03 · P2 · 金额模式忽略 `round_digits=False`

金额 `[1.123456,2.234567]` 即使关闭格式化也返回 `[1.12,2.23]`；比例、WOE、IV、LIFT 同样被提前截断。这违反公共参数含义，影响二次聚合和精确对账。

修复后关闭格式化保留原始精度，默认展示仍保持金额两位、比例六位和 LIFT/改善四位等现有格式。

### M04 · P1 · KS 分桶表的累计比例分母错误

位置：[`classification.py`](../hscredit/core/metrics/classification.py) 的 `ks_bucket`。

8 人中 4 好 4 坏，高风险半区 3 坏 1 好。该半区的累计坏/好比例应为 3/4、1/4；累计到全量时两列都应为 **1**。旧表全量行却是 **0.5/0.5**，因为它计算了累计人群内部的坏率与好率。

修复为各自除以全量坏数和好数，并检查每行 `KS贡献≈abs(累计坏比例-累计好比例)`；类别累计比例的定义也与 [scikit-learn ROC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) 一致。不要求原始 KS 等于分箱 KS。

### M05 · P1 · 连续目标合计行沿用首箱

位置：[`_binning.py`](../hscredit/core/metrics/_binning.py) 的 `add_margins`。

两箱目标 `[1,3]`、`[10,14]`，总体总和 **28**、均值 **7**，旧“合计”却为总和 **4**、均值 **2**，标准差、最值和累计数也继承首行。旧合计实现只处理二分类列。

修复按样本权重合并连续目标矩，并在大基数（`1e12`）测试下避免用 `E[y²]-E[y]²` 计算方差；总体标准差采用 [NumPy `ddof=0` 定义](https://numpy.org/doc/stable/reference/generated/numpy.std.html)。同时覆盖单层及多级表头，不修改调用方输入。

### M06 · P1 · RuleMetrics 丢弃金额参数

位置：[`report/mining/metrics.py`](../hscredit/report/mining/metrics.py)。`_calculate_metrics` 将传入的 `amount` 立即设为 `None`，金额分支永远不执行；该分支本身还混用了人数和金额。

手算：命中 2 人，其中好金额 9、坏金额 1，件数坏率 50%，金额损失率应为 **10%**。旧输出缺少全部金额指标。

修复保持原“命中样本数/坏样本率/分类指标”为件数，额外通过 `Rule.report` 计算明确命名的金额指标。训练/测试金额分别绑定，数组和 Series、批量并行入口都验证。

### M07 · P1/P2 · RuleMetrics 正类参数与单类混淆矩阵

同一实现从不读取 `target_positive`。指定 0 为正类、规则恰好命中所有 0 类时，旧召回率仍为 **0**，应为 **1**。另外全好且全部未命中时，混淆矩阵是 1×1，直接拆成四个元素报错。

修复在进入指标计算前统一正类为 1，显式指定二元混淆矩阵标签；单类和多类输入分别按明确契约处理。

### M08 · P1/P2 · 阈值权衡表静默丢行和阈值不可复现

位置：[`core/eda/strategy.py`](../hscredit/core/eda/strategy.py) 的 `approval_badrate_tradeoff`。

- 标签 2 行、评分 3 行时，旧实现借助 pandas 索引交集只计算前 2 行，没有报错。
- 无效 `score_low_risk` 直接进入另一方向；无效点数、空样本和无限值没有统一的中文校验。
- 评分 `0..9`、两档阈值时，旧表显示阈值 **6.0、通过 3 人**；按显示阈值执行 `score>=6.0` 却有 **4 人**。实际阈值是略大于 6 的浮点数，被四舍五入后改变了可执行含义。

修复先校验一维等长输入、方向、点数和有效有限评分，输出保留实际用于决策的阈值。展示层可以格式化，但计算表不能丢失该值。

## 7. 观察到但未擅自改动的口径

| 项目 | 已确认现状 | 尚需确定的预期 |
|---|---|---|
| 全拒绝后的坏账改善 | 剩余人数为零时安全除法取 0，因此可显示改善 100%；旧测试明确依赖这一行为 | 展示 100%、NaN 还是“不适用”；应与策略通过量一起解释 |
| 单类标签的 KS | `ks([0,0,0],[1,2,3])=0`，分箱表最大 KS 约为 1 | 统一返回 0 哨兵、NaN 或拒绝计算；不能在尚未选定策略时只改一个报告 |
| 灰客户与逾期缺失 | 显式 target 的无效值与 overdue 生成标签采用已文档化但不同的策略 | 特定业务是否把缺失/负 MOB 视为未成熟；不能直接把全项目改为剔除 |
| AUC 自动方向 | 独立 `metrics.auc` 默认 auto；模型概率评估默认 higher_risk | 这是已披露设计差异。固定概率应显式方向，不能把 auto 结果直接当作标准概率 AUC |
| 非单调风险的累计方向 | 按正常箱两端坏率决定从哪端累计，保留箱随后加入 | 是否额外提供按风险排序的累计表；当前表不代表最优拒绝组合 |
| WOE 截断与 IV | 截断后的 WOE 参与 IV 计算 | 是否同时展示未截断 IV 与实际编码 IV，避免一个名称承载两种用途 |
| PSI 箱基准 | 默认 quantile 使用冻结基准；其他部分方法采用两期联合分箱并有 attrs 标识 | 两种数值不能直接跨期比较；长期监控应固定 MonitoringBaseline |

金额与件数、灰客户、日期、方向、阈值的选择应进入报告元数据，而不只留在调用代码中。

## 8. 仍需完善的工程设计

1. **指标语义注册。** `MetricSpec` 已提供部分聚合口径，但“单位、方向、分母、边界策略、累计顺序、精度”尚未统一覆盖全部报告列。应让表格、图和 Excel 共用可查询定义。
2. **测试从存在性扩展到守恒。** 现有真实数据矩阵大量验证合计、列名和文件生成；本次新增逐箱公式、单位金额等价、CDF 终点和实际阈值复现检查，暴露了旧测试漏掉的错误。例如 [`test_bin_stats_formatting.py`](../tests/test_report/test_bin_stats_formatting.py) 的格式化差异检查只打印警告，打印失败文字也不会令 pytest 失败；不应把此类输出当成数值验收。
3. **保留原始精度到输出末端。** 合计和跨表比较应使用未格式化数据；列格式由 Excel/图表决定，不能对中间结果反复 round。
4. **减少独立公式副本。** 本次收敛二分类/金额分箱；规则报告、策略报告、模型报告仍需持续做同口径交叉检查。不能仅因有 `compute_bin_stats` 就假定全库都已完全复用。
5. **错误状态覆盖。** ReportResult 已能记录章节失败，但部分模型摘要和图形兼容分支仍有宽泛降级。应逐条故障注入确认用户能否识别降级；不宜机械删除所有 `except Exception`。
6. **模块拆分与类型检查。** model_report 约 5,700 行、scorecard 约 5,500 行、feature_analyzer 约 4,100 行；应按计算/状态/展示拆分。当前 Makefile 的 mypy 只覆盖三个基础设施文件，并非全库类型证明。
7. **环境可复现。** 当前默认 Anaconda 与验收环境依赖差异显著，需记录解释器、版本、测试集合和跳过原因。源码兼容不能修复旧二进制扩展 ABI；本次没有升级共享环境。
8. **大数据链路。** 已有流式数据库、分块统计与并行预算，但训练、宽表物化、图形与 Excel 导出仍有各自内存边界。970 行真实数据只能验证业务语义，不能证明百万行性能。
9. **更新导航资料。** AGENTS/ROADMAP 的历史测试数、算法数与计划状态应附日期或指向此类实测记录。反事实、折外编码、数据库能力已存在，不能继续整体列为未实现。
10. **基础依赖定位。** 当前基础依赖已包含 SHAP、OR-Tools、PyArrow 等；懒加载减少导入时开销，并不减少安装体积。关于“基础安装轻量”和解释功能可选安装的旧描述需要对照 pyproject.toml，不能把未声明的 `explain` extra 当作当前安装入口。

## 9. 验证与复现

验证数据为本地 `examples/hscredit_yyp.xlsx`：970 行、18 列，FPD 为 834 个好样本和 136 个坏样本。原始工作簿未修改、未加入版本控制，证据只使用聚合结果。

新增独立测试：

- [`test_metric_audit_contracts.py`](../tests/test_binning/test_metric_audit_contracts.py)：单位金额等价、手算风险拒绝比、单类 WOE/IV、精度、KS 累计分母和连续目标矩。
- [`test_rule_metrics_amounts.py`](../tests/test_rules/test_rule_metrics_amounts.py)：金额与人数分离、训练/测试隔离、并行、正类和单类。
- [`test_tradeoff_input_contracts.py`](../tests/test_report/test_tradeoff_input_contracts.py)：非法输入、方向和可执行阈值。
- [`test_metric_audit_real_data.py`](../tests/test_report/test_metric_audit_real_data.py)：4 字段 × 3 标签/剔灰模式 × 2 统计口径，共 24 组逐箱核算，另含 3 组 AUC/KS 核对及规则金额核对。

修复前，最初 17 项合成用例有 16 项失败，28 项真实数据用例有 16 项失败；失败与手算及源码根因一致。输入与正类专项另行复现，不将重复运行累加成独立测试数。

修复后的聚合样例、环境版本与四个变更模块的 SHA256 保存于 [`metric-audit-2026-10-27.json`](validation/metric-audit-2026-10-27.json)。该文件不包含逐行客户明细。

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$auditPython = '.\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
& $auditPython -m pytest tests/test_binning/test_metric_audit_contracts.py tests/test_rules/test_rule_metrics_amounts.py tests/test_report/test_tradeoff_input_contracts.py tests/test_report/test_metric_audit_real_data.py -q
```

### 最终执行结果

| 检查 | 结果 | 范围与限制 |
|---|---|---|
| 默认 Anaconda 环境基线 | 收集阶段 5 个错误 | 旧 statsmodels 的 NumPy 私有路径、CatBoost 二进制 ABI、Numba/SHAP 兼容冲突；未修改共享环境 |
| 兼容验收环境、修复前常规测试 | 6,482 通过、9 跳过、27 排除，903.10 秒 | 产品代码未修改时执行；不含后来新增的审计用例 |
| 新增审计与原累计方向专项 | 66 通过 | 62 个新增用例与 4 个已有用例；不重复加入全套总数 |
| 兼容验收环境、修复后常规测试 | **6,544 通过、0 失败、0 错误、9 跳过、27 排除**，868.43 秒 | `-m "not slow and not integration"`；含新增 62 项和原有真实数据参数/报告导出矩阵 |
| Black | 562 个文件通过 | `black --check hscredit tests` |
| 关键 flake8 | 通过 | `E9,F63,F7,F82,F601`，不称作全量风格检查 |
| mypy | 3 个指定基础设施文件通过 | 与 Makefile 的既有检查范围一致，不代表全库类型完备 |
| Python 3.9 语法 | 4 个变更库模块通过 AST 检查 | 实际运行环境为 Python 3.11.5，未把语法检查称作多版本运行验收 |
| 文档与证据 | 13 个本地链接有效，四个模块源码指纹匹配 | 本次没有重新发布文档网站或构建发行包 |

9 项跳过分别对应 ClickHouse、Hive、Impala、MaxCompute、MongoDB、MySQL、Oracle、Redis、StarRocks 的真实服务配置缺失，详见验证 JSON 的“跳过明细”。最终 pytest 仍报告 630 条警告；测试通过不等于没有依赖/弃用警告。

验收解释器为仓库原有 `.audit_tmp/defect-repairs/runtime/Scripts/python.exe`：Python 3.11.5、NumPy 2.4.6、Pandas 2.3.3、scikit-learn 1.6.1、pytest 8.3.5。运行完成后再次核对四个库模块的 SHA256，确认结果对应最终源码。

完整常规回归复现：

```powershell
& $auditPython -m pytest tests/ -m "not slow and not integration" -q --tb=short --durations=20 --junitxml=.audit_tmp/metric-audit-final-20261027.xml
```

本地日志位于 `.audit_tmp/metric-audit-final-20261027.log`，JUnit 位于同名 `.xml`。初始化说明也已更新到本地 `AGENTS.md`；该文件原本被仓库忽略，本次没有改变其版本控制策略。

## 10. 定义核对来源

项目自身口径以当前源码、公开 docstring 与 [`overdue_labels.md`](overdue_labels.md)、[`analysis_parameters.md`](analysis_parameters.md) 为准。外部来源仅用于核对基础统计定义，不替代本项目业务决策：

- [scikit-learn ROC 曲线](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)：正负类累计比例与评分方向。
- [scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)：固定正类评分的 AUC；本次真实数据独立比对使用该实现。
- [NumPy 标准差](https://numpy.org/doc/stable/reference/generated/numpy.std.html)：`ddof=0` 总体标准差及精度说明；本次连续目标合计采用相同总体矩定义。

“风险拒绝比”按照本项目 `Rule.report` 和二分类统计的明确公式判断，不宣称存在跨机构统一命名标准。
