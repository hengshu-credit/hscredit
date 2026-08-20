# HSCredit 模型可解释性增强设计

## 1. 背景与目标

HSCredit 已有 `ModelExplainer`、基础 SHAP 图、原生特征重要性、逻辑回归统计量、评分卡原因和规则命中原因，
但当前 SHAP 仍是可选依赖，解释结果以易失的二维数组缓存为主，类别、输出尺度、样本身份和背景数据等元信息
没有统一保存；`ModelReport` 也没有把 SHAP 全局解释和单样本解释纳入正式报告。

本次改造目标是建立一套面向信贷模型的完整解释体系：

1. 将 SHAP 变为基础依赖，安装 HSCredit 后即可执行全局和单样本解释；
2. 用结构化解释结果统一管理贡献值、样本、类别、输出尺度、背景数据和计算方法；
3. 补齐全局重要性、相关性、聚类、交互、稳定性、代表样本和单样本分析；
4. 将解释能力接入 `ModelReport`，生成中文表格和图片；
5. 在 SHAP 贡献之外提供可审计的业务原因码和受约束反事实建议；
6. 保留现有公开方法，修复其类别、缓存、交互聚合和中文输出问题。

参考材料包括：

- [SHAP 模型解释与可视化](https://mp.weixin.qq.com/s/1Y7LNZfO7S3tLbGjXGT-6A)
- [SHAP 高级分析与稳定性](https://mp.weixin.qq.com/s/ARET-vlvZcUOxRPAXn9I2w)
- [组合重要性、相关性与交互分析](https://mp.weixin.qq.com/s/pLbBdDyrYH8Z9VzcpIRWdw)
- [多纵轴分布图参考](https://mp.weixin.qq.com/s/VxrsojDPlX1jPQt8akeVbA)

第四篇参考文章主要讨论多纵轴直方图和正态拟合，不属于模型解释主链路，本次不移植该图形。

## 2. 明确范围与非目标

### 2.1 本次实现

- SHAP 基础依赖和 Python 版本兼容矩阵；
- 统一解释结果与解释器工厂；
- 全局、局部、相关性、聚类、交互和稳定性分析；
- 代表样本选择与批量到单样本的结构化下钻数据；
- 中文可视化和综合解释总览；
- `ModelReport` 可选解释页；
- 通用模型与评分卡原因码；
- 无额外第三方解释依赖的受约束反事实解释；
- 公共导出、文档、示例、单元测试和真实数据验证；
- NumPy 2.x 下类别分箱排序键溢出的前置兼容修复。

### 2.2 明确不实现

- 不保留空的 `explain = []`，`explain` extra 直接删除；
- 不引入 LIME，也不提供 LIME 兼容层；
- 不内置 Streamlit、Web 服务或交互式前端；结构化 API 应足以供外部页面下钻；
- 不把 SHAP 或反事实建议表述为因果结论、授信结论或监管合规证明；
- 不实现第四篇文章中与解释无关的多纵轴正态拟合图。

## 3. 依赖与兼容策略

### 3.1 SHAP 作为基础依赖

在 `project.dependencies` 中新增带 Python 标记的 SHAP 约束：

```toml
"shap>=0.49.1,<0.50; python_version < '3.14'",
"shap>=0.51,<0.53; python_version >= '3.14'",
```

从 `[project.optional-dependencies]` 中删除 `explain` 项，并从 `all` extra 的自引用列表中删除 `explain`。
README、安装文档和 API 文档中所有 `hscredit[explain]` 说明同步删除。

SHAP 虽然是基础安装依赖，仍在代码层使用延迟导入，避免 `import hscredit` 时立即加载 SHAP、Numba 和绘图库。
SHAP 加载失败时保留中文依赖兼容错误，不再提示用户安装 `hscredit[explain]`。

### 3.2 Python 与 NumPy

项目继续支持 Python 3.9+。Python 3.14 使用支持该解释器和 NumPy 2 的较新 SHAP 分支；较低 Python 版本使用
兼容 Python 3.9 的 SHAP 0.49.x。

为使 Python 3.14 / NumPy 2 验证可执行，本次一并修复 `compute_bin_stats` 的类别分箱排序键：在执行
`bin_index * 10000` 前将 NumPy 标量显式转换为 Python `int`。该修复只消除整数溢出，不改变分箱排序语义。

## 4. 分层架构

解释能力分为三层，三者不得混用概念：

```text
模型贡献层：SHAP 全局/局部贡献、交互、相关性、稳定性
    ↓ 提供候选驱动
业务原因层：风险方向过滤、业务名称映射、原因码、审计字段
    ↓ 结合可变约束
行动建议层：满足目标且遵守约束的反事实候选
```

- 模型贡献层回答“模型为什么给出该输出”；
- 业务原因层回答“哪些不利因素可以作为本次决策的规范化原因”；
- 行动建议层回答“在模型和给定约束下，哪些最小变更可能达到目标”。

所有输出中均注明模型输出尺度和目标类别。反事实输出额外注明“非因果建议”。

## 5. 模块组织与公共导出

保留 `hscredit.core.models.evaluation.interpretability` 作为现有导入路径和兼容门面，并按职责拆分新增模块：

```text
hscredit/core/models/evaluation/
├── explanation.py        # ExplanationResult 和规范化工具
├── explainer.py          # ModelExplainer、解释器选择和结构化分析
├── explanation_plots.py  # 中文 SHAP 图和综合总览
├── reason_codes.py       # 通用模型原因码
├── counterfactual.py     # CounterfactualExplainer
└── interpretability.py   # 旧 API 兼容包装及轻量非 SHAP 报告
```

以下对象从 `hscredit.core.models.evaluation` 和 `hscredit.core.models` 公开导出；顶层 `hscredit` 继续沿用项目的
懒加载导出机制：

```python
ModelExplainer
ExplanationResult
CounterfactualExplainer
model_explain_report
plot_feature_importance
plot_shap_importance
plot_importance_comparison
```

不存在的文档路径 `hscredit.core.models.interpretability` 全部修正为真实路径。旧的
`hscredit.core.models.evaluation.interpretability` 导入继续有效。

## 6. 统一解释结果

新增只读语义的 `ExplanationResult` 数据类，核心字段为：

```python
@dataclass
class ExplanationResult:
    explanation: "shap.Explanation"
    data: pd.DataFrame
    sample_ids: pd.Index
    target_class: Any
    output_index: Optional[int]
    model_output: str
    explainer_type: str
    background_summary: dict
    dataset_fingerprint: str
    metadata: dict
```

它提供 `values`、`base_values`、`feature_names` 等只读便捷属性，并保证选定类别后的 `values` 统一为
`(样本数, 特征数)`。多输出 SHAP 的 list、三维数组和新式 `Explanation` 都在一个规范化入口处理。

元信息至少包含模型类型、SHAP 版本、解释算法、计算时间、随机种子、样本数、特征数、特征顺序、数据类型、
目标类别、请求输出尺度、实际输出尺度和近似算法参数。输出尺度只能是明确记录的 `raw`、`probability` 或
模型特定 `score`；当某个解释器不能原生满足请求尺度时必须报错或记录实际尺度，禁止静默错标。

数据指纹包含列顺序、数据类型、索引和值哈希。绘图和旧式缓存只复用指纹完全一致的结果，解决不同数据集间
误复用旧 SHAP 值的问题。

## 7. ModelExplainer API

### 7.1 构造与计算

主要接口为：

```python
explainer = ModelExplainer(
    model,
    background_data=X_background,
    feature_names=None,
    algorithm="auto",
    model_output="probability",
    target_class=1,
    max_background=200,
    random_state=42,
)

result = explainer.explain(X_test)
```

规则如下：

- 二分类默认解释标签值 `1`；如 `classes_` 不含 `1`，使用正类位置并在元信息中记录真实标签；
- 多分类必须显式提供 `target_class`，避免任意选择第二类；
- DataFrame 保留原始索引、列名、列顺序和数据类型；数组必须提供特征名或生成中文占位名；
- 背景数据采用固定随机种子抽样，抽样行号和摘要写入元信息；
- `max_samples`、`max_evals` 等成本上限公开配置，超过上限时确定性抽样并记录；
- 默认不在同一调用中隐式重训模型。

### 7.2 解释器选择

`algorithm="auto"` 按以下顺序选择：

1. 可直接接收原始入参且支持请求输出尺度的树模型使用 `TreeExplainer`；概率尺度需要背景数据时，使用显式背景
   数据，或从本次解释数据中确定性抽样并记录背景来源；
2. 可直接接收原始入参的线性模型在 raw / log-odds 尺度使用 `LinearExplainer`；请求 probability 尺度时改用
   模型公开概率函数和 `PermutationExplainer`，避免把 log-odds 贡献误标成概率贡献；
3. sklearn Pipeline、带内部预处理的 HSCredit 包装器或未知模型使用模型公开预测函数和
   `PermutationExplainer`，保证贡献落在原始输入特征；
4. `KernelExplainer` 仅作为显式 `algorithm="kernel"` 的兼容选项，不再作为未知模型默认项。

若用户强制选择不适用的算法，抛出包含模型类型、算法和原因的中文错误。解释器不得通过宽泛
`except Exception` 静默降级。

### 7.3 兼容接口

以下现有方法保留，并在内部调用 `explain()`：

- `compute_shap_values()`：继续返回选定类别的二维 NumPy 数组；
- `get_shap_importance()`：继续返回 Series，但名称改为中文并保持确定性排序；
- 现有 summary、bar、violin、dependence、force、waterfall、combined importance 和 interaction API；
- `model_explain_report()` 继续提供不依赖 SHAP 计算的轻量原生重要性报告。

兼容方法的参数和返回类型原则上不变；新增参数均提供默认值。明显错误的类别选择、陈旧缓存和交互总和语义
作为缺陷修复，不保留旧行为。

## 8. 结构化分析能力

### 8.1 全局解释报告

`get_global_report(result)` 返回中文 DataFrame，至少包含：

- 特征、平均绝对 SHAP 值、SHAP 重要性占比、平均 SHAP 值；
- 正向影响占比、负向影响占比、影响标准差和主要分位数；
- 原生特征重要性、原生排名、SHAP 排名和排名差；
- 特征值与自身 SHAP 值的 Pearson / Spearman 相关系数。

原生重要性不可用时保留对应列并填充 `NaN`，不得用 SHAP 值冒充原生重要性。

### 8.2 单样本解释

`get_sample_report(result, sample_id=..., top_n=...)` 返回长表，包含样本索引、目标类别、模型输出、基准值、
特征、特征值、SHAP 值、绝对贡献、贡献方向、累计贡献和排名。输出同时支持 sample_id 和位置索引，二者冲突时
报中文错误。

### 8.3 代表样本

`select_representative_samples(result, threshold=0.5)` 至少选择：

- 最高风险、最低风险；
- 最接近决策阈值；
- 最接近总体中位输出；
- 最不确定样本；
- 总绝对贡献最大样本。

重复样本去重并合并选择理由。返回表包含样本索引、选择理由、模型输出、风险排名和阈值距离，可直接传给
单样本报告和绘图方法，形成批量到样本的下钻数据链路。

### 8.4 相关性、聚类与交互

新增：

- `get_correlation_report(result, kind="feature_shap")`：特征值与自身 SHAP 相关性；
- `get_correlation_report(result, kind="shap_shap")`：SHAP 值之间的相关矩阵；
- `get_feature_clusters(result)`：基于 SHAP 贡献相关性生成可复现层次聚类和叶序；
- `get_feature_interactions()`：树模型精确交互；
- `get_approximate_interactions()`：非树模型基于 SHAP 依赖的近似交互排序。

交互强度使用样本均值而不是样本总和，使不同样本量报告可比较。交互表使用无重复特征对、中文列名，并区分
主效应与交互效应。

### 8.5 稳定性

`get_stability_report()` 明确支持两种不可混淆的模式：

- `mode="sample"`：对固定解释结果做样本 Bootstrap，评估平均绝对 SHAP、置信区间、排名均值、排名标准差、
  Top-K 入选率；该模式不重训模型；
- `mode="refit"`：clone 模型并对训练集 Bootstrap 重训，在固定验证集上重新解释，评估模型重训后的解释稳定性。

`refit` 模式要求提供训练数据、标签和可 clone 模型；不满足条件时清晰报错，不自动退化为 sample 模式。
两种模式均接受 `n_bootstrap`、`confidence_level`、`top_k`、`random_state` 和成本上限，并在输出表中写明模式。

## 9. 中文可视化

所有图函数接受 `ExplanationResult`，也允许兼容式传入 `X` 后显式计算。新增：

- 决策路径图 `plot_decision()`；
- 多样本热力图 `plot_heatmap()`；
- 单特征 SHAP 分布图 `plot_distribution()`；
- SHAP 相关性热力图 `plot_correlation()`；
- 特征层次聚类图 `plot_feature_clustering()`；
- 交互强度热力图和气泡图；
- 组合小提琴图 + 重要性条形图 `plot_importance_overview()`；
- 包含全局重要性、贡献方向、相关性和代表样本的 `plot_explanation_overview()` 综合总览。

现有 summary、bar、violin、dependence、force、waterfall 和 combined importance 同步改为中文标题、坐标轴和图例。
函数返回 Matplotlib Figure/Axes 或 SHAP 可视对象，`show=False` 时不产生不可控副作用，便于 Excel 报告复用。

## 10. 业务原因码

### 10.1 通用模型

`ModelExplainer.get_reason_codes()` 从局部贡献中筛选风险方向一致的因素，而不是简单按绝对值取前 N 个。
调用方必须声明或使用模型适配器给出的风险方向：

- `higher_output_higher_risk`：正 SHAP 是不利贡献；
- `higher_output_lower_risk`：负 SHAP 是不利贡献。

支持业务字段名、原因码和原因描述映射，并返回中文长表：样本索引、原因排名、特征、特征值、SHAP 值、风险贡献、
原因码、原因描述、目标类别、输出尺度和风险方向。没有不利贡献时返回空原因集合并保留样本审计记录，不拿有利因素
凑满数量。

### 10.2 评分卡

`ScoreCard` 和 `RoundScoreCard` 新增结构化 `get_reason_codes()`，按各特征实际得分相对基准贡献筛选真正导致风险升高
或分数降低的因素。现有 `get_reason()` 保留原返回形态以兼容下游，但内部复用正确的方向排序，不再使用绝对偏差把
有利因素当作不利原因。RoundScoreCard 使用实际取整后的分值，确保原因与最终计分一致。

## 11. 反事实解释

新增 `CounterfactualExplainer`，不引入新解释依赖：

```python
counter = CounterfactualExplainer(
    model,
    reference_data=X_reference,
    constraints={
        "年龄": {"mutable": False},
        "收入": {"min": 0, "direction": "increase_only"},
    },
    random_state=42,
)

counter.generate(X_one, target_probability=0.20, max_changes=3, top_n=5)
```

搜索策略分为：

- 评分卡：枚举可变特征的可达分箱，以真实分箱分数变化做确定性组合搜索；
- 通用表格模型：从参考数据的数值分位点和类别取值构造候选，使用确定性束搜索寻找满足目标的最小变更。

约束支持不可变特征、数值上下界、允许类别、只增/只减方向、单特征成本权重和最大变更特征数。默认距离为按参考
数据尺度归一化后的加权距离；类别变化计固定成本。

输出中文表包含方案编号、是否达标、变更特征数、总成本、预测前值、预测后值、目标值、特征、原值、新值、变化方向、
约束检查和说明。找不到可行方案时返回带失败原因的结构化结果，不伪造建议。所有结果注明“模型条件下的非因果建议”。

## 12. ModelReport 集成

`ModelReport` 构造或 `to_excel()` 增加显式解释配置，默认关闭高成本计算：

```python
explain_config = {
    "enabled": True,
    "data": X_explain,
    "background_data": X_background,
    "target_class": 1,
    "model_output": "probability",
    "max_samples": 500,
    "representative_count": 6,
    "stability_mode": "sample",
    "n_bootstrap": 100,
}
```

启用后在现有 1–6 页之后追加 `7-模型解释`，不改变已有工作表名称和编号。该页包含：

1. 解释范围、算法、类别、尺度、样本与背景数据元信息；
2. 全局 SHAP 重要性表和组合重要性图；
3. 相关性、聚类和主要交互表/图；
4. 指定模式的稳定性表；
5. 代表样本清单、单样本贡献表和 waterfall/decision 图；
6. 原因码表及“模型贡献不等于因果或审批依据”的说明。

解释计算失败时默认抛出带阶段上下文的中文异常，不能用宽泛捕获静默生成缺页报告；只有显式
`on_explain_error="warn"` 时才记录警告并在解释页写入失败原因。`to_dict()` 在启用解释时新增结构化 `模型解释`
节点，未启用时保持现有结构。

ModelReport 不默认执行 `refit` 稳定性和反事实批量搜索；用户显式配置时才加入，避免报告生成时间失控。

## 13. 性能、确定性与错误处理

- 背景数据、解释样本、Bootstrap 和候选搜索统一使用公开 `random_state`；
- 默认背景样本 200、解释样本上限 500，报告中记录抽样；
- 非树模型公开 `max_evals` 和并行配置，默认值按特征数计算并设置上限；
- 绘图复用已计算的 `ExplanationResult`，不重复执行模型预测；
- 大矩阵相关性、聚类和热力图按解释样本上限运行；
- 输入缺列、列顺序错误、目标类别不存在、输出尺度不支持、模型未拟合、约束矛盾等均抛中文的项目异常；
- 不新增静默的 `except Exception: pass`；需要转换外部异常时保留原异常链和阶段信息。

## 14. 测试策略

### 14.1 单元与契约测试

1. SHAP 不再使用 `pytest.importorskip`，基础安装环境必须直接运行解释测试；
2. RandomForest、GradientBoosting、LogisticRegression、sklearn Pipeline 和 HSCredit 包装模型均可解释；
3. 二分类类别标签、三维多输出和现代 `shap.Explanation` 规范化正确；
4. 多分类未指定类别时报错，指定类别后贡献维度和元信息正确；
5. `base_value + sum(SHAP)` 与记录的目标输出尺度满足允许误差；
6. DataFrame 索引、中文列名、列顺序和类型保留；
7. 不同数据集不复用旧缓存，相同指纹可安全复用；
8. 全局、单样本、代表样本、相关性、聚类和交互表列名与排序稳定；
9. sample/refit 两种稳定性模式语义、置信区间和错误路径正确；
10. 原因码只选择不利贡献，不用有利贡献补位；
11. ScoreCard 与 RoundScoreCard 原因和实际分值方向一致；
12. 反事实遵守不可变、边界、类别、方向和最大变更数，找不到方案时有明确结果；
13. 新旧绘图在无界面后端下 smoke test 通过且 `show=False`；
14. `ModelReport` 仅在启用时追加 `7-模型解释`，现有 1–6 页不变；
15. 公共导入路径、旧方法返回类型和 pickle/artifact 基本契约不破坏。

### 14.2 依赖矩阵

构建验证至少覆盖：

- Python 3.9 + SHAP 0.49.x；
- 项目主支持 Python + 锁定分支最高 SHAP；
- Python 3.14 + SHAP 0.51/0.52 + NumPy 2。

构建产物元数据中不得再出现 `hscredit[explain]`，普通 `pip install hscredit` 必须解析并安装 SHAP。

### 14.3 真实数据验证

按仓库约定使用 `examples/hscredit_yyp.xlsx`：

- 目标列使用 `FPD`；
- 单特征使用 `衡枢鉴真分老客版`；
- 多特征使用 `衡枢鉴真分老客版`、`近六个月非银多头机构数`、`青云24`；
- 验证全局报告、代表样本、单样本原因码、反事实和带解释 Excel 报告；
- 确认模块导入、输出中文、图片可渲染和工作簿可重新打开；
- 完整非 slow、非 integration 回归不新增失败，并确认 NumPy 2 类别分箱已知失败被修复。

## 15. 验收标准

1. 基础安装含兼容版本 SHAP，`explain` extra 完全不存在；
2. 项目不新增任何 LIME 依赖、导入、API 或文档；
3. `from hscredit.core.models import ModelExplainer, ExplanationResult, CounterfactualExplainer` 可用；
4. 全局和单样本解释无需额外安装即可执行，类别和输出尺度可审计；
5. 文章涉及且属于模型解释范围的决策图、热力图、分布、聚类、相关性、交互、稳定性和综合视图均有 API；
6. `ModelReport` 可追加完整中文解释页，默认不开启高成本解释；
7. 原因码只陈述模型中的不利驱动，反事实严格遵守约束并标注非因果；
8. 现有解释方法和现有报告 1–6 页保持兼容；
9. 单元测试、回归测试、构建检查和真实放款数据验证均通过或只保留与本次无关的已知失败。
