# 统一标签输入与灰客户

所有原先支持 overdue+dpds 的分析入口都接受两种互斥模式：

- `target`：列名、直接的一维标签数据、callable，或 `{标签名称: 标签数据或 callable}`。
- `overdue + dpds`：独立指定逾期字段和阈值，按 `overdue_operator` 生成标签。

显式混用两种模式会报错。`target` 字典不再解释逾期配置。公开标签参数统一为 `target`，
原 `target_col` 已移除；效率分析的原 `dpd` 改为 `dpds`。阈值支持标量或非空序列。
显式进入逾期模式时，`overdue` 与 `dpds` 必须同时提供。

## 适用入口

以下公开入口统一提供 `target`、`overdue_operator` 和 `del_grey`。类配置适用于后续拟合或报告。

| 范围 | 入口 |
| --- | --- |
| 特征分析 | `feature_bin_stats`、`feature_bin_stats_2d`、`feature_binning_summary`、`feature_group_binning_summary`、`benchmark_binning_methods`、`feature_efficiency_analysis`、`auto_feature_analysis` |
| 规则与策略 | `Rule.report`、`ruleset_analysis`、`rule_group_compare`、`swap_out_report`、`rule_swap_analysis` |
| 逾期预测与置换 | `OverduePredictor`、`overdue_prediction_report`、`swap_analysis` |
| 模型报告 | `ModelReport`、`auto_model_report`、`BaseRiskModel.report`、`LogisticRegression.report`、`ModelWorkflowMixin.model_report`、`ModelWorkflowMixin.report` |
| 树规则报告 | `DecisionTreeAnalyzer.report/get_rule_table`、`ManualTreeExtractor.report/get_rule_table` |
| EDA | `feature_summary`、`DataFrame.summary`、`Series.summary`、`bad_rate_overall`、`bad_rate_by_dimension`、`bad_rate_trend`、`bad_rate_by_bins` |
| 绘图 | `distribution_plot`、`bin_overdues_plot`、`bad_rate_trend_plot` |

继承统一模型工作流的组件可在 `model.model_report(...)` 中直接传入上述参数，IDE 与 `help()` 可查看完整签名。
`ProbabilityScoreCard` 和 `ProbabilityCalibrator` 的专用 `report()` 保持其原有用途，完整模型报告使用 `model_report()`。

## target 模式

```python
from hscredit.report import feature_bin_stats

single = feature_bin_stats(data, "评分", target="FPD")
direct = feature_bin_stats(data, "评分", target=data["FPD"].to_numpy())
computed = feature_bin_stats(data, "评分", target=lambda df: df["MOB1"] > 7)
multiple = feature_bin_stats(
    data, "评分",
    target={
        "首逾": data["FPD"],
        "七天逾期": lambda df: df["MOB1"] > 7,
        "自定义标签": custom_labels,  # Series、ndarray 或 list
    },
    del_grey=True, margins=True,
)
```

字典键决定输出名称和顺序，`xxx_7+`、`xxx@7` 不会被逾期命名规则改写。
`分箱详情` 是表头保留名。字典值不能是列名字符串，需读取列时请传 Series 或 callable。

| 设置 | target 中非缺失、非 0/1 值的处理 |
| --- | --- |
| `del_grey=False` | 按负样本 0 计算 |
| `del_grey=True` | 仅从当前标签的有效样本中剔除 |

数值 1/True 为正样本，0/False 为负样本，其余非缺失值为灰客户。该规则不受 `overdue_operator` 影响。
原始标签缺失由 `handle_missing_target` 单独处理。默认 `"drop"` 从当前标签的指标中剔除该行，`"positive"` 计为 1，`"negative"` 计为 0，`"raise"` 遇到缺失即报错。
各标签的样本数、金额、坏率和模型指标使用各自分母，不能把某标签的灰客户从其它标签同时删除。

金额的数值校验针对各标签的有效人群，已剔除行不会混入当前标签的分母。结果表与报告不额外输出指标状态或原因说明。

## callable 的作用数据与对齐

统一调用 `fn(data)`，传入与该入口生成逾期标签时相同的作用 DataFrame，保留原始字段、行顺序和索引。
参数名不参与字段推断。每个标签在每份数据上求值一次，后续分组、规则筛选、特征过滤和并行任务只切片已绑定标签。
callable 接收独立 DataFrame 副本，常规列修改不会污染调用方或其它标签。

```python
from functools import partial

def make_label(data, *, threshold):
    return data["MOB1"].gt(threshold)

target = partial(make_label, threshold=7)
target = lambda df, threshold=7: df["MOB1"].gt(threshold)
```

返回值必须是等长的一维 `Series`、`ndarray` 或 `list`。唯一索引的 Series 按完整索引对齐；
重复索引必须完全同序。数组和列表按位置对应。长度不一致、索引覆盖不完整、二维结果或标量均报错，
不广播、不截断、不取索引交集。函数内部错误保留异常链，不更换参数重试。

比较表达式可能已将原始缺失变成 False。需要按缺失策略处理未表现样本时，应让 callable 显式返回缺失值。

```python
target = lambda df: df["MOB1"].gt(7).astype(float).where(df["MOB1"].notna())
```

模型报告的各数据集分别执行 callable，追加数据集只解析新增数据。
置换分析的参考集和分析集分别生成标签。跨数据集建议使用 callable；静态数组必须确实与各自作用数据逐行对应。
`ModelReport` 的显式外部 `y` 保留原有优先级。

`rule_swap_analysis` 未指定参考集时，直接复用分析集已经生成的标签；callable 只执行一次，
重复业务索引归一化前先完成标签绑定。参考坏率逐标签排除未表现行，不会因为首个标签缺失而删除其它标签的有效样本。
分析集中的未表现样本仍遵循置换规则：关闭剔灰时，仅 OUT-IN 样本可由参考坏率估计表现。
开启剔灰时，只过滤风险统计中的无效实际表现；审批决策仍保留所有输入样本，
OUT-IN 使用预测表现，不能因实际标签缺失而被删除。
`决策样本数` 与各阶段通过率跨标签共享，`样本总数` 则为各标签的有效风险样本数。
完整计算与单位见 [规则置换通过率口径](rule_swap_approval.md)。

## overdue + dpds 模式

```python
params = dict(overdue=["MOB1", "MOB3"], dpds=[7, 30], overdue_operator=">", del_grey=True)
table = feature_bin_stats(data, "评分", margins=True, **params)
```

| overdue_operator | 正样本条件 | del_grey=True 剔除的灰客户 |
| --- | --- | --- |
| `>` | 逾期天数大于 DPD | `(0, DPD]` |
| `>=` | 逾期天数大于或等于 DPD | `(0, DPD)` |
| `<` | 逾期天数小于 DPD | 无 |
| `<=` | 逾期天数小于或等于 DPD | 无 |

不剔灰时，满足比较条件为 1，其余为 0。阈值为 0 时灰区为空；`>=0` 和 `<=0` 包含恰好为 0 的样本。
负数和缺失不会因上述灰区被删除；置换分析对未表现样本的独立处理继续保留。

大多数入口默认 `>`；`distribution_plot`、`bin_overdues_plot` 保留默认 `>=`。
后两者保留逾期列与阈值一一对应，其余组合式入口按 `overdue × dpds` 展开。
`3.0`、`"3.0"` 和 `3` 规范为同一阈值；组合模式去重阈值，配对模式保留各字段对应阈值。

## 输出与参数优先级

(feature-summary-labels)=
### 单标签与多标签特征摘要

`feature_summary(data, ...)` 与 `data.summary(...)` 使用同一套统计口径。无标签或实际只有一个标签时返回单层列。多个标签时，每个特征占一行，一级表头依次为“统计详情”、各标签、“分布详情”。

```python
from hscredit import feature_summary, hscredit_dataset

data = hscredit_dataset(columns=["age_year", "DK_X1995_2", "MOB3", "MOB6"])
features = ["age_year", "DK_X1995_2"]
single = data.summary(features=features, overdue="MOB3", dpds=30)
multi = feature_summary(data, features=features, overdue=["MOB3", "MOB6"], dpds=30, fillna=-99999)
mob3 = multi["MOB3@30"][["IV", "KS", "AUC", "趋势"]]
```

样本数、特征缺失率、PSI 和分位数按完整特征样本计算一次。IV、KS、AUC、趋势及自动训练的模型重要性分别使用各标签的有效样本。`fillna` 仅填补 KS 和 AUC 的数值特征缺失值。默认比较符 `>` 的分组名如 `MOB3@30`，其他比较符直接显示为 `MOB3>=30` 等。

已有标签列可以传 `target="是否逾期"` 或 `target=["三期逾期", "六期逾期"]`。需要自定义名称时使用标签字典。使用 callable 时可显式指定 `features`，选取授信时点的特征。旧参数 `y` 保留按位置匹配单标签向量的行为。

分开保存 X 和逾期天数时，可直接传 `X.summary(overdue=days[["MOB3", "MOB6"]], dpds=30)`。单特征也可用 `X["age_year"].summary(overdue=days["MOB3"], dpds=30)`。外部逾期数据按索引对齐。

使用 `summary(return_type="dict")` 时，每个特征返回一条记录。多标签记录使用 `("MOB3@30", "KS")` 这样的二元组键，单标签记录沿用 `"KS"` 等字符串键。

### 其他报告

- 分箱和规则复用现有多级表头；长表继续使用 `逾期标签` 列，自定义名称原样显示。
- 公共分箱在首个可拟合标签上学习一次，其余标签复用分箱重新统计。某标签缺少的箱不会导致其它标签丢箱。
- 模型报告单独查看只有一个类别的标签时，若同一数据集有其它可拟合标签，评分及特征分箱复用其箱边界，统计仍使用当前标签；没有可拟合标签时不虚构训练标签。
- 自定义多标签的单标签视图默认使用首标签，不自动做 OR。原逾期多标签的 combined 继续使用 OR，并排除全部标签无效的行。
- 效率分析支持多个标签，返回 `targets`、`comparison_figures` 及带标签列的 `manual_table/auto_table` 汇总长表。
- `auto_feature_analysis`、`OverduePredictor`、`rule_swap_analysis` 的显式 `del_grey/overdue_operator` 优先于 `bin_params`，缺省时沿用配置。
- 分箱方法汇总、分组汇总及效率分析的顶层标签定义和剔灰口径统一作用于各方法；算法参数不能改写标签。
- 模型训练和选择器的 `fit(X, y)` 继续接收单标签，本次扩展用于分析和报告。

## 技能接口与迁移

技能支持列表标签和同进程对象引用。`inputs.target` 可引用整个定义；也可在 `parameters.target`
字典值中用 `{"input": "输入名称"}` 引用数组、Series 或 callable。
JSON 不执行 lambda 字符串；Python callable 通过 `object_ref` 在 `environment.mode=current` 下传入。
显式标签摘要记录 `target_mode/target_names`；逾期模式记录 `overdue_operator/label_combinations`。
两种模式均记录 `del_grey`，沿用与执行入口一致的显式参数与 `bin_params` 优先级；重复标签组合只记录一次。

旧的 `target={"overdue": "MOB1", "dpds": [7, 30]}` 需拆成独立参数；删除同时指定的 `target`。
将 `target_col`、`dpd` 关键字改为 `target`、`dpds`，省略的阈值改为显式 `dpds=0`。
可执行示例见 `examples/29_target_inputs.ipynb`。
