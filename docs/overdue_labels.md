# 统一标签输入与灰客户

所有原先支持 overdue+dpds 的分析入口都接受两种互斥模式：

- `target`：列名、直接的一维标签数据、callable，或 `{标签名称: 标签数据或 callable}`。
- `overdue + dpds`：独立指定逾期字段和阈值，按 `overdue_operator` 生成标签。

显式混用两种模式会报错。`target` 字典不再解释逾期配置。公开标签参数统一为 `target`，
原 `target_col` 已移除；效率分析的原 `dpd` 改为 `dpds`。阈值支持标量或非空序列。
显式进入逾期模式时，`overdue` 与 `dpds` 必须同时提供。

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

| 设置 | target 中非 0/1 值（包含缺失）的处理 |
| --- | --- |
| `del_grey=False` | 按负样本 0 计算 |
| `del_grey=True` | 仅从当前标签的有效样本中剔除 |

数值 1/True 为正样本，0/False 为负样本；其它值为灰客户。该规则不受 `overdue_operator` 影响。
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

比较表达式可能已将原始缺失变成 False；需要标记未表现样本时应显式返回缺失，并开启剔灰：

```python
target = lambda df: df["MOB1"].gt(7).astype(float).where(df["MOB1"].notna())
```

模型报告的各数据集分别执行 callable，追加数据集只解析新增数据。
置换分析的参考集和分析集分别生成标签。跨数据集建议使用 callable；静态数组必须确实与各自作用数据逐行对应。
`ModelReport` 的显式外部 `y` 保留原有优先级。

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

旧的 `target={"overdue": "MOB1", "dpds": [7, 30]}` 需拆成独立参数；删除同时指定的 `target`。
将 `target_col`、`dpd` 关键字改为 `target`、`dpds`，省略的阈值改为显式 `dpds=0`。
可执行示例见 `examples/29_target_inputs.ipynb`。
