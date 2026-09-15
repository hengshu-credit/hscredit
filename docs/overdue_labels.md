# 逾期标签与灰客户

使用 `overdue + dpds` 生成标签时，通过 `overdue_operator` 指定比较符。
对一次调用中的所有逾期字段和 DPD 阈值使用相同比较符；满足条件的样本记为 1，其余记为 0。

| overdue_operator | 标签为 1 的条件 | del_grey=True 剔除的灰客户 |
| --- | --- | --- |
| `>` | 逾期天数大于 DPD | `(0, DPD]` |
| `>=` | 逾期天数大于或等于 DPD | `(0, DPD)` |
| `<` | 逾期天数小于 DPD | 无，参数保留供后续扩展 |
| `<=` | 逾期天数小于或等于 DPD | 无，参数保留供后续扩展 |

`del_grey=False` 保留灰客户。多个标签会按各自的 DPD 独立剔灰，分别计算样本数、金额和坏样本率分母。
`<`、`<=` 下，两种 `del_grey` 设置结果相同。
阈值为 0 时，两个灰客户区间均为空；`>= 0` 和 `<= 0` 会把恰好为 0 的样本记为 1。
灰客户只由表中区间确定，负数和缺失值不会因剔灰而被删除；各入口原有的未表现样本处理仍独立生效。

## 用法

```python
from hscredit.report import feature_bin_stats
from hscredit.core.rules import Rule

params = dict(
    overdue="MOB1",
    dpds=[7, 3, 0],
    overdue_operator=">=",
    del_grey=True,
)

bins = feature_bin_stats(data, "评分", method="quantile", margins=True, **params)
rules = Rule("评分 < 600").report(data, margins=True, **params)
```

例如，阈值为 3、逾期天数为 `[0, 1, 3, 4]`：

| 比较符 | 原始标签 | 开启 del_grey 后保留的逾期天数 |
| --- | --- | --- |
| `>` | `[0, 0, 0, 1]` | `[0, 4]` |
| `>=` | `[0, 0, 1, 1]` | `[0, 3, 4]` |
| `<` | `[1, 1, 0, 0]` | `[0, 1, 3, 4]` |
| `<=` | `[1, 1, 1, 0]` | `[0, 1, 3, 4]` |

## 支持入口

- 特征分析：`feature_bin_stats`、`feature_bin_stats_2d`、`feature_binning_summary`、`feature_group_binning_summary`、`benchmark_binning_methods`、`feature_efficiency_analysis`、`auto_feature_analysis`。
- 规则分析：`Rule.report`、`ruleset_analysis`、`rule_group_compare`、`swap_out_report`、`rule_swap_analysis`、`swap_analysis`，以及树分析器的 `report`、`get_rule_table`。
- 模型报告：`ModelReport`、`auto_model_report`、模型的 `report` 方法。
- 逾期预测：`OverduePredictor`、`overdue_prediction_report` 及其兼容别名。
- EDA：`bad_rate_overall`、`bad_rate_by_dimension`、`bad_rate_trend`、`bad_rate_by_bins`。
- 绘图：`bad_rate_trend_plot`、`distribution_plot`、`bin_overdues_plot`。

`feature_efficiency_analysis` 继续使用单阈值参数 `dpd`；其他支持多阈值的报告继续使用 `dpds`。
绘图中的 `distribution_plot`、`bin_overdues_plot` 保留逾期列与阈值一一对应的方式，其他入口保留各自的组合方式。
传入现成标签、现成分箱表的路径不会重新定义标签。

## 默认值、配置优先级和表头

大多数接口默认使用 `>`。`distribution_plot` 和 `bin_overdues_plot` 历史默认值为 `>=`，本次保留；
如需全流程采用同一比较方式，请显式传入 `overdue_operator`。

`auto_feature_analysis`、`OverduePredictor`、`rule_swap_analysis` 支持从 `bin_params` 读取比较符，显式参数优先。
分箱方法对比中的 `feature_binning_summary`、`feature_group_binning_summary` 使用顶层比较符统一所有方法的标签定义。
`feature_efficiency_analysis` 的手工与自动分箱共同使用顶层 `overdue_operator` 和 `del_grey`，`auto_kwargs` 不覆盖标签口径。
`ModelReport` 及自动模型报告支持在 `target` 字典中配置，显式参数优先：

```python
target = {"overdue": "MOB1", "dpds": [7, 3], "overdue_operator": ">="}
```

`>` 保留已有的 `MOB1_7+`、`MOB1 7+`、`MOB1>7`、`MOB1@7` 等表头。
其他比较符显式显示为 `MOB1>=7`、`MOB1<7`、`MOB1<=7`，便于识别标签含义。
使用 `target_names`、预测系数或现成分箱表时，应使用对应比较符的标签名称。
整数值阈值统一命名，例如 `3.0` 与 `3` 都生成 `MOB1>=3`；非整数小数保留原值。
恢复本次扩展之前保存的模型报告或逾期预测器时，继续使用原有的 `>` 定义。
通过 hsbin / hsreport 生成多标签分箱或自动特征报告时，运行摘要中的 `overdue_operator` 记录实际比较符，与 `label_combinations` 一起说明标签定义。
不支持的比较符会抛出中文 `ValueError`。
