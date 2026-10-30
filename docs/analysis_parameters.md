# 分析参数约定与迁移

报告、EDA、规则挖掘和绘图接口统一使用下面的参数名称。旧名称需要在调用代码中同步迁移。

| 含义 | 参数 |
| --- | --- |
| 单字段 | `feature` |
| 字段集合 | `features`，支持单字段自动转为列表 |
| 二维字段 | `features=["a", "b"]` 或 `x1="a", x2="b"` |
| 分箱方法、最大箱数 | `method`、`max_n_bins` |
| 多方法比较 | `methods` |
| 扩展分箱配置 | `bin_params` |
| 时间、分组、金额 | `date`、`group`、`amount` |
| 日期频率 | `freq`，默认按具体入口约定，通常为 `"M"` |
| 绘图尺寸、字号、方向、保存路径 | `figsize`、`fontsize`、`orientation`、`save` |
| 图片保存分辨率 | `dpi` |
| 绘图入口返回统计数据 | `result` |
| 树/策略报告工作表名 | `sheet_name` |
| 树/策略报告保存路径 | `save`；模型一键报告仍按该入口签名使用 `excel_path`，二者不互作别名 |

## 字段选择

`features="score"` 与 `features=["score"]` 等价，也支持一维 tuple、NumPy 数组和 pandas Index。已有整数列名仍然可用。`feature_bin_stats` 和分箱汇总同时提供 `feature` 与 `features` 两种入口，两者不能同时指定。

二维分箱器、二维统计、二维绘图和交叉规则接口支持 `features=np.array(["score", "age"])`，也支持 `x1="score", x2="age"`。两个字段必须不同；`x1/x2` 必须成对出现，不能与 `features` 混用。

规则挖掘器的构造和 `fit` 使用 `features`。模型报告的公开字段入参也使用 `features`；sklearn 的 `feature_names_in_` 和绘图使用的模型特征名称元数据保持原有含义。编码器需要处理的字段由 `cols` 改为 `features`。

手工二叉树、多叉树及 DecisionTreeAnalyzer 的 `features` 同样支持字符串或一维 list、tuple、ndarray、Index，树规则中的字段名须为字符串。`fit` 显式字段优先于构造配置。`report` 和 `get_rule_table` 使用 `datasets`，支持自带标签列的单表、表格列表/元组或名称到 DataFrame 的字典。

树报告显式提供 `amount`、`feature_map`、`apply_rule_filter`；导出使用 `save`、`sheet_name`、`excel_writer`。完整 Excel 报告的日期、分组和金额从业务字段读取；单表规则效果评估也支持外部金额向量。分裂与筛选配置在初始化时声明，并行使用 `n_jobs`、`parallel_backend`、`parallel_config`。规范参数与优先级见[多叉树与统一树报告](multiway_tree.rst)。

多叉树最大普通分支数为 `max_n_bins`，独立缺失分支不计入；不再接受 `max_branches`。树报告不接受 `sheet`、`data` 等旧参数别名。规则筛选的指标值统一为 `lift`、`badrate`，不使用中文列名或大写别名作为入参；报告输出仍使用中文指标名称。

## date、group、amount 的数据输入

存在 DataFrame 作用数据的接口支持：

- 列名：`date="放款时间"`、`group="客群"`、`amount="放款金额"`。
- Series：索引集合必须与作用数据完全一致，顺序不同时先对齐；存在重复索引时必须完全同序。
- 一维 ndarray、list、tuple、Index：按行位置绑定，长度必须与作用数据一致。

数据在筛选、分组和剔灰之前绑定，随后与原始行一起筛选、排序，不修改调用方的 DataFrame，也不覆盖同名但值不同的字段。

`freq` 使用 pandas 日期 offset / `Grouper` 的分组规则，兼容 pandas 新旧频率写法，不再受 `Period` 支持范围限制：

| 含义 | 旧写法 | 新写法 |
| --- | --- | --- |
| 月末、业务月末、半月末 | `M`、`BM`、`SM` | `ME`、`BME`、`SME` |
| 季末、业务季末 | `Q`、`BQ` | `QE`、`BQE` |
| 年末、业务年末 | `A/Y`、`BA/BY` | `YE`、`BYE` |
| 年初、业务年初 | `AS`、`BAS` | `YS`、`BYS` |
| 小时、分钟、秒 | `H`、`T`、`S` | `h`、`min`、`s` |
| 毫秒、微秒、纳秒 | `L`、`U`、`N` | `ms`、`us`、`ns` |
| 业务小时、自定义业务小时 | `BH`、`CBH` | `bh`、`cbh` |

同样支持 `MS`、`QS-APR`、`YS-JUL`、`BMS`、`B`、`W-MON`、`WOM-2MON` 等频率，保留倍数、锚点和组合，例如 `2ME`、`2W-MON`、`15min`、`2h30min`、`1.5h`。`MS`（月初）与 `ms`（毫秒）严格区分。也可传入 pandas 可用于时间分组的 `DateOffset` / `Timedelta` 对象；自定义工作日对象的节假日及工作周配置会被保留。

分组采用 pandas 默认的 `closed/label/origin` 边界，`2ME` 等倍数频率形成实际的不重叠分组。固定间隔的统计只计算已有样本所属的桶，避免稀疏高频数据展开大量空桶；分布图保留原生 resample 的空桶行为。原有年月周展示格式保留，其他频率显示相应时间边界，小时、分钟、亚秒和时区信息不会被截断为日期。

非法、非正或 pandas 本身不能用于当前日期分组的频率会抛出中文错误。支持结构化执行状态的报告仍会记录频率失败，严格模式禁止发布不完整报告。新旧别名对应关系参见 [pandas 日期频率](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) 与 [2.2 版本迁移说明](https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets)。

多维分组入口同样使用 `group`，包括特征分组摘要、规则流报告和分箱趋势图。它们还支持字段列表和按组名定义的数据字典，例如 `group=["客群", "渠道"]` 或 `group={"客群": customer_groups, "渠道": channels}`。一维字符串列表恰好全部是现有列名时解释为字段集合；要明确传入逐行字符串分组数据，可以使用 Series。

`ModelReport.get_bin_table`、`get_feature_bin_table` 和 `to_excel` 支持辅助数据。多数据集导出可按数据集名称提供字典，例如 `date={"train": train_dates, "test": test_dates}`。临时数据和派生缓存不会覆盖原报告的数据集。

单标签剔灰也保留数据集的原始行和索引，外部日期、分组、金额数据仍按原始数据对齐；计算指标时再同步应用有效标签掩码。`Rule.report` 和 `ruleset_analysis` 同样在规则筛选前绑定外部金额数据。

置换分析中，金额同时作用于实际和参考数据。传入一份数据时需要同时匹配两个作用数据集；也可使用 `amount={"swap_df": actual_amounts, "reference_df": reference_amounts}`。`rule_swap_analysis` 对应的字典键为 `data` 与 `reference_data`。

## 分箱和绘图

分箱器参数继续使用 `min_n_bins`、`min_bin_size`、`max_bin_size`、`missing_separate`、`prebinning`、`prebinning_params`、`special_codes` 等原有名称。直方图箱数、校准桶数、预分箱数量不与最终风险分箱数混为同一参数。

分箱统计、方法比较、绘图、特征摘要（含 DataFrame/Series 扩展）、效率分析、预估器和置换分析统一支持 `bin_params`。优先级为：显式参数 > `bin_params` > 函数默认值。预估器保留配置字典，克隆后配置仍一致。多方法汇总中的按方法配置用于覆盖各方法的公共设置。

报告规则使用 `rules`，分箱器构造使用 `user_splits`。规则的边界格式仍遵循对应入口的语义：固定分箱表的内部切点与含无穷边界的区间定义不能不经转换直接混用。

`bin_plot(result=True)` 保持返回 `(图对象, 分箱统计表)`。`psi_plot(result=True)` 和 `distribution_plot(result=True)` 保持返回各自的统计表。只统一开关名称，不改变这些返回格式。

`colors` 是颜色列表，`cmap` 是连续颜色映射；`save` 是单张图片路径，批量图片目录使用 `save_dir`；这些不同含义的参数保留区分。

模型报告原先的预测方法 `method` 改为 `prediction_method`，预测方法的额外参数改为 `prediction_kwargs`，从而让报告中的 `method` 明确表示分箱方法。

## 指标边界、合计与精度

`feature_bin_stats` 和 `feature_bin_stats_2d` 支持 `round_digits=False` 返回原始精度；默认在合计完成后才舍入展示值。`return_cols` 也在合计完成后生效，不会因隐藏计数列而丢失总体坏率。多特征和长表多标签分别合计，重复添加合计不会叠加已有合计行。

结果表和报告只展示指标数值，不添加指标状态、原因列或额外的“不适用”说明。坏账改善沿用兼容值：全拒绝且总体坏率大于 0 时为 1，总体坏率为 0 时为 0。全部拒绝时，风险拒绝比和累计风险拒绝比为 NaN；未拒绝时风险拒绝比为 0。NaN 在 Excel 中显示为空。单类分箱 KS 保留 0。

模型 TOP 表的坏账改善仍使用拒绝 TOP 后的计算口径及上述兼容值，不使用 `LIFT−1`。Swap 客群比较表保持其总体比较口径。

启用 `woe_clip` 后，原有 IV 列仍使用截断后的 WOE；同时提供 `原始分档WOE值`、`原始分档IV值` 和 `原始指标IV值`。连续目标含负数或没有正总量时，普通统计矩仍可计算，贡献分布指标为空。

详细定义和复核证据见 [指标逐项修复与逻辑复核](HSCREDIT_METRIC_RECHECK_2026-10-27.md)。

## 主要旧名称

`date_col/date_freq/psi_date_col/psi_freq` → `date/freq`；`group_col/group_cols/dimension_col/dimension_cols/psi_group_col` → `group`；`amount_col` → `amount`。

风险分箱 `n_bins` → `max_n_bins`；`bin_method/binning_method/auto_method/bin_type` → `method`；`hscredit_methods` → `methods`；`binning_params/binner_kwargs/auto_kwargs` → `bin_params`。

`return_frame` → `result`；`figure_size` → `figsize`；表格图的 `font_size` → `fontsize`；单图 `save_path` → `save`；柱状图 `horizontal` 布尔值 → `orientation="horizontal"/"vertical"`。

技能入口与 Python 接口使用相同名称，并支持通过 `inputs.date/group/amount` 或 `{"input": "对象名"}` 引用辅助数据。
