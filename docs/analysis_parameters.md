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

## 字段选择

`features="score"` 与 `features=["score"]` 等价，也支持一维 tuple、NumPy 数组和 pandas Index。已有整数列名仍然可用。`feature_bin_stats` 和分箱汇总同时提供 `feature` 与 `features` 两种入口，两者不能同时指定。

二维分箱器、二维统计、二维绘图和交叉规则接口支持 `features=np.array(["score", "age"])`，也支持 `x1="score", x2="age"`。两个字段必须不同；`x1/x2` 必须成对出现，不能与 `features` 混用。

规则挖掘器的构造和 `fit` 使用 `features`。模型报告的公开字段入参也使用 `features`；sklearn 的 `feature_names_in_` 和绘图使用的模型特征名称元数据保持原有含义。编码器需要处理的字段由 `cols` 改为 `features`。

## date、group、amount 的数据输入

存在 DataFrame 作用数据的接口支持：

- 列名：`date="放款时间"`、`group="客群"`、`amount="放款金额"`。
- Series：索引集合必须与作用数据完全一致，顺序不同时先对齐；存在重复索引时必须完全同序。
- 一维 ndarray、list、tuple、Index：按行位置绑定，长度必须与作用数据一致。

数据在筛选、分组和剔灰之前绑定，随后与原始行一起筛选、排序，不修改调用方的 DataFrame，也不覆盖同名但值不同的字段。

多维分组入口同样使用 `group`，包括特征分组摘要、规则流报告和分箱趋势图。它们还支持字段列表和按组名定义的数据字典，例如 `group=["客群", "渠道"]` 或 `group={"客群": customer_groups, "渠道": channels}`。一维字符串列表恰好全部是现有列名时解释为字段集合；要明确传入逐行字符串分组数据，可以使用 Series。

`ModelReport.get_bin_table`、`get_feature_bin_table` 和 `to_excel` 支持辅助数据。多数据集导出可按数据集名称提供字典，例如 `date={"train": train_dates, "test": test_dates}`。临时数据和派生缓存不会覆盖原报告的数据集。

置换分析中，金额同时作用于实际和参考数据。传入一份数据时需要同时匹配两个作用数据集；也可使用 `amount={"swap_df": actual_amounts, "reference_df": reference_amounts}`。`rule_swap_analysis` 对应的字典键为 `data` 与 `reference_data`。

## 分箱和绘图

分箱器参数继续使用 `min_n_bins`、`min_bin_size`、`max_bin_size`、`missing_separate`、`prebinning`、`prebinning_params`、`special_codes` 等原有名称。直方图箱数、校准桶数、预分箱数量不与最终风险分箱数混为同一参数。

`feature_bin_stats`、`feature_bin_stats_2d`、`bin_plot`、`psi_plot`、分箱趋势图和二维分箱图支持 `bin_params`。这些入口的优先级为：显式参数 > `bin_params` > 函数默认值。多方法汇总中的按方法配置仍用于覆盖各方法的公共设置。

报告规则使用 `rules`，分箱器构造使用 `user_splits`。规则的边界格式仍遵循对应入口的语义：固定分箱表的内部切点与含无穷边界的区间定义不能不经转换直接混用。

`bin_plot(result=True)` 保持返回 `(图对象, 分箱统计表)`。`psi_plot(result=True)` 和 `distribution_plot(result=True)` 保持返回各自的统计表。只统一开关名称，不改变这些返回格式。

`colors` 是颜色列表，`cmap` 是连续颜色映射；`save` 是单张图片路径，批量图片目录使用 `save_dir`；这些不同含义的参数保留区分。

模型报告原先的预测方法 `method` 改为 `prediction_method`，预测方法的额外参数改为 `prediction_kwargs`，从而让报告中的 `method` 明确表示分箱方法。

## 主要旧名称

`date_col/date_freq/psi_date_col/psi_freq` → `date/freq`；`group_col/group_cols/dimension_col/dimension_cols/psi_group_col` → `group`；`amount_col` → `amount`。

风险分箱 `n_bins` → `max_n_bins`；`bin_method/binning_method/auto_method/bin_type` → `method`；`hscredit_methods` → `methods`；`binning_params/binner_kwargs/auto_kwargs` → `bin_params`。

`return_frame` → `result`；`figure_size` → `figsize`；表格图的 `font_size` → `fontsize`；单图 `save_path` → `save`；柱状图 `horizontal` 布尔值 → `orientation="horizontal"/"vertical"`。

技能入口与 Python 接口使用相同名称，并支持通过 `inputs.date/group/amount` 或 `{"input": "对象名"}` 引用辅助数据。
