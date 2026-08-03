# 分箱参数优先级与二维单调方向统一设计

## 目标

统一 `OptimalBinning2D` 和 `feature_summary` 的参数覆盖规则，确保调用者通过显式参数表达的约束不会被透传参数覆盖，并使二维分箱的单调方向沿用一维分箱器的定义与拟合结果。

## 范围

本次修改包含：

- `OptimalBinning2D` 的全局参数、`x_params` / `y_params` 与 `_x` / `_y` 显式参数优先级。
- `OptimalBinning2D` 一维预分箱和二维合并阶段共用有效轴向参数。
- `OptimalBinning2D` 自动单调方向复用内部 `OptimalBinning` 的拟合结果。
- `feature_summary` 外层分箱参数与 `binning_params` 的优先级。
- 相关中文 docstring、代码注释和回归测试。

本次不修改 `feature_binning_summary`。该接口的 `bin_params` 支持按分箱方法分别配置，属于不同的 API 语义。

## 参数优先级

### OptimalBinning2D

每个轴先构造一份有效配置，覆盖顺序由低到高为：

1. 全局参数，例如 `max_n_bins`、`min_bin_size`、`method`、`monotonic`。
2. 轴向透传字典，即 `x_params` 或 `y_params`。
3. 非 `None` 的轴向显式参数，例如 `max_n_bins_x`、`method_y`、`monotonic_x`。

因此最终优先级为：

`显式 _x/_y 参数 > x_params/y_params > 全局参数`

缺失值分箱继续遵循已有的 `missing_separate_x/y > missing_separate` 规则。`user_splits_x/y`、`special_codes_x/y` 和 `dtype_x/y` 本身就是轴向显式配置，不由全局参数覆盖。

有效轴向配置在创建内部 `OptimalBinning` 时一次性传入，不再先构造对象后通过 `setattr` 修改，以保证构造期参数校验和实际拟合配置一致。未知的透传参数继续产生警告并忽略，保持现有兼容行为。

### feature_summary

`feature_summary` 的有效分箱配置覆盖顺序由低到高为：

1. `binning_params` 中的透传参数。
2. 外层 `binning_method`、`max_n_bins`、`random_state`。

因此最终优先级为：

`外层参数 > binning_params`

外层没有对应入口的配置，例如 `user_splits`、`strict_user_splits`、`min_bin_size`，仍由 `binning_params` 传递。配置合并不得修改调用者传入的字典。

## 单调方向

`ascending` 和 `descending` 完全沿用 `BaseBinning` 的定义：

- `ascending`：特征值或分箱索引增大时，坏样本率递增，即越大风险越高、越大越差。
- `descending`：特征值或分箱索引增大时，坏样本率递减，即越大风险越低、越大越好。

二维合并阶段不再根据二维边际分箱首尾坏样本率单独推断方向。两个内部一维分箱器拟合后，二维阶段读取各自的有效 `monotonic` 配置及 `monotonic_trend_`：

- 显式 `ascending` / `descending` 直接沿用。
- `True`、`auto`、`auto_asc_desc`、`auto_heuristic` 使用内部一维分箱器实际识别出的趋势。
- 一维结果为 `peak`、`valley`、`convex` 或 `concave` 时，不强行投影成递增或递减；一维预分箱保留相应约束，二维硬约束只处理能够直接表达的 `ascending` / `descending`。
- 未启用单调约束时，二维阶段不添加轴向单调硬约束。

这样一维与二维不会对同一参数产生相反解释，也不会维持两套自动方向算法。

## 实现边界

`OptimalBinning2D` 增加私有的轴向配置解析逻辑。内部一维分箱器从完整的有效轴向配置创建；二维单调方向读取同一配置和对应一维分箱器的拟合结果。最终二维分箱数仍由 `max_n_bins_2d`（未提供时回退 `max_n_bins`）控制，最终二维分箱的最小样本量仍由全局 `min_bin_size` 控制，因为这两个约束作用于联合区域而非单一轴。实现保持现有公开构造函数签名，不新增公开参数。

`feature_summary` 仅调整 `_normalize_binning_config` 的合并顺序、说明文字与示例，不改变指标计算和并行流程。

## 错误处理与兼容性

- 显式轴向参数和 `x_params` / `y_params` 同时提供时，不报冲突错误，按既定优先级选择显式值。
- `x_params` / `y_params` 中不存在于 `OptimalBinning` 的键继续警告并忽略。
- 有效参数必须在构造内部 `OptimalBinning` 时接受原有校验，避免通过 `setattr` 绕过校验。
- `feature_summary` 继续在并行任务开始前校验最终配置。
- 不修改已有用户工作区中的类别分箱改动。

## 测试策略

采用回归测试覆盖以下行为：

1. `OptimalBinning2D` 显式 `_x` / `_y` 参数覆盖轴向 params 和全局参数。
2. 未提供显式轴向参数时，`x_params` / `y_params` 覆盖全局参数。
3. 未提供轴向配置时回退全局参数。
4. 传入无效但随后被显式轴向参数覆盖的值时，实际构造配置以显式参数为准；最终有效的非法值仍由 `OptimalBinning` 报中文参数错误或原有错误。
5. 二维自动方向等于内部一维分箱器的 `monotonic_trend_`，不再由二维边际首尾值重新判断。
6. `ascending` / `descending` 的二维违例检查与 `BaseBinning` 坏样本率方向定义一致。
7. `feature_summary` 的外层 `binning_method`、`max_n_bins`、`random_state` 覆盖 `binning_params` 同名键，同时保留其扩展键且不修改原字典。
8. 运行二维分箱与 EDA 专项测试，并使用 `examples/hscredit_yyp.xlsx` 对指定三个特征和 `FPD` 做真实数据验证。
