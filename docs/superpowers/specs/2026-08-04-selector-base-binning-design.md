# 特征筛选器统一前置分箱设计

## 目标

在 `BaseFeatureSelector` 中统一实现“先分箱、再筛选”，让所有特征筛选器同时支持 `binner` 和 `binning_params`。调用者可以复用已训练分箱器、传入配置好的未训练分箱器，或仅通过参数字典让筛选器内部创建 `OptimalBinning`。`IVSelector` 由此基于分箱 index 计算 IV，`CorrSelector` 删除自身重复的分箱器创建和训练逻辑。

## 范围

本次修改包含：

- `BaseFeatureSelector` 对分箱器来源、拟合状态、训练和 index 转换的统一管理。
- 所有公开特征筛选器构造函数显式支持并转发 `binner`、`binning_params`。
- `IVSelector` 的参数说明和三种分箱用法示例。
- `CorrSelector` 通过基类默认分箱器直接计算，不再自行实例化 `OptimalBinning`。
- 已训练、未训练、参数优先级、sklearn 参数发现和中文错误信息的回归测试。
- 使用 `examples/hscredit_yyp.xlsx` 验证真实数据上的分箱后筛选流程。

本次不改变各筛选器的筛选算法、阈值定义以及 `transform` 的公开输出语义，也不重构无关选择器代码。

## 公开参数与优先级

`BaseFeatureSelector` 及其所有公开子类增加以下通用参数：

- `binner`：配置完成的分箱器实例，不接受分箱器类。实例可以已经训练，也可以尚未训练。
- `binning_params`：传给 `OptimalBinning` 构造函数的参数字典；`None` 表示不通过参数字典启用分箱，空字典 `{}` 表示使用 `OptimalBinning` 的默认构造参数。

有效分箱器按以下规则解析：

1. `binner` 不为 `None` 时直接使用该实例，`binning_params` 无论是否传入都被忽略。
2. 未传 `binner` 且 `binning_params` 不为 `None` 时，使用 `OptimalBinning(**dict(binning_params))` 创建实例；复制字典以避免修改调用者对象。
3. 两者均未传入时不做前置分箱，保持筛选器现有行为。

因此优先级固定为：

`binner > binning_params > 不分箱`

`binner` 只接受实例，避免“类对象如何构造”和实例参数覆盖产生第二套优先级。若传入类或缺少必要的转换接口，抛出中文参数错误。

## 基类数据流

`BaseFeatureSelector.fit` 继续先通过 `_check_input` 统一处理 sklearn 风格 `fit(X, y)` 和 scorecardpipeline 风格 `fit(df)`。完成目标列分离后，由基类执行以下流程：

1. 解析有效分箱器并保存到 `_binner_instance`。
2. 判断分箱器是否已经训练。HSCredit 分箱器以 `_is_fitted` 为准，同时兼容常见的 `is_fitted_`、`fitted_` 和 sklearn 拟合状态检查。
3. 未训练实例调用 `fit(X, y)`；没有 `y` 时调用 `fit(X)`，由具体分箱器决定是否支持无监督拟合。
4. 已训练实例不再拟合，直接复用已有规则。
5. 优先调用 `transform(X, metric="indices")`，把原始值转换为分箱 index；兼容不接受 `metric` 的外部分箱器时回退到 `transform(X)`，保留已有 `apply(X)` 兼容入口。
6. 将分箱后的 DataFrame 交给子类 `_fit_impl` 计算得分和选中特征。

分箱结果必须保持输入行索引和字段名。分箱器返回 ndarray 时由基类恢复为 DataFrame；返回字段数与输入不一致时抛出中文错误，避免基于错位数据筛选。

`BaseFeatureSelector.transform` 仍只根据 `selected_features_` 从调用者输入中选列，返回原始字段值，不返回分箱 index。这样分箱仅影响筛选口径，不改变下游数据形态。

## 全部筛选器接入方式

所有公开筛选器在现有构造参数末尾显式增加 `binner=None`、`binning_params=None`，并原样转发给 `BaseFeatureSelector`。显式签名保证 sklearn `get_params`、`set_params`、`clone` 和 Pipeline 能识别这两个参数；不使用动态签名、元类或仅靠实例属性注入。

除构造参数转发外，各筛选器不单独创建、训练或转换公共分箱器。未配置分箱的调用路径保持不变，以控制修改范围。

组合筛选器在进入其内部阶段前由自身的基类流程统一完成一次分箱。内部阶段接收已分箱的字段，不自动继承和重复应用外层分箱器；只有内部选择器被调用者单独显式配置时才会再次分箱。

## IVSelector

`IVSelector` 显式公开 `binner` 和 `binning_params`，其 `_fit_impl` 不感知分箱器来源，只对基类传入的数据按唯一分箱 index 计算 IV。

文档至少说明三种用法：

1. `IVSelector(binning_params={"method": "best_iv", "max_n_bins": 5})`：内部创建并训练 `OptimalBinning`。
2. `IVSelector(binner=OptimalBinning(...))`：传入配置好的未训练实例，由基类训练。
3. `IVSelector(binner=trained_binner)`：复用已训练实例，不重新训练。

同时传入 `binner` 和 `binning_params` 时，示例明确说明只使用 `binner`。

## CorrSelector

`CorrSelector` 删除 `_compute_metric_weights` 中单独构造和训练 `OptimalBinning` 的逻辑，统一使用 `BaseFeatureSelector` 管理的 `_binner_instance`：

- 相关矩阵基于分箱 index 计算。
- 未显式传入 `weights` 时，从同一分箱器的 `bin_tables_` 提取 IV、KS、Lift 或坏样本率作为特征保留权重。
- 显式 `weights` 继续优先于指标权重，但不阻止显式配置的基类分箱作用于相关矩阵。

`CorrSelector` 在构造时为每个实例初始化一份独立的默认分箱参数：

```python
{
    "method": "best_iv",
    "max_n_bins": 5,
    "min_bin_size": 0.01,
    "missing_separate": True,
}
```

`CorrSelector` 的构造签名直接使用上述默认配置；构造函数复制字典，不修改共享默认对象。调用者不传 `binner` 和 `binning_params`、且拟合时提供 `y` 或目标列时，默认配置经由基类创建并训练 `OptimalBinning`。显式字典替换整份默认配置，显式 `binning_params=None` 表示关闭 `CorrSelector` 的构造默认分箱，显式 `binner` 仍具有最高优先级。

为了保持 `CorrSelector.fit(X)` 的既有无目标调用，未提供 `y` 且当前配置来自构造默认值时，跳过该默认监督分箱，继续按原始数值相关性和等权规则筛选。调用者显式提供未训练分箱器或显式 `binning_params` 时不静默跳过，其无目标拟合能力由具体分箱器决定；已训练分箱器可以在没有 `y` 时直接转换。

`ScorecardFeatureSelection.corr_binning_params` 继续控制内部 `CorrSelector` 的筛选前分箱配置，不再触发第二个指标分箱器：外层未分箱且未配置该参数时省略关键字，让内部 `CorrSelector` 使用默认配置；外层已经分箱且该参数为 `None` 时显式传入 `binning_params=None`，避免对 index 再次分箱；调用者显式配置 `corr_binning_params` 时按其要求执行内部二次分箱。

## 错误处理与兼容性

- `binner` 为类时抛出中文错误，提示应传入配置好的实例，例如 `OptimalBinning(...)`。
- `binning_params` 在需要使用时必须是字典；`binner` 已提供时按优先级直接忽略 `binning_params`，不校验或修改它。
- 分箱器缺少 `transform` 和 `apply` 时抛出中文错误，不再静默返回原始数据。
- 未训练的监督分箱器缺少目标变量时保留分箱器原有中文错误。
- 已训练 HSCredit 分箱器通过 `_is_fitted` 正确识别，修复当前被重复拟合的问题。
- 参数字典在解析时复制，构造函数仍保存公开参数，保证调用者字典不被修改并兼容 sklearn 克隆。
- 未配置分箱的筛选器保持现有计算与输出；`CorrSelector` 按上述默认配置和无目标兼容规则处理。

## 测试策略

采用测试驱动开发覆盖以下行为：

1. `IVSelector(binning_params=...)` 创建 `OptimalBinning`，按指定 `best_iv` 参数拟合，并基于 index 而非连续原值计算 IV。
2. 配置好的未训练 `binner` 被训练一次；训练后的分箱器再次传入时不被重新拟合。
3. 同时传入 `binner` 和冲突或无效的 `binning_params` 时只使用 `binner`。
4. 传入分箱器类、非法 `binning_params`、无转换接口对象和错位转换结果时产生明确中文错误。
5. `binning_params={}` 使用默认 `OptimalBinning`，且调用者参数字典不被修改。
6. 代表性无监督、监督、模型型和组合筛选器的 `get_params`/`clone` 均包含并保留 `binner`、`binning_params`。
7. `CorrSelector()` 在有 `y` 时使用 `best_iv`、`max_n_bins=5`、`min_bin_size=0.01`、`missing_separate=True` 的默认配置，相关矩阵基于 index，指标来自同一 `bin_tables_`。
8. `CorrSelector.fit(X)` 无目标时保留原始相关性和等权路径；显式已训练分箱器在无目标时可以直接使用。
9. `ScorecardFeatureSelection` 不发生重复分箱，阶段报告和最终选中特征接口保持正常。
10. 运行特征筛选专项测试和非慢速、非集成回归测试；已知失败按 `AGENTS.md` 记录区分。
11. 使用 `examples/hscredit_yyp.xlsx`，以 `FPD` 为目标，验证 `衡枢鉴真分老客版` 单特征以及 `[衡枢鉴真分老客版, 近六个月非银多头机构数, 青云24]` 多特征的分箱后 IV 与相关性筛选，检查模块导入、输出字段、输出格式和中文报告。
