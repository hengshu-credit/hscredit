# 分箱约束策略与统一评价目标

日期：2026-10-27（Asia/Shanghai）。本次在已有分箱修复之上增加可选行为，默认仍保持严格约束。

## 1. 约束冲突策略

全部 17 种一维分箱器、`OptimalBinning` 和 `OptimalBinning2D` 增加 `constraint_policy`：

| 取值 | 行为 |
|---|---|
| `"raise"`（默认） | 尝试满足箱数、样本占比、坏率和单调约束；无法同时满足时明确报错 |
| `"legacy"` | 同样先尝试满足约束；最终仍不可行时，保留原生算法尽力满足的规则。没有可行分区的优化器可沿用单箱回退，允许部分约束未满足 |

`legacy` 只控制约束冲突，不恢复旧排序、边界、精度和 IV 计算问题。输入非法、参数非法（例如最少箱数大于最多箱数）、目标函数抛错/返回非法值仍然报错。超时沿用原有处理：已有可行解时明确返回未证明最优的结果，未找到可行解时抛错。显式 `target_bad_rates` 配合 `strict_mode=True` 的分档要求保持严格，需放宽时使用其自身的 `strict_mode=False`。

固定规则保持优先级，二维缺失语义组不会为减少箱数而丢弃。未满足项在 `constraint_violations_` 中按字段记录；必要时 `solver_info_` 标识 `LEGACY`。没有向业务分箱表增加状态或原因列。

```python
binner = OptimalBinning(
    method="tree", max_n_bins=5, min_n_bins=3,
    min_bin_size=0.1, monotonic="descending",
    constraint_policy="legacy",
).fit(X_train, y_train)

print(binner.constraint_violations_)
```

## 2. 哪些方法支持自定义评价目标

扩展前只有 `ORBinning` 支持 `objective="custom" + custom_objective`；CP-SAT 和遗传分箱只能从固定字符串中选择。

| 方法 | 当前原生目标 | 名称快捷调用 / 任意评价函数 |
|---|---|---|
| `or_tools` / `ORBinning` | `iv, ks, gini, entropy, chi2` | 支持 |
| `cp_sat` / `CPSATBinning` | `iv, ks, gini` | 支持；Python 回调走有时限的候选枚举，实际后端记为 `enumeration` |
| `genetic` / `GeneticBinning` | `iv, ks, gini` | 支持，作为遗传适应度 |
| 其他一维方法 | 各自原生算法准则；`best_lift` 另有 `optimization` 专用选项 | 不接受任意 `objective`，传入会明确拒绝 |
| `OptimalBinning2D` | 区域合并使用既有准则 | 可以通过 `x_params/y_params` 配置支持目标的轴方法 |

三种支持方法统一提供 `objective` 和可选的 `objective_params`。直接类与统一入口使用同一目标解析与回调契约；比较结果时需对齐各方法的默认参数。

```python
from hscredit.core.binning import OptimalBinning, CustomObjectives

# 名称快捷选择，权重单独配置。
binner = OptimalBinning(
    method="or_tools",
    objective="max_lift_iv",
    objective_params={"iv_weight": 1.0, "lift_weight": 0.2},
    constraint_policy="legacy",
).fit(X_train, y_train)

# 同一个参数接收自定义函数；额外参数也通过 objective_params 传入。
def my_objective(bin_stats, total_good, total_bad, penalty=0.01):
    return sum(row["分档IV值"] for row in bin_stats) - penalty * len(bin_stats)

custom = OptimalBinning(
    method="genetic",
    objective=my_objective,
    objective_params={"penalty": 0.02},
    random_state=42,
).fit(X_train, y_train)

# 既有工厂函数生成的回调可直接传入。
OptimalBinning(method="cp_sat", objective=CustomObjectives.max_ks_iv(ks_weight=0.3))

# 查看可用名称。
CustomObjectives.names()
OptimalBinning.available_objectives("or_tools")
```

原 `objective="custom", custom_objective=函数` 写法继续可用。不能同时选择其他目标并额外传入 `custom_objective`，以免静默忽略其中一个配置。`objective_params` 对命名复合目标配置工厂参数，对自定义函数传额外关键字；原生目标不接收额外权重。

## 3. 八个预定义复合目标

| 名称 | 最大化内容（各项按配置权重相加） |
|---|---|
| `max_lift_iv` | IV + 最大 Lift |
| `min_lift_iv` | IV + 最小 Lift |
| `max_min_lift_sum_iv` | IV + 最大 Lift + 最小 Lift |
| `max_lift_distance_iv` | IV + 各箱 Lift 离 1 的最大距离 |
| `min_lift_distance_iv` | IV + 各箱 Lift 离 1 的最小距离 |
| `lift_distance_sum_iv` | IV + 上述最大距离 + 最小距离 |
| `max_ks_iv` | IV + 最大绝对累计 KS |
| `woe_variance_iv` | IV + 箱 WOE 方差 |

前六项使用 `iv_weight/lift_weight`，KS 项使用 `iv_weight/ks_weight`，WOE 方差项使用 `iv_weight/variance_weight`。

原实现的复合 IV 项混用了箱内坏率与好率，而且 WOE 符号与共享内核相反。本次将预定义复合目标统一接入分档 IV。可手算样本“两箱好坏数分别为 4/1、1/4”应得到 `1.2 × ln(4) ≈ 1.663553`；不等人数样本“4/1、6/9”应得到 `0.3 × ln(6) ≈ 0.537528`。关闭附加权重后八个目标均与独立公式及共享统计一致。

回调的 `bin_stats` 为按箱号排序的字典列表，包含未舍入的中文指标列，并保留英文键：

- `count/good/bad`：样本数、好数、坏数。
- `bad_rate/good_rate`：各箱内部坏率/好率。
- `bad_dist/good_dist`：分别以总体坏数/好数为分母的分布。
- `woe/iv/lift`：对应统一统计中的分档 WOE、IV、Lift。WOE 方向为 `log(坏分布/好分布)`。

回调应直接读取 `iv` 或 `分档IV值`，不能将 `bad_rate-good_rate` 当作 IV 中的分布差。普通区间搜索不包含缺失/特殊箱，原生 `gini` 保持各算法既有实现；跨方法要完全相同的评价口径时使用同一个回调或复合目标。

回调必须返回有限实数，统一取最大值；需要最小化损失时返回其相反数。NaN、无穷、数组、布尔值及回调异常会明确报错，兼容约束策略也不会吞掉这些错误。

## 4. 工程行为与验证

- OR/CP 的自定义目标一次排序建立前缀计数，候选评价只聚合各箱计数，并复用 `compute_bin_stats` 的派生指标内核。
- 常量特征会评价唯一可行单箱；普通搜索样本因缺失/特殊值剔除而成为单类时，也不绕过用户回调。整体标签输入校验保持原约定。
- 遗传分箱的自定义目标结果默认不会再被通用纯箱合并改写；公共硬约束仍进行最终验收。
- CP 原生目标仍走 CP-SAT；任意 Python 回调不伪装成可编译的 CP 目标，使用带时间限制的候选枚举。遗传算法保持近似搜索性质。
- clone、set_params、串行/进程并行、原有 `custom_objective` 入口、joblib artifact 保存/载入均有回归验证。回调仅在序列化状态中使用现有 joblib/cloudpickle 包装，载入恢复原函数，不改变构造参数。原 artifact 的 `trusted=True` 要求保持不变。
- 目标参数与约束策略可由现有 hsbin `optimal_binning_fit` 结构化操作传入；函数回调用于本地 Python/Notebook，JSON 请求使用注册名称与参数字典。

真实数据示例：[Notebook](../examples/30_binning_objectives_and_constraints.ipynb)。聚合验证结果：[JSON](validation/binning-objectives-and-constraints-2026-10-27.json)。Notebook 为唯一演示入口，没有并行 `.py` 演示。

## 5. 验证结果

2026-10-27（Asia/Shanghai），使用已有解释器 `E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe`（Python 3.11.5），没有安装或升级共享环境。

- 新增 96 项专项测试，覆盖三种方法的名称/函数等价、参数传递、clone/set_params、旧入口、序列化、进程并行、严格/兼容约束、类别/二维、常量与普通样本单类等边界。
- 最终相关回归 **1,390 通过、0 失败、0 跳过，125.75 秒**。范围：`tests/test_binning/`、`tests/test_parallel_api_contract.py`、hsbin 的分箱器/表格测试，以及文档参数/构建校验测试。新增专项包含在总数中，不能重复累加。本轮未重跑全仓测试。
- `examples/hscredit_yyp.xlsx` 的 970 行、136 个坏样本：3 字段 × 3 方法 × 8 个指标，**72 组名称/函数对照（144 次拟合）全部一致**，逐箱人数和坏数与原始行聚合一致。
- `商品类别` 的三种自定义目标、二维轴目标核验均完成，样本总数保持 970；另完成 16 组手算 IV 对照。
- Notebook 的 **9 个代码单元全部执行，0 个执行错误**；hsbin 结构化调用成功，实际生成模型和 Excel 两项制品。
- Black 检查、Flake8 的 E9/F63/F7/F82 检查和 Python 3.9 语法检查通过；源码 SHA256 与执行时记录一致。未声明重新测量覆盖率或完成全量类型检查。

原始 JUnit：`.audit_tmp/binning-objective-policy-final.xml`。复现：

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$auditPython = '.\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
$binningScopes = @(
    'tests/test_binning/',
    'tests/test_parallel_api_contract.py',
    'tests/test_skills/test_hsbin_binners.py',
    'tests/test_skills/test_hsbin_tables.py',
    'tests/test_docs_parameter_rendering.py',
    'tests/test_docs_build_validation.py'
)
& $auditPython -m pytest @binningScopes -q --tb=short --junitxml=.audit_tmp/binning-objective-policy-final.xml
```
