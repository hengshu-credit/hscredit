# 分箱算法修复与复核记录

> 后续增加了可选的 `constraint_policy="legacy"` 和统一名称/函数评价目标入口，见 [约束策略与评价目标](HSCREDIT_BINNING_OBJECTIVES_2026-10-27.md)。本文保留此前修复和验证记录。

日期：2026-10-27（Asia/Shanghai）。修复依据：[修复前审计](HSCREDIT_BINNING_AUDIT_2026-10-27.md)。
修复前 Notebook/JSON 保持不变；修复后入口：[复核 Notebook](validation/binning-method-repair-2026-10-27.ipynb)，聚合证据：[JSON](validation/binning-method-repair-2026-10-27.json)。

## 1. 修复清单

| 审计项 | 修复 |
|---|---|
| B01：CP 输入未排序 | x/y 成对稳定排序；候选位置使用与 transform 相同的左边界。CP-SAT 数值路径改为真正的区间路径模型，任意输入排列代表同一问题。 |
| B02：默认 Lift 改写算法目标 | 默认关闭额外 Lift 调整；只有 `lift_refine=True` 显式启用。OR 的声明目标不再混入隐藏的 Lift、形状或箱数利用奖励。 |
| B03：克隆丢失参数 | 基类公开经 `**kwargs` 接收的公共构造参数；统一入口公开控制选项；MDLP 正确转发精度/WOE 参数。clone、set_params、事务拟合保持配置。 |
| B04：约束不保证成立 | 最终统一核验数值箱数、人数、坏率和形状。先保留/合并原边界，必要时在合法候选上求解；不可行时明确报错。最少人数向上取整，最多人数向下取整。 |
| B05：卡方按位置裁剪 | 超过最大箱数时继续选择相邻卡方最小的合并，不再删除最左侧一段切点。 |
| B06：低基数丢边界 | 取相邻数值中点，OR/CP 不再因唯一值少于最大箱数返回无候选。 |
| B07：边界和精度不一致 | 搜索、统计、应用统一为 `[a,b)`；修正指标辅助函数和 `pd.cut` 路径；实际切点不舍入，`decimal` 只影响显示，重复切点去重。中间统计不再使用展示精度。 |
| B08：目标坏率克隆失败 | 构造参数原样保存，拟合时验证并规范化；明确按最终区间坏率匹配档位，不再用累计前缀坏率替代。 |

其他已处理项：

- 预分箱先形成不可拆分的原子箱，再运行主算法并还原对应边界；移除当前流程中的“最近切点映射”。CP 预箱结果用小候选集的原始行穷举核验。
- `cp_sat` 及 `or_tools(use_cp_sat=True)` 调用真实 `CpSolver`；`solver_info_` 记录后端、目标、求解状态、整数目标和界限。没有给业务统计表添加状态或原因列。
- CP 先在短预算内验证可行候选并作为求解提示。短时限只完成预求解时，可以返回已逐项验证的可行分区，明确标记 `FEASIBLE` 和来源，保留求解器的 `UNKNOWN` 原状态，不冒充已证明最优。
- `or_tools.n_prebins`、`cp_sat.max_candidates` 控制实际候选数；CP 工作线程使用已有统一预算；候选搜索及求解器保留时间限制。
- `best_ks` 在全局 KS 相同的候选之间以 IV 决定次序，减少按位置任意补箱。
- MDLP 强制补箱只补到 `min_n_bins`；候选只能位于不同数值之间；特殊值不进入普通区间搜索。
- `convex/concave` 最终按坏率二阶差分验收，不能仅用峰/谷代替。
- 遗传适应度阶段检查最终闭开边界、大小、最小坏率和形状；修正初始染色体多选一个切点的上界。
- KMeans 使用局部随机状态，不再修改全局 NumPy 随机种子。
- `smooth.merge_criterion` 实际选择 IV/卡方/联合准则，统一入口通过 `smoothing_method` 选择平滑方式。
- KDE 的 `iv_weight` 参与候选谷的 IV/谷深权衡，`fallback_to_iv=False` 不再绕过配置进入 IV 后备分支。
- 非零 `best_lift.min_lift` 明确拒绝。覆盖全部样本的分箱器不执行“丢弃其他箱”的筛选；应在结果表或规则分析中筛选。
- 默认 KDE 带宽更名为 `adaptive_reference`；旧 `isj` 保留弃用提示的兼容别名，不再宣称实现完整 ISJ 算法。
- 集成回归暴露的 `RoundScoreCard` 同 WOE 箱标签歧义已修复：原始输入优先使用真实分箱标签，不能用相同 WOE 反推出第一个箱。PMML 测试改为核对模型实际边界，移除旧默认算法切点的固定快照。

主要代码：[共享最终约束](../hscredit/core/binning/base.py)、[候选搜索](../hscredit/core/binning/_candidate_search.py)、[CP 区间求解器](../hscredit/core/binning/_partition_solver.py)、[统一入口](../hscredit/core/binning/optimal_binning.py)。

## 2. 兼容性与约束边界

1. 新拟合默认不进行额外 Lift 调整，需要原策略时显式传入 `lift_refine=True`。历史模型已有规则不会因导入/加载而重新训练。
2. `decimal` 不再改变实际切点。标签不足以区分相邻边界时提高显示精度；规则导出和转换使用完整切点。
3. 完全固定的用户规则、导入规则保留原有优先级。`min_bin_size=None` 的等宽分箱仍可保留规则空间中的空箱，不将合法空区间与重复切点混为一谈。
4. 唯一值不足时最少箱数以可区分的取值数为上限；MDLP 显式关闭强制补箱时保留统计停止准则。其他约束冲突不静默放宽。
5. 目标坏率列表定义坏率档位。严格模式每个占用档位最多一个箱，`merge_empty_bins=False` 要求所有档位均被覆盖；宽松模式优先覆盖更多档位，允许同档多个箱。
6. CP 的 `OPTIMAL` 指候选空间中整数化目标的最优解，不代表连续切点空间上的无条件全局最优。`FEASIBLE` 明确表示尚未证明最优；无可行解或到期未找到解会抛错。

具体 API 说明见 [分箱文档](api/binning.rst)。MDLP 的 IV 候选排序、树/CART 差异、KMeans 初始化方式、二维连通区域合并等明确的算法选择没有被强行改成另一家库的实现。

## 3. 验证方法

新增 65 项 [分箱审计回归测试](../tests/test_binning/test_algorithm_audit_repairs.py)，覆盖输入排列、低基数、精度/clone、明确不可行约束、统计与转换边界、卡方、坏率档位、预箱穷举、真实求解器调用、候选预算、随机状态、固定边界和再次拟合；另新增评分卡相同 WOE 对应不同箱的回归用例。

旧测试中“60% 最大箱允许 100% 单箱”“最小箱 40% 仍保留 1/99 分区”“最少补箱补到最多箱”“显示精度修改实际规则”等预期已更正。MDLP 仍保留独立逐行计数与前缀和的差分验证。

真实数据使用 `examples/hscredit_yyp.xlsx` 的 970 行，目标 FPD；三个约定数值字段、`商品类别` 和二维交互均纳入。三库箱号统一用 `compute_bin_stats(round_digits=False)` 统计；逐箱人数/坏数和非退化 IV 用原始行独立复核。

修复后复核包含 123 组完整样本配置、150 组留出比较，以及结构反例、类别矩阵、二维人口守恒和参数实验。5 次留出切分仅用于敏感性比较，不视为五份独立总体；没有据训练 IV 宣称普遍泛化提升。

## 4. 执行记录

解释器：`E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe`，Python 3.11.5。继续使用已有验收环境，没有安装或升级共享 Anaconda 依赖。

2026-10-27 的实际执行结果：

| 范围 | 结果 | 解释 |
|---|---|---|
| 中间版本的完整 `tests/` | 6,817 通过、3 失败、9 跳过，1,206.95 秒 | 发现 CP 两秒预算、评分卡同 WOE 箱标签、PMML 旧阈值快照三项问题，随后全部处理 |
| 最终分箱目录 + 8 个评分卡/运行时测试模块 | **1,190 通过、0 失败、0 跳过，189.54 秒** | 覆盖全部分箱测试和上述三项失败；JSON 按测试类/测试名逐项验证三项已通过 |
| 修复后真实数据 Notebook | **11 个代码单元全部执行，0 个执行错误** | 123 组完整样本配置、150 组留出比较无执行错误；33 组结构反例按成功或预期不可行异常验收 |
| 静态检查 | Black 检查通过（36 文件）；Flake8 的 E9/F63/F7/F82 检查通过；26 个分箱源文件通过 Python 3.9 语法解析 | 没有重新测量测试覆盖率；没有声称全量 Flake8 或 mypy 检查通过 |

上述测试结果存在重叠，不能相加。最后三项修复后进行了相关模块复测，没有重新运行第二次完整全库测试。9 项跳过为需要真实数据库的既有测试。原始记录保存在 `.audit_tmp/binning-repairs-full.xml`、`.audit_tmp/binning-repairs-final.xml` 和对应日志。

Notebook 保存了 26 个分箱源文件、共享指标模块、评分卡模块及第三方对照源码的 SHA256，核验与当前最终源码一致。修复前后的目标坏率显式参数均分别验证；修复后直接类和统一入口都能完成拟合。

### 4.1 真实数据默认配置对照

下表为训练 IV，“修复前 → 修复后”。两侧保持相同数据及公共箱数/人数设置，但默认行为已按修复目标调整，因此不等同于单个算法参数的孤立因果实验。

| 方法 | 衡枢鉴真分老客版 | 近六个月非银多头机构数 | 青云24 |
|---|---:|---:|---:|
| tree | 0.277028 → 0.336751 | 0.075403 → 0.100863 | 0.086573 → 0.258563 |
| chi | 0.212552 → 0.124904 | 0.072122 → 0.068038 | 0.047486 → 0.016681 |
| best_iv | 0.287019 → 0.317699 | 0.069181 → 0.117802 | 0.080784 → 0.214704 |
| cp_sat | 0.113479 → 0.291376 | 0.004437 → 0.117802 | 0.104560 → 0.180757 |
| or_tools | 0.246818 → 0.283993 | 0.085942 → 0.091503 | 0.203371 → 0.141280 |

部分训练 IV 下降是取消额外 Lift 调整、恢复明确目标和候选预算后的真实结果，没有筛掉这些记录。方法之间仍有预箱、候选与统计约束的设计差别，不把某个训练 IV 排名作为所有方法的正确性标准。

三个真实字段在对齐叶节点人数与精度后，树分箱箱号继续与 toad 完全一致。预箱小样本中，hscredit、独立穷举、optbinning 均得到 `[4.0, 5.5, 10.0]`，IV 同为 `1.3687899576175322`。

复现命令：

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$auditPython = '.\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
& $auditPython -m pytest tests/ -q --tb=short --durations=15 --junitxml=.audit_tmp/binning-repairs-full.xml
```

最终相关模块复测：

```powershell
$binningScopes = @(
    'tests/test_binning/',
    'tests/test_models/test_round_scorecard.py',
    'tests/test_models/test_scorecard_export_consistency.py',
    'tests/test_models/test_score_runtime_contract_repairs.py',
    'tests/test_models/test_scorecard_contract_repairs.py',
    'tests/test_models/test_scorecard_points.py',
    'tests/test_models/test_probability_scorecard.py',
    'tests/test_models/test_default_probability_scorecard.py',
    'tests/test_models/test_scorecard_refactored.py'
)
& $auditPython -m pytest @binningScopes -q --tb=short --junitxml=.audit_tmp/binning-repairs-final.xml
```

Notebook 使用同一解释器的 Jupyter 内核执行；不维护并行 `.py` 演示。修复前结果继续保留在原审计 JSON，避免把历史问题的结果覆盖成修复后结果。
