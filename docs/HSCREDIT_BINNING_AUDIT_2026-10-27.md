# 分箱实现、可控性与第三方差异审计

> 本文保留修复前证据。后续代码修复、兼容行为变化与复测结果见 [分箱修复记录](HSCREDIT_BINNING_REPAIRS_2026-10-27.md)。

审查日期：2026-10-27（Asia/Shanghai）。审查对象：当前工作区 `hscredit/core/binning/` 的 25 个 Python 文件、17 种一维方法、统一入口和二维分箱主要调用链。HEAD：`17e212c3641c6bafd21f79b48e9574ba1f97b76b`。本次新增研究材料，未修改分箱算法。

实验入口：[已执行 Notebook](validation/binning-method-audit-2026-10-27.ipynb)；完整聚合结果、参数、异常、版本及源码 SHA256：[证据 JSON](validation/binning-method-audit-2026-10-27.json)。文中位置均指本次记录的源码版本。

## 1. 结论

“结果不受控”的感受有明确实现依据。问题集中在**统一入口额外改变目标、约束不能保证最终成立、参数在克隆时丢失、候选搜索与实际分箱使用不同边界**；另外存在 `cp_sat` 的输入排序错误和卡方分箱的任意裁剪。

但不能得出“hscredit 的所有同名算法都比 toad / optbinning 差”：

- 对齐叶节点人数和切点精度，并关闭 Lift 后处理后，三个真实字段的 hscredit `tree` 与 toad `dt` **全部样本的箱号完全一致**。
- 默认后处理会显著降低部分训练 IV，也可能改善部分留出集 AUC；它应是明确可选择的策略，不能混同于原算法的效果。
- toad 卡方的高 IV 有时来自极小箱或纯好箱。本数据的 `青云24` 出现 49 人、0 坏样本的箱，统一平滑口径下 IV 为 1.5806；这不是它有更好泛化能力的证据。
- hscredit 的 `OptimalBinning` 是方法分派入口；optbinning 的同名类主要是在预分箱候选上建立约束优化问题。类名相同并不意味着实现同一算法。[optbinning 官方定义](https://gnpalencia.org/optbinning/binning_binary.html)、[数学规划论文](https://arxiv.org/abs/2001.08025)。

优先修复正确性与可控性，再比较预测效果。继续叠加 Lift、形状或平滑启发式，会使不同方法更难解释和调参。

## 2. 实验口径与版本边界

解释器：`E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe`，Python 3.11.5。没有安装、升级或修改共享 Anaconda 依赖。

| 依赖 | 实测版本 |
|---|---|
| hscredit | 0.1.3，导入当前仓库源码 |
| toad | 0.1.4 |
| optbinning | 0.20.1 |
| NumPy / pandas | 2.4.6 / 2.3.3 |
| scikit-learn / OR-Tools | 1.6.1 / 9.7.2996 |

第三方数值结果只代表上述已安装版本。在线文档和主分支可能更新；例如 toad 当前更新记录已有新的 ChiMerge 约束选项，不能将本机旧版行为推广到所有版本。[toad 更新记录](https://github.com/amphibian-dev/toad/blob/master/CHANGELOG.md)。

数据为 `examples/hscredit_yyp.xlsx`：970 行、18 列，FPD=0 有 834 行，FPD=1 有 136 行。三个指定数值字段和 `商品类别` 均无缺失。原始客户行未写入审计产物。

执行范围：

1. 三个数值字段 × 17 方法 × 开关默认调整，以及 toad 5 种方法、optbinning 两种趋势：**123 组完整样本比较**。
2. 三个字段 × 5 个分层随机切分 × 10 种配置：**150 组留出比较**。训练/测试为 679/291 行，种子为 7、19、42、83、137。
3. **33 组结构与约束反例**，另有参数、调用链、候选穷举、精度对齐实验。
4. `商品类别` 的 17 种方法和一组真实二维分箱的人口守恒验证。
5. 4 个既有测试模块：**102 通过，0 失败，0 跳过，47.88 秒**，详见第 8 节。没有重新测量覆盖率。

hscredit 数值基线统一使用 `max_n_bins=5, min_n_bins=2, min_bin_size=0.05, monotonic=False, random_state=42, n_jobs=1`，比较 `lift_refine=True/False`，其他算法参数保持各自默认。optbinning 使用 CART 20 预箱、最小预箱 5%、最小最终箱 5%、2–5 箱，分别测试 `monotonic_trend=None/auto`。toad `dt` 最小叶节点 5%；其他方法没有等价的完整约束接口，尤其不能把卡方 `min_samples` 当成同时满足所有约束的保证。

所有库输出的箱号统一调用 `compute_bin_stats(round_digits=False)`。逐箱人数和坏数用原始行另行聚合校验；没有纯好/纯坏箱时，再按原始计数独立计算 IV。这样隔离“分箱边界不同”与“指标平滑公式不同”。

### 2.1 训练 IV：默认调整改变了什么

| 字段 | hs tree 默认 | hs tree 关闭调整 | toad dt | optbinning 无单调 | optbinning auto |
|---|---:|---:|---:|---:|---:|
| 衡枢鉴真分老客版 | 0.277028 | 0.322019 | 0.336751 | 0.342736 | 0.287043 |
| 近六个月非银多头机构数 | 0.075403 | 0.100863 | 0.100863 | 0.117802 | 0.100863 |
| 青云24 | 0.086573 | 0.258563 | 0.258563 | 0.258563 | 0.189083 |

`青云24` 默认树分箱把 5 箱变为 4 箱，IV 下降约 66.5%。该差异发生在同一实现、同一数据、只切换 `lift_refine` 的对照中，能够定位到统一入口后处理。

老客评分即使关闭调整仍与 toad 有差异，主要要继续对齐切点精度和叶节点人数。将两库的最小叶节点都设为 `ceil(970×5%)=49`，hscredit 使用 `decimal=12` 并关闭调整后，三个字段的最终箱号完全一致，IV 分别为 **0.336751、0.100863、0.258563**。

### 2.2 留出表现：不能以训练 IV 排名代替效果判断

下表是 5 次分层随机切分的留出 AUC 均值。分箱和箱坏率均只使用训练集；测试标签不参与方向选择。

| 字段 | hs tree 默认 | hs tree 关闭调整 | toad dt | optbinning 无单调 | optbinning auto |
|---|---:|---:|---:|---:|---:|
| 衡枢鉴真分老客版 | 0.6283 | 0.6018 | 0.5834 | 0.5703 | 0.6099 |
| 近六个月非银多头机构数 | 0.5343 | 0.5243 | 0.5243 | 0.5214 | 0.5349 |
| 青云24 | 0.5341 | 0.5466 | 0.5489 | 0.5464 | 0.5342 |

这组结果不支持“统一关闭后处理一定更好”，也不支持“第三方一定更好”。第三个字段各方法波动较大，例如 hs tree 关闭调整的标准差约 0.0547。970 行只有 136 个坏样本，重复切分也不是独立的五份数据；本次未做客户分组或严格时间外验证，不能声称统计显著优劣。

## 3. 已复现的高优先级问题

### B01 · P1 · `cp_sat` 在未排序数组上做有序候选搜索

位置：[cp_sat_binning.py:364](../hscredit/core/binning/cp_sat_binning.py#L364)、[_candidate_search.py:129](../hscredit/core/binning/_candidate_search.py#L129)。

`_get_candidate_splits()` 内部排序计算候选，但 `_cp_sat_numerical()` 传给 `search_candidate_splits()` 的仍是原始 `x_vals/y_vals`。后者假定输入已排序，直接使用 `np.searchsorted(x_values, candidates)` 和标签前缀和。因此候选人数、好坏数都可能对应错误的区间。

对 12 种数值、每种 20 行的同一份 240 行样本，只改变行顺序：

| 输入顺序 | 最终切点 | 实际箱数 |
|---|---|---:|
| 按 x 升序 | 2.5、5.5、8.5 | 4 |
| 固定种子打乱 | 5.5 | 2 |
| 按 x 倒序 | 无 | 1 |

同条件 `or_tools(use_cp_sat=True)` 在三种顺序下均保持 4 箱。真实数据上 `cp_sat` 的低 IV 不能直接解释为它不适合这些变量，应先修正这个确定错误。

### B02 · P1 · 统一入口默认将原目标换成 Lift 复合评分

位置：[optimal_binning.py:591](../hscredit/core/binning/optimal_binning.py#L591)、[1388](../hscredit/core/binning/optimal_binning.py#L1388)、[1471](../hscredit/core/binning/optimal_binning.py#L1471)。

多数方法完成原始拟合后，会进入 `_refine_splits_for_lift_stability()`。评分默认综合尾部 Lift、样本占比、坏率转折次数、IV 和箱数；其中尾部项权重 3.0，IV 权重只有 0.08。它既删除算法切点，也从 5%–95% 分位点池增加新切点。

因此 `method='best_iv'` 的最终结果未必保留较高 IV，`tree` 最终切点未必来自树，`kmeans` 最终结果也可能依赖标签。`lift_refine=False` 只关掉这一层，其他大小、坏率、单调性后处理仍会执行。

这可以作为独立策略，但默认混入多个算法会破坏同名比较和参数预期。三字段的训练 IV 对照已直接复现其影响；留出数据也表明不能简单把它定义为“总是有害”。

### B03 · P1 · sklearn 克隆丢失控制选项与精度

位置：[optimal_binning.py:502](../hscredit/core/binning/optimal_binning.py#L502)、[base.py:378](../hscredit/core/binning/base.py#L378)。

`OptimalBinning.get_params()` 没有返回 `_fit_control_options`。因此：

```python
b = OptimalBinning(method='tree', lift_refine=False)
clone(b)._fit_control_options  # {}
```

交叉验证、网格搜索或其他 sklearn 克隆场景中，显式关闭的后处理会恢复默认。当前 `fit()` 的事务包装为控制选项做了额外复制，所以直接 `b.fit()` 能保留该设置；外部 `clone(b)` 仍会丢失。

另有 6 个直接分箱器通过 `**kwargs` 接收 `decimal`，但未在 sklearn 参数中完整保留：`best_ks、best_iv、or_tools、cp_sat、cart、best_lift`。设置 `decimal=8` 后，`clone()` 都变回 4；而每次 `fit()` 本身就会克隆。实验中 `best_iv(decimal=8)` 仍产生 0 切点，小数精度配置没有真正到达拟合算法。

MDLP 的直接构造还会忽略 `decimal/woe_clip`；这点已有局部 docstring 说明，但统一入口的公共精度参数没有体现这种能力差异。

### B04 · P1 · 数值约束以尽量调整代替最终验收

位置：[base.py:1778](../hscredit/core/binning/base.py#L1778)、[1999](../hscredit/core/binning/base.py#L1999)、[2358](../hscredit/core/binning/base.py#L2358)。

大小、单调性、坏率、最大箱数依次修改规则；到达最小箱数、没有切点或没有可调整动作时直接返回，没有统一证明最终约束同时成立。类别字段有显式验证器，数值字段没有对应的统一验收。

可复现结果：

- 三种数值各 100 行，坏率 10%/50%/90%，设置 `min_n_bins=2, max_bin_size=0.4`，`cp_sat` 返回 **1 箱、占比 100%**；明明三个 1/3 箱是可行解。
- 坏率随 x 严格增加，却要求至少 3 箱且 `monotonic='descending'`；`tree/best_iv/monotonic` 返回仍然递增的 3 箱，没有报不可行。
- 90/10 两个不可拆分取值，要求每箱至少 20%、至少 2 箱；部分方法保留 10% 的箱，另一些退化为一箱，均未给出明确不可行结果。

此外，最小占比通常取 `floor(n×比例)`，最大占比在基类取 `ceil(n×比例)`。970 行的 5% 被允许为 48 行，真实 `best_ks` 出现 48/970=4.9485% 的箱。若参数含义是严格的最少/最多占比，应分别采用向上/向下取整并统一分母。

### B05 · P1 · 卡方提前停止后直接裁掉左侧切点

位置：[chi_merge_binning.py:293](../hscredit/core/binning/chi_merge_binning.py#L293)、[603](../hscredit/core/binning/chi_merge_binning.py#L603)。

`_chi_merge()` 遇到最小卡方超过阈值时退出，即使箱数还超标。随后 `_adjust_bins()` 通过 `splits = splits[n_remove:]` 直接删除最左侧若干边界，没有继续按卡方或 IV 损失选择合并。

8 种值各 100 行，坏率为 10%/90%/20%/80%/30%/70%/40%/60%，要求最多 4 箱：默认阈值下，调整前边界 `[1,2,3,4,5,6]` 被直接裁成 `[4,5,6]`，前四个差异很大的群体被合为一个 400 人箱，IV 为 **0.169460**。把阈值放宽、允许卡方流程继续合并时，结果为 `[1,2,3]`，IV 为 **1.123008**。后一个结果也不声称全局最优；对照证明前一个结果包含与卡方依据无关的裁剪。

### B06 · P1 · 低基数数值特征的边界生成错误

位置：[best_iv_binning.py:255](../hscredit/core/binning/best_iv_binning.py#L255)、[best_ks_binning.py:248](../hscredit/core/binning/best_ks_binning.py#L248)、[chi_merge_binning.py:241](../hscredit/core/binning/chi_merge_binning.py#L241)、[or_binning.py:440](../hscredit/core/binning/or_binning.py#L440)、[cp_sat_binning.py:311](../hscredit/core/binning/cp_sat_binning.py#L311)。

部分方法使用 `unique_values[:-1]` 作为左闭右开分箱的切点，将最小值变成空箱边界，并漏掉最后两种取值之间的边界；OR/CP 在唯一值数不超过 `max_n_bins` 时直接返回无候选。

三种取值 0/1/2，各 100 行，坏率 10%/50%/90%：`quantile` 得到正确的 `[0.5,1.5]` 三箱；`chi/best_iv/best_ks` 最终只保留 `[1.0]` 两箱；`or_tools/cp_sat` 返回一箱。该问题对计数变量和已离散化指标尤其直接。

### B07 · P1 · 搜索边界和应用边界不一致，且舍入改变实际规则

位置：[best_iv_binning.py:373](../hscredit/core/binning/best_iv_binning.py#L373)、[541](../hscredit/core/binning/best_iv_binning.py#L541)、[best_ks_binning.py:336](../hscredit/core/binning/best_ks_binning.py#L336)、[base.py:2615](../hscredit/core/binning/base.py#L2615)。

搜索阶段使用 `searchsorted(x_sorted, cut, side='right')`，即左侧包含等于 cut 的样本；应用阶段使用 `searchsorted(cuts, x, side='right')`，等于 cut 的样本属于右箱。两者只在切点不等于任何观测值时等价。

0/1/2 各 10 行，坏数 1/7/8，边界为 1：搜索阶段左箱 20 人、IV **0.639914**；最终应用左箱 10 人、IV **1.912762**。这个反例直接证明“优化的分区”与“返回的分区”不是同一个。遗传和单调算法中的 `pd.cut` 默认右闭，也需要统一检查。

另外 `_round_splits()` 直接舍入用于 transform 的边界，没有在舍入后统一去重、校验区间。数值缩小到微量级后，`quantile` 的四个切点都变成 0，`n_bins_=5`，实际只有 1 个有样本的箱。4 位精度不是纯展示选项；老客评分的第三方差异中也已看到边界舍入改变人数。

### B08 · P1 · 显式 `target_bad_rates` 无法正常拟合

位置：[target_bad_rate_binning.py:193](../hscredit/core/binning/target_bad_rate_binning.py#L193)、[base.py:378](../hscredit/core/binning/base.py#L378)。

构造函数替换了传入的目标坏率列表，不符合 sklearn 克隆对构造参数的要求。直接类和统一入口传 `target_bad_rates=[0.1,0.5,0.9]`，均在 `fit()` 的事务克隆阶段抛出 `RuntimeError: ... modifies parameter target_bad_rates`，尚未进入算法。

因此基线矩阵中不指定目标坏率的默认模式能跑通，并不能证明该方法的主要配置入口可用。

## 4. 与 optbinning 差异最大的设计环节

### 4.1 预分箱是结果映射，并非候选集上的联合寻优

当前真实路径是 `_fit_with_prebinning()` → `_fit_with_method(X,y)` → `_restrict_to_prebin_candidates()`。[源码](../hscredit/core/binning/optimal_binning.py#L930)。

主算法先在原始数据上独立拟合，再把每个结果切点映射到数值距离最近的预分箱切点。它不会直接把预分箱的好坏计数和可选边界作为主算法优化问题。多个切点可能映射到同一个候选；为了补足最少箱数，还会从剩余候选的前部补点。最后一次映射后没有再做完整约束验收。[映射实现](../hscredit/core/binning/optimal_binning.py#L1076)。

调用追踪确认 `cp_sat/or_tools/tree/genetic` 在预分箱模式下均进入各自主算法。文件中的 `_fit_with_method_and_prebins()` 是未被当前主路径调用的遗留实现，不能把它的分支当作当前行为。

一个无纯箱、同候选、同 2–4 箱和 5% 人数约束的反例：hscredit 映射后的 IV 为 **1.364011**；枚举全部合法候选分区的最优值为 **1.368790**；optbinning 使用同一 `user_splits` 得到同样的最优规则和 `OPTIMAL` 状态。差值本身不大，但证明映射并不等于候选上的最优求解。

### 4.2 名称为 CP-SAT / OR-Tools，不代表当前路径使用这些求解器

- `CPSATBinning` 的当前数值主路径调用 `_candidate_search.py`。无单调 IV 使用动态规划；其他目标或单调要求使用组合枚举和时间限制。对 `CpSolver` 设置拦截后，正常拟合仍完成，实际调用次数 **0**。
- `ORBinning` 默认 `use_cp_sat=False`，组合 DP、贪心与局部搜索，再按 `_score_splits()` 评价。即使 `objective='iv'`，最终评分也加了形状、Lift 和箱数利用项，因此不是纯 IV 最大化。[评分实现](../hscredit/core/binning/or_binning.py#L1654)。
- `use_cp_sat=True` 的 IV/KS/Gini 分支仍调用上述候选搜索。真实 `_cp_sat_solve()` 的遗留实现不能作为当前主路径已经建立 CP 模型的证据。
- 约束组合枚举随候选数和最大箱数迅速膨胀；超时得到当前可行解与已证明最优解应有可查询的差别。当前主要通过 warning 表达，缺少 optbinning 那样的求解结果状态接口。
- `or_tools` 的 `n_prebins` 仅被保存，当前实现不读取；默认非 CP 路径的 `time_limit` 也未传给其 DP/局部搜索。`cp_sat` 的 `num_workers` 当前没有实际使用。

动态规划本身并不低于 CP，也可以在明确条件下证明最优。问题是实际算法、支持的约束、目标函数和接口承诺没有一致表达。optbinning 0.20.1 的 CP 路径确实建立模型、调用 `CpSolver` 并取得求解状态。[对应版本源码](https://github.com/guillermo-navas-palencia/optbinning/blob/v0.20.1/optbinning/binning/cp.py)。

## 5. 17 种方法逐项对照

下表中的“风险”区分已经复现的行为与源码可确认的算法选择，不把不同的设计一律列为公式错误。

| 方法 | 当前实现与第三方关系 | 主要不足或使用边界 |
|---|---|---|
| `uniform` | 等宽边界，接近 toad `step` | 统一入口可为人数约束合并边界，因此最终不一定等宽；直接类与统一入口行为也不完全相同。小数舍入可造成重复边界。 |
| `quantile` | 分位数切点，重复边界合并 | 常规真实数值与 toad quantile 一致；低基数分支、显式分位数与普通分支约束处理不同；最大箱人数和不可行状态缺少统一验收。 |
| `tree` | sklearn 分类树，最接近 toad `dt` | 默认最小叶节点 5%，还有独立的 `min_bin_size`；取整、精度、默认 Lift 调整造成差异。完全对齐后本次三字段箱号一致。 |
| `cart` | sklearn 树先生成 `max(4×max_n_bins,20)` 个叶节点，再做事件率、显著性、人数和通用后处理 | 不是 optbinning 完整最优分箱；也不等于简单限制 5 叶的树。最大人数的局部处理仍尝试合并过大箱，方向值得重构；通用层不能保证后续修复成功。 |
| `chi` | 相邻 Pearson 卡方合并；较多唯一值时先压到约 100 个分位预箱 | 与 toad 0.1.4 的全唯一值起始、`balance=True` 样本量加权及并列最小值合并不同；另有 B05/B06 的确定错误。 |
| `best_ks` | 贪心添加使全局 KS 最大的切点 | 一旦包含全局 KS 最大位置，很多新增切点都得同样的 KS，后续选择容易被候选顺序支配；不是对每个子区间递归求局部 KS。还有边界和精度问题。 |
| `best_iv` | 最多约 100 个候选点，逐次贪心添加 | 不保证全局最优；搜索评分忽略纯好/纯坏箱，和最终统计的平滑口径不同；默认 Lift 后处理还会改掉其目标。 |
| `mdlp` | 以 IV 选择候选，再用 MDL 熵准则停止；阈值默认乘 0.7，并强制补箱 | 是已在模块说明中声明的改造版，不能当作标准 MDLP。`force_min_bins` 补箱目标实际用 `max_n_bins-1`；200 个交替 0/1 标签，关闭强制补箱无切点，开启后补出 4 个边界。 |
| `or_tools` | 默认多种启发式/DP结果加复合评分；可选择候选搜索路径 | IV 目标混入其他评分、低基数退化、控制参数未使用；不是默认建立整数规划模型。 |
| `cp_sat` | 当前为候选 DP/组合枚举 | B01 的未排序输入是首要错误；另有低基数退化、边界语义、参数和状态接口问题。 |
| `kmeans` | SciPy `kmeans2(minit='points')` 多次初始化；toad 使用 sklearn KMeans | 初始化和收敛流程不同；实现会设置全局 NumPy 随机种子，多特征并行下应隔离随机状态。人数与 Lift 后处理会使最终边界脱离聚类中心中点。 |
| `monotonic` | 等频/等距预箱后逐次合并违规相邻箱 | 是贪心形状修整，不是带单调约束的 IV 联合优化；最小箱数可能阻止完成单调合并。`convex/concave` 通过谷/峰处理，并不保证二阶凸凹约束。 |
| `genetic` | 分位候选的布尔染色体，选择/交叉/变异，停滞 20 代早停 | 随机近似算法，不保证全局最优；适应度用右闭 `pd.cut`，最终应用边界需要对齐；部分约束只靠后处理。 |
| `smooth` | 分位数+树候选，收缩坏率辅助挑选合并对象 | 最终分箱表仍是经验坏率，不是收缩概率表；`merge_criterion` 只保存未使用。统一入口的 method 名与直接类平滑 method 还共用同一参数名，不能直接以同名 kwargs 选择平滑类型。 |
| `kernel_density` | KDE 寻谷，默认 `use_target=True, fallback_to_iv=True`，可转 IV 搜索 | 虽被统一入口归为无监督，默认实际使用 y；`iv_weight` 未使用；`isj` 实现是基于参考带宽加偏度/峰度系数的启发式，而非可由该名称推断出的完整 ISJ 求解流程。 |
| `best_lift` | 分位预箱后贪心保留极端 Lift/最大跨度等 | 适合特定策略目标，不应以最大训练 IV 判断优劣；预箱按排序行位置切割，会涉及重复值不可拆分问题；`min_lift` 未使用。 |
| `target_bad_rate` | 默认按两侧坏率差挑切点；显式目标路径匹配从最左端累计的坏率 | 累计前缀坏率不是每个最终区间坏率；`strict_mode` 未使用；显式列表还会触发 B08 的克隆错误。 |

源码入口均位于 [binning 模块](../hscredit/core/binning/__init__.py)。标准 MDLP 原始工作以熵最小化选择切点，并用描述长度准则停止，和这里的 IV 候选排序有本质区别。[Fayyad–Irani 原论文](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf)。

需要额外注意的实现事实：

- `strict_mode、min_lift、merge_criterion、iv_weight、or_tools.n_prebins` 在对应当前类中只有赋值，没有读取；全仓相关调用链搜索和端到端切换参数实验均支持“当前不生效”的判断。
- 数值单调与合并过程多处调用 `_compute_bin_stats()`，后者未设置 `round_digits=False`，会用默认展示精度的坏率参与中间比较。应统一使用未舍入的充分统计量。
- MDLP 普通区间搜索仅去掉缺失，没有排除 `special_codes`。补充实验添加特殊值后边界从 3.5 变为 4.0，本例普通样本箱号没有改变；这是预处理一致性问题，不把该例夸大成已证实的 IV 损失。

### 5.1 类别和二维分箱

类别字段先按训练坏率排序编码，再调用数值算法，最后还原分组并做类别约束校验。这是有监督类别排序，必须在训练折内拟合；它也意味着把数值字段改成类别字段，可能走不同的边界修复和不可行处理。

真实 `商品类别` 的 17 种方法均完成，未知类别输出箱号也已检查。此结果不能替代任意长尾类别、高基数或未知类别分布的验收。

`OptimalBinning2D` 先分别拟合两轴、生成网格，再按相邻连通区域贪心合并，选择顺序综合小箱、形状违规和 IV 损失。输出区域可以不是矩形；名称中的 Optimal 不代表在全部二维分区上求全局最优。真实数据最终 5 箱人数为 570/67/96/180/57，合计 970；本次没有做与 optbinning 二维求解器的等候选最优性对照。

## 6. 建议修复顺序与验收条件

### 第一阶段：修正可确定的错误

1. 修正 CP 输入排序；让候选搜索对已排序的 x/y 成对计算，增加乱序、倒序和重复值的置换不变性验证。
2. 统一边界闭开约定、合法切点生成和低基数处理。搜索得到的训练分区必须和 transform 逐行一致。
3. 修复 sklearn 参数完整保留。直接类、统一入口、clone、Pipeline/CV 应保留完全相同的业务参数；构造函数不能替换参数对象以破坏克隆契约。
4. 建立数值最终约束验收：箱数、人数、好坏人数、单调性、固定切点同时检查。无可行解时明确抛错或提供机器可读的求解状态，不返回伪合规结果。
5. 卡方超箱时继续按明确定义的合并目标处理，删除任意按位置裁剪逻辑。
6. 精度只用于标签显示；若必须改变实际切点，必须去重并重新验收人数、形状和候选可行性。

这些验收信息应放在分箱器属性、异常或审计元数据中，不要求给业务结果表新增状态、原因或“不适用”列。

### 第二阶段：拆开算法目标与可选后处理

1. 将 Lift 调整变为明确的可选步骤，保存原始算法切点和最终切点，说明每次改变的目标。变更默认值需要兼容策略，不直接用本次实验决定全项目切换。
2. 将预分箱结果转换为候选边界及好坏充分统计，主算法只在这些原子箱上搜索；不先自由拟合再映射。
3. `or_tools/cp_sat` 明确声明实际后端、目标、可支持的约束和求解结果；纯 IV 模式不得混入 Lift 或形状奖励。
4. `best_ks` 明确采用递归局部 KS 还是全局 KS 加次级目标，避免等分值下用候选顺序代替有意义的准则。
5. MDLP 提供标准模式与当前改造模式的明确区别，修正“最少箱数”补到“最多箱数”的行为。
6. 无效参数要实现或拒绝，避免“成功接收但不影响结果”。统一各方法的能力说明。

### 第三阶段：重新做公平效果比较

使用相同训练/验证/OOT 划分、候选集、箱数范围、人数阈值、纯箱政策、单调趋势、特殊值、边界精度和统计平滑参数，再比较 IV、KS、训练箱风险映射的留出 AUC、稳定性和耗时。

可以把 toad 和 optbinning 作为独立参考实现，但不能仅因为它们输出更高训练 IV 就将其作为正确答案。小候选集先用穷举确认最优值；大候选集再测最优界、时间限制和可行性。

## 7. 当前可执行的使用建议

如果目的是复核同名树分箱，已验证下面的配置能在三个真实字段上与本机 toad DT 对齐：

```python
min_count = int(np.ceil(len(X_train) * 0.05))
binner = OptimalBinning(
    method='tree', max_n_bins=5, min_n_bins=2,
    min_bin_size=min_count, min_samples_leaf=min_count,
    decimal=12, lift_refine=False, monotonic=False,
    random_state=42, n_jobs=1,
)
binner.fit(X_train, y_train)
```

这是本次数据上的可比基线，不是所有变量的最佳参数。若进入 sklearn CV，目前还必须防止 B03 的外部 clone 丢失控制选项，不能据此宣称完整 Pipeline 已解决。

`cp_sat`、显式目标坏率、严格大小/方向约束应先处理相应缺陷。仅预先排序输入或提高 `decimal` 不能解决其余问题，也不能作为最终修复。

## 8. 复现与验证记录

已执行的既有测试命令：

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$auditPython = '.\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
& $auditPython -m pytest tests/test_binning/test_binning_contracts.py tests/test_binning/test_binning_constraints_regression.py tests/test_binning/test_or_binning_dual_api.py tests/test_binning/test_best_binning.py -q --tb=short --junitxml=.audit_tmp/binning-audit-existing-tests.xml
```

2026-10-27，Python 3.11.5，以上四模块：102 passed，1 warning，47.88 秒。原始日志保存在 `.audit_tmp/binning-audit-existing-tests.log`。该通过结果与反例并不矛盾：现有用例没有覆盖这些输入排列、克隆选项、精度和冲突约束。

Notebook 按单元保存了实际输出，也以指定解释器启动真实 Jupyter 内核整体执行。复现入口是 Notebook，不维护对应 `.py` 演示。可用同一解释器作为 Jupyter 内核运行，或从仓库根目录执行：

```powershell
@'
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = Path('docs/validation/binning-method-audit-2026-10-27.ipynb').resolve()
nb = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(nb, timeout=300, km=manager,
               resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(nb, path)
'@ | & $auditPython -
```

本次记录了依赖的弃用/Windows 核数探测警告；这些环境警告没有被当作业务算法错误。未重新运行全仓测试，未测量覆盖率，也未验证最新版本第三方库、百万行性能或生产业务收益。分箱源码在审计首尾通过 SHA256 一致性检查，其他原有工作区改动不纳入本次结论。
