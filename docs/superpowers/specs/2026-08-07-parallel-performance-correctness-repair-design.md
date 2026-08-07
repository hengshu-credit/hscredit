# HSCredit 并行性能与预算传递修复设计

## 1. 背景

HSCredit 已按 `2026-08-04-unified-parallel-execution-design.md` 建立统一并行 API，
但实际使用 `CorrSelector` 处理 `(6194, 67793)` 的超宽数据时，配置
`n_jobs=12` 仍仅观察到少量活动任务，运行二十分钟未完成。当前相关性实现将
`corr_block_size=512` 同时作为内存上限和任务粒度，并按列块顺序推进；前几个列块
只能产生 1 到 5 个外层任务。每个任务中的 NumPy 矩阵乘法还可能再次启动 BLAS
线程，造成外层 joblib 与内层 BLAS 超额并发。

当前实现还存在两个同源问题：

- `FeatureImportanceSelector` 和 `RFESelector` 暴露 `n_jobs`，但没有把解析后的预算
  传给底层 estimator；
- `BorutaSelector` 只为内部创建的默认随机森林设置 `n_jobs`，用户传入自定义
  estimator 时该配置失效。

共享执行器在任务数收缩为 1 后会提前走串行路径，部分后端冲突或未知后端因此没有
在执行前暴露。现有测试主要验证参数存在、串并行结果一致及异常事务性，没有覆盖
真实 worker 参与、底层 estimator 收到的预算或 BLAS 超额并发。

## 2. 目标

本次修复必须实现：

1. `CorrSelector(n_jobs=N)` 在超宽数据的排名和相关计算阶段实际使用不超过调用者
   预算的多核能力，而不是由默认块大小意外退化为少量 worker；
2. 保持相关性方法、权重稳定排序、严格 `correlation > threshold` 判断、强制保留/
   剔除规则及贪心选择顺序不变，不抽样、不降精度、不减少候选；
3. 已淘汰特征不再参加后续跨块矩阵乘法，减少确定无用的相关计算；
4. joblib 外层任务与 BLAS/OpenMP 内层线程共享同一个 `n_jobs` 总预算，避免乘法式
   超额并发；
5. `FeatureImportanceSelector`、`RFESelector` 和自定义 estimator 的
   `BorutaSelector` 将解析后的预算传给底层 estimator，同时保持用户对象不变；
6. 单任务路径仍完整验证 `parallel_backend` 和 `parallel_config`；
7. 为全部公开并行入口补充“实际工作发生在哪里”的契约审计，防止再次出现只暴露
   参数但昂贵计算未使用预算的情况。

## 3. 非目标与性能边界

本次不引入近似最近邻、随机投影、行抽样、float32 降精度、GPU 依赖或静默算法切换。
`67793` 个特征包含 `2,297,911,528` 个唯一特征对；若绝大多数特征最终都被保留，
任何精确算法仍具有二次级最坏计算量。修复保证消除当前无效计算和预算浪费，但不承诺
对任意独立特征集合实现线性复杂度。

单个联合模型若没有公开 `n_jobs` 参数、没有使用 joblib 且自身不支持多线程，
HSCredit 不创建无意义的单元素外层任务来伪造并行。此时仍验证公共并行配置，并保持
模型原有执行语义。

## 4. CorrSelector 精确并行流水线

### 4.1 工作预算

在相关计算开始时，根据 `self.n_jobs`、活动嵌套预算、物理核心数和特征数解析本阶段
总预算 `total_workers`。显式正整数继续遵循现有公共契约；自动模式受活动预算限制。

每一轮列块相关计算根据实际任务数得到：

```text
outer_workers = min(total_workers, current_task_count)
native_threads = max(1, floor(total_workers / outer_workers))
```

默认共享内存线程后端下，joblib 最多启动 `outer_workers` 个任务，并通过
`threadpoolctl.threadpool_limits` 将该轮 BLAS/OpenMP 线程限制为
`native_threads`。因此一项大型矩阵乘法可以使用完整原生线程预算，五项任务会均分
预算，任务数达到预算后每项只使用一个原生线程。

显式 `loky`/`multiprocessing` 后端继续由 joblib 进程隔离和
`inner_max_num_threads` 控制内层线程；父进程的线程限制不冒充对子进程生效。

### 4.2 Spearman 排名

无缺失 Spearman 快速路径不再调用一次性的全表 `DataFrame.rank`。改为按输入列顺序
提交独立排名任务，每项返回列序号和 `float64` 排名数组，主线程按序写入预分配的
二维数组。默认使用 `threading`，因为 pandas 排名释放 GIL 且列任务共享输入数据；
用户显式兼容后端仍可覆盖默认值。

Pearson 不执行排名。Kendall、含缺失 Pearson/Spearman 继续使用兼容相关计算路径，
不得因本次优化改变 pandas 的缺失值语义。

### 4.3 归一化与跨块计算

归一化继续使用 `float64`，常量列继续写为 `NaN`。矩阵块上限仍由公开
`corr_block_size` 控制，避免创建完整的 `p × p` 相关矩阵。

处理一个后续列块时，worker 必须先根据主线程已经确定的 `kept_rows` 选择实际保留的
左侧列，再执行矩阵乘法：

```text
left = values[:, row_start:row_stop][:, kept_positions]
corr = left.T @ right
```

当前代码在完整左块矩阵乘法之后才过滤 `kept_positions`，会持续计算已淘汰特征；
修复后返回的相关特征索引必须映射回原排序位置。当前列块内部仍先计算对角相关块，再由
主线程按特征顺序执行相同的贪心保留决策，从而保持依赖语义。

### 4.4 确定性

任务完成顺序不得影响结果。所有排名列、相关块候选和报告记录按提交序号归并。串行与
并行必须得到相同的选中特征、剔除特征、关联特征及稳定平局顺序；浮点相关系数使用
现有严格局部容差验证，不放宽整张报告比较。

## 5. estimator 预算传递

筛选器基类新增内部 helper，职责限定为：

1. 克隆调用者 estimator；
2. 解析当前筛选器的有效 worker 预算；
3. 查找 estimator 参数树中的 `n_jobs`；
4. 将最浅并行层设置为有效预算，更深嵌套层设置为 1；
5. 返回隔离克隆，不修改调用者传入的 estimator。

`FeatureImportanceSelector` 使用该克隆完成一次联合拟合；`RFESelector` 将该克隆交给
sklearn RFE，使每轮底层拟合继承预算；`BorutaSelector` 无论使用默认模型还是自定义
模型，都在每轮 clone 前基于同一 helper 准备基础 estimator。

如果 estimator 参数树不存在 `n_jobs`，helper 保持模型原状。公共
`parallel_backend`/`parallel_config` 仍在昂贵拟合前完成验证；能通过 joblib 后端上下文
影响的 estimator 继承所选后端，但 HSCredit 不假定任意第三方 estimator 必然使用
joblib。

## 6. 共享运行时修复

`parallel_execute` 在空任务或单 worker 返回之前完成以下工作：

- 校验通用配置键；
- 解析有效后端；
- 校验 threading 与 `inner_max_num_threads` 等已知冲突；
- 验证显式 joblib 后端是否已注册或可构造。

空任务仍不创建 worker，单任务仍直接执行同一个 worker 函数，但配置错误不得因为任务
数量少而被静默忽略。

原有任务顺序、异常链、事务提交和 `n_jobs=None` 串行兼容语义保持不变。

## 7. 全量并行审计

以 `tests/test_parallel_api_contract.py` 的公开并行入口清单为基准，为每类入口声明并验证
其昂贵计算采用以下一种真实策略：

- 按特征、列、规则、标签、数据集或候选调用共享执行器；
- 顺序外层中的当前独立候选调用共享执行器；
- 将预算传给真正执行计算的底层 estimator、binner 或 child component；
- 算法本质顺序且底层不支持并行时，明确记录该边界，不创建 identity worker。

分箱器继续以 `BaseBinning._fit_features` 和 `_transform_features` 的按特征任务为统一
入口，并验证多特征输入至少有两个真实 worker 参与。已有 `OptimalBinning` 与
`OptimalBinning2D` 子分箱器预算传递测试保留，并补充回归审计，避免 wrapper 参数只停留
在构造器。

## 8. 错误处理

新增或调整的用户可见错误继续使用中文 `ValidationError`。worker 中的特征名、块序号
或 estimator 拟合错误保留原始异常链。任何失败不得提交部分筛选状态，也不得修改用户
传入 estimator、权重字典或并行配置字典。

资源不足、临时目录不足或进程序列化失败时不自动降精度、抽样或切换近似算法。

## 9. 测试与验收

### 9.1 TDD 回归测试

先添加并观察以下测试在当前代码上按预期失败：

- 默认 `corr_block_size`、特征数小于 512 时，Spearman 排名由多个 worker 线程执行；
- 超宽分块计算在只有少量外层任务时仍按总预算分配原生线程，在任务增多后限制内层线程；
- 已淘汰左侧特征不会进入后续矩阵乘法；
- CorrSelector 串行与 threading/loky 的选择结果、相关特征及报告顺序一致；
- 三个 estimator 型筛选器在 `n_jobs=12` 时，底层模型观察到 12，而调用者原始模型仍
  保持原值；
- 单任务和空任务使用非法后端配置时仍抛出中文校验错误；
- 代表性筛选器和分箱器的批量输入实际出现多个 worker 身份。

### 9.2 性能验证

新增可独立运行的 CorrSelector 基准脚本，接受行数、特征数、相关簇比例、`n_jobs`、
后端和块大小参数，打印：

- 排名、归一化、跨块相关和报告阶段耗时；
- 每轮外层 worker/原生线程预算；
- 串并行总耗时与加速比；
- 实际参与的线程或进程身份数量；
- 选中与剔除特征数量及一致性摘要。

常规 slow 测试使用能在 CI 内完成的缩小数据验证并行中位数不退化。用户实际
`(6194, 67793)` 规模由基准脚本运行；若本地内存或执行窗口不足以完成串行对照，至少
执行并行路径并记录阶段进度、CPU 预算和已完成块数量，不能伪造速度结论。

### 9.3 完成标准

- 新增回归测试完成红绿循环；
- CorrSelector 定向测试、全部筛选器测试、全部分箱器并行测试和公共 API 契约通过；
- 非 slow、非 integration 全量测试无新增失败；
- `git diff --check`、格式和相关静态检查通过；
- 基准证明确实使用请求的总 CPU 预算，且不存在 joblib × BLAS 乘法式超额并发；
- 文档明确说明“单 Python 进程可以通过线程使用多个核心”，不再以进程数量作为唯一
  并行判断标准。
