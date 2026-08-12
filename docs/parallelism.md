# 并行执行指南

HSCredit 的分箱器、特征筛选器、编码器、规则执行与规则挖掘组件，以及特征、规则、逾期、漂移和模型报告，统一使用 joblib 调度独立任务。并行只改变任务调度方式，不改变算法、样本、候选集合、迭代次数或数值精度。

## 并行数 `n_jobs`

设机器可用的物理 CPU 核数为 `P`。HSCredit 优先通过 joblib 查询物理核数；旧版 joblib 不支持该查询时依次退回逻辑核数和 `os.cpu_count()`，仍不可用时按 1 核处理。

| 配置 | 含义 |
|---|---|
| `-1`（默认） | 自动并行。先建立 `min(P - 1, ceil(0.8 × P))` 的总预算，再根据任务数、行列规模、数据字节数、单项成本和实现能力决定实际 worker；小任务会自动串行。 |
| 正整数 | 使用明确指定的 worker 总预算；例如 `4` 表示最多 4 个 worker。显式设置优先于自动工作量策略，仅受任务数、实现能力和活动嵌套总预算约束。 |
| 整数值浮点数 | 与对应整数完全相同；`1.0` 与 `1` 都只使用 1 个 worker，`4.0` 与 `4` 相同。 |
| `0 < n_jobs < 1` | 按 `ceil(P × n_jobs)` 向上取整；例如 8 个物理核上的 `0.5` 为 4 个 worker。 |
| `None` | 兼容旧调用，强制串行执行。新代码建议明确使用 `1` 表达串行意图。 |

`bool`、`0`、小于 `-1` 的整数、非整数且大于等于 1 的浮点数、非有限浮点数以及其他类型会在创建 worker 前抛出中文 `ValidationError`。

## 后端选择

`parallel_backend` 可显式选择 joblib 已注册的后端。常用值为：

- `threading`：线程共享同一个 DataFrame，不需要序列化大对象，适合 NumPy、pandas 或第三方实现会释放 GIL 的列级计算，也适合 I/O 较多的任务。字符串/分类列只有在底层操作释放 GIL 时才能获得计算加速，但仍可避免进程序列化和重复内存。
- `loky`：隔离的进程 worker，适合可序列化、Python 计算较重的纯函数任务。进程启动和数据传输有固定成本，小任务不一定更快。
- `multiprocessing`：保留兼容性；Windows 创建子进程时应从正常 Python 模块或 `if __name__ == "__main__":` 入口运行，不要把不可序列化的局部函数传给 worker。
- `None`：由具体模块选择适合工作负载的默认后端，或使用 joblib 默认值。

显式后端与 `parallel_config["prefer"]`、`parallel_config["require"]` 必须兼容。例如进程后端不能同时要求 `sharedmem`。不兼容或未注册的后端会在实际执行前转换为中文配置错误。

## `parallel_config`

支持以下稳定配置键。HSCredit 会复制配置再交给 joblib，不会修改调用者传入的字典。

| 键 | 允许值与用途 |
|---|---|
| `adaptive` | 布尔值，默认 `True`。仅在 `n_jobs=-1` 且未显式选择后端时按工作量收缩 worker；显式 `n_jobs` 或显式后端始终优先。设为 `False` 可关闭自动工作量收缩。 |
| `prefer` | `"threads"`、`"processes"` 或 `None`，给 joblib 的软后端偏好。 |
| `require` | `"sharedmem"` 或 `None`，给 joblib 的硬约束。 |
| `batch_size` | `"auto"` 或正整数，控制每批任务数量。 |
| `pre_dispatch` | `"all"`、整数或 joblib 表达式字符串，控制预提交任务数。 |
| `max_nbytes` | 整数、容量字符串或 `None`，超过阈值的 NumPy 数组可使用内存映射。 |
| `mmap_mode` | `"r"`、`"r+"`、`"w+"`、`"c"` 或 `None`。只读计算通常使用 `"r"`。 |
| `temp_folder` | 内存映射临时目录或 `None`。目录应有足够空间，并允许所有 worker 访问。 |
| `timeout` | 单个批次超时秒数或 `None`。 |
| `verbose` | joblib 调度日志级别整数。 |
| `inner_max_num_threads` | 进程后端中每个 worker 的第三方线程上限；`threading` 后端不支持。 |
| `backend_kwargs` | 独立的后端专属字典，例如 loky 的 worker 生命周期配置。 |

`n_jobs` 和 `backend` 不允许重复写入 `parallel_config`，应分别使用公共参数 `n_jobs` 和 `parallel_backend`。未知键、非字符串键、非字典 `backend_kwargs` 及已知冲突都会抛出中文 `ValidationError`。具体值的最终兼容性仍取决于已安装的 joblib 版本和所选后端；HSCredit 保持 `joblib>=1.0` 兼容，不依赖仅在新版本存在的接口。

大数组使用进程后端时，可设置：

```python
parallel_config={
    "max_nbytes": "64M",
    "mmap_mode": "r",
    "temp_folder": r"D:\joblib-cache",
}
```

线程后端天然共享内存，通常不需要 mmap。内存映射只减少大数组的进程间复制，不能避免 pandas 对象序列化、结果对象占用内存或最终表格合并所需内存。任务过细时，可调整 `batch_size` 和 `pre_dispatch`，但不应通过抽样、降低精度或减少候选换取速度。

## 嵌套预算

只有“外层 worker 正在并行执行，且每个外层 worker 同时再次启动并行子任务”才属于真实嵌套。设当前可用预算为 `C`、外层独立任务数为 `T`：

```text
outer_workers = min(T, ceil(sqrt(C)))
child_budget = max(1, floor(C / outer_workers))
```

该预算上下文会传播到线程和进程 worker。`n_jobs=-1`、显式正整数、比例预算以及第三方模型的 `thread_count`/`num_workers` 都受活动子预算约束；因此父子层不会各自重复使用整机核数。`inner_max_num_threads` 仍可用于进一步限制进程 worker 内的 BLAS/OpenMP 线程。

按顺序执行的多个阶段不拆分预算。例如 Composite 筛选器逐级筛选、Stepwise/RFE 的依赖轮次、报告的先计算后写入、多个 Excel sheet 顺序渲染，都在当前阶段开始时使用完整可用预算。只有外层特征/标签/数据集任务与其内部候选计算确实重叠时才切分。

## 一致性、随机性与失败行为

- 串行和并行调用相同的单任务 worker；结果按提交顺序返回，由主线程按原特征、规则、标签和数据集顺序提交。
- 不按数据行拆分浮点归约，不改变候选、迭代次数或输入 dtype。DataFrame 的索引、列名、列顺序、dtype 和缺失值语义应保持一致。
- 拟合任务先写临时结果，所有任务成功后才由主线程一次性提交。首次拟合失败不留下半成品；重新拟合失败保留旧的有效状态。
- 有 `random_state` 的算法应固定基础种子；子任务种子由稳定任务顺序决定，而不是 worker 编号或完成顺序。`random_state=None` 保留算法原有的非确定语义。
- worker 失败时不会静默重试、跳过或改用近似算法。错误包含中文阶段/任务标识，并保留原始异常链。
- `timeout` 在单 worker 路径同样生效；不会因为自适应退化为串行而被绕过。
- Excel 写入、绘图和最终文件替换在主线程中按固定顺序完成；并行 worker 只计算表格、指标或绘图所需数据。

第三方模型可能使用自己的 OpenMP、BLAS 或 GPU 线程，HSCredit 无法完全控制其内部归约顺序。因此固定随机种子仍可能出现平台级末位浮点差异；这类差异应只对明确字段使用严格局部容差，不能把整张表降级为宽松比较。

## 使用示例

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.report import ModelReport

# 默认自动并行：总预算约为物理核的 80%，小任务自动串行。
binner = OptimalBinning(n_jobs=-1)

# 物理核数的 50%，线程共享输入 DataFrame。
encoder = WOEEncoder(
    n_jobs=0.5,
    parallel_backend="threading",
    parallel_config={"batch_size": 4, "pre_dispatch": "2*n_jobs"},
)

# CPU 较重的多数据集模型报告使用进程，并限制每个进程的内部线程。
report = ModelReport(
    model,
    datasets=data,
    n_jobs=4,
    parallel_backend="loky",
    parallel_config={
        "batch_size": 2,
        "max_nbytes": "64M",
        "mmap_mode": "r",
        "inner_max_num_threads": 1,
    },
)
```

同一组公共参数也适用于公开分箱器、筛选器、编码器、`Rule`/`RuleFlow`、`RuleSet`/`RulesClassifier`、规则挖掘器、`OverduePredictor`、Swap/漂移报告、`ModelReport`、`auto_model_report` 和 `compare_models`。单次标量辅助对象或函数（如规则条件、单个指标计算、Excel writer）不创建批量任务，因此不增加无意义的并行参数。

EDA 的批量 IV、特征重要性、异常值、稀有类别、集中度、时间稳定性、PSI、特征/评分漂移、客群画像与客群监控等入口同样接受 `n_jobs`、`parallel_backend` 和 `parallel_config`。统计计算可并行，Excel 写入和最终结果组装仍在主线程按固定顺序完成。

`ModelReport` 对已经训练的模型只做读取和预测，不会为并行执行改写模型参数。VIF 计算继续使用原生 statsmodels 逻辑，不为并行化重写算法。

## pandas apply 并行扩展

导入 `hscredit` 后，DataFrame、Series、DataFrameGroupBy 和 SeriesGroupBy 都会注册
`hscredit` 链式代理。代理后的 `apply` 保留当前 pandas 版本的原生参数边界和结果装配：

```python
import hscredit

rows = df.hscredit(n_jobs=-1, bar=True).apply(score_row, axis=1)
values = df["amount"].hscredit(n_jobs=4, bar=False).apply(normalize)
summary = (
    df.groupby(["month", "channel"])
    .hscredit(
        n_jobs=-1,
        bar=True,
        parallel_backend="loky",
        parallel_config={"batch_size": "auto", "pre_dispatch": "2*n_jobs", "timeout": 300},
    )
    .apply(summarize, include_groups=False)
)
```

- `n_jobs=-1` 使用统一自适应预算；任务很小或只有一项时自动串行，足够大时才启动多 worker。
- 后端决策严格发生在执行前：纯 Python 且可序列化的 callable 使用 `loky` 进程，明确的 NumPy/内置归约使用 `threading`，不可序列化的 callable 回退到线程。显式 `parallel_backend` 优先。
- 不抽样、不测速、不调用用户函数进行能力探测。每一行、列、元素或分组在单次 `apply` 中最多调用一次，失败不会静默重试。
- `bar=True` 只按已真实完成的逻辑项推进；失败、超时或中断时立即关闭，不伪造 100% 完成。
- 用户函数异常、`TimeoutError` 和 `KeyboardInterrupt` 保留原始异常类型。Jupyter 中断只终止当前调用并把错误交回当前 kernel，不主动终止或重启 kernel；已经存在的变量仍保留。
- 多进程 worker 中对普通 Python 全局变量的修改不会回写主进程；需要共享副作用时应显式选择 `threading`，或让函数返回结果后在主进程合并。

`raw=True`、`result_type="broadcast"`、NumPy ufunc、非 Python engine 和 pandas 需要整体对象语义的调用会安全委托给原生 `apply`。这些兼容路径仍只执行一次，但不会拆分为并行任务。

Windows 上第一次使用 `loky` 需要启动 worker 并导入依赖，可能明显慢于后续调用。严格单次契约下 HSCredit 不会隐藏执行一次 UDF 来预热或测速，因此短任务应保留 `n_jobs=-1` 让静态成本模型自动串行；确认任务足够重时可显式设置正整数 `n_jobs`。同一 Python/Jupyter 进程内 joblib 会复用可用 worker，稳定进程池下的重 CPU UDF 才是多进程加速的主要场景。

## 性能和资源限制

并行并不保证每个工作负载都更快。默认 `n_jobs=-1` 且未显式选择后端时，自适应策略会让小数据、少特征、少规则和已向量化任务保持串行；计算量足够大时才扩展 worker。高维混合类型数据通常优先使用共享内存的 `threading`，纯 Python CPU 重且可序列化的任务使用 `loky`。用户显式设置 `n_jobs`、后端或 `adaptive=False` 时以显式设置为准。基准测试应先预热，并对串行和并行分别运行至少三次后比较中位数。

### CorrSelector 超宽表

`CorrSelector` 的无缺失 Spearman 路径会按列并行排名；分块相关计算则把同一份
`n_jobs` 总预算分配给外层 joblib 任务和矩阵乘法使用的 BLAS/OpenMP 线程。例如
`n_jobs=12` 且当前只有 3 个相关块任务时，每个任务最多使用约 4 个原生线程；任务数达到
12 后，每个任务限制为 1 个原生线程。这样可避免形成“12 个外层任务 × 每项 12 个
BLAS 线程”的超额并发。

默认 `threading` 后端共享大型输入矩阵，因此任务管理器中通常仍只显示一个 Python
进程；一个进程内的多个线程可以同时运行在多个 CPU 核心上，不能仅根据 Python 进程数
判断是否使用了多核。Windows 上对数 GB 矩阵强制使用 `loky` 可能产生较大的序列化、
内存映射和临时磁盘开销。

精确相关筛选在最坏情况下仍需比较所有特征对，复杂度随特征数平方增长。实现会在后续
块计算前排除已经淘汰的特征，但不会抽样、降精度或切换近似算法。可用以下命令生成
与实际规模一致的诊断；串行对照仅在显式增加 `--compare-serial` 时执行：

```powershell
python scripts/benchmark_corr_selector.py --rows 6194 --features 67793 --n-jobs 12
```

当内存不足、临时目录空间不足、对象不能 pickle、第三方库不支持多进程或后端超时时，HSCredit 会报告错误而不会损失精度继续运行。可依次减小 `n_jobs`、降低 `pre_dispatch`、启用 mmap、选择 `threading` 或将第三方模型内部线程设为 1；不要用降低样本量、分箱候选数或迭代次数掩盖资源问题。
