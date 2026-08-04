# HSCredit 统一并行执行设计

## 1. 背景与目标

HSCredit 的分箱、筛选、编码、规则和报告模块目前只有少量独立的 joblib 调用：分箱基类使用线程按特征拟合，部分筛选器自行创建 `Parallel`，EDA 特征概览另有一套 CPU 解析逻辑，编码器、规则器和大部分报告入口没有统一并行接口。这会造成默认值、后端、嵌套行为、异常处理和输出顺序不一致。

本次改造建立统一并行运行时，并让以下模块在存在安全、独立的任务边界时默认启用并行：

- `hscredit.core.binning` 的所有公开分箱器，包括 `OptimalBinning2D`；
- `hscredit.core.selectors` 的所有公开筛选器和组合筛选器；
- `hscredit.core.encoders` 的所有公开编码器；
- `hscredit.core.rules`、`hscredit.core.models.rules` 和 `hscredit.report.mining` 的规则执行、规则分类和规则挖掘组件；
- `hscredit.report` 的公开分析函数、分析器和报告生成器。

并行化不得减少样本、候选、迭代次数或数值精度。串行和并行必须使用同一个单任务算法，并尽可能产生完全相同的拟合状态、转换结果、规则命中和报告内容。

## 2. 公共 API

所有具有实际批量计算能力的公开组件统一暴露以下参数：

```python
n_jobs: Union[int, float] = -1
parallel_backend: Optional[str] = None
parallel_config: Optional[Dict[str, Any]] = None
```

含义如下：

- `n_jobs=-1`：自动并行，按物理核心数计算保守 worker 预算；
- `n_jobs` 为正整数：使用指定数量的 worker；
- `n_jobs` 为整数值浮点数，如 `1.0`、`2.0`：分别等价于 1、2 个 worker；
- `0 < n_jobs < 1`：使用物理核心数乘该比例并向上取整；
- `parallel_backend`：显式选择 `loky`、`threading`、`multiprocessing` 或已注册的 joblib 后端；
- `parallel_config`：配置 joblib 调度、内存映射和后端细节。

为保持旧调用兼容，`n_jobs=None` 暂时接受并按串行解释；新组件和已有目标模块的默认值统一为 `-1`。`bool`、`0`、小于 `-1` 的整数、非整数且大于 1 的浮点数及其他类型均抛出中文 `ValidationError`。

sklearn 估计器的构造函数必须显式声明上述参数，并将调用者传入的对象原样保存到同名属性。验证、规范化和字典复制延迟到执行阶段，以保持 `get_params()`、`set_params()`、`clone()`、Pipeline 和序列化契约。

单次标量辅助对象不增加无意义参数，例如 `RuleCondition`、`MinedRule` 和单个指标函数；其批量调用者负责并行调度。

## 3. 并行配置

`parallel_config` 支持下列稳定配置：

```python
{
    "prefer": "threads" | "processes" | None,
    "require": "sharedmem" | None,
    "batch_size": "auto" | int,
    "pre_dispatch": "all" | int | str,
    "max_nbytes": int | str | None,
    "mmap_mode": "r" | "r+" | "w+" | "c" | None,
    "temp_folder": str | None,
    "timeout": float | None,
    "verbose": int,
    "inner_max_num_threads": int | None,
    "backend_kwargs": dict,
}
```

`n_jobs` 和 `backend` 不允许重复放入 `parallel_config`，避免参数来源冲突。未知配置、互斥的 `prefer`/`require`/`parallel_backend` 组合以及当前 joblib 版本不支持的配置均在创建 worker 前抛出中文参数错误。

运行时保持 `joblib>=1.0.0` 兼容：公共适配器根据已安装 joblib 的调用签名选择 `Parallel` 或后端上下文支持的参数，不依赖仅在新版本中存在的 API。`backend_kwargs` 作为一个独立字典传给后端，不与通用参数混合。

当 `parallel_backend=None` 时，调用模块选择适合其工作负载的默认后端：共享大型 DataFrame 或 I/O 密集型任务优先 `threading`；Python 密集且 worker 已重构为纯函数的任务优先 `loky`。用户显式配置的兼容后端覆盖模块默认值。

## 4. CPU 预算解析

设物理核心数为 `P`，当前独立任务数为 `T`。物理核心数优先通过 joblib 获取；旧版 joblib 不支持物理核心查询时回退逻辑核心数；仍无法取得时回退 1。

自动模式的根预算为：

```python
if P == 1:
    B = 1
else:
    B = min(T, P - 1, ceil(P * 0.8))
```

因此默认模式不会占满全部 CPU。显式固定 worker 数和比例分别为：

```python
# 正整数或整数值浮点数
B = min(T, int(n_jobs))

# 0 < n_jobs < 1
B = min(T, ceil(P * n_jobs))
```

任务数始终是最终上限。显式正整数允许超过物理核心数，表示调用者主动选择超额订阅；比例模式不会超过物理核心数。

## 5. 真实嵌套并行预算

只有外层 worker 执行期间会再次启动内部并行 worker 时，才进行嵌套预算分配。调用点必须明确声明 worker 是否具有同时并行的子任务，不能仅根据内部步骤数量或调用深度推测。

设当前可用预算为 `C`，外层任务数为 `T`，且 worker 会同时启动并行子任务：

```python
outer_workers = min(T, ceil(sqrt(C)))
child_budget = max(1, floor(C / outer_workers))
```

子任务若继续产生真实并行，再递归应用同一规则。线程和进程 worker 都通过可序列化的预算上下文获得 `child_budget`。

以下场景不切分预算：

- Composite 筛选器依次执行多个筛选阶段；
- RFE、Sequential、Stepwise 的轮次有先后依赖，但当前轮候选并行；
- 报告先计算一类指标，再计算另一类指标；
- Excel 写入、图表渲染、状态合并等顺序步骤。

这些场景在当前可并行步骤开始时可使用完整预算。只有诸如“报告同时并行多个特征，而每个特征内部又同时并行多个分箱候选”才切分外层和内层预算。

某一层显式指定 worker 数时，该层配置优先。父层和子层都显式指定较大并发时视为调用者主动覆盖自动保护；其余自动层只使用上下文分配的预算。

## 6. 公共运行时结构

`hscredit/utils/parallel.py` 承担以下职责：

- 解析和验证 `n_jobs`；
- 探测物理核心数；
- 验证 `parallel_backend` 和 `parallel_config`；
- 计算普通及真实嵌套 worker 预算；
- 在线程和进程 worker 中传播剩余预算；
- 统一串行与 joblib 执行路径；
- 为失败任务补充中文任务上下文并保留原始异常链。

公共运行时提供内部 mixin，供估计器基类生成执行参数；函数式报告入口直接调用同一执行器。调用者提供稳定的任务标识、输入顺序、模块默认后端以及是否存在同时并行的子任务。

无论串行还是并行，执行器都返回按提交顺序排列的结果。它不会根据完成顺序合并状态，也不会捕获异常后静默跳过任务。

## 7. 确定性、精度与事务状态

串行和并行路径必须调用完全相同的单任务 worker。并行只改变调度，不维护第二套算法实现。

worker 不得直接并发修改估计器。拟合结果先写入临时状态容器；全部任务成功后，主线程按输入顺序一次性提交到 `splits_`、`bin_tables_`、`mapping_`、`scores_` 等公开拟合状态。任一任务失败时，不提交本轮部分状态；已拟合对象重新拟合失败时保留调用前的有效拟合状态。

转换 worker 只读取拟合状态，并按原始索引及列顺序返回结果。OneHot 等扩展列由主线程按照原特征顺序和已学习类别顺序拼接。

默认只沿相互独立的特征、规则、标签、数据集、候选或阈值拆分，不沿数据行拆分浮点归约。必须执行全局归约时，在主线程按固定顺序完成，避免改变浮点累加次序。

有 `random_state` 的算法按基础种子和原始任务序号派生稳定子种子，不能使用 worker 编号或完成顺序派生。`random_state=None` 保留原有非确定性语义。

离散状态、DataFrame、索引、列名、列顺序和报告结构要求串并行精确相等。浮点结果优先要求精确相等；仅第三方算法存在平台级浮点差异时使用明确、严格且逐字段记录的容差。

## 8. 数据传递与内存

线程后端共享 DataFrame，不复制整表。进程后端优先向单任务 worker 传入需要的列数组；可通过 `max_nbytes`、`mmap_mode` 和 `temp_folder` 使用 joblib 内存映射。

执行器按任务数和 worker 数使用 joblib 的批调度，不预先构造全部结果副本。`batch_size` 和 `pre_dispatch` 可由用户覆盖。并行化不得通过抽样、降精度、减少候选或缩短迭代换取速度。

## 9. 模块改造

### 9.1 分箱

`BaseBinning`、全部 18 种具体分箱器、`OptimalBinning` 和 `OptimalBinning2D` 统一公共参数。`fit` 和 `transform` 按特征或二维特征对并行。

当前 `_fit_features` 的线程 worker 会直接修改 `self`，需要改为单特征结果返回和主线程事务合并。`OptimalBinning` 向实际算法透传完整配置，并在预分箱、二次分箱和约束后处理真正同时并行时使用嵌套预算。

### 9.2 筛选

全部 22 种筛选器、`CompositeFeatureSelector` 和 `ScorecardFeatureSelection` 默认 `n_jobs=-1`。

- IV、Lift、PSI、VIF、缺失率、众数率、基数等按特征并行；
- RFE、Sequential 和 Stepwise 保留轮次依赖，只并行当前轮候选评分或拟合；
- Null Importance 并行独立随机实验；
- Boruta 并行影子特征或底层模型允许的独立评估；
- Stability 并行独立窗口或特征评估；
- 组合筛选阶段有依赖时保持顺序，每个阶段内部使用完整当前预算。

改造必须融合工作区已有的 selector 分箱生命周期变更，不覆盖、回退或重复实现它们。

### 9.3 编码

`BaseEncoder` 和 9 种公开编码器统一公共参数。映射学习及转换按列并行，主线程恢复原始列顺序。

Target、WOE 和 CatBoost 编码保持平滑、缺失值、未知值和防泄漏逻辑。GBMEncoder 可并行独立列模型，底层模型同时并行时参与真实嵌套预算。OneHotEncoder 的类别与扩展列顺序保持稳定。

### 9.4 规则与规则挖掘

- `Rule` 的多标签、多 DPD 和分组报告并行；单表达式预测保持串行；
- `RuleFlow` 并行独立规则预测，串联过滤顺序不变；
- `RuleSet` 和 `RulesClassifier` 并行规则评估，再按声明顺序进行逻辑聚合；
- Single/Multi/MultiLabel rule miner 按特征、组合和标签并行；
- Tree rule miner 按独立树、规则或数据集评估并行；
- 人工树节点构建保持依赖顺序，节点报告和独立规则评价并行。

### 9.5 报告

所有公开特征分析、规则分析、Swap、逾期预测、漂移、模型报告和模型比较入口统一公共配置。

- 特征报告按特征、目标或分箱方法并行；
- 规则报告按规则、标签或 DPD 并行；
- 模型报告按数据集和独立指标并行；
- `compare_models` 按模型并行；
- 漂移报告按特征和时间窗口并行；
- `OverduePredictor` 按标签/DPD 或可独立特征计算并行。

Excel、图表和同一输出文件的最终写入在主线程按固定顺序执行。并行阶段只生成 DataFrame、指标及绘图所需数据。函数的 `**kwargs` 入口显式提取并行参数，不能把它们误传给 Excel、绘图或底层模型。

## 10. 异常处理

并行参数错误使用中文 `ValidationError`。worker 失败时，主线程错误包含模块阶段和任务标识，例如特征名、规则名、标签名或报告章节，同时通过异常链保留原始异常。

执行失败不得静默重试为近似算法、吞掉异常、跳过失败项或提交部分拟合状态。显式后端与 `require="sharedmem"` 等配置冲突时在执行前报错。

## 11. 测试策略

### 11.1 公共运行时

- 模拟 1、2、8、16 个物理核心验证自动预算；
- 覆盖整数、整数值浮点数、比例、`None`、任务上限及非法输入；
- 验证只有真实同时并行才切分预算；
- 验证线程及进程 worker 获得正确子预算；
- 验证 joblib 参数透传、冲突检测、异常链和事务提交。

### 11.2 API 契约

枚举目标模块全部公开组件，检查统一参数和 `n_jobs=-1` 默认值。sklearn 组件检查 `get_params()`、`set_params()`、`clone()`、Pipeline 和 joblib 序列化，且构造函数不能修改调用者传入的配置字典。

### 11.3 串并行一致性

各组件分别以 `n_jobs=1`、`n_jobs=-1`、固定整数和比例运行。至少覆盖 `threading` 与 `loky`，并对 `multiprocessing` 做兼容性冒烟测试。

测试数值型、类别型、缺失值、未知类别、单列、多列、多标签和固定随机种子，比较切点、分箱表、编码映射、筛选结果、规则命中、模型预测、报告表、Excel sheet 名和关键单元格。

类别分箱一致性测试需要先对 `hscredit/core/metrics/_binning.py` 的 NumPy 2.x 排序键溢出做最小修复：排序键使用 Python `int`，不进行其他无关重构。

### 11.4 真实数据

使用 `examples/hscredit_yyp.xlsx` 完成串行与并行比较：

- 单特征 `衡枢鉴真分老客版`，标签 `FPD`；
- 多特征 `[衡枢鉴真分老客版, 近六个月非银多头机构数, 青云24]`；
- `overdue=['MOB1']`，`dpds=[7, 3, 0]`；
- 金额 `放款金额`；
- 日期 `放款时间`；
- 类别特征 `商品类别`。

### 11.5 性能

新增 `slow` 基准，覆盖宽表、大样本、多规则和多标签。每项先预热，再至少运行 3 次并比较中位数。

物理核心少于 4 时跳过速度门槛，但仍验证多个 worker 确实参与执行。4 核以上环境中，至少一个代表性 CPU 密集型宽表流程达到 1.2 倍加速；其他已并行化代表流程的并行中位数不得比串行慢超过 5%。

## 12. 完成标准

- 公共运行时、API 契约和串并行一致性测试全部通过；
- 分箱、筛选、编码、规则、规则挖掘和报告专项测试通过；
- 非 slow、非 integration 全量测试没有新增失败；
- 真实 Excel 数据的所有规定场景通过且串并行结果一致；
- 性能基准达到第 11.5 节门槛，或明确记录受限环境无法执行门槛的证据；
- 构建检查和 `git diff --check` 通过；
- 所有新增用户可见参数、错误和文档使用中文，并保持 Python 3.9+ 与 joblib 1.0+ 兼容。

## 13. 实施阶段

1. 建立公共并行运行时、参数验证、真实嵌套预算和契约测试；
2. 改造全部分箱器和筛选器，融合当前 selector 分箱生命周期变更；
3. 改造全部编码器、规则器和规则挖掘器；
4. 改造报告模块，执行完整一致性、真实数据和性能回归。
