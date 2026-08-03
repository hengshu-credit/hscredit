# Feature Summary 并行性能优化设计

## 目标

在不删减任何现有统计指标、不改变指标触发条件和返回列语义的前提下，优化
`feature_summary` 在大样本与超高维数据集上的执行效率，并让
`pd.DataFrame.summary`、`pd.Series.summary` 具有一致、完整的配置能力。

当前基线中，5000 个数值字段、50 行数据的基础统计约需 9.47 秒；100 个字段、
200 行数据的默认随机拆分 PSI 约需 1.22 秒，其中 PSI 约占总耗时的 70%。主要
原因是每个字段被重复扫描、多次独立计算分位数，以及 PSI 和趋势为每个字段反复
创建分箱器。

## 兼容性要求

- 保留基础统计、IV、KS、趋势、PSI 和模型重要性，不提供跳过指标的快速模式。
- 保持现有触发规则：传入 `y` 时计算 IV、KS、趋势；满足现有 PSI 条件时计算
  PSI；传入模型或 `model_type` 时计算模型重要性。
- 保持返回字段、字段顺序、百分比与小数位处理以及输入特征顺序。
- `y` 支持目标列名、`numpy.ndarray`、列表、元组及 `pd.Series`。外部数组型目标
  按位置与 `df` 对齐；长度不一致时抛出中文 `ValueError`。
- 缺失字段继续跳过，单字段指标计算失败时继续使用当前的 `NaN`/`unknown`
  降级语义，不中断其他字段。
- 不增加新的运行时依赖；项目现有依赖 `joblib>=1.0.0`。

## 公共接口

`feature_summary` 增加并行配置：

```python
def feature_summary(
    df: pd.DataFrame,
    features: List[str] = None,
    y: Optional[Union[str, Sequence, np.ndarray, pd.Series]] = None,
    val_df: Optional[pd.DataFrame] = None,
    models: Optional[Dict[str, Any]] = None,
    model_type: Optional[Literal[
        "xgboost", "lightgbm", "catboost", "randomforest"
    ]] = None,
    model_params: Optional[Dict] = None,
    max_n_bins: int = 5,
    psi_method: Literal["random_split", "group_col", "date_col"] = "random_split",
    psi_group_col: Optional[str] = None,
    psi_date_col: Optional[str] = None,
    psi_freq: str = "M",
    psi_test_size: float = 0.3,
    percentiles: List[float] = None,
    random_state: int = 42,
    numeric_as_categorical: Optional[List[str]] = None,
    force_numeric: Optional[List[str]] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    ...
```

`n_jobs` 语义：

- `-1`：自动模式。使用物理核心数的约 75%，始终至少保留一个物理核心，并受
  当前任务批次数限制。例如 16 个物理核心最多使用 12 个工作线程。
- `1`：完全串行，用于调试、结果对照或受限环境。
- 正整数：用户明确指定的工作线程数，但不创建超过实际任务批数的空闲工作线程。
- `0` 或小于 `-1`：抛出中文 `ValueError`。

`pd.DataFrame.summary` 与公共函数保持一致，补齐当前遗漏的
`numeric_as_categorical`、`force_numeric` 和新增的 `n_jobs` 参数，并原样转发。

注册 `pd.Series.summary`。Series 作为单字段 DataFrame 进入同一计算引擎；支持
数组、列表、元组和 Series 形式的 `y`。Series 没有可解析目标列的上下文，因此
其 `y` 不接受列名字符串。具名 Series 保留原名称，匿名 Series 沿用
`Series.to_frame()` 的列名 `0`。`return_type='dict'` 与 DataFrame 扩展一致。

## 实现架构

保留 `hscredit.core.eda.overview.feature_summary` 作为稳定公共入口，将高复杂度计算
提取到私有模块 `hscredit/core/eda/_feature_summary.py`。公共入口负责文档、签名和
转发；私有模块负责输入归一化、任务切分、统计计算和结果装配，避免继续扩大已经
较长的 `overview.py`。

私有计算引擎按阶段执行：

1. 输入归一化：过滤有效字段，按位置构造与 `df.index` 对齐的目标 Series，解析
   百分位数及强制类型配置。
2. 基础统计：把字段切成有限数量的批次，通过 joblib 线程共享原 DataFrame。
   数值字段在批内一次计算全部百分位数，并批量计算 count、min、max、mean、std、
   零值和负值；类别字段复用单次 `value_counts` 同时得到唯一值数、众数、众数频数
   和类别分位点。类型推断复用已经计算的唯一值数，避免额外全表扫描。
3. 预测指标：对数值字段按批并行。每个字段任务集中计算 IV、KS 和趋势，避免三轮
   独立任务调度，并保持 quantile IV 与 monotonic 趋势的现有算法。
4. PSI：先在主线程一次性生成训练/验证、分组或期间的行位置，再对字段按批并行。
   随机拆分只拆分行位置，不复制两份完整宽表；分组与日期方式不再为每个字段重复
   构造布尔条件和 DataFrame 子集。
5. 模型重要性：保持当前模型级调用。自动训练模型不与字段任务嵌套并行，避免模型
   自身并行与 joblib 工作线程产生 CPU 过度订阅。
6. 结果装配：joblib 按输入任务顺序返回批次，最终仍显式按原 `features` 顺序重排，
   再执行现有重要性、KS/IV/PSI/趋势列排序。

joblib 使用线程后端。Windows 下进程后端需要序列化或复制超宽 DataFrame，容易产生
高内存峰值；线程后端可以共享原始数据。任务不是“一字段一任务”，而是把字段划分为
有界批次，使调度任务数远少于字段数。批大小根据有效字段数、工作线程数和每个阶段的
工作量推断，并设置上下限，以兼顾 10 万字段的调度成本和高基数字段的负载均衡。

## 大数据与高维控制

- 不对整个 10 万字段表执行无必要的 `.copy()`；随机 PSI 只保存行索引。
- 数值统计按字段块处理，避免为了向量化一次创建与原表同量级的多个临时矩阵。
- 每个并行阶段结束后再进入下一阶段，禁止基础统计、预测指标、PSI 和模型训练嵌套
  开池。
- 自动工作线程数同时受物理 CPU 和任务批次数限制；小数据不会启动大量线程。
- 对全空、常数、高基数、数值、类别和日期字段分别保持现有输出语义。
- 并行异常限制在当前字段，聚合阶段不会因单个坏字段丢失其他结果。

## 测试与性能验收

功能测试使用真实函数，不模拟统计结果：

- `n_jobs=1` 与自动并行在混合类型数据上的完整结果逐列一致。
- `y` 分别使用列名、数组、列表、元组和带自定义索引的 Series，验证按位置对齐及
  长度错误。
- 验证 random split、验证集、分组列和日期列四类 PSI 路径。
- 验证 IV、KS、趋势、PSI 以及模型重要性列仍按原条件产生。
- 验证 `numeric_as_categorical`、`force_numeric` 和 `n_jobs` 从
  `DataFrame.summary` 透传。
- 验证具名及匿名 `Series.summary`、直接传入 `y` 和字典返回形式。
- 验证自动并行不使用全部物理核心，串行模式不进入 joblib 并行调度。

性能验证不设置依赖机器速度的脆弱 CI 秒数，而使用相同环境、固定随机种子和固定数据
的串并行中位数对比，并报告结果：

- 超宽场景：至少 50 行、10000 个数值字段，验证基础统计的重复扫描消除效果。
- 大样本场景：至少 100000 行、100 个混合字段，验证分块内存和统计吞吐。
- 完整指标场景：至少 1000 行、500 个数值字段并传入 `y`，验证 IV、KS、趋势和
  默认 PSI 的端到端加速。
- 在可用多核环境中，并行完整指标场景必须快于同一实现的 `n_jobs=1`；同时记录相对
  当前提交基线的耗时与加速比。若环境只提供一个可用核心，则只验证结果一致性和无
  额外明显回退。

## 非目标

- 不改变 IV、KS、趋势或 PSI 的数学定义、分箱方法和默认参数。
- 不引入近似统计、采样统计或自动跳过字段。
- 不并行训练多个用户模型，也不覆盖用户通过 `model_params` 指定的模型线程参数。
- 不改变其他 EDA API 或 pandas 扩展方法。
