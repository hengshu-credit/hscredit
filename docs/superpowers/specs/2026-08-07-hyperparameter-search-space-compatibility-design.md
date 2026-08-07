# HSCredit 多框架超参数搜索空间兼容设计

## 1. 背景

HSCredit 的模型调优统一由 Optuna 执行，并已提供一组模仿 Optuna、scikit-optimize
（skopt）和 Hyperopt 的搜索空间声明对象。当前相关测试可以通过，但实现只覆盖了
自定义样例，没有完整遵循各框架的常用真实签名和取值语义。

已确认的缺陷包括：

- `Real`、`Integer`、`Categorical` 缺少 skopt 的 `base`、`transform`、`dtype` 等
  参数位置，且对象的 `name` 没有参与参数名解析；
- bayesian-optimization 的 `(low, high, int)` 显式整数声明和字符串类别元组无法解析；
- Hyperopt `randint(label, low, upper)` 的位置参数顺序不兼容；
- Hyperopt `loguniform`、`qloguniform` 的 `low`、`high` 本应表示对数域边界，当前却被
  当成最终值边界；
- `qloguniform` 没有真正执行量化；
- `lognormal` 使用最终值边界计算对数空间 CDF，导致采样越界；
- 正态族借助 `参数名__u` 采样后，单目标 `best_params_`、trial 重评估和预设搜索点仍
  直接使用 Optuna 潜变量，无法构造真实模型；
- 同名声明只在 `hscredit.core.models.tuning.search_space` 暴露，不能按预期从
  `hscredit` 顶层统一导入。

## 2. 目标

本次改造必须实现：

1. 用户不安装 Optuna 之外的搜索框架，只从 `hscredit` 导入与原框架同名的构造器，
   即可复用常见搜索空间声明代码；
2. 保留 Optuna、GridSearch、skopt、bayesian-optimization 和 Hyperopt 的常用声明
   格式、参数名称、边界、类别、对数、步长、量化和主要采样语义；
3. 所有搜索仍统一由 HSCredit 现有 Optuna 后端执行，不引入或切换到其他搜索引擎；
4. Optuna 无法原生表达的分布通过潜变量和确定性变换近似，模型、公共结果和重评估 API
   只看到真实参数名与变换后的有效值；
5. 修复现有兼容代码中的已知错误，并用真实调用形式而非内部实现细节建立回归测试；
6. 在 `examples/04_models.ipynb` 中分别演示五种框架风格以及混合风格，并实际执行整个
   笔记本确认输出无异常；
7. 接受各框架常见的手工搜索点格式，将模型最终参数值统一转换后通过 Optuna
   `study.enqueue_trial` 入队；
8. 不新增任何第三方依赖，不改动用户当前工作区中与本任务无关的文件。

## 3. 非目标与兼容边界

- GridSearch 的列表仅表示完整候选集合；默认 TPE 后端不承诺像 `GridSearchCV` 一样穷举
  所有组合或保持遍历顺序。若用户显式选择 Optuna GridSampler，仍由现有 sampler 接口
  控制搜索算法。
- skopt 的 `transform` 是其优化器内部编码方式，在 Optuna 后端中只做合法性校验；
  `base` 不改变给定最终值边界上的对数均匀分布；`dtype` 用于最终模型参数转换。
- skopt 类别 `prior` 和 Hyperopt 非原生分布通过一维潜变量近似。该近似保持候选值、
  权重或目标分布的基本语义，但不承诺与原框架优化器产生相同的 trial 序列。
- Hyperopt 的嵌套条件 pyll 图、`pchoice`、任意确定性表达式和自定义分布不在本次范围；
  仅覆盖用户明确列出的 `uniform`、`choice`、`randint`、`quniform`、`loguniform`、
  `qloguniform`、`normal`、`qnormal`、`lognormal`、`qlognormal`。
- 不安装 `skopt`、`hyperopt` 或 `bayesian-optimization` 来做运行时类型判断；兼容对象
  全部由 HSCredit 自身实现。

## 4. 公共 API

### 4.1 导入入口

以下符号同时从 `hscredit`、`hscredit.core.models.tuning` 和现有
`hscredit.core.models.tuning.search_space` 路径公开：

- `Dimension`、`Real`、`Integer`、`Categorical`；
- `IntDistribution`、`FloatDistribution`、`CategoricalDistribution`；
- `suggest_int`、`suggest_float`、`suggest_categorical`、
  `suggest_discrete_uniform`、`suggest_loguniform`；
- `uniform`、`choice`、`randint`、`quniform`、`loguniform`、`qloguniform`、
  `normal`、`qnormal`、`lognormal`、`qlognormal`；
- `normalize_search_space`。

顶层导出继续使用现有懒加载机制，不让普通 `import hscredit` 提前导入可选重依赖。

### 4.2 搜索空间容器

`normalize_search_space`、`ModelTuner` 和模型 `tune()` 接受两类容器：

1. 映射：参数名到声明对象、列表或元组，覆盖 GridSearch、bayesian-optimization 及
   各框架常见字典写法；
2. 具名维度序列：每一项必须是带非空 `name` 的 HSCredit `Dimension`，覆盖 skopt
   `dimensions=[Real(..., name='C'), ...]` 和同名 Optuna/Hyperopt 声明的复用场景。

映射值含内部名称时，该名称必须与映射键一致。名称不一致、序列项无名称或存在重复名称
时抛出中文 `ValueError`，不得静默覆盖。

### 4.3 手工搜索点入口

`ModelTuner` 保留现有 `trial_points=dict/list[dict]` 和 `enqueue_trials(...)`，并新增
以下兼容入口：

```python
# Optuna
tuner.enqueue_trial(
    {"C": 1.0},
    user_attrs={"来源": "经验值"},
    skip_if_exists=True,
)

# GridSearch：按 ParameterGrid 展开后逐点入队
tuner.enqueue_trials(param_grid={"C": [0.1, 1.0], "solver": ["liblinear"]})

# skopt：按搜索空间维度顺序传入 x0
tuner.enqueue_trials(x0=[[0.1, "liblinear"], [1.0, "lbfgs"]])

# bayesian-optimization
tuner.probe(params={"C": 1.0, "max_iter": 200}, lazy=True)
tuner.probe(params=[1.0, 200], lazy=True)

# Hyperopt
tuner = ModelTuner(
    ...,
    points_to_evaluate=[
        {"C": 0.1, "solver": "liblinear"},
        {"C": 1.0, "solver": "lbfgs"},
    ],
)
```

Optuna 同名方法签名为
`enqueue_trial(params, user_attrs=None, skip_if_exists=False)`。`probe` 接受字典或按搜索空间
顺序排列的序列；`lazy` 参数保持 bayesian-optimization 的调用兼容，但统一 Optuna 后端
下无论取值均表示加入待评估队列。`x0` 支持单点一维序列或多点二维序列，并利用搜索空间
维度数量消除二者歧义。

`trial_points`、`points_to_evaluate` 和 Optuna 字典点允许只指定部分参数，其余参数在 trial
中正常采样。GridSearch 展开点、skopt `x0` 和 bayesian-optimization 序列点必须提供完整
维度。所有格式中的值均表示模型最终收到的参数值，不要求用户了解内部潜变量。

## 5. 同名构造器语义

### 5.1 Optuna

`suggest_int`、`suggest_float`、`suggest_categorical` 保持 Optuna 的名称和常用参数形式。
`suggest_int` 的 `step`、`log` 以及 `suggest_float` 的 `step`、`log` 遵循互斥约束。
兼容的 `IntDistribution` 和 `FloatDistribution` 调整为 Optuna 当前构造顺序：
`(low, high, log=False, step=...)`。旧的弃用别名继续可用，但新示例优先使用
`suggest_float(step=...)` 和 `suggest_float(log=True)`。

### 5.2 GridSearch

`dict(参数名=[候选值...])` 归一化为类别候选。空候选拒绝；元组不作为 GridSearch
候选容器，以避免与 bayesian-optimization 边界元组产生不可判定歧义。

### 5.3 skopt

兼容常用构造签名：

```python
Real(low, high, prior="uniform", base=10, transform=None, name=None, dtype=float)
Integer(low, high, prior="uniform", base=10, transform=None, name=None, dtype=np.int64)
Categorical(categories, prior=None, transform=None, name=None)
```

`Real` 和 `Integer` 支持 `uniform`、`log-uniform`；对数先验要求正数边界。
`Categorical.prior` 为 `None` 时等权，有值时必须与类别数量一致、每项非负且总和大于零，
归一化后用于潜变量区间划分。`transform` 只接受对应 skopt 的常用合法值。最终值按
`dtype` 转换后传给模型。

### 5.4 bayesian-optimization

字典值按以下规则解析：

- 两个数值的元组 `(low, high)`：两端均为非布尔整数时为整数范围，否则为浮点范围；
- `(low, high, int)`：显式整数范围；
- `(low, high, float)`：显式浮点范围；
- 全部为字符串的元组：类别候选；
- 三元素字符串 prior 元组继续兼容现有 skopt 简写 `(low, high, 'uniform')` 或
  `(low, high, 'log-uniform')`。

类型、长度或范围无法判定时抛出中文错误，不根据模型参数名称猜测。

### 5.5 Hyperopt

函数位置参数与 Hyperopt 常用形式一致：首个参数为 `label`。`randint` 同时支持
`randint(label, upper)` 和 `randint(label, low, upper)`，最终整数区间为
`[low, upper)`。

分布转换如下：

- `uniform`：直接映射最终值域上的均匀浮点；
- `quniform`：最终值为 `round(uniform(low, high) / q) * q`，并限制在量化后的有效
  边界内；
- `loguniform`：潜变量在 `[low, high]` 均匀采样，最终值为 `exp(latent)`；
- `qloguniform`：最终值为 `round(exp(latent) / q) * q`；
- `normal`、`qnormal`：通过 `[epsilon, 1-epsilon]` 上的均匀潜变量和 SciPy 正态分布
  逆 CDF 得到近似无界正态，再按需量化；
- `lognormal`、`qlognormal`：先按正态族生成对数空间值，再取指数并按需量化；
- `choice`：无嵌套声明时映射为类别候选。

`epsilon` 使用模块内固定数值常量，避免逆 CDF 产生无穷值。该常量不作为新的公共配置。

## 6. 内部适配架构

### 6.1 公开声明对象

`search_space.py` 只负责同名类型、构造签名、局部参数校验和无 Optuna 依赖的声明数据。
每个声明对象保留 `name`、来源框架、最终值类型和构造参数，并通过稳定内部方法导出原始
规格；它不负责创建 trial 或解释搜索空间容器。

### 6.2 归一化与采样适配器

新增内部适配模块，职责包括：

1. 将映射或具名序列解析为有序参数映射；
2. 校验并解析内部名称；
3. 将列表、元组、声明对象、SciPy 冻结分布和真实 Optuna Distribution 统一为内部规格；
4. 判断参数能否直接使用原名称交给 `trial.suggest_*`；
5. 为变换分布生成不会与用户参数冲突的内部潜变量名；
6. 在采样、最佳 trial、指定 trial、历史记录和预设点之间执行可逆转换；
7. 将 NumPy 标量转换为模型和 Optuna 存储可接受的 Python 标量。

内部潜变量名称使用保留前缀，并在用户参数名与保留前缀冲突时立即报错。公共 API 不把
潜变量名称作为模型参数返回。

### 6.3 ModelTuner 数据流

`ModelTuner` 初始化时归一化一次搜索空间。每个 trial 的数据流为：

```text
公开搜索空间 -> 内部规格 -> Optuna 原生参数或潜变量
             -> 确定性变换 -> 模型有效参数 -> 模型评估
```

trial 完成后，最佳参数和历史展示由同一个规格将 `trial.params` 还原为模型有效参数。
单目标与多目标走同一还原函数。`evaluate_study_trials` 不再直接把原始 `trial.params`
传给模型。

`trial_points` 在入队前执行反向转换：直接参数保持原名，变换参数转换为潜变量。不可逆的
量化值选择对应区间中点，确保重新采样能还原为用户指定的有效值；超出范围或不属于类别
候选的值以中文错误拒绝。

### 6.4 手工点归一化与入队

适配器用一个内部入队记录统一保存：

- 模型最终参数字典；
- 可选 `user_attrs`；
- `skip_if_exists`；
- 输入来源，仅用于中文错误上下文和调试，不改变搜索结果。

各入口先转换为有序的模型最终参数字典：

- Optuna、现有 `trial_points` 和 Hyperopt `points_to_evaluate` 直接读取字典；
- GridSearch `param_grid` 使用项目已有的 sklearn `ParameterGrid` 完整展开笛卡尔积，
  不限制或抽样候选数量；
- skopt `x0` 和 bayesian-optimization 序列按归一化搜索空间的稳定顺序映射参数名；
- bayesian-optimization 字典与 Optuna 字典走相同名称路径。

入队前，适配器验证参数名、候选、边界、类型和量化结果，再将模型最终值反向转换为 Optuna
实际采样名称和值。直接分布保持原参数名；对数、正态、量化和带权类别分布转换为对应潜
变量。量化值的反向映射选择能稳定还原该有效值的区间中点，避免浮点边界误差。

如果 study 尚未创建，入队记录保存在 tuner 中；每次 `fit` 新建或加载 study 后，记录均
逐项调用 `study.enqueue_trial`。如果 study 已存在，新增记录也立即提交给当前 study，且
仍保留为 tuner 配置，保证后续重新创建 study 时不会丢失。`skip_if_exists` 原样传给
Optuna；未显式指定时保持 Optuna 的 `False` 默认值。

`n_trials` 继续采用 Optuna 语义，包含本次实际执行的已入队 trial。如果队列数量大于
`n_trials`，剩余点保留为待评估状态，不自动扩大用户请求的 trial 数。

用户仍可直接访问 `tuner.study_`，但对包含潜变量的搜索空间，应使用 tuner 的兼容入口；
直接调用 `tuner.study_.enqueue_trial` 不执行模型最终值到潜变量的反向转换。

### 6.5 LightGBM 特殊约束

现有 `num_leaves <= 2 ** max_depth` 动态约束继续保留，但通过适配器提供的直接采样入口
执行。若用户为这两个参数声明变换分布而无法安全收紧动态上界，则在初始化时给出明确
中文错误，不允许潜变量和动态约束产生不一致结果。

## 7. 错误处理

下列情况必须在开始优化前失败：

- 空候选、重复名称、名称不一致、无名称序列项；
- `low > high`、非正 `q` 或 `sigma`；
- 对数分布最终值边界不合法，或 Optuna `log=True` 与非默认 step 同时出现；
- skopt 非法 prior、transform、dtype 或类别 prior；
- bayesian-optimization 元组类型无法识别；
- Hyperopt `randint` 空区间、choice 含不支持的嵌套维度；
- 用户参数使用内部保留前缀；
- 手工点参数名未知、值超出范围、值不符合量化结果或类别不存在；
- skopt/bayesian-optimization 序列点维数不等于搜索空间维数；
- GridSearch `param_grid` 的值不是合法候选序列；
- 预设搜索点无法稳定反向映射到搜索空间。

所有新增用户可见错误使用中文 `ValueError`，并包含参数名与失败原因。

## 8. 测试策略

### 8.1 TDD 回归测试

先添加并观察以下行为在当前实现上失败：

- `from hscredit import ...` 可导入全部同名入口；
- Optuna、skopt、Hyperopt 构造器接受真实常用位置参数和关键字参数；
- skopt 具名维度序列可直接归一化，名称冲突和重复名称被拒绝；
- bayesian-optimization 显式 `int`/`float` 元组和字符串类别元组正确转换；
- Hyperopt `randint(label, low, upper)`、对数域、量化和正态族产生正确类型与范围；
- `qloguniform`、`qnormal`、`qlognormal` 的结果是 `q` 的整数倍；
- lognormal 中位潜变量还原到 `exp(mu)`，不会越出实现自己的有效范围；
- 变换分布完成调优后，`best_params_`、`get_best_model`、单目标和多目标结果中不存在
  潜变量名；
- `evaluate_study_trials` 使用还原后的模型参数；
- `trial_points` 对直接、对数、量化、正态和带权类别参数完成往返。
- Optuna `enqueue_trial` 保留 `user_attrs` 和 `skip_if_exists` 并调用真实 study；
- GridSearch `param_grid` 按确定顺序展开所有组合并逐点入队；
- skopt 单点/多点 `x0` 按维度顺序转换，错误维数被拒绝；
- bayesian-optimization `probe` 的字典和序列形式得到相同最终点；
- Hyperopt `points_to_evaluate` 支持部分参数点；
- 入队的变换分布点在执行时还原为用户提交的模型最终值；
- study 创建前后追加点均不会在下一次 `fit` 时丢失；
- `n_trials` 小于待评估队列长度时不被适配器静默扩大。

测试断言模型实际收到的参数和值，不把内部规格字典文本作为主要验收目标。

### 8.2 端到端搜索

使用 sklearn 小型分类数据和轻量估计器，为 Optuna、GridSearch、skopt、
bayesian-optimization、Hyperopt 各建立至少一个真实 `ModelTuner.fit` 用例。每个用例
运行少量 trial，验证搜索完成、最佳参数名称正确、值符合声明且最佳模型可构造。

混合格式用例覆盖同一搜索空间同时包含多个来源声明，并验证归一化顺序稳定。

### 8.3 笔记本

`examples/04_models.ipynb` 的超参数搜索空间章节改为从 `hscredit` 顶层导入，增加：

- Optuna `suggest_*`；
- GridSearch 列表字典；
- skopt 具名 `Dimension` 序列和字典；
- bayesian-optimization 连续、显式整数与类别元组；
- Hyperopt 十种分布的声明与代表性实际调优；
- 五种框架各自的手工搜索点写法，并展示它们进入同一个 Optuna study；
- 多框架混合搜索空间。

示例控制 trial 数和数据规模，避免重复演示显著增加运行时间。修改后使用现有开发依赖
执行整个笔记本并保留输出；不得只执行新增单元格。

## 9. 完成标准

- 每项生产行为均有先失败后通过的回归测试；
- 搜索空间和调优定向测试全部通过；
- 五类手工点入口都经过真实 `study.enqueue_trial` 并优先得到评估；
- 非 slow、非 integration 测试无新增失败；
- `examples/04_models.ipynb` 从头到尾执行成功；
- `git diff --check`、Black、flake8 和相关 mypy 检查无本次新增问题；
- `pyproject.toml` 与其他依赖文件没有新增依赖；
- `git status --short` 中原有无关修改保持内容不变。
