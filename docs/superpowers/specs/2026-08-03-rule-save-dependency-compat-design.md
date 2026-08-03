# Rule.save 与依赖版本兼容层设计

## 背景

当前 `Rule.save` 仍按已经废弃的 ExcelWriter 接口实现：相对导入会解析到不存在的
`hscredit.core.report`，路径参数被当作样式模板传入，随后调用的 `add_dataframe()` 和
`close()` 也不属于当前 `ExcelWriter` API。

当前环境还存在两组已经复现的依赖组合问题：

- LightGBM 3.3.5 导入可选 Dask 2022.7.0 时，Dask 引用 Pandas 2.0 已移除的
  `pandas.core.strings.StringMethods`，导致 LightGBM 在业务模型代码执行前导入失败。
- Seaborn 0.11.2 调用 `mode.use_inf_as_null`，而 Pandas 2.0 已删除该配置项，导致
  `hist_plot` 失败。

项目目标不是收紧依赖版本，而是在尽可能宽松的依赖范围内，通过明确、可测试的版本矩阵
选择兼容适配器，并保持 hscredit 对外接口及基本功能一致。

## 设计原则

1. 兼容路径必须由安装包的明确版本号决定，不以 API 探测、异常捕获、错误消息解析或
   “先调用、失败后重试”决定。
2. 版本读取统一使用 `importlib.metadata` 与 `packaging.version.Version`；预发布版本遵循
   PEP 440 比较语义。
3. 适配器只处理已经确认的版本组合，每个适配器必须声明最小版本、最大版本和移除条件。
4. 对外继续暴露 `LightGBMRiskModel`、`GBMEncoder`、selector 和绘图函数等现有接口，
   不把第三方版本差异泄漏给调用方。
5. 兼容适配不得把导入或运行异常转换为 `None`、`AttributeError` 或静默降级。缺少可选依赖
   可以保留明确的 `ImportError`/`DependencyError`，其他异常保留原始异常链。
6. Pandas 的 MultiIndex、多层级列和 Index 语义是必须保持的能力；不引入 Polars，也不把
   Polars 列为后续数据后端。

## 方案选择

采用集中式内部兼容层，版本矩阵与兼容动作放在 `hscredit/_compat.py`。调用点只表达“准备
某依赖”，不自行判断版本，也不维护本地 fallback。

不采用以下方案：

- 各模块分别通过 `try/except TypeError` 重试不同参数：行为依赖实际失败顺序，难以证明
  覆盖范围，且会隐藏第三方库内部真实错误。
- 收紧 `pyproject.toml` 的上下界：会直接违背宽依赖范围目标。
- 通过 `inspect.signature()` 判断兼容分支：即使比异常重试稳定，仍不是明确版本契约，
  无法审计每个版本组合的预期行为。

## Rule.save 契约

`Rule.save(report, excel_writer, sheet_name=None, excel_params=None)` 保持方法名和返回类型不变：

- `excel_writer` 支持 `str`、`os.PathLike` 和现有 `ExcelWriter`。
- 路径输入时创建 `ExcelWriter`，使用 `dataframe2excel()` 写入，随后保存到目标路径。
- 对象输入时写入同一实例，不自动保存或关闭，便于继续追加其他工作表。
- `excel_params` 作为 `dataframe2excel()` 参数透传；与显式参数冲突时，显式的
  `sheet_name` 和 `excel_writer` 优先。
- 返回实际使用的 `ExcelWriter` 实例。
- DataFrame 的单层列、多层级列、普通 Index 和 MultiIndex 均由现有 Excel 写入核心处理，
  `Rule.save` 不展平列名或重排索引。
- 非支持类型抛出中文 `TypeError`。

## 版本矩阵

### LightGBM 导入 Dask 的 Pandas 兼容

启用条件全部满足时才应用：

- `3.2.0 <= lightgbm < 4.7.0`；
- `pandas >= 2.0.0`；
- 已安装 `dask < 2023.2.0`。

适配动作是在 LightGBM 首次导入前，将 Pandas 2.x 中真实的
`pandas.core.strings.accessor.StringMethods` 暴露到旧 Dask 使用的
`pandas.core.strings.StringMethods` 路径。Dask 2023.2.0 已改为使用 `pd.Series.str`，
因此不再应用。LightGBM 主线已移除基础兼容模块中的 Dask 强制导入，`4.7.0` 作为该适配器
的移除边界；在正式版本变化时通过矩阵测试调整，不猜测未来接口。

### LightGBM 与 scikit-learn 参数名兼容

启用条件：

- `lightgbm < 4.6.0`；
- `scikit-learn >= 1.8.0`。

适配动作将 LightGBM 内部对 `_LGBMCheckXY`、`_LGBMCheckArray` 的
`force_all_finite` 关键字转译为 `ensure_all_finite`。scikit-learn 1.6 开始重命名但仍接受
旧名，1.8 才移除旧名；LightGBM 4.6.0 已引入自己的 `validate_data` 兼容实现，因此只在
上述交叉区间启用。

### LightGBM fit 与早停

项目声明的 LightGBM 最低版本为 3.2.0：

- `3.2.0 <= lightgbm < 4.0.0`：使用 `early_stopping_rounds` 和 `verbose` 参数；
- `lightgbm >= 4.0.0`：使用 `callbacks`、`early_stopping()` 和 `log_evaluation()`；
- 不再捕获 `TypeError` 后删除参数重试。

无验证集时同样按版本决定是否向 `fit()` 传入 `verbose`。

### Seaborn 与 Pandas 无穷值配置

启用条件：

- `0.11.0 <= seaborn < 0.12.2`；
- `pandas >= 2.0.0`。

适配动作在 Seaborn 首次加载前注册兼容配置名 `mode.use_inf_as_null`。为了保持旧配置的业务
语义，hscredit 传给分布图的数值数据会先把正负无穷转换为缺失值；兼容配置只负责满足旧版
Seaborn 的上下文访问，不依赖配置回调改变 Pandas 全局行为。Seaborn 0.12.2 已切换为
`mode.use_inf_as_na`，不再启用该适配。

## 延迟导入与异常语义

- `hscredit` 初始化兼容注册时只读取已安装包元数据并执行命中的轻量适配，不导入
  LightGBM、Seaborn、Dask 等重依赖。
- `LazyModule` 在真正加载 Seaborn 前再次按同一幂等矩阵准备兼容环境。
- boosting、tuning 和 `core.models` 的模块级 `__getattr__` 删除宽泛
  `except Exception` 与 `getattr(..., None)`；成功后才缓存真实对象。
- 未安装依赖与不在已知矩阵内的第三方错误不触发兼容 fallback，不被伪装为属性不存在。

## 测试策略

严格按 TDD 增加以下测试：

1. `Rule.save` 路径输入写出可重新打开的 xlsx，并返回 `ExcelWriter`。
2. `Rule.save` 对象输入复用同一 writer、不提前落盘，且支持多层级列与 MultiIndex。
3. `excel_params` 正确透传，非法 writer 类型给出中文错误。
4. 版本矩阵测试直接传入版本字符串，验证边界前后是否命中，不依赖当前环境碰巧安装的版本。
5. 当前 `LightGBM 3.3.5 + Dask 2022.7.0 + Pandas 2.3.3` 可以导入并完成
   `LightGBMRiskModel`、GBMEncoder 和 selector 的基础训练。
6. 当前 `Seaborn 0.11.2 + Pandas 2.3.3` 下 `hist_plot` 成功，并验证正负无穷按缺失值处理。
7. 模拟新版本分支时不安装旧适配、不修改第三方对象。
8. 懒加载失败时保留真实异常，不缓存 `None`。

验证顺序为定向单测、当前 8 个失败用例、相关模块测试、去除慢速/集成标记的完整测试。

## 非目标

- 本轮不全面枚举所有 XGBoost、CatBoost、statsmodels 等历史版本；先建立统一矩阵和注册机制，
  后续每发现一个明确版本断点就新增一条有上下界、有测试、有移除条件的规则。
- 不保证用户在导入 hscredit 之前直接导入一个自身已经损坏的第三方组合也能成功；保证的是
  hscredit 公开入口及经 hscredit 准备后的原生模型集成路径。
- 不修改 Pandas 多层级列、MultiIndex 和行序语义，不引入 Polars。

## 上游依据

- Pandas 2.0 删除 `mode.use_inf_as_null`：
  https://pandas.pydata.org/pandas-docs/version/2.0/whatsnew/v2.0.0.html
- Dask 2023.1.1 仍使用 `pd.core.strings.StringMethods`，2023.2.0 改为 `pd.Series.str`：
  https://github.com/dask/dask/blob/2023.1.1/dask/dataframe/accessor.py
  https://github.com/dask/dask/blob/2023.2.0/dask/dataframe/accessor.py
- Seaborn 0.12.1 仍使用旧配置名，0.12.2 切换到 `mode.use_inf_as_na`：
  https://github.com/mwaskom/seaborn/blob/v0.12.1/seaborn/_oldcore.py
  https://github.com/mwaskom/seaborn/blob/v0.12.2/seaborn/_oldcore.py
- scikit-learn 1.6 声明旧参数在 1.8 移除：
  https://scikit-learn.org/stable/whats_new/v1.6.html
- LightGBM 4.6.0 的兼容实现：
  https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/python-package/lightgbm/compat.py
