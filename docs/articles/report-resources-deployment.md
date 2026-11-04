# 报告口径与资源部署

本次更新保持现有模型报告章节，完善标签、分箱、指标和资源接口。可执行演示见 `examples/33_report_resources_deployment.ipynb`。

## 标签与报告口径

`handle_missing_target` 独立控制原始缺失标签，支持 `drop`（默认剔除）、`positive`（归为1）、`negative`（归为0）、`raise`（报错）。`del_grey` 继续只控制业务灰样本；将缺失归为正或负样本不会再被剔灰开关覆盖。多标签分别保留有效样本分母。回归仅支持缺失剔除或报错。

报告中的订单数、坏率、金额、分箱、分群及图形按真实样本计算。模型训练使用的样本权重只改变模型学习。若需要额外查看加权性能，显式提供报告的 `sample_weight`，并设置 `show_weighted_metrics=True`；整体 KS、AUC 或回归误差会在实际指标右侧并列显示。

模型报告默认使用 `binning_reference="train"`。训练数据集为 `datasets` 的首项，或者 `X_train`；其他数据集只应用已拟合规则。`binning_reference="independent"` 用于独立分位分析。`rules` 与 `feature_bin_stats` 一致：列表作用于评分，字典作用于指定特征，`score`/`__score__` 为评分键。明确的规则优先于自动分箱。

如果模型已有名为 `score` 的输入特征，字典的 `score` 键只表示该特征；模型评分规则使用 `__score__` 或列表，避免混淆两种输入。

无效金额列、负金额、非数值、缺失和无穷金额直接报错。报告表不新增错误页；`mode="strict"` 拒绝失败报告，`best_effort` 静默保留成功内容。对象级调试状态仍可通过 `return_result=True` 读取。

数字字符串、日期和分类金额字段需由调用者先显式转换为数值，避免字符串拼接或类别求和被当作金额统计。

## 训练与评估接口

TargetEncoder 的 `fit_transform` 默认采用折外编码，随机种子默认42；`training_mode="in_sample"` 明确选择全量映射。小样本应设置合适的 `cv`。`fit(...).transform(...)` 继续使用最终完整映射，因此不等同于折外训练编码。

调参器整数 `cv` 加 `groups` 使用分组划分；自定义 splitter 仍须保证组间隔离，否则明确报错。WOEEncoder 的传统评分卡默认方式保持不变，可显式选择 OOF。

四种校准器支持拟合权重。ProbabilityCalibrator 内部分割时分别向基础模型训练集和校准集传入相应权重；`calib_ratio=None` 时基础模型不重训，未消费的额外基础训练参数会报错。

## 报告内存管理

`ModelReport(cache_policy="full")` 保留现有缓存行为。`cache_policy="none"` 在一次公开调用结束后释放派生缓存，同次导出内仍复用中间计算。冻结分箱规则保持不变。

- `release_cache()` 释放派生表与解释缓存，可以从现有数据重新计算。
- `release_data()` 先保存整体指标和摘要，再释放报告引用的逐行数据；此后可读 `get_metrics()`、`summary()`，生成明细或导出需重新构建报告。
- `resource_usage()` 返回可见数据的估算保留字节及行数，不包含外部模型原生内存，也不是操作系统 RSS。
- `last_execution_` 记录最近一次公开操作的名称、耗时和缓存策略，不自动追加报告章节。

调用者自己持有的 DataFrame、模型训练状态和自定义函数捕获的数据不会因释放报告数据而被删除。

## 流式汇总

数据库查询使用 `retain=False` 可避免 QueryStream 保留所有已读分块。`aggregate(accumulator, transform=...)` 逐块更新已有累加器并返回最终汇总；转换函数返回 tuple 时展开为 update 的位置参数。

默认 `require_complete=True`，查询被中断时抛异常，避免把已读部分当成完整总体。显式允许部分结果时，DataFrame 的 `attrs["数据源"]` 标明完整性。循环结束或异常均关闭底层资源。`release_results()` 可释放已有分块并切换到不保留模式；完整保留模式仍然可用。

查询流合并后的 DataFrame 保留 `hscredit_query_source` 完整性标记，分箱和编码后的 DataFrame 继续保留；明确未完成的数据源不能直接进入模型或标签报告。该标记与 WOE 等编码器的 `hscredit_source` 类型标记独立。

## 部署包

支持 ArtifactSerializableMixin 的组件可调用 `save_deployment(directory, validation_data=X)`。部署包包含模型、字段顺序与类型、类别映射、预测函数、前处理组件、依赖版本和验证记录，通过不可变版本目录与原子完成清单发布。

有 `save_inference` 的模型采用推理制品；其他组件保留原对象格式，并在清单中标记。提供验证数据时，会比较保存前后预测；不将验证样本明细写入包中。

推理导出移除已知训练历史与调参引用；自定义函数捕获的数据沿用现有 `save_inference` 契约，可能仍随对象保存。部署清单记录实际使用的后端依赖及前处理指纹，完整模型模式保留原对象能力。

`type(model).load_deployment(manifest_path)` 校验全部文件摘要、类型和元数据后加载。复制整个输出目录可以迁移包，文件路径按清单目录解析。`metadata_policy="strict"` 要求依赖元数据匹配；序列化对象仍须来自可信来源。旧 `save_artifact/load_artifact` 和原生格式继续保留。

## 导入与环境

设置环境变量 `HSCREDIT_LIGHT_IMPORT=1` 可启用按需公共 API 加载；基础指标、pandas 扩展、LR/随机森林训练推理和等频分箱无需启动绘图、解释或求解器。绘图和 OR 求解时才加载对应后端。默认导入保留既有 API 和样式设置；轻量模式需要全局样式设置时显式调用 `init_setting()`。此选项减少导入开销，不改变当前基础安装依赖集合。

`check_environment()` 或 `python -m hscredit check-environment --strict` 检查依赖声明及实际导入版本。部署/训练快照优先记录已经加载模块的实际版本，能识别旧安装元数据残留问题。CI 单独验证基础安装、Boost/Tune extras 和构建后的 wheel。

`scripts/benchmark_workflows.py` 在独立进程测量冷启动、指标、模型报告和流式聚合，输出耗时、进程树采样峰值 RSS、依赖版本和结果摘要；先检查两种导入模式结果一致，再比较耗时。性能阈值应在固定环境建立，采样结果不是内存硬限制。
