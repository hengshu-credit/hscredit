# models 模块设计复盘

## 结论

难用的主要原因是公共封装没有完整维持 sklearn 与原生框架的契约。单独一次数值数组训练容易通过，
但把同一个模型放进 Pipeline、调参、中文 DataFrame、保存恢复或重复训练流程后，配置和状态会发生变化。
因此本次保留已有模型与扩展功能，修复共同基础并补齐完整工作流。

## 模块职责与设计取舍

| 模块 | 当前职责 | 本次处理 |
| --- | --- | --- |
| `base.py` | 输入、评估、报告、保存、调参的公共入口 | 补齐参数管理、字段契约、稀疏输入和保存完整性 |
| `classical/` | sklearn 分类器与含统计能力的逻辑回归 | 保留原生参数/fit 能力，修复重复训练状态和 LR 字段对齐 |
| `boosting/` | 四个框架的参数和训练适配 | 处理版本差异、别名、回调、评估函数、类别类型和重要性顺序 |
| `tuning/` | 搜索空间、目标、Optuna Study、分析绘图 | 支持原生扩展入口并保存逐折训练内容 |
| `scorecard/`、`scorecard_support.py` | 概率评分、传统评分卡、刻度与漂移 | 保留全部能力，保存时保留评分转换器和相关模型状态 |
| `losses/` | 自定义损失与框架适配器 | 保留现有实现和公开名称，通过训练/保存契约提升可复用性 |
| `calibration/` | 概率校准与校准后模型 | 修正 sklearn 分类器标签继承顺序 |
| `explainability/` | SHAP、原因码、反事实与报告 | 保留原有接口，以回归测试验证影响 |
| `rules/` | 规则分类器 | 保留规则语义，修正 sklearn 分类器标签继承顺序 |

包装模型继续使用组合方式管理原生模型，逻辑回归保留继承原生 sklearn 类的方式。
强行把所有原生接口变成相同的内部实现会扩大兼容风险；统一的是入口、参数往返、状态和保存协议，
各框架的专属参数与回调协议仍由框架自身决定。

## 已复现的问题与修复

| 问题 | 对用户的影响 | 修复 |
| --- | --- | --- |
| `BaseEstimator` 位于 `ClassifierMixin` 前面 | sklearn 1.6 把 XGBoost、LightGBM、CatBoost、RandomForest 包装器识别成非分类器 | 调整模型、规则与校准类的继承顺序 |
| `**kwargs`、隐式 target 没进入 get/set 参数流程 | `clone` 丢 `max_bin`、`max_samples`、target；`set_params` 拒绝这些原生参数 | 完整保留构造输入、扩展参数和嵌套参数 |
| 构造和训练时改写参数 | 自动权重变成固定值、CatBoost 修改外部字典、旧 params 覆盖新搜索结果 | 分离构造配置和实际训练参数；复制外部字典；固定合并优先级 |
| 自定义 eval_metric 被当成列表 | XGBoost/LightGBM 报 `'function' object is not iterable` | 保留原生 callable 协议 |
| XGBoost 新旧 fit 参数位置不同 | early_stopping_rounds、callbacks、eval_metric 在不同版本报参数错误 | 1.6 以前走旧 fit API，之后走构造器 API |
| XGBoost KS 回调与早停回调无序 | 早停可能先读取尚未生成的 KS | KS 的生成与对应早停按固定顺序执行 |
| 验证集直接转数组，训练字段顺序没有复用 | 列顺序变化导致验证值错误；类别 dtype 丢失 | 验证集与预测共用字段校验；保留支持框架的 DataFrame |
| Boosting 重要性数组使用排序后的值 | sklearn 特征筛选把重要性配给错误字段 | 数组使用训练字段顺序，展示 Series 才排序 |
| importance_type 形参没有生效 | 用户要求 gain，实际拿到 split 等默认结果 | XGBoost/LightGBM 使用相应原生重要性接口 |
| sklearn 包装器忽略 fit_params | warm_start、monitor 等使用效果与原生不符，拼写错误被忽略 | 传递 fit 参数，并在 warm_start 时复用底层模型 |
| 训练失败仍可能保留旧的“已训练”状态 | 失败后预测拿到旧模型或不完整模型 | 训练状态明确失效，记录错误类型和信息 |
| LR 的目标列、重排字段及 WOE 状态处理不一致 | 显式 y 时标签泄漏；恢复后换列序报错；重复拟合复用旧 WOE 方向 | 与模型公共字段契约对齐，拟合前清理派生统计状态 |
| LR 评估捕获异常后静默继续 | 错误指标被忽略、显式请求少量指标也额外返回其他指标 | 复用统一评估实现，指标、权重和异常行为保持一致 |
| NGBoost 丢弃原生 evals_result | 训练完成后学习曲线丢失 | 保留框架实际输出的损失曲线 |
| JSON 保存过滤字典、函数和对象，LightGBM 检查错层级 | 恢复后配置/训练记录不完整，部分模型不能保存 | JSON 清单配套完整状态文件；使用框架原生导出入口 |
| 默认 pickle 引擎无法保存局部函数 | Notebook 自定义指标、损失或调参函数使保存失败 | 默认 cloudpickle；保持旧文件与显式引擎支持；单文件原子替换 |
| 直接加载原生模型未恢复实际字段名 | CatBoost 丢失字段数量，LightGBM/XGBoost 以人工字段名覆盖中文字段 | 从原生模型恢复字段，并替换此前模型的旧字段契约 |
| LightGBM 自定义损失原生加载后误处理概率 | 原始分值进入两列概率，数值可能超出 [0, 1] | 识别自定义目标的原始分值输出并保持原来的概率变换 |
| 调参只保留平均分，最终模型每取一次重新训练 | 无法复盘各折、重用最佳模型；中断后公开历史缺失 | 保存逐折模型和预测，缓存最佳模型，中断也更新已产生历史 |
| 整数 cv 与封闭的 Optuna 参数面 | GroupKFold、条件搜索、Study 回调、剪枝无法在 tune 使用 | 开放 Study、原生 Trial 函数、回调、剪枝器、分割器与 fit 参数工厂 |

## 调参与训练记录的边界

保留已有 `ModelTuner.fit()` 返回参数字典、`model.tune()` 返回最佳模型的行为。
`objective(y_true, y_probability)` 仍是评分目标；新增 `trial_objective(trial)` 表达原生 Optuna 目标，
避免通过函数签名猜测而破坏旧调用。

每折结果保存训练/验证位置、真实标签、预测、指标、曲线、可选模型；失败和剪枝结果也保留。
`artifact_dir` 写入逐折更新的试验制品，完整调参器可保存再加载续跑。
SQLite 自身只保存 Optuna 数据，不能替代模型与预测制品。

最佳 Boosting 模型默认利用各折最佳轮数，在完整输入上重训一次；之后复用缓存。
Pipeline 末端的模型同样保留训练记录并应用这一重训规则；未启用早停的模型保留原迭代数。
最终训练的参数和尝试模型保存在 `refit_params_` / `refit_model_`，重训失败也可检查原因。

## 兼容性验证与实际边界

本次在已有环境和独立虚拟环境验证，未更改用户原 Python 环境的依赖版本。

| 依赖 | 已有环境 | 隔离环境 |
| --- | --- | --- |
| Python | 3.11.5 | 3.11.5 |
| numpy / pandas | 1.23.5 / 2.2.3 | 1.23.5 / 2.2.3 |
| scikit-learn | 1.6.1 | 1.7.2 |
| XGBoost | 2.0.2 | 3.0.5 |
| LightGBM | 4.1.0 | 4.6.0 |
| CatBoost | 1.2.2 | 1.2.8 |
| NGBoost | 0.3.12 | 0.5.8 |
| Optuna | 4.9.0 | 4.9.0 |

验证包括模型回归、参数 clone/set、原生函数保存、条件搜索、剪枝、分组验证、Pipeline、
最佳模型重用，以及指定 `hscredit_yyp.xlsx` 的真实数据工作流。
旧版本分支还使用现有的兼容契约测试覆盖；没有声称实测了所有历史版本组合。

2026-09-16 最终验证结果：

- 当前环境的模型、特征筛选衔接和模型报告回归：**756 通过，1 项单独排除**。
- 第二套依赖组合的训练、保存、调参、评分卡和特征筛选测试：**212 通过**。
- 新增使用场景回归覆盖 **48 项**；既有公开方法无删除。
- 指定真实数据 **970 条**，其中训练 776 条、测试 194 条：十类分类器、类别字段、模型/评分保存、
  报告与 Optuna 续跑通过；另验证了概率校准、规则分类器和概率评分卡的保存往返。
- 关键静态错误检查、Git diff 空白检查、文档构建和 CI YAML 解析通过。
- 第二套固定依赖组合通过 pip 安装解析的 dry run；本机环境未实际升级。

原模型测试基线存在两个环境问题：未安装开发包时，独立示例进程找不到 hscredit；
本机 PyTorch 的 `c10.dll` 加载失败。前者可通过开发安装或显式 PYTHONPATH 解决，
后者不在这次树模型与调参改动中伪装为通过。

已新增固定依赖组合的 CI 工作流，覆盖参数、原生加载、保存、调参、评分卡和特征筛选的衔接。
该工作流已经写入仓库；远程 CI 尚未运行，本地实测结果单独列出。

GPU、分布式训练及任意 Python 制品的跨版本迁移未在本次本机验证。
当前包装器维持项目既有二分类、坏标签为 1 的边界；这不等同于覆盖原生框架的多分类、回归和排序模型。
模型输出的概率和评分语义、现有损失/规则/解释/报告入口都保留。

## 参考依据

- [sklearn 估计器开发契约](https://scikit-learn.org/stable/developers/develop.html)：混入类顺序、构造参数和 clone 的要求。
- [XGBoost 2.1 变更说明](https://xgboost.readthedocs.io/en/stable/changes/v2.1.0.html)：fit 中移除 eval_metric、early_stopping_rounds 和 callbacks。
- [LightGBM early_stopping](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html)：回调和指标方向。
- [Optuna Study 接口](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)：回调、catch、剪枝和 Study 复用。

具体用法见 [模型工作流指南](model-workflow.md)，可运行示例为 `examples/28_model_workflow.py`。
