# 模型训练、调参与保存

## 1. 最短训练路径

```python
import pandas as pd
from hscredit import LightGBM

features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
df = pd.read_excel("examples/hscredit_yyp.xlsx")
model = LightGBM(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=4,
    target="FPD",
    eval_metric=["auc", "ks"],
    early_stopping_rounds=20,
)
model.fit(df[features + ["FPD"]])
probability = model.predict_proba(df[features])[:, 1]
score = model.predict_score(df[features])
```

这段代码演示接口，效果评估应另留测试集。标签必须同时包含 `0` 和 `1`，`1` 表示坏样本。
也可以使用 `fit(X, y, sample_weight=weights)`；显式 `y` 优先，并从特征中移除 `target` 列。
DataFrame 预测和验证集会按训练字段重新排序；缺字段、重名列、标签或权重长度不符会直接报中文错误。

早停需要验证数据。包装模型未收到 `eval_set` 时，会按 `validation_fraction` 从训练输入划出验证集；
不启用早停时使用全部输入。训练权重、XGBoost `base_margin`、LightGBM `init_score`、
CatBoost `baseline` 会跟随自动划分同步切分。

类别字段保留原来的 DataFrame 类型：LightGBM 可使用 pandas `category`，XGBoost 使用其原生
`enable_categorical=True` 配置；CatBoost 可传 `cat_features=["商品类别"]`。
包装模型以 DataFrame、二维数组或框架支持的稀疏矩阵为输入；框架专属容器需按该框架支持情况使用。

## 2. 参数如何复用

```python
from sklearn.base import clone
from xgboost import XGBClassifier
from hscredit import XGBoost

native_params = XGBClassifier(
    n_estimators=50, max_depth=3, max_bin=64, n_jobs=2, random_state=42,
).get_params()
source = XGBoost(**native_params, target="FPD")
model = clone(source)                   # 保留 target、原生扩展参数和自定义配置
model.set_params(max_bin=128)           # 可用于 Pipeline / GridSearchCV
model.fit(df[features], df["FPD"])
actual_params = model.get_native_params()
native_model = model.get_native_model()
```

- `get_params()` 返回构造配置，适合克隆、搜索和重新配置。
- `get_native_params()` 返回实际训练使用的原生构造参数；自动类别权重等计算值在这里检查。
- `get_native_model()` 返回已训练的底层对象，可以直接使用原生方法。
- `params={...}` 继续支持。初始化时保留历史规则：该字典优先于同名关键字；后续 `set_params()`
  的显式更新优先于旧字典，不会修改调用者传入的字典。
- 常用别名会被规范化，例如 CatBoost 的 `n_estimators`、`max_depth`、`reg_lambda`，
  LightGBM 的 `num_iterations`、`feature_fraction`，XGBoost 的 `eta`、`lambda`。
- 各框架的专属参数、损失函数和评估函数保留原生语义。XGBoost 的自定义评估函数返回数值，
  LightGBM 返回 `(名称, 数值, 是否越大越好)`，CatBoost 接收原生评估对象。
  `predict()` / `predict_proba()` 的额外预测参数也会传递给原生模型。

特征重要性有两个用途：`feature_importances_` 始终与输入字段顺序一致，供 sklearn 筛选器使用；
`get_feature_importances(importance_type=...)` 返回排序后的具名 Series，供查看和报告使用。

## 3. 保存完整模型与训练过程

```python
model.save("artifacts/model.joblib")
restored = type(model).load("artifacts/model.joblib")

restored.training_summary_              # 最近一次训练的状态、参数、用时、依赖版本等
restored.training_history_              # 历次训练，包括失败和中断
restored.evals_result_                  # 框架实际产生的逐轮指标
restored.best_iteration_
restored.predict_score(df[features])
```

默认保存使用 cloudpickle，支持 Notebook 中定义的函数、损失和回调，原有 joblib/pickle/dill
引擎选项与旧文件加载方式保留。完整模型文件写入成功后才替换目标，序列化失败不会破坏原文件。
未实际计算的指标不会伪造为训练曲线；查看逐轮验证曲线时，应配置 `eval_set` 或启用自动早停。

| 保存方式 | 内容与用途 |
| --- | --- |
| `save("model.joblib")` | 完整 Python 对象：模型、评分转换器、训练历史及关联调参器 |
| `save_artifact(...)` | 既有统一制品接口，可显式指定 `engine="cloudpickle"` |
| `save("model.json")` | 支持框架的 JSON 清单，配套原生模型、评分转换器和 `.state.pkl` 完整状态文件；需一起移动 |
| `save_model(...)` | 原生模型文件与评分转换器附属文件，保留已有接口 |
| `get_native_model()` | 直接调用底层框架的导出、树结构、叶节点、预测或其他专属能力 |

输入预处理也是复用的一部分：推荐把编码、标准化与模型放进 sklearn Pipeline 一起保存；
如果预处理在外部完成，必须同时保存已拟合的预处理器和字段顺序。
训练记录保存配置和模型过程，不自动复制作为位置参数传入的原始训练矩阵。
CatBoost 验证集 Pool 的原生句柄不能深拷贝，训练记录保留其标签、权重、基线与字段结构。

Python 完整对象不承诺任意跨版本加载。版本迁移时，先在新环境运行验证脚本；
若原框架的对象结构变化，使用该框架支持的原生导出和加载方式。记录中的 `依赖版本` 用于定位来源环境。

## 4. 在 tune 中直接使用 Optuna

`tune()` 返回最佳模型；原始模型和返回模型的 `.tuner` 指向同一调参器。
原始模型本身不会被替换为最佳模型。

```python
import optuna
from pathlib import Path
from hscredit import RandomForest

Path("artifacts").mkdir(exist_ok=True)
study = optuna.create_study(
    direction="maximize",
    study_name="FPD随机森林",
    storage="sqlite:///artifacts/tuning.db",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)

def search(trial):
    depth = trial.suggest_int("max_depth", 2, 8)
    return {
        "max_depth": depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        # 可继续调用 suggest_float、suggest_categorical，或用 if 定义条件空间。
    }

def on_trial(study, trial):
    print(trial.number, trial.state, trial.values)

model = RandomForest(n_estimators=100, target="FPD", random_state=42, n_jobs=4)
best = model.tune(
    df[features + ["FPD"]],
    search_space=search,
    study=study,
    callbacks=[on_trial],
    metric="auc",
    cv=3,
    n_trials=20,
    catch=(ValueError,),
    gc_after_trial=True,
    artifact_dir="artifacts/trials",
)
```

先创建数据库父目录。传入已有 Study 时，采样器、剪枝器与持久化连接由该 Study 管理；
不要在包装器上重复配置另一套 Study 设置。`direction` 必须与已有 Study 一致。
已有字典搜索空间、Optuna 分布对象、多目标、手工点、Hyperopt/skopt 风格声明和各类可视化保持可用。

| 需求 | 入口 |
| --- | --- |
| 原生 Trial 条件采样 | `search_space=lambda trial: {...}` |
| 已有 Study、采样器、剪枝器 | `study=study`；自行创建 Study 时也可用 `sampler=`、`pruner=` |
| Trial 完成后的回调 | `callbacks=[fn]`，签名与 `Study.optimize` 一致 |
| 异常策略、垃圾回收 | `catch=(...)`、`gc_after_trial=True` |
| 底层模型 fit 参数 | `fit_params={"callbacks": [...], ...}` |
| 每折创建原生训练回调 | `fit_params_factory(trial, fold_index) -> dict` |
| 分组、时间、重复验证 | `cv=GroupKFold(...)`、`cv=TimeSeriesSplit(...)` 或索引迭代器；必要时 `groups=` |
| 原生完整目标函数 | `trial_objective=fn`，`fn(trial)` 完全接管试验；与既有 `objective(y, p)` 评分函数区分 |
| 原生高级操作 | `model.tuner.study_.enqueue_trial(...)`、`stop()`、`ask()`、`tell()` 等原生 Study 方法 |

`fit_params_factory` 每折调用一次；最终重训会以 `(None, None)` 调用，应在此分支返回最终训练参数，
避免把绑定某个 Trial 的剪枝回调复用到最终模型上。已知样本级参数如 `sample_weight`、`base_margin`、
`init_score`、`baseline` 及 Pipeline 中对应的 `步骤名__参数` 会按折同步切分。
外部固定 `eval_set` 会按原生语义传递，应由调用者确保它不包含外层验证或最终测试样本。

自动评分采用外层交叉验证；包装模型内部早停和原生提升器的内部分割均只使用训练折。
单目标试验每完成一折执行一次 `trial.report()` / `should_prune()`。
多目标继续返回各目标结果与 Pareto 前沿，不调用 Optuna 不支持的多目标 `report()`。

为了维持现有 CPU 预算和可中断行为，Trial 依旧顺序执行，`n_jobs` 分配给当前模型。
这不是将 `n_jobs` 直接当成 `Study.optimize` 的并行试验数。

## 5. 保留每折内容、继续搜索、复用最佳模型

```python
tuner = best.tuner
tuner.get_trial_result(0)                # 参数、每折模型、训练/验证预测、指标、曲线和异常
tuner.get_oof_predictions()              # 样本位置、真实标签、折外预测概率、验证次数
tuner.get_optimization_history()         # 保留原有 Optuna 风格历史表
tuner.get_best_model() is best           # True：不会再次训练

tuner.save("artifacts/tuner.joblib")
restored = type(tuner).load("artifacts/tuner.joblib")
restored.enqueue_trial({"max_depth": 4})
restored.fit(df[features + ["FPD"]], n_trials=10)
new_best = restored.get_best_model()
```

完整调参制品包括输入数据、标签和权重，适合继续原任务；换数据集或改变验证口径时，应创建新的 Study。
SQLite 保留 Optuna 试验状态；`artifact_dir` 额外逐折写入模型过程，可在新调参器中通过 Study 记录的路径恢复。
只使用 SQLite、不保存完整对象或配置制品目录，不能恢复已丢失的 Python 模型和预测数组。
同样，只调用原生 `trial_objective` 时，内部训练过程由用户函数管理，自动逐折记录不适用。

默认 `store_models=True` 会保留所有训练折的模型，适合复盘；试验较大时可显式设置 `store_models=False`
仅保留预测、指标和训练记录。配置 `artifact_dir` 会同时写磁盘，当前实现仍保留内存记录。
自定义函数返回不能放入 Optuna JSON 属性的模型对象或损失对象时，完整配置保存在调参制品中。

最佳模型只训练一次并缓存。Boosting 最终默认使用各折最佳轮数的中位数关闭内部早停，在全部输入上训练；
可用 `get_best_model(refit=True, full_data=False)` 显式重训并保留内部验证。
中断会继续抛出 `KeyboardInterrupt`，不会悄悄开始最终重训，但已经完成的试验和历史仍可检查、保存。

## 6. 可运行的真实数据验证

```console
python examples/28_model_workflow.py --output artifacts/model-workflow
```

脚本使用 `examples/hscredit_yyp.xlsx` 的指定三项特征和 `FPD`，覆盖十类分类器、中文类别字段、
概率/评分保存往返、学习曲线、报告入口、Optuna 持久化与继续搜索；同时保存标准化器和字段顺序。
省略 `--output` 时输出到系统临时目录。
