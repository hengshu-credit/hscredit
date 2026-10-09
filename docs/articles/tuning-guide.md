# 超参数搜索：入门、指标、损失与结果复盘

`ModelTuner` 使用 Optuna 采样候选参数，在外层交叉验证中评估候选，保留试验过程，再用选中的参数训练最终模型。已有 HSCredit 模型时，优先用 `model.tune(...)`，直接得到可预测的最佳模型。

完整可运行演示：[tuning_workflow.ipynb](../../examples/tuning_workflow.ipynb)。所有代码格使用仓库真实放款数据；模型数量和训练轮数控制在演示规模。损失选择与框架回调详见 [损失指南](losses-guide.md) 和 [losses_workflow.ipynb](../../examples/losses_workflow.ipynb)。

## 先选对入口

| 需求 | 推荐入口 | 返回内容 |
| --- | --- | --- |
| 已有 HSCredit 模型，想得到最优模型 | `best = model.tune(X, y, ...)` | 已重训的模型，搜索器在 `best.tuner` |
| 要管理 Study、逐折结果或 sklearn Pipeline | `tuner = ModelTuner(model_or_class, ...)`，再 `fit(...)` | `fit` 返回参数字典；模型通过 `get_best_model()` 获取 |
| 已有参数化评估函数 | `make_metric(fn, name=..., greater_is_better=...)` | 可离线评估、训练监控及调参的 `BaseMetric` |
| 已有 BaseLoss | `loss=loss, metric=None` | 用该损失训练，按其同口径指标搜索 |
| loss 与业务评价不同 | `loss=loss, metric="auc"` 或业务指标对象 | 训练优化代理损失，搜索按业务口径选优 |
| 搜索需要关联/条件参数 | `search_space(trial) -> dict` | 每个 trial 的模型参数字典 |
| 完全自定义训练和评分循环 | `trial_objective(trial)` | 原始单目标数值或多目标序列，由函数自行管理训练过程 |

### 区分训练目标、搜索指标与试验函数

| 名称 | 接收内容 | 用途 |
| --- | --- | --- |
| `ModelTuner.loss` | `BaseLoss` 实例 | 注入 HSCredit XGBoost、LightGBM、CatBoost 的训练目标 |
| `ModelTuner.metric` | 字符串、`BaseMetric`、明确方向的函数，或上述列表 | 决定 trial 的优劣 |
| 模型 `eval_metric` | 对应模型支持的评估配置 | 训练曲线与早停；不自动等于搜索 metric |
| `ModelTuner.objective` | 历史评分目标名/函数 | 保留兼容的 `metric` 替代入口；不是训练 loss |
| `ModelTuner.trial_objective` | `(trial) -> float` 或序列 | 完全接管一轮搜索，可用于外部训练流程 |

`loss.metric()` 是同口径训练代理损失，始终最小化；`loss.business_metric()` 是对应真实业务指标，方向随指标变化。不要把训练 loss 的低值直接当作 AUC、利润或坏账率的改进。

## 环境与数据准备

安装 `pip install -e ".[boost,tune,dev]"`，从仓库根目录或 `examples` 目录执行。下文代码按顺序复用变量。

数据约定：三项特征为 `衡枢鉴真分老客版`、`近六个月非银多头机构数`、`青云24`；`FPD=1` 表示坏样本。先划出独立测试集，再在训练数据内做搜索和早停。示例使用 LightGBM 原生缺失值处理，Pipeline 示例则逐折拟合预处理器。


```python
from pathlib import Path
from tempfile import mkdtemp
import sys

repository_root = next(
    (path for path in (Path.cwd(), Path.cwd().parent) if (path / "hscredit").is_dir()),
    None,
)
if repository_root is None:
    raise FileNotFoundError("请从仓库根目录或 examples 目录启动 Notebook")
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from hscredit import LightGBM, ModelTuner
from hscredit.core.models.tuning import Integer, Real, Categorical
from hscredit.core.models.losses import (
    BaseLoss, BaseMetric, FocalLoss, WeightedBCELoss, AUCMetric, make_metric,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
data = pd.read_excel(repository_root / "examples" / "hscredit_yyp.xlsx")
data = data.loc[data["FPD"].isin([0, 1])].copy()
X = data[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
y = data["FPD"].astype(int)
amount = pd.to_numeric(data["放款金额"], errors="raise")
X_train, X_test, y_train, y_test, amount_train, amount_test = train_test_split(
    X, y, amount, test_size=0.3, stratify=y, random_state=42,
)
print(f"训练样本数：{len(X_train)}；测试样本数：{len(X_test)}")
```

## 1. 已有模型：用 `.tune()` 获取最佳模型

最少只需模型、搜索空间和评估指标。`metric="auc"` 自动最大化，`metric="log_loss"` 自动最小化，不必重复写方向。已有模型的其他设置继续保留；搜索参数覆盖原配置，显式 `fixed_params` 的优先级最高。

`tune()` 返回已经重训好的最佳模型。原模型与返回模型均通过 `.tuner` 查看同一次搜索，原对象本身不会被替换为最佳模型。

```python
model = LightGBM(n_estimators=30, num_leaves=7, n_jobs=1, random_state=42)
best = model.tune(
    X_train, y_train,
    search_space={"max_depth": [3, 4], "learning_rate": [0.03, 0.08]},
    metric="auc", cv=3, n_trials=3, n_jobs=1,
    retention="predictions", show_progress_bar=False,
)
tuner = best.tuner
print("最佳参数：", tuner.best_params_)
print(f"CV 平均 AUC：{tuner.best_score_:.6f}")
print(f"独立测试 AUC：{AUCMetric().evaluate(y_test, best.predict_proba(X_test)):.6f}")
assert tuner.get_best_model() is best
```

## 2. 独立调参器：理解 `fit` 的返回值

`ModelTuner.fit()` 为兼容旧 API 返回**最佳参数字典**，不是调参器或模型。先保存 `tuner`，再调用 `get_best_model()`。首次取模型会在全部输入训练数据上重训并缓存，后续调用复用缓存；`refit=True` 显式要求再训练。

以下辅助函数仅减少本 Notebook 的重复配置。每次都新建调参器；不同数据、指标或 CV 口径使用不同 Study。

```python
def make_tuner(metric="auc", **kwargs):
    """创建小规模演示搜索器，默认不做内部早停。"""
    options = dict(
        model_class=LightGBM,
        search_space={"max_depth": [3, 4]},
        fixed_params={"n_estimators": 25, "num_leaves": 7, "validation_fraction": 0.0},
        metric=metric, cv=3, n_jobs=1, random_state=42,
        early_stopping_rounds=None, retention="predictions",
    )
    options.update(kwargs)
    return ModelTuner(**options)

loss_score_tuner = make_tuner(metric="log_loss")
best_params = loss_score_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
assert isinstance(best_params, dict)
logloss_best = loss_score_tuner.get_best_model()
print(f"CV 平均交叉熵：{loss_score_tuner.best_score_:.6f}；越小越好")
```

## 3. 搜索空间：先学三种声明，再学条件空间

`Integer` 是整数范围，`Real` 是连续范围，`Categorical` 是离散选项。列表也是离散候选，**列表声明不意味着穷举所有组合**；默认仍由 TPE 采样。固定参数放入 `fixed_params`，避免当作搜索维度。

`Integer(3, 4)` 包含两端；学习率跨数量级时可用 `Real(..., prior="log-uniform")`。调参器的 `n_jobs` 主要控制模型并行，trial 按顺序运行。

```python
space = {
    "max_depth": Integer(3, 4),
    "learning_rate": Real(0.02, 0.1, prior="log-uniform"),
    "reg_lambda": Categorical([0.0, 1.0, 3.0]),
}
space_tuner = make_tuner(search_space=space)
space_tuner.enqueue_trial({"max_depth": 3, "learning_rate": 0.05, "reg_lambda": 1.0})
space_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
print("搜索空间示例最佳参数：", space_tuner.best_params_)
```

复杂关联用 `search_space(trial) -> dict`：例如叶子数不能超过 `2 ** max_depth`，或某模型参数仅在另一参数取特定值时出现。返回的键必须是模型构造参数；Trial 内部参数名也会保留在 Study 中。

```python
def conditional_space(trial):
    depth = trial.suggest_int("max_depth", 2, 4)
    return {
        "max_depth": depth,
        "num_leaves": trial.suggest_int("num_leaves", 3, min(15, 2 ** depth)),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
    }

conditional_tuner = make_tuner(
    search_space=conditional_space,
    fixed_params={"n_estimators": 25, "validation_fraction": 0.0},
)
conditional_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
print("条件空间最佳参数：", conditional_tuner.best_params_)
```

## 4. 自定义评估指标：普通函数只需加名称和方向

评估函数签名是 `(y_true, y_prob, sample_weight=None, **固定参数) -> float`；收到的 `y_prob` 是一维坏样本概率，不是 logits，也不是两列概率。

用 `make_metric` 包装后，离线评估、训练监控、ModelTuner 和 sklearn scorer 都可复用同一个对象。没有额外参数时可以直接使用 sklearn 的 `brier_score_loss` 等概率指标。

```python
def probability_error(y_true, y_prob, sample_weight=None, scale=1.0):
    """均方概率误差，scale 是固定业务系数。"""
    error = (np.asarray(y_prob) - np.asarray(y_true)) ** 2
    return float(scale * np.average(error, weights=sample_weight))

brier = make_metric(
    probability_error, name="均方概率误差", greater_is_better=False, scale=1.0,
)
function_tuner = make_tuner(metric=brier)
function_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
function_model = function_tuner.get_best_model()
print(f"自定义指标 CV：{function_tuner.best_score_:.6f}")
print(f"自定义指标测试值：{brier.evaluate(y_test, function_model.predict_proba(X_test)):.6f}")
assert brier.direction == "minimize"

# 也可直接传裸函数，但必须显式声明方向，避免误把误差最大化。
raw_function_tuner = make_tuner(metric=probability_error, direction="minimize")
raw_function_tuner.fit(X_train, y_train, n_trials=1, show_progress_bar=False)
```

## 5. 需要封装状态：继承 BaseMetric

指标需要多个参数、方法或业务语义时再定义类。实现 `__call__` 即可，基类 `evaluate()` 会校验输入、支持两列概率及显式原始分数转换。额外权重要在 `__call__` 中明确接收并使用，不能静默忽略。

```python
class ProbabilityError(BaseMetric):
    """可复用的均方概率误差。"""
    def __init__(self):
        super().__init__(name="均方概率误差", greater_is_better=False)

    def __call__(self, y_true, y_pred, sample_weight=None):
        error = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
        return float(np.average(error, weights=sample_weight))

custom_metric = ProbabilityError()
class_tuner = make_tuner(metric=custom_metric)
class_tuner.fit(X_train, y_train, n_trials=1, show_progress_bar=False)
assert np.isclose(custom_metric.evaluate(y_test, best.predict_proba(X_test)), brier.evaluate(y_test, best.predict_proba(X_test)))
print("BaseMetric 自动采用方向：", class_tuner.directions)
```

## 6. 自定义 loss 训练，固定 metric 选择模型

`loss=...` 便捷参数支持 HSCredit `LightGBM` / `XGBoost` / `CatBoost` 包装器。设 `metric=None` 会使用 `loss.metric()`；设 `metric="auc"` 则训练优化该 loss、搜索最大化 AUC。

`ModelTuner(objective=...)` 是历史上的**搜索评分目标**别名，不是模型训练 loss。原生 sklearn estimator 请在模型构造器或 `fixed_params` 中按该框架签名配置训练目标。

```python
loss = FocalLoss(alpha=0.75, gamma=2.0)
focal_tuner = make_tuner(loss=loss, metric=None)
focal_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
focal_model = focal_tuner.get_best_model()
print(f"Focal CV 损失：{focal_tuner.best_score_:.6f}")
print(f"Focal 测试损失：{loss.evaluate(y_test, focal_model.predict_proba(X_test)):.6f}")
assert focal_tuner.directions == ["minimize"]
```

搜索 loss 自身参数时，在条件搜索函数返回模型的 `objective` 对象。**评估指标必须固定**：下例搜索 alpha/gamma，但始终用 AUC 比较。不能让每个 trial 的 loss 参数同时改变评价标尺。

```python
def loss_parameter_space(trial):
    return {
        "objective": FocalLoss(
            alpha=trial.suggest_float("alpha", 0.4, 0.8),
            gamma=trial.suggest_float("gamma", 0.5, 2.0),
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 4),
    }

loss_parameter_tuner = make_tuner(metric=AUCMetric(), search_space=loss_parameter_space)
loss_parameter_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
print(f"搜索 loss 参数后的 CV AUC：{loss_parameter_tuner.best_score_:.6f}")
```

## 7. 从零实现一个自定义训练 loss

Brier loss 对概率求导，适配器负责转换为原始分数导数。逐样本贡献的均值与 `__call__` 相同，`gradient` / `hessian` 是“样本数 × 平均损失”的导数；`is_additive=True` 表示允许分批与外部权重。

```python
class BrierLoss(BaseLoss):
    """均方概率误差训练目标。"""
    is_additive = True

    def __init__(self):
        super().__init__("均方概率损失")

    def loss_values(self, y_true, y_pred):
        return (np.asarray(y_pred) - np.asarray(y_true)) ** 2

    def __call__(self, y_true, y_pred):
        return float(np.mean(self.loss_values(y_true, y_pred)))

    def gradient(self, y_true, y_pred):
        return 2 * (np.asarray(y_pred) - np.asarray(y_true))

    def hessian(self, y_true, y_pred):
        return np.full(len(y_true), 2.0)

custom_loss_tuner = make_tuner(loss=BrierLoss(), metric=brier)
custom_loss_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
custom_loss_model = custom_loss_tuner.get_best_model()
print(f"自定义 loss 模型测试 Brier：{brier.evaluate(y_test, custom_loss_model.predict_proba(X_test)):.6f}")
```

## 8. 金额加权：区分训练权重和评估权重

`sample_weight` 控制训练；`evaluation_weight` 控制 CV 选优。只传训练权重不会暗中改变评估口径。两者都应与 `X_train` 按行位置对齐，调参器按训练/验证折分别切分。

需要金额加权 BCE 时，使用普通 `WeightedBCELoss` 加权即可。不把全量金额数组绑定进跨折复用的 `AmountWeightedLoss`；绑定数组需要手动逐折创建损失和指标。

```python
weighted_loss = WeightedBCELoss()
weighted_tuner = make_tuner(loss=weighted_loss, metric=weighted_loss.metric())
weight_train = amount_train.to_numpy(dtype=float)
weighted_tuner.fit(
    X_train, y_train, sample_weight=weight_train, evaluation_weight=weight_train,
    n_trials=2, show_progress_bar=False,
)
weighted_model = weighted_tuner.get_best_model()
weighted_test = weighted_loss.evaluate(
    y_test, weighted_model.predict_proba(X_test), sample_weight=amount_test.to_numpy(dtype=float),
)
print(f"CV 金额加权 BCE：{weighted_tuner.best_score_:.6f}；测试值：{weighted_test:.6f}")
```

## 9. sklearn Pipeline：预处理在每折内部拟合

模型可传 estimator 实例，Pipeline 内的参数用 `步骤名__参数`。使用自定义空间，不依赖 HSCredit 特定模型的默认空间。不要先对全量数据拟合填充器/标准化器再做 CV。

```python
pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
pipeline_tuner = ModelTuner(
    pipeline, search_space={"logisticregression__C": Real(0.01, 10.0, prior="log-uniform")},
    metric=brier, cv=3, n_jobs=1, random_state=42, retention="predictions",
)
pipeline_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
pipeline_best = pipeline_tuner.get_best_model()
print(f"Pipeline 测试 Brier：{brier.evaluate(y_test, pipeline_best.predict_proba(X_test)):.6f}")
```

## 10. 多目标：保留所有不能同时被超越的候选

`metric=[...]` 声明多个目标，各指标自动给出方向。这里同时最大化 AUC、最小化 Brier。`get_pareto_front()` 返回 Pareto 候选；默认 `get_best_model()` 按指标列表顺序优先第一个指标，同分时看下一个，不能代替业务权衡。

```python
multi_tuner = make_tuner(metric=[AUCMetric(), brier], sampler="nsgaii")
multi_tuner.fit(X_train, y_train, n_trials=3, show_progress_bar=False)
pareto = pd.DataFrame([
    {"试验编号": trial.number, "AUC": trial.values[0], "均方概率误差": trial.values[1]}
    for trial in multi_tuner.get_pareto_front()
])
print(pareto.to_string(index=False))
print("默认最优试验：", multi_tuner.best_trial_.number)
assert multi_tuner.directions == ["maximize", "minimize"]
```

## 11. 时间交叉验证：按放款时间模拟向未来预测

按 `放款时间` 排序，最后 20% 保留作时间外测试。`TimeSeriesSplit` 只用过去训练、未来验证，不打乱时间。AUC 要求每个验证折同时含好坏两类，先验证这个条件。时间 CV 的最早一段不会获得 OOF 预测，输出保留缺失值是正常现象。

客户/门店分组场景则使用 `cv=GroupKFold(...)` 并在 `fit(..., groups=分组数组)` 中传入真实组标识。

```python
time_data = data.assign(放款时间=pd.to_datetime(data["放款时间"], errors="coerce"))
time_data = time_data.dropna(subset=["放款时间"]).sort_values("放款时间", kind="stable")
cut = int(len(time_data) * 0.8)
time_train, time_test = time_data.iloc[:cut], time_data.iloc[cut:]
X_time = time_train[FEATURES].apply(pd.to_numeric, errors="coerce")
y_time = time_train["FPD"].astype(int)
time_cv = TimeSeriesSplit(n_splits=3)
assert all(y_time.iloc[valid].nunique() == 2 for _, valid in time_cv.split(X_time))
time_tuner = make_tuner(cv=time_cv)
time_tuner.fit(X_time, y_time, n_trials=2, show_progress_bar=False)
time_model = time_tuner.get_best_model()
time_probability = time_model.predict_proba(time_test[FEATURES].apply(pd.to_numeric, errors="coerce"))
print(f"时间外测试 AUC：{AUCMetric().evaluate(time_test['FPD'], time_probability):.6f}")
```

## 12. 结果复盘、OOF 和复训

`best_score_` 是外层 CV 各折指标均值，不是最终模型在训练集或测试集的得分。`get_trial_result()` 查看逐折记录，`get_oof_predictions()` 查看折外预测；OOF 只用于训练样本的折外估计。

`get_optimization_history()` 保留兼容的 Optuna 历史字段；`retention="predictions"` 保存预测但不保留各折模型，适合本例。最终模型另有缓存。

```python
record = tuner.get_trial_result(tuner.best_trial_.number)
print("最佳试验状态：", record["状态"], "；折数：", len(record["各折"]))
oof = tuner.get_oof_predictions()
print("OOF 列：", list(oof.columns), "；行数：", len(oof))
assert len(oof) == len(X_train)
print(tuner.get_optimization_history().head().to_string(index=False))
assert tuner.get_best_model() is best
refitted = tuner.get_best_model(refit=True)
print(f"显式复训后的测试 AUC：{AUCMetric().evaluate(y_test, refitted.predict_proba(X_test)):.6f}")
```

## 13. 保存与恢复：完整对象、SQLite 和推理模型

完整 `tuner.save()` 包含搜索配置、原始训练数据、指标函数、Study 和当前保留的结果，适合继续同一个任务。SQLite 保存 Optuna 的 trial 状态，不自动保存 Python 模型和预测；需要跨进程读取折结果时同时设置 `artifact_dir`。

此例在新临时目录写文件，不覆盖现有结果。恢复后传入相同数据和 CV/指标配置再 `fit(n_trials=1)` 表示**追加 1 次 trial**。数据、行顺序或评估口径发生变化，应新建调参器与 Study。

```python
output_dir = Path(mkdtemp(prefix="hscredit_tuning_workflow_"))
storage_url = "sqlite:///" + (output_dir / "search.db").as_posix()
persistent_tuner = make_tuner(
    metric=brier, storage=storage_url, study_name="FPD概率误差演示",
    artifact_dir=str(output_dir / "trials"), retention="disk",
)
persistent_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
persistent_best = persistent_tuner.get_best_model()
tuner_path = output_dir / "tuner.joblib"
persistent_tuner.save(tuner_path)
restored = ModelTuner.load(tuner_path)
np.testing.assert_allclose(restored.get_best_model().predict_proba(X_test), persistent_best.predict_proba(X_test))
old_count = len(restored.study_.trials)
restored.fit(X_train, y_train, n_trials=1, show_progress_bar=False)
assert len(restored.study_.trials) == old_count + 1
restored.get_best_model().save_inference(output_dir / "最佳预测模型.joblib")
print("验证输出目录：", output_dir)
print("恢复后试验总数：", len(restored.study_.trials))
```

## 14. 可视化与扩展入口

完整搜索过程保留在 `tuner.get_study()`。下列方法生成图对象；在 Jupyter 末行展示或调用 `.show()` 即可。少量 trial 的参数重要性不稳定，本例只构造历史图并核验对象。

更复杂的训练逻辑可使用 `fit_params_factory(trial, fold_index)` 每折创建训练回调，或 `trial_objective(trial)` 完全接管试验。后者需要自己管理切分、训练、评价与记录，不享有自动逐折保存。

```python
figure = tuner.visualization.plot_optimization_history()
assert len(figure.data) > 0
print("已生成搜索历史图，可运行 figure.show() 查看。")
```

## 15. 常见误用检查

- 调参结果“反着选”：裸函数必须声明方向；指标对象通过 `greater_is_better` 声明一次。
- `metric="log_loss"` 是正的交叉熵，越小越好；旧 `"logloss"` / `"neg_log_loss"` 是负值，越大越好，保留作兼容。
- `loss` 决定训练，`metric` 决定候选优劣，模型的 `eval_metric` 决定训练监控/早停；三者可以不同。
- `ModelTuner(objective=...)` 是旧评分入口；自定义训练用 `loss=...`，完全接管 trial 才用 `trial_objective=...`。
- 自定义空间的名字必须是模型参数名；Pipeline 用 `步骤名__参数`；列表空间不是自动穷举。
- 不同 loss 参数的损失数值不可直接比大小；调 loss 参数时使用固定 AUC、固定 Brier 或固定业务指标。
- 加权训练和加权评价分别传 `sample_weight`、`evaluation_weight`；自定义指标必须支持权重。
- 测试集不传入搜索/早停；绑定金额等行级数组不能直接跨折复用。
- 只保存 SQLite 不能恢复 Python 模型；完整制品、折制品目录、外部数据库各自保留。


## 16. 常用参数按职责查阅

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `model_class` | `ModelTuner` | 模型类或可克隆 estimator 实例；Pipeline 用实例 |
| `search_space` | 构造器 / `.tune` | 字典、声明对象或 `(trial) -> dict`；推荐 `Integer` / `Real` / `Categorical` |
| `model_params` | 构造器 | 模型基准配置；`model.tune` 自动带入原模型配置 |
| `fixed_params` | 构造器 / `.tune` | 固定模型配置，覆盖同名采样值；避免在两处定义同名参数 |
| `metric` | 构造器 / `.tune` | 单目标或列表；`None` 配合 loss 自动生成配套指标 |
| `direction` | 构造器 / `.tune` | 默认从内置指标或指标对象推断；裸函数必须指定 |
| `metric_names` | 构造器 | 需要覆盖指标显示名时提供同长度列表，不得重名 |
| `cv` | 构造器 / `.tune` | 整数分层 K 折、splitter 对象或 `(train_idx, valid_idx)` 迭代器 |
| `groups` | `fit` / `.tune` | 分组 splitter 需要的行级组标签；不属于模型参数 |
| `n_trials` | `fit` / `.tune` | 本次新增 trial 数，不是整个 Study 的累计上限 |
| `timeout` | `fit` / `.tune` | 本次优化的总时间预算；正在运行的模型 fit 不保证即时被打断 |
| `n_jobs` | 构造器 / `.tune` | 模型侧并行预算；不代表并行 trial 数 |
| `sampler` / `sampler_kwargs` | 构造器 | 默认 TPE，可传 `random` / `nsgaii` 或原生采样器实例 |
| `trial_points` / `enqueue_trial` | 构造器 / 方法 | 优先评估已有经验参数；部分键未提供时其余维度仍采样 |
| `early_stopping_rounds` | 构造器 | 支持早停模型的默认监控轮数；监控只使用外层训练折内部数据 |
| `sample_weight` | `fit` / `.tune` | 训练样本权重，按训练折切分 |
| `evaluation_weight` | `fit` / `.tune` | 显式评估权重，按训练/验证评分对应数据切分 |
| `fit_params` | 构造器 | 传给底层模型 fit 的固定参数；支持已识别的样本级参数切分 |
| `fit_params_factory` | 构造器 | 每折创建新训练回调；最终重训传 `(None, None)` |
| `callbacks` / `pruner` | 构造器 | Optuna 完成回调与剪枝器；区别于模型训练回调 |
| `catch` | `fit` / `.tune` | 允许某类异常记作 trial 失败后继续；默认让错误暴露 |
| `storage` / `study_name` | 构造器 | Optuna 存储和名字，支持 SQLite |
| `study` | 构造器 / `.tune` | 直接使用原生 Study，方向、采样器和存储以该对象为准 |
| `artifact_dir` | 构造器 / `.tune` | 持久化逐折 Python 制品，不等同于 Optuna storage |
| `retention` | 构造器 / `.tune` | `full` / `predictions` / `summary` / `best` / `disk` |

### 默认空间与跨库写法

初学时为 1–3 个关键参数给出小范围、固定其他配置。已知 HSCredit 模型可省略空间使用自适应默认值；未知模型或 Pipeline 应显式给空间。

内部统一使用 Optuna 搜索。接收 sklearn 列表、skopt、Hyperopt、Bayesian Optimization 等声明，表示**兼容空间写法**，不表示切换到那些库的搜索算法。`sampler` 才决定 Optuna 如何采样。

```python
# 等价表达：整数区间包含两端，列表表示离散候选。
spaces = {
    "简写": {"max_depth": (3, 5), "learning_rate": [0.03, 0.05, 0.1]},
    "声明对象": {"max_depth": Integer(3, 5), "learning_rate": Categorical([0.03, 0.05, 0.1])},
    "字典": {"max_depth": {"type": "int", "low": 3, "high": 5},
             "learning_rate": {"type": "categorical", "choices": [0.03, 0.05, 0.1]}},
}

# 确实需要完整网格时，显式入队所有点，并给足 trial 数。
grid_tuner = make_tuner(search_space=spaces["简写"])
grid_tuner.enqueue_trials(param_grid={"max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.1]})
grid_tuner.fit(X_train, y_train, n_trials=9, show_progress_bar=False)
```

### 分组验证

以下 `商品类别` 仅用于演示按类别隔离，不是通用的客户防泄漏分组。实际客户有多条记录时应传真实客户标识；至少有 `n_splits` 个独立组，并检查每个验证折的标签。

```python
from sklearn.model_selection import GroupKFold

groups = data.loc[X_train.index, "商品类别"].fillna("未标注类别").astype(str).to_numpy()
group_cv = GroupKFold(n_splits=3)
if len(np.unique(groups)) < 3:
    raise ValueError("可用分组不足3个，请减少折数或选择合适的真实分组标识")
group_tuner = make_tuner(cv=group_cv, metric="log_loss")
group_tuner.fit(X_train, y_train, groups=groups, n_trials=2, show_progress_bar=False)
```

### 每折独立生成原生回调

```python
import lightgbm as lgb

def fit_options(trial, fold_index):
    if trial is None:  # 最终全数据重训，没有内部早停监控集
        return {}
    return {"callbacks": [lgb.log_evaluation(period=0)]}

callback_tuner = make_tuner(fit_params_factory=fit_options)
callback_tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False)
```

固定外部 `eval_set` 会按模型原生语义传递；调用方必须确保它不包含外层验证或最终测试样本。普通使用无需手动提供，包装模型会在外层训练折内部切出监控集。最终重训默认结合各折最佳轮数，在全部输入训练数据上拟合；`get_best_model(refit=True, full_data=False)` 显式保留内部验证流程。

### 内置指标与优化方向

| 指标 | 方向 | 数值含义 |
| --- | --- | --- |
| `"auc"` | 最大化 | 风险方向 AUC；反向预测不会自动翻转为高 AUC |
| `"ks"` | 最大化 | 好坏风险分布区分度 |
| `"ks_diff"` | 最小化 | 训练与验证 KS 差异；建议与区分度搭配，单独低差异不保证有效 |
| `"log_loss"` | 最小化 | 正的二元交叉熵；新代码优先用此名称 |
| `"logloss"` / `"neg_log_loss"` | 最大化 | 负的二元交叉熵；保留旧比较方向 |
| `BaseMetric` / `make_metric(...)` | 对象声明 | 原始数值；不会自动取负 |
| `loss.metric()` | 最小化 | 固定损失参数的同口径代理损失 |
| 裸函数 | 必须指定 | 返回一个有限标量；不猜测业务方向 |

仅 sklearn 的 `metric.to_scorer()` 将最小化指标取负，不能把负 scorer 直接当作 ModelTuner 的概率评估函数。ModelTuner 评分函数签名是 `(y_true, y_prob)`，sklearn scorer 签名是 `(estimator, X, y)`。

## 17. 留存、恢复与适用边界

| retention | 保存内容 | 适合 |
| --- | --- | --- |
| `full` | 各折模型、预测、完整训练记录 | 小规模详细复盘，默认 |
| `predictions` | 预测和指标，不留各折模型 | OOF 分析，本教程默认 |
| `summary` | 指标、曲线、样本数、错误摘要 | 较大搜索；不提供 OOF |
| `best` | 当前最佳试验完整结果，其他仅摘要 | 控制内存并重点复盘最佳候选 |
| `disk` | 内存留摘要，完整折结果写入 artifact_dir | 大模型，需要保留磁盘目录 |

`get_trial_result(number, load=False)` 读取摘要；`disk` 默认 `load=True` 按需读取完整折结果。`get_oof_predictions()` 会合并重复验证预测，未验证的位置保留缺失值。`evaluate_trials` / `evaluate_study_trials` 会重新训练评估，不是只读查询；换数据时传入新的 `X/y/groups/evaluation_weight`，不会复用旧 CV 行位置。

`release_training_data()` 释放输入与行级训练参数后，缓存模型仍可预测；最终重训和完整 OOF 需要重新提供训练数据。`save_inference()` 保存不带整个搜索历史的预测模型。完整 `save()` 适合复盘、继续同一个任务，并会保留训练数据。

自动续跑会验证数据、CV 和评分配置的一致性，避免把不同实验的分数混进同一 Study。已存在的 Study 必须与当前指标方向一致；没有可确认上下文的外部历史，不要当作当前实验的可续跑记录。数据、顺序、标签、权重、CV 或指标参数更改时，新建 Study 最清晰。

Optuna 4.9 已弃用 `terminator` 模块，普通调参不会再自动调用它来记录专用 CV 数据；
各折指标、中间值、中文指标名称和常规可视化仍然保留。只有需要旧版终止改进诊断时，
才显式使用 `record_terminator_scores=True`，并接受上游对该功能的弃用提示。
指标名称仍通过公开的 `set_metric_names` 接口设置；包装器只在这一调用内接管其自身的实验性提示，
不会屏蔽用户函数、训练异常或其它库的警告。

## 18. 本次整理后的使用约定

- 指标方向从内置名称或对象声明推断；普通函数须明确方向，输入或指标错误不再用零分代表成功。
- 自定义评估使用 `make_metric`，需要完整类型封装时使用 `BaseMetric`；自定义训练使用 `BaseLoss`。
- `loss=` 明确区分训练目标与历史 `objective=` 评分入口；收到混淆用法会说明正确参数。
- 外层 CV 验证数据用于搜索评分，内部监控只来自训练折；训练权重与评价权重独立指定。
- 保留原有空间声明和历史 `logloss` 负值语义；推荐新代码统一 `Integer/Real/Categorical` 与 `log_loss`。
- Notebook 使用少量 trial 验证接口与产物，不声称此参数组合已达到最佳业务效果。

API 逐项说明见 [超参数调优 API](../api/tuning.rst)。模型保存、原生训练参数、制品版本限制详见 [模型工作流](model-workflow.md)。
