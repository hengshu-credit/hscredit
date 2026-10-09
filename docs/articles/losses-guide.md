# 损失函数：训练、评估与调参

`hscredit.core.models.losses` 提供 16 种二分类损失。每一种损失都可以用同一个对象生成训练目标、配套评估指标和 sklearn 评分器。

```python
from hscredit import LightGBM
from hscredit.core.models.losses import FocalLoss

loss = FocalLoss(alpha=0.75, gamma=2)
model = LightGBM(objective=loss, n_estimators=100, n_jobs=1)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
value = loss.evaluate(y_valid, model.predict_proba(X_valid))
```

`X_train` 等数据的准备方式见下一节。`LightGBM` / `XGBoost` 在 `objective=loss` 且未指定 `eval_metric` 时自动使用 `loss.metric()`；CatBoost 保留默认 `AUC`，可显式传入 `eval_metric=loss.metric()`。

## 1. 先区分三个用途

| 入口 | 含义 | 优劣方向 |
| --- | --- | --- |
| `loss` / `objective=loss` | 训练时提供梯度和二阶导数 | 优化损失 |
| `loss.metric()` | 与该损失相同参数、相同数值定义的评估指标 | 始终越小越好 |
| `loss.business_metric()` | 对应真实业务口径，例如 AUC、捕获率、利润；没有专用口径时返回损失指标 | 由 `greater_is_better` / `direction` 声明 |
| `metric.to_scorer()` | sklearn CV / GridSearchCV 评分器 | sklearn 要求越大越好，因此最小化指标自动取负 |

`loss.metric()` 返回 `LossMetric`，它保存损失参数的独立快照。指标评估不会修改训练损失；如果随后修改了损失参数，应重新生成指标。`loss.to_metric()` 与 `loss.metric()` 等价，`loss.to_scorer()` 与 `loss.metric().to_scorer()` 等价。

同一个指标可以用于训练曲线、早停、测试集评估和超参数搜索；比较模型时必须保持指标参数、样本范围、样本权重一致。不同 `alpha`、成本参数或金额口径的损失值不能直接横向比较。即使某种自定义损失不用于训练，仍可用它评估任意提供坏样本概率的二分类模型。

全模块约定：`y=0` 是好样本，`y=1` 是坏样本，概率越大风险越高；低风险客户才通过审批。这里的概率不是“越高越好”的信用评分。

## 2. 准备训练集、验证集和测试集

以下代码从项目根目录运行，后续样例复用这些变量。验证集用于早停与选参数，测试集留作最后一次效果评估。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
data = pd.read_excel("examples/hscredit_yyp.xlsx")
data = data.dropna(subset=features + ["FPD"]).copy()
X = data[features].astype(float)
y = data["FPD"].to_numpy()

train_idx, other_idx = train_test_split(
    np.arange(len(data)), test_size=0.4, stratify=y, random_state=42,
)
valid_idx, test_idx = train_test_split(
    other_idx, test_size=0.5, stratify=y[other_idx], random_state=42,
)
X_train, y_train = X.iloc[train_idx], y[train_idx]
X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
X_test, y_test = X.iloc[test_idx], y[test_idx]
```

## 3. 每种损失对应什么指标

所有行均支持 `loss.metric()`，返回与训练损失同定义的标量，方向均为 `minimize`。下面的业务指标通过 `loss.business_metric()` 获取，自动继承该损失的相关参数。

| 损失 | 主要用途 | `business_metric()` | 业务方向 |
| --- | --- | --- | --- |
| `FocalLoss` | 降低容易样本的权重、关注难分样本 | 聚焦损失本身 | 越小越好 |
| `AsymmetricFocalLoss` | 好坏样本分别设置聚焦强度 | 非对称聚焦损失本身 | 越小越好 |
| `BalancedFocalLoss` | 有效样本数平衡、标签平滑 | 平衡聚焦损失本身 | 越小越好 |
| `WeightedBCELoss` | 好坏样本使用固定类别权重 | 加权交叉熵本身 | 越小越好 |
| `CostSensitiveLoss` | 漏抓与误拒成本不同 | `ClassificationCostMetric`，固定阈值实际误判成本 | 越小越好 |
| `BadDebtLoss` | 固定人数通过率下控制坏账 | `BadDebtMetric`，通过客户观察坏账率 | 越小越好 |
| `ApprovalRateLoss` | 观察坏账约束下争取通过率 | `ApprovalRateMetric`，回看最大通过率 | 越大越好 |
| `ProfitMaxLoss` | 根据好客户收益和坏账损失加权 BCE | `ProfitMetric`，实际人均审批利润 | 越大越好 |
| `ExpectedProfitLoss` | 软审批概率、收益、损失与 BCE 联合优化 | `ProfitMetric`，固定审批阈值实际利润 | 越大越好 |
| `OrdinalRankLoss` | 好坏样本配对排序 | `AUCMetric` | 越大越好 |
| `RankingAUCProxyLoss` | 排序间隔与难配对约束 | `AUCMetric` | 越大越好 |
| `LiftFocusedLoss` | 强调高风险头部名单 | `TopKLiftMetric`，头部坏率 / 整体坏率 | 越大越好 |
| `KSFocusedLoss` | 分离好坏样本预测分布 | `KSMetric` | 越大越好 |
| `TopKBadCaptureLoss` | 高风险头部捕获坏客户 | `TopKCaptureMetric`，头部坏样本 / 全体坏样本 | 越大越好 |
| `AmountWeightedLoss` | 按放款金额加权 BCE | 金额加权损失本身 | 越小越好 |
| `ExpectedValueLoss` | 按 EAD、LGD、利率和成本构造权重 | 期望价值损失本身 | 越小越好 |

例如选择实际头部捕获率评估排序效果：

```python
from hscredit.core.models.losses import TopKBadCaptureLoss

capture_loss = TopKBadCaptureLoss(top_ratio=0.05, miss_penalty=5)
capture_metric = capture_loss.business_metric()
print(capture_metric.name, capture_metric.direction)
# 头部坏样本捕获率 maximize
```

业务口径有以下明确约定：

- `AUCMetric` 保留风险方向，反向预测会低于 0.5；不会自动翻转分数。有效权重下缺少好或坏任一类别时无法计算。
- 头部捕获、Lift 和固定通过率坏账指标先按人数选取 `ceil(n * ratio)` 人。边界同分者等比例计入，避免结果受行顺序影响；统计时使用样本权重。没有有效坏样本时，捕获率和 Lift 无定义并报错。
- `ApprovalRateMetric` 按同分组搜索所有可行风险阈值，累计通过率与坏账率均使用权重。这是在已知标签数据上的回看结果，不是未来坏账率保证；部署阈值应在验证集确定，在独立测试集评估。
- `ProfitMetric` 对 `p < cutoff` 的好客户计收益、坏客户计负损失，拒绝客户计零，分母是全量申请客户权重。它与训练用的利润代理损失不是同一个数。
- `ClassificationCostMetric` 对 `p >= threshold` 判为坏客户，默认阈值为 0.5。可以独立创建不同阈值的业务指标。

## 4. HSCredit 模型：最少配置的用法

### LightGBM：自动配套指标与早停

```python
from hscredit import LightGBM
from hscredit.core.models.losses import FocalLoss

loss = FocalLoss(alpha=0.75, gamma=2)
model = LightGBM(
    objective=loss,
    n_estimators=300,
    learning_rate=0.05,
    early_stopping_rounds=20,
    n_jobs=1,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
p_test = model.predict_proba(X_test)[:, 1]
print(loss.evaluate(y_test, p_test))
print(model.evals_result_)
```

同时监控多个指标，并指定早停口径：

```python
metric = loss.metric()
model = LightGBM(
    objective=loss,
    eval_metric=["auc", metric],
    early_stopping_metric=metric.name,
    early_stopping_rounds=20,
    n_estimators=300,
    n_jobs=1,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
```

### XGBoost：自动处理原始分数与评估方向

```python
from hscredit import XGBoost
from hscredit.core.models.losses import ExpectedProfitLoss

profit_loss = ExpectedProfitLoss(revenue=1, default_cost=8, cutoff=0.2)
profit_metric = profit_loss.business_metric()
model = XGBoost(
    objective=profit_loss,
    eval_metric=profit_metric,
    n_estimators=300,
    early_stopping_rounds=20,
    n_jobs=1,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
print(profit_metric.evaluate(y_test, model.predict_proba(X_test)))
```

这里训练优化可微的利润代理，早停选择实际利润最高的轮次。HSCredit 自动配置指标方向；也可将 `eval_metric` 改成 `profit_loss.metric()`，按训练损失选择轮次。

XGBoost 的 `eval_metric` 支持单个指标对象，或原有的内置指标名称/名称列表；不能混合多个指标对象。训练后可一次计算多个离线指标。

### CatBoost：显式选择配套指标

```python
from hscredit import CatBoost
from hscredit.core.models.losses import WeightedBCELoss

loss = WeightedBCELoss(pos_weight=3, neg_weight=1)
model = CatBoost(
    objective=loss,
    eval_metric=loss.metric(),
    iterations=300,
    early_stopping_rounds=20,
    n_jobs=1,
    allow_writing_files=False,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
```

CatBoost 未指定 `eval_metric` 时仍使用原有默认 `AUC`；显式 `eval_metric=None` 时才自动配套损失指标。CatBoost 只支持一个自定义主评估指标；额外的监控项使用其内置指标名称。可用损失限制见第 10 节。

### 只更换评价口径，保留模型内置训练目标

```python
metric = WeightedBCELoss(pos_weight=3).metric()
model = LightGBM(eval_metric=metric, n_estimators=100, n_jobs=1)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
```

此时模型仍训练原有内置目标，配套指标仅用于监控或早停。不要把“换了评估指标”理解为“换了训练损失”。

## 5. 独立评估、概率与原始分数

```python
from hscredit.core.models.losses import AUCMetric, KSMetric, ClassificationCostMetric

p_test = model.predict_proba(X_test)
metric = WeightedBCELoss(pos_weight=3).metric()
print(metric.evaluate(y_test, p_test))       # 支持两列 [P(好), P(坏)]
print(metric(y_test, p_test[:, 1]))         # 支持一维坏样本概率
print(model.evaluate(
    X_test, y_test,
    metrics=["auc", metric, ClassificationCostMetric(fn_cost=8, fp_cost=1)],
))

comparison = pd.DataFrame([
    {"指标": item.name, "数值": item.evaluate(y_test, p_test), "优化方向": item.direction}
    for item in [metric, AUCMetric(), KSMetric()]
])
print(comparison)
```

原始分数必须显式声明，不能根据数值范围猜测是否已经过 sigmoid：

```python
from scipy.special import logit

margin = logit(np.clip(p_test[:, 1], 1e-7, 1 - 1e-7))
value = metric.evaluate(y_test, margin, raw_score=True)
```

概率必须有限且落在 `[0, 1]`，标签必须是 `0/1`。`evaluate` 也检查权重、数组长度和有限性。两个概率列须按 `[0, 1]` 顺序且每行和为 1；sklearn 评分器会通过模型 `classes_` 找到坏样本概率列。

## 6. 直接使用原生训练框架

原生框架的目标回调与评估回调不是同一个签名。明确选择 `api`，可以避免记忆标签与预测值的传入顺序。

| 框架入口 | 目标函数 | 评估函数 |
| --- | --- | --- |
| `xgb.train` | `loss.to_xgboost(api="native")` | `metric.to_xgboost(api="native", raw_score=True)` |
| `xgb.XGBClassifier` | `loss.to_xgboost(api="sklearn")` | `metric.to_xgboost(api="sklearn", raw_score=True)` |
| `lgb.train` | `loss.to_lightgbm(api="native")` | `metric.to_lightgbm(api="native", raw_score=True)` |
| `lgb.LGBMClassifier` | `loss.to_lightgbm(api="sklearn")` | `metric.to_lightgbm(api="sklearn", raw_score=True)` |
| `cb.CatBoostClassifier` | `loss.to_catboost()` | `metric.to_catboost()` |

表中 `raw_score=True` 适用于同时使用自定义目标的情况。使用框架内置二分类概率目标时，XGBoost / LightGBM 的指标回调应使用默认 `raw_score=False`。CatBoost 的指标适配器始终按其原始分数协议转换。

### XGBoost 原生训练与 sklearn 接口

```python
import xgboost as xgb
from scipy.special import expit
from hscredit.core.models.losses import FocalLoss

loss = FocalLoss(alpha=0.75)
metric = loss.metric()
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
booster = xgb.train(
    {"objective": "binary:logistic", "max_depth": 3, "nthread": 1,
     "disable_default_eval_metric": 1},
    dtrain,
    num_boost_round=300,
    obj=loss.to_xgboost(),
    custom_metric=metric.to_xgboost(raw_score=True),
    evals=[(dvalid, "验证集")],
    early_stopping_rounds=20,
    maximize=metric.greater_is_better,
    verbose_eval=False,
)
p_valid = expit(booster.predict(
    dvalid, output_margin=True, iteration_range=(0, booster.best_iteration + 1),
))

native_model = xgb.XGBClassifier(
    objective=loss.to_xgboost(api="sklearn"),
    eval_metric=metric.to_xgboost(api="sklearn", raw_score=True),
    n_estimators=300,
    n_jobs=1,
    callbacks=[xgb.callback.EarlyStopping(
        rounds=20, metric_name=metric.name, maximize=metric.greater_is_better,
        save_best=True,
    )],
)
native_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
```

示例使用支持 `custom_metric` 的 XGBoost API。原生 sklearn 自定义目标是否接收样本权重取决于 XGBoost 版本；需要加权训练时，HSCredit 包装器提供兼容处理，也可以使用带 `weight` 的 `DMatrix` 与 `xgb.train`。

### LightGBM 原生训练与 sklearn 接口

```python
import lightgbm as lgb

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
booster = lgb.train(
    {"objective": loss.to_lightgbm(api="native"), "metric": "None",
     "verbosity": -1, "num_threads": 1},
    train_set,
    num_boost_round=300,
    valid_sets=[valid_set],
    feval=metric.to_lightgbm(api="native", raw_score=True),
    callbacks=[lgb.early_stopping(20, verbose=False)],
)
p_valid = expit(booster.predict(X_valid, num_iteration=booster.best_iteration))

native_model = lgb.LGBMClassifier(
    objective=loss.to_lightgbm(), metric="None", n_estimators=300,
    n_jobs=1, verbosity=-1,
)
native_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=metric.to_lightgbm(raw_score=True),
    callbacks=[lgb.early_stopping(20, verbose=False)],
)
# 原生 LGBMClassifier 的自定义目标预测可能返回一维原始分数。
p_valid = expit(native_model.predict(X_valid, raw_score=True))
```

原生 `lgb.train` 示例采用 LightGBM 4.x 的 `params["objective"]` 自定义目标接口。HSCredit 包装器的 `predict_proba` 会自动返回两列概率。

### CatBoost 原生接口

```python
import catboost as cb

native_model = cb.CatBoostClassifier(
    loss_function=loss.to_catboost(),
    eval_metric=metric.to_catboost(),
    iterations=300, thread_count=1, verbose=False,
    early_stopping_rounds=20, allow_writing_files=False,
)
native_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
p_valid = native_model.predict_proba(X_valid)[:, 1]
```

CatBoost 的指标对象自己声明最大化/最小化方向，无须再手动设置 `maximize`。

## 7. CV、GridSearchCV、ModelTuner 与 Optuna

### 评估任意 sklearn 二分类器

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

metric = WeightedBCELoss(pos_weight=3).metric()
estimator = LogisticRegression(max_iter=1000)
scores = cross_val_score(estimator, X_train, y_train, cv=3, scoring=metric.to_scorer())
print("每折损失：", -scores)

search = GridSearchCV(
    estimator, {"C": [0.1, 1, 10]},
    scoring={"损失": metric.to_scorer(), "AUC": "roc_auc"},
    refit="损失", cv=3, n_jobs=1,
)
search.fit(X_train, y_train)
print("最优交叉验证损失：", -search.best_score_)
```

`to_scorer()` 自动取负仅是 sklearn 的比较约定。训练日志、`metric.evaluate` 和 ModelTuner 中的指标仍是原始数值。对 `ProfitMetric`、AUC 等越大越好的指标，`to_scorer()` 保留原值。

### ModelTuner 自动继承指标名称和方向

```python
from hscredit import LightGBM, ModelTuner

loss = WeightedBCELoss(pos_weight=3)
tuner = ModelTuner(
    LightGBM,
    search_space={"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.05]},
    fixed_params={"n_estimators": 100, "n_jobs": 1},
    loss=loss, metric=None,  # 自动使用 loss.metric()，方向为 minimize
    cv=3, n_jobs=1,
)
tuner.fit(X_train, y_train, n_trials=10, show_progress_bar=False)
print(tuner.best_params_, tuner.best_score_)  # 原始损失，越小越好
```

传入 `BaseMetric` 或调参器自己的 `Metric` 对象时，ModelTuner 采用对象自带方向；不需要重复设置 `direction="minimize"`。普通自定义函数须显式指定方向，也可用下节的 `make_metric` 一次包装。多目标搜索可传 `metric=[loss.metric(), loss.business_metric()]`，每个对象分别提供方向；当两个入口返回同一种损失指标时没有必要重复添加。

`loss=` 是训练目标；`metric` 决定候选优劣；`ModelTuner(objective=...)` 是历史评分别名，不要将 loss 放入此参数。已有模型可直接 `best = model.tune(X_train, y_train, loss=loss, metric=loss.metric(), ...)`。搜索 `alpha`、`gamma` 等损失参数时，固定评估指标为 AUC、Brier 或同一业务口径，不能让各个 trial 使用不同损失标尺。详见 [调参指南](tuning-guide.md)。

### 原生 Optuna：直接使用 direction

```python
import optuna

metric = loss.metric()

def objective(trial):
    candidate = LightGBM(
        objective=loss,
        num_leaves=trial.suggest_int("num_leaves", 7, 31),
        n_estimators=100, n_jobs=1,
    )
    candidate.fit(X_train, y_train)
    return metric.evaluate(y_valid, candidate.predict_proba(X_valid))

study = optuna.create_study(direction=metric.direction)
study.optimize(objective, n_trials=10)
```

这里返回原始损失，不要额外取负。只有使用 sklearn scorer 的场景需要其统一的“越大越好”约定。

## 8. 样本权重、放款金额和样本参数对齐

### 固定类别权重与额外样本权重

```python
loss = WeightedBCELoss(pos_weight=3)
metric = loss.metric()
w_train = np.where(y_train == 1, 2.0, 1.0)
w_valid = np.where(y_valid == 1, 2.0, 1.0)

model = LightGBM(objective=loss, eval_metric=metric, n_estimators=100, n_jobs=1)
model.fit(
    X_train, y_train, sample_weight=w_train,
    eval_set=[(X_valid, y_valid)], eval_sample_weight=[w_valid],
)
value = metric.evaluate(y_valid, model.predict_proba(X_valid), sample_weight=w_valid)
```

`pos_weight` 与 `sample_weight` 会共同起作用，不要无意中对同一成本重复加权。权重必须非负、有限、非空且总和大于零。XGBoost 验证权重参数为 `sample_weight_eval_set`，LightGBM 为 `eval_sample_weight`，CatBoost 可使用带 `weight` 的验证 `Pool`。

普通可拆为逐样本贡献的损失按 `np.average(loss_values, weights=sample_weight)` 评估；`AmountWeightedLoss` / `ExpectedValueLoss` 则将内部金额或价值权重与外部 `sample_weight` 相乘后重新整体归一化。依赖全体样本排序或分布、未定义逐样本贡献的损失会在训练和评估时拒绝额外权重，不会静默忽略或把梯度加权伪装成严格的加权全局损失。

### 金额数组必须跟随数据切分

```python
from hscredit.core.models.losses import AmountWeightedLoss

amount = data["放款金额"].to_numpy(dtype=float)
a_train, a_valid, a_test = amount[train_idx], amount[valid_idx], amount[test_idx]
train_loss = AmountWeightedLoss(amounts=a_train)
valid_metric = train_loss.metric(amounts=a_valid)

model = LightGBM(
    objective=train_loss, eval_metric=valid_metric,
    n_estimators=100, validation_fraction=0, n_jobs=1,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
print(train_loss.evaluate(y_test, model.predict_proba(X_test), amounts=a_test))
```

参数覆盖创建独立快照，不会改动训练金额。`ExpectedValueLoss` 的 `ead`、`lgd`、`rate`、`cost` 数组也必须按训练/验证/测试的行位置分别提供，例如 `loss.metric(ead=valid_ead, lgd=valid_lgd)`。

内部数组按行位置对齐，长度相同但顺序不同仍会算错；DataFrame 索引不会自动重新对齐这些数组。绑定训练数组的损失不能直接复用到自动验证切分或其他长度的验证集；调用这类指标的 `to_scorer()` 会直接报中文错误，以防 sklearn CV 将全量数组错误用于每一折。请先手动切分、分别创建指标，并只为该指标对应的验证集记录曲线。

常规金额加权建模也可用 `WeightedBCELoss()` 加 `fit(sample_weight=金额权重)`，这样 CV 可以按折切分训练权重。评估权重仍需显式传递；不要假定 sklearn 自动把训练权重交给 scorer。若需要每折不同的业务数组，使用显式的折循环构造每折损失和指标。

## 9. 自定义指标、自定义损失与其他框架

### 只需要新评价口径：先写普通函数

评估指标不需要梯度。`make_metric` 让一个接收标签、坏样本概率并返回有限标量的函数，成为通用指标对象；名称、方向和固定业务参数只声明一次。

```python
from hscredit.core.models.losses import make_metric
from sklearn.metrics import brier_score_loss

metric = make_metric(
    brier_score_loss, name="均方概率误差", greater_is_better=False,
)
value = metric.evaluate(y_test, model.predict_proba(X_test))
scorer = metric.to_scorer()  # sklearn CV 自动用负值；离线评估仍是正误差

monitor_model = LightGBM(eval_metric=metric, n_estimators=100, n_jobs=1)
monitor_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
```

自写函数签名为 `(y_true, y_prob, sample_weight=None, **固定业务参数)`，`y_prob` 是一维坏样本概率。额外参数通过 `make_metric(..., 参数=值)` 绑定；`sample_weight` 是每次评估的行级输入，不应绑定成全量数组。

```python
def weighted_probability_error(y_true, y_prob, sample_weight=None, scale=1.0):
    error = (np.asarray(y_prob) - np.asarray(y_true)) ** 2
    return float(scale * np.average(error, weights=sample_weight))

metric = make_metric(
    weighted_probability_error, name="加权概率误差", greater_is_better=False, scale=1.0,
)
print(metric.evaluate(y_test, model.predict_proba(X_test)))
```

不支持权重的函数在收到 `sample_weight` 时明确报错；不要静默忽略权重。直接把 sklearn scorer 传给 `make_metric` 或 ModelTuner 不成立：scorer 接收 `(estimator, X, y)`，这里需要的是 `(y_true, y_prob)`。

### 需要可复用类：实现 BaseMetric

```python
from hscredit.core.models.losses import BaseMetric

class BrierMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="均方概率误差", greater_is_better=False)

    def __call__(self, y_true, y_pred, sample_weight=None):
        error = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
        return float(np.average(error, weights=sample_weight))

metric = BrierMetric()
print(metric.evaluate(y_test, model.predict_proba(X_test)))
```

基类 `evaluate()` 负责输入校验、两列概率提取与显式原始分数转换；`__call__()` 实现指标计算。`to_scorer`、`to_xgboost`、`to_lightgbm`、`to_catboost` 则生成不同调用场景所需的适配器。HSCredit 模型和 ModelTuner 可直接传指标对象。

### NGBoost 与 TabNet

NGBoost 的 `Score` 需要与对应分布配套，使用 `ngboost_params()` 一次提供：

```python
from ngboost import NGBClassifier

loss = WeightedBCELoss(pos_weight=3)
model = NGBClassifier(**loss.ngboost_params(), n_estimators=100, verbose=False)
model.fit(X_train, y_train)
print(loss.evaluate(y_test, model.predict_proba(X_test)))
```

TabNet 通过可反向传播的 PyTorch 适配器训练，完成后仍使用同一离线指标：

```python
from pytorch_tabnet.tab_model import TabNetClassifier
from hscredit.core.models.losses import TabNetLossAdapter

model = TabNetClassifier(verbose=0)
model.fit(
    X_train.to_numpy(dtype="float32"), y_train,
    loss_fn=TabNetLossAdapter(loss).loss_fn(),
    max_epochs=10, batch_size=256,
)
p_test = model.predict_proba(X_test.to_numpy(dtype="float32"))
print(loss.metric().evaluate(y_test, p_test))
```

### 需要改变训练目标：实现 BaseLoss

自定义损失继承 `BaseLoss`，实现概率上的损失、梯度、二阶导数即可自动获得配套 `LossMetric`。梯度与二阶导数采用 `n × 平均损失` 的标度；框架适配器负责 sigmoid 链式求导和训练需要的正对角曲率近似。如果确实逐样本可加，还应实现 `loss_values` 并设 `is_additive=True`，才能安全用于分批框架及加权评估。

```python
from hscredit.core.models.losses import BaseLoss

class BrierLoss(BaseLoss):
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

metric = BrierLoss().metric(name="均方概率误差")
print(metric.evaluate([0, 1], [0.1, 0.9]))
```

## 10. 数学限制与旧代码迁移

### 框架适用范围

XGBoost / LightGBM 的目标回调可以获得整批训练预测，支持全局排序或分布损失。此类损失通常含跨样本耦合项；树框架仍要求对角 Hessian，适配器使用正对角近似并裁剪非正曲率。因此它们是实验性优化目标，不能承诺等价于完整二阶优化或必然提升业务指标。

CatBoost 目标回调、NGBoost 和 TabNet 涉及分块或小批量运算，只接受 `is_additive=True` 的损失。固定参数的 Focal、AsymmetricFocal、WeightedBCE、CostSensitive、ProfitMax、ExpectedProfit 可以使用；动态类别平衡、排序、TopK、KS、固定通过率以及绑定样本数组的金额损失不应在这些框架中分批训练。`WeightedBCELoss(auto_balance=True)`、`BalancedFocalLoss(auto_alpha=True)` 依赖本批标签分布；可改用预先确定的固定类别权重。`ExpectedValueLoss` 依赖整体归一化，不可分批。

`BadDebtLoss` / `ApprovalRateLoss` 的业务部分使用硬排序，现有训练梯度来自 BCE 代理，业务项不会产生可用的排序梯度；调整业务参数会改变评估口径，不等于直接求得目标策略最优解。若主要目标是优化审批收益，可比较 `ExpectedProfitLoss`；若主要目标是提升排序，则比较排序代理损失，并使用真实业务指标验收。

TopK 分位点、Lift 头部边界和排序难配对选择都可能分段不可微。固定排序区域的导数验证不代表同分或排序切换点存在唯一导数。不同框架、随机种子和参数仍需以独立数据效果比较。

### 行为变化

| 旧用法或现象 | 当前用法或变化 |
| --- | --- |
| 不同适配器混用回调签名 | 显式 `api="native"` / `api="sklearn"`；优先使用 HSCredit 模型直接传对象 |
| 每次手写一个损失指标函数 | `loss.metric()` 统一生成训练、离线与调参指标 |
| 自定义目标下直接把原始分数当概率 | 原生 metric 回调声明 `raw_score=True`；HSCredit 自动判断 |
| CatBoost 自定义评估返回值不符合协议 | `metric.to_catboost()` 返回符合 `(error_sum, weight_sum)` 协议的对象 |
| `CostSensitiveLoss(y, p)` 返回硬分类成本 | 默认变为成本加权 BCE；实际成本用 `loss.business_metric()` / `loss.classification_cost(y, p, threshold=0.5)`；显式 `loss(y, p, threshold=0.5)` 保留旧口径 |
| `ProfitMaxLoss` 混合训练代理与实际利润 | 默认返回利润成本加权 BCE；实际利润用 `loss.business_metric()` / `loss.profit(y, p, threshold=0.5)` |
| 非对称聚焦的 `clip_value` 同时改变两类概率 | 现在仅对负类采用 `max(p-clip_value, 0)`；正类概率保持原定义 |
| BalancedFocal 标签平滑未实际作用 | 标签平滑参与损失、梯度和二阶导数 |
| KS 聚焦项仅改变梯度 | 聚焦权重现在同时进入标量损失，保证评价与导数一致 |
| `loss.to_ngboost()` 直接搭默认分布 | `NGBClassifier(**loss.ngboost_params())` 同时配套分布和 Score |

上述数学修复会改变部分历史训练结果和损失数值；旧模型与新模型应在同一份独立评估数据上重新计算统一指标，不能直接比较不同版本保存的训练曲线。低损失只说明某个代理目标更好，不能代替实际 AUC、捕获率、坏账率、利润以及校准表现。

## 11. 可运行样例与接口索引

仓库提供 [losses_workflow.ipynb](../../examples/losses_workflow.ipynb)，覆盖全部损失的独立评估、金额对齐、CV、框架训练、自定义 `BaseMetric` 和 `BaseLoss`。超参数搜索完整演示见 [tuning_workflow.ipynb](../../examples/tuning_workflow.ipynb) 和 [调参指南](tuning-guide.md)。完整 API 见 [损失函数与评估指标](../api/losses.rst)。

框架回调协议可查阅官方说明：[XGBoost 自定义目标和指标](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)、[LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/Python-API.html)、[CatBoost 用户自定义损失和指标](https://catboost.ai/docs/en/concepts/python-usages-examples#user-defined-loss-function)、[sklearn 自定义评分](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-callable)。
