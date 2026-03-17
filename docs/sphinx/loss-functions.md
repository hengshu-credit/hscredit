# 自定义损失函数模块文档

## 概述

hscredit自定义损失函数模块为金融风控场景提供了丰富的损失函数和评估指标，支持主流机器学习框架：

- **XGBoost**
- **LightGBM**
- **CatBoost**
- **TabNet** (需要PyTorch)

## 核心特性

### 1. 不平衡数据处理
- **FocalLoss**: 通过降低易分类样本权重，专注于难分类样本
- **WeightedBCELoss**: 为正负样本分配不同权重
- **CostSensitiveLoss**: 根据错误成本分配权重

### 2. 风控业务损失
- **BadDebtLoss**: 最小化坏账率，同时保持通过率
- **ApprovalRateLoss**: 在保证坏账率的前提下最大化通过率
- **ProfitMaxLoss**: 最大化总利润（利息收益 - 坏账损失）

### 3. 自定义评估指标
- **KSMetric**: KS值（Kolmogorov-Smirnov）
- **GiniMetric**: Gini系数
- **PSIMetric**: PSI（Population Stability Index）

### 4. 框架适配器
- **XGBoostLossAdapter**: XGBoost适配器
- **LightGBMLossAdapter**: LightGBM适配器
- **CatBoostLossAdapter**: CatBoost适配器
- **TabNetLossAdapter**: TabNet适配器

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install numpy pandas scikit-learn

# 根据需要安装模型框架
pip install xgboost        # XGBoost
pip install lightgbm       # LightGBM
pip install catboost       # CatBoost
pip install pytorch-tabnet # TabNet（还需要PyTorch）
```

### 基础用法

```python
from hscredit.core.models import FocalLoss, KSMetric, XGBoostLossAdapter
import xgboost as xgb

# 创建损失函数
loss = FocalLoss(alpha=0.75, gamma=2.0)
adapter = XGBoostLossAdapter(loss)

# 训练模型
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
bst = xgb.train(
    params,
    dtrain,
    obj=adapter.objective(),  # 使用自定义损失
    num_boost_round=100
)

# 使用自定义评估指标
ks_metric = KSMetric()
bst = xgb.train(
    params,
    dtrain,
    custom_metric=adapter.metric(ks_metric),
    num_boost_round=100
)
```

## 详细使用指南

### 1. Focal Loss - 处理不平衡数据

Focal Loss通过调整样本权重来解决类别不平衡问题，特别适合金融风控场景中坏账率很低的情况。

```python
from hscredit.core.models import FocalLoss, LightGBMLossAdapter
import lightgbm as lgb

# 创建Focal Loss
# alpha: 正样本权重，通常设置为正样本占比的反比
# gamma: 聚焦参数，越大易分类样本权重越小
loss = FocalLoss(alpha=0.75, gamma=2.0)
adapter = LightGBMLossAdapter(loss)

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'max_depth': 6
}

bst = lgb.train(
    params,
    train_data,
    fobj=adapter.objective(),
    num_boost_round=100
)
```

**参数说明:**
- `alpha`: 正样本权重，范围[0, 1]
  - 如果正样本占比10%，建议设置alpha=0.75（即1-0.25，让正负样本总体权重平衡）
- `gamma`: 聚焦参数，通常取值[0, 5]
  - gamma=0: 等价于标准交叉熵
  - gamma=2: 推荐值，适合大多数场景
  - gamma越大，模型越专注于难分类样本

### 2. 成本敏感损失

根据业务成本为不同类型的错误分配不同权重。

```python
from hscredit.core.models import CostSensitiveLoss, CatBoostLossAdapter
from catboost import CatBoostClassifier

# 假设漏抓一个坏客户损失10000元，误拒一个好客户损失100元
# 成本比例为100:1
loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
adapter = CatBoostLossAdapter(loss)

model = CatBoostClassifier(
    iterations=1000,
    loss_function=adapter.objective(),
    eval_metric='AUC',
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

**成本矩阵:**

|            | 预测负 | 预测正 |
|------------|--------|--------|
| **实际负** | 0      | fp_cost |
| **实际正** | fn_cost| 0      |

- `fn_cost`: 假阴性成本（漏抓坏客户）
- `fp_cost`: 假阳性成本（误拒好客户）

### 3. 坏账率优化

在信贷审批场景中，优化坏账率同时保持合理通过率。

```python
from hscredit.core.models import BadDebtLoss

# 目标通过率30%，重点优化坏账率
loss = BadDebtLoss(
    target_approval_rate=0.3,  # 目标通过率30%
    bad_debt_weight=1.0,       # 坏账率权重
    approval_weight=0.5        # 通过率偏离惩罚权重
)
```

**参数说明:**
- `target_approval_rate`: 目标通过率
- `bad_debt_weight`: 坏账率损失权重，越大越重视降低坏账率
- `approval_weight`: 通过率偏离惩罚权重，越大越倾向于保持目标通过率

### 4. 利润最大化

综合考虑利息收益和坏账损失，最大化总利润。

```python
from hscredit.core.models import ProfitMaxLoss

# 假设每笔贷款利息收益100元，坏账损失1000元
loss = ProfitMaxLoss(
    interest_income=100,   # 单位利息收益
    bad_debt_loss=1000     # 单位坏账损失
)
```

**利润模型:**
```
总利润 = 通过好客户数 × 利息收益 - 通过坏客户数 × 坏账损失
```

### 5. 自定义评估指标

#### KS值

```python
from hscredit.core.models import KSMetric, LightGBMLossAdapter

ks_metric = KSMetric()
adapter = LightGBMLossAdapter(loss)

bst = lgb.train(
    params,
    train_data,
    feval=adapter.metric(ks_metric),  # 使用KS作为评估指标
    num_boost_round=100
)
```

**KS值解释:**
- KS < 0.2: 模型区分能力较弱
- 0.2 ≤ KS < 0.4: 模型区分能力一般
- 0.4 ≤ KS < 0.6: 模型区分能力较强
- KS ≥ 0.6: 模型区分能力很强

#### Gini系数

```python
from hscredit.core.models import GiniMetric

gini_metric = GiniMetric()
gini_value = gini_metric(y_true, y_pred)
```

**关系:** Gini = 2 × AUC - 1

#### PSI监控

```python
from hscredit.core.models import PSIMetric

# 使用训练集作为基准
psi_metric = PSIMetric(expected=y_train_pred, n_bins=10)

# 计算测试集PSI
psi_value = psi_metric(y_test, y_test_pred)

# PSI解读
if psi_value < 0.1:
    print("分布稳定")
elif psi_value < 0.25:
    print("分布有轻微变化")
else:
    print("分布变化显著，需要关注")
```

**PSI解释:**
- PSI < 0.1: 分布稳定
- 0.1 ≤ PSI < 0.25: 分布有轻微变化
- PSI ≥ 0.25: 分布变化显著

### 6. TabNet使用示例

```python
from hscredit.core.models import FocalLoss, TabNetLossAdapter
from pytorch_tabnet.tab_model import TabNetClassifier

loss = FocalLoss(alpha=0.75, gamma=2.0)
adapter = TabNetLossAdapter(loss)

model = TabNetClassifier(
    n_d=8, n_a=8,
    n_steps=3,
    gamma=1.3,
    seed=42
)

model.fit(
    X_train.values, y_train.values,
    eval_set=[(X_test.values, y_test.values)],
    eval_metric=['auc'],
    loss_fn=adapter.loss_fn(),
    max_epochs=100,
    patience=10,
    batch_size=1024
)
```

## 最佳实践

### 1. 损失函数选择指南

| 场景 | 推荐损失函数 | 参数建议 |
|------|--------------|----------|
| 不平衡数据（坏账率<5%） | FocalLoss | alpha=0.75-0.9, gamma=2.0 |
| 成本敏感场景 | CostSensitiveLoss | 根据实际成本设置fn_cost和fp_cost |
| 坏账率优化 | BadDebtLoss | 根据业务目标设置通过率 |
| 利润最大化 | ProfitMaxLoss | 根据业务模型设置收益和损失 |
| 自动平衡权重 | WeightedBCELoss | auto_balance=True |

### 2. 参数调优建议

#### Focal Loss
- **alpha**: 通常设置为 1 - 正样本占比/2
  - 正样本占比10%: alpha=0.75
  - 正样本占比5%: alpha=0.85
  
- **gamma**: 从2.0开始尝试
  - gamma=1.0: 适合轻度不平衡
  - gamma=2.0: 适合中度不平衡（推荐）
  - gamma=3.0-5.0: 适合极度不平衡

#### Cost Sensitive Loss
- 先量化业务成本，然后按比例设置
- 例如: 漏抓坏客户损失10000元，误拒好客户损失100元
  - fn_cost=100, fp_cost=1

#### Bad Debt Loss
- 根据业务策略设置target_approval_rate
- 通过调整bad_debt_weight和approval_weight平衡两个目标

### 3. 性能优化

#### 并行计算
```python
# XGBoost支持多线程
params = {
    'objective': 'binary:logistic',
    'nthread': 8,  # 使用8个线程
    'tree_method': 'hist'  # 使用histogram算法加速
}
```

#### 早停策略
```python
# LightGBM早停
bst = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)]  # 50轮无改善则停止
)

# XGBoost早停
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dvalid, 'valid')],
    early_stopping_rounds=50
)
```

### 4. 模型评估

使用多个指标综合评估:

```python
from hscredit.core.models import KSMetric, GiniMetric, PSIMetric

# 计算多个指标
ks_metric = KSMetric()
gini_metric = GiniMetric()
psi_metric = PSIMetric(expected=y_train_pred)

metrics = {
    'KS': ks_metric(y_test, y_pred),
    'Gini': gini_metric(y_test, y_pred),
    'PSI': psi_metric(y_test, y_pred)
}

print("模型评估指标:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

## 常见问题

### Q1: 为什么需要自定义损失函数？

**A:** 在金融风控场景中:
1. 标准损失函数没有考虑类别不平衡
2. 没有考虑不同错误的业务成本
3. 无法直接优化业务指标（如坏账率、利润）

### Q2: 如何选择合适的损失函数？

**A:** 根据业务场景:
- **坏账率很低（<5%）**: 使用FocalLoss
- **不同错误成本差异大**: 使用CostSensitiveLoss
- **有明确业务目标**: 使用BadDebtLoss或ProfitMaxLoss
- **不确定**: 先用WeightedBCELoss(auto_balance=True)

### Q3: 损失函数和评估指标的区别？

**A:**
- **损失函数**: 用于训练时优化模型参数，需要有梯度
- **评估指标**: 用于评估模型性能，不需要梯度
- 可以同时使用自定义损失和自定义指标

### Q4: 如何验证自定义损失函数是否有效？

**A:**
1. 与标准损失函数对比KS、AUC等指标
2. 观察学习曲线是否正常收敛
3. 在验证集上评估业务指标（坏账率、通过率等）

## 注意事项

1. **概率转换**: 框架适配器会自动处理概率转换，无需手动sigmoid
2. **数值稳定性**: 所有损失函数都做了数值稳定性处理
3. **二阶导数**: 某些框架（如XGBoost）需要二阶导数，如果未实现会使用近似值
4. **TabNet依赖**: TabNet适配器需要PyTorch环境
5. **内存占用**: 大数据集时注意内存占用，可以分批处理

## 参考资料

1. **Focal Loss**: Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
2. **Cost-Sensitive Learning**: Elkan, C. "The foundations of cost-sensitive learning." IJCAI 2001.
3. **KS Statistic**: 有人于1933年提出，用于检验两个分布是否有显著差异
4. **PSI**: 用于模型监控的标准指标

## 更新日志

- **v0.1.0** (2026-03-15)
  - 初始版本
  - 支持XGBoost、LightGBM、CatBoost、TabNet
  - 实现6种损失函数和3种评估指标
