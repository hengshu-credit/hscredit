# hscredit风控模型框架设计文档

## 概述

hscredit风控模型框架提供统一的风控建模接口，支持多种机器学习模型，并集成风控专用的评估指标、特征分析和超参数调优功能。

## 架构设计

### 1. 核心组件

```
core/models/
├── base.py              # BaseRiskModel - 模型基类
├── report.py            # ModelReport - 模型评估报告
├── tuning.py            # ModelTuner/AutoTuner - 超参数调优
├── xgboost_model.py     # XGBoostRiskModel
├── lightgbm_model.py    # LightGBMRiskModel
├── catboost_model.py    # CatBoostRiskModel
├── sklearn_models.py    # RandomForest/ExtraTrees/GradientBoosting
├── logistic_regression.py  # LogisticRegression
├── scorecard.py         # ScoreCard
└── rule_classifier.py   # RulesClassifier
```

### 2. 统一接口设计

所有风控模型继承 `BaseRiskModel`，提供统一API：

```python
# 统一接口
model.fit(X, y, sample_weight=None, eval_set=None)
model.predict(X) -> np.ndarray
model.predict_proba(X) -> np.ndarray
model.predict_score(X) -> np.ndarray  # 0-1000评分
model.evaluate(X, y) -> Dict[str, float]
model.get_feature_importances() -> pd.Series
model.get_model_info() -> Dict[str, Any]
model.generate_report(X_train, y_train, X_test, y_test) -> ModelReport
```

### 3. 支持的模型

| 模型 | 类名 | 特点 |
|------|------|------|
| XGBoost | XGBoostRiskModel | 高效梯度提升，支持自定义损失 |
| LightGBM | LightGBMRiskModel | 训练速度快，内存占用少 |
| CatBoost | CatBoostRiskModel | 对类别特征友好 |
| RandomForest | RandomForestRiskModel | 稳定，不易过拟合 |
| ExtraTrees | ExtraTreesRiskModel | 更随机，训练更快 |
| GradientBoosting | GradientBoostingRiskModel | sklearn原生实现 |
| LogisticRegression | LogisticRegression | 可解释性强，支持统计检验 |

### 4. 评估指标体系

内置风控专用评估指标：

- **KS**: Kolmogorov-Smirnov统计量，衡量区分能力
- **AUC**: ROC曲线下面积
- **Gini**: 基尼系数，Gini = 2*AUC - 1
- **Lift**: Lift值，衡量排序能力
- **PSI**: Population Stability Index，稳定性指标
- **标准指标**: Accuracy, Precision, Recall, F1, LogLoss

### 5. ModelReport报告系统

提供完整的模型评估报告：

```python
report = model.generate_report(X_train, y_train, X_test, y_test)

# 获取各项分析
metrics = report.get_metrics()
importance = report.get_feature_importance(top_n=10)
distribution = report.get_score_distribution(n_bins=10)
psi = report.get_psi()
roc_data = report.get_roc_curve()
lift_data = report.get_lift_curve(n_bins=10)

# 打印完整报告
report.print_report()
```

### 6. Optuna超参数调优

统一超参数调优接口：

```python
from hscredit.core.models import AutoTuner

# 创建调优器
tuner = AutoTuner.create(
    model_type='xgboost',  # 或 'lightgbm', 'randomforest'等
    metric='auc',
    direction='maximize'
)

# 执行调优
best_params = tuner.fit(X_train, y_train, n_trials=100)

# 获取最佳模型
best_model = tuner.get_best_model()
```

预定义的搜索空间：
- XGBoost: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda
- LightGBM: num_leaves, max_depth, learning_rate, n_estimators, min_child_samples, subsample
- RandomForest: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

### 7. 自定义损失函数

支持风控业务场景的损失函数：

```python
from hscredit.core.models import XGBoostRiskModel
from hscredit.core.models.losses import FocalLoss, CostSensitiveLoss

# 使用Focal Loss处理不平衡数据
model = XGBoostRiskModel(
    objective=FocalLoss(alpha=0.25, gamma=2.0),
    eval_metric='auc'
)
```

### 8. scorecardpipeline风格集成

与hscredit分箱、WOE编码无缝集成：

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.models import LogisticRegression, ScoreCard

# 分箱
binner = OptimalBinning(method='chi', max_n_bins=5)
binner.fit(X_train, y_train)

# WOE转换
X_train_woe = binner.transform(X_train, metric='woe')

# 训练LR
lr = LogisticRegression(C=1.0, calculate_stats=True)
lr.fit(X_train_woe, y_train)

# 转换为评分卡
scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750, 
                      lr_model=lr, combiner=binner)
```

## 使用示例

### 快速开始

```python
from hscredit.core.models import XGBoostRiskModel

# 创建模型
model = XGBoostRiskModel(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    eval_metric=['auc', 'ks']
)

# 训练
model.fit(X_train, y_train)

# 评估
metrics = model.evaluate(X_test, y_test)
print(f"AUC: {metrics['AUC']:.4f}, KS: {metrics['KS']:.4f}")

# 生成报告
report = model.generate_report(X_train, y_train, X_test, y_test)
report.print_report()
```

### 多模型对比

```python
models = {
    'XGBoost': XGBoostRiskModel(max_depth=5),
    'LightGBM': LightGBMRiskModel(num_leaves=31),
    'RandomForest': RandomForestRiskModel(n_estimators=100),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(f"{name}: AUC={metrics['AUC']:.4f}, KS={metrics['KS']:.4f}")
```

## 扩展指南

### 添加新模型

1. 继承 `BaseRiskModel`
2. 实现 `fit`, `predict`, `predict_proba`, `get_feature_importances` 方法
3. 在 `__init__.py` 中导出

```python
from .base import BaseRiskModel

class MyCustomModel(BaseRiskModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义参数
    
    def fit(self, X, y, sample_weight=None, eval_set=None, **fit_params):
        # 训练逻辑
        return self
    
    def predict(self, X):
        # 预测逻辑
        pass
    
    def predict_proba(self, X):
        # 概率预测
        pass
    
    def get_feature_importances(self, importance_type='gain'):
        # 特征重要性
        pass
```

### 自定义评估指标

```python
def custom_metric(y_true, y_proba):
    # 自定义指标计算
    return score

# 在evaluate中使用
metrics = model.evaluate(X_test, y_test, metrics=['auc', 'ks', custom_metric])
```

## 依赖要求

- numpy >= 1.20
- pandas >= 1.3
- scikit-learn >= 1.0
- xgboost >= 1.5 (可选)
- lightgbm >= 3.3 (可选)
- catboost >= 1.0 (可选)
- optuna >= 3.0 (可选，用于超参数调优)

## 性能建议

1. **大数据集**: 优先使用 LightGBM 或 XGBoost 的 histogram 模式
2. **类别特征**: 使用 CatBoost 或对类别特征进行编码
3. **不平衡数据**: 使用 `scale_pos_weight` 或 Focal Loss
4. **可解释性**: 使用 LogisticRegression 或 RandomForest
5. **超参数调优**: 使用 AutoTuner 的预定义搜索空间作为起点

## 注意事项

1. 所有模型都支持 numpy 数组和 pandas DataFrame 输入
2. 预测概率范围 [0, 1]，风险评分范围 [0, 1000]
3. PSI < 0.1 表示模型稳定，0.1-0.25 表示略有变化，> 0.25 表示不稳定
4. 建议始终使用验证集评估模型性能
