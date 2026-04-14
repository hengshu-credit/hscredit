"""模型模块 (core/models).

提供风控建模相关的模型和工具，支持多种机器学习模型和统一接口。

**核心功能**

- 统一模型基类 (BaseRiskModel): 所有风控模型的抽象基类
- 多种风控模型: XGBoost、LightGBM、CatBoost、RandomForest、GradientBoosting
- 逻辑回归模型: 扩展sklearn，支持统计信息
- 评分卡模型: 将LR模型转换为评分卡
- 规则集分类模型: 基于规则集的分类器
- 超参数调优: 基于Optuna的自动调参
- 模型评估报告: 统一的模型性能评估

**支持的模型**

| 模型 | 类名 | 说明 |
|------|------|------|
| XGBoost | XGBoostRiskModel | 高效梯度提升树 |
| LightGBM | LightGBMRiskModel | 快速梯度提升树 |
| CatBoost | CatBoostRiskModel | 对类别特征友好的提升树 |
| RandomForest | RandomForestRiskModel | 随机森林 |
| ExtraTrees | ExtraTreesRiskModel | 极端随机树 |
| GradientBoosting | GradientBoostingRiskModel | 梯度提升树 |
| NGBoost | NGBoostRiskModel | 自然梯度提升（概率预测） |
| LogisticRegression | LogisticRegression | 扩展逻辑回归 |
| ScoreCard | ScoreCard | 评分卡模型 |

**快速开始**

**1. 基础模型训练**

>>> from hscredit.core.models import XGBoostRiskModel
>>> model = XGBoostRiskModel(
...     max_depth=5,
...     learning_rate=0.1,
...     n_estimators=100,
...     eval_metric=['auc', 'ks']
... )
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_test)

**2. 模型评估**

>>> metrics = model.evaluate(X_test, y_test)
>>> print(f"AUC: {metrics['AUC']:.4f}, KS: {metrics['KS']:.4f}")

**3. 生成完整报告**

>>> report = model.generate_report(X_train, y_train, X_test, y_test)
>>> report.print_report()

**4. 超参数调优**

>>> from hscredit.core.models import AutoTuner
>>> tuner = AutoTuner.create('xgboost', metric='auc')
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)
>>> best_model = tuner.get_best_model()

**5. 统一接口使用不同模型**

>>> from hscredit.core.models import (
...     XGBoostRiskModel,
...     LightGBMRiskModel,
...     CatBoostRiskModel
... )
>>>
>>> models = {
...     'xgboost': XGBoostRiskModel(max_depth=5),
...     'lightgbm': LightGBMRiskModel(num_leaves=31),
...     'catboost': CatBoostRiskModel(depth=6),
... }
>>>
>>> for name, model in models.items():
...     model.fit(X_train, y_train)
...     metrics = model.evaluate(X_test, y_test)
...     print(f"{name}: AUC={metrics['AUC']:.4f}")
"""

# 导入损失函数
from .losses import (
    # 基类
    BaseLoss,
    BaseMetric,
    # 不平衡数据处理
    FocalLoss,
    AsymmetricFocalLoss,
    WeightedBCELoss,
    # 成本敏感
    CostSensitiveLoss,
    # 风控业务损失
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    # 自定义评估指标
    KSMetric,
    GiniMetric,
    PSIMetric,
    # 框架适配器
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
)

# 导入模型基类
from .base import BaseRiskModel

# 导入提升树模型 (boosting/, 可选依赖)
try:
    from .boosting import XGBoostRiskModel
except (ImportError, Exception):
    XGBoostRiskModel = None

try:
    from .boosting import LightGBMRiskModel
except (ImportError, Exception):
    LightGBMRiskModel = None

try:
    from .boosting import CatBoostRiskModel
except (ImportError, Exception):
    CatBoostRiskModel = None

try:
    from .boosting import NGBoostRiskModel
except (ImportError, Exception):
    NGBoostRiskModel = None

# 导入经典模型 (classical/)
from .classical import (
    RandomForestRiskModel,
    ExtraTreesRiskModel,
    GradientBoostingRiskModel,
    LogisticRegression,
)

# 导入评分卡 (scorecard/)
from .scorecard import ScoreCard, RoundScoreCard


# 导入规则集分类模型 (rules/)
from .rules import (
    RuleSet,
    RulesClassifier,
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules,
)

# 导入评估报告 (evaluation/)
from .evaluation import ModelReport

# 导入超参数调优 (tuning/, 可选依赖)
try:
    from .tuning import ModelTuner, AutoTuner, TuningObjective
    TUNING_AVAILABLE = True
except ImportError:
    TUNING_AVAILABLE = False
    ModelTuner = None
    AutoTuner = None
    TuningObjective = None

__all__ = [
    # 损失函数基类
    "BaseLoss",
    "BaseMetric",
    # 不平衡数据处理
    "FocalLoss",
    "AsymmetricFocalLoss",
    "WeightedBCELoss",
    # 成本敏感
    "CostSensitiveLoss",
    # 风控业务损失
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    "OrdinalRankLoss",
    "LiftFocusedLoss",
    # 自定义评估指标
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    # 框架适配器
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",
    # 模型基类
    "BaseRiskModel",
    # 各模型类
    "XGBoostRiskModel",
    "LightGBMRiskModel",
    "CatBoostRiskModel",
    "NGBoostRiskModel",
    "RandomForestRiskModel",
    "ExtraTreesRiskModel",
    "GradientBoostingRiskModel",
    # 逻辑回归
    "LogisticRegression",
    # 评分卡
    "ScoreCard",
    # 规则集分类
    "RuleSet",
    "RulesClassifier",
    "LogicOperator",
    "RuleResult",
    "create_and_ruleset",
    "create_or_ruleset",
    "combine_rules",
    # 评估报告
    "ModelReport",
]

# 如果optuna可用，添加调优类
if TUNING_AVAILABLE:
    __all__.extend(["ModelTuner", "AutoTuner", "TuningObjective"])
