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
| XGBoost | XGBoost | 高效梯度提升树 |
| LightGBM | LightGBM | 快速梯度提升树 |
| CatBoost | CatBoost | 对类别特征友好的提升树 |
| RandomForest | RandomForest | 随机森林 |
| ExtraTrees | ExtraTrees | 极端随机树 |
| GradientBoosting | GradientBoosting | 梯度提升树 |
| SVM | SVM | 支持向量机概率分类 |
| DecisionTree | DecisionTreeClassifier | 单棵决策树分类 |
| NGBoost | NGBoost | 自然梯度提升（概率预测） |
| LogisticRegression | LogisticRegression | 扩展逻辑回归 |
| ScoreCard | ScoreCard | 评分卡模型 |

**快速开始**

**1. 基础模型训练**

>>> from hscredit.core.models import XGBoost
>>> model = XGBoost(
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
...     XGBoost,
...     LightGBM,
...     CatBoost
... )
>>>
>>> models = {
...     'xgboost': XGBoost(max_depth=5),
...     'lightgbm': LightGBM(num_leaves=31),
...     'catboost': CatBoost(depth=6),
... }
>>>
>>> for name, model in models.items():
...     model.fit(X_train, y_train)
...     metrics = model.evaluate(X_test, y_test)
...     print(f"{name}: AUC={metrics['AUC']:.4f}")
"""

import importlib

# 导入损失函数
from .losses import (
    # 基类
    BaseLoss,
    BaseMetric,
    # 不平衡数据处理
    FocalLoss,
    AsymmetricFocalLoss,
    BalancedFocalLoss,
    WeightedBCELoss,
    # 成本敏感
    CostSensitiveLoss,
    # 风控业务损失
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    ExpectedProfitLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    RankingAUCProxyLoss,
    KSFocusedLoss,
    TopKBadCaptureLoss,
    AmountWeightedLoss,
    ExpectedValueLoss,
    # 自定义评估指标
    KSMetric,
    GiniMetric,
    PSIMetric,
    # 框架适配器
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
    NGBoostLossAdapter,
)

# 导入模型基类
from .base import BaseRiskModel

# 导入提升树模型 (boosting/, 可选重依赖 xgboost/lightgbm/catboost/ngboost，懒加载)
_LAZY_BOOSTING_MODELS = ("XGBoost", "LightGBM", "CatBoost", "NGBoost")

# 导入经典模型 (classical/)
from .classical import (
    RandomForest,
    ExtraTrees,
    GradientBoosting,
    SVM,
    DecisionTreeClassifier,
    LogisticRegression,
)

# 导入评分卡 (scorecard/)
from .scorecard import (
    ScoreCard,
    RoundScoreCard,
    ProbabilityScoreCard,
    BaseScoreTransformer,
    StandardScoreTransformer,
    LinearScoreTransformer,
    QuantileScoreTransformer,
    BoxCoxScoreTransformer,
    ScoreTransformer,
    transform_probability_to_score,
    ScoreDriftCalibrator,
)

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

# 导入解释工具 (evaluation/)；ModelReport 已统一为 hscredit.report.ModelReport，
# 通过 __getattr__ 懒加载兼容别名，避免 import hscredit 期间触发 hscredit.report 循环导入。
from .evaluation import CounterfactualExplainer, ExplanationResult, model_explain_report

# 导入超参数调优 (tuning/, 可选重依赖 optuna，懒加载)
_LAZY_TUNING_MODELS = ("ModelTuner", "AutoTuner", "TuningObjective", "TuningSampler")

# 搜索空间兼容符号本身不依赖 Optuna，可安全地作为模型模块常规公开 API 导出。
from .tuning.search_space import (  # noqa: E402
    Categorical,
    CategoricalDistribution,
    Dimension,
    FloatDistribution,
    IntDistribution,
    Integer,
    Real,
    choice,
    lognormal,
    loguniform,
    normal,
    qlognormal,
    qloguniform,
    qnormal,
    quniform,
    randint,
    suggest_categorical,
    suggest_discrete_uniform,
    suggest_float,
    suggest_int,
    suggest_loguniform,
    suggest_uniform,
    uniform,
)


def __getattr__(name):
    """懒加载 boosting/tuning 子包及 ModelReport 兼容别名，避免 import hscredit 时即时加载重依赖."""
    if name in _LAZY_BOOSTING_MODELS:
        value = getattr(importlib.import_module(".boosting", __name__), name)
    elif name in _LAZY_TUNING_MODELS:
        value = getattr(importlib.import_module(".tuning", __name__), name)
    elif name in {"ModelExplainer"}:
        value = getattr(importlib.import_module(".evaluation", __name__), name)
    elif name == "ModelReport":
        value = getattr(importlib.import_module(".evaluation", __name__), "ModelReport")
    else:
        raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")
    globals()[name] = value
    return value


__all__ = [
    # 损失函数基类
    "BaseLoss",
    "BaseMetric",
    # 不平衡数据处理
    "FocalLoss",
    "AsymmetricFocalLoss",
    "BalancedFocalLoss",
    "WeightedBCELoss",
    # 成本敏感
    "CostSensitiveLoss",
    # 风控业务损失
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    "ExpectedProfitLoss",
    # 排序与 AUC 优化
    "OrdinalRankLoss",
    "LiftFocusedLoss",
    "RankingAUCProxyLoss",
    # KS 分布分离
    "KSFocusedLoss",
    # 头部捕获优化
    "TopKBadCaptureLoss",
    # 金额/敞口加权
    "AmountWeightedLoss",
    "ExpectedValueLoss",
    # 自定义评估指标
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    # 框架适配器
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",
    "NGBoostLossAdapter",
    # 模型基类
    "BaseRiskModel",
    # 各模型类（boosting 系列为懒加载，不放入 __all__ 以避免 `import *` 触发重依赖加载，
    # 可通过 from hscredit.core.models import XGBoost 等方式显式访问）
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "SVM",
    "DecisionTreeClassifier",
    # 逻辑回归
    "LogisticRegression",
    # 评分卡
    "ScoreCard",
    "RoundScoreCard",
    "ProbabilityScoreCard",
    # 概率转评分
    "BaseScoreTransformer",
    "StandardScoreTransformer",
    "LinearScoreTransformer",
    "QuantileScoreTransformer",
    "BoxCoxScoreTransformer",
    "ScoreTransformer",
    "transform_probability_to_score",
    # 评分漂移校准
    "ScoreDriftCalibrator",
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
    "model_explain_report",
    "ExplanationResult",
    "CounterfactualExplainer",
    "ModelExplainer",
    # 统一超参数搜索空间声明
    "Dimension",
    "Real",
    "Integer",
    "Categorical",
    "IntDistribution",
    "FloatDistribution",
    "CategoricalDistribution",
    "suggest_int",
    "suggest_float",
    "suggest_categorical",
    "suggest_uniform",
    "suggest_discrete_uniform",
    "suggest_loguniform",
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "choice",
    "randint",
    "normal",
    "qnormal",
    "lognormal",
    "qlognormal",
    # 超参数调优为懒加载，不放入 __all__（可通过 from hscredit.core.models import AutoTuner 等方式显式访问）
]
