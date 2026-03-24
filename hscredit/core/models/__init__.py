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
- 模型可解释性: SHAP分析和特征重要性可视化
- 概率校准: Platt Scaling、Isotonic Regression等校准方法
- 概率转评分: 支持信用评分(越大越好)和欺诈评分(越小越好)
- 评分漂移校准: 解决生产环境与训练分布不一致问题

**支持的模型**

| 模型 | 类名 | 说明 |
|------|------|------|
| XGBoost | XGBoostRiskModel | 高效梯度提升树 |
| LightGBM | LightGBMRiskModel | 快速梯度提升树 |
| CatBoost | CatBoostRiskModel | 对类别特征友好的提升树 |
| RandomForest | RandomForestRiskModel | 随机森林 |
| ExtraTrees | ExtraTreesRiskModel | 极端随机树 |
| GradientBoosting | GradientBoostingRiskModel | 梯度提升树 |
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

**3. 特征重要性可视化**

>>> # 传统特征重要性
>>> fig = model.plot_feature_importance(top_n=15)
>>> fig.savefig('importance.png')

>>> # SHAP特征重要性
>>> fig = model.plot_feature_importance(X_test, method='shap', top_n=15)

>>> # 组合对比图
>>> fig = model.plot_feature_importance(X_test, method='combined', top_n=10)

**4. SHAP可解释性分析**

>>> from hscredit.core.models.interpretability import ModelExplainer
>>> explainer = ModelExplainer(model)
>>> shap_values = explainer.compute_shap_values(X_test)
>>> explainer.plot_shap_summary(X_test)
>>> explainer.plot_shap_dependence('feature_name', X_test)

**5. 概率校准**

>>> from hscredit.core.models.calibration import ProbabilityCalibrator
>>> calibrator = ProbabilityCalibrator(method='isotonic')
>>> calibrator.fit(model, X_calib, y_calib)  # 或 calibrator.fit(model, df_calib)
>>> proba_calib = calibrator.predict_proba(X_test)
>>> calibrator.plot_reliability_diagram(X_test, y_test)

**6. 概率转评分**

>>> from hscredit.core.models.probability_to_score import ScoreTransformer
>>> # 信用评分(概率越低分越高): 300-1000分
>>> transformer = ScoreTransformer(
...     method='standard',
...     lower=300,
...     upper=1000,
...     direction='descending',
...     base_odds=0.02,
...     base_score=600,
...     pdo=20,
...     precision=0
... )
>>> transformer.fit(model, X_train)
>>> credit_scores = transformer.predict_score(X_test)
>>> transformer.plot_transformation_curve()

>>> # 欺诈评分(概率越高分越高): 0-100分
>>> fraud_transformer = ScoreTransformer(
...     method='linear',
...     lower=0,
...     upper=100,
...     direction='ascending',
...     precision=0
... )
>>> fraud_transformer.fit(model, X_train)
>>> fraud_scores = fraud_transformer.predict_score(X_test)

**7. 评分漂移校准**

>>> from hscredit.core.models.score_drift import ScoreDriftCalibrator
>>> # 检测生产环境评分漂移
>>> calibrator = ScoreDriftCalibrator(method='linear')
>>> calibrator.fit(model, X_production, X_reference=X_train)
>>> scores_calib = calibrator.predict_score(X_test)
>>> drift_report = calibrator.detect_drift(X_train, X_production)

>>> # 分位数对齐校准
>>> quantile_calibrator = ScoreDriftCalibrator(
...     method='quantile',
...     n_quantiles=100
... )
>>> quantile_calibrator.fit(model, X_production, X_reference=X_train)

>>> # 分箱重校准(需要标签)
>>> binning_calibrator = ScoreDriftCalibrator(method='binning', n_bins=10)
>>> binning_calibrator.fit(
...     model, X_production, y_production,
...     X_reference=X_train, y_reference=y_train
... )

**8. 生成完整报告**

>>> report = model.generate_report(X_train, y_train, X_test, y_test)
>>> report.print_report()

**9. 超参数调优**

>>> from hscredit.core.models import AutoTuner
>>> tuner = AutoTuner.create('xgboost', metric='auc')
>>> best_params = tuner.fit(X_train, y_train, n_trials=100)
>>> best_model = tuner.get_best_model()

**10. 统一接口使用不同模型**

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
    WeightedBCELoss,
    # 成本敏感
    CostSensitiveLoss,
    # 风控业务损失
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
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

# 导入各模型类
from .xgboost_model import XGBoostRiskModel
from .lightgbm_model import LightGBMRiskModel
from .catboost_model import CatBoostRiskModel
from .sklearn_models import (
    RandomForestRiskModel,
    ExtraTreesRiskModel,
    GradientBoostingRiskModel,
)

# 导入逻辑回归模型
from .logistic_regression import LogisticRegression

# 导入评分卡模型
from .scorecard import ScoreCard

# 导入规则集分类模型
from .rule_classifier import (
    RuleSet,
    RulesClassifier,
    RuleSetClassifier,
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules,
)

# 导入评估报告
from .report import ModelReport

# 导入可解释性模块 (可选依赖)
try:
    from .interpretability import (
        ModelExplainer,
        plot_feature_importance,
        plot_shap_importance,
        plot_importance_comparison,
    )
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    ModelExplainer = None
    plot_feature_importance = None
    plot_shap_importance = None
    plot_importance_comparison = None

# 导入概率校准模块 (可选依赖)
try:
    from .calibration import (
        BaseCalibrator,
        PlattCalibrator,
        IsotonicCalibrator,
        BetaCalibrator,
        HistogramCalibrator,
        ProbabilityCalibrator,
        CalibratedModel,
        plot_calibration_comparison,
        calibrate_model,
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    BaseCalibrator = None
    PlattCalibrator = None
    IsotonicCalibrator = None
    BetaCalibrator = None
    HistogramCalibrator = None
    ProbabilityCalibrator = None
    CalibratedModel = None
    plot_calibration_comparison = None
    calibrate_model = None

# 导入概率转评分模块 (可选依赖)
try:
    from .probability_to_score import (
        BaseScoreTransformer,
        StandardScoreTransformer,
        LinearScoreTransformer,
        QuantileScoreTransformer,
        ScoreTransformer,
        transform_probability_to_score,
        plot_score_transformation_curve,
        compare_score_transformers,
    )
    SCORE_CONVERSION_AVAILABLE = True
except ImportError:
    SCORE_CONVERSION_AVAILABLE = False
    BaseScoreTransformer = None
    StandardScoreTransformer = None
    LinearScoreTransformer = None
    QuantileScoreTransformer = None
    ScoreTransformer = None
    transform_probability_to_score = None
    plot_score_transformation_curve = None
    compare_score_transformers = None

# 导入超参数调优 (可选依赖)
try:
    from .tuning import ModelTuner, AutoTuner
    TUNING_AVAILABLE = True
except ImportError:
    TUNING_AVAILABLE = False
    ModelTuner = None
    AutoTuner = None

__all__ = [
    # 损失函数基类
    "BaseLoss",
    "BaseMetric",
    # 不平衡数据处理
    "FocalLoss",
    "WeightedBCELoss",
    # 成本敏感
    "CostSensitiveLoss",
    # 风控业务损失
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
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
    "RandomForestRiskModel",
    "ExtraTreesRiskModel",
    "GradientBoostingRiskModel",
    # 逻辑回归
    "LogisticRegression",
    # 评分卡
    "ScoreCard",
    # 规则集分类
    "RuleSet",
    "RuleSetClassifier",
    "RulesClassifier",
    "LogicOperator",
    "RuleResult",
    "create_and_ruleset",
    "create_or_ruleset",
    "combine_rules",
    # 评估报告
    "ModelReport",
]

# 如果shap可用，添加可解释性类
if INTERPRETABILITY_AVAILABLE:
    __all__.extend([
        "ModelExplainer",
        "plot_feature_importance",
        "plot_shap_importance",
        "plot_importance_comparison",
    ])

# 如果校准模块可用，添加校准类
if CALIBRATION_AVAILABLE:
    __all__.extend([
        "BaseCalibrator",
        "PlattCalibrator",
        "IsotonicCalibrator",
        "BetaCalibrator",
        "HistogramCalibrator",
        "ProbabilityCalibrator",
        "CalibratedModel",
        "plot_calibration_comparison",
        "calibrate_model",
    ])

# 如果概率转评分模块可用，添加相关类
if SCORE_CONVERSION_AVAILABLE:
    __all__.extend([
        "BaseScoreTransformer",
        "StandardScoreTransformer",
        "LinearScoreTransformer",
        "QuantileScoreTransformer",
        "ScoreTransformer",
        "transform_probability_to_score",
        "plot_score_transformation_curve",
        "compare_score_transformers",
    ])

# 如果评分漂移校准模块可用，添加相关类
try:
    from .score_drift import (
        BaseDriftCalibrator,
        LinearDriftCalibrator,
        QuantileAligner,
        BinningRecalibrator,
        ScoreDriftCalibrator,
        plot_drift_comparison,
        compare_drift_methods,
    )
    DRIFT_CALIBRATION_AVAILABLE = True
except ImportError:
    DRIFT_CALIBRATION_AVAILABLE = False
    BaseDriftCalibrator = None
    LinearDriftCalibrator = None
    QuantileAligner = None
    BinningRecalibrator = None
    ScoreDriftCalibrator = None
    plot_drift_comparison = None
    compare_drift_methods = None

if DRIFT_CALIBRATION_AVAILABLE:
    __all__.extend([
        "BaseDriftCalibrator",
        "LinearDriftCalibrator",
        "QuantileAligner",
        "BinningRecalibrator",
        "ScoreDriftCalibrator",
        "plot_drift_comparison",
        "compare_drift_methods",
    ])

# 如果optuna可用，添加调优类
if TUNING_AVAILABLE:
    __all__.extend(["ModelTuner", "AutoTuner"])
