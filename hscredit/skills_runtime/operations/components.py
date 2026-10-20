"""可声明式创建的 hscredit 组件白名单；不接受模块路径。"""

import importlib

from .common import invalid, resolve

MODELS = (
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "SVM",
    "DecisionTreeClassifier",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NGBoost",
    "ScoreCard",
    "RoundScoreCard",
    "ProbabilityScoreCard",
    "RulesClassifier",
)
SELECTORS = (
    "TypeSelector",
    "RegexSelector",
    "NullSelector",
    "ModeSelector",
    "CardinalitySelector",
    "VarianceSelector",
    "CorrSelector",
    "VIFSelector",
    "IVSelector",
    "KSSelector",
    "LiftSelector",
    "PSISelector",
    "FeatureImportanceSelector",
    "NullImportanceSelector",
    "RFESelector",
    "SequentialFeatureSelector",
    "StepwiseSelector",
    "BorutaSelector",
    "MutualInfoSelector",
    "Chi2Selector",
    "FTestSelector",
    "StabilityAwareSelector",
    "ScorecardFeatureSelection",
    "CompositeFeatureSelector",
)
LOSSES = (
    "FocalLoss",
    "AsymmetricFocalLoss",
    "BalancedFocalLoss",
    "WeightedBCELoss",
    "CostSensitiveLoss",
    "BadDebtLoss",
    "ApprovalRateLoss",
    "ProfitMaxLoss",
    "ExpectedProfitLoss",
    "OrdinalRankLoss",
    "LiftFocusedLoss",
    "RankingAUCProxyLoss",
    "KSFocusedLoss",
    "TopKBadCaptureLoss",
    "AmountWeightedLoss",
    "ExpectedValueLoss",
    "LossMetric",
    "KSMetric",
    "GiniMetric",
    "PSIMetric",
    "AUCMetric",
    "TopKCaptureMetric",
    "TopKLiftMetric",
    "BadDebtMetric",
    "ApprovalRateMetric",
    "ProfitMetric",
    "ClassificationCostMetric",
    "XGBoostLossAdapter",
    "LightGBMLossAdapter",
    "CatBoostLossAdapter",
    "TabNetLossAdapter",
    "NGBoostLossAdapter",
)
TRANSFORMERS = (
    "StandardScoreTransformer",
    "LinearScoreTransformer",
    "QuantileScoreTransformer",
    "BoxCoxScoreTransformer",
    "ScoreTransformer",
    "ScoreDriftCalibrator",
    "ProbabilityCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "HistogramCalibrator",
    "RuleSet",
)
COMPONENTS = {
    **{name: "hscredit.core.models" for name in MODELS + LOSSES + TRANSFORMERS},
    **{name: "hscredit.core.selectors" for name in SELECTORS},
    **{name: "hscredit.core.rules" for name in ("Rule", "RuleFlow")},
    **{name: "hscredit.core.binning" for name in ("OptimalBinning", "OptimalBinning2D")},
}


def component_class(name):
    if name not in COMPONENTS:
        raise invalid(f"组件名称未登记：{name}", "component")
    return getattr(importlib.import_module(COMPONENTS[name]), name)


def rule(value, context=None):
    from ...core.rules import Rule

    if isinstance(value, Rule):
        return value
    if isinstance(value, str):
        return Rule(value)
    if isinstance(value, dict):
        if "component" in value or "input" in value:
            return resolve(context, value)
        if set(value) <= {"expr", "name", "description", "weight", "n_jobs"} and "expr" in value:
            return Rule(**value)
    raise invalid("规则必须是表达式、Rule 配置或显式输入引用")


def rules(value, context=None):
    return [rule(item, context) for item in (value if isinstance(value, list) else [value])]


def create_component(context, name, params=None):
    cls = component_class(name)
    kwargs = resolve(context, dict(params or {}))
    if name in {"RulesClassifier", "RuleSet", "RuleFlow"} and "rules" in kwargs:
        kwargs["rules"] = [
            rule(item, context) if isinstance(item, (str, dict)) else item
            for item in (kwargs["rules"] if isinstance(kwargs["rules"], list) else [kwargs["rules"]])
        ]
    if name == "CompositeFeatureSelector":
        kwargs["selectors"] = [
            tuple(item) if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) else item
            for item in kwargs.get("selectors", [])
        ]
    return cls(**kwargs)
