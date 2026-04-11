"""hscredit - 金融信贷风险策略和模型开发库.

一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。
"""

__version__ = "0.1.0"
__author__ = "hscredit team"
__email__ = "hscredit@hengshucredit.com"

from .exceptions import (
    HSCreditError,
    ValidationError,
    InputValidationError,
    InputTypeError,
    FeatureNotFoundError,
    StateError,
    NotFittedError,
    DependencyError,
    SerializationError,
)


# ========== sklearn Pipeline 和集成学习组件 ==========
# 为了方便用户，直接从hscredit导入sklearn的Pipeline相关组件

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.preprocessing import FunctionTransformer

# ========== 核心模块导入 (先导入core，避免循环导入) ==========

# 核心分箱模块
from .core.binning import (
    BaseBinning,
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    ORBinning,
    CustomObjectives,
    OptimalBinning,
    CartBinning,
    KMeansBinning,
    MonotonicBinning,
    GeneticBinning,
    SmoothBinning,
    KernelDensityBinning,
    BestLiftBinning,
    TargetBadRateBinning,
)

# 核心编码器模块
from .core.encoders import (
    BaseEncoder,
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    GBMEncoder,
    CardinalityEncoder,
)

# 核心特征筛选模块
from .core.selectors import (
    BaseFeatureSelector,
    SelectionReportCollector,
    VarianceSelector,
    NullSelector,
    ModeSelector,
    CorrSelector,
    VIFSelector,
    IVSelector,
    LiftSelector,
    PSISelector,
    CardinalitySelector,
    TypeSelector,
    RegexSelector,
    FeatureImportanceSelector,
    NullImportanceSelector,
    RFESelector,
    SequentialFeatureSelector,
    BorutaSelector,
    MutualInfoSelector,
    Chi2Selector,
    FTestSelector,
    CompositeFeatureSelector,
    StepwiseSelector,
    StabilityAwareSelector,
)

# 核心模型/损失函数模块
from .core.models import (
    BaseLoss,
    BaseMetric,
    FocalLoss,
    AsymmetricFocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    KSMetric,
    GiniMetric,
    PSIMetric,
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
    LogisticRegression,
    ScoreCard,
    # 规则集分类
    RuleSet,
    RulesClassifier,
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules,
)


def get_version():
    """获取版本号."""
    return __version__


def info():
    """打印包信息."""
    print(f"hscredit version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print("一个完整的金融信贷风险建模工具包")
    print()
    print("Pipeline 和集成学习组件 (从sklearn导入):")
    print("  - Pipeline: 管道，串联多个转换器和模型")
    print("  - make_pipeline: 快速创建Pipeline")
    print("  - VotingClassifier/VotingRegressor: 投票分类器/回归器")
    print("  - StackingClassifier/StackingRegressor: 堆叠分类器/回归器")
    print("  - ColumnTransformer: 列转换器，对不同列应用不同转换")
    print("  - FunctionTransformer: 函数转换器，将函数包装成Transformer")
    print()
    print("核心模块 (core):")
    print("  - core.binning: 分箱算法 (Uniform/Quantile/Tree/ChiMerge/BestKS/BestIV/MDLP)")
    print("  - core.selectors: 特征筛选 (Variance/Null/IV/Corr/VIF/Lift/PSI...)")
    print("  - core.encoders: 编码器 (WOE/Target/Count/OneHot...)")
    print("  - core.models: 自定义损失函数和评估指标")
    print("  - core.viz: 可视化 (bin_plot/ks_plot/corr_plot...)")
    print("  - core.feature_engineering: 特征工程 (NumExprDerive)")
    print("  - core.rules: 规则引擎 (Rule)")
    print("  - core.financial: 金融计算 (FV/PV/PMT/NPER/IRR/NPV)")
    print("  - core.eda: 数据探索分析 (EDAReport/DataOverview/TargetAnalysis/IV/PSI...)")
    print()
    print("报告模块 (report):")
    print("  - excel: Excel报告生成")
    print("  - report.feature_analyzer: 特征分箱统计与自动分析")
    print("  - report.rule_analysis: 规则集与多标签规则分析")
    print()
    print("工具模块 (utils):")
    print("  - utils: 工具函数 (随机种子、数据集、pickleIO)")
    print()
    print("待实现模块:")
    print("  - core.encoding: 编码转换")
    print("  - core.metrics: 指标计算 (KS/AUC/PSI/IV/Gini)")
