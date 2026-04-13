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

# ========== 顶层公开 API 聚合 ==========

from .core import binning as _binning
from .core import encoders as _encoders
from .core import selectors as _selectors
from .core import models as _models
from .core import metrics as _metrics
from .core import viz as _viz
from .core import eda as _eda
from .core import rules as _rules
from .core import financial as _financial
from .core import feature_engineering as _feature_engineering
from . import excel as _excel
from . import report as _report
from . import utils as _utils

from .core.binning import *
from .core.encoders import *
from .core.selectors import *
from .core.models import *
from .core.metrics import *
from .core.viz import *
# EDA 放在 metrics 之后导入，顶层 feature_summary 默认指向更常用的 EDA API。
from .core.eda import *
from .core.rules import *
from .core.financial import *
from .core.feature_engineering import *
from .excel import *
from .report import *
from .utils import *


def _collect_public_exports(*modules):
    """汇总模块 __all__，过滤私有符号并去重。"""
    exports = []
    seen = set()
    for module in modules:
        for name in getattr(module, "__all__", []):
            if name.startswith("_") or name in seen:
                continue
            exports.append(name)
            seen.add(name)
    return exports


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


_BASE_EXPORTS = [
    "HSCreditError",
    "ValidationError",
    "InputValidationError",
    "InputTypeError",
    "FeatureNotFoundError",
    "StateError",
    "NotFittedError",
    "DependencyError",
    "SerializationError",
    "Pipeline",
    "make_pipeline",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "ColumnTransformer",
    "make_column_selector",
    "make_column_transformer",
    "FunctionTransformer",
    "get_version",
    "info",
]

_MODULE_EXPORTS = _collect_public_exports(
    _binning,
    _encoders,
    _selectors,
    _models,
    _metrics,
    _viz,
    _eda,
    _rules,
    _financial,
    _feature_engineering,
    _excel,
    _report,
    _utils,
)

__all__ = _BASE_EXPORTS + [name for name in _MODULE_EXPORTS if name not in _BASE_EXPORTS]
