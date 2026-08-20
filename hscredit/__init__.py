"""hscredit - 金融信贷风险策略和模型开发库.

一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。
"""

__version__ = "0.1.2"
__author__ = "hscredit"
__email__ = "hscredit@hengshucredit.com"

from ._compat import prepare_runtime_compatibility

prepare_runtime_compatibility()

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
from .core import model_selection as _model_selection
from . import excel as _excel
from . import report as _report
from . import utils as _utils

from .core.binning import *
from .core.encoders import *
from .core.selectors import *
from .core.models import *
from .core.metrics import *
from .core.viz import *
from .core.eda import *
from .core.rules import *
from .core.financial import *
from .core.feature_engineering import *
from .core.model_selection import *
from .excel import *
from .report import *
from .utils import *

init_setting()


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


# boosting/tuning 模型为懒加载（core.models 不将其放入 __all__，避免本文件顶部的
# `from .core.models import *` 在 import hscredit 时即时触发 xgboost/lightgbm/
# catboost/ngboost/optuna 的加载）。这里通过 __getattr__ 保留 hscredit.XGBoost
# 等顶层直接访问方式，首次访问时才委托给 core.models 完成真正的导入。
_LAZY_MODEL_NAMES = (
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NGBoost",
    "ModelTuner",
    "AutoTuner",
    "TuningObjective",
    "TuningSampler",
)


def __getattr__(name):
    if name in _LAZY_MODEL_NAMES:
        value = getattr(_models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"模块 {__name__!r} 不存在属性 {name!r}")


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
    print("  - core.binning: 分箱算法 (Uniform/Quantile/Tree/ChiMerge/BestKS/BestIV/MDLP/2D)")
    print("  - core.selectors: 特征筛选 (Variance/Null/IV/Corr/VIF/Lift/PSI...)")
    print("  - core.encoders: 编码器 (WOE/Target/Count/OneHot...)")
    print("  - core.models: 风控模型、评分卡、自定义损失函数和评估指标")
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
    print("  - utils: 工具函数 (随机种子、数据集、pickleIO、输入校验)")


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
    _model_selection,
    _excel,
    _report,
    _utils,
)

__all__ = _BASE_EXPORTS + [name for name in _MODULE_EXPORTS if name not in _BASE_EXPORTS]
