"""特征筛选模块.

提供多种特征筛选方法，从多个维度评估和筛选特征。

**筛选方法分类**

- 过滤法: 方差、相关性、VIF、缺失率、单一值率、基数、IV、Lift、PSI
- 包装法: 穷举搜索、逐步回归、Boruta
- 嵌入法: 特征重要性、Lasso、树模型重要性、Permutation Importance

**通用特性**

- 独立使用: fit/transform 接口
- sklearn Pipeline 集成
- 统一的中文筛选报告

**参考样例**

>>> from hscredit.core.selectors import VarianceSelector, IVSelector, CorrSelector
>>> from hscredit.core.selectors.base import SelectionReportCollector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'f{i}' for i in range(10)])
>>> y = pd.Series(np.random.randint(0, 2, 1000))
>>> selector = VarianceSelector(threshold=0.01)
>>> selector.fit(X, y)
VarianceSelector(...)
>>> report = selector.get_selection_report()
>>> print(f"输入特征数: {report['输入特征数']}, 选中特征数: {report['选中特征数']}")
"""

from typing import Union, List, Dict, Optional, Any, Callable
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector, CompositeFeatureSelector, SelectionReportCollector
from .variance_selector import VarianceSelector
from .null_selector import NullSelector
from .mode_selector import ModeSelector
from .corr_selector import CorrSelector
from .vif_selector import VIFSelector
from .iv_selector import IVSelector
from .lift_selector import LiftSelector
from .psi_selector import PSISelector
from .cardinality_selector import CardinalitySelector
from .type_selector import TypeSelector
from .regex_selector import RegexSelector
from .importance_selector import FeatureImportanceSelector
from .null_importance_selector import NullImportanceSelector
from .rfe_selector import RFESelector
from .sequential_selector import SequentialFeatureSelector
from .boruta_selector import BorutaSelector
from .mutual_info_selector import MutualInfoSelector
from .chi2_selector import Chi2Selector
from .f_test_selector import FTestSelector
from .stepwise_selector import StepwiseSelector
from .stability_selector import StabilityAwareSelector
from .scorecard_feature_selection import ScorecardFeatureSelection

# 导出所有筛选器
__all__ = [
    # 基类
    'BaseFeatureSelector',
    'SelectionReportCollector',
    
    # 过滤法 - 基础筛选
    'TypeSelector',
    'RegexSelector',
    'NullSelector',
    'ModeSelector',
    'CardinalitySelector',
    'VarianceSelector',
    
    # 过滤法 - 相关性筛选
    'CorrSelector',
    'VIFSelector',
    
    # 过滤法 - 目标导向筛选
    'IVSelector',
    'LiftSelector',
    'PSISelector',
    
    # 嵌入法 - 特征重要性
    'FeatureImportanceSelector',
    'NullImportanceSelector',
    'RFESelector',
    'SequentialFeatureSelector',
    'StepwiseSelector',
    
    # 高级方法
    'BorutaSelector',
    'MutualInfoSelector',
    'Chi2Selector',
    'FTestSelector',
    'StabilityAwareSelector',
    'ScorecardFeatureSelection',
    
    # 组合筛选器
    'CompositeFeatureSelector',
]
