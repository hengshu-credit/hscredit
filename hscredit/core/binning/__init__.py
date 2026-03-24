"""分箱算法模块 - 统一接口整合所有分箱方法.

提供多种分箱算法的统一接口，所有方法都可以通过 OptimalBinning 访问：

**基础方法:**
- uniform: 等宽分箱
- quantile: 等频分箱
- tree: 决策树分箱
- chi_merge: 卡方分箱

**优化方法:**
- optimal_ks: 最优KS分箱
- optimal_iv: 最优IV分箱
- mdlp: MDLP分箱（基于信息论）

**运筹规划方法:**
- or_tools: OR-Tools 最优化分箱（基于 Google OR-Tools）

**高级方法:**
- cart: CART分箱（参考optbinning实现）
- monotonic: 单调性约束分箱（支持U型/倒U型/凸/凹）
- genetic: 遗传算法分箱
- smooth: 平滑/正则化分箱
- kernel_density: 核密度分箱
- best_lift: Best Lift分箱
- target_bad_rate: 目标坏样本率分箱

**主要类**

- BaseBinning: 分箱算法基类
- OptimalBinning: 统一分箱接口（推荐）
- 各具体分箱类: UniformBinning, QuantileBinning, TreeBinning, CartBinning,
  ChiMergeBinning, BestKSBinning, BestIVBinning, MDLPBinning, ORBinning,
  KMeansBinning, MonotonicBinning, GeneticBinning, SmoothBinning,
  KernelDensityBinning, BestLiftBinning, TargetBadRateBinning

**快速开始**

>>> from hscredit.core.binning import OptimalBinning
>>> # 使用最优IV分箱
>>> binner = OptimalBinning(method='optimal_iv', max_n_bins=5)
>>> binner.fit(X_train, y_train)
>>> X_binned = binner.transform(X_test)
>>> bin_table = binner.get_bin_table('feature_name')

>>> # 使用单调性约束（支持U型/倒U型）
>>> binner = OptimalBinning(method='monotonic', monotonic='peak')
>>> binner.fit(X, y)

>>> # 指定切分点
>>> binner = OptimalBinning(user_splits={'age': [25, 35, 45]})
>>> binner.fit(X, y)

>>> # 自动选择最优方法
>>> best_method = OptimalBinning.auto_select_method(X, y, 'feature')
>>> binner = OptimalBinning(method=best_method)
>>> binner.fit(X, y)
"""

from .base import BaseBinning
from .uniform_binning import UniformBinning
from .quantile_binning import QuantileBinning
from .tree_binning import TreeBinning
from .cart_binning import CartBinning
from .chi_merge_binning import ChiMergeBinning
from .best_ks_binning import BestKSBinning
from .best_iv_binning import BestIVBinning
from .optimal_binning import OptimalBinning
from .mdlp_binning import MDLPBinning
from .or_binning import ORBinning, CustomObjectives
from .kmeans_binning import KMeansBinning
from .monotonic_binning import MonotonicBinning
from .genetic_binning import GeneticBinning
from .smooth_binning import SmoothBinning
from .kernel_density_binning import KernelDensityBinning
from .best_lift_binning import BestLiftBinning
from .target_bad_rate_binning import TargetBadRateBinning

__all__ = [
    'BaseBinning',
    'UniformBinning',
    'QuantileBinning',
    'TreeBinning',
    'CartBinning',
    'ChiMergeBinning',
    'BestKSBinning',
    'BestIVBinning',
    'OptimalBinning',
    'MDLPBinning',
    'ORBinning',
    'CustomObjectives',
    'KMeansBinning',
    'MonotonicBinning',
    'GeneticBinning',
    'SmoothBinning',
    'KernelDensityBinning',
    'BestLiftBinning',
    'TargetBadRateBinning',
]
