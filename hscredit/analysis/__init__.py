"""分析模块.

提供特征分析、策略分析和规则挖掘功能。

子模块:
- feature: 特征分析（分箱统计、IV计算等）
- strategy: 策略分析（待开发）
"""

from .feature_analyzer import FeatureAnalyzer, feature_bin_stats

__all__ = [
    # 特征分析
    "FeatureAnalyzer",
    "feature_bin_stats",
]
