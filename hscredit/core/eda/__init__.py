# -*- coding: utf-8 -*-
"""
EDA (Exploratory Data Analysis) 模块 - 金融信贷数据探索性分析.

提供金融风控场景下的完整数据探索功能，包括:
- 数据概览与质量评估
- 目标变量分析 (逾期率、分布)
- 特征分析 (数值/类别特征分布)
- 特征与标签关系 (IV、逾期率分箱)
- 特征稳定性分析 (PSI、时间稳定性)
- 相关性分析
- 可视化图表

参考:
    - 金融风控建模标准流程
    - 评分卡开发规范
"""

from .data_overview import DataOverview
from .target_analysis import TargetAnalysis
from .feature_analysis import FeatureAnalysis
from .feature_label_relationship import FeatureLabelRelationship
from .stability_analysis import StabilityAnalysis
from .correlation_analysis import CorrelationAnalysis
from .eda_report import EDAReport

__all__ = [
    'DataOverview',
    'TargetAnalysis',
    'FeatureAnalysis',
    'FeatureLabelRelationship',
    'StabilityAnalysis',
    'CorrelationAnalysis',
    'EDAReport',
]
