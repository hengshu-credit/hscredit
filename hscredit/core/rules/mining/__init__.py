"""规则挖掘模块.

提供从数据中自动挖掘规则的功能，包括：
- 单特征规则挖掘
- 多特征交叉规则挖掘  
- 决策树规则提取（支持DT、RF、GBDT、XGBoost、孤立森林）
- 规则评估指标
- 决策树可视化

代码风格参考hscredit的binning模块和Rule模块，fit方法兼容scorecardpipeline风格。

示例:
    >>> from hscredit.core.rules.mining import SingleFeatureRuleMiner
    >>> miner = SingleFeatureRuleMiner(target='ISBAD')
    >>> miner.fit(df)
    >>> rules = miner.get_top_rules(top_n=10)
    
    >>> from hscredit.core.rules.mining import TreeRuleExtractor
    >>> extractor = TreeRuleExtractor(algorithm='rf', max_depth=5)
    >>> extractor.fit(X, y)
    >>> rules = extractor.extract_rules()
"""

from .single_feature import SingleFeatureRuleMiner
from .multi_feature import MultiFeatureRuleMiner
from .tree_extractor import TreeRuleExtractor
from .metrics import RuleMetrics, calculate_rule_metrics
from .visualization import TreeVisualizer, plot_decision_tree

__all__ = [
    'SingleFeatureRuleMiner',
    'MultiFeatureRuleMiner', 
    'TreeRuleExtractor',
    'RuleMetrics',
    'calculate_rule_metrics',
    'TreeVisualizer',
    'plot_decision_tree',
]
