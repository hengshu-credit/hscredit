"""规则挖掘模块.

提供从数据中自动挖掘规则的功能，包括：
- 单特征规则挖掘
- 多特征交叉规则挖掘
- 决策树规则提取（支持DT、RF、GBDT、XGBoost、孤立森林）
- 规则评估指标

代码风格参考hscredit的binning模块和Rule模块，fit方法兼容scorecardpipeline风格。

示例:
    >>> from hscredit.report.mining import SingleFeatureRuleMiner
    >>> miner = SingleFeatureRuleMiner(target='target')
    >>> miner.fit(df)
    >>> rules = miner.get_top_rules(top_n=10)

    >>> from hscredit.report.mining import TreeRuleExtractor
    >>> extractor = TreeRuleExtractor(algorithm='rf', max_depth=5)
    >>> extractor.fit(X, y)
    >>> rules = extractor.extract_rules()

    >>> from hscredit.report.mining import DecisionTreeAnalyzer
    >>> analyzer = DecisionTreeAnalyzer(target='target', features=['age', 'income'])
    >>> analyzer.fit(df_train)
    >>> print(analyzer.evaluate([('测试', df_test)], metric_type='ks'))
"""

from .single_feature import SingleFeatureRuleMiner
from .multi_feature import MultiFeatureRuleMiner
from .multi_label import MultiLabelRuleMiner
from .tree_extractor import TreeRuleExtractor
from .manual_tree_extractor import DecisionTreeAnalyzer, ManualTreeExtractor
from .metrics import RuleMetrics, calculate_rule_metrics

__all__ = [
    'SingleFeatureRuleMiner',
    'MultiFeatureRuleMiner',
    'MultiLabelRuleMiner',
    'TreeRuleExtractor',
    'DecisionTreeAnalyzer',
    'ManualTreeExtractor',
    'RuleMetrics',
    'calculate_rule_metrics',
]
