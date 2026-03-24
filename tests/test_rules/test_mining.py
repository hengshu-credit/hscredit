"""规则挖掘模块测试."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.rules.mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    TreeRuleExtractor,
    RuleMetrics,
    calculate_rule_metrics
)
from hscredit.core.rules.rule import Rule


@pytest.fixture
def sample_data():
    """创建测试数据."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names


@pytest.fixture
def train_test_data(sample_data):
    """创建训练集和测试集."""
    df, feature_names = sample_data
    
    train_df = df[:800]
    test_df = df[800:]
    
    return train_df, test_df, feature_names


class TestSingleFeatureRuleMiner:
    """测试单特征规则挖掘器."""
    
    def test_init(self):
        """测试初始化."""
        miner = SingleFeatureRuleMiner(target='target', method='quantile')
        assert miner.target == 'target'
        assert miner.method == 'quantile'
        assert miner.max_n_bins == 20
        assert miner.min_lift == 1.1
        
        with pytest.raises(ValueError):
            SingleFeatureRuleMiner(method='invalid')
    
    def test_fit(self, sample_data):
        """测试拟合."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target', method='quantile')
        miner.fit(df)
        
        assert miner._is_fitted
        assert len(miner.results_) > 0
        assert miner.overall_badrate_ >= 0
    
    def test_fit_with_xy(self, sample_data):
        """测试sklearn风格的fit."""
        df, feature_names = sample_data
        
        X = df[feature_names]
        y = df['target']
        
        miner = SingleFeatureRuleMiner(target='target')
        miner.fit(X, y)
        
        assert miner._is_fitted
    
    def test_get_top_rules(self, sample_data):
        """测试获取top规则."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target')
        miner.fit(df)
        
        rules = miner.get_top_rules(top_n=5)
        
        assert isinstance(rules, pd.DataFrame)
        assert len(rules) <= 5
        assert 'lift' in rules.columns
        assert 'rule_description' in rules.columns
    
    def test_get_rules(self, sample_data):
        """测试获取规则对象."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target')
        miner.fit(df)
        
        rules = miner.get_rules(min_lift=1.0)
        
        assert isinstance(rules, list)
        
        if rules:
            from hscredit.core.rules.mining.base import MinedRule
            assert isinstance(rules[0], MinedRule)
    
    def test_get_rule_objects(self, sample_data):
        """测试获取Rule对象."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target')
        miner.fit(df)
        
        rules = miner.get_rule_objects(min_lift=1.0)
        
        assert isinstance(rules, list)
        if rules:
            assert isinstance(rules[0], Rule)
    
    def test_analyze_feature(self, sample_data):
        """测试分析单个特征."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target', method='quantile')
        miner.fit(df)
        
        result = miner.analyze_feature('feature_0')
        
        assert isinstance(result, pd.DataFrame)
        # 可能有结果为空的情况
        if not result.empty:
            assert 'lift' in result.columns
            assert 'badrate' in result.columns
    
    def test_get_feature_summary(self, sample_data):
        """测试特征摘要."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target')
        miner.fit(df)
        
        summary = miner.get_feature_summary()
        
        assert isinstance(summary, pd.DataFrame)
        if not summary.empty:
            assert 'max_lift' in summary.columns
            assert 'feature' in summary.columns
    
    def test_chi2_algorithm(self, sample_data):
        """测试卡方算法."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target', method='chi2')
        miner.fit(df)
        
        assert miner._is_fitted
        # 由于数据问题可能为空
        assert isinstance(miner.results_, dict)
    
    def test_best_iv_method(self, sample_data):
        """测试最优IV分箱方法."""
        df, _ = sample_data
        
        miner = SingleFeatureRuleMiner(target='target', method='best_iv', max_n_bins=5)
        miner.fit(df)
        
        assert miner._is_fitted
        assert isinstance(miner.results_, dict)


class TestMultiFeatureRuleMiner:
    """测试多特征交叉规则挖掘器."""
    
    def test_init(self):
        """测试初始化."""
        miner = MultiFeatureRuleMiner(target='target')
        assert miner.target == 'target'
        assert miner.method == 'quantile'
        assert miner.max_n_bins == 5
    
    def test_fit(self, sample_data):
        """测试拟合."""
        df, _ = sample_data
        
        miner = MultiFeatureRuleMiner(target='target')
        miner.fit(df)
        
        assert miner._is_fitted
    
    def test_generate_cross_matrix(self, sample_data):
        """测试生成交叉矩阵."""
        df, _ = sample_data
        
        miner = MultiFeatureRuleMiner(target='target')
        miner.fit(df)
        
        matrix = miner.generate_cross_matrix('feature_0', 'feature_1')
        
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.columns.nlevels == 2  # MultiIndex
    
    def test_get_cross_rules(self, sample_data):
        """测试获取交叉规则."""
        df, _ = sample_data
        
        miner = MultiFeatureRuleMiner(target='target')
        miner.fit(df)
        
        rules = miner.get_cross_rules('feature_0', 'feature_1', top_n=5)
        
        assert isinstance(rules, pd.DataFrame)
    
    def test_get_all_cross_rules(self, sample_data):
        """测试获取所有交叉规则."""
        df, _ = sample_data
        
        miner = MultiFeatureRuleMiner(target='target')
        miner.fit(df)
        
        rules = miner.get_all_cross_rules(top_n=5)
        
        assert isinstance(rules, pd.DataFrame)
    
    def test_plot_cross_heatmap(self, sample_data):
        """测试绘制热力图."""
        df, _ = sample_data
        
        miner = MultiFeatureRuleMiner(target='target')
        miner.fit(df)
        
        # 测试是否能运行不报错
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            fig = miner.plot_cross_heatmap('feature_0', 'feature_1')
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib未安装")


class TestTreeRuleExtractor:
    """测试决策树规则提取器."""
    
    def test_init(self):
        """测试初始化."""
        extractor = TreeRuleExtractor(algorithm='dt')
        assert extractor.algorithm == 'dt'
        assert extractor.max_depth == 5
        
        with pytest.raises(ValueError):
            TreeRuleExtractor(algorithm='invalid')
    
    def test_fit_dt(self, sample_data):
        """测试决策树拟合."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='dt', max_depth=3)
        extractor.fit(df)
        
        assert extractor.is_fitted_
        assert extractor.model_ is not None
    
    def test_fit_rf(self, sample_data):
        """测试随机森林拟合."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='rf', n_estimators=5)
        extractor.fit(df)
        
        assert extractor.is_fitted_
    
    def test_extract_rules_dt(self, sample_data):
        """测试从决策树提取规则."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='dt', max_depth=3)
        extractor.fit(df)
        rules = extractor.extract_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        # 检查规则结构
        if rules:
            assert 'conditions' in rules[0]
            assert 'predicted_class' in rules[0]
    
    def test_get_rules(self, sample_data):
        """测试获取规则对象."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='dt', max_depth=3)
        extractor.fit(df)
        
        rules = extractor.get_rules(top_n=10)
        
        assert isinstance(rules, list)
        
        if rules:
            from hscredit.core.rules.mining.base import MinedRule
            assert isinstance(rules[0], MinedRule)
    
    def test_get_rule_objects(self, sample_data):
        """测试获取Rule对象."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='dt', max_depth=3)
        extractor.fit(df)
        
        rules = extractor.get_rule_objects(top_n=10)
        
        assert isinstance(rules, list)
        if rules:
            assert isinstance(rules[0], Rule)
    
    def test_get_rules_dataframe(self, sample_data):
        """测试获取规则DataFrame."""
        df, _ = sample_data
        
        extractor = TreeRuleExtractor(algorithm='dt', max_depth=3)
        extractor.fit(df)
        
        rules_df = extractor.get_rules_dataframe(top_n=10)
        
        assert isinstance(rules_df, pd.DataFrame)
        assert 'rule' in rules_df.columns
    
    def test_kwargs_support(self, sample_data):
        """测试**kwargs参数支持."""
        df, _ = sample_data
        
        # 测试传入额外的sklearn参数
        extractor = TreeRuleExtractor(
            algorithm='rf',
            n_estimators=5,
            max_depth=3,
            criterion='entropy',  # 额外的sklearn参数
            bootstrap=True,
            oob_score=True
        )
        extractor.fit(df)
        
        assert extractor.is_fitted_
        assert hasattr(extractor.model_, 'oob_score_')


class TestRuleMetrics:
    """测试规则评估指标."""
    
    def test_init(self):
        """测试初始化."""
        metrics = RuleMetrics()
        assert metrics.target_positive == 1
    
    def test_calculate_metrics(self, sample_data):
        """测试计算指标."""
        df, _ = sample_data
        
        # 创建一个简单规则
        rule = Rule("feature_0 > 0", name="测试规则")
        
        metrics = RuleMetrics()
        result = metrics.evaluate_rule(
            rule,
            X_train=df.drop('target', axis=1),
            y_train=df['target']
        )
        
        assert isinstance(result, dict)
        assert 'train_lift' in result
        assert 'train_precision' in result
        assert 'train_recall' in result
    
    def test_calculate_metrics_with_test(self, train_test_data):
        """测试计算带测试集的指标."""
        train_df, test_df, _ = train_test_data
        
        rule = Rule("feature_0 > 0", name="测试规则")
        
        metrics = RuleMetrics()
        result = metrics.evaluate_rule(
            rule,
            X_train=train_df.drop('target', axis=1),
            y_train=train_df['target'],
            X_test=test_df.drop('target', axis=1),
            y_test=test_df['target']
        )
        
        assert 'test_lift' in result
        assert 'psi' in result
        assert 'badrate_diff' in result
    
    def test_evaluate_rules(self, sample_data):
        """测试批量评估规则."""
        df, _ = sample_data
        
        rules = [
            Rule("feature_0 > 0", name="规则1"),
            Rule("feature_1 < 0", name="规则2")
        ]
        
        metrics = RuleMetrics()
        results = metrics.evaluate_rules(
            rules,
            X_train=df.drop('target', axis=1),
            y_train=df['target']
        )
        
        # 返回的是DataFrame
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2


def test_calculate_rule_metrics_function(sample_data):
    """测试便捷函数."""
    df, _ = sample_data
    
    rule = Rule("feature_0 > 0", name="测试规则")
    
    result = calculate_rule_metrics(
        rule,
        df.drop('target', axis=1),
        df['target']
    )
    
    assert isinstance(result, dict)
    # 返回的指标带有train_前缀
    assert 'train_lift' in result


class TestTreeVisualizer:
    """测试决策树可视化器."""
    
    def test_init(self):
        """测试初始化."""
        from hscredit.core.rules.mining import TreeVisualizer
        visualizer = TreeVisualizer(feature_names=['a', 'b'])
        assert visualizer.feature_names == ['a', 'b']
    
    def test_plot_matplotlib(self, sample_data):
        """测试matplotlib可视化."""
        df, _ = sample_data
        
        from hscredit.core.rules.mining import TreeVisualizer
        
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(df.drop('target', axis=1), df['target'])
        
        visualizer = TreeVisualizer()
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            fig = visualizer.plot_matplotlib(tree, feature_names=list(df.drop('target', axis=1).columns))
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib未安装")
    
    def test_plot_feature_importance(self, sample_data):
        """测试特征重要性图."""
        df, _ = sample_data
        
        from hscredit.core.rules.mining import TreeVisualizer
        
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(df.drop('target', axis=1), df['target'])
        
        visualizer = TreeVisualizer()
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig = visualizer.plot_feature_importance(
                tree,
                feature_names=list(df.drop('target', axis=1).columns)
            )
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib未安装")
