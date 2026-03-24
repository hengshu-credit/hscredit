"""规则集分类模型测试."""

import pytest
import numpy as np
import pandas as pd

from hscredit.core.rules import Rule
from hscredit.core.models import (
    RuleSet, 
    RulesClassifier,  # 统一入口
    RuleSetClassifier,  # 别名
    LogicOperator,
    RuleResult,
    create_and_ruleset,
    create_or_ruleset,
    combine_rules
)


class TestRuleSet:
    """测试 RuleSet 类."""
    
    def test_basic_creation(self):
        """测试基本创建."""
        rule = Rule("age > 18")
        rs = RuleSet(name="测试规则集", logic="and", rules=[rule])
        
        assert rs.name == "测试规则集"
        assert rs.logic == LogicOperator.AND
        assert len(rs.rules) == 1
        
    def test_add_rule(self):
        """测试添加规则."""
        rs = RuleSet(name="测试")
        rule1 = Rule("age > 18")
        rule2 = Rule("income > 5000")
        
        rs.add_rule(rule1).add_rule(rule2)
        assert len(rs.rules) == 2
        
    def test_and_logic(self):
        """测试且逻辑."""
        df = pd.DataFrame({
            'age': [16, 20, 25, 17],
            'income': [3000, 6000, 4000, 7000]
        })
        
        rule1 = Rule("age > 18")
        rule2 = Rule("income > 5000")
        rs = RuleSet(name="且测试", logic="and", rules=[rule1, rule2])
        
        result, details = rs.evaluate(df)
        
        # 只有第2行(20, 6000)和第4行(17不符合，但25, 4000不符合)符合条件
        # 实际上：20>18且6000>5000 = True
        #        25>18且4000>5000 = False
        expected = np.array([False, True, False, False])
        assert np.array_equal(result, expected)
        
    def test_or_logic(self):
        """测试或逻辑."""
        df = pd.DataFrame({
            'age': [16, 20, 25, 17],
            'income': [3000, 6000, 4000, 7000]
        })
        
        rule1 = Rule("age > 18")
        rule2 = Rule("income > 5000")
        rs = RuleSet(name="或测试", logic="or", rules=[rule1, rule2])
        
        result, details = rs.evaluate(df)
        
        # 20>18, 6000>5000, 25>18, 7000>5000 = 第2,3,4行应该命中
        expected = np.array([False, True, True, True])
        assert np.array_equal(result, expected)
        
    def test_nested_ruleset(self):
        """测试嵌套规则集."""
        df = pd.DataFrame({
            'age': [16, 20, 25, 30],
            'income': [3000, 6000, 4000, 8000],
            'score': [500, 700, 600, 400]
        })
        
        # 创建内层规则集
        inner_rules = RuleSet(
            name="年龄收入规则",
            logic="and",
            rules=[Rule("age > 18"), Rule("income > 5000")]
        )
        
        # 创建外层规则集
        outer_rules = RuleSet(
            name="综合规则",
            logic="or",
            rules=[inner_rules, Rule("score < 500")]
        )
        
        result, details = outer_rules.evaluate(df)
        
        # inner: row2(20,6000), row4(30,8000) 命中
        # score<500: row4(400) 命中
        # or逻辑：row2, row4 命中
        expected = np.array([False, True, False, True])
        assert np.array_equal(result, expected)
        
    def test_ruleset_operators(self):
        """测试规则集操作符."""
        rs1 = RuleSet(name="rs1", logic="or", rules=[Rule("age > 18")])
        rs2 = RuleSet(name="rs2", logic="or", rules=[Rule("income > 5000")])
        
        # 测试与操作
        combined_and = rs1 & rs2
        assert combined_and.logic == LogicOperator.AND
        assert len(combined_and.rules) == 2
        
        # 测试或操作
        combined_or = rs1 | rs2
        assert combined_or.logic == LogicOperator.OR
        assert len(combined_or.rules) == 2


class TestRulesClassifier:
    """测试 RulesClassifier 类（统一入口）."""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据."""
        return pd.DataFrame({
            'age': [16, 20, 25, 30, 35, 40, 45, 50],
            'income': [3000, 6000, 4000, 8000, 10000, 5000, 7000, 9000],
            'score': [400, 700, 600, 800, 300, 650, 750, 550]
        })
    
    def test_basic_fit_predict(self, sample_data):
        """测试基本的拟合和预测."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[
                Rule("age < 20"),
                Rule("income > 7000"),
                Rule("score < 400")
            ],
            logic='or',
            output_mode='final'
        )
        
        clf.fit(X)
        result = clf.predict(X)
        
        # 验证输出形状
        assert len(result) == len(X)
        assert set(np.unique(result)).issubset({0, 1})
        
    def test_output_mode_individual(self, sample_data):
        """测试 individual 输出模式."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[
                Rule("age < 20", name="年轻"),
                Rule("income > 7000", name="高收入"),
            ],
            logic='or',
            output_mode='individual'
        )
        
        clf.fit(X)
        result = clf.predict(X)
        
        # 验证输出是DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(X), 2)
        
    def test_output_mode_both(self, sample_data):
        """测试 both 输出模式."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("age < 20"), Rule("income > 7000")],
            logic='or',
            output_mode='both'
        )
        
        clf.fit(X)
        final, individual = clf.predict(X)
        
        assert isinstance(final, np.ndarray)
        assert isinstance(individual, pd.DataFrame)
        assert len(final) == len(X)
        
    def test_output_mode_reason(self, sample_data):
        """测试 reason 输出模式."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[
                Rule("age < 20", name="年轻规则"),
                Rule("income > 7000", name="高收入规则"),
            ],
            logic='or',
            output_mode='reason'
        )
        
        clf.fit(X)
        result, reasons = clf.predict(X, return_reason=True)
        
        assert len(result) == len(X)
        assert len(reasons) == len(X)
        # 验证每个样本都有原因列表
        assert all(isinstance(r, list) for r in reasons)
        
    def test_and_logic_predict(self, sample_data):
        """测试且逻辑预测."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("age > 25"), Rule("income > 6000")],
            logic='and',
            output_mode='final'
        )
        
        clf.fit(X)
        result = clf.predict(X)
        
        # 手动验证
        # age>25: rows 3,4,5,6,7 (30,35,40,45,50)
        # income>6000: rows 1,3,4,6,7 (6000,8000,10000,7000,9000)
        # and: rows 3,4,6,7 (30,8000), (35,10000), (45,7000), (50,9000)
        expected = np.array([False, False, False, True, True, False, True, True])
        assert np.array_equal(result.astype(bool), expected)
        
    def test_with_ruleset_objects(self, sample_data):
        """测试使用 RuleSet 对象."""
        X = sample_data
        
        rs = RuleSet(
            name="高风险",
            logic="and",
            rules=[Rule("age < 25"), Rule("income > 5000")]
        )
        
        clf = RulesClassifier(
            rules=[rs, Rule("score < 350")],
            logic='or',
            output_mode='both'
        )
        
        clf.fit(X)
        final, individual = clf.predict(X)
        
        assert len(final) == len(X)
        assert '高风险' in individual.columns
        
    def test_predict_proba(self, sample_data):
        """测试概率预测."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("age < 20"), Rule("income > 7000")],
            logic='or',
            weights=[1.0, 2.0]  # 加权
        )
        
        clf.fit(X)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        
    def test_get_rule_summary(self, sample_data):
        """测试获取规则摘要."""
        rs = RuleSet(
            name="嵌套规则集",
            logic="and",
            rules=[Rule("age > 18"), Rule("income > 5000")]
        )
        
        clf = RulesClassifier(
            rules=[rs, Rule("score < 500")],
            logic='or'
        )
        
        clf.fit(sample_data)
        summary = clf.get_rule_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert '名称' in summary.columns
        assert '类型' in summary.columns
        assert len(summary) == 4  # 2个顶层 + 2个嵌套
        
    def test_validation_missing_columns(self, sample_data):
        """测试验证缺失列."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("nonexistent_column > 10")],
            logic='or'
        )
        
        with pytest.raises(ValueError, match="不存在的特征"):
            clf.fit(X)
            
    def test_add_rule_chaining(self, sample_data):
        """测试添加规则链式调用."""
        clf = RulesClassifier(logic='or')
        
        result = clf.add_rule(Rule("age > 18")).add_rule(Rule("income > 5000"))
        
        assert result is clf
        assert len(clf.rules) == 2
        
    def test_nested_ruleset_with_ruleset(self, sample_data):
        """测试规则集中的规则集嵌套."""
        X = sample_data
        
        # 创建多层嵌套
        inner = RuleSet(
            name="内层",
            logic="or",
            rules=[Rule("age < 20"), Rule("score > 700")]
        )
        
        middle = RuleSet(
            name="中层",
            logic="and",
            rules=[inner, Rule("income > 5000")]
        )
        
        outer = RuleSet(
            name="外层",
            logic="or",
            rules=[middle, Rule("score < 300")]
        )
        
        clf = RulesClassifier(rules=[outer], logic='or')
        clf.fit(X)
        
        result, details = outer.evaluate(X)
        
        # 验证结果
        assert len(result) == len(X)
        # 应该有详细结果
        assert len(details) > 0


class TestConvenienceFunctions:
    """测试便捷函数."""
    
    def test_create_and_ruleset(self):
        """测试 create_and_ruleset."""
        rules = [Rule("age > 18"), Rule("income > 5000")]
        rs = create_and_ruleset(rules, name="测试")
        
        assert rs.logic == LogicOperator.AND
        assert rs.name == "测试"
        
    def test_create_or_ruleset(self):
        """测试 create_or_ruleset."""
        rules = [Rule("age > 18"), Rule("income > 5000")]
        rs = create_or_ruleset(rules, name="测试")
        
        assert rs.logic == LogicOperator.OR
        assert rs.name == "测试"
        
    def test_combine_rules(self):
        """测试 combine_rules."""
        rule1 = Rule("age > 18")
        rule2 = Rule("income > 5000")
        rule3 = Rule("score > 600")
        
        rs = combine_rules(rule1, rule2, rule3, logic='and', name="组合规则")
        
        assert rs.logic == LogicOperator.AND
        assert len(rs.rules) == 3
        assert rs.name == "组合规则"


class TestRuleWithNewAttributes:
    """测试 Rule 类的新属性."""
    
    def test_rule_with_name(self):
        """测试带名称的规则."""
        rule = Rule("age > 18", name="成年规则", description="判断成年", weight=1.5)
        
        assert rule.name == "成年规则"
        assert rule.description == "判断成年"
        assert rule.weight == 1.5
        
    def test_combined_rule_preserves_attributes(self):
        """测试组合规则保留属性."""
        rule1 = Rule("age > 18", name="规则1", description="成年", weight=1.0)
        rule2 = Rule("income > 5000", name="规则2", description="高收入", weight=2.0)
        
        combined = rule1 & rule2
        
        assert "规则1" in combined.name
        assert "规则2" in combined.name
        assert combined.weight == 2.0  # max of both
        
    def test_default_name_is_expression(self):
        """测试默认名称为表达式."""
        rule = Rule("age > 18")
        assert rule.name == "age > 18"


class TestRulesClassifierNoFit:
    """测试 RulesClassifier 无需fit即可使用."""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据."""
        return pd.DataFrame({
            'age': [16, 20, 25, 30, 35, 40, 45, 50],
            'income': [3000, 6000, 4000, 8000, 10000, 5000, 7000, 9000],
            'score': [400, 700, 600, 800, 300, 650, 750, 550]
        })
    
    def test_predict_without_fit(self, sample_data):
        """测试无需fit直接predict."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("age < 20"), Rule("income > 7000")],
            logic='or'
        )
        
        # 不调用fit，直接predict
        result = clf.predict(X)
        
        assert len(result) == len(X)
        assert set(np.unique(result)).issubset({0, 1})
    
    def test_predict_proba_without_fit(self, sample_data):
        """测试无需fit直接predict_proba."""
        X = sample_data
        
        clf = RulesClassifier(
            rules=[Rule("age < 20"), Rule("income > 7000")],
            logic='or'
        )
        
        # 不调用fit，直接predict_proba
        proba = clf.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_single_rule_without_fit(self, sample_data):
        """测试单条规则无需fit."""
        X = sample_data
        
        clf = RulesClassifier(rules=Rule("age < 20"))
        result = clf.predict(X)
        
        assert len(result) == len(X)
        assert result[0] == 1  # 第一行满足条件
    
    def test_ruleset_without_fit(self, sample_data):
        """测试规则集无需fit."""
        X = sample_data
        
        rs = RuleSet(
            name="测试",
            logic="and",
            rules=[Rule("age < 25"), Rule("income > 5000")]
        )
        
        clf = RulesClassifier(rules=rs)
        result = clf.predict(X)
        
        assert len(result) == len(X)


class TestRulesClassifierFitModes:
    """测试 RulesClassifier fit的三种模式."""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据."""
        return pd.DataFrame({
            'age': [16, 20, 25, 30],
            'income': [3000, 6000, 4000, 8000],
            'target': [1, 0, 0, 1]
        })
    
    def test_fit_with_x_only(self, sample_data):
        """测试只传入X的fit."""
        X = sample_data.drop(columns=['target'])
        
        clf = RulesClassifier(rules=[Rule("age < 20")])
        clf.fit(X)
        
        assert hasattr(clf, 'n_features_in_')
        assert clf.n_features_in_ == 2
        assert hasattr(clf, '_is_fitted')
        assert clf._is_fitted
    
    def test_fit_with_x_and_y(self, sample_data):
        """测试传入X和y的fit."""
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        
        clf = RulesClassifier(rules=[Rule("age < 20")])
        clf.fit(X, y)
        
        assert hasattr(clf, 'classes_')
        assert set(clf.classes_) == {0, 1}
        assert clf.n_features_in_ == 2
    
    def test_fit_scorecardpipeline_style(self, sample_data):
        """测试scorecardpipeline风格的fit（target在DataFrame中）."""
        # 不分离target，直接传入完整DataFrame
        clf = RulesClassifier(rules=[Rule("age < 20")], target='target')
        clf.fit(sample_data)  # 自动从sample_data中提取target列
        
        assert hasattr(clf, 'classes_')
        assert clf.n_features_in_ == 2  # 排除target列
        assert 'target' not in clf.feature_names_in_


class TestRulesClassifierAdvanced:
    """测试 RulesClassifier 高级功能."""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据."""
        return pd.DataFrame({
            'age': [16, 20, 25, 30, 35, 40, 45, 50],
            'income': [3000, 6000, 4000, 8000, 10000, 5000, 7000, 9000],
            'score': [400, 700, 600, 800, 300, 650, 750, 550]
        })
    
    def test_single_rule_input(self, sample_data):
        """测试传入单条规则（非列表）."""
        X = sample_data
        
        # 传入单条规则而非列表
        clf = RulesClassifier(
            rules=Rule("age < 20", name="年轻"),
            logic='or'
        )
        
        clf.fit(X)
        result = clf.predict(X)
        
        assert len(result) == len(X)
        # 只有第一行(16)满足条件
        assert result[0] == 1
        assert sum(result) == 1
    
    def test_single_ruleset_input(self, sample_data):
        """测试传入单个规则集（非列表）."""
        X = sample_data
        
        rs = RuleSet(
            name="测试规则集",
            logic="and",
            rules=[Rule("age > 18"), Rule("income > 5000")]
        )
        
        # 传入单个规则集而非列表
        clf = RulesClassifier(rules=rs, logic='or')
        clf.fit(X)
        result = clf.predict(X)
        
        assert len(result) == len(X)
    
    def test_mixed_rules_and_rulesets(self, sample_data):
        """测试混合使用单规则和规则集."""
        X = sample_data
        
        # 混合使用
        rs = RuleSet(
            name="年龄收入规则",
            logic="and",
            rules=[Rule("age < 25"), Rule("income > 4000")]
        )
        
        clf = RulesClassifier(
            rules=[rs, Rule("score < 350", name="低分规则")],
            logic='or',
            output_mode='both'
        )
        
        clf.fit(X)
        final, individual = clf.predict(X)
        
        assert '年龄收入规则' in individual.columns
        assert '低分规则' in individual.columns
    
    def test_deeply_nested_ruleset(self, sample_data):
        """测试深层嵌套规则集."""
        X = sample_data
        
        # 创建三层嵌套
        level3 = RuleSet(
            name="第三层",
            logic="or",
            rules=[Rule("age < 20"), Rule("age > 60")]
        )
        
        level2 = RuleSet(
            name="第二层",
            logic="and",
            rules=[level3, Rule("income > 5000")]
        )
        
        level1 = RuleSet(
            name="第一层",
            logic="or",
            rules=[level2, Rule("score < 400")]
        )
        
        clf = RulesClassifier(rules=[level1], logic='or', verbose=True)
        clf.fit(X)
        
        summary = clf.get_rule_summary()
        
        # 验证层级结构
        # 第一层(1) + 第二层(1) + 第三层(1) + age<20 + age>60 + income>5000 + score<400 = 7
        assert len(summary) == 7
        assert 0 in summary['层级'].values
        assert 1 in summary['层级'].values
        assert 2 in summary['层级'].values
        assert 3 in summary['层级'].values
    
    def test_weights_validation(self, sample_data):
        """测试权重验证."""
        X = sample_data
        
        # 权重数量与规则数量不匹配
        clf = RulesClassifier(
            rules=[Rule("age < 20"), Rule("income > 7000")],
            weights=[1.0]  # 只有一个权重，但有两条规则
        )
        
        with pytest.raises(ValueError, match="weights长度"):
            clf.fit(X)
    
    def test_count_total_rules(self, sample_data):
        """测试统计总规则数."""
        # 嵌套规则集
        inner = RuleSet(
            name="内层",
            logic="or",
            rules=[Rule("age < 20"), Rule("score > 700")]
        )
        
        outer = RuleSet(
            name="外层",
            logic="and",
            rules=[inner, Rule("income > 5000")]
        )
        
        clf = RulesClassifier(rules=[outer, Rule("score < 300")])
        clf.fit(sample_data)
        
        # 总计: inner(2) + outer的income规则(1) + 外层单规则(1) = 4
        assert clf._count_total_rules() == 4
    
    def test_rules_classifier_alias(self, sample_data):
        """测试 RuleSetClassifier 是 RulesClassifier 的别名."""
        # 验证两者是同一个类
        assert RuleSetClassifier is RulesClassifier
        
        # 验证可以使用 RuleSetClassifier 创建实例
        clf = RuleSetClassifier(
            rules=[Rule("age < 20")],
            logic='or'
        )
        clf.fit(sample_data)
        result = clf.predict(sample_data)
        
        assert len(result) == len(sample_data)


class TestConvenienceFunctionsAdvanced:
    """测试便捷函数高级功能."""
    
    def test_create_and_ruleset_with_description(self):
        """测试 create_and_ruleset 带描述."""
        rules = [Rule("age > 18"), Rule("income > 5000")]
        rs = create_and_ruleset(rules, name="测试", description="测试描述")
        
        assert rs.logic == LogicOperator.AND
        assert rs.name == "测试"
        assert rs.description == "测试描述"
        
    def test_create_or_ruleset_with_description(self):
        """测试 create_or_ruleset 带描述."""
        rules = [Rule("age > 18"), Rule("income > 5000")]
        rs = create_or_ruleset(rules, name="测试", description="测试描述")
        
        assert rs.logic == LogicOperator.OR
        assert rs.name == "测试"
        assert rs.description == "测试描述"
        
    def test_combine_rules_with_description(self):
        """测试 combine_rules 带描述."""
        rule1 = Rule("age > 18")
        rule2 = Rule("income > 5000")
        
        rs = combine_rules(rule1, rule2, logic='or', name="组合规则", description="组合描述")
        
        assert rs.logic == LogicOperator.OR
        assert len(rs.rules) == 2
        assert rs.name == "组合规则"
        assert rs.description == "组合描述"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
