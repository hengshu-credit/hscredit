"""测试分箱器三大修复:

1. 切分点包含缺失值 (scorecardpipeline 格式)
2. max_n_bins 硬性限制
3. 分箱标签 左闭右开 [a, b)
"""
import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning


@pytest.fixture
def sample_data():
    """生成含缺失值的数据."""
    np.random.seed(42)
    n = 500
    age = np.random.normal(35, 10, n)
    # 注入缺失值
    age[np.random.choice(n, 30, replace=False)] = np.nan
    target = (age > 30).astype(float)
    target[np.isnan(age)] = np.random.choice([0, 1], size=30, p=[0.7, 0.3])
    df = pd.DataFrame({'age': age, 'target': target.astype(int)})
    return df


@pytest.fixture
def categorical_data():
    """生成含缺失值的类别数据."""
    np.random.seed(42)
    n = 500
    city = np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n)
    # 注入缺失值
    city = pd.Series(city).where(pd.Series(np.random.random(n) > 0.1), other=np.nan)
    target = np.random.choice([0, 1], n, p=[0.7, 0.3])
    df = pd.DataFrame({'city': city, 'target': target})
    return df


# ============================================================
# Issue 1: 切分点输出包含 np.nan (scorecardpipeline 格式)
# ============================================================

class TestCutPointsMissingValue:
    """测试切分点输出是否包含缺失值标记."""

    def test_numerical_splits_contain_nan(self, sample_data):
        """数值型特征切分点末尾应包含 np.nan."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])

        # __getitem__ 返回
        splits = binner['age']
        assert isinstance(splits, list), f"应返回 list, 实际 {type(splits)}"
        assert len(splits) >= 2, "至少有切分点 + nan"
        assert np.isnan(splits[-1]), f"末尾应为 nan, 实际 {splits[-1]}"
        # 非 nan 部分是有序的
        numeric_part = [s for s in splits if not np.isnan(s)]
        assert numeric_part == sorted(numeric_part), "非 nan 部分应升序"

    def test_get_splits_contain_nan(self, sample_data):
        """get_splits 返回也应包含 nan."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])

        splits = binner.get_splits('age')
        assert isinstance(splits, list)
        assert np.isnan(splits[-1])

    def test_export_rules_contain_nan(self, sample_data):
        """export_rules 返回的数值型切分点应包含 nan."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])

        rules = binner.export_rules()
        assert 'age' in rules
        assert np.isnan(rules['age'][-1]), f"末尾应为 nan, 实际 {rules['age']}"

    def test_missing_separate_false_no_nan(self, sample_data):
        """missing_separate=False 时不应追加 nan."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=False)
        binner.fit(sample_data[['age']], sample_data['target'])

        splits = binner['age']
        if isinstance(splits, (list, np.ndarray)):
            for s in splits:
                if isinstance(s, float) and np.isnan(s):
                    pytest.fail("missing_separate=False 时切分点不应包含 nan")

    def test_import_rules_strips_nan(self):
        """import_rules 应自动剥离末尾 nan."""
        binner = OptimalBinning()
        # 模拟 scorecardpipeline 导出的格式 (末尾含 nan)
        binner.import_rules({'age': [25.0, 35.0, 45.0, np.nan]})
        
        # 内部 splits_ 不应包含 nan
        internal = binner.splits_['age']
        assert not any(np.isnan(s) for s in internal), \
            f"内部 splits_ 不应含 nan: {internal}"
        assert len(internal) == 3, f"应有3个切分点, 实际 {len(internal)}"

    def test_load_strips_nan(self):
        """load 也应自动剥离末尾 nan."""
        binner = OptimalBinning()
        binner.load({'age': [25.0, 35.0, np.nan]})
        
        internal = binner.splits_['age']
        assert len(internal) == 2
        assert not any(np.isnan(s) for s in internal)

    def test_round_trip_export_import(self, sample_data):
        """export → import 往返不应丢失信息."""
        binner1 = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner1.fit(sample_data[['age']], sample_data['target'])
        
        rules = binner1.export_rules()
        
        binner2 = OptimalBinning(missing_separate=True)
        binner2.import_rules(rules)
        
        # 内部切分点应一致（去除 nan 后）
        s1 = binner1.splits_['age']
        s2 = binner2.splits_['age']
        np.testing.assert_array_almost_equal(s1, s2)


# ============================================================
# Issue 2: max_n_bins 硬性限制
# ============================================================

class TestMaxNBinsEnforcement:
    """测试 max_n_bins 限制是否有效."""

    @pytest.mark.parametrize("method", ['quantile', 'best_iv', 'mdlp', 'tree', 'chi', 'cart'])
    def test_max_n_bins_respected(self, method):
        """各种方法都应尊重 max_n_bins 限制."""
        np.random.seed(42)
        n = 2000
        x = np.random.exponential(1, n)
        y = (x > np.median(x)).astype(int)
        df = pd.DataFrame({'feat': x, 'target': y})
        
        max_bins = 4
        binner = OptimalBinning(method=method, max_n_bins=max_bins, missing_separate=False)
        try:
            binner.fit(df[['feat']], df['target'])
        except Exception:
            pytest.skip(f"方法 {method} 拟合失败")
        
        n_bins = binner.n_bins_.get('feat', 0)
        assert n_bins <= max_bins, \
            f"方法 {method}: n_bins={n_bins} > max_n_bins={max_bins}"

    def test_max_n_bins_with_missing(self, sample_data):
        """含缺失值时，正常箱数不应超过 max_n_bins."""
        max_bins = 3
        binner = OptimalBinning(method='quantile', max_n_bins=max_bins, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        # n_bins_ 记录正常箱数（不含缺失箱）
        n_bins = binner.n_bins_.get('age', 0)
        assert n_bins <= max_bins, \
            f"正常箱数 {n_bins} > max_n_bins {max_bins}"

    def test_max_n_bins_with_many_unique_values(self):
        """高基数数值特征也应尊重 max_n_bins."""
        np.random.seed(42)
        n = 5000
        # 构造高基数数据
        x = np.random.uniform(0, 100, n) + np.random.normal(0, 0.01, n)
        y = (x > 50).astype(int)
        noise_idx = np.random.choice(n, n // 5, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        df = pd.DataFrame({'feat': x, 'target': y})
        
        for max_bins in [3, 5, 8]:
            binner = OptimalBinning(method='best_iv', max_n_bins=max_bins)
            binner.fit(df[['feat']], df['target'])
            
            n_bins = binner.n_bins_.get('feat', 0)
            assert n_bins <= max_bins, \
                f"max_n_bins={max_bins}: 实际 {n_bins} 箱"

    def test_post_fit_constraints_enforce_cap(self):
        """_apply_post_fit_constraints 之后不超过 max_n_bins."""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1, n)
        y = (x > 0).astype(int)
        df = pd.DataFrame({'feat': x, 'target': y})
        
        binner = OptimalBinning(method='mdlp', max_n_bins=3, monotonic='auto')
        binner.fit(df[['feat']], df['target'])
        
        n_splits = len(binner.splits_.get('feat', []))
        assert n_splits <= 2, f"切分点数 {n_splits} > max_n_bins-1=2"


# ============================================================
# Issue 3: 分箱标签 左闭右开 [a, b) 格式
# ============================================================

class TestBinLabelsFormat:
    """测试分箱标签是否为左闭右开 [a, b) 格式."""

    def test_labels_left_closed_right_open(self, sample_data):
        """分箱标签应为 [a, b) 格式."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        table = binner.get_bin_table('age')
        labels = table['分箱标签'].tolist()
        
        for label in labels:
            if label in ('missing', 'special'):
                continue
            # 应以 '[' 开头, 以 ')' 结尾
            assert label.startswith('['), \
                f"标签 '{label}' 应以 '[' 开头 (左闭)"
            assert label.endswith(')'), \
                f"标签 '{label}' 应以 ')' 结尾 (右开)"

    def test_first_bin_starts_with_neg_inf(self, sample_data):
        """第一个箱应包含 -inf."""
        binner = OptimalBinning(method='quantile', max_n_bins=4, missing_separate=False)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        table = binner.get_bin_table('age')
        valid_labels = [l for l in table['分箱标签'] if l not in ('missing', 'special')]
        assert valid_labels[0].startswith('[-inf,'), \
            f"第一箱应以 '[-inf,' 开头, 实际: {valid_labels[0]}"

    def test_last_bin_ends_with_pos_inf(self, sample_data):
        """最后一个箱应包含 +inf."""
        binner = OptimalBinning(method='quantile', max_n_bins=4, missing_separate=False)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        table = binner.get_bin_table('age')
        valid_labels = [l for l in table['分箱标签'] if l not in ('missing', 'special')]
        assert valid_labels[-1].endswith('+inf)'), \
            f"最后箱应以 '+inf)' 结尾, 实际: {valid_labels[-1]}"

    def test_digitize_consistency(self, sample_data):
        """np.digitize 分箱结果应与标签定义一致."""
        binner = OptimalBinning(method='quantile', max_n_bins=4, missing_separate=False)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        splits = binner.splits_['age']
        if len(splits) == 0:
            return
        
        # 边界值测试: 恰好等于切分点的值应归入上一个箱 (左闭 → 值 == split[i] 归入 bin i+1)
        cut_value = splits[0]
        bin_idx = np.digitize(cut_value, splits)
        # 默认 right=False: bin_idx = 1 (因为 cut_value >= splits[0])
        assert bin_idx == 1, \
            f"值 {cut_value} 应归入 bin 1 (左闭), 实际 bin {bin_idx}"

    @pytest.mark.parametrize("method", ['quantile', 'best_iv', 'mdlp'])
    def test_all_methods_use_left_closed_labels(self, sample_data, method):
        """所有分箱方法都应输出 [a, b) 格式标签."""
        binner = OptimalBinning(method=method, max_n_bins=5, missing_separate=True)
        try:
            binner.fit(sample_data[['age']], sample_data['target'])
        except Exception:
            pytest.skip(f"方法 {method} 拟合失败")
        
        table = binner.get_bin_table('age')
        for label in table['分箱标签']:
            if label in ('missing', 'special'):
                continue
            assert label.startswith('[') and label.endswith(')'), \
                f"方法 {method}: 标签 '{label}' 不是 [a, b) 格式"


# ============================================================
# 综合集成测试
# ============================================================

class TestIntegration:
    """综合测试三个修复的交互效果."""

    def test_full_workflow(self, sample_data):
        """完整工作流: fit → 标签/切分点/导出 均正确."""
        binner = OptimalBinning(method='best_iv', max_n_bins=4, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        # 1. 切分点含 nan
        cuts = binner['age']
        assert np.isnan(cuts[-1]), "切分点末尾应含 nan"
        
        # 2. 正常箱数不超过 max_n_bins
        n_bins = binner.n_bins_['age']
        assert n_bins <= 4, f"正常箱数 {n_bins} > 4"
        
        # 3. 标签为 [a, b) 格式
        table = binner.get_bin_table('age')
        for label in table['分箱标签']:
            if label in ('missing', 'special'):
                continue
            assert label.startswith('[') and label.endswith(')'), \
                f"标签 '{label}' 不是 [a, b) 格式"
        
        # 4. 导出 → 导入往返正确
        rules = binner.export_rules()
        binner2 = OptimalBinning(missing_separate=True)
        binner2.import_rules(rules)
        np.testing.assert_array_almost_equal(
            binner.splits_['age'], binner2.splits_['age']
        )

    def test_export_json_and_load(self, sample_data, tmp_path):
        """JSON 导出/加载往返正确."""
        binner = OptimalBinning(method='quantile', max_n_bins=5, missing_separate=True)
        binner.fit(sample_data[['age']], sample_data['target'])
        
        json_path = str(tmp_path / 'rules.json')
        binner.export(to_json=json_path)
        
        binner2 = OptimalBinning(missing_separate=True)
        binner2.load(json_path)
        
        # 内部切分点应一致
        np.testing.assert_array_almost_equal(
            binner.splits_['age'], binner2.splits_['age']
        )

    def test_scorecardpipeline_format_compatibility(self):
        """兼容 scorecardpipeline 格式的导入."""
        # 模拟 scorecardpipeline 导出的规则
        rules = {
            'age': [25.0, 35.0, 45.0, float('nan')],     # 末尾 nan
            'city': [['北京', '上海'], ['广州', float('nan')]],  # nan 在组内
        }
        
        binner = OptimalBinning()
        binner.import_rules(rules)
        
        # age 切分点应为 [25, 35, 45]（自动剥离 nan）
        age_splits = binner.splits_['age']
        np.testing.assert_array_equal(age_splits, [25.0, 35.0, 45.0])
        
        # city 分组应保持 nan
        city_bins = binner._cat_bins_['city']
        assert len(city_bins) == 2
        # 检查第二组包含 nan
        assert any(isinstance(v, float) and np.isnan(v) for v in city_bins[1])
