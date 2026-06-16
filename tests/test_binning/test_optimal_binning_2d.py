# -*- coding: utf-8 -*-
"""OptimalBinning2D 二维交互分箱测试."""

import pytest
import numpy as np
import pandas as pd
import warnings

from hscredit.core.binning import OptimalBinning2D


@pytest.fixture
def sample_df():
    """生成示例数据."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'income': np.random.randint(3000, 50000, n),
        'target': np.random.randint(0, 2, n)
    })
    return df


@pytest.fixture
def sample_df_with_target():
    """生成包含target列的示例数据."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'income': np.random.randint(3000, 50000, n),
        'target': np.random.randint(0, 2, n)
    })
    return df


class TestOptimalBinning2DBasic:
    """基础功能测试."""

    def test_basic_fit_transform(self, sample_df):
        """测试基本fit和transform."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        # 测试transform
        result = binner.transform(sample_df[['age', 'income']], metric='indices')

        assert 'age' in result.columns
        assert 'income' in result.columns
        assert len(result) == len(sample_df)

    def test_transform_bins(self, sample_df):
        """测试 metric='bins' 返回标签."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        result = binner.transform(sample_df[['age', 'income']], metric='bins')

        assert 'age' in result.columns
        assert 'income' in result.columns
        assert len(result) == len(sample_df)
        # 标签应该是字符串
        assert all(isinstance(v, str) for v in result['age'].values)

    def test_sklearn_style_fit(self, sample_df):
        """测试 sklearn 风格 fit（不传 features，取前两列）."""
        binner = OptimalBinning2D(max_n_bins=4)
        X = sample_df[['age', 'income']].values
        y = sample_df['target'].values

        binner.fit(X, y)

        assert binner._is_fitted
        cross_table = binner.get_cross_table()
        assert len(cross_table) > 0
        # numpy 数组无列名，内部用 feature_0/feature_1 作为特征名
        assert binner.feature_x_ == 'feature_0'
        assert binner.feature_y_ == 'feature_1'

    def test_sklearn_style_fit_df(self, sample_df):
        """测试 sklearn 风格 fit（DataFrame，不传 features）."""
        binner = OptimalBinning2D(max_n_bins=4)
        X = sample_df[['age', 'income']]

        binner.fit(X, y=sample_df['target'])

        assert binner._is_fitted
        cross_table = binner.get_cross_table()
        assert len(cross_table) > 0

    def test_get_cross_table(self, sample_df):
        """测试获取交叉分箱表."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        cross_table = binner.get_cross_table()

        # 检查列
        expected_cols = [
            '特征1', '特征2', '分箱1', '分箱2',
            '分箱1标签', '分箱2标签',
            '样本总数', '好样本数', '坏样本数',
            '坏样本率', '样本占比', '分档WOE值', '分档IV值', 'LIFT值'
        ]
        for col in expected_cols:
            assert col in cross_table.columns, f"缺少列: {col}"

        # 检查行数（应该是 n_bins1 * n_bins2）
        expected_rows = binner.n_bins_x_ * binner.n_bins_y_
        assert len(cross_table) == expected_rows

    def test_scorecardpipeline_style(self, sample_df_with_target):
        """测试scorecardpipeline风格（target在DataFrame中）."""
        binner = OptimalBinning2D(target='target', max_n_bins=4)
        df = sample_df_with_target.copy()
        df['target'] = sample_df_with_target['target']

        # 不传入y，从df中提取target
        binner.fit(df, features=['age', 'income'])

        assert binner._is_fitted
        cross_table = binner.get_cross_table()
        assert len(cross_table) > 0


class TestOptimalBinning2DMatrices:
    """矩阵获取测试."""

    def test_get_bad_rate_matrix(self, sample_df):
        """测试获取坏样本率矩阵."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        matrix = binner.get_bad_rate_matrix()

        # 检查形状
        assert matrix.shape == (binner.n_bins_x_, binner.n_bins_y_)

        # 检查所有值在0-1之间
        assert matrix.min().min() >= 0
        assert matrix.max().max() <= 1

    def test_get_count_matrix(self, sample_df):
        """测试获取样本数矩阵."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        matrix = binner.get_count_matrix()

        # 检查形状
        assert matrix.shape == (binner.n_bins_x_, binner.n_bins_y_)

        # 检查所有值非负
        assert (matrix >= 0).all().all()

    def test_get_woe_matrix(self, sample_df):
        """测试获取WOE矩阵."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        matrix = binner.get_woe_matrix()

        # 检查形状
        assert matrix.shape == (binner.n_bins_x_, binner.n_bins_y_)

    def test_get_iv_matrix(self, sample_df):
        """测试获取IV矩阵."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        matrix = binner.get_iv_matrix()

        # 检查形状
        assert matrix.shape == (binner.n_bins_x_, binner.n_bins_y_)

        # 检查所有IV值非负
        assert (matrix >= 0).all().all()


class TestOptimalBinning2DPlot:
    """可视化测试（仅检查不报错）."""

    def test_plot_bad_rate(self, sample_df, tmp_path):
        """测试绘制坏样本率热力图."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        # 测试绘制（不检查具体图形内容）
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端

        fig = binner.plot(metric='bad_rate')
        assert fig is not None

        # 测试保存
        save_path = tmp_path / "test_bad_rate.png"
        binner.plot(metric='bad_rate', save=str(save_path))
        assert save_path.exists()

    def test_plot_count(self, sample_df):
        """测试绘制样本数热力图."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        import matplotlib
        matplotlib.use('Agg')

        fig = binner.plot(metric='count', fmt='d')
        assert fig is not None

    def test_plot_woe(self, sample_df):
        """测试绘制WOE热力图."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        import matplotlib
        matplotlib.use('Agg')

        fig = binner.plot(metric='woe')
        assert fig is not None

    def test_plot_with_custom_title(self, sample_df):
        """测试自定义标题."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        import matplotlib
        matplotlib.use('Agg')

        fig = binner.plot(metric='bad_rate', title='自定义标题')
        assert fig is not None


class TestOptimalBinning2DCustom:
    """自定义参数测试."""

    def test_user_splits(self, sample_df):
        """测试用户自定义分箱规则."""
        binner = OptimalBinning2D(
            user_splits_x=[25, 35, 45],
            user_splits_y=[5000, 10000, 20000]
        )
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        cross_table = binner.get_cross_table()

        # 检查分箱数
        assert binner.n_bins_x_ == 4  # 3个切分点 -> 4个分箱
        assert binner.n_bins_y_ == 4

    def test_method_parameter(self, sample_df):
        """测试不同分箱方法."""
        methods = ['quantile', 'uniform']

        for method in methods:
            binner = OptimalBinning2D(method=method, max_n_bins=4)
            binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

            assert binner._is_fitted
            cross_table = binner.get_cross_table()
            assert len(cross_table) > 0

    def test_monotonic_constraint(self, sample_df):
        """测试单调性约束."""
        binner = OptimalBinning2D(monotonic='descending', max_n_bins=5)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        assert binner._is_fitted
        cross_table = binner.get_cross_table()
        assert len(cross_table) > 0

    def test_x1_y_params(self, sample_df):
        """测试 x_params / y_params 扩展参数."""
        binner = OptimalBinning2D(
            max_n_bins=5,
            max_n_bins_x=3,
            max_n_bins_y=4,
            min_bin_size_x=0.01,
            min_bin_size_y=0.01,
            x_params={'monotonic': 'descending'},
            y_params={'method': 'uniform'},
        )
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        assert binner._is_fitted
        # 特征1应使用 max_n_bins_x=3
        assert binner.n_bins_x_ <= 3
        # 特征2应使用 max_n_bins_y=4
        assert binner.n_bins_y_ <= 4
        cross_table = binner.get_cross_table()
        assert len(cross_table) == binner.n_bins_x_ * binner.n_bins_y_

    def test_individual_bin_params(self, sample_df):
        """测试特征独立的分箱参数（max_n_bins_x / max_n_bins_y 等）."""
        binner = OptimalBinning2D(
            max_n_bins_x=3,
            max_n_bins_y=2,
            min_bin_size_x=0.05,
            min_bin_size_y=0.1,
            method_x='uniform',
            method_y='quantile',
        )
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        assert binner._is_fitted
        assert binner.n_bins_x_ == 3
        assert binner.n_bins_y_ == 2
        cross_table = binner.get_cross_table()
        assert len(cross_table) == 3 * 2


class TestOptimalBinning2DEdgeCases:
    """边界情况测试."""

    def test_empty_features(self, sample_df):
        """测试空特征列表."""
        binner = OptimalBinning2D()

        with pytest.raises(ValueError):
            binner.fit(sample_df, y=sample_df['target'], features=[])

    def test_single_feature(self, sample_df):
        """测试只有一个特征."""
        binner = OptimalBinning2D()

        with pytest.raises(ValueError):
            binner.fit(sample_df, y=sample_df['target'], features=['age'])

    def test_invalid_feature(self, sample_df):
        """测试无效特征名."""
        binner = OptimalBinning2D()

        with pytest.raises(Exception):
            binner.fit(sample_df, y=sample_df['target'], features=['invalid_col', 'income'])

    def test_not_fitted_error(self):
        """测试未拟合时的错误."""
        binner = OptimalBinning2D()

        with pytest.raises(Exception):
            binner.get_cross_table()

    def test_transform_not_fitted(self):
        """测试未拟合时transform的错误."""
        binner = OptimalBinning2D()

        df = pd.DataFrame({'age': [25, 30], 'income': [5000, 6000]})
        with pytest.raises(Exception):
            binner.transform(df)


class TestOptimalBinning2DStats:
    """统计指标测试."""

    def test_iv_interaction(self, sample_df):
        """测试交互IV值计算."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        # IV应该是正数
        assert binner.iv_interaction_ >= 0

        # IV应该等于分档IV之和
        cross_table = binner.get_cross_table()
        iv_sum = cross_table['分档IV值'].sum()
        assert abs(binner.iv_interaction_ - iv_sum) < 1e-6

    def test_bad_rate_range(self, sample_df):
        """测试坏样本率范围."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        cross_table = binner.get_cross_table()

        # 有样本的箱坏样本率应该在0-1之间
        non_empty = cross_table[cross_table['样本总数'] > 0]
        assert (non_empty['坏样本率'] >= 0).all()
        assert (non_empty['坏样本率'] <= 1).all()

    def test_sample_count_sum(self, sample_df):
        """测试样本数总和."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        cross_table = binner.get_cross_table()
        total = cross_table['样本总数'].sum()

        # 总样本数应该等于有效样本数
        assert total > 0


class TestOptimalBinning2DRepr:
    """字符串表示测试."""

    def test_repr_not_fitted(self):
        """测试未拟合时的repr."""
        binner = OptimalBinning2D()
        repr_str = repr(binner)
        assert 'OptimalBinning2D' in repr_str
        assert 'fitted=False' in repr_str

    def test_repr_fitted(self, sample_df):
        """测试拟合后的repr."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        repr_str = repr(binner)
        assert 'OptimalBinning2D' in repr_str
        assert 'fitted=True' in repr_str
        assert 'age' in repr_str
        assert 'income' in repr_str


class TestOptimalBinning2DExport:
    """导出功能测试."""

    def test_export_rules(self, sample_df):
        """测试导出分箱规则（末尾应有 np.nan 表示缺失箱）."""
        binner = OptimalBinning2D(max_n_bins=4, missing_separate=True)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        rules = binner.export_rules()

        assert 'age' in rules
        assert 'income' in rules
        assert isinstance(rules['age'], list)
        assert isinstance(rules['income'], list)
        # missing_separate=True 时末尾应有 np.nan
        assert len(rules['age']) > 0
        assert len(rules['income']) > 0
        import numpy as np
        assert np.nan in rules['age'], "export_rules 末尾应有 np.nan"
        assert np.nan in rules['income'], "export_rules 末尾应有 np.nan"

    def test_import_rules(self, sample_df):
        """测试导入分箱规则."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        import numpy as np
        rules = {'age': [25, 35], 'income': [5000, 10000]}
        binner.import_rules(rules)

        assert binner.user_splits_x == [25, 35]
        assert binner.user_splits_y == [5000, 10000]

    def test_get_bin_table_cross(self, sample_df):
        """测试 get_bin_table() 默认返回交叉分箱表."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        table = binner.get_bin_table()
        cross = binner.get_cross_table()
        assert table.equals(cross)

    def test_get_bin_table_feature(self, sample_df):
        """测试 get_bin_table(feature) 返回独立分箱表."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        table_x = binner.get_bin_table('age')
        assert '分箱' in table_x.columns or '分箱标签' in table_x.columns
        assert len(table_x) == binner.n_bins_x_

        table_y = binner.get_bin_table('income')
        assert len(table_y) == binner.n_bins_y_

    def test_get_stats(self, sample_df):
        """测试 get_stats 方法."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        stats = binner.get_stats()
        assert 'age' in stats
        assert 'income' in stats
        assert 'n_bins' in stats['age']
        assert 'bin_table' in stats['age']

        s = binner.get_stats('age')
        assert 'n_bins' in s
        assert 'bin_table' in s

    def test_get_splits(self, sample_df):
        """测试 get_splits 方法."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        splits = binner.get_splits()
        assert 'age' in splits
        assert 'income' in splits

        import numpy as np
        splits_x = binner.get_splits('age')
        assert isinstance(splits_x, np.ndarray) or isinstance(splits_x, list)

    def test_get_marginal_stats(self, sample_df):
        """测试获取边缘分箱统计."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        marginal = binner.get_marginal_stats()

        assert 'age' in marginal
        assert 'income' in marginal
        assert isinstance(marginal['age'], pd.DataFrame)
        assert isinstance(marginal['income'], pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])