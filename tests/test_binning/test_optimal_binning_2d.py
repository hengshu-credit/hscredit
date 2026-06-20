# -*- coding: utf-8 -*-
"""OptimalBinning2D 二维交互分箱测试."""

import pytest
import numpy as np
import pandas as pd

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

        assert list(result.columns) == ['ageXincome']
        assert len(result) == len(sample_df)
        assert result['ageXincome'].between(0, binner.n_bins_2d_ - 1).all()

    def test_transform_bins(self, sample_df):
        """测试 metric='bins' 返回标签."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        result = binner.transform(sample_df[['age', 'income']], metric='bins')

        assert list(result.columns) == ['ageXincome']
        assert len(result) == len(sample_df)
        # 标签应该是字符串
        assert all(isinstance(v, str) for v in result['ageXincome'].values)

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
            '分箱', '分箱标签',
            '特征1名称', '特征1分箱', '特征1标签',
            '特征2名称', '特征2分箱', '特征2标签',
            '样本总数', '好样本数', '坏样本数',
            '坏样本率', '样本占比', '分档WOE值', '分档IV值', 'LIFT值'
        ]
        for col in expected_cols:
            assert col in cross_table.columns, f"缺少列: {col}"

        bin_table = binner.get_bin_table()
        assert list(cross_table.columns[8:]) == list(bin_table.columns[4:])

        # 检查行数（应该是 n_bins1 * n_bins2）
        expected_rows = binner.n_bins_x_ * binner.n_bins_y_
        assert len(cross_table) == expected_rows

    def test_both_tables_use_shared_compute_bin_stats(self, sample_df):
        binner = OptimalBinning2D(max_n_bins=4)
        original = binner._compute_bin_stats
        calls = []

        def spy(*args, **kwargs):
            calls.append(np.asarray(args[0]).copy())
            return original(*args, **kwargs)

        binner._compute_bin_stats = spy
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        assert len(calls) == 2
        assert list(binner.get_cross_table().columns[8:]) == list(binner.get_bin_table().columns[4:])

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

        # IV应该等于合并后二维分箱的分档IV之和
        binning_table = binner.get_bin_table()
        iv_sum = binning_table.loc[binning_table['分箱'] >= 0, '分档IV值'].sum()
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

        rules = {'age': [25, 35], 'income': [5000, 10000]}
        binner.import_rules(rules)

        assert binner.user_splits_x == [25, 35]
        assert binner.user_splits_y == [5000, 10000]

    def test_get_bin_table_cross(self, sample_df):
        """测试 get_bin_table() 默认返回合并后二维分箱表."""
        binner = OptimalBinning2D(max_n_bins=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        table = binner.get_bin_table()
        cross = binner.get_cross_table()
        assert len(table[table['分箱'] >= 0]) == binner.n_bins_2d_
        assert list(table.columns[:6]) == [
            '指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比'
        ]
        assert table['指标名称'].eq('ageXincome').all()
        assert table['指标含义'].isna().all()
        assert {'分箱', '分箱标签'}.issubset(cross.columns)


class TestOptimalBinning2DMerge:
    """最终二维合并分箱测试."""

    def test_merge_limit_and_connected_regions(self, sample_df):
        binner = OptimalBinning2D(max_n_bins=5, max_n_bins_2d=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        assert 1 <= binner.n_bins_2d_ <= 4
        for bin_id in range(binner.n_bins_2d_):
            cells = set(map(tuple, np.argwhere(binner.solution_ == bin_id)))
            visited = {next(iter(cells))}
            pending = list(visited)
            while pending:
                i, j = pending.pop()
                for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if neighbour in cells and neighbour not in visited:
                        visited.add(neighbour)
                        pending.append(neighbour)
            assert visited == cells

    def test_transform_uses_merged_solution(self, sample_df):
        binner = OptimalBinning2D(max_n_bins=4, max_n_bins_2d=3)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])

        x_bins = binner.binner_x_.transform(sample_df[['age']], metric='indices')['age'].to_numpy()
        y_bins = binner.binner_y_.transform(sample_df[['income']], metric='indices')['income'].to_numpy()
        expected = binner.solution_[x_bins, y_bins]
        actual = binner.transform(sample_df[['age', 'income']], metric='indices')['ageXincome'].to_numpy()
        np.testing.assert_array_equal(actual, expected)

        woe = binner.transform(sample_df[['age', 'income']], metric='woe')['ageXincome'].to_numpy()
        np.testing.assert_allclose(woe, binner._woe_2d_[expected])

    def test_transform_maps_observed_missing_bin_metrics(self, sample_df):
        """训练中出现的缺失箱应返回其统计值，而不是 NaN."""
        df = sample_df.copy()
        df.loc[df.index[:20], 'age'] = np.nan
        binner = OptimalBinning2D(max_n_bins=4, missing_separate=True)
        binner.fit(df, y=df['target'], features=['age', 'income'])

        transformed = {
            metric: binner.transform(df[['age', 'income']], metric=metric)['ageXincome']
            for metric in ('indices', 'bins', 'woe', 'event_rate')
        }
        missing = df['age'].isna()

        assert (transformed['indices'][missing] >= 0).all()
        assert (transformed['indices'][~missing] >= 0).all()
        assert transformed['bins'][missing].str.startswith('缺失值 × ').all()
        assert transformed['woe'][missing].notna().all()
        assert transformed['event_rate'][missing].notna().all()
        table = binner.get_bin_table().set_index('分箱')
        expected_woe = transformed['indices'][missing].map(table['分档WOE值'])
        expected_rate = transformed['indices'][missing].map(table['坏样本率'])
        np.testing.assert_allclose(transformed['woe'][missing], expected_woe)
        np.testing.assert_allclose(transformed['event_rate'][missing], expected_rate)

    def test_binning_table_uses_hscredit_metrics(self, sample_df):
        binner = OptimalBinning2D(max_n_bins=4, max_n_bins_2d=4)
        binner.fit(sample_df, y=sample_df['target'], features=['age', 'income'])
        table = binner.get_bin_table()

        expected_columns = [
            '指标名称', '指标含义', '分箱', '分箱标签', '样本总数', '样本占比', '好样本数', '坏样本数',
            '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '坏账改善',
            '风险拒绝比', '累积LIFT值', '分档KS值',
        ]
        assert set(expected_columns).issubset(table.columns)
        assert table['样本总数'].sum() == len(sample_df)

    def test_missing_bins_form_cartesian_product_in_required_order(self, sample_df):
        df = sample_df.copy()
        df.loc[df.index[:40], 'age'] = np.nan
        df.loc[df.index[40:80], 'income'] = np.nan
        df.loc[df.index[80:100], ['age', 'income']] = np.nan

        binner = OptimalBinning2D(max_n_bins=4, max_n_bins_2d=5, missing_separate=True)
        binner.fit(df, y=df['target'], features=['age', 'income'])
        cross = binner.get_cross_table()

        assert len(cross) == (binner.n_bins_x_ + 1) * (binner.n_bins_y_ + 1)
        assert cross['样本总数'].sum() == len(df)
        assert binner.get_bin_table()['样本总数'].sum() == len(df)

        groups = np.select(
            [
                (cross['特征1分箱'] >= 0) & (cross['特征2分箱'] >= 0),
                (cross['特征1分箱'] >= 0) & (cross['特征2分箱'] == -1),
                (cross['特征1分箱'] == -1) & (cross['特征2分箱'] >= 0),
            ],
            [0, 1, 2],
            default=3,
        )
        assert groups.tolist() == sorted(groups.tolist())
        assert cross.loc[cross['特征1分箱'] == -1, '特征1标签'].eq('缺失值').all()
        assert cross.loc[cross['特征2分箱'] == -1, '特征2标签'].eq('缺失值').all()
        assert cross['分箱'].ge(0).all()
        assert cross['分箱标签'].notna().all()
        assert list(cross.columns[8:]) == list(binner.get_bin_table().columns[4:])

    def test_missing_row_and_column_follow_non_missing_axis_monotonicity(self, sample_df):
        df = sample_df.copy()
        df.loc[df.index[:120], 'age'] = np.nan
        df.loc[df.index[120:240], 'income'] = np.nan
        binner = OptimalBinning2D(
            max_n_bins=4,
            max_n_bins_2d=6,
            monotonic_x='ascending',
            monotonic_y='descending',
            missing_separate=True,
        )
        binner.fit(df, y=df['target'], features=['age', 'income'])
        table = binner.get_bin_table().set_index('分箱')
        counts = {
            int(bin_id): (float(row['坏样本数']), float(row['好样本数']))
            for bin_id, row in table.iterrows()
        }
        assert not binner._monotonic_violations(
            binner.solution_, counts, 'ascending', 'descending'
        )

    def test_axis_monotonic_constraint(self):
        rng = np.random.RandomState(7)
        n = 2500
        age = rng.uniform(18, 70, n)
        income = rng.uniform(2000, 50000, n)
        probability = 1 / (1 + np.exp(-(-5 + age / 18 + income / 25000)))
        target = rng.binomial(1, probability)
        df = pd.DataFrame({'age': age, 'income': income, 'target': target})

        binner = OptimalBinning2D(
            max_n_bins=5,
            max_n_bins_2d=6,
            monotonic_x='ascending',
            monotonic_y='ascending',
        )
        binner.fit(df, y=df['target'], features=['age', 'income'])
        normal = binner.get_bin_table().query('分箱 >= 0')
        counts = {
            int(row['分箱']): (float(row['坏样本数']), float(row['好样本数']))
            for _, row in normal.iterrows()
        }
        assert not binner._monotonic_violations(
            binner.solution_, counts, 'ascending', 'ascending')

    def test_monotonic_hard_constraint_on_noisy_data(self):
        """硬约束：即便目标完全随机（非单调），合并结果也必须零单调违例."""
        rng = np.random.RandomState(123)
        n = 1500
        df = pd.DataFrame({
            'a': rng.rand(n),
            'c': rng.rand(n),
            'target': rng.randint(0, 2, n),  # 与特征无关，天然非单调
        })

        for trend_x, trend_y in [('ascending', 'ascending'),
                                 ('descending', 'descending'),
                                 ('ascending', 'descending')]:
            binner = OptimalBinning2D(
                max_n_bins=5, max_n_bins_2d=8,
                monotonic_x=trend_x, monotonic_y=trend_y,
            )
            binner.fit(df, y=df['target'], features=['a', 'c'])
            normal = binner.get_bin_table().query('分箱 >= 0')
            counts = {
                int(row['分箱']): (float(row['坏样本数']), float(row['好样本数']))
                for _, row in normal.iterrows()
            }
            # 硬约束：合并后必须零违例（软约束在随机数据上通常无法保证）
            assert not binner._monotonic_violations(
                binner.solution_, counts, trend_x, trend_y), \
                f"单调硬约束未满足: trend_x={trend_x}, trend_y={trend_y}"
            # 仍保持连通区域
            for bin_id in range(binner.n_bins_2d_):
                cells = set(map(tuple, np.argwhere(binner.solution_ == bin_id)))
                visited = {next(iter(cells))}
                pending = list(visited)
                while pending:
                    i, j = pending.pop()
                    for nb in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                        if nb in cells and nb not in visited:
                            visited.add(nb)
                            pending.append(nb)
                assert visited == cells

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
