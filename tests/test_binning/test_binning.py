"""分箱模块测试.

测试各种分箱方法的正确性和一致性.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from hscredit.core.binning import (
    BaseBinning,
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    BestKSBinning,
    BestIVBinning,
    MDLPBinning,
    OptimalBinning,
)


class TestDataGenerator:
    """测试数据生成器."""

    @staticmethod
    def create_binary_data(n_samples=1000, n_features=3, random_state=42):
        """创建二分类测试数据."""
        np.random.seed(random_state)

        # 创建特征
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.randint(0, 100, n_samples),
        })

        # 创建目标变量（与特征相关）
        prob = 1 / (1 + np.exp(-(0.5 * X['feature_1'] - 0.3 * X['feature_2'] + 0.01 * X['feature_3'])))
        y = pd.Series(np.random.binomial(1, prob))

        return X, y

    @staticmethod
    def create_data_with_missing(n_samples=1000, missing_rate=0.1):
        """创建包含缺失值的数据."""
        X, y = TestDataGenerator.create_binary_data(n_samples)

        # 随机添加缺失值
        n_missing = int(n_samples * missing_rate)
        missing_idx = np.random.choice(X.index, n_missing, replace=False)
        X.loc[missing_idx, 'feature_1'] = np.nan

        return X, y


class TestBaseBinning(unittest.TestCase):
    """测试分箱基类."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()

    def test_base_binning_is_abstract(self):
        """测试基类是抽象类."""
        with self.assertRaises(TypeError):
            BaseBinning()


class TestUniformBinning(unittest.TestCase):
    """测试等距分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = UniformBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)
        self.assertIn('feature_1', self.binner.splits_)

    def test_transform(self):
        """测试转换."""
        self.binner.fit(self.X, self.y)
        X_transformed = self.binner.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)

    def test_transform_woe(self):
        """测试 WOE 转换."""
        self.binner.fit(self.X, self.y)
        X_woe = self.binner.transform(self.X, metric='woe')
        self.assertEqual(X_woe.shape, self.X.shape)

    def test_get_bin_table(self):
        """测试获取分箱表."""
        self.binner.fit(self.X, self.y)
        bin_table = self.binner.get_bin_table('feature_1')
        self.assertIn('分箱', bin_table.columns)
        self.assertIn('分档WOE值', bin_table.columns)
        self.assertIn('指标IV值', bin_table.columns)

    def test_bin_count(self):
        """测试分箱数量."""
        self.binner.fit(self.X, self.y)
        for feature in self.X.columns:
            self.assertLessEqual(
                self.binner.n_bins_[feature],
                self.binner.max_n_bins
            )


class TestQuantileBinning(unittest.TestCase):
    """测试等频分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = QuantileBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_equal_frequency(self):
        """测试等频特性."""
        self.binner.fit(self.X, self.y)
        bin_table = self.binner.get_bin_table('feature_1')
        # 各箱样本数应该大致相等（允许一定误差）
        counts = bin_table['样本总数'].values
        mean_count = np.mean(counts)
        for count in counts:
            self.assertLess(
                abs(count - mean_count) / mean_count,
                0.5  # 允许 50% 的偏差
            )


class TestTreeBinning(unittest.TestCase):
    """测试决策树分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = TreeBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_transform(self):
        """测试转换."""
        self.binner.fit(self.X, self.y)
        X_transformed = self.binner.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)


class TestChiMergeBinning(unittest.TestCase):
    """测试卡方分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = ChiMergeBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_transform(self):
        """测试转换."""
        self.binner.fit(self.X, self.y)
        X_transformed = self.binner.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)


class TestBestKSBinning(unittest.TestCase):
    """测试最优 KS 分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = BestKSBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_ks_statistic(self):
        """测试 KS 统计量."""
        self.binner.fit(self.X, self.y)
        bin_table = self.binner.get_bin_table('feature_1')
        self.assertIn('分档KS值', bin_table.columns)


class TestBestIVBinning(unittest.TestCase):
    """测试最优 IV 分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = BestIVBinning(max_n_bins=5)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_iv_calculation(self):
        """测试 IV 计算."""
        self.binner.fit(self.X, self.y)
        bin_table = self.binner.get_bin_table('feature_1')
        self.assertIn('指标IV值', bin_table.columns)
        iv = bin_table['指标IV值'].iloc[0]
        self.assertGreaterEqual(iv, 0)


class TestMDLPBinning(unittest.TestCase):
    """测试 MDLP 分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = MDLPBinning(max_n_bins=10)

    def test_fit(self):
        """测试拟合."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_transform(self):
        """测试转换."""
        self.binner.fit(self.X, self.y)
        X_transformed = self.binner.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)


class TestOptimalBinning(unittest.TestCase):
    """测试统一分箱接口."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()

    def test_uniform_method(self):
        """测试 uniform 方法."""
        binner = OptimalBinning(method='uniform', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_uniform_method_accepts_legacy_n_bins(self):
        """测试 uniform 方法兼容 legacy n_bins 参数."""
        binner = OptimalBinning(method='uniform', n_bins=4)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)
        self.assertEqual(binner.n_bins_['feature_1'], 4)

    def test_quantile_method(self):
        """测试 quantile 方法."""
        binner = OptimalBinning(method='quantile', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_uniform_method_default_decimal_precision(self):
        """测试默认切分点精度为4位小数."""
        X = pd.DataFrame({'feature_1': [0.123456, 0.223456, 0.323456, 0.423456, 0.523456, 0.623456]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        binner = OptimalBinning(method='uniform', max_n_bins=4)
        binner.fit(X, y)

        np.testing.assert_allclose(
            binner.splits_['feature_1'],
            np.array([0.2485, 0.3735, 0.4985])
        )

    def test_uniform_method_custom_decimal_precision(self):
        """测试可通过 decimal 调整切分点精度."""
        X = pd.DataFrame({'feature_1': [0.123456, 0.223456, 0.323456, 0.423456, 0.523456, 0.623456]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        binner = OptimalBinning(method='uniform', max_n_bins=4, decimal=2)
        binner.fit(X, y)

        np.testing.assert_allclose(
            binner.splits_['feature_1'],
            np.array([0.25, 0.37, 0.5])
        )

    def test_tree_method(self):
        """测试 tree 方法."""
        binner = OptimalBinning(method='tree', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_chi_method(self):
        """测试 chi 方法."""
        binner = OptimalBinning(method='chi', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_best_ks_method(self):
        """测试 best_ks 方法."""
        binner = OptimalBinning(method='best_ks', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_best_iv_method(self):
        """测试 best_iv 方法."""
        binner = OptimalBinning(method='best_iv', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_mdlp_method(self):
        """测试 mdlp 方法."""
        binner = OptimalBinning(method='mdlp', max_n_bins=5)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_user_splits_with_nan_for_numerical(self):
        """测试数值型自定义切分点包含np.nan时可正常工作."""
        X = self.X[['feature_1']].copy()
        y = self.y.copy()

        binner = OptimalBinning(user_splits={'feature_1': [-0.5, 0.0, 0.5, np.nan]})
        binner.fit(X, y)

        # 能成功拟合并生成分箱表（含缺失箱标签体系）
        self.assertTrue(binner._is_fitted)
        bin_table = binner.get_bin_table('feature_1')
        self.assertIn('分箱标签', bin_table.columns)

    def test_import_rules_respects_decimal_precision(self):
        """测试 import_rules 也遵循 decimal 精度设置."""
        binner = OptimalBinning(decimal=2)
        binner.import_rules({'feature_1': [np.float64(0.123456), np.float64(0.987654), np.nan]})

        np.testing.assert_allclose(
            binner.splits_['feature_1'],
            np.array([0.12, 0.99])
        )

    def test_check_input_align_by_common_index(self):
        """测试X/y长度不一致时按共同索引自动对齐."""
        X = self.X[['feature_1']].copy()
        y_filtered = self.y.loc[X['feature_1'] > 0]

        binner = OptimalBinning(method='mdlp', max_n_bins=5)
        # 不应抛出长度不匹配错误
        binner.fit(X, y_filtered)
        self.assertTrue(binner._is_fitted)

    def test_getitem_returns_feature_rules(self):
        """测试通过[]获取特征分箱规则."""
        X = self.X[['feature_1']].copy()
        y = self.y.copy()

        binner = OptimalBinning(user_splits={'feature_1': [-0.5, 0.0, 0.5]})
        binner.fit(X, y)

        rules = np.asarray(binner['feature_1'])
        self.assertIsNotNone(rules)
        # user_splits 会与数据 min/max 合并，且相邻切分可能被合并，故只校验关键分界仍存在
        self.assertGreaterEqual(len(rules), 2)
        self.assertTrue(np.any(np.isclose(rules, -0.5)), rules)
        self.assertTrue(np.any(np.isclose(rules, 0.5)), rules)

    def test_invalid_method(self):
        """测试无效方法."""
        with self.assertRaises(ValueError):
            OptimalBinning(method='invalid_method')


class TestMissingValues(unittest.TestCase):
    """测试缺失值处理."""

    def setUp(self):
        """设置包含缺失值的测试数据."""
        self.X, self.y = TestDataGenerator.create_data_with_missing()
        self.binner = BestIVBinning(max_n_bins=5)

    def test_fit_with_missing(self):
        """测试拟合包含缺失值的数据."""
        self.binner.fit(self.X, self.y)
        self.assertTrue(self.binner._is_fitted)

    def test_transform_with_missing(self):
        """测试转换包含缺失值的数据."""
        self.binner.fit(self.X, self.y)
        X_transformed = self.binner.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)

    def test_missing_in_bin_table(self):
        """测试分箱表包含缺失值统计."""
        self.binner.fit(self.X, self.y)
        bin_table = self.binner.get_bin_table('feature_1')
        # 检查是否有缺失值分箱
        has_missing = any('missing' in str(bin_val) for bin_val in bin_table['分箱标签'])
        self.assertTrue(has_missing)


class TestBinningConsistency(unittest.TestCase):
    """测试分箱一致性."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()

    def test_reproducibility(self):
        """测试结果可复现性."""
        binner1 = UniformBinning(max_n_bins=5, random_state=42)
        binner2 = UniformBinning(max_n_bins=5, random_state=42)

        binner1.fit(self.X, self.y)
        binner2.fit(self.X, self.y)

        # 比较切分点
        for feature in self.X.columns:
            np.testing.assert_array_equal(
                binner1.splits_[feature],
                binner2.splits_[feature]
            )

    def test_bin_counts_consistency(self):
        """测试分箱数量一致性."""
        binner = QuantileBinning(max_n_bins=5)
        binner.fit(self.X, self.y)

        # 转换数据
        X_transformed = binner.transform(self.X)

        # 检查转换后的分箱数量
        for feature in self.X.columns:
            unique_bins = X_transformed[feature].nunique()
            # 允许缺失值导致的额外分箱
            self.assertLessEqual(unique_bins, binner.n_bins_[feature] + 1)


class TestWOEEncoding(unittest.TestCase):
    """测试 WOE 编码."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = BestIVBinning(max_n_bins=5)
        self.binner.fit(self.X, self.y)

    def test_woe_values(self):
        """测试 WOE 值计算."""
        X_woe = self.binner.transform(self.X, metric='woe')

        # WOE 值应该是数值型
        self.assertTrue(np.issubdtype(X_woe.dtypes['feature_1'], np.floating))

    def test_woe_finite(self):
        """测试 WOE 值有限."""
        X_woe = self.binner.transform(self.X, metric='woe')
        self.assertTrue(np.isfinite(X_woe.values).all())


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
