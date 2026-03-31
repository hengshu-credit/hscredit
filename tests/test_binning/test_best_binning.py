"""BestIVBinning 和 BestKSBinning 测试.

测试 BestIVBinning 和 BestKSBinning 分箱类.
"""

import unittest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.binning import (
    BestIVBinning,
    BestKSBinning,
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


class TestBestIVBinning(unittest.TestCase):
    """测试 BestIVBinning."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = BestIVBinning(max_n_bins=5)

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
        table = self.binner.get_bin_table('feature_1')
        self.assertIsNotNone(table)
        self.assertIn('分档WOE值', table.columns)

    def test_iv_calculation(self):
        """测试 IV 计算."""
        self.binner.fit(self.X, self.y)
        table = self.binner.get_bin_table('feature_1')
        self.assertIn('指标IV值', table.columns)
        iv_value = table['指标IV值'].iloc[0]
        self.assertGreater(iv_value, 0)

    def test_low_unique_values_should_keep_multiple_bins(self):
        """测试低唯一值数值特征不会退化成单箱."""
        X = pd.DataFrame({'x': [1, 1, 2, 2, 3, 3, 4, 4]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

        binner = BestIVBinning(max_n_bins=5)
        binner.fit(X, y)

        # 至少应有2个箱（切分点>=1）
        self.assertGreaterEqual(binner.n_bins_['x'], 2)


class TestBestKSBinning(unittest.TestCase):
    """测试 BestKSBinning."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = BestKSBinning(max_n_bins=5)

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

    def test_ks_statistic(self):
        """测试 KS 统计量计算."""
        self.binner.fit(self.X, self.y)
        table = self.binner.get_bin_table('feature_1')
        self.assertIn('分档KS值', table.columns)


    def test_low_unique_values_should_keep_multiple_bins(self):
        """测试低唯一值数值特征不会退化成单箱."""
        X = pd.DataFrame({'x': [1, 1, 2, 2, 3, 3, 4, 4]})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

        binner = BestKSBinning(max_n_bins=5)
        binner.fit(X, y)

        self.assertGreaterEqual(binner.n_bins_['x'], 2)


class TestBestBinningAPI(unittest.TestCase):
    """测试新风格 API."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()

    def test_best_iv_style(self):
        """测试 BestIVBinning 风格."""
        binner = BestIVBinning(max_n_bins=5, min_n_bins=2)
        binner.fit(self.X, self.y)
        
        # 验证分箱数
        for feature in self.X.columns:
            self.assertLessEqual(
                binner.n_bins_[feature],
                5,
                f"{feature} 的分箱数应不超过 5"
            )

    def test_best_ks_style(self):
        """测试 BestKSBinning 风格."""
        binner = BestKSBinning(max_n_bins=5, min_n_bins=2)
        binner.fit(self.X, self.y)
        
        # 验证分箱数
        for feature in self.X.columns:
            self.assertLessEqual(
                binner.n_bins_[feature],
                5,
                f"{feature} 的分箱数应不超过 5"
            )


if __name__ == '__main__':
    unittest.main()
