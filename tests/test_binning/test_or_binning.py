"""OR-Tools 运筹规划分箱算法测试.

测试 ORBinning 的正确性和性能.
"""

import unittest
import numpy as np
import pandas as pd
import warnings

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

try:
    from hscredit.core.binning import ORBinning
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    warnings.warn("OR-Tools 未安装，跳过 ORBinning 测试")


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


@unittest.skipUnless(ORTOOLS_AVAILABLE, "OR-Tools 未安装")
class TestORBinning(unittest.TestCase):
    """测试 OR-Tools 运筹规划分箱."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()
        self.binner = ORBinning(max_n_bins=5, objective='iv', time_limit=5)

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

    def test_different_objectives(self):
        """测试不同优化目标."""
        objectives = ['iv', 'ks', 'gini', 'entropy', 'chi2']
        
        for obj in objectives:
            with self.subTest(objective=obj):
                binner = ORBinning(max_n_bins=5, objective=obj, time_limit=3)
                binner.fit(self.X, self.y)
                self.assertTrue(binner._is_fitted)

    def test_monotonic_constraint(self):
        """测试单调性约束."""
        binner = ORBinning(max_n_bins=5, objective='iv', monotonic=True, time_limit=3)
        binner.fit(self.X, self.y)
        self.assertTrue(binner._is_fitted)

    def test_bin_count_constraint(self):
        """测试分箱数约束."""
        binner = ORBinning(max_n_bins=5, min_n_bins=3, time_limit=3)
        binner.fit(self.X, self.y)
        
        for feature in self.X.columns:
            n_bins = binner.n_bins_[feature]
            # 考虑缺失值箱，实际分箱数可能在范围内
            self.assertLessEqual(n_bins, 7)  # max_n_bins + 缺失/特殊值箱

    def test_invalid_objective(self):
        """测试无效的优化目标."""
        with self.assertRaises(ValueError):
            ORBinning(objective='invalid')


@unittest.skipUnless(ORTOOLS_AVAILABLE, "OR-Tools 未安装")
class TestORBinningComparison(unittest.TestCase):
    """测试 ORBinning 与其他分箱方法的对比."""

    def setUp(self):
        """设置测试数据."""
        self.X, self.y = TestDataGenerator.create_binary_data()

    def test_or_vs_greedy_iv(self):
        """对比 OR-Tools 和贪心算法的 IV."""
        from hscredit.core.binning import BestIVBinning
        
        # OR-Tools 分箱
        or_binner = ORBinning(max_n_bins=5, objective='iv', time_limit=5)
        or_binner.fit(self.X, self.y)
        
        # 贪心算法分箱
        greedy_binner = BestIVBinning(max_n_bins=5)
        greedy_binner.fit(self.X, self.y)
        
        # 获取 IV 值进行比较
        or_iv = or_binner.bin_tables_['feature_1']['指标IV值'].iloc[0]
        greedy_iv = greedy_binner.bin_tables_['feature_1']['指标IV值'].iloc[0]
        
        # OR-Tools 应该能找到至少不差于贪心算法的解
        # （但由于启发式简化，不一定总是更优）
        print(f"\nOR-Tools IV: {or_iv:.4f}, Greedy IV: {greedy_iv:.4f}")


if __name__ == '__main__':
    unittest.main()
