"""测试 ORBinning 双 API 支持 (sklearn + scorecardpipeline 风格)."""

import unittest
import numpy as np
import pandas as pd
import warnings

try:
    from hscredit.core.binning import ORBinning
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


@unittest.skipUnless(ORTOOLS_AVAILABLE, "OR-Tools not installed")
class TestORBinningDualAPI(unittest.TestCase):
    """测试 ORBinning 双 API 支持."""
    
    def setUp(self):
        """设置测试数据."""
        np.random.seed(42)
        n_samples = 240
        
        # 创建特征
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n_samples) * 10 + 50,
            'feature2': np.random.randn(n_samples) * 5 + 30,
        })
        
        # 创建目标变量
        y_prob = 1 / (1 + np.exp(-(self.X['feature1'] - 50) / 10))
        self.y = (np.random.random(n_samples) < y_prob).astype(int)
        
        # 创建完整 DataFrame (scorecardpipeline 风格)
        self.df = self.X.copy()
        self.df['target'] = self.y
    
    def test_sklearn_style_fit(self):
        """测试 sklearn 风格: fit(X, y)."""
        binner = ORBinning(max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.X, self.y)
        
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature1', binner.splits_)
        self.assertIn('feature2', binner.splits_)
    
    def test_scorecardpipeline_style_fit(self):
        """测试 scorecardpipeline 风格: fit(df) 从 df 中提取 target."""
        binner = ORBinning(target='target', max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.df)  # 不传 y，从 df 中提取 'target' 列
        
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature1', binner.splits_)
        self.assertIn('feature2', binner.splits_)
    
    def test_sklearn_style_priority(self):
        """测试 y 参数优先级: fit(df, y) 时优先使用 y."""
        # 创建不同的 y
        y_alt = 1 - self.y  # 翻转标签
        
        binner = ORBinning(target='target', max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.df, y=y_alt)  # 应该使用 y_alt，忽略 df['target']
        
        self.assertTrue(binner._is_fitted)
    
    def test_numpy_array_input(self):
        """测试 numpy 数组输入."""
        X_np = self.X.values
        y_np = self.y.values
        
        binner = ORBinning(max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(X_np, y_np)
        
        self.assertTrue(binner._is_fitted)
    
    def test_transform_after_fit(self):
        """测试拟合后的转换功能."""
        # sklearn 风格
        binner = ORBinning(max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.X, self.y)
        
        X_binned = binner.transform(self.X, metric='indices')
        self.assertIsInstance(X_binned, pd.DataFrame)
        self.assertEqual(len(X_binned), len(self.X))
    
    def test_scorecardpipeline_transform(self):
        """测试 scorecardpipeline 风格拟合后的转换."""
        binner = ORBinning(target='target', max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.df)
        
        # 转换时应该传入不含 target 的数据
        X_features = self.df.drop(columns=['target'])
        X_binned = binner.transform(X_features, metric='indices')
        
        self.assertIsInstance(X_binned, pd.DataFrame)
        self.assertEqual(len(X_binned), len(self.df))
    
    def test_fit_transform(self):
        """测试 fit_transform 方法."""
        binner = ORBinning(max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        X_binned = binner.fit_transform(self.X, self.y, metric='indices')
        
        self.assertIsInstance(X_binned, pd.DataFrame)
        self.assertTrue(binner._is_fitted)
    
    def test_get_bin_table(self):
        """测试获取分箱表."""
        binner = ORBinning(max_n_bins=5, objective='iv', time_limit=1, max_candidates=20)
        binner.fit(self.X, self.y)
        
        bin_table = binner.get_bin_table('feature1')
        self.assertIsNotNone(bin_table)
        self.assertGreater(len(bin_table), 0)


if __name__ == '__main__':
    unittest.main()
