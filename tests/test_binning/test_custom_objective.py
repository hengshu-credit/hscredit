"""测试 ORBinning 自定义目标函数功能."""

import unittest
import numpy as np
import pandas as pd
import warnings

try:
    from hscredit.core.binning import ORBinning, CustomObjectives
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


@unittest.skipUnless(ORTOOLS_AVAILABLE, "OR-Tools not installed")
class TestCustomObjective(unittest.TestCase):
    """测试自定义目标函数."""
    
    def setUp(self):
        """设置测试数据."""
        np.random.seed(42)
        n_samples = 1000
        
        # 创建一个简单的特征
        self.X = pd.DataFrame({
            'feature': np.random.randn(n_samples) * 10 + 50
        })
        
        # 创建目标变量（与特征相关）
        y_prob = 1 / (1 + np.exp(-(self.X['feature'] - 50) / 5))
        self.y = (np.random.random(n_samples) < y_prob).astype(int)
    
    def test_custom_objective_basic(self):
        """测试基本自定义目标函数."""
        # 定义一个简单的自定义目标：最大化 IV
        def simple_iv_objective(bin_stats, total_good, total_bad):
            total_iv = 0.0
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
            return total_iv
        
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=simple_iv_objective,
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_max_lift_iv(self):
        """测试最大 LIFT + IV 目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.max_lift_iv(lift_weight=0.5, iv_weight=1.0),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
        
        # 验证生成了分箱表
        bin_table = binner.get_bin_table('feature')
        self.assertIsNotNone(bin_table)
        self.assertGreater(len(bin_table), 0)
    
    def test_custom_objective_min_lift_iv(self):
        """测试最小 LIFT + IV 目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.min_lift_iv(lift_weight=0.5, iv_weight=1.0),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_max_min_lift_sum_iv(self):
        """测试最大LIFT + 最小LIFT + IV 目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.max_min_lift_sum_iv(
                lift_weight=0.1, 
                iv_weight=1.0
            ),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_lift_distance_sum_iv(self):
        """测试最大/最小LIFT离1距离求和 + IV 目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.lift_distance_sum_iv(
                lift_weight=0.5, 
                iv_weight=1.0
            ),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_max_ks_iv(self):
        """测试 KS + IV 复合目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.max_ks_iv(ks_weight=0.5, iv_weight=1.0),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_woe_variance(self):
        """测试 WOE 方差 + IV 目标."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.woe_variance_iv(
                variance_weight=0.1, 
                iv_weight=1.0
            ),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 验证拟合成功
        self.assertTrue(binner._is_fitted)
        self.assertIn('feature', binner.splits_)
    
    def test_custom_objective_without_custom_func(self):
        """测试不提供自定义函数时的错误."""
        with self.assertRaises(ValueError) as context:
            ORBinning(
                max_n_bins=5,
                objective='custom',
                custom_objective=None
            )
        
        self.assertIn("必须提供 custom_objective 参数", str(context.exception))
    
    def test_transform_with_custom_objective(self):
        """测试使用自定义目标拟合后的转换功能."""
        binner = ORBinning(
            max_n_bins=5,
            objective='custom',
            custom_objective=CustomObjectives.max_lift_iv(lift_weight=0.5),
            time_limit=10
        )
        
        binner.fit(self.X, self.y)
        
        # 测试转换为分箱索引
        X_binned = binner.transform(self.X, metric='indices')
        self.assertIsInstance(X_binned, pd.DataFrame)
        self.assertIn('feature', X_binned.columns)
        
        # 测试转换为 WOE
        X_woe = binner.transform(self.X, metric='woe')
        self.assertIsInstance(X_woe, pd.DataFrame)
        self.assertIn('feature', X_woe.columns)


if __name__ == '__main__':
    unittest.main()
