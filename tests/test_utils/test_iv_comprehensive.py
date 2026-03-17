"""全面测试IV计算修复.

测试所有分箱方法的IV计算，确保：
1. IV值永远非负
2. 平滑处理正确
3. 公式顺序正确
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent / "hscredit"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from hscredit.core.binning import (
    OptimalBinning,
    KernelDensityBinning,
    GeneticBinning,
)
from hscredit.core.metrics.binning_metrics import woe_iv_vectorized, compute_bin_stats
from hscredit.core.metrics.importance import IV, IV_table


def test_extreme_imbalance():
    """测试极端不平衡情况."""
    print("=" * 80)
    print("测试：极端不平衡数据")
    print("=" * 80)
    
    np.random.seed(42)
    n_samples = 1000
    
    # 创建极端不平衡数据
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.exponential(1, n_samples)
    })
    
    # 目标变量 - 极端不平衡
    y = pd.Series(np.zeros(n_samples, dtype=int))
    y[X['feature1'] < -2] = 1
    y[X['feature1'] > 2] = 1
    
    print(f"\n总样本数: {n_samples}")
    print(f"好样本数: {(y == 0).sum()}")
    print(f"坏样本数: {(y == 1).sum()}")
    print(f"坏样本率: {y.mean():.2%}")
    
    # 测试不同分箱方法
    methods = [
        ('optimal_iv', OptimalBinning(method='optimal_iv', max_n_bins=5)),
        ('kernel_density', KernelDensityBinning(max_n_bins=5)),
        ('genetic', GeneticBinning(max_n_bins=5)),
    ]
    
    for method_name, binner in methods:
        print(f"\n{'-' * 80}")
        print(f"测试 {method_name} 方法:")
        
        try:
            binner.fit(X[['feature1']], y)
            bin_table = binner.get_bin_table('feature1')
            
            print(f"\n分箱表:")
            print(bin_table[['分箱', '样本总数', '好样本数', '坏样本数', '坏样本率', '分档WOE值', '分档IV值']])
            
            iv_values = bin_table['分档IV值'].values
            total_iv = bin_table['分档IV值'].sum()
            
            print(f"\n各bin IV值: {iv_values}")
            print(f"总IV值: {total_iv:.6f}")
            print(f"是否存在负值: {np.any(iv_values < 0)}")
            print(f"✓ 测试{'通过' if np.all(iv_values >= 0) else '失败'}")
            
            assert np.all(iv_values >= 0), f"{method_name} 存在负的IV值: {iv_values}"
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            raise


def test_zero_bins():
    """测试包含空bins的情况."""
    print("\n" + "=" * 80)
    print("测试：包含空bins（某些bin只有好样本或只有坏样本）")
    print("=" * 80)
    
    # 手动构造包含空bin的情况
    good_counts = np.array([100, 200, 0, 50])
    bad_counts = np.array([20, 0, 150, 10])
    
    print(f"\n好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    
    # 使用woe_iv_vectorized计算
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    
    print(f"\nWOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"是否存在负值: {np.any(bin_iv < 0)}")
    print(f"✓ 测试{'通过' if np.all(bin_iv >= 0) else '失败'}")
    
    assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"


def test_iv_table_function():
    """测试IV_table函数."""
    print("\n" + "=" * 80)
    print("测试：IV_table 函数")
    print("=" * 80)
    
    np.random.seed(42)
    n_samples = 1000
    
    # 创建测试数据
    feature = np.random.randn(n_samples)
    y_true = np.zeros(n_samples, dtype=int)
    y_true[feature < -1] = 1
    y_true[feature > 1] = 1
    
    print(f"\n总样本数: {n_samples}")
    print(f"好样本数: {np.sum(y_true == 0)}")
    print(f"坏样本数: {np.sum(y_true == 1)}")
    
    # 计算IV表
    iv_table = IV_table(y_true, feature, bins=5, feature_name="test_feature")
    
    print(f"\nIV表:")
    print(iv_table)
    
    # 计算总IV
    iv_value = IV(y_true, feature, bins=5)
    print(f"\n总IV值: {iv_value:.6f}")
    print(f"✓ 测试{'通过' if iv_value >= 0 else '失败'}")
    
    assert iv_value >= 0, f"IV值为负: {iv_value}"


def test_monotonic_binning():
    """测试单调分箱的IV计算."""
    print("\n" + "=" * 80)
    print("测试：单调分箱")
    print("=" * 80)
    
    np.random.seed(42)
    n_samples = 1000
    
    # 创建单调关系的特征
    X = pd.DataFrame({
        'feature': np.linspace(0, 100, n_samples)
    })
    
    # 目标变量与特征单调相关
    y = pd.Series((X['feature'] > 50).astype(int))
    
    print(f"\n总样本数: {n_samples}")
    print(f"好样本数: {(y == 0).sum()}")
    print(f"坏样本数: {(y == 1).sum()}")
    
    # 使用单调分箱
    binner = OptimalBinning(method='monotonic', max_n_bins=5, monotonic='auto')
    binner.fit(X, y)
    
    bin_table = binner.get_bin_table('feature')
    
    print(f"\n分箱表:")
    print(bin_table[['分箱', '样本总数', '好样本数', '坏样本数', '坏样本率', '分档WOE值', '分档IV值']])
    
    iv_values = bin_table['分档IV值'].values
    total_iv = bin_table['分档IV值'].sum()
    
    print(f"\n各bin IV值: {iv_values}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"是否存在负值: {np.any(iv_values < 0)}")
    print(f"✓ 测试{'通过' if np.all(iv_values >= 0) else '失败'}")
    
    assert np.all(iv_values >= 0), f"存在负的IV值: {iv_values}"


def test_all_zeros_in_bin():
    """测试某个bin完全为空的情况."""
    print("\n" + "=" * 80)
    print("测试：某个bin完全为空（good=0, bad=0）")
    print("=" * 80)
    
    good_counts = np.array([100, 0, 200, 0, 50])
    bad_counts = np.array([20, 0, 40, 0, 10])
    
    print(f"\n好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    
    print(f"\nWOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"是否存在负值: {np.any(bin_iv < 0)}")
    print(f"✓ 测试{'通过' if np.all(bin_iv >= 0) else '失败'}")
    
    # 注意：当bin完全为空时，平滑后的IV值非常小但理论上非负
    assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"


if __name__ == '__main__':
    try:
        test_extreme_imbalance()
        test_zero_bins()
        test_iv_table_function()
        test_monotonic_binning()
        test_all_zeros_in_bin()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        print("\n修复总结：")
        print("1. IV公式：(bad_dist - good_dist) * log(bad_dist / good_dist)")
        print("2. 平滑方法：将0替换为epsilon，然后重新归一化")
        print("3. 修复文件：")
        print("   - binning_metrics.py (woe_iv_vectorized)")
        print("   - optimal_iv_binning.py (_calc_iv)")
        print("   - kernel_density_binning.py (_calculate_iv)")
        print("   - genetic_binning.py (_calculate_iv)")
        print("   - importance.py (IV, IV_table)")
        
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        raise
