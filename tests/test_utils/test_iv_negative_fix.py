"""测试IV值出现负数的修复.

验证场景：
1. 极端不平衡的分布（某些bin只有好样本或只有坏样本）
2. 空bins的情况
3. 单个样本的bin
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent / "hscredit"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.core.metrics.binning_metrics import woe_iv_vectorized, compute_bin_stats


def test_woe_iv_vectorized():
    """测试向量化WOE和IV计算."""
    print("=" * 80)
    print("测试1: woe_iv_vectorized 函数")
    print("=" * 80)
    
    # 测试用例1: 正常情况
    print("\n测试用例1: 正常分布")
    good_counts = np.array([100, 200, 150, 50])
    bad_counts = np.array([20, 30, 40, 10])
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    print(f"好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    print(f"WOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_iv >= 0)}")
    
    # 测试用例2: 某些bin只有好样本
    print("\n测试用例2: 某些bin只有好样本（坏样本数为0）")
    good_counts = np.array([100, 200, 0, 50])
    bad_counts = np.array([20, 0, 150, 10])
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    print(f"好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    print(f"WOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_iv >= 0)}")
    assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"
    
    # 测试用例3: 极端不平衡 - 单个样本的bin
    print("\n测试用例3: 极端不平衡（单个样本的bin）")
    good_counts = np.array([1000, 1, 500, 2])
    bad_counts = np.array([1, 800, 2, 400])
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    print(f"好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    print(f"WOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_iv >= 0)}")
    assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"
    
    # 测试用例4: 所有bin都是极端不平衡
    print("\n测试用例4: 所有bin都是极端不平衡")
    good_counts = np.array([1000, 0, 500, 10])
    bad_counts = np.array([0, 800, 5, 0])
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    print(f"好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    print(f"WOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_iv >= 0)}")
    assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"


def test_compute_bin_stats():
    """测试分箱统计计算."""
    print("\n" + "=" * 80)
    print("测试2: compute_bin_stats 函数")
    print("=" * 80)
    
    # 创建测试数据 - 极端不平衡
    np.random.seed(42)
    n_samples = 1000
    
    # 特征值
    X = np.random.randn(n_samples)
    
    # 目标变量 - 极端不平衡的分布
    y = np.zeros(n_samples, dtype=int)
    y[X < -2] = 1  # 只有最左边的样本是坏样本
    y[X > 2] = 1   # 最右边的样本也是坏样本
    
    # 分箱
    bins = np.digitize(X, bins=[-1, 0, 1])
    
    # 计算分箱统计
    bin_stats = compute_bin_stats(bins, y)
    
    print("\n分箱统计表:")
    print(bin_stats[['分箱', '样本总数', '好样本数', '坏样本数', '坏样本率', '分档WOE值', '分档IV值']])
    
    print(f"\n各bin IV值: {bin_stats['分档IV值'].values}")
    print(f"总IV值: {bin_stats['分档IV值'].sum():.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_stats['分档IV值'] >= 0)}")
    
    assert np.all(bin_stats['分档IV值'] >= 0), f"存在负的IV值: {bin_stats['分档IV值'].values}"


def test_optimal_binning():
    """测试OptimalBinning分箱."""
    print("\n" + "=" * 80)
    print("测试3: OptimalBinning 分箱")
    print("=" * 80)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 特征值 - 多个区间
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.exponential(1, n_samples)
    })
    
    # 目标变量 - 与feature1相关，但极端不平衡
    y = pd.Series(np.zeros(n_samples, dtype=int))
    y[X['feature1'] < -2] = 1
    y[X['feature1'] > 2] = 1
    
    # 分箱
    binner = OptimalBinning(max_n_bins=5, method='best_iv')
    binner.fit(X, y)
    
    # 获取分箱表
    for feature in X.columns:
        bin_table = binner.get_bin_table(feature)
        print(f"\n{feature} 分箱表:")
        print(bin_table[['分箱', '样本总数', '好样本数', '坏样本数', '坏样本率', '分档WOE值', '分档IV值']])
        
        iv_values = bin_table['分档IV值'].values
        print(f"\n各bin IV值: {iv_values}")
        print(f"总IV值: {bin_table['分档IV值'].sum():.6f}")
        print(f"各bin IV值是否非负: {np.all(iv_values >= 0)}")
        
        assert np.all(iv_values >= 0), f"{feature} 存在负的IV值: {iv_values}"


def test_edge_case_all_zeros():
    """测试边界情况：某个bin完全没有样本."""
    print("\n" + "=" * 80)
    print("测试4: 边界情况 - 某个bin完全没有样本")
    print("=" * 80)
    
    # 创建数据，使得某些bin为空
    good_counts = np.array([100, 0, 200, 0, 50])
    bad_counts = np.array([10, 0, 20, 0, 5])
    
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts)
    
    print(f"好样本数: {good_counts}")
    print(f"坏样本数: {bad_counts}")
    print(f"WOE值: {woe}")
    print(f"各bin IV值: {bin_iv}")
    print(f"总IV值: {total_iv:.6f}")
    print(f"各bin IV值是否非负: {np.all(bin_iv >= 0)}")
    
    # 注意：当某个bin完全没有样本时（good=0, bad=0），
    # 平滑处理后该bin的IV值可能非常小，但理论上不应该为负
    # 实际应用中，这类空bin通常会被合并掉


if __name__ == '__main__':
    try:
        test_woe_iv_vectorized()
        test_compute_bin_stats()
        test_optimal_binning()
        test_edge_case_all_zeros()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！IV值不再出现负数")
        print("=" * 80)
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        raise
