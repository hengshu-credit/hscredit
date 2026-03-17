"""测试SmoothBinning和KernelDensityBinning的优化效果."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.binning import SmoothBinning, KernelDensityBinning
from hscredit.core.binning import OptimalBinning  # 作为对比


def generate_test_data(n_samples=5000, seed=42):
    """生成测试数据."""
    np.random.seed(seed)
    
    # 特征1: 单调递增关系
    x1 = np.random.randn(n_samples)
    y1 = (x1 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # 特征2: 单调递减关系
    x2 = np.random.randn(n_samples)
    y2 = (-x2 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # 特征3: 多峰分布 (确保长度一致)
    n_per_peak = n_samples // 3
    x3 = np.concatenate([
        np.random.normal(-3, 1, n_per_peak),
        np.random.normal(0, 1, n_per_peak),
        np.random.normal(3, 1, n_samples - 2 * n_per_peak)  # 确保总数正确
    ])
    y3 = ((x3 > -1.5) & (x3 < 1.5)).astype(int)
    
    # 特征4: U型关系
    x4 = np.random.randn(n_samples)
    y4 = (np.abs(x4) > 1).astype(int)
    
    df = pd.DataFrame({
        'monotonic_inc': x1,
        'monotonic_dec': x2,
        'multimodal': x3,
        'u_shaped': x4
    })
    
    y_target = pd.DataFrame({
        'monotonic_inc': y1,
        'monotonic_dec': y2,
        'multimodal': y3,
        'u_shaped': y4
    })
    
    return df, y_target


def _test_binner_impl(binner_class, name, X, y, **kwargs):
    """测试分箱器实现."""
    print(f"\n{'='*60}")
    print(f"测试 {name}")
    print('='*60)
    
    binner = binner_class(max_n_bins=5, min_n_bins=2, verbose=False, **kwargs)
    
    try:
        binner.fit(X, y['monotonic_inc'])
        
        for feature in X.columns:
            bin_table = binner.bin_tables_[feature]
            iv = bin_table['指标IV值'].iloc[0] if '指标IV值' in bin_table.columns else 0
            
            print(f"\n特征: {feature}")
            print(f"  分箱数: {binner.n_bins_[feature]}")
            print(f"  总IV: {iv:.4f}")
            
            # 显示分箱详情
            if len(bin_table) > 0:
                print(f"  分箱统计:")
                for idx, row in bin_table.iterrows():
                    if '分箱标签' in bin_table.columns:
                        bin_iv = row['分档IV值'] if '分档IV值' in bin_table.columns else 0
                        print(f"    {row['分箱标签']}: 样本数={row['样本总数']}, 坏样本率={row['坏样本率']:.4f}, IV={bin_iv:.4f}")
        
        return binner
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数."""
    print("生成测试数据...")
    X, y_target = generate_test_data(n_samples=5000)
    
    print(f"\n数据形状: {X.shape}")
    print(f"目标变量分布:")
    for col in y_target.columns:
        print(f"  {col}: 坏样本率 = {y_target[col].mean():.4f}")
    
    # 测试 SmoothBinning - 不同方法
    print("\n" + "="*80)
    print("测试 SmoothBinning 优化版本")
    print("="*80)
    
    # Laplace方法
    _test_binner_impl(SmoothBinning, "SmoothBinning (Laplace)", X, y_target,
                method='laplace', smoothing_param=1.0)

    # Bayesian方法
    _test_binner_impl(SmoothBinning, "SmoothBinning (Bayesian)", X, y_target,
                method='bayesian', smoothing_param=1.0)

    # IV优化方法
    _test_binner_impl(SmoothBinning, "SmoothBinning (IV Optimized)", X, y_target,
                method='iv_optimized', smoothing_param=0.01)

    # 带单调性约束
    _test_binner_impl(SmoothBinning, "SmoothBinning (单调递增)", X[['monotonic_inc']], y_target,
                method='iv_optimized', monotonic='ascending')

    _test_binner_impl(SmoothBinning, "SmoothBinning (单调递减)", X[['monotonic_dec']], y_target,
                method='iv_optimized', monotonic='descending')
    
    # 测试 KernelDensityBinning
    print("\n" + "="*80)
    print("测试 KernelDensityBinning 优化版本")
    print("="*80)
    
    # 基本设置
    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (基础)", X, y_target,
                kernel='gaussian', bandwidth='scott')

    # 结合目标变量
    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (结合目标变量)", X, y_target,
                kernel='gaussian', bandwidth='scott', use_target=True)

    # 带单调性约束
    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (单调递增)", X[['monotonic_inc']], y_target,
                kernel='gaussian', bandwidth='scott', use_target=True, monotonic='ascending')

    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (单调递减)", X[['monotonic_dec']], y_target,
                kernel='gaussian', bandwidth='scott', use_target=True, monotonic='descending')

    # 多峰分布
    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (多峰)", X[['multimodal']], y_target,
                kernel='gaussian', bandwidth='scott', use_target=True, min_peak_distance=0.15)

    # U型分布
    _test_binner_impl(KernelDensityBinning, "KernelDensityBinning (U型)", X[['u_shaped']], y_target,
                kernel='gaussian', bandwidth='scott', use_target=True, monotonic='valley')
    
    # 对比 OptimalBinning
    print("\n" + "="*80)
    print("对比 OptimalBinning (基准)")
    print("="*80)
    
    _test_binner_impl(OptimalBinning, "OptimalBinning", X, y_target)
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == '__main__':
    main()
