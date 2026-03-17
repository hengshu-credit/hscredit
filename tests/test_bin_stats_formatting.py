"""测试 compute_bin_stats 的数值格式化功能."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.metrics.binning_metrics import compute_bin_stats


def test_formatting():
    """测试数值格式化."""
    print("="*80)
    print("测试 compute_bin_stats 数值格式化")
    print("="*80)
    
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    
    # 创建分箱
    bins = np.random.randint(0, 5, n)
    y = np.random.randint(0, 2, n)
    
    print(f"\n测试数据: {n} 个样本, 5 个分箱")
    print(f"坏样本率: {y.mean():.4f}")
    
    # 测试默认行为（round_digits=True）
    print("\n" + "-"*80)
    print("测试 1: round_digits=True (默认)")
    print("-"*80)
    df_formatted = compute_bin_stats(bins, y, round_digits=True)
    
    print("\n分箱统计表（格式化）:")
    print(df_formatted.to_string())
    
    # 检查是否有科学计数法
    print("\n检查科学计数法:")
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'float32']:
            # 检查是否有科学计数法
            values = df_formatted[col].values
            has_scientific = any(abs(v) < 1e-4 or abs(v) > 1e4 for v in values if v != 0)
            print(f"  {col}: 最大值={values.max():.6f}, 最小值={values.min():.6f}, 可能科学计数={has_scientific}")
    
    # 测试不格式化
    print("\n" + "-"*80)
    print("测试 2: round_digits=False")
    print("-"*80)
    df_raw = compute_bin_stats(bins, y, round_digits=False)
    
    print("\n分箱统计表（原始精度）:")
    print(df_raw.to_string())
    
    # 验证 WOE 值是否一致
    print("\n" + "-"*80)
    print("验证 WOE 值一致性")
    print("-"*80)
    woe_diff = (df_formatted['分档WOE值'] - df_raw['分档WOE值']).abs().max()
    iv_diff = (df_formatted['指标IV值'] - df_raw['指标IV值']).abs().max()
    
    print(f"WOE 最大差异: {woe_diff:.10f}")
    print(f"IV 最大差异: {iv_diff:.10f}")
    
    if woe_diff < 1e-5 and iv_diff < 1e-5:
        print("✓ 格式化不影响 WOE 和 IV 的计算精度")
    else:
        print("✗ 警告：格式化可能影响计算精度")
    
    # 测试极值情况
    print("\n" + "="*80)
    print("测试 3: 极值情况")
    print("="*80)
    
    # 创建极端不平衡的数据
    y_extreme = np.zeros(n)
    y_extreme[:10] = 1  # 只有 1% 的坏样本
    bins_extreme = np.random.randint(0, 3, n)
    
    print(f"\n极端不平衡数据: 坏样本率 = {y_extreme.mean():.4f}")
    
    df_extreme = compute_bin_stats(bins_extreme, y_extreme, round_digits=True)
    print("\n分箱统计表（极端不平衡）:")
    print(df_extreme.to_string())
    
    print("\n检查是否还有科学计数法:")
    for col in df_extreme.columns:
        if df_extreme[col].dtype in ['float64', 'float32']:
            values = df_extreme[col].values
            # 转换为字符串检查是否有 'e'
            str_values = [str(v) for v in values]
            has_scientific = any('e' in s.lower() for s in str_values)
            if has_scientific:
                print(f"  ✗ {col}: 发现科学计数法")
            else:
                print(f"  ✓ {col}: 无科学计数法")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == '__main__':
    test_formatting()
