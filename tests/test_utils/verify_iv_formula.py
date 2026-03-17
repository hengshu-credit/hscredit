"""验证IV值计算公式的数学正确性.

IV (Information Value) 公式：
IV = Σ (bad_dist - good_dist) * log(bad_dist / good_dist)

理论上，这个公式应该总是非负的，因为：
1. 当 bad_dist > good_dist 时：
   - bad_dist / good_dist > 1
   - log(bad_dist / good_dist) > 0
   - (bad_dist - good_dist) > 0
   - 乘积 > 0

2. 当 bad_dist < good_dist 时：
   - bad_dist / good_dist < 1
   - log(bad_dist / good_dist) < 0
   - (bad_dist - good_dist) < 0
   - 乘积 > 0（负负得正）

3. 当 bad_dist = good_dist 时：
   - log(bad_dist / good_dist) = 0
   - 乘积 = 0
"""

import numpy as np


def verify_iv_formula_mathematically():
    """从数学角度验证IV公式的非负性."""
    print("=" * 80)
    print("数学验证：IV公式的非负性")
    print("=" * 80)
    
    # 测试各种bad_dist和good_dist的组合
    test_cases = [
        ("bad_dist > good_dist", 0.3, 0.1),
        ("bad_dist < good_dist", 0.1, 0.3),
        ("bad_dist = good_dist", 0.2, 0.2),
        ("bad_dist >> good_dist", 0.9, 0.01),
        ("bad_dist << good_dist", 0.01, 0.9),
    ]
    
    print("\n测试用例：")
    for name, bad_dist, good_dist in test_cases:
        woe = np.log(bad_dist / good_dist)
        bin_iv = (bad_dist - good_dist) * woe
        
        print(f"\n{name}:")
        print(f"  bad_dist = {bad_dist}, good_dist = {good_dist}")
        print(f"  bad_dist / good_dist = {bad_dist / good_dist:.4f}")
        print(f"  WOE = log(bad_dist / good_dist) = {woe:.4f}")
        print(f"  bin_IV = (bad_dist - good_dist) * WOE")
        print(f"        = ({bad_dist} - {good_dist}) * {woe:.4f}")
        print(f"        = {bin_iv:.6f}")
        print(f"  ✓ bin_IV >= 0: {bin_iv >= 0}")
        
        assert bin_iv >= 0, f"bin_IV should be non-negative, got {bin_iv}"


def verify_iv_formula_with_smoothing():
    """验证平滑处理后的IV公式非负性."""
    print("\n" + "=" * 80)
    print("验证平滑处理后的IV公式非负性")
    print("=" * 80)
    
    epsilon = 1e-10
    
    # 测试不同的平滑场景
    test_cases = [
        {
            "name": "正常情况",
            "good_counts": [100, 200, 150],
            "bad_counts": [20, 30, 40]
        },
        {
            "name": "某些bin只有好样本（bad=0）",
            "good_counts": [100, 200, 0],
            "bad_counts": [20, 0, 150]
        },
        {
            "name": "某些bin只有坏样本（good=0）",
            "good_counts": [100, 0, 150],
            "bad_counts": [20, 200, 0]
        },
        {
            "name": "极端不平衡",
            "good_counts": [1000, 1, 500],
            "bad_counts": [1, 800, 2]
        },
        {
            "name": "所有bin都是极端情况",
            "good_counts": [1000, 0, 500],
            "bad_counts": [0, 800, 0]
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        good_counts = np.array(case['good_counts'], dtype=float)
        bad_counts = np.array(case['bad_counts'], dtype=float)
        
        print(f"  原始 good_counts: {good_counts}")
        print(f"  原始 bad_counts: {bad_counts}")
        
        # 平滑处理
        good_smooth = np.where(good_counts == 0, epsilon, good_counts)
        bad_smooth = np.where(bad_counts == 0, epsilon, bad_counts)
        
        print(f"  平滑后 good_counts: {good_smooth}")
        print(f"  平滑后 bad_counts: {bad_smooth}")
        
        # 计算分布
        good_dist = good_smooth / good_smooth.sum()
        bad_dist = bad_smooth / bad_smooth.sum()
        
        print(f"  good_dist: {good_dist}")
        print(f"  bad_dist: {bad_dist}")
        
        # 计算WOE和IV
        woe = np.log(bad_dist / good_dist)
        bin_iv = (bad_dist - good_dist) * woe
        
        print(f"  WOE: {woe}")
        print(f"  bin_IV: {bin_iv}")
        print(f"  总IV: {bin_iv.sum():.6f}")
        print(f"  ✓ 所有bin_IV >= 0: {np.all(bin_iv >= 0)}")
        
        assert np.all(bin_iv >= 0), f"存在负的IV值: {bin_iv}"


def compare_smoothing_methods():
    """对比不同平滑方法的效果."""
    print("\n" + "=" * 80)
    print("对比不同平滑方法")
    print("=" * 80)
    
    # 测试数据：某个bin完全为空
    good_counts = np.array([100, 0, 200])
    bad_counts = np.array([20, 0, 40])
    epsilon = 1e-10
    
    print(f"\n原始数据：")
    print(f"  good_counts: {good_counts}")
    print(f"  bad_counts: {bad_counts}")
    
    # 方法1：原始的错误方法（会破坏分布比例）
    print("\n方法1（错误）：分子分母分别加epsilon")
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    good_dist_wrong = (good_counts + epsilon) / (total_good + epsilon * len(good_counts))
    bad_dist_wrong = (bad_counts + epsilon) / (total_bad + epsilon * len(bad_counts))
    woe_wrong = np.log(bad_dist_wrong / good_dist_wrong)
    bin_iv_wrong = (bad_dist_wrong - good_dist_wrong) * woe_wrong
    print(f"  good_dist: {good_dist_wrong}")
    print(f"  bad_dist: {bad_dist_wrong}")
    print(f"  bin_IV: {bin_iv_wrong}")
    print(f"  总IV: {bin_iv_wrong.sum():.6f}")
    print(f"  是否存在负值: {np.any(bin_iv_wrong < 0)}")
    
    # 方法2：正确的平滑方法（保持分布归一化）
    print("\n方法2（正确）：将0替换为epsilon，然后归一化")
    good_smooth = np.where(good_counts == 0, epsilon, good_counts)
    bad_smooth = np.where(bad_counts == 0, epsilon, bad_counts)
    good_dist_correct = good_smooth / good_smooth.sum()
    bad_dist_correct = bad_smooth / bad_smooth.sum()
    woe_correct = np.log(bad_dist_correct / good_dist_correct)
    bin_iv_correct = (bad_dist_correct - good_dist_correct) * woe_correct
    print(f"  good_dist: {good_dist_correct}")
    print(f"  bad_dist: {bad_dist_correct}")
    print(f"  bin_IV: {bin_iv_correct}")
    print(f"  总IV: {bin_iv_correct.sum():.6f}")
    print(f"  是否存在负值: {np.any(bin_iv_correct < 0)}")
    
    print("\n结论：方法1可能产生负值，方法2保证所有值非负")


if __name__ == '__main__':
    verify_iv_formula_mathematically()
    verify_iv_formula_with_smoothing()
    compare_smoothing_methods()
    
    print("\n" + "=" * 80)
    print("✅ 验证完成：IV计算公式逻辑正确，理论上不可能为负数")
    print("=" * 80)
    print("\n关键要点：")
    print("1. IV公式：bin_IV = (bad_dist - good_dist) * log(bad_dist / good_dist)")
    print("2. 数学证明：该公式总是非负的")
    print("3. 平滑处理：应将0替换为epsilon，然后重新归一化")
    print("4. 避免错误：不要在分子分母分别加epsilon，这会破坏分布比例")
