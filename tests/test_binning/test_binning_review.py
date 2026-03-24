"""
hscredit 分箱方法全面测试脚本

测试所有分箱方法的以下方面：
1. 参数设置 max_n_bins/n_bins 是否生效
2. 特征类型检测是否正确（numerical vs categorical）
3. 缺失值和特殊值处理是否正确
4. WOE/IV计算是否正确
5. transform 方法是否正常工作
6. 边界条件处理（如所有值相同、只有一个唯一值等）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 加载测试数据
print("=" * 80)
print("加载测试数据")
print("=" * 80)
df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/utils/hscredit.xlsx')
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)

# 选择几个特征进行测试
numerical_features = ['青云24', '同盾分', '百融分']
categorical_features = ['学历', '婚姻状况'] if '学历' in df.columns else []

print(f"数据形状: {df.shape}")
print(f"目标变量分布:\n{df['target'].value_counts()}")
print(f"数值型特征: {numerical_features}")
print(f"类别型特征: {categorical_features}")

# 导入所有分箱方法
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.binning import (
    UniformBinning, QuantileBinning, TreeBinning, 
    ChiMergeBinning, BestKSBinning, BestIVBinning,
    MDLPBinning, OptimalBinning
)

# 测试结果记录
results = {
    'UniformBinning': {'status': 'PENDING', 'issues': []},
    'QuantileBinning': {'status': 'PENDING', 'issues': []},
    'TreeBinning': {'status': 'PENDING', 'issues': []},
    'ChiMergeBinning': {'status': 'PENDING', 'issues': []},
    'BestKSBinning': {'status': 'PENDING', 'issues': []},
    'BestIVBinning': {'status': 'PENDING', 'issues': []},
    'MDLPBinning': {'status': 'PENDING', 'issues': []},
    'OptimalBinning': {'status': 'PENDING', 'issues': []},
}

def check_max_bins(binner, expected_max_bins, feature_name):
    """检查最大分箱数是否生效"""
    actual_bins = binner.n_bins_.get(feature_name, 0)
    # 考虑缺失值和特殊值箱
    if actual_bins > expected_max_bins + 2:  # +2 for missing and special bins
        return False, f"分箱数 {actual_bins} 超过预期最大值 {expected_max_bins} + 2"
    return True, f"分箱数 {actual_bins} 符合预期"

def check_woe_iv_calculation(bin_table):
    """检查WOE和IV计算是否正确"""
    issues = []
    
    if 'woe' not in bin_table.columns or 'bin_iv' not in bin_table.columns:
        issues.append("缺少woe或bin_iv列")
        return issues
    
    # 检查WOE是否为有限值
    if not np.all(np.isfinite(bin_table['woe'].values)):
        issues.append("WOE值包含非有限值(inf或nan)")
    
    # 检查IV是否为非负值
    if not np.all(bin_table['bin_iv'].values >= -1e-10):  # 允许小的数值误差
        issues.append("IV值包含负值")
    
    # 检查total_iv是否一致
    if 'total_iv' in bin_table.columns:
        unique_total_iv = bin_table['total_iv'].unique()
        if len(unique_total_iv) != 1:
            issues.append(f"total_iv不一致: {unique_total_iv}")
    
    return issues

def check_transform_works(binner, X_test, feature_name):
    """检查transform方法是否正常工作"""
    issues = []
    
    try:
        # 测试不同的metric
        for metric in ['indices', 'bins', 'woe']:
            result = binner.transform(X_test, metric=metric)
            if result is None or len(result) != len(X_test):
                issues.append(f"transform(metric='{metric}')返回结果异常")
            
            # 检查是否有nan值（除了输入中的nan）
            if metric == 'woe':
                if result[feature_name].isna().any():
                    issues.append(f"transform(metric='woe')返回结果包含nan值")
    except Exception as e:
        issues.append(f"transform方法抛出异常: {str(e)}")
    
    return issues

# ==================== 1. 测试 UniformBinning ====================
print("\n" + "=" * 80)
print("1. 测试 UniformBinning (等距分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = UniformBinning(max_n_bins=5)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 5, '青云24')
    if not ok:
        results['UniformBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['UniformBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['UniformBinning']['issues'].extend(transform_issues)
    
    # 测试2: 边界条件 - 所有值相同
    X_same = pd.DataFrame({'same_value': [100] * 100})
    y_same = pd.Series([0] * 50 + [1] * 50)
    try:
        binner_same = UniformBinning(max_n_bins=5)
        binner_same.fit(X_same, y_same)
        print("  ✓ 所有值相同的情况处理正常")
    except Exception as e:
        results['UniformBinning']['issues'].append(f"所有值相同的情况处理失败: {str(e)}")
    
    # 测试3: 缺失值处理
    X_missing = X.copy()
    X_missing.loc[:10, '青云24'] = np.nan
    try:
        binner_missing = UniformBinning(max_n_bins=5, missing_separate=True)
        binner_missing.fit(X_missing, y)
        bin_table_missing = binner_missing.get_bin_table('青云24')
        # 检查是否有缺失值箱
        if 'missing' not in str(bin_table_missing['bin'].values):
            results['UniformBinning']['issues'].append("缺失值未正确处理为单独一箱")
        print("  ✓ 缺失值处理正常")
    except Exception as e:
        results['UniformBinning']['issues'].append(f"缺失值处理失败: {str(e)}")
    
    # 测试4: 特殊值处理
    X_special = X.copy()
    X_special.loc[:5, '青云24'] = -999
    try:
        binner_special = UniformBinning(max_n_bins=5, special_codes=[-999])
        binner_special.fit(X_special, y)
        print("  ✓ 特殊值处理正常")
    except Exception as e:
        results['UniformBinning']['issues'].append(f"特殊值处理失败: {str(e)}")
    
    # 测试5: 截断功能
    try:
        binner_clip = UniformBinning(max_n_bins=5, left_clip=0.05, right_clip=0.95)
        binner_clip.fit(X, y)
        print("  ✓ 截断功能正常")
    except Exception as e:
        results['UniformBinning']['issues'].append(f"截断功能失败: {str(e)}")
    
    if not results['UniformBinning']['issues']:
        results['UniformBinning']['status'] = 'PASSED'
    else:
        results['UniformBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['UniformBinning']['status'] = 'ERROR'
    results['UniformBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 2. 测试 QuantileBinning ====================
print("\n" + "=" * 80)
print("2. 测试 QuantileBinning (等频分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = QuantileBinning(n_bins=5, max_n_bins=10)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 10, '青云24')
    if not ok:
        results['QuantileBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['QuantileBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['QuantileBinning']['issues'].extend(transform_issues)
    
    # 测试2: 自定义分位点
    try:
        binner_custom = QuantileBinning(quantiles=[0, 0.2, 0.5, 0.8, 1.0])
        binner_custom.fit(X, y)
        print("  ✓ 自定义分位点功能正常")
    except Exception as e:
        results['QuantileBinning']['issues'].append(f"自定义分位点功能失败: {str(e)}")
    
    # 测试3: 边界条件 - 大量重复值
    X_dup = pd.DataFrame({'dup_feature': [50] * 200 + list(range(800))})
    y_dup = pd.Series([0] * 500 + [1] * 500)
    try:
        binner_dup = QuantileBinning(n_bins=5)
        binner_dup.fit(X_dup, y_dup)
        print("  ✓ 大量重复值处理正常")
    except Exception as e:
        results['QuantileBinning']['issues'].append(f"大量重复值处理失败: {str(e)}")
    
    if not results['QuantileBinning']['issues']:
        results['QuantileBinning']['status'] = 'PASSED'
    else:
        results['QuantileBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['QuantileBinning']['status'] = 'ERROR'
    results['QuantileBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 3. 测试 TreeBinning ====================
print("\n" + "=" * 80)
print("3. 测试 TreeBinning (决策树分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = TreeBinning(max_depth=5, max_n_bins=5)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 5, '青云24')
    if not ok:
        results['TreeBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['TreeBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['TreeBinning']['issues'].extend(transform_issues)
    
    # 测试2: 单调性约束
    try:
        binner_mono = TreeBinning(max_depth=5, monotonic='ascending')
        binner_mono.fit(X, y)
        # 检查坏样本率是否单调
        bin_table_mono = binner_mono.get_bin_table('青云24')
        bad_rates = bin_table_mono['bad_rate'].values
        is_monotonic = all(bad_rates[i] <= bad_rates[i+1] for i in range(len(bad_rates)-1))
        if not is_monotonic:
            results['TreeBinning']['issues'].append("单调递增约束未生效")
        print("  ✓ 单调性约束功能正常")
    except Exception as e:
        results['TreeBinning']['issues'].append(f"单调性约束功能失败: {str(e)}")
    
    if not results['TreeBinning']['issues']:
        results['TreeBinning']['status'] = 'PASSED'
    else:
        results['TreeBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['TreeBinning']['status'] = 'ERROR'
    results['TreeBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 4. 测试 ChiMergeBinning ====================
print("\n" + "=" * 80)
print("4. 测试 ChiMergeBinning (卡方分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = ChiMergeBinning(n_bins=5, max_n_bins=10)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 10, '青云24')
    if not ok:
        results['ChiMergeBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['ChiMergeBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['ChiMergeBinning']['issues'].extend(transform_issues)
    
    # 测试2: 自定义卡方阈值
    try:
        binner_chi = ChiMergeBinning(n_bins=5, min_chi2_threshold=5.0)
        binner_chi.fit(X, y)
        print("  ✓ 自定义卡方阈值功能正常")
    except Exception as e:
        results['ChiMergeBinning']['issues'].append(f"自定义卡方阈值功能失败: {str(e)}")
    
    if not results['ChiMergeBinning']['issues']:
        results['ChiMergeBinning']['status'] = 'PASSED'
    else:
        results['ChiMergeBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['ChiMergeBinning']['status'] = 'ERROR'
    results['ChiMergeBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 5. 测试 BestKSBinning ====================
print("\n" + "=" * 80)
print("5. 测试 BestKSBinning (最优KS分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = BestKSBinning(max_n_bins=5)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 5, '青云24')
    if not ok:
        results['BestKSBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['BestKSBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['BestKSBinning']['issues'].extend(transform_issues)
    
    if not results['BestKSBinning']['issues']:
        results['BestKSBinning']['status'] = 'PASSED'
    else:
        results['BestKSBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['BestKSBinning']['status'] = 'ERROR'
    results['BestKSBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 6. 测试 BestIVBinning ====================
print("\n" + "=" * 80)
print("6. 测试 BestIVBinning (最优IV分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = BestIVBinning(max_n_bins=5)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 5, '青云24')
    if not ok:
        results['BestIVBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['BestIVBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['BestIVBinning']['issues'].extend(transform_issues)
    
    if not results['BestIVBinning']['issues']:
        results['BestIVBinning']['status'] = 'PASSED'
    else:
        results['BestIVBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['BestIVBinning']['status'] = 'ERROR'
    results['BestIVBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 7. 测试 MDLPBinning ====================
print("\n" + "=" * 80)
print("7. 测试 MDLPBinning (MDLP分箱)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试1: 基本功能
    binner = MDLPBinning(max_n_bins=5)
    binner.fit(X, y)
    
    # 检查分箱数
    ok, msg = check_max_bins(binner, 5, '青云24')
    if not ok:
        results['MDLPBinning']['issues'].append(f"max_n_bins不生效: {msg}")
    
    # 检查WOE/IV计算
    bin_table = binner.get_bin_table('青云24')
    woe_issues = check_woe_iv_calculation(bin_table)
    results['MDLPBinning']['issues'].extend(woe_issues)
    
    # 检查transform
    transform_issues = check_transform_works(binner, X, '青云24')
    results['MDLPBinning']['issues'].extend(transform_issues)
    
    if not results['MDLPBinning']['issues']:
        results['MDLPBinning']['status'] = 'PASSED'
    else:
        results['MDLPBinning']['status'] = 'FAILED'
    
    print(f"  分箱统计表:\n{bin_table}")
    
except Exception as e:
    results['MDLPBinning']['status'] = 'ERROR'
    results['MDLPBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 8. 测试 OptimalBinning ====================
print("\n" + "=" * 80)
print("8. 测试 OptimalBinning (统一接口)")
print("=" * 80)

try:
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    # 测试所有方法
    methods = ['uniform', 'quantile', 'tree', 'chi_merge', 'optimal_ks', 'optimal_iv', 'mdlp']
    
    for method in methods:
        try:
            binner = OptimalBinning(method=method, max_n_bins=5)
            binner.fit(X, y)
            
            # 检查分箱数
            ok, msg = check_max_bins(binner, 5, '青云24')
            if not ok:
                results['OptimalBinning']['issues'].append(f"method={method}: max_n_bins不生效: {msg}")
            
            # 检查transform
            transform_issues = check_transform_works(binner, X, '青云24')
            for issue in transform_issues:
                results['OptimalBinning']['issues'].append(f"method={method}: {issue}")
            
            print(f"  ✓ method='{method}' 正常")
        except Exception as e:
            results['OptimalBinning']['issues'].append(f"method={method}: 测试失败: {str(e)}")
            print(f"  ✗ method='{method}' 失败: {str(e)}")
    
    if not results['OptimalBinning']['issues']:
        results['OptimalBinning']['status'] = 'PASSED'
    else:
        results['OptimalBinning']['status'] = 'FAILED'
    
except Exception as e:
    results['OptimalBinning']['status'] = 'ERROR'
    results['OptimalBinning']['issues'].append(f"测试执行失败: {str(e)}")
    print(f"  ✗ 测试失败: {str(e)}")

# ==================== 汇总测试结果 ====================
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

for method, result in results.items():
    status = result['status']
    issues = result['issues']
    
    if status == 'PASSED':
        print(f"✓ {method}: 通过")
    elif status == 'FAILED':
        print(f"✗ {method}: 失败 ({len(issues)} 个问题)")
        for issue in issues:
            print(f"    - {issue}")
    elif status == 'ERROR':
        print(f"✗ {method}: 错误")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"? {method}: 未测试")

# 保存详细结果
print("\n" + "=" * 80)
print("保存详细测试结果到 binning_test_results.txt")
print("=" * 80)

with open('/Users/xiaoxi/CodeBuddy/hscredit/binning_test_results.txt', 'w') as f:
    f.write("hscredit 分箱方法测试报告\n")
    f.write("=" * 80 + "\n\n")
    
    for method, result in results.items():
        f.write(f"\n{method}\n")
        f.write("-" * 40 + "\n")
        f.write(f"状态: {result['status']}\n")
        if result['issues']:
            f.write("问题列表:\n")
            for i, issue in enumerate(result['issues'], 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("问题列表: 无\n")
        f.write("\n")

print("测试完成！")
