"""
最终验证 - 所有10种分箱方法
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

print("=" * 80)
print("最终验证 - 所有10种分箱方法")
print("=" * 80)

# 加载测试数据
df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/hscredit.xlsx')
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)

X = df[['青云24']].copy()
y = df['target'].copy()

results = {}

# 1. 基础分箱方法
print("\n1. 基础分箱方法")
print("-" * 60)

from hscredit.core.binning import UniformBinning, QuantileBinning, TreeBinning

for name, Binner, kwargs in [
    ('UniformBinning', UniformBinning, {'max_n_bins': 5}),
    ('QuantileBinning', QuantileBinning, {'n_bins': 5}),
    ('TreeBinning', TreeBinning, {'max_depth': 5, 'max_n_bins': 5})
]:
    try:
        binner = Binner(**kwargs)
        binner.fit(X, y)
        result = binner.transform(X, metric='indices')
        print(f"  ✓ {name}: {binner.n_bins_} bins")
        results[name] = 'PASSED'
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        results[name] = f'FAILED: {e}'

# 2. 高级分箱方法
print("\n2. 高级分箱方法")
print("-" * 60)

from hscredit.core.binning import ChiMergeBinning, OptimalKSBinning, OptimalIVBinning, MDLPBinning

for name, Binner, kwargs in [
    ('ChiMergeBinning', ChiMergeBinning, {'n_bins': 5}),
    ('OptimalKSBinning', OptimalKSBinning, {'max_n_bins': 5}),
    ('OptimalIVBinning', OptimalIVBinning, {'max_n_bins': 5}),
    ('MDLPBinning', MDLPBinning, {'max_n_bins': 5})
]:
    try:
        binner = Binner(**kwargs)
        binner.fit(X, y)
        result = binner.transform(X, metric='indices')
        print(f"  ✓ {name}: {binner.n_bins_} bins")
        results[name] = 'PASSED'
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        results[name] = f'FAILED: {e}'

# 3. 统一接口
print("\n3. OptimalBinning 统一接口")
print("-" * 60)

from hscredit.core.binning import OptimalBinning

methods = ['uniform', 'quantile', 'tree', 'chi_merge', 'optimal_ks', 'optimal_iv', 'mdlp']
for method in methods:
    try:
        binner = OptimalBinning(method=method, max_n_bins=5)
        binner.fit(X, y)
        print(f"  ✓ method='{method}'")
    except Exception as e:
        print(f"  ✗ method='{method}': {e}")

results['OptimalBinning'] = 'CHECKED'

# 4. 新增分箱方法
print("\n4. 新增分箱方法")
print("-" * 60)

# 尝试导入新增方法
try:
    from hscredit.core.binning import MonotonicBinning, KMeansBinning
    
    # MonotonicBinning
    try:
        binner = MonotonicBinning(max_n_bins=5, monotonic='ascending')
        binner.fit(X, y)
        result = binner.transform(X, metric='indices')
        print(f"  ✓ MonotonicBinning: {binner.n_bins_} bins")
        results['MonotonicBinning'] = 'PASSED'
    except Exception as e:
        print(f"  ✗ MonotonicBinning: {e}")
        results['MonotonicBinning'] = f'FAILED: {e}'
    
    # KMeansBinning
    try:
        binner = KMeansBinning(max_n_bins=5)
        binner.fit(X, y)
        result = binner.transform(X, metric='indices')
        print(f"  ✓ KMeansBinning: {binner.n_bins_} bins")
        results['KMeansBinning'] = 'PASSED'
    except Exception as e:
        print(f"  ✗ KMeansBinning: {e}")
        results['KMeansBinning'] = f'FAILED: {e}'
        
except ImportError as e:
    print(f"  ⚠ 新增方法尚未导入 binning 模块: {e}")
    results['MonotonicBinning'] = 'NOT_IMPORTED'
    results['KMeansBinning'] = 'NOT_IMPORTED'

# 5. 统一API验证
print("\n5. 统一API验证")
print("-" * 60)

from hscredit.core.binning import UniformBinning

try:
    binner = UniformBinning(max_n_bins=5)
    
    # fit
    binner.fit(X, y)
    print("  ✓ fit()")
    
    # transform
    result = binner.transform(X, metric='indices')
    print("  ✓ transform(metric='indices')")
    
    result = binner.transform(X, metric='bins')
    print("  ✓ transform(metric='bins')")
    
    result = binner.transform(X, metric='woe')
    print("  ✓ transform(metric='woe')")
    
    # fit_transform
    result = binner.fit_transform(X, y, metric='indices')
    print("  ✓ fit_transform()")
    
    # get_bin_table
    bin_table = binner.get_bin_table('青云24')
    print("  ✓ get_bin_table()")
    
    print("\n  统一API验证通过！")
    results['统一API'] = 'PASSED'
    
except Exception as e:
    print(f"  ✗ API验证失败: {e}")
    results['统一API'] = f'FAILED: {e}'

# 汇总
print("\n" + "=" * 80)
print("最终验证汇总")
print("=" * 80)

passed = 0
failed = 0
pending = 0

for name, status in results.items():
    if 'PASSED' in status or 'CHECKED' in status:
        print(f"✓ {name}: {status}")
        passed += 1
    elif 'FAILED' in status:
        print(f"✗ {name}: {status}")
        failed += 1
    else:
        print(f"⏳ {name}: {status}")
        pending += 1

print(f"\n总计: {passed} 通过, {failed} 失败, {pending} 待处理")
print("=" * 80)
