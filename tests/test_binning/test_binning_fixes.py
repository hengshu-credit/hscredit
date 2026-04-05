"""
验证分箱方法修复情况
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hscredit.core.binning import (
    UniformBinning, QuantileBinning, TreeBinning, 
    ChiMergeBinning, BestKSBinning, BestIVBinning,
    MDLPBinning, OptimalBinning
)

# 加载测试数据
df = pd.read_excel(_PROJECT_ROOT / "examples" / "hscredit.xlsx")
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)

X = df[['青云24']].copy()
y = df['target'].copy()

print("=" * 80)
print("验证分箱方法修复情况")
print("=" * 80)

results = {}

# 1. 测试 QuantileBinning 修复
print("\n1. 测试 QuantileBinning (force_numerical 和自定义分位点)")
print("-" * 60)
try:
    # 测试 force_numerical
    binner = QuantileBinning(max_n_bins=5, force_numerical=True)
    binner.fit(X, y)
    print(f"  ✓ force_numerical=True 工作正常")
    print(f"    特征类型: {binner.feature_types_}")
    
    # 测试自定义分位点
    binner2 = QuantileBinning(quantiles=[0, 0.2, 0.5, 0.8, 1.0])
    binner2.fit(X, y)
    print(f"  ✓ 自定义分位点工作正常")
    print(f"    切分点: {binner2.splits_}")
    
    results['QuantileBinning'] = 'PASSED'
except Exception as e:
    print(f"  ✗ 失败: {e}")
    results['QuantileBinning'] = f'FAILED: {e}'

# 2. 测试 TreeBinning 修复
print("\n2. 测试 TreeBinning (force_numerical)")
print("-" * 60)
try:
    binner = TreeBinning(max_depth=5, max_n_bins=5, force_numerical=True)
    binner.fit(X, y)
    print(f"  ✓ force_numerical 工作正常")
    print(f"    特征类型: {binner.feature_types_}")
    print(f"    切分点: {binner.splits_}")
    
    # 测试 transform
    result = binner.transform(X, metric='indices')
    print(f"  ✓ transform 工作正常，结果形状: {result.shape}")
    
    results['TreeBinning'] = 'PASSED'
except Exception as e:
    print(f"  ✗ 失败: {e}")
    results['TreeBinning'] = f'FAILED: {e}'

# 3. 测试 ChiMergeBinning 修复
print("\n3. 测试 ChiMergeBinning (max_n_bins)")
print("-" * 60)
try:
    binner = ChiMergeBinning(max_n_bins=5)
    binner.fit(X, y)
    print(f"  ✓ max_n_bins 工作正常")
    print(f"    分箱数: {binner.n_bins_}")
    print(f"    切分点: {binner.splits_}")
    
    # 检查分箱统计表行数
    bin_table = binner.get_bin_table('青云24')
    print(f"    分箱统计表行数: {len(bin_table)}")
    
    if len(bin_table) <= 7:  # 5 bins + missing + special
        print(f"  ✓ 分箱数控制在预期范围内")
        results['ChiMergeBinning'] = 'PASSED'
    else:
        print(f"  ⚠ 分箱数仍然过多: {len(bin_table)}")
        results['ChiMergeBinning'] = f'WARNING: 分箱数 {len(bin_table)}'
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    results['ChiMergeBinning'] = f'FAILED: {e}'

# 4. 测试 OptimalBinning 统一接口修复
print("\n4. 测试 OptimalBinning 统一接口 (参数冲突修复)")
print("-" * 60)
methods = ['uniform', 'quantile', 'tree', 'chi', 'best_ks', 'best_iv', 'mdlp']

for method in methods:
    try:
        binner = OptimalBinning(method=method, max_n_bins=5)
        binner.fit(X, y)
        print(f"  ✓ method='{method}' 工作正常")
    except Exception as e:
        print(f"  ✗ method='{method}' 失败: {e}")

results['OptimalBinning'] = 'CHECKED'

# 5. 测试待修复问题：WOE极端值
print("\n5. 测试 WOE 极端值问题 (待修复)")
print("-" * 60)
try:
    binner = UniformBinning(max_n_bins=5)
    binner.fit(X, y)
    bin_table = binner.get_bin_table('青云24')
    
    woe_min = bin_table['woe'].min()
    woe_max = bin_table['woe'].max()
    
    print(f"  WOE 范围: [{woe_min:.4f}, {woe_max:.4f}]")
    
    # 检查极端值
    extreme = bin_table[abs(bin_table['woe']) > 10]
    if len(extreme) > 0:
        print(f"  ⚠ 发现极端WOE值:")
        print(extreme[['bin', 'count', 'woe']])
        results['WOE极端值'] = 'PENDING'
    else:
        print(f"  ✓ 无极端WOE值")
        results['WOE极端值'] = 'FIXED'
except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    results['WOE极端值'] = f'ERROR: {e}'

# 6. 测试待修复问题：_get_bin_labels
print("\n6. 测试 _get_bin_labels 标签映射 (待修复)")
print("-" * 60)
try:
    binner = UniformBinning(max_n_bins=5)
    binner.fit(X, y)
    
    # 测试 transform with metric='bins'
    result = binner.transform(X, metric='bins')
    print(f"  ✓ transform(metric='bins') 工作正常")
    print(f"    结果形状: {result.shape}")
    print(f"    示例值: {result['青云24'].unique()[:5]}")
    results['_get_bin_labels'] = 'FIXED'
except Exception as e:
    print(f"  ✗ 失败: {e}")
    results['_get_bin_labels'] = f'PENDING: {e}'

# 汇总
print("\n" + "=" * 80)
print("修复验证汇总")
print("=" * 80)
for item, status in results.items():
    if 'PASSED' in status or 'FIXED' in status:
        print(f"✓ {item}: {status}")
    elif 'PENDING' in status:
        print(f"⏳ {item}: {status}")
    else:
        print(f"✗ {item}: {status}")
