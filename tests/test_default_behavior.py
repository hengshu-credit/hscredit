"""
验证 force_numerical 默认值为 False 后的行为
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.binning import (
    UniformBinning, QuantileBinning, TreeBinning
)

# 加载测试数据
df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/hscredit.xlsx')
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)

X = df[['青云24']].copy()
y = df['target'].copy()

print("=" * 80)
print("验证 force_numerical 默认行为")
print("=" * 80)

# 测试1: UniformBinning 默认行为
print("\n1. UniformBinning - 默认行为 (force_numerical=False)")
print("-" * 60)
try:
    binner = UniformBinning(max_n_bins=5)
    binner.fit(X, y)
    print(f"  特征类型: {binner.feature_types_}")
    print(f"  切分点: {binner.splits_}")
    print(f"  分箱数: {binner.n_bins_}")
    
    # 检查是否正确识别为数值型
    if binner.feature_types_['青云24'] == 'numerical':
        print("  ✓ 正确识别为数值型")
    else:
        print("  ⚠ 被识别为类别型")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试2: QuantileBinning 默认行为
print("\n2. QuantileBinning - 默认行为 (force_numerical=False)")
print("-" * 60)
try:
    binner = QuantileBinning(n_bins=5)
    binner.fit(X, y)
    print(f"  特征类型: {binner.feature_types_}")
    print(f"  切分点: {binner.splits_}")
    print(f"  分箱数: {binner.n_bins_}")
    
    if binner.feature_types_['青云24'] == 'numerical':
        print("  ✓ 正确识别为数值型")
    else:
        print("  ⚠ 被识别为类别型")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试3: TreeBinning 默认行为
print("\n3. TreeBinning - 默认行为 (force_numerical=False)")
print("-" * 60)
try:
    binner = TreeBinning(max_depth=5, max_n_bins=5)
    binner.fit(X, y)
    print(f"  特征类型: {binner.feature_types_}")
    print(f"  切分点: {binner.splits_}")
    print(f"  分箱数: {binner.n_bins_}")
    
    if binner.feature_types_['青云24'] == 'numerical':
        print("  ✓ 正确识别为数值型")
    else:
        print("  ⚠ 被识别为类别型")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试4: 显式设置 force_numerical=True
print("\n4. 显式设置 force_numerical=True")
print("-" * 60)
try:
    binner = UniformBinning(max_n_bins=5, force_numerical=True)
    binner.fit(X, y)
    print(f"  UniformBinning: {binner.feature_types_}")
    
    binner2 = QuantileBinning(n_bins=5, force_numerical=True)
    binner2.fit(X, y)
    print(f"  QuantileBinning: {binner2.feature_types_}")
    
    binner3 = TreeBinning(max_depth=5, max_n_bins=5, force_numerical=True)
    binner3.fit(X, y)
    print(f"  TreeBinning: {binner3.feature_types_}")
    
    print("  ✓ 显式设置 force_numerical=True 工作正常")
except Exception as e:
    print(f"  ✗ 失败: {e}")

# 测试5: 测试类别型特征检测
print("\n5. 测试类别型特征检测")
print("-" * 60)
try:
    # 创建一个类别型特征
    X_cat = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 100})
    y_cat = pd.Series([0, 1, 0, 1, 0, 1] * 100)
    
    binner = UniformBinning(max_n_bins=5)
    binner.fit(X_cat, y_cat)
    print(f"  类别型特征检测结果: {binner.feature_types_}")
    
    if binner.feature_types_['category'] == 'categorical':
        print("  ✓ 正确识别类别型特征")
    else:
        print("  ⚠ 类别型特征未被正确识别")
except Exception as e:
    print(f"  ✗ 失败: {e}")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
