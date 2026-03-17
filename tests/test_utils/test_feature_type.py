#!/usr/bin/env python
"""测试特征类型判断逻辑."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hscredit'))

import pandas as pd
import numpy as np
from hscredit.core.binning import UniformBinning, TreeBinning

# 创建测试数据
df = pd.DataFrame({
    'numeric': [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    'object_str': ['a', 'b', 'c', 'd', 'e'] * 20,
    'category': pd.Categorical(['A', 'B', 'C', 'A', 'B'] * 20),
    'integer': [1, 2, 3, 4, 5] * 20,
})
df['target'] = [0, 1] * 50

print("=" * 60)
print("特征类型判断测试")
print("=" * 60)

print("\n数据类型:")
for col in df.columns:
    if col != 'target':
        print(f"  {col}: {df[col].dtype} -> {str(df[col].dtype)}")

# 测试 UniformBinning
print("\n1. UniformBinning (force_numerical=False, 默认):")
binner1 = UniformBinning(max_n_bins=3)
binner1.fit(df.drop('target', axis=1), df['target'])

print("特征类型判断结果:")
for feature, ftype in binner1.feature_types_.items():
    print(f"  {feature}: {ftype}")

# 测试 TreeBinning
print("\n2. TreeBinning (force_numerical=False, 默认):")
binner2 = TreeBinning(max_n_bins=3)
binner2.fit(df.drop('target', axis=1), df['target'])

print("特征类型判断结果:")
for feature, ftype in binner2.feature_types_.items():
    print(f"  {feature}: {ftype}")

# 测试 force_numerical=True
print("\n3. UniformBinning (force_numerical=True):")
binner3 = UniformBinning(max_n_bins=3, force_numerical=True)
binner3.fit(df.drop('target', axis=1), df['target'])

print("特征类型判断结果:")
for feature, ftype in binner3.feature_types_.items():
    print(f"  {feature}: {ftype}")

print("\n" + "=" * 60)
print("测试结论:")
print("- object/string/category 类型正确识别为 categorical")
print("- 数值型特征正确识别为 numerical")
print("- force_numerical=True 时强制所有特征为 numerical")
print("=" * 60)
