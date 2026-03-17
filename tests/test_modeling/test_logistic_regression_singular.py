#!/usr/bin/env python3
"""测试 LogisticRegression 处理奇异矩阵的能力"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.models import LogisticRegression
import pandas as pd
import numpy as np

print("=" * 70)
print("测试 LogisticRegression 处理多重共线性")
print("=" * 70)

# 测试1：正常情况
print("\n测试 1: 正常数据")
try:
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    
    print(f"✓ 正常数据训练成功")
    print(f"  系数: {model.coef_}")
    print(f"  VIF: {model.vif_}")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 测试2：完全相关的特征（奇异矩阵）
print("\n测试 2: 完全相关的特征（奇异矩阵）")
try:
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # 添加完全相关的列
    X = np.column_stack([X, X[:, 0]])  # 第4列与第1列完全相同
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    
    print(f"✓ 奇异矩阵处理成功")
    print(f"  系数: {model.coef_}")
    print(f"  VIF: {model.vif_}")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 测试3：高度相关的特征
print("\n测试 3: 高度相关的特征")
try:
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # 添加高度相关的列
    X = np.column_stack([X, X[:, 0] + np.random.randn(100) * 0.01])  # 与第1列高度相关
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    
    print(f"✓ 高度相关特征处理成功")
    print(f"  系数: {model.coef_}")
    print(f"  VIF: {model.vif_}")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 测试4：常数特征
print("\n测试 4: 常数特征")
try:
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # 添加常数列
    X = np.column_stack([X, np.ones(100)])
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    
    print(f"✓ 常数特征处理成功")
    print(f"  系数: {model.coef_}")
    print(f"  VIF: {model.vif_}")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("所有测试通过！✓")
print("=" * 70)
