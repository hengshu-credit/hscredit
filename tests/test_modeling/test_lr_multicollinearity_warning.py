#!/usr/bin/env python3
"""测试 LogisticRegression 的多重共线性警告"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

# 确保显示所有警告
import warnings
warnings.simplefilter("always")

from hscredit.core.models import LogisticRegression
import numpy as np

print("=" * 70)
print("测试 LogisticRegression 多重共线性检测和警告")
print("=" * 70)

# 测试1：完全相关的特征
print("\n测试 1: 完全相关的特征（应该触发 VIF=inf 警告）")
np.random.seed(42)
X = np.random.randn(100, 3)
X = np.column_stack([X, X[:, 0]])  # 第4列与第1列完全相同
y = np.random.randint(0, 2, 100)

try:
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    print(f"✓ 模型训练成功")
    print(f"  VIF: {model.vif_}")
    print(f"  inf VIF 数量: {np.sum(np.isinf(model.vif_))}")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试2：高度相关的特征
print("\n测试 2: 高度相关的特征（应该触发 VIF > 10 警告）")
np.random.seed(42)
X = np.random.randn(100, 3)
X = np.column_stack([X, X[:, 0] + np.random.randn(100) * 0.01])
y = np.random.randint(0, 2, 100)

try:
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=100, random_state=42)
    model.fit(X, y)
    print(f"✓ 模型训练成功")
    print(f"  VIF: {model.vif_}")
    print(f"  高 VIF 数量: {np.sum((model.vif_ > 10) & (model.vif_ != np.inf))}")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
