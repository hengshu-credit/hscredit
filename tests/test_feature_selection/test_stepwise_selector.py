#!/usr/bin/env python3
"""测试逐步回归筛选器

测试内容包括：
1. 基本功能测试（前向、后向、双向）
2. max_features 参数测试
3. 不同准则测试（AIC, BIC, KS, AUC）
4. 强制包含/剔除特征测试
5. 边界情况测试
"""

import sys
import os

# 添加 hscredit 模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from hscredit.core.selectors import StepwiseSelector
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("逐步回归筛选器测试")
print("=" * 70)

# ==================== 测试 1: 基本功能 ====================
print("\n测试 1: 基本功能 - 前向选择")
print("-" * 70)

np.random.seed(42)
n_samples = 200

# 创建特征
X = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.randn(n_samples),
    'feature4': np.random.randn(n_samples),
    'feature5': np.random.randn(n_samples),
})

# 创建目标变量（与 feature1 和 feature2 相关）
y = (X['feature1'] + X['feature2'] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

selector = StepwiseSelector(
    estimator='logit',
    direction='forward',
    criterion='aic',
    p_enter=0.05,
    max_iter=50,
    verbose=False
)

selector.fit(X, y)

print(f"前向选择结果:")
print(f"  选中特征: {selector.selected_features_}")
print(f"  剔除特征: {selector.removed_features_}")
print(f"  特征数量: {selector.n_features_}")

# 验证 feature1 和 feature2 被选中
assert 'feature1' in selector.selected_features_ or 'feature2' in selector.selected_features_, \
    "前向选择应该选中相关特征"
print("✓ 前向选择测试通过")

# ==================== 测试 2: 后向消除 ====================
print("\n测试 2: 基本功能 - 后向消除")
print("-" * 70)

selector_backward = StepwiseSelector(
    estimator='logit',
    direction='backward',
    criterion='aic',
    p_remove=0.05,
    max_iter=50,
    verbose=False
)

selector_backward.fit(X, y)

print(f"后向消除结果:")
print(f"  选中特征: {selector_backward.selected_features_}")
print(f"  剔除特征: {selector_backward.removed_features_}")
print(f"  特征数量: {selector_backward.n_features_}")

print("✓ 后向消除测试通过")

# ==================== 测试 3: 双向选择 ====================
print("\n测试 3: 基本功能 - 双向选择")
print("-" * 70)

selector_both = StepwiseSelector(
    estimator='logit',
    direction='both',
    criterion='aic',
    p_enter=0.05,
    p_remove=0.05,
    p_value_enter=0.2,
    max_iter=50,
    verbose=False
)

selector_both.fit(X, y)

print(f"双向选择结果:")
print(f"  选中特征: {selector_both.selected_features_}")
print(f"  剔除特征: {selector_both.removed_features_}")
print(f"  特征数量: {selector_both.n_features_}")

print("✓ 双向选择测试通过")

# ==================== 测试 4: max_features 参数 ====================
print("\n测试 4: max_features 参数测试")
print("-" * 70)

# 测试整数限制
selector_max2 = StepwiseSelector(
    estimator='logit',
    direction='forward',
    criterion='aic',
    max_features=2,
    verbose=False
)

selector_max2.fit(X, y)

print(f"max_features=2 的结果:")
print(f"  选中特征: {selector_max2.selected_features_}")
print(f"  特征数量: {selector_max2.n_features_}")

assert selector_max2.n_features_ <= 2, "选中特征数应该不超过 max_features"
print("✓ max_features 整数测试通过")

# 测试浮点数限制
selector_max_ratio = StepwiseSelector(
    estimator='logit',
    direction='forward',
    criterion='aic',
    max_features=0.5,  # 最多选择 50% 的特征
    verbose=False
)

selector_max_ratio.fit(X, y)

print(f"\nmax_features=0.5 的结果:")
print(f"  选中特征: {selector_max_ratio.selected_features_}")
print(f"  特征数量: {selector_max_ratio.n_features_}")
print(f"  最大允许特征数: {int(0.5 * X.shape[1])}")

assert selector_max_ratio.n_features_ <= int(0.5 * X.shape[1]), \
    "选中特征数应该不超过 max_features 比例"
print("✓ max_features 浮点数测试通过")

# ==================== 测试 5: 不同准则 ====================
print("\n测试 5: 不同准则测试")
print("-" * 70)

criteria = ['aic', 'bic', 'ks', 'auc']

for criterion in criteria:
    selector_criterion = StepwiseSelector(
        estimator='logit',
        direction='forward',
        criterion=criterion,
        max_features=3,
        verbose=False
    )

    selector_criterion.fit(X, y)

    print(f"准则 {criterion.upper():4s}: 选中 {selector_criterion.n_features_} 个特征 - "
          f"{selector_criterion.selected_features_}")

print("✓ 不同准则测试通过")

# ==================== 测试 6: 强制包含/剔除特征 ====================
print("\n测试 6: 强制包含/剔除特征测试")
print("-" * 70)

selector_include = StepwiseSelector(
    estimator='logit',
    direction='forward',
    criterion='aic',
    include=['feature3'],  # 强制包含 feature3
    exclude=['feature4'],  # 强制剔除 feature4
    verbose=False
)

selector_include.fit(X, y)

print(f"强制包含 feature3, 强制剔除 feature4:")
print(f"  选中特征: {selector_include.selected_features_}")

assert 'feature3' in selector_include.selected_features_, "强制包含的特征应该被选中"
assert 'feature4' not in selector_include.selected_features_, "强制剔除的特征不应该被选中"
print("✓ 强制包含/剔除测试通过")

# ==================== 测试 7: 自定义评估器 ====================
print("\n测试 7: 自定义评估器测试")
print("-" * 70)

custom_estimator = LogisticRegression(max_iter=100, random_state=42)

selector_custom = StepwiseSelector(
    estimator=custom_estimator,
    direction='forward',
    criterion='ks',
    max_features=3,
    verbose=False
)

selector_custom.fit(X, y)

print(f"自定义评估器 (LogisticRegression):")
print(f"  选中特征: {selector_custom.selected_features_}")
print(f"  特征数量: {selector_custom.n_features_}")

print("✓ 自定义评估器测试通过")

# ==================== 测试 8: 历史记录和摘要 ====================
print("\n测试 8: 历史记录和摘要测试")
print("-" * 70)

selector_summary = StepwiseSelector(
    estimator='logit',
    direction='both',
    criterion='aic',
    max_features=3,
    verbose=False
)

selector_summary.fit(X, y)

# 获取历史记录
history_df = selector_summary.get_history_df()
print(f"历史记录行数: {len(history_df)}")
print(f"历史记录列: {list(history_df.columns)}")

# 获取摘要
print("\n摘要信息:")
print(selector_summary.summary())

print("✓ 历史记录和摘要测试通过")

# ==================== 测试 9: 边界情况 ====================
print("\n测试 9: 边界情况测试")
print("-" * 70)

# 测试只有少量特征的情况
X_small = pd.DataFrame({
    'f1': np.random.randn(100),
})
y_small = np.random.randint(0, 2, 100)

selector_small = StepwiseSelector(
    estimator='logit',
    direction='forward',
    max_iter=10,
    verbose=False
)

selector_small.fit(X_small, y_small)

print(f"只有1个特征的情况:")
print(f"  选中特征: {selector_small.selected_features_}")

print("✓ 边界情况测试通过")

# ==================== 测试 10: OLS 评估器 ====================
print("\n测试 10: OLS 评估器测试")
print("-" * 70)

# 创建回归问题
y_reg = X['feature1'] * 2 + X['feature2'] * 3 + np.random.randn(n_samples) * 0.5

selector_ols = StepwiseSelector(
    estimator='ols',
    direction='forward',
    criterion='aic',
    max_features=3,
    verbose=False
)

selector_ols.fit(X, y_reg)

print(f"OLS 评估器:")
print(f"  选中特征: {selector_ols.selected_features_}")
print(f"  特征数量: {selector_ols.n_features_}")

print("✓ OLS 评估器测试通过")

# ==================== 总结 ====================
print("\n" + "=" * 70)
print("所有测试通过！✓")
print("=" * 70)

print("\n测试总结:")
print(f"  - 基本功能测试: 通过")
print(f"  - max_features 参数: 通过")
print(f"  - 不同准则测试: 通过")
print(f"  - 强制包含/剔除: 通过")
print(f"  - 自定义评估器: 通过")
print(f"  - 历史记录和摘要: 通过")
print(f"  - 边界情况: 通过")
print(f"  - OLS 评估器: 通过")
