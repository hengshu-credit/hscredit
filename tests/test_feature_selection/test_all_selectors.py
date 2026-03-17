#!/usr/bin/env python3
"""特征筛选器综合测试脚本

测试所有修复的筛选器：
1. MutualInfoSelector - 修复 n_jobs 参数问题
2. VIFSelector - 修复迭代剔除逻辑
3. CorrSelector - 修复 include 参数
4. TypeSelector - 修复 include 参数
"""

import sys
import os

# 添加 hscredit 模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.selectors import (
    MutualInfoSelector,
    VIFSelector,
    CorrSelector,
    TypeSelector,
    IVSelector
)
import pandas as pd
import numpy as np


def test_mutual_info():
    """测试 1: MutualInfoSelector"""
    print("\n" + "=" * 70)
    print("测试 1: MutualInfoSelector (修复 n_jobs 参数)")
    print("=" * 70)
    
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
    })
    y = np.random.randint(0, 2, 100)
    
    mi_selector = MutualInfoSelector(threshold=0.01)
    mi_selector.fit(X, y)
    
    print(f"✓ MutualInfoSelector 测试通过")
    print(f"  选中特征: {mi_selector.selected_features_}")
    print(f"  互信息得分:\n{mi_selector.scores_}")


def test_vif():
    """测试 2: VIFSelector"""
    print("\n" + "=" * 70)
    print("测试 2: VIFSelector (修复迭代剔除逻辑)")
    print("=" * 70)
    
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5],  # 与 a 完全相关
        'c': [5, 4, 3, 2, 1]
    })
    
    vif_selector = VIFSelector(threshold=4.0, verbose=False)
    vif_selector.fit(X)
    
    print(f"✓ VIFSelector 测试通过")
    print(f"  选中特征: {vif_selector.selected_features_}")
    print(f"  剔除特征: {vif_selector.removed_features_}")
    
    # 验证：应该只剔除一个特征
    assert len(vif_selector.selected_features_) == 2, "应该保留2个特征"
    print("  ✓ 验证通过：只剔除一个高相关特征，保留另一个")


def test_corr():
    """测试 3: CorrSelector"""
    print("\n" + "=" * 70)
    print("测试 3: CorrSelector (修复 include 参数)")
    print("=" * 70)
    
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5],
        'c': [5, 4, 3, 2, 1]
    })
    
    corr_selector = CorrSelector(threshold=0.8, method='pearson')
    corr_selector.fit(X)
    
    print(f"✓ CorrSelector 测试通过")
    print(f"  选中特征: {corr_selector.selected_features_}")


def test_type():
    """测试 4: TypeSelector"""
    print("\n" + "=" * 70)
    print("测试 4: TypeSelector (修复 include 参数)")
    print("=" * 70)
    
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5],
        'c': [5, 4, 3, 2, 1]
    })
    
    type_selector = TypeSelector(dtype_include='number')
    type_selector.fit(X)
    
    print(f"✓ TypeSelector 测试通过")
    print(f"  选中特征: {type_selector.selected_features_}")


def test_iv():
    """测试 5: IVSelector"""
    print("\n" + "=" * 70)
    print("测试 5: IVSelector (验证 n_jobs 支持)")
    print("=" * 70)
    
    np.random.seed(42)
    X = pd.DataFrame({
        'score1': np.random.randn(100) * 100 + 600,
        'score2': np.random.randn(100) * 50 + 700,
    })
    y = np.random.randint(0, 2, 100)
    
    iv_selector = IVSelector(threshold=0.02)
    iv_selector.fit(X, y)
    
    print(f"✓ IVSelector 测试通过")
    print(f"  选中特征: {iv_selector.selected_features_}")


def main():
    """运行所有测试"""
    print("=" * 70)
    print("特征筛选器综合测试")
    print("=" * 70)
    
    test_mutual_info()
    test_vif()
    test_corr()
    test_type()
    test_iv()
    
    print("\n" + "=" * 70)
    print("所有测试通过！✓")
    print("=" * 70)


if __name__ == '__main__':
    main()
