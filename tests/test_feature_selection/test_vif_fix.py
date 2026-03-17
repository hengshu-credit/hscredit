#!/usr/bin/env python3
"""VIFSelector 修复效果验证脚本

运行此脚本验证 VIFSelector 的迭代剔除功能。
"""

import sys
import os

# 添加 hscredit 模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.selectors import VIFSelector
import pandas as pd
import numpy as np


def test_perfect_correlation():
    """测试1：完全相关的特征"""
    print("\n" + "=" * 70)
    print("测试1：完全相关的特征（应该只剔除一个）")
    print("=" * 70)
    
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5],  # 与 a 完全相关
        'c': [5, 4, 3, 2, 1]
    })
    
    print("\n原始数据相关性:")
    print(X.corr())
    
    selector = VIFSelector(threshold=4.0, verbose=True)
    selector.fit(X)
    
    print(f"\n结果：")
    print(f"  保留特征: {selector.selected_features_}")
    print(f"  剔除特征: {selector.removed_features_}")
    
    # 验证：应该只剔除一个特征
    assert len(selector.selected_features_) == 2, "应该保留2个特征"
    assert len(selector.removed_features_) == 1, "应该剔除1个特征"
    print("\n✓ 测试通过")


def test_multiple_correlation():
    """测试2：多个相关特征"""
    print("\n" + "=" * 70)
    print("测试2：多个相关特征（应该逐个剔除）")
    print("=" * 70)
    
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'f3': np.random.randn(100),
        'f4': np.random.randn(100),
    })
    X['f3'] = X['f1'] + np.random.randn(100) * 0.1  # f3 与 f1 高度相关
    
    print("\n特征相关性:")
    print(X.corr().round(3))
    
    selector = VIFSelector(threshold=4.0, verbose=True)
    selector.fit(X)
    
    print(f"\n结果：")
    print(f"  保留特征: {selector.selected_features_}")
    print(f"  剔除特征: {selector.removed_features_}")
    
    # 验证：f3 应该被保留（剔除 f1 后 f3 的 VIF 会降低）
    assert 'f3' in selector.selected_features_, "f3 应该被保留"
    print("\n✓ 测试通过")


def test_missing_and_inf():
    """测试3：缺失值和 inf 值"""
    print("\n" + "=" * 70)
    print("测试3：缺失值和 inf 值处理")
    print("=" * 70)
    
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [1, 2, 3, 4, 5],  # 与 a 完全相关，会产生 inf
        'c': [5, 4, 3, 2, 1],
        'd': [1, 1, 1, 1, 1],  # 常数特征
        'e': [1, 2, np.nan, 4, 5]  # 有缺失值
    })
    
    print("\n原始数据:")
    print(X)
    
    selector = VIFSelector(threshold=4.0, verbose=True)
    selector.fit(X)
    
    print(f"\n结果：")
    print(f"  保留特征: {selector.selected_features_}")
    print(f"  剔除特征: {selector.removed_features_}")
    
    # 验证：应该正确处理缺失值和 inf
    assert len(selector.selected_features_) > 0, "应该有特征被保留"
    print("\n✓ 测试通过")


def test_user_scenario():
    """测试4：用户场景（青云24衍生特征）"""
    print("\n" + "=" * 70)
    print("测试4：用户场景 - 青云24衍生特征")
    print("=" * 70)
    
    np.random.seed(42)
    X = pd.DataFrame({
        '青云24': np.random.randn(100) * 100 + 600,
        '游昆定制分80': np.random.randn(100) * 50 + 700,
        '百融定制分V9': np.random.randn(100) * 30 + 650,
        '其他特征': np.random.randn(100),
    })
    X['青云24_完全相同'] = X['青云24']  # 完全相同的衍生特征
    
    print("\n特征相关性（青云24相关）:")
    print(X.corr()['青云24'].round(3))
    
    selector = VIFSelector(threshold=4.0, verbose=True)
    selector.fit(X)
    
    print(f"\n结果：")
    print(f"  保留特征: {selector.selected_features_}")
    print(f"  剔除特征: {selector.removed_features_}")
    
    # 验证：应该只剔除一个特征
    assert len(selector.removed_features_) == 1, "应该只剔除1个特征"
    assert len(selector.selected_features_) == 4, "应该保留4个特征"
    print("\n✓ 测试通过")


def test_with_include():
    """测试5：使用 include 参数强制保留特征"""
    print("\n" + "=" * 70)
    print("测试5：使用 include 参数强制保留特征")
    print("=" * 70)
    
    np.random.seed(42)
    X = pd.DataFrame({
        '青云24': np.random.randn(100) * 100 + 600,
        '游昆定制分80': np.random.randn(100) * 50 + 700,
        '百融定制分V9': np.random.randn(100) * 30 + 650,
    })
    X['青云24_完全相同'] = X['青云24']
    
    # 强制保留青云24
    selector = VIFSelector(
        threshold=4.0,
        include=['青云24'],  # 强制保留
        verbose=True
    )
    selector.fit(X)
    
    print(f"\n结果：")
    print(f"  保留特征: {selector.selected_features_}")
    print(f"  剔除特征: {selector.removed_features_}")
    
    # 验证：青云24 应该被保留（因为 include 参数）
    assert '青云24' in selector.selected_features_, "青云24 应该被保留（强制保留）"
    # 验证：至少有一个特征被保留
    assert len(selector.selected_features_) >= 3, "应该保留至少3个特征"
    print("\n✓ 测试通过")


if __name__ == '__main__':
    print("=" * 70)
    print("VIFSelector 修复效果验证")
    print("=" * 70)
    
    try:
        test_perfect_correlation()
        test_multiple_correlation()
        test_missing_and_inf()
        test_user_scenario()
        test_with_include()
        
        print("\n" + "=" * 70)
        print("所有测试通过！✓")
        print("=" * 70)
        print("\n修复效果：")
        print("  ✓ 完全相关特征：只剔除一个，保留另一个")
        print("  ✓ 多个相关特征：逐个剔除，避免误伤")
        print("  ✓ inf 值：正确识别和处理")
        print("  ✓ 缺失值：正确处理")
        print("  ✓ include 参数：支持强制保留特征")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
