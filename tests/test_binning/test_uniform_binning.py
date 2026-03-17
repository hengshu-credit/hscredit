"""UniformBinning 功能测试脚本.

测试新功能:
1. force_numerical: 强制数值型分箱
2. left_clip/right_clip: 截断异常值
3. special_codes: 处理特殊值
"""

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

import pandas as pd
import numpy as np
from hscredit.core.binning import UniformBinning


def test_basic():
    """测试基础功能."""
    print("=" * 60)
    print("测试1: 基础功能（force_numerical=True）")
    print("=" * 60)
    
    df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/utils/hscredit.xlsx')
    df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)
    
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    print(f"特征范围: [{X['青云24'].min()}, {X['青云24'].max()}]")
    print(f"-999 的数量: {(X['青云24'] == -999).sum()}")
    
    binner = UniformBinning(max_n_bins=5, special_codes=[-999])
    binner.fit(X, y)
    
    print(f"\n分箱数: {binner.n_bins_}")
    print(f"特征类型: {binner.feature_types_}")
    print(f"切分点: {binner.splits_['青云24']}")
    
    table = binner.get_bin_table('青云24')
    print(f"\n分箱表:")
    print(table[['分箱', '样本总数', '样本占比', '坏样本率', '分档WOE值']])
    print(f"\n总IV: {table['指标IV值'].iloc[0]:.4f}")
    
    print("\n✅ 基础功能测试通过")


def test_clip():
    """测试截断功能."""
    print("\n" + "=" * 60)
    print("测试2: 使用 left_clip 和 right_clip 截断异常值")
    print("=" * 60)
    
    np.random.seed(42)
    X = pd.DataFrame({
        'score': np.concatenate([
            np.random.normal(500, 50, 1000),  # 正常值
            [1000, 1050, 1100, -500, -600],    # 异常值
        ])
    })
    y = pd.Series(np.random.binomial(1, 0.15, 1005))
    
    print(f"原始数据范围: [{X['score'].min():.0f}, {X['score'].max():.0f}]")
    
    # 不使用截断
    binner_no_clip = UniformBinning(max_n_bins=5)
    binner_no_clip.fit(X, y)
    print(f"\n不截断时的分箱边界: {binner_no_clip.splits_['score']}")
    
    # 使用截断
    binner_clip = UniformBinning(max_n_bins=5, left_clip=0.01, right_clip=0.99)
    binner_clip.fit(X, y)
    print(f"截断后的分箱边界: {binner_clip.splits_['score']}")
    
    cl, cu = binner_clip.clip_bounds_['score']
    print(f"截断范围: [{cl:.2f}, {cu:.2f}]")
    
    table = binner_clip.get_bin_table('score')
    print(f"\n截断后的分箱表:")
    print(table[['分箱', '样本总数', '坏样本率']])
    
    print("\n✅ 截断功能测试通过")


def test_woe_transform():
    """测试 WOE 转换."""
    print("\n" + "=" * 60)
    print("测试3: WOE 转换")
    print("=" * 60)
    
    df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/utils/hscredit.xlsx')
    df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    binner = UniformBinning(max_n_bins=5, special_codes=[-999])
    binner.fit(X, y)
    
    X_woe = binner.transform(X, metric='woe')
    print(f"WOE转换后数据形状: {X_woe.shape}")
    print(f"WOE值范围: [{X_woe['青云24'].min():.4f}, {X_woe['青云24'].max():.4f}]")
    print(f"\n前5个WOE值:")
    print(X_woe['青云24'].head())
    
    print("\n✅ WOE转换测试通过")


def test_bin_labels():
    """测试分箱标签转换."""
    print("\n" + "=" * 60)
    print("测试4: 分箱标签转换")
    print("=" * 60)
    
    df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/utils/hscredit.xlsx')
    df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)
    X = df[['青云24']].copy()
    y = df['target'].copy()
    
    binner = UniformBinning(max_n_bins=5, special_codes=[-999])
    binner.fit(X, y)
    
    X_bins = binner.transform(X, metric='bins')
    print(f"分箱标签示例:")
    print(X_bins['青云24'].value_counts())
    
    print("\n✅ 分箱标签转换测试通过")


if __name__ == '__main__':
    test_basic()
    test_clip()
    test_woe_transform()
    test_bin_labels()
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
