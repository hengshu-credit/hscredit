"""
深入测试分箱方法的问题
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

from hscredit.core.binning import (
    UniformBinning, QuantileBinning, TreeBinning, 
    ChiMergeBinning, OptimalKSBinning, OptimalIVBinning,
    MDLPBinning, OptimalBinning
)

# 加载测试数据
df = pd.read_excel('/Users/xiaoxi/CodeBuddy/hscredit/hscredit/examples/utils/hscredit.xlsx')
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)

X = df[['青云24']].copy()
y = df['target'].copy()

print("=" * 80)
print("问题1: OptimalBinning统一接口 - uniform和quantile方法参数冲突")
print("=" * 80)

# 查看optimal_binning.py中的UniformBinning和QuantileBinning定义
print("\n问题分析:")
print("在optimal_binning.py中:")
print("- UniformBinning的__init__接收n_bins参数，并调用super().__init__(max_n_bins=n_bins)")
print("- QuantileBinning的__init__接收n_bins参数，并调用super().__init__(max_n_bins=n_bins)")
print("- 但OptimalBinning.fit()在创建这些实例时，传递了common_params包含max_n_bins")
print("- 导致max_n_bins被传递两次")

print("\n" + "=" * 80)
print("问题2: TreeBinning/ChiMergeBinning - transform返回结果长度不匹配")
print("=" * 80)

# 测试TreeBinning
try:
    binner = TreeBinning(max_depth=5, max_n_bins=5)
    binner.fit(X, y)
    
    print(f"\nTreeBinning splits_: {binner.splits_}")
    print(f"TreeBinning n_bins_: {binner.n_bins_}")
    print(f"TreeBinning feature_types_: {binner.feature_types_}")
    
    # 检查bin_table
    bin_table = binner.get_bin_table('青云24')
    print(f"\nBin table rows: {len(bin_table)}")
    print(bin_table)
    
    # 尝试transform
    try:
        result = binner.transform(X, metric='indices')
        print(f"\nTransform result shape: {result.shape}")
    except Exception as e:
        print(f"\nTransform error: {e}")
        
        # 调试_transform_bins
        splits = binner.splits_['青云24']
        print(f"\nSplits: {splits}")
        print(f"Splits type: {type(splits)}")
        print(f"Splits length: {len(splits)}")
        
        # 手动调用_apply_bins
        bins = binner._apply_bins(X['青云24'], splits)
        print(f"\nBins from _apply_bins: {len(bins)}")
        print(f"Unique bins: {np.unique(bins)}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("问题3: OptimalKSBinning/OptimalIVBinning - 产生过多分箱")
print("=" * 80)

try:
    binner = OptimalKSBinning(max_n_bins=5)
    binner.fit(X, y)
    
    print(f"\nOptimalKSBinning splits_: {binner.splits_}")
    print(f"OptimalKSBinning n_bins_: {binner.n_bins_}")
    
    bin_table = binner.get_bin_table('青云24')
    print(f"\nBin table rows: {len(bin_table)}")
    print(f"Expected max bins: 5, Actual: {len(bin_table)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("问题4: 检查 _get_bin_labels 方法")
print("=" * 80)

# 检查base.py中的_get_bin_labels方法
from hscredit.core.binning.base import BaseBinning

class TestBinning(BaseBinning):
    def fit(self, X, y=None, **kwargs):
        return self
    def transform(self, X, metric='indices', **kwargs):
        return X

# 测试_get_bin_labels
test_binner = TestBinning()
test_binner.splits_ = {'test': np.array([100, 200, 300])}

# 测试正常情况
bins_normal = np.array([0, 1, 2, 3, -1, -2])
labels = test_binner._get_bin_labels(np.array([100, 200, 300]), bins_normal)
print(f"\nNormal bins {bins_normal} -> labels: {labels}")

# 测试空splits
bins_empty = np.array([0, 0, 0, -1])
labels_empty = test_binner._get_bin_labels(np.array([]), bins_empty)
print(f"Empty splits with bins {bins_empty} -> labels: {labels_empty}")

print("\n" + "=" * 80)
print("问题5: 检查 TreeBinning 的 _apply_bins 方法")
print("=" * 80)

try:
    binner = TreeBinning(max_depth=5, max_n_bins=5)
    binner.fit(X, y)
    
    splits = binner.splits_['青云24']
    print(f"\nSplits for '青云24': {splits}")
    print(f"Feature type: {binner.feature_types_['青云24']}")
    
    # 手动测试_apply_bins
    x_series = X['青云24']
    print(f"\nX series length: {len(x_series)}")
    
    # 检查_apply_bins的实现
    if isinstance(splits, list):
        print("Splits is a list (categorical)")
        bins = np.zeros(len(x_series), dtype=int)
        for i, cat in enumerate(splits):
            bins[x_series == cat] = i
        bins[x_series.isna()] = -1
        print(f"Bins length: {len(bins)}")
    else:
        print("Splits is an array (numerical)")
        bins = np.zeros(len(x_series), dtype=int)
        
        if binner.missing_separate:
            bins[x_series.isna()] = -1
        
        if binner.special_codes:
            for code in binner.special_codes:
                bins[x_series == code] = -2
        
        mask = x_series.notna()
        if binner.special_codes:
            for code in binner.special_codes:
                mask = mask & (x_series != code)
        
        bins[mask] = np.digitize(x_series[mask], splits)
        print(f"Bins length: {len(bins)}")
        print(f"Unique bins: {np.unique(bins)}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("问题6: 检查 OptimalBinning 的参数传递")
print("=" * 80)

# 检查optimal_binning.py中的参数传递问题
print("\n在OptimalBinning.fit()中:")
print("common_params = {")
print("    'target': self.target,")
print("    'max_n_bins': self.max_n_bins,")
print("    ...")
print("}")
print("")
print("当method='uniform'时:")
print("  self._binner = UniformBinning(**common_params)")
print("")
print("但UniformBinning的__init__是:")
print("  def __init__(self, target='target', n_bins=5, **kwargs):")
print("      super().__init__(target=target, max_n_bins=n_bins, **kwargs)")
print("")
print("所以max_n_bins被传递了两次:")
print("  1. 从common_params传递的max_n_bins")
print("  2. 从super().__init__(max_n_bins=n_bins)传递的max_n_bins")

print("\n" + "=" * 80)
print("问题7: 检查 WOE/IV 计算中的极端值")
print("=" * 80)

# 检查WOE计算中的极端值
binner = UniformBinning(max_n_bins=5)
binner.fit(X, y)
bin_table = binner.get_bin_table('青云24')

print("\nWOE值检查:")
print(f"WOE min: {bin_table['分档WOE值'].min()}")
print(f"WOE max: {bin_table['分档WOE值'].max()}")
print(f"WOE values:\n{bin_table['分档WOE值']}")

# 检查是否有极端WOE值（如-13.11）
extreme_woe = bin_table[abs(bin_table['分档WOE值']) > 10]
if len(extreme_woe) > 0:
    print(f"\n极端WOE值发现:")
    print(extreme_woe[['分箱', '样本总数', '好样本数', '坏样本数', '坏样本率', '分档WOE值']])
    print("\n原因分析: 当某个分箱的good或bad数量为0时，WOE计算会添加1e-10平滑值")
    print("但log(1e-10)仍然是一个很大的负数，导致WOE值极端")

print("\n" + "=" * 80)
print("详细分析报告完成")
print("=" * 80)
