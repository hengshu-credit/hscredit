#!/usr/bin/env python3
"""测试 bin_plot 的两种使用方式"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.viz.binning_plots import bin_plot
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("测试 bin_plot 的两种使用方式")
print("=" * 70)

# 创建测试数据
np.random.seed(42)
df = pd.DataFrame({
    'score': np.random.randn(200) * 100 + 600,
    'age': np.random.randint(20, 60, 200),
    'target': np.random.randint(0, 2, 200)
})

# 方式1：传入原始数据（toad 模式）
print("\n测试 1: toad 模式 - DataFrame + 列名")
try:
    fig = bin_plot(df, target='target', feature='score')
    print("✓ toad 模式测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 方式2：传入 Series + 目标数组
print("\n测试 2: toad 模式 - Series + 目标数组")
try:
    fig = bin_plot(df['score'], target=df['target'])
    print("✓ Series 模式测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 方式3：使用 desc 参数
print("\n测试 3: toad 模式 - 使用 desc 参数")
try:
    fig = bin_plot(
        df['score'], 
        target=df['target'],
        desc='信用评分'
    )
    print("✓ desc 参数测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 方式4：使用 title 参数
print("\n测试 4: toad 模式 - 使用 title 参数")
try:
    fig = bin_plot(
        df['score'], 
        target=df['target'],
        title='信用评分分箱图'
    )
    print("✓ title 参数测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 方式5：用户原来的调用方式
print("\n测试 5: 用户的调用方式")
try:
    X_train_selected = df[['score']]
    y_train = df['target']
    feature_to_check = 'score'
    
    fig = bin_plot(
        X_train_selected[feature_to_check],
        y_train,
        desc=f'{feature_to_check} 分箱图',
        show_data_points=False
    )
    print("✓ 用户调用方式测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

# 方式6：传入分箱统计表
print("\n测试 6: 统计表模式")
try:
    feature_table = pd.DataFrame({
        '分箱': ['[0, 100)', '[100, 200)', '[200, 300)'],
        '好样本数': [100, 150, 200],
        '坏样本数': [10, 15, 20],
        '样本总数': [110, 165, 220],
        '坏样本率': [0.091, 0.091, 0.091]
    })
    fig = bin_plot(feature_table, desc="特征A")
    print("✓ 统计表模式测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("所有测试通过！✓")
print("=" * 70)
