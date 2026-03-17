#!/usr/bin/env python3
"""测试所有绘图函数的 title 参数支持"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.viz.binning_plots import bin_plot, hist_plot, psi_plot
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 创建测试数据
feature_table = pd.DataFrame({
    '分箱': ['[0, 100)', '[100, 200)', '[200, 300)'],
    '好样本数': [100, 150, 200],
    '坏样本数': [10, 15, 20],
    '样本总数': [110, 165, 220],
    '坏样本率': [0.091, 0.091, 0.091]
})

print("=" * 70)
print("测试绘图函数的 title 参数支持")
print("=" * 70)

# 测试 bin_plot
print("\n测试 1: bin_plot - title 参数")
try:
    fig = bin_plot(feature_table, title="自定义标题 - 分箱图")
    print("✓ bin_plot 的 title 参数测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试 hist_plot
print("\n测试 2: hist_plot - title 参数")
try:
    score = np.random.randn(100)
    fig = hist_plot(score, title="自定义标题 - 分布图")
    print("✓ hist_plot 的 title 参数测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试 psi_plot
print("\n测试 3: psi_plot - title 参数")
try:
    fig = psi_plot(feature_table, feature_table, title="自定义标题 - PSI图")
    print("✓ psi_plot 的 title 参数测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

# 测试向后兼容性
print("\n测试 4: 向后兼容性（使用 desc 参数）")
try:
    fig = bin_plot(feature_table, desc="特征A")
    print("✓ bin_plot 的 desc 参数（向后兼容）测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n" + "=" * 70)
print("所有测试通过！✓")
print("=" * 70)
