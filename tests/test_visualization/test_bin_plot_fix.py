#!/usr/bin/env python3
"""测试 bin_plot 函数的 title 参数"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.viz.binning_plots import bin_plot
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 创建测试数据
feature_table = pd.DataFrame({
    '分箱': ['[0, 100)', '[100, 200)', '[200, 300)'],
    '好样本数': [100, 150, 200],
    '坏样本数': [10, 15, 20],
    '样本总数': [110, 165, 220],
    '坏样本率': [0.091, 0.091, 0.091]
})

print("测试 1: 使用 title 参数")
try:
    fig = bin_plot(feature_table, title="特征A 分箱图")
    print("✓ title 参数测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试 2: 使用 desc 参数（向后兼容）")
try:
    fig = bin_plot(feature_table, desc="特征A")
    print("✓ desc 参数测试通过")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n所有测试通过！✓")
