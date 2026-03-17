#!/usr/bin/env python3
"""测试 psi_plot 函数"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
hscredit_pkg = os.path.dirname(script_dir)
sys.path.insert(0, hscredit_pkg)

from hscredit.core.viz.binning_plots import psi_plot
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# 创建正确的测试数据
expected = pd.DataFrame({
    '分箱': ['[0, 100)', '[100, 200)', '[200, 300)'],
    '样本总数': [110, 165, 220],
    '样本占比': [0.22, 0.33, 0.44],
    '坏样本率': [0.091, 0.091, 0.091]
})

actual = pd.DataFrame({
    '分箱': ['[0, 100)', '[100, 200)', '[200, 300)'],
    '样本总数': [120, 170, 210],
    '样本占比': [0.24, 0.34, 0.42],
    '坏样本率': [0.092, 0.092, 0.092]
})

print("测试 psi_plot - title 参数:")
try:
    fig = psi_plot(expected, actual, title="自定义标题 - PSI图")
    print("✓ psi_plot 的 title 参数测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败:")
    traceback.print_exc()

print("\n测试 psi_plot - desc 参数:")
try:
    fig = psi_plot(expected, actual, desc="特征A")
    print("✓ psi_plot 的 desc 参数测试通过")
except Exception as e:
    import traceback
    print(f"✗ 失败:")
    traceback.print_exc()
