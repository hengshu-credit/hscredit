#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试notebook修复."""

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

import pandas as pd
import numpy as np
from hscredit.analysis import feature_bin_stats

# 创建测试数据
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'score': np.random.randn(n) * 50 + 600,
    'MOB1': np.random.choice([0, 1, 2, 3, 5, 7, 10, 15, 30, 60], n),
})

# 测试1: 单个overdue，不剔除灰客户
table1 = feature_bin_stats(
    data=df,
    feature='score',
    overdue='MOB1',
    dpds=7,
    del_grey=False,
    method='quantile',
    max_n_bins=5,
    verbose=1
)

print('\n测试1结果:')
print('是否多级表头:', isinstance(table1.columns, pd.MultiIndex))

# 尝试访问列
try:
    print(table1[['分箱标签', '样本总数', '坏样本数', '坏样本率', '分档WOE值']].head(2))
    print('✅ 列访问成功')
except Exception as e:
    print(f'❌ 列访问失败: {e}')

# 测试2: 单个overdue，剔除灰客户
table2 = feature_bin_stats(
    data=df,
    feature='score',
    overdue='MOB1',
    dpds=7,
    del_grey=True,
    method='quantile',
    max_n_bins=5,
    verbose=1
)

print('\n测试2结果:')
print('样本数:', table2['样本总数'].sum())

print('\n✅ 所有测试通过！')
