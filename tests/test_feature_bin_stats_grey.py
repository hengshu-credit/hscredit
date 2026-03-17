#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试feature_bin_stats的灰客户剔除功能
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent / "hscredit"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from hscredit.analysis import feature_bin_stats

print("="*80)
print("测试feature_bin_stats的灰客户剔除功能")
print("="*80)

# 创建测试数据
np.random.seed(42)
n_samples = 1000

# 创建逾期天数数据
# 假设使用MOB1表示逾期天数
# 0: 好样本（未逾期）
# 1-7: 轻微逾期（灰客户）
# 8+: 严重逾期（坏样本）
mob1 = np.zeros(n_samples)
mob1[:200] = 0  # 好样本
mob1[200:400] = np.random.randint(1, 8, 200)  # 灰客户 (1-7天)
mob1[400:600] = np.random.randint(8, 30, 200)  # 坏样本 (8+天)
mob1[600:800] = 0  # 更多好样本
mob1[800:900] = np.random.randint(1, 8, 100)  # 更多灰客户
mob1[900:1000] = np.random.randint(8, 60, 100)  # 更多坏样本

# 创建特征
score = np.random.randint(300, 850, n_samples)

df = pd.DataFrame({
    'score': score,
    'MOB1': mob1.astype(int)
})

print("\n数据概况:")
print(f"总样本数: {len(df)}")
print(f"好样本数 (MOB1 == 0): {(df['MOB1'] == 0).sum()}")
print(f"灰客户数 (0 < MOB1 <= 7): {((df['MOB1'] > 0) & (df['MOB1'] <= 7)).sum()}")
print(f"坏样本数 (MOB1 > 7): {(df['MOB1'] > 7).sum()}")

# 测试1: 不剔除灰客户
print("\n" + "="*80)
print("测试1: 不剔除灰客户 (del_grey=False)")
print("="*80)

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

print("\n分箱结果:")
print(table1[['分箱标签', '样本总数', '坏样本数', '坏样本率', '分档WOE值']].to_string(index=False))

# 测试2: 剔除灰客户
print("\n" + "="*80)
print("测试2: 剔除灰客户 (del_grey=True)")
print("="*80)

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

print("\n分箱结果:")
print(table2[['分箱标签', '样本总数', '坏样本数', '坏样本率', '分档WOE值']].to_string(index=False))

# 测试3: 多目标分析（不剔除灰客户）
print("\n" + "="*80)
print("测试3: 多目标分析（不剔除灰客户）")
print("="*80)

table3 = feature_bin_stats(
    data=df,
    feature='score',
    overdue=['MOB1', 'MOB1'],
    dpds=[7, 15],
    del_grey=False,
    method='quantile',
    max_n_bins=5,
    verbose=1
)

print("\n分箱结果（多级表头）:")
print(table3.to_string(index=False))

# 测试4: 多目标分析（剔除灰客户）
print("\n" + "="*80)
print("测试4: 多目标分析（剔除灰客户）")
print("="*80)

table4 = feature_bin_stats(
    data=df,
    feature='score',
    overdue=['MOB1', 'MOB1'],
    dpds=[7, 15],
    del_grey=True,
    method='quantile',
    max_n_bins=5,
    verbose=1
)

print("\n分箱结果（多级表头，注意样本数不同）:")
print(table4.to_string(index=False))

# 验证
print("\n" + "="*80)
print("验证结果")
print("="*80)

total_samples_no_grey = len(df) - ((df['MOB1'] > 0) & (df['MOB1'] <= 7)).sum()
total_samples_with_grey = len(df)

print(f"不剔除灰客户 - 样本总数: {total_samples_with_grey}")
print(f"剔除灰客户 - 样本总数: {total_samples_no_grey}")
print(f"剔除的灰客户数: {total_samples_with_grey - total_samples_no_grey}")

# 检查table1和table2的样本数
if '样本总数' in table1.columns:
    sum1 = table1['样本总数'].sum()
    print(f"\nTable1 (不剔除) 样本总数: {sum1}")
    assert sum1 == total_samples_with_grey, f"样本数不一致: {sum1} != {total_samples_with_grey}"

if '样本总数' in table2.columns:
    sum2 = table2['样本总数'].sum()
    print(f"Table2 (剔除) 样本总数: {sum2}")
    assert sum2 == total_samples_no_grey, f"样本数不一致: {sum2} != {total_samples_no_grey}"

print("\n✅ 所有测试通过！")
print("="*80)
print("\n功能说明:")
print("1. del_grey=False: 保留所有样本，包括逾期天数在(0, dpd]的灰客户")
print("2. del_grey=True: 剔除灰客户，只保留好样本(overdue==0)和坏样本(overdue>dpd)")
print("3. 多目标分析时，不同目标下样本数可能不同（当del_grey=True时）")
print("4. 参考scp实现，merge_columns根据del_grey动态调整")
