#!/usr/bin/env python
"""测试 MonotonicBinning 单调性约束分箱."""

from pathlib import Path

import numpy as np
import pandas as pd
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hscredit.core.binning import MonotonicBinning

# 读取测试数据
df = pd.read_excel(_PROJECT_ROOT / "examples" / "hscredit.xlsx")
df['target'] = ((df['MOB1'] > 15) | (df['MOB2'] > 15)).astype(int)
X = df[['青云24']].copy()
y = df['target'].copy()

print("=" * 60)
print("MonotonicBinning 单调性约束分箱测试")
print("=" * 60)

# 测试递增
print("\n1. 测试递增约束 (monotonic='ascending'):")
binner = MonotonicBinning(monotonic='ascending', max_n_bins=5, verbose=True)
binner.fit(X, y)

table = binner.get_bin_table('青云24')
print("\n分箱统计表:")
print(table)

bad_rates = table[table['分箱标签'] != 'missing']['坏样本率'].tolist()
print(f"\n坏样本率: {[round(r, 4) for r in bad_rates]}")
is_asc = all(bad_rates[i] <= bad_rates[i+1] + 1e-10 for i in range(len(bad_rates)-1))
print(f"是否递增: {is_asc}")
print(f"✓ 测试通过" if is_asc else "✗ 测试失败")

# 测试递减
print("\n2. 测试递减约束 (monotonic='descending'):")
binner2 = MonotonicBinning(monotonic='descending', max_n_bins=5, verbose=True)
binner2.fit(X, y)

table2 = binner2.get_bin_table('青云24')
print("\n分箱统计表:")
print(table2)

bad_rates2 = table2[table2['分箱标签'] != 'missing']['坏样本率'].tolist()
print(f"\n坏样本率: {[round(r, 4) for r in bad_rates2]}")
is_desc = all(bad_rates2[i] >= bad_rates2[i+1] - 1e-10 for i in range(len(bad_rates2)-1))
print(f"是否递减: {is_desc}")
print(f"✓ 测试通过" if is_desc else "✗ 测试失败")

# 测试自动检测
print("\n3. 测试自动检测 (monotonic='auto'):")
binner3 = MonotonicBinning(monotonic='auto', max_n_bins=5, verbose=True)
binner3.fit(X, y)

table3 = binner3.get_bin_table('青云24')
print("\n分箱统计表:")
print(table3)

bad_rates3 = table3[table3['分箱标签'] != 'missing']['坏样本率'].tolist()
print(f"\n坏样本率: {[round(r, 4) for r in bad_rates3]}")

# 检查单调性
is_asc3 = all(bad_rates3[i] <= bad_rates3[i+1] + 1e-10 for i in range(len(bad_rates3)-1))
is_desc3 = all(bad_rates3[i] >= bad_rates3[i+1] - 1e-10 for i in range(len(bad_rates3)-1))
print(f"是否递增: {is_asc3}")
print(f"是否递减: {is_desc3}")
print(f"是否单调: {is_asc3 or is_desc3}")
print(f"✓ 测试通过" if (is_asc3 or is_desc3) else "✗ 测试失败")

print("\n" + "=" * 60)
print("MonotonicBinning 测试完成!")
print("=" * 60)
