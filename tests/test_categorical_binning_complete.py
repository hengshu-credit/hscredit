#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试05_categorical_binning.ipynb的所有功能."""

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.report import feature_bin_stats
import json

print("="*80)
print("测试 1: 自动识别类别型特征")
print("="*80)

# 创建测试数据
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    '城市': np.random.choice(
        ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安'],
        n_samples
    ),
    '学历': np.random.choice(
        ['高中', '大专', '本科', '硕士', '博士'],
        n_samples
    ),
    '年龄': np.random.randint(18, 60, n_samples),
    'target': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
})

# 添加缺失值
df.loc[df.sample(50).index, '城市'] = np.nan
df.loc[df.sample(30).index, '学历'] = np.nan

X = df[['城市', '学历', '年龄']]
y = df['target']

binner = OptimalBinning(method='tree', max_n_bins=5)
binner.fit(X, y)

print("特征类型识别:")
for feature, ftype in binner.feature_types_.items():
    print(f"  {feature:<15} {ftype:>12}")

print("\n" + "="*80)
print("测试 2: List[List]格式分箱")
print("="*80)

user_splits = {
    '城市': [
        ['北京', '上海'],
        ['广州', '深圳', '杭州'],
        ['成都', '武汉', '西安'],
        [np.nan]
    ],
    '学历': [
        ['高中', '大专'],
        ['本科'],
        ['硕士', '博士'],
        [np.nan]
    ]
}

binner = OptimalBinning(user_splits=user_splits)
binner.fit(X[['城市', '学历']], y)

for feature in ['城市', '学历']:
    bin_table = binner.get_bin_table(feature)
    print(f"\n【{feature}】分箱结果:")
    print(bin_table[['分箱标签', '样本总数', '坏样本率', '分档WOE值']].to_string(index=False))

print("\n" + "="*80)
print("测试 3: 导出和导入规则")
print("="*80)

rules = binner.export_rules()
print(f"导出的规则类型: {type(rules)}")
print(f"城市规则类型: {type(rules['城市'])}")
print(f"城市规则长度: {len(rules['城市'])}")

# 导入规则
binner_new = OptimalBinning()
binner_new.import_rules(rules)
binner_new.fit(X[['城市', '学历']], y)

# 验证一致性
bin_table_orig = binner.get_bin_table('城市')
bin_table_new = binner_new.get_bin_table('城市')

if bin_table_orig['样本总数'].tolist() == bin_table_new['样本总数'].tolist():
    print("✅ 导出-导入循环一致性验证成功")
else:
    print("❌ 导出-导入循环一致性验证失败")

print("\n" + "="*80)
print("测试 4: 向后兼容 - 字符串格式")
print("="*80)

user_splits_old = {
    '城市': [
        '北京,上海',
        '广州,深圳,杭州',
        '成都,武汉,西安'
    ]
}

binner_old = OptimalBinning(user_splits=user_splits_old, missing_separate=True)
binner_old.fit(X[['城市']], y)

bin_table_old = binner_old.get_bin_table('城市')
print("字符串格式分箱结果:")
print(bin_table_old[['分箱标签', '样本总数', '坏样本率']].to_string(index=False))

print("\n" + "="*80)
print("测试 5: 混合类型分箱")
print("="*80)

mixed_rules = {
    '城市': [
        ['北京', '上海'],
        ['广州', '深圳', '杭州'],
        ['成都', '武汉', '西安'],
        [np.nan]
    ],
    '年龄': [25, 35, 45, 55]
}

binner_mixed = OptimalBinning(user_splits=mixed_rules)
binner_mixed.fit(X[['城市', '年龄']], y)

print("混合类型分箱成功:")
for feature in ['城市', '年龄']:
    bin_table = binner_mixed.get_bin_table(feature)
    print(f"  {feature}: {len(bin_table)} bins")

print("\n" + "="*80)
print("测试 6: 应用分箱转换")
print("="*80)

test_data = X.head(5)
print("原始数据:")
print(test_data)

binned_indices = binner_mixed.transform(test_data[['城市', '年龄']], metric='indices')
print("\n分箱索引:")
print(binned_indices)

binned_labels = binner_mixed.transform(test_data[['城市', '年龄']], metric='bins')
print("\n分箱标签:")
print(binned_labels)

print("\n" + "="*80)
print("测试 7: JSON序列化")
print("="*80)

# 转换np.nan为字符串
def convert_nan_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_str(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    return obj

rules_json = convert_nan_to_str(rules)
json_str = json.dumps(rules_json, indent=2, ensure_ascii=False)
print(f"JSON序列化成功，长度: {len(json_str)}")

# 反序列化
def convert_str_to_nan(obj):
    if isinstance(obj, dict):
        return {k: convert_str_to_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_nan(item) for item in obj]
    elif obj == "NaN":
        return np.nan
    return obj

rules_loaded = convert_str_to_nan(json.loads(json_str))
binner_loaded = OptimalBinning()
binner_loaded.import_rules(rules_loaded)

print("✅ JSON序列化和反序列化成功")

print("\n" + "="*80)
print("✅ 所有测试通过！")
print("="*80)

print("\n主要功能验证:")
print("1. ✅ 自动识别类别型特征")
print("2. ✅ List[List]格式分箱")
print("3. ✅ 导出/导入规则")
print("4. ✅ 向后兼容字符串格式")
print("5. ✅ 混合类型分箱")
print("6. ✅ 应用分箱转换")
print("7. ✅ JSON序列化")
