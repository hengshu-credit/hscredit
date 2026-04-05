#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试05_categorical_binning.ipynb的所有功能."""

import sys
import numpy as np
import pandas as pd
import pytest
from hscredit.core.binning import OptimalBinning
from hscredit.report import feature_bin_stats
import json


class TestCategoricalBinning:
    """测试类别型特征分箱功能."""

    def setup_method(self):
        """每个测试方法前的准备工作."""
        np.random.seed(42)
        n_samples = 1000
        
        self.df = pd.DataFrame({
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
        self.df.loc[self.df.sample(50).index, '城市'] = np.nan
        self.df.loc[self.df.sample(30).index, '学历'] = np.nan
        
        self.X = self.df[['城市', '学历', '年龄']]
        self.y = self.df['target']

    def test_auto_detect_categorical(self):
        """测试自动识别类别型特征."""
        binner = OptimalBinning(method='tree', max_n_bins=5)
        binner.fit(self.X, self.y)
        
        print("特征类型识别:")
        for feature, ftype in binner.feature_types_.items():
            print(f"  {feature:<15} {ftype:>12}")
        
        # 验证特征类型识别
        assert '城市' in binner.feature_types_
        assert '学历' in binner.feature_types_
        assert '年龄' in binner.feature_types_

    def test_list_format_binning(self):
        """测试List[List]格式分箱."""
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
        binner.fit(self.X[['城市', '学历']], self.y)
        
        for feature in ['城市', '学历']:
            bin_table = binner.get_bin_table(feature)
            print(f"\n【{feature}】分箱结果:")
            print(bin_table[['分箱标签', '样本总数', '坏样本率', '分档WOE值']].to_string(index=False))
            
            # 验证分箱表不为空
            assert len(bin_table) > 0

    def test_export_import_rules(self):
        """测试导出和导入规则."""
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
        binner.fit(self.X[['城市', '学历']], self.y)
        
        # 导出规则
        rules = binner.export_rules()
        print(f"导出的规则类型: {type(rules)}")
        print(f"城市规则类型: {type(rules['城市'])}")
        print(f"城市规则长度: {len(rules['城市'])}")
        
        # 导入规则
        binner_new = OptimalBinning()
        binner_new.import_rules(rules)
        binner_new.fit(self.X[['城市', '学历']], self.y)
        
        # 验证一致性
        bin_table_orig = binner.get_bin_table('城市')
        bin_table_new = binner_new.get_bin_table('城市')
        
        assert bin_table_orig['样本总数'].tolist() == bin_table_new['样本总数'].tolist()
        print("✅ 导出-导入循环一致性验证成功")

    def test_backward_compatibility_string_format(self):
        """测试向后兼容 - 字符串格式."""
        user_splits_old = {
            '城市': [
                '北京,上海',
                '广州,深圳,杭州',
                '成都,武汉,西安'
            ]
        }
        
        binner_old = OptimalBinning(user_splits=user_splits_old, missing_separate=True)
        binner_old.fit(self.X[['城市']], self.y)
        
        bin_table_old = binner_old.get_bin_table('城市')
        print("字符串格式分箱结果:")
        print(bin_table_old[['分箱标签', '样本总数', '坏样本率']].to_string(index=False))
        
        # 验证分箱表不为空
        assert len(bin_table_old) > 0

    def test_mixed_type_binning(self):
        """测试混合类型分箱."""
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
        binner_mixed.fit(self.X[['城市', '年龄']], self.y)
        
        print("混合类型分箱成功:")
        for feature in ['城市', '年龄']:
            bin_table = binner_mixed.get_bin_table(feature)
            print(f"  {feature}: {len(bin_table)} bins")
            
            # 验证分箱表不为空
            assert len(bin_table) > 0

    def test_apply_binning_transform(self):
        """测试应用分箱转换."""
        # 先进行分箱
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
        binner_mixed.fit(self.X[['城市', '年龄']], self.y)
        
        test_data = self.X.head(5)
        print("原始数据:")
        print(test_data)
        
        # 测试索引转换
        binned_indices = binner_mixed.transform(test_data[['城市', '年龄']], metric='indices')
        print("\n分箱索引:")
        print(binned_indices)
        assert binned_indices.shape == (5, 2)
        
        # 测试标签转换
        binned_labels = binner_mixed.transform(test_data[['城市', '年龄']], metric='bins')
        print("\n分箱标签:")
        print(binned_labels)
        assert binned_labels.shape == (5, 2)

    def test_json_serialization(self):
        """测试JSON序列化."""
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
        binner.fit(self.X[['城市', '学历']], self.y)
        
        rules = binner.export_rules()
        
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
        assert len(json_str) > 0
        
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


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
