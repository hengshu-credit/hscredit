"""简化测试：类别型变量分箱规则的导入导出.

先测试import_rules功能，确保List[List]格式可以正常工作。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent / "hscredit"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning


def test_import_categorical_rules():
    """测试导入类别型变量分箱规则."""
    print("=" * 80)
    print("测试1: 导入类别型变量分箱规则（List[List]格式）")
    print("=" * 80)
    
    # 手动定义分箱规则
    rules = {
        'city': [['北京', '上海'], ['广州', '深圳'], ['杭州', '南京'], [np.nan]],
        'age': [25, 35, 45, 55]  # 数值型变量
    }
    
    print(f"\n导入的分箱规则:")
    for feature, rule in rules.items():
        print(f"  {feature}: {rule}")
    
    # 导入规则
    binner = OptimalBinning()
    binner.import_rules(rules)
    
    print(f"\n导入成功！")
    print(f"特征类型: {binner.feature_types_}")
    print(f"分箱数: {binner.n_bins_}")
    
    # 创建测试数据
    df = pd.DataFrame({
        'city': ['北京', '上海', '广州', '深圳', '杭州', '南京', np.nan, '北京'],
        'age': [20, 30, 40, 50, 60, 70, 45, 25]
    })
    
    print(f"\n测试数据:")
    print(df)
    
    # 应用分箱
    df_binned = binner.transform(df, metric='indices')
    print(f"\n分箱索引:")
    print(df_binned)
    
    # 应用分箱标签
    df_labels = binner.transform(df, metric='bins')
    print(f"\n分箱标签:")
    print(df_labels)


def test_export_import_cycle():
    """测试导出-导入循环."""
    print("\n" + "=" * 80)
    print("测试2: 导出-导入循环测试")
    print("=" * 80)
    
    # 创建规则
    rules_original = {
        'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]],
        'income': [5000, 10000, 20000]
    }
    
    print(f"\n原始规则:")
    for feature, rule in rules_original.items():
        print(f"  {feature}: {rule}")
    
    # 导入
    binner1 = OptimalBinning()
    binner1.import_rules(rules_original)
    
    # 导出
    rules_exported = binner1.export_rules()
    
    print(f"\n导出的规则:")
    for feature, rule in rules_exported.items():
        print(f"  {feature}: {rule}")
    
    # 再次导入
    binner2 = OptimalBinning()
    binner2.import_rules(rules_exported)
    
    print(f"\n再次导入成功！")
    
    # 测试应用
    df = pd.DataFrame({
        'city': ['北京', '上海', '广州', '深圳', np.nan],
        'income': [3000, 8000, 15000, 25000, 5000]
    })
    
    df_binned1 = binner1.transform(df, metric='bins')
    df_binned2 = binner2.transform(df, metric='bins')
    
    print(f"\n第一次分箱结果:")
    print(df_binned1)
    print(f"\n第二次分箱结果:")
    print(df_binned2)
    
    # 验证结果一致
    assert df_binned1.equals(df_binned2), "导出-导入循环后结果不一致"
    print(f"\n✓ 导出-导入循环测试通过！")


def test_json_serialization():
    """测试JSON序列化."""
    print("\n" + "=" * 80)
    print("测试3: JSON序列化测试")
    print("=" * 80)
    
    import json
    import numpy as np
    
    # 创建规则
    rules = {
        'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]],
        'age': [25, 35, 45]
    }
    
    # 处理np.nan以便JSON序列化
    def convert_nan_to_str(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_str(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return "NaN"
        return obj
    
    def convert_str_to_nan(obj):
        if isinstance(obj, dict):
            return {k: convert_str_to_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_str_to_nan(item) for item in obj]
        elif obj == "NaN":
            return np.nan
        return obj
    
    # 序列化
    rules_json = convert_nan_to_str(rules)
    json_str = json.dumps(rules_json, indent=2, ensure_ascii=False)
    
    print(f"\nJSON字符串:")
    print(json_str)
    
    # 反序列化
    rules_loaded = json.loads(json_str)
    rules_final = convert_str_to_nan(rules_loaded)
    
    print(f"\n反序列化后的规则:")
    for feature, rule in rules_final.items():
        print(f"  {feature}: {rule}")
    
    # 验证
    binner = OptimalBinning()
    binner.import_rules(rules_final)
    
    print(f"\n✓ JSON序列化测试通过！")


if __name__ == '__main__':
    try:
        test_import_categorical_rules()
        test_export_import_cycle()
        test_json_serialization()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise
