"""测试类别型变量的分箱规则格式.

验证List[List]格式的导入导出功能。
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


def test_categorical_rules_export():
    """测试类别型变量分箱规则的导出."""
    print("=" * 80)
    print("测试：类别型变量分箱规则导出")
    print("=" * 80)
    
    # 创建类别型数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '南京'], n_samples),
        'gender': np.random.choice(['M', 'F', np.nan], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    print(f"\n数据概览:")
    print(df.head(10))
    print(f"\n目标变量分布:")
    print(df['target'].value_counts())
    
    # 分箱 - 使用tree方法（支持类别型变量）
    binner = OptimalBinning(max_n_bins=5, method='tree')
    binner.fit(df[['city', 'gender']], df['target'])
    
    # 导出规则
    rules = binner.export_rules()
    
    print(f"\n导出的分箱规则:")
    for feature, rule in rules.items():
        feature_type = binner.feature_types_[feature]
        print(f"\n{feature} ({feature_type}):")
        print(f"  类型: {type(rule)}")
        print(f"  内容: {rule}")
        
        # 验证类别型变量的规则格式
        if feature_type == 'categorical':
            assert isinstance(rule, list), f"类别型变量规则应为list，实际为{type(rule)}"
            assert all(isinstance(r, list) for r in rule), f"类别型变量规则的元素应为list"
            print(f"  ✓ 格式正确: List[List]")


def test_categorical_rules_import():
    """测试类别型变量分箱规则的导入."""
    print("\n" + "=" * 80)
    print("测试：类别型变量分箱规则导入")
    print("=" * 80)
    
    # 手动定义分箱规则
    rules = {
        'city': [['北京', '上海'], ['广州', '深圳'], ['杭州', '南京'], [np.nan]],
        'gender': [['M'], ['F'], [np.nan]]
    }
    
    print(f"\n导入的分箱规则:")
    for feature, rule in rules.items():
        print(f"  {feature}: {rule}")
    
    # 导入规则
    binner = OptimalBinning()
    binner.import_rules(rules)
    
    print(f"\n导入成功！")
    
    # 创建测试数据
    df = pd.DataFrame({
        'city': ['北京', '上海', '广州', '深圳', '杭州', '南京', np.nan],
        'gender': ['M', 'F', 'M', 'F', 'M', np.nan, np.nan]
    })
    
    # 应用分箱
    df_binned = binner.transform(df, metric='indices')
    
    print(f"\n分箱结果:")
    print(df_binned)
    
    # 应用分箱标签
    df_labels = binner.transform(df, metric='bins')
    print(f"\n分箱标签:")
    print(df_labels)


def test_mixed_type_rules():
    """测试混合类型变量的分箱规则."""
    print("\n" + "=" * 80)
    print("测试：混合类型变量分箱规则")
    print("=" * 80)
    
    # 创建混合类型数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    print(f"\n数据概览:")
    print(df.head(10))
    
    # 分箱 - 使用tree方法（支持类别型变量）
    binner = OptimalBinning(max_n_bins=5, method='tree')
    binner.fit(df[['age', 'city']], df['target'])
    
    # 导出规则
    rules = binner.export_rules()
    
    print(f"\n导出的分箱规则:")
    for feature, rule in rules.items():
        feature_type = binner.feature_types_[feature]
        print(f"\n{feature} ({feature_type}):")
        print(f"  {rule}")
    
    # 导入规则并应用
    binner2 = OptimalBinning()
    binner2.import_rules(rules)
    
    df_binned = binner2.transform(df[['age', 'city']], metric='bins')
    print(f"\n应用分箱后的结果:")
    print(df_binned.head(10))


if __name__ == '__main__':
    try:
        test_categorical_rules_export()
        test_categorical_rules_import()
        test_mixed_type_rules()
        
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
