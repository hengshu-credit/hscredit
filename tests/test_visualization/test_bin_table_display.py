"""测试分箱表美化展示功能."""

import sys
import pandas as pd
import numpy as np
from hscredit.report import feature_bin_stats
from hscredit.utils import register_extensions, style_bin_table, BinTableDisplay


def test_basic_functionality():
    """测试基本功能."""
    print("="*80)
    print("测试分箱表美化展示功能")
    print("="*80)
    
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'score': np.random.randn(n),
        'age': np.random.randint(20, 60, n),
        'target': (np.random.randn(n) + np.random.randn(n) * 0.5 > 0).astype(int),
    })
    
    print(f"\n测试数据: {n} 个样本")
    print(f"坏样本率: {data['target'].mean():.4f}")
    
    # 测试 1: 启用 DataFrame.show() 方法
    print("\n" + "-"*80)
    print("测试 1: 启用 DataFrame.show() 方法")
    print("-"*80)
    register_extensions()
    
    # 测试 2: 生成分箱表
    print("\n" + "-"*80)
    print("测试 2: 生成分箱表")
    print("-"*80)
    table = feature_bin_stats(
        data,
        'score',
        target='target',
        method='mdlp',
        max_n_bins=5,
        desc='信用评分'
    )
    
    print(f"\n分箱表形状: {table.shape}")
    print(f"列名: {list(table.columns)}")
    
    # 测试 3: 使用 style_bin_table 函数
    print("\n" + "-"*80)
    print("测试 3: 使用 style_bin_table 函数")
    print("-"*80)
    try:
        styler = style_bin_table(table, compact=True)
        print(f"✓ style_bin_table 成功创建 Styler 对象")
        print(f"  Styler 类型: {type(styler)}")
    except Exception as e:
        print(f"✗ style_bin_table 失败: {e}")
    
    # 测试 4: 使用 BinTableDisplay 类
    print("\n" + "-"*80)
    print("测试 4: 使用 BinTableDisplay 类")
    print("-"*80)
    try:
        display_obj = BinTableDisplay(table)
        print(f"✓ BinTableDisplay 成功创建")
        print(f"  对象类型: {type(display_obj)}")
        
        # 测试导出功能
        try:
            display_obj.show(compact=True)
            print(f"✓ show() 方法可用")
        except Exception as e:
            print(f"  注: show() 在非 Jupyter 环境中可能无法正常显示 ({type(e).__name__})")
    except Exception as e:
        print(f"✗ BinTableDisplay 失败: {e}")
    
    # 测试 5: 检查 DataFrame 是否有 show 方法
    print("\n" + "-"*80)
    print("测试 5: 检查 DataFrame.show() 方法")
    print("-"*80)
    if hasattr(pd.DataFrame, 'show'):
        print("✓ DataFrame 已注入 show 方法")
        print(f"  方法类型: {type(pd.DataFrame.show)}")
    else:
        print("✗ DataFrame 未注入 show 方法")
    
    # 测试 6: 多特征分析
    print("\n" + "-"*80)
    print("测试 6: 多特征分析")
    print("-"*80)
    multi_table = feature_bin_stats(
        data,
        ['score', 'age'],
        target='target',
        method='mdlp',
        max_n_bins=5
    )
    print(f"多特征分箱表形状: {multi_table.shape}")
    
    try:
        styler = style_bin_table(multi_table, compact=True)
        print(f"✓ 多特征表格成功创建 Styler")
    except Exception as e:
        print(f"✗ 多特征表格失败: {e}")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    print("\n使用说明:")
    print("1. 在 Jupyter Notebook 中导入:")
    print("   from hscredit.utils import register_extensions")
    print("   register_extensions()")
    print("\n2. 使用方法:")
    print("   table = feature_bin_stats(data, 'feature', target='target')")
    print("   table.show()  # 美化展示")
    print("   table.show(compact=True)  # 紧凑模式")


if __name__ == '__main__':
    test_basic_functionality()
