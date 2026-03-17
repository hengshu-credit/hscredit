#!/usr/bin/env python3
"""分箱方法演示脚本.

快速验证所有分箱方法是否正常工作.
"""

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

import numpy as np
import pandas as pd
from hscredit.core.binning import (
    UniformBinning,
    QuantileBinning,
    TreeBinning,
    ChiMergeBinning,
    OptimalKSBinning,
    OptimalIVBinning,
    MDLPBinning,
    OptimalBinning,
)

def create_sample_data(n_samples=1000, random_state=42):
    """创建示例数据."""
    np.random.seed(random_state)

    # 创建特征
    X = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'income': np.random.lognormal(8, 0.5, n_samples),
        'score': np.random.normal(600, 100, n_samples).astype(int),
    })

    # 创建目标变量
    prob = 1 / (1 + np.exp(-(0.02 * (X['age'] - 40) - 0.1 * (np.log(X['income']) - 8))))
    y = pd.Series(np.random.binomial(1, prob))

    return X, y


def test_binner(binner_class, name, X, y):
    """测试单个分箱器."""
    try:
        binner = binner_class(max_n_bins=5)
        binner.fit(X, y)

        # 转换数据
        X_woe = binner.transform(X, metric='woe')

        # 获取第一个特征的统计
        feature = X.columns[0]
        bin_table = binner.get_bin_table(feature)
        iv = bin_table['total_iv'].iloc[0]
        n_bins = binner.n_bins_[feature]

        print(f"✅ {name:20s} - 分箱数: {n_bins}, IV: {iv:.4f}")
        return True
    except Exception as e:
        print(f"❌ {name:20s} - 错误: {e}")
        return False


def main():
    """主函数."""
    print("=" * 60)
    print("分箱方法演示")
    print("=" * 60)

    # 创建数据
    print("\n创建示例数据...")
    X, y = create_sample_data()
    print(f"数据形状: {X.shape}")
    print(f"坏样本率: {y.mean():.2%}")

    # 测试各种分箱方法
    print("\n测试分箱方法:")
    print("-" * 60)

    binners = [
        (UniformBinning, "等距分箱"),
        (QuantileBinning, "等频分箱"),
        (TreeBinning, "决策树分箱"),
        (ChiMergeBinning, "卡方分箱"),
        (OptimalKSBinning, "最优KS分箱"),
        (OptimalIVBinning, "最优IV分箱"),
        (MDLPBinning, "MDLP分箱"),
    ]

    results = []
    for binner_class, name in binners:
        success = test_binner(binner_class, name, X, y)
        results.append((name, success))

    # 测试统一接口
    print("\n测试统一接口 (OptimalBinning):")
    print("-" * 60)

    methods = ['uniform', 'quantile', 'tree', 'chi_merge', 'optimal_ks', 'optimal_iv', 'mdlp']
    for method in methods:
        try:
            binner = OptimalBinning(method=method, max_n_bins=5)
            binner.fit(X, y)
            feature = X.columns[0]
            iv = binner.bin_tables_[feature]['total_iv'].iloc[0]
            n_bins = binner.n_bins_[feature]
            print(f"✅ {method:20s} - 分箱数: {n_bins}, IV: {iv:.4f}")
            results.append((f"OptimalBinning({method})", True))
        except Exception as e:
            print(f"❌ {method:20s} - 错误: {e}")
            results.append((f"OptimalBinning({method})", False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n✅ 所有分箱方法测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")

    print("\n使用示例:")
    print("-" * 60)
    print("""
from hscredit.core.binning import OptimalBinning

# 创建分箱器
binner = OptimalBinning(method='optimal_iv', max_n_bins=5)

# 拟合数据
binner.fit(X_train, y_train)

# 转换为 WOE 值
X_train_woe = binner.transform(X_train, metric='woe')
X_test_woe = binner.transform(X_test, metric='woe')

# 查看分箱统计
bin_table = binner.get_bin_table('feature_name')
print(bin_table[['bin', 'count', 'bad_rate', 'woe']])
""")


if __name__ == '__main__':
    main()
