"""update 方法使用示例.

演示如何在分箱器训练完成后手工修改切分点.
"""
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.utils.datasets import germancredit


def main():
    # 加载数据
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df[['age_in_years', 'credit_amount', 'duration_in_month', 'purpose']].copy()

    print("=" * 60)
    print("update 方法使用示例")
    print("=" * 60)
    print()

    # 使用 user_splits 初始化分箱器
    print("1. 初始分箱（使用 user_splits）")
    print("-" * 60)
    user_splits = {
        'age_in_years': [25, 35, 45, 55],
        'credit_amount': [1000, 3000, 6000, 10000],
        'duration_in_month': [6, 12, 24, 36],
        'purpose': [['car'], ['furniture'], ['radio/tv'], ['domestic appliances']],
    }
    binner = OptimalBinning(user_splits=user_splits)
    binner.fit(X, y)

    print("初始切分点:")
    for feature in X.columns:
        print(f"  {feature}: {binner.splits_[feature]}")
    print()

    # 只更新切分点（不重新计算统计表）
    print("2. 只更新切分点（不重新计算统计表）")
    print("-" * 60)
    binner.update({
        'age_in_years': [20, 30, 40, 50, 60],  # 增加一个切分点
    })
    print("更新后的 age_in_years 切分点:", binner.splits_['age_in_years'])
    print()

    # 更新切分点并重新计算统计表
    print("3. 更新切分点并重新计算统计表")
    print("-" * 60)
    binner.update({
        'credit_amount': [2000, 5000, 8000],  # 修改切分点
        'purpose': [['car', 'furniture'], ['radio/tv', 'domestic appliances']],  # 合并类别
    }, X=X, y=y)

    print("更新后的切分点:")
    for feature in ['credit_amount', 'purpose']:
        print(f"  {feature}: {binner.splits_[feature]}")
    print()

    # 查看更新后的分箱统计表
    print("4. 查看更新后的分箱统计表")
    print("-" * 60)
    bin_table = binner.get_bin_table('credit_amount')
    print(bin_table[['分箱', '分箱标签', '样本总数', '坏样本率']])
    print()

    # 链式调用
    print("5. 链式调用")
    print("-" * 60)
    X_binned = binner.update({
        'duration_in_month': [12, 24, 36]
    }).transform(X.head(5))
    print("分箱后的数据:")
    print(X_binned)
    print()

    # 数值型转类别型示例
    print("6. 将数值型特征改为类别型分箱")
    print("-" * 60)
    # 注意：这会将特征从数值型改为类别型
    binner.update({
        'age_in_years': [['20-30', '30-40'], ['40-50'], ['50-60']],  # 改为类别型分箱
    })
    print("更新后的 age_in_years 类型:", binner.feature_types_['age_in_years'])
    print("更新后的 age_in_years 分箱:", binner.splits_['age_in_years'])
    print()

    print("=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
