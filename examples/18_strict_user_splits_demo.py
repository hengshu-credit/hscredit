"""strict_user_splits 参数使用示例.

演示如何使用 strict_user_splits 参数强制保留用户指定的分箱切分点。
"""
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.utils.datasets import germancredit


def main():
    # 加载数据
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    print("=" * 60)
    print("strict_user_splits 参数使用示例")
    print("=" * 60)
    print()

    # 定义用户指定的切分点
    user_splits = {
        'age_in_years': [20, 30, 40, 50, 60, 70],  # 包括超出数据范围的点
        'credit_amount': [1000.123, 2500.456, 5000.789],  # 带小数的切分点
    }

    print("用户指定的切分点:")
    for feature, splits in user_splits.items():
        print(f"  {feature}: {splits}")
    print()

    # 获取数据实际范围
    print("数据实际范围:")
    for feature in user_splits.keys():
        x_min = X[feature].min()
        x_max = X[feature].max()
        print(f"  {feature}: [{x_min}, {x_max}]")
    print()

    # 使用默认设置 (strict_user_splits=False)
    print("-" * 60)
    print("strict_user_splits=False (默认):")
    print("-" * 60)
    binner_default = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=False,
    )
    binner_default.fit(X, y)

    for feature in user_splits.keys():
        actual_splits = list(binner_default.splits_[feature])
        print(f"  {feature}: {actual_splits}")
    print("  注意: 超出范围的切分点被过滤，切分点被四舍五入")
    print()

    # 使用严格模式 (strict_user_splits=True)
    print("-" * 60)
    print("strict_user_splits=True:")
    print("-" * 60)
    binner_strict = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=True,
    )
    binner_strict.fit(X, y)

    for feature in user_splits.keys():
        actual_splits = list(binner_strict.splits_[feature])
        print(f"  {feature}: {actual_splits}")
    print("  注意: 所有切分点完全保留，包括精度")
    print()

    # 验证严格模式完全保留了切分点
    print("=" * 60)
    print("验证结果:")
    print("=" * 60)

    all_match = True
    for feature, expected_splits in user_splits.items():
        actual_splits = binner_strict.splits_[feature]
        expected_array = np.array(expected_splits, dtype=float)
        if np.array_equal(actual_splits, expected_array):
            print(f"  [OK] {feature}: 切分点完全匹配")
        else:
            print(f"  [FAIL] {feature}: 切分点不匹配")
            print(f"    期望: {expected_splits}")
            print(f"    实际: {list(actual_splits)}")
            all_match = False

    print()
    if all_match:
        print("所有切分点完全保留!")
    else:
        print("有切分点未完全保留!")


if __name__ == '__main__':
    main()
