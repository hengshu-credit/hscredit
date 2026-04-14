"""测试 strict_user_splits 功能.

验证当传入 strict_user_splits=True 时，分箱器会完全保留用户指定的切分点。
"""
import numpy as np
import pandas as pd

from hscredit.core.binning import OptimalBinning
from hscredit.utils.datasets import germancredit


def test_strict_user_splits_preserves_exact_splits():
    """测试 strict_user_splits=True 时完全保留用户指定的切分点."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    # 定义精确的切分点
    user_splits = {
        'age_in_years': [25, 35, 45, 55, 65],  # 精确到整数
        'credit_amount': [1000, 2500, 5000, 7500, 10000],  # 精确金额
    }

    # 使用 strict_user_splits=True
    # 当指定了 user_splits 时，直接使用用户指定的切分点
    binner_strict = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=True,
    )
    binner_strict.fit(X, y)

    # 验证切分点完全保留
    for feature, expected_splits in user_splits.items():
        actual_splits = binner_strict.splits_[feature]
        np.testing.assert_array_equal(
            actual_splits,
            np.array(expected_splits, dtype=float),
            err_msg=f"特征 {feature} 的切分点未完全保留"
        )
        print(f"[OK] 特征 {feature}: 切分点完全匹配 {list(actual_splits)}")


def test_strict_user_splits_with_out_of_range():
    """测试 strict_user_splits=True 时，超出数据范围的切分点也被保留."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    # 获取数据实际范围
    age_min = X['age_in_years'].min()
    age_max = X['age_in_years'].max()
    print(f"age_in_years 实际范围: [{age_min}, {age_max}]")

    # 定义包含超出范围的切分点
    user_splits = {
        'age_in_years': [10, 20, 30, 100],  # 10 和 100 超出实际范围
    }

    # strict_user_splits=False (默认): 超出范围的切分点会被过滤
    binner_default = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=False,
    )
    binner_default.fit(X, y)
    print(f"strict_user_splits=False: 切分点 = {list(binner_default.splits_['age_in_years'])}")

    # strict_user_splits=True: 切分点完全保留
    binner_strict = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=True,
    )
    binner_strict.fit(X, y)
    print(f"strict_user_splits=True: 切分点 = {list(binner_strict.splits_['age_in_years'])}")

    # 验证 strict 模式保留了所有切分点
    np.testing.assert_array_equal(
        binner_strict.splits_['age_in_years'],
        np.array([10, 20, 30, 100], dtype=float)
    )

    # 验证默认模式过滤了超出范围的切分点
    assert len(binner_default.splits_['age_in_years']) < len(user_splits['age_in_years']), \
        "默认模式应该过滤超出范围的切分点"

    print("[OK] strict_user_splits=True 正确保留了超出数据范围的切分点")


def test_strict_user_splits_no_rounding():
    """测试 strict_user_splits=True 时，不对切分点进行四舍五入."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    # 定义带小数的切分点
    user_splits = {
        'duration_in_month': [6.12345, 12.98765, 24.11111, 36.99999],
    }

    # strict_user_splits=True: 切分点完全保留
    binner_strict = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=True,
    )
    binner_strict.fit(X, y)

    # 验证切分点没有被四舍五入
    expected = np.array([6.12345, 12.98765, 24.11111, 36.99999], dtype=float)
    np.testing.assert_array_almost_equal(
        binner_strict.splits_['duration_in_month'],
        expected,
        decimal=5,
        err_msg="切分点被意外四舍五入"
    )
    print(f"[OK] strict_user_splits=True 保留了原始精度: {list(binner_strict.splits_['duration_in_month'])}")

    # strict_user_splits=False: 切分点会被四舍五入
    binner_default = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=False,
        decimal=2,  # 设置两位小数
    )
    binner_default.fit(X, y)
    print(f"strict_user_splits=False (decimal=2): 切分点 = {list(binner_default.splits_['duration_in_month'])}")


def test_strict_user_splits_categorical():
    """测试 strict_user_splits=True 时对类别型特征的支持."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    # 定义类别型分箱
    user_splits = {
        'purpose': [['car', 'furniture'], ['radio/tv'], ['domestic appliances'], ['repairs']],
    }

    # strict_user_splits=True
    binner_strict = OptimalBinning(
        user_splits=user_splits,
        strict_user_splits=True,
    )
    binner_strict.fit(X, y)

    # 验证类别分箱完全保留
    assert binner_strict.splits_['purpose'] == user_splits['purpose'], \
        "类别型分箱未完全保留"
    print(f"[OK] 类别型分箱完全保留: {binner_strict.splits_['purpose']}")


if __name__ == '__main__':
    print("=" * 60)
    print("测试 strict_user_splits 功能")
    print("=" * 60)

    test_strict_user_splits_preserves_exact_splits()
    print()

    test_strict_user_splits_with_out_of_range()
    print()

    test_strict_user_splits_no_rounding()
    print()

    test_strict_user_splits_categorical()
    print()

    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
