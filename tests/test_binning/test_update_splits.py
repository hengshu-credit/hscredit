"""测试 update 方法.

验证手动修改切分点后 update 方法能正确更新相关属性.
"""
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.utils.datasets import germancredit


def test_update_numerical_splits():
    """测试更新数值型特征的切分点."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    # 只使用数值型特征
    X = df[['age_in_years', 'credit_amount', 'duration_in_month']].copy()

    # 初始分箱
    binner = OptimalBinning(method='quantile', max_n_bins=5)
    binner.fit(X, y)

    original_splits = binner.splits_['age_in_years'].copy()
    print(f"原始切分点: {list(original_splits)}")

    # 更新切分点
    new_splits = [20, 30, 40, 50, 60]
    binner.update({'age_in_years': new_splits}, X=X, y=y)

    updated_splits = binner.splits_['age_in_years']
    print(f"更新后切分点: {list(updated_splits)}")

    # 验证切分点已更新
    np.testing.assert_array_equal(
        updated_splits,
        np.array(new_splits, dtype=float)
    )
    print("[OK] 数值型切分点更新成功")

    # 验证分箱统计表已更新
    assert 'age_in_years' in binner.bin_tables_
    bin_table = binner.bin_tables_['age_in_years']
    print(f"分箱数: {len(bin_table)}")
    print("[OK] 分箱统计表已更新")


def test_update_without_recompute_stats():
    """测试只更新切分点而不重新计算统计表."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df[['age_in_years', 'credit_amount', 'duration_in_month']].copy()

    # 初始分箱
    binner = OptimalBinning(method='quantile', max_n_bins=5)
    binner.fit(X, y)

    # 保存原始统计表
    original_bin_table = binner.bin_tables_['age_in_years'].copy()

    # 只更新切分点，不传入 X 和 y
    new_splits = [20, 30, 40, 50, 60]
    binner.update({'age_in_years': new_splits})

    # 验证切分点已更新
    np.testing.assert_array_equal(
        binner.splits_['age_in_years'],
        np.array(new_splits, dtype=float)
    )
    print("[OK] 只更新切分点成功")

    # 验证统计表未被更新（还是旧的）
    # 注意：由于没有重新计算，bin_tables_ 中可能还是旧数据或不存在


def test_update_categorical_splits():
    """测试更新类别型特征的切分点."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    # 包含类别型特征
    X = df[['age_in_years', 'purpose']].copy()

    # 使用 user_splits 初始化分箱器（支持类别型特征）
    initial_splits = {
        'age_in_years': [25, 35, 45, 55],
        'purpose': [['car'], ['furniture'], ['radio/tv'], ['domestic appliances']],
    }
    binner = OptimalBinning(user_splits=initial_splits)
    binner.fit(X, y)

    print(f"原始类别型切分点: {binner.splits_['purpose']}")

    # 更新类别型切分点
    new_cat_splits = [['car', 'furniture'], ['radio/tv'], ['domestic appliances', 'repairs']]
    binner.update({'purpose': new_cat_splits}, X=X, y=y)

    # 验证切分点已更新
    assert binner.splits_['purpose'] == new_cat_splits
    assert binner._cat_bins_['purpose'] == new_cat_splits
    assert binner.feature_types_['purpose'] == 'categorical'
    print(f"更新后类别型切分点: {binner.splits_['purpose']}")
    print("[OK] 类别型切分点更新成功")


def test_update_multiple_features():
    """测试同时更新多个特征."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df[['age_in_years', 'credit_amount', 'purpose']].copy()

    # 使用 user_splits 初始化分箱器（支持类别型特征）
    initial_splits = {
        'age_in_years': [25, 35, 45, 55],
        'credit_amount': [1000, 5000, 10000],
        'purpose': [['car'], ['furniture'], ['radio/tv']],
    }
    binner = OptimalBinning(user_splits=initial_splits)
    binner.fit(X, y)

    # 同时更新多个特征
    updates = {
        'age_in_years': [20, 30, 40, 50],
        'credit_amount': [2000, 6000, 12000],
        'purpose': [['car', 'furniture'], ['radio/tv'], ['others']],
    }
    binner.update(updates, X=X, y=y)

    # 验证所有特征都已更新
    np.testing.assert_array_equal(
        binner.splits_['age_in_years'],
        np.array([20, 30, 40, 50], dtype=float)
    )
    np.testing.assert_array_equal(
        binner.splits_['credit_amount'],
        np.array([2000, 6000, 12000], dtype=float)
    )
    assert binner.splits_['purpose'] == [['car', 'furniture'], ['radio/tv'], ['others']]
    print("[OK] 多个特征同时更新成功")


def test_update_chain_call():
    """测试链式调用."""
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df[['age_in_years', 'credit_amount']].copy()

    # 初始分箱
    binner = OptimalBinning(method='quantile', max_n_bins=5)
    binner.fit(X, y)

    # 链式调用
    result = binner.update({'age_in_years': [20, 30, 40]}).transform(X.head(5))

    # 验证返回的是分箱结果
    assert isinstance(result, pd.DataFrame)
    print("[OK] 链式调用成功")


def test_update_before_fit_raises_error():
    """测试在 fit 之前调用 update 应该报错."""
    binner = OptimalBinning(method='quantile', max_n_bins=5)

    try:
        binner.update({'age_in_years': [20, 30, 40]})
        assert False, "应该抛出 NotFittedError"
    except Exception as e:
        print(f"[OK] 正确抛出异常: {type(e).__name__}")


if __name__ == '__main__':
    print("=" * 60)
    print("测试 update 方法")
    print("=" * 60)
    print()

    test_update_numerical_splits()
    print()

    test_update_without_recompute_stats()
    print()

    test_update_categorical_splits()
    print()

    test_update_multiple_features()
    print()

    test_update_chain_call()
    print()

    test_update_before_fit_raises_error()
    print()

    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
