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
import pytest
from hscredit.core.binning import OptimalBinning


def test_categorical_rules_export():
    """测试类别型变量分箱规则的导出."""
    print("=" * 80)
    print("测试：类别型变量分箱规则导出")
    print("=" * 80)

    # 创建类别型数据
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame(
        {
            "city": np.random.choice(["北京", "上海", "广州", "深圳", "杭州", "南京"], n_samples),
            "gender": np.random.choice(["M", "F", np.nan], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )

    print(f"\n数据概览:")
    print(df.head(10))
    print(f"\n目标变量分布:")
    print(df["target"].value_counts())

    # 分箱 - 使用tree方法（支持类别型变量）
    binner = OptimalBinning(max_n_bins=5, method="tree")
    binner.fit(df[["city", "gender"]], df["target"])

    # 导出规则
    rules = binner.export_rules()

    print(f"\n导出的分箱规则:")
    for feature, rule in rules.items():
        feature_type = binner.feature_types_[feature]
        print(f"\n{feature} ({feature_type}):")
        print(f"  类型: {type(rule)}")
        print(f"  内容: {rule}")

        # 验证类别型变量的规则格式
        if feature_type == "categorical":
            assert isinstance(rule, list), f"类别型变量规则应为list，实际为{type(rule)}"
            assert all(isinstance(r, list) for r in rule), f"类别型变量规则的元素应为list"
            print(f"  ✓ 格式正确: List[List]")


def test_categorical_rules_import():
    """测试类别型变量分箱规则的导入."""
    print("\n" + "=" * 80)
    print("测试：类别型变量分箱规则导入")
    print("=" * 80)

    # 手动定义分箱规则
    rules = {"city": [["北京", "上海"], ["广州", "深圳"], ["杭州", "南京"], [np.nan]], "gender": [["M"], ["F"], [np.nan]]}

    print(f"\n导入的分箱规则:")
    for feature, rule in rules.items():
        print(f"  {feature}: {rule}")

    # 导入规则
    binner = OptimalBinning()
    binner.import_rules(rules)

    print(f"\n导入成功！")

    # 创建测试数据
    df = pd.DataFrame(
        {"city": ["北京", "上海", "广州", "深圳", "杭州", "南京", np.nan], "gender": ["M", "F", "M", "F", "M", np.nan, np.nan]}
    )

    # 应用分箱
    df_binned = binner.transform(df, metric="indices")

    print(f"\n分箱结果:")
    print(df_binned)

    # 应用分箱标签
    df_labels = binner.transform(df, metric="bins")
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

    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, n_samples),
            "city": np.random.choice(["北京", "上海", "广州", "深圳"], n_samples),
            "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
    )

    print(f"\n数据概览:")
    print(df.head(10))

    # 分箱 - 使用tree方法（支持类别型变量）
    binner = OptimalBinning(max_n_bins=5, method="tree")
    binner.fit(df[["age", "city"]], df["target"])

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

    df_binned = binner2.transform(df[["age", "city"]], metric="bins")
    print(f"\n应用分箱后的结果:")
    print(df_binned.head(10))


def _make_explicit_rule_data():
    X = pd.DataFrame({"category": ["c1", "c2", "c3", "c4", np.nan] * 4})
    y = pd.Series([0, 0, 1, 1, 1, 0, 1, 0, 1, 0] * 2, name="target")
    return X, y


@pytest.mark.parametrize(
    ("groups", "expected_missing_bin", "expected_c4_bin"),
    [
        ([["c1", "c2"], ["c3", "c4"], [np.nan]], 2, 1),
        ([["c1", "c2"], ["c3"], ["c4", np.nan]], 2, 2),
    ],
)
def test_strict_custom_groups_support_missing_alone_or_merged(groups, expected_missing_bin, expected_c4_bin):
    """防止显式缺失组在拟合、转换或导出时被移出用户指定箱。"""
    X, y = _make_explicit_rule_data()
    binner = OptimalBinning(user_splits={"category": groups}, strict_user_splits=True).fit(X, y)

    transformed = binner.transform(pd.DataFrame({"category": [np.nan, "c4"]}), metric="indices")["category"]
    exported = binner.export_rules()["category"]

    assert transformed.tolist() == [expected_missing_bin, expected_c4_bin]
    assert len(exported) == len(groups)
    assert any(pd.isna(value) for value in exported[expected_missing_bin])


@pytest.mark.parametrize(
    "groups",
    [
        [["c1"], [], ["c2", "c3", "c4", np.nan]],
        [["c1", "c2"], ["c2", "c3"], ["c4", np.nan]],
        [["c1", "c2"], ["c3", np.nan]],
        [["c1", "c2", np.nan], ["c3", "c4", None]],
    ],
    ids=["empty-group", "duplicate-category", "uncovered-category", "duplicate-missing"],
)
def test_invalid_strict_custom_groups_raise_value_error(groups):
    """防止非法自定义类别规则静默生成未知训练箱或相互覆盖。"""
    X, y = _make_explicit_rule_data()

    with pytest.raises(ValueError, match="category"):
        OptimalBinning(user_splits={"category": groups}, strict_user_splits=True).fit(X, y)


def test_custom_groups_require_explicit_missing_when_not_separate():
    """防止 missing_separate=False 时缺失训练值静默成为未知类别。"""
    X, y = _make_explicit_rule_data()
    groups = [["c1", "c2"], ["c3", "c4"]]

    with pytest.raises(ValueError, match="category.*缺失"):
        OptimalBinning(
            user_splits={"category": groups},
            strict_user_splits=True,
            missing_separate=False,
        ).fit(X, y)


def test_user_splits_take_priority_over_category_order():
    """防止显式分箱因无关的 category_order 校验而无法拟合。"""
    X, y = _make_explicit_rule_data()
    groups = [["c1", "c2"], ["c3"], ["c4", np.nan]]

    binner = OptimalBinning(
        user_splits={"category": groups},
        strict_user_splits=True,
        category_order={"category": ["not-used"]},
    ).fit(X, y)

    assert binner.n_bins_["category"] == 3


def test_non_strict_custom_groups_are_merged_as_atomic_units_by_method():
    """防止非严格模式忽略 max_n_bins，或拆散用户已经定义的类别组。"""
    X = pd.DataFrame({"category": ["a", "b", "c", "d", "e", "f", "g", "h"] * 10})
    y = pd.Series(([0, 0, 0, 1, 0, 1, 1, 1] * 10), name="target")
    groups = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]

    binner = OptimalBinning(
        method="uniform",
        min_n_bins=2,
        max_n_bins=2,
        user_splits={"category": groups},
        strict_user_splits=False,
    ).fit(X, y)

    assert binner.export_rules()["category"] == [["a", "b", "c", "d"], ["e", "f", "g", "h"]]


def test_optimal_binning_forwards_reverse_category_order_to_native_method():
    """防止 wrapper 只保存用户排序但底层方法仍按默认坏样本率顺序拟合。"""
    X = pd.DataFrame({"category": ["A", "B", "C", "D"] * 20})
    y = pd.Series(([0, 0, 1, 1] * 20), name="target")
    order = ["D", "C", "B", "A"]

    binner = OptimalBinning(
        method="uniform",
        min_n_bins=2,
        max_n_bins=2,
        category_order={"category": order},
    ).fit(X, y)

    assert binner.export_rules()["category"] == [["D", "C"], ["B", "A"]]


def test_export_import_preserves_mixed_missing_group_and_neutral_woe():
    """无训练统计的导入规则也应保留缺失位置并支持三种转换指标。"""
    rules = {"工资": [["1000-3000"], ["3000-5000", np.nan]]}
    loaded = OptimalBinning(missing_separate=False)

    loaded.import_rules(rules)

    assert loaded.export_rules()["工资"] == rules["工资"]
    values = pd.DataFrame({"工资": ["1000-3000", "3000-5000", np.nan, "未知"]})
    assert loaded.transform(values, metric="indices")["工资"].tolist() == [0, 1, 1, -3]
    assert loaded.transform(values, metric="bins")["工资"].tolist() == [
        "1000-3000",
        "3000-5000, nan",
        "3000-5000, nan",
        "unknown",
    ]
    assert loaded.transform(values, metric="woe")["工资"].tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "rules",
    [
        {"工资": [["低"], []]},
        {"工资": [["低"], ["低", "高"]]},
        {"工资": [["低", np.nan], [None]]},
    ],
    ids=["empty-group", "duplicate-category", "duplicate-missing"],
)
def test_import_rules_rejects_invalid_category_structure(rules):
    """导入没有训练数据时仍必须完成可独立判断的结构校验。"""
    with pytest.raises(ValueError, match="工资"):
        OptimalBinning().import_rules(rules)


def test_update_validates_category_coverage_when_training_data_is_available():
    """update 传入训练数据时不能接受漏掉训练类别的规则。"""
    X, y = _make_explicit_rule_data()
    binner = OptimalBinning(
        user_splits={"category": [["c1", "c2"], ["c3"], ["c4", np.nan]]},
        strict_user_splits=True,
    ).fit(X, y)

    with pytest.raises(ValueError, match="category.*未覆盖"):
        binner.update({"category": [["c1", "c2"], ["c3", np.nan]]}, X=X, y=y)


def test_prebinning_preserves_complete_category_state():
    """防止预分箱只复制 splits_，导致类别转换状态缺失。"""
    X = pd.DataFrame({"category": ["A", "B", "C", "D"] * 20})
    y = pd.Series(([0, 0, 1, 1] * 20), name="target")
    binner = OptimalBinning(
        method="best_iv",
        prebinning="uniform",
        min_n_bins=2,
        max_n_bins=2,
        category_order={"category": ["A", "B", "C", "D"]},
    ).fit(X, y)

    assert binner._cat_bins_["category"] == binner.splits_["category"]
    assert binner._category_orders_["category"] == ["A", "B", "C", "D"]
    assert "category" in binner._category_code_maps_
    assert "category" in binner._categorical_numeric_splits_
    assert binner.transform(X, metric="indices")["category"].ge(0).all()


if __name__ == "__main__":
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
