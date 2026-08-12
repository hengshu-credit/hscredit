"""分组描述统计的统一并行与混合类型回归测试。"""

import pandas as pd

from hscredit.utils.describe import groupby_feature_describe


def test_groupby_feature_describe_mixed_types_matches_serial():
    """字符串、分类和数值列的线程结果必须与串行完全一致。"""
    data = pd.DataFrame(
        {
            "分组": ["甲", "甲", "甲", "乙", "乙", "乙"],
            "数值": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "字符串": ["A", "B", "A", "B", None, "C"],
            "分类": pd.Categorical(
                ["低", "中", "低", "高", "中", None],
                categories=["低", "中", "高"],
                ordered=True,
            ),
        }
    )

    serial = groupby_feature_describe(data, by="分组", n_jobs=1)
    parallel = groupby_feature_describe(
        data,
        by="分组",
        n_jobs=2,
        parallel_backend="threading",
        parallel_config={"batch_size": 1},
    )

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert serial.index.names == ["特征名称", "统计指标"]


def test_groupby_feature_describe_multiple_group_columns_preserves_column_index():
    """多分组列并行后仍保留原来的 MultiIndex 列契约。"""
    data = pd.DataFrame(
        {
            "地区": ["东", "东", "西", "西"],
            "月份": [1, 2, 1, 2],
            "类别": ["A", "B", "A", "C"],
        }
    )

    result = groupby_feature_describe(
        data,
        by=["地区", "月份"],
        n_jobs=2,
        parallel_backend="threading",
    )

    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == [("东", 1), ("东", 2), ("西", 1), ("西", 2)]
