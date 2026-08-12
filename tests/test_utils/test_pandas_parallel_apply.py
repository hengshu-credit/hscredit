"""pandas ``hscredit(...).apply`` 并行扩展测试。"""

import os
import threading
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import hscredit  # noqa: F401 - 导入即注册 pandas 扩展
from hscredit.exceptions import ValidationError


def _row_total(row, offset=0):
    return row["a"] + row["b"] + offset


def _column_span(column, multiplier=1):
    return (column.max() - column.min()) * multiplier


def _expand_row(row, offset):
    return pd.Series({"合计": row["a"] + row["b"] + offset, "差值": row["b"] - row["a"]})


def _expand_value(value):
    return pd.Series({"原值": value, "平方": value * value})


def _worker_process_id(_):
    return os.getpid()


def _group_sum(group):
    return group["x"].sum()


def _group_frame(group):
    return group.assign(两倍=group["x"] * 2)


def _series_group_range(group):
    return group.max() - group.min()


def _group_name_result(group):
    return pd.Series({"组名": str(group.name), "合计": group["x"].sum()})


def test_import_registers_hscredit_proxy_on_all_apply_objects():
    """缺少任一 pandas apply 对象的注册都会破坏统一链式入口。"""
    df = pd.DataFrame({"g": [1, 1, 2], "x": [2, 3, 4]})
    objects = [df, df["x"], df.groupby("g"), df.groupby("g")["x"]]

    for obj in objects:
        proxy = obj.hscredit(n_jobs=1, bar=False)
        assert type(proxy).__name__ == "HSCreditApplyProxy"


def test_hscredit_returns_fresh_configuration_without_mutating_source():
    """复用或写回代理配置会让前一次调用被后一次配置污染。"""
    series = pd.Series([1, 2], name="x")

    first = series.hscredit(n_jobs=1, bar=False)
    second = series.hscredit(n_jobs=2, bar=True)

    assert first is not second
    assert first.n_jobs == 1
    assert first.bar is False
    assert second.n_jobs == 2
    assert second.bar is True
    pd.testing.assert_series_equal(series, pd.Series([1, 2], name="x"))


def test_dataframe_axis_one_scalar_result_preserves_duplicate_index():
    """按行并行若按标签重组，会丢失或覆盖重复索引。"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["r", "r", "s"])

    actual = df.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(
        _row_total,
        axis=1,
        offset=5,
    )

    expected = pd.Series([16, 27, 38], index=pd.Index(["r", "r", "s"]), dtype="int64")
    pd.testing.assert_series_equal(actual, expected)


def test_dataframe_axis_zero_forwards_function_kwargs():
    """并行配置与 UDF kwargs 混用时不能吞掉原生 apply 参数。"""
    df = pd.DataFrame({"a": [1, 4], "b": [10, 16]})

    actual = df.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(
        _column_span,
        axis=0,
        multiplier=3,
    )

    expected = pd.Series({"a": 9, "b": 18}, dtype="int64")
    pd.testing.assert_series_equal(actual, expected)


def test_dataframe_series_result_uses_global_pandas_expansion():
    """各分块独立推断返回类型会破坏按行返回 Series 的展开结果。"""
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]}, index=["x", "y"])

    actual = df.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(
        _expand_row,
        axis=1,
        args=(3,),
    )

    expected = pd.DataFrame({"合计": [14, 25], "差值": [9, 18]}, index=["x", "y"])
    pd.testing.assert_frame_equal(actual, expected)


def test_dataframe_raw_apply_matches_native_shape_and_labels():
    """raw 路径回退原生 pandas 时仍必须接受 hscredit 链式入口。"""
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]}, index=["x", "y"])

    actual = df.hscredit(n_jobs=2, bar=False).apply(np.sum, axis=1, raw=True)
    expected = pd.Series([11, 22], index=["x", "y"])

    pd.testing.assert_series_equal(actual, expected)


def test_series_scalar_result_preserves_duplicate_index_and_name():
    """按元素并行必须按位置恢复重复索引并保留 Series 名称。"""
    series = pd.Series([1, 2, 3], index=["x", "x", "y"], name="金额")

    actual = series.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(lambda value: value + 7)
    expected = pd.Series([8, 9, 10], index=["x", "x", "y"], name="金额")

    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2], dtype="int64"),
        pd.Series([1.5, 2.5], dtype="float64"),
        pd.Series([True, False], dtype="bool"),
        pd.Series([1, pd.NA], dtype="Int64"),
        pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"])),
    ],
)
def test_series_udf_receives_same_boxed_scalar_types_as_native(series):
    """并行 Series apply 传给 UDF 的标量类型必须与 pandas 原生装箱一致。"""
    expected = series.apply(type)

    actual = series.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(type)

    pd.testing.assert_series_equal(actual, expected)


def test_series_returning_series_expands_once_using_global_first_result():
    """Series 返回值应与 pandas 一样统一展开为 DataFrame。"""
    series = pd.Series([2, 3], index=["a", "b"], name="值")

    actual = series.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(_expand_value)
    expected = pd.DataFrame({"原值": [2, 3], "平方": [4, 9]}, index=["a", "b"])

    pd.testing.assert_frame_equal(actual, expected)


def test_series_threading_executes_each_element_exactly_once():
    """自动试跑或分块重试会让元素调用次数超过一次。"""
    calls = []

    def record(value):
        calls.append(value)
        return value * 10

    series = pd.Series([1, 2, 3, 4])
    actual = series.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(record)

    pd.testing.assert_series_equal(actual, pd.Series([10, 20, 30, 40]))
    assert sorted(calls) == [1, 2, 3, 4]


def test_plain_python_callable_automatically_uses_processes():
    """普通 Python UDF 若误选线程，会全部返回主进程 PID。"""
    series = pd.Series([1, 2, 3, 4])

    worker_pids = series.hscredit(n_jobs=2, bar=False).apply(_worker_process_id)

    assert set(worker_pids).isdisjoint({os.getpid()})


def test_dataframe_groupby_scalar_result_matches_pandas():
    """按组结果必须恢复原始排序和分组索引。"""
    df = pd.DataFrame({"g": ["b", "a", "b", "a"], "x": [2, 3, 5, 7]})
    grouped = df.groupby("g", sort=False)

    actual = grouped.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(
        _group_sum,
        include_groups=False,
    )
    expected = pd.Series([7, 10], index=pd.Index(["b", "a"], name="g"), dtype="int64")

    pd.testing.assert_series_equal(actual, expected)


def test_dataframe_groupby_dataframe_result_preserves_group_keys():
    """DataFrame 返回值若简单 concat 会丢失 group_keys MultiIndex。"""
    df = pd.DataFrame({"g": [1, 1, 2], "x": [2, 3, 5]}, index=["a", "b", "c"])
    grouped = df.groupby("g", group_keys=True)

    actual = grouped.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(
        _group_frame,
        include_groups=False,
    )
    expected = pd.DataFrame(
        {"x": [2, 3, 5], "两倍": [4, 6, 10]},
        index=pd.MultiIndex.from_tuples([(1, "a"), (1, "b"), (2, "c")], names=["g", None]),
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_series_groupby_apply_preserves_series_name():
    """SeriesGroupBy 标量结果必须保留选择列名称。"""
    series = pd.Series([2, 8, 3, 9], name="金额")
    keys = pd.Series(["a", "a", "b", "b"], name="组")

    actual = series.groupby(keys, sort=False).hscredit(n_jobs=2, bar=False).apply(_series_group_range)
    expected = pd.Series([6, 6], index=pd.Index(["a", "b"], name="组"), name="金额")

    pd.testing.assert_series_equal(actual, expected)


def test_multikey_groupby_function_receives_native_group_name():
    """未设置 group.name 会破坏依赖多键组名的 UDF。"""
    df = pd.DataFrame({"g1": ["a", "a", "b"], "g2": [1, 2, 1], "x": [3, 4, 5]})
    grouped = df.groupby(["g1", "g2"], sort=False)

    actual = grouped.hscredit(n_jobs=2, bar=False, parallel_backend="loky").apply(
        _group_name_result,
        include_groups=False,
    )
    expected = pd.DataFrame(
        {"组名": ["('a', 1)", "('a', 2)", "('b', 1)"], "合计": [3, 4, 5]},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["g1", "g2"]),
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_groupby_type_error_is_not_retried():
    """pandas 2.x 的排除分组列重试会让已经开始的组执行第二次。"""
    calls = []

    def fail(group):
        calls.append(group.name)
        raise TypeError(f"bad-group-{group.name}")

    grouped = pd.DataFrame({"g": [1, 1, 2, 2], "x": [2, 3, 4, 5]}).groupby("g")

    with pytest.raises(TypeError, match="bad-group"):
        grouped.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(
            fail,
            include_groups=False,
        )

    counts = Counter(calls)
    assert counts
    assert max(counts.values()) == 1


def test_dataframe_progress_reports_exact_completed_rows(capsys):
    """进度条若按提交而非完成更新，会在 UDF 尚未结束时提前显示 100%。"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    result = df.hscredit(n_jobs=1, bar=True).apply(_row_total, axis=1)

    pd.testing.assert_series_equal(result, pd.Series([11, 22, 33]))
    progress = capsys.readouterr().err
    assert "DataFrame 行计算" in progress
    assert "3/3" in progress


def test_bar_false_creates_no_series_progress_output(capsys):
    """bar=False 不应创建或刷新 tqdm 输出。"""
    series = pd.Series([1, 2, 3])

    result = series.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(lambda value: value + 1)

    pd.testing.assert_series_equal(result, pd.Series([2, 3, 4]))
    assert "Series 元素计算" not in capsys.readouterr().err


def test_groupby_failure_closes_progress_without_false_completion(capsys):
    """异常路径必须关闭进度条且不能把失败分组计为完成。"""

    def fail(group):
        if group.name == 1:
            raise RuntimeError("group failed")
        return group["x"].sum()

    grouped = pd.DataFrame({"g": [1, 1, 2, 2], "x": [2, 3, 4, 5]}).groupby("g", sort=False)

    with pytest.raises(RuntimeError, match="group failed"):
        grouped.hscredit(n_jobs=1, bar=True).apply(fail, include_groups=False)

    progress = capsys.readouterr().err
    assert "GroupBy 分组计算" in progress
    assert "2/2" not in progress


def test_dataframe_result_type_expand_matches_literal_frame():
    """result_type=expand 必须在全量结果上扩展，不能按 worker 分块推断。"""
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]}, index=["x", "y"])

    actual = df.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(
        lambda row: [row["a"], row["b"]],
        axis=1,
        result_type="expand",
    )
    expected = pd.DataFrame([[1, 10], [2, 20]], index=["x", "y"])

    pd.testing.assert_frame_equal(actual, expected)


def test_series_by_row_false_delegates_whole_series_once():
    """by_row=False 若被拆成元素任务，会改变函数输入类型和调用次数。"""
    calls = []

    def whole(value):
        calls.append(type(value))
        return value.sum()

    series = pd.Series([1, 2, 3], name="值")
    actual = series.hscredit(n_jobs=2, bar=False).apply(whole, by_row=False)

    assert actual == 6
    assert calls == [pd.Series]


def test_unserializable_python_callable_falls_back_to_threading_without_probe():
    """cloudpickle 失败时不能执行探针，也不能让 loky 序列化错误覆盖结果。"""
    lock = threading.Lock()
    calls = []

    def locked(value):
        with lock:
            calls.append(value)
        return os.getpid()

    result = pd.Series([1, 2, 3]).hscredit(n_jobs=2, bar=False).apply(locked)

    assert result.tolist() == [os.getpid(), os.getpid(), os.getpid()]
    assert sorted(calls) == [1, 2, 3]


def test_loky_progress_reaches_exact_series_total(capsys):
    """进程进度事件必须回到主进程，不能只在线程路径显示。"""
    result = pd.Series([1, 2, 3]).hscredit(n_jobs=2, bar=True, parallel_backend="loky").apply(lambda value: value * 2)

    pd.testing.assert_series_equal(result, pd.Series([2, 4, 6]))
    progress = capsys.readouterr().err
    assert "Series 元素计算" in progress
    assert "3/3" in progress


def test_parallel_timeout_is_forwarded_and_preserves_timeout_type():
    """parallel_config.timeout 若被代理吞掉，长任务会一直运行而不是及时退出。"""

    def slow(value):
        import time

        time.sleep(1)
        return value

    with pytest.raises(MultiprocessingTimeoutError):
        pd.Series([1, 2]).hscredit(
            n_jobs=2,
            bar=False,
            parallel_backend="loky",
            parallel_config={"timeout": 0.05},
        ).apply(slow)


@pytest.mark.parametrize(
    ("configuration", "message"),
    [({"bar": "yes"}, "bar"), ({"n_jobs": 0}, "n_jobs"), ({"parallel_backend": "missing"}, "并行后端")],
)
def test_invalid_proxy_configuration_raises_chinese_validation(configuration, message):
    """原生回退路径也必须校验 hscredit 配置，不能静默忽略拼写错误。"""
    with pytest.raises(ValidationError, match=message):
        pd.Series([1, 2]).hscredit(**configuration).apply(np.sqrt)


def test_groupby_dropna_false_and_as_index_false_match_literal_result():
    """GroupBy 装配必须保留缺失组和 as_index=False 的分组列。"""
    df = pd.DataFrame({"g": ["a", None, "a"], "x": [2, 5, 7]})
    grouped = df.groupby("g", dropna=False, as_index=False, sort=False)

    actual = grouped.hscredit(n_jobs=2, bar=False, parallel_backend="threading").apply(
        _group_sum,
        include_groups=False,
    )
    expected = pd.DataFrame({"g": ["a", np.nan], None: [9, 5]})

    pd.testing.assert_frame_equal(actual, expected)
