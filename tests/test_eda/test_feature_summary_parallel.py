"""feature_summary 并行计算与 pandas 扩展回归测试。"""

import inspect
import threading

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import hscredit  # noqa: F401  # 导入即注册 pandas 扩展
from hscredit.core.eda import _feature_summary as feature_summary_impl
from hscredit.core.eda import feature_summary


def _row(result: pd.DataFrame, feature):
    """按特征名取得一行摘要。"""
    return result.set_index("特征名").loc[feature]


def test_feature_summary_returns_unrounded_ratio_values():
    """比例字段应保留 0~1 原始小数，供 Excel 层设置百分比格式。"""
    df = pd.DataFrame(
        {
            "num": [0.0, -1.0, np.nan, 2.0],
            "cat": ["a", "a", None, "b"],
        }
    )

    result = feature_summary(df, n_jobs=1)

    num = _row(result, "num")
    cat = _row(result, "cat")
    assert num["缺失率"] == 1 / 4
    assert num["众数占比"] == 1 / 3
    assert num["零值率"] == 1 / 3
    assert num["负值率"] == 1 / 3
    assert num["重复率"] == 0.0
    assert cat["众数占比"] == 2 / 3
    assert cat["重复率"] == 1 / 3


def test_categorical_unused_categories_do_not_count_as_observed_values():
    """Categorical 中未使用的类别不能污染唯一值数和重复率。"""
    df = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["a", "a", "b"],
                categories=["a", "b", "unused"],
            )
        }
    )

    row = _row(feature_summary(df, n_jobs=1), "category")

    assert row["唯一值数"] == 2
    assert row["重复数"] == 1
    assert row["重复率"] == 1 / 3


@pytest.mark.parametrize("target_factory", [np.asarray, list, tuple, pd.Series])
def test_feature_summary_accepts_positional_array_like_target(target_factory):
    """外部 y 按位置匹配，即使 DataFrame 或 Series 使用自定义索引。"""
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=[10, 20, 30, 40])
    y = target_factory([0, 0, 1, 1])

    result = feature_summary(df, y=y, n_jobs=1)

    assert {"IV", "KS", "趋势"}.issubset(result.columns)


def test_feature_summary_rejects_target_with_wrong_length():
    """目标长度错误应在进入统计任务前给出明确中文异常。"""
    with pytest.raises(ValueError, match="目标变量长度与数据不匹配"):
        feature_summary(pd.DataFrame({"x": [1, 2, 3]}), y=[0, 1], n_jobs=1)


def test_dataframe_summary_forwards_type_and_parallel_configuration():
    """DataFrame 扩展应透传类型覆盖及并行配置。"""
    df = pd.DataFrame({"code": [1, 1, 2, 2]})

    result = df.summary(
        numeric_as_categorical=["code"],
        n_jobs=1,
        show_progress=False,
    )

    assert _row(result, "code")["字段类型"] == "categorical"


def test_dataframe_summary_forwards_binning_configuration():
    """DataFrame.summary 应完整透传统一分箱配置。"""
    x = np.concatenate([np.linspace(0, 1, 80), np.linspace(2, 10, 20)])
    y = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 10 + [1] * 10)
    df = pd.DataFrame({"x": x})

    expected = feature_summary(
        df,
        y=y,
        binning_method="uniform",
        binning_params={"max_n_bins": 3, "min_bin_size": 0.05},
        n_jobs=1,
    )
    actual = df.summary(
        y=y,
        binning_method="uniform",
        binning_params={"max_n_bins": 3, "min_bin_size": 0.05},
        n_jobs=1,
    )

    pdt.assert_frame_equal(actual, expected)


def test_dataframe_summary_preserves_positional_return_type_compatibility():
    """旧版最后一个位置参数 return_type 仍应返回 records。"""
    records = pd.DataFrame({"x": [1, 2, 3]}).summary(
        None,
        None,
        None,
        None,
        None,
        None,
        5,
        "random_split",
        None,
        None,
        "M",
        0.3,
        None,
        42,
        "dict",
    )

    assert isinstance(records, list)
    assert records[0]["特征名"] == "x"


def test_named_series_summary_accepts_direct_target():
    """具名 Series 可直接传入数组型 y 计算完整预测指标。"""
    series = pd.Series([1.0, 2.0, 3.0, 4.0], name="score")

    result = series.summary(y=[0, 0, 1, 1], n_jobs=1)

    assert result.loc[0, "特征名"] == "score"
    assert {"IV", "KS", "趋势"}.issubset(result.columns)


def test_unnamed_series_summary_uses_pandas_default_column_name():
    """匿名 Series 沿用 to_frame 的列名 0。"""
    result = pd.Series([1, 2, 3]).summary(n_jobs=1)

    assert result.loc[0, "特征名"] == 0


def test_series_summary_rejects_string_target():
    """Series 没有目标列上下文，字符串 y 应给出明确错误。"""
    with pytest.raises(ValueError, match="Series.summary 的 y 不支持列名"):
        pd.Series([1, 2, 3], name="score").summary(y="target", n_jobs=1)


def test_series_summary_supports_record_dict_return_type():
    """Series 扩展应与 DataFrame 扩展保持 records 返回契约。"""
    records = pd.Series([1, 2, 3], name="score").summary(n_jobs=1, return_type="dict")

    assert isinstance(records, list)
    assert records[0]["特征名"] == "score"


def test_series_summary_forwards_binning_configuration():
    """Series.summary 应支持与 DataFrame.summary 相同的统一分箱参数。"""
    series = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 40), name="score")
    rates = [0.1, 0.9, 0.2, 0.8]
    y = np.concatenate([np.array([1] * int(40 * rate) + [0] * (40 - int(40 * rate))) for rate in rates])

    result = series.summary(
        y=y,
        binning_params={
            "user_splits": {"score": [1.5, 3.5]},
            "strict_user_splits": True,
        },
        n_jobs=1,
    )

    assert _row(result, "score")["趋势"] == "ascending"


def test_outer_binning_args_override_binning_params_without_mutation():
    """外层分箱参数优先级最高，同时不得修改调用者字典。"""
    params = {
        "method": "uniform",
        "max_n_bins": 3,
        "random_state": 99,
        "min_bin_size": 0.05,
    }
    snapshot = params.copy()

    result = feature_summary_impl._normalize_binning_config(
        binning_method="quantile",
        max_n_bins=10,
        random_state=42,
        binning_params=params,
    )

    assert result["method"] == "quantile"
    assert result["max_n_bins"] == 10
    assert result["random_state"] == 42
    assert result["min_bin_size"] == 0.05
    assert params == snapshot


def test_all_summary_apis_default_to_quantile_ten_bins():
    """三个公开入口必须统一默认使用等频10分箱。"""
    for summary_api in (feature_summary, pd.DataFrame.summary, pd.Series.summary):
        parameters = inspect.signature(summary_api).parameters
        assert parameters["binning_method"].default == "quantile"
        assert parameters["max_n_bins"].default == 10


def test_batch_binning_config_only_keeps_current_user_splits():
    """并行批次不得重复序列化整张超宽表的显式切分点字典。"""
    config = {
        "method": "quantile",
        "max_n_bins": 10,
        "user_splits": {"a": [1], "b": [2], "c": [3]},
    }

    sliced = feature_summary_impl._slice_binning_config(config, ["a", "c"])

    assert sliced["user_splits"] == {"a": [1], "c": [3]}
    assert config["user_splits"] == {"a": [1], "b": [2], "c": [3]}


@pytest.mark.parametrize(
    "binning_params",
    [
        {"n_bins": 3},
        {"prebinning": "missing_method"},
        {"prebinning": {"method": "missing_method"}},
        {"user_splits": "x:1,2,3"},
    ],
)
def test_invalid_binning_config_is_rejected_before_feature_work(binning_params):
    """最终生效的非法分箱配置应在进入字段任务前给出中文错误。"""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="分箱"):
        feature_summary(
            df,
            y=[0, 0, 1, 1],
            binning_params=binning_params,
            n_jobs=1,
        )


@pytest.mark.parametrize(
    "outer_params",
    [
        {"binning_method": "missing_method"},
        {"max_n_bins": 0},
        {"max_n_bins": -1},
    ],
)
def test_invalid_outer_binning_config_is_rejected_before_feature_work(outer_params):
    """最终生效的非法外层分箱参数应在进入字段任务前给出中文错误。"""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="分箱"):
        feature_summary(
            df,
            y=[0, 0, 1, 1],
            n_jobs=1,
            **outer_params,
        )


def test_auto_n_jobs_reserves_cpu(monkeypatch):
    """自动模式使用统一的保守物理核预算，并受任务数约束。"""
    monkeypatch.setattr(
        "hscredit.utils.parallel.get_physical_cpu_count",
        lambda: 16,
    )

    assert feature_summary_impl._resolve_n_jobs(-1, task_count=100) == 13
    assert feature_summary_impl._resolve_n_jobs(-1, task_count=2) == 2
    assert feature_summary_impl._resolve_n_jobs(-1, task_count=0) == 1


@pytest.mark.parametrize("n_jobs", [True, 0, -2, 1.5, "2", object()])
def test_invalid_n_jobs_is_rejected(n_jobs):
    """非法工作数必须抛出共享的中文校验异常。"""
    from hscredit.exceptions import ValidationError

    with pytest.raises(ValidationError, match="n_jobs"):
        feature_summary_impl._resolve_n_jobs(n_jobs, task_count=10)


@pytest.mark.parametrize(
    ("n_jobs", "expected"),
    [(None, None), (1, 1), (1.0, 1), (0.25, 4)],
)
def test_eda_n_jobs_uses_shared_legacy_and_ratio_semantics(monkeypatch, n_jobs, expected):
    """EDA 必须接受共享解析器规定的旧串行值、浮点一核和比例值。"""
    monkeypatch.setattr("hscredit.utils.parallel.get_physical_cpu_count", lambda: 16)

    assert feature_summary_impl._resolve_n_jobs(n_jobs, task_count=100) == expected


def test_progress_displays_an_active_feature_and_exact_completion(capsys):
    """完成当前字段后，应切换到另一个仍活跃的真实字段。"""
    reporter = feature_summary_impl._FeatureProgressReporter(enabled=True, total=2)
    reporter.start("feature_a")
    reporter.start("feature_b")
    assert reporter.current_feature == "feature_b"

    reporter.complete("feature_b")
    assert reporter.completed == 1
    assert reporter.current_feature == "feature_a"
    reporter.close()

    output = capsys.readouterr().err
    assert "1/2" in output
    assert "当前处理字段" in output
    assert "feature_a" in output
    assert all(metric not in output for metric in ("IV", "KS", "PSI", "趋势"))


def test_progress_uses_fixed_bar_width_and_throttles_fast_updates(capsys):
    """字段名变化不能改变前部条形宽度，高频字段也不能逐个强制重绘。"""
    reporter = feature_summary_impl._FeatureProgressReporter(enabled=True, total=100)
    for index in range(100):
        suffix = "x" * (70 if index % 2 == 0 else 1)
        feature = f"feature_{index}_{suffix}"
        reporter.start(feature)
        reporter.complete(feature)
    reporter.close()

    output = capsys.readouterr().err
    frames = [frame for frame in output.split("\r") if "特征计算:" in frame]
    bar_widths = [len(frame.split("|", 2)[1]) for frame in frames]

    assert frames
    assert set(bar_widths) == {20}
    assert output.count("\r") < 20
    assert "] , 当前处理字段" not in output


def test_progress_prefix_and_time_width_stay_fixed(capsys):
    """不同耗时、剩余时间和速度不能改变当前字段的起始显示列。"""
    from tqdm import tqdm
    from tqdm.utils import disp_len

    total = 150_767
    reporter = feature_summary_impl._FeatureProgressReporter(enabled=True, total=total)
    bar_format = reporter._bar.bar_format
    reporter.close()
    capsys.readouterr()

    lines = [
        tqdm.format_meter(
            n=28,
            total=total,
            elapsed=elapsed,
            rate=rate,
            unit="字段",
            bar_format=bar_format,
            postfix="当前处理字段=完整字段名",
            prefix="特征计算",
        )
        for elapsed, rate in ((4, 4.14), (4, 8.86), (40_000, 31.42))
    ]
    prefix_widths = [disp_len(line.split("当前处理字段=", 1)[0]) for line in lines]

    assert len(set(prefix_widths)) == 1


def test_progress_escapes_feature_control_characters_without_truncating(capsys):
    """字段控制字符应显示为完整转义文本，不能在刷新帧内部产生换行。"""
    feature = "字段第一段\n字段第二段\r字段第三段\t字段末尾"
    reporter = feature_summary_impl._FeatureProgressReporter(enabled=True, total=1)

    reporter.start(feature)
    reporter.close()

    output = capsys.readouterr().err
    output_without_close_newline = output[:-1] if output.endswith("\n") else output
    assert "字段第一段\\n字段第二段\\r字段第三段\\t字段末尾" in output
    assert "\n" not in output_without_close_newline


def test_progress_keeps_full_variable_length_feature_suffix(capsys):
    """固定前部进度条后，后部当前字段应保留完整的可变长度名称。"""
    long_feature = "超长字段_" + "x" * 120
    reporter = feature_summary_impl._FeatureProgressReporter(enabled=True, total=1)

    reporter.start(long_feature)
    reporter.close()

    output = capsys.readouterr().err
    assert f"当前处理字段={long_feature}" in output


def test_batch_progress_does_not_preannounce_pending_fields():
    """批量预聚合后，进度事件只能进入当前实际计算的单字段。"""

    class RecordingReporter:
        def __init__(self):
            self.events = []

        def start(self, feature):
            self.events.append(("start", feature))

        def complete(self, feature):
            self.events.append(("complete", feature))

    reporter = RecordingReporter()
    df = pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]})

    feature_summary_impl._summarize_complete_batch(
        df=df,
        batch=["first", "second"],
        percentiles=[0.5],
        numeric_as_categorical=set(),
        force_numeric=set(),
        y_series=None,
        psi_context=None,
        max_n_bins=5,
        reporter=reporter,
    )

    first_complete = reporter.events.index(("complete", "first"))
    second_start = reporter.events.index(("start", "second"))
    assert first_complete < second_start


def test_numeric_batch_failure_falls_back_to_individual_fields(monkeypatch):
    """某个批量聚合失败时不能丢失整个字段批次。"""
    monkeypatch.setattr(
        feature_summary_impl,
        "_prepare_numeric_batch_stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("unsupported dtype")),
    )
    df = pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]})

    results = feature_summary_impl._summarize_complete_batch(
        df=df,
        batch=["first", "second"],
        percentiles=[0.5],
        numeric_as_categorical=set(),
        force_numeric=set(),
        y_series=None,
        psi_context=None,
        max_n_bins=5,
        reporter=None,
    )

    assert [result["特征名"] for result in results] == ["first", "second"]


def test_feature_summary_preserves_order_and_mixed_statistics():
    """并行摘要应保持有效字段顺序以及数值/类别分位语义。"""
    df = pd.DataFrame(
        {
            "category": ["b", "a", "a", None],
            "number": [0.0, -1.0, 2.0, np.nan],
            "constant": [1, 1, 1, 1],
        }
    )

    result = feature_summary(
        df,
        features=["number", "missing", "category", "constant"],
        percentiles=[0.25, 0.5, 0.75],
        n_jobs=2,
    )

    assert result["特征名"].tolist() == ["number", "category", "constant"]
    number = _row(result, "number")
    assert number["25%"] == -0.5
    assert number["50%"] == 0.0
    assert number["75%"] == 1.0
    assert _row(result, "category")["50%"] == "a"


def test_parallel_and_serial_basic_results_are_identical():
    """并行基础统计应与串行结果完全一致。"""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 40)), columns=[f"f{i}" for i in range(40)])

    serial = feature_summary(df, n_jobs=1)
    parallel = feature_summary(df, n_jobs=2)

    pdt.assert_frame_equal(serial, parallel)


def test_feature_summary_rejects_invalid_n_jobs_through_public_api():
    """公共入口不能静默忽略非法并行配置。"""
    with pytest.raises(ValueError, match="n_jobs"):
        feature_summary(pd.DataFrame({"x": [1, 2, 3]}), n_jobs=0)


def test_feature_summary_progress_reports_current_field(capsys):
    """真实摘要调用仅展示字段计数和当前活跃字段。"""
    df = pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [3, 2, 1]})

    feature_summary(df, n_jobs=2, show_progress=True)

    output = capsys.readouterr().err
    assert "2/2" in output
    assert "当前处理字段" in output
    assert "feature_" in output
    assert all(metric not in output for metric in ("IV", "KS", "PSI", "趋势"))


def test_parallel_and_serial_complete_metrics_are_identical():
    """IV、KS、趋势和 PSI 的串并行结果应完全一致。"""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"f{i}" for i in range(12)])
    y = rng.integers(0, 2, size=len(df))

    serial = feature_summary(df, y=y, n_jobs=1, random_state=9)
    parallel = feature_summary(df, y=y, n_jobs=2, random_state=9)

    pdt.assert_frame_equal(serial, parallel)
    assert {"IV", "KS", "趋势", "PSI"}.issubset(serial.columns)


def test_predictive_metrics_run_in_joblib_worker_threads(monkeypatch):
    """传入 y 后共用的 IV/趋势分箱必须由多个 joblib 工作线程承担。"""
    from hscredit.core.binning import OptimalBinning

    original_fit = OptimalBinning.fit
    thread_ids = set()
    lock = threading.Lock()

    def recording_fit(self, *args, **kwargs):
        with lock:
            thread_ids.add(threading.get_ident())
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(OptimalBinning, "fit", recording_fit)
    rng = np.random.default_rng(71)
    df = pd.DataFrame(rng.normal(size=(200, 16)), columns=[f"f{i}" for i in range(16)])
    y = rng.integers(0, 2, size=len(df))

    feature_summary(df, y=y, n_jobs=2)

    assert len(thread_ids) > 1


@pytest.mark.parametrize(
    "psi_method, extra",
    [
        ("group_col", {"psi_group_col": "group"}),
        ("date_col", {"psi_date_col": "date", "psi_freq": "M"}),
    ],
)
def test_parallel_grouped_psi_matches_serial(psi_method, extra):
    """分组和时间 PSI 应复用行位置且保持串并行一致。"""
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=240),
            "group": np.repeat(["a", "b"], 120),
            "date": pd.date_range("2025-01-01", periods=240, freq="D"),
        }
    )

    serial = feature_summary(df, features=["x"], psi_method=psi_method, n_jobs=1, **extra)
    parallel = feature_summary(df, features=["x"], psi_method=psi_method, n_jobs=2, **extra)

    pdt.assert_frame_equal(serial, parallel)


def test_validation_dataframe_psi_matches_serial():
    """显式验证集 PSI 应保持串并行一致。"""
    train = pd.DataFrame({"x": np.arange(120, dtype=float)})
    valid = pd.DataFrame({"x": np.arange(20, 140, dtype=float)})

    pdt.assert_frame_equal(
        feature_summary(train, val_df=valid, n_jobs=1),
        feature_summary(train, val_df=valid, n_jobs=2),
    )


def test_basic_statistics_and_percentiles_keep_full_precision():
    """基础数值指标不得在摘要层截断为四位小数。"""
    series = pd.Series([0.123456789, 1.987654321, 3.141592653, 4.765432198], name="x")

    result = series.summary(percentiles=[0.37], n_jobs=1)
    row = result.iloc[0]

    assert row["最小值"] == series.min()
    assert row["最大值"] == series.max()
    assert row["平均值"] == series.mean()
    assert row["标准差"] == series.std()
    assert row["37%"] == series.quantile(0.37)


def test_predictive_metrics_keep_full_precision():
    """IV 与 KS 应等于底层算法原始结果，不在摘要层 round。"""
    from hscredit.core.binning import OptimalBinning
    from hscredit.core.metrics import ks

    rng = np.random.default_rng(81)
    df = pd.DataFrame({"x": rng.normal(size=37)})
    y = pd.Series(rng.integers(0, 2, size=len(df)), index=df.index)
    binner = OptimalBinning(method="quantile", max_n_bins=5)
    binner.fit(df[["x"]], y)
    expected_iv = binner.bin_tables_["x"]["分档IV值"].sum()
    expected_ks = ks(y, df["x"])

    row = _row(feature_summary(df, y=y, max_n_bins=5, n_jobs=1), "x")

    assert row["IV"] == expected_iv
    assert row["KS"] == expected_ks


def test_outer_binning_args_control_iv_and_keep_extension_params():
    """IV 应使用外层方法、箱数和随机种子，同时保留 params 扩展参数。"""
    from hscredit.core.binning import OptimalBinning

    x = np.concatenate([np.linspace(0, 1, 80), np.linspace(2, 10, 20)])
    y = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 10 + [1] * 10)
    df = pd.DataFrame({"x": x})
    params = {
        "method": "uniform",
        "max_n_bins": 3,
        "min_bin_size": 0.05,
        "random_state": 99,
    }
    expected_binner = OptimalBinning(
        method="quantile",
        max_n_bins=10,
        min_bin_size=params["min_bin_size"],
        random_state=42,
    ).fit(df[["x"]], y)
    expected_iv = expected_binner.bin_tables_["x"]["分档IV值"].sum()

    result = feature_summary(
        df,
        y=y,
        binning_method="quantile",
        max_n_bins=10,
        random_state=42,
        binning_params=params,
        n_jobs=1,
    )

    assert _row(result, "x")["IV"] == expected_iv


def test_shared_binning_user_splits_control_trend():
    """趋势必须来自与 IV 相同的显式分箱结构。"""
    x = np.repeat([1.0, 2.0, 3.0, 4.0], 40)
    rates = [0.1, 0.9, 0.2, 0.8]
    y = np.concatenate([np.array([1] * int(40 * rate) + [0] * (40 - int(40 * rate))) for rate in rates])
    df = pd.DataFrame({"x": x})

    result = feature_summary(
        df,
        y=y,
        binning_params={
            "user_splits": {"x": [1.5, 3.5]},
            "strict_user_splits": True,
        },
        n_jobs=1,
    )

    assert _row(result, "x")["趋势"] == "ascending"


def test_custom_binning_user_splits_are_mapped_to_psi():
    """PSI 内部 value 字段应使用按原字段名传入的显式切分点。"""
    from hscredit.core.metrics import psi_table

    train = pd.DataFrame({"x": np.linspace(0, 10, 200)})
    valid = pd.DataFrame({"x": np.linspace(1, 13, 180)})
    splits = [2.0, 5.0, 8.0]
    expected = psi_table(
        train["x"],
        valid["x"],
        method="uniform",
        max_n_bins=3,
        min_bin_size=0.05,
        random_state=99,
        user_splits={"value": splits},
        strict_user_splits=True,
    )["PSI贡献"].sum()

    result = feature_summary(
        train,
        val_df=valid,
        binning_method="quantile",
        max_n_bins=10,
        binning_params={
            "method": "uniform",
            "max_n_bins": 3,
            "min_bin_size": 0.05,
            "random_state": 99,
            "user_splits": {"x": splits},
            "strict_user_splits": True,
        },
        n_jobs=1,
    )

    assert _row(result, "x")["PSI"] == expected


def test_multi_feature_user_splits_match_between_serial_and_loky():
    """多字段显式切分点经批次裁剪后，loky 结果仍须与串行完全一致。"""
    rng = np.random.default_rng(83)
    columns = [f"f{index}" for index in range(128)]
    df = pd.DataFrame(rng.normal(size=(80, len(columns))), columns=columns)
    y = rng.integers(0, 2, size=len(df))
    binning_params = {
        "user_splits": {
            "f0": [-0.5, 0.25],
            "f63": [-0.25, 0.75],
            "f127": [0.0, 1.0],
        },
        "strict_user_splits": True,
    }

    serial = feature_summary(df, y=y, binning_params=binning_params, n_jobs=1)
    parallel = feature_summary(df, y=y, binning_params=binning_params, n_jobs=2)

    pdt.assert_frame_equal(parallel, serial)


def test_psi_keeps_full_precision():
    """PSI 应保留 psi_table 求和后的完整精度。"""
    from hscredit.core.metrics import psi_table

    rng = np.random.default_rng(82)
    train = pd.DataFrame({"x": rng.normal(size=120)})
    valid = pd.DataFrame({"x": rng.normal(loc=0.25, size=120)})
    expected = psi_table(train["x"], valid["x"], max_n_bins=5)["PSI贡献"].sum()

    row = _row(feature_summary(train, val_df=valid, max_n_bins=5, n_jobs=1), "x")

    assert row["PSI"] == expected


def test_model_importance_keeps_full_precision():
    """已训练模型的重要性不应被摘要层限制为六位小数。"""

    class ExactImportanceModel:
        def get_feature_importances(self):
            return pd.Series({"x": 0.123456789123})

    result = feature_summary(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        models={"模型": ExactImportanceModel()},
        n_jobs=1,
    )

    assert _row(result, "x")["模型重要性"] == 0.123456789123


def test_parallel_strategy_adapts_to_workload(monkeypatch):
    """自动模式应跳过小任务开销，并只为重任务启用保守进程数。"""
    monkeypatch.setattr("hscredit.utils.parallel.get_physical_cpu_count", lambda: 16)

    assert feature_summary_impl._select_parallel_strategy(
        n_jobs=-1,
        feature_count=60,
        row_count=1000,
        has_expensive_metrics=True,
    ) == (1, "sequential")
    assert feature_summary_impl._select_parallel_strategy(
        n_jobs=-1,
        feature_count=240,
        row_count=1000,
        has_expensive_metrics=True,
    ) == (4, "processes")
    assert feature_summary_impl._select_parallel_strategy(
        n_jobs=-1,
        feature_count=100_000,
        row_count=50,
        has_expensive_metrics=False,
    ) == (1, "sequential")
    assert feature_summary_impl._select_parallel_strategy(
        n_jobs=2,
        feature_count=16,
        row_count=200,
        has_expensive_metrics=True,
    ) == (2, "threads")
    assert feature_summary_impl._select_parallel_strategy(
        n_jobs=4,
        feature_count=16,
        row_count=20_000,
        has_expensive_metrics=True,
        has_python_objects=True,
    ) == (4, "processes")


def test_feature_summary_honors_explicit_backend_and_config(monkeypatch):
    """公共 EDA 与 pandas 扩展必须把用户 backend/config 传到统一执行器。"""
    observed = []
    original = feature_summary_impl.parallel_execute

    def recording_execute(*args, **kwargs):
        observed.append((kwargs.get("parallel_backend"), kwargs.get("parallel_config")))
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_summary_impl, "parallel_execute", recording_execute)
    df = pd.DataFrame({"数值": np.arange(20), "分类": pd.Categorical(np.tile(["甲", "乙"], 10))})
    config = {"adaptive": False, "batch_size": 1}

    direct = feature_summary(
        df,
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=config,
    )
    via_pandas = df.summary(
        n_jobs=2,
        parallel_backend="threading",
        parallel_config=config,
    )

    pdt.assert_frame_equal(direct, via_pandas)
    assert observed
    assert all(backend == "threading" for backend, _ in observed)
    assert all(call_config == config for _, call_config in observed)


def test_feature_summary_marks_nested_binning_children(monkeypatch):
    """分箱批次漏标 Genetic 子并行时，父层不会拆分总预算。"""
    captured = []
    original_execute = feature_summary_impl.parallel_execute

    def recording_execute(function, tasks, **kwargs):
        captured.append(kwargs["workload"])
        return original_execute(function, tasks, **kwargs)

    def fake_batch(task):
        return [{"特征名": feature, "字段类型": "numeric", "样本数": len(task.df)} for feature in task.batch]

    monkeypatch.setattr(feature_summary_impl, "parallel_execute", recording_execute)
    monkeypatch.setattr(feature_summary_impl, "_run_feature_summary_batch", fake_batch)
    df = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = pd.Series(np.tile([0, 1], 10), index=df.index)

    feature_summary_impl.build_feature_summary_fields(
        df=df,
        features=["x"],
        percentiles=[0.25, 0.5, 0.75],
        numeric_as_categorical=None,
        force_numeric=None,
        y_series=y,
        val_df=None,
        max_n_bins=5,
        psi_method="split",
        psi_group_col=None,
        psi_date_col=None,
        psi_freq="M",
        psi_test_size=0.2,
        random_state=42,
        n_jobs=1,
        parallel_backend=None,
        parallel_config=None,
        show_progress=False,
        binning_method="genetic",
        binning_params={"population_size": 10, "generations": 2},
    )

    assert len(captured) == 1
    assert captured[0].has_parallel_children is True


def test_loky_progress_reports_all_fields(capsys):
    """进程后端应通过事件队列准确更新同一条字段进度。"""
    df = pd.DataFrame(
        np.arange(3 * 128, dtype=float).reshape(3, 128),
        columns=[f"feature_{index}" for index in range(128)],
    )

    serial = feature_summary(df, n_jobs=1)
    result = feature_summary(df, n_jobs=2, show_progress=True)

    output = capsys.readouterr().err
    pdt.assert_frame_equal(serial, result)
    assert len(result) == 128
    assert "128/128" in output
    assert "当前处理字段" in output
    assert "feature_" in output
