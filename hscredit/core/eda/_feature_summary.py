"""综合特征摘要的输入归一化与并行计算实现。"""

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...utils.parallel import ParallelWorkload, parallel_execute, resolve_n_jobs


TargetLike = Union[str, Sequence[Any], np.ndarray, pd.Series]
_NESTED_BINNING_METHODS = frozenset({"genetic", "or_tools", "cp_sat"})


def _normalize_binning_config(
    binning_method: str,
    max_n_bins: int,
    random_state: int,
    binning_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并并校验摘要指标共用的分箱配置。"""
    if binning_params is not None and not isinstance(binning_params, dict):
        raise ValueError("binning_params 分箱参数必须是字典")

    effective = dict(binning_params or {})
    # 外层便捷参数表达调用者在当前入口的最终选择，优先于 params 中的同名透传值。
    effective.update(
        {
            "method": binning_method,
            "max_n_bins": max_n_bins,
            "random_state": random_state,
        }
    )

    from ..binning import BaseBinning, OptimalBinning

    method = effective.get("method")
    try:
        effective["method"] = OptimalBinning.validate_method(method)
    except ValueError:
        raise ValueError(f"无效的分箱方法: {method!r}，可选值为 {OptimalBinning.VALID_METHODS}") from None

    bins = effective.get("max_n_bins")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) <= 0:
        raise ValueError("分箱参数 max_n_bins 必须为正整数")
    effective["max_n_bins"] = int(bins)

    user_splits = effective.get("user_splits")
    if user_splits is not None and not isinstance(user_splits, dict) and not callable(user_splits):
        raise ValueError("分箱配置 user_splits 必须是按字段名配置的字典或可调用对象")

    prebinning = effective.get("prebinning")
    if isinstance(prebinning, str):
        prebinning_method = prebinning
    elif isinstance(prebinning, dict):
        prebinning_method = prebinning.get("method", "cart")
    elif prebinning is None or isinstance(prebinning, BaseBinning):
        prebinning_method = None
    else:
        raise ValueError("分箱配置 prebinning 必须是有效方法名、配置字典或分箱器实例")
    if prebinning_method is not None:
        try:
            normalized_prebinning_method = OptimalBinning.validate_method(prebinning_method)
        except ValueError:
            raise ValueError(f"无效的预分箱方法: {prebinning_method!r}，可选值为 {OptimalBinning.VALID_METHODS}") from None
        if isinstance(prebinning, str):
            effective["prebinning"] = normalized_prebinning_method
        elif isinstance(prebinning, dict):
            effective["prebinning"] = {**prebinning, "method": normalized_prebinning_method}

    # 字段任务会容错降级数据相关异常，但用户配置错误必须在并行和全表扫描前暴露。
    # 先构造一次分箱器，可复用其构造参数校验（例如已移除的 n_bins、非法 decimal），
    # 避免这些错误在单字段 IV/PSI 的异常保护中被静默转换成 NaN。
    try:
        OptimalBinning(**effective)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"分箱配置无效: {exc}") from exc

    return effective


def _binning_config_for_feature(
    binning_config: Dict[str, Any],
    feature: Any,
    internal_feature: Optional[str] = None,
) -> Dict[str, Any]:
    """复制公共分箱配置，并仅保留当前字段对应的显式切分点。"""
    feature_config = dict(binning_config)
    user_splits = feature_config.get("user_splits")
    if not isinstance(user_splits, dict):
        return feature_config

    if feature not in user_splits:
        # 宽表按字段并行时不能把其他字段的切分点带入当前任务，否则
        # strict_user_splits 会改变未配置字段的默认分箱路径。
        feature_config.pop("user_splits", None)
        return feature_config

    # IV 使用原字段名；psi_table 内部固定将字段重命名为 value，因此 PSI
    # 需要把调用者以原字段名配置的切分点映射到内部字段名。
    target_feature = feature if internal_feature is None else internal_feature
    feature_config["user_splits"] = {target_feature: user_splits[feature]}
    return feature_config


def _slice_binning_config(binning_config: Dict[str, Any], features: Sequence[Any]) -> Dict[str, Any]:
    """为字段批次裁剪显式切分点，避免进程重复序列化超宽配置。"""
    batch_config = dict(binning_config)
    user_splits = batch_config.get("user_splits")
    if not isinstance(user_splits, dict):
        return batch_config

    batch_splits = {feature: user_splits[feature] for feature in features if feature in user_splits}
    if batch_splits:
        batch_config["user_splits"] = batch_splits
    else:
        batch_config.pop("user_splits", None)
    return batch_config


def _binning_has_parallel_children(binning_config: Dict[str, Any]) -> bool:
    """判断字段摘要使用的分箱配置是否会启动真实子并行。"""

    def can_spawn(method: Any, n_jobs: Any) -> bool:
        return method in _NESTED_BINNING_METHODS and n_jobs is not None and n_jobs not in (1, 1.0)

    inherited_n_jobs = binning_config.get("n_jobs", -1)
    if can_spawn(binning_config.get("method"), inherited_n_jobs):
        return True

    prebinning = binning_config.get("prebinning")
    if isinstance(prebinning, str):
        return can_spawn(prebinning, inherited_n_jobs)
    if isinstance(prebinning, dict):
        return can_spawn(
            prebinning.get("method", "cart"),
            prebinning.get("n_jobs", inherited_n_jobs),
        )

    prebinning_name = prebinning.__class__.__name__.lower() if prebinning is not None else ""
    prebinning_method = next(
        (method for method in _NESTED_BINNING_METHODS if method.replace("_", "") in prebinning_name),
        None,
    )
    return can_spawn(prebinning_method, getattr(prebinning, "n_jobs", inherited_n_jobs))


def _metric_binning_arguments(binning_config: Dict[str, Any]) -> Tuple[str, int, float, Dict[str, Any]]:
    """拆出指标函数的显式参数，避免与透传参数产生重复关键字。"""
    metric_kwargs = dict(binning_config)
    method = metric_kwargs.pop("method", "quantile")
    max_n_bins = metric_kwargs.pop("max_n_bins", 10)
    min_bin_size = metric_kwargs.pop("min_bin_size", 0.01)
    # iv_table/psi_table 为避免底层分箱器打印日志会固定传 verbose=False。
    metric_kwargs.pop("verbose", None)
    return method, max_n_bins, min_bin_size, metric_kwargs


def _normalize_target(df: pd.DataFrame, y: Optional[TargetLike]) -> Optional[pd.Series]:
    """将外部目标变量按位置对齐到输入数据索引。"""
    if y is None:
        return None

    if isinstance(y, str):
        if y not in df.columns:
            raise ValueError(f"目标列 '{y}' 不存在")
        return df[y]

    values = np.asarray(y)
    if values.ndim != 1 or len(values) != len(df):
        raise ValueError("目标变量长度与数据不匹配")

    return pd.Series(values, index=df.index)


def _ratio(numerator: int, denominator: int) -> float:
    """返回不缩放、不舍入的原始比例。"""
    return numerator / denominator if denominator else 0.0


def _resolve_n_jobs(n_jobs: Optional[Union[int, float]], task_count: int) -> Optional[int]:
    """使用全库共享规则解析字段摘要的并行工作数。"""
    return resolve_n_jobs(n_jobs, task_count=task_count)


def _select_parallel_strategy(
    n_jobs: Optional[Union[int, float]],
    feature_count: int,
    row_count: int,
    has_expensive_metrics: bool,
    has_python_objects: bool = False,
) -> Tuple[int, str]:
    """根据任务规模选择串行、线程或 loky 进程执行。"""
    available_workers = _resolve_n_jobs(n_jobs, feature_count)
    if available_workers is None or available_workers == 1:
        return 1, "sequential"

    if n_jobs is not None and n_jobs > 0:
        backend = "processes" if feature_count >= 128 or (has_python_objects and row_count >= 10_000) else "threads"
        return available_workers, backend

    if has_expensive_metrics:
        if feature_count < 128 and row_count < 10_000:
            return 1, "sequential"
        inferred = max(2, math.ceil(feature_count / 64), math.ceil(row_count / 50_000))
        return min(available_workers, 4, inferred), "processes"

    return 1, "sequential"


def _progress_feature_text(feature: Any) -> str:
    """完整保留字段文本，并将可能破坏单行刷新的控制字符转成可见形式。"""
    named_escapes = {
        "\t": r"\t",
        "\n": r"\n",
        "\r": r"\r",
        "\v": r"\v",
        "\f": r"\f",
    }
    escaped = []
    for character in str(feature):
        code_point = ord(character)
        if character in named_escapes:
            escaped.append(named_escapes[character])
        elif code_point < 32 or code_point == 127:
            escaped.append(f"\\x{code_point:02x}")
        elif code_point in (0x85, 0x2028, 0x2029):
            escaped.append(f"\\u{code_point:04x}")
        else:
            escaped.append(character)
    return "".join(escaped)


class _FeatureProgressReporter:
    """线程安全的字段级进度报告器。"""

    _BAR_WIDTH = 20
    _TIME_WIDTH = 12
    _RATE_WIDTH = 20
    _FEATURE_REFRESH_INTERVAL = 0.2

    def __init__(self, enabled: bool, total: int):
        self.enabled = bool(enabled)
        self.total = int(total)
        self._completed = 0
        self._active = OrderedDict()
        self._completed_fields = set()
        self._lock = threading.Lock()
        self._bar = None
        self._last_feature_refresh_at = 0.0

        if self.enabled:
            from tqdm.auto import tqdm

            count_width = max(1, len(str(self.total)))
            bar_format = "{desc}: {percentage:3.0f}%|{bar:20}| " f"{{n_fmt:>{count_width}}}/{{total_fmt}} " f"[{{elapsed:>{self._TIME_WIDTH}.{self._TIME_WIDTH}}}" f"<{{remaining:>{self._TIME_WIDTH}.{self._TIME_WIDTH}}}, " f"{{rate_fmt:>{self._RATE_WIDTH}.{self._RATE_WIDTH}}}]{{postfix}}"
            self._bar = tqdm(
                total=self.total,
                desc="特征计算",
                unit="字段",
                dynamic_ncols=False,
                mininterval=self._FEATURE_REFRESH_INTERVAL,
                bar_format=bar_format,
            )

    @property
    def completed(self) -> int:
        """已处理字段数。"""
        with self._lock:
            return self._completed

    @property
    def current_feature(self):
        """当前展示的一个活跃字段。"""
        with self._lock:
            return self._current_feature_unlocked()

    def start(self, feature) -> None:
        """登记一个正在处理的字段。"""
        if not self.enabled:
            return
        with self._lock:
            self._active.pop(feature, None)
            self._active[feature] = None
            self._set_postfix_unlocked()
            now = time.monotonic()
            if now - self._last_feature_refresh_at >= self._FEATURE_REFRESH_INTERVAL:
                self._bar.refresh()
                self._last_feature_refresh_at = now

    def complete(self, feature) -> None:
        """完成字段并将计数增加一次。"""
        if not self.enabled:
            return
        with self._lock:
            self._active.pop(feature, None)
            if feature not in self._completed_fields:
                self._completed_fields.add(feature)
                self._completed += 1
                self._set_postfix_unlocked(fallback_feature=feature)
                self._bar.update(1)
            self._set_postfix_unlocked()

    def close(self) -> None:
        """关闭进度条。"""
        if not self.enabled:
            return
        with self._lock:
            self._set_postfix_unlocked()
            self._bar.close()

    def _current_feature_unlocked(self):
        if not self._active:
            return None
        return next(reversed(self._active))

    def _set_postfix_unlocked(self, fallback_feature=None) -> None:
        current = self._current_feature_unlocked()
        if current is None:
            current = fallback_feature
        if current is None:
            self._bar.set_postfix_str("当前处理字段=-", refresh=False)
            return
        self._bar.set_postfix_str(f"当前处理字段={_progress_feature_text(current)}", refresh=False)


class _QueueProgressReporter:
    """供 loky 工作进程发送字段状态事件的轻量代理。"""

    def __init__(self, queue):
        self._queue = queue

    def start(self, feature) -> None:
        self._queue.put(("start", feature))

    def complete(self, feature) -> None:
        self._queue.put(("complete", feature))


def _monitor_progress_events(queue, reporter: _FeatureProgressReporter) -> None:
    """在主进程线程中消费 loky 工作进程的进度事件。"""
    while True:
        action, feature = queue.get()
        if action == "stop":
            return
        if action == "start":
            reporter.start(feature)
        elif action == "complete":
            reporter.complete(feature)


def _feature_batches(features: Sequence[Any], workers: int) -> Iterable[List[Any]]:
    """将超高维字段切成数量有界、负载可均衡的批次。"""
    feature_count = len(features)
    if feature_count == 0:
        return

    # 每个工作单元保留 4 个批次即可兼顾负载均衡；更多小批次在 Windows loky
    # 下会显著放大 DataFrame 切片序列化和任务调度成本。
    target_batch_count = max(1, workers * 4)
    batch_size = max(1, min(512, math.ceil(feature_count / target_batch_count)))
    for start in range(0, feature_count, batch_size):
        yield list(features[start : start + batch_size])


def _mode_from_counts(series: pd.Series, value_counts: pd.Series):
    """复用频数表并保持 pandas mode 的最小众数语义。"""
    if value_counts.empty:
        return None, 0

    mode_frequency = int(value_counts.max())
    candidates = value_counts.index[value_counts.to_numpy() == mode_frequency]
    if len(candidates) == 1:
        return candidates[0], mode_frequency

    try:
        return candidates.min(), mode_frequency
    except (TypeError, ValueError):
        modes = series.mode()
        return (modes.iloc[0] if not modes.empty else candidates[0]), mode_frequency


@dataclass
class _NumericBatchStats:
    """一个字段批次共享的数值统计缓存。"""

    zero_count: pd.Series
    negative_count: pd.Series
    minimum: pd.Series
    maximum: pd.Series
    mean: pd.Series
    std: pd.Series
    quantiles: pd.DataFrame


def _prepare_numeric_batch_stats(
    df: pd.DataFrame,
    batch: Sequence[Any],
    percentiles: Sequence[float],
) -> Optional[_NumericBatchStats]:
    """批量计算数值列统计，避免逐列重复进入 pandas 聚合器。"""
    numeric_features = [feature for feature in batch if pd.api.types.is_numeric_dtype(df[feature])]
    if not numeric_features:
        return None

    numeric = df.loc[:, numeric_features]
    return _NumericBatchStats(
        zero_count=(numeric == 0).sum(),
        negative_count=(numeric < 0).sum(),
        minimum=numeric.min(),
        maximum=numeric.max(),
        mean=numeric.mean(),
        std=numeric.std(),
        quantiles=numeric.quantile(list(percentiles)),
    )


def _infer_feature_type(
    feature,
    series: pd.Series,
    unique_count: int,
    total: int,
    numeric_as_categorical: Sequence[Any],
    force_numeric: Sequence[Any],
) -> str:
    """复用摘要唯一值数，并保持公共类型推断的判定顺序。"""
    if unique_count <= 1:
        return "constant"
    if total and unique_count / total > 0.95:
        return "id"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == "object":
        try:
            pd.to_datetime(series.dropna().iloc[:100])
            return "datetime"
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(series):
        return "categorical" if feature in numeric_as_categorical else "numerical"
    return "numerical" if feature in force_numeric else "categorical"


def _summarize_basic_feature(
    feature,
    series: pd.Series,
    total: int,
    percentiles: Sequence[float],
    numeric_as_categorical: Sequence[Any],
    force_numeric: Sequence[Any],
    numeric_stats: Optional[_NumericBatchStats] = None,
) -> Dict[str, Any]:
    """完成单字段基础统计，并复用高成本中间结果。"""
    non_null_series = series.dropna()
    non_null = len(non_null_series)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    value_counts = non_null_series.value_counts(sort=not is_numeric)
    # Categorical.value_counts 会包含频数为 0 的未使用类别；摘要只统计实际观察值。
    if isinstance(series.dtype, pd.CategoricalDtype):
        value_counts = value_counts[value_counts > 0]
    unique_count = len(value_counts)
    mode_value, mode_frequency = _mode_from_counts(non_null_series, value_counts)
    feature_type = _infer_feature_type(
        feature,
        series,
        unique_count,
        total,
        numeric_as_categorical,
        force_numeric,
    )

    result = {
        "特征名": feature,
        "字段类型": feature_type,
        "样本数": total,
        "缺失数": total - non_null,
        "缺失率": _ratio(total - non_null, total),
        "唯一值数": unique_count,
        "众数": mode_value,
        "众数频数": mode_frequency,
        "众数占比": _ratio(mode_frequency, non_null),
    }

    if is_numeric:
        if numeric_stats is not None:
            zero_count = int(numeric_stats.zero_count.loc[feature])
            negative_count = int(numeric_stats.negative_count.loc[feature])
        else:
            zero_count = int((non_null_series == 0).sum())
            negative_count = int((non_null_series < 0).sum())
    else:
        zero_count = 0
        negative_count = 0

    duplicate_count = non_null - unique_count
    result.update(
        {
            "零值数": zero_count,
            "零值率": _ratio(zero_count, non_null),
            "负值数": negative_count,
            "负值率": _ratio(negative_count, non_null),
            "重复数": duplicate_count,
            "重复率": _ratio(duplicate_count, non_null),
        }
    )

    if is_numeric:
        if numeric_stats is not None:
            minimum = numeric_stats.minimum.loc[feature]
            maximum = numeric_stats.maximum.loc[feature]
            mean = numeric_stats.mean.loc[feature]
            std = numeric_stats.std.loc[feature]
        else:
            minimum = non_null_series.min() if non_null else np.nan
            maximum = non_null_series.max() if non_null else np.nan
            mean = non_null_series.mean() if non_null else np.nan
            std = non_null_series.std() if non_null else np.nan
        result.update(
            {
                "最小值": minimum if not pd.isna(minimum) else None,
                "最大值": maximum if not pd.isna(maximum) else None,
                "平均值": mean if not pd.isna(mean) else None,
                "标准差": std if not pd.isna(std) else None,
            }
        )
        for percentile in percentiles:
            if numeric_stats is not None:
                value = numeric_stats.quantiles.loc[percentile, feature]
            else:
                value = series.quantile(percentile)
            result[f"{int(percentile * 100)}%"] = value
        return result

    result.update({"最小值": None, "最大值": None, "平均值": None, "标准差": None})
    if value_counts.empty:
        for percentile in percentiles:
            result[f"{int(percentile * 100)}%"] = None
        return result

    cumulative_counts = value_counts.to_numpy().cumsum()
    for percentile in percentiles:
        target_count = int(non_null * percentile)
        position = int(np.searchsorted(cumulative_counts, target_count, side="left"))
        position = min(position, len(value_counts) - 1)
        result[f"{int(percentile * 100)}%"] = value_counts.index[position]
    return result


@dataclass
class _PsiContext:
    """所有字段共享的 PSI 分组位置。"""

    kind: str
    pairs: List[Tuple[np.ndarray, np.ndarray]]
    val_df: Optional[pd.DataFrame] = None


def _non_null_row_mask(df: pd.DataFrame, features: Sequence[Any]) -> np.ndarray:
    """分块计算任一特征非空的行，避免复制整张超宽布尔表。"""
    mask = np.zeros(len(df), dtype=bool)
    for batch in _feature_batches(features, workers=1):
        mask |= df.loc[:, batch].notna().any(axis=1).to_numpy()
    return mask


def _positions_by_value(values: pd.Series, ordered_values: Sequence[Any]) -> Dict[Any, np.ndarray]:
    """一次构建分组值对应的整数行位置。"""
    array = values.to_numpy()
    return {value: np.flatnonzero(array == value) for value in ordered_values}


def _prepare_psi_context(
    df: pd.DataFrame,
    features: Sequence[Any],
    val_df: Optional[pd.DataFrame],
    psi_method: str,
    psi_group_col: Optional[str],
    psi_date_col: Optional[str],
    psi_freq: str,
    psi_test_size: float,
    random_state: int,
) -> Optional[_PsiContext]:
    """预计算 PSI 所需行位置，供全部字段复用。"""
    if val_df is not None:
        return _PsiContext(kind="validation", pairs=[], val_df=val_df)

    if psi_method == "random_split" and len(df) >= 100:
        row_mask = _non_null_row_mask(df, features)
        row_positions = np.flatnonzero(row_mask)
        if len(row_positions) < 100:
            return None
        expected_positions, actual_positions = train_test_split(
            row_positions,
            test_size=psi_test_size,
            random_state=random_state,
        )
        return _PsiContext(kind="random_split", pairs=[(expected_positions, actual_positions)])

    if psi_method == "group_col" and psi_group_col is not None and psi_group_col in df.columns:
        groups = df[psi_group_col].dropna().unique().tolist()
        positions = _positions_by_value(df[psi_group_col], groups)
        pairs = [(positions[first], positions[second]) for index, first in enumerate(groups) for second in groups[index + 1 :]]
        return _PsiContext(kind="group_col", pairs=pairs)

    if psi_method == "date_col" and psi_date_col is not None and psi_date_col in df.columns:
        try:
            dates = pd.to_datetime(df[psi_date_col])
            if psi_freq == "M":
                periods = dates.dt.to_period("M").astype(str)
            elif psi_freq == "W":
                periods = dates.dt.to_period("W").astype(str)
            elif psi_freq == "Q":
                periods = dates.dt.to_period("Q").astype(str)
            else:
                periods = dates.dt.date.astype(str)
            ordered_periods = sorted(periods.dropna().unique().tolist())
            positions = _positions_by_value(periods, ordered_periods)
            pairs = [(positions[first], positions[second]) for index, first in enumerate(ordered_periods) for second in ordered_periods[index + 1 :]]
            return _PsiContext(kind="date_col", pairs=pairs)
        except Exception:
            return _PsiContext(kind="date_col", pairs=[])

    return None


def _predictive_metrics(
    feature,
    series: pd.Series,
    y_series: Optional[pd.Series],
    max_n_bins: int,
    binning_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """在一个字段任务内完成 IV、KS，并由 IV 的同一分箱表识别趋势。"""
    if y_series is None:
        return {}

    is_numeric = pd.api.types.is_numeric_dtype(series)
    result = {
        "IV": np.nan,
        "KS": np.nan,
        "趋势": "unknown" if is_numeric else "categorical",
    }
    if not is_numeric:
        return result

    from ..binning import OptimalBinning
    from ..metrics import iv as iv_metric
    from ..metrics import ks as ks_metric

    effective_config = binning_config or {
        "method": "quantile",
        "max_n_bins": max_n_bins,
    }
    feature_config = _binning_config_for_feature(effective_config, feature)
    bin_table = pd.DataFrame()
    try:
        binner = OptimalBinning(**feature_config)
        binner.fit(series.to_frame(name=feature), y_series)
        bin_table = binner.bin_tables_.get(feature, pd.DataFrame())
        if not bin_table.empty and "分档IV值" in bin_table.columns:
            result["IV"] = bin_table["分档IV值"].sum()
    except Exception:
        try:
            fallback_config = _binning_config_for_feature(effective_config, feature, internal_feature="feature")
            method, bins, min_bin_size, metric_kwargs = _metric_binning_arguments(fallback_config)
            result["IV"] = iv_metric(
                y_series,
                series,
                method=method,
                max_n_bins=bins,
                min_bin_size=min_bin_size,
                **metric_kwargs,
            )
        except Exception:
            result["IV"] = np.nan

    try:
        result["KS"] = ks_metric(y_series, series)
    except Exception:
        result["KS"] = np.nan

    result["趋势"] = _trend_from_bin_table(bin_table)
    return result


def _trend_from_bin_table(bin_table: pd.DataFrame) -> str:
    """根据 IV 共用分箱表中常规箱的坏样本率识别整体趋势。"""
    if bin_table.empty or "坏样本率" not in bin_table.columns:
        return "unknown"

    regular_bins = bin_table
    if "分箱" in regular_bins.columns:
        bin_numbers = pd.to_numeric(regular_bins["分箱"], errors="coerce")
        # -1/-2 分别代表缺失箱和特殊值箱，不参与数值区间趋势判断。
        regular_bins = regular_bins.loc[bin_numbers >= 0]

    bad_rates = pd.to_numeric(regular_bins["坏样本率"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(bad_rates) < 2:
        return "unknown"

    differences = np.diff(bad_rates)
    tolerance = 1e-12
    non_decreasing = differences >= -tolerance
    non_increasing = differences <= tolerance
    if np.all(non_decreasing):
        return "ascending"
    if np.all(non_increasing):
        return "descending"

    for pivot in range(1, len(bad_rates) - 1):
        if np.all(non_decreasing[:pivot]) and np.all(non_increasing[pivot:]):
            return "peak"
        if np.all(non_increasing[:pivot]) and np.all(non_decreasing[pivot:]):
            return "valley"
    return "unknown"


def _psi_for_feature(
    df: pd.DataFrame,
    feature,
    context: Optional[_PsiContext],
    max_n_bins: int,
    binning_config: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """使用预计算行位置完成一个字段的 PSI。"""
    if context is None:
        return None

    from ..metrics import psi_table

    effective_config = binning_config or {
        "method": "quantile",
        "max_n_bins": max_n_bins,
    }
    # psi_table 会把任意输入统一命名为 value，显式切分点需同步映射。
    feature_config = _binning_config_for_feature(effective_config, feature, internal_feature="value")
    method, bins, min_bin_size, metric_kwargs = _metric_binning_arguments(feature_config)

    if context.kind == "validation":
        if context.val_df is None or feature not in context.val_df.columns:
            return np.nan
        try:
            table = psi_table(
                df[feature],
                context.val_df[feature],
                method=method,
                max_n_bins=bins,
                min_bin_size=min_bin_size,
                **metric_kwargs,
            )
            return table["PSI贡献"].sum()
        except Exception:
            return np.nan

    values = []
    for expected_positions, actual_positions in context.pairs:
        expected = df[feature].iloc[expected_positions]
        actual = df[feature].iloc[actual_positions]
        if context.kind in ("group_col", "date_col"):
            expected = expected.dropna()
            actual = actual.dropna()
            if len(expected) <= 10 or len(actual) <= 10:
                continue
        try:
            table = psi_table(
                expected,
                actual,
                method=method,
                max_n_bins=bins,
                min_bin_size=min_bin_size,
                **metric_kwargs,
            )
            values.append(table["PSI贡献"].sum())
        except Exception:
            continue

    if not values:
        return np.nan
    if context.kind == "random_split":
        return values[0]
    return float(np.mean(values))


def _summarize_complete_batch(
    df: pd.DataFrame,
    batch: Sequence[Any],
    percentiles: Sequence[float],
    numeric_as_categorical: Sequence[Any],
    force_numeric: Sequence[Any],
    y_series: Optional[pd.Series],
    psi_context: Optional[_PsiContext],
    max_n_bins: int,
    reporter,
    binning_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """在一个工作单元内完成字段批次的所有适用指标。"""
    results = []
    total = len(df)
    if reporter is not None and batch:
        # 批量预聚合会真实处理批内全部字段；只展示首字段，避免预告尚未进入
        # 单字段指标阶段的其他字段。
        reporter.start(batch[0])
    try:
        numeric_stats = _prepare_numeric_batch_stats(df, batch, percentiles)
    except Exception:
        # 某个扩展数值 dtype 不支持批量比较/quantile 时，逐字段路径仍可能可用。
        numeric_stats = None
    for feature in batch:
        if reporter is not None:
            reporter.start(feature)
        try:
            series = df[feature]
            result = _summarize_basic_feature(
                feature,
                series,
                total,
                percentiles,
                numeric_as_categorical,
                force_numeric,
                numeric_stats,
            )
            result.update(_predictive_metrics(feature, series, y_series, max_n_bins, binning_config))
            psi_value = _psi_for_feature(df, feature, psi_context, max_n_bins, binning_config)
            if psi_context is not None:
                result["PSI"] = psi_value
            results.append(result)
        finally:
            if reporter is not None:
                reporter.complete(feature)
    return results


def _slice_psi_context(context: Optional[_PsiContext], batch: Sequence[Any]) -> Optional[_PsiContext]:
    """为进程任务裁剪显式验证集，避免重复序列化整张宽表。"""
    if context is None or context.kind != "validation" or context.val_df is None:
        return context
    columns = [feature for feature in batch if feature in context.val_df.columns]
    return _PsiContext(kind=context.kind, pairs=context.pairs, val_df=context.val_df.loc[:, columns])


@dataclass
class _FeatureSummaryBatchTask:
    """字段摘要 worker 的完整输入，在线程与进程后端间共用。"""

    df: pd.DataFrame
    batch: Sequence[Any]
    percentiles: Sequence[float]
    numeric_as_categorical: Sequence[Any]
    force_numeric: Sequence[Any]
    y_series: Optional[pd.Series]
    psi_context: Optional[_PsiContext]
    max_n_bins: int
    reporter: Any
    binning_config: Optional[Dict[str, Any]] = None


def _run_feature_summary_batch(
    task: _FeatureSummaryBatchTask,
) -> List[Dict[str, Any]]:
    """执行一个字段摘要批次；串行、线程与进程使用同一 worker。"""
    return _summarize_complete_batch(
        task.df,
        task.batch,
        task.percentiles,
        task.numeric_as_categorical,
        task.force_numeric,
        task.y_series,
        task.psi_context,
        task.max_n_bins,
        task.reporter,
        task.binning_config,
    )


def build_feature_summary_fields(
    df: pd.DataFrame,
    features: Sequence[Any],
    percentiles: Sequence[float],
    numeric_as_categorical: Optional[Sequence[Any]],
    force_numeric: Optional[Sequence[Any]],
    y_series: Optional[pd.Series],
    val_df: Optional[pd.DataFrame],
    max_n_bins: int,
    psi_method: str,
    psi_group_col: Optional[str],
    psi_date_col: Optional[str],
    psi_freq: str,
    psi_test_size: float,
    random_state: int,
    n_jobs: int,
    parallel_backend: Optional[str],
    parallel_config: Optional[Dict[str, Any]],
    show_progress: bool,
    binning_method: str,
    binning_params: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """按字段批次并行计算全部字段级摘要指标。"""
    # 在任何全表 PSI 预处理前校验，避免非法参数触发无意义的大表扫描。
    _resolve_n_jobs(n_jobs, len(features))
    # 默认等频 10 箱；外层便捷参数在此覆盖 binning_params 中的同名透传值。
    effective_binning_config = _normalize_binning_config(
        binning_method,
        max_n_bins,
        random_state,
        binning_params,
    )
    numeric_as_categorical = set(numeric_as_categorical or [])
    force_numeric = set(force_numeric or [])
    psi_context = _prepare_psi_context(
        df,
        features,
        val_df,
        psi_method,
        psi_group_col,
        psi_date_col,
        psi_freq,
        psi_test_size,
        random_state,
    )
    workers, backend = _select_parallel_strategy(
        n_jobs=n_jobs,
        feature_count=len(features),
        row_count=len(df),
        has_expensive_metrics=y_series is not None or psi_context is not None,
        has_python_objects=any(not pd.api.types.is_numeric_dtype(df[feature]) for feature in features),
    )
    if parallel_backend is not None:
        if workers == 1:
            backend = "sequential"
        elif parallel_backend in {"loky", "multiprocessing"}:
            backend = "processes"
        else:
            backend = "threads"

    batches = list(_feature_batches(features, workers))
    reporter = _FeatureProgressReporter(show_progress, len(features))
    manager = None
    event_queue = None
    monitor = None
    try:
        task_reporter = reporter
        if backend == "processes":
            task_reporter = None
            if show_progress:
                from joblib.externals.loky.backend.context import get_context

                manager = get_context().Manager()
                event_queue = manager.Queue()
                task_reporter = _QueueProgressReporter(event_queue)
                monitor = threading.Thread(
                    target=_monitor_progress_events,
                    args=(event_queue, reporter),
                    daemon=True,
                )
                monitor.start()

        tasks = [
            _FeatureSummaryBatchTask(
                df=df.loc[:, batch] if backend == "processes" else df,
                batch=batch,
                percentiles=percentiles,
                numeric_as_categorical=numeric_as_categorical,
                force_numeric=force_numeric,
                y_series=y_series,
                psi_context=(_slice_psi_context(psi_context, batch) if backend == "processes" else psi_context),
                max_n_bins=max_n_bins,
                reporter=task_reporter,
                binning_config=_slice_binning_config(effective_binning_config, batch),
            )
            for batch in batches
        ]
        backend_name = (
            parallel_backend
            or {
                "sequential": None,
                "threads": "threading",
                "processes": "loky",
            }[backend]
        )
        execution_config = {"batch_size": 1}
        execution_config.update(dict(parallel_config or {}))
        has_parallel_children = _binning_has_parallel_children(effective_binning_config)
        batch_results = parallel_execute(
            _run_feature_summary_batch,
            tasks,
            n_jobs=1 if backend == "sequential" else workers,
            parallel_backend=backend_name,
            parallel_config=execution_config,
            task_labels=[f"字段批次 {index + 1}: {task.batch[0] if task.batch else ''}" for index, task in enumerate(tasks)],
            workload=ParallelWorkload(
                task_count=len(tasks),
                rows=len(df),
                columns=len(features),
                data_bytes=int(df.loc[:, features].memory_usage(deep=True).sum()),
                cost_per_item=(12.0 if y_series is not None or psi_context is not None else 1.0),
                capability={
                    "sequential": "serial_only",
                    "threads": "thread_safe",
                    "processes": "process_safe",
                }[backend],
                releases_gil=backend == "threads",
                has_parallel_children=has_parallel_children,
                operation="综合特征摘要字段批次",
            ),
            has_parallel_children=has_parallel_children,
        )
    finally:
        if event_queue is not None:
            event_queue.put(("stop", None))
        if monitor is not None:
            monitor.join()
        if manager is not None:
            manager.shutdown()
        reporter.close()

    results = [result for batch in batch_results for result in batch]
    if not results:
        empty = pd.DataFrame(columns=["字段类型", "样本数", "缺失数", "缺失率"])
        empty.index.name = "特征名"
        return empty
    return pd.DataFrame(results).set_index("特征名")
