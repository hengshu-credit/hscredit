"""pandas 新旧日期频率别名兼容及保留行位置的时间分组。"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

_ALIASES = {
    "M": "ME",
    "BM": "BME",
    "CBM": "CBME",
    "SM": "SME",
    "Q": "QE",
    "BQ": "BQE",
    "A": "YE",
    "Y": "YE",
    "BA": "BYE",
    "BY": "BYE",
    "AS": "YS",
    "BAS": "BYS",
    "H": "h",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "BH": "bh",
    "CBH": "cbh",
    "d": "D",
    "w": "W",
    "b": "B",
    "c": "C",
    "m": "ME",
    "q": "QE",
    "y": "YE",
    "a": "YE",
}
_OLD_ALIASES = {
    "ME": "M",
    "BME": "BM",
    "CBME": "CBM",
    "SME": "SM",
    "QE": "Q",
    "BQE": "BQ",
    "YE": "A",
    "BYE": "BA",
    "YS": "AS",
    "BYS": "BAS",
    "h": "H",
    "min": "T",
    "s": "S",
    "ms": "L",
    "us": "U",
    "ns": "N",
    "bh": "BH",
    "cbh": "CBH",
}
_FREQUENCY_TOKEN = re.compile(r"([A-Za-z]+)((?:-[A-Za-z0-9]+)*)")


@lru_cache(maxsize=None)
def _runtime_alias(unit, anchor=""):
    """探测新别名支持，旧 pandas 使用其等价旧名，不依赖版本号比较。"""
    if unit in _OLD_ALIASES:
        try:
            probe = "1.5" + unit if unit in {"h", "min", "s", "ms", "us"} else unit + anchor
            to_offset(probe)
        except ValueError:
            return _OLD_ALIASES[unit] + anchor
    return unit + anchor


def normalize_freq(freq):
    """返回 pandas offset，兼容新旧别名、倍数、复合频率及 DateOffset 对象。"""
    value = freq
    if isinstance(freq, str):
        if not freq.strip():
            raise ValueError("freq 必须是非空 pandas 日期频率")

        def replace(match):
            unit = _ALIASES.get(match[1], match[1])
            return _runtime_alias(unit, match[2].upper())

        value = _FREQUENCY_TOKEN.sub(replace, freq.strip())
    try:
        offset = to_offset(value)
        if offset is None or offset.n <= 0:
            raise ValueError("日期分组需要正频率")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"不支持的 freq 日期频率：{freq!r}") from exc
    return offset


def frequency_name(freq):
    """用于标题及分组字段名；自定义 offset 的配置仍由原对象保留。"""
    value = normalize_freq(freq).freqstr

    def display(match):
        unit = _ALIASES.get(match[1], match[1])
        anchor = match[2]
        # 已有报告使用 M/Q/Y/W 作为标题简称；新旧输入产生同一展示名。
        unit = {"ME": "M", "QE": "Q", "YE": "Y"}.get(unit, unit)
        if (unit in {"Q", "Y"} and anchor == "-DEC") or (unit == "W" and anchor == "-SUN"):
            anchor = ""
        return unit + anchor

    return _FREQUENCY_TOKEN.sub(display, value)


def format_time_labels(labels, freq, *, period=True):
    """常规年月周保留原展示格式；其他频率保留时刻、亚秒和时区信息。"""
    offset = normalize_freq(freq)
    period_freq = None
    if period and offset.n == 1:
        if isinstance(offset, (pd.offsets.MonthEnd, pd.offsets.QuarterEnd, pd.offsets.YearEnd)):
            period_freq = offset
        elif isinstance(offset, pd.offsets.Week) and offset.weekday is not None:
            period_freq = offset
    formatted = []
    for value in labels:
        if pd.isna(value):
            formatted.append(np.nan)
            continue
        stamp = pd.Timestamp(value)
        if period_freq is not None:
            # 仅显示本地日历周期，不让展示转换丢弃时区的警告影响日期分组。
            local = stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
            formatted.append(str(local.to_period(period_freq)))
        elif stamp == stamp.normalize() and not isinstance(
            offset, (pd.offsets.Tick, pd.offsets.BusinessHour, pd.offsets.CustomBusinessHour)
        ):
            formatted.append(stamp.strftime("%Y-%m-%d"))
        elif isinstance(offset, pd.offsets.Day):
            formatted.append(stamp.strftime("%Y-%m-%d"))
        else:
            formatted.append(stamp.isoformat(sep=" "))
    return formatted


def date_groups(values, freq, *, timestamps=False):
    """按 pandas Grouper 的边界分组，并将标签放回原行；缺失日期保持缺失。

    采用 pandas 的默认 closed/label/origin，倍数频率形成不重叠的分组。
    日期可乱序、重复，Series 的索引也可重复；分组不改变标签与数据的对应关系。
    """
    offset = normalize_freq(freq)
    dates = pd.to_datetime(values)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    result = pd.Series(np.nan, index=dates.index, name=dates.name, dtype=object)
    valid = dates.notna().to_numpy()
    if not valid.any():
        return result
    positions = np.flatnonzero(valid)
    index = pd.DatetimeIndex(dates.iloc[positions])
    try:
        # 固定间隔只计算实际样本的桶，避免稀疏的分钟/纳秒数据展开海量空桶。
        # 带时区的 Day 必须交给 pandas 按本地日历处理夏令时。
        if isinstance(offset, pd.offsets.Tick) and not (isinstance(offset, pd.offsets.Day) and index.tz is not None):
            ticks = index.as_unit("ns").asi8
            origin = index.min().normalize().value
            step = offset.nanos
            remainders = (ticks % step - origin % step) % step
            boundaries = ticks - remainders
            if np.any(boundaries > ticks):
                raise ValueError("日期分组边界超出纳秒时间戳范围")
            labels = pd.to_datetime(boundaries, utc=True)
            labels = labels.tz_localize(None) if index.tz is None else labels.tz_convert(index.tz)
            result.iloc[positions] = list(labels) if timestamps else format_time_labels(labels, offset)
            return result
        rows = pd.Series(positions, index=index)
        for label, members in rows.groupby(pd.Grouper(freq=offset)):
            if len(members):
                value = label if timestamps else format_time_labels([label], offset)[0]
                result.iloc[members.to_numpy(dtype=int)] = value
    except (TypeError, ValueError, NotImplementedError, OverflowError) as exc:
        raise ValueError(f"freq={frequency_name(offset)!r} 无法用于当前日期分组：{exc}") from exc
    return result
