"""数据探索的时间频率校验与全量样本重采样，保留空档及无效值口径。"""

import re

import pandas as pd


def exploration_offset(freq):
    """兼容月／季度／年的新旧别名，其余交由 pandas 解析。"""
    if not isinstance(freq, str) or not 1 <= len(freq.strip()) <= 32:
        raise ValueError("时间汇总频率需要为 1 到 32 个字符")
    value = freq.strip()
    match = re.fullmatch(r"(\d*)(M|ME|Q|QE|Y|YE|A)", value)
    try:
        if match:
            number, unit = match.groups()
            factory = {"M": pd.offsets.MonthEnd, "ME": pd.offsets.MonthEnd, "Q": pd.offsets.QuarterEnd, "QE": pd.offsets.QuarterEnd, "Y": pd.offsets.YearEnd, "YE": pd.offsets.YearEnd, "A": pd.offsets.YearEnd}[unit]
            offset = factory(int(number or 1))
        else:
            offset = pd.tseries.frequencies.to_offset(value)
        if offset.n <= 0:
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("时间汇总频率无效，请填写 D、W、M、Q、Y 或 2W、MS、15D 等正向频率") from exc
    return offset


def time_distribution(series, freq="M"):
    """按 resample 的边界统计样本数，限制时间段数量以免细粒度导致过量分配。"""
    offset = exploration_offset(freq)
    # pandas 2 支持混合日期格式；旧版逐值解析已接受这些格式。
    options = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    dates = pd.to_datetime(series, errors="coerce", utc=True, **options)
    valid = dates.dropna().dt.tz_convert(None)
    missing = int(series.isna().sum())
    invalid = int(dates.isna().sum()) - missing
    points = []
    if not valid.empty:
        first, last = valid.min(), valid.max()
        try:
            if isinstance(offset, pd.offsets.Tick):
                if (last - first).value // offset.nanos > 9999:
                    raise ValueError("当前频率会产生超过 10000 个时间段，请选择更粗的汇总频率")
            else:
                cursor = offset.rollback(first)
                for _ in range(10001):
                    if cursor > last:
                        break
                    cursor += offset
                else:
                    raise ValueError("当前频率会产生超过 10000 个时间段，请选择更粗的汇总频率")
            counts = pd.Series(1, index=pd.DatetimeIndex(valid)).resample(offset).size()
        except (OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
            raise ValueError("当前时间范围或汇总频率超出可计算范围，请调整频率") from exc
        if len(counts) > 10000:
            raise ValueError("当前频率会产生超过 10000 个时间段，请选择更粗的汇总频率")
        for stamp, count in counts.items():
            if isinstance(offset, (pd.offsets.MonthEnd, pd.offsets.MonthBegin)):
                label = stamp.strftime("%Y-%m")
            elif isinstance(offset, (pd.offsets.YearEnd, pd.offsets.YearBegin)):
                label = str(stamp.year)
            else:
                label = stamp.isoformat(sep=" ") if isinstance(offset, pd.offsets.Tick) and offset.nanos < pd.Timedelta(days=1).value else stamp.strftime("%Y-%m-%d")
            points.append({"区间": label, "样本数": int(count)})
    return {"distribution": points, "valid": len(valid), "missing": missing, "invalid": invalid, "freq": freq, "distribution_note": f"按 {freq} 汇总全量有效日期的样本数，空档补零；缺失 {missing} 行、无效日期 {invalid} 行不参与分布。时间边界使用 pandas resample 默认规则；带时区时间统一为 UTC。"}
