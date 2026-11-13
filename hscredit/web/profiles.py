"""数据预览的单列摘要与按类型聚合的全量分布，不使用预览分页样本。"""

import numpy as np
import pandas as pd

from .storage import records


def column_profile(data, column, kind=None):
    """直接计算无标签 summary，分布独立保留缺失与无效值的计数。"""
    series = data[column]
    summary = records(data[[column]].summary(n_jobs=1, show_progress=False))[0]
    missing = int(series.isna().sum())
    invalid = 0
    if pd.api.types.is_bool_dtype(series):
        kind = "boolean"
        points = [{"label": label, "count": int(series.eq(value).sum())} for value, label in [(True, "真"), (False, "假")]]
        points.append({"label": "缺失", "count": missing})
        note = "真、假及缺失值占全部样本的比例。"
    elif kind == "date" or pd.api.types.is_datetime64_any_dtype(series):
        kind = "date"
        dates = pd.to_datetime(series, errors="coerce", utc=True)
        invalid = int((dates.isna() & series.notna()).sum())
        valid = dates.dropna().dt.tz_convert(None)
        points = []
        unit = "日"
        if not valid.empty:
            span = (valid.max() - valid.min()).total_seconds() / 86400
            freq, unit = ("h", "小时") if span <= 2 else ("D", "日") if span <= 120 else ("M", "月") if span <= 365 * 6 else ("Y", "年")
            counts = valid.dt.to_period(freq).value_counts().sort_index()
            counts = counts.reindex(pd.period_range(counts.index[0], counts.index[-1], freq=freq), fill_value=0)
            points = [{"label": str(period), "count": int(count)} for period, count in counts.items()]
        note = f"按{unit}统计样本量，空档补零；缺失与无效时间不参与分布。带时区时间统一为 UTC。"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
        values = series.dropna().astype(float)
        invalid = int((~np.isfinite(values)).sum())
        values = values[np.isfinite(values)]
        points = []
        if not values.empty:
            hist, edges = np.histogram(values, bins=min(20, values.nunique()))
            points = [{"label": f"{left:.6g} ~ {right:.6g}", "count": int(count), "lower": float(left), "upper": float(right)} for left, right, count in zip(edges[:-1], edges[1:], hist)]
        note = "数值直方图，最多 20 个等宽区间；缺失与无穷值不参与分布，最右区间包含右端点。"
    else:
        kind = "category"
        counts = series.dropna().astype(str).value_counts()
        points = [{"label": str(value), "count": int(count)} for value, count in counts.head(12).items()]
        if len(counts) > 12:
            points.append({"label": "其余类别（合计）", "count": int(counts.iloc[12:].sum())})
        if missing:
            points.append({"label": "缺失值（合计）", "count": missing})
        note = "展示频数最高的 12 类，其余类别合并；缺失值单独计数。"
    return {"column": column, "kind": kind, "dtype": str(series.dtype), "rows": len(series), "missing": missing, "invalid": invalid, "summary": summary, "distribution": points, "note": note}
