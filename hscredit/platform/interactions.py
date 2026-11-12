"""按字段类型生成一至三变量的交互分布，保留标签缺失与采样口径。"""

from itertools import combinations

import numpy as np
import pandas as pd

from .analysis import explore_charts, validate_columns
from .explore_time import exploration_offset
from .storage import records


def _pair_summary(axes, values, a, b, groups, freq):
    """用两字段的全量有效配对计算相关系数或分组均值，不使用散点抽样。"""
    continuous = {"numeric", "text"}
    if axes[a]["kind"] in continuous and axes[b]["kind"] in continuous:
        both = values.iloc[:, [a, b]].dropna()
        corr = both.iloc[:, 0].corr(both.iloc[:, 1]) if len(both) > 1 and both.nunique().min() > 1 else None
        return {"pearson": float(corr) if corr is not None and np.isfinite(corr) else None}
    for dimension, measure in ((a, b), (b, a)):
        if axes[measure]["kind"] not in continuous:
            continue
        axis = axes[dimension]
        coordinate = values.iloc[:, dimension]
        if axis["kind"] == "date":
            buckets = pd.to_datetime(coordinate, unit="ms", utc=True)
            key = pd.Grouper(key="维度", freq=exploration_offset(freq))
        else:
            buckets = coordinate.map(dict(enumerate(axis["categories"])))
            key = "维度"
        frame = pd.DataFrame({"维度": buckets, "数值": values.iloc[:, measure], "标签": groups}).dropna()
        if frame.empty:
            rows = []
        else:
            table = frame.groupby([key, "标签"], observed=True)["数值"].agg(["count", "mean"]).reset_index()
            rows = records(table.rename(columns={"count": "样本数", "mean": "均值"}))
        return {"means": rows, "dimension": dimension, "measure": measure}
    return {}


def _axis_values(series, axis):
    kind = axis["kind"]
    if kind == "text":
        return series.astype("string").str.len().astype(float)
    if kind == "numeric":
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if kind == "date":
        dates = pd.to_datetime(series, errors="coerce", utc=True, **({"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}))
        return dates.map(lambda value: value.timestamp() * 1000 if pd.notna(value) else np.nan)
    values = series.astype("string").fillna("（缺失）")
    categories = axis["categories"]
    return values.where(values.isin(categories), "（其余类别）").map({v: i for i, v in enumerate(categories)})


def _axis_buckets(values, axis, bins):
    if "categories" in axis:
        categories = axis["categories"]
        return values.map(dict(enumerate(categories))), categories
    if axis["kind"] == "date":
        values = pd.to_datetime(values, unit="ms", utc=True)
    if not values.notna().any():
        return pd.Series(pd.NA, index=values.index, dtype="string"), []
    cuts = pd.cut(values, bins=bins, duplicates="drop")
    return cuts.astype("string"), [str(interval) for interval in cuts.cat.categories]


def interactions(data, features, target=None, bins=20, freq="M", kinds=None):
    validate_columns(data, features)
    if len(features) > 3:
        raise ValueError("最多选择三个交互变量")
    if target:
        validate_columns(data, [target])
        if target in features:
            raise ValueError("标签不能同时作为交互变量")
    groups = data[target].astype("string").fillna("（标签缺失）") if target else pd.Series("全部样本", index=data.index)
    if groups.nunique() > 30:
        raise ValueError("图表标签最多支持 30 个类别，请先加工标签")
    kinds, axes, encoded, distributions = kinds or {}, [], pd.DataFrame(index=data.index), []
    for feature in features:
        series = data[feature]
        kind = kinds.get(feature)
        if pd.api.types.is_bool_dtype(series):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "date"
        elif not kind:
            kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "category"
        if kind == "category" and pd.api.types.is_string_dtype(series.dropna().dtype) and not isinstance(series.dtype, pd.CategoricalDtype):
            text = series.dropna().astype("string")
            if text.nunique() > 30 or (len(text) and text.str.len().median() > 32):
                kind = "text"
        if kind == "text":
            lengths = series.astype("string").str.len().astype(float)
            chart = explore_charts(lengths.to_frame(feature), feature, bins=bins, feature_kind="numeric")
            distributions.append({**chart, "kind": "text"})
        elif kind == "boolean":
            distributions.append({"feature": feature, "kind": kind, "distribution": [{"区间": title, "样本数": int(count)} for title, count in [("真", series.eq(True).sum()), ("假", series.eq(False).sum()), ("缺失", series.isna().sum())]]})
        else:
            distributions.append(explore_charts(data[[feature]], feature, bins=bins, freq=freq, feature_kind=kind))
        axis = {"name": feature, "kind": kind}
        if kind == "text":
            axis["unit"] = "字符数"
        elif kind not in {"numeric", "date"}:
            values = series.astype("string").fillna("（缺失）")
            categories = values.value_counts().head(30).index.tolist()
            if (~values.isin(categories)).any() and "（其余类别）" not in categories:
                categories.append("（其余类别）")
            axis["categories"] = categories
        encoded[feature] = lengths if kind == "text" else _axis_values(series, axis)
        dist = distributions[-1]
        dist.setdefault("missing", int(series.isna().sum()))
        dist.setdefault("invalid", 0)
        dist.setdefault("valid", len(series) - dist["missing"] - dist["invalid"])
        dist["unique"] = int(series.nunique())
        if kind in {"numeric", "text"}:
            finite = encoded[feature].dropna()
            dist["five_number"] = finite.quantile([0, .25, .5, .75, 1]).astype(float).tolist() if len(finite) else []
        if target:
            observed = encoded[feature].where(series.notna()).groupby(groups)
            grouped = pd.DataFrame({"样本数": observed.size(), "有效数": observed.count(), "缺失数": series.isna().groupby(groups).sum()})
            if kind in {"numeric", "text"}:
                grouped["平均字符数" if kind == "text" else "均值"] = observed.mean()
            dist["by_group"] = records(grouped.rename_axis("分组标签").reset_index())
        axes.append(axis)

    def scatter(frame):
        valid = frame.dropna()
        sample = valid.sample(min(len(valid), 3000), random_state=42)
        return len(valid), [{"values": row, "label": label} for row, label in zip(sample.to_numpy().tolist(), groups.loc[sample.index])]

    valid_count, points = scatter(encoded)
    pairs = []
    for a, b in combinations(range(len(features)), 2):
        # 连续变量使用等宽格，类别变量保留类别坐标；二维密度来自全部有效样本。
        cell = pd.DataFrame(index=encoded.index)
        order = []
        for position in (a, b):
            feature = features[position]
            cell[position], levels = _axis_buckets(encoded[feature], axes[position], bins)
            order.append(levels)
        cell["label"] = groups
        density = cell.dropna().groupby([a, b, "label"], observed=True, sort=False).size().reset_index(name="样本数").rename(columns={a: "x", b: "y", "label": "标签"})
        pair_valid, pair_points = scatter(encoded.iloc[:, [a, b]]) if len(features) == 3 else (valid_count, points)
        pairs.append({"indices": [a, b], "valid": pair_valid, "points": pair_points, "order": order, "density": records(density), **_pair_summary(axes, encoded, a, b, groups, freq)})
    return {"axes": axes, "distributions": distributions, "pairs": pairs, "points": points, "target": target, "total": len(data), "valid": valid_count, "sampled": len(points), "note": "单变量和二维密度使用全量样本；散点固定种子最多抽取 3000 行。离散类别保留前 30 类及其余，类别坐标不表示数值距离。自由文本按字符数分析，日期轴使用 UTC。"}
