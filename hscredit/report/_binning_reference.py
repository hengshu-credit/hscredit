"""模型报告评分与特征分箱的训练基准及共享规则。"""

import numpy as np
import pandas as pd
from joblib import hash as stable_hash


def reference_binner(report, feature, method, max_n_bins, label=None, labels=None, supplied=None, *, is_score=False):
    """训练集拟合一次；明确的 rules 优先，独立模式保留每集拟合入口。"""
    rules = report.rules
    if isinstance(rules, dict):
        selected_rules = rules.get(feature)
        if is_score:
            selected_rules = rules.get("__score__", rules.get("score") if "score" not in report.feature_names else None)
    else:
        selected_rules = rules if is_score else None
    if report.binning_reference == "independent" and selected_rules is None:
        return None
    if selected_rules is None and supplied is not None:
        return supplied
    key = (is_score, feature, method, max_n_bins, label, tuple(labels or ()), stable_hash(selected_rules))
    if key in report._reference_binners:
        return report._reference_binners[key]
    from threading import RLock

    lock = report._reference_locks.setdefault(key, RLock())
    with lock:
        if key in report._reference_binners:
            return report._reference_binners[key]
        return _fit_reference_binner(report, feature, method, max_n_bins, label, labels, selected_rules, is_score, key)


def _fit_reference_binner(report, feature, method, max_n_bins, label, labels, selected_rules, is_score, key):
    from .feature_analyzer import feature_bin_stats

    train = report._datasets[report._train_key]
    values = train.score if is_score else train.X[feature]
    # 仅携带本次分箱需要的列，不复制所有原始字段。
    frame = pd.DataFrame({feature: np.asarray(values)}, index=train.X.index)
    choices = labels or [label]
    targets = [report._get_y(report._train_key, name) for name in choices]
    target = next((y for y in targets if pd.Series(y).dropna().nunique() >= 2), targets[0])
    if pd.Series(target).dropna().nunique() < 2:
        if method not in ("quantile", "uniform", "kmeans", "kernel_density") and selected_rules is None:
            raise ValueError("训练基准没有可用于监督分箱的双类标签，请指定 rules 或无监督分箱方法")
        if len(frame) == 1:
            frame = pd.concat([frame, frame], ignore_index=True)
        target = np.arange(len(frame)) % 2
    _, fitted = feature_bin_stats(
        frame,
        feature=feature,
        target=np.asarray(target),
        method=method,
        max_n_bins=max_n_bins,
        rules=selected_rules,
        del_grey=True,
        return_binner=True,
        n_jobs=report.n_jobs,
        parallel_backend=report.parallel_backend,
        parallel_config=report.parallel_config,
    )
    result = fitted[feature]
    report._reference_binners[key] = result
    return result
