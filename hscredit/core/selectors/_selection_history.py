"""筛选器有界诊断事件，避免报告历史随候选组合无限增长。"""

import numbers
import sys

import numpy as np


def initialize_history(selector):
    """校验保留策略并初始化当前拟合的事件容器。"""
    if selector.report_history not in {"summary", "diagnostics", "full"}:
        raise ValueError("report_history 必须为 'summary'、'diagnostics' 或 'full'")
    limit = selector.max_report_events
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, numbers.Integral) or limit < 1:
        raise ValueError("max_report_events 必须为正整数")
    byte_limit = selector.max_report_bytes
    if isinstance(byte_limit, (bool, np.bool_)) or not isinstance(byte_limit, numbers.Integral) or byte_limit < 1:
        raise ValueError("max_report_bytes 必须为正整数")
    selector.selection_events_ = []
    selector.report_events_truncated_ = 0
    selector.selection_events_truncated_ = 0
    selector.report_events_bytes_ = 0
    selector.report_candidates_truncated_ = 0
    selector.report_decisions_truncated_ = 0


def event_bytes(value, limit):
    """保守估算一条事件的深层内存大小，超过预算即提前终止遍历。"""
    pending = [value]
    seen = set()
    size = 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if size > limit:
            return size
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return size


def record_event(selector, event, *, diagnostic=False):
    """按模式及事件数量上限保留信息；最终特征结果不受截断影响。"""
    if not hasattr(selector, "selection_events_"):
        initialize_history(selector)
    if diagnostic and selector.report_history == "summary":
        return
    size = event_bytes(event, selector.max_report_bytes)
    if (
        len(selector.selection_events_) >= selector.max_report_events
        or selector.report_events_bytes_ + size > selector.max_report_bytes
    ):
        selector.report_events_truncated_ += 1
        selector.selection_events_truncated_ += 1
        if diagnostic:
            selector.report_candidates_truncated_ += 1
        else:
            selector.report_decisions_truncated_ += 1
        return
    selector.selection_events_.append(dict(event))
    selector.report_events_bytes_ += size
