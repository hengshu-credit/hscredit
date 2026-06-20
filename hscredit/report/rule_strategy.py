"""拒绝规则策略文档表格转换工具."""

from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_DETAIL_GROUPS = ("分箱详情", "规则详情")
_DETAIL_COLUMNS = ("规则分类", "指标名称", "指标含义", "分箱")
_BIN_ORDER = ("命中", "未命中", "合计")
_DEFAULT_METRICS = ("样本总数", "样本占比", "坏样本率", "LIFT值")
_DEFAULT_TARGET_METRICS = ("样本总数", "样本占比", "坏样本率", "LIFT值", "坏账改善", "风险拒绝比")
_DEFAULT_GROUP_METRICS = OrderedDict((("坏样本率", "坏样本率"), ("样本占比", "样本占比"), ("LIFT指标", "LIFT值"), ("样本总数", "样本总数")))


def _ordered_unique(values: Sequence[str]) -> list:
    return list(dict.fromkeys(values))


def _resolve_detail_group(report: pd.DataFrame) -> str:
    groups = _ordered_unique(report.columns.get_level_values(0).tolist())
    for group in _DETAIL_GROUPS:
        if group in groups and (group, "分箱") in report.columns:
            return group
    raise ValueError("rule.report 结果缺少分箱详情列")


def _display_target(target: str, target_names: Optional[Mapping[str, str]]) -> str:
    return target_names.get(target, target) if target_names else target


def _extract_rule_name(report: pd.DataFrame, rule_name: Optional[str], detail_group: Optional[str]) -> str:
    if rule_name:
        return rule_name

    column = (detail_group, "指标名称") if detail_group else "指标名称"
    if column in report.columns:
        values = report[column].dropna().astype(str)
        values = values[values != "合计"]
        if not values.empty:
            return values.iloc[0]
    return "规则"


def _value_from_row(row: pd.Series, target: str, metric: str, detail_group: Optional[str]):
    if detail_group is None:
        return row.get(metric, np.nan)

    target_column = (target, metric)
    detail_column = (detail_group, metric)
    if target_column in row.index:
        return row[target_column]
    if detail_column in row.index:
        return row[detail_column]
    return np.nan


def _total_metric(rows: pd.DataFrame, metric: str):
    if metric in {"样本总数", "好样本数", "坏样本数"}:
        return rows[metric].sum(min_count=1)
    if metric in {"样本占比", "好样本占比", "坏样本占比"}:
        value = rows[metric].sum(min_count=1)
        return min(float(value), 1.0) if pd.notna(value) else np.nan
    if metric == "坏样本率":
        if {"坏样本数", "样本总数"}.issubset(rows.columns):
            total = rows["样本总数"].sum(min_count=1)
            bad = rows["坏样本数"].sum(min_count=1)
            if pd.notna(total) and total != 0 and pd.notna(bad):
                return bad / total
        if "样本总数" in rows.columns:
            weights = pd.to_numeric(rows["样本总数"], errors="coerce")
            values = pd.to_numeric(rows[metric], errors="coerce")
            valid = weights.notna() & values.notna()
            if valid.any() and weights[valid].sum() != 0:
                return np.average(values[valid], weights=weights[valid])
    if metric == "LIFT值":
        return 1.0
    if metric in {"坏账改善", "风险拒绝比"}:
        return 0.0
    return np.nan


def _normalize_report(
    report: pd.DataFrame,
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    target_name: str = "target",
) -> Tuple[pd.DataFrame, str]:
    """将单层或多层 ``Rule.report`` 结果转换为统一长表."""
    if not isinstance(report, pd.DataFrame) or report.empty:
        raise ValueError("report 必须是非空的 DataFrame")

    detail_group = _resolve_detail_group(report) if isinstance(report.columns, pd.MultiIndex) else None
    bin_column = (detail_group, "分箱") if detail_group else "分箱"
    if bin_column not in report.columns:
        raise ValueError("rule.report 结果缺少分箱列")

    if detail_group:
        groups = _ordered_unique(report.columns.get_level_values(0).tolist())
        targets = [group for group in groups if group != detail_group]
        detail_metrics = [column[1] for column in report.columns if column[0] == detail_group]
        target_metrics = [column[1] for column in report.columns if column[0] in targets]
        metrics = _ordered_unique(detail_metrics + target_metrics)
    else:
        targets = [target_name]
        metrics = report.columns.tolist()

    metrics = [metric for metric in metrics if metric not in _DETAIL_COLUMNS]
    resolved_rule_name = _extract_rule_name(report, rule_name, detail_group)
    records = []
    for target in targets:
        display_target = _display_target(str(target), target_names)
        for bin_name in _BIN_ORDER:
            matched = report.loc[report[bin_column] == bin_name]
            if matched.empty:
                continue
            row = matched.iloc[0]
            record = {"规则名称": resolved_rule_name, "逾期指标": display_target, "命中情况": bin_name}
            record.update({metric: _value_from_row(row, target, metric, detail_group) for metric in metrics})
            records.append(record)

        target_rows = [record for record in records if record["逾期指标"] == display_target]
        if not any(record["命中情况"] == "合计" for record in target_rows):
            detail = pd.DataFrame([record for record in target_rows if record["命中情况"] in {"命中", "未命中"}])
            total = {"规则名称": resolved_rule_name, "逾期指标": display_target, "命中情况": "合计"}
            total.update({metric: _total_metric(detail, metric) for metric in metrics})
            records.append(total)

    normalized = pd.DataFrame(records)
    if normalized.empty:
        raise ValueError("rule.report 结果中没有可用的命中、未命中或合计数据")
    target_order = _ordered_unique(normalized["逾期指标"].tolist())
    normalized["逾期指标"] = pd.Categorical(normalized["逾期指标"], categories=target_order, ordered=True)
    normalized["命中情况"] = pd.Categorical(normalized["命中情况"], categories=_BIN_ORDER, ordered=True)
    normalized = normalized.sort_values(["逾期指标", "命中情况"], kind="stable").reset_index(drop=True)
    normalized["逾期指标"] = normalized["逾期指标"].astype(object)
    normalized["命中情况"] = normalized["命中情况"].astype(object)
    return normalized, resolved_rule_name


def rule_report_table(
    report: pd.DataFrame,
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    metrics: Optional[Sequence[str]] = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """生成按逾期指标横向展开的规则详情表.

    :param report: ``Rule.report`` 返回的 DataFrame
    :param rule_name: 展示用规则名称，默认使用报告中的指标名称
    :param target_names: 逾期指标名称映射，如 ``{'MOB1 1+': 'fpd1'}``
    :param metrics: 每个逾期指标需要展示的字段
    :param target_name: 单标签报告的逾期指标名称
    :return: 两层列头的规则详情表
    """
    normalized, resolved_rule_name = _normalize_report(report, rule_name, target_names, target_name)
    metrics = list(metrics or _DEFAULT_METRICS)
    _validate_metrics(normalized, metrics)

    targets = _ordered_unique(normalized["逾期指标"].tolist())
    rows = []
    for bin_name in _BIN_ORDER:
        row = [resolved_rule_name, bin_name]
        for target in targets:
            matched = normalized[(normalized["逾期指标"] == target) & (normalized["命中情况"] == bin_name)]
            row.extend([matched.iloc[0][metric] if not matched.empty else np.nan for metric in metrics])
        rows.append(row)

    columns = [("规则详情", "规则名称"), ("规则详情", "分箱")]
    columns.extend((target, metric) for target in targets for metric in metrics)
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))


def rule_target_analysis(
    report: pd.DataFrame,
    current_pass_rate: Optional[float] = 1.0,
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """生成拒绝规则目标分析表.

    绝对逾期改善为合计坏样本率减未命中坏样本率，绝对通过率为规则样本内的
    未命中占比；相对逾期改善以合计坏样本率为分母，相对通过率则在现有策略
    通过率基础上计算，即 ``current_pass_rate * 绝对通过率``。

    :param report: ``Rule.report`` 返回的 DataFrame
    :param current_pass_rate: 规则执行前的当前通过率，取值范围为[0, 1]
    :param rule_name: 展示用规则名称
    :param target_names: 逾期指标名称映射
    :param target_name: 单标签报告的逾期指标名称
    :return: 两层列头的目标分析表
    """
    if not isinstance(current_pass_rate, (int, float, np.integer, np.floating)) or not 0 <= current_pass_rate <= 1:
        raise ValueError("current_pass_rate 必须是[0, 1]范围内的数值")

    normalized, resolved_rule_name = _normalize_report(report, rule_name, target_names, target_name)
    required = ["坏样本率", "样本总数", "样本占比", "LIFT值", "风险拒绝比"]
    _validate_metrics(normalized, required)

    rows = []
    for target in _ordered_unique(normalized["逾期指标"].tolist()):
        target_rows = normalized[normalized["逾期指标"] == target].set_index("命中情况")
        hit, miss, total = target_rows.loc["命中"], target_rows.loc["未命中"], target_rows.loc["合计"]
        absolute_overdue = total["坏样本率"] - miss["坏样本率"]
        absolute_pass = miss["样本占比"]
        relative_overdue = absolute_overdue / total["坏样本率"] if total["坏样本率"] else 0.0
        relative_pass = current_pass_rate * absolute_pass
        rows.append(
            [
                target,
                total["坏样本率"],
                miss["坏样本率"],
                hit["坏样本率"],
                total["样本总数"],
                miss["样本总数"],
                hit["样本总数"],
                hit["样本占比"],
                hit["LIFT值"],
                hit["风险拒绝比"],
                absolute_overdue,
                absolute_pass,
                relative_overdue,
                relative_pass,
            ]
        )

    columns = [
        (resolved_rule_name, "逾期指标"),
        ("坏样本率", "合计"),
        ("坏样本率", "未命中"),
        ("坏样本率", "命中"),
        ("样本总数", "合计"),
        ("样本总数", "未命中"),
        ("样本总数", "命中"),
        ("样本占比", "命中"),
        ("规则指标", "拒绝LIFT"),
        ("规则指标", "风险拒绝比"),
        ("绝对比例", "逾期改善"),
        ("绝对比例", "通过率"),
        ("相对比例", "逾期改善"),
        ("相对比例", "通过率"),
    ]
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))


def rule_target_table(
    report: pd.DataFrame,
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    metrics: Optional[Sequence[str]] = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """生成规则、逾期指标和命中情况组成的纵向明细表."""
    normalized, _ = _normalize_report(report, rule_name, target_names, target_name)
    metrics = list(metrics or _DEFAULT_TARGET_METRICS)
    _validate_metrics(normalized, metrics)
    return normalized[["规则名称", "逾期指标", "命中情况"] + metrics].rename(columns={"规则名称": "规则详情"})


def rule_group_hit_table(
    group_reports: Mapping[str, pd.DataFrame],
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    metrics: Optional[Mapping[str, str]] = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """生成多个样本分组下的规则命中效果对比表.

    :param group_reports: 分组名称到 ``Rule.report`` 结果的映射
    :param rule_name: 展示用规则名称
    :param target_names: 逾期指标名称映射
    :param metrics: 顶层展示名称到 ``Rule.report`` 字段名的映射
    :param target_name: 单标签报告的逾期指标名称
    :return: 两层列头的分组命中对比表，不包含合计行
    """
    if not isinstance(group_reports, Mapping) or not group_reports:
        raise ValueError("group_reports 必须是非空的分组名称到 DataFrame 的映射")

    metrics = OrderedDict(metrics or _DEFAULT_GROUP_METRICS)
    normalized_groups: Dict[str, pd.DataFrame] = {}
    resolved_rule_name = rule_name
    for group, report in group_reports.items():
        normalized, extracted_name = _normalize_report(report, resolved_rule_name, target_names, target_name)
        _validate_metrics(normalized, list(metrics.values()))
        normalized_groups[str(group)] = normalized
        resolved_rule_name = resolved_rule_name or extracted_name

    targets = _ordered_unique(
        [target for normalized in normalized_groups.values() for target in normalized["逾期指标"].tolist()]
    )
    rows = []
    for target in targets:
        for bin_name in ("命中", "未命中"):
            row = [resolved_rule_name, target, bin_name]
            for source_metric in metrics.values():
                for normalized in normalized_groups.values():
                    matched = normalized[(normalized["逾期指标"] == target) & (normalized["命中情况"] == bin_name)]
                    row.append(matched.iloc[0][source_metric] if not matched.empty else np.nan)
            rows.append(row)

    columns = [("规则详情", "规则名称"), ("规则详情", "逾期指标"), ("规则详情", "是否命中")]
    columns.extend((display_metric, group) for display_metric in metrics for group in normalized_groups.keys())
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))


def _validate_metrics(normalized: pd.DataFrame, metrics: Sequence[str]) -> None:
    missing = [metric for metric in metrics if metric not in normalized.columns]
    if missing:
        raise ValueError(f"rule.report 结果缺少以下指标列: {missing}")


__all__ = ["rule_report_table", "rule_target_analysis", "rule_target_table", "rule_group_hit_table"]
