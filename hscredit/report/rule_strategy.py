"""拒绝规则策略文档表格转换工具."""

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core.rules import Rule


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
    :param metrics: 每个逾期指标需要展示的字段，默认使用内置 ``_DEFAULT_METRICS``
    :param target_name: 单标签报告的逾期指标名称，默认 ``"target"``
    :return: 两层列头的规则详情表

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report import rule_report_table
    >>> rep = Rule("score < 600").report(data, target='FPD')
    >>> rule_report_table(rep, rule_name='低分拒绝')
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
    :param target_name: 单标签报告的逾期指标名称，默认 ``"target"``
    :return: 两层列头的目标分析表

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report import rule_target_analysis
    >>> rep = Rule("score < 600").report(data, target='FPD')
    >>> # 当前通过率 0.8 时，评估该拒绝规则带来的逾期改善与通过率变化
    >>> rule_target_analysis(rep, current_pass_rate=0.8, rule_name='低分拒绝')
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
    """生成规则、逾期指标和命中情况组成的纵向明细表.

    与 :func:`rule_report_table` 的横向展开不同，本函数按
    ``规则 × 逾期指标 × 命中情况`` 逐行纵向罗列各项指标，便于直接落库或透视。

    :param report: ``Rule.report`` 返回的 DataFrame
    :param rule_name: 展示用规则名称，默认使用报告中的指标名称
    :param target_names: 逾期指标名称映射，如 ``{'MOB1 1+': 'fpd1'}``
    :param metrics: 需要展示的字段列表，默认使用内置 ``_DEFAULT_TARGET_METRICS``
    :param target_name: 单标签报告的逾期指标名称，默认 ``"target"``
    :return: 含 ``规则详情`` / ``逾期指标`` / ``命中情况`` 及各指标列的纵向明细表

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report import rule_target_table
    >>> rep = Rule("score < 600").report(data, target='FPD')
    >>> rule_target_table(rep, rule_name='低分拒绝')
    """
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
    :param target_name: 单标签报告的逾期指标名称，默认 ``"target"``
    :return: 两层列头的分组命中对比表，不包含合计行

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report import rule_group_hit_table
    >>> r = Rule("score < 600")
    >>> # 对比训练集 / 测试集 / OOT 三个分组下同一规则的命中效果
    >>> reports = {
    ...     '训练集': r.report(train, target='FPD'),
    ...     '测试集': r.report(test, target='FPD'),
    ...     'OOT': r.report(oot, target='FPD'),
    ... }
    >>> rule_group_hit_table(reports, rule_name='低分拒绝')
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


_FREQ_LABELS = OrderedDict((("D", "日"), ("W", "周"), ("M", "月"), ("Q", "季度")))
_VALID_GROUP_ORDER = ("asc", "desc", "appearance")

# group_order 接受：``None``/``"asc"``（升序，默认）、``"desc"``（降序）、
# ``"appearance"``（数据出现顺序）、可调用排序键、或显式分组名称序列
GroupOrder = Union[None, str, Callable[[Any], Any], Sequence[Any]]


def _order_groups(present: Sequence[Any], group_order: GroupOrder) -> list:
    """按 ``group_order`` 规则对已出现的分组标签排序，缺省升序排列."""
    present = _ordered_unique(list(present))  # 去重并保留出现顺序

    if group_order is None or (isinstance(group_order, str) and group_order == "asc"):
        try:
            return sorted(present)
        except TypeError:
            return present
    if isinstance(group_order, str):
        if group_order == "desc":
            try:
                return sorted(present, reverse=True)
            except TypeError:
                return list(reversed(present))
        if group_order == "appearance":
            return present
        raise ValueError(f"group_order 字符串仅支持 {_VALID_GROUP_ORDER} 之一")
    if callable(group_order):
        return sorted(present, key=group_order)
    if isinstance(group_order, (list, tuple, pd.Index, np.ndarray)):
        specified = list(group_order)
        specified_set = set(specified)
        ordered = [group for group in specified if group in present]
        remaining = [group for group in present if group not in specified_set]
        return ordered + remaining
    raise ValueError("group_order 必须是 'asc'/'desc'/'appearance'、可调用对象或分组名称序列")


def _resolve_group_labels(
    data: pd.DataFrame,
    date_col: Optional[str],
    freq: str,
    group_col: Optional[str],
    dropna: bool,
    group_order: GroupOrder,
) -> Tuple[pd.Series, list]:
    """根据日期+频率或分组字段，解析每条样本所属的分组标签与有序分组列表."""
    if (date_col is None) == (group_col is None):
        raise ValueError("date_col 与 group_col 必须且只能指定其中一个")

    if group_col is not None:
        if group_col not in data.columns:
            raise ValueError(f"数据集缺少分组字段列: {group_col}")
        labels = data[group_col]
    else:
        if date_col not in data.columns:
            raise ValueError(f"数据集缺少日期列: {date_col}")
        if freq not in _FREQ_LABELS:
            raise ValueError("freq必须是'D'/'W'/'M'/'Q'之一")
        parsed = pd.to_datetime(data[date_col], errors="coerce")
        labels = parsed.dt.to_period(freq).astype(str)
        labels = labels.where(parsed.notna(), other=np.nan)

    labels = pd.Series(labels, index=data.index)
    valid = labels.notna()
    if not dropna and not valid.all():
        labels = labels.where(valid, other="缺失")
        valid = pd.Series(True, index=data.index)

    unique_values = labels[valid].dropna().unique().tolist()
    return labels, _order_groups(unique_values, group_order)


def rule_group_compare(
    data: pd.DataFrame,
    rule: Union[str, Rule],
    date_col: Optional[str] = None,
    freq: str = "M",
    group_col: Optional[str] = None,
    target: str = "target",
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    rule_name: Optional[str] = None,
    target_names: Optional[Mapping[str, str]] = None,
    metrics: Optional[Mapping[str, str]] = None,
    prior_rules: Optional[Rule] = None,
    amount: Optional[str] = None,
    del_grey: bool = False,
    dropna: bool = True,
    group_order: GroupOrder = "asc",
    **kwargs: Any,
) -> pd.DataFrame:
    """直接从原始数据生成分组下的规则命中效果对比表.

    相比 :func:`rule_group_hit_table` 需要在函数外手工切分样本并逐组调用
    ``Rule.report``，本函数接收原始明细数据，按 ``日期列 + 频率`` 或 ``分组字段``
    自动切分样本，对每个分组调用同一规则的 ``Rule.report``（支持
    ``target`` 单标签或 ``overdue + dpds`` 多标签口径），再汇总为分组对比表。

    :param data: 原始明细数据 DataFrame，需包含规则所需字段、目标/逾期字段及分组依据列
    :param rule: 规则表达式字符串或 :class:`~hscredit.core.rules.Rule` 实例
    :param date_col: 日期列名，与 ``freq`` 配合按时间周期分组（与 ``group_col`` 二选一）
    :param freq: 时间频率，``'D'`` 日 / ``'W'`` 周 / ``'M'`` 月 / ``'Q'`` 季度，默认 ``'M'``
    :param group_col: 分组字段列名，按其取值分组（与 ``date_col`` 二选一）
    :param target: 目标变量列名，默认 ``"target"``，0=好样本，1=坏样本
    :param overdue: 逾期天数字段名（可选，传入时以逾期天数>DPD定义坏样本，支持多标签）
    :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本，可传入列表支持多DPD联合分析
    :param rule_name: 展示用规则名称，默认使用规则自身名称或报告中的指标名称
    :param target_names: 逾期指标名称映射，如 ``{'MOB1 1+': 'fpd1'}``
    :param metrics: 顶层展示名称到 ``Rule.report`` 字段名的映射，默认 ``_DEFAULT_GROUP_METRICS``
    :param prior_rules: 先验规则（可选），每个分组内先排除命中先验规则的样本再评估
    :param amount: 金额字段名（可选），传入时以金额口径而非样本数口径统计
    :param del_grey: 是否删除逾期天数在(0, DPD]区间内的灰度样本，默认为False
    :param dropna: 是否丢弃分组依据缺失的样本，默认为True；为False时缺失样本归入“缺失”分组
    :param group_order: 分组排列方式，默认 ``"asc"`` 升序。支持：

        * ``"asc"`` / ``"desc"`` — 按分组标签升序 / 降序
        * ``"appearance"`` — 按分组在数据中首次出现的顺序
        * 可调用对象 — 作为 ``sorted`` 的 ``key`` 排序键
        * 分组名称序列 — 按给定顺序排列，未列出的分组按出现顺序追加在末尾

    :param kwargs: 透传给 ``Rule.report`` 的其他参数（如 ``desc``、``filter_cols``、``margins`` 等）
    :return: 两层列头的分组命中对比表，不包含合计行；列头第二层为各分组名称
    :raises ValueError: ``date_col`` 与 ``group_col`` 未二选一，或所需列缺失时

    **参考样例**

    >>> from hscredit.report import rule_group_compare
    >>> # 按放款月份对比同一拒绝规则在各月样本上的命中效果（多标签口径）
    >>> rule_group_compare(
    ...     data, "score < 600", date_col='放款时间', freq='M',
    ...     overdue=['MOB1'], dpds=[7, 0], rule_name='低分拒绝',
    ... )
    >>> # 按商品类别分组、金额口径，并自定义分组展示顺序
    >>> rule_group_compare(
    ...     data, "score < 600", group_col='商品类别', target='FPD',
    ...     amount='放款金额', group_order=['手机通讯', '电脑数码', '家用电器'],
    ... )
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data 必须是非空的 DataFrame")

    rule = rule if isinstance(rule, Rule) else Rule(rule)
    labels, ordered_groups = _resolve_group_labels(data, date_col, freq, group_col, dropna, group_order)
    if not ordered_groups:
        raise ValueError("根据 date_col/group_col 未能切分出任何有效分组")

    group_reports: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for group in ordered_groups:
        subset = data.loc[labels == group]
        if subset.empty:
            continue
        group_reports[str(group)] = rule.report(
            subset,
            target=target,
            overdue=overdue,
            dpds=dpds,
            del_grey=del_grey,
            prior_rules=prior_rules,
            amount=amount,
            **kwargs,
        )

    return rule_group_hit_table(
        group_reports,
        rule_name=rule_name or rule.name,
        target_names=target_names,
        metrics=metrics,
        target_name=target,
    )


def _validate_metrics(normalized: pd.DataFrame, metrics: Sequence[str]) -> None:
    missing = [metric for metric in metrics if metric not in normalized.columns]
    if missing:
        raise ValueError(f"rule.report 结果缺少以下指标列: {missing}")


__all__ = [
    "rule_report_table",
    "rule_target_analysis",
    "rule_target_table",
    "rule_group_hit_table",
    "rule_group_compare",
]
