"""规则分析模块.

提供规则集综合评估与多标签规则分析功能，以及规则置入置出分析。
"""

from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from ..core.rules import Rule
from .mining.multi_label import MultiLabelRuleMiner
from .feature_analyzer import feature_bin_stats
from .mining.base import _binning_has_parallel_children, _mining_workload
from .rule_strategy import (
    _configured_rule_copy,
    _configured_rule_report,
    _plan_report_parallel,
    _rule_report_task_count,
)
from ..utils.parallel import parallel_execute, resolve_n_jobs, validate_parallel_config


def _swap_score_bin_table_call(task):
    """计算并规范化单个评分分箱表。"""
    name, col, reference_data, target, overdue, dpds, merged_params = task
    if col not in reference_data.columns:
        raise ValueError(f"reference_data 中缺少评分列 '{col}'")
    table = feature_bin_stats(
        reference_data,
        feature=col,
        target=target,
        overdue=overdue,
        dpds=dpds,
        amount=None,
        margins=True,
        **merged_params,
    )
    return name, _normalize_bin_table(table, label=name)


def _swap_normalize_bin_table_call(task):
    """规范化调用方提供的单个评分分箱表。"""
    name, table = task
    return name, _normalize_bin_table(table, label=name)


def _swap_score_prediction_call(task):
    """计算单个评分对应的逐样本预测坏概率。"""
    name, data, score_col, table, target_name = task
    bad_rate_cols = _extract_bad_rate_cols(table, [target_name])
    return name, _compute_predicted_bad_prob(data, score_col, table, bad_rate_cols[target_name])


def _swap_rule_mask_call(task):
    """计算一条独立规则的命中掩码。"""
    _, position, rule, data = task
    mask = rule.predict(data)
    if not isinstance(mask, pd.Series):
        mask = pd.Series(np.asarray(mask, dtype=bool), index=data.index)
    else:
        mask = mask.reindex(data.index, fill_value=False).astype(bool)
    return position, mask


def _evaluate_swap_rule_masks(
    rules,
    data,
    mode,
    n_jobs,
    parallel_backend,
    parallel_config,
):
    """按独立或严格漏斗语义计算规则掩码。"""
    if not rules:
        return []
    if mode == "independent":
        tasks = [(rule.name, position, rule, data) for position, rule in enumerate(rules)]
        results = parallel_execute(
            _swap_rule_mask_call,
            tasks,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            task_labels=[task[0] for task in tasks],
            default_backend="threading",
            has_parallel_children=False,
            workload=_mining_workload(
                data,
                len(tasks),
                operation="独立规则命中计算",
                cost_per_item=6.0,
            ),
        )
        return [mask for _, mask in sorted(results, key=lambda item: item[0])]

    active = pd.Series(True, index=data.index)
    masks = []
    for rule in rules:
        subset = data.loc[active]
        predicted = rule.predict(subset)
        if not isinstance(predicted, pd.Series):
            predicted = pd.Series(np.asarray(predicted, dtype=bool), index=subset.index)
        else:
            predicted = predicted.reindex(subset.index, fill_value=False).astype(bool)
        mask = pd.Series(False, index=data.index)
        mask.loc[subset.index] = predicted
        masks.append(mask)
        active &= ~mask
    return masks


def _combine_swap_masks(masks, index):
    """按顺序合并若干布尔掩码。"""
    combined = pd.Series(False, index=index)
    for mask in masks:
        combined |= mask.reindex(index, fill_value=False).astype(bool)
    return combined


@dataclass
class _SwapStages:
    """规则置换各阶段的互斥样本掩码。"""

    base_masks: List[pd.Series]
    out_masks: List[pd.Series]
    in_masks: List[pd.Series]
    out_out: pd.Series
    in_out: pd.Series
    in_in: pd.Series
    out_in: pd.Series
    s1: pd.Series
    s2: pd.Series


_SWAP_ATOMIC_GROUPS = ("out_out", "in_out", "in_in", "out_in")


def _resolve_target_series(data, target, overdue, dpds):
    """解析分析样本上的一个或多个实际表现标签。"""
    if target is not None:
        if target not in data.columns:
            return {target: pd.Series(np.nan, index=data.index, dtype=float)}
        return {target: pd.to_numeric(data[target], errors="coerce").astype(float)}

    if overdue is None or dpds is None:
        raise ValueError("必须传入 target 或 overdue + dpds 才能计算规则置换风险")

    overdue_cols = [overdue] if isinstance(overdue, str) else list(overdue)
    thresholds = [dpds] if np.isscalar(dpds) else list(dpds)
    targets = {}
    for overdue_col in overdue_cols:
        if overdue_col not in data.columns:
            overdue_values = pd.Series(np.nan, index=data.index, dtype=float)
        else:
            overdue_values = pd.to_numeric(data[overdue_col], errors="coerce")
        for threshold in thresholds:
            label = f"{overdue_col}_{threshold}+"
            targets[label] = (overdue_values > threshold).where(overdue_values.notna()).astype(float)
    return targets


def _resolve_risk_uplifts(out_in_uplift, risk_uplifts):
    """合并并校验四个原子客群的风险上浮系数。"""
    resolved = {
        "out_out": 1.0,
        "in_out": 1.0,
        "in_in": 1.0,
        "out_in": out_in_uplift,
    }
    unknown = set(risk_uplifts or {}) - set(_SWAP_ATOMIC_GROUPS)
    if unknown:
        raise ValueError(f"risk_uplifts 包含不支持的客群: {sorted(unknown)}")
    resolved.update(risk_uplifts or {})
    for group, value in resolved.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{group} 的风险上浮系数必须为有限且非负数") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{group} 的风险上浮系数必须为有限且非负数")
        resolved[group] = value
    return resolved


def _swap_risk_source(mask, out_in_mask, fallback_mask=None):
    """根据报告行包含的原子客群标记风险来源。"""
    predicted_count = int((mask & out_in_mask).sum())
    observed_count = int((mask & ~out_in_mask).sum())
    has_fallback = (
        fallback_mask is not None and bool((mask & out_in_mask & fallback_mask).any())
    )
    predicted_label = "评分预测(含评分缺失回退)" if has_fallback else "评分预测"
    if predicted_count and observed_count:
        return f"实际表现+{predicted_label}"
    if predicted_count:
        return predicted_label
    return "实际表现"


def _swap_relative_change(before, after):
    """计算相对变化；零基线发生变化时返回 NaN 表示未定义。"""
    if before == 0:
        return 0.0 if after == 0 else np.nan
    return (after - before) / before


def _expand_swap_masks(masks, index):
    """把阶段局部掩码扩展到全量索引后取并集。"""
    expanded = pd.Series(False, index=index)
    for mask in masks:
        expanded.loc[mask.index] |= mask.astype(bool)
    return expanded


def _build_swap_stages(
    data,
    rules_base,
    rules_out,
    rules_in,
    mode,
    n_jobs,
    parallel_backend,
    parallel_config,
):
    """按基础拒绝、置出、置入顺序构建互斥阶段样本。"""
    all_mask = pd.Series(True, index=data.index)

    base_masks = _evaluate_swap_rule_masks(
        rules_base, data, mode, n_jobs, parallel_backend, parallel_config
    )
    out_out = _expand_swap_masks(base_masks, data.index)
    s1 = all_mask & ~out_out

    out_masks = _evaluate_swap_rule_masks(
        rules_out, data.loc[s1], mode, n_jobs, parallel_backend, parallel_config
    )
    in_out = _expand_swap_masks(out_masks, data.index)
    s2 = s1 & ~in_out

    in_masks = _evaluate_swap_rule_masks(
        rules_in, data.loc[s2], mode, n_jobs, parallel_backend, parallel_config
    )
    out_in = _expand_swap_masks(in_masks, data.index)
    in_in = s2 & ~out_in

    return _SwapStages(
        base_masks=base_masks,
        out_masks=out_masks,
        in_masks=in_masks,
        out_out=out_out,
        in_out=in_out,
        in_in=in_in,
        out_in=out_in,
        s1=s1,
        s2=s2,
    )


def _merge_swap_target_pipelines(pipelines: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把多目标流水线合并为共享阶段列加目标指标列。"""
    shared_columns = [
        "规则分类",
        "指标名称",
        "规则详情",
        "行类型",
        "样本总数",
        "样本占比",
        "阶段前样本数",
        "阶段后样本数",
        "阶段前生产通过率",
        "生产通过率",
        "通过率",
        "通过率(绝对值)",
        "通过率(相对值)",
        "通过率变化",
        "样本总额",
        "金额占比",
    ]
    first = next(iter(pipelines.values())).reset_index(drop=True)
    shared = [column for column in shared_columns if column in first.columns]
    pieces = [first[shared].copy()]
    pieces[0].columns = pd.MultiIndex.from_tuples([("分箱详情", column) for column in shared])

    for target_name, pipeline in pipelines.items():
        pipeline = pipeline.reset_index(drop=True)
        metrics = [
            column
            for column in pipeline.columns
            if column not in shared and column != "目标标签"
        ]
        target_table = pipeline[metrics].copy()
        target_table.columns = pd.MultiIndex.from_tuples([(target_name, column) for column in metrics])
        pieces.append(target_table)
    return pd.concat(pieces, axis=1)


def _get_detail_group_name(table: pd.DataFrame) -> str:
    """兼容旧版 `规则详情` 和新版 `分箱详情` 顶层分组名。"""
    if not isinstance(table.columns, pd.MultiIndex):
        return ""

    level0_names = set(table.columns.get_level_values(0))
    if "分箱详情" in level0_names:
        return "分箱详情"
    if "规则详情" in level0_names:
        return "规则详情"
    raise KeyError("未找到多层表头中的详情分组列")


# ============================================================================
# 简化版 Swap 分析辅助函数（整合自 scorecardpipeline 的 swapin_report 和 ruleset_analysis）
# ============================================================================


def _resolve_bin_table(
    reference_data: Optional[pd.DataFrame],
    bin_table: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    score: Union[str, Dict[str, str]],
    target: Optional[str],
    overdue: Optional[Union[str, List[str]]],
    dpds: Optional[Union[int, List[int]]],
    bin_method: str,
    max_n_bins: int,
    min_bin_size: float,
    missing_separate: bool,
    bin_params: Optional[dict],
    data: Optional[pd.DataFrame] = None,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> Dict[str, pd.DataFrame]:
    """解析或计算分箱表，统一转换为 {评分名: 分箱表} 结构（简化版）。

    优先级：bin_table > reference_data > data 自动生成
    """
    if isinstance(score, str):
        score_map = {'_default': score}
    else:
        score_map = score

    # 1. bin_table 优先
    if bin_table is not None:
        if isinstance(bin_table, pd.DataFrame):
            if len(score_map) != 1:
                raise ValueError("多评分场景必须通过字典为每个评分逐评分提供 bin_table")
            name = list(score_map.keys())[0]
            return {name: _normalize_bin_table(bin_table, label=name)}
        elif isinstance(bin_table, dict):
            if set(bin_table) != set(score_map):
                raise ValueError(
                    "多评分场景下 bin_table 与 score 的评分名必须完全一致，"
                    f"score={sorted(score_map)}，bin_table={sorted(bin_table)}"
                )
            invalid = [name for name, table in bin_table.items() if not isinstance(table, pd.DataFrame)]
            if invalid:
                raise TypeError(f"bin_table 中以下评分对应的值不是 DataFrame: {sorted(invalid)}")
            tasks = [(name, tbl) for name, tbl in bin_table.items() if isinstance(tbl, pd.DataFrame)]
        else:
            raise TypeError("bin_table 必须为 pandas DataFrame 或评分名到 DataFrame 的字典")
        return dict(
            parallel_execute(
                _swap_normalize_bin_table_call,
                tasks,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                parallel_config=parallel_config,
                task_labels=[name for name, _ in tasks],
                default_backend="threading",
                has_parallel_children=False,
                workload=_mining_workload(
                    tasks[0][1] if tasks else data,
                    len(tasks),
                    operation="评分分箱表规范化",
                    cost_per_item=4.0,
                ),
            )
        )

    # 2. 从 reference_data 计算
    if reference_data is None:
        if data is not None:
            if target is None and (overdue is None or dpds is None):
                raise ValueError("从 data 自动生成 bin_table 时，必须传入 target 或 (overdue + dpds) 参数")
            reference_data = data.copy()
            ref_col = target if target else overdue[0] if isinstance(overdue, list) else overdue
            if ref_col and ref_col in reference_data.columns:
                reference_data = reference_data.dropna(subset=[ref_col])
        else:
            raise ValueError("必须传入 bin_table 或 reference_data 参数之一")

    if target is None and (overdue is None or dpds is None):
        raise ValueError("从 reference_data 计算分箱表时，必须传入 target 或 (overdue + dpds) 参数")

    extra_params = dict(bin_params) if bin_params else {}
    merged_params = {**extra_params, 'method': bin_method, 'max_n_bins': max_n_bins,
                      'min_bin_size': min_bin_size, 'missing_separate': missing_separate}
    merged_params.setdefault("n_jobs", -1)
    merged_params.setdefault("parallel_backend", parallel_backend)
    merged_params.setdefault("parallel_config", parallel_config)

    tasks = [
        (name, col, reference_data, target, overdue, dpds, merged_params)
        for name, col in score_map.items()
    ]
    has_parallel_children = _binning_has_parallel_children(bin_method, merged_params)
    return dict(
        parallel_execute(
            _swap_score_bin_table_call,
            tasks,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            task_labels=[name for name, *_ in tasks],
            default_backend="threading",
            has_parallel_children=has_parallel_children,
            workload=_mining_workload(
                reference_data,
                len(tasks),
                operation="评分分箱表计算",
                cost_per_item=16.0,
                has_parallel_children=has_parallel_children,
            ),
        )
    )


def _build_swap_pipeline(
    data: pd.DataFrame,
    score_map: Dict[str, str],
    score_weights: Optional[Dict[str, float]],
    bin_table_result: Dict[str, pd.DataFrame],
    rules_base: List[Rule],
    rules_out: Optional[List[Rule]],
    rules_in: Optional[List[Rule]],
    target: Optional[str],
    overdue: Optional[Union[str, List[str]]],
    dpds: Optional[Union[int, List[int]]],
    amount: Optional[str],
    out_in_uplift: float,
    risk_uplifts: Optional[Dict[str, float]],
    sample_survival_rate: float,
    reverse_order: bool,
    rule_analysis_mode: str,
    out_in_amount_fill: Optional[float],
    out_in_amount_col: Optional[str],
    y: Optional[Union[np.ndarray, pd.Series, Dict[str, pd.Series]]] = None,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按显式阶段集合构建置入置出流水线。"""
    n_total = len(data)
    all_mask = pd.Series(True, index=data.index)

    if isinstance(y, dict):
        target_series = y
    elif y is not None:
        target_name = target or "目标标签"
        target_series = {target_name: pd.Series(y, index=data.index, dtype=float)}
    else:
        target_series = _resolve_target_series(data, target, overdue, dpds)

    if len(target_series) > 1:
        pipelines = {}
        results = []
        for target_name, actual_target in target_series.items():
            pipeline, result = _build_swap_pipeline(
                data=data,
                score_map=score_map,
                score_weights=score_weights,
                bin_table_result=bin_table_result,
                rules_base=rules_base,
                rules_out=rules_out,
                rules_in=rules_in,
                target=target,
                overdue=overdue,
                dpds=dpds,
                amount=amount,
                out_in_uplift=out_in_uplift,
                risk_uplifts=risk_uplifts,
                sample_survival_rate=sample_survival_rate,
                reverse_order=reverse_order,
                rule_analysis_mode=rule_analysis_mode,
                out_in_amount_fill=out_in_amount_fill,
                out_in_amount_col=out_in_amount_col,
                y={target_name: actual_target},
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                parallel_config=parallel_config,
            )
            pipelines[target_name] = pipeline
            result = result.copy()
            result.insert(0, "目标标签", target_name)
            results.append(result)
        return _merge_swap_target_pipelines(pipelines), pd.concat(results, ignore_index=True)

    target_name, actual_risk = next(iter(target_series.items()))

    score_tasks = [
        (name, data, score_map[name], table, target_name)
        for name, table in bin_table_result.items()
    ]
    score_bad_probs = dict(
        parallel_execute(
            _swap_score_prediction_call,
            score_tasks,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            task_labels=[task[0] for task in score_tasks],
            default_backend="threading",
            has_parallel_children=False,
            workload=_mining_workload(
                data,
                len(score_tasks),
                operation="评分坏概率预测",
                cost_per_item=6.0,
            ),
        )
    )
    if not score_bad_probs:
        raise ValueError("未解析到任何评分分箱表，无法预测 OUT-IN 风险")
    prediction_fallback_mask = pd.Series(False, index=data.index)
    active_weights = score_weights or {
        name: 1.0 / len(score_bad_probs) for name in score_bad_probs
    }
    for name, probability in score_bad_probs.items():
        if active_weights.get(name, 0.0) > 0:
            fallback = probability.attrs.get("风险回退掩码")
            if fallback is not None:
                prediction_fallback_mask |= fallback.reindex(data.index, fill_value=False).astype(bool)
    if len(score_bad_probs) == 1:
        full_bad_probs = next(iter(score_bad_probs.values()))
    else:
        full_bad_probs = sum(
            (prob * active_weights[name] for name, prob in score_bad_probs.items()),
            start=pd.Series(0.0, index=data.index),
        )
    stages = _build_swap_stages(
        data,
        rules_base,
        rules_out,
        rules_in,
        rule_analysis_mode,
        n_jobs,
        parallel_backend,
        parallel_config,
    )

    actual_risk = pd.to_numeric(actual_risk, errors="coerce").reindex(data.index)
    observed_mask = ~stages.out_in
    if actual_risk.loc[observed_mask].isna().any():
        raise ValueError(f"目标标签 '{target_name}' 的非OUT-IN样本缺少实际表现")
    observed_values = actual_risk.loc[observed_mask].dropna()
    if not observed_values.isin([0, 1]).all():
        raise ValueError(f"目标标签 '{target_name}' 的实际表现必须为 0/1")
    if full_bad_probs.loc[stages.out_in].isna().any():
        raise ValueError("OUT-IN样本无法从评分分箱表映射到预测坏概率")

    raw_risk = actual_risk.copy()
    raw_risk.loc[stages.out_in] = full_bad_probs.loc[stages.out_in]
    resolved_uplifts = _resolve_risk_uplifts(out_in_uplift, risk_uplifts)
    adjusted_risk = raw_risk.copy()
    atomic_masks = {
        "out_out": stages.out_out,
        "in_out": stages.in_out,
        "in_in": stages.in_in,
        "out_in": stages.out_in,
    }
    for group, mask in atomic_masks.items():
        adjusted_risk.loc[mask] = raw_risk.loc[mask] * resolved_uplifts[group]
    full_bad_rate = float(adjusted_risk.mean()) if n_total else 0.0

    effective_amount = None
    raw_bad_amount = None
    adjusted_bad_amount = None
    full_amount = 0.0
    if amount is not None:
        if amount not in data.columns:
            raise ValueError(f"数据集中缺少金额字段 '{amount}'")
        effective_amount = pd.to_numeric(data[amount], errors="coerce").astype(float)
        if out_in_amount_col is not None:
            if out_in_amount_col not in data.columns:
                raise ValueError(f"数据集中缺少 OUT-IN 金额字段 '{out_in_amount_col}'")
            candidate = pd.to_numeric(data[out_in_amount_col], errors="coerce")
            effective_amount.loc[stages.out_in] = candidate.loc[stages.out_in].combine_first(
                effective_amount.loc[stages.out_in]
            )
        if out_in_amount_fill is not None:
            try:
                amount_fill = float(out_in_amount_fill)
            except (TypeError, ValueError) as exc:
                raise ValueError("out_in_amount_fill 必须为有限数值") from exc
            if not np.isfinite(amount_fill):
                raise ValueError("out_in_amount_fill 必须为有限数值")
            effective_amount.loc[stages.out_in] = effective_amount.loc[stages.out_in].fillna(amount_fill)
        raw_bad_amount = effective_amount * raw_risk
        adjusted_bad_amount = effective_amount * adjusted_risk
        full_amount = float(effective_amount.sum())

    def production_rate(mask):
        if n_total == 0:
            return 0.0
        return float(sample_survival_rate * mask.sum() / n_total * 100.0)

    def expanded_mask(local_mask):
        result = pd.Series(False, index=data.index)
        result.loc[local_mask.index] = local_mask.astype(bool)
        return result

    rows = []

    def append_row(
        rule_class,
        rule_name,
        row_mask,
        before_mask,
        after_mask,
        row_type,
        rule_detail="",
    ):
        n_samples = int(row_mask.sum())
        raw_bad = float(raw_risk.loc[row_mask].sum())
        adjusted_bad = float(adjusted_risk.loc[row_mask].sum())
        row = _make_swap_row(
            rule_class,
            rule_name,
            n_samples,
            adjusted_bad,
            n_total_full=n_total,
            n_bad_full=float(adjusted_risk.sum()),
            full_bad_rate=full_bad_rate,
            rule_detail=rule_detail,
        )
        before_rate = production_rate(before_mask)
        after_rate = production_rate(after_mask)
        delta = after_rate - before_rate
        row.update(
            {
                "行类型": row_type,
                "目标标签": target_name,
                "风险来源": _swap_risk_source(row_mask, stages.out_in, prediction_fallback_mask),
                "原始好样本数": n_samples - raw_bad,
                "原始坏样本数": raw_bad,
                "原始坏样本率": raw_bad / n_samples if n_samples else 0.0,
                "调整后好样本数": n_samples - adjusted_bad,
                "调整后坏样本数": adjusted_bad,
                "调整后坏样本率": adjusted_bad / n_samples if n_samples else 0.0,
                "阶段前样本数": int(before_mask.sum()),
                "阶段后样本数": int(after_mask.sum()),
                "阶段前生产通过率": before_rate,
                "生产通过率": after_rate,
                "通过率": after_rate,
                "通过率(绝对值)": after_rate,
                "通过率(相对值)": _swap_relative_change(before_rate, after_rate),
                "通过率变化": delta,
                "样本占比": n_samples / n_total if n_total else 0.0,
            }
        )
        if effective_amount is not None:
            row_amount = float(effective_amount.loc[row_mask].sum())
            row_raw_bad_amount = float(raw_bad_amount.loc[row_mask].sum())
            row_adjusted_bad_amount = float(adjusted_bad_amount.loc[row_mask].sum())
            row.update(
                {
                    "样本总额": row_amount,
                    "金额占比": row_amount / full_amount if full_amount else 0.0,
                    "原始坏样本总额": row_raw_bad_amount,
                    "调整后坏样本总额": row_adjusted_bad_amount,
                    "原始坏样本率(金额)": row_raw_bad_amount / row_amount if row_amount else 0.0,
                    "调整后坏样本率(金额)": (
                        row_adjusted_bad_amount / row_amount if row_amount else 0.0
                    ),
                }
            )
        rows.append(row)

    append_row("全量样本", "", all_mask, all_mask, all_mask, "状态")

    def append_rejection_stage(rule_class, rules, masks, parent, remaining):
        rolling_before = parent.copy()
        for rule, local_mask in zip(rules, masks):
            mask = expanded_mask(local_mask)
            if rule_analysis_mode == "sequential":
                before = rolling_before
                after = before & ~mask
                rolling_before = after
            else:
                before = parent
                after = parent & ~mask
            append_row(
                rule_class,
                rule.name,
                mask,
                before,
                after,
                "规则明细",
                rule.expr,
            )
        hit = parent & ~remaining
        append_row(rule_class, "合计", hit, parent, remaining, "阶段合计")

    if rules_base:
        append_rejection_stage(
            "OUT-OUT拒绝", rules_base, stages.base_masks, all_mask, stages.s1
        )
        append_row("剩余样本", "", stages.s1, stages.s1, stages.s1, "状态")

    if rules_out:
        append_rejection_stage(
            "IN-OUT置出", rules_out, stages.out_masks, stages.s1, stages.s2
        )

    if rules_out or rules_in:
        append_row(
            "total通过样本", "", stages.s2, stages.s2, stages.s2, "状态"
        )

    if rules_out or rules_in:
        in_in_before = stages.s2 if rules_in else stages.in_in
        append_row(
            "IN-IN通过",
            "",
            stages.in_in,
            in_in_before,
            stages.in_in,
            "状态",
        )

    if rules_in:
        rolling_before = stages.in_in.copy()
        for rule, local_mask in zip(rules_in, stages.in_masks):
            mask = expanded_mask(local_mask)
            if rule_analysis_mode == "sequential":
                before = rolling_before
                after = before | mask
                rolling_before = after
            else:
                before = stages.in_in
                after = stages.in_in | mask
            append_row(
                "OUT-IN置入",
                rule.name,
                mask,
                before,
                after,
                "规则明细",
                rule.expr,
            )
        append_row(
            "OUT-IN置入",
            "合计",
            stages.out_in,
            stages.in_in,
            stages.s2,
            "阶段合计",
        )
        append_row("ALL-IN置换", "", stages.s2, stages.s2, stages.s2, "状态")

    pipeline_df = pd.DataFrame(rows)
    pipeline_df["LIFT值"] = pipeline_df["坏样本率"].apply(
        lambda value: value / full_bad_rate if full_bad_rate > 0 else 0.0
    )
    pipeline_df["好样本数"] = pipeline_df["样本总数"] - pipeline_df["坏样本数"]
    pipeline_df["好样本占比"] = 1.0 - pipeline_df["坏样本率"]
    pipeline_df["坏样本占比"] = pipeline_df["坏样本率"]
    pipeline_df["坏账改善"] = pipeline_df["坏样本率"].apply(
        lambda value: (full_bad_rate - value) / full_bad_rate
        if full_bad_rate > 0
        else 0.0
    )
    pipeline_df["风险拒绝比"] = pipeline_df.apply(
        lambda row: row["坏账改善"] / row["样本占比"]
        if row["样本占比"] > 0
        else 0.0,
        axis=1,
    )

    before_mask = stages.in_in if rules_in else stages.s2
    after_mask = stages.s2
    pass_rate_before = production_rate(before_mask)
    pass_rate_after = production_rate(after_mask)
    bad_rate_before = (
        float(adjusted_risk.loc[before_mask].mean()) if before_mask.any() else 0.0
    )
    bad_rate_after = (
        float(adjusted_risk.loc[after_mask].mean()) if after_mask.any() else 0.0
    )
    raw_bad_before = float(raw_risk.loc[before_mask].sum())
    raw_bad_after = float(raw_risk.loc[after_mask].sum())
    effective_uplift_before = (
        float(adjusted_risk.loc[before_mask].sum()) / raw_bad_before
        if raw_bad_before > 0
        else 1.0
    )
    effective_uplift_after = (
        float(adjusted_risk.loc[after_mask].sum()) / raw_bad_after
        if raw_bad_after > 0
        else 1.0
    )
    swap_result = pd.DataFrame(
        [
            {
                "指标": "通过率",
                "变化前": pass_rate_before,
                "变化后": pass_rate_after,
                "绝对变化": pass_rate_after - pass_rate_before,
                "相对变化": _swap_relative_change(pass_rate_before, pass_rate_after),
            },
            {
                "指标": "逾期率",
                "变化前": bad_rate_before,
                "变化后": bad_rate_after,
                "绝对变化": bad_rate_after - bad_rate_before,
                "相对变化": _swap_relative_change(bad_rate_before, bad_rate_after),
            },
            {
                "指标": "风险上浮系数",
                "变化前": effective_uplift_before,
                "变化后": effective_uplift_after,
                "绝对变化": effective_uplift_after - effective_uplift_before,
                "相对变化": _swap_relative_change(effective_uplift_before, effective_uplift_after),
            },
            {
                "指标": "样本集幸存比例",
                "变化前": sample_survival_rate,
                "变化后": sample_survival_rate,
                "绝对变化": 0.0,
                "相对变化": 0.0,
            },
        ]
    )
    if reverse_order:
        pipeline_df = pipeline_df.iloc[::-1].reset_index(drop=True)

    col_order = [
        "规则分类",
        "指标名称",
        "规则详情",
        "行类型",
        "目标标签",
        "风险来源",
        "样本总数",
        "样本占比",
        "原始好样本数",
        "调整后好样本数",
        "好样本数",
        "好样本占比",
        "坏样本数",
        "坏样本占比",
        "坏样本率",
        "原始坏样本数",
        "原始坏样本率",
        "调整后坏样本数",
        "调整后坏样本率",
        "LIFT值",
        "坏账改善",
        "风险拒绝比",
        "阶段前样本数",
        "阶段后样本数",
        "阶段前生产通过率",
        "生产通过率",
        "通过率",
        "通过率(绝对值)",
        "通过率(相对值)",
        "通过率变化",
        "样本总额",
        "金额占比",
        "原始坏样本总额",
        "调整后坏样本总额",
        "原始坏样本率(金额)",
        "调整后坏样本率(金额)",
    ]
    existing = [column for column in col_order if column in pipeline_df.columns]
    extra = [column for column in pipeline_df.columns if column not in existing]
    return pipeline_df[existing + extra], swap_result


def _make_swap_row(
    rule_class: str,
    rule_name: str,
    n_samples: int,
    n_bad: float,
    n_total_full: int,
    n_bad_full: float,
    full_bad_rate: float,
    rule_detail: str = '',
    amount: Optional[str] = None,
    amount_col: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    n_good: Optional[int] = None,
) -> dict:
    """构建单行 swap pipeline 数据。

    当传入金额字段时，使用金额口径计算所有指标，列名与订单口径保持一致。

    :param rule_class: 规则分类（如 '全量样本', 'OUT-OUT拒绝' 等）
    :param rule_name: 规则名称
    :param n_samples: 样本数（订单口径）
    :param n_bad: 坏样本数（订单口径）
    :param n_total_full: 全量样本数
    :param n_bad_full: 全量坏样本数
    :param full_bad_rate: 全量坏样本率
    :param rule_detail: 规则详情（表达式）
    :param amount: 金额字段名
    :param amount_col: 金额字段名（别名）
    :param data: 数据集
    :param y: 目标变量（0/1），用于计算金额口径指标
    :param n_good: 好样本数（订单口径），自动计算或显式传入
    :return: 单行字典
    """
    n_samples = max(0, n_samples)
    n_bad = max(0.0, float(n_bad))

    # 金额口径：使用金额计算所有指标
    # - 样本总数 = 金额总数
    # - 好样本数 = 好样本金额
    # - 坏样本数 = 坏样本金额
    # - 坏样本率 = 坏样本金额 / 金额总数
    if amount and data is not None and amount in data.columns and y is not None:
        amt_values = data[amount].values
        amt_total = float(amt_values.sum())
        # 好样本金额 = 金额 * (1 - target)
        good_amt = float((amt_values * (1 - y)).sum())
        # 坏样本金额 = 金额 * target
        bad_amt = float((amt_values * y).sum())
        # 坏样本率(金额口径) = 坏样本金额 / 金额总数
        bad_rate = bad_amt / amt_total if amt_total > 0 else 0.0

        row = {
            '规则分类': rule_class,
            '指标名称': rule_name,
            '规则详情': rule_detail,
            '样本总数': int(round(amt_total)),
            '好样本数': int(round(good_amt)),
            '坏样本数': int(round(bad_amt)),
            '坏样本率': bad_rate,
        }
    else:
        # 订单口径
        bad_rate = n_bad / n_samples if n_samples > 0 else 0.0
        # 自动计算好样本数（如果未传入）
        if n_good is None:
            n_good = max(0.0, n_samples - n_bad)

        row = {
            '规则分类': rule_class,
            '指标名称': rule_name,
            '规则详情': rule_detail,
            '样本总数': n_samples,
            '好样本数': n_good,
            '坏样本数': n_bad,
            '坏样本率': bad_rate,
        }

    return row


def ruleset_analysis(
    datasets: pd.DataFrame,
    rules: List[Rule],
    target: str = "target",
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    filter_cols: Optional[List[str]] = None,
    amount: Optional[str] = None,
    n_jobs: Union[int, float] = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """用于D类调优时的规则集效果分析.

    分析规则集在数据集上的应用效果，展示原始样本、每条规则命中效果、
    各规则剩余样本以及所有规则合计命中效果。

    :param datasets: 数据集
    :param rules: 规则列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称（支持多标签，传入列表）
    :param dpds: 逾期定义方式（支持多标签，传入列表）
    :param filter_cols: 指定返回的字段列表
    :param amount: 金额字段名称，用于金额口径分析
    :return: 规则集效果评估表。单标签时返回单层列结构，多标签时返回多层列结构（MultiIndex）

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report import ruleset_analysis
    >>> rules = [Rule("score < 600", name="低分"), Rule("多头 > 5", name="多头高")]
    >>> # 单标签
    >>> ruleset_analysis(df, rules, target='FPD')
    >>> # 多逾期标签 + 金额口径
    >>> ruleset_analysis(df, rules, overdue=['MOB1', 'MOB3'], dpds=[7, 0], amount='放款金额')
    """
    datasets = datasets.copy()

    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if len(feature_names_missing) > 0:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()
    all_rules = reduce(lambda r1, r2: r1 | r2, rules)
    report_plan = _plan_report_parallel(
        n_jobs,
        outer_task_count=1,
        inner_task_count=_rule_report_task_count(overdue, dpds),
    )
    execution_kwargs = dict(
        target=target,
        overdue=overdue,
        dpds=dpds,
        filter_cols=filter_cols,
        amount=amount,
        n_jobs=report_plan.child_workers,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        **kwargs,
    )
    table_total = _configured_rule_report(
        all_rules,
        datasets,
        {**execution_kwargs, "margins": True},
    )

    if isinstance(table_total.columns, pd.MultiIndex):
        detail_group = _get_detail_group_name(table_total)
        table_total[(detail_group, "分箱")] = ["所有规则", "剩余样本", "原始样本"]
        cols_to_drop = [(detail_group, "规则分类"), (detail_group, "指标名称")]
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])
        original_row = table_total.loc[table_total[(detail_group, "分箱")] == "原始样本", :]
    else:
        table_total["分箱"] = ["所有规则", "剩余样本", "原始样本"]
        cols_to_drop = ["规则分类", "指标名称"]
        table_total = table_total.drop(columns=[c for c in cols_to_drop if c in table_total.columns])
        original_row = table_total.loc[table_total["分箱"] == "原始样本", :]
    report = pd.concat([report, original_row])

    for rule in rules:
        table = _configured_rule_report(
            rule,
            datasets,
            {**execution_kwargs, "margins": False},
        )

        if isinstance(table.columns, pd.MultiIndex):
            detail_group = _get_detail_group_name(table)
            table[(detail_group, "分箱")] = [rule.expr, "剩余样本"]
            cols_to_drop = [(detail_group, "规则分类"), (detail_group, "指标名称")]
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])
        else:
            table["分箱"] = [rule.expr, "剩余样本"]
            cols_to_drop = ["规则分类", "指标名称"]
            table = table.drop(columns=[c for c in cols_to_drop if c in table.columns])

        report = pd.concat([report, table])
        prediction_rule = _configured_rule_copy(
            rule,
            report_plan.child_workers,
            parallel_backend,
            parallel_config,
        )
        datasets = datasets[~prediction_rule.predict(datasets)]

    if isinstance(table_total.columns, pd.MultiIndex):
        detail_group = _get_detail_group_name(table_total)
        summary_row = table_total.loc[table_total[(detail_group, "分箱")] == "所有规则", :]
    else:
        summary_row = table_total.loc[table_total["分箱"] == "所有规则", :]

    report = pd.concat([report, summary_row]).reset_index(drop=True)
    return report


def multi_label_rule_analysis(
    df: pd.DataFrame,
    features: List[str],
    labels: Dict[str, str],
    miner_params: Optional[dict] = None,
    output_path: str = 'rule_analysis.xlsx',
    n_jobs: Union[int, float] = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Dict] = None,
) -> str:
    """多标签规则分析（Excel 输出）.

    报告包含：
    - 规则汇总：各规则在每个标签下的覆盖率/坏率/LIFT/有效性分类
    - 有效性矩阵：行=规则，列=标签，格=LIFT值
    - 规则分类统计：按规则类型分组的汇总统计

    :param df: 输入数据 DataFrame
    :param features: 参与挖掘的特征列表
    :param labels: 标签映射 {中文名: 列名}
    :param miner_params: 传递给 MultiLabelRuleMiner 的额外参数（如 min_support、min_lift）
    :param output_path: 输出 Excel 文件路径，默认 ``'rule_analysis.xlsx'``
    :return: 输出文件路径（即 ``output_path``）

    **参考样例**

    >>> from hscredit.report import multi_label_rule_analysis
    >>> multi_label_rule_analysis(
    ...     df,
    ...     features=['score', '近六个月非银多头机构数', '青云24'],
    ...     labels={'首逾7+': 'fpd7', '首逾0+': 'fpd0'},
    ...     output_path='多标签规则分析.xlsx',
    ... )
    """
    label_cols = list(labels.values())
    label_names = list(labels.keys())

    params = dict(
        labels=label_cols,
        label_names=label_names,
        min_support=0.02,
        min_lift=1.5,
    )
    if miner_params:
        params.update(miner_params)
    params.setdefault('n_jobs', n_jobs)
    params.setdefault('parallel_backend', parallel_backend)
    params.setdefault('parallel_config', parallel_config)

    miner = MultiLabelRuleMiner(**params)
    miner.fit(df, features=features)

    all_rules = miner.get_report()
    matrix = miner.get_effectiveness_matrix()

    if len(all_rules) > 0:
        category_stats = all_rules.groupby('规则类型').agg(
            规则条数=('规则', 'count'),
            平均覆盖率=('覆盖率', 'mean'),
        ).reset_index()
    else:
        category_stats = pd.DataFrame(columns=['规则类型', '规则条数', '平均覆盖率'])

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        all_rules.to_excel(writer, sheet_name='规则汇总', index=False)
        matrix.to_excel(writer, sheet_name='有效性矩阵', index=False)
        category_stats.to_excel(writer, sheet_name='规则分类统计', index=False)

    return output_path


def _merge_label_tables(tables: List[pd.DataFrame], label_names: List[str]) -> pd.DataFrame:
    """将多标签的 rule.report() 结果合并为多层列头DataFrame。

    参考 feature_analyzer.py 的多标签合并逻辑：
    - merge_columns（分箱详情）作为左侧固定列
    - 每张表的非merge列按标签名作为顶层列名合并
    """
    if len(tables) == 0:
        return pd.DataFrame()
    if len(tables) == 1:
        return tables[0]

    detail_group = "分箱详情"
    base_table = tables[0].copy()

    # 重建列结构：第一层为标签名，第二层为列名
    multi_cols = []
    for col in base_table.columns:
        if isinstance(col, tuple) and col[0] == detail_group:
            multi_cols.append(col)
        elif col in ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]:
            multi_cols.append((detail_group, col))
        else:
            multi_cols.append((label_names[0] if label_names else "标签0", col))
    base_table.columns = pd.MultiIndex.from_tuples(multi_cols)

    merge_on = [(detail_group, c) for c in ["规则分类", "指标名称", "分箱"]]

    for tbl, lbl in zip(tables[1:], label_names[1:]):
        tbl_copy = tbl.copy()
        tc_cols = []
        for col in tbl.columns:
            if isinstance(col, tuple) and col[0] == detail_group:
                tc_cols.append(col)
            elif col in ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]:
                tc_cols.append((detail_group, col))
            else:
                tc_cols.append((lbl, col))
        tbl_copy.columns = pd.MultiIndex.from_tuples(tc_cols)
        try:
            base_table = base_table.merge(tbl_copy, on=merge_on)
        except Exception:
            pass

    return base_table


def rule_swap_analysis(
    data: pd.DataFrame,
    score: Union[str, Dict[str, str]],
    rules_in: Optional[List[Rule]] = None,
    rules_out: Optional[List[Rule]] = None,
    rules_base: Optional[List[Rule]] = None,
    reference_data: Optional[pd.DataFrame] = None,
    bin_table: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    score_weights: Optional[Dict[str, float]] = None,
    out_in_uplift: float = 2.0,
    amount: Optional[str] = None,
    sample_survival_rate: float = 1.0,
    reverse_order: bool = False,
    out_in_amount_fill: Optional[float] = None,
    out_in_amount_col: Optional[str] = None,
    bin_method: str = 'quantile',
    max_n_bins: int = 10,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    bin_params: Optional[dict] = None,
    rule_analysis_mode: str = 'independent',
    n_jobs: Union[int, float] = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Dict] = None,
    risk_uplifts: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """规则置入置出（Swap）分析。

    整合自 scorecardpipeline 的 ``swapin_report`` 和 ``ruleset_analysis``（即 swapout_report），
    只输出 ``swap_pipeline`` 和 ``swap_result``，支持金额和订单口径。

    **阶段与四象限定义**

    ``rules_base`` 先从输入样本剔除 OUT-OUT；``rules_out`` 再在剩余样本中剔除
    IN-OUT，得到 ``total通过样本``；``rules_in`` 仅在 total通过样本中拆分
    IN-IN（不置入）和 OUT-IN（本次置入），置入后的 ALL-IN 等于二者之和。

    ==========  ==========  ====================================
    象限        含义        风险说明
    ==========  ==========  ====================================
    in_in      total通过 & 未置入     不置入时通过客群，使用实际表现
    in_out     本次置出规则命中       置出客群，使用实际表现
    out_in     本次置入规则命中       置入客群，使用评分预测风险
    out_out    生产基础规则命中       基础拒绝客群，使用实际表现
    ==========  ==========  ====================================

    :param data: 全量样本集（包含 score 列 + rules_in/rules_out/rules_base 用到的所有特征列）
    :param score: 评分字段名（str）或多评分映射（Dict）
    :param rules_in: 置入规则集（List[Rule]），对应 out_in 象限
    :param rules_out: 置出规则集（可选），对应 in_out 象限
    :param rules_base: 基准拒绝规则集（可选），对应 out_out 象限
    :param reference_data: 历史有表现参考数据集（包含 target 或 overdue+dpds）
    :param bin_table: 现成分箱表，支持：
        - pd.DataFrame：单评分分箱表
        - Dict[str, pd.DataFrame]：多评分分箱表 ``{评分名: 分箱表}``
        - None：自动从 reference_data 计算
    :param target: 目标变量名（与 bin_table 二选一）
    :param overdue: 逾期天数字段（多标签场景）
    :param dpds: 逾期天数阈值
    :param score_weights: 多模型权重（可选）
    :param out_in_uplift: 置入风险上浮系数，默认 2.0
    :param risk_uplifts: 四象限风险上浮映射，可选键为 out_out/in_out/in_in/out_in
    :param amount: 金额字段（可选），传入后同时输出金额口径报告
    :param sample_survival_rate: 分析样本进入生产时的通过比例，取值 (0, 1]，用于校准生产通过率漏斗
    :param reverse_order: 是否逆序展示；仅改变流水线行顺序，不影响置换结论
    :param out_in_amount_fill: out_in 置入样本额度填充定值（可选）
    :param out_in_amount_col: out_in 置入样本额度填充字段名（可选）
    :param bin_method: 分箱方法，默认 'quantile'（仅 reference_data 模式生效）
    :param max_n_bins: 最大分箱数，默认 10（仅 reference_data 模式生效）
    :param min_bin_size: 每箱最小样本占比，默认 0.05（仅 reference_data 模式生效）
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param bin_params: 额外分箱参数 dict，会透传给 ``feature_bin_stats``
    :param rule_analysis_mode: 规则分析模式，默认 'independent'。
        - 'independent'：每条规则应用到所属阶段的同一父样本，明细可重叠，合计按命中并集去重。
        - 'sequential'：每条规则在同阶段前一条规则处理后的剩余样本上分析。
    :return: 包含两张表的字典

        - ``swap_pipeline``：分阶段通过率及实际/预测风险变化，支持订单/金额双口径；多目标使用多层列
        - ``swap_result``：IN-IN 与 ALL-IN 的置换前后对比；多目标按 ``目标标签`` 输出长表

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report.rule_analysis import rule_swap_analysis
    >>>
    >>> # 置入规则分析（传入历史参考数据，自动计算分箱表）
    >>> result = rule_swap_analysis(
    ...     data=swap_data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     rules_base=[rule_base],
    ...     reference_data=hist_data,
    ...     target='target',
    ...     amount='放款金额',
    ... )
    >>>
    >>> print(result['swap_pipeline'])   # 分步骤报告
    >>> print(result['swap_result'])      # 置换前后对比

    >>> # 多逾期标签分析
    >>> result = rule_swap_analysis(
    ...     data=swap_data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     reference_data=hist_data,
    ...     overdue='MOB1',
    ...     dpds=[0, 7, 30],
    ...     amount='放款金额',
    ... )
    """
    # ── 第一步：校验轻量输入 ─────────────────────────────────────────────
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须为 pandas DataFrame")
    if data.empty:
        raise ValueError("data 不能为空")
    if not data.index.is_unique:
        data = data.reset_index(drop=True)
    try:
        sample_survival_rate = float(sample_survival_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("样本集幸存比例必须位于 (0, 1] 区间") from exc
    if not np.isfinite(sample_survival_rate) or not 0 < sample_survival_rate <= 1:
        raise ValueError("样本集幸存比例必须位于 (0, 1] 区间")
    if target is None:
        overdue_values = [overdue] if isinstance(overdue, str) else list(overdue or [])
        dpd_values = [dpds] if np.isscalar(dpds) and dpds is not None else list(dpds or [])
        if not overdue_values or not dpd_values:
            raise ValueError("overdue 和 dpds 不能为空")

    validate_parallel_config(parallel_backend, parallel_config)
    resolve_n_jobs(n_jobs, task_count=1)
    if rule_analysis_mode not in {"independent", "sequential"}:
        raise ValueError("rule_analysis_mode 必须为 'independent' 或 'sequential'")
    _resolve_risk_uplifts(out_in_uplift, risk_uplifts)

    if isinstance(score, str):
        score_map = {"_default": score}
    elif isinstance(score, dict) and score:
        score_map = dict(score)
    else:
        raise TypeError("score 必须为评分字段名或非空评分映射字典")
    missing_scores = [column for column in score_map.values() if column not in data.columns]
    if missing_scores:
        raise ValueError(f"data 中缺少评分列：{sorted(set(missing_scores))}")

    rules_in, rules_out, rules_base = _validate_rules(
        data=data,
        rules_in=rules_in,
        rules_out=rules_out,
        rules_base=rules_base,
    )
    score_weights = _normalize_score_weights(score_weights, score_map)

    if amount is not None and amount not in data.columns:
        raise ValueError(f"数据集中缺少金额字段 '{amount}'")
    if out_in_amount_col is not None and out_in_amount_col not in data.columns:
        raise ValueError(f"数据集中缺少 OUT-IN 金额字段 '{out_in_amount_col}'")
    if (out_in_amount_col is not None or out_in_amount_fill is not None) and amount is None:
        raise ValueError("使用 OUT-IN 金额回填参数时必须同时传入 amount")

    # ── 第二步：解析与计算分箱表 ─────────────────────────────────────────
    resolved_bin_params = dict(bin_params) if bin_params else {}
    resolved_bin_params.setdefault('n_jobs', n_jobs)
    resolved_bin_params.setdefault('parallel_backend', parallel_backend)
    resolved_bin_params.setdefault('parallel_config', parallel_config)
    bin_table_result = _resolve_bin_table(
        reference_data=reference_data,
        bin_table=bin_table,
        score=score,
        target=target,
        overdue=overdue,
        dpds=dpds,
        bin_method=bin_method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        missing_separate=missing_separate,
        bin_params=resolved_bin_params,
        data=data,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
    )

    # ── 第三步半：解析分析样本的实际表现标签 ──────────────────────────────
    y = _resolve_target_series(data, target, overdue, dpds)
    for table in bin_table_result.values():
        _validate_swap_bin_labels(table)
    if len(y) > 1:
        for table in bin_table_result.values():
            _extract_bad_rate_cols(table, list(y))

    # ── 第四步：构建 swap_pipeline（核心逻辑）────────────────────────────
    # 整合 ruleset_analysis（swapout）和 swapin_report（swapin）逻辑
    swap_pipeline, swap_result = _build_swap_pipeline(
        data=data,
        score_map=score_map,
        score_weights=score_weights,
        bin_table_result=bin_table_result,
        rules_base=rules_base,
        rules_out=rules_out,
        rules_in=rules_in,
        target=target,
        overdue=overdue,
        dpds=dpds,
        amount=amount,
        out_in_uplift=out_in_uplift,
        risk_uplifts=risk_uplifts,
        sample_survival_rate=sample_survival_rate,
        reverse_order=reverse_order,
        rule_analysis_mode=rule_analysis_mode,
        out_in_amount_fill=out_in_amount_fill,
        out_in_amount_col=out_in_amount_col,
        y=y,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
    )

    # ── 返回结果 ────────────────────────────────────────────────────────────
    return {
        'swap_pipeline': swap_pipeline,
        'swap_result': swap_result,
    }


def _store_splits_from_labels(tbl: pd.DataFrame) -> None:
    """从分箱标签解析切分点，存入 ``tbl._splits`` 属性。

    解析规则（基于 ``feature_bin_stats`` 的标签生成逻辑）：
    - ``[-inf, x)`` → 切分点 x
    - ``[-inf, +inf)`` → 跳过（+inf 不是有效切分）
    - ``缺失`` / ``特殊`` → 跳过
    - ``箱{i}`` → 跳过（无切分点）

    解析顺序：从左到右收集各分箱的右边界作为切分点。

    :param tbl: 标准化后的分箱表（inplace 修改，添加 _splits 属性）
    """
    import re

    # 找出分箱标签：列中找 → MultiIndex列中找 → MultiIndex行索引中找
    if '分箱标签' in tbl.columns:
        labels = tbl['分箱标签'].tolist()
    elif isinstance(tbl.columns, pd.MultiIndex):
        detail_group = _get_detail_group_name(tbl)
        bin_label_col = next(
            (c for c in tbl.columns
             if isinstance(c, tuple) and c[0] == detail_group and c[1] == '分箱标签'),
            None
        )
        labels = tbl[bin_label_col].tolist() if bin_label_col else []
    elif isinstance(tbl.index, pd.MultiIndex):
        # MultiIndex 行（金额口径场景）：分箱标签在 level=1
        labels = tbl.index.get_level_values(1).tolist()
    else:
        labels = []

    splits = []
    for lbl in labels:
        if lbl in ('missing', 'special', '合计'):
            continue
        # 格式: [x, y) 或 [x, +inf)
        m = re.search(r', *(.+?)\)', str(lbl))
        if m:
            val_str = m.group(1).strip()
            if val_str.lower() == '+inf' or val_str == '∞':
                continue  # +inf 不是有效切分点
            try:
                val = float(val_str)
                if not np.isnan(val) and not np.isinf(val):
                    splits.append(val)
            except (ValueError, TypeError):
                continue

    splits = sorted(set(splits))
    object.__setattr__(tbl, '_splits', np.array(splits) if splits else np.array([]))


def _normalize_bin_table(
    tbl: pd.DataFrame,
    label: str = '_default',
) -> pd.DataFrame:
    """标准化分箱表，确保列结构统一，并提取切分点存储到属性中。

    标准化规则：
    - 单层列：检查是否有 '分箱标签' 列，有则保留，无则添加
    - 多层列（MultiIndex）：确保顶层分组名为 '分箱详情'，子层列名统一
    - 统一添加 '分箱' 别名列（兼容旧代码）
    - 从分箱标签解析切分点，存入 ``tbl._splits`` 属性（供坏样本预测使用）

    :param tbl: 原始分箱表
    :param label: 标签名（用于单层表头的默认分组）
    :return: 标准化后的分箱表
    """
    tbl = tbl.copy()

    # ── 提取切分点 ──────────────────────────────────────────────────────────
    _store_splits_from_labels(tbl)

    if isinstance(tbl.columns, pd.MultiIndex):
        # MultiIndex 列：确保顶层分组名为 '分箱详情'
        detail_group = _get_detail_group_name(tbl)

        # 提取各标签下的坏样本率列，构建统一结构
        # 保留 '分箱详情' 公共列 + 各标签的坏样本率
        available_merge = [c for c in tbl.columns
                          if isinstance(c, tuple) and c[0] == detail_group
                          and c[1] in ['指标名称', '指标含义', '分箱标签', '样本总数', '样本占比']]
        non_merge = [c for c in tbl.columns if c not in available_merge]

        # 构建新的列结构
        new_cols = available_merge.copy()
        for col in non_merge:
            if isinstance(col, tuple) and col[0] != detail_group:
                new_cols.append(col)

        return tbl[new_cols] if new_cols else tbl

    else:
        # 单层列：检查必要列
        if '分箱标签' not in tbl.columns and '分箱' not in tbl.columns:
            # 生成分箱标签
            tbl['分箱标签'] = [f'箱{i + 1}' for i in range(len(tbl))]

        # 添加分箱别名（兼容旧代码）
        if '分箱标签' in tbl.columns and '分箱' not in tbl.columns:
            tbl['分箱'] = tbl['分箱标签']

        return tbl


def _normalize_rules_input(
    rules: Union[Rule, List[Rule], None],
) -> List[Rule]:
    """将规则参数统一规范化为 List[Rule]。

    支持传入单条 Rule、List[Rule] 或 None。

    :param rules: 规则集输入
    :return: 规范化的 List[Rule]（空列表当输入为 None 时）
    """
    if rules is None:
        return []
    if isinstance(rules, Rule):
        return [rules]
    if isinstance(rules, list):
        return rules
    raise TypeError(
        f"规则参数类型错误，期望 Rule 或 List[Rule]，实际为 {type(rules).__name__}"
    )


def _validate_rules(
    data: pd.DataFrame,
    rules_in: Union[Rule, List[Rule]],
    rules_out: Optional[Union[Rule, List[Rule]]],
    rules_base: Optional[Union[Rule, List[Rule]]],
) -> tuple:
    """校验并规范化三个规则集。

    处理逻辑：
    1. 将三个规则集统一规范化为 List[Rule]
    2. 要求至少有一个规则集非空
    3. 从所有规则中提取所需特征列
    4. 校验 data 中是否包含全部所需特征列

    :param data: 样本数据集
    :param rules_in: 置入规则集
    :param rules_out: 置出规则集（可选）
    :param rules_base: 基准拒绝规则集（可选）
    :return: (rules_in, rules_out, rules_base) 均为 List[Rule]
    :raises ValueError: 三个规则集均为空时
    :raises FeatureNotFoundError: data 缺少规则所需列时
    """
    from hscredit.core.rules import get_columns_from_query
    from hscredit.exceptions import FeatureNotFoundError

    # 统一规范化
    rules_in = _normalize_rules_input(rules_in)
    rules_out = _normalize_rules_input(rules_out) if rules_out is not None else []
    rules_base = _normalize_rules_input(rules_base) if rules_base is not None else []

    # 收集所有规则所需特征列
    all_rules = rules_in + rules_out + rules_base
    if not all_rules:
        raise ValueError("rules_in、rules_out、rules_base 至少需要提供一个非空规则集")
    if all_rules:
        required_cols: set = set()
        for rule in all_rules:
            required_cols.update(get_columns_from_query(rule.expr))

        # 校验 data 包含全部所需列
        missing = required_cols - set(data.columns)
        if missing:
            raise FeatureNotFoundError(
                f"data 中缺少规则所需的列：{sorted(missing)}，"
                f"请检查规则表达式是否引用了不存在的字段"
            )

    return rules_in, rules_out, rules_base


def _normalize_score_weights(
    score_weights: Optional[Union[float, Dict[str, float], List[float]]],
    score_map: Dict[str, str],
) -> Optional[Dict[str, float]]:
    """将 score_weights 统一规范化为 {评分名: 权重} 字典，并归一化到 [0, 1] 区间。

    支持三种输入形式：
    - 单个 float：对所有评分使用相同权重
    - Dict[str, float]：键为评分名（与 score_map 的 key 对应），覆盖对应评分权重
    - List[float]：与 score_map 的 key 按顺序一一对应

    归一化方法：将权重之和缩放，使 max(weight) = 1.0，
    即 w_normalized = w / sum(all_weights)。

    :param score_weights: 原始权重（支持单值、字典、列表）
    :param score_map: 评分映射字典 {评分名: 实际列名}
    :return: 归一化后的权重字典 {评分名: 归一化权重}，或 None（当 score_weights 为 None 时）
    :raises ValueError: 字典键不在 score_map 中，或列表长度与 score_map 不匹配时
    """
    if score_weights is None:
        return None

    score_names = list(score_map.keys())

    if isinstance(score_weights, (int, float)):
        raw_weights = {name: float(score_weights) for name in score_names}
    elif isinstance(score_weights, dict):
        if set(score_weights) != set(score_names):
            raise ValueError(
                "score_weights 与 score 的评分名必须完全一致，"
                f"score={sorted(score_names)}，score_weights={sorted(score_weights)}"
            )
        raw_weights = {name: float(score_weights[name]) for name in score_names}
    elif isinstance(score_weights, (list, tuple)):
        if len(score_weights) != len(score_names):
            raise ValueError(
                f"score_weights 列表长度 ({len(score_weights)}) "
                f"与 score_map 中的评分数量 ({len(score_names)}) 不匹配"
            )
        raw_weights = {name: float(w) for name, w in zip(score_names, score_weights)}
    else:
        raise TypeError(
            f"score_weights 参数类型错误，期望 float / Dict / List，"
            f"实际为 {type(score_weights).__name__}"
        )

    # 归一化：w_normalized = w / sum(all_weights)，使 sum = 1
    invalid = [name for name, value in raw_weights.items() if not np.isfinite(value) or value < 0]
    if invalid:
        raise ValueError(f"score_weights 必须为有限且非负数，异常评分名：{invalid}")
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("score_weights 所有权重之和必须大于 0")

    return {name: w / total for name, w in raw_weights.items()}


def _extract_bad_rate_cols(df_bin: pd.DataFrame, target_names: List[str]) -> Dict[str, object]:
    """按目标标签显式解析分箱表中的坏样本率列。"""
    if df_bin.empty:
        raise ValueError("分箱表为空，无法预测 OUT-IN 坏概率")

    if not isinstance(df_bin.columns, pd.MultiIndex):
        bad_col = next((col for col in ["坏样本率", "坏样本率(金额)"] if col in df_bin.columns), None)
        if bad_col is None:
            raise ValueError("分箱表缺少坏样本率列")
        if len(target_names) != 1:
            raise ValueError("多目标分析必须为每个目标提供对应的坏样本率列")
        return {target_names[0]: bad_col}

    result = {}
    for target_name in target_names:
        candidates = [
            col
            for col in df_bin.columns
            if isinstance(col, tuple) and col[0] == target_name and "坏样本率" in str(col[1])
        ]
        if not candidates:
            raise ValueError(f"分箱表缺少目标标签 '{target_name}' 对应的坏样本率列")
        result[target_name] = candidates[0]
    return result


def _validate_swap_bin_labels(df_bin: pd.DataFrame) -> None:
    """多箱评分分箱必须提供每箱可解析的数值区间。"""
    import re

    labels = _extract_swap_bin_labels(df_bin).astype(str)
    numeric_labels = labels[~labels.str.lower().isin(["missing", "缺失", "合计"])]
    if len(numeric_labels) <= 1:
        return
    interval_pattern = re.compile(r"^\s*[\[\(]\s*[^,]+\s*,\s*[^\]\)]+\s*[\]\)]\s*$")
    invalid = [label for label in numeric_labels if not interval_pattern.match(label)]
    if invalid:
        raise ValueError(f"多箱分箱表存在无法映射非缺失评分的分箱标签：{invalid}")


def _extract_bad_rate_col(df_bin: pd.DataFrame) -> Tuple[Optional[object], List[object]]:
    """兼容旧内部接口，返回分箱表中的坏样本率列。"""
    if not isinstance(df_bin.columns, pd.MultiIndex):
        cols = [col for col in ["坏样本率", "坏样本率(金额)"] if col in df_bin.columns]
    else:
        cols = [col for col in df_bin.columns if isinstance(col, tuple) and "坏样本率" in str(col[1])]
    return (cols[0] if len(cols) == 1 else None), cols


def _extract_swap_bin_labels(df_bin: pd.DataFrame) -> pd.Series:
    """兼容新旧列名及行索引，提取逐行分箱标签。"""
    for label_name in ["分箱标签", "分箱"]:
        if label_name in df_bin.columns:
            return pd.Series(df_bin[label_name].values, index=df_bin.index)
    if isinstance(df_bin.columns, pd.MultiIndex):
        for column in df_bin.columns:
            if isinstance(column, tuple) and column[1] in {"分箱标签", "分箱"}:
                return pd.Series(df_bin[column].values, index=df_bin.index)
    if isinstance(df_bin.index, pd.MultiIndex):
        for level in range(df_bin.index.nlevels - 1, -1, -1):
            values = df_bin.index.get_level_values(level)
            if any("inf" in str(value).lower() or str(value) in {"missing", "缺失"} for value in values):
                return pd.Series(values, index=df_bin.index)
    return pd.Series([f"箱{position + 1}" for position in range(len(df_bin))], index=df_bin.index)


def _swap_bin_fallback_rate(df_bin, bad_rates, labels, mapped_probability=None):
    """按合计、箱样本数、分析样本映射结果的顺序计算总体坏率。"""
    normalized_labels = labels.astype(str).str.lower()
    total_rows = normalized_labels == "合计"
    total_rates = bad_rates.loc[total_rows].dropna()
    if not total_rates.empty:
        return float(total_rates.iloc[0])

    valid = ~normalized_labels.isin(["missing", "缺失", "合计"])
    rates = bad_rates.loc[valid].dropna()
    count_col = None
    for column in df_bin.columns:
        column_name = column[1] if isinstance(column, tuple) else column
        if column_name == "样本总数":
            count_col = column
            break
    if count_col is not None:
        weights = pd.to_numeric(df_bin.loc[rates.index, count_col], errors="coerce").fillna(0.0)
        if float(weights.sum()) > 0:
            return float(np.average(rates, weights=weights))
    if mapped_probability is not None:
        mapped = pd.to_numeric(mapped_probability, errors="coerce").dropna()
        if not mapped.empty:
            return float(mapped.mean())
    if len(rates) == 1:
        return float(rates.iloc[0])
    return np.nan


def _compute_predicted_bad_prob(
    data: pd.DataFrame,
    score_col: str,
    df_bin: pd.DataFrame,
    single_bad_col: object,
) -> pd.Series:
    """按分箱区间映射评分坏概率，并为缺失评分提供安全回退。"""
    import re

    if df_bin.empty:
        return pd.Series(np.nan, index=data.index, dtype=float)
    if score_col not in data.columns:
        raise ValueError(f"data 中缺少评分列 '{score_col}'")
    if single_bad_col not in df_bin.columns:
        raise ValueError(f"分箱表缺少坏样本率列 '{single_bad_col}'")

    all_labels = _extract_swap_bin_labels(df_bin)
    all_bad_rates = pd.to_numeric(df_bin[single_bad_col], errors="coerce")
    invalid_rates = all_bad_rates.dropna()[
        (~np.isfinite(all_bad_rates.dropna()))
        | (all_bad_rates.dropna() < 0)
        | (all_bad_rates.dropna() > 1)
    ]
    if not invalid_rates.empty:
        raise ValueError("分箱表坏样本率必须为 0 到 1 之间的有限数值")
    keep = all_labels.astype(str) != "合计"
    labels = all_labels.loc[keep]
    bad_rates = all_bad_rates.loc[keep]
    scores = pd.to_numeric(data[score_col], errors="coerce")
    prob = pd.Series(np.nan, index=data.index, dtype=float)
    fallback_used = pd.Series(False, index=data.index)

    missing_rows = labels.astype(str).str.lower().isin(["missing", "缺失"])
    numeric_rows = ~missing_rows
    if int(numeric_rows.sum()) == 1:
        prob.loc[scores.notna()] = float(bad_rates.loc[numeric_rows].iloc[0])
    else:
        interval_pattern = re.compile(r"^\s*([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])\s*$")

        def parse_boundary(value):
            normalized = value.strip().lower().replace("∞", "inf")
            if normalized in {"-inf", "-infinity"}:
                return -np.inf
            if normalized in {"+inf", "inf", "+infinity", "infinity"}:
                return np.inf
            return float(normalized)

        for index, label in labels.loc[numeric_rows].items():
            match = interval_pattern.match(str(label))
            if not match:
                continue
            try:
                lower = parse_boundary(match.group(2))
                upper = parse_boundary(match.group(3))
            except (TypeError, ValueError):
                continue
            lower_ok = scores.ge(lower) if match.group(1) == "[" else scores.gt(lower)
            upper_ok = scores.le(upper) if match.group(4) == "]" else scores.lt(upper)
            match_mask = scores.notna() & lower_ok & upper_ok & prob.isna()
            prob.loc[match_mask] = float(bad_rates.loc[index])

    fallback_rate = _swap_bin_fallback_rate(
        df_bin,
        all_bad_rates,
        all_labels,
        mapped_probability=prob.loc[scores.notna()],
    )
    explicit_missing_rates = bad_rates.loc[missing_rows].dropna()
    if not explicit_missing_rates.empty:
        missing_rate = float(explicit_missing_rates.iloc[0])
    else:
        missing_rate = fallback_rate
        fallback_used.loc[scores.isna()] = True
    prob.loc[scores.isna()] = missing_rate

    unmapped = scores.notna() & prob.isna()
    if unmapped.any():
        raise ValueError(
            f"评分列 '{score_col}' 有 {int(unmapped.sum())} 条非缺失评分无法映射到分箱区间"
        )
    if prob.isna().any():
        raise ValueError("分箱表缺少可用于评分缺失或未映射值的总体坏样本率")
    prob.attrs["风险回退掩码"] = fallback_used
    return prob
