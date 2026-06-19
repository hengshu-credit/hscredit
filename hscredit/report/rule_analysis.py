"""规则分析模块.

提供规则集综合评估与多标签规则分析功能，以及规则置入置出分析。
"""

from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from ..core.rules import Rule
from ..core.binning import OptimalBinning
from ..core.metrics._binning import compute_bin_stats
from .mining.multi_label import MultiLabelRuleMiner
from .overdue_predictor import OverduePredictor
from .feature_analyzer import feature_bin_stats


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
            if len(score_map) == 1:
                name = list(score_map.keys())[0]
                return {name: _normalize_bin_table(bin_table, label=name)}
            else:
                return {name: _normalize_bin_table(bin_table, label=name) for name in score_map}
        elif isinstance(bin_table, dict):
            result = {}
            for name, tbl in bin_table.items():
                if isinstance(tbl, pd.DataFrame):
                    result[name] = _normalize_bin_table(tbl, label=name)
            return result

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

    result = {}
    for name, col in score_map.items():
        if col not in reference_data.columns:
            raise ValueError(f"reference_data 中缺少评分列 '{col}'")

        # 订单口径
        tbl_count = feature_bin_stats(
            reference_data, feature=col, target=target, overdue=overdue, dpds=dpds,
            amount=None, margins=True, **merged_params,
        )
        result[name] = _normalize_bin_table(tbl_count, label=name)

    return result


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
    sample_survival_rate: float,
    reverse_order: bool,
    rule_analysis_mode: str,
    out_in_amount_fill: Optional[float],
    out_in_amount_col: Optional[str],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构建简化的 swap_pipeline 和 swap_result（整合自 scorecardpipeline）。

    核心逻辑：
    1. ruleset_analysis（swapout）：评估 rules_out 置出效果
    2. swapin_report（swapin）：评估 rules_in 置入效果，基于 base 分箱预测

    :return: (swap_pipeline, swap_result)
    """
    n_total = len(data)

    # ── 计算每个样本的预测坏概率 ──────────────────────────────────────────────
    score_bad_probs = {}
    for name, df_bin in bin_table_result.items():
        score_col = score_map[name]
        single_bad_col, _ = _extract_bad_rate_col(df_bin)
        score_bad_probs[name] = _compute_predicted_bad_prob(data, score_col, df_bin, single_bad_col)

    if len(score_bad_probs) == 1:
        full_bad_probs = list(score_bad_probs.values())[0]
    else:
        if score_weights:
            weights = {name: score_weights[name] for name in score_bad_probs}
        else:
            weights = {name: 1.0 / len(score_bad_probs) for name in score_bad_probs}
        prob_sum = None
        for name, prob in score_bad_probs.items():
            prob_sum = prob * weights[name] if prob_sum is None else prob_sum + prob * weights[name]
        full_bad_probs = prob_sum

    # 全量坏样本率
    full_bad_rate = float(full_bad_probs.mean()) if len(full_bad_probs) > 0 else 0.0
    n_total_full = int(n_total / sample_survival_rate) if sample_survival_rate > 0 else n_total
    n_bad_full = full_bad_rate * n_total_full

    # ── SWAPOUT 分析（基于 ruleset_analysis）───────────────────────────────
    # 样本分为：OUT-OUT拒绝、剩余样本
    all_rows = []

    # 判断哪些区域需要显示
    has_rules_out = rules_out is not None and len(rules_out) > 0
    has_rules_in = rules_in is not None and len(rules_in) > 0
    has_rules_base = rules_base is not None and len(rules_base) > 0

    # 辅助函数：根据全量数据掩码过滤 y（y 可能是 numpy 数组或 pandas Series）
    def _filter_y(mask):
        """根据掩码过滤 y，返回过滤后的 y。

        mask: 可能是 numpy 数组或 pandas Series (布尔索引)
        """
        if y is None:
            return None
        if isinstance(y, np.ndarray):
            # numpy 数组：需要用布尔数组
            # mask 可能是 numpy 数组或 pandas Series
            if isinstance(mask, np.ndarray):
                return y[mask]
            else:
                return y[mask.values]
        else:
            # pandas Series：直接用布尔 Series 索引
            return y[mask]

    # 1. 全量样本（始终显示）
    full_n = n_total
    full_n_bad = float(full_bad_probs.sum())
    full_n_bad = max(0.0, min(full_n_bad, float(full_n)))
    all_rows.append(_make_swap_row(
        '全量样本', '', full_n, full_n_bad,
        n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
        amount=amount, amount_col=amount, data=data, y=y,
    ))

    # 2. OUT-OUT拒绝样本（rules_base）
    if has_rules_base:
        combined_base = reduce(lambda r1, r2: r1 | r2, rules_base)
        base_hit = combined_base.predict(data)
        base_n = int(base_hit.sum())
        base_n_bad = float(full_bad_probs.loc[base_hit].sum())
        base_n_bad = max(0.0, min(base_n_bad, float(base_n)))

        for rule in rules_base:
            mask = rule.predict(data)
            n_hit = int(mask.sum())
            n_bad = float(full_bad_probs.loc[mask].sum())
            n_bad = max(0.0, min(n_bad, float(n_hit)))
            # 金额口径：传入过滤后的数据和 y
            filtered_data = data[mask].copy()
            filtered_y = _filter_y(mask)
            all_rows.append(_make_swap_row(
                'OUT-OUT拒绝', rule.name, n_hit, n_bad,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                rule_detail=rule.expr, amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
            ))

        # OUT-OUT合计
        filtered_data = data[base_hit].copy()
        filtered_y = _filter_y(base_hit)
        all_rows.append(_make_swap_row(
            'OUT-OUT拒绝', '合计', base_n, base_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))

    # 计算剩余样本掩码
    if has_rules_base:
        remain_mask = ~base_hit
    else:
        remain_mask = pd.Series(True, index=data.index)

    # 将 remain_mask 转换为 DataFrame 用于 rules_out 预测
    remain_data = data[remain_mask].copy()

    # 3. 剩余样本
    # 条件：有 rules_base 时显示剩余样本（场景2, 3, 6）
    # 场景分析：
    # - 场景2 (rules_base): 全量 → OUT-OUT拒绝 → 剩余样本
    # - 场景3 (rules_base + rules_in): 全量 → OUT-OUT拒绝 → 剩余样本 → OUT-IN置入 → ALL-IN
    # - 场景6 (全部): 全量 → OUT-OUT拒绝 → 剩余样本 → IN-OUT置出 → IN-IN通过 → OUT-IN置入 → ALL-IN
    if has_rules_base:
        remain_n = int(remain_mask.sum())
        remain_n_bad = float(full_bad_probs.loc[remain_mask].sum())
        remain_n_bad = max(0.0, min(remain_n_bad, float(remain_n)))
        # 金额口径：传入过滤后的数据和 y
        filtered_data = data[remain_mask].copy()
        filtered_y = _filter_y(remain_mask)
        all_rows.append(_make_swap_row(
            '剩余样本', '', remain_n, remain_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))

    # 4. IN-OUT置出样本（rules_out）
    # 注意：IN-OUT 在 remain_data 范围内计算
    if has_rules_out:
        combined_out = reduce(lambda r1, r2: r1 | r2, rules_out)
        out_hit = combined_out.predict(remain_data)

        for rule in rules_out:
            mask = rule.predict(remain_data)
            # 构建全量数据上的掩码：remain_mask AND mask
            full_mask_indices = remain_data.index[mask.values]
            full_mask = data.index.isin(full_mask_indices)
            n_hit = int(mask.sum())
            n_bad = float(full_bad_probs.loc[full_mask].sum())
            n_bad = max(0.0, min(n_bad, float(n_hit)))
            # 金额口径：传入过滤后的数据和 y
            filtered_data = remain_data[mask].copy()
            filtered_y = _filter_y(full_mask)
            all_rows.append(_make_swap_row(
                'IN-OUT置出', rule.name, n_hit, n_bad,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                rule_detail=rule.expr, amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
            ))

        # IN-OUT合计
        out_n = int(out_hit.sum())
        out_hit_indices = remain_data.index[out_hit.values]
        out_hit_full_mask = data.index.isin(out_hit_indices)
        out_n_bad = float(full_bad_probs.loc[out_hit_full_mask].sum())
        out_n_bad = max(0.0, min(out_n_bad, float(out_n)))
        # 金额口径：传入过滤后的数据和 y
        filtered_data = remain_data[out_hit].copy()
        filtered_y = _filter_y(out_hit_full_mask)
        all_rows.append(_make_swap_row(
            'IN-OUT置出', '合计', out_n, out_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))

        # 计算 IN-IN 通过样本掩码：剩余样本中未被 IN-OUT 拒绝的样本
        # 注意：out_hit 有 remain_data 的索引，需要映射回 data 的索引
        outin_not_hit_indices = remain_data.index[~out_hit.values]
        inin_mask = pd.Series(data.index.isin(outin_not_hit_indices), index=data.index)
    elif has_rules_base:
        # 无 rules_out 但有 rules_base 时，IN-IN = 剩余样本
        inin_mask = remain_mask
    else:
        # 无 rules_out 且无 rules_base 时，IN-IN = 全量
        inin_mask = pd.Series(True, index=data.index)

    # 5. IN-IN通过样本
    # 条件：只有有 rules_out 时才显示 IN-IN（场景4, 5, 6）
    if has_rules_out:
        inin_n = int(inin_mask.sum())
        inin_n_bad = float(full_bad_probs.loc[inin_mask].sum())
        inin_n_bad = max(0.0, min(inin_n_bad, float(inin_n)))
        # 金额口径：传入过滤后的数据和 y
        filtered_data = data[inin_mask].copy()
        filtered_y = _filter_y(inin_mask)
        all_rows.append(_make_swap_row(
            'IN-IN通过', '', inin_n, inin_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))
    else:
        # 只有 rules_in 或只有 rules_base：计算 IN-IN 用于 ALL-IN
        inin_n = int(inin_mask.sum())
        inin_n_bad = float(full_bad_probs.loc[inin_mask].sum())
        inin_n_bad = max(0.0, min(inin_n_bad, float(inin_n)))

    # 6. OUT-IN置入样本（rules_in）
    # OUT-IN：在 rules_base 拒绝范围外的样本（即 remain_mask）中，满足 rules_in 的样本
    # 注意：OUT-IN 行显示预测坏样本数（无 uplift），只在 ALL-IN 阶段应用一次 uplift
    if has_rules_in:
        combined_in = reduce(lambda r1, r2: r1 | r2, rules_in)
        # OUT-IN = remain_mask AND rules_in
        outin_mask = remain_mask & combined_in.predict(data)

        for rule in rules_in:
            mask = rule.predict(data)
            # 单条规则的 OUT-IN：remain_mask AND mask
            single_outin_mask = remain_mask & mask
            n_hit = int(single_outin_mask.sum())
            # OUT-IN 显示预测坏样本数（无 uplift）
            n_bad = float(full_bad_probs.loc[single_outin_mask].sum())
            n_bad = max(0.0, min(n_bad, float(n_hit)))
            # 金额口径：传入过滤后的数据和 y
            filtered_data = data[single_outin_mask].copy()
            filtered_y = _filter_y(single_outin_mask)
            all_rows.append(_make_swap_row(
                'OUT-IN置入', rule.name, n_hit, n_bad,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                rule_detail=rule.expr, amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
            ))

        # OUT-IN合计
        outin_total_n = int(outin_mask.sum())
        # OUT-IN 预测坏样本总数（无 uplift）
        outin_total_n_bad = float(full_bad_probs.loc[outin_mask].sum())
        outin_total_n_bad = max(0.0, min(outin_total_n_bad, float(outin_total_n)))
        # 金额口径：传入过滤后的数据和 y
        filtered_data = data[outin_mask].copy()
        filtered_y = _filter_y(outin_mask)
        all_rows.append(_make_swap_row(
            'OUT-IN置入', '合计', outin_total_n, outin_total_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))

        # 7. ALL-IN置换样本
        # 当有 rules_in 时显示 ALL-IN
        # ALL-IN = IN-IN（不包含被OUT-OUT拒绝的样本）+ OUT-IN（应用一次 uplift）
        all_in_n = inin_n + outin_total_n
        # 只在 ALL-IN 阶段应用一次 uplift
        all_in_n_bad = inin_n_bad + outin_total_n_bad * out_in_uplift
        all_in_n_bad = max(0.0, min(all_in_n_bad, float(all_in_n)))
        # 金额口径：IN-IN + OUT-IN（无 uplift，因为 uplift 只影响预测，不影响实际金额）
        all_in_mask = inin_mask | outin_mask
        filtered_data = data[all_in_mask].copy()
        filtered_y = _filter_y(all_in_mask)
        all_rows.append(_make_swap_row(
            'ALL-IN置换', '', all_in_n, all_in_n_bad,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            amount=amount, amount_col=amount, data=filtered_data, y=filtered_y,
        ))

    # ── 构建 swap_pipeline ──────────────────────────────────────────────────
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    pipeline_df = pd.DataFrame(all_rows)

    # 判断是否为金额口径模式：检查是否有金额总数列（由 _make_swap_row 在金额口径下写入）
    has_amount = (
        '金额总数' in pipeline_df.columns
        and pipeline_df['金额总数'].notna().any()
        and len(pipeline_df) > 0
    )

    # 先计算通过率(绝对值)和通过率，金额口径用金额，订单口径用样本数
    if has_amount:
        # 金额口径：各行金额 / 全量金额 = 通过率(绝对值)，直接是 0~100 的百分比数值
        amount_total_full = float(pipeline_df['金额总数'].iloc[0])
        pipeline_df['通过率(绝对值)'] = (
            pipeline_df['金额总数'] / amount_total_full * 100.0 if amount_total_full > 0
            else 0.0
        )
        pipeline_df['通过率'] = pipeline_df['通过率(绝对值)']
        # 金额口径下"样本占比"的语义改为金额占比
        pipeline_df['样本占比'] = pipeline_df['通过率(绝对值)'] / 100.0
        # 通过率(相对值)：金额口径下父行金额未知，用 1.0 填充
        pipeline_df['通过率(相对值)'] = 1.0
        # 好/坏样本数在金额口径下已是金额数，无需重算
    else:
        # 订单口径：各行样本数 / 全量样本数 = 通过率(绝对值)
        pipeline_df['通过率(绝对值)'] = pipeline_df['样本总数'] / n_total_full * 100.0
        pipeline_df['通过率'] = pipeline_df['通过率(绝对值)']
        pipeline_df['样本占比'] = pipeline_df['样本总数'] / n_total
        pipeline_df['通过率(相对值)'] = pipeline_df['样本占比'] / (n_total / n_total_full)

    # 计算通过率变化（基于原始比例 diff，再乘以 100 转换为百分点）
    rate_abs_ratio = pipeline_df['通过率(绝对值)'] / 100.0
    pipeline_df['通过率变化'] = rate_abs_ratio.diff() * 100.0
    pipeline_df.loc[pipeline_df.index[0], '通过率变化'] = pipeline_df.loc[pipeline_df.index[0], '通过率(绝对值)']

    # 计算LIFT值
    pipeline_df['LIFT值'] = pipeline_df.apply(
        lambda r: r['坏样本率'] / full_bad_rate if full_bad_rate > 0 else 0.0, axis=1
    )

    # 填充其他指标（仅在订单口径下基于样本数计算）
    if not has_amount:
        pipeline_df['好样本数'] = pipeline_df['样本总数'] - pipeline_df['坏样本数']
        pipeline_df['好样本占比'] = 1 - pipeline_df['坏样本率']
        pipeline_df['坏样本占比'] = pipeline_df['坏样本率']

    # 计算坏账改善和风险拒绝比
    pipeline_df['坏账改善'] = pipeline_df.apply(
        lambda r: (full_bad_rate - r['坏样本率']) / full_bad_rate if full_bad_rate > 0 else 0.0, axis=1
    )
    pipeline_df['风险拒绝比'] = pipeline_df.apply(
        lambda r: r['坏账改善'] / r['样本占比'] if r['样本占比'] > 0 else 0.0, axis=1
    )

    # 逆序处理
    if reverse_order:
        pipeline_df = pipeline_df.iloc[::-1].reset_index(drop=True)

    # 按照 rule.report 的指标顺序调整列顺序
    # 参考: 指标名称, 规则详情, 分箱, 样本总数, 样本占比, 好样本数, 好样本占比, 坏样本数, 坏样本占比, 坏样本率, LIFT值, 坏账改善
    # 通过率相关列在最后，顺序为: 通过率 → 通过率(绝对值) → 通过率(相对值) → 通过率变化
    col_order = [
        '规则分类', '指标名称', '规则详情',
        '样本总数', '样本占比',
        '好样本数', '好样本占比',
        '坏样本数', '坏样本占比', '坏样本率',
        'LIFT值', '坏账改善', '风险拒绝比',
        '通过率', '通过率(绝对值)', '通过率(相对值)', '通过率变化',
    ]
    # 只保留存在的列
    existing_cols = [c for c in col_order if c in pipeline_df.columns]
    other_cols = [c for c in pipeline_df.columns if c not in col_order]
    pipeline_df = pipeline_df[existing_cols + other_cols]

    # ── 构建 swap_result ──────────────────────────────────────────────────
    # 提取关键指标
    inin_row = pipeline_df[pipeline_df['规则分类'] == 'IN-IN通过']
    outin_sum_row = pipeline_df[pipeline_df['规则分类'] == 'OUT-IN置入']

    if not inin_row.empty:
        pass_rate_before = float(inin_row.iloc[0]['通过率(绝对值)'])
        bad_rate_before = float(inin_row.iloc[0]['坏样本率'])
    else:
        pass_rate_before = 1.0
        bad_rate_before = full_bad_rate

    if not outin_sum_row.empty:
        n_outin = int(outin_sum_row.iloc[0]['样本总数'])
        bad_rate_outin = float(outin_sum_row.iloc[0]['坏样本率'])
        pass_rate_after = pass_rate_before + float(outin_sum_row.iloc[0]['通过率变化'])
        bad_rate_after = (inin_n * bad_rate_before + n_outin * bad_rate_outin) / (inin_n + n_outin) if (inin_n + n_outin) > 0 else bad_rate_before
    else:
        pass_rate_after = pass_rate_before
        bad_rate_after = bad_rate_before

    swap_result_rows = [
        {'指标': '通过率', '变化前': pass_rate_before, '变化后': pass_rate_after,
         '绝对变化': pass_rate_after - pass_rate_before,
         '相对变化': (pass_rate_after - pass_rate_before) / max(pass_rate_before, 1e-9)},
        {'指标': '逾期率', '变化前': bad_rate_before, '变化后': bad_rate_after,
         '绝对变化': bad_rate_after - bad_rate_before,
         '相对变化': (bad_rate_after - bad_rate_before) / max(bad_rate_before, 1e-9)},
        {'指标': '风险上浮系数', '变化前': 1.0, '变化后': out_in_uplift,
         '绝对变化': out_in_uplift - 1.0, '相对变化': out_in_uplift - 1.0},
        {'指标': '样本集幸存比例', '变化前': sample_survival_rate, '变化后': sample_survival_rate,
         '绝对变化': 0.0, '相对变化': 0.0},
    ]
    swap_result = pd.DataFrame(swap_result_rows)

    return pipeline_df, swap_result


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
    n_bad = max(0.0, min(n_bad, float(n_samples)))

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
            n_good = max(0, n_samples - int(round(n_bad)))

        row = {
            '规则分类': rule_class,
            '指标名称': rule_name,
            '规则详情': rule_detail,
            '样本总数': n_samples,
            '好样本数': n_good,
            '坏样本数': int(round(n_bad)),
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
    """
    datasets = datasets.copy()

    feature_names_missing = set([f for rule in rules for f in rule.feature_names_in_]) - set(datasets.columns)
    if len(feature_names_missing) > 0:
        raise ValueError(f"数据集字段缺少以下字段: {feature_names_missing}")

    report = pd.DataFrame()
    all_rules = reduce(lambda r1, r2: r1 | r2, rules)

    table_total = all_rules.report(
        datasets,
        target=target,
        overdue=overdue,
        dpds=dpds,
        filter_cols=filter_cols,
        margins=True,
        amount=amount,
        **kwargs,
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
        table = rule.report(
            datasets,
            target=target,
            overdue=overdue,
            dpds=dpds,
            filter_cols=filter_cols,
            margins=False,
            amount=amount,
            **kwargs,
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
        datasets = datasets[~rule.predict(datasets)]

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
) -> str:
    """多标签规则分析（Excel 输出）.

    报告包含：
    - 规则汇总：各规则在每个标签下的覆盖率/坏率/LIFT/有效性分类
    - 有效性矩阵：行=规则，列=标签，格=LIFT值
    - 规则分类统计：按规则类型分组的汇总统计

    :param df: 输入数据 DataFrame
    :param features: 参与挖掘的特征列表
    :param labels: 标签映射 {中文名: 列名}
    :param miner_params: 传递给 MultiLabelRuleMiner 的额外参数
    :param output_path: 输出 Excel 文件路径
    :return: 输出文件路径
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
) -> Dict[str, pd.DataFrame]:
    """规则置入置出（Swap）分析。

    整合自 scorecardpipeline 的 ``swapin_report`` 和 ``ruleset_analysis``（即 swapout_report），
    只输出 ``swap_pipeline`` 和 ``swap_result``，支持金额和订单口径。

    **四象限定义**

    ==========  ==========  ====================================
    象限        含义        风险说明
    ==========  ==========  ====================================
    in_in      模型通过 & 规则通过   基准客群，最终放款
    in_out     模型通过 & 规则拒绝   置出样本，误拒损失
    out_in     模型拒绝 & 规则通过   置入样本，核心风险敞口
    out_out    模型拒绝 & 规则拒绝   仍拒绝，无影响
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
    :param amount: 金额字段（可选），传入后同时输出金额口径报告
    :param sample_survival_rate: 样本集幸存比例，默认 1.0
    :param reverse_order: 是否逆序展示（True: 从置入效果开始展示）
    :param out_in_amount_fill: out_in 置入样本额度填充定值（可选）
    :param out_in_amount_col: out_in 置入样本额度填充字段名（可选）
    :param bin_method: 分箱方法，默认 'quantile'（仅 reference_data 模式生效）
    :param max_n_bins: 最大分箱数，默认 10（仅 reference_data 模式生效）
    :param min_bin_size: 每箱最小样本占比，默认 0.05（仅 reference_data 模式生效）
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param bin_params: 额外分箱参数 dict，会透传给 ``feature_bin_stats``
    :param rule_analysis_mode: 规则分析模式，默认 'independent'。
        - 'independent'：每条规则独立应用到全量 data，分别统计命中好坏分布。
        - 'sequential'：漏斗模式，每条规则在前一条拒绝后的剩余样本上分析。
    :return: 包含两张表的字典

        - ``swap_pipeline``：分步骤通过率与逾期率变化（可逆序），支持订单/金额双口径
        - ``swap_result``：置换前后对比与业务增益

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
    # ── 第一步：解析与计算分箱表 ─────────────────────────────────────────
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
        bin_params=bin_params,
        data=data,
    )

    # ── 第二步：规则集预处理 ─────────────────────────────────────────────
    if isinstance(score, str):
        score_map = {'_default': score}
    else:
        score_map = score

    rules_in, rules_out, rules_base = _validate_rules(
        data=data,
        rules_in=rules_in,
        rules_out=rules_out,
        rules_base=rules_base,
    )

    # ── 第三步：权重归一化 ───────────────────────────────────────────────
    score_weights = _normalize_score_weights(score_weights, score_map)

    # ── 第三步半：计算目标变量 y（用于金额口径计算）──────────────────────────
    y = None
    if amount is not None and amount in data.columns:
        # 计算 y：目标变量（0/1）
        if target is not None and target in data.columns:
            y = data[target].values
        elif overdue is not None and dpds is not None:
            # 多逾期标签场景：只支持单逾期单DPD
            if isinstance(overdue, list):
                overdue_col = overdue[0] if len(overdue) > 0 else None
            else:
                overdue_col = overdue
            if isinstance(dpds, list):
                dpd_val = dpds[0] if len(dpds) > 0 else 0
            else:
                dpd_val = dpds
            if overdue_col is not None and overdue_col in data.columns:
                y = (data[overdue_col] > dpd_val).astype(int).values

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
        sample_survival_rate=sample_survival_rate,
        reverse_order=reverse_order,
        rule_analysis_mode=rule_analysis_mode,
        out_in_amount_fill=out_in_amount_fill,
        out_in_amount_col=out_in_amount_col,
        y=y,
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
    tbl._splits = np.array(splits) if splits else np.array([])


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
        # 校验键
        unknown = set(score_weights.keys()) - set(score_names)
        if unknown:
            raise ValueError(
                f"score_weights 字典中包含不在 score_map 中的评分名：{sorted(unknown)}，"
                f"有效评分名：{score_names}"
            )
        raw_weights = {name: float(score_weights.get(name, 0.0)) for name in score_names}
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
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("score_weights 所有权重之和必须大于 0")

    return {name: w / total for name, w in raw_weights.items()}


def _extract_bad_rate_col(
    df_bin: pd.DataFrame,
) -> Tuple[Optional[str], List[str]]:
    """从分箱表中提取坏样本率列名。

    处理多种列结构：
    - 单层列：直接查找 '坏样本率' 或带金额后缀的 '坏样本率(金额)'
    - MultiIndex 列：查找各标签下的坏样本率列

    :param df_bin: 单个评分的分箱表（标准化后）
    :return: (单一坏样本率列名或None, 所有坏样本率列名列表)
    """
    if df_bin.empty:
        return None, []

    # 方案A：单层列
    if not isinstance(df_bin.columns, pd.MultiIndex):
        # 优先找 '坏样本率'，其次 '坏样本率(金额)'
        for col in ['坏样本率', '坏样本率(金额)']:
            if col in df_bin.columns:
                return col, [col]
        return None, []

    # 方案B：MultiIndex 列（多标签场景）
    # 顶层分组：标签名 + '分箱详情'
    level0 = df_bin.columns.get_level_values(0)
    label_names = [l for l in level0 if l != '分箱详情']
    bad_rate_cols = []
    for label in label_names:
        for col in df_bin.columns:
            if isinstance(col, tuple) and col[0] == label and '坏样本率' in col[1]:
                bad_rate_cols.append(col)
                break

    if len(bad_rate_cols) == 1:
        return bad_rate_cols[0], bad_rate_cols
    return None, bad_rate_cols


def _compute_predicted_bad_prob(
    data: pd.DataFrame,
    score_col: str,
    df_bin: pd.DataFrame,
    single_bad_col: Optional[str],
) -> pd.Series:
    """根据分箱表计算每个样本的预测坏概率。

    :param data: 数据集
    :param score_col: 评分列名
    :param df_bin: 该评分的分箱表（已标准化，合计行已移除）
    :param single_bad_col: 单一坏样本率列名
    :return: 每行样本的预测坏概率（0~1）
    """
    if df_bin.empty:
        return pd.Series(0.0, index=data.index)

    # 提取切分点（由 _store_splits_from_labels 解析 bin 标签得到）
    splits_arr: np.ndarray = getattr(df_bin, '_splits', np.array([]))
    if splits_arr is None or len(splits_arr) == 0:
        # 回退：从分箱标签解析（兜底）
        import re as _re
        labels = None
        if '分箱标签' in df_bin.columns:
            labels = df_bin['分箱标签'].tolist()
        elif isinstance(df_bin.index, pd.MultiIndex):
            # MultiIndex 行 (amount case): 分箱标签在 level=1
            labels = df_bin.index.get_level_values(1).tolist()
        labels = labels or []

        _splits_list = []
        for lbl in labels:
            if lbl in ('missing', 'special', '合计'):
                continue
            m = _re.search(r', *(.+?)\)', str(lbl))
            if m:
                val_str = m.group(1).strip()
                if val_str.lower() not in ('+inf', '∞'):
                    try:
                        v = float(val_str)
                        if not np.isnan(v) and not np.isinf(v):
                            _splits_list.append(v)
                    except (ValueError, TypeError):
                        pass
        splits_arr = np.array(sorted(set(_splits_list))) if _splits_list else np.array([])

    if len(splits_arr) == 0:
        return pd.Series(0.0, index=data.index)

    scores = data[score_col].values.copy()
    missing_mask = pd.isna(scores)
    bins = np.digitize(scores, splits_arr, right=False)
    bins = bins.astype(float)
    bins[missing_mask] = -1

    # 构建 bin → bad_rate 映射（按行位置）
    df_valid = df_bin.copy()

    # 过滤合计行（分箱标签可能在列中，也可能在 MultiIndex 行中）
    try:
        if '分箱标签' in df_valid.columns:
            df_valid = df_valid[df_valid['分箱标签'] != '合计']
        elif isinstance(df_valid.index, pd.MultiIndex):
            df_valid = df_valid[df_valid.index.get_level_values(1) != '合计']
    except KeyError:
        # '分箱标签' 不在列中（可能被 MultiIndex 列或其他结构占用），跳过过滤
        pass

    n_bins = len(df_valid)
    if single_bad_col and single_bad_col in df_valid.columns:
        bad_rates = df_valid[single_bad_col].values
    else:
        bad_rates = df_valid.iloc[:, 0].values

    # bins 取值范围 [0, n_bins-1]，超出范围的 clamp
    bins_clipped = np.clip(bins.astype(int), 0, n_bins - 1)

    prob = pd.Series(bad_rates[bins_clipped], index=data.index)
    prob.iloc[missing_mask] = np.nan
    return prob
