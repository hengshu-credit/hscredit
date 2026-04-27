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

    # 找出可合并的列（merge columns）
    available_merge = [c for c in base_table.columns
                      if isinstance(c, tuple) and c[0] == detail_group and c[1] in ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]
                      or isinstance(c, str) and c in ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]]
    non_merge = [c for c in base_table.columns if c not in available_merge]

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
    df: pd.DataFrame,
    score: Union[str, Dict[str, str]],
    rules_in: List[Rule],
    rules_out: Optional[List[Rule]] = None,
    rules_base: Optional[List[Rule]] = None,
    bin_table: Optional[pd.DataFrame] = None,
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
) -> Dict[str, pd.DataFrame]:
    """规则置入置出（Swap）分析.

    在已有模型评分体系上，新增/替换规则后，评估样本在各象限之间的流转情况。
    核心关注 Out-In（置入）带来的风险敞口变化，以及 In-Out（置出）带来的业务损失。

    **四象限定义**

    ==========  ========== ====================================
    象限        含义       风险说明
    ==========  ========== ====================================
    in_in      模型通过 & 规则通过  最终放款，基准客群
    in_out     模型通过 & 规则拒绝  规则置出，误拒损失
    out_in     模型拒绝 & 规则通过  规则置入，核心风险敞口
    out_out    模型拒绝 & 规则拒绝  仍拒绝，无影响
    ==========  ========== ====================================

    **输入说明**

    - ``rules_in``: 置入规则集（生产规则通过后，标记数据集中哪些样本为 out_in）
    - ``rules_out``: 置出规则集（可选，标记 in_out 样本；若不传则不分析 in_out）
    - ``rules_base``: 基准拒绝规则集（可选，标记 out_out 样本；若不传则不含 out_out）
    - ``score``: 模型评分字段，支持单模型（str）或多模型加权（Dict[str, str]）
      - Dict 格式为 ``{评分名: 字段路径}``，如 ``{'model_a': 'score_a', 'model_b': 'score_b'}``
    - ``score_weights``: 多模型权重（与 score key 对应），用于综合预估风险
    - ``bin_table``: 现成分箱表，若不传则从 df 自动生成分箱表
    - ``target`` / ``overdue+dpds``: 目标变量配置（与 bin_table 二选一）
    - ``amount``: 金额字段，支持后增加金额口径分析
    - ``sample_survival_rate``: 样本集幸存比例，默认 1.0（100%%）；若传 0.5，
      则表示本数据集仅为全量样本集的 50%%，通过率从 50%% 开始计算
    - ``out_in_uplift``: 置入风险上浮系数，默认 2.0；仅对 out_in 样本生效
    - ``out_in_amount_fill``: out_in 置入样本的额度填充定值；当客户无额度时使用该值填充
    - ``out_in_amount_col``: out_in 置入样本的额度填充字段名（从 df 中取每客户额度）；
      优先级：先取 ``out_in_amount_col``，再取 ``out_in_amount_fill``，都无则按 0 处理

    :param df: 全量样本集（包含 score 列 + rules_in/rules_out/rules_base 用到的所有特征列）
    :param score: 评分字段名（str）或多评分映射（Dict）
    :param rules_in: 置入规则集（List[Rule]）
    :param rules_out: 置出规则集（可选）
    :param rules_base: 基准拒绝规则集（可选）
    :param bin_table: 现成分箱表（可选）
    :param target: 目标变量名（与 bin_table 二选一）
    :param overdue: 逾期天数字段（多标签场景）
    :param dpds: 逾期天数阈值
    :param score_weights: 多模型权重（可选）
    :param out_in_uplift: 置入风险上浮系数，默认 2.0
    :param amount: 金额字段（可选）
    :param sample_survival_rate: 样本集幸存比例，默认 1.0
    :param reverse_order: 是否逆序展示（True: 从置入效果开始展示）
    :param out_in_amount_fill: out_in 置入样本额度填充定值（可选）
    :param out_in_amount_col: out_in 置入样本额度填充字段名（可选，优先级高于 out_in_amount_fill）
    :return: 包含三张表的字典

        - ``swap_summary``：四象限样本汇总
        - ``swap_pipeline``：分步骤通过率与逾期率变化（可逆序）
        - ``swap_result``：置换前后对比与业务增益

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report.rule_analysis import rule_swap_analysis
    >>>
    >>> # 置入规则：score_a > 350
    >>> rule_in = Rule("score_a > 350", name="新客提额规则")
    >>>
    >>> # 基准拒绝规则（out_out）
    >>> rule_base = Rule("score_a <= 350", name="基准拒绝规则")
    >>>
    >>> result = rule_swap_analysis(
    ...     df=data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     rules_base=[rule_base],
    ...     target='target',
    ...     bin_method='quantile',
    ...     max_n_bins=10,
    ...     out_in_uplift=2.0,
    ... )
    >>>
    >>> print(result['swap_summary'])      # 四象限样本汇总
    >>> print(result['swap_pipeline'])      # 分步骤通过率与逾期率
    >>> print(result['swap_result'])        # 业务增益
    """
    # ------------------------------------------------------------------
    # 0. 参数预处理与输入验证
    # ------------------------------------------------------------------
    df = df.copy()

    # 统一 score 参数为 Dict 格式
    if isinstance(score, str):
        score_map = {'_default': score}
    else:
        score_map = score

    # 验证所有评分列存在
    for name, col in score_map.items():
        if col not in df.columns:
            raise ValueError(f"评分列 '{col}' 不在数据集中")

    # 验证规则所需特征列存在
    for rules in [rules_in, rules_out or [], rules_base or []]:
        for rule in rules:
            missing = set(rule.feature_names_in_) - set(df.columns)
            if missing:
                raise ValueError(f"数据集缺少以下规则特征列: {missing}")

    # 解析多模型权重
    if score_weights is None:
        score_weights = {k: 1.0 for k in score_map}
    else:
        score_weights = {k: score_weights.get(k, 1.0) for k in score_map}

    total_weight = sum(score_weights.values())
    if total_weight <= 0:
        raise ValueError("score_weights 总和必须大于 0")

    # ------------------------------------------------------------------
    # 1. 分类样本象限
    # ------------------------------------------------------------------
    # 1.1 标记模型通过（所有评分均达到阈值，取各评分中位数以上为通过）
    score_pass_mask = pd.Series(True, index=df.index)
    for name, col in score_map.items():
        median_val = df[col].median()
        score_pass_mask = score_pass_mask & (df[col] >= median_val)

    # 1.2 标记规则通过（rules_in 命中即通过；rules_out 命中即拒绝）
    rule_in_hit = pd.Series(False, index=df.index)
    for r in rules_in:
        rule_in_hit = rule_in_hit | r.predict(df)

    # 1.3 标记 rules_out（置出）
    rule_out_hit = pd.Series(False, index=df.index)
    if rules_out:
        for r in rules_out:
            rule_out_hit = rule_out_hit | r.predict(df)

    # 1.4 标记 rules_base（out_out 基准拒绝）
    out_out_mask = pd.Series(False, index=df.index)
    if rules_base:
        for r in rules_base:
            out_out_mask = out_out_mask | r.predict(df)

    # 1.5 构建四象限
    model_pass = score_pass_mask.values  # already boolean
    rule_pass = rule_in_hit.values & (~rule_out_hit.values)

    conditions = [
        model_pass & rule_pass,       # in_in: 模型通过 + 规则通过
        model_pass & ~rule_pass,      # in_out: 模型通过 + 规则拒绝
        ~model_pass & rule_pass,       # out_in: 模型拒绝 + 规则通过
    ]
    choices = ['in_in', 'in_out', 'out_in']
    df['_swap_quadrant'] = np.select(conditions, choices, default='out_out')

    # ------------------------------------------------------------------
    # 2. 构建多模型加权综合评分（归一化到 [0,1]）
    # ------------------------------------------------------------------
    df['_swap_score_combined'] = 0.0
    for name, col in score_map.items():
        w = score_weights[name] / total_weight
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f'_tmp_norm_{name}'] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f'_tmp_norm_{name}'] = 0.5
        df['_swap_score_combined'] += w * df[f'_tmp_norm_{name}']

    # ------------------------------------------------------------------
    # 3. 构建评分→逾期率映射（分箱表或自动生成）
    # ------------------------------------------------------------------
    if bin_table is not None:
        predictor = OverduePredictor(feature='_swap_score_combined')
        predictor.fit(bin_table)
    else:
        if target is None and (overdue is None or dpds is None):
            raise ValueError("未传入 bin_table 时，必须传入 target 或 overdue+dpds 参数")
        predictor = OverduePredictor(
            feature='_swap_score_combined',
            target=target,
            overdue=overdue,
            dpds=dpds,
            method='quantile',
            max_n_bins=10,
            missing_separate=True,
        )
        predictor.fit(df)

    # ------------------------------------------------------------------
    # 4. 计算各象限逾期率
    # ------------------------------------------------------------------
    def _calc_bad_rate(q_df: pd.DataFrame, amount_col: Optional[str] = None) -> float:
        """计算象限的坏样本率（订单口径或金额口径）。"""
        if q_df.empty:
            return 0.0
        if target is not None:
            col = target
            if amount_col is not None and amount_col in q_df.columns:
                y = q_df[col]
                amt = q_df[amount_col]
                amt_sum = float(amt.sum())
                # out_in 象限：额度为 0 时使用填充值
                if amt_sum == 0.0 and (q_df['_swap_quadrant'] == 'out_in').any():
                    if out_in_amount_col and out_in_amount_col in q_df.columns:
                        amt = q_df[out_in_amount_col]
                    elif out_in_amount_fill is not None:
                        amt = pd.Series(out_in_amount_fill, index=q_df.index)
                if float(amt.sum()) > 0:
                    return float((y * amt).sum() / amt.sum())
            if col in q_df.columns:
                return float(q_df[col].mean())
        elif overdue is not None:
            mob_col = overdue[0] if isinstance(overdue, list) else overdue
            dpd_val = dpds[0] if isinstance(dpds, list) else dpds
            if mob_col in q_df.columns:
                y = (q_df[mob_col] > dpd_val).astype(float)
                if amount_col is not None and amount_col in q_df.columns:
                    amt = q_df[amount_col]
                    amt_sum = float(amt.sum())
                    # out_in 象限：额度为 0 时使用填充值
                    if amt_sum == 0.0 and (q_df['_swap_quadrant'] == 'out_in').any():
                        if out_in_amount_col and out_in_amount_col in q_df.columns:
                            amt = q_df[out_in_amount_col]
                        elif out_in_amount_fill is not None:
                            amt = pd.Series(out_in_amount_fill, index=q_df.index)
                    if float(amt.sum()) > 0:
                        return float((y * amt).sum() / amt.sum())
                return float(y.mean())
        return 0.0

    # 基准坏样本率：使用 IN-IN 象限的实际坏样本率作为基准
    _overall_bad_rate = _calc_bad_rate(df[df['_swap_quadrant'] == 'in_in'], amount)
    if _overall_bad_rate <= 0:
        _overall_bad_rate = df[target].mean() if target in df.columns else 0.05

    # ------------------------------------------------------------------
    # 5. 获取预测逾期率（使用 OverduePredictor）
    # ------------------------------------------------------------------
    _pred_df = predictor.predict(df[['_swap_score_combined']])
    if isinstance(_pred_df, dict):
        _pred_df = _pred_df.get('_default', pd.Series(0.0, index=df.index))
    _pred_df = pd.Series(_pred_df, index=df.index)

    _overall_pred_bad = float(_pred_df.mean()) if len(_pred_df) > 0 else _overall_bad_rate
    if _overall_pred_bad <= 0:
        _overall_pred_bad = _overall_bad_rate

    # ------------------------------------------------------------------
    # 6. 构建 Swap Summary 表
    # ------------------------------------------------------------------
    quadrants = ['in_in', 'in_out', 'out_in', 'out_out']
    if reverse_order:
        quadrants = ['out_out', 'out_in', 'in_out', 'in_in']

    summary_rows = []
    for q in quadrants:
        q_mask = df['_swap_quadrant'] == q
        q_df = df[q_mask]
        n = q_mask.sum()
        if n == 0:
            continue

        bad_rate = _calc_bad_rate(q_df, amount)
        if q == 'out_in':
            bad_rate = min(bad_rate * out_in_uplift, 1.0)

        lift = bad_rate / _overall_bad_rate if _overall_bad_rate > 0 else 0.0
        n_bad = int(n * bad_rate)
        n_good = n - n_bad
        good_rate = 1 - bad_rate

        if amount is not None and amount in q_df.columns:
            amt = q_df[amount]
            amt_sum = float(amt.sum())
            # out_in 象限：额度为 0 时使用填充值
            if amt_sum == 0.0 and q == 'out_in':
                if out_in_amount_col and out_in_amount_col in q_df.columns:
                    amt_sum = float(q_df[out_in_amount_col].sum())
                elif out_in_amount_fill is not None:
                    amt_sum = float(out_in_amount_fill * n)
            amt_total = amt_sum
            amt_bad = float(amt_total * bad_rate)
        else:
            amt_total = None
            amt_bad = None

        summary_rows.append({
            '象限': q,
            '样本数': int(n),
            '样本占比': float(n / len(df)),
            '好样本数': n_good,
            '好样本占比': good_rate,
            '坏样本数': n_bad,
            '坏样本占比': bad_rate,
            '坏样本率': bad_rate,
            'LIFT': lift,
            '金额总数': float(amt_total) if amt_total is not None else None,
            '预估坏金额': float(amt_bad) if amt_bad is not None else None,
        })

    swap_summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # 7. 构建 Swap Pipeline 表（分析流程 + 规则集 + 样本占比 + 通过率列）
    # ------------------------------------------------------------------
    scale = 1.0 / sample_survival_rate if sample_survival_rate > 0 else 1.0
    N_total_actual = len(df)  # 实际样本数（不乘 scale）
    N_total = int(N_total_actual * scale)  # 调整后总样本数（用于通过率计算）

    quad_counts = {
        q: int((df['_swap_quadrant'] == q).sum())
        for q in ['in_in', 'in_out', 'out_in', 'out_out']
    }
    N_ii = quad_counts['in_in']
    N_io = quad_counts['in_out']
    N_oi = quad_counts['out_in']
    N_oo = quad_counts['out_out']

    # 获取多标签列结构（用于构建 MultiIndex 列）
    # 先用一个 dummy call 探测 rule.report() 返回的列结构
    _demo_tbl = None
    for _r in (rules_in or []):
        _demo_tbl = _r.report(
            df.head(1), target=target, overdue=overdue, dpds=dpds, margins=False, amount=amount
        )
        break

    def _is_multi_label(tbl: pd.DataFrame) -> bool:
        return isinstance(tbl.columns, pd.MultiIndex)

    def _get_label_names(tbl: pd.DataFrame) -> List[str]:
        if not isinstance(tbl.columns, pd.MultiIndex):
            return []
        return [c for c in tbl.columns.get_level_values(0)
                if c not in ('分箱详情', '规则详情')]

    is_multi = _is_multi_label(_demo_tbl) if _demo_tbl is not None else False
    label_names = _get_label_names(_demo_tbl) if _demo_tbl is not None else []

    def _build_detail_col(tbl: pd.DataFrame, expr_text: str = '') -> pd.DataFrame:
        """追加 规则详情 列。
        MultiIndex 场景：新建 ('分箱详情', '规则详情') 列并拼回原表。
        非 MultiIndex 场景：追加单层 '规则详情' 列。
        """
        if isinstance(tbl.columns, pd.MultiIndex):
            detail_group = _get_detail_group_name(tbl)
            # 创建新的规则详情列（1行，与 tbl 等长）
            detail_series = pd.DataFrame({(detail_group, '规则详情'): [expr_text] * len(tbl)},
                                          index=tbl.index)
            # concat 方式合并（避免混合单层/多层列）
            return pd.concat([tbl.reset_index(drop=True),
                              detail_series.reset_index(drop=True)], axis=1)
        else:
            tbl = tbl.copy()
            if '规则详情' not in tbl.columns:
                tbl['规则详情'] = expr_text
            return tbl

    def _filter_hit_rows(tbl: pd.DataFrame) -> pd.DataFrame:
        """只保留 命中 行（移除 未命中 和 合计 行）。"""
        tbl = tbl.copy()
        if isinstance(tbl.columns, pd.MultiIndex):
            detail_group = _get_detail_group_name(tbl)
            bin_col = (detail_group, '分箱')
            if bin_col in tbl.columns:
                mask = tbl[bin_col] == '命中'
                tbl = tbl[mask]
        else:
            if '分箱' in tbl.columns:
                tbl = tbl[tbl['分箱'] == '命中']
        return tbl

    def _append_rate_change_col(tbl: pd.DataFrame, prev_rate: float) -> Tuple[pd.DataFrame, float]:
        """追加 通过率变化 列，返回 (新表, 当前通过率)。"""
        tbl = tbl.copy()
        cur_rate = float(tbl['通过率(绝对值)'].iloc[0]) if len(tbl) > 0 else prev_rate
        change = cur_rate - prev_rate if prev_rate is not None else None
        tbl['通过率变化'] = change if change is not None else None
        return tbl, cur_rate

    def _build_rule_row(rule_name, rule_obj, df_subset, step_label, adj_dir,
                        parent_n, cum_remain, prev_rate, is_out_in=False):
        """生成单条规则的命中行，附加分析流程、规则详情、通过率变化等信息。"""
        if len(df_subset) == 0:
            return None, prev_rate

        # 直接从预测计算命中数（避免 rule.report() 金额口径导致样本数口径错乱）
        hit_mask = rule_obj.predict(df_subset)
        n_hit = int(hit_mask.sum())
        if n_hit == 0:
            return None, prev_rate

        # 调用 rule.report() 获取坏样本率等指标（不用 amount，避免样本数/金额口径混乱）
        raw_tbl = rule_obj.report(
            df_subset, target=target, overdue=overdue, dpds=dpds,
            margins=False,
        )
        raw_tbl = raw_tbl.reset_index(drop=True)

        # 只保留 命中 行
        tbl = _filter_hit_rows(raw_tbl)
        if len(tbl) == 0:
            return None, prev_rate

        # 扁平化 MultiIndex → 单层列（必须在任何标量列赋值之前完成）
        expr_text = rule_obj.expr
        if isinstance(tbl.columns, pd.MultiIndex):
            detail_group = _get_detail_group_name(tbl)
            # 显式构建单层字符串列名列表，避免 list(tbl.columns) 返回 tuple 导致 MultiIndex 保留
            new_col_names = []
            for col in tbl.columns:
                name = col[1]
                if col[0] == detail_group:
                    if name == '规则分类':
                        name = '规则分类_out'
                    new_col_names.append(name)
                else:
                    new_col_names.append(f'{name}({col[0]})')
            tbl.columns = new_col_names
            tbl['规则详情'] = expr_text
        else:
            if '规则分类' in tbl.columns:
                tbl = tbl.rename(columns={'规则分类': '规则分类_out'})
            if '规则详情' not in tbl.columns:
                tbl['规则详情'] = ''
            if '指标名称' not in tbl.columns:
                tbl['指标名称'] = ''

        # 样本数使用直接计算的命中数（不从 rule.report() 取，避免金额口径覆盖）
        sample_ratio = n_hit / N_total_actual if N_total_actual > 0 else 0.0
        sample_ratio_rel = n_hit / parent_n if parent_n > 0 else 0.0

        # 样本数用实际值，但通过率用调整后值
        abs_pass = cum_remain / N_total if N_total > 0 else 0.0
        rel_pass = cum_remain / parent_n if parent_n > 0 else 0.0

        # 添加固定列
        tbl['分析流程'] = step_label
        tbl['规则集'] = rule_name
        tbl['样本总数'] = n_hit
        tbl['样本占比'] = sample_ratio
        tbl['样本占比(相对)'] = sample_ratio_rel
        tbl['通过率(绝对值)'] = abs_pass
        tbl['通过率(相对值)'] = rel_pass
        tbl['通过率'] = abs_pass
        tbl['调整方向'] = adj_dir

        # out_in 特殊处理：基于 OverduePredictor 预测应用 uplift
        if is_out_in:
            # hit_mask 与 _pred_df 索引不同，使用 positional index 避免对齐问题
            # 先获取 df_subset 在 df 中的位置索引
            df_subset_idx = df_subset.index
            pred_subset = _pred_df.loc[df_subset_idx]  # reindex _pred_df to df_subset's index
            if '坏样本率' in tbl.columns:
                pred_bad = float(pred_subset[hit_mask].mean()) if hit_mask.sum() > 0 else _overall_pred_bad
                uplifted_bad = min(pred_bad * out_in_uplift, 1.0)
                tbl['坏样本率'] = uplifted_bad
            if 'LIFT值' in tbl.columns:
                base_bad = _overall_pred_bad if _overall_pred_bad > 0 else 0.01
                tbl['LIFT值'] = uplifted_bad / base_bad

        # 金额口径支持（独立于 rule.report() 结果）
        if amount is not None and amount in df_subset.columns:
            amt_hit = float(df_subset.loc[hit_mask, amount].sum())
            if amt_hit == 0.0 and is_out_in:
                n_oi = hit_mask.sum()
                if out_in_amount_col and out_in_amount_col in df_subset.columns:
                    amt_hit = float(df_subset.loc[hit_mask, out_in_amount_col].sum())
                elif out_in_amount_fill is not None:
                    amt_hit = float(out_in_amount_fill * n_oi)
            tbl['金额总数'] = amt_hit
            bad_rate_val = float(tbl['坏样本率'].iloc[0]) if len(tbl) > 0 and '坏样本率' in tbl.columns else 0.0
            tbl['预估坏金额'] = float(amt_hit * bad_rate_val)

        # 计算通过率变化
        cur_rate = abs_pass
        change = cur_rate - prev_rate if prev_rate is not None else None
        tbl['通过率变化'] = change if change is not None else None

        return tbl, cur_rate

    def _build_subtotal_row(step_label, sub_n, parent_n, cum_remain, prev_rate, adj_dir, label_suffix=''):
        """生成合计小计行（只显示样本数、占比、通过率等，不含 rule.report 详情）。"""
        if sub_n == 0:
            return None, prev_rate

        # 计算小计行的坏样本率（基于 OverduePredictor 预测）
        q_name_map = {
            'OUT-OUT 拒绝样本': 'out_out',
            'IN-OUT 置出样本': 'in_out',
            'OUT-IN 置入样本': 'out_in',
        }
        q_key = q_name_map.get(step_label, step_label)
        q_mask = df['_swap_quadrant'] == q_key
        q_pred = _pred_df[q_mask]
        sub_bad_rate = float(q_pred.mean()) if len(q_pred) > 0 else _overall_pred_bad
        if q_key == 'out_in':
            sub_bad_rate = min(sub_bad_rate * out_in_uplift, 1.0)

        sub_lift = sub_bad_rate / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
        sub_good = int(sub_n * (1 - sub_bad_rate))
        sub_bad = int(sub_n * sub_bad_rate)
        sample_ratio = sub_n / N_total_actual if N_total_actual > 0 else 0.0
        sample_ratio_rel = sub_n / parent_n if parent_n > 0 else 0.0
        abs_pass = cum_remain / N_total if N_total > 0 else 0.0
        rel_pass = cum_remain / parent_n if parent_n > 0 else 0.0
        cur_rate = abs_pass
        change = cur_rate - prev_rate if prev_rate is not None else None

        # 构建列结构（与 rule.report 保持一致）
        row = {}
        # 先填入指标列
        if is_multi and label_names:
            for lbl in label_names:
                for metric in ('好样本数', '坏样本数', '坏样本率', 'LIFT值',
                              '好样本占比', '坏样本占比'):
                    row[f'{metric}({lbl})'] = None  # placeholder，合计行不展开
            # 小计行只显示综合坏样本率
            row['坏样本率'] = sub_bad_rate
            row['LIFT值'] = sub_lift
            row['好样本数'] = sub_good
            row['坏样本数'] = sub_bad
            row['好样本占比'] = 1 - sub_bad_rate
            row['坏样本占比'] = sub_bad_rate
        else:
            row = {
                '分析流程': step_label,
                '规则集': '合计',
                '规则分类_out': step_label,
                '规则详情': '',
                '指标名称': '合计',
                '分箱': '合计',
                '样本总数': sub_n,
                '样本占比': sample_ratio,
                '样本占比(相对)': sample_ratio_rel,
                '好样本数': sub_good,
                '好样本占比': 1 - sub_bad_rate,
                '坏样本数': sub_bad,
                '坏样本占比': sub_bad_rate,
                '坏样本率': sub_bad_rate,
                'LIFT值': sub_lift,
                '通过率': abs_pass,
                '通过率(绝对值)': abs_pass,
                '通过率(相对值)': rel_pass,
                '通过率变化': change,
                '调整方向': adj_dir,
            }
            # 金额口径支持
            if amount is not None and amount in df.columns:
                quad_name_map = {
                    'OUT-OUT 拒绝样本': 'out_out',
                    'IN-OUT 置出样本': 'in_out',
                    'OUT-IN 置入样本': 'out_in',
                    'IN-IN 通过样本': 'in_in',
                }
                q_key = quad_name_map.get(step_label, step_label)
                quad_amt = float(df.loc[df['_swap_quadrant'] == q_key, amount].sum())
                row['金额总数'] = quad_amt
                row['预估坏金额'] = quad_amt * sub_bad_rate
            return pd.DataFrame([row]), cur_rate

        # 多标签分支
        row.update({
            '分析流程': step_label,
            '规则集': '合计',
            '样本总数': sub_n,
            '样本占比': sample_ratio,
            '样本占比(相对)': sample_ratio_rel,
            '通过率': abs_pass,
            '通过率(绝对值)': abs_pass,
            '通过率(相对值)': rel_pass,
            '通过率变化': change,
            '调整方向': adj_dir,
        })
        # 金额口径支持
        if amount is not None and amount in df.columns:
            quad_name_map = {
                'OUT-OUT 拒绝样本': 'out_out',
                'IN-OUT 置出样本': 'in_out',
                'OUT-IN 置入样本': 'out_in',
                'IN-IN 通过样本': 'in_in',
            }
            q_key = quad_name_map.get(step_label, step_label)
            quad_amt = float(df.loc[df['_swap_quadrant'] == q_key, amount].sum())
            row['金额总数'] = quad_amt
            row['预估坏金额'] = quad_amt * sub_bad_rate
        return pd.DataFrame([row]), cur_rate

    def _build_remaining_row(step_label, rem_n, parent_n, cum_remain, prev_rate):
        """生成剩余样本行。"""
        if rem_n == 0:
            return None, prev_rate
        rem_bad_rate = _overall_pred_bad
        rem_lift = rem_bad_rate / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
        rem_good = int(rem_n * (1 - rem_bad_rate))
        rem_bad = int(rem_n * rem_bad_rate)
        sample_ratio = rem_n / N_total_actual if N_total_actual > 0 else 0.0
        sample_ratio_rel = rem_n / parent_n if parent_n > 0 else 0.0
        abs_pass = cum_remain / N_total if N_total > 0 else 0.0
        rel_pass = cum_remain / parent_n if parent_n > 0 else 0.0
        cur_rate = abs_pass
        change = cur_rate - prev_rate if prev_rate is not None else None

        row = {
            '分析流程': step_label,
            '规则集': '',
            '规则分类_out': step_label,
            '规则详情': '',
            '指标名称': '',
            '分箱': '剩余样本',
            '样本总数': rem_n,
            '样本占比': sample_ratio,
            '样本占比(相对)': sample_ratio_rel,
            '好样本数': rem_good,
            '好样本占比': 1 - rem_bad_rate,
            '坏样本数': rem_bad,
            '坏样本占比': rem_bad_rate,
            '坏样本率': rem_bad_rate,
            'LIFT值': rem_lift,
            '通过率': abs_pass,
            '通过率(绝对值)': abs_pass,
            '通过率(相对值)': rel_pass,
            '通过率变化': change,
            '调整方向': '-',
        }
        # 金额口径支持：各剩余阶段累计金额
        if amount is not None and amount in df.columns:
            if 'OUT-OUT' in step_label:
                rem_quads = ['in_out', 'in_in', 'out_in']
            elif 'IN-OUT' in step_label and '后' in step_label:
                rem_quads = ['in_in', 'out_in']
            elif 'IN-IN' in step_label and '后' in step_label:
                rem_quads = ['out_in']
            else:
                rem_quads = ['in_in', 'out_in']
            amt_total = float(df.loc[df['_swap_quadrant'].isin(rem_quads), amount].sum())
            row['金额总数'] = amt_total
            row['预估坏金额'] = amt_total * rem_bad_rate
        return pd.DataFrame([row]), cur_rate

    def _build_total_row(step_label, parent_n, prev_rate):
        """生成全部样本汇总行。"""
        if parent_n == 0:
            return None, prev_rate
        total_bad = _overall_pred_bad
        sample_ratio = parent_n / N_total_actual if N_total_actual > 0 else 1.0
        abs_pass = 1.0
        rel_pass = 1.0
        cur_rate = 1.0
        change = cur_rate - prev_rate if prev_rate is not None else None
        row = {
            '分析流程': step_label,
            '规则集': '',
            '规则分类_out': step_label,
            '规则详情': '',
            '指标名称': '',
            '分箱': '全部样本',
            '样本总数': parent_n,
            '样本占比': sample_ratio,
            '样本占比(相对)': 1.0,
            '好样本数': int(parent_n * (1 - total_bad)),
            '好样本占比': 1 - total_bad,
            '坏样本数': int(parent_n * total_bad),
            '坏样本占比': total_bad,
            '坏样本率': total_bad,
            'LIFT值': 1.0,
            '通过率': abs_pass,
            '通过率(绝对值)': abs_pass,
            '通过率(相对值)': rel_pass,
            '通过率变化': change,
            '调整方向': '-',
        }
        # 金额口径支持
        if amount is not None and amount in df.columns:
            row['金额总数'] = float(df[amount].sum())
            row['预估坏金额'] = row['金额总数'] * total_bad
        return pd.DataFrame([row]), cur_rate

    def _build_all_in_row(cum_remain, prev_rate):
        """生成 ALL-IN 置换样本最终汇总行。"""
        all_in_n = N_ii + N_oi
        if all_in_n == 0:
            return None, prev_rate
        all_in_pass = cum_remain / N_total if N_total > 0 else 0.0
        all_in_bad = (N_ii * _overall_pred_bad + N_oi * min(_overall_pred_bad * out_in_uplift, 1.0)) \
            / max(all_in_n, 1)
        all_in_lift = all_in_bad / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
        sample_ratio = all_in_n / N_total_actual if N_total_actual > 0 else 0.0
        cur_rate = all_in_pass
        change = cur_rate - prev_rate if prev_rate is not None else None
        row = {
            '分析流程': 'ALL-IN 置换样本',
            '规则集': '',
            '规则分类_out': 'ALL-IN 置换样本',
            '规则详情': '',
            '指标名称': '',
            '分箱': 'ALL-IN',
            '样本总数': all_in_n,
            '样本占比': sample_ratio,
            '样本占比(相对)': 1.0,
            '好样本数': int(all_in_n * (1 - all_in_bad)),
            '好样本占比': 1 - all_in_bad,
            '坏样本数': int(all_in_n * all_in_bad),
            '坏样本占比': all_in_bad,
            '坏样本率': all_in_bad,
            'LIFT值': all_in_lift,
            '通过率': all_in_pass,
            '通过率(绝对值)': all_in_pass,
            '通过率(相对值)': 1.0,
            '通过率变化': change,
            '调整方向': '-',
        }
        # 金额口径支持
        if amount is not None and amount in df.columns:
            all_in_amt = float(df.loc[df['_swap_quadrant'].isin(['in_in', 'out_in']), amount].sum())
            row['金额总数'] = all_in_amt
            row['预估坏金额'] = all_in_amt * all_in_bad
        return pd.DataFrame([row]), cur_rate

    # 统计各象限内每条规则的命中数（用于决定规则展示顺序）
    def _count_rule_hits(df_quad, rules_list):
        result = []
        for r in (rules_list or []):
            hit = r.predict(df_quad)
            n_hit = int(hit.sum())
            result.append((n_hit, r.name or r.expr, r))
        return sorted(result, key=lambda x: -x[0])

    out_in_hits = _count_rule_hits(df[df['_swap_quadrant'] == 'out_in'], rules_in)
    out_out_hits = _count_rule_hits(df[df['_swap_quadrant'] == 'out_out'], rules_base)
    in_out_hits = _count_rule_hits(df[df['_swap_quadrant'] == 'in_out'], rules_out)

    # ── 正序流程 ────────────────────────────────────────────────────────
    if not reverse_order:
        pipeline_tables = []
        prev_rate = None

        # 1. 全部样本
        t, prev_rate = _build_total_row('全部样本', N_total, prev_rate)
        if t is not None:
            pipeline_tables.append(t)

        # 2. OUT-OUT 拒绝样本（per-rule）
        parent_oo = N_total
        cum_oo = N_total
        for n_hit, rule_name, rule_obj in out_out_hits:
            df_oo_subset = df[df['_swap_quadrant'] == 'out_out']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_oo_subset,
                step_label='OUT-OUT 拒绝样本',
                adj_dir='收紧',
                parent_n=parent_oo,
                cum_remain=cum_oo,
                prev_rate=prev_rate,
                is_out_in=False,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_oo > 0:
            t, prev_rate = _build_subtotal_row('OUT-OUT 拒绝样本', N_oo, parent_oo, cum_oo, prev_rate, '收紧')
            if t is not None:
                pipeline_tables.append(t)

        # 3. OUT-OUT 后剩余样本
        rem_after_oo = N_total - N_oo
        if rem_after_oo > 0:
            t, prev_rate = _build_remaining_row('OUT-OUT后剩余样本', rem_after_oo, parent_oo, rem_after_oo, prev_rate)
            if t is not None:
                pipeline_tables.append(t)

        # 4. IN-OUT 置出样本（per-rule）
        parent_io = rem_after_oo
        cum_io = rem_after_oo
        for n_hit, rule_name, rule_obj in in_out_hits:
            df_io_subset = df[df['_swap_quadrant'] == 'in_out']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_io_subset,
                step_label='IN-OUT 置出样本',
                adj_dir='释放',
                parent_n=parent_io,
                cum_remain=cum_io,
                prev_rate=prev_rate,
                is_out_in=False,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_io > 0:
            t, prev_rate = _build_subtotal_row('IN-OUT 置出样本', N_io, parent_io, cum_io, prev_rate, '释放')
            if t is not None:
                pipeline_tables.append(t)

        # 5. IN-OUT 后剩余样本
        rem_after_io = rem_after_oo - N_io
        if rem_after_io > 0:
            t, prev_rate = _build_remaining_row('IN-OUT后剩余样本', rem_after_io, parent_io, rem_after_io, prev_rate)
            if t is not None:
                pipeline_tables.append(t)

        # 6. IN-IN 通过样本
        if N_ii > 0:
            t, prev_rate = _build_subtotal_row('IN-IN 通过样本', N_ii, rem_after_io, rem_after_io, prev_rate, '释放')
            if t is not None:
                pipeline_tables.append(t)

        # 7. OUT-IN 置入样本（per-rule）
        parent_oi = N_oi
        cum_oi = rem_after_io
        for n_hit, rule_name, rule_obj in out_in_hits:
            df_oi_subset = df[df['_swap_quadrant'] == 'out_in']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_oi_subset,
                step_label='OUT-IN 置入样本',
                adj_dir='收紧',
                parent_n=parent_oi,
                cum_remain=cum_oi,
                prev_rate=prev_rate,
                is_out_in=True,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_oi > 0:
            t, prev_rate = _build_subtotal_row('OUT-IN 置入样本', N_oi, parent_oi, cum_oi, prev_rate, '收紧')
            if t is not None:
                pipeline_tables.append(t)

        # 8. ALL-IN 置换样本
        t, prev_rate = _build_all_in_row(cum_oi, prev_rate)
        if t is not None:
            pipeline_tables.append(t)

    # ── 逆序流程 ────────────────────────────────────────────────────────
    else:
        pipeline_tables = []
        prev_rate = None

        # 1. 全部样本
        t, prev_rate = _build_total_row('全部样本', N_total, prev_rate)
        if t is not None:
            pipeline_tables.append(t)

        # OUT-IN 置入样本（per-rule）— 逆序先展示
        parent_oi_rev = N_oi
        cum_oi_rev = N_total
        for n_hit, rule_name, rule_obj in out_in_hits:
            df_oi_subset = df[df['_swap_quadrant'] == 'out_in']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_oi_subset,
                step_label='OUT-IN 置入样本',
                adj_dir='收紧',
                parent_n=parent_oi_rev,
                cum_remain=cum_oi_rev,
                prev_rate=prev_rate,
                is_out_in=True,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_oi > 0:
            t, prev_rate = _build_subtotal_row('OUT-IN 置入样本', N_oi, parent_oi_rev, cum_oi_rev, prev_rate, '收紧')
            if t is not None:
                pipeline_tables.append(t)

        # OUT-IN 后剩余样本（IN-IN）
        rem_after_oi = N_oi + N_ii
        if rem_after_oi > 0:
            t, prev_rate = _build_remaining_row('IN-IN后剩余样本', rem_after_oi, cum_oi_rev, rem_after_oi, prev_rate)
            if t is not None:
                pipeline_tables.append(t)

        # IN-IN 通过样本
        if N_ii > 0:
            t, prev_rate = _build_subtotal_row('IN-IN 通过样本', N_ii, rem_after_oi, rem_after_oi, prev_rate, '释放')
            if t is not None:
                pipeline_tables.append(t)

        # IN-OUT 置出样本（per-rule）
        parent_io_rev = N_io
        cum_io_rev = rem_after_oi
        for n_hit, rule_name, rule_obj in in_out_hits:
            df_io_subset = df[df['_swap_quadrant'] == 'in_out']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_io_subset,
                step_label='IN-OUT 置出样本',
                adj_dir='释放',
                parent_n=parent_io_rev,
                cum_remain=cum_io_rev,
                prev_rate=prev_rate,
                is_out_in=False,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_io > 0:
            t, prev_rate = _build_subtotal_row('IN-OUT 置出样本', N_io, parent_io_rev, cum_io_rev, prev_rate, '释放')
            if t is not None:
                pipeline_tables.append(t)

        # IN-OUT 后剩余样本
        rem_after_io_rev = N_oi + N_ii + N_io
        if rem_after_io_rev > 0:
            t, prev_rate = _build_remaining_row('IN-OUT后剩余样本', rem_after_io_rev, cum_io_rev, rem_after_io_rev, prev_rate)
            if t is not None:
                pipeline_tables.append(t)

        # OUT-OUT 拒绝样本（per-rule）
        parent_oo_rev = N_oo
        cum_oo_rev = rem_after_io_rev
        for n_hit, rule_name, rule_obj in out_out_hits:
            df_oo_subset = df[df['_swap_quadrant'] == 'out_out']
            t, prev_rate = _build_rule_row(
                rule_name, rule_obj, df_oo_subset,
                step_label='OUT-OUT 拒绝样本',
                adj_dir='收紧',
                parent_n=parent_oo_rev,
                cum_remain=cum_oo_rev,
                prev_rate=prev_rate,
                is_out_in=False,
            )
            if t is not None:
                pipeline_tables.append(t)
        if N_oo > 0:
            t, prev_rate = _build_subtotal_row('OUT-OUT 拒绝样本', N_oo, parent_oo_rev, cum_oo_rev, prev_rate, '收紧')
            if t is not None:
                pipeline_tables.append(t)

        # ALL-IN 置换样本
        t, prev_rate = _build_all_in_row(cum_oo_rev, prev_rate)
        if t is not None:
            pipeline_tables.append(t)

    # 合并所有表，统一列顺序
    if len(pipeline_tables) > 0:
        swap_pipeline = pd.concat(pipeline_tables, ignore_index=True)

        # 确保关键列存在
        standard_cols = [
            '分析流程', '规则集', '规则详情',
            '样本总数', '样本占比', '样本占比(相对)',
            '好样本数', '好样本占比', '坏样本数', '坏样本占比',
            '坏样本率', 'LIFT值', '坏账改善', '风险拒绝比',
            '准确率', '精确率', '召回率', 'F1分数',
            '通过率', '通过率(绝对值)', '通过率(相对值)', '通过率变化', '调整方向',
            # rule.report MultiIndex 列兼容
            '规则分类_out', '指标名称', '分箱',
        ]
        for col in standard_cols:
            if col not in swap_pipeline.columns:
                swap_pipeline[col] = None

        # 按标准顺序排列列（保留出现的列，未出现的跳过）
        final_cols = [c for c in standard_cols if c in swap_pipeline.columns]
        # 补充任何额外列
        extra_cols = [c for c in swap_pipeline.columns if c not in standard_cols]
        swap_pipeline = swap_pipeline[final_cols + extra_cols]
    else:
        swap_pipeline = pd.DataFrame(columns=[
            '分析流程', '规则集', '规则详情',
            '样本总数', '样本占比', '样本占比(相对)',
            '好样本数', '好样本占比', '坏样本数', '坏样本占比',
            '坏样本率', 'LIFT值', '通过率', '通过率(绝对值)', '通过率(相对值)',
            '通过率变化', '调整方向',
        ])

    # ------------------------------------------------------------------
    # 8. 构建 Swap Result 表（置换前后对比与业务增益）
    # ------------------------------------------------------------------
    # 基于预测逾期率计算（使用 _pred_df 保持一致性）
    ii_mask = df['_swap_quadrant'] == 'in_in'
    oi_mask = df['_swap_quadrant'] == 'out_in'
    io_mask = df['_swap_quadrant'] == 'in_out'

    # 置换前：仅 in_in 通过，基准逾期率
    pass_rate_before = N_ii / N_total if N_total > 0 else 0.0
    pass_rate_after = (N_ii + N_oi) / N_total if N_total > 0 else 0.0

    bad_rate_before = float(_pred_df[ii_mask].mean()) if ii_mask.sum() > 0 else _overall_pred_bad
    io_pred = float(_pred_df[io_mask].mean()) if io_mask.sum() > 0 else bad_rate_before
    oi_pred_raw = float(_pred_df[oi_mask].mean()) if oi_mask.sum() > 0 else bad_rate_before
    bad_rate_oi = min(oi_pred_raw * out_in_uplift, 1.0)

    # 置换后整体逾期率（加权平均）
    n_after_total = N_ii + N_oi
    bad_rate_after = (N_ii * bad_rate_before + N_oi * bad_rate_oi) / n_after_total if n_after_total > 0 else 0.0

    # 业务增益（支持金额口径：优先用金额计算，fallback 到样本数）
    if amount is not None and amount in df.columns:
        # 金额口径
        amt_ii = float(df[ii_mask][amount].sum())
        amt_oi_raw = float(df[oi_mask][amount].sum())
        amt_io = float(df[io_mask][amount].sum())
        # out_in 填充
        n_oi = oi_mask.sum()
        if amt_oi_raw == 0.0 and n_oi > 0:
            if out_in_amount_col and out_in_amount_col in df.columns:
                amt_oi_raw = float(df[oi_mask][out_in_amount_col].sum())
            elif out_in_amount_fill is not None:
                amt_oi_raw = float(out_in_amount_fill * n_oi)
        amt_oi = amt_oi_raw
        loan_increase_abs = amt_oi
        loan_increase_rel = amt_oi / amt_ii if amt_ii > 0 else 0.0
        bad_in_out = int(amt_io * io_pred)
        bad_out_in = int(amt_oi * oi_pred_raw * out_in_uplift)
    else:
        loan_increase_abs = N_oi
        loan_increase_rel = N_oi / N_ii if N_ii > 0 else 0.0
        bad_in_out = int(N_io * io_pred)
        bad_out_in = int(N_oi * oi_pred_raw * out_in_uplift)
    risk_delta_abs = bad_out_in - bad_in_out
    risk_delta_rel = risk_delta_abs / max(N_ii, 1)

    swap_result_rows = [
        {'指标': '通过率变化', '变化前': pass_rate_before, '变化后': pass_rate_after,
         '绝对变化': pass_rate_after - pass_rate_before,
         '相对变化': (pass_rate_after - pass_rate_before) / max(abs(pass_rate_before), 1e-9)},
        {'指标': '逾期率变化', '变化前': bad_rate_before, '变化后': bad_rate_after,
         '绝对变化': bad_rate_after - bad_rate_before,
         '相对变化': (bad_rate_after - bad_rate_before) / max(abs(bad_rate_before), 1e-9)},
        {'指标': '风险上浮系数', '变化前': 1.0, '变化后': out_in_uplift,
         '绝对变化': out_in_uplift - 1.0, '相对变化': out_in_uplift - 1.0},
        {'指标': '放款增量（绝对）', '变化前': 0, '变化后': loan_increase_abs,
         '绝对变化': loan_increase_abs, '相对变化': loan_increase_rel},
        {'指标': '放款增量（相对）', '变化前': 0, '变化后': loan_increase_rel,
         '绝对变化': loan_increase_rel, '相对变化': loan_increase_rel},
        {'指标': '坏样本变化（绝对）', '变化前': 0, '变化后': risk_delta_abs,
         '绝对变化': risk_delta_abs, '相对变化': risk_delta_rel},
        {'指标': '坏样本变化（相对）', '变化前': 0, '变化后': risk_delta_rel,
         '绝对变化': risk_delta_rel, '相对变化': risk_delta_rel},
        {'指标': '样本集幸存比例', '变化前': sample_survival_rate, '变化后': sample_survival_rate,
         '绝对变化': 0.0, '相对变化': 0.0},
    ]

    swap_result = pd.DataFrame(swap_result_rows)

    # ------------------------------------------------------------------
    # 9. 清理临时列
    # ------------------------------------------------------------------
    drop_cols = [c for c in df.columns if c.startswith('_swap') or c.startswith('_tmp_norm')]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return {
        'swap_summary': swap_summary,
        'swap_pipeline': swap_pipeline,
        'swap_result': swap_result,
    }


def rule_swap_analysis_v2(
    data: pd.DataFrame,
    score: Union[str, Dict[str, str]],
    rules_in: List[Rule],
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
    """规则置入置出（Swap）分析 v2.

    API 参数与 ``rule_swap_analysis`` 基本一致，新增/变更说明：

    - 新增 ``reference_data`` 参数：历史有表现样本集，用于计算评分的风险表现分箱表
    - ``bin_table`` 参数支持 dict 格式：``{评分名: 分箱表}``，多评分时各评分独立计算
    - ``reference_data`` 与 ``bin_table`` 二选一；优先使用 ``bin_table``
    - 分箱表结果通过 ``self.bin_table_`` 属性存储（标准化后的单/多层列 DataFrame）

    :param data: 全量样本集（包含 score 列 + rules_in/rules_out/rules_base 用到的所有特征列）
    :param score: 评分字段名（str）或多评分映射（Dict）
    :param rules_in: 置入规则集（List[Rule]）
    :param rules_out: 置出规则集（可选）
    :param rules_base: 基准拒绝规则集（可选）
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
    :param amount: 金额字段（可选）
    :param sample_survival_rate: 样本集幸存比例，默认 1.0
    :param reverse_order: 是否逆序展示（True: 从置入效果开始展示）
    :param out_in_amount_fill: out_in 置入样本额度填充定值（可选）
    :param out_in_amount_col: out_in 置入样本额度填充字段名（可选）
    :param bin_method: 分箱方法，默认 'quantile'（仅 reference_data 模式生效）
    :param max_n_bins: 最大分箱数，默认 10（仅 reference_data 模式生效）
    :param min_bin_size: 每箱最小样本占比，默认 0.05（仅 reference_data 模式生效）
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param bin_params: 额外分箱参数 dict，会透传给 ``feature_bin_stats``
        常用键：rules（自定义切分点）、monotonic（单调性约束）等
    :param rule_analysis_mode: 规则分析模式，默认 'independent'。
        - 'independent'：每条规则独立应用到全量 data，分别统计命中好坏分布。
          OUT-OUT合计 = 所有规则合并（reduce |）后的联合命中。
        - 'sequential'：漏斗模式，每条规则在前一条拒绝后的剩余样本上分析，
          规则命中（拒绝）时样本量减少，未命中（通过）时保持剩余样本。
          适合分析多规则叠加的拒绝效果。
    :return: 包含三张表的字典

        - ``swap_summary``：四象限样本汇总
        - ``swap_pipeline``：分步骤通过率与逾期率变化（可逆序）
        - ``swap_result``：置换前后对比与业务增益

    **参考样例**

    >>> from hscredit.core.rules import Rule
    >>> from hscredit.report.rule_analysis import rule_swap_analysis_v2
    >>>
    >>> # 方式一：传入历史参考数据，自动计算分箱表
    >>> result = rule_swap_analysis_v2(
    ...     data=data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     rules_base=[rule_base],
    ...     reference_data=hist_data,   # 有表现样本，包含 target 列
    ...     target='target',
    ... )
    >>>
    >>> # 方式二：直接传入现成分箱表
    >>> result = rule_swap_analysis_v2(
    ...     data=data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     rules_base=[rule_base],
    ...     bin_table=score_bin_table,   # pd.DataFrame 或 Dict[str, pd.DataFrame]
    ... )
    >>>
    >>> # 方式三：多标签场景（overdue+dpds）
    >>> result = rule_swap_analysis_v2(
    ...     data=data,
    ...     score='score_a',
    ...     rules_in=[rule_in],
    ...     reference_data=hist_data,
    ...     overdue='MOB1',
    ...     dpds=[0, 7, 30],
    ... )
    >>>
    >>> print(result['swap_summary'])
    >>> print(result['swap_pipeline'])
    >>> print(result['swap_result'])
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
        amount=amount,
        data=data,
    )

    # ── 第二步：规则集预处理 ─────────────────────────────────────────────
    # 统一 score 为 Dict[str, str] 格式（供后续使用）
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

    # ── 第四步：构建OUT-OUT拒绝报告 ────────────────────────────────────────
    base_report, full_bad_probs = _build_base_report(
        data=data,
        rules_base=rules_base,
        bin_table_result=bin_table_result,
        score_map=score_map,
        score_weights=score_weights,
        amount=amount,
        rule_analysis_mode=rule_analysis_mode,
        sample_survival_rate=sample_survival_rate,
    )

    # 从 base_report 提取最终通过样本（剩余样本）及其通过率
    if rules_base:
        # 剩余样本是 base_report 最后一行
        final_remain_row = base_report[base_report['规则分类'] == '剩余样本']
        if not final_remain_row.empty:
            # 获取最终通过率（从调整方向列或通过率列）
            final_pass_rate = float(final_remain_row.iloc[0]['通过率'])
        else:
            final_pass_rate = sample_survival_rate
        # 获取剩余样本的掩码
        combined_base = reduce(lambda r1, r2: r1 | r2, rules_base)
        in_remaining_mask = ~combined_base.predict(data)
        in_remaining_data = data[in_remaining_mask].reset_index(drop=True)
    else:
        # 无rules_base时，全部数据进入IN-OUT
        final_pass_rate = sample_survival_rate
        in_remaining_data = data.copy().reset_index(drop=True)

    # ── 第五步：构建IN-OUT置出报告 ────────────────────────────────────────
    # 计算全量样本的扩展坏样本数（用于IN-OUT分析）
    n_total = len(data)
    n_total_full_v2 = int(n_total / sample_survival_rate)
    n_bad_full_v2 = float(base_report[base_report['规则分类'] == '全量样本'].iloc[0]['坏样本数']) \
        if not base_report.empty else 0.0
    # 全量坏样本数（预测值，非整数化）需要从full_bad_probs获取
    full_bad_rate_v2 = float(base_report[base_report['规则分类'] == '全量样本'].iloc[0]['坏样本率']) \
        if not base_report.empty else 0.0

    # 全量坏样本数（扩展到全量）
    inout_n_bad_full = full_bad_rate_v2 * n_total_full_v2 if n_total_full_v2 > 0 else 0.0

    inout_report = pd.DataFrame()  # 空DataFrame，rules_out为空时返回空
    if rules_out and len(in_remaining_data) > 0:
        inout_report = _build_inout_report(
            in_remaining_data=in_remaining_data,
            rules_out=rules_out,
            bin_table_result=bin_table_result,
            score_map=score_map,
            score_weights=score_weights,
            rule_analysis_mode=rule_analysis_mode,
            n_total_full=n_total_full_v2,
            n_bad_full=inout_n_bad_full,
            full_bad_rate=full_bad_rate_v2,
            sample_survival_rate=final_pass_rate,
        )

    # ── 返回结果 ────────────────────────────────────────────────────────────
    # swap_summary: 合并 base_report 和 inout_report
    # swap_pipeline / swap_result: 暂时返回空，后续逐步完善
    swap_summary_rows = []
    swap_summary_rows.extend(base_report.to_dict('records'))
    swap_summary_rows.extend(inout_report.to_dict('records'))

    swap_summary = pd.DataFrame(swap_summary_rows) if swap_summary_rows else pd.DataFrame()

    return {
        'swap_summary': swap_summary,
        'swap_pipeline': pd.DataFrame(),
        'swap_result': pd.DataFrame(),
    }


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
    amount: Optional[str],
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """解析或计算分箱表，统一转换为 {评分名: 分箱表} 结构。

    优先级：bin_table > reference_data > data 自动生成
    - bin_table 为 DataFrame：视为单评分，key 为 '_default'
    - bin_table 为 Dict：直接使用
    - bin_table 为 None：尝试从 reference_data 计算
    - reference_data 也为 None：尝试从 data 自动生成（需 data 有 target 或 overdue+dpds）

    分箱表标准化：
    - 单标签：单层列 DataFrame
    - 多标签：MultiIndex 列 DataFrame，顶层为标签名（如 'MOB1_0+'）

    金额口径支持：
    - 当 reference_data 中包含 amount 字段时，自动同时计算订单口径和金额口径
    - 金额口径的坏样本率列添加后缀 '(金额)' 以示区分
    - bin_table 传入时直接标准化，不额外处理金额口径（由用户自行确保传入完整）
    """
    # 统一 score 为 Dict[str, str] 格式
    if isinstance(score, str):
        score_map = {'_default': score}
    else:
        score_map = score

    # 1. bin_table 优先（直接标准化，不处理金额口径）
    if bin_table is not None:
        if isinstance(bin_table, pd.DataFrame):
            # 单 DataFrame：根据 score_map 决定返回结构
            if len(score_map) == 1:
                name = list(score_map.keys())[0]
                return {name: _normalize_bin_table(bin_table, label=name)}
            else:
                # 多评分 + 单分箱表：同一张表复制给所有评分
                return {name: _normalize_bin_table(bin_table, label=name) for name in score_map}
        elif isinstance(bin_table, dict):
            result = {}
            for name, tbl in bin_table.items():
                if isinstance(tbl, pd.DataFrame):
                    result[name] = _normalize_bin_table(tbl, label=name)
            return result
        else:
            raise TypeError(
                f"bin_table 参数类型错误，期望 pd.DataFrame 或 Dict[str, pd.DataFrame]，"
                f"实际为 {type(bin_table).__name__}"
            )

    # 2. 从 reference_data 计算
    if reference_data is None:
        # 尝试从 data 自动生成（data 需包含 target 或 overdue+dpds）
        if data is not None:
            if target is None and (overdue is None or dpds is None):
                raise ValueError(
                    "从 data 自动生成 bin_table 时，必须传入 target 或 (overdue + dpds) 参数"
                )
            reference_data = data.copy()
            # 剔除标签缺失的行（target 为 NaN 或 overdue 字段缺失）
            ref_col = target if target else overdue[0] if isinstance(overdue, list) else overdue
            if ref_col and ref_col in reference_data.columns:
                reference_data = reference_data.dropna(subset=[ref_col])
        else:
            raise ValueError(
                "必须传入 bin_table 或 reference_data 参数之一，"
                "也可传入 data（附带 target 或 overdue+dpds）自动生成。"
            )

    if target is None and (overdue is None or dpds is None):
        raise ValueError(
            "从 reference_data 计算分箱表时，必须传入 target 或 (overdue + dpds) 参数"
        )

    # 判断是否需要计算金额口径（amount 字段存在于 reference_data）
    has_amount = (
        amount is not None
        and amount in reference_data.columns
        and not reference_data[amount].isna().all()
    )

    # 合并 bin_params（显式参数优先级高于 bin_params 中的同名键）
    extra_params = dict(bin_params) if bin_params else {}
    explicit_params = {
        'method': bin_method,
        'max_n_bins': max_n_bins,
        'min_bin_size': min_bin_size,
        'missing_separate': missing_separate,
    }
    merged_params = {**extra_params, **explicit_params}

    result = {}
    for name, col in score_map.items():
        if col not in reference_data.columns:
            raise ValueError(
                f"reference_data 中缺少评分列 '{col}'（映射名：'{name}'）"
            )

        # ① 订单口径（必须）
        tbl_count = feature_bin_stats(
            reference_data,
            feature=col,
            target=target,
            overdue=overdue,
            dpds=dpds,
            amount=None,
            margins=True,
            **merged_params,
        )

        # ② 金额口径（可选，仅当 amount 字段存在时）
        if has_amount:
            tbl_amount = feature_bin_stats(
                reference_data,
                feature=col,
                target=target,
                overdue=overdue,
                dpds=dpds,
                amount=amount,
                margins=True,  # 金额口径同样需要合计行
                **merged_params,
            )
            tbl_merged = _merge_amount_bad_rate(tbl_count, tbl_amount)
        else:
            tbl_merged = tbl_count

        result[name] = _normalize_bin_table(tbl_merged, label=name)

    return result


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
        if lbl in ('缺失', '特殊', '合计'):
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

        # 确保有分箱标签列（可能在 '分箱详情' 下）
        bin_label_col = None
        for col in tbl.columns:
            col_name = col[-1] if isinstance(col, tuple) else col
            if col_name in ('分箱标签', 'bin', '分箱'):
                bin_label_col = col
                break

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


def _merge_amount_bad_rate(
    tbl_count: pd.DataFrame,
    tbl_amount: pd.DataFrame,
) -> pd.DataFrame:
    """合并订单口径和金额口径的分箱表，通过行 index 进行区分。

    合并后的结构：
    - 行索引为 MultiIndex：level0='口径'(订单口径/金额口径)，level1='分箱标签'
    - 列保持原有结构（单层或 MultiIndex），不添加额外层级

    该结构可自然适配多标签场景（多标签时列本身为 MultiIndex，行索引附加口径维度）。

    :param tbl_count: 订单口径分箱表（样本数加权，margins=True）
    :param tbl_amount: 金额口径分箱表（金额加权，margins=True）
    :return: 合并后的分箱表，index 为 MultiIndex (口径 × 分箱标签)
    """
    if tbl_count.empty or tbl_amount.empty:
        return tbl_count.copy()

    if '分箱标签' not in tbl_count.columns or '分箱标签' not in tbl_amount.columns:
        return tbl_count.copy()

    # 以分箱标签为索引，垂直拼接
    tbl_c_idx = tbl_count.set_index('分箱标签')
    tbl_a_idx = tbl_amount.set_index('分箱标签')

    result = pd.concat(
        [tbl_c_idx, tbl_a_idx],
        keys=['订单口径', '金额口径'],
        names=['口径', '分箱标签'],
    )
    return result


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

    # 要求至少有一个规则集非空
    if not rules_in and not rules_out and not rules_base:
        raise ValueError(
            "rules_in、rules_out、rules_base 三个规则集至少需要传入一个"
        )

    # 收集所有规则所需特征列
    all_rules = rules_in + rules_out + rules_base
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


def _get_bin_bad_rate_map(
    df_bin: pd.DataFrame,
    single_bad_col: Optional[str],
) -> pd.Series:
    """从分箱表构建 bin_index → bad_rate 映射，忽略合计行。

    处理 index 类型：
    - 整数索引（分箱索引）：直接用位置映射
    - MultiIndex (口径, 分箱标签)：提取分箱标签行后用位置映射

    :param df_bin: 分箱表
    :param single_bad_col: 单一坏样本率列名（单标签或 amount 场景）
    :return: bin_position → bad_rate Series，索引为分箱位置
    """
    df = df_bin.copy()

    # 过滤合计行
    if '分箱标签' in df.columns:
        df = df[df['分箱标签'] != '合计']
    elif isinstance(df.index, pd.MultiIndex):
        # MultiIndex 行：口径 × 分箱标签，合计行 index[1] == '合计'
        df = df[df.index.get_level_values(1) != '合计']
    elif isinstance(df.index, pd.RangeIndex) or not isinstance(df.index, pd.Index):
        pass  # 整数索引保持不变

    if single_bad_col and single_bad_col in df.columns:
        return df[single_bad_col].reset_index(drop=True)
    elif single_bad_col is None:
        # 多标签：返回所有坏样本率列的子集（取第一列作为代理）
        # caller 应根据 multi_label_cols 处理
        return df.iloc[:, 0].reset_index(drop=True)
    return pd.Series(dtype=float)


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
            if lbl in ('缺失', '特殊', '合计'):
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


def _build_base_report(
    data: pd.DataFrame,
    rules_base: List[Rule],
    bin_table_result: Dict[str, pd.DataFrame],
    score_map: Dict[str, str],
    score_weights: Optional[Dict[str, float]] = None,
    amount: Optional[str] = None,
    rule_analysis_mode: str = 'independent',
    sample_survival_rate: float = 1.0,
) -> pd.DataFrame:
    """基于分箱表预测的好坏样本，构建基础报告（四段式结构）。

    报告结构（全量样本 > OUT-OUT规则 > OUT-OUT合计 > 剩余样本）：
    - 全量样本：全部 data 的好坏分布，坏样本率从 data 本身计算
    - OUT-OUT规则：每条规则命中的好坏分布
    - OUT-OUT合计：所有规则合并（reduce |）后的命中好坏分布
    - 剩余样本：未被任何规则命中的样本好坏分布

    预测逻辑：
    1. 对每个样本，根据其评分值在 bin_table 中查找对应分箱的坏样本率
    2. 坏样本数 = sum(bad_rate × count_in_bin)，好样本数 = 总样本 - 坏样本
    3. 多评分时按 score_weights 加权坏概率后求和
    4. 忽略 bin_table 的合计行（数据分布变化后已失效）

    :param data: 全部样本集
    :param rules_base: 基准拒绝规则集
    :param bin_table_result: {评分名: 分箱表}，由 _resolve_bin_table 返回
    :param score_map: {评分名: 实际列名}
    :param score_weights: 多评分权重（可选）
    :param amount: 金额字段（可选，用于口径判断）
    :param rule_analysis_mode: 规则分析模式。
        - 'independent'（默认）：每条规则独立应用到全量 data
        - 'sequential'：漏斗模式，每条规则在前一条的剩余样本上分析，
          根据规则命中（拒绝）减少样本量，未命中（通过）保持样本量。
    :param sample_survival_rate: 基准通过率（生产策略通过率，默认1.0）
    :return: 4-part 报告 DataFrame
    """
    n_total = len(data)
    rows = []

    # ── 预处理：计算每个样本在每个评分下的坏概率 ────────────────────────────
    score_bad_probs = {}  # score_name → prob Series
    for name, df_bin in bin_table_result.items():
        score_col = score_map[name]
        single_bad_col, _ = _extract_bad_rate_col(df_bin)
        score_bad_probs[name] = _compute_predicted_bad_prob(
            data, score_col, df_bin, single_bad_col
        )

    # ── 计算全量样本坏概率（从 data 本身，非 bin_table 合计）─────────────────
    if len(score_bad_probs) == 1:
        only_name = list(score_bad_probs.keys())[0]
        full_bad_probs = score_bad_probs[only_name]
    else:
        if score_weights:
            w_names = set(score_weights.keys())
            p_names = set(score_bad_probs.keys())
            if w_names != p_names:
                missing_w = p_names - w_names
                missing_p = w_names - p_names
                raise ValueError(
                    f"score_weights 与 bin_table_result 的评分名不匹配。"
                    f"weights 缺少：{missing_w}，bin_table 缺少：{missing_p}"
                )
            weights = {name: score_weights[name] for name in score_bad_probs}
        else:
            n_scores = len(score_bad_probs)
            weights = {name: 1.0 / n_scores for name in score_bad_probs}

        prob_sum = None
        for name, prob in score_bad_probs.items():
            w = weights[name]
            prob_sum = prob * w if prob_sum is None else prob_sum + prob * w

        full_bad_probs = prob_sum

    # 全量样本坏样本数（从 data 本身计算）
    full_total_bad = float(full_bad_probs.sum())
    full_total_bad = round(full_total_bad, 6)  # 消除浮点精度误差
    full_total_bad = max(0.0, min(full_total_bad, n_total))

    # 全量样本数 = data样本数 / sample_survival_rate
    # data是生产策略通过后的样本，sample_survival_rate是生产策略的通过率
    n_total_full = int(n_total / sample_survival_rate)
    # 全量坏样本数也需要按比例扩展到全量
    n_bad_full = full_total_bad * (n_total_full / n_total) if n_total > 0 else 0.0
    full_bad_rate = full_total_bad / n_total if n_total > 0 else 0.0

    rows.append(_make_report_row(
        '全量样本', 'OUT-OUT基准', full_total_bad, n_total,
        n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
        sample_survival_rate=sample_survival_rate,
    ))

    # ── 单规则命中统计 ───────────────────────────────────────────────────
    # 基准通过率，sequential 模式下会随规则链式递减
    current_pass_rate = sample_survival_rate
    if rules_base:
        if rule_analysis_mode == 'sequential':
            # 漏斗模式：每条规则在前一条拒绝后的剩余样本上分析
            # 当前存活样本 = 尚未被任何规则拒绝的样本
            current_remaining = pd.Series(True, index=data.index)
            # 基准通过率 = sample_survival_rate（生产策略通过率）
            current_pass_rate = sample_survival_rate

            for rule in rules_base:
                # 当前规则在当前存活样本范围内进行预测
                # 命中的样本 = 规则拒绝（predict=True）且当前存活
                # 未命中的样本 = 规则通过（predict=False）且当前存活
                rule_hit = rule.predict(data)
                current_rule_hit = rule_hit & current_remaining  # 仅限当前存活样本中命中

                n_hit = int(current_rule_hit.sum())
                n_good, n_bad = _calc_good_bad_from_prob(
                    full_bad_probs, current_rule_hit, n_total, n_hit
                )
                n_bad = round(n_bad, 1)
                rows.append(_make_report_row(
                    'OUT-OUT规则', rule.name, n_bad, n_hit,
                    rule_detail=rule.expr,
                    n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                    sample_survival_rate=current_pass_rate,
                ))

                # 更新存活样本：排除被当前规则拒绝的样本
                current_remaining = current_remaining & ~rule_hit
                # 更新基准通过率：当前通过率 = (当前通过样本数 - 拒绝样本数) / 全量样本数
                # 当前通过样本数 = n_total_full × current_pass_rate（来自上一步的通过样本，以全量计）
                # 拒绝样本数 = n_hit（被当前规则拒绝的样本数）
                base_pass_n = n_total_full * current_pass_rate
                current_pass_rate = max(0.0, (base_pass_n - n_hit) / n_total_full)

            # OUT-OUT合计（sequential 模式下等于所有被拒绝样本的累计）
            combined_mask = ~current_remaining
            n_combined_hit = int(combined_mask.sum())
            n_good_comb, n_bad_comb = _calc_good_bad_from_prob(
                full_bad_probs, combined_mask, n_total, n_combined_hit
            )
            n_bad_comb = round(n_bad_comb, 1)
            rows.append(_make_report_row(
                'OUT-OUT合计', 'OUT-OUT合计', n_bad_comb, n_combined_hit,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                sample_survival_rate=current_pass_rate,
            ))

        else:
            # independent 模式（默认）：每条规则独立应用到全量 data
            for rule in rules_base:
                mask = rule.predict(data)
                n_hit = int(mask.sum())
                n_good, n_bad = _calc_good_bad_from_prob(
                    full_bad_probs, mask, n_total, n_hit
                )
                n_bad = round(n_bad, 1)
                rows.append(_make_report_row(
                    'OUT-OUT规则', rule.name, n_bad, n_hit,
                    rule_detail=rule.expr,
                    n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                    sample_survival_rate=sample_survival_rate,
                ))

            # OUT-OUT 合计：合并所有规则
            combined_rule = reduce(lambda r1, r2: r1 | r2, rules_base)
            combined_mask = combined_rule.predict(data)
            n_combined_hit = int(combined_mask.sum())
            n_good_comb, n_bad_comb = _calc_good_bad_from_prob(
                full_bad_probs, combined_mask, n_total, n_combined_hit
            )
            n_bad_comb = round(n_bad_comb, 1)
            rows.append(_make_report_row(
                'OUT-OUT合计', 'OUT-OUT合计', n_bad_comb, n_combined_hit,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                sample_survival_rate=sample_survival_rate,
            ))

    # ── 剩余样本：未被任何规则命中的样本 ───────────────────────────────────
    if rules_base:
        combined_rule = reduce(lambda r1, r2: r1 | r2, rules_base)
        hit_mask = combined_rule.predict(data)
    else:
        hit_mask = pd.Series(False, index=data.index)
    remain_mask = ~hit_mask
    n_remain = int(remain_mask.sum())
    n_good_rem, n_bad_rem = _calc_good_bad_from_prob(
        full_bad_probs, remain_mask, n_total, n_remain
    )
    n_bad_rem = round(n_bad_rem, 1)
    # 剩余样本的通过率基准：在 sequential 模式下用最终累计通过率，independent 模式下用原始基准
    remain_pass_rate = current_pass_rate if rule_analysis_mode == 'sequential' else sample_survival_rate
    rows.append(_make_report_row(
        '剩余样本', '剩余样本', n_bad_rem, n_remain,
        n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
        sample_survival_rate=remain_pass_rate,
    ))

    return pd.DataFrame(rows), full_bad_probs


def _build_inout_report(
    in_remaining_data: pd.DataFrame,
    rules_out: List[Rule],
    bin_table_result: Dict[str, pd.DataFrame],
    score_map: Dict[str, str],
    score_weights: Optional[Dict[str, float]] = None,
    rule_analysis_mode: str = 'independent',
    n_total_full: int = 0,
    n_bad_full: float = 0.0,
    full_bad_rate: float = 0.0,
    sample_survival_rate: float = 1.0,
) -> pd.DataFrame:
    """构建IN-OUT置出分析报告。

    报告结构（IN-IN通过 > IN-OUT置出规则 > IN-OUT置出合计 > IN-IN通过）：
    - IN-IN通过样本：作为基准的全量（rules_out为空时即全部通过，否则为剩余通过样本）
    - IN-OUT置出规则：每条置出规则的拒绝效果
    - IN-OUT置出合计：所有置出规则合并后的总拒绝
    - IN-IN通过（最终）：经过所有置出规则后的剩余通过样本

    :param in_remaining_data: 经过OUT-OUT拒绝后的剩余样本（进入IN-OUT的样本）
    :param rules_out: 置出规则集
    :param bin_table_result: {评分名: 分箱表}
    :param score_map: {评分名: 实际列名}
    :param score_weights: 多评分权重（可选）
    :param rule_analysis_mode: 规则分析模式（independent/sequential）
    :param n_total_full: 全量样本总数（来自Step2，用于通过率计算）
    :param n_bad_full: 全量坏样本数（来自Step2）
    :param full_bad_rate: 全量坏样本率
    :param sample_survival_rate: 基准通过率（Step2结束后的累计通过率）
    :return: IN-OUT分析报告 DataFrame
    """
    n_total = len(in_remaining_data)
    rows = []

    # ── 计算in_remaining样本的坏概率 ──────────────────────────────────────
    score_bad_probs = {}
    for name, df_bin in bin_table_result.items():
        score_col = score_map[name]
        single_bad_col, _ = _extract_bad_rate_col(df_bin)
        score_bad_probs[name] = _compute_predicted_bad_prob(
            in_remaining_data, score_col, df_bin, single_bad_col
        )

    if len(score_bad_probs) == 1:
        only_name = list(score_bad_probs.keys())[0]
        full_bad_probs_inout = score_bad_probs[only_name]
    else:
        if score_weights:
            w_names = set(score_weights.keys())
            p_names = set(score_bad_probs.keys())
            if w_names != p_names:
                missing_w = p_names - w_names
                missing_p = w_names - p_names
                raise ValueError(
                    f"score_weights 与 bin_table_result 的评分名不匹配。"
                    f"weights 缺少：{missing_w}，bin_table 缺少：{missing_p}"
                )
            weights = {name: score_weights[name] for name in score_bad_probs}
        else:
            n_scores = len(score_bad_probs)
            weights = {name: 1.0 / n_scores for name in score_bad_probs}

        prob_sum = None
        for name, prob in score_bad_probs.items():
            w = weights[name]
            prob_sum = prob * w if prob_sum is None else prob_sum + prob * w
        full_bad_probs_inout = prob_sum

    # IN-IN通过样本数（来自Step2的剩余样本）
    in_total_bad = float(full_bad_probs_inout.sum())
    in_total_bad = round(in_total_bad, 6)
    in_total_bad = max(0.0, min(in_total_bad, n_total))

    # IN-IN通过行（基准）
    rows.append(_make_report_row(
        'IN-IN通过', 'IN-IN通过', in_total_bad, n_total,
        n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
        sample_survival_rate=sample_survival_rate,
    ))

    if not rules_out:
        return pd.DataFrame(rows)

    # ── 单规则命中统计 ───────────────────────────────────────────────────
    current_pass_rate = sample_survival_rate
    if rule_analysis_mode == 'sequential':
        current_remaining = pd.Series(True, index=in_remaining_data.index)
        for rule in rules_out:
            rule_hit = rule.predict(in_remaining_data)
            current_rule_hit = rule_hit & current_remaining

            n_hit = int(current_rule_hit.sum())
            n_good, n_bad = _calc_good_bad_from_prob(
                full_bad_probs_inout, current_rule_hit, n_total, n_hit
            )
            n_bad = round(n_bad, 1)
            rows.append(_make_report_row(
                'IN-OUT置出', rule.name, n_bad, n_hit,
                rule_detail=rule.expr,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                sample_survival_rate=current_pass_rate,
            ))

            current_remaining = current_remaining & ~rule_hit
            base_pass_n = n_total_full * current_pass_rate
            current_pass_rate = max(0.0, (base_pass_n - n_hit) / n_total_full)

        combined_mask = ~current_remaining
        n_combined_hit = int(combined_mask.sum())
        n_good_comb, n_bad_comb = _calc_good_bad_from_prob(
            full_bad_probs_inout, combined_mask, n_total, n_combined_hit
        )
        n_bad_comb = round(n_bad_comb, 1)
        rows.append(_make_report_row(
            'IN-OUT置出', 'IN-OUT置出合计', n_bad_comb, n_combined_hit,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            sample_survival_rate=current_pass_rate,
        ))
    else:
        for rule in rules_out:
            mask = rule.predict(in_remaining_data)
            n_hit = int(mask.sum())
            n_good, n_bad = _calc_good_bad_from_prob(
                full_bad_probs_inout, mask, n_total, n_hit
            )
            n_bad = round(n_bad, 1)
            rows.append(_make_report_row(
                'IN-OUT置出', rule.name, n_bad, n_hit,
                rule_detail=rule.expr,
                n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
                sample_survival_rate=sample_survival_rate,
            ))

        combined_rule = reduce(lambda r1, r2: r1 | r2, rules_out)
        combined_mask = combined_rule.predict(in_remaining_data)
        n_combined_hit = int(combined_mask.sum())
        n_good_comb, n_bad_comb = _calc_good_bad_from_prob(
            full_bad_probs_inout, combined_mask, n_total, n_combined_hit
        )
        n_bad_comb = round(n_bad_comb, 1)
        rows.append(_make_report_row(
            'IN-OUT置出', 'IN-OUT置出合计', n_bad_comb, n_combined_hit,
            n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
            sample_survival_rate=sample_survival_rate,
        ))

    # ── IN-IN通过（最终）：未被任何置出规则拒绝的样本 ──────────────────────
    if rules_out:
        combined_rule = reduce(lambda r1, r2: r1 | r2, rules_out)
        hit_mask = combined_rule.predict(in_remaining_data)
    else:
        hit_mask = pd.Series(False, index=in_remaining_data.index)
    remain_mask = ~hit_mask
    n_remain = int(remain_mask.sum())
    n_good_rem, n_bad_rem = _calc_good_bad_from_prob(
        full_bad_probs_inout, remain_mask, n_total, n_remain
    )
    n_bad_rem = round(n_bad_rem, 1)
    # IN-IN通过基准：sequential模式用最终累计通过率，independent模式用原始基准
    in_pass_rate = current_pass_rate if rule_analysis_mode == 'sequential' else sample_survival_rate
    rows.append(_make_report_row(
        'IN-IN通过', 'IN-IN通过', n_bad_rem, n_remain,
        n_total_full=n_total_full, n_bad_full=n_bad_full, full_bad_rate=full_bad_rate,
        sample_survival_rate=in_pass_rate,
    ))

    return pd.DataFrame(rows)


def _calc_good_bad_from_prob(
    full_bad_probs: pd.Series,
    mask: pd.Series,
    n_total: int,
    n_hit: int,
) -> Tuple[float, float]:
    """根据坏概率和命中掩码计算好/坏样本数。

    对于命中的样本：累加其坏概率得到预测坏样本数
    好样本数 = 命中数 - 坏样本数

    :param full_bad_probs: 全量样本的预测坏概率（与 data 行对齐）
    :param mask: 命中掩码（True = 命中）
    :param n_total: 全量样本总数
    :param n_hit: 命中样本数
    :return: (好样本数, 坏样本数)
    """
    if n_hit == 0:
        return 0.0, 0.0
    hit_probs = full_bad_probs[mask.values]
    n_bad = float(hit_probs.sum())
    n_bad = max(0.0, min(n_bad, float(n_hit)))
    n_good = float(n_hit) - n_bad
    return float(n_good), float(n_bad)


def _make_report_row(
    rule_class: str,
    rule_name: str,
    n_bad: float,
    n_total: int,
    rule_detail: Optional[str] = None,
    n_total_full: Optional[int] = None,
    n_bad_full: Optional[float] = None,
    full_bad_rate: Optional[float] = None,
    sample_survival_rate: float = 1.0,
) -> dict:
    """构建基础报告的一行。

    所有指标基于预测的坏样本数和好样本数计算。

    :param rule_class: 规则分类（如 '全量样本'、'OUT-OUT规则'、'OUT-OUT合计'、'剩余样本'）
    :param rule_name: 规则名称
    :param n_bad: 预测坏样本数（已round到1位小数）
    :param n_total: 当前行总样本数
    :param rule_detail: 规则表达式（rule.expr），全量样本和剩余样本传 None
    :param n_total_full: 全量样本总数（用于计算样本占比）
    :param n_bad_full: 全量样本预测坏样本总数（用于计算LIFT、坏账改善、风险拒绝比）
    :param full_bad_rate: 全量样本坏样本率（用于计算LIFT、坏账改善）
    :param sample_survival_rate: 基准通过率（生产策略通过率，默认1.0）
    :return: 指标字典
    """
    # 好样本数 = 总样本数 - 预测坏样本数
    # n_bad_rounded 与 n_good 使用同一基准，保证 n_good + n_bad_rounded = n_total
    n_bad_rounded = round(n_bad)
    n_good = n_total - n_bad_rounded

    n_total_f = float(n_total)
    n_total_full_val = n_total_full if n_total_full is not None else n_total

    # 坏样本率 = 预测坏样本数(整化) / 当前行总样本数
    bad_rate = n_bad_rounded / n_total_f if n_total_f > 0 else 0.0

    # 基础占比（相对于全量样本）
    p_total = n_total_f / n_total_full_val if n_total_full_val > 0 else 0.0
    p_good = n_good / n_total_f if n_total_f > 0 else 0.0
    p_bad = bad_rate

    # LIFT = 当前行坏账率 / 全量样本坏账率（反映该规则群体的风险相对水平）
    lift = 0.0
    if full_bad_rate is not None and full_bad_rate > 0:
        lift = bad_rate / full_bad_rate

    # 坏账改善 = (全量坏样本率 - 拒绝后剩余样本坏样本率) / 全量坏样本率
    # 拒绝后剩余样本坏样本率 = (全量坏样本数 - 当前行坏样本数) / (全量样本数 - 当前行样本数)
    bad_improve = 0.0
    if full_bad_rate is not None and full_bad_rate > 0:
        n_remaining = n_total_full_val - n_total
        n_bad_remaining = (n_bad_full if n_bad_full is not None else 0.0) - n_bad_rounded
        remaining_bad_rate = n_bad_remaining / n_remaining if n_remaining > 0 else 0.0
        bad_improve = (full_bad_rate - remaining_bad_rate) / full_bad_rate

    # 风险拒绝比 = 坏账改善 / 当前箱样本占比
    # 反映"每拒绝1%样本能带来多少坏账改善"
    risk_reject = 0.0
    if p_total > 0 and full_bad_rate is not None and full_bad_rate > 0:
        risk_reject = bad_improve / p_total

    # 准确率、精确率、召回率、F1
    # 基于预测值（TP=预测坏, TN=预测好），不使用真实标签
    # TP = n_bad_rounded（命中的预测坏样本）
    # FP = n_good（命中的预测好样本）
    # FN = 全量预测坏 - 当前行预测坏 = n_bad_full - n_bad_rounded
    # TN = 全量预测好 - 当前行预测好 = (n_total_full - n_bad_full) - n_good
    tp = n_bad_rounded
    fp = n_good
    fn = 0.0
    tn = 0.0
    if n_bad_full is not None:
        fn = max(0.0, n_bad_full - n_bad_rounded)
    if n_total_full is not None and n_bad_full is not None:
        total_good_full = n_total_full - n_bad_full
        tn = max(0.0, total_good_full - n_good)

    accuracy = 0.0
    precision_val = 0.0
    recall_val = 0.0
    total_pos = tp + fp
    total_actual = tp + fn
    if (tp + fp + fn + tn) > 0:
        accuracy = (tp + tn) / (tp + fp + fn + tn)
    if total_pos > 0:
        precision_val = tp / total_pos
    if total_actual > 0:
        recall_val = tp / total_actual

    f1 = 0.0
    if (precision_val + recall_val) > 0:
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val)

    # ── 通过率分析 ───────────────────────────────────────────────────────
    # 通过率 = (基准通过样本数 - 当前行拒绝样本数) / 全量样本数
    # 基准通过样本数 = 全量样本数 × sample_survival_rate（生产策略通过率）
    # 当前行拒绝样本数 = 当前行样本数（当前行 = 被拒绝的样本）
    base_pass_n = n_total_full_val * sample_survival_rate
    current_pass_rate = (base_pass_n - n_total_f) / n_total_full_val if n_total_full_val > 0 else 0.0
    current_pass_rate = max(0.0, min(current_pass_rate, 1.0))

    # 通过率%(绝对值) = 通过率 × 100
    pass_rate_abs = current_pass_rate * 100

    # 通过率%(相对值) = (当前通过率 - 基准通过率) / 基准通过率 × 100
    base_pass_rate = sample_survival_rate
    pass_rate_rel = 0.0
    if base_pass_rate > 0:
        pass_rate_rel = (current_pass_rate - base_pass_rate) / base_pass_rate * 100

    # 调整方向：仅规则行有值，全量样本/剩余样本/合计为空
    direction = ''
    if rule_class == 'OUT-OUT规则':
        if current_pass_rate < base_pass_rate - 1e-10:
            direction = '收紧'
        elif current_pass_rate > base_pass_rate + 1e-10:
            direction = '放松'
        else:
            direction = '不变'

    return {
        '规则分类': rule_class,
        '指标名称': rule_name,
        '规则详情': rule_detail if rule_detail else '',
        '分箱': '',
        '样本总数': int(n_total),
        '样本占比': p_total,
        '好样本数': int(n_good),
        '好样本占比': p_good,
        '坏样本数': int(n_bad_rounded),
        '坏样本占比': p_bad,
        '坏样本率': bad_rate,
        'LIFT值': lift,
        '坏账改善': bad_improve,
        '风险拒绝比': risk_reject,
        '准确率': accuracy,
        '精确率': precision_val,
        '召回率': recall_val,
        'F1分数': f1,
        '通过率': current_pass_rate,
        '通过率%(绝对值)': pass_rate_abs,
        '通过率%(相对值)': pass_rate_rel,
        '调整方向': direction,
    }
