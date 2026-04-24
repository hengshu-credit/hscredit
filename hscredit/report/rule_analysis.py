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