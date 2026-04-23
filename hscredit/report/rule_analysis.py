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
    # 0. 参数预处理
    # ------------------------------------------------------------------
    df = df.copy()

    # 统一 score 参数为 Dict 格式
    if isinstance(score, str):
        score_map = {'_default': score}
    else:
        score_map = score

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
    df['_swap_in'] = np.nan   # 模型通过标记
    df['_swap_rule'] = np.nan  # 规则通过标记

    # 1.1 标记模型通过（所有评分均达到阈值）
    # 阈值策略：取各评分在中位数以上的样本为"通过"
    score_pass_mask = pd.Series(True, index=df.index)
    for name, col in score_map.items():
        if col in df.columns:
            median_val = df[col].median()
            score_pass_mask = score_pass_mask & (df[col] >= median_val)
        else:
            raise ValueError(f"评分列 '{col}' 不在数据集中")
    df['_swap_in'] = score_pass_mask.astype(int)

    # 1.2 标记规则通过（rules_in 命中即通过；rules_out 命中即拒绝）
    rule_in_hit = pd.Series(False, index=df.index)
    for r in rules_in:
        rule_in_hit = rule_in_hit | r.predict(df)
    df['_swap_rule'] = rule_in_hit.astype(int)

    # 若有 rules_out，叠加拒绝标记
    if rules_out:
        rule_out_hit = pd.Series(False, index=df.index)
        for r in rules_out:
            rule_out_hit = rule_out_hit | r.predict(df)
        df.loc[rule_out_hit, '_swap_rule'] = 0

    # 1.3 若有 rules_base，标记 out_out（基准拒绝）
    out_out_mask = pd.Series(False, index=df.index)
    if rules_base:
        for r in rules_base:
            out_out_mask = out_out_mask | r.predict(df)

    # 1.4 构建四象限
    def _get_quadrant(row):
        in_model = bool(row['_swap_in'])
        in_rule = bool(row['_swap_rule'])
        if out_out_mask is not None and out_out_mask.get(row.name, False):
            return 'out_out'
        if in_model and in_rule:
            return 'in_in'
        elif in_model and not in_rule:
            return 'in_out'
        elif not in_model and in_rule:
            return 'out_in'
        else:
            return 'out_out'

    df['_swap_quadrant'] = df.apply(_get_quadrant, axis=1)

    # ------------------------------------------------------------------
    # 2. 构建评分→逾期率映射（分箱表或自动生成）
    # ------------------------------------------------------------------
    if bin_table is not None:
        # 复用 OverduePredictor 从现成分箱表提取逾期率
        predictor = OverduePredictor(feature='_swap_score_combined')
        predictor.fit(bin_table)
        # 构建 {分箱标签: 逾期率} 字典
        bin_rates = predictor.bin_rates_.get('_default', {})
    else:
        # 自动分箱：从 df（含 target/overdue+dpds）生成分箱表
        predictor = OverduePredictor(
            feature=list(score_map.values())[0],
            target=target,
            overdue=overdue,
            dpds=dpds,
            method='quantile',
            max_n_bins=10,
            missing_separate=True,
        )
        predictor.fit(df)
        bin_rates = predictor.bin_rates_.get('_default', {})

    # 多模型加权综合评分（归一化）
    df['_swap_score_combined'] = 0.0
    for name, col in score_map.items():
        w = score_weights[name] / total_weight
        # 归一化到 [0,1]（按 max-min）
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f'_tmp_norm_{name}'] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f'_tmp_norm_{name}'] = 0.5
        df['_swap_score_combined'] += w * df[f'_tmp_norm_{name}']

    # ------------------------------------------------------------------
    # 3. 计算各象限逾期率
    # ------------------------------------------------------------------
    def _calc_bad_rate(q_df: pd.DataFrame, amount_col: Optional[str] = None) -> float:
        """计算象限的坏样本率（加权/金额口径）。"""
        if q_df.empty:
            return 0.0
        if target is not None:
            col = target
        elif overdue is not None:
            mob_col = overdue[0] if isinstance(overdue, list) else overdue
            dpd_val = dpds[0] if isinstance(dpds, list) else dpds
            col = mob_col
            if col in q_df.columns:
                y = (q_df[col] > dpd_val).astype(int)
                return float(y.mean())
        if col in q_df.columns:
            return float(q_df[col].mean())
        return 0.0

    # 为 out_in 应用 uplift
    _overall_bad_rate = _calc_bad_rate(df[df['_swap_quadrant'] == 'in_in'], amount)
    if _overall_bad_rate <= 0:
        _overall_bad_rate = df[target].mean() if target in df.columns else 0.05

    quadrants = ['in_in', 'in_out', 'out_in', 'out_out']
    if reverse_order:
        quadrants = ['out_out', 'out_in', 'in_out', 'in_in']

    # ------------------------------------------------------------------
    # 4. 构建 Swap Summary 表
    # ------------------------------------------------------------------
    summary_rows = []
    for q in quadrants:
        q_df = df[df['_swap_quadrant'] == q]
        n = len(q_df)
        if n == 0:
            continue

        bad_rate = _calc_bad_rate(q_df, amount)
        # 对 out_in 应用 uplift
        if q == 'out_in':
            bad_rate = min(bad_rate * out_in_uplift, 1.0)

        lift = bad_rate / _overall_bad_rate if _overall_bad_rate > 0 else 0.0

        n_bad = int(n * bad_rate)
        n_good = n - n_bad
        good_rate = 1 - bad_rate

        if amount is not None and amount in q_df.columns:
            amt_total = q_df[amount].sum()
            amt_bad = amt_total * bad_rate
        else:
            amt_total = None
            amt_bad = None

        summary_rows.append({
            '象限': q,
            '样本数': n,
            '样本占比': n / len(df),
            '好样本数': n_good,
            '好样本占比': good_rate,
            '坏样本数': n_bad,
            '坏样本占比': bad_rate,
            '坏样本率': bad_rate,
            'LIFT': lift,
            '金额总数': amt_total,
            '预估坏金额': amt_bad,
        })

    swap_summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # 5. 构建 Swap Pipeline 表（Rule.report() 风格 + 三列额外指标）
    # ------------------------------------------------------------------
    # 7步顺序（reverse_order=False）:
    #   全部样本 → OUT-OUT拒绝 → 剩余 → IN-OUT置出 → 剩余 → IN-IN通过 → OUT-IN置入
    # reverse_order=True 时逆序遍历，先展示置入效果，逐步到全量样本
    n_total = len(df) / sample_survival_rate  # 还原全量样本数（支持幸存比例缩放）
    scale = 1.0 / sample_survival_rate if sample_survival_rate > 0 else 1.0
    N_total = int(n_total)

    # 归一化各象限到全量口径
    def _quadrant_n(name):
        return int(len(df[df['_swap_quadrant'] == name]))

    n_ii = _quadrant_n('in_in')
    n_io = _quadrant_n('in_out')
    n_oi = _quadrant_n('out_in')
    n_oo = _quadrant_n('out_out')
    N_oo = int(n_oo * scale)
    N_io = int(n_io * scale)
    N_ii = int(n_ii * scale)
    N_oi = int(n_oi * scale)

    # 使用 OverduePredictor.predict() 获取每个样本的预测逾期率
    # _swap_score_combined 在前面多模型加权综合评分步骤中已创建于 df
    _pred_df = predictor.predict(df[['_swap_score_combined']])

    # 全量预测逾期率（用于 全部样本 步骤的 lift 基准）
    _overall_pred_bad = float(_pred_df.mean()) if len(_pred_df) > 0 else 0.0
    if _overall_pred_bad <= 0:
        _overall_pred_bad = df[target].mean() if target in df.columns else 0.05

    # Extra columns to add to each step
    _extra_cols = ['调整方向', '通过率', '通过率(绝对值)', '通过率(相对值)']

    def _build_full_step(step_name, quadrant, step_n, cum_remain, is_rem_step=False, adj_dir='-'):
        """构建完整一步（含命中/未命中/合计三行 + extra列）。"""
        if step_n == 0:
            return pd.DataFrame()

        cur_pass_rate = cum_remain / N_total if N_total > 0 else 0.0
        extra = {
            '调整方向': adj_dir,
            '通过率': cur_pass_rate,
            '通过率(绝对值)': cur_pass_rate,
            '通过率(相对值)': cur_pass_rate,
        }

        if is_rem_step:
            rem_mask = df['_swap_quadrant'] == quadrant
            rem_pred = _pred_df[rem_mask]
            step_bad_rate = float(rem_pred.mean()) if len(rem_pred) > 0 else 0.0
            lift_val = step_bad_rate / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
            step_rows = [
                {**{
                    '规则分类': quadrant,
                    '指标名称': step_name,
                    '分箱': '命中',
                    '样本总数': step_n,
                    '样本占比': step_n / N_total,
                    '好样本数': int(step_n * (1 - step_bad_rate)),
                    '好样本占比': 1 - step_bad_rate,
                    '坏样本数': int(step_n * step_bad_rate),
                    '坏样本占比': step_bad_rate,
                    '坏样本率': step_bad_rate,
                    'LIFT值': lift_val,
                    '坏账改善': 0.0,
                    '风险拒绝比': 0.0,
                    '准确率': 0.0,
                    '精确率': 0.0,
                    '召回率': 0.0,
                    'F1分数': 0.0,
                }, **extra},
                {**{
                    '规则分类': quadrant,
                    '指标名称': step_name,
                    '分箱': '未命中',
                    '样本总数': 0,
                    '样本占比': 0.0,
                    '好样本数': 0,
                    '好样本占比': 0.0,
                    '坏样本数': 0,
                    '坏样本占比': 0.0,
                    '坏样本率': 0.0,
                    'LIFT值': 0.0,
                    '坏账改善': 0.0,
                    '风险拒绝比': 0.0,
                    '准确率': 0.0,
                    '精确率': 0.0,
                    '召回率': 0.0,
                    'F1分数': 0.0,
                }, **extra},
            ]
            table = pd.DataFrame(step_rows)
        else:
            quad_mask = df['_swap_quadrant'] == quadrant
            quad_pred = _pred_df[quad_mask]
            step_bad_rate = float(quad_pred.mean()) if len(quad_pred) > 0 else 0.0
            if quadrant == 'out_in':
                step_bad_rate = min(step_bad_rate * out_in_uplift, 1.0)
            lift_hit = step_bad_rate / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
            step_rows = [
                {**{
                    '规则分类': quadrant,
                    '指标名称': step_name,
                    '分箱': '命中',
                    '样本总数': step_n,
                    '样本占比': step_n / N_total,
                    '好样本数': int(step_n * (1 - step_bad_rate)),
                    '好样本占比': 1 - step_bad_rate,
                    '坏样本数': int(step_n * step_bad_rate),
                    '坏样本占比': step_bad_rate,
                    '坏样本率': step_bad_rate,
                    'LIFT值': lift_hit,
                    '坏账改善': 0.0,
                    '风险拒绝比': 0.0,
                    '准确率': 0.0,
                    '精确率': 0.0,
                    '召回率': 0.0,
                    'F1分数': 0.0,
                }, **extra},
                {**{
                    '规则分类': quadrant,
                    '指标名称': step_name,
                    '分箱': '未命中',
                    '样本总数': 0,
                    '样本占比': 0.0,
                    '好样本数': 0,
                    '好样本占比': 0.0,
                    '坏样本数': 0,
                    '坏样本占比': 0.0,
                    '坏样本率': 0.0,
                    'LIFT值': 0.0,
                    '坏账改善': 0.0,
                    '风险拒绝比': 0.0,
                    '准确率': 0.0,
                    '精确率': 0.0,
                    '召回率': 0.0,
                    'F1分数': 0.0,
                }, **extra},
            ]
            table = pd.DataFrame(step_rows)

        table_total = {
            **{'规则分类': quadrant, '指标名称': step_name, '分箱': '合计',
               '样本总数': step_n, '样本占比': 1.0,
               '好样本数': int(step_n * (1 - step_bad_rate)),
               '好样本占比': 1.0, '坏样本数': int(step_n * step_bad_rate),
               '坏样本占比': 1.0, '坏样本率': step_bad_rate,
               'LIFT值': 1.0, '坏账改善': 0.0, '风险拒绝比': 0.0,
               '准确率': 0.0, '精确率': 0.0, '召回率': 0.0, 'F1分数': 0.0},
            **extra,
        }
        return pd.concat([table, pd.DataFrame([table_total])], ignore_index=True)

    def _build_total_step(step_name, step_n, cum_remain):
        """构建 全部样本 步骤的表（用 predictor 计算预测逾期率）。"""
        if step_n == 0:
            return pd.DataFrame()
        step_bad_rate = _overall_pred_bad
        lift_hit = step_bad_rate / _overall_pred_bad if _overall_pred_bad > 0 else 1.0
        cur_pass_rate = cum_remain / N_total if N_total > 0 else 0.0
        extra = {
            '调整方向': '-',
            '通过率': cur_pass_rate,
            '通过率(绝对值)': cur_pass_rate,
            '通过率(相对值)': cur_pass_rate,
        }
        step_rows = [
            {**{'规则分类': '_all_', '指标名称': step_name, '分箱': '命中',
                '样本总数': step_n, '样本占比': 1.0,
                '好样本数': int(step_n * (1 - step_bad_rate)),
                '好样本占比': 1 - step_bad_rate,
                '坏样本数': int(step_n * step_bad_rate),
                '坏样本占比': step_bad_rate,
                '坏样本率': step_bad_rate,
                'LIFT值': lift_hit,
                '坏账改善': 0.0, '风险拒绝比': 0.0,
                '准确率': 0.0, '精确率': 0.0, '召回率': 0.0, 'F1分数': 0.0},
             **extra},
            {**{'规则分类': '_all_', '指标名称': step_name, '分箱': '未命中',
                '样本总数': 0, '样本占比': 0.0,
                '好样本数': 0, '好样本占比': 0.0,
                '坏样本数': 0, '坏样本占比': 0.0,
                '坏样本率': 0.0, 'LIFT值': 0.0,
                '坏账改善': 0.0, '风险拒绝比': 0.0,
                '准确率': 0.0, '精确率': 0.0, '召回率': 0.0, 'F1分数': 0.0},
             **extra},
        ]
        table = pd.DataFrame(step_rows)
        table_total = {
            **{'规则分类': '_all_', '指标名称': step_name, '分箱': '合计',
               '样本总数': step_n, '样本占比': 1.0,
               '好样本数': int(step_n * (1 - step_bad_rate)),
               '好样本占比': 1.0, '坏样本数': int(step_n * step_bad_rate),
               '坏样本占比': 1.0, '坏样本率': step_bad_rate,
               'LIFT值': 1.0, '坏账改善': 0.0, '风险拒绝比': 0.0,
               '准确率': 0.0, '精确率': 0.0, '召回率': 0.0, 'F1分数': 0.0},
            **extra,
        }
        return pd.concat([table, pd.DataFrame([table_total])], ignore_index=True)

    # 步骤定义: (quadrant, step_name, is_rem_step, adj_direction)
    # 全部样本步骤用 _ALL_ sentinel 值表示
    # 注意：两个"剩余样本"步骤使用不同 step_name，避免 cum_map 键冲突
    if not reverse_order:
        step_defs = [
            ('_all_',    '全部样本',             None),
            ('out_out',  'OUT-OUT 拒绝样本',      None),
            ('_rem_oo',  'OUT-OUT后剩余样本',     'out_out'),
            ('in_out',   'IN-OUT 置出样本',       None),
            ('_rem_io',  'IN-OUT后剩余样本',      'in_out'),
            ('in_in',    'IN-IN 通过样本',        None),
            ('out_in',   'OUT-IN 置入样本',       None),
        ]
        adj_dirs = {
            '全部样本': '-',
            'OUT-OUT 拒绝样本': '收紧',
            'OUT-OUT后剩余样本': '-',
            'IN-OUT 置出样本': '释放',
            'IN-OUT后剩余样本': '-',
            'IN-IN 通过样本': '释放',
            'OUT-IN 置入样本': '收紧',
        }
    else:
        step_defs = [
            ('out_in',   'OUT-IN 置入样本',       None),
            ('in_in',    'IN-IN 通过样本',         None),
            ('_rem_ii',  'IN-IN后剩余样本',        'in_in'),
            ('in_out',   'IN-OUT 置出样本',        None),
            ('_rem_io2', 'IN-OUT后剩余样本',       'in_out'),
            ('out_out',  'OUT-OUT 拒绝样本',       None),
            ('_all_',    '全部样本',               None),
        ]
        adj_dirs = {
            '全部样本': '-',
            'OUT-OUT 拒绝样本': '收紧',
            'IN-OUT 置出样本': '释放',
            'IN-IN后剩余样本': '-',
            'IN-OUT后剩余样本': '-',
            'IN-IN 通过样本': '释放',
            'OUT-IN 置入样本': '收紧',
        }

    # 预计算每个象限的样本数（全量口径）
    quad_counts = {q: int((df['_swap_quadrant'] == q).sum() * scale) for q in ['in_in', 'in_out', 'out_in', 'out_out']}
    N_ii = quad_counts['in_in']
    N_io = quad_counts['in_out']
    N_oi = quad_counts['out_in']
    N_oo = quad_counts['out_out']

    # 正序累计剩余（用于计算通过率）
    # 各 step 的 cum_remain = 经过该步骤之前所有步骤后的剩余样本数
    cum_map_normal = {
        '全部样本': N_total,
        'OUT-OUT 拒绝样本': N_total,
        'OUT-OUT后剩余样本': N_total - N_oo,
        'IN-OUT 置出样本': N_total - N_oo,
        'IN-OUT后剩余样本': N_total - N_oo - N_io,
        'IN-IN 通过样本': N_total - N_oo - N_io - N_ii,
        'OUT-IN 置入样本': N_total - N_oo - N_io - N_ii - N_oi,
    }
    cum_map_reverse = {
        '全部样本': N_total,
        'OUT-IN 置入样本': N_total,
        'IN-IN 通过样本': N_total - N_oi,
        'IN-IN后剩余样本': N_oi + N_ii,
        'IN-OUT 置出样本': N_oi + N_ii,
        'IN-OUT后剩余样本': N_oi + N_ii + N_io,
        'OUT-OUT 拒绝样本': N_oi + N_ii + N_io + N_oo,
    }

    cum_map = cum_map_reverse if reverse_order else cum_map_normal

    pipeline_tables = []
    for quadrant, step_name, rem_q in step_defs:
        if quadrant == '_all_':
            step_n = N_total
            cum_remain = cum_map.get(step_name, N_total)
            t = _build_total_step(step_name, step_n, cum_remain)
        elif rem_q is not None:
            # 剩余样本步骤
            cum_remain = cum_map.get(step_name, N_total)
            step_n = cum_remain
            t = _build_full_step(step_name, rem_q, step_n, cum_remain, is_rem_step=True, adj_dir='-')
        else:
            cum_remain = cum_map.get(step_name, N_total)
            step_n = quad_counts.get(quadrant, 0)
            adj_dir = adj_dirs.get(step_name, '-')
            t = _build_full_step(step_name, quadrant, step_n, cum_remain, is_rem_step=False, adj_dir=adj_dir)

        if len(t) == 0:
            continue

        # 更新 extra 列
        for col in _extra_cols:
            if col not in t.columns:
                t[col] = '-'
        pipeline_tables.append(t)

    swap_pipeline = pd.concat(pipeline_tables, ignore_index=True)

    # ------------------------------------------------------------------
    # 6. 构建 Swap Result 表（置换对比与业务增益）
    # ------------------------------------------------------------------
    pass_rate_before = N_ii / N_total if N_total > 0 else 0.0
    pass_rate_after = (N_ii + N_oi) / N_total if N_total > 0 else 0.0

    # 从 predictor 获取 in_in 的预测不良率
    ii_mask = df['_swap_quadrant'] == 'in_in'
    bad_rate_before = float(_pred_df[ii_mask].mean()) if ii_mask.sum() > 0 else _overall_pred_bad
    oi_mask = df['_swap_quadrant'] == 'out_in'
    bad_rate_after_raw = float(_pred_df[oi_mask].mean()) if oi_mask.sum() > 0 else bad_rate_before
    bad_rate_after = min(bad_rate_after_raw * out_in_uplift, 1.0)

    loan_increase_abs = N_oi
    loan_increase_rel = N_oi / N_ii if N_ii > 0 else 0.0

    bad_in_out = int(N_io * bad_rate_before)
    bad_out_in = int(N_oi * bad_rate_before * out_in_uplift)
    risk_delta_abs = bad_out_in - bad_in_out
    risk_delta_rel = risk_delta_abs / N_ii if N_ii > 0 else 0.0

    swap_result_rows = [
        {'指标': '通过率变化', '变化前': pass_rate_before, '变化后': pass_rate_after,
         '绝对变化': pass_rate_after - pass_rate_before,
         '相对变化': (pass_rate_after - pass_rate_before) / max(pass_rate_before, 1e-9)},
        {'指标': '逾期率变化', '变化前': bad_rate_before, '变化后': bad_rate_after,
         '绝对变化': bad_rate_after - bad_rate_before,
         '相对变化': (bad_rate_after - bad_rate_before) / max(bad_rate_before, 1e-9)},
        {'指标': '风险上浮系数', '变化前': 1.0, '变化后': out_in_uplift,
         '绝对变化': out_in_uplift - 1.0, '相对变化': out_in_uplift - 1.0},
        {'指标': '放款增量（绝对）', '变化前': 0, '变化后': loan_increase_abs,
         '绝对变化': loan_increase_abs, '相对变化': loan_increase_rel},
        {'指标': '放款增量（相对）', '变化前': 0, '变化后': loan_increase_rel,
         '绝对变化': loan_increase_rel, '相对变化': loan_increase_rel},
        {'指标': '坏样本变化（绝对）', '变化前': 0, '变化后': risk_delta_abs,
         '绝对变化': risk_delta_abs, '相对变化': risk_delta_abs / max(N_ii, 1)},
        {'指标': '坏样本变化（相对）', '变化前': 0, '变化后': risk_delta_rel,
         '绝对变化': risk_delta_rel, '相对变化': risk_delta_rel},
        {'指标': '样本集幸存比例', '变化前': sample_survival_rate, '变化后': sample_survival_rate,
         '绝对变化': 0.0, '相对变化': 0.0},
    ]

    swap_result = pd.DataFrame(swap_result_rows)

    # ------------------------------------------------------------------
    # 6. 构建 Swap Result 表（置换对比与业务增益）
    # ------------------------------------------------------------------
    # 使用 section 5 中已归一化到全量口径的计数（N_ii, N_io, N_oi, N_oo, N_total）
    pass_rate_before = N_ii / N_total if N_total > 0 else 0.0   # 仅 IN-IN 通过
    pass_rate_after = (N_ii + N_oi) / N_total if N_total > 0 else 0.0

    # bad_rate_before 从 df 的 in_in 部分计算（坏样本率不受样本数缩放影响）
    bad_rate_before = _calc_bad_rate(df[df['_swap_quadrant'] == 'in_in'], amount)
    bad_rate_after = (
        N_ii * bad_rate_before + N_oi * min(bad_rate_before * out_in_uplift, 1.0)
    ) / (N_ii + N_oi) if (N_ii + N_oi) > 0 else 0.0

    # 放款增量（基于全量口径）
    loan_increase_abs = N_oi
    loan_increase_rel = N_oi / N_ii if N_ii > 0 else 0.0

    # 坏样本增量（基于全量口径）
    bad_in_out = int(N_io * bad_rate_before)
    bad_out_in = int(N_oi * bad_rate_before * out_in_uplift)
    risk_delta_abs = bad_out_in - bad_in_out
    risk_delta_rel = risk_delta_abs / N_ii if N_ii > 0 else 0.0

    swap_result_rows = [
        {'指标': '通过率变化', '变化前': pass_rate_before, '变化后': pass_rate_after,
         '绝对变化': pass_rate_after - pass_rate_before,
         '相对变化': (pass_rate_after - pass_rate_before) / max(pass_rate_before, 1e-9)},
        {'指标': '逾期率变化', '变化前': bad_rate_before, '变化后': bad_rate_after,
         '绝对变化': bad_rate_after - bad_rate_before,
         '相对变化': (bad_rate_after - bad_rate_before) / max(bad_rate_before, 1e-9)},
        {'指标': '风险上浮系数', '变化前': 1.0, '变化后': out_in_uplift,
         '绝对变化': out_in_uplift - 1.0, '相对变化': out_in_uplift - 1.0},
        {'指标': '放款增量（绝对）', '变化前': 0, '变化后': loan_increase_abs,
         '绝对变化': loan_increase_abs, '相对变化': loan_increase_rel},
        {'指标': '放款增量（相对）', '变化前': 0, '变化后': loan_increase_rel,
         '绝对变化': loan_increase_rel, '相对变化': loan_increase_rel},
        {'指标': '坏样本变化（绝对）', '变化前': 0, '变化后': risk_delta_abs,
         '绝对变化': risk_delta_abs, '相对变化': risk_delta_abs / max(N_ii, 1)},
        {'指标': '坏样本变化（相对）', '变化前': 0, '变化后': risk_delta_rel,
         '绝对变化': risk_delta_rel, '相对变化': risk_delta_rel},
        {'指标': '样本集幸存比例', '变化前': sample_survival_rate, '变化后': sample_survival_rate,
         '绝对变化': 0.0, '相对变化': 0.0},
    ]

    swap_result = pd.DataFrame(swap_result_rows)

    # ------------------------------------------------------------------
    # 7. 清理临时列
    # ------------------------------------------------------------------
    drop_cols = [c for c in df.columns if c.startswith('_swap') or c.startswith('_tmp_norm')]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return {
        'swap_summary': swap_summary,
        'swap_pipeline': swap_pipeline,
        'swap_result': swap_result,
    }