"""策略分析模块.

提供从模型评分到业务策略的链路分析工具，
包括通过率-坏率权衡、策略仿真、Vintage、
滚动率矩阵、标签泄露检测以及多标签相关性分析。

主要函数:
- approval_badrate_tradeoff: 通过率 vs 坏率权衡分析
- score_strategy_simulation: 评分阈值策略仿真
- vintage_performance_summary: Vintage 账龄表汇总
- roll_rate_matrix: DPD 滚动率矩阵
- label_leakage_check: 标签泄露检测
- multi_label_correlation: 多标签相关性矩阵
"""

import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from .utils import validate_dataframe


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _check_binary_target(y: pd.Series, name: str = 'target') -> None:
    """检查是否为 0/1 二值标签."""
    uniq = set(y.dropna().unique())
    if not uniq.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(f"'{name}' 须为 0/1 二值变量，当前唯一值: {uniq}")


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom != 0 else np.nan


# ---------------------------------------------------------------------------
# 1. approval_badrate_tradeoff
# ---------------------------------------------------------------------------

def approval_badrate_tradeoff(
    y_true: pd.Series,
    score: pd.Series,
    n_points: int = 100,
    score_low_risk: str = 'high',
) -> pd.DataFrame:
    """通过率 vs 坏率权衡分析.

    生成「通过率 - 坏率」权衡曲线数据，每行对应一个评分阈值，
    输出该阈值下的通过率、拒绝率、通过人群坏率、拒绝人群坏率
    以及 KS 值，用于向业务方解释策略调整影响。

    :param y_true: 真实标签序列（0/1）
    :param score: 模型评分序列（越高越优质）
    :param n_points: 阈值点数量，默认 100
    :param score_low_risk: 'high' 表示高分为低风险（通过），'low' 表示低分为低风险
    :return: 权衡表 DataFrame

    Example:
        >>> tradeoff = approval_badrate_tradeoff(df['fpd15'], df['score'])
        >>> # 找坏率≤3% 时通过率最高的阈值
        >>> tradeoff[tradeoff['通过人群坏率(%)'] <= 3].head(1)
    """
    y = pd.to_numeric(y_true, errors='coerce')
    s = pd.to_numeric(score, errors='coerce')
    mask = y.notna() & s.notna()
    y, s = y[mask], s[mask]
    _check_binary_target(y, 'y_true')

    total = len(y)
    total_bad = int(y.sum())
    total_good = total - total_bad

    thresholds = np.percentile(s, np.linspace(0, 100, n_points + 2)[1:-1])
    thresholds = np.unique(thresholds)

    rows = []
    for thr in thresholds:
        if score_low_risk == 'high':
            approved = s >= thr
        else:
            approved = s <= thr
        approved_n = int(approved.sum())
        rejected_n = total - approved_n
        approved_bad = int(y[approved].sum())
        rejected_bad = int(y[~approved].sum())

        approved_rate = round(approved_n / total * 100, 2)
        app_bad_rate = round(_safe_div(approved_bad, approved_n) * 100, 4)
        rej_bad_rate = round(_safe_div(rejected_bad, rejected_n) * 100, 4) if rejected_n > 0 else np.nan

        row: Dict[str, Any] = {
            '评分阈值': round(float(thr), 4),
            '通过率(%)': approved_rate,
            '拒绝率(%)': round(100 - approved_rate, 2),
            '通过人数': approved_n,
            '拒绝人数': rejected_n,
            '通过人群坏率(%)': app_bad_rate,
            '拒绝人群坏率(%)': rej_bad_rate,
            '通过人群坏样本数': approved_bad,
            '拒绝人群坏样本数': rejected_bad,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values('通过率(%)', ascending=True).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 2. score_strategy_simulation
# ---------------------------------------------------------------------------

def score_strategy_simulation(
    df: pd.DataFrame,
    score_col: str,
    target: str,
    thresholds: List[float],
    amount_col: Optional[str] = None,
    score_low_risk: str = 'high',
) -> pd.DataFrame:
    """评分阈值策略仿真.

    对一组指定的评分阈值，分别计算每档策略对应的
    通过量（件/金额）、坏率、坏账量，
    便于与当前策略对比，做"如果把阈值调整到X会怎样"的仿真。

    :param df: 输入 DataFrame
    :param score_col: 评分列名
    :param target: 目标变量列名（0/1）
    :param thresholds: 评分阈值列表（通过/拒绝切割点）
    :param amount_col: 金额列名（可选），提供时计算通过金额和坏账金额
    :param score_low_risk: 'high' 表示高分为低风险，'low' 表示低分为低风险
    :return: 各阈值下的策略仿真结果 DataFrame

    Example:
        >>> result = score_strategy_simulation(df, score_col='score', target='fpd15',
        ...     thresholds=[500, 520, 540, 560], amount_col='loan_amount')
    """
    validate_dataframe(df, required_cols=[score_col, target])
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[score_col, target])
    _check_binary_target(df[target], target)

    total = len(df)
    total_bad = int(df[target].sum())

    rows = []
    for thr in thresholds:
        if score_low_risk == 'high':
            approved = df[score_col] >= thr
        else:
            approved = df[score_col] <= thr
        app_df = df[approved]
        rej_df = df[~approved]

        app_n = len(app_df)
        app_bad = int(app_df[target].sum())
        rej_n = len(rej_df)
        rej_bad = int(rej_df[target].sum())

        row: Dict[str, Any] = {
            '评分阈值': thr,
            '通过量(笔)': app_n,
            '通过率(%)': round(_safe_div(app_n, total) * 100, 2),
            '通过人群坏率(%)': round(_safe_div(app_bad, app_n) * 100, 4),
            '捕获坏样本数': app_bad,
            '拒绝量(笔)': rej_n,
            '拒绝率(%)': round(_safe_div(rej_n, total) * 100, 2),
            '拒绝坏样本数': rej_bad,
            '坏样本拦截率(%)': round(_safe_div(rej_bad, total_bad) * 100, 2) if total_bad > 0 else np.nan,
        }

        if amount_col is not None and amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            app_amt = float(app_df[amount_col].sum())
            app_bad_amt = float(app_df.loc[app_df[target] == 1, amount_col].sum())
            row['通过金额'] = round(app_amt, 2)
            row['通过人群坏账金额'] = round(app_bad_amt, 2)
            row['坏账率(金额,%)'] = round(_safe_div(app_bad_amt, app_amt) * 100, 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    return result


# ---------------------------------------------------------------------------
# 3. vintage_performance_summary
# ---------------------------------------------------------------------------

def vintage_performance_summary(
    df: pd.DataFrame,
    vintage_col: str,
    mob_col: str,
    target_col: str,
    mob_points: Optional[List[int]] = None,
    amount_col: Optional[str] = None,
) -> pd.DataFrame:
    """Vintage 账龄绩效汇总表.

    以 vintage（放款批次，如放款月份）为行，MOB（账龄）为列，
    计算不同账龄下的累计坏率，支持金额加权。
    可传入多个 MOB 观测点（如 [3,6,9,12]）。

    :param df: 输入 DataFrame
    :param vintage_col: Vintage 列名（如放款年月 '2024-01'）
    :param mob_col: 当前账龄（月）列名
    :param target_col: 坏标签列名（0/1）
    :param mob_points: 关注的 MOB 观测点列表，None 时自动取所有整数 MOB
    :param amount_col: 金额列名（可选），提供时输出金额加权坏率
    :return: Vintage 绩效汇总宽表，行=vintage，列=MOB_X_坏率

    Example:
        >>> summary = vintage_performance_summary(df, vintage_col='loan_month',
        ...     mob_col='mob', target_col='fpd15', mob_points=[3,6,9,12])
    """
    validate_dataframe(df, required_cols=[vintage_col, mob_col, target_col])
    df = df.copy()
    df[mob_col] = pd.to_numeric(df[mob_col], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[mob_col, target_col])
    _check_binary_target(df[target_col], target_col)

    all_mobs = sorted(df[mob_col].dropna().astype(int).unique())
    if mob_points is None:
        mob_points = all_mobs
    else:
        mob_points = [m for m in mob_points if m in all_mobs]

    vintages = sorted(df[vintage_col].dropna().unique())
    records = []
    for vintage in vintages:
        row: Dict[str, Any] = {'Vintage': vintage}
        vdf = df[df[vintage_col] == vintage]
        for mob in mob_points:
            mob_df = vdf[vdf[mob_col] <= mob]
            n = len(mob_df)
            if n == 0:
                row[f'MOB{mob}_坏率(%)'] = np.nan
                if amount_col and amount_col in df.columns:
                    row[f'MOB{mob}_坏账率(金额,%)'] = np.nan
                row[f'MOB{mob}_样本数'] = 0
                continue
            bad_n = int(mob_df[target_col].sum())
            row[f'MOB{mob}_样本数'] = n
            row[f'MOB{mob}_坏率(%)'] = round(_safe_div(bad_n, n) * 100, 4)
            if amount_col is not None and amount_col in df.columns:
                amt_col = pd.to_numeric(mob_df[amount_col], errors='coerce')
                total_amt = float(amt_col.sum())
                bad_amt = float(amt_col[mob_df[target_col] == 1].sum())
                row[f'MOB{mob}_坏账率(金额,%)'] = round(_safe_div(bad_amt, total_amt) * 100, 4)
        records.append(row)

    result = pd.DataFrame(records)
    return result


# ---------------------------------------------------------------------------
# 4. roll_rate_matrix
# ---------------------------------------------------------------------------

def _roll_build_matrix(
    raw: pd.DataFrame,
    labels: List[str],
    obs_label: str,
    perf_label: str,
    row_totals: pd.Series,
    total_suffix: str,
) -> pd.DataFrame:
    """内部辅助：将原始迁移计数/金额表构造成带多层列头和变好/保持/变坏汇总的 DataFrame.

    行索引：外层名称 = obs_label（观察点标识），内层 = DPD 状态标签。
    列索引：外层 = perf_label（表现期）或 '汇总'，内层 = DPD 状态或汇总标签。
    """
    state_cols = pd.MultiIndex.from_tuples(
        [(perf_label, lbl) for lbl in labels]
    )
    summary_cols = pd.MultiIndex.from_tuples([
        ('汇总', f'变好（{total_suffix}）'),
        ('汇总', f'保持（{total_suffix}）'),
        ('汇总', f'变坏（{total_suffix}）'),
        ('汇总', f'合计（{total_suffix}）'),
    ])
    col_idx = state_cols.append(summary_cols)
    # 行的外层名称改为 obs_label，这样表现期列下只展示 D0/D1-7/…
    row_idx = pd.MultiIndex.from_product(
        [[obs_label], labels],
        names=['观察点', 'DPD状态'],
    )

    data = []
    for state in labels:
        i = labels.index(state)
        row_vals = [
            int(raw.loc[state, col]) if state in raw.index and col in raw.columns else 0
            for col in labels
        ]
        back = sum(row_vals[:i])
        keep = row_vals[i]
        fwd  = sum(row_vals[i + 1:])
        total = int(row_totals.get(state, 0))
        data.append(row_vals + [back, keep, fwd, total])

    return pd.DataFrame(data, index=row_idx, columns=col_idx)


def roll_rate_matrix(
    df: pd.DataFrame,
    dpd_t0: str,
    dpd_t1: str,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    mob_t0: Optional[int] = None,
    mob_t1: Optional[int] = None,
    date_t0: Optional[str] = None,
    date_t1: Optional[str] = None,
    amount_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """DPD 滚动率矩阵.

    分析借款人在观察点（t0）到表现点（t1）之间逾期状态的迁移规律，
    同时输出订单与金额（可选）两个口径的计数矩阵、比例矩阵及整体汇总，
    量化各 DPD 状态段的资产质量改善（变好）、保持与恶化（变坏）程度，
    辅助信贷风险的动态监控与预警。

    **样本选取说明**：
    应选取观察点之前已放款、且在观察点之后仍有未结清余额（贷款表现）的样本。
    调用方需在传入 DataFrame 前完成此过滤：剔除观察点前已提前结清的贷款，
    以及观察点之后才新发放的贷款，确保每笔样本在观察期（t0）和表现期（t1）
    均有有效 DPD 记录，避免已结清或未成熟贷款混入造成分布失真。

    :param df: 输入 DataFrame（须已完成上述样本过滤）
    :param dpd_t0: 观察期 DPD 列名（t0 时刻逾期天数）
    :param dpd_t1: 表现期 DPD 列名（t1 时刻逾期天数）
    :param bins: DPD 分档断点，默认 [0, 1, 7, 15, 30, 60, 90, 120, inf]
    :param labels: 各档标签，默认 ['D0','D1-7','D8-15','D16-30','D31-60','D61-90','D91-120','D120+']
    :param mob_t0: 观察点账龄（MOB），用于表头标注，如 12 表示 "MOB12"
    :param mob_t1: 表现点账龄（MOB），用于表头标注，如 18 表示 "MOB18"
    :param date_t0: 观察点日期字符串（可选），如 '2023-06'，用于表头描述
    :param date_t1: 表现点日期字符串（可选），如 '2023-12'，用于表头描述
    :param amount_col: 贷款金额列名（可选），提供时额外输出金额口径矩阵与比例矩阵
    :return: 包含以下键的字典：

        - ``'元信息'``: 观察点/表现点描述、总订单数、订单口径变好/保持/变坏笔数
          及占比；若提供 amount_col 则同时包含金额口径汇总
        - ``'计数矩阵'``: 多层列索引的迁移计数宽表，含变好/保持/变坏/合计汇总列
        - ``'订单比例矩阵'``: 基于订单数行归一化的迁移概率（原始小数），含变好/保持/变坏汇总列
        - ``'金额矩阵'``: 当 ``amount_col`` 提供时，金额口径的迁移宽表，含汇总列
        - ``'金额比例矩阵'``: 当 ``amount_col`` 提供时，基于金额行归一化的迁移比例（原始小数）

    Example:
        >>> result = roll_rate_matrix(
        ...     df, dpd_t0='dpd_mob12', dpd_t1='dpd_mob18',
        ...     mob_t0=12, mob_t1=18, amount_col='loan_amount'
        ... )
        >>> display(result['元信息'])
        >>> display(result['计数矩阵'])
        >>> display(result['订单比例矩阵'])
        >>> display(result['金额矩阵'])
        >>> display(result['金额比例矩阵'])
    """
    validate_dataframe(df, required_cols=[dpd_t0, dpd_t1])

    if bins is None:
        bins = [0, 1, 7, 15, 30, 60, 90, 120, float('inf')]
    if labels is None:
        default_labels = ['D0', 'D1-7', 'D8-15', 'D16-30', 'D31-60', 'D61-90', 'D91-120', 'D120+']
        n_intervals = len(bins) - 1
        labels = default_labels[:n_intervals] if n_intervals <= len(default_labels) else \
            [f'D_bin{i}' for i in range(n_intervals)]

    df = df.copy()
    df[dpd_t0] = pd.to_numeric(df[dpd_t0], errors='coerce')
    df[dpd_t1] = pd.to_numeric(df[dpd_t1], errors='coerce')
    df = df.dropna(subset=[dpd_t0, dpd_t1])

    df['__state_t0__'] = pd.cut(df[dpd_t0], bins=bins, labels=labels, right=False)
    df['__state_t1__'] = pd.cut(df[dpd_t1], bins=bins, labels=labels, right=False)
    df = df.dropna(subset=['__state_t0__', '__state_t1__'])

    # ── Section labels ─────────────────────────────────────────────────────
    obs_label = f'MOB{mob_t0}' if mob_t0 is not None else '观察期'
    perf_label = f'MOB{mob_t1}' if mob_t1 is not None else '表现期'
    if date_t0:
        obs_label = f'{obs_label} {date_t0}'
    if date_t1:
        perf_label = f'{perf_label} {date_t1}'

    # ── Raw count table ────────────────────────────────────────────────────
    raw_cnt = pd.crosstab(df['__state_t0__'], df['__state_t1__'])
    raw_cnt = raw_cnt.reindex(index=labels, columns=labels, fill_value=0)
    cnt_row_totals = raw_cnt.sum(axis=1)

    # ── Count matrix (multi-level) ─────────────────────────────────────────
    count_mat = _roll_build_matrix(raw_cnt, labels, obs_label, perf_label, cnt_row_totals, '笔')

    # ── Order proportion matrix ────────────────────────────────────────────
    prop_cnt = raw_cnt.div(cnt_row_totals.replace(0, np.nan), axis=0).fillna(0.0)
    state_cols_idx = pd.MultiIndex.from_tuples(
        [(perf_label, lbl) for lbl in labels]
    )
    summary_cols_idx = pd.MultiIndex.from_tuples([
        ('汇总', '变好'), ('汇总', '保持'), ('汇总', '变坏'),
    ])
    # row index: outer = obs_label, inner = DPD 状态
    prop_row_idx = pd.MultiIndex.from_product(
        [[obs_label], labels],
        names=['观察点', 'DPD状态'],
    )
    order_prop_data = []
    for state in labels:
        i = labels.index(state)
        row_vals = [
            round(float(prop_cnt.loc[state, col]), 6) if state in prop_cnt.index and col in prop_cnt.columns else 0.0
            for col in labels
        ]
        back = round(sum(row_vals[:i]), 6)
        keep = round(row_vals[i], 6)
        fwd  = round(sum(row_vals[i + 1:]), 6)
        order_prop_data.append(row_vals + [back, keep, fwd])

    order_prop_mat = pd.DataFrame(
        order_prop_data,
        index=prop_row_idx,
        columns=state_cols_idx.append(summary_cols_idx),
    )

    # ── State ordering for meta ────────────────────────────────────────────
    state_order_map = {lbl: i for i, lbl in enumerate(labels)}
    t0_ord = df['__state_t0__'].map(state_order_map)
    t1_ord = df['__state_t1__'].map(state_order_map)
    n_total = len(df)

    back_mask = t1_ord < t0_ord
    keep_mask = t1_ord == t0_ord
    fwd_mask  = t1_ord > t0_ord

    # 订单口径元信息行
    order_meta_row: Dict[str, Any] = {
        '口径': '订单',
        '观察点': obs_label,
        '表现点': perf_label,
        '观察期DPD列': dpd_t0,
        '表现期DPD列': dpd_t1,
        '总量': n_total,
        '变好量': int(back_mask.sum()),
        '保持量': int(keep_mask.sum()),
        '变坏量': int(fwd_mask.sum()),
        '变好率': round(_safe_div(back_mask.sum(), n_total), 4),
        '保持率': round(_safe_div(keep_mask.sum(), n_total), 4),
        '变坏率': round(_safe_div(fwd_mask.sum(), n_total), 4),
    }

    result: Dict[str, pd.DataFrame] = {
        '计数矩阵': count_mat,
        '订单比例矩阵': order_prop_mat,
    }

    # ── Amount matrices ────────────────────────────────────────────────────
    if amount_col is not None and amount_col in df.columns:
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)

        amt_raw = (
            df.groupby(['__state_t0__', '__state_t1__'], observed=True)[amount_col]
            .sum()
            .unstack(fill_value=0.0)
        )
        amt_raw = amt_raw.reindex(index=labels, columns=labels, fill_value=0.0)
        amt_row_totals = amt_raw.sum(axis=1)

        amt_mat = _roll_build_matrix(amt_raw, labels, obs_label, perf_label, amt_row_totals, '金额')
        result['金额矩阵'] = amt_mat

        # Amount proportion
        prop_amt = amt_raw.div(amt_row_totals.replace(0, np.nan), axis=0).fillna(0.0)
        amt_prop_data = []
        for state in labels:
            i = labels.index(state)
            row_vals = [
                round(float(prop_amt.loc[state, col]), 6) if state in prop_amt.index and col in prop_amt.columns else 0.0
                for col in labels
            ]
            back = round(sum(row_vals[:i]), 6)
            keep = round(row_vals[i], 6)
            fwd  = round(sum(row_vals[i + 1:]), 6)
            amt_prop_data.append(row_vals + [back, keep, fwd])

        amt_prop_mat = pd.DataFrame(
            amt_prop_data,
            index=prop_row_idx,
            columns=state_cols_idx.append(summary_cols_idx),
        )
        result['金额比例矩阵'] = amt_prop_mat

        # 金额口径元信息行
        total_amt = float(df[amount_col].sum())
        back_amt = float(df.loc[back_mask, amount_col].sum())
        keep_amt = float(df.loc[keep_mask, amount_col].sum())
        fwd_amt  = float(df.loc[fwd_mask, amount_col].sum())
        amt_meta_row: Dict[str, Any] = {
            '口径': '金额',
            '观察点': obs_label,
            '表现点': perf_label,
            '观察期DPD列': dpd_t0,
            '表现期DPD列': dpd_t1,
            '总量': round(total_amt, 2),
            '变好量': round(back_amt, 2),
            '保持量': round(keep_amt, 2),
            '变坏量': round(fwd_amt, 2),
            '变好率': round(_safe_div(back_amt, total_amt), 4),
            '保持率': round(_safe_div(keep_amt, total_amt), 4),
            '变坏率': round(_safe_div(fwd_amt, total_amt), 4),
        }
        meta_rows = [order_meta_row, amt_meta_row]
    else:
        meta_rows = [order_meta_row]

    result['元信息'] = pd.DataFrame(meta_rows).reset_index(drop=True)
    # reorder so 元信息 is first
    result = {k: result[k] for k in ['元信息', '计数矩阵', '订单比例矩阵']
              + [k for k in result if k not in ('元信息', '计数矩阵', '订单比例矩阵')]}

    return result


# ---------------------------------------------------------------------------
# 5. label_leakage_check
# ---------------------------------------------------------------------------

def label_leakage_check(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    threshold_iv: float = 0.5,
    threshold_auc: float = 0.9,
    n_bins: int = 10,
) -> pd.DataFrame:
    """标签泄露检测.

    对每个特征计算 IV 和 AUC，超过阈值则标记为疑似泄露，
    可快速识别训练数据中存在未来信息的特征。

    :param df: 输入 DataFrame
    :param features: 待检测特征列表
    :param target: 目标变量列名（0/1）
    :param threshold_iv: IV 泄露阈值，超过则告警，默认 0.5
    :param threshold_auc: AUC 泄露阈值，超过则告警，默认 0.9
    :param n_bins: IV 计算分箱数
    :return: 标签泄露检测结果 DataFrame

    Example:
        >>> result = label_leakage_check(df, features=df.columns[:-1].tolist(), target='fpd15')
        >>> suspected = result[result['疑似泄露'] == True]
        >>> print(suspected[['特征名', 'IV', 'AUC', '泄露原因']])
    """
    validate_dataframe(df, required_cols=[target] + features)

    from ..metrics import iv, auc  # type: ignore[attr-defined]

    _check_binary_target(df[target], target)

    rows = []
    for feat in features:
        if feat not in df.columns or feat == target:
            continue
        col = pd.to_numeric(df[feat], errors='coerce')
        y = pd.to_numeric(df[target], errors='coerce')
        mask = col.notna() & y.notna()
        n_valid = mask.sum()

        if n_valid < 30 or y[mask].nunique() < 2:
            rows.append({
                '特征名': feat,
                'IV': np.nan,
                'AUC': np.nan,
                '样本量': n_valid,
                '疑似泄露': False,
                '泄露原因': '样本不足，跳过',
            })
            continue

        try:
            iv_val = round(float(iv(y[mask], col[mask])), 4)
        except Exception:
            iv_val = np.nan
        try:
            auc_val = round(float(auc(y[mask], col[mask])), 4)
        except Exception:
            auc_val = np.nan

        reasons = []
        if not np.isnan(iv_val) and iv_val > threshold_iv:
            reasons.append(f'IV={iv_val} > 阈值{threshold_iv}')
        if not np.isnan(auc_val) and auc_val > threshold_auc:
            reasons.append(f'AUC={auc_val} > 阈值{threshold_auc}')
        suspected = len(reasons) > 0

        rows.append({
            '特征名': feat,
            'IV': iv_val,
            'AUC': auc_val,
            '样本量': n_valid,
            '疑似泄露': suspected,
            '泄露原因': '；'.join(reasons) if reasons else '正常',
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values('IV', ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 6. multi_label_correlation
# ---------------------------------------------------------------------------

def multi_label_correlation(
    df: pd.DataFrame,
    labels: List[str],
    method: str = 'pearson',
    threshold: float = 0.7,
) -> pd.DataFrame:
    """多标签相关性矩阵.

    计算多个标签之间的相关性（皮尔逊或斯皮尔曼），
    并标注高相关标签对，用于判断多个目标变量是否相互依赖，
    以及选择最优目标变量。

    :param df: 输入 DataFrame
    :param labels: 标签列名列表（均须为 0/1）
    :param method: 相关系数方法，'pearson' 或 'spearman'
    :param threshold: 高相关告警阈值（绝对值），默认 0.7
    :return: 相关性矩阵 DataFrame，附带高相关标签对描述

    Example:
        >>> corr_df = multi_label_correlation(df, labels=['fpd7', 'fpd15', 'fpd30'])
        >>> print(corr_df)
    """
    validate_dataframe(df, required_cols=labels)
    assert method in ('pearson', 'spearman'), "method 须为 'pearson' 或 'spearman'"

    label_df = df[labels].apply(pd.to_numeric, errors='coerce')
    corr_matrix = label_df.corr(method=method).round(4)

    # 高相关标签对
    high_corr_pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) >= threshold:
                high_corr_pairs.append(f'{labels[i]}-{labels[j]}: {val:.4f}')

    # 在结果中附加一列汇总
    result = corr_matrix.reset_index().rename(columns={'index': '标签'})

    if high_corr_pairs:
        notes = pd.Series([''] * len(result), name='高相关配对')
        notes.iloc[0] = '；'.join(high_corr_pairs)
        result = pd.concat([result, notes], axis=1)

    return result
