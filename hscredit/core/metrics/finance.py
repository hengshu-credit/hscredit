"""金融风控专用指标.

提供金融风控场景下的评估指标。

主要指标:
- lift: 模型提升度
- lift_at: 指定覆盖率下的LIFT值（支持1%/3%/5%/10%/任意值）
- lift_table: Lift详细统计表
- lift_curve: Lift曲线数据（扩充默认比例，支持tail参数）
- lift_monotonicity_check: LIFT单调性检验（头部/尾部）
- rule_lift: 规则Lift指标
- badrate: 坏账率计算
- badrate_by_group: 分组坏账率
- badrate_trend: 坏账率时间趋势
- badrate_by_score_bin: 按评分分箱的坏账率
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List

from ._base import _validate_same_length, _validate_binary_target
from ._binning import compute_bin_stats


def lift(y_true: Union[np.ndarray, pd.Series],
         y_prob: Union[np.ndarray, pd.Series],
         threshold: float = 0.5) -> float:
    """计算Lift值.

    Lift = 命中样本坏账率 / 总体坏账率
    Lift > 1 表示模型效果优于随机。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率或评分
    :param threshold: 分类阈值，默认0.5
    :return: Lift值
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    overall_badrate = np.mean(y_true)
    if overall_badrate == 0:
        return 0.0

    hit_mask = y_prob >= threshold
    hit_count = np.sum(hit_mask)

    if hit_count == 0:
        return 0.0

    hit_badrate = np.mean(y_true[hit_mask])
    return hit_badrate / overall_badrate


def lift_table(y_true: Union[np.ndarray, pd.Series],
               y_prob: Union[np.ndarray, pd.Series],
               method: str = 'quantile',
               max_n_bins: int = 10,
               min_bin_size: float = 0.01,
               **kwargs) -> pd.DataFrame:
    """计算Lift详细统计表.

    按分箱计算每箱的Lift值、累积Lift值、坏账改善等指标。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率或评分
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: Lift统计表
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    valid_mask = ~(pd.isna(y_prob) | pd.isna(y_true))
    y_prob_clean = y_prob[valid_mask]
    y_true_clean = y_true[valid_mask]

    if len(y_prob_clean) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")

    from ..binning import OptimalBinning

    df = pd.DataFrame({'prob': y_prob_clean, 'target': y_true_clean})

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df[['prob']], df['target'])
    bins = binner.transform(df[['prob']], metric='indices').values.flatten()

    bin_labels = None
    if 'prob' in binner.bin_tables_:
        bin_table = binner.bin_tables_['prob']
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()

    stats_df = compute_bin_stats(bins, y_true_clean, bin_labels=bin_labels, round_digits=False)

    result_rows = []
    for _, row in stats_df.iterrows():
        bin_idx = int(row['分箱'])
        mask = bins == bin_idx

        prob_min = y_prob_clean[mask].min() if mask.sum() > 0 else np.nan
        prob_max = y_prob_clean[mask].max() if mask.sum() > 0 else np.nan

        result_rows.append({
            '分箱': bin_idx + 1,
            '最小概率': round(prob_min, 6) if not np.isnan(prob_min) else np.nan,
            '最大概率': round(prob_max, 6) if not np.isnan(prob_max) else np.nan,
            '样本数': int(row['样本总数']),
            '好样本数': int(row['好样本数']),
            '坏样本数': int(row['坏样本数']),
            '坏样本率': round(row['坏样本率'], 6),
            '样本占比': round(row['样本占比'], 6),
            'Lift值': round(row['LIFT值'], 4),
            '坏账改善': round(row['坏账改善'], 4),
            '累积Lift值': round(row['累积LIFT值'], 4),
            '累积坏账改善': round(row['累积坏账改善'], 4),
        })

    return pd.DataFrame(result_rows)


def lift_curve(y_true: Union[np.ndarray, pd.Series],
               y_prob: Union[np.ndarray, pd.Series],
               percentages: List[float] = None,
               tail: bool = False) -> pd.DataFrame:
    """计算Lift曲线数据.

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率或评分
    :param percentages: 百分比切分点列表，默认 [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
    :param tail: False=从高概率端截取（头部高风险），True=从低概率端截取（尾部低风险客群）
    :return: Lift曲线数据
    """
    if percentages is None:
        percentages = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    total = len(y_true)
    total_bad = np.sum(y_true)
    overall_badrate = total_bad / total if total > 0 else 0.0

    # tail=True时从低概率端截取（分析低风险客群纯净度）
    if tail:
        sorted_indices = np.argsort(y_prob)   # 升序，低概率在前
    else:
        sorted_indices = np.argsort(y_prob)[::-1]  # 降序，高概率在前
    y_true_sorted = y_true[sorted_indices]

    results = []
    for pct in percentages:
        n_samples = int(total * pct)
        n_samples = max(1, min(n_samples, total))

        hit_y_true = y_true_sorted[:n_samples]
        hit_bad = np.sum(hit_y_true)
        hit_badrate = hit_bad / n_samples if n_samples > 0 else 0.0

        lift_value = hit_badrate / overall_badrate if overall_badrate > 0 else 0.0
        capture_rate = hit_bad / total_bad if total_bad > 0 else 0.0

        results.append({
            '覆盖率': f"{pct:.0%}" if pct >= 0.01 else f"{pct*100:.1f}%",
            '样本数': n_samples,
            '坏样本数': int(hit_bad),
            '坏样本捕获率': f"{capture_rate:.2%}",
            'Lift值': round(lift_value, 4),
        })

    return pd.DataFrame(results)


def lift_at(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    ratios: Union[float, List[float]] = None,
    ascending: bool = False,
) -> Union[float, pd.DataFrame]:
    """计算指定覆盖率下的LIFT值.

    支持 1%/3%/5%/10% 等任意比例，适配风控场景中头部/尾部区分能力分析。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率
    :param ratios: 覆盖率，标量（如 0.05）或列表（如 [0.01, 0.03, 0.05, 0.10]）。
        默认 [0.01, 0.03, 0.05, 0.10]
    :param ascending: False=高概率排前（风险模型头部），True=低概率排前（尾部分析）
    :return: 单个 float（ratios 为标量时）或 DataFrame（ratios 为列表时）

    Example:
        >>> from hscredit.core.metrics import lift_at
        >>> lift_at(y_true, y_prob, ratios=0.05)          # 单值，返回float
        3.42
        >>> lift_at(y_true, y_prob, ratios=[0.01, 0.03, 0.05, 0.10])  # 返回DataFrame
           覆盖率  样本数  坏样本数  坏样本率  坏样本捕获率  LIFT值
        0    1%      50      35   70.00%     12.50%    5.83
        1    3%     150      98   65.33%     35.00%    4.83
    """
    _return_scalar = isinstance(ratios, (int, float))
    if ratios is None:
        ratios = [0.01, 0.03, 0.05, 0.10]
    elif _return_scalar:
        ratios = [float(ratios)]
    else:
        ratios = list(ratios)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    total = len(y_true)
    total_bad = int(np.sum(y_true))
    overall_badrate = total_bad / total if total > 0 else 0.0

    if ascending:
        sorted_indices = np.argsort(y_prob)
    else:
        sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_indices]

    results = []
    for ratio in ratios:
        n_samples = max(1, min(int(total * ratio), total))
        hit_y = y_sorted[:n_samples]
        hit_bad = int(np.sum(hit_y))
        hit_badrate = hit_bad / n_samples if n_samples > 0 else 0.0
        lift_value = hit_badrate / overall_badrate if overall_badrate > 0 else 0.0
        capture_rate = hit_bad / total_bad if total_bad > 0 else 0.0

        pct_label = f"{ratio*100:.0f}%" if ratio >= 0.01 else f"{ratio*100:.1f}%"
        results.append({
            '覆盖率': pct_label,
            '样本数': n_samples,
            '坏样本数': hit_bad,
            '坏样本率': f"{hit_badrate:.2%}",
            '坏样本捕获率': f"{capture_rate:.2%}",
            'LIFT值': round(lift_value, 4),
        })

    if _return_scalar:
        return results[0]['LIFT值']
    return pd.DataFrame(results)


def lift_monotonicity_check(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    direction: str = 'both',
) -> Dict[str, Any]:
    """检查LIFT单调性.

    风控场景理想状态：高风险端（头部）坏率单调递减，低风险端（尾部）坏率单调递增。
    违反单调性意味着评分在某区间区分能力弱，可作为调参约束目标。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率
    :param n_bins: 分箱数，默认10
    :param direction: 检验方向。'head'=仅检验头部（高概率→低概率坏率是否递减），
        'tail'=仅检验尾部，'both'=头尾均检验
    :return: 字典，含以下键：
        - head_monotonic (bool): 头部是否单调
        - tail_monotonic (bool): 尾部是否单调
        - head_lift_values (list): 头部各分箱LIFT值（由高风险到低风险）
        - tail_lift_values (list): 尾部各分箱LIFT值（由低风险到高风险）
        - head_violations (list): 头部违反单调性的分箱对 [(i, j, 差值)]
        - tail_violations (list): 尾部违反单调性的分箱对
        - head_violation_ratio (float): 头部违反比例 0.0~1.0
        - tail_violation_ratio (float): 尾部违反比例 0.0~1.0
        - head_bin_table (pd.DataFrame): 头部分箱坏率统计表
        - tail_bin_table (pd.DataFrame): 尾部分箱坏率统计表

    Example:
        >>> from hscredit.core.metrics import lift_monotonicity_check
        >>> result = lift_monotonicity_check(y_true, y_prob, n_bins=10)
        >>> print(result['head_monotonic'])        # True / False
        >>> print(result['head_violation_ratio'])  # 0.0 ~ 1.0
        >>> print(result['head_bin_table'])
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    total = len(y_true)
    total_bad = int(np.sum(y_true))
    overall_badrate = total_bad / total if total > 0 else 0.0

    def _check_direction(sorted_indices: np.ndarray, label: str) -> Dict[str, Any]:
        """对给定排序（头部/尾部）计算各分箱坏率和单调性."""
        y_sorted = y_true[sorted_indices]
        prob_sorted = y_prob[sorted_indices]
        bin_size = total // n_bins

        bin_badrates = []
        bin_lifts = []
        bin_rows = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else total
            seg = y_sorted[start:end]
            seg_prob = prob_sorted[start:end]
            n = len(seg)
            n_bad = int(np.sum(seg))
            br = n_bad / n if n > 0 else 0.0
            lv = br / overall_badrate if overall_badrate > 0 else 0.0
            bin_badrates.append(br)
            bin_lifts.append(round(lv, 4))
            bin_rows.append({
                '分箱': i + 1,
                '样本数': n,
                '坏样本数': n_bad,
                '坏样本率': round(br, 6),
                'LIFT值': round(lv, 4),
                '概率均值': round(float(seg_prob.mean()), 6),
            })

        bin_table = pd.DataFrame(bin_rows)

        # 检测单调性：对于头部，坏率应单调递减（每个箱 <= 前一个箱）
        violations = []
        for i in range(1, len(bin_badrates)):
            if bin_badrates[i] > bin_badrates[i - 1] + 1e-8:  # 允许极小数值误差
                diff = bin_badrates[i] - bin_badrates[i - 1]
                violations.append((i, i + 1, round(diff, 6)))

        n_pairs = n_bins - 1
        violation_ratio = len(violations) / n_pairs if n_pairs > 0 else 0.0
        is_monotonic = len(violations) == 0

        return {
            'monotonic': is_monotonic,
            'lift_values': bin_lifts,
            'violations': violations,
            'violation_ratio': round(violation_ratio, 4),
            'bin_table': bin_table,
        }

    result: Dict[str, Any] = {}

    if direction in ('head', 'both'):
        # 头部：高概率排前，坏率应递减
        head_indices = np.argsort(y_prob)[::-1]
        head = _check_direction(head_indices, 'head')
        result['head_monotonic'] = head['monotonic']
        result['head_lift_values'] = head['lift_values']
        result['head_violations'] = head['violations']
        result['head_violation_ratio'] = head['violation_ratio']
        result['head_bin_table'] = head['bin_table']
    else:
        result['head_monotonic'] = None
        result['head_lift_values'] = []
        result['head_violations'] = []
        result['head_violation_ratio'] = None
        result['head_bin_table'] = pd.DataFrame()

    if direction in ('tail', 'both'):
        # 尾部：低概率排前，坏率应递增（即低风险区纯净度递增）
        # 注：对尾部客群而言，越靠前概率越低，坏率应越小，
        # 若坏率单调递增（从低风险到高风险）即合理
        tail_indices = np.argsort(y_prob)          # 升序
        tail = _check_direction(tail_indices, 'tail')
        # 尾部单调性：坏率应单调递增（violations定义为坏率[i] < 坏率[i-1]）
        tail_violations = []
        br_list = [row['坏样本率'] for _, row in tail['bin_table'].iterrows()]
        for i in range(1, len(br_list)):
            if br_list[i] < br_list[i - 1] - 1e-8:
                tail_violations.append((i, i + 1, round(br_list[i - 1] - br_list[i], 6)))
        n_pairs = n_bins - 1
        tail_vr = len(tail_violations) / n_pairs if n_pairs > 0 else 0.0
        result['tail_monotonic'] = len(tail_violations) == 0
        result['tail_lift_values'] = tail['lift_values']
        result['tail_violations'] = tail_violations
        result['tail_violation_ratio'] = round(tail_vr, 4)
        result['tail_bin_table'] = tail['bin_table']
    else:
        result['tail_monotonic'] = None
        result['tail_lift_values'] = []
        result['tail_violations'] = []
        result['tail_violation_ratio'] = None
        result['tail_bin_table'] = pd.DataFrame()

    return result


def rule_lift(y_true: Union[np.ndarray, pd.Series],
              rule_mask: Union[np.ndarray, pd.Series],
              amount: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """计算规则的Lift指标.

    :param y_true: 真实标签 (0/1)
    :param rule_mask: 规则命中掩码
    :param amount: 金额数据（可选）
    :return: Lift指标字典
    """
    y_true = np.asarray(y_true)
    rule_mask = np.asarray(rule_mask)

    _validate_same_length(y_true, rule_mask, ("y_true", "rule_mask"))

    total = len(y_true)
    total_bad = np.sum(y_true)
    overall_badrate = total_bad / total if total > 0 else 0.0

    hit_count = np.sum(rule_mask)
    hit_rate = hit_count / total if total > 0 else 0.0

    if hit_count == 0:
        return {
            'hit_count': 0, 'hit_rate': 0.0, 'bad_count': 0, 'good_count': 0,
            'badrate': 0.0, 'overall_badrate': overall_badrate,
            'lift': 0.0, 'bad_improve': 0.0,
        }

    hit_bad = np.sum(y_true[rule_mask])
    hit_good = hit_count - hit_bad
    hit_badrate = hit_bad / hit_count

    lift_value = hit_badrate / overall_badrate if overall_badrate > 0 else 0.0

    other_bad = total_bad - hit_bad
    other_total = total - hit_count
    other_badrate = other_bad / other_total if other_total > 0 else 0.0
    bad_improve = (overall_badrate - other_badrate) / overall_badrate if overall_badrate > 0 else 0.0

    result = {
        'hit_count': int(hit_count),
        'hit_rate': round(hit_rate, 6),
        'bad_count': int(hit_bad),
        'good_count': int(hit_good),
        'badrate': round(hit_badrate, 6),
        'overall_badrate': round(overall_badrate, 6),
        'lift': round(lift_value, 4),
        'bad_improve': round(bad_improve, 4),
    }

    if amount is not None:
        amount = np.asarray(amount)
        if len(amount) != len(y_true):
            raise ValueError("amount长度必须与其他参数相同")

        total_amount = np.sum(amount)
        hit_amount = np.sum(amount[rule_mask])
        bad_amount = np.sum(amount[rule_mask] * y_true[rule_mask])

        overall_bad_amount_ratio = np.sum(amount * y_true) / total_amount if total_amount > 0 else 0.0
        hit_bad_amount_ratio = bad_amount / hit_amount if hit_amount > 0 else 0.0
        amount_lift = hit_bad_amount_ratio / overall_bad_amount_ratio if overall_bad_amount_ratio > 0 else 0.0

        result.update({
            'total_amount': total_amount,
            'hit_amount': hit_amount,
            'bad_amount': bad_amount,
            'amount_lift': round(amount_lift, 4),
        })

    return result


def badrate(y_true: Union[np.ndarray, pd.Series],
            weights: Optional[Union[np.ndarray, pd.Series]] = None) -> float:
    """计算总体坏账率.

    :param y_true: 真实标签 (0/1)
    :param weights: 样本权重（可选）
    :return: 坏账率
    """
    y_true = np.asarray(y_true)

    if weights is not None:
        weights = np.asarray(weights)
        return np.average(y_true, weights=weights)

    return np.mean(y_true)


def badrate_by_group(y_true: Union[np.ndarray, pd.Series],
                     group: Union[np.ndarray, pd.Series],
                     weights: Optional[Union[np.ndarray, pd.Series]] = None) -> pd.DataFrame:
    """按分组计算坏账率.

    :param y_true: 真实标签 (0/1)
    :param group: 分组标签
    :param weights: 样本权重（可选）
    :return: 各组坏账率统计
    """
    y_true = np.asarray(y_true)
    group = np.asarray(group)

    _validate_same_length(y_true, group, ("y_true", "group"))

    total = len(y_true)
    overall_badrate = badrate(y_true, weights)

    results = []
    for g in np.unique(group):
        mask = group == g
        group_y_true = y_true[mask]
        group_count = len(group_y_true)
        group_bad = np.sum(group_y_true)
        group_good = group_count - group_bad

        if weights is not None:
            group_weights = weights[mask]
            group_badrate = np.average(group_y_true, weights=group_weights)
        else:
            group_badrate = group_bad / group_count if group_count > 0 else 0.0

        results.append({
            '分组': g,
            '样本数': group_count,
            '好样本数': group_good,
            '坏样本数': group_bad,
            '坏账率': round(group_badrate, 6),
            '样本占比': round(group_count / total, 6),
            '与总体差异': round(group_badrate - overall_badrate, 6),
        })

    return pd.DataFrame(results).sort_values('坏账率', ascending=False)


def badrate_trend(y_true: Union[np.ndarray, pd.Series],
                  date: Union[np.ndarray, pd.Series],
                  freq: str = 'M') -> pd.DataFrame:
    """计算坏账率时间趋势.

    :param y_true: 真实标签 (0/1)
    :param date: 日期数据
    :param freq: 时间频率，'D'日/'W'周/'M'月/'Q'季度
    :return: 时间趋势数据
    """
    y_true = np.asarray(y_true)
    date = pd.to_datetime(date)

    _validate_same_length(y_true, date, ("y_true", "date"))

    df = pd.DataFrame({'date': date, 'y_true': y_true})

    if freq == 'D':
        df['period'] = df['date'].dt.date
    elif freq == 'W':
        df['period'] = df['date'].dt.to_period('W').astype(str)
    elif freq == 'M':
        df['period'] = df['date'].dt.to_period('M').astype(str)
    elif freq == 'Q':
        df['period'] = df['date'].dt.to_period('Q').astype(str)
    else:
        raise ValueError("freq必须是'D'/'W'/'M'/'Q'之一")

    results = []
    cum_good = 0
    cum_bad = 0
    prev_badrate = None

    for period, group in df.groupby('period'):
        count = len(group)
        bad = group['y_true'].sum()
        good = count - bad
        badrate_val = bad / count if count > 0 else 0.0

        change = badrate_val - prev_badrate if prev_badrate is not None else np.nan
        prev_badrate = badrate_val

        cum_good += good
        cum_bad += bad
        cum_total = cum_good + cum_bad
        cum_badrate = cum_bad / cum_total if cum_total > 0 else 0.0

        results.append({
            '时间周期': period,
            '样本数': count,
            '好样本数': good,
            '坏样本数': bad,
            '坏账率': round(badrate_val, 6),
            '环比变化': round(change, 6) if not np.isnan(change) else None,
            '累积坏账率': round(cum_badrate, 6),
        })

    return pd.DataFrame(results)


def badrate_by_score_bin(y_true: Union[np.ndarray, pd.Series],
                         score: Union[np.ndarray, pd.Series],
                         method: str = 'quantile',
                         max_n_bins: int = 10,
                         min_bin_size: float = 0.01,
                         **kwargs) -> pd.DataFrame:
    """按评分分箱计算坏账率.

    :param y_true: 真实标签 (0/1)
    :param score: 评分或概率
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 分箱坏账率统计
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    _validate_same_length(y_true, score, ("y_true", "score"))

    valid_mask = ~(pd.isna(score) | pd.isna(y_true))
    score_clean = score[valid_mask]
    y_true_clean = y_true[valid_mask]

    if len(score_clean) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")

    overall_badrate = badrate(y_true_clean)

    from ..binning import OptimalBinning

    df = pd.DataFrame({'score': score_clean, 'target': y_true_clean})

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df[['score']], df['target'])
    bins = binner.transform(df[['score']], metric='indices').values.flatten()

    bin_labels = None
    if 'score' in binner.bin_tables_:
        bin_table = binner.bin_tables_['score']
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()

    stats_df = compute_bin_stats(bins, y_true_clean, bin_labels=bin_labels, round_digits=False)

    result_rows = []
    for _, row in stats_df.iterrows():
        bin_idx = int(row['分箱'])
        mask = bins == bin_idx

        score_min = score_clean[mask].min() if mask.sum() > 0 else np.nan
        score_max = score_clean[mask].max() if mask.sum() > 0 else np.nan

        result_rows.append({
            '分箱': bin_idx + 1,
            '最小评分': round(score_min, 2) if not np.isnan(score_min) else np.nan,
            '最大评分': round(score_max, 2) if not np.isnan(score_max) else np.nan,
            '样本数': int(row['样本总数']),
            '好样本数': int(row['好样本数']),
            '坏样本数': int(row['坏样本数']),
            '坏账率': round(row['坏样本率'], 6),
            '与总体差异': round(row['坏样本率'] - overall_badrate, 6),
        })

    return pd.DataFrame(result_rows)


def score_stats(score: Union[np.ndarray, pd.Series],
                y_true: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """计算评分统计信息.

    :param score: 评分数据
    :param y_true: 真实标签（可选）
    :return: 评分统计字典
    """
    score = np.asarray(score)

    result = {
        '样本数': len(score),
        '缺失数': np.sum(pd.isna(score)),
        '缺失率': np.mean(pd.isna(score)),
        '均值': np.nanmean(score),
        '标准差': np.nanstd(score),
        '最小值': np.nanmin(score),
        '最大值': np.nanmax(score),
        '中位数': np.nanmedian(score),
        '分位数_25': np.nanquantile(score, 0.25),
        '分位数_75': np.nanquantile(score, 0.75),
    }

    if y_true is not None:
        y_true = np.asarray(y_true)
        valid_mask = ~(pd.isna(score) | pd.isna(y_true))
        if valid_mask.sum() > 0:
            try:
                from .classification import ks, auc
                result['KS'] = ks(y_true[valid_mask], score[valid_mask])
                result['AUC'] = auc(y_true[valid_mask], score[valid_mask])
            except:
                pass

    return result


def score_stability(score_train: Union[np.ndarray, pd.Series],
                    score_test: Union[np.ndarray, pd.Series],
                    method: str = 'quantile',
                    max_n_bins: int = 10,
                    min_bin_size: float = 0.01,
                    **kwargs) -> pd.DataFrame:
    """计算评分稳定性.

    :param score_train: 训练集评分
    :param score_test: 测试集评分
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 稳定性分析表
    """
    from .stability import psi_table

    return psi_table(score_train, score_test, method, max_n_bins, min_bin_size, **kwargs)
