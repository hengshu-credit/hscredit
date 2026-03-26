"""金融风控专用指标.

提供金融风控场景下的评估指标。

主要指标:
- lift: 模型提升度
- lift_table: Lift详细统计表
- lift_curve: Lift曲线数据
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
               percentages: List[float] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]) -> pd.DataFrame:
    """计算Lift曲线数据.

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率或评分
    :param percentages: 百分比切分点列表
    :return: Lift曲线数据
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    total = len(y_true)
    total_bad = np.sum(y_true)
    overall_badrate = total_bad / total if total > 0 else 0.0

    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]

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
            '覆盖率': f"{pct:.0%}",
            '样本数': n_samples,
            '坏样本数': hit_bad,
            '坏样本捕获率': f"{capture_rate:.2%}",
            'Lift值': round(lift_value, 4),
        })

    return pd.DataFrame(results)


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
