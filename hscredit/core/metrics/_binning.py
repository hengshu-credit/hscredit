"""分箱统计内部实现.

提供高效的分箱后指标计算，供其他指标模块使用。
支持二元目标（0/1）、连续目标（金额、余额等）和金额加权模式。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Literal
from scipy import stats

from ._base import _woe_iv_vectorized


def compute_bin_stats(
    bins: np.ndarray,
    y: np.ndarray,
    target_type: Literal['binary', 'continuous', 'amount_weighted'] = 'binary',
    amount: Optional[np.ndarray] = None,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
) -> pd.DataFrame:
    """计算分箱统计信息.

    一次性计算所有分箱指标，支持三种模式：
    1. 二元目标（0/1）：标准的分类问题，计算WOE/IV/LIFT等
    2. 连续目标：如逾期金额、余额等，计算金额统计
    3. 金额加权：基于二元标签，但所有统计按金额加权

    :param bins: 分箱索引数组
    :param y: 目标变量
        - target_type='binary': 0/1数组
        - target_type='continuous': 连续值数组（如逾期金额、余额）
        - target_type='amount_weighted': 0/1数组，配合amount参数使用
    :param target_type: 目标变量类型，'binary'/'continuous'/'amount_weighted'，默认'binary'
    :param amount: 金额数组，仅在target_type='amount_weighted'时需要
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化，默认为True
    :return: 分箱统计DataFrame，包含中文列名
    
    Example:
        >>> # 二元目标（0/1）
        >>> bins = np.array([0, 0, 1, 1, 2, 2])
        >>> y_binary = np.array([0, 1, 0, 1, 0, 1])
        >>> stats_df = compute_bin_stats(bins, y_binary, target_type='binary')
        
        >>> # 连续目标（逾期金额）
        >>> y_amount = np.array([0, 1000, 0, 2000, 0, 1500])
        >>> stats_df = compute_bin_stats(bins, y_amount, target_type='continuous')
        
        >>> # 金额加权（基于逾期金额加权的坏账统计）
        >>> y_flag = np.array([0, 1, 0, 1, 0, 1])
        >>> amount = np.array([100, 1000, 200, 2000, 150, 1500])
        >>> stats_df = compute_bin_stats(bins, y_flag, target_type='amount_weighted', amount=amount)
    """
    bins = np.asarray(bins)
    y = np.asarray(y, dtype=np.float64)
    
    if target_type == 'binary':
        return _compute_bin_stats_binary(bins, y, epsilon, bin_labels, round_digits)
    elif target_type == 'continuous':
        return _compute_bin_stats_continuous(bins, y, epsilon, bin_labels, round_digits)
    elif target_type == 'amount_weighted':
        if amount is None:
            raise ValueError("target_type='amount_weighted'时必须提供amount参数")
        amount = np.asarray(amount, dtype=np.float64)
        return _compute_bin_stats_amount_weighted(bins, y, amount, epsilon, bin_labels, round_digits)
    else:
        raise ValueError(f"target_type必须是'binary'/'continuous'/'amount_weighted'，得到: {target_type}")


def _compute_bin_stats_binary(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
) -> pd.DataFrame:
    """计算二元目标的分箱统计.
    
    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :return: 分箱统计DataFrame
    """
    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)

    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前

    sort_order = np.argsort([sk[0] * 10000 + sk[1] for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)
    good_counts = np.bincount(bin_indices, weights=(y == 0).astype(int), minlength=n_bins)
    bad_counts = np.bincount(bin_indices, weights=y, minlength=n_bins)
    counts = good_counts + bad_counts

    bad_rate = np.where(counts > 0, bad_counts / counts, 0.0)

    # 计算WOE和IV
    woe, bin_iv, total_iv = _woe_iv_vectorized(good_counts, bad_counts, epsilon)

    # 计算占比
    total = counts.sum()
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()

    count_distr = counts / total if total > 0 else np.zeros(n_bins)
    good_distr = good_counts / total_good if total_good > 0 else np.zeros(n_bins)
    bad_distr = bad_counts / total_bad if total_bad > 0 else np.zeros(n_bins)

    # 计算总体坏样本率（用于LIFT）
    overall_bad_rate = total_bad / total if total > 0 else 0.0

    # 计算LIFT值
    lift = np.where(bad_rate > 0, bad_rate / (overall_bad_rate + epsilon), 0.0)

    # 计算坏账改善
    other_bad = total_bad - bad_counts
    other_total = total - counts
    bad_improve = np.where(
        other_total > 0,
        (overall_bad_rate - other_bad / other_total) / (overall_bad_rate + epsilon),
        0.0
    )

    # 按分箱顺序计算累积指标
    cum_good = np.cumsum(good_counts)
    cum_bad = np.cumsum(bad_counts)
    cum_total = cum_good + cum_bad

    cum_lift = np.where(cum_total > 0, (cum_bad / cum_total) / (overall_bad_rate + epsilon), 0.0)
    other_cum_bad = total_bad - cum_bad
    other_cum_total = total - cum_total
    cum_bad_improve = np.where(
        other_cum_total > 0,
        (overall_bad_rate - other_cum_bad / other_cum_total) / (overall_bad_rate + epsilon),
        0.0
    )

    # 计算KS值
    cum_good_rate = cum_good / (total_good + epsilon)
    cum_bad_rate = cum_bad / (total_bad + epsilon)
    ks_values = np.abs(cum_bad_rate - cum_good_rate)

    # 构建DataFrame
    data = {'分箱': unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels

    data.update({
        '样本总数': counts.astype(int),
        '好样本数': good_counts.astype(int),
        '坏样本数': bad_counts.astype(int),
        '样本占比': count_distr,
        '好样本占比': good_distr,
        '坏样本占比': bad_distr,
        '坏样本率': bad_rate,
        '分档WOE值': woe,
        '分档IV值': bin_iv,
        '指标IV值': total_iv,
        'LIFT值': lift,
        '坏账改善': bad_improve,
        '累积LIFT值': cum_lift,
        '累积坏账改善': cum_bad_improve,
        '累积好样本数': cum_good.astype(int),
        '累积坏样本数': cum_bad.astype(int),
        '分档KS值': ks_values,
    })

    df = pd.DataFrame(data)

    # 对浮点数列进行四舍五入格式化
    if round_digits:
        float_columns = {
            '样本占比': 6, '好样本占比': 6, '坏样本占比': 6,
            '坏样本率': 6, '分档WOE值': 6, '分档IV值': 6, '指标IV值': 6,
            'LIFT值': 4, '坏账改善': 4, '累积LIFT值': 4, '累积坏账改善': 4,
            '分档KS值': 6,
        }
        for col, digits in float_columns.items():
            if col in df.columns:
                df[col] = np.round(df[col], digits)

    return df


def _compute_bin_stats_continuous(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
) -> pd.DataFrame:
    """计算连续目标的分箱统计（如逾期金额、余额等）.
    
    :param bins: 分箱索引数组
    :param y: 目标变量（连续值，如逾期金额）
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :return: 分箱统计DataFrame
    """
    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)

    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前

    sort_order = np.argsort([sk[0] * 10000 + sk[1] for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])

    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]

    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    n_bins = len(unique_bins)
    
    # 计算每箱的样本数
    counts = np.bincount(bin_indices, minlength=n_bins)
    
    # 计算每箱的目标值统计
    y_sum = np.bincount(bin_indices, weights=y, minlength=n_bins)
    y_mean = np.where(counts > 0, y_sum / counts, 0.0)
    
    # 计算每箱的方差和标准差
    y_squared_sum = np.bincount(bin_indices, weights=y**2, minlength=n_bins)
    y_var = np.where(counts > 0, y_squared_sum / counts - y_mean**2, 0.0)
    y_std = np.sqrt(np.maximum(y_var, 0))
    
    # 计算每箱的最小值和最大值
    y_min = np.array([y[bin_indices == i].min() if counts[i] > 0 else 0 for i in range(n_bins)])
    y_max = np.array([y[bin_indices == i].max() if counts[i] > 0 else 0 for i in range(n_bins)])
    
    # 计算占比
    total = counts.sum()
    total_y = y_sum.sum()
    
    count_distr = counts / total if total > 0 else np.zeros(n_bins)
    y_distr = y_sum / (total_y + epsilon) if total_y > 0 else np.zeros(n_bins)
    
    # 计算总体均值（用于LIFT）
    overall_mean = total_y / total if total > 0 else 0.0
    
    # 计算LIFT值（连续目标的LIFT = 该箱均值 / 总体均值）
    lift = np.where(y_mean > 0, y_mean / (overall_mean + epsilon), 0.0)
    
    # 计算"改善"指标（连续目标的改善 = 该箱贡献度与该箱样本占比的差异）
    bad_improve = np.where(
        count_distr > 0,
        (y_distr - count_distr) / (count_distr + epsilon),
        0.0
    )

    # 按分箱顺序计算累积指标
    cum_counts = np.cumsum(counts)
    cum_y = np.cumsum(y_sum)
    cum_mean = np.where(cum_counts > 0, cum_y / cum_counts, 0.0)

    # 累积LIFT
    cum_lift = np.where(cum_mean > 0, cum_mean / (overall_mean + epsilon), 0.0)
    
    # 累积占比
    cum_count_distr = cum_counts / total if total > 0 else np.zeros(n_bins)
    cum_y_distr = cum_y / (total_y + epsilon) if total_y > 0 else np.zeros(n_bins)
    
    # 累积改善
    cum_bad_improve = np.where(
        cum_count_distr > 0,
        (cum_y_distr - cum_count_distr) / (cum_count_distr + epsilon),
        0.0
    )

    # 计算KS值（对于连续目标，使用金额累积占比 vs 样本累积占比的差）
    ks_values = np.abs(cum_y_distr - cum_count_distr)

    # 构建DataFrame
    data = {'分箱': unique_bins}

    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels

    data.update({
        '样本总数': counts.astype(int),
        '样本占比': count_distr,
        '目标值总和': y_sum,
        '目标值均值': y_mean,
        '目标值标准差': y_std,
        '目标值最小值': y_min,
        '目标值最大值': y_max,
        '目标值占比': y_distr,
        '平均LIFT值': lift,
        '贡献改善': bad_improve,
        '累积样本数': cum_counts.astype(int),
        '累积目标值': cum_y,
        '累积平均值': cum_mean,
        '累积LIFT值': cum_lift,
        '累积贡献改善': cum_bad_improve,
        '分档KS值': ks_values,
    })

    df = pd.DataFrame(data)

    # 对浮点数列进行四舍五入格式化
    if round_digits:
        float_columns = {
            '样本占比': 6,
            '目标值均值': 2, '目标值标准差': 2, '目标值最小值': 2, '目标值最大值': 2,
            '目标值占比': 6, '平均LIFT值': 4, '贡献改善': 4,
            '累积平均值': 2, '累积LIFT值': 4, '累积贡献改善': 4,
            '分档KS值': 6,
        }
        for col, digits in float_columns.items():
            if col in df.columns:
                df[col] = np.round(df[col], digits)

    return df


def _compute_bin_stats_amount_weighted(
    bins: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
) -> pd.DataFrame:
    """计算金额加权的分箱统计（基于二元标签，但按金额加权）.
    
    适用于风控场景：基于逾期金额加权的坏账统计分析。
    输出列名与binary模式保持一致（样本/好样本/坏样本表示金额），便于统一处理。
    
    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param amount: 金额数组（如放款金额、余额等）
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表
    :param round_digits: 是否对浮点数进行四舍五入格式化
    :return: 分箱统计DataFrame（列名与binary模式保持一致，便于对比）
    """
    bins = np.asarray(bins)
    y = np.asarray(y)
    amount = np.asarray(amount)
    
    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)
    
    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前
    
    sort_order = np.argsort([sk[0] * 10000 + sk[1] for sk in sort_keys])
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    unique_bins_sorted = unique_bins[sort_order]
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])
    
    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]
    
    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted
    
    n_bins = len(unique_bins)
    
    # 计算每个分箱的好金额和坏金额（金额口径的核心）
    good_amounts = np.bincount(bin_indices, weights=(y == 0).astype(float) * amount, minlength=n_bins)
    bad_amounts = np.bincount(bin_indices, weights=y.astype(float) * amount, minlength=n_bins)
    amount_totals = good_amounts + bad_amounts
    
    # 同时计算样本数（用于参考）
    good_counts = np.bincount(bin_indices, weights=(y == 0).astype(int), minlength=n_bins)
    bad_counts = np.bincount(bin_indices, weights=y.astype(int), minlength=n_bins)
    counts = good_counts + bad_counts
    
    # 计算占比（基于金额）
    total_amount = amount_totals.sum()
    total_good_amount = good_amounts.sum()
    total_bad_amount = bad_amounts.sum()
    
    amount_ratios = amount_totals / total_amount if total_amount > 0 else np.zeros(n_bins)
    good_amount_ratios = good_amounts / total_good_amount if total_good_amount > 0 else np.zeros(n_bins)
    bad_amount_ratios = bad_amounts / total_bad_amount if total_bad_amount > 0 else np.zeros(n_bins)
    
    # 金额口径坏账率 = 坏金额 / 总金额
    bad_rate = np.where(amount_totals > 0, bad_amounts / amount_totals, 0.0)
    
    # 计算WOE和IV（基于金额占比）
    good_amounts_smooth = np.where(good_amounts == 0, epsilon, good_amounts)
    bad_amounts_smooth = np.where(bad_amounts == 0, epsilon, bad_amounts)
    total_good_smooth = good_amounts_smooth.sum()
    total_bad_smooth = bad_amounts_smooth.sum()
    
    good_distr = good_amounts_smooth / total_good_smooth if total_good_smooth > 0 else np.zeros(n_bins)
    bad_distr = bad_amounts_smooth / total_bad_smooth if total_bad_smooth > 0 else np.zeros(n_bins)
    
    woe = np.log(bad_distr / good_distr)
    bin_iv = (bad_distr - good_distr) * woe
    total_iv = bin_iv.sum()
    
    # 计算LIFT值（金额口径）
    overall_bad_rate = total_bad_amount / total_amount if total_amount > 0 else 0.0
    lift = np.where(bad_rate > 0, bad_rate / (overall_bad_rate + epsilon), 0.0)
    
    # 计算坏账改善
    other_bad = total_bad_amount - bad_amounts
    other_total = total_amount - amount_totals
    bad_improve = np.where(
        other_total > 0,
        (overall_bad_rate - other_bad / other_total) / (overall_bad_rate + epsilon),
        0.0
    )
    
    # 按分箱顺序计算累积指标
    cum_good = np.cumsum(good_amounts)
    cum_bad = np.cumsum(bad_amounts)
    cum_total = cum_good + cum_bad
    
    cum_lift = np.where(cum_total > 0, (cum_bad / cum_total) / (overall_bad_rate + epsilon), 0.0)
    other_cum_bad = total_bad_amount - cum_bad
    other_cum_total = total_amount - cum_total
    cum_bad_improve = np.where(
        other_cum_total > 0,
        (overall_bad_rate - other_cum_bad / other_cum_total) / (overall_bad_rate + epsilon),
        0.0
    )
    
    # 计算KS值（基于金额累积占比）
    cum_good_rate = cum_good / (total_good_amount + epsilon)
    cum_bad_rate = cum_bad / (total_bad_amount + epsilon)
    ks_values = np.abs(cum_bad_rate - cum_good_rate)
    
    # 构建DataFrame（使用与样本口径统一的列名，便于统一处理）
    data = {'分箱': unique_bins}
    
    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels
    
    # 使用与样本口径相同的列名，便于统一处理
    data.update({
        '样本总数': np.round(amount_totals, 2),
        '好样本数': np.round(good_amounts, 2),
        '坏样本数': np.round(bad_amounts, 2),
        '样本占比': np.round(amount_ratios, 6),
        '好样本占比': np.round(good_amount_ratios, 6),
        '坏样本占比': np.round(bad_amount_ratios, 6),
        '坏样本率': np.round(bad_rate, 6),
        '分档WOE值': np.round(woe, 6),
        '分档IV值': np.round(bin_iv, 6),
        '指标IV值': np.round(total_iv, 6),
        'LIFT值': np.round(lift, 4),
        '坏账改善': np.round(bad_improve, 4),
        '累积LIFT值': np.round(cum_lift, 4),
        '累积坏账改善': np.round(cum_bad_improve, 4),
        '累积好样本数': np.round(cum_good, 2),
        '累积坏样本数': np.round(cum_bad, 2),
        '分档KS值': np.round(ks_values, 6),
    })
    
    return pd.DataFrame(data)


def add_margins(table: pd.DataFrame) -> pd.DataFrame:
    """为分箱表添加合计行.
    
    缺失值和特殊值放在正常分箱之后、合计之前。
    列名已统一，同时支持样本口径和金额口径。
    
    :param table: 分箱统计表
    :return: 添加合计行后的分箱表
    """
    if table.empty:
        return table
    
    # 查找分箱标签列（支持多级表头和单层表头）
    bin_label_col = None
    is_multi = isinstance(table.columns, pd.MultiIndex)
    
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '分箱标签':
            bin_label_col = col
            break
    
    if bin_label_col is None:
        return table
    
    # 分离正常分箱、缺失值、特殊值
    normal_bins = []
    missing_bin = None
    special_bin = None
    
    for idx, row in table.iterrows():
        label = row[bin_label_col]
        if label == '缺失':
            missing_bin = row
        elif label == '特殊':
            special_bin = row
        else:
            normal_bins.append(row)
    
    # 计算合计行
    total_row = table.iloc[0].copy()
    total_row[bin_label_col] = '合计'
    
    # 需要汇总的数值列（列名已统一）
    numeric_cols = []
    sample_total_col = None
    bad_sample_col = None
    
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ['样本总数', '好样本数', '坏样本数', '累积好样本数', '累积坏样本数']:
            numeric_cols.append(col)
        if col_name == '样本总数':
            sample_total_col = col
        if col_name == '坏样本数':
            bad_sample_col = col
    
    # 对每一列求和
    for col in numeric_cols:
        total_row[col] = table[col].sum()
    
    # 计算占比类指标 = 1
    ratio_cols = []
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ['样本占比', '好样本占比', '坏样本占比']:
            ratio_cols.append(col)
    
    for col in ratio_cols:
        total_row[col] = 1.0
    
    # 计算坏样本率
    bad_rate_col = None
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '坏样本率':
            bad_rate_col = col
            break
    
    if bad_rate_col and sample_total_col is not None and total_row[sample_total_col] > 0:
        total_row[bad_rate_col] = total_row[bad_sample_col] / total_row[sample_total_col]
    
    # 计算LIFT和坏账改善（合计行LIFT=1，坏账改善=1）
    lift_cols = []
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name in ['LIFT值', '坏账改善', '累积LIFT值', '累积坏账改善']:
            lift_cols.append(col)
    
    for col in lift_cols:
        total_row[col] = 1.0
    
    # WOE和IV值：分档WOE=0，分档IV=0，指标IV=各分档IV之和
    woe_col = None
    bin_iv_col = None
    total_iv_col = None
    
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '分档WOE值':
            woe_col = col
        elif col_name == '分档IV值':
            bin_iv_col = col
        elif col_name == '指标IV值':
            total_iv_col = col
    
    if woe_col:
        total_row[woe_col] = 0.0
    if bin_iv_col:
        total_row[bin_iv_col] = 0.0
    if total_iv_col:
        total_row[total_iv_col] = table[total_iv_col].iloc[0]
    
    # KS值：取最大KS
    ks_col = None
    for col in table.columns:
        col_name = col[1] if is_multi else col
        if col_name == '分档KS值':
            ks_col = col
            break
    
    if ks_col:
        total_row[ks_col] = table[ks_col].max()
    
    # 重新组合：正常分箱 -> 缺失值 -> 特殊值 -> 合计
    result_rows = []
    
    # 正常分箱
    for row in normal_bins:
        result_rows.append(row)
    
    # 缺失值
    if missing_bin is not None:
        result_rows.append(missing_bin)
    
    # 特殊值
    if special_bin is not None:
        result_rows.append(special_bin)
    
    # 合计
    result_rows.append(total_row)
    
    # 重建DataFrame
    result_table = pd.DataFrame(result_rows)
    result_table = result_table.reset_index(drop=True)
    
    return result_table


def _ks_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """按分箱计算KS统计量.

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (max_ks, ks_array)
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    unique_bins = np.unique(bins)
    n_bins = len(unique_bins)

    total_good = (y == 0).sum()
    total_bad = y.sum()

    if total_good == 0 or total_bad == 0:
        return 0.0, np.zeros(n_bins)

    good_counts = np.array([((y == 0) & (bins == b)).sum() for b in unique_bins])
    bad_counts = np.array([y[bins == b].sum() for b in unique_bins])

    cum_good = np.cumsum(good_counts)
    cum_bad = np.cumsum(bad_counts)

    cum_good_rate = cum_good / total_good
    cum_bad_rate = cum_bad / total_bad

    ks_values = np.abs(cum_good_rate - cum_bad_rate)
    max_ks = ks_values.max()

    return max_ks, ks_values


def _chi2_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """按分箱计算卡方统计量.

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (chi2_stat, p_value, chi2_contrib)
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    contingency = pd.crosstab(bins, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 1.0, np.zeros(contingency.shape[0])

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    chi2_contrib = ((contingency - expected) ** 2 / expected).sum(axis=1)

    return chi2_stat, p_value, chi2_contrib
