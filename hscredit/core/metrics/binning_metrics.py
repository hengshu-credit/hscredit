"""分箱专用指标计算.

提供高效的分箱后指标计算，用于评分卡开发。

主要指标:
- WOE (Weight of Evidence): 衡量每个分箱的预测能力
- IV (Information Value): 衡量特征整体预测能力
- KS_by_bin: 按分箱计算KS统计量
- Gini_by_bin: 按分箱计算Gini系数
- Chi2_by_bin: 按分箱计算卡方统计量
- Divergence: 分箱差异度

优化特性:
- 向量化计算，避免循环
- 支持批量处理多特征
- 向量化计算WOE和IV
- 缓存机制减少重复计算
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
from scipy import stats


def woe_iv_vectorized(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """向量化计算WOE和IV.

    高效计算每个分箱的WOE值和IV贡献。

    :param good_counts: 每个箱的好样本数
    :param bad_counts: 每个箱的坏样本数
    :param epsilon: 平滑参数，避免log(0)
    :return: (woe_array, bin_iv_array, total_iv)
        - woe_array: 每个箱的WOE值
        - bin_iv_array: 每个箱的IV贡献
        - total_iv: 总IV值
    """
    good_counts = np.asarray(good_counts, dtype=np.float64)
    bad_counts = np.asarray(bad_counts, dtype=np.float64)

    total_good = good_counts.sum()
    total_bad = bad_counts.sum()

    if total_good == 0 or total_bad == 0:
        return np.zeros(len(good_counts)), np.zeros(len(good_counts)), 0.0

    # 平滑处理：将0替换为epsilon，避免log(0)和除零错误
    # 参考scorecardpy的做法：.replace(0, 0.9)
    # 这样可以保持分布的归一化，避免破坏好坏样本分布的比例关系
    good_counts_smooth = np.where(good_counts == 0, epsilon, good_counts)
    bad_counts_smooth = np.where(bad_counts == 0, epsilon, bad_counts)
    
    # 重新计算平滑后的总数
    total_good_smooth = good_counts_smooth.sum()
    total_bad_smooth = bad_counts_smooth.sum()

    # 计算占比（保持归一化）
    good_distr = good_counts_smooth / total_good_smooth
    bad_distr = bad_counts_smooth / total_bad_smooth

    # 向量化计算WOE
    woe = np.log(bad_distr / good_distr)

    # 向量化计算IV
    # 理论上：bin_iv = (bad_distr - good_distr) * woe 总是 >= 0
    # 当 bad_distr > good_distr 时，woe > 0，乘积 > 0
    # 当 bad_distr < good_distr 时，woe < 0，乘积 > 0
    # 当 bad_distr = good_distr 时，woe = 0，乘积 = 0
    bin_iv = (bad_distr - good_distr) * woe
    total_iv = bin_iv.sum()

    return woe, bin_iv, total_iv


def compute_bin_stats(
    bins: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-10,
    bin_labels: Optional[List[str]] = None,
    round_digits: bool = True
) -> pd.DataFrame:
    """计算分箱统计信息（高效版本，参考scorecardpipeline扩充指标）.

    一次性计算所有分箱指标，包含基础统计、WOE/IV、LIFT、KS等完整指标。
    列名使用中文，与scorecardpipeline保持一致。
    
    注意：缺失值(-1)和特殊值(-2)会放在分箱表的最后，参考optbinning的展示方式。

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param epsilon: 平滑参数
    :param bin_labels: 可选的分箱标签列表，如果提供则在"分箱"列后添加"分箱标签"列
    :param round_digits: 是否对浮点数进行四舍五入格式化，默认为True。
        设置为False可保留原始精度用于计算，设置为True时会对展示列进行合理的四舍五入。
    :return: 分箱统计DataFrame，包含中文列名:
        - 分箱: 分箱索引
        - 分箱标签: 分箱区间标签（如果提供bin_labels）
        - 样本总数: 样本数
        - 好样本数: 好样本数
        - 坏样本数: 坏样本数
        - 样本占比: 样本占比
        - 好样本占比: 好样本占比
        - 坏样本占比: 坏样本占比
        - 坏样本率: 坏样本率
        - 分档WOE值: WOE值
        - 分档IV值: 箱IV贡献
        - 指标IV值: 总IV
        - LIFT值: Lift值
        - 坏账改善: 坏账改善
        - 累积LIFT值: 累积Lift值
        - 累积坏账改善: 累积坏账改善
        - 累积好样本数: 累积好样本数
        - 累积坏样本数: 累积坏样本数
        - 分档KS值: KS值
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    # 使用np.unique获取唯一的bin索引和计数
    unique_bins, bin_indices = np.unique(bins, return_inverse=True)
    
    # 重新排序：将缺失值(-1)和特殊值(-2)放在最后
    # 排序规则：正常分箱(>=0)在前，按升序排列；缺失(-1)放倒数第二，特殊(-2)放最后
    # 创建排序键: (组, 值) - 组0是正常分箱，组1是缺失，组2是特殊
    sort_keys = []
    for b in unique_bins:
        if b == -2:
            sort_keys.append((2, b))  # 特殊值最后
        elif b == -1:
            sort_keys.append((1, b))  # 缺失值倒数第二
        else:
            sort_keys.append((0, b))  # 正常分箱在前，按值排序
    
    # 获取排序后的索引
    sort_order = np.argsort([sk[0] * 10000 + sk[1] for sk in sort_keys])
    
    # 创建从旧位置到新位置的映射
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    
    # 重新排列 unique_bins
    unique_bins_sorted = unique_bins[sort_order]
    
    # 重新映射 bin_indices
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])
    
    # 如果提供了分箱标签，也按相同顺序重新排列
    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels_sorted = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]
        bin_labels = bin_labels_sorted
    
    # 使用排序后的值继续计算
    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted

    # 向量化计算每个bin的好/坏样本数
    n_bins = len(unique_bins)
    good_counts = np.bincount(bin_indices, weights=(y == 0).astype(int), minlength=n_bins)
    bad_counts = np.bincount(bin_indices, weights=y, minlength=n_bins)
    counts = good_counts + bad_counts

    # 计算坏样本率
    bad_rate = np.where(counts > 0, bad_counts / counts, 0.0)

    # 计算WOE和IV
    woe, bin_iv, total_iv = woe_iv_vectorized(good_counts, bad_counts, epsilon)

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

    # 按分箱顺序计算累积指标（假设分箱已按顺序排列）
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

    # 构建DataFrame（使用中文列名）
    # 先构建基础数据
    data = {
        '分箱': unique_bins,
    }
    
    # 如果提供了分箱标签，添加到"分箱"列后面
    if bin_labels is not None and len(bin_labels) == n_bins:
        data['分箱标签'] = bin_labels
    
    # 添加其他列
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

    # 对浮点数列进行四舍五入格式化，避免科学计数法显示
    if round_digits:
        # 定义各列的小数位数
        float_columns = {
            # 占比类：保留6位小数
            '样本占比': 6,
            '好样本占比': 6,
            '坏样本占比': 6,
            # 比率类：保留6位小数
            '坏样本率': 6,
            # WOE和IV：保留6位小数
            '分档WOE值': 6,
            '分档IV值': 6,
            '指标IV值': 6,
            # LIFT和改善：保留4位小数
            'LIFT值': 4,
            '坏账改善': 4,
            '累积LIFT值': 4,
            '累积坏账改善': 4,
            # KS：保留6位小数
            '分档KS值': 6,
        }
        
        for col, digits in float_columns.items():
            if col in df.columns:
                df[col] = np.round(df[col], digits)
    
    return df


def ks_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """按分箱计算KS统计量.

    KS = max|累积好样本率 - 累积坏样本率|

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (max_ks, ks_array)
        - max_ks: 最大KS值
        - ks_array: 每个分箱的KS贡献
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    unique_bins = np.unique(bins)
    n_bins = len(unique_bins)

    total_good = (y == 0).sum()
    total_bad = y.sum()

    if total_good == 0 or total_bad == 0:
        return 0.0, np.zeros(n_bins)

    # 计算每个箱的好/坏样本数
    good_counts = np.array([((y == 0) & (bins == b)).sum() for b in unique_bins])
    bad_counts = np.array([(y[bins == b]).sum() for b in unique_bins])

    # 计算累积分布
    cum_good = np.cumsum(good_counts)
    cum_bad = np.cumsum(bad_counts)

    cum_good_rate = cum_good / total_good
    cum_bad_rate = cum_bad / total_bad

    # KS = |累积好占比 - 累积坏占比|
    ks_values = np.abs(cum_good_rate - cum_bad_rate)
    max_ks = ks_values.max()

    return max_ks, ks_values


def chi2_by_bin(bins: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """按分箱计算卡方统计量.

    用于评估分箱与目标变量的独立性。

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: (chi2_stat, p_value, chi2_contrib)
        - chi2_stat: 卡方统计量
        - p_value: p值
        - chi2_contrib: 每个箱的卡方贡献
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    # 构建列联表
    contingency = pd.crosstab(bins, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 1.0, np.zeros(contingency.shape[0])

    # 计算卡方统计量和p值
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

    # 计算每个箱的卡方贡献
    chi2_contrib = ((contingency - expected) ** 2 / expected).sum(axis=1)

    return chi2_stat, p_value, chi2_contrib


def divergence_by_bin(bins: np.ndarray, y: np.ndarray) -> float:
    """计算分箱差异度（Divergence）.

    Divergence衡量分箱后好坏样本分布的差异程度。

    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :return: Divergence值
    """
    bins = np.asarray(bins)
    y = np.asarray(y)

    unique_bins = np.unique(bins)

    total_good = (y == 0).sum()
    total_bad = y.sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    divergence = 0.0
    for b in unique_bins:
        mask = bins == b
        bin_good = ((y == 0) & mask).sum()
        bin_bad = (y & mask).sum()

        if bin_good > 0 and bin_bad > 0:
            # 差异度 = (坏样本率 - 总体坏样本率)^2 * 箱样本占比
            bin_bad_rate = bin_bad / mask.sum()
            overall_bad_rate = total_bad / len(y)
            bin_weight = mask.sum() / len(y)

            divergence += (bin_bad_rate - overall_bad_rate) ** 2 * bin_weight

    return divergence


def iv_for_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray
) -> float:
    """根据切分点快速计算IV值.

    :param x: 特征值
    :param y: 目标变量 (0/1)
    :param splits: 切分点数组
    :return: IV值
    """
    if len(splits) == 0:
        return 0.0

    # 使用digitize分箱
    bins = np.digitize(x, splits, right=True)

    # 计算IV
    _, _, total_iv = woe_iv_vectorized(
        np.bincount(bins, weights=(y == 0).astype(int)),
        np.bincount(bins, weights=y)
    )

    return total_iv


def ks_for_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray
) -> float:
    """根据切分点快速计算KS值.

    :param x: 特征值
    :param y: 目标变量 (0/1)
    :param splits: 切分点数组
    :return: KS值
    """
    if len(splits) == 0:
        return 0.0

    # 使用digitize分箱
    bins = np.digitize(x, splits, right=True)

    # 计算KS
    max_ks, _ = ks_by_bin(bins, y)

    return max_ks


def batch_iv(
    X: pd.DataFrame,
    y: pd.Series,
    bins_dict: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """批量计算多特征的IV值.

    :param X: 特征DataFrame
    :param y: 目标变量
    :param bins_dict: 每个特征的分箱索引字典
    :return: 每个特征的IV值字典
    """
    y_arr = np.asarray(y)
    iv_dict = {}

    for feature, bins in bins_dict.items():
        if feature in X.columns:
            _, _, iv = woe_iv_vectorized(
                np.bincount(bins, weights=(y_arr == 0).astype(int)),
                np.bincount(bins, weights=y_arr)
            )
            iv_dict[feature] = iv

    return iv_dict


def compare_splits_iv(
    x: np.ndarray,
    y: np.ndarray,
    splits_before: np.ndarray,
    splits_after: np.ndarray
) -> float:
    """比较两个切分方案的IV差异（用于预分箱优化）.

    :param x: 特征值
    :param y: 目标变量
    :param splits_before: 合并前的切分点
    :param splits_after: 合并后的切分点
    :return: IV损失 (合并前IV - 合并后IV)
    """
    iv_before = iv_for_splits(x, y, splits_before)
    iv_after = iv_for_splits(x, y, splits_after)
    return iv_before - iv_after


def compare_splits_ks(
    x: np.ndarray,
    y: np.ndarray,
    splits_before: np.ndarray,
    splits_after: np.ndarray
) -> float:
    """比较两个切分方案的KS差异（用于预分箱优化）.

    :param x: 特征值
    :param y: 目标变量
    :param splits_before: 合并前的切分点
    :param splits_after: 合并后的切分点
    :return: KS损失 (合并前KS - 合并后KS)
    """
    ks_before = ks_for_splits(x, y, splits_before)
    ks_after = ks_for_splits(x, y, splits_after)
    return ks_before - ks_after
