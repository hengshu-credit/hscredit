"""特征分析器.

提供特征分箱统计分析功能，支持多逾期标签和逾期天数组合分析。
参考 scorecardpipeline 的 feature_bin_stats 实现。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from copy import deepcopy

from ..core.binning import OptimalBinning
from ..core.binning.base import BaseBinning
from ..core.metrics.binning_metrics import compute_bin_stats


def _calculate_lift(bad_rate: float, overall_bad_rate: float) -> float:
    """计算LIFT值."""
    if overall_bad_rate <= 0:
        return 1.0
    return bad_rate / overall_bad_rate


def _calculate_ks(cum_bad_rate: np.ndarray, cum_good_rate: np.ndarray) -> np.ndarray:
    """计算KS值."""
    return np.abs(cum_bad_rate - cum_good_rate)


def _add_margins(table: pd.DataFrame) -> pd.DataFrame:
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


def _create_bin_table(
    bins: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    desc: str = "",
    splits: Optional[np.ndarray] = None,
    amount: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """创建分箱统计表.
    
    :param bins: 分箱索引
    :param y: 目标变量
    :param feature_name: 特征名称
    :param desc: 特征描述
    :param splits: 切分点
    :param amount: 金额数组，用于金额口径分析
    :return: 分箱统计表
    """
    # 生成分箱标签
    bin_labels = _get_bin_labels(splits, bins)
    
    # 根据是否有金额字段，选择统计口径
    if amount is not None:
        # 金额口径：所有统计基于金额而非样本数
        stats = _compute_bin_stats_amount(bins, y, amount, bin_labels=bin_labels)
    else:
        # 样本数口径：使用 metrics 模块计算分箱统计
        stats = compute_bin_stats(bins, y, bin_labels=bin_labels)
    
    # 添加特征信息（插入到最前面）
    stats.insert(0, '指标含义', desc if desc else feature_name)
    stats.insert(0, '指标名称', feature_name)
    
    # 删除"分箱"列，只保留"分箱标签"（分箱标签更具可读性）
    if '分箱' in stats.columns:
        stats = stats.drop(columns=['分箱'])
    
    return stats


def _compute_bin_stats_amount(
    bins: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    bin_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """计算金额口径的分箱统计信息.
    
    参考 scorecardpipeline 实现，所有指标基于金额而非样本数计算。
    列名与样本口径保持一致，便于统一处理。
    
    :param bins: 分箱索引数组
    :param y: 目标变量 (0/1)
    :param amount: 金额数组
    :param bin_labels: 可选的分箱标签列表
    :return: 分箱统计DataFrame，列名与样本口径统一
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
    unique_bins_sorted = unique_bins[sort_order]
    
    # 重新映射 bin_indices
    old_to_new = {int(old_pos): new_pos for new_pos, old_pos in enumerate(sort_order)}
    bin_indices_sorted = np.array([old_to_new[int(idx)] for idx in bin_indices])
    
    # 如果提供了分箱标签，也按相同顺序重新排列
    if bin_labels is not None and len(bin_labels) == len(unique_bins):
        bin_labels_sorted = [bin_labels[int(sort_order[i])] for i in range(len(sort_order))]
        bin_labels = bin_labels_sorted
    
    unique_bins = unique_bins_sorted
    bin_indices = bin_indices_sorted
    
    n_bins = len(unique_bins)
    
    # 计算每个分箱的好金额和坏金额（金额口径的核心）
    good_amounts = np.bincount(bin_indices, weights=(y == 0).astype(float) * amount, minlength=n_bins)
    bad_amounts = np.bincount(bin_indices, weights=y.astype(float) * amount, minlength=n_bins)
    amount_totals = good_amounts + bad_amounts
    
    # 计算占比
    total_amount = amount_totals.sum()
    total_good_amount = good_amounts.sum()
    total_bad_amount = bad_amounts.sum()
    
    amount_ratios = amount_totals / total_amount if total_amount > 0 else np.zeros(n_bins)
    good_amount_ratios = good_amounts / total_good_amount if total_good_amount > 0 else np.zeros(n_bins)
    bad_amount_ratios = bad_amounts / total_bad_amount if total_bad_amount > 0 else np.zeros(n_bins)
    
    # 计算金额口径坏样本率
    bad_rate = np.where(amount_totals > 0, bad_amounts / amount_totals, 0.0)
    
    # 计算WOE和IV（基于金额占比）
    epsilon = 1e-10
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
    
    # 构建DataFrame（使用与样本口径统一的列名）
    data = {
        '分箱': unique_bins,
    }
    
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


def _get_bin_labels(splits: Optional[np.ndarray], bins: np.ndarray) -> List[str]:
    """根据切分点生成分箱标签.
    
    :param splits: 切分点数组
    :param bins: 分箱索引数组
    :return: 分箱标签列表
    """
    unique_bins = np.unique(bins)
    n_unique = len(unique_bins)
    
    # 没有切分点的情况
    if splits is None or len(splits) == 0:
        labels = []
        for b in unique_bins:
            b_int = int(b)
            if b_int == -1:
                labels.append('缺失')
            elif b_int == -2:
                labels.append('特殊')
            else:
                labels.append('(-inf, +inf)')
        return labels
    
    # 有切分点的情况
    n_splits = len(splits)
    labels = []
    
    for b in unique_bins:
        b_int = int(b)
        if b_int == -1:
            labels.append('缺失')
        elif b_int == -2:
            labels.append('特殊')
        elif b_int == 0:
            labels.append(f'(-inf, {splits[0]}]')
        elif b_int >= n_splits:
            labels.append(f'({splits[-1]}, +inf)')
        else:
            labels.append(f'({splits[b_int-1]}, {splits[b_int]}]')
    
    return labels


def _merge_multi_target_tables(
    tables: List[pd.DataFrame],
    target_names: List[str],
    merge_columns: List[str]
) -> pd.DataFrame:
    """合并多目标的分箱表.
    
    参考 scp 实现：使用 merge 基于共同列合并，确保列名对齐正确。
    修复：确保列名顺序正确，分箱详情列在前，各目标列连续排列。
    
    :param tables: 各目标的分箱表列表
    :param target_names: 目标名称列表
    :param merge_columns: 合并键列名（会作为公共列出现在所有目标中）
    :return: 合并后的多级表头DataFrame
    """
    if not tables:
        return pd.DataFrame()
    
    if len(tables) == 1:
        return tables[0]
    
    # 第一个表作为基础
    base_table = tables[0].copy()
    
    # 确定哪些列是公共列（merge_columns）
    available_merge_cols = [c for c in merge_columns if c in base_table.columns]
    # 非公共列
    non_merge_cols = [c for c in base_table.columns if c not in available_merge_cols]
    
    # 修复：重新排列列顺序，确保 merge_columns 在前，非 merge 列在后
    base_table = base_table[available_merge_cols + non_merge_cols]
    
    # 为第一个表设置多级表头
    multi_cols = []
    for col in base_table.columns:
        if col in available_merge_cols:
            multi_cols.append(('分箱详情', col))
        else:
            multi_cols.append((target_names[0], col))
    base_table.columns = pd.MultiIndex.from_tuples(multi_cols)
    
    # 合并后续表 - 使用 merge 基于 merge_columns 合并（参考 scp 实现）
    for i, (table, target_name) in enumerate(zip(tables[1:], target_names[1:]), 1):
        # 为当前表设置多级表头
        table_multi_cols = []
        for col in table.columns:
            if col in available_merge_cols:
                table_multi_cols.append(('分箱详情', col))
            else:
                table_multi_cols.append((target_name, col))
        table_copy = table.copy()
        table_copy.columns = pd.MultiIndex.from_tuples(table_multi_cols)
        
        # 使用 merge 基于分箱详情列合并（参考 scp）
        merge_on = [('分箱详情', c) for c in available_merge_cols]
        base_table = base_table.merge(table_copy, on=merge_on)
    
    return base_table


def feature_bin_stats(
    data: pd.DataFrame,
    feature: Union[str, List[str]],
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    rules: Optional[Union[List, Dict[str, List]]] = None,
    method: str = 'mdlp',
    desc: Optional[Union[str, Dict[str, str]]] = None,
    binner: Optional[BaseBinning] = None,
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    return_cols: Optional[List[str]] = None,
    return_rules: bool = False,
    del_grey: bool = False,
    margins: bool = False,
    amount: Optional[str] = None,
    verbose: int = 0,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """特征分箱统计表，汇总统计特征每个分箱的各项指标信息.
    
    支持单特征或多特征，支持单目标或多逾期标签+逾期天数组合分析。
    当传入 overdue 和 dpds 时，会生成多级表头展示不同标签组合下的分箱统计。
    
    :param data: 数据集
    :param feature: 特征名称或特征名称列表
    :param target: 目标变量名称，默认 None
    :param overdue: 逾期天数字段名称或列表，如 'MOB1' 或 ['MOB1', 'MOB3']
    :param dpds: 逾期定义天数或列表，如 7 或 [0, 7, 30]
        - 逾期天数 > dpds 为坏样本(1)，其他为好样本(0)
    :param rules: 自定义分箱规则，支持 list（单个特征）或 dict（多个特征）
    :param method: 分箱方法，可选：
        - 基础方法: 'uniform'(等宽), 'quantile'(等频), 'tree'(决策树), 'chi_merge'(卡方)
        - 优化方法: 'optimal_ks'(最优KS), 'optimal_iv'(最优IV), 'mdlp'(信息论)
        - 高级方法: 'cart'(CART), 'monotonic'(单调性), 'genetic'(遗传算法),
                    'smooth'(平滑), 'kernel_density'(核密度), 
                    'best_lift'(Best Lift), 'target_bad_rate'(目标坏样本率)
        - 聚类方法: 'kmeans'
        默认: 'mdlp'
    :param desc: 特征描述，支持 str（单个特征）或 dict（多个特征）
    :param binner: 预训练的分箱器，优先级高于 method
    :param max_n_bins: 最大分箱数，默认 5
    :param min_bin_size: 每箱最小样本占比，默认 0.05
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param return_cols: 指定返回的列名列表，默认返回所有列
    :param return_rules: 是否返回分箱规则，默认 False
    :param del_grey: 是否删除逾期天数 (0, dpds] 的灰样本，仅 overdue 起作用时有用
        - True: 剔除灰样本，不同目标下样本数不同，样本数相关列按目标单独显示
        - False: 保留灰样本，不同目标下样本数相同，样本数相关列作为公共列
    :param margins: 是否在分箱表最后添加合计行，默认 False
        - True: 在最后一行显示合计，缺失值和特殊值放在正常分箱之后、合计之前
    :param amount: 金额字段名称，用于金额口径分析。传入后会增加金额总数、金额占比等指标
    :param verbose: 是否输出详细信息，默认 0
    :param kwargs: 其他分箱器参数，如 monotonic='peak' 等
    
    :return: 
        - pd.DataFrame: 特征分箱统计表
        - Tuple[pd.DataFrame, Dict]: 当 return_rules=True 时返回 (统计表, 分箱规则)
    
    **示例**
    
    >>> # 单特征单目标分析
    >>> table = feature_bin_stats(data, 'score', target='target', method='mdlp')
    >>> 
    >>> # 单特征多逾期标签分析
    >>> table = feature_bin_stats(data, 'score', overdue=['MOB1', 'MOB3'], dpds=[0, 7])
    >>> 
    >>> # 多特征分析
    >>> table = feature_bin_stats(data, ['score', 'age'], overdue='MOB1', dpds=7)
    >>> 
    >>> # 使用自定义分箱规则
    >>> table = feature_bin_stats(data, 'score', rules=[300, 500, 700])
    >>> 
    >>> # 使用单调性分箱
    >>> table = feature_bin_stats(data, 'score', method='monotonic', monotonic='peak')
    >>> 
    >>> # 金额口径分析
    >>> table = feature_bin_stats(data, 'score', target='target', amount='loan_amount')
    """
    # 统一处理 feature 参数
    if isinstance(feature, str):
        features = [feature]
    else:
        features = feature
    
    # 统一处理 desc 参数
    if desc is None:
        desc_dict = {f: f for f in features}
    elif isinstance(desc, str):
        desc_dict = {f: desc if f == features[0] else f for f in features}
    else:
        desc_dict = desc
    
    # 检查 overdue 和 dpds 参数
    if overdue is not None and dpds is None:
        raise ValueError("传入 overdue 参数时必须同时传入 dpds")
    
    # 构建目标变量列表
    target_configs = []
    if overdue is not None:
        # 逾期分析模式
        if isinstance(overdue, str):
            overdue = [overdue]
        if isinstance(dpds, int):
            dpds = [dpds]
        
        for mob_col in overdue:
            for d in dpds:
                target_name = f"{mob_col}_{d}+"
                target_configs.append({
                    'name': target_name,
                    'mob_col': mob_col,
                    'dpd': d
                })
    elif target is not None:
        # 普通目标模式
        target_configs = [{'name': target, 'mob_col': None, 'dpd': None}]
    else:
        raise ValueError("必须传入 target 或 overdue+dpds 参数")
    
    # 存储所有特征的结果
    all_feature_tables = []
    all_feature_rules = {}
    
    for feat in features:
        # 获取当前特征的分箱器
        if binner is not None:
            current_binner = deepcopy(binner)
        else:
            # 根据 method 创建分箱器
            # 定义方法到类的映射
            method_mapping = {
                # 基础方法
                'uniform': 'UniformBinning',
                'quantile': 'QuantileBinning',
                'tree': 'TreeBinning',
                'chi_merge': 'ChiMergeBinning',
                # 优化方法
                'optimal_ks': 'OptimalKSBinning',
                'optimal_iv': 'OptimalIVBinning',
                'mdlp': 'MDLPBinning',
                # 高级方法
                'cart': 'CartBinning',
                'monotonic': 'MonotonicBinning',
                'genetic': 'GeneticBinning',
                'smooth': 'SmoothBinning',
                'kernel_density': 'KernelDensityBinning',
                'best_lift': 'BestLiftBinning',
                'target_bad_rate': 'TargetBadRateBinning',
                # 聚类方法
                'kmeans': 'KMeansBinning',
                # 兼容旧版本
                'optimal': 'OptimalBinning',
            }
            
            if method not in method_mapping:
                raise ValueError(
                    f"不支持的分箱方法: {method}。"
                    f"可选方法: {', '.join(sorted(method_mapping.keys()))}"
                )
            
            # 动态导入分箱类
            class_name = method_mapping[method]
            from ..core import binning as binning_module
            binning_class = getattr(binning_module, class_name)
            
            # 构建分箱器参数
            binner_params = {
                'max_n_bins': max_n_bins,
                'min_bin_size': min_bin_size,
                'missing_separate': missing_separate
            }
            
            # 添加额外的kwargs参数（如monotonic='peak'等）
            binner_params.update(kwargs)
            
            # 创建分箱器实例
            current_binner = binning_class(**binner_params)
        
        # 第一个目标用于训练分箱器
        first_target = target_configs[0]
        
        # 准备训练数据
        if first_target['mob_col'] is not None:
            # 逾期模式
            train_data = data[[feat, first_target['mob_col']]].copy()
            y_train = (train_data[first_target['mob_col']] > first_target['dpd']).astype(int)
            
            if del_grey:
                mask = (train_data[first_target['mob_col']] > first_target['dpd']) | (train_data[first_target['mob_col']] == 0)
                train_data = train_data[mask]
                y_train = y_train[mask]
        else:
            # 普通目标模式
            train_data = data[[feat, first_target['name']]].copy()
            y_train = train_data[first_target['name']]
        
        # 应用自定义规则或拟合分箱器
        if rules is not None:
            # 使用自定义规则
            if isinstance(rules, dict) and feat in rules:
                custom_splits = np.array(rules[feat])
            elif isinstance(rules, list):
                custom_splits = np.array(rules)
            else:
                custom_splits = np.array([])
            
            # 手动设置分箱器状态
            current_binner.splits_ = {feat: custom_splits}
            current_binner.feature_types_ = {feat: 'numerical'}
            current_binner.n_bins_ = {feat: len(custom_splits) + 1}
            current_binner._is_fitted = True
            
            # 生成bin_table用于后续的transform
            bins = np.digitize(train_data[feat].values, custom_splits, right=True)
            temp_stats = compute_bin_stats(bins, y_train.values)
            current_binner.bin_tables_ = {feat: temp_stats}
        else:
            # 拟合分箱器
            current_binner.fit(train_data[[feat]], y_train)
        
        # 为每个目标生成分箱表
        feat_tables = []
        target_names = []
        
        # 根据del_grey确定merge_columns
        # merge_columns: 这些列在不同目标下是相同的，放在"分箱详情"层级下
        # 当 del_grey=True 时，不同目标下样本数不同，样本数相关列不应该合并
        # 当 del_grey=False 时，样本数相同，可以合并样本数相关列
        # 注意：样本占比也受 del_grey 影响，因为分母（总样本数）可能不同
        # 列名已统一，无论金额口径还是样本口径都使用相同的列名
        base_merge_cols = ['指标名称', '指标含义', '分箱标签']
        
        if isinstance(del_grey, bool) and del_grey:
            # 剔除灰样本：只保留基础分箱信息作为公共列
            merge_cols = base_merge_cols
        else:
            # 保留灰样本或单目标：样本数和占比也是公共列
            merge_cols = base_merge_cols + ['样本总数', '样本占比']
        
        for target_cfg in target_configs:
            target_name = target_cfg['name']
            target_names.append(target_name)
            
            # 准备数据
            if target_cfg['mob_col'] is not None:
                # 逾期模式：需要包含金额字段（如果有）
                cols_to_select = [feat, target_cfg['mob_col']]
                if amount is not None and amount in data.columns:
                    cols_to_select.append(amount)
                analysis_data = data[cols_to_select].copy()
                y = (analysis_data[target_cfg['mob_col']] > target_cfg['dpd']).astype(int)
                
                # 剔除灰客户：只保留好样本(overdue==0)和坏样本(overdue>dpd)
                # 参考 scp: _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)")
                if isinstance(del_grey, bool) and del_grey:
                    mask = (analysis_data[target_cfg['mob_col']] > target_cfg['dpd']) | (analysis_data[target_cfg['mob_col']] == 0)
                    analysis_data = analysis_data[mask].reset_index(drop=True)
                    y = y[mask].reset_index(drop=True)
            else:
                # 普通目标模式：需要包含金额字段（如果有）
                cols_to_select = [feat, target_name]
                if amount is not None and amount in data.columns:
                    cols_to_select.append(amount)
                analysis_data = data[cols_to_select].copy()
                y = analysis_data[target_name]
            
            # 分箱转换
            X_feat = analysis_data[[feat]]
            # 使用 digitize 进行分箱转换，与分箱器内部逻辑一致
            splits = current_binner.splits_.get(feat, np.array([]))
            x_values = X_feat[feat].values
            # 处理缺失值
            missing_mask = pd.isna(x_values)
            bins = np.digitize(x_values, splits, right=True)
            bins = bins.astype(float)
            bins[missing_mask] = -1  # 缺失值标记为 -1
            
            # 准备金额数据（如果有）
            amount_values = analysis_data[amount].values if amount is not None and amount in analysis_data.columns else None
            
            # 创建分箱表
            splits = current_binner.splits_.get(feat)
            bin_table = _create_bin_table(
                bins=bins,
                y=y.values,
                feature_name=feat,
                desc=desc_dict.get(feat, feat),
                splits=splits,
                amount=amount_values
            )
            
            # 筛选指定列
            if return_cols is not None:
                # 确保基础列存在
                base_cols = ['指标名称', '指标含义', '分箱标签']
                available_cols = [c for c in base_cols + return_cols if c in bin_table.columns]
                bin_table = bin_table[available_cols]
            
            feat_tables.append(bin_table)
            
            if verbose > 0:
                n_samples = len(analysis_data)
                n_bad = y.sum()
                bad_rate = y.mean()
                print(f"特征 {feat} - 目标 {target_name}: 样本数 {n_samples}, 坏样本数 {n_bad}, 坏样本率 {bad_rate:.4f}, 分箱数 {len(bin_table)}")
        
        # 合并多目标表
        if len(feat_tables) > 1:
            merged_table = _merge_multi_target_tables(feat_tables, target_names, merge_cols)
        else:
            merged_table = feat_tables[0]
        
        all_feature_tables.append(merged_table)
        
        # 保存分箱规则
        if return_rules:
            all_feature_rules[feat] = current_binner.splits_.get(feat, np.array([])).tolist()
    
    # 合并多特征表
    if len(all_feature_tables) == 1:
        final_table = all_feature_tables[0]
    else:
        final_table = pd.concat(all_feature_tables, axis=0, ignore_index=True)
    
    # 添加合计行
    if margins:
        final_table = _add_margins(final_table)
    
    if return_rules:
        return final_table, all_feature_rules
    return final_table


class FeatureAnalyzer:
    """特征分析器.
    
    提供批量特征分析、多维度对比分析等功能。
    """
    
    def __init__(
        self,
        method: str = 'mdlp',
        max_n_bins: int = 5,
        min_bin_size: float = 0.05,
        missing_separate: bool = True
    ):
        """初始化特征分析器.
        
        :param method: 分箱方法
        :param max_n_bins: 最大分箱数
        :param min_bin_size: 每箱最小样本占比
        :param missing_separate: 是否将缺失值单独分箱
        """
        self.method = method
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.missing_separate = missing_separate
        self.binners_: Dict[str, BaseBinning] = {}
    
    def analyze(
        self,
        data: pd.DataFrame,
        feature: Union[str, List[str]],
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        target: Optional[str] = None,
        desc: Optional[Union[str, Dict[str, str]]] = None,
        amount: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """批量分析多个特征.
        
        :param data: 数据集
        :param feature: 特征名称或特征名称列表
        :param overdue: 逾期天数字段
        :param dpds: 逾期定义天数
        :param target: 目标变量（可选，与overdue+dpds二选一）
        :param desc: 特征描述
        :param amount: 金额字段名称，用于金额口径分析
        :param kwargs: 其他参数传递给 feature_bin_stats
        :return: 合并的分箱统计表
        """
        return feature_bin_stats(
            data=data,
            feature=feature,
            overdue=overdue,
            dpds=dpds,
            target=target,
            method=self.method,
            desc=desc,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            missing_separate=self.missing_separate,
            amount=amount,
            **kwargs
        )
    
    def compare_targets(
        self,
        data: pd.DataFrame,
        feature: str,
        overdue_list: List[str],
        dpds_list: List[int],
        desc: str = "",
        **kwargs
    ) -> pd.DataFrame:
        """对比不同目标下的特征分箱效果.
        
        :param data: 数据集
        :param feature: 特征名称
        :param overdue_list: 多个逾期天数字段列表
        :param dpds_list: 多个逾期定义天数列表
        :param desc: 特征描述
        :param kwargs: 其他参数
        :return: 对比分析表
        """
        return feature_bin_stats(
            data=data,
            feature=feature,
            overdue=overdue_list,
            dpds=dpds_list,
            method=self.method,
            desc=desc,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            missing_separate=self.missing_separate,
            **kwargs
        )
    
    def get_iv_summary(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str
    ) -> pd.DataFrame:
        """获取特征IV值汇总表.
        
        :param data: 数据集
        :param features: 特征名称列表
        :param target: 目标变量
        :return: IV值汇总表
        """
        results = []
        
        for feat in features:
            table = feature_bin_stats(
                data=data,
                feature=feat,
                target=target,
                method=self.method,
                max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size,
                return_cols=['指标IV值']
            )
            
            # 处理单层列和多层级列的情况
            if isinstance(table.columns, pd.MultiIndex):
                # 多层级列：获取第一行的指标IV值
                iv_col = [c for c in table.columns if c[1] == '指标IV值']
                if iv_col:
                    iv_value = table[iv_col[0]].iloc[0]
                else:
                    iv_value = 0.0
            else:
                # 单层列
                iv_value = table['指标IV值'].iloc[0]
            
            results.append({
                '特征名称': feat,
                'IV值': iv_value,
                '分箱数': len(table)
            })
        
        # 排序
        iv_summary = pd.DataFrame(results).sort_values('IV值', ascending=False)
        
        # 添加IV解释
        def iv_interpret(iv):
            if iv < 0.02:
                return '无预测力'
            elif iv < 0.1:
                return '弱预测力'
            elif iv < 0.3:
                return '中等预测力'
            elif iv < 0.5:
                return '强预测力'
            else:
                return '超强预测力(需检查)'
        
        iv_summary['预测力'] = iv_summary['IV值'].apply(iv_interpret)
        
        return iv_summary
