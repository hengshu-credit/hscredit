"""特征分析模块.

提供特征分箱统计分析与自动化特征分析输出功能，支持多逾期标签、
多逾期天数组合分析以及 Excel 报告生成。
"""

import os
import traceback
from copy import deepcopy
from typing import Union, List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet
from tqdm import tqdm

from ..core.binning import OptimalBinning
from ..core.binning.base import BaseBinning
from ..core.viz import bin_plot, corr_plot, distribution_plot, hist_plot, ks_plot
from ..core.metrics._binning import compute_bin_stats, add_margins
from ..excel import ExcelWriter, dataframe2excel
from ..utils import init_setting


def _create_bin_table(
    bins: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    desc: str = "",
    splits: Optional[np.ndarray] = None,
    amount: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """创建分箱统计表。"""
    bin_labels = _get_bin_labels(splits, bins)

    if amount is not None:
        stats = compute_bin_stats(
            bins,
            y,
            target_type='amount_weighted',
            amount=amount,
            bin_labels=bin_labels,
        )
    else:
        stats = compute_bin_stats(bins, y, target_type='binary', bin_labels=bin_labels)

    stats.insert(0, '指标含义', desc if desc else feature_name)
    stats.insert(0, '指标名称', feature_name)

    if '分箱' in stats.columns:
        stats = stats.drop(columns=['分箱'])

    return stats


def _get_bin_labels(splits: Optional[np.ndarray], bins: np.ndarray) -> List[str]:
    """根据切分点生成分箱标签。"""
    unique_bins = np.unique(bins)

    if splits is None or len(splits) == 0:
        labels = []
        for current_bin in unique_bins:
            bin_value = int(current_bin)
            if bin_value == -1:
                labels.append('缺失')
            elif bin_value == -2:
                labels.append('特殊')
            else:
                labels.append('[-inf, +inf)')
        return labels

    n_splits = len(splits)
    labels = []
    for current_bin in unique_bins:
        bin_value = int(current_bin)
        if bin_value == -1:
            labels.append('缺失')
        elif bin_value == -2:
            labels.append('特殊')
        elif bin_value == 0:
            labels.append(f'[-inf, {splits[0]})')
        elif bin_value >= n_splits:
            labels.append(f'[{splits[-1]}, +inf)')
        else:
            labels.append(f'[{splits[bin_value - 1]}, {splits[bin_value]})')

    return labels


def _merge_multi_target_tables(
    tables: List[pd.DataFrame],
    target_names: List[str],
    merge_columns: List[str],
) -> pd.DataFrame:
    """合并多目标的分箱表。"""
    if not tables:
        return pd.DataFrame()

    if len(tables) == 1:
        return tables[0]

    base_table = tables[0].copy()
    available_merge_cols = [column for column in merge_columns if column in base_table.columns]
    non_merge_cols = [column for column in base_table.columns if column not in available_merge_cols]
    base_table = base_table[available_merge_cols + non_merge_cols]

    multi_cols = []
    for column in base_table.columns:
        if column in available_merge_cols:
            multi_cols.append(('分箱详情', column))
        else:
            multi_cols.append((target_names[0], column))
    base_table.columns = pd.MultiIndex.from_tuples(multi_cols)

    for table, target_name in zip(tables[1:], target_names[1:]):
        table_multi_cols = []
        for column in table.columns:
            if column in available_merge_cols:
                table_multi_cols.append(('分箱详情', column))
            else:
                table_multi_cols.append((target_name, column))
        table_copy = table.copy()
        table_copy.columns = pd.MultiIndex.from_tuples(table_multi_cols)

        merge_on = [('分箱详情', column) for column in available_merge_cols]
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
    binner: Optional[Union[BaseBinning, Dict[str, BaseBinning]]] = None,
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    prebinning: Optional[Union[str, BaseBinning, Dict]] = 'quantile',
    prebinning_params: Optional[Dict[str, Any]] = None,
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
    :param rules: 自定义分箱规则，支持 list（所有特征统一规则）或 dict（按特征名映射规则）。
        对 rules 中未包含的特征，按 method 参数重新训练分箱器。
        优先级: binner > rules > method
    :param method: 分箱方法，可选：
        - 基础方法: 'uniform'(等宽), 'quantile'(等频), 'tree'(决策树), 'chi'(卡方)
        - 优化方法: 'best_ks'(最优KS), 'best_iv'(最优IV), 'mdlp'(信息论)
        - 运筹规划方法: 'or_tools'(OR-Tools整数规划，需安装 ortools)
        - 高级方法: 'cart'(CART), 'monotonic'(单调性), 'genetic'(遗传算法),
                    'smooth'(平滑), 'kernel_density'(核密度), 
                    'best_lift'(Best Lift), 'target_bad_rate'(目标坏样本率)
        - 聚类方法: 'kmeans'
        默认: 'mdlp'
    :param desc: 特征描述，支持 str（单个特征）或 dict（多个特征）
    :param binner: 分箱器，支持以下三种传入方式：
        - BaseBinning（已训练）: 对其中已包含的特征直接使用，未包含的特征按 method 参数重新训练
        - BaseBinning（未训练）: 作为模板，对每个特征 deepcopy 后 fit
        - Dict[str, BaseBinning]: 按特征名映射的已训练分箱器字典，未包含的特征按 method 参数重新训练
        优先级: binner > rules > method
    :param max_n_bins: 最大分箱数，默认 5
    :param min_bin_size: 每箱最小样本占比，默认 0.05
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param prebinning: 预分箱配置，参数格式与 OptimalBinning 保持一致，默认 'quantile'。
        - None: 不使用预分箱
        - str: 预分箱方法名（如 'quantile' / 'tree'）
        - BaseBinning: 预分箱器实例
        - Dict: 预分箱配置字典
    :param prebinning_params: 预分箱参数（传给 OptimalBinning.prebinning_params）。
        默认 None，此时会使用 {'max_n_bins': 100}，即先等频100箱再合并。
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
    
    # 构建默认分箱器参数（在循环外，避免重复计算）
    method_for_binner = 'mdlp' if method == 'optimal' else method
    effective_prebinning_params = {'max_n_bins': 100}
    if prebinning_params:
        effective_prebinning_params.update(prebinning_params)

    default_binner_params = {
        'method': method_for_binner,
        'max_n_bins': max_n_bins,
        'min_bin_size': min_bin_size,
        'missing_separate': missing_separate,
        'prebinning': prebinning,
        'prebinning_params': effective_prebinning_params,
    }

    # MDLP默认开启后处理微调，用户可通过 kwargs 覆盖
    if method_for_binner == 'mdlp':
        default_binner_params.setdefault('lift_refine', True)
        default_binner_params.setdefault('lift_focus_weight', 3.0)
        default_binner_params.setdefault('sample_stability_weight', 0.2)
        default_binner_params.setdefault('monotonic_bonus_weight', 0.4)
        default_binner_params.setdefault('lift_refine_max_bins', max_n_bins)

    # 添加额外参数（如monotonic='auto_asc_desc'）
    default_binner_params.update(kwargs)

    # 显式关闭预分箱
    if prebinning is None:
        default_binner_params.pop('prebinning', None)
        default_binner_params.pop('prebinning_params', None)

    for feat in features:
        # === 确定当前特征的分箱器 ===
        # 优先级: binner(已训练且覆盖该特征) > rules(覆盖该特征) > binner(未训练模板) > method(新建)
        current_binner = None
        need_fit = False

        # 1. 检查 binner 是否覆盖该特征
        if binner is not None:
            if isinstance(binner, dict):
                # 按特征名映射的分箱器字典
                if feat in binner:
                    feat_binner = binner[feat]
                    if getattr(feat_binner, '_is_fitted', False) and hasattr(feat_binner, 'splits_') and feat in feat_binner.splits_:
                        current_binner = feat_binner  # 直接使用已训练的分箱器
            elif isinstance(binner, BaseBinning):
                if getattr(binner, '_is_fitted', False) and hasattr(binner, 'splits_') and feat in binner.splits_:
                    # 已训练的分箱器且包含该特征 → 直接使用
                    current_binner = binner
                elif not getattr(binner, '_is_fitted', False):
                    # 未训练的分箱器 → 作为模板 deepcopy 后训练
                    current_binner = deepcopy(binner)
                    need_fit = True

        # 2. 检查 rules 是否覆盖该特征
        feat_rule = None
        if current_binner is None and rules is not None:
            if isinstance(rules, dict) and feat in rules:
                feat_rule = np.array(rules[feat])
            elif isinstance(rules, list):
                feat_rule = np.array(rules)

        # 3. 如果 binner 和 rules 都没覆盖，创建新的分箱器
        if current_binner is None and feat_rule is None:
            current_binner = OptimalBinning(**default_binner_params)
            need_fit = True

        # 需要训练或应用规则时，准备训练数据
        if need_fit or feat_rule is not None:
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

            if feat_rule is not None:
                # 从规则生成分箱器
                current_binner = OptimalBinning(method='quantile')
                current_binner.splits_ = {feat: feat_rule}
                current_binner.feature_types_ = {feat: 'numerical'}
                current_binner.n_bins_ = {feat: len(feat_rule) + 1}
                current_binner._is_fitted = True

                # 生成bin_table用于后续的transform
                bins_tmp = np.digitize(train_data[feat].values, feat_rule, right=True)
                temp_stats = compute_bin_stats(bins_tmp, y_train.values, target_type='binary')
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
        final_table = add_margins(final_table)
    
    if return_rules:
        return final_table, all_feature_rules
    return final_table


def benchmark_binning_methods(
    data: pd.DataFrame,
    feature: str,
    overdue_col: str = 'MOB1',
    dpds: Optional[List[int]] = None,
    max_n_bins: int = 5,
    min_bin_size: float = 0.01,
    monotonic: str = 'auto_asc_desc',
    hscredit_methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """逐方法对比 hscredit 内部分箱效果。

    仅使用 hscredit 内置分箱器，不依赖额外第三方分箱库。
    重点指标：头部/尾部 Lift、头尾差(edge_gap)、是否单调。
    """
    if dpds is None:
        dpds = [3, 0]
    if hscredit_methods is None:
        hscredit_methods = ['mdlp', 'cart', 'chi', 'tree', 'kmeans', 'best_ks', 'best_iv', 'quantile']

    x = pd.to_numeric(data[feature], errors='coerce')

    def _eval_splits(x_s: pd.Series, y_s: pd.Series, splits: Optional[List], model_name: str, dpd: int) -> Dict[str, Any]:
        mask = x_s.notna() & y_s.notna()
        xv = x_s[mask].values.astype(float)
        yv = y_s[mask].values.astype(int)
        if len(xv) == 0:
            return {'method': model_name, 'dpd': dpd, 'error': 'no valid samples'}

        sp = np.array(splits if splits is not None else [], dtype=float)
        bins = np.digitize(xv, sp, right=True)
        n_bins = len(sp) + 1

        counts = np.bincount(bins, minlength=n_bins).astype(float)
        bad = np.bincount(bins, weights=yv, minlength=n_bins).astype(float)
        bad_rate = bad / np.maximum(counts, 1.0)
        overall_bad_rate = max(yv.mean(), 1e-12)
        lift = bad_rate / overall_bad_rate

        diffs = np.diff(bad_rate)
        asc = bool(np.all(diffs >= -1e-12))
        desc = bool(np.all(diffs <= 1e-12))
        nz = np.sign(diffs)
        nz = nz[nz != 0]
        turns = 0 if len(nz) <= 1 else int(np.sum(nz[1:] * nz[:-1] < 0))

        return {
            'method': model_name,
            'dpd': dpd,
            'n_bins': int(n_bins),
            'head_lift': float(lift[0]),
            'tail_lift': float(lift[-1]),
            'edge_gap': float(abs(lift[-1] - lift[0])),
            'max_lift': float(np.max(lift)),
            'min_lift': float(np.min(lift)),
            'monotonic': bool(asc or desc),
            'turns': turns,
            'splits': sp.tolist(),
        }

    rows = []

    for d in dpds:
        y = (data[overdue_col] > d).astype(int)

        for method in hscredit_methods:
            try:
                binner = OptimalBinning(
                    method=method,
                    max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size,
                    monotonic=monotonic,
                    prebinning='quantile',
                    prebinning_params={'max_n_bins': 100},
                    lift_refine=True,
                )
                binner.fit(pd.DataFrame({feature: x}), y)
                rows.append(_eval_splits(x, y, binner.splits_.get(feature, []), f'hscredit-{method}', d))
            except Exception as e:
                rows.append({'method': f'hscredit-{method}', 'dpd': d, 'error': str(e)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if 'error' in result.columns:
        ok = result[result['error'].isna()] if result['error'].notna().any() else result
    else:
        ok = result

    if not ok.empty:
        ok = ok.sort_values(['dpd', 'monotonic', 'edge_gap', 'head_lift'], ascending=[True, False, False, False])
        return ok.reset_index(drop=True)

    return result.reset_index(drop=True)


def auto_feature_analysis(
    data: pd.DataFrame,
    features=None,
    target="target",
    overdue=None,
    dpds=None,
    date=None,
    data_summary_comment="",
    freq="M",
    excel_writer=None,
    sheet="分析报告",
    start_col=2,
    start_row=2,
    dropna=False,
    writer_params=None,
    bin_params=None,
    feature_map=None,
    corr=False,
    pictures=None,
    suffix="",
    output_dir="model_report",
    margins=False,
    amount=None,
    image_table_gap_rows=None,
):
    """自动特征分析.

    用于三方数据评估或自有评分效果评估。生成包含数据集概况、特征分箱统计、
    KS 曲线、分布图等内容的 Excel 分析结果。

    :param data: 需要评估的数据集，需要包含目标变量
    :param features: 需要进行分析的特征名称，支持单个字符串或列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称，传入时会覆盖 target 参数
    :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本
    :param date: 日期列，用于时间维度分布分析
    :param freq: 日期统计粒度，默认按月 "M"
    :param data_summary_comment: 数据备注信息
    :param excel_writer: Excel 文件路径或 ExcelWriter 对象
    :param sheet: 工作表名称
    :param start_col: 起始列
    :param start_row: 起始行
    :param dropna: 是否剔除缺失值
    :param writer_params: Excel 写入器初始化参数
    :param bin_params: 分箱统计参数，支持 feature_bin_stats 的参数
    :param feature_map: 特征名称映射字典
    :param corr: 是否计算特征相关性
    :param pictures: 需要生成的图片列表，支持 ["ks", "hist", "bin"]
    :param suffix: 文件名后缀，避免同名文件被覆盖
    :param output_dir: 图片输出目录
    :param margins: 是否在每个特征分箱表末尾添加合计行，默认 False
    :param amount: 放款金额或余额字段名称。传入后同时生成订单口径和金额口径两张分箱表
    :param image_table_gap_rows: 图片区与分箱表之间的额外空行数
    :return: (end_row, end_col) 分析结束位置

    示例::

        >>> from hscredit.report.feature_analyzer import auto_feature_analysis
        >>> auto_feature_analysis(data, features=['feature1'], target='target', excel_writer='分析结果.xlsx')
    """
    if writer_params is None:
        writer_params = {}
    if bin_params is None:
        bin_params = {}
    if feature_map is None:
        feature_map = {}
    if pictures is None:
        pictures = ["bin", "ks", "hist"]

    init_setting()

    data = data.copy()
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(features, (list, tuple)):
        features = [features]

    if overdue and not isinstance(overdue, list):
        overdue = [overdue]

    if dpds and not isinstance(dpds, list):
        dpds = [dpds]

    if overdue:
        target = f"{overdue[0]} {dpds[0]}+"
        data[target] = (data[overdue[0]] > dpds[0]).astype(int)

    if date is not None and date in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date]):
        converted_date = pd.to_datetime(data[date], errors='coerce')
        if converted_date.notna().sum() < len(data) * 0.5 and pd.api.types.is_numeric_dtype(data[date]):
            converted_date_excel = pd.to_datetime(data[date], unit='D', errors='coerce')
            if converted_date_excel.notna().sum() > converted_date.notna().sum():
                converted_date = converted_date_excel
        data[date] = converted_date

    if isinstance(excel_writer, ExcelWriter):
        writer = excel_writer
    else:
        writer = ExcelWriter(**writer_params)

    worksheet = writer.get_sheet_by_name(sheet)

    if image_table_gap_rows is None:
        image_table_gap_rows = 2 if getattr(writer, "system", "windows") == "windows" else 1

    if bin_params and "del_grey" in bin_params and bin_params.get("del_grey"):
        merge_columns = ["指标名称", "指标含义", "分箱标签"]
    else:
        merge_columns = ["指标名称", "指标含义", "分箱标签", "样本总数", "样本占比"]

    return_cols = []
    if bin_params:
        if "return_cols" in bin_params and bin_params.get("return_cols"):
            return_cols = bin_params.pop("return_cols")
            if not isinstance(return_cols, (list, np.ndarray)):
                return_cols = [return_cols]
            return_cols = list(set(return_cols) - set(merge_columns))
        else:
            return_cols = []

    max_columns_len = len(merge_columns) + len(return_cols) * len(overdue) * len(dpds) \
        if overdue and len(overdue) > 0 else len(merge_columns) + len(return_cols)

    end_row, end_col = writer.insert_value2sheet(
        worksheet, (start_row, start_col), value="数据有效性分析报告",
        style="header_middle", end_space=(start_row, start_col + max_columns_len - 1)
    )

    if date is not None and date in data.columns:
        if data[date].dtype.name in ["str", "object"]:
            start_date = pd.to_datetime(data[date]).min().strftime("%Y-%m-%d")
            end_date = pd.to_datetime(data[date]).max().strftime("%Y-%m-%d")
        else:
            start_date = data[date].min().strftime("%Y-%m-%d")
            end_date = data[date].max().strftime("%Y-%m-%d")

        dataset_summary = pd.DataFrame(
            [[start_date, end_date, len(data), data[target].sum(),
              data[target].sum() / len(data), data_summary_comment]],
            columns=["开始时间", "结束时间", "样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(
            dataset_summary, writer, worksheet, percent_cols=["坏客户占比"],
            start_row=end_row + 2, title="样本总体分布情况"
        )

        distribution = distribution_plot(
            data, date=date, freq=freq, target=target,
            save=os.path.join(output_dir, f"sample_time_distribution{suffix}.png"), result=True
        )
        end_row, end_col = writer.insert_value2sheet(
            worksheet, (end_row + 2, start_col), value="样本时间分布情况", style="header",
            end_space=(end_row + 2, start_col + len(distribution.columns) - 1)
        )
        end_row, end_col = writer.insert_pic2sheet(
            worksheet, os.path.join(output_dir, f"sample_time_distribution{suffix}.png"),
            (end_row + 1, start_col), figsize=(720, 370)
        )
        end_row, end_col = dataframe2excel(
            distribution, writer, worksheet,
            percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率"],
            condition_cols=["坏样本率"], start_row=end_row
        )
        end_row += 2
    else:
        dataset_summary = pd.DataFrame(
            [[len(data), data[target].sum(), data[target].sum() / len(data), data_summary_comment]],
            columns=["样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(
            dataset_summary, writer, worksheet, percent_cols=["坏客户占比"],
            start_row=end_row + 2, title="样本总体分布情况"
        )
        end_row += 2

    if corr:
        temp = data[features].select_dtypes(include="number")
        corr_plot(
            temp, save=os.path.join(output_dir, f"auto_report_corr_plot{suffix}.png"),
            annot=True if len(temp.columns) <= 10 else False,
            fontsize=14 if len(temp.columns) <= 10 else 12
        )
        end_row, end_col = dataframe2excel(
            temp.corr(), writer, worksheet, color_cols=list(temp.columns),
            start_row=end_row, figures=[os.path.join(output_dir, f"auto_report_corr_plot{suffix}.png")],
            title="数值类变量相关性",
            figsize=(min(60 * len(temp.columns), 1080), min(55 * len(temp.columns), 950)),
            index=True, custom_cols=list(temp.columns), custom_format="0.00"
        )
        end_row += 2

    end_row, end_col = writer.insert_value2sheet(
        worksheet, (end_row, start_col), value="数值类特征 OR 评分效果评估",
        style="header_middle", end_space=(end_row, start_col + max_columns_len - 1)
    )

    use_amount = amount is not None and amount in data.columns

    features_iter = tqdm(features)
    for col in features_iter:
        features_iter.set_postfix(feature=feature_map.get(col, col))
        try:
            if overdue is None:
                cols_needed = [col, target]
            else:
                cols_needed = list(set([col, target] + overdue))
            if use_amount:
                cols_needed = list(set(cols_needed + [amount]))
            temp = data[cols_needed]

            if isinstance(dropna, bool) and dropna is True:
                temp = temp.dropna(subset=col).reset_index(drop=True)
            elif isinstance(dropna, (float, int, str)):
                temp = temp[temp[col] != dropna].reset_index(drop=True)

            actual_target = target
            if overdue:
                actual_target = f"{overdue[0]} {dpds[0]}+"

            sample_table = feature_bin_stats(
                temp, col, overdue=overdue, dpds=dpds,
                desc=f"{feature_map.get(col, col)}", target=target,
                margins=margins,
                **bin_params
            )

            if use_amount:
                amount_table = feature_bin_stats(
                    temp, col, overdue=overdue, dpds=dpds,
                    desc=f"{feature_map.get(col, col)}", target=target,
                    amount=amount,
                    margins=margins,
                    **bin_params
                )
            else:
                amount_table = None

            sample_title_columns_len = len(sample_table.columns)
            amount_title_columns_len = len(amount_table.columns) if (use_amount and amount_table is not None) else 0

            if return_cols:
                if sample_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                    _merge_cols_for_title = [("分箱详情", c) for c in merge_columns]
                else:
                    _merge_cols_for_title = merge_columns
                sample_title_columns_len = len(
                    _merge_cols_for_title + [
                        c for c in sample_table.columns
                        if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                        or (not isinstance(c, (tuple, list)) and c in return_cols)
                        or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                    ]
                )

                if use_amount and amount_table is not None:
                    if amount_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                        _merge_cols_amt_for_title = [("分箱详情", c) for c in merge_columns]
                    else:
                        _merge_cols_amt_for_title = merge_columns
                    amount_title_columns_len = len(
                        _merge_cols_amt_for_title + [
                            c for c in amount_table.columns
                            if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                            or (not isinstance(c, (tuple, list)) and c in return_cols)
                            or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                        ]
                    )

            if pictures and len(pictures) > 0:
                if "bin" in pictures:
                    if sample_table.columns.nlevels > 1:
                        level1_cols = sample_table.columns.get_level_values(0).unique().tolist()
                        target_col = actual_target if actual_target in level1_cols else level1_cols[-1] if len(level1_cols) > 1 else level1_cols[0]
                        plot_table = sample_table[["分箱详情", target_col]]
                        plot_table.columns = [c[-1] for c in plot_table.columns]
                    else:
                        plot_table = sample_table.copy()

                    if "分箱标签" in plot_table.columns:
                        plot_table.rename(columns={"分箱标签": "分箱"}, inplace=True)

                    bin_plot(
                        plot_table, desc=f"{feature_map.get(col, col)}", figsize=(10, 5),
                        anchor=0.935, save=os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png")
                    )

                if temp[col].dtypes.name not in ['object', 'str', 'category']:
                    if "ks" in pictures:
                        plot_source = temp.dropna().reset_index(drop=True)
                        has_ks = len(plot_source) > 0 and plot_source[col].nunique() > 1 and plot_source[actual_target].nunique() > 1
                        if has_ks:
                            ks_plot(
                                plot_source[col], plot_source[actual_target], figsize=(10, 5),
                                title=f"{feature_map.get(col, col)}",
                                save=os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png")
                            )
                    if "hist" in pictures:
                        plot_source = temp.dropna().reset_index(drop=True)
                        if len(plot_source) > 0:
                            hist_plot(
                                plot_source[col], y_true=plot_source[actual_target], figsize=(10, 6),
                                desc=f"{feature_map.get(col, col)} 好客户 VS 坏客户",
                                bins=30, anchor=1.11, fontsize=14,
                                labels={0: "好客户", 1: "坏客户"},
                                save=os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png")
                            )

            if use_amount and amount_table is not None:
                title_span = sample_title_columns_len + 1 + amount_title_columns_len
            else:
                title_span = sample_title_columns_len

            if (len(temp) < len(data)) and (isinstance(dropna, bool) and dropna is True) or \
               isinstance(dropna, (float, int, str)):
                end_row, end_col = writer.insert_value2sheet(
                    worksheet, (end_row + 2, start_col),
                    value=f"数据字段: {feature_map.get(col, col)} (缺失率: {round((1 - len(temp) / len(data)) * 100, 2)}%)",
                    style="header", end_space=(end_row + 2, start_col + title_span - 1)
                )
            else:
                end_row, end_col = writer.insert_value2sheet(
                    worksheet, (end_row + 2, start_col),
                    value=f"数据字段: {feature_map.get(col, col)}",
                    style="header", end_space=(end_row + 2, start_col + title_span - 1)
                )

            if pictures and len(pictures) > 0:
                chart_row = end_row + 1
                if "bin" in pictures:
                    end_row, end_col = writer.insert_pic2sheet(
                        worksheet, os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png"),
                        (chart_row, start_col), figsize=(600, 350)
                    )
                if temp[col].dtypes.name not in ['object', 'str', 'category'] and temp[col].isnull().sum() != len(temp):
                    if "ks" in pictures and has_ks:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png"),
                            (chart_row, end_col - 1), figsize=(600, 350)
                        )
                    if "hist" in pictures:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png"),
                            (chart_row, end_col - 1), figsize=(600, 350)
                        )

            table_start_row = end_row + image_table_gap_rows
            if return_cols:
                if sample_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                    sample_merge_cols = [("分箱详情", c) for c in merge_columns]
                else:
                    sample_merge_cols = merge_columns
                end_row, end_col = dataframe2excel(
                    sample_table[
                        sample_merge_cols + [
                            c for c in sample_table.columns
                            if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                            or (not isinstance(c, (tuple, list)) and c in return_cols)
                            or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                        ]
                    ], writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=table_start_row
                )
            else:
                end_row, end_col = dataframe2excel(
                    sample_table, writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=table_start_row
                )

            if use_amount and amount_table is not None:
                amount_start_col = end_col + 1
                if return_cols:
                    if amount_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                        amount_merge_cols = [("分箱详情", c) for c in merge_columns]
                    else:
                        amount_merge_cols = merge_columns
                    dataframe2excel(
                        amount_table[
                            amount_merge_cols + [
                                c for c in amount_table.columns
                                if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                                or (not isinstance(c, (tuple, list)) and c in return_cols)
                                or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                            ]
                        ], writer, worksheet,
                        percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                        condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                        merge=True, fill=True,
                        start_row=table_start_row, start_col=amount_start_col
                    )
                else:
                    dataframe2excel(
                        amount_table, writer, worksheet,
                        percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                        condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                        merge=True, fill=True,
                        start_row=table_start_row, start_col=amount_start_col
                    )

        except Exception:
            print(f"数据字段 {col} 分析时发生异常，请排查数据中是否存在异常:\n{traceback.format_exc()}")

    if not isinstance(excel_writer, ExcelWriter) and not isinstance(sheet, Worksheet):
        writer.save(excel_writer)

    return end_row, end_col
