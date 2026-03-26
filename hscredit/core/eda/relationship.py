"""特征标签关系模块.

提供IV分析、WOE分箱、单调性检验等特征与目标变量关系分析功能.
主要复用 hscredit.core.metrics 的功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

from .utils import validate_dataframe, validate_binary_target, iv_rating


def iv_analysis(df: pd.DataFrame,
                feature: str,
                target: str,
                n_bins: int = 10,
                method: str = 'quantile') -> Dict[str, Union[str, float, pd.DataFrame]]:
    """单变量IV分析.
    
    复用 hscredit.core.metrics.IV_table
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param n_bins: 分箱数
    :param method: 分箱方法
    :return: IV分析结果字典，包含[特征名, IV值, 预测能力, 分箱明细]
    
    Example:
        >>> result = iv_analysis(df, 'age', 'fpd15')
        >>> print(f"IV值: {result['IV值']}, 预测能力: {result['预测能力']}")
        >>> print(result['分箱明细'])
    """
    validate_dataframe(df, required_cols=[feature, target])
    validate_binary_target(df[target])
    
    # 复用metrics模块
    from ..metrics import iv_table
    
    # 计算IV表
    iv_df = iv_table(df[target], df[feature], method=method, max_n_bins=n_bins)
    
    # 计算总IV
    iv_value = iv_df['分档IV值'].sum()
    
    return {
        '特征名': feature,
        'IV值': round(iv_value, 4),
        '预测能力': iv_rating(iv_value),
        '分箱数': len(iv_df),
        '分箱明细': iv_df,
    }


def batch_iv_analysis(df: pd.DataFrame,
                     features: List[str],
                     target: str,
                     n_bins: int = 10,
                     method: str = 'quantile',
                     return_details: bool = False) -> pd.DataFrame:
    """批量IV分析.
    
    复用 hscredit.core.metrics.batch_iv
    
    :param df: 输入数据
    :param features: 特征列表
    :param target: 目标变量名
    :param n_bins: 分箱数
    :param method: 分箱方法
    :param return_details: 是否返回详细分箱结果
    :return: IV分析结果DataFrame，列包括[特征名, IV值, 预测能力, 分箱数]
    
    Example:
        >>> iv_result = batch_iv_analysis(df, ['age', 'income', 'score'], 'fpd15')
        >>> print(iv_result[['特征名', 'IV值', '预测能力']].sort_values('IV值', ascending=False))
    """
    validate_dataframe(df, required_cols=[target])
    validate_binary_target(df[target])
    
    # 批量计算IV
    results = []
    detail_results = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        try:
            result = iv_analysis(df, feature, target, n_bins, method)
            results.append({
                '特征名': result['特征名'],
                'IV值': result['IV值'],
                '预测能力': result['预测能力'],
                '分箱数': result['分箱数'],
            })
            
            if return_details:
                detail_results[feature] = result['分箱明细']
                
        except Exception as e:
            results.append({
                '特征名': feature,
                'IV值': np.nan,
                '预测能力': '计算失败',
                '分箱数': 0,
            })
    
    result_df = pd.DataFrame(results).sort_values('IV值', ascending=False).reset_index(drop=True)
    
    if return_details:
        return result_df, detail_results
    
    return result_df


def woe_analysis(df: pd.DataFrame,
                feature: str,
                target: str,
                n_bins: int = 10,
                method: str = 'quantile') -> pd.DataFrame:
    """WOE分箱分析.
    
    复用 hscredit.core.metrics.IV_table 获取WOE值
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param n_bins: 分箱数
    :param method: 分箱方法
    :return: WOE分析DataFrame
    
    Example:
        >>> woe_df = woe_analysis(df, 'age', 'fpd15')
        >>> print(woe_df[['分箱', '分箱标签', 'WOE值', 'IV值']])
    """
    validate_dataframe(df, required_cols=[feature, target])
    validate_binary_target(df[target])
    
    # 复用metrics模块
    from ..metrics import iv_table
    
    # 计算IV表（包含WOE值）
    iv_df = iv_table(df[target], df[feature], method=method, max_n_bins=n_bins)
    
    # 选择关键列并重命名
    result = iv_df[['分箱标签', '样本总数', '好样本数', '坏样本数', 
                    '坏样本率', '分档WOE值', '分档IV值']].copy()
    result.columns = ['分箱', '样本数', '好样本数', '坏样本数', 
                     '逾期率', 'WOE值', 'IV值']
    
    # 添加LIFT值
    overall_bad_rate = df[target].mean()
    result['LIFT值'] = (result['逾期率'] / overall_bad_rate).round(4)
    
    return result


def binning_bad_rate(df: pd.DataFrame,
                    feature: str,
                    target: str,
                    n_bins: int = 10,
                    method: str = 'quantile') -> pd.DataFrame:
    """分箱逾期率分析.
    
    复用 hscredit.core.metrics.compute_bin_stats
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param n_bins: 分箱数
    :param method: 分箱方法
    :return: 分箱逾期率DataFrame
    
    Example:
        >>> bin_df = binning_bad_rate(df, 'score', 'fpd15', n_bins=10)
        >>> print(bin_df[['分箱', '样本数', '逾期率', 'LIFT值']])
    """
    validate_dataframe(df, required_cols=[feature, target])
    validate_binary_target(df[target])
    
    # 复用metrics模块
    from ..metrics import compute_bin_stats
    from ..binning import OptimalBinning
    
    # 数据清洗
    valid_mask = df[feature].notna() & df[target].notna()
    X = df.loc[valid_mask, feature]
    y = df.loc[valid_mask, target]
    
    # 分箱 - 使用DataFrame保持列名一致性
    X_df = X.to_frame(name=feature)
    binner = OptimalBinning(method=method, max_n_bins=n_bins, verbose=False)
    binner.fit(X_df, y.values)
    bins = binner.transform(X_df, metric='indices')
    
    # 计算分箱统计
    bin_labels = None
    if feature in binner.bin_tables_:
        bin_table = binner.bin_tables_[feature]
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()
    
    stats_df = compute_bin_stats(bins[feature].values, y.values, 
                                 target_type='binary', 
                                 bin_labels=bin_labels)
    
    return stats_df


def monotonicity_check(df: pd.DataFrame,
                      feature: str,
                      target: str,
                      n_bins: int = 10) -> Dict[str, Union[str, float]]:
    """单调性检验.
    
    检查特征分箱后的逾期率是否单调变化
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param n_bins: 分箱数
    :return: 单调性检验结果
    
    Example:
        >>> result = monotonicity_check(df, 'score', 'fpd15')
        >>> print(f"单调性: {result['单调性']}, 相关系数: {result['Spearman相关系数']}")
    """
    validate_dataframe(df, required_cols=[feature, target])
    validate_binary_target(df[target])
    
    # 获取分箱统计
    bin_stats = binning_bad_rate(df, feature, target, n_bins)
    
    if len(bin_stats) < 3:
        return {
            '特征名': feature,
            '单调性': '无法判断',
            '单调方向': '-',
            'Spearman相关系数': np.nan,
            '说明': '分箱数太少',
        }
    
    # 计算单调性
    bad_rates = bin_stats['坏样本率'].values
    x = np.arange(len(bad_rates))
    
    # Spearman相关系数
    from scipy import stats
    spearman_corr, pvalue = stats.spearmanr(x, bad_rates)
    
    # 判断单调性
    if abs(spearman_corr) < 0.5:
        monotonicity = '非单调'
        direction = '-'
    elif spearman_corr > 0:
        monotonicity = '单调'
        direction = '递增'
    else:
        monotonicity = '单调'
        direction = '递减'
    
    return {
        '特征名': feature,
        '单调性': monotonicity,
        '单调方向': direction,
        'Spearman相关系数': round(spearman_corr, 4),
        'P值': round(pvalue, 4),
        '说明': '单调性良好' if abs(spearman_corr) >= 0.8 else '需检查业务含义',
    }


def univariate_auc(df: pd.DataFrame,
                  feature: str,
                  target: str) -> Dict[str, Union[str, float]]:
    """单变量AUC分析.
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :return: AUC分析结果
    
    Example:
        >>> result = univariate_auc(df, 'score', 'fpd15')
        >>> print(f"AUC: {result['AUC值']}, 区分能力: {result['区分能力']}")
    """
    validate_dataframe(df, required_cols=[feature, target])
    validate_binary_target(df[target])
    
    # 复用metrics模块
    from ..metrics import auc
    
    # 数据清洗
    valid_mask = df[feature].notna() & df[target].notna()
    X = df.loc[valid_mask, feature]
    y = df.loc[valid_mask, target]
    
    # 计算AUC
    auc_value = auc(y.values, X.values)
    
    # 评级
    if auc_value < 0.5:
        level = '无区分能力(倒置)'
    elif auc_value < 0.6:
        level = '弱区分能力'
    elif auc_value < 0.7:
        level = '中等区分能力'
    elif auc_value < 0.8:
        level = '良好区分能力'
    elif auc_value < 0.9:
        level = '强区分能力'
    else:
        level = '极强区分能力'
    
    return {
        '特征名': feature,
        'AUC值': round(auc_value, 4),
        '区分能力': level,
    }


def feature_importance_ranking(df: pd.DataFrame,
                              features: List[str],
                              target: str,
                              metrics: List[str] = ['iv', 'auc']) -> pd.DataFrame:
    """综合特征重要性排序.
    
    综合IV和AUC等多个指标评估特征重要性
    
    :param df: 输入数据
    :param features: 特征列表
    :param target: 目标变量名
    :param metrics: 评估指标列表
    :return: 特征重要性DataFrame
    
    Example:
        >>> ranking = feature_importance_ranking(df, feature_list, 'fpd15')
        >>> print(ranking[['特征名', 'IV值', 'AUC值', '综合得分', '排名']])
    """
    validate_dataframe(df, required_cols=[target])
    
    results = []
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        result = {'特征名': feature}
        
        # IV值
        if 'iv' in metrics:
            try:
                iv_result = iv_analysis(df, feature, target)
                result['IV值'] = iv_result['IV值']
                result['预测能力'] = iv_result['预测能力']
            except:
                result['IV值'] = np.nan
        
        # AUC值
        if 'auc' in metrics:
            try:
                auc_result = univariate_auc(df, feature, target)
                result['AUC值'] = auc_result['AUC值']
            except:
                result['AUC值'] = np.nan
        
        results.append(result)
    
    result_df = pd.DataFrame(results)
    
    # 计算综合得分（标准化后的平均值）
    score_cols = []
    if 'IV值' in result_df.columns:
        result_df['IV得分'] = (result_df['IV值'] / result_df['IV值'].max()).fillna(0)
        score_cols.append('IV得分')
    
    if 'AUC值' in result_df.columns:
        # AUC需要处理倒置情况（AUC<0.5）
        auc_norm = result_df['AUC值'].apply(lambda x: max(x, 1-x))
        result_df['AUC得分'] = (auc_norm / auc_norm.max()).fillna(0)
        score_cols.append('AUC得分')
    
    if score_cols:
        result_df['综合得分'] = result_df[score_cols].mean(axis=1).round(4)
        result_df = result_df.sort_values('综合得分', ascending=False)
        result_df['排名'] = range(1, len(result_df) + 1)
    
    return result_df.reset_index(drop=True)
