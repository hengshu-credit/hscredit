"""稳定性分析模块.

提供PSI、CSI、时间稳定性等分析功能.
主要复用 hscredit.core.metrics 的PSI/CSI功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

from .utils import validate_dataframe, psi_rating


def psi_analysis(base_df: pd.DataFrame,
                current_df: pd.DataFrame,
                feature: str,
                n_bins: int = 10) -> Dict[str, Union[str, float, pd.DataFrame]]:
    """单变量PSI稳定性分析.
    
    复用 hscredit.core.metrics.psi_table
    
    :param base_df: 基准数据集
    :param current_df: 当前数据集
    :param feature: 特征名
    :param n_bins: 分箱数
    :return: PSI分析结果字典
    
    Example:
        >>> result = psi_analysis(train_df, test_df, 'age')
        >>> print(f"PSI: {result['PSI值']}, 稳定性: {result['稳定性']}")
    """
    from ..metrics import psi_table
    
    # 计算PSI表
    psi_df = psi_table(base_df[feature], current_df[feature], bins=n_bins)
    
    # 计算总PSI
    psi_value = psi_df['PSI贡献'].sum()
    
    return {
        '特征名': feature,
        'PSI值': round(psi_value, 4),
        '稳定性': psi_rating(psi_value),
        '分箱明细': psi_df,
    }


def batch_psi_analysis(df: pd.DataFrame,
                      features: List[str],
                      date_col: str,
                      base_period: str,
                      compare_periods: List[str],
                      n_bins: int = 10) -> pd.DataFrame:
    """批量PSI时间稳定性分析.
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param base_period: 基准期（如'2023-01'）
    :param compare_periods: 对比期列表
    :param n_bins: 分箱数
    :return: 批量PSI结果DataFrame
    
    Example:
        >>> psi_result = batch_psi_analysis(df, ['age', 'income'], 'apply_month', 
        ...                                 '2023-01', ['2023-02', '2023-03'])
        >>> print(psi_result[['特征名', '对比期', 'PSI值', '稳定性']])
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
    
    # 基准期数据
    base_df = df[df[date_col] == base_period]
    
    results = []
    
    for period in compare_periods:
        period_df = df[df[date_col] == period]
        
        if len(period_df) == 0:
            continue
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            try:
                psi_result = psi_analysis(base_df, period_df, feature, n_bins)
                results.append({
                    '特征名': psi_result['特征名'],
                    '基准期': base_period,
                    '对比期': period,
                    'PSI值': psi_result['PSI值'],
                    '稳定性': psi_result['稳定性'],
                })
            except Exception as e:
                results.append({
                    '特征名': feature,
                    '基准期': base_period,
                    '对比期': period,
                    'PSI值': np.nan,
                    '稳定性': '计算失败',
                })
    
    return pd.DataFrame(results)


def csi_analysis(df: pd.DataFrame,
                feature: str,
                target: str,
                date_col: str,
                base_period: str,
                compare_period: str) -> Dict[str, Union[str, float]]:
    """CSI特征稳定性分析.
    
    复用 hscredit.core.metrics.csi
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param date_col: 日期列名
    :param base_period: 基准期
    :param compare_period: 对比期
    :return: CSI分析结果
    
    Example:
        >>> result = csi_analysis(df, 'score', 'fpd15', 'apply_month', '2023-01', '2023-02')
        >>> print(f"CSI: {result['CSI值']}")
    """
    from ..metrics import csi
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
    
    base_df = df[df[date_col] == base_period]
    compare_df = df[df[date_col] == compare_period]
    
    # 计算CSI
    csi_value = csi(base_df[feature], compare_df[feature], 
                   base_df[target], compare_df[target])
    
    return {
        '特征名': feature,
        '基准期': base_period,
        '对比期': compare_period,
        'CSI值': round(csi_value, 4),
    }


def time_psi_tracking(df: pd.DataFrame,
                     features: List[str],
                     date_col: str,
                     freq: str = 'M',
                     n_bins: int = 10) -> pd.DataFrame:
    """PSI时序追踪分析.
    
    计算每个特征在不同时间段的PSI值变化趋势
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param freq: 时间频率
    :param n_bins: 分箱数
    :return: PSI时序追踪DataFrame
    
    Example:
        >>> tracking = time_psi_tracking(df, ['age', 'income'], 'apply_date')
        >>> print(tracking.pivot(index='时间周期', columns='特征名', values='PSI值'))
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建时间周期
    if freq == 'M':
        df['时间周期'] = df[date_col].dt.to_period('M').astype(str)
    elif freq == 'W':
        df['时间周期'] = df[date_col].dt.to_period('W').astype(str)
    elif freq == 'Q':
        df['时间周期'] = df[date_col].dt.to_period('Q').astype(str)
    
    # 获取所有周期
    periods = sorted(df['时间周期'].unique())
    
    if len(periods) < 2:
        return pd.DataFrame({'信息': ['时间周期不足，无法计算PSI']})
    
    # 以第一个周期为基准
    base_period = periods[0]
    base_df = df[df['时间周期'] == base_period]
    
    results = []
    
    for period in periods[1:]:
        period_df = df[df['时间周期'] == period]
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            try:
                psi_result = psi_analysis(base_df, period_df, feature, n_bins)
                results.append({
                    '特征名': psi_result['特征名'],
                    '基准期': base_period,
                    '时间周期': period,
                    'PSI值': psi_result['PSI值'],
                    '稳定性': psi_result['稳定性'],
                })
            except:
                pass
    
    return pd.DataFrame(results)


def stability_report(df: pd.DataFrame,
                    features: List[str],
                    date_col: str,
                    target: str = None,
                    psi_threshold: float = 0.1) -> pd.DataFrame:
    """综合稳定性报告.
    
    包含PSI分析和时间稳定性评估
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param target: 目标变量名（可选）
    :param psi_threshold: PSI阈值
    :return: 综合稳定性报告DataFrame
    
    Example:
        >>> report = stability_report(df, feature_list, 'apply_date')
        >>> print(report[['特征名', '平均PSI', '最大PSI', '不稳定期数']])
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # PSI时序追踪
    psi_tracking = time_psi_tracking(df, features, date_col)
    
    if psi_tracking.empty or '信息' in psi_tracking.columns:
        return pd.DataFrame({'信息': ['数据不足以生成稳定性报告']})
    
    # 汇总统计
    results = []
    
    for feature in features:
        feature_psi = psi_tracking[psi_tracking['特征名'] == feature]
        
        if len(feature_psi) == 0:
            continue
        
        avg_psi = feature_psi['PSI值'].mean()
        max_psi = feature_psi['PSI值'].max()
        unstable_count = (feature_psi['PSI值'] > psi_threshold).sum()
        
        results.append({
            '特征名': feature,
            '平均PSI': round(avg_psi, 4),
            '最大PSI': round(max_psi, 4),
            '统计期数': len(feature_psi),
            '不稳定期数': int(unstable_count),
            '不稳定期占比(%)': round(unstable_count / len(feature_psi) * 100, 2),
            '稳定性评级': psi_rating(avg_psi),
        })
    
    return pd.DataFrame(results).sort_values('平均PSI', ascending=False)


def psi_cross_analysis(
    df: pd.DataFrame,
    features: List[str],
    date_col: Optional[str] = None,
    group_col: Optional[str] = None,
    freq: str = 'M',
    n_bins: int = 10,
    return_matrix: bool = True
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """PSI交叉分析 - 计算两两组之间的PSI矩阵.
    
    支持两种方式分组：
    1. 自动分组：指定日期列(date_col)和频率(freq)，按时间自动分组
    2. 手工分组：指定分组列(group_col)，使用已有分组
    
    计算每两组之间的PSI值，返回PSI矩阵或长格式DataFrame.
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名（自动分组模式）
    :param group_col: 分组列名（手工分组模式）
    :param freq: 时间频率，'D'日/'W'周/'M'月/'Q'季度，默认'M'
    :param n_bins: PSI计算分箱数，默认10
    :param return_matrix: 是否返回矩阵格式，True返回方阵矩阵，False返回长格式
    :return: 
        - return_matrix=True: 返回 {特征名: PSI矩阵DataFrame} 字典
        - return_matrix=False: 返回长格式DataFrame [特征名, 组1, 组2, PSI值, 稳定性]
    
    Example:
        >>> # 自动按月份分组
        >>> result = psi_cross_analysis(df, ['age', 'income'], 
        ...                             date_col='apply_date', freq='M')
        >>> print(result['age'])  # 查看age特征的PSI矩阵
        
        >>> # 手工分组
        >>> result = psi_cross_analysis(df, ['score'], group_col='channel')
        
        >>> # 返回长格式
        >>> result = psi_cross_analysis(df, ['age'], date_col='apply_date', 
        ...                             freq='M', return_matrix=False)
    """
    from ..metrics import psi_table
    
    validate_dataframe(df)
    
    # 检查参数
    if date_col is None and group_col is None:
        raise ValueError("必须指定 date_col（自动分组）或 group_col（手工分组）之一")
    
    if date_col is not None and group_col is not None:
        raise ValueError("不能同时指定 date_col 和 group_col，请选择一种分组方式")
    
    df = df.copy()
    
    # 确定分组列
    if date_col is not None:
        # 自动分组模式
        if date_col not in df.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 根据频率创建分组
        if freq == 'D':
            df['_psi_group'] = df[date_col].dt.date.astype(str)
        elif freq == 'W':
            df['_psi_group'] = df[date_col].dt.to_period('W').astype(str)
        elif freq == 'M':
            df['_psi_group'] = df[date_col].dt.to_period('M').astype(str)
        elif freq == 'Q':
            df['_psi_group'] = df[date_col].dt.to_period('Q').astype(str)
        else:
            raise ValueError("freq 必须是 'D'/'W'/'M'/'Q' 之一")
        
        group_col = '_psi_group'
    else:
        # 手工分组模式
        if group_col not in df.columns:
            raise ValueError(f"分组列 '{group_col}' 不存在")
    
    # 获取所有分组
    groups = sorted(df[group_col].unique())
    
    if len(groups) < 2:
        return pd.DataFrame({'信息': ['分组数不足，至少需要2个组']})
    
    results = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        # 为每个特征计算PSI矩阵
        psi_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
        stability_matrix = pd.DataFrame(index=groups, columns=groups, dtype=object)
        
        # 计算两两之间的PSI
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i == j:
                    psi_matrix.loc[group1, group2] = 0.0
                    stability_matrix.loc[group1, group2] = '相同组'
                elif i < j:
                    # 只计算上三角，下三角对称
                    data1 = df[df[group_col] == group1][feature].dropna()
                    data2 = df[df[group_col] == group2][feature].dropna()
                    
                    if len(data1) == 0 or len(data2) == 0:
                        psi_value = np.nan
                        stability = '数据不足'
                    else:
                        try:
                            # 使用 psi_table 计算 PSI
                            psi_df = psi_table(data1, data2, max_n_bins=n_bins)
                            psi_value = psi_df['PSI贡献'].sum()
                            stability = psi_rating(psi_value)
                        except Exception as e:
                            psi_value = np.nan
                            stability = '计算失败'
                    
                    psi_matrix.loc[group1, group2] = psi_value
                    psi_matrix.loc[group2, group1] = psi_value
                    stability_matrix.loc[group1, group2] = stability
                    stability_matrix.loc[group2, group1] = stability
        
        if return_matrix:
            results[feature] = psi_matrix
        else:
            # 转换为长格式
            long_results = []
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if i != j:  # 排除对角线
                        long_results.append({
                            '特征名': feature,
                            '基准组': group1,
                            '对比组': group2,
                            'PSI值': psi_matrix.loc[group1, group2],
                            '稳定性': stability_matrix.loc[group1, group2]
                        })
            results[feature] = pd.DataFrame(long_results)
    
    # 如果只有一个特征，直接返回结果；否则返回字典
    if len(results) == 1:
        return list(results.values())[0]
    return results
