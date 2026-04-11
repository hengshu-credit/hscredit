"""目标变量分析模块.

提供目标变量分布、逾期率分析、时间趋势等功能.
专注于金融风控场景.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple

from .utils import validate_dataframe, validate_binary_target, safe_divide


def _build_overdue_labels(overdue: Union[str, List[str]],
                          dpds: Union[int, List[int]]) -> List[Tuple[str, int, str]]:
    """构建逾期标签列表.
    
    :param overdue: 逾期天数字段名或列表
    :param dpds: 逾期天数或列表
    :return: 标签列表 [(标签名, dpd天数, 逾期字段), ...]
    """
    if isinstance(overdue, str):
        overdue = [overdue]
    if isinstance(dpds, int):
        dpds = [dpds]
    
    labels = []
    for od_field in overdue:
        for dpd in dpds:
            if dpd == 0:
                label_name = f"{od_field}>0"
            else:
                label_name = f"{od_field}>{dpd}"
            labels.append((label_name, dpd, od_field))
    
    return labels


def _create_binary_target(df: pd.DataFrame,
                          overdue_col: str,
                          dpd: int,
                          del_grey: bool = False) -> pd.Series:
    """根据逾期天数创建二元目标变量.
    
    :param df: 输入数据
    :param overdue_col: 逾期天数字段名
    :param dpd: 逾期定义天数
    :param del_grey: 是否删除灰样本
    :return: 二元目标变量 (0/1)，灰样本为NaN
    """
    overdue = df[overdue_col].copy()
    
    if del_grey:
        # 删除灰样本：逾期天数在 (0, dpd] 区间设为NaN
        target = pd.Series(np.nan, index=df.index)
        target[overdue > dpd] = 1  # 坏样本
        target[overdue <= 0] = 0   # 好样本
    else:
        # 保留灰样本作为好样本
        target = (overdue > dpd).astype(int)
    
    return target


def target_distribution(df: pd.DataFrame,
                       target_col: str) -> pd.DataFrame:
    """目标变量分布统计.
    
    :param df: 输入数据
    :param target_col: 目标变量列名
    :return: 目标分布DataFrame，列包括[类别, 样本数, 占比, 累计占比]
    
    Example:
        >>> dist = target_distribution(df, 'fpd15')
        >>> print(dist)
           类别   样本数   占比(%)  累计占比(%)
        0   0    8500   85.00      85.00
        1   1    1500   15.00     100.00
    """
    validate_dataframe(df, required_cols=[target_col])
    
    value_counts = df[target_col].value_counts().sort_index()
    total = len(df)
    
    results = []
    cumsum = 0
    
    for value, count in value_counts.items():
        ratio = count / total * 100
        cumsum += ratio
        results.append({
            '类别': value,
            '样本数': int(count),
            '占比(%)': round(ratio, 2),
            '累计占比(%)': round(cumsum, 2),
        })
    
    return pd.DataFrame(results)


def bad_rate_overall(df: pd.DataFrame,
                    target_col: Optional[str] = None,
                    overdue: Optional[Union[str, List[str]]] = None,
                    dpds: Optional[Union[int, List[int]]] = None,
                    del_grey: bool = False,
                    *,
                    target: Optional[str] = None) -> Union[Dict, pd.DataFrame]:
    """计算整体逾期率.
    
    支持单标签分析（通过target_col）或多标签分析（通过overdue+dpds）。
    
    :param df: 输入数据
    :param target_col: 目标变量列名（单标签模式）
    :param overdue: 逾期天数字段名或列表，如 'MOB1' 或 ['MOB1', 'MOB3']
    :param dpds: 逾期定义天数或列表，如 7 或 [0, 7, 30]
        - 逾期天数 > dpds 为坏样本(1)，其他为好样本(0)
    :param del_grey: 是否删除逾期天数在 (0, dpd] 区间的灰样本
    :return: 单标签返回字典，多标签返回DataFrame
    
    Example:
        >>> # 单标签分析
        >>> result = bad_rate_overall(df, target_col='fpd15')
        >>> print(result)
        {'样本总数': 10000, '好样本数': 8500, '坏样本数': 1500, '逾期率(%)': 15.0}
        
        >>> # 多标签分析
        >>> result = bad_rate_overall(df, overdue=['MOB1', 'MOB3'], dpds=[7, 30])
        >>> print(result)
             标签      样本总数  好样本数  坏样本数  逾期率(%)
        0  MOB1>7     9800    8820     980    10.00
        1  MOB1>30    9800    9400     400     4.08
    """
    target_col = target or target_col
    validate_dataframe(df)
    
    # 单标签模式
    if target_col is not None:
        validate_binary_target(df[target_col])
        total = len(df)
        bad_count = df[target_col].sum()
        good_count = total - bad_count
        bad_rate = bad_count / total * 100
        
        return {
            '样本总数': int(total),
            '好样本数': int(good_count),
            '坏样本数': int(bad_count),
            '逾期率(%)': round(bad_rate, 2),
        }
    
    # 多标签模式
    if overdue is None or dpds is None:
        raise ValueError("必须指定 target_col 或 (overdue + dpds)")
    
    labels = _build_overdue_labels(overdue, dpds)
    results = []
    
    for label_name, dpd, od_field in labels:
        target = _create_binary_target(df, od_field, dpd, del_grey)
        valid_mask = target.notna()
        
        total = valid_mask.sum()
        if total == 0:
            results.append({
                '标签': label_name,
                '样本总数': 0,
                '好样本数': 0,
                '坏样本数': 0,
                '逾期率(%)': np.nan,
            })
            continue
        
        bad_count = target[valid_mask].sum()
        good_count = total - bad_count
        bad_rate = bad_count / total * 100
        
        results.append({
            '标签': label_name,
            '样本总数': int(total),
            '好样本数': int(good_count),
            '坏样本数': int(bad_count),
            '逾期率(%)': round(bad_rate, 2),
        })
    
    return pd.DataFrame(results)


def bad_rate_by_dimension(df: pd.DataFrame,
                         dim_col: str = None,
                         target_col: Optional[str] = None,
                         overdue: Optional[Union[str, List[str]]] = None,
                         dpds: Optional[Union[int, List[int]]] = None,
                         del_grey: bool = False,
                         sort_by: str = 'bad_rate',
                         *,
                         target: Optional[str] = None,
                         segment_col: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """分维度逾期率分析.
    
    支持单标签或多标签分析。
    
    :param df: 输入数据
    :param dim_col: 维度列名（如渠道、产品类型）
    :param target_col: 目标变量列名（单标签模式）
    :param overdue: 逾期天数字段名或列表
    :param dpds: 逾期定义天数或列表
    :param del_grey: 是否删除灰样本
    :param sort_by: 排序方式，'bad_rate'或'count'
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典
    
    Example:
        >>> # 单标签
        >>> result = bad_rate_by_dimension(df, 'channel', target_col='fpd15')
        
        >>> # 多标签
        >>> result = bad_rate_by_dimension(df, 'channel', overdue='MOB1', dpds=[7, 30])
        >>> print(result['MOB1>7'])
    """
    target_col = target or target_col
    dim_col = segment_col or dim_col
    if dim_col is None:
        raise ValueError("必须指定 dim_col 或 segment_col 参数")
    validate_dataframe(df, required_cols=[dim_col])
    
    # 单标签模式
    if target_col is not None:
        validate_binary_target(df[target_col])
        total = len(df)
        
        grouped = df.groupby(dim_col).agg({
            target_col: ['count', 'sum', 'mean']
        }).reset_index()
        
        grouped.columns = ['维度值', '样本数', '坏样本数', '逾期率']
        grouped['好样本数'] = grouped['样本数'] - grouped['坏样本数']
        grouped['逾期率(%)'] = (grouped['逾期率'] * 100).round(2)
        grouped['样本占比(%)'] = (grouped['样本数'] / total * 100).round(2)
        
        grouped = grouped.drop('逾期率', axis=1)
        
        if sort_by == 'bad_rate':
            grouped = grouped.sort_values('逾期率(%)', ascending=False)
        else:
            grouped = grouped.sort_values('样本数', ascending=False)
        
        return grouped.reset_index(drop=True)
    
    # 多标签模式
    if overdue is None or dpds is None:
        raise ValueError("必须指定 target_col 或 (overdue + dpds)")
    
    labels = _build_overdue_labels(overdue, dpds)
    results = {}
    
    for label_name, dpd, od_field in labels:
        target = _create_binary_target(df, od_field, dpd, del_grey)
        valid_mask = target.notna()
        
        df_valid = df[valid_mask].copy()
        df_valid['_target'] = target[valid_mask]
        
        total = len(df_valid)
        if total == 0:
            results[label_name] = pd.DataFrame()
            continue
        
        grouped = df_valid.groupby(dim_col).agg({
            '_target': ['count', 'sum', 'mean']
        }).reset_index()
        
        grouped.columns = ['维度值', '样本数', '坏样本数', '逾期率']
        grouped['好样本数'] = grouped['样本数'] - grouped['坏样本数']
        grouped['逾期率(%)'] = (grouped['逾期率'] * 100).round(2)
        grouped['样本占比(%)'] = (grouped['样本数'] / total * 100).round(2)
        grouped = grouped.drop('逾期率', axis=1)
        
        if sort_by == 'bad_rate':
            grouped = grouped.sort_values('逾期率(%)', ascending=False)
        else:
            grouped = grouped.sort_values('样本数', ascending=False)
        
        results[label_name] = grouped.reset_index(drop=True)
    
    return results if len(results) > 1 else list(results.values())[0]


def bad_rate_trend(df: pd.DataFrame,
                  date_col: str,
                  target_col: Optional[str] = None,
                  overdue: Optional[Union[str, List[str]]] = None,
                  dpds: Optional[Union[int, List[int]]] = None,
                  del_grey: bool = False,
                  freq: str = 'M',
                  dimensions: Optional[List[str]] = None,
                  *,
                  target: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """逾期率时间趋势分析.
    
    支持单标签或多标签分析。
    
    :param df: 输入数据
    :param date_col: 日期列名
    :param target_col: 目标变量列名（单标签模式）
    :param overdue: 逾期天数字段名或列表
    :param dpds: 逾期定义天数或列表
    :param del_grey: 是否删除灰样本
    :param freq: 时间频率，'D'日/'W'周/'M'月/'Q'季度
    :param dimensions: 分维度分析列表
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典
    
    Example:
        >>> # 单标签
        >>> trend = bad_rate_trend(df, 'apply_date', target_col='fpd15', freq='M')
        
        >>> # 多标签
        >>> trend = bad_rate_trend(df, 'apply_date', overdue=['MOB1', 'MOB3'], dpds=30, freq='M')
        >>> print(trend['MOB1>30'])
    """
    target_col = target or target_col
    validate_dataframe(df, required_cols=[date_col])
    
    # 确保日期格式正确
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建时间周期
    period_col = '时间周期'
    if freq == 'D':
        df[period_col] = df[date_col].dt.date
    elif freq == 'W':
        df[period_col] = df[date_col].dt.to_period('W').astype(str)
    elif freq == 'M':
        df[period_col] = df[date_col].dt.to_period('M').astype(str)
    elif freq == 'Q':
        df[period_col] = df[date_col].dt.to_period('Q').astype(str)
    else:
        raise ValueError("freq必须是'D'/'W'/'M'/'Q'之一")
    
    # 单标签模式
    if target_col is not None:
        validate_binary_target(df[target_col])
        
        grouped = df.groupby(period_col).agg({
            target_col: ['count', 'sum', 'mean']
        }).reset_index()
        grouped.columns = [period_col, '样本数', '坏样本数', '逾期率']
        grouped['好样本数'] = grouped['样本数'] - grouped['坏样本数']
        grouped['逾期率(%)'] = (grouped['逾期率'] * 100).round(2)
        
        # 计算环比变化
        grouped['环比变化(%)'] = grouped['逾期率(%)'].diff().round(2)
        
        # 计算同比变化（如果数据跨度足够）
        if len(grouped) > 12:
            grouped['同比变化(%)'] = (grouped['逾期率(%)'] - 
                                     grouped['逾期率(%)'].shift(12)).round(2)
        
        return grouped[[period_col, '样本数', '好样本数', '坏样本数', 
                       '逾期率(%)', '环比变化(%)']].reset_index(drop=True)
    
    # 多标签模式
    if overdue is None or dpds is None:
        raise ValueError("必须指定 target_col 或 (overdue + dpds)")
    
    labels = _build_overdue_labels(overdue, dpds)
    results = {}
    
    for label_name, dpd, od_field in labels:
        target = _create_binary_target(df, od_field, dpd, del_grey)
        valid_mask = target.notna()
        
        df_valid = df[valid_mask].copy()
        df_valid['_target'] = target[valid_mask]
        
        grouped = df_valid.groupby(period_col).agg({
            '_target': ['count', 'sum', 'mean']
        }).reset_index()
        
        if grouped.empty:
            results[label_name] = pd.DataFrame()
            continue
        
        grouped.columns = [period_col, '样本数', '坏样本数', '逾期率']
        grouped['好样本数'] = grouped['样本数'] - grouped['坏样本数']
        grouped['逾期率(%)'] = (grouped['逾期率'] * 100).round(2)
        grouped['环比变化(%)'] = grouped['逾期率(%)'].diff().round(2)
        
        if len(grouped) > 12:
            grouped['同比变化(%)'] = (grouped['逾期率(%)'] - 
                                     grouped['逾期率(%)'].shift(12)).round(2)
        
        results[label_name] = grouped[[period_col, '样本数', '好样本数', '坏样本数', 
                                      '逾期率(%)', '环比变化(%)']].reset_index(drop=True)
    
    return results if len(results) > 1 else list(results.values())[0]


def bad_rate_by_bins(df: pd.DataFrame,
                    score_col: str,
                    target_col: Optional[str] = None,
                    overdue: Optional[Union[str, List[str]]] = None,
                    dpds: Optional[Union[int, List[int]]] = None,
                    del_grey: bool = False,
                    n_bins: int = 10,
                    method: str = 'quantile') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """评分分箱逾期率分析.
    
    支持单标签或多标签分析。
    
    :param df: 输入数据
    :param score_col: 评分列名
    :param target_col: 目标变量列名（单标签模式）
    :param overdue: 逾期天数字段名或列表
    :param dpds: 逾期定义天数或列表
    :param del_grey: 是否删除灰样本
    :param n_bins: 分箱数
    :param method: 分箱方法，'quantile'等频/'uniform'等距
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典
    
    Example:
        >>> # 单标签
        >>> bins = bad_rate_by_bins(df, 'score', target_col='fpd15', n_bins=10)
        
        >>> # 多标签
        >>> bins = bad_rate_by_bins(df, 'score', overdue=['MOB1', 'MOB3'], dpds=30, n_bins=10)
        >>> print(bins['MOB1>30'])
    """
    validate_dataframe(df, required_cols=[score_col])
    
    series = df[score_col].dropna()
    
    if len(series) == 0:
        return pd.DataFrame()
    
    # 分箱
    if method == 'quantile':
        bins = pd.qcut(df[score_col], q=n_bins, duplicates='drop')
    else:
        bins = pd.cut(df[score_col], bins=n_bins)
    
    # 单标签模式
    if target_col is not None:
        validate_binary_target(df[target_col])
        
        total_bad_rate = df[target_col].mean()
        
        result = df.groupby(bins).agg({
            target_col: ['count', 'sum', 'mean']
        }).reset_index()
        
        result.columns = ['分箱区间', '样本数', '坏样本数', '逾期率']
        result['好样本数'] = result['样本数'] - result['坏样本数']
        result['逾期率(%)'] = (result['逾期率'] * 100).round(2)
        result['样本占比(%)'] = (result['样本数'] / len(df) * 100).round(2)
        result['提升度'] = (result['逾期率'] / total_bad_rate).round(4)
        result['分箱'] = range(1, len(result) + 1)
        
        return result[['分箱', '分箱区间', '样本数', '好样本数', '坏样本数',
                      '逾期率(%)', '样本占比(%)', '提升度']].reset_index(drop=True)
    
    # 多标签模式
    if overdue is None or dpds is None:
        raise ValueError("必须指定 target_col 或 (overdue + dpds)")
    
    labels = _build_overdue_labels(overdue, dpds)
    results = {}
    
    for label_name, dpd, od_field in labels:
        target = _create_binary_target(df, od_field, dpd, del_grey)
        valid_mask = target.notna()
        
        df_valid = df[valid_mask].copy()
        df_valid['_target'] = target[valid_mask]
        
        bins_valid = bins[valid_mask]
        total_bad_rate = df_valid['_target'].mean()
        
        result = df_valid.groupby(bins_valid).agg({
            '_target': ['count', 'sum', 'mean']
        }).reset_index()
        
        if result.empty:
            results[label_name] = pd.DataFrame()
            continue
        
        result.columns = ['分箱区间', '样本数', '坏样本数', '逾期率']
        result['好样本数'] = result['样本数'] - result['坏样本数']
        result['逾期率(%)'] = (result['逾期率'] * 100).round(2)
        result['样本占比(%)'] = (result['样本数'] / len(df_valid) * 100).round(2)
        result['提升度'] = (result['逾期率'] / total_bad_rate).round(4)
        result['分箱'] = range(1, len(result) + 1)
        
        results[label_name] = result[['分箱', '分箱区间', '样本数', '好样本数', '坏样本数',
                                     '逾期率(%)', '样本占比(%)', '提升度']].reset_index(drop=True)
    
    return results if len(results) > 1 else list(results.values())[0]


def sample_distribution(df: pd.DataFrame,
                       date_col: str,
                       freq: str = 'M',
                       target_col: Optional[str] = None,
                       *,
                       target: Optional[str] = None) -> pd.DataFrame:
    """样本时间分布分析.
    
    :param df: 输入数据
    :param date_col: 日期列名
    :param freq: 时间频率
    :param target_col: 目标变量列名（如有）
    :return: 样本分布DataFrame
    
    Example:
        >>> dist = sample_distribution(df, 'apply_date', target_col='fpd15')
        >>> print(dist[['时间周期', '样本数', '坏样本数', '逾期率(%)']])
    """
    target_col = target or target_col
    validate_dataframe(df, required_cols=[date_col])
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建时间周期
    period_col = '时间周期'
    if freq == 'D':
        df[period_col] = df[date_col].dt.date
    elif freq == 'W':
        df[period_col] = df[date_col].dt.to_period('W').astype(str)
    elif freq == 'M':
        df[period_col] = df[date_col].dt.to_period('M').astype(str)
    elif freq == 'Q':
        df[period_col] = df[date_col].dt.to_period('Q').astype(str)
    
    # 计算分布
    if target_col:
        validate_binary_target(df[target_col])
        grouped = df.groupby(period_col).agg({
            target_col: ['count', 'sum', 'mean']
        }).reset_index()
        grouped.columns = [period_col, '样本数', '坏样本数', '逾期率']
        grouped['好样本数'] = grouped['样本数'] - grouped['坏样本数']
        grouped['逾期率(%)'] = (grouped['逾期率'] * 100).round(2)
        grouped = grouped.drop('逾期率', axis=1)
    else:
        grouped = df.groupby(period_col).size().reset_index(name='样本数')
    
    # 计算环比
    grouped['样本环比(%)'] = (grouped['样本数'].pct_change() * 100).round(2)
    
    return grouped.reset_index(drop=True)
