"""Vintage分析模块.

提供账龄(Vintage)分析、滚动率分析等金融风控特有功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

from .utils import validate_dataframe, validate_binary_target


def vintage_analysis(df: pd.DataFrame,
                    vintage_col: str,
                    mob_col: str,
                    target_col: str,
                    max_mob: int = 12) -> pd.DataFrame:
    """Vintage账龄分析.
    
    追踪不同放款批次（Vintage）随账龄（MOB）的风险表现变化
    
    :param df: 输入数据
    :param vintage_col: Vintage批次列（如放款月份）
    :param mob_col: 账龄列（Month on Book）
    :param target_col: 目标变量列（如是否逾期）
    :param max_mob: 最大账龄
    :return: Vintage分析DataFrame
    
    Example:
        >>> vintage = vintage_analysis(df, 'issue_month', 'mob', 'ever_dpd30', max_mob=12)
        >>> print(vintage.pivot(index='MOB', columns='Vintage批次', values='累积坏账率(%)'))
    """
    validate_dataframe(df, required_cols=[vintage_col, mob_col, target_col])
    validate_binary_target(df[target_col])
    
    # 按Vintage和MOB汇总
    grouped = df.groupby([vintage_col, mob_col]).agg({
        target_col: ['count', 'sum']
    }).reset_index()
    
    grouped.columns = ['Vintage批次', 'MOB', '开户数', '坏账户数']
    
    # 筛选最大账龄
    grouped = grouped[grouped['MOB'] <= max_mob]
    
    # 计算累积坏账率
    results = []
    
    for vintage in grouped['Vintage批次'].unique():
        vintage_data = grouped[grouped['Vintage批次'] == vintage].sort_values('MOB')
        
        # 计算累积
        total_accounts = vintage_data['开户数'].sum()
        
        cum_bad = 0
        for _, row in vintage_data.iterrows():
            cum_bad += row['坏账户数']
            bad_rate = cum_bad / total_accounts * 100 if total_accounts > 0 else 0
            
            results.append({
                'Vintage批次': vintage,
                'MOB': int(row['MOB']),
                '当期开户数': int(row['开户数']),
                '当期坏账户数': int(row['坏账户数']),
                '累积坏账户数': int(cum_bad),
                '累积坏账率(%)': round(bad_rate, 2),
            })
    
    return pd.DataFrame(results)


def vintage_summary(df: pd.DataFrame,
                   vintage_col: str,
                   mob_col: str,
                   target_col: str,
                   max_mob: int = 12) -> pd.DataFrame:
    """Vintage汇总统计.
    
    :param df: 输入数据
    :param vintage_col: Vintage批次列
    :param mob_col: 账龄列
    :param target_col: 目标变量列
    :param max_mob: 最大账龄
    :return: Vintage汇总DataFrame
    
    Example:
        >>> summary = vintage_summary(df, 'issue_month', 'mob', 'ever_dpd30')
        >>> print(summary[['Vintage批次', '总开户数', f'MOB{max_mob}坏账率']])
    """
    vintage_df = vintage_analysis(df, vintage_col, mob_col, target_col, max_mob)
    
    if vintage_df.empty:
        return pd.DataFrame()
    
    # 汇总每个Vintage
    results = []
    
    for vintage in vintage_df['Vintage批次'].unique():
        vintage_data = vintage_df[vintage_df['Vintage批次'] == vintage]
        
        total_accounts = vintage_data['当期开户数'].sum()
        
        # 各MOB点的坏账率
        result = {
            'Vintage批次': vintage,
            '总开户数': int(total_accounts),
        }
        
        for mob in range(max_mob + 1):
            mob_data = vintage_data[vintage_data['MOB'] == mob]
            if len(mob_data) > 0:
                result[f'MOB{mob}坏账率(%)'] = mob_data['累积坏账率(%)'].values[0]
            else:
                result[f'MOB{mob}坏账率(%)'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


def roll_rate_analysis(df: pd.DataFrame,
                      overdue_cols: List[str],
                      labels: List[str] = None) -> pd.DataFrame:
    """滚动率分析.
    
    分析不同逾期状态之间的转化情况
    
    :param df: 输入数据
    :param overdue_cols: 各期逾期状态列（如['mob1_status', 'mob2_status', 'mob3_status']）
    :param labels: 状态标签（如['M0', 'M1', 'M2+']）
    :return: 滚动率分析DataFrame
    
    Example:
        >>> roll = roll_rate_analysis(df, ['mob1', 'mob2', 'mob3'], ['M0', 'M1', 'M2+'])
        >>> print(roll)
    """
    validate_dataframe(df, required_cols=overdue_cols)
    
    if labels is None:
        labels = [f'状态{i+1}' for i in range(len(overdue_cols))]
    
    # 计算各状态分布
    results = []
    
    for i, col in enumerate(overdue_cols):
        value_counts = df[col].value_counts()
        total = len(df)
        
        for status, count in value_counts.items():
            results.append({
                '期数': labels[i],
                '状态': status,
                '户数': int(count),
                '占比(%)': round(count / total * 100, 2),
            })
    
    return pd.DataFrame(results)
