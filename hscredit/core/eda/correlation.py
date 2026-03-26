"""相关性分析模块.

提供相关性矩阵、高相关筛选、VIF分析等功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Literal

from .utils import validate_dataframe, vif_rating


def correlation_matrix(df: pd.DataFrame,
                       features: List[str] = None,
                       method: Literal['pearson', 'spearman', 'kendall'] = 'pearson') -> pd.DataFrame:
    """计算相关性矩阵.
    
    :param df: 输入数据
    :param features: 指定分析的特征，None则分析全部数值型
    :param method: 相关计算方法
    :return: 相关性矩阵DataFrame
    
    Example:
        >>> corr = correlation_matrix(df, ['age', 'income', 'score'])
        >>> print(corr)
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # 过滤非数值列
        features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if len(features) < 2:
        return pd.DataFrame()
    
    # 计算相关性
    corr_matrix = df[features].corr(method=method)
    
    return corr_matrix


def high_correlation_pairs(df: pd.DataFrame,
                          features: List[str] = None,
                          threshold: float = 0.8,
                          method: Literal['pearson', 'spearman', 'kendall'] = 'pearson') -> pd.DataFrame:
    """高相关性特征对检测.
    
    :param df: 输入数据
    :param features: 指定分析的特征
    :param threshold: 高相关阈值
    :param method: 相关计算方法
    :return: 高相关特征对DataFrame
    
    Example:
        >>> pairs = high_correlation_pairs(df, threshold=0.8)
        >>> print(pairs[['特征1', '特征2', '相关系数', '相关评级']])
    """
    corr_matrix = correlation_matrix(df, features, method)
    
    if corr_matrix.empty:
        return pd.DataFrame()
    
    # 提取上三角矩阵的高相关对
    results = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                results.append({
                    '特征1': corr_matrix.columns[i],
                    '特征2': corr_matrix.columns[j],
                    '相关系数': round(corr_value, 4),
                    '绝对相关系数': round(abs(corr_value), 4),
                    '相关评级': vif_rating(abs(corr_value)),
                })
    
    if not results:
        return pd.DataFrame({'信息': [f'未发现相关系数>={threshold}的特征对']})
    
    return pd.DataFrame(results).sort_values('绝对相关系数', ascending=False)


def correlation_filter(df: pd.DataFrame,
                      features: List[str],
                      target: str,
                      threshold: float = 0.8,
                      method: Literal['pearson', 'spearman', 'kendall'] = 'pearson') -> List[str]:
    """相关性筛选，剔除高相关特征.
    
    策略：保留与目标变量相关性高的特征
    
    :param df: 输入数据
    :param features: 特征列表
    :param target: 目标变量
    :param threshold: 高相关阈值
    :param method: 相关计算方法
    :return: 筛选后的特征列表
    
    Example:
        >>> selected = correlation_filter(df, feature_list, 'fpd15', threshold=0.8)
        >>> print(f"从{len(feature_list)}个特征中筛选出{len(selected)}个")
    """
    validate_dataframe(df, required_cols=[target])
    
    # 计算与目标的相关性
    target_corr = {}
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            corr = df[feature].corr(df[target], method=method)
            target_corr[feature] = abs(corr) if not pd.isna(corr) else 0
    
    # 按与目标相关性排序
    sorted_features = sorted(target_corr.keys(), key=lambda x: target_corr[x], reverse=True)
    
    # 逐步筛选
    selected = []
    removed = []
    
    for feature in sorted_features:
        if feature in removed:
            continue
        
        selected.append(feature)
        
        # 找出与该特征高相关的其他特征
        for other in sorted_features:
            if other != feature and other not in removed:
                corr = df[feature].corr(df[other], method=method)
                if abs(corr) >= threshold:
                    removed.append(other)
    
    return selected


def vif_analysis(df: pd.DataFrame,
                features: List[str] = None,
                threshold: float = 10.0) -> pd.DataFrame:
    """VIF多重共线性分析.
    
    :param df: 输入数据
    :param features: 指定分析的特征，None则分析全部数值型
    :param threshold: VIF阈值
    :return: VIF分析DataFrame
    
    Example:
        >>> vif_df = vif_analysis(df, threshold=10)
        >>> print(vif_df[['特征名', 'VIF值', '共线性评级']])
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if len(features) < 2:
        return pd.DataFrame({'信息': ['数值型特征少于2个，无法计算VIF']})
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # 准备数据（填充缺失值）
    X = df[features].fillna(df[features].median())
    
    # 计算VIF
    results = []
    
    for i, feature in enumerate(features):
        try:
            vif_value = variance_inflation_factor(X.values, i)
            
            results.append({
                '特征名': feature,
                'VIF值': round(vif_value, 2),
                '共线性评级': vif_rating(vif_value),
                '建议': '剔除' if vif_value > threshold else '保留',
            })
        except Exception as e:
            results.append({
                '特征名': feature,
                'VIF值': np.nan,
                '共线性评级': '计算失败',
                '建议': '检查数据',
            })
    
    result_df = pd.DataFrame(results).sort_values('VIF值', ascending=False)
    
    return result_df
