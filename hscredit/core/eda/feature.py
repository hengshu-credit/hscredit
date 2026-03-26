"""特征分析模块.

提供特征分布、异常值检测、集中度分析等功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Literal, Any

from .utils import (
    infer_feature_types,
    validate_dataframe,
    calculate_gini,
    remove_outliers_iqr
)


def feature_type_inference(df: pd.DataFrame,
                          categorical_threshold: int = 20,
                          unique_ratio_threshold: float = 0.05,
                          numeric_as_categorical: Optional[List[str]] = None,
                          force_numeric: Optional[List[str]] = None) -> pd.DataFrame:
    """自动推断特征类型.

    默认严格按照实际数据类型判断：
    - 数值类型（int/float）-> 'numerical'
    - 非数值类型（object/string/category）-> 'categorical'

    仅当用户指定参数时才进行特殊处理：
    - numeric_as_categorical: 将指定的数值列视为 categorical
    - force_numeric: 将指定的列视为 numerical

    :param df: 输入数据
    :param categorical_threshold: 保留参数，不再用于默认类型判断
    :param unique_ratio_threshold: 保留参数，不再用于默认类型判断
    :param numeric_as_categorical: 强制视为分类变量的数值列名列表
    :param force_numeric: 强制视为数值变量的列名列表
    :return: 特征类型DataFrame

    Example:
        >>> types = feature_type_inference(df)
        >>> print(types[['特征名', '特征类型', '唯一值数', '建议处理方式']])
        >>> # 将特定数值列视为分类
        >>> types = feature_type_inference(df, numeric_as_categorical=['education_level'])
    """
    validate_dataframe(df)

    feature_types = infer_feature_types(df, categorical_threshold, unique_ratio_threshold,
                                       numeric_as_categorical, force_numeric)
    
    results = []
    for col, ftype in feature_types.items():
        n_unique = df[col].nunique(dropna=True)
        
        # 建议处理方式
        if ftype == 'constant':
            suggestion = '考虑删除'
        elif ftype == 'id':
            suggestion = '不参与建模'
        elif ftype == 'datetime':
            suggestion = '提取时间特征'
        elif ftype == 'categorical':
            suggestion = '编码处理'
        elif ftype == 'text':
            suggestion = '文本特征工程'
        else:
            suggestion = '标准化/归一化'
        
        results.append({
            '特征名': col,
            '特征类型': ftype,
            '数据类型': str(df[col].dtype),
            '唯一值数': n_unique,
            '建议处理方式': suggestion,
        })
    
    return pd.DataFrame(results)


def numeric_distribution(df: pd.DataFrame,
                        feature: str,
                        n_bins: int = 20) -> pd.DataFrame:
    """数值特征分布统计.
    
    :param df: 输入数据
    :param feature: 特征名
    :param n_bins: 分箱数
    :return: 分布统计DataFrame
    
    Example:
        >>> dist = numeric_distribution(df, 'age', n_bins=10)
        >>> print(dist[['分箱区间', '频数', '频率(%)', '累计频率(%)']])
    """
    validate_dataframe(df, required_cols=[feature])
    
    series = df[feature].dropna()
    
    if len(series) == 0:
        return pd.DataFrame()
    
    # 分箱统计
    counts, bin_edges = np.histogram(series, bins=n_bins)
    total = counts.sum()
    
    results = []
    cumsum = 0
    
    for i in range(len(counts)):
        freq = counts[i]
        ratio = freq / total * 100 if total > 0 else 0
        cumsum += ratio
        
        results.append({
            '分箱编号': i + 1,
            '分箱区间': f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})',
            '频数': int(freq),
            '频率(%)': round(ratio, 2),
            '累计频率(%)': round(cumsum, 2),
        })
    
    return pd.DataFrame(results)


def categorical_distribution(df: pd.DataFrame,
                            feature: str,
                            top_n: int = None) -> pd.DataFrame:
    """类别特征分布统计.
    
    :param df: 输入数据
    :param feature: 特征名
    :param top_n: 仅显示前N个类别
    :return: 分布统计DataFrame
    
    Example:
        >>> dist = categorical_distribution(df, 'education', top_n=5)
        >>> print(dist[['类别值', '频数', '频率(%)']])
    """
    validate_dataframe(df, required_cols=[feature])
    
    series = df[feature]
    total = len(series)
    
    value_counts = series.value_counts()
    
    if top_n:
        value_counts = value_counts.head(top_n)
    
    results = []
    cumsum = 0
    
    for value, count in value_counts.items():
        ratio = count / total * 100
        cumsum += ratio
        
        results.append({
            '类别值': value,
            '频数': int(count),
            '频率(%)': round(ratio, 2),
            '累计频率(%)': round(cumsum, 2),
        })
    
    result_df = pd.DataFrame(results)
    
    # 添加其他类别统计
    if top_n and len(value_counts) < series.nunique():
        other_count = total - value_counts.sum()
        other_ratio = other_count / total * 100
        
        other_row = pd.DataFrame({
            '类别值': ['其他'],
            '频数': [other_count],
            '频率(%)': [round(other_ratio, 2)],
            '累计频率(%)': [100.0],
        })
        result_df = pd.concat([result_df, other_row], ignore_index=True)
    
    return result_df


def outlier_detection(df: pd.DataFrame,
                     features: List[str] = None,
                     method: Literal['iqr', 'zscore', 'mad'] = 'iqr',
                     threshold: float = 1.5) -> pd.DataFrame:
    """异常值检测.
    
    :param df: 输入数据
    :param features: 指定检测的特征，None则检测全部数值型
    :param method: 检测方法，'iqr'/'zscore'/'mad'
    :param threshold: 阈值
    :return: 异常值统计DataFrame
    
    Example:
        >>> outliers = outlier_detection(df, method='iqr')
        >>> print(outliers[['特征名', '异常值数', '异常值率(%)', '正常范围']])
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for col in features:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (series < lower) | (series > upper)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold
            lower = series.mean() - threshold * series.std()
            upper = series.mean() + threshold * series.std()
            
        elif method == 'mad':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / mad if mad != 0 else 0
            outliers = np.abs(modified_z) > threshold
            lower = median - threshold * mad / 0.6745
            upper = median + threshold * mad / 0.6745
        
        outlier_count = outliers.sum()
        outlier_rate = outlier_count / len(series) * 100
        
        results.append({
            '特征名': col,
            '异常值数': int(outlier_count),
            '异常值率(%)': round(outlier_rate, 2),
            '正常范围': f'[{lower:.2f}, {upper:.2f}]',
            '最小值': round(series.min(), 4),
            '最大值': round(series.max(), 4),
        })
    
    return pd.DataFrame(results).sort_values('异常值率(%)', ascending=False)


def rare_category_detection(df: pd.DataFrame,
                           features: List[str] = None,
                           threshold: float = 0.01) -> pd.DataFrame:
    """稀有类别检测.
    
    :param df: 输入数据
    :param features: 指定检测的特征，None则检测全部分类别
    :param threshold: 稀有阈值（频率低于此值视为稀有）
    :return: 稀有类别统计DataFrame
    
    Example:
        >>> rare = rare_category_detection(df, threshold=0.01)
        >>> print(rare[['特征名', '稀有类别', '频数', '频率(%)', '建议']])
    """
    validate_dataframe(df)
    
    if features is None:
        feature_types = infer_feature_types(df)
        features = [f for f, t in feature_types.items() if t == 'categorical']
    
    results = []
    total = len(df)
    
    for col in features:
        if col not in df.columns:
            continue
        
        value_counts = df[col].value_counts()
        
        for value, count in value_counts.items():
            rate = count / total
            
            if rate < threshold:
                results.append({
                    '特征名': col,
                    '稀有类别': value,
                    '频数': int(count),
                    '频率(%)': round(rate * 100, 3),
                    '建议': '合并或删除',
                })
    
    if not results:
        return pd.DataFrame({'信息': ['未发现稀有类别']})
    
    return pd.DataFrame(results).sort_values('频率(%)')


def concentration_analysis(df: pd.DataFrame,
                          features: List[str] = None) -> pd.DataFrame:
    """集中度分析(Gini系数).
    
    :param df: 输入数据
    :param features: 指定分析的特征，None则分析全部数值型
    :return: 集中度分析DataFrame
    
    Example:
        >>> concentration = concentration_analysis(df)
        >>> print(concentration[['特征名', 'Gini系数', '集中度评级']])
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for col in features:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        gini = calculate_gini(series.values)
        
        # 评级
        if gini < 0.2:
            level = '低集中'
        elif gini < 0.4:
            level = '中等集中'
        elif gini < 0.6:
            level = '高集中'
        else:
            level = '极高集中'
        
        results.append({
            '特征名': col,
            'Gini系数': round(gini, 4),
            '集中度评级': level,
        })
    
    if not results:
        return pd.DataFrame(columns=['特征名', 'Gini系数', '集中度评级'])
    
    return pd.DataFrame(results).sort_values('Gini系数', ascending=False)


def feature_stability_over_time(df: pd.DataFrame,
                               features: List[str],
                               date_col: str,
                               freq: str = 'M') -> pd.DataFrame:
    """特征时序稳定性分析.
    
    :param df: 输入数据
    :param features: 指定分析的特征列表
    :param date_col: 日期列名
    :param freq: 时间频率
    :return: 时序稳定性DataFrame
    
    Example:
        >>> stability = feature_stability_over_time(df, ['age', 'income'], 'apply_date')
        >>> print(stability[['特征名', '均值标准差', '变异系数', '稳定性评级']])
    """
    validate_dataframe(df, required_cols=[date_col])
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建时间周期
    period_col = '时间周期'
    if freq == 'M':
        df[period_col] = df[date_col].dt.to_period('M').astype(str)
    elif freq == 'W':
        df[period_col] = df[date_col].dt.to_period('W').astype(str)
    elif freq == 'Q':
        df[period_col] = df[date_col].dt.to_period('Q').astype(str)
    
    results = []
    
    for col in features:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # 计算每期的统计量
        period_stats = df.groupby(period_col)[col].agg(['mean', 'std', 'count']).reset_index()
        
        if len(period_stats) < 2:
            continue
        
        mean_std = period_stats['mean'].std()
        mean_mean = period_stats['mean'].mean()
        cv = mean_std / mean_mean if mean_mean != 0 else 0
        
        # 评级
        if cv < 0.05:
            level = '非常稳定'
        elif cv < 0.1:
            level = '相对稳定'
        else:
            level = '不稳定'
        
        results.append({
            '特征名': col,
            '均值标准差': round(mean_std, 4),
            '均值变异系数': round(cv, 4),
            '稳定性评级': level,
            '统计期数': len(period_stats),
        })
    
    return pd.DataFrame(results).sort_values('均值变异系数', ascending=False)
