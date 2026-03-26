# -*- coding: utf-8 -*-
"""
特征分析模块 - 单变量特征探索.

功能:
- 数值特征分布分析 (均值、分位数、偏度、峰度)
- 类别特征分布分析 (频次、占比)
- 离群值检测
- 特征趋势分析 (时间维度)
- 集中度分析 (基尼系数、熵)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
import warnings


class FeatureAnalysis:
    """特征分析类.
    
    提供金融信贷特征的单变量分析，包括分布、趋势、离群值检测等。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param date_col: 日期列名，用于时间趋势分析，可选
    
    **示例**
    
        >>> from hscredit.core.eda import FeatureAnalysis
        >>> fa = FeatureAnalysis(df, date_col='loan_date')
        >>> # 数值特征分析
        >>> num_stats = fa.numeric_analysis('age')
        >>> # 离群值检测
        >>> outliers = fa.detect_outliers('income')
    """
    
    def __init__(self, df: pd.DataFrame, date_col: Optional[str] = None):
        """初始化特征分析器."""
        self.df = df.copy()
        self.date_col = date_col
        
        # 尝试转换日期列
        if date_col and date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                except:
                    warnings.warn(f"无法将 '{date_col}' 转换为日期类型")
    
    def numeric_analysis(self, feature: str) -> Dict:
        """数值特征分布分析.
        
        :param feature: 特征列名
        :return: 统计信息字典
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        series = self.df[feature].dropna()
        
        if len(series) == 0:
            return {'错误': '该特征无有效值'}
        
        stats_dict = {
            '样本数': len(series),
            '缺失数': self.df[feature].isna().sum(),
            '缺失率(%)': round(self.df[feature].isna().sum() / len(self.df) * 100, 4),
            '均值': round(series.mean(), 4),
            '标准差': round(series.std(), 4),
            '最小值': series.min(),
            '最大值': series.max(),
            '中位数': series.median(),
            '25分位数': series.quantile(0.25),
            '75分位数': series.quantile(0.75),
            '偏度': round(series.skew(), 4),
            '峰度': round(series.kurtosis(), 4),
            '变异系数': round(series.std() / max(series.mean(), 1e-10), 4),
        }
        
        # 计算极值比例
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        stats_dict['离群值数量(IQR)'] = len(outliers)
        stats_dict['离群值比例(IQR)'] = round(len(outliers) / len(series) * 100, 4)
        
        return stats_dict
    
    def category_analysis(self, feature: str, top_n: int = 10) -> pd.DataFrame:
        """类别特征分布分析.
        
        :param feature: 特征列名
        :param top_n: 显示前N个类别
        :return: 类别统计 DataFrame
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        value_counts = self.df[feature].value_counts(dropna=False)
        value_ratio = value_counts / len(self.df) * 100
        
        result = pd.DataFrame({
            '分类值': value_counts.index.astype(str),
            '样本数': value_counts.values,
            '占比(%)': round(value_ratio.values, 4),
            '累计占比(%)': round(value_ratio.cumsum().values, 4),
        })
        
        return result.head(top_n)
    
    def detect_outliers(self, feature: str, method: str = 'iqr') -> pd.DataFrame:
        """离群值检测.
        
        :param feature: 特征列名
        :param method: 检测方法，'iqr'或'zscore'
        :return: 离群值 DataFrame
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        series = self.df[feature].dropna()
        
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = self.df[(self.df[feature] < lower_bound) | 
                              (self.df[feature] > upper_bound)][[feature]]
            outliers = outliers.copy()
            outliers['检测方法'] = 'IQR'
            outliers['下限'] = lower_bound
            outliers['上限'] = upper_bound
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outlier_idx = series.index[z_scores > 3]
            
            outliers = self.df.loc[outlier_idx, [feature]].copy()
            outliers['检测方法'] = 'Z-Score'
            outliers['Z分数'] = z_scores[z_scores > 3]
            outliers['阈值'] = 3
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        return outliers
    
    def time_trend(self, feature: str, freq: str = 'M', agg_func: str = 'mean') -> pd.DataFrame:
        """特征时间趋势分析.
        
        :param feature: 特征列名
        :param freq: 时间频率，'D'日, 'W'周, 'M'月, 'Q'季度
        :param agg_func: 聚合函数，'mean', 'median', 'sum', 'count'
        :return: 时间趋势 DataFrame
        """
        if not self.date_col:
            raise ValueError("未指定日期列，无法进行时间趋势分析")
        
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        df_copy = self.df[[self.date_col, feature]].copy()
        df_copy = df_copy.dropna()
        df_copy['period'] = df_copy[self.date_col].dt.to_period(freq)
        
        if agg_func == 'mean':
            trend = df_copy.groupby('period')[feature].mean()
        elif agg_func == 'median':
            trend = df_copy.groupby('period')[feature].median()
        elif agg_func == 'sum':
            trend = df_copy.groupby('period')[feature].sum()
        elif agg_func == 'count':
            trend = df_copy.groupby('period')[feature].count()
        else:
            raise ValueError(f"不支持的聚合函数: {agg_func}")
        
        result = pd.DataFrame({
            '时间周期': trend.index.astype(str),
            f'{feature}_{agg_func}': trend.values.round(4),
            '环比变化': trend.pct_change().values.round(4) * 100,
        })
        
        return result
    
    def concentration_analysis(self, feature: str) -> Dict:
        """特征集中度分析.
        
        :param feature: 特征列名
        :return: 集中度指标字典
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        series = self.df[feature].dropna()
        
        result = {}
        
        # 基尼系数 (用于评估分布不平等程度)
        def gini_coefficient(x):
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        try:
            result['基尼系数'] = round(gini_coefficient(series), 4)
        except:
            result['基尼系数'] = None
        
        # 熵 (用于评估类别分布)
        if series.dtype == 'object' or series.nunique() < 20:
            value_counts = series.value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            result['熵'] = round(entropy, 4)
            result['集中度(Top1占比%)'] = round(series.value_counts(normalize=True).iloc[0] * 100, 4)
        
        # 零值占比
        zero_ratio = (series == 0).sum() / len(series)
        result['零值占比(%)'] = round(zero_ratio * 100, 4)
        
        return result
    
    def distribution_shift(self, feature: str, time_split: str) -> Dict:
        """特征分布漂移分析.
        
        :param feature: 特征列名
        :param time_split: 时间分割点，用于划分前后两个时期
        :return: 分布漂移指标
        """
        if not self.date_col:
            raise ValueError("未指定日期列，无法进行分布漂移分析")
        
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        split_date = pd.to_datetime(time_split)
        
        period1 = self.df[self.df[self.date_col] < split_date][feature].dropna()
        period2 = self.df[self.df[self.date_col] >= split_date][feature].dropna()
        
        result = {
            '时期1样本数': len(period1),
            '时期2样本数': len(period2),
            '时期1均值': round(period1.mean(), 4),
            '时期2均值': round(period2.mean(), 4),
            '均值变化(%)': round((period2.mean() - period1.mean()) / period1.mean() * 100, 4) if period1.mean() != 0 else None,
            '时期1标准差': round(period1.std(), 4),
            '时期2标准差': round(period2.std(), 4),
        }
        
        # KS检验
        try:
            ks_stat, p_value = stats.ks_2samp(period1, period2)
            result['KS统计量'] = round(ks_stat, 4)
            result['KS检验P值'] = round(p_value, 4)
            result['分布是否显著变化'] = p_value < 0.05
        except:
            result['KS统计量'] = None
            result['KS检验P值'] = None
        
        return result
    
    def generate_report(self, features: Optional[List[str]] = None) -> Dict:
        """生成特征分析报告.
        
        :param features: 需要分析的特征列表，None则分析所有特征
        :return: 包含所有分析结果的字典
        """
        if features is None:
            features = self.df.columns.tolist()
        
        report = {
            '数值特征分析': {},
            '类别特征分析': {},
        }
        
        for feature in features:
            if feature == self.date_col:
                continue
                
            if feature not in self.df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(self.df[feature]):
                report['数值特征分析'][feature] = {
                    '基础统计': self.numeric_analysis(feature),
                    '集中度': self.concentration_analysis(feature),
                }
            else:
                report['类别特征分析'][feature] = {
                    '分布': self.category_analysis(feature),
                    '集中度': self.concentration_analysis(feature),
                }
        
        return report
