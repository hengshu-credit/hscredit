# -*- coding: utf-8 -*-
"""
数据概览模块 - 数据质量评估与基础统计.

功能:
- 基础信息统计 (样本数、特征数、数据类型)
- 缺失值分析 (缺失率、缺失模式)
- 重复值检测
- 内存使用分析
- 数据类型推断与优化建议
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings


class DataOverview:
    """数据概览分析类.
    
    提供金融信贷数据的全面概览，包括数据质量、缺失值、重复值等分析。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param target_col: 目标变量列名，可选
    
    **示例**
    
        >>> from hscredit.core.eda import DataOverview
        >>> overview = DataOverview(df, target_col='FPD15')
        >>> # 基础信息报告
        >>> report = overview.basic_info()
        >>> # 缺失值分析
        >>> missing_report = overview.missing_analysis()
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """初始化数据概览分析器."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame")
        
        self.df = df.copy()
        self.target_col = target_col
        self.n_samples = len(df)
        self.n_features = len(df.columns)
        
    def basic_info(self) -> Dict:
        """获取数据基础信息.
        
        :return: 包含基础统计信息的字典
        """
        info = {
            '样本总数': self.n_samples,
            '特征总数': self.n_features,
            '数值特征数': len(self.df.select_dtypes(include=[np.number]).columns),
            '类别特征数': len(self.df.select_dtypes(include=['object', 'category']).columns),
            '日期特征数': len(self.df.select_dtypes(include=['datetime']).columns),
            '内存使用(MB)': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            '重复样本数': self.df.duplicated().sum(),
            '重复样本占比': round(self.df.duplicated().sum() / self.n_samples * 100, 4),
        }
        
        if self.target_col and self.target_col in self.df.columns:
            info['目标变量'] = self.target_col
            info['目标缺失数'] = self.df[self.target_col].isnull().sum()
            info['目标缺失率'] = round(info['目标缺失数'] / self.n_samples * 100, 4)
        
        return info
    
    def missing_analysis(self, threshold: float = 0.0) -> pd.DataFrame:
        """缺失值分析.
        
        :param threshold: 缺失率阈值，只返回缺失率>=阈值的特征
        :return: 缺失值分析报告 DataFrame
        """
        missing_count = self.df.isnull().sum()
        missing_rate = missing_count / self.n_samples * 100
        
        # 数据类型
        dtypes = self.df.dtypes.astype(str)
        
        # 唯一值数量
        n_unique = self.df.nunique()
        
        report = pd.DataFrame({
            '数据类型': dtypes,
            '缺失数量': missing_count,
            '缺失率(%)': round(missing_rate, 4),
            '唯一值数量': n_unique,
            '唯一值占比(%)': round(n_unique / self.n_samples * 100, 4),
        })
        
        # 过滤
        if threshold > 0:
            report = report[report['缺失率(%)'] >= threshold]
        
        # 按缺失率排序
        report = report.sort_values('缺失率(%)', ascending=False)
        
        return report
    
    def data_types_summary(self) -> pd.DataFrame:
        """数据类型汇总分析.
        
        :return: 各数据类型的特征统计
        """
        dtype_summary = []
        
        for dtype_kind in ['numeric', 'category', 'datetime', 'bool', 'object']:
            if dtype_kind == 'numeric':
                cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            elif dtype_kind == 'category':
                cols = self.df.select_dtypes(include=['category']).columns.tolist()
            elif dtype_kind == 'datetime':
                cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
            elif dtype_kind == 'bool':
                cols = self.df.select_dtypes(include=['bool']).columns.tolist()
            else:
                cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            if cols:
                dtype_summary.append({
                    '数据类型': dtype_kind,
                    '特征数量': len(cols),
                    '特征列表': ', '.join(cols[:5]) + ('...' if len(cols) > 5 else ''),
                })
        
        return pd.DataFrame(dtype_summary)
    
    def numeric_summary(self) -> pd.DataFrame:
        """数值特征统计摘要.
        
        :return: 数值特征的统计描述
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        stats = self.df[numeric_cols].describe().T
        stats['缺失率(%)'] = round(self.df[numeric_cols].isnull().sum() / self.n_samples * 100, 4)
        stats['偏度'] = self.df[numeric_cols].skew()
        stats['峰度'] = self.df[numeric_cols].kurtosis()
        stats['零值数量'] = (self.df[numeric_cols] == 0).sum()
        stats['零值占比(%)'] = round(stats['零值数量'] / self.n_samples * 100, 4)
        stats['负值数量'] = (self.df[numeric_cols] < 0).sum()
        stats['负值占比(%)'] = round(stats['负值数量'] / self.n_samples * 100, 4)
        
        # 重命名列
        stats = stats.rename(columns={
            'count': '非空计数',
            'mean': '均值',
            'std': '标准差',
            'min': '最小值',
            '25%': '25分位数',
            '50%': '中位数',
            '75%': '75分位数',
            'max': '最大值',
        })
        
        return stats.round(4)
    
    def category_summary(self, max_categories: int = 10) -> Dict[str, pd.DataFrame]:
        """类别特征统计摘要.
        
        :param max_categories: 每个类别特征最多显示的分类数
        :return: 字典，key为特征名，value为分类统计DataFrame
        """
        category_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        result = {}
        for col in category_cols:
            value_counts = self.df[col].value_counts(dropna=False)
            value_ratio = value_counts / self.n_samples * 100
            
            summary = pd.DataFrame({
                '分类值': value_counts.index.astype(str),
                '样本数': value_counts.values,
                '占比(%)': round(value_ratio.values, 4),
            })
            
            # 限制显示数量
            if len(summary) > max_categories:
                summary = summary.head(max_categories)
                summary.loc[len(summary)] = ['...', '...', '...']
            
            result[col] = summary
        
        return result
    
    def constant_features(self) -> pd.DataFrame:
        """识别常数/准常数特征.
        
        :return: 常数/准常数特征报告
        """
        constant_cols = []
        quasi_constant_cols = []
        
        for col in self.df.columns:
            if col == self.target_col:
                continue
                
            n_unique = self.df[col].nunique(dropna=False)
            
            # 完全常数
            if n_unique == 1:
                constant_cols.append({
                    '特征名': col,
                    '类型': '完全常数',
                    '唯一值': str(self.df[col].iloc[0]),
                    '占比': '100%',
                })
            # 准常数 (某个值占比>99%)
            elif n_unique > 1:
                mode_ratio = self.df[col].value_counts(normalize=True).iloc[0]
                if mode_ratio > 0.99:
                    quasi_constant_cols.append({
                        '特征名': col,
                        '类型': '准常数',
                        '主导值': str(self.df[col].value_counts().index[0]),
                        '主导占比': f'{mode_ratio*100:.2f}%',
                    })
        
        return pd.DataFrame(constant_cols + quasi_constant_cols)
    
    def data_quality_score(self) -> Dict:
        """计算数据质量评分.
        
        :return: 数据质量评分字典
        """
        scores = {}
        
        # 1. 完整性评分 (基于缺失率)
        missing_rates = self.df.isnull().mean()
        completeness_score = (1 - missing_rates.mean()) * 100
        scores['完整性评分'] = round(completeness_score, 2)
        
        # 2. 唯一性评分 (基于重复率)
        duplicate_rate = self.df.duplicated().sum() / self.n_samples
        uniqueness_score = (1 - duplicate_rate) * 100
        scores['唯一性评分'] = round(uniqueness_score, 2)
        
        # 3. 有效性评分 (基于常数特征比例)
        constant_count = len(self.constant_features())
        validity_score = (1 - constant_count / max(self.n_features, 1)) * 100
        scores['有效性评分'] = round(validity_score, 2)
        
        # 4. 综合评分
        scores['综合数据质量评分'] = round(
            (completeness_score + uniqueness_score + validity_score) / 3, 2
        )
        
        # 评级
        total_score = scores['综合数据质量评分']
        if total_score >= 90:
            scores['质量评级'] = '优秀'
        elif total_score >= 75:
            scores['质量评级'] = '良好'
        elif total_score >= 60:
            scores['质量评级'] = '及格'
        else:
            scores['质量评级'] = '较差'
        
        return scores
    
    def generate_report(self) -> Dict:
        """生成完整的数据概览报告.
        
        :return: 包含所有分析结果的字典
        """
        return {
            '基础信息': self.basic_info(),
            '数据质量评分': self.data_quality_score(),
            '缺失值分析': self.missing_analysis(),
            '数据类型汇总': self.data_types_summary(),
            '数值特征统计': self.numeric_summary(),
            '类别特征统计': self.category_summary(),
            '常数特征': self.constant_features(),
        }
