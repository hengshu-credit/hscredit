# -*- coding: utf-8 -*-
"""
目标变量分析模块 - 逾期率分析与标签分布.

功能:
- 目标变量分布分析
- 逾期率统计
- 时间维度逾期率趋势
- 逾期率按不同维度拆解
- 目标变量与业务指标关系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings


class TargetAnalysis:
    """目标变量分析类.
    
    分析金融信贷场景中的目标变量（如逾期标签），提供逾期率统计、趋势分析等功能。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param target_col: 目标变量列名
    :param date_col: 日期列名，用于时间趋势分析，可选
    
    **示例**
    
        >>> from hscredit.core.eda import TargetAnalysis
        >>> target_analyzer = TargetAnalysis(df, target_col='FPD15', date_col='loan_date')
        >>> # 基础逾期率统计
        >>> stats = target_analyzer.basic_stats()
        >>> # 时间趋势
        >>> trend = target_analyzer.time_trend(freq='M')
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, date_col: Optional[str] = None):
        """初始化目标变量分析器."""
        if target_col not in df.columns:
            raise ValueError(f"目标变量 '{target_col}' 不在数据中")
        
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        # 尝试转换日期列
        if date_col and date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                except:
                    warnings.warn(f"无法将 '{date_col}' 转换为日期类型")
    
    def basic_stats(self) -> Dict:
        """目标变量基础统计.
        
        :return: 基础统计信息字典
        """
        target = self.df[self.target_col]
        
        # 检查是否为二分类问题
        unique_values = target.dropna().unique()
        is_binary = len(unique_values) == 2 and set(unique_values).issubset({0, 1})
        
        stats = {
            '总样本数': len(target),
            '有效样本数': target.notna().sum(),
            '缺失样本数': target.isna().sum(),
            '缺失率(%)': round(target.isna().sum() / len(target) * 100, 4),
            '唯一值数量': target.nunique(dropna=False),
            '唯一值列表': list(target.dropna().unique())[:10],
        }
        
        if is_binary:
            # 二分类问题 - 计算逾期率
            bad_count = (target == 1).sum()
            good_count = (target == 0).sum()
            
            stats['正样本数(逾期)'] = int(bad_count)
            stats['负样本数(正常)'] = int(good_count)
            stats['逾期率(%)'] = round(bad_count / target.notna().sum() * 100, 4)
            stats['样本不平衡比例'] = round(good_count / max(bad_count, 1), 2)
        else:
            # 连续变量或其他
            stats['均值'] = round(target.mean(), 4)
            stats['标准差'] = round(target.std(), 4)
            stats['最小值'] = target.min()
            stats['最大值'] = target.max()
            stats['中位数'] = target.median()
        
        return stats
    
    def distribution_by_group(self, group_col: str) -> pd.DataFrame:
        """按分组变量分析目标分布.
        
        :param group_col: 分组列名
        :return: 分组统计 DataFrame
        """
        if group_col not in self.df.columns:
            raise ValueError(f"分组变量 '{group_col}' 不在数据中")
        
        # 分组统计
        grouped = self.df.groupby(group_col)[self.target_col].agg([
            ('样本数', 'count'),
            ('逾期数', lambda x: (x == 1).sum()),
            ('逾期率(%)', lambda x: (x == 1).sum() / x.notna().sum() * 100),
            ('缺失数', lambda x: x.isna().sum()),
        ]).reset_index()
        
        grouped['样本占比(%)'] = round(grouped['样本数'] / len(self.df) * 100, 4)
        
        return grouped.round(4)
    
    def time_trend(self, freq: str = 'M') -> pd.DataFrame:
        """时间维度逾期率趋势分析.
        
        :param freq: 时间频率，'D'日, 'W'周, 'M'月, 'Q'季度
        :return: 时间趋势 DataFrame
        """
        if not self.date_col:
            raise ValueError("未指定日期列，无法进行时间趋势分析")
        
        if self.date_col not in self.df.columns:
            raise ValueError(f"日期列 '{self.date_col}' 不在数据中")
        
        # 复制数据避免修改
        df_copy = self.df[[self.date_col, self.target_col]].copy()
        df_copy = df_copy.dropna(subset=[self.date_col])
        
        # 按时间频率分组
        df_copy['period'] = df_copy[self.date_col].dt.to_period(freq)
        
        trend = df_copy.groupby('period')[self.target_col].agg([
            ('样本数', 'count'),
            ('逾期数', lambda x: (x == 1).sum()),
            ('逾期率(%)', lambda x: (x == 1).sum() / x.notna().sum() * 100),
        ]).reset_index()
        
        trend['period'] = trend['period'].astype(str)
        trend['逾期率变化'] = trend['逾期率(%)'].diff()
        
        return trend.round(4)
    
    def cross_analysis(self, cross_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """交叉维度逾期率分析.
        
        :param cross_cols: 交叉分析的列名列表
        :return: 字典，包含各维度的交叉分析结果
        """
        results = {}
        
        for col in cross_cols:
            if col not in self.df.columns:
                continue
            
            result = self.distribution_by_group(col)
            results[col] = result
        
        return results
    
    def bad_rate_by_bins(self, feature_col: str, n_bins: int = 10) -> pd.DataFrame:
        """按特征分箱分析逾期率.
        
        :param feature_col: 特征列名
        :param n_bins: 分箱数
        :return: 分箱逾期率分析
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"特征 '{feature_col}' 不在数据中")
        
        df_copy = self.df[[feature_col, self.target_col]].copy()
        df_copy = df_copy.dropna()
        
        # 等频分箱
        try:
            df_copy['bin'] = pd.qcut(df_copy[feature_col], q=n_bins, duplicates='drop')
        except:
            # 如果等频分箱失败，使用等宽分箱
            df_copy['bin'] = pd.cut(df_copy[feature_col], bins=n_bins)
        
        bin_stats = df_copy.groupby('bin').agg({
            self.target_col: ['count', 'sum', 'mean'],
            feature_col: ['min', 'max']
        }).reset_index()
        
        bin_stats.columns = ['分箱区间', '样本数', '逾期数', '逾期率', '特征最小值', '特征最大值']
        bin_stats['逾期率(%)'] = round(bin_stats['逾期率'] * 100, 4)
        bin_stats['样本占比(%)'] = round(bin_stats['样本数'] / len(df_copy) * 100, 4)
        
        return bin_stats[['分箱区间', '样本数', '样本占比(%)', '逾期数', '逾期率(%)', '特征最小值', '特征最大值']]
    
    def performance_metrics(self, prob_col: Optional[str] = None) -> Dict:
        """计算业务绩效指标.
        
        :param prob_col: 预测概率列名，可选
        :return: 业务指标字典
        """
        metrics = {}
        
        # 基础逾期率
        bad_rate = self.df[self.target_col].mean()
        metrics['整体逾期率'] = round(bad_rate, 4)
        
        # 如果提供了预测概率，计算排序性指标
        if prob_col and prob_col in self.df.columns:
            from sklearn.metrics import roc_auc_score
            
            valid_data = self.df[[self.target_col, prob_col]].dropna()
            if len(valid_data) > 0:
                auc = roc_auc_score(valid_data[self.target_col], valid_data[prob_col])
                metrics['AUC'] = round(auc, 4)
        
        # 如果存在金额列，计算金额加权指标
        amount_cols = [c for c in self.df.columns if 'amount' in c.lower() or 'amt' in c.lower()]
        for amt_col in amount_cols[:1]:  # 只取第一个金额列
            valid_data = self.df[[self.target_col, amt_col]].dropna()
            if len(valid_data) > 0:
                total_amt = valid_data[amt_col].sum()
                bad_amt = valid_data[valid_data[self.target_col] == 1][amt_col].sum()
                metrics[f'{amt_col}_总额'] = round(total_amt, 2)
                metrics[f'{amt_col}_逾期金额'] = round(bad_amt, 2)
                metrics[f'{amt_col}_金额逾期率'] = round(bad_amt / total_amt, 4)
        
        return metrics
    
    def generate_report(self) -> Dict:
        """生成完整的目标变量分析报告.
        
        :return: 包含所有分析结果的字典
        """
        report = {
            '基础统计': self.basic_stats(),
            '绩效指标': self.performance_metrics(),
        }
        
        if self.date_col:
            report['时间趋势(按月)'] = self.time_trend(freq='M')
        
        return report
