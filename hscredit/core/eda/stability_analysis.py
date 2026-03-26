# -*- coding: utf-8 -*-
"""
特征稳定性分析模块 - PSI、时间稳定性评估.

功能:
- PSI (Population Stability Index) 计算
- 特征分布时间稳定性
- 逾期率时间稳定性
- 特征稳定性评级
- 跨时间段对比分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings

from ...core.metrics import PSI, PSI_table


class StabilityAnalysis:
    """特征稳定性分析类.
    
    分析金融信贷特征随时间的稳定性，主要用于模型监控和特征筛选。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param date_col: 日期列名
    
    **示例**
    
        >>> from hscredit.core.eda import StabilityAnalysis
        >>> sa = StabilityAnalysis(df, date_col='loan_date')
        >>> # 计算PSI
        >>> psi_result = sa.calculate_psi('age', base_period='2023-01', test_period='2023-06')
        >>> # 批量PSI分析
        >>> all_psi = sa.batch_psi_analysis(['age', 'income'], base_period='2023-01')
    """
    
    def __init__(self, df: pd.DataFrame, date_col: str):
        """初始化稳定性分析器."""
        if date_col not in df.columns:
            raise ValueError(f"日期列 '{date_col}' 不在数据中")
        
        self.df = df.copy()
        self.date_col = date_col
        
        # 转换为日期类型
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
            except:
                raise ValueError(f"无法将 '{date_col}' 转换为日期类型")
    
    def calculate_psi(self, feature: str, base_period: str, test_period: str,
                      n_bins: int = 10) -> Dict:
        """计算特征PSI值.
        
        :param feature: 特征列名
        :param base_period: 基准时间段 (如 '2023-01', '2023-01-01')
        :param test_period: 测试时间段
        :param n_bins: 分箱数
        :return: PSI分析结果
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        # 划分时间段
        base_data = self._get_period_data(base_period)
        test_data = self._get_period_data(test_period)
        
        if len(base_data) == 0 or len(test_data) == 0:
            return {
                '特征名': feature,
                'PSI': np.nan,
                '稳定性': '数据不足',
                '基准样本数': len(base_data),
                '测试样本数': len(test_data),
            }
        
        try:
            # 使用hscredit的PSI函数
            psi_value = PSI(
                base_data[feature].dropna(),
                test_data[feature].dropna(),
                n_bins=n_bins
            )
            
            return {
                '特征名': feature,
                'PSI': round(psi_value, 4),
                '稳定性': self._psi_stability(psi_value),
                '基准样本数': len(base_data),
                '测试样本数': len(test_data),
                '基准时间段': base_period,
                '测试时间段': test_period,
            }
        except Exception as e:
            return {
                '特征名': feature,
                'PSI': np.nan,
                '稳定性': f'计算错误: {str(e)}',
                '基准样本数': len(base_data),
                '测试样本数': len(test_data),
            }
    
    def _get_period_data(self, period: str) -> pd.DataFrame:
        """根据时间段字符串获取数据.
        
        :param period: 时间段字符串 (如 '2023-01', '2023-01-01')
        :return: 过滤后的数据
        """
        period_dt = pd.to_datetime(period)
        
        # 判断是月还是日
        if len(period) <= 7:  # 年月格式
            mask = (self.df[self.date_col].dt.year == period_dt.year) & \
                   (self.df[self.date_col].dt.month == period_dt.month)
        else:  # 具体日期
            mask = self.df[self.date_col].dt.date == period_dt.date
        
        return self.df[mask]
    
    def _psi_stability(self, psi: float) -> str:
        """根据PSI值判断稳定性.
        
        :param psi: PSI值
        :return: 稳定性描述
        """
        if psi < 0.1:
            return '非常稳定'
        elif psi < 0.25:
            return '相对稳定'
        else:
            return '不稳定'
    
    def batch_psi_analysis(self, features: List[str], base_period: str,
                           test_periods: Optional[List[str]] = None,
                           n_bins: int = 10) -> pd.DataFrame:
        """批量PSI分析.
        
        :param features: 特征列表
        :param base_period: 基准时间段
        :param test_periods: 测试时间段列表，None则自动获取后续月份
        :param n_bins: 分箱数
        :return: PSI分析结果 DataFrame
        """
        if test_periods is None:
            # 自动获取所有月份
            test_periods = self._get_all_periods()
            # 排除基准月份
            test_periods = [p for p in test_periods if p != base_period]
        
        results = []
        for feature in features:
            for test_period in test_periods:
                result = self.calculate_psi(feature, base_period, test_period, n_bins)
                results.append(result)
        
        return pd.DataFrame(results)
    
    def _get_all_periods(self) -> List[str]:
        """获取数据中的所有时间段（月份）."""
        periods = self.df[self.date_col].dt.to_period('M').unique()
        return [str(p) for p in sorted(periods)]
    
    def time_stability_trend(self, feature: str, freq: str = 'M',
                             n_bins: int = 10) -> pd.DataFrame:
        """特征稳定性时间趋势.
        
        :param feature: 特征列名
        :param freq: 时间频率
        :param n_bins: 分箱数
        :return: 稳定性趋势 DataFrame
        """
        periods = self._get_all_periods()
        
        if len(periods) < 2:
            return pd.DataFrame()
        
        base_period = periods[0]
        results = []
        
        for test_period in periods[1:]:
            result = self.calculate_psi(feature, base_period, test_period, n_bins)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def bad_rate_stability(self, feature: str, n_bins: int = 10,
                           freq: str = 'M') -> pd.DataFrame:
        """逾期率稳定性分析.
        
        :param feature: 特征列名
        :param n_bins: 分箱数
        :param freq: 时间频率
        :return: 逾期率稳定性分析
        """
        from ...core.metrics import KS
        
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        target_col = None
        for col in ['target', 'label', 'FPD15', 'FPD30', 'bad']:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("数据中未找到目标变量列")
        
        # 按时间段分析
        self.df['period'] = self.df[self.date_col].dt.to_period(freq)
        
        results = []
        for period, group in self.df.groupby('period'):
            if len(group) < 100:  # 跳过样本过少的时间段
                continue
            
            try:
                bad_rate = group[target_col].mean()
                
                # 计算该时间段内的KS
                ks_value = KS(group[target_col], group[feature])
                
                results.append({
                    '时间段': str(period),
                    '样本数': len(group),
                    '逾期率(%)': round(bad_rate * 100, 4),
                    'KS值': round(ks_value, 4),
                })
            except:
                continue
        
        result_df = pd.DataFrame(results)
        
        # 计算变异系数
        if len(result_df) > 1:
            result_df['逾期率变化'] = result_df['逾期率(%)'].diff()
            result_df['逾期率CV'] = round(result_df['逾期率(%)'].std() / result_df['逾期率(%)'].mean(), 4)
        
        return result_df
    
    def stability_rating(self, features: List[str], base_period: str) -> pd.DataFrame:
        """特征稳定性评级.
        
        :param features: 特征列表
        :param base_period: 基准时间段
        :return: 稳定性评级结果
        """
        all_periods = self._get_all_periods()
        test_periods = [p for p in all_periods if p != base_period]
        
        results = []
        for feature in features:
            psi_values = []
            
            for test_period in test_periods:
                result = self.calculate_psi(feature, base_period, test_period)
                if not np.isnan(result['PSI']):
                    psi_values.append(result['PSI'])
            
            if len(psi_values) > 0:
                avg_psi = np.mean(psi_values)
                max_psi = np.max(psi_values)
                
                results.append({
                    '特征名': feature,
                    '平均PSI': round(avg_psi, 4),
                    '最大PSI': round(max_psi, 4),
                    'PSI标准差': round(np.std(psi_values), 4),
                    '稳定性评级': self._stability_rating_grade(avg_psi, max_psi),
                })
        
        result_df = pd.DataFrame(results)
        return result_df.sort_values('平均PSI')
    
    def _stability_rating_grade(self, avg_psi: float, max_psi: float) -> str:
        """计算稳定性评级.
        
        :param avg_psi: 平均PSI
        :param max_psi: 最大PSI
        :return: 评级字符串
        """
        if avg_psi < 0.05 and max_psi < 0.1:
            return 'A - 非常稳定'
        elif avg_psi < 0.1 and max_psi < 0.25:
            return 'B - 稳定'
        elif avg_psi < 0.25 and max_psi < 0.35:
            return 'C - 一般稳定'
        else:
            return 'D - 不稳定'
    
    def distribution_comparison(self, feature: str, period1: str, period2: str) -> Dict:
        """两个时间段的分布对比.
        
        :param feature: 特征列名
        :param period1: 第一个时间段
        :param period2: 第二个时间段
        :return: 分布对比结果
        """
        data1 = self._get_period_data(period1)
        data2 = self._get_period_data(period2)
        
        series1 = data1[feature].dropna()
        series2 = data2[feature].dropna()
        
        result = {
            '特征名': feature,
            '时间段1': period1,
            '时间段2': period2,
            '样本数1': len(series1),
            '样本数2': len(series2),
            '均值1': round(series1.mean(), 4),
            '均值2': round(series2.mean(), 4),
            '均值变化(%)': round((series2.mean() - series1.mean()) / series1.mean() * 100, 4) if series1.mean() != 0 else None,
            '标准差1': round(series1.std(), 4),
            '标准差2': round(series2.std(), 4),
            '中位数1': series1.median(),
            '中位数2': series2.median(),
        }
        
        # KS检验
        from scipy import stats
        try:
            ks_stat, p_value = stats.ks_2samp(series1, series2)
            result['KS统计量'] = round(ks_stat, 4)
            result['KS检验P值'] = round(p_value, 4)
            result['分布是否显著变化'] = p_value < 0.05
        except:
            result['KS统计量'] = None
            result['KS检验P值'] = None
        
        return result
    
    def generate_report(self, features: List[str], base_period: str) -> Dict:
        """生成稳定性分析报告.
        
        :param features: 特征列表
        :param base_period: 基准时间段
        :return: 包含所有分析结果的字典
        """
        return {
            '稳定性评级': self.stability_rating(features, base_period),
            'PSI详细分析': self.batch_psi_analysis(features, base_period),
        }
