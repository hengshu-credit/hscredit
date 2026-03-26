# -*- coding: utf-8 -*-
"""
特征与标签关系分析模块 - 预测能力评估.

功能:
- IV (Information Value) 计算
- 分箱逾期率分析
- 单调性检验
- 特征重要性排序
- 单变量AUC分析
- 分箱WOE值分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

from ...core.metrics import IV, IV_table


class FeatureLabelRelationship:
    """特征与标签关系分析类.
    
    分析特征与目标变量的关系，评估特征的预测能力。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param target_col: 目标变量列名
    
    **示例**
    
        >>> from hscredit.core.eda import FeatureLabelRelationship
        >>> flr = FeatureLabelRelationship(df, target_col='FPD15')
        >>> # 计算IV
        >>> iv_results = flr.calculate_iv('age')
        >>> # 批量IV分析
        >>> all_iv = flr.batch_iv_analysis(feature_cols=['age', 'income', 'score'])
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        """初始化特征标签关系分析器."""
        if target_col not in df.columns:
            raise ValueError(f"目标变量 '{target_col}' 不在数据中")
        
        self.df = df.copy()
        self.target_col = target_col
    
    def calculate_iv(self, feature: str, n_bins: int = 10, 
                     method: str = 'quantile') -> Dict:
        """计算单特征IV值.
        
        :param feature: 特征列名
        :param n_bins: 分箱数
        :param method: 分箱方法，'quantile'等频或'uniform'等宽
        :return: IV分析结果字典
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        # 使用hscredit的IV_table函数
        try:
            iv_table_result = IV_table(
                self.df[self.target_col],
                self.df[feature],
                bins=n_bins,
                feature_name=feature
            )
            
            feature_iv = iv_table_result['IV贡献'].sum() if 'IV贡献' in iv_table_result.columns else 0
            
            return {
                '特征名': feature,
                'IV值': round(feature_iv, 4),
                '预测能力': self._iv_strength(feature_iv),
                '分箱详情': iv_table_result,
            }
        except Exception as e:
            return {
                '特征名': feature,
                'IV值': 0,
                '预测能力': '计算失败',
                '错误': str(e),
            }
    
    def _iv_strength(self, iv: float) -> str:
        """根据IV值判断预测能力强弱.
        
        :param iv: IV值
        :return: 预测能力描述
        """
        if iv < 0.02:
            return '无预测能力'
        elif iv < 0.1:
            return '弱预测能力'
        elif iv < 0.3:
            return '中等预测能力'
        elif iv < 0.5:
            return '强预测能力'
        else:
            return '极强预测能力(需检查)'
    
    def batch_iv_analysis(self, feature_cols: List[str], 
                          n_bins: int = 10) -> pd.DataFrame:
        """批量计算IV值.
        
        :param feature_cols: 特征列名列表
        :param n_bins: 分箱数
        :return: IV分析结果 DataFrame
        """
        results = []
        
        for feature in feature_cols:
            if feature == self.target_col:
                continue
            
            try:
                iv_result = self.calculate_iv(feature, n_bins=n_bins)
                results.append({
                    '特征名': iv_result['特征名'],
                    'IV值': iv_result['IV值'],
                    '预测能力': iv_result['预测能力'],
                })
            except Exception as e:
                results.append({
                    '特征名': feature,
                    'IV值': np.nan,
                    '预测能力': f'错误: {str(e)}',
                })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('IV值', ascending=False)
        
        return result_df
    
    def bin_bad_rate_analysis(self, feature: str, n_bins: int = 10) -> pd.DataFrame:
        """分箱逾期率分析.
        
        :param feature: 特征列名
        :param n_bins: 分箱数
        :return: 分箱逾期率分析 DataFrame
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        df_copy = self.df[[feature, self.target_col]].copy()
        df_copy = df_copy.dropna()
        
        # 分箱
        try:
            df_copy['bin'] = pd.qcut(df_copy[feature], q=n_bins, duplicates='drop')
        except:
            df_copy['bin'] = pd.cut(df_copy[feature], bins=n_bins)
        
        # 计算每个箱的统计
        bin_stats = []
        for bin_label, group in df_copy.groupby('bin'):
            stats = {
                '分箱区间': str(bin_label),
                '样本数': len(group),
                '逾期数': (group[self.target_col] == 1).sum(),
                '正常数': (group[self.target_col] == 0).sum(),
                '逾期率(%)': round((group[self.target_col] == 1).sum() / len(group) * 100, 4),
                '占比(%)': round(len(group) / len(df_copy) * 100, 4),
                '特征最小值': group[feature].min(),
                '特征最大值': group[feature].max(),
                '特征均值': round(group[feature].mean(), 4),
            }
            bin_stats.append(stats)
        
        result = pd.DataFrame(bin_stats)
        
        # 计算单调性
        if len(result) > 2:
            bad_rates = result['逾期率(%)'].values
            mono_increasing = all(x <= y for x, y in zip(bad_rates, bad_rates[1:]))
            mono_decreasing = all(x >= y for x, y in zip(bad_rates, bad_rates[1:]))
            result['单调性'] = '单调递增' if mono_increasing else ('单调递减' if mono_decreasing else '非单调')
        
        return result
    
    def monotonicity_test(self, feature: str, n_bins: int = 10) -> Dict:
        """特征单调性检验.
        
        :param feature: 特征列名
        :param n_bins: 分箱数
        :return: 单调性检验结果
        """
        bin_analysis = self.bin_bad_rate_analysis(feature, n_bins)
        
        if len(bin_analysis) < 3:
            return {
                '特征名': feature,
                '是否单调': False,
                '单调类型': '样本不足',
                '斯皮尔曼相关系数': None,
            }
        
        bad_rates = bin_analysis['逾期率(%)'].values
        bin_indices = np.arange(len(bad_rates))
        
        # 斯皮尔曼相关
        corr, pvalue = stats.spearmanr(bin_indices, bad_rates)
        
        mono_increasing = all(x <= y for x, y in zip(bad_rates, bad_rates[1:]))
        mono_decreasing = all(x >= y for x, y in zip(bad_rates, bad_rates[1:]))
        
        return {
            '特征名': feature,
            '是否单调': mono_increasing or mono_decreasing,
            '单调类型': '递增' if mono_increasing else ('递减' if mono_decreasing else '非单调'),
            '斯皮尔曼相关系数': round(corr, 4),
            'P值': round(pvalue, 4),
            '分箱数': len(bin_analysis),
        }
    
    def univariate_auc(self, feature: str) -> Dict:
        """单变量AUC分析.
        
        :param feature: 特征列名
        :return: AUC分析结果
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        try:
            from sklearn.metrics import roc_auc_score
            
            valid_data = self.df[[feature, self.target_col]].dropna()
            
            if len(valid_data) < 10:
                return {'特征名': feature, 'AUC': None, '错误': '有效样本不足'}
            
            auc = roc_auc_score(valid_data[self.target_col], valid_data[feature])
            
            return {
                '特征名': feature,
                'AUC': round(auc, 4),
                'Gini': round(2 * auc - 1, 4),
                '有效样本数': len(valid_data),
            }
        except Exception as e:
            return {'特征名': feature, 'AUC': None, '错误': str(e)}
    
    def woe_analysis(self, feature: str, n_bins: int = 10) -> pd.DataFrame:
        """WOE值分析.
        
        :param feature: 特征列名
        :param n_bins: 分箱数
        :return: WOE分析结果
        """
        if feature not in self.df.columns:
            raise ValueError(f"特征 '{feature}' 不在数据中")
        
        df_copy = self.df[[feature, self.target_col]].copy()
        df_copy = df_copy.dropna()
        
        # 整体好坏分布
        total_bad = (df_copy[self.target_col] == 1).sum()
        total_good = (df_copy[self.target_col] == 0).sum()
        
        # 分箱
        try:
            df_copy['bin'] = pd.qcut(df_copy[feature], q=n_bins, duplicates='drop')
        except:
            df_copy['bin'] = pd.cut(df_copy[feature], bins=n_bins)
        
        # 计算WOE
        woe_stats = []
        for bin_label, group in df_copy.groupby('bin'):
            bin_bad = (group[self.target_col] == 1).sum()
            bin_good = (group[self.target_col] == 0).sum()
            
            # 避免除零
            bad_dist = bin_bad / total_bad if total_bad > 0 else 0.001
            good_dist = bin_good / total_good if total_good > 0 else 0.001
            
            woe = np.log(good_dist / bad_dist) if bad_dist > 0 and good_dist > 0 else 0
            iv = (good_dist - bad_dist) * woe
            
            woe_stats.append({
                '分箱区间': str(bin_label),
                '样本数': len(group),
                '逾期数': bin_bad,
                '正常数': bin_good,
                '逾期率(%)': round(bin_bad / len(group) * 100, 4) if len(group) > 0 else 0,
                'WOE值': round(woe, 4),
                'IV值': round(iv, 4),
            })
        
        result = pd.DataFrame(woe_stats)
        result['累计IV'] = result['IV值'].cumsum()
        
        return result
    
    def feature_importance_ranking(self, feature_cols: List[str]) -> pd.DataFrame:
        """特征重要性综合排序.
        
        :param feature_cols: 特征列名列表
        :return: 综合排序结果
        """
        results = []
        
        for feature in feature_cols:
            if feature == self.target_col:
                continue
            
            try:
                # IV
                iv_result = self.calculate_iv(feature)
                iv_value = iv_result['IV值']
                
                # AUC
                auc_result = self.univariate_auc(feature)
                auc_value = auc_result.get('AUC', np.nan)
                
                # 单调性
                mono_result = self.monotonicity_test(feature)
                is_mono = mono_result['是否单调']
                
                results.append({
                    '特征名': feature,
                    'IV值': iv_value,
                    'AUC': auc_value,
                    '是否单调': is_mono,
                    '单调类型': mono_result['单调类型'],
                })
            except:
                continue
        
        result_df = pd.DataFrame(results)
        
        # 综合得分 (IV标准化 + AUC标准化)
        if len(result_df) > 0:
            result_df['IV排名'] = result_df['IV值'].rank(ascending=False)
            result_df['AUC排名'] = result_df['AUC'].rank(ascending=False)
            result_df['综合排名'] = (result_df['IV排名'] + result_df['AUC排名']) / 2
            result_df = result_df.sort_values('综合排名')
        
        return result_df
    
    def generate_report(self, feature_cols: List[str]) -> Dict:
        """生成特征与标签关系分析报告.
        
        :param feature_cols: 需要分析的特征列表
        :return: 包含所有分析结果的字典
        """
        return {
            'IV分析': self.batch_iv_analysis(feature_cols),
            '特征重要性排序': self.feature_importance_ranking(feature_cols),
        }
