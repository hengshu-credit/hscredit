# -*- coding: utf-8 -*-
"""
相关性分析模块 - 特征间关系分析.

功能:
- 皮尔逊相关系数
- 斯皮尔曼相关系数
- 相关性热力图数据
- 高相关性特征对识别
- 多重共线性检测 (VIF)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings


class CorrelationAnalysis:
    """相关性分析类.
    
    分析特征间的相关性，用于特征去重和多重共线性检测。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    
    **示例**
    
        >>> from hscredit.core.eda import CorrelationAnalysis
        >>> ca = CorrelationAnalysis(df)
        >>> # 计算相关系数矩阵
        >>> corr_matrix = ca.correlation_matrix(method='pearson')
        >>> # 识别高相关性特征对
        >>> high_corr = ca.high_correlation_pairs(threshold=0.8)
    """
    
    def __init__(self, df: pd.DataFrame):
        """初始化相关性分析器."""
        self.df = df.copy()
        # 只保留数值列
        self.numeric_df = df.select_dtypes(include=[np.number])
    
    def correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """计算相关系数矩阵.
        
        :param method: 相关方法，'pearson', 'spearman', 'kendall'
        :return: 相关系数矩阵
        """
        if len(self.numeric_df.columns) < 2:
            return pd.DataFrame()
        
        return self.numeric_df.corr(method=method)
    
    def high_correlation_pairs(self, threshold: float = 0.8,
                               method: str = 'pearson') -> pd.DataFrame:
        """识别高相关性特征对.
        
        :param threshold: 相关系数阈值
        :param method: 相关方法
        :return: 高相关性特征对 DataFrame
        """
        corr_matrix = self.correlation_matrix(method=method)
        
        # 获取上三角矩阵
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        '特征1': corr_matrix.columns[i],
                        '特征2': corr_matrix.columns[j],
                        '相关系数': round(corr_value, 4),
                        '绝对相关系数': round(abs(corr_value), 4),
                    })
        
        result = pd.DataFrame(high_corr_pairs)
        
        if len(result) > 0:
            result = result.sort_values('绝对相关系数', ascending=False)
        
        return result
    
    def correlation_with_target(self, target_col: str,
                                method: str = 'pearson') -> pd.DataFrame:
        """计算特征与目标变量的相关性.
        
        :param target_col: 目标变量列名
        :param method: 相关方法
        :return: 相关性分析结果
        """
        if target_col not in self.df.columns:
            raise ValueError(f"目标变量 '{target_col}' 不在数据中")
        
        if not pd.api.types.is_numeric_dtype(self.df[target_col]):
            raise ValueError(f"目标变量 '{target_col}' 必须是数值类型")
        
        correlations = []
        
        for col in self.numeric_df.columns:
            if col == target_col:
                continue
            
            # 计算相关性
            valid_data = self.df[[col, target_col]].dropna()
            
            if len(valid_data) < 10:
                continue
            
            if method == 'pearson':
                corr = valid_data[col].corr(valid_data[target_col], method='pearson')
            elif method == 'spearman':
                corr = valid_data[col].corr(valid_data[target_col], method='spearman')
            else:
                corr = valid_data[col].corr(valid_data[target_col], method='kendall')
            
            correlations.append({
                '特征名': col,
                '相关系数': round(corr, 4),
                '绝对相关系数': round(abs(corr), 4),
            })
        
        result = pd.DataFrame(correlations)
        
        if len(result) > 0:
            result = result.sort_values('绝对相关系数', ascending=False)
        
        return result
    
    def vif_analysis(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """VIF (方差膨胀因子) 分析.
        
        :param features: 需要分析的特征列表，None则分析所有数值特征
        :return: VIF分析结果
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            warnings.warn("statsmodels未安装，无法计算VIF")
            return pd.DataFrame()
        
        if features is None:
            features = self.numeric_df.columns.tolist()
        
        # 过滤有效特征
        features = [f for f in features if f in self.numeric_df.columns]
        
        if len(features) < 2:
            return pd.DataFrame()
        
        # 处理缺失值
        data = self.numeric_df[features].dropna()
        
        if len(data) < len(features):
            warnings.warn(f"VIF分析使用 {len(data)} 条完整样本")
        
        try:
            vif_data = pd.DataFrame()
            vif_data['特征名'] = features
            vif_data['VIF'] = [variance_inflation_factor(data.values, i) 
                               for i in range(len(features))]
            vif_data['VIF'] = vif_data['VIF'].round(4)
            vif_data['共线性程度'] = vif_data['VIF'].apply(self._vif_interpretation)
            
            return vif_data.sort_values('VIF', ascending=False)
        except Exception as e:
            warnings.warn(f"VIF计算失败: {str(e)}")
            return pd.DataFrame()
    
    def _vif_interpretation(self, vif: float) -> str:
        """VIF值解释.
        
        :param vif: VIF值
        :return: 共线性程度描述
        """
        if vif < 5:
            return '无共线性'
        elif vif < 10:
            return '中度共线性'
        else:
            return '严重共线性'
    
    def redundancy_analysis(self, threshold: float = 0.95) -> pd.DataFrame:
        """冗余特征分析 (高相关性特征组).
        
        :param threshold: 相关系数阈值
        :return: 冗余特征分析结果
        """
        high_corr_pairs = self.high_correlation_pairs(threshold=threshold)
        
        if len(high_corr_pairs) == 0:
            return pd.DataFrame()
        
        # 使用并查集找出相关特征组
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 合并高相关特征
        for _, row in high_corr_pairs.iterrows():
            union(row['特征1'], row['特征2'])
        
        # 分组
        groups = {}
        for feature in parent:
            root = find(feature)
            if root not in groups:
                groups[root] = []
            groups[root].append(feature)
        
        # 筛选出多于1个特征的组
        result = []
        for group_id, features in groups.items():
            if len(features) > 1:
                result.append({
                    '特征组ID': group_id,
                    '特征数量': len(features),
                    '特征列表': ', '.join(features),
                })
        
        return pd.DataFrame(result)
    
    def partial_correlation(self, x: str, y: str, control: List[str]) -> Dict:
        """偏相关分析 (控制其他变量后的相关).
        
        :param x: 第一个特征
        :param y: 第二个特征
        :param control: 控制变量列表
        :return: 偏相关结果
        """
        try:
            from scipy import stats
            from sklearn.linear_model import LinearRegression
        except ImportError:
            return {'错误': '需要sklearn包'}
        
        # 准备数据
        cols = [x, y] + control
        data = self.numeric_df[cols].dropna()
        
        if len(data) < len(cols) + 10:
            return {'错误': '样本不足'}
        
        # 残差化
        lr_x = LinearRegression()
        lr_x.fit(data[control], data[x])
        residual_x = data[x] - lr_x.predict(data[control])
        
        lr_y = LinearRegression()
        lr_y.fit(data[control], data[y])
        residual_y = data[y] - lr_y.predict(data[control])
        
        # 计算残差相关
        corr, p_value = stats.pearsonr(residual_x, residual_y)
        
        return {
            '特征1': x,
            '特征2': y,
            '控制变量': ', '.join(control),
            '偏相关系数': round(corr, 4),
            'P值': round(p_value, 4),
            '显著性': '显著' if p_value < 0.05 else '不显著',
        }
    
    def generate_report(self, target_col: Optional[str] = None,
                       threshold: float = 0.8) -> Dict:
        """生成相关性分析报告.
        
        :param target_col: 目标变量列名
        :param threshold: 高相关性阈值
        :return: 包含所有分析结果的字典
        """
        report = {
            '相关系数矩阵(皮尔逊)': self.correlation_matrix(method='pearson'),
            '高相关性特征对': self.high_correlation_pairs(threshold=threshold, method='pearson'),
            'VIF分析': self.vif_analysis(),
            '冗余特征分析': self.redundancy_analysis(threshold=threshold),
        }
        
        if target_col:
            report['与目标变量相关性'] = self.correlation_with_target(target_col, method='pearson')
        
        return report
