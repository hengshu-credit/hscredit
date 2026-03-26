# -*- coding: utf-8 -*-
"""
EDA 报告生成模块 - 整合所有分析结果.

功能:
- 一键生成完整EDA报告
- Excel/JSON格式导出
- 可视化图表生成
- 数据质量评估报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings
import os

from .data_overview import DataOverview
from .target_analysis import TargetAnalysis
from .feature_analysis import FeatureAnalysis
from .feature_label_relationship import FeatureLabelRelationship
from .stability_analysis import StabilityAnalysis
from .correlation_analysis import CorrelationAnalysis


class EDAReport:
    """EDA报告生成类.
    
    整合所有EDA分析模块，生成完整的数据探索报告。
    
    **参数**
    
    :param df: 输入数据 DataFrame
    :param target_col: 目标变量列名，可选
    :param date_col: 日期列名，用于时间分析，可选
    
    **示例**
    
        >>> from hscredit.core.eda import EDAReport
        >>> eda = EDAReport(df, target_col='FPD15', date_col='loan_date')
        >>> # 生成完整报告
        >>> report = eda.generate_full_report()
        >>> # 导出到Excel
        >>> eda.export_to_excel('eda_report.xlsx')
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                 date_col: Optional[str] = None):
        """初始化EDA报告生成器."""
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        # 初始化各个分析器
        self.data_overview = DataOverview(df, target_col)
        self.feature_analysis = FeatureAnalysis(df, date_col)
        self.correlation_analysis = CorrelationAnalysis(df)
        
        if target_col:
            self.target_analysis = TargetAnalysis(df, target_col, date_col)
            self.feature_label_rel = FeatureLabelRelationship(df, target_col)
        
        if date_col:
            self.stability_analysis = StabilityAnalysis(df, date_col)
    
    def generate_full_report(self, feature_cols: Optional[List[str]] = None) -> Dict:
        """生成完整EDA报告.
        
        :param feature_cols: 需要详细分析的特征列表，None则分析所有
        :return: 完整报告字典
        """
        if feature_cols is None:
            feature_cols = [c for c in self.df.columns 
                          if c != self.target_col and c != self.date_col]
        
        print("="*60)
        print("开始生成EDA报告...")
        print("="*60)
        
        report = {}
        
        # 1. 数据概览
        print("[1/6] 数据概览分析...")
        report['数据概览'] = self.data_overview.generate_report()
        
        # 2. 目标变量分析
        if self.target_col:
            print("[2/6] 目标变量分析...")
            report['目标变量分析'] = self.target_analysis.generate_report()
        
        # 3. 特征分析
        print("[3/6] 特征分析...")
        report['特征分析'] = self.feature_analysis.generate_report(feature_cols[:20])  # 限制前20个
        
        # 4. 特征与标签关系
        if self.target_col:
            print("[4/6] 特征与标签关系分析...")
            report['特征与标签关系'] = self.feature_label_rel.generate_report(feature_cols[:20])
        
        # 5. 稳定性分析 (需要date_col)
        if self.date_col and hasattr(self, 'stability_analysis'):
            print("[5/6] 稳定性分析...")
            periods = self.stability_analysis._get_all_periods()
            if len(periods) >= 2:
                report['稳定性分析'] = self.stability_analysis.generate_report(
                    feature_cols[:10], periods[0])
        
        # 6. 相关性分析
        print("[6/6] 相关性分析...")
        report['相关性分析'] = self.correlation_analysis.generate_report(
            self.target_col, threshold=0.8)
        
        print("="*60)
        print("EDA报告生成完成!")
        print("="*60)
        
        return report
    
    def export_to_excel(self, output_path: str, feature_cols: Optional[List[str]] = None):
        """导出EDA报告到Excel.
        
        :param output_path: 输出文件路径
        :param feature_cols: 需要分析的特征列表
        """
        if feature_cols is None:
            feature_cols = [c for c in self.df.columns 
                          if c != self.target_col and c != self.date_col]
        
        report = self.generate_full_report(feature_cols)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 数据概览
            if '数据概览' in report:
                # 基础信息
                basic_info = pd.DataFrame([report['数据概览']['基础信息']])
                basic_info.to_excel(writer, sheet_name='1-基础信息', index=False)
                
                # 缺失值分析
                report['数据概览']['缺失值分析'].to_excel(
                    writer, sheet_name='2-缺失值分析')
                
                # 数值特征统计
                if len(report['数据概览']['数值特征统计']) > 0:
                    report['数据概览']['数值特征统计'].to_excel(
                        writer, sheet_name='3-数值特征统计')
                
                # 数据质量评分
                quality_score = pd.DataFrame([report['数据概览']['数据质量评分']])
                quality_score.to_excel(writer, sheet_name='4-数据质量评分', index=False)
            
            # 2. 目标变量分析
            if '目标变量分析' in report and self.target_col:
                # 基础统计
                target_stats = pd.DataFrame([report['目标变量分析']['基础统计']])
                target_stats.to_excel(writer, sheet_name='5-目标变量统计', index=False)
                
                # 时间趋势
                if '时间趋势(按月)' in report['目标变量分析']:
                    report['目标变量分析']['时间趋势(按月)'].to_excel(
                        writer, sheet_name='6-逾期率趋势', index=False)
            
            # 3. IV分析
            if '特征与标签关系' in report and 'IV分析' in report['特征与标签关系']:
                report['特征与标签关系']['IV分析'].to_excel(
                    writer, sheet_name='7-IV分析', index=False)
                
                # 特征重要性排序
                if '特征重要性排序' in report['特征与标签关系']:
                    report['特征与标签关系']['特征重要性排序'].to_excel(
                        writer, sheet_name='8-特征重要性', index=False)
            
            # 4. 相关性分析
            if '相关性分析' in report:
                # 高相关性特征对
                if len(report['相关性分析']['高相关性特征对']) > 0:
                    report['相关性分析']['高相关性特征对'].to_excel(
                        writer, sheet_name='9-高相关性特征', index=False)
                
                # VIF分析
                if len(report['相关性分析']['VIF分析']) > 0:
                    report['相关性分析']['VIF分析'].to_excel(
                        writer, sheet_name='10-VIF分析', index=False)
                
                # 与目标变量相关性
                if '与目标变量相关性' in report['相关性分析']:
                    report['相关性分析']['与目标变量相关性'].to_excel(
                        writer, sheet_name='11-与目标相关性', index=False)
            
            # 5. 稳定性分析
            if '稳定性分析' in report:
                if '稳定性评级' in report['稳定性分析']:
                    report['稳定性分析']['稳定性评级'].to_excel(
                        writer, sheet_name='12-稳定性评级', index=False)
        
        print(f"报告已导出至: {output_path}")
    
    def get_data_quality_summary(self) -> Dict:
        """获取数据质量摘要.
        
        :return: 数据质量摘要字典
        """
        basic_info = self.data_overview.basic_info()
        quality_score = self.data_overview.data_quality_score()
        missing_analysis = self.data_overview.missing_analysis()
        
        # 找出高缺失率特征
        high_missing = missing_analysis[missing_analysis['缺失率(%)'] > 50]
        
        # 找出常数特征
        constant_features = self.data_overview.constant_features()
        
        return {
            '样本总数': basic_info.get('样本总数', 0),
            '特征总数': basic_info.get('特征总数', 0),
            '缺失率超50%特征数': len(high_missing),
            '缺失率超50%特征': high_missing.index.tolist() if len(high_missing) > 0 else [],
            '常数特征数': len(constant_features),
            '常数特征': constant_features['特征名'].tolist() if len(constant_features) > 0 else [],
            '数据质量评级': quality_score.get('质量评级', '未知'),
            '综合质量评分': quality_score.get('综合数据质量评分', 0),
        }
    
    def get_feature_recommendations(self, feature_cols: Optional[List[str]] = None) -> Dict:
        """获取特征处理建议.
        
        :param feature_cols: 特征列表
        :return: 处理建议字典
        """
        if feature_cols is None:
            feature_cols = [c for c in self.df.columns 
                          if c != self.target_col and c != self.date_col]
        
        recommendations = {
            '建议删除': [],
            '需要处理缺失值': [],
            '高相关性特征组': [],
            '不稳定特征': [],
            '强预测特征': [],
        }
        
        # 1. 缺失率过高的特征
        missing_analysis = self.data_overview.missing_analysis()
        high_missing = missing_analysis[missing_analysis['缺失率(%)'] > 70].index.tolist()
        recommendations['建议删除'].extend(high_missing)
        
        # 2. 常数特征
        constant_features = self.data_overview.constant_features()
        recommendations['建议删除'].extend(constant_features['特征名'].tolist() if len(constant_features) > 0 else [])
        
        # 3. 需要处理缺失值的特征
        need_impute = missing_analysis[
            (missing_analysis['缺失率(%)'] > 0) & 
            (missing_analysis['缺失率(%)'] <= 70)
        ].index.tolist()
        recommendations['需要处理缺失值'] = need_impute
        
        # 4. 高相关性特征
        high_corr = self.correlation_analysis.redundancy_analysis(threshold=0.9)
        if len(high_corr) > 0:
            recommendations['高相关性特征组'] = high_corr.to_dict('records')
        
        # 5. IV分析 (如果存在目标变量)
        if self.target_col and hasattr(self, 'feature_label_rel'):
            iv_results = self.feature_label_rel.batch_iv_analysis(feature_cols[:50])
            
            # 强预测特征 (IV > 0.1)
            strong_predictors = iv_results[iv_results['IV值'] > 0.1]['特征名'].tolist()
            recommendations['强预测特征'] = strong_predictors[:10]
            
            # 无预测能力特征 (IV < 0.02)
            weak_predictors = iv_results[iv_results['IV值'] < 0.02]['特征名'].tolist()
            recommendations['建议删除'].extend(weak_predictors[:10])
        
        # 去重
        recommendations['建议删除'] = list(set(recommendations['建议删除']))
        
        return recommendations
    
    def print_summary(self):
        """打印EDA摘要."""
        print("\n" + "="*60)
        print("EDA 数据探索摘要")
        print("="*60)
        
        # 数据质量
        quality = self.get_data_quality_summary()
        print("\n【数据质量】")
        print(f"  样本总数: {quality['样本总数']:,}")
        print(f"  特征总数: {quality['特征总数']}")
        print(f"  数据质量评级: {quality['数据质量评级']}")
        print(f"  综合质量评分: {quality['综合质量评分']}")
        
        if quality['缺失率超50%特征数'] > 0:
            print(f"  ⚠️ 缺失率超50%的特征: {quality['缺失率超50%特征数']}个")
        
        if quality['常数特征数'] > 0:
            print(f"  ⚠️ 常数特征: {quality['常数特征数']}个")
        
        # 目标变量
        if self.target_col:
            target_stats = self.target_analysis.basic_stats()
            print("\n【目标变量】")
            print(f"  逾期率: {target_stats.get('逾期率(%)', 0)}%")
            print(f"  样本不平衡比例: {target_stats.get('样本不平衡比例', 0)}:1")
        
        # 建议
        recommendations = self.get_feature_recommendations()
        print("\n【处理建议】")
        print(f"  建议删除特征: {len(recommendations['建议删除'])}个")
        print(f"  需要处理缺失值: {len(recommendations['需要处理缺失值'])}个")
        print(f"  强预测特征: {len(recommendations['强预测特征'])}个")
        
        print("\n" + "="*60)
