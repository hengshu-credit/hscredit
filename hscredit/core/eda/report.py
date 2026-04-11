"""综合报告模块.

提供EDA综合报告生成和导出功能.
整合所有模块的分析结果.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime

from .utils import validate_dataframe
from .overview import data_info, missing_analysis, feature_summary, data_quality_report
from .target import target_distribution, bad_rate_overall, bad_rate_trend
from .relationship import batch_iv_analysis
from .correlation import high_correlation_pairs

from ...excel import ExcelWriter, dataframe2excel


def eda_summary(df: pd.DataFrame,
               target: str = None,
               features: List[str] = None,
               date_col: str = None) -> Dict[str, pd.DataFrame]:
    """EDA分析摘要.
    
    快速生成数据集的关键分析结果
    
    :param df: 输入数据
    :param target: 目标变量名（可选）
    :param features: 特征列表（可选）
    :param date_col: 日期列名（可选）
    :return: EDA摘要字典
    
    Example:
        >>> summary = eda_summary(df, target='fpd15', date_col='apply_date')
        >>> for key, value in summary.items():
        ...     print(f"\n=== {key} ===")
        ...     print(value)
    """
    validate_dataframe(df)
    
    if features is None:
        features = [c for c in df.columns if c != target]
    
    summary = {}
    
    # 1. 数据基础信息
    summary['数据基础信息'] = data_info(df)
    
    # 2. 缺失值分析
    summary['缺失值分析'] = missing_analysis(df, threshold=0.0)
    
    # 3. 特征描述统计
    summary['特征描述统计'] = feature_summary(df, features)
    
    # 4. 数据质量问题
    summary['数据质量问题'] = data_quality_report(df)
    
    # 5. 目标变量分析
    if target and target in df.columns:
        summary['目标变量分布'] = target_distribution(df, target)
        summary['整体逾期率'] = pd.DataFrame([bad_rate_overall(df, target)])
        
        # 时间趋势
        if date_col and date_col in df.columns:
            try:
                summary['逾期率趋势'] = bad_rate_trend(df, target, date_col)
            except:
                pass
    
    return summary


def generate_report(df: pd.DataFrame,
                   target: str = None,
                   features: List[str] = None,
                   date_col: str = None,
                   config: Dict = None) -> Dict[str, pd.DataFrame]:
    """生成完整EDA报告.
    
    :param df: 输入数据
    :param target: 目标变量名
    :param features: 特征列表
    :param date_col: 日期列名
    :param config: 配置参数
    :return: 完整报告字典
    
    Example:
        >>> report = generate_report(df, target='fpd15', date_col='apply_date',
        ...                          config={'iv_threshold': 0.02})
        >>> export_report_to_excel(report, 'eda_report.xlsx')
    """
    validate_dataframe(df)
    
    if features is None:
        features = [c for c in df.columns if c != target and c != date_col]
    
    if config is None:
        config = {}
    
    iv_threshold = config.get('iv_threshold', 0.02)
    psi_threshold = config.get('psi_threshold', 0.1)
    corr_threshold = config.get('corr_threshold', 0.8)
    
    report = {}
    
    # 1. 数据概览
    report['1.数据基础信息'] = data_info(df)
    report['2.缺失值分析'] = missing_analysis(df)
    report['3.特征描述统计'] = feature_summary(df, features)
    report['4.数据质量问题'] = data_quality_report(df)
    
    # 2. 目标变量分析
    if target and target in df.columns:
        report['5.目标变量分布'] = target_distribution(df, target)
        report['6.整体逾期率'] = pd.DataFrame([bad_rate_overall(df, target)])
        
        if date_col and date_col in df.columns:
            try:
                report['7.逾期率趋势'] = bad_rate_trend(df, target, date_col)
            except:
                pass
        
        # 3. IV分析
        try:
            iv_result = batch_iv_analysis(df, features, target)
            report['8.IV分析'] = iv_result[iv_result['IV值'] >= iv_threshold]
        except:
            pass
    
    # 4. 相关性分析
    try:
        corr_pairs = high_correlation_pairs(df, features, threshold=corr_threshold)
        if '信息' not in corr_pairs.columns:
            report['9.高相关性特征对'] = corr_pairs
    except:
        pass
    
    return report


def export_report_to_excel(report: Dict[str, pd.DataFrame],
                          filepath: str,
                          sheet_name_mapping: Dict[str, str] = None,
                          theme_color: str = '2639E9',
                          auto_width: bool = True) -> None:
    """导出报告到Excel.
    
    使用 hscredit 的 ExcelWriter 生成专业格式的 Excel 报告。
    
    :param report: 报告字典
    :param filepath: 导出文件路径
    :param sheet_name_mapping: 工作表名称映射
    :param theme_color: 主题颜色，默认 '2639E9'（蓝色）
    :param auto_width: 是否自动调整列宽，默认 True
    
    Example:
        >>> export_report_to_excel(report, 'eda_report.xlsx')
        >>> export_report_to_excel(report, 'eda_report.xlsx', theme_color='00A651')
    """
    # 处理sheet名称（Excel限制：最多31个字符，不能包含特殊字符）
    def clean_sheet_name(name: str) -> str:
        # 移除特殊字符
        name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
        name = name.replace('?', '').replace('*', '').replace('[', '').replace(']', '')
        # 截取前31个字符
        return name[:31]
    
    with ExcelWriter(theme_color=theme_color) as writer:
        for section_name, df in report.items():
            if df is None or df.empty:
                continue
            
            # 处理工作表名称
            if sheet_name_mapping and section_name in sheet_name_mapping:
                sheet_name = clean_sheet_name(sheet_name_mapping[section_name])
            else:
                sheet_name = clean_sheet_name(section_name)
            
            # 获取或创建工作表
            worksheet = writer.get_sheet_by_name(sheet_name)
            
            # 写入标题
            writer.insert_value2sheet(
                worksheet, 'B2', 
                value=section_name, 
                style='header'
            )
            
            # 写入DataFrame
            writer.insert_df2sheet(
                worksheet, df, 'B4',
                header=True,
                index=False,
                auto_width=auto_width,
                fill=False
            )
        
        # 保存文件
        writer.save(filepath)
    
    print(f"报告已导出至: {filepath}")


def generate_html_report(report: Dict[str, pd.DataFrame],
                        filepath: str,
                        title: str = "EDA分析报告") -> None:
    """生成HTML报告.
    
    :param report: 报告字典
    :param filepath: 导出文件路径
    :param title: 报告标题
    
    Example:
        >>> generate_html_report(report, 'eda_report.html', title='信贷数据EDA报告')
    """
    html_parts = []
    
    # HTML头部
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #007bff; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .timestamp {{ color: #666; font-size: 12px; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """)
    
    # 各章节
    for section_name, df in report.items():
        html_parts.append(f"<h2>{section_name}</h2>")
        if not df.empty:
            html_parts.append(df.to_html(index=False, classes='data-table'))
        else:
            html_parts.append("<p>无数据</p>")
    
    # HTML尾部
    html_parts.append(f"""
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """)
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"HTML报告已导出至: {filepath}")
