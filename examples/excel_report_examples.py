"""
Excel报告生成示例

演示如何使用hscredit.report.excel模块生成专业的Excel报告。
"""

import sys
from pathlib import Path

# 添加项目路径到sys.path（使用绝对路径）
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from hscredit.report.excel import ExcelWriter, dataframe2excel


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("基本使用示例")
    print("="*60)
    
    # 创建写入器
    writer = ExcelWriter(theme_color='3f1dba')
    
    # 获取工作表
    worksheet = writer.get_sheet_by_name("模型报告")
    
    # 插入标题
    writer.insert_value2sheet(worksheet, "B2", value="模型报告", style="header")
    
    # 插入副标题
    writer.insert_value2sheet(
        worksheet, "B3",
        value="这是一个评分卡模型报告示例",
        style="content",
        auto_width=True
    )
    
    # 创建示例数据
    df = pd.DataFrame({
        '特征名称': ['age', 'income', 'education', 'marriage'],
        'IV值': [0.15, 0.23, 0.08, 0.12],
        'KS值': [0.32, 0.41, 0.25, 0.28],
        '缺失率': [0.02, 0.05, 0.01, 0.03]
    })
    
    # 插入DataFrame
    writer.insert_df2sheet(worksheet, df, "B5")
    
    # 保存文件
    writer.save("examples/output/basic_report.xlsx")
    print("✓ 基本报告已保存到: examples/output/basic_report.xlsx")


def example_dataframe2excel():
    """使用dataframe2excel便捷函数"""
    print("\n" + "="*60)
    print("dataframe2excel便捷函数示例")
    print("="*60)
    
    # 创建示例数据
    df = pd.DataFrame({
        'feature': ['年龄', '收入', '学历', '婚姻状况', '职业'],
        'iv': [0.15, 0.23, 0.08, 0.12, 0.18],
        'ks': [0.32, 0.41, 0.25, 0.28, 0.35],
        'auc': [0.68, 0.72, 0.63, 0.65, 0.70],
        'missing_rate': [0.02, 0.05, 0.01, 0.03, 0.04]
    })
    
    # 快速写入Excel
    dataframe2excel(
        df,
        "examples/output/feature_analysis.xlsx",
        sheet_name="特征分析",
        title="特征重要性统计表",
        percent_cols=['missing_rate'],      # 百分比格式
        condition_cols=['iv', 'ks', 'auc'], # 条件格式（数据条）
        auto_width=True
    )
    print("✓ 特征分析报告已保存到: examples/output/feature_analysis.xlsx")


def example_advanced_features():
    """高级功能示例"""
    print("\n" + "="*60)
    print("高级功能示例")
    print("="*60)
    
    # 创建写入器
    writer = ExcelWriter(theme_color='2639E9')
    worksheet = writer.get_sheet_by_name("高级报告")
    
    # 1. 插入合并单元格的标题
    writer.insert_value2sheet(
        worksheet, "B2",
        value="金融风控模型报告",
        style="header",
        end_space="F2"
    )
    
    # 2. 插入带超链接的内容
    writer.insert_value2sheet(
        worksheet, "B4",
        value="点击查看特征分析",
        style="content",
        auto_width=True
    )
    writer.insert_hyperlink2sheet(worksheet, "B4", target_space="B10")
    
    # 3. 创建多层索引数据
    arrays = [
        ['训练集', '训练集', '训练集', '测试集', '测试集', '测试集'],
        ['样本数', '坏账率', 'KS值', '样本数', '坏账率', 'KS值']
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=['数据集', '指标'])
    
    df_multi = pd.DataFrame(
        np.random.rand(5, 6),
        index=['年龄', '收入', '学历', '婚姻', '职业'],
        columns=columns
    )
    
    # 插入多层索引数据
    writer.insert_df2sheet(
        worksheet, df_multi,
        "B10",
        fill=True,
        index=True
    )
    
    # 4. 创建分组数据
    df_group = pd.DataFrame({
        '特征': ['年龄', '年龄', '年龄', '收入', '收入', '收入'],
        '分箱': ['[18,30)', '[30,50)', '[50,100)', '[0,5000)', '[5000,10000)', '[10000,inf)'],
        '样本数': [1000, 2000, 500, 1500, 1500, 500],
        '坏账率': [0.15, 0.08, 0.12, 0.10, 0.05, 0.03],
        'WOE': [0.45, -0.32, 0.18, 0.12, -0.67, -1.05]
    })
    
    # 插入分组数据（自动合并相同特征）
    writer.insert_df2sheet(
        worksheet, df_group,
        "B18",
        merge_column='特征',
        merge=True,
        fill=True
    )
    
    # 5. 设置冻结窗格
    writer.set_freeze_panes(worksheet, "B3")
    
    # 保存
    writer.save("examples/output/advanced_report.xlsx")
    print("✓ 高级报告已保存到: examples/output/advanced_report.xlsx")


def example_multiple_sheets():
    """多工作表示例"""
    print("\n" + "="*60)
    print("多工作表示例")
    print("="*60)
    
    writer = ExcelWriter(theme_color='2E86AB')
    
    # Sheet 1: 模型概述
    ws1 = writer.get_sheet_by_name("模型概述")
    writer.insert_value2sheet(ws1, "B2", value="模型概述", style="header")
    
    summary_df = pd.DataFrame({
        '指标': ['样本数', '特征数', '坏账率', 'KS值', 'AUC值'],
        '训练集': [10000, 50, 0.12, 0.45, 0.78],
        '测试集': [3000, 50, 0.11, 0.43, 0.76]
    })
    writer.insert_df2sheet(ws1, summary_df, "B4")
    
    # Sheet 2: 特征重要性
    ws2 = writer.get_sheet_by_name("特征重要性")
    writer.insert_value2sheet(ws2, "B2", value="特征重要性", style="header")
    
    feature_df = pd.DataFrame({
        '特征': [f'feature_{i}' for i in range(1, 11)],
        'IV值': np.random.uniform(0.01, 0.3, 10),
        '重要性': np.random.uniform(0.1, 1.0, 10)
    })
    dataframe2excel(
        feature_df, writer,
        sheet_name=ws2,
        start_row=4,
        condition_cols=['IV值', '重要性']
    )
    
    # Sheet 3: 模型性能
    ws3 = writer.get_sheet_by_name("模型性能")
    writer.insert_value2sheet(ws3, "B2", value="模型性能", style="header")
    
    # 插入超链接到其他sheet
    writer.insert_value2sheet(ws3, "B4", value="返回模型概述", style="content")
    writer.insert_hyperlink2sheet(ws3, "B4", sheet="模型概述", target_space="B2")
    
    writer.insert_value2sheet(ws3, "B5", value="查看特征重要性", style="content")
    writer.insert_hyperlink2sheet(ws3, "B5", sheet="特征重要性", target_space="B2")
    
    # 调整sheet顺序
    writer.move_sheet("模型性能", index=2)
    
    writer.save("examples/output/multi_sheet_report.xlsx")
    print("✓ 多工作表报告已保存到: examples/output/multi_sheet_report.xlsx")


def example_styled_report():
    """样式定制示例"""
    print("\n" + "="*60)
    print("样式定制示例")
    print("="*60)
    
    # 自定义主题色
    writer = ExcelWriter(
        theme_color='E63946',  # 红色主题
        fontsize=11,
        font='微软雅黑',
        opacity=0.7
    )
    
    worksheet = writer.get_sheet_by_name("样式报告")
    
    # 插入不同样式的标题
    writer.insert_value2sheet(
        worksheet, "B2",
        value="自定义样式报告",
        style="header"
    )
    
    # 创建数据
    df = pd.DataFrame({
        '指标': ['准确率', '精确率', '召回率', 'F1分数', 'AUC'],
        '训练集': [0.85, 0.82, 0.88, 0.85, 0.89],
        '测试集': [0.83, 0.80, 0.86, 0.83, 0.87],
        '验证集': [0.84, 0.81, 0.87, 0.84, 0.88]
    })
    
    # 使用颜色填充
    writer.insert_df2sheet(
        worksheet, df,
        "B4",
        fill=True,
        auto_width=True
    )
    
    # 添加条件格式
    writer.add_conditional_formatting(worksheet, "C6", "E8")
    
    # 设置列宽
    writer.set_column_width(worksheet, 'B', 15)
    writer.set_column_width(worksheet, 'C', 12)
    
    writer.save("examples/output/styled_report.xlsx")
    print("✓ 样式定制报告已保存到: examples/output/styled_report.xlsx")


def example_append_mode():
    """追加模式示例"""
    print("\n" + "="*60)
    print("追加模式示例")
    print("="*60)
    
    # 第一次写入
    writer1 = ExcelWriter(theme_color='2A9D8F')
    ws1 = writer1.get_sheet_by_name("Sheet1")
    writer1.insert_value2sheet(ws1, "B2", value="第一次写入", style="header")
    writer1.save("examples/output/append_report.xlsx")
    print("✓ 第一次写入完成")
    
    # 追加模式写入新sheet
    writer2 = ExcelWriter(mode='append')
    ws2 = writer2.get_sheet_by_name("Sheet2")
    writer2.insert_value2sheet(ws2, "B2", value="第二次追加", style="header")
    writer2.save("examples/output/append_report.xlsx")
    print("✓ 追加写入完成")
    print("✓ 追加模式报告已保存到: examples/output/append_report.xlsx")


if __name__ == "__main__":
    import os
    
    # 创建输出目录
    os.makedirs("examples/output", exist_ok=True)
    
    print("\n" + "="*60)
    print("hscredit Excel报告生成示例")
    print("="*60)
    
    # 运行所有示例
    example_basic_usage()
    example_dataframe2excel()
    example_advanced_features()
    example_multiple_sheets()
    example_styled_report()
    example_append_mode()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - examples/output/basic_report.xlsx")
    print("  - examples/output/feature_analysis.xlsx")
    print("  - examples/output/advanced_report.xlsx")
    print("  - examples/output/multi_sheet_report.xlsx")
    print("  - examples/output/styled_report.xlsx")
    print("  - examples/output/append_report.xlsx")
