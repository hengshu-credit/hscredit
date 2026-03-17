# -*- coding: utf-8 -*-
"""
自动特征分析报告.

用于三方数据评估或自有评分效果评估的自动化报告生成工具。
能够快速生成数据集情况、特征分布以及特征字段有效性分析报告。

参考 scorecardpipeline 实现优化而来。
"""

import traceback
import warnings
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openpyxl.worksheet.worksheet import Worksheet

from ..core.viz import (
    bin_plot,
    corr_plot,
    ks_plot,
    hist_plot,
    distribution_plot,
)
from ..utils import init_setting
from .feature_analyzer import feature_bin_stats
from .excel import ExcelWriter, dataframe2excel


def auto_feature_analysis_report(
    data: pd.DataFrame,
    features=None,
    target="target",
    overdue=None,
    dpds=None,
    date=None,
    data_summary_comment="",
    freq="M",
    excel_writer=None,
    sheet="分析报告",
    start_col=2,
    start_row=2,
    dropna=False,
    writer_params=None,
    bin_params=None,
    feature_map=None,
    corr=False,
    pictures=None,
    suffix="",
    output_dir="model_report"
):
    """
    自动特征分析报告.

    用于三方数据评估或自有评分效果评估。生成包含数据集概况、特征分箱统计、
    KS曲线、分布图等内容的Excel报告。

    :param data: 需要评估的数据集，需要包含目标变量
    :param features: 需要进行分析的特征名称，支持单个字符串或列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称，传入时会覆盖 target 参数
    :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本
    :param date: 日期列，用于时间维度分布分析
    :param freq: 日期统计粒度，默认按月 "M"
    :param data_summary_comment: 数据备注信息
    :param excel_writer: Excel文件路径或ExcelWriter对象
    :param sheet: 工作表名称
    :param start_col: 起始列
    :param start_row: 起始行
    :param dropna: 是否剔除缺失值
    :param writer_params: Excel写入器初始化参数
    :param bin_params: 分箱统计参数，支持 feature_bin_stats 的参数
    :param feature_map: 特征名称映射字典
    :param corr: 是否计算特征相关性
    :param pictures: 需要生成的图片列表，支持 ["ks", "hist", "bin"]
    :param suffix: 文件名后缀，避免同名文件被覆盖
    :param output_dir: 图片输出目录
    :return: (end_row, end_col) 报告结束位置

    **使用示例**

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hscredit.report.feature_report import auto_feature_analysis_report

    >>> # 准备数据
    >>> data = pd.DataFrame({
    ...     'feature1': np.random.randn(1000),
    ...     'feature2': np.random.randint(0, 10, 1000),
    ...     'target': np.random.randint(0, 2, 1000)
    ... })

    >>> # 生成报告
    >>> auto_feature_analysis_report(
    ...     data,
    ...     features=['feature1', 'feature2'],
    ...     target='target',
    ...     excel_writer='数据测试报告.xlsx',
    ...     pictures=['bin', 'ks', 'hist'],
    ...     corr=True
    ... )
    """
    if writer_params is None:
        writer_params = {}
    if bin_params is None:
        bin_params = {}
    if feature_map is None:
        feature_map = {}
    if pictures is None:
        pictures = ["bin", "ks", "hist"]

    init_setting()

    data = data.copy()
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(features, (list, tuple)):
        features = [features]

    if overdue and not isinstance(overdue, list):
        overdue = [overdue]

    if dpds and not isinstance(dpds, list):
        dpds = [dpds]

    if overdue:
        target = f"{overdue[0]} {dpds[0]}+"
        data[target] = (data[overdue[0]] > dpds[0]).astype(int)

    if isinstance(excel_writer, ExcelWriter):
        writer = excel_writer
    else:
        writer = ExcelWriter(**writer_params)

    worksheet = writer.get_sheet_by_name(sheet)

    if bin_params and "del_grey" in bin_params and bin_params.get("del_grey"):
        merge_columns = ["指标名称", "指标含义", "分箱"]
    else:
        merge_columns = ["指标名称", "指标含义", "分箱", "样本总数", "样本占比"]

    return_cols = []
    if bin_params:
        if "return_cols" in bin_params and bin_params.get("return_cols"):
            return_cols = bin_params.pop("return_cols")
            if not isinstance(return_cols, (list, np.ndarray)):
                return_cols = [return_cols]
            return_cols = list(set(return_cols) - set(merge_columns))
        else:
            return_cols = []

    max_columns_len = len(merge_columns) + len(return_cols) * len(overdue) * len(dpds) \
        if overdue and len(overdue) > 0 else len(merge_columns) + len(return_cols)

    end_row, end_col = writer.insert_value2sheet(
        worksheet, (start_row, start_col), value="数据有效性分析报告",
        style="header_middle", end_space=(start_row, start_col + max_columns_len - 1)
    )

    # 数据集概况
    if date is not None and date in data.columns:
        if data[date].dtype.name in ["str", "object"]:
            start_date = pd.to_datetime(data[date]).min().strftime("%Y-%m-%d")
            end_date = pd.to_datetime(data[date]).max().strftime("%Y-%m-%d")
        else:
            start_date = data[date].min().strftime("%Y-%m-%d")
            end_date = data[date].max().strftime("%Y-%m-%d")

        dataset_summary = pd.DataFrame(
            [[start_date, end_date, len(data), data[target].sum(), 
              data[target].sum() / len(data), data_summary_comment]],
            columns=["开始时间", "结束时间", "样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(
            dataset_summary, writer, worksheet, percent_cols=["坏客户占比"],
            start_row=end_row + 2, title="样本总体分布情况"
        )

        distribution = distribution_plot(
            data, date=date, freq=freq, target=target,
            save=os.path.join(output_dir, f"sample_time_distribution{suffix}.png"), result=True
        )
        end_row, end_col = writer.insert_value2sheet(
            worksheet, (end_row + 2, start_col), value="样本时间分布情况", style="header",
            end_space=(end_row + 2, start_col + len(distribution.columns) - 1)
        )
        end_row, end_col = writer.insert_pic2sheet(
            worksheet, os.path.join(output_dir, f"sample_time_distribution{suffix}.png"),
            (end_row + 1, start_col), figsize=(720, 370)
        )
        end_row, end_col = dataframe2excel(
            distribution, writer, worksheet,
            percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率"],
            condition_cols=["坏样本率"], start_row=end_row
        )
        end_row += 2
    else:
        dataset_summary = pd.DataFrame(
            [[len(data), data[target].sum(), data[target].sum() / len(data), data_summary_comment]],
            columns=["样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(
            dataset_summary, writer, worksheet, percent_cols=["坏客户占比"],
            start_row=end_row + 2, title="样本总体分布情况"
        )
        end_row += 2

    # 变量相关性分析
    if corr:
        temp = data[features].select_dtypes(include="number")
        corr_plot(
            temp, save=os.path.join(output_dir, f"auto_report_corr_plot{suffix}.png"),
            annot=True if len(temp.columns) <= 10 else False,
            fontsize=14 if len(temp.columns) <= 10 else 12
        )
        end_row, end_col = dataframe2excel(
            temp.corr(), writer, worksheet, color_cols=list(temp.columns),
            start_row=end_row, figures=[os.path.join(output_dir, f"auto_report_corr_plot{suffix}.png")],
            title="数值类变量相关性",
            figsize=(min(60 * len(temp.columns), 1080), min(55 * len(temp.columns), 950)),
            index=True, custom_cols=list(temp.columns), custom_format="0.00"
        )
        end_row += 2

    end_row, end_col = writer.insert_value2sheet(
        worksheet, (end_row, start_col), value="数值类特征 OR 评分效果评估",
        style="header_middle", end_space=(end_row, start_col + max_columns_len - 1)
    )

    features_iter = tqdm(features)
    for col in features_iter:
        features_iter.set_postfix(feature=feature_map.get(col, col))
        try:
            if overdue is None:
                temp = data[[col, target]]
            else:
                temp = data[list(set([col, target] + overdue))]

            if isinstance(dropna, bool) and dropna is True:
                temp = temp.dropna(subset=col).reset_index(drop=True)
            elif isinstance(dropna, (float, int, str)):
                temp = temp[temp[col] != dropna].reset_index(drop=True)

            # 确定实际的目标变量名（当有overdue时，target会被更新）
            actual_target = target
            if overdue:
                actual_target = f"{overdue[0]} {dpds[0]}+"

            score_table_train = feature_bin_stats(
                temp, col, overdue=overdue, dpds=dpds,
                desc=f"{feature_map.get(col, col)}", target=target, **bin_params
            )

            # 根据 score_table_train 的实际列数计算 end_space 宽度
            actual_columns_len = len(score_table_train.columns)

            if pictures and len(pictures) > 0:
                if "bin" in pictures:
                    if score_table_train.columns.nlevels > 1:
                        # 获取第一级列名，找到实际的目标列
                        level1_cols = score_table_train.columns.get_level_values(0).unique().tolist()
                        target_col = actual_target if actual_target in level1_cols else level1_cols[-1] if len(level1_cols) > 1 else level1_cols[0]
                        _ = score_table_train[["分箱详情", target_col]]
                        _.columns = [c[-1] for c in _.columns]
                    else:
                        _ = score_table_train.copy()

                    # 兼容列名：分箱标签 -> 分箱
                    if "分箱标签" in _.columns:
                        _.rename(columns={"分箱标签": "分箱"}, inplace=True)

                    bin_plot(
                        _, desc=f"{feature_map.get(col, col)}", figsize=(10, 5),
                        anchor=0.935, save=os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png")
                    )

                if temp[col].dtypes.name not in ['object', 'str', 'category']:
                    if "ks" in pictures:
                        _ = temp.dropna().reset_index(drop=True)
                        has_ks = len(_) > 0 and _[col].nunique() > 1 and _[actual_target].nunique() > 1
                        if has_ks:
                            ks_plot(
                                _[col], _[actual_target], figsize=(10, 5),
                                title=f"{feature_map.get(col, col)}",
                                save=os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png")
                            )
                    if "hist" in pictures:
                        _ = temp.dropna().reset_index(drop=True)
                        if len(_) > 0:
                            hist_plot(
                                _[col], y_true=_[actual_target], figsize=(10, 6),
                                desc=f"{feature_map.get(col, col)} 好客户 VS 坏客户",
                                bins=30, anchor=1.11, fontsize=14,
                                labels={0: "好客户", 1: "坏客户"},
                                save=os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png")
                            )

            if (len(temp) < len(data)) and (isinstance(dropna, bool) and dropna is True) or \
               isinstance(dropna, (float, int, str)):
                end_row, end_col = writer.insert_value2sheet(
                    worksheet, (end_row + 2, start_col),
                    value=f"数据字段: {feature_map.get(col, col)} (缺失率: {round((1 - len(temp) / len(data)) * 100, 2)}%)",
                    style="header", end_space=(end_row + 2, start_col + actual_columns_len - 1)
                )
            else:
                end_row, end_col = writer.insert_value2sheet(
                    worksheet, (end_row + 2, start_col),
                    value=f"数据字段: {feature_map.get(col, col)}",
                    style="header", end_space=(end_row + 2, start_col + actual_columns_len - 1)
                )

            if pictures and len(pictures) > 0:
                ks_row = end_row + 1
                if "bin" in pictures:
                    end_row, end_col = writer.insert_pic2sheet(
                        worksheet, os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png"),
                        (ks_row, start_col), figsize=(600, 350)
                    )
                if temp[col].dtypes.name not in ['object', 'str', 'category'] and \
                   temp[col].isnull().sum() != len(temp):
                    if "ks" in pictures and has_ks:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png"),
                            (ks_row, end_col - 1), figsize=(600, 350)
                        )
                    if "hist" in pictures:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png"),
                            (ks_row, end_col - 1), figsize=(600, 350)
                        )

            if return_cols:
                if score_table_train.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                    merge_columns = [("分箱详情", c) for c in merge_columns]

                end_row, end_col = dataframe2excel(
                    score_table_train[
                        merge_columns + [
                            c for c in score_table_train.columns 
                            if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                            or (not isinstance(c, (tuple, list)) and c in return_cols)
                            or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                        ]
                    ], writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=end_row
                )
            else:
                end_row, end_col = dataframe2excel(
                    score_table_train, writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=end_row
                )
        except Exception as e:
            print(f"数据字段 {col} 分析时发生异常，请排查数据中是否存在异常:\n{traceback.format_exc()}")

    if not isinstance(excel_writer, ExcelWriter) and not isinstance(sheet, Worksheet):
        writer.save(excel_writer)

    return end_row, end_col
