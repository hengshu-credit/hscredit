"""综合报告模块.

提供EDA综合报告生成和导出功能.
整合所有模块的分析结果.
"""

from hscredit.utils.arguments import analysis_inputs

import logging
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

logger = logging.getLogger(__name__)


def _execute_eda_report(df, target, features, date, config, n_jobs, parallel_backend, parallel_config, mode, full):
    # 延迟导入，避免 core.eda 初始化时反向触发 report 包聚合。
    from ...report.result import ReportResult

    validate_dataframe(df)
    if features is None:
        features = [c for c in df.columns if c != target and (not full or c != date)]
    parallel = dict(n_jobs=n_jobs, parallel_backend=parallel_backend, parallel_config=parallel_config)
    result = ReportResult(
        mode=mode, metadata={"目标": target, "日期字段": date, "特征": list(features), "输入行数": len(df)}
    )
    names = ["数据基础信息", "缺失值分析", "特征描述统计", "数据质量问题", "目标变量分布", "整体逾期率", "逾期率趋势"]
    if full:
        names += ["IV分析", "高相关性特征对"]
    keys = {name: f"{i + 1}.{name}" if full else name for i, name in enumerate(names)}
    for key in keys.values():
        result.plan(key, input_rows=len(df))

    def iv_table():
        table = batch_iv_analysis(df, features, target, **parallel)
        return table[table["IV值"] >= config.get("iv_threshold", 0.02)]

    operations = {
        "数据基础信息": lambda: data_info(df),
        "缺失值分析": lambda: missing_analysis(df, threshold=0.0),
        "特征描述统计": lambda: feature_summary(df, features, **parallel),
        "数据质量问题": lambda: data_quality_report(df),
        "目标变量分布": lambda: target_distribution(df, target),
        "整体逾期率": lambda: pd.DataFrame([bad_rate_overall(df, target)]),
        "逾期率趋势": lambda: bad_rate_trend(df, date=date, target=target),
        "IV分析": iv_table,
        "高相关性特征对": lambda: high_correlation_pairs(df, features, threshold=config.get("corr_threshold", 0.8)),
    }
    for name in names:
        if name in {"目标变量分布", "整体逾期率", "逾期率趋势", "IV分析"} and target is None:
            result.skip(keys[name], "未指定目标变量")
        elif name == "逾期率趋势" and date is None:
            result.skip(keys[name], "未指定日期字段")
        else:
            result.run(keys[name], operations[name])
    return result


@analysis_inputs("df", roles=("date",))
def eda_summary(
    df: pd.DataFrame,
    target: str = None,
    features: List[str] = None,
    date: str = None,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
    *,
    mode: str = "best_effort",
    return_result: bool = False,
) -> Dict[str, pd.DataFrame]:
    """EDA分析摘要.

    快速生成数据集的关键分析结果。

    :param df: 输入数据

    :param target: 目标变量名（可选）

    :param features: 特征列表（可选）

    :param date: 日期列名（可选）

    :param mode: strict 遇必需章节失败抛异常；best_effort 记录原因并继续

    :param return_result: True 返回兼容字典的 ReportResult，含完整章节状态
    :return: EDA摘要字典；return_result=True 时返回兼容字典的 ReportResult。

    **参考样例**

    >>> summary = eda_summary(df, target='fpd15', date='apply_date')
    >>> for key, value in summary.items():
    ...     print(f"=== {key} ===")
    ...     print(value)

    **参数**

    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。
    """
    result = _execute_eda_report(df, target, features, date, {}, n_jobs, parallel_backend, parallel_config, mode, False)
    return result if return_result else dict(result)


@analysis_inputs("df", roles=("date",))
def generate_report(
    df: pd.DataFrame,
    target: str = None,
    features: List[str] = None,
    date: str = None,
    config: Dict = None,
    n_jobs=-1,
    parallel_backend=None,
    parallel_config=None,
    *,
    mode: str = "best_effort",
    return_result: bool = False,
) -> Dict[str, pd.DataFrame]:
    """生成完整EDA报告.

    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :param df: 输入数据
    :type df: pd.DataFrame
    :param target: 默认值为 ``None`` 。目标变量名
    :type target: str
    :param features: 默认值为 ``None`` 。特征列表
    :type features: List[str]
    :param date: 默认值为 ``None`` 。日期列名
    :type date: str
    :param config: 默认值为 ``None`` 。配置参数
    :type config: Dict
    :param mode: 默认值为 ``'best_effort'`` 。strict 拒绝必需章节失败；best_effort 保留成功结果并记录原因
    :type mode: str
    :param return_result: 默认值为 ``False`` 。True 返回含章节状态和发布清单的 ReportResult
    :type return_result: bool
    :return: 完整报告字典

    **参考样例**

    >>> report = generate_report(df, target='fpd15', date='apply_date',
    ...                          config={'iv_threshold': 0.02})
    >>> export_report_to_excel(report, 'eda_report.xlsx')
    """
    result = _execute_eda_report(
        df, target, features, date, config or {}, n_jobs, parallel_backend, parallel_config, mode, True
    )
    return result if return_result else dict(result)


def export_report_to_excel(
    report: Dict[str, pd.DataFrame],
    filepath: str,
    sheet_name_mapping: Dict[str, str] = None,
    theme_color: str = "2639E9",
    auto_width: bool = True,
) -> None:
    """导出报告到Excel.

    使用 hscredit 的 ExcelWriter 生成专业格式的 Excel 报告。

    :param report: 报告字典
    :type report: Dict[str, pd.DataFrame]
    :param filepath: 导出文件路径
    :type filepath: str
    :param sheet_name_mapping: 默认值为 ``None`` 。工作表名称映射
    :type sheet_name_mapping: Dict[str, str]
    :param theme_color: 主题颜色，默认 '2639E9'（蓝色）
    :type theme_color: str
    :param auto_width: 是否自动调整列宽，默认 True
    :type auto_width: bool

    **参考样例**

    >>> export_report_to_excel(report, 'eda_report.xlsx')
    >>> export_report_to_excel(report, 'eda_report.xlsx', theme_color='00A651')
    """
    from ...report.result import ReportResult

    structured = isinstance(report, ReportResult)
    if structured:
        report.ensure_publishable()
    # 清洗后再做不区分大小写的唯一化，避免截断/特殊字符碰撞覆盖章节。
    used_names = set()

    def clean_sheet_name(name: str) -> str:
        name = str(name)
        # 移除特殊字符
        name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        name = name.replace("?", "").replace("*", "").replace("[", "").replace("]", "")
        # 截取前31个字符
        base = name.strip().strip("'")[:31] or "章节"
        candidate = base
        count = 2
        while candidate.casefold() in used_names:
            suffix = f"_{count}"
            candidate = base[: 31 - len(suffix)] + suffix
            count += 1
        used_names.add(candidate.casefold())
        return candidate

    sheet_mapping = {}
    with ExcelWriter(theme_color=theme_color) as writer:
        if structured:
            status_sheet = writer.get_sheet_by_name(clean_sheet_name("报告执行状态"))
            writer.insert_value2sheet(
                status_sheet,
                "B2",
                "报告完整" if report.complete else "报告不完整：请检查失败或未执行章节",
                style="header",
            )
            writer.insert_df2sheet(status_sheet, report.status_table(), "B4", index=False, auto_width=auto_width)
        for section_name, df in report.items():
            if df is None or df.empty:
                continue

            # 处理工作表名称
            if sheet_name_mapping and section_name in sheet_name_mapping:
                sheet_name = clean_sheet_name(sheet_name_mapping[section_name])
            else:
                sheet_name = clean_sheet_name(section_name)
            sheet_mapping[section_name] = sheet_name

            # 获取或创建工作表
            worksheet = writer.get_sheet_by_name(sheet_name)

            # 写入标题
            writer.insert_value2sheet(worksheet, "B2", value=section_name, style="header")

            # 写入DataFrame
            writer.insert_df2sheet(worksheet, df, "B4", header=True, index=False, auto_width=auto_width, fill=False)

        # 保存文件
        writer.save(filepath)
    if structured:
        report.sheet_mapping = sheet_mapping
        report.artifacts.append(
            {"类型": "Excel", "路径": str(filepath), "完整": report.complete, "章节工作表": dict(sheet_mapping)}
        )

    logger.info("报告已导出至: %s", filepath)


def generate_html_report(report: Dict[str, pd.DataFrame], filepath: str, title: str = "EDA分析报告") -> None:
    """生成HTML报告.

    :param report: 报告字典
    :type report: Dict[str, pd.DataFrame]
    :param filepath: 导出文件路径
    :type filepath: str
    :param title: 默认值为 ``'EDA分析报告'`` 。报告标题
    :type title: str

    **参考样例**

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
            html_parts.append(df.to_html(index=False, classes="data-table"))
        else:
            html_parts.append("<p>无数据</p>")

    # HTML尾部
    html_parts.append(f"""
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """)

    # 写入文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("HTML报告已导出至: %s", filepath)
