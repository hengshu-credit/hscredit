from __future__ import annotations

"""Pandas DataFrame/Series 扩展模块.

为 pandas DataFrame 和 Series 提供额外的方法，包括：
- df.summary(): 综合特征描述统计
- s.summary(): 单字段综合特征描述统计
- df.eda_info(): EDA基础信息
- df.missing_analysis(): 缺失值分析
- df.show(): 美化展示分箱表
- df.save(): 保存到Excel
- df.hscredit.apply() / df.hscredit(...).apply(): DataFrame/Series/GroupBy 严格单次并行 apply

使用方式:
    >>> import pandas as pd
    >>> import hscredit  # 自动注册所有扩展
    >>> 
    >>> df = pd.DataFrame({...})
    >>> 
    >>> # 数据摘要
    >>> summary = df.summary()
    >>> series_summary = df['age'].summary(y=df['target'])
    >>> 
    >>> # 保存到Excel
    >>> df.save("report.xlsx", sheet_name="数据", title="统计表")
    >>> 
    >>> # 美化展示分箱表
    >>> table.show(compact=True)
    >>>
    >>> # 并行 apply（Series 和 GroupBy 用法相同）
    >>> default_result = df.hscredit.apply(func, axis=1)
    >>> result = df.hscredit(n_jobs=-1, bar=True).apply(func, axis=1)
"""

import logging
import re
from functools import partial
from html import escape
import pandas as pd
from .._compat import map_styler, prepare_dependency
from .arguments import binning_options

logger = logging.getLogger(__name__)
import numpy as np
from typing import Optional, List, Dict, Any, Literal, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..excel import ExcelWriter


# =============================================================================
# 1. EDA相关扩展方法 (原pandas_ext.py)
# =============================================================================


@binning_options(keep_config=True)
def _summary_method(
    self,
    features: List[str] = None,
    y: Optional[Union[str, pd.Series, np.ndarray, List, Tuple]] = None,
    val_df: Optional[pd.DataFrame] = None,
    models: Optional[Dict[str, Any]] = None,
    model_type: Optional[Literal["xgboost", "lightgbm", "catboost", "randomforest"]] = None,
    model_params: Optional[Dict] = None,
    max_n_bins: int = 10,
    psi_method: Literal["random_split", "group", "date"] = "random_split",
    group: Optional[str] = None,
    date: Optional[str] = None,
    freq: str = "M",
    psi_test_size: float = 0.3,
    percentiles: List[float] = None,
    random_state: int = 42,
    return_type: Literal["dataframe", "dict"] = "dataframe",
    numeric_as_categorical: Optional[List[str]] = None,
    force_numeric: Optional[List[str]] = None,
    n_jobs: int = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
    method: str = "quantile",
    bin_params: Optional[Dict[str, Any]] = None,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """DataFrame 数据分布摘要统计.

    快速获取数据集特征详情，包括基础统计、IV、KS、趋势、PSI和特征重要性。

    :param features: 特征列表，None则分析全部
    :param y: 目标变量，支持列名、数组、列表、元组或Series，不传则不计算IV/KS/趋势/特征重要性
    :param val_df: 验证集，用于计算PSI
    :param models: 已训练好的模型字典，用于获取特征重要性
    :param model_type: 模型类型，用于自动训练模型提取特征重要性
    :param model_params: 模型参数
    :param max_n_bins: IV、趋势和PSI共用的最大分箱数，默认10
    :param psi_method: PSI计算方式
    :param group: 分组列名（当psi_method='group'时使用）
    :param date: 日期列名（当psi_method='date'时使用）
    :param freq: 时间频率
    :param psi_test_size: 随机拆分比例
    :param percentiles: 分位数点，默认[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    :param random_state: 随机种子
    :param return_type: 返回类型，'dataframe' 或 'dict'
    :param numeric_as_categorical: 强制视为分类变量的数值列
    :param force_numeric: 强制标记为数值变量的列
    :param n_jobs: 并行工作数，-1为保守自动推断
    :param show_progress: 是否显示字段级处理进度
    :param method: IV、趋势和PSI共用的分箱方法，默认'quantile'（等频分箱）
    :param bin_params: 传给OptimalBinning的完整参数；显式参数优先于配置，配置优先于默认值。
        user_splits字典按原DataFrame字段名配置

    :return: 综合特征描述DataFrame或字典

    Example:
        >>> # 基础统计
        >>> summary = df.summary()

        >>> # 包含IV、KS、趋势（传入目标变量）
        >>> summary = df.summary(y='target')

        >>> # 按日期分组计算PSI
        >>> summary = df.summary(y='target', psi_method='date', date='apply_date')
    """
    from ..core.eda import feature_summary

    result = feature_summary(
        df=self,
        features=features,
        y=y,
        val_df=val_df,
        models=models,
        model_type=model_type,
        model_params=model_params,
        max_n_bins=max_n_bins,
        psi_method=psi_method,
        group=group,
        date=date,
        freq=freq,
        psi_test_size=psi_test_size,
        percentiles=percentiles,
        random_state=random_state,
        numeric_as_categorical=numeric_as_categorical,
        force_numeric=force_numeric,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        show_progress=show_progress,
        method=method,
        bin_params=bin_params,
    )

    if return_type == "dict":
        return result.to_dict(orient="records")
    return result


@binning_options(keep_config=True)
def _series_summary_method(
    self,
    y: Optional[Union[pd.Series, np.ndarray, List, Tuple]] = None,
    val_df: Optional[Union[pd.DataFrame, pd.Series]] = None,
    models: Optional[Dict[str, Any]] = None,
    model_type: Optional[Literal["xgboost", "lightgbm", "catboost", "randomforest"]] = None,
    model_params: Optional[Dict] = None,
    max_n_bins: int = 10,
    psi_method: Literal["random_split", "group", "date"] = "random_split",
    group: Optional[str] = None,
    date: Optional[str] = None,
    freq: str = "M",
    psi_test_size: float = 0.3,
    percentiles: List[float] = None,
    random_state: int = 42,
    numeric_as_categorical: Optional[List[str]] = None,
    force_numeric: Optional[List[str]] = None,
    n_jobs: int = -1,
    parallel_backend: Optional[str] = None,
    parallel_config: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
    return_type: Literal["dataframe", "dict"] = "dataframe",
    method: str = "quantile",
    bin_params: Optional[Dict[str, Any]] = None,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Series 单字段综合摘要统计。

    y 支持数组、列表、元组或 Series。IV、趋势和 PSI 默认共用 quantile 等频
    10 分箱；显式参数优先于 bin_params，配置优先于默认值。
    user_splits 字典使用当前 Series 名称作为字段键。
    """
    if isinstance(y, str):
        raise ValueError("Series.summary 的 y 不支持列名，请直接传入数组、列表或 Series")

    frame = self.to_frame()
    if isinstance(val_df, pd.Series):
        val_df = val_df.to_frame(name=frame.columns[0])

    return _summary_method(
        frame,
        y=y,
        val_df=val_df,
        models=models,
        model_type=model_type,
        model_params=model_params,
        max_n_bins=max_n_bins,
        psi_method=psi_method,
        group=group,
        date=date,
        freq=freq,
        psi_test_size=psi_test_size,
        percentiles=percentiles,
        random_state=random_state,
        numeric_as_categorical=numeric_as_categorical,
        force_numeric=force_numeric,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        parallel_config=parallel_config,
        show_progress=show_progress,
        return_type=return_type,
        method=method,
        bin_params=bin_params,
    )


def _eda_info_method(self) -> Dict[str, Any]:
    """DataFrame EDA 基础信息.

    快速获取数据集基础信息，包括样本数、特征数、缺失值等。

    :return: 字典格式的数据集信息

    Example:
        >>> info = df.eda_info()
        >>> print(info['样本数'])
    """
    from ..core.eda import data_info

    result = data_info(self)
    return dict(zip(result["信息项"], result["值"]))


def _missing_analysis_method(self, threshold: float = 0.0) -> pd.DataFrame:
    """DataFrame 缺失值分析.

    :param threshold: 缺失率阈值，仅返回缺失率>=该值的特征
    :return: 缺失值分析DataFrame

    Example:
        >>> missing = df.missing_analysis(threshold=0.05)
    """
    from ..core.eda import missing_analysis

    return missing_analysis(self, threshold=threshold)


# =============================================================================
# 2. Excel保存相关扩展方法 (原report/excel/pandas_extension.py)
# =============================================================================


def _dataframe_save(
    self,
    excel_writer: Union[str, ExcelWriter],
    worksheet: Optional[Any] = None,
    sheet_name: Optional[str] = None,
    title: Optional[str] = None,
    header: bool = True,
    theme_color: str = "2639E9",
    condition_color: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None,
    fill: bool = True,
    percent_cols: Optional[List] = None,
    condition_cols: Optional[List] = None,
    custom_cols: Optional[List] = None,
    custom_format: str = "#,##0",
    color_cols: Optional[List] = None,
    percent_rows: Optional[List] = None,
    condition_rows: Optional[List] = None,
    custom_rows: Optional[List] = None,
    color_rows: Optional[List] = None,
    start_col: int = 2,
    start_row: int = 2,
    mode: str = "replace",
    figures: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[int, int] = (600, 350),
    image_bottom_padding_rows: int = 1,
    writer_params: Optional[Dict] = None,
    **kwargs,
) -> Union[Tuple[int, int], ExcelWriter]:
    """将DataFrame保存到Excel文件或已有的ExcelWriter中。

    :param excel_writer: 文件路径或ExcelWriter对象
    :param worksheet: 工作表对象（如果提供，将写入该worksheet而不保存文件）
    :param sheet_name: 工作表名称，默认为None
    :param title: 标题，默认为None
    :param header: 是否保存列名，默认为True
    :param theme_color: 主题颜色，默认为"2639E9"
    :param condition_color: 条件格式颜色，默认None（使用ExcelWriter的condition_color）。支持 str（统一）、list/tuple（2或3色异色锚点，仅颜色渐变列/行生效，构成双/三色阶）或 dict（按列名/行索引分别指定，值可为 str 或 list/tuple，未匹配时回退ExcelWriter的condition_color）
    :param fill: 是否使用颜色填充，默认为True
    :param percent_cols: 需要显示为百分数的列，默认为None
    :param condition_cols: 需要显示数据条的列，默认为None
    :param custom_cols: 需要自定义格式的列，默认为None
    :param custom_format: 自定义格式，默认为"#,##0"
    :param color_cols: 需要显示颜色渐变的列，默认为None
    :param percent_rows: 需要显示为百分数的行，默认为None
    :param condition_rows: 需要显示数据条的行，默认为None
    :param custom_rows: 需要自定义格式的行，默认为None
    :param color_rows: 需要显示颜色渐变的行，默认为None
    :param start_col: 起始列，默认为2
    :param start_row: 起始行，默认为2
    :param mode: 写入模式，默认为"replace"
    :param figures: 需要插入的图片路径，默认为None
    :param figsize: 图片大小，默认为(600, 350)
    :param image_bottom_padding_rows: 图片区与下方表格之间的额外空行数，默认为1
    :param writer_params: ExcelWriter参数，默认为None
    :param kwargs: 其他参数，传递给insert_df2sheet
    :return: 如果传入文件路径返回(end_row, end_col)；如果传入ExcelWriter且提供了worksheet，返回ExcelWriter

    **参考样例**

    >>> import pandas as pd
    >>> from hscredit.excel import ExcelWriter
    >>>
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>>
    >>> # 方式1：直接保存到文件
    >>> df.save("report.xlsx", sheet_name="数据", title="统计表")
    >>>
    >>> # 方式2：写入已有的ExcelWriter
    >>> writer = ExcelWriter()
    >>> worksheet = writer.get_sheet_by_name("Sheet1")
    >>> df.save(writer, worksheet=worksheet)
    >>> writer.save("report.xlsx")
    """
    from ..excel import ExcelWriter, dataframe2excel, resolve_condition_color
    from openpyxl.worksheet.worksheet import Worksheet
    from openpyxl.utils import get_column_letter

    # 如果提供了worksheet，说明要使用已有的writer
    if worksheet is not None and isinstance(excel_writer, ExcelWriter):
        # 直接插入到指定的worksheet
        writer = excel_writer

        # 插入标题
        if title:
            col_width = len(self.columns) + self.index.nlevels if kwargs.get("index", False) else len(self.columns)
            _start_row, _end_col = writer.insert_value2sheet(
                worksheet,
                (start_row, start_col),
                value=title,
                style="header",
                end_space=(start_row, start_col + col_width - 1),
            )
            start_row += 1

        # 插入图片
        if figures is not None:
            if isinstance(figures, str):
                figures = [figures]
            figures = [pic for pic in figures if pic]
            if figures:
                pic_row = start_row
                for i, pic in enumerate(figures):
                    if i == 0:
                        start_row, end_col = writer.insert_pic2sheet(
                            worksheet, pic, (pic_row, start_col), figsize=figsize
                        )
                    else:
                        start_row, end_col = writer.insert_pic2sheet(
                            worksheet, pic, (pic_row, end_col - 1), figsize=figsize
                        )

                start_row += 0 if image_bottom_padding_rows is None else max(int(image_bottom_padding_rows), 0)

        # 处理merge_column参数
        if "merge_column" in kwargs and kwargs["merge_column"]:
            if not isinstance(kwargs["merge_column"][0], (tuple, list)):
                kwargs["merge_column"] = [
                    c
                    for c in self.columns
                    if (isinstance(c, tuple) and c[-1] in kwargs["merge_column"])
                    or (not isinstance(c, tuple) and c in kwargs["merge_column"])
                ]

        # 插入DataFrame
        end_row, end_col = writer.insert_df2sheet(
            worksheet, self, (start_row, start_col), fill=fill, header=header, **kwargs
        )

        # 设置百分比格式列
        if percent_cols:
            if not isinstance(percent_cols[0], (tuple, list)):
                percent_cols = [
                    c
                    for c in self.columns
                    if (isinstance(c, tuple) and c[-1] in percent_cols)
                    or (not isinstance(c, tuple) and c in percent_cols)
                ]
            for c in [c for c in percent_cols if c in self.columns]:
                conditional_column = get_column_letter(
                    start_col + self.columns.get_loc(c) + self.index.nlevels
                    if kwargs.get("index", False)
                    else start_col + self.columns.get_loc(c)
                )
                writer.set_number_format(
                    worksheet, f"{conditional_column}{end_row - len(self)}:{conditional_column}{end_row - 1}", "0.00%"
                )

        # 设置自定义格式列
        if custom_cols:
            if not isinstance(custom_cols[0], (tuple, list)):
                custom_cols = [
                    c
                    for c in self.columns
                    if (isinstance(c, tuple) and c[-1] in custom_cols)
                    or (not isinstance(c, tuple) and c in custom_cols)
                ]
            for c in [c for c in custom_cols if c in self.columns]:
                conditional_column = get_column_letter(
                    start_col + self.columns.get_loc(c) + self.index.nlevels
                    if kwargs.get("index", False)
                    else start_col + self.columns.get_loc(c)
                )
                writer.set_number_format(
                    worksheet,
                    f"{conditional_column}{end_row - len(self)}:{conditional_column}{end_row - 1}",
                    custom_format,
                )

        # 设置条件格式列
        if condition_cols:
            if not isinstance(condition_cols[0], (tuple, list)):
                condition_cols = [
                    c
                    for c in self.columns
                    if (isinstance(c, tuple) and c[-1] in condition_cols)
                    or (not isinstance(c, tuple) and c in condition_cols)
                ]
            for c in [c for c in condition_cols if c in self.columns]:
                conditional_column = get_column_letter(
                    start_col + self.columns.get_loc(c) + self.index.nlevels
                    if kwargs.get("index", False)
                    else start_col + self.columns.get_loc(c)
                )
                writer.add_conditional_formatting(
                    worksheet,
                    f"{conditional_column}{end_row - len(self)}",
                    f"{conditional_column}{end_row - 1}",
                    condition_color=resolve_condition_color(condition_color, c, writer.condition_color),
                )

        return writer

    # 使用dataframe2excel函数（传入文件路径或ExcelWriter但没有worksheet）
    return dataframe2excel(
        data=self,
        excel_writer=excel_writer,
        sheet_name=sheet_name,
        title=title,
        header=header,
        theme_color=theme_color,
        condition_color=condition_color,
        fill=fill,
        percent_cols=percent_cols,
        condition_cols=condition_cols,
        custom_cols=custom_cols,
        custom_format=custom_format,
        color_cols=color_cols,
        percent_rows=percent_rows,
        condition_rows=condition_rows,
        custom_rows=custom_rows,
        color_rows=color_rows,
        start_col=start_col,
        start_row=start_row,
        mode=mode,
        figures=figures,
        figsize=figsize,
        image_bottom_padding_rows=image_bottom_padding_rows,
        writer_params=writer_params,
        **kwargs,
    )


def _series_save(
    self,
    excel_writer: Union[str, ExcelWriter],
    worksheet: Optional[Any] = None,
    sheet_name: Optional[str] = None,
    title: Optional[str] = None,
    header: bool = True,
    theme_color: str = "2639E9",
    condition_color: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None,
    fill: bool = True,
    percent_cols: Optional[List] = None,
    condition_cols: Optional[List] = None,
    custom_cols: Optional[List] = None,
    custom_format: str = "#,##0",
    color_cols: Optional[List] = None,
    percent_rows: Optional[List] = None,
    condition_rows: Optional[List] = None,
    custom_rows: Optional[List] = None,
    color_rows: Optional[List] = None,
    start_col: int = 2,
    start_row: int = 2,
    mode: str = "replace",
    figures: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[int, int] = (600, 350),
    image_bottom_padding_rows: int = 1,
    writer_params: Optional[Dict] = None,
    **kwargs,
) -> Union[Tuple[int, int], ExcelWriter]:
    """将Series保存到Excel文件或已有的ExcelWriter中。
    Series会被转换为单列DataFrame（保留name作为列名）。

    :param excel_writer: 文件路径或ExcelWriter对象
    :param worksheet: 工作表对象（如果提供，将写入该worksheet而不保存文件）
    :param sheet_name: 工作表名称，默认为None
    :param title: 标题，默认为None
    :param header: 是否保存列名，默认为True
    :param theme_color: 主题颜色，默认为"2639E9"
    :param condition_color: 条件格式颜色，默认None（使用ExcelWriter的condition_color）。支持 str（统一）、list/tuple（2或3色异色锚点，仅颜色渐变列/行生效，构成双/三色阶）或 dict（按列名/行索引分别指定，值可为 str 或 list/tuple，未匹配时回退ExcelWriter的condition_color）
    :param fill: 是否使用颜色填充，默认为True
    :param percent_cols: 需要显示为百分数的列，默认为None
    :param condition_cols: 需要显示数据条的列，默认为None
    :param custom_cols: 需要自定义格式的列，默认为None
    :param custom_format: 自定义格式，默认为"#,##0"
    :param color_cols: 需要显示颜色渐变的列，默认为None
    :param percent_rows: 需要显示为百分数的行，默认为None
    :param condition_rows: 需要显示数据条的行，默认为None
    :param custom_rows: 需要自定义格式的行，默认为None
    :param color_rows: 需要显示颜色渐变的行，默认为None
    :param start_col: 起始列，默认为2
    :param start_row: 起始行，默认为2
    :param mode: 写入模式，默认为"replace"
    :param figures: 需要插入的图片路径，默认为None
    :param figsize: 图片大小，默认为(600, 350)
    :param image_bottom_padding_rows: 图片区与下方表格之间的额外空行数，默认为1
    :param writer_params: ExcelWriter参数，默认为None
    :param kwargs: 其他参数，传递给insert_df2sheet
    :return: 如果传入文件路径返回(end_row, end_col)；如果传入ExcelWriter且提供了worksheet，返回ExcelWriter

    **参考样例**

    >>> import pandas as pd
    >>> from hscredit.excel import ExcelWriter
    >>>
    >>> s = pd.Series([1, 2, 3], name='数值', index=['a', 'b', 'c'])
    >>>
    >>> # 方式1：直接保存到文件
    >>> s.save("report.xlsx", sheet_name="数据", title="序列数据")
    >>>
    >>> # 方式2：写入已有的ExcelWriter
    >>> writer = ExcelWriter()
    >>> worksheet = writer.get_sheet_by_name("Sheet1")
    >>> s.save(writer, worksheet=worksheet)
    >>> writer.save("report.xlsx")
    """
    # 将Series转换为DataFrame
    df = self.to_frame()

    # 调用DataFrame的save方法
    return df.save(
        excel_writer=excel_writer,
        worksheet=worksheet,
        sheet_name=sheet_name,
        title=title,
        header=header,
        theme_color=theme_color,
        condition_color=condition_color,
        fill=fill,
        percent_cols=percent_cols,
        condition_cols=condition_cols,
        custom_cols=custom_cols,
        custom_format=custom_format,
        color_cols=color_cols,
        percent_rows=percent_rows,
        condition_rows=condition_rows,
        custom_rows=custom_rows,
        color_rows=color_rows,
        start_col=start_col,
        start_row=start_row,
        mode=mode,
        figures=figures,
        figsize=figsize,
        image_bottom_padding_rows=image_bottom_padding_rows,
        writer_params=writer_params,
        **kwargs,
    )


# =============================================================================
# 3. 分箱表展示相关扩展方法 (原bin_table_display.py)
# =============================================================================


def _format_table_value(value, spec=None):
    """格式化标量指标，容器按文本展示，且不改变原始单元格。"""
    if not pd.api.types.is_scalar(value):
        return escape(str(value))
    if pd.isna(value):
        return "-"
    is_number = pd.api.types.is_number(value) and not isinstance(value, (bool, np.bool_, complex, np.complexfloating))
    if is_number and spec is not None:
        return format(value, spec)
    if isinstance(value, (float, np.floating)):
        return format(value, f".{pd.get_option('styler.format.precision')}f")
    return escape(str(value))


def _is_numeric_table_column(series):
    """仅对实数数值列启用指标数据条，排除容器、文本、布尔及复数。"""
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _select_table_columns(df, columns):
    """按完整列名或多级表头末级名称选择列，与 dataframe2excel 一致。"""
    if columns is None:
        return []
    if isinstance(columns, str):
        columns = [columns]
    selectors = [tuple(col) if isinstance(col, list) else col for col in columns]
    return [col for col in df.columns if col in selectors or (isinstance(col, tuple) and col[-1] in selectors)]


def _table_hex_color(color):
    """校验并规范化表格使用的六位十六进制颜色。"""
    if not isinstance(color, str) or re.fullmatch(r"#?[0-9a-fA-F]{6}", color) is None:
        raise ValueError("表格颜色必须为六位十六进制字符串，可带 # 前缀")
    return "#" + color.lstrip("#").upper()


def _table_condition_colors(condition_color, column, default):
    """复用 Excel 的完整列名、由内向外层级和缺省颜色优先级。"""
    from ..excel.writer import _coerce_color_anchors, _resolve_condition_value

    value = _resolve_condition_value(condition_color, column, default)
    return [_table_hex_color(color) for color in _coerce_color_anchors(value)] or [_table_hex_color(default)]


def _apply_table_color_blocks(styler, df, columns, condition_color):
    """按数值生成颜色块；缺失、无穷和复杂类型不着色。"""
    if not columns:
        return styler
    prepare_dependency("matplotlib")
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_hex

    for col in columns:
        if not _is_numeric_table_column(df[col]):
            continue
        values = df[col].dropna().astype(float)
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        anchors = _table_condition_colors(condition_color, col, "F76E6C")
        low, high = float(values.min()), float(values.max())
        if len(anchors) == 1:
            # 单色以零为白色基准，负值和正值两端均使用指定色。
            anchors = [anchors[0], "#FFFFFF", anchors[0]]
            bound = max(abs(low), abs(high)) or 1.0
            norm = Normalize(-bound, bound)
        elif len(anchors) == 3 and low < float(values.median()) < high:
            norm = TwoSlopeNorm(float(values.median()), vmin=low, vmax=high)
        else:
            norm = Normalize(low, high)
        cmap = LinearSegmentedColormap.from_list("table_condition", anchors, N=257)

        def color_cell(value, *, norm=norm, cmap=cmap):
            if pd.isna(value) or not np.isfinite(value):
                return ""
            rgba = cmap(norm(float(value)))
            luminance = sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), rgba[:3]))
            text_color = "#000000" if luminance > 0.5 else "#ffffff"
            return f"background-color: {to_hex(rgba)}; color: {text_color}"

        styler = map_styler(styler, color_cell, subset=[col])
    return styler


def _style_table_column_groups(styler, rgb):
    """用层级底色和贯穿子列的分隔线区分多级表头的相邻分组。"""
    columns = styler.data.columns
    if not isinstance(columns, pd.MultiIndex) or columns.nlevels < 2:
        return styler

    last_level = columns.nlevels - 1
    theme = "#" + "".join(f"{component:02x}" for component in rgb)
    light = "#" + "".join(f"{round(255 * 0.88 + component * 0.12):02x}" for component in rgb)
    dark = "#" + "".join(f"{round(component * 0.55):02x}" for component in rgb)
    styles = [
        {
            "selector": f"thead th.level{last_level}",
            "props": [("background-color", light), ("color", dark), ("border-bottom", f"2px solid {theme}")],
        }
    ]
    # 由内向外设置，确保父分组边界不会被较细的子分组边界覆盖。
    for level in reversed(range(last_level)):
        group_number = -1
        previous = None
        for position in range(len(columns)):
            # 使用 codes 区分完整父路径，也兼容缺失的层级名称。
            group = tuple(int(columns.codes[index][position]) for index in range(level + 1))
            is_start = group != previous
            if is_start:
                group_number += 1
            shade = min(0.65 + 0.2 * level + 0.2 * (group_number % 2), 1.0)
            background = "#" + "".join(f"{round(component * shade):02x}" for component in rgb)
            styles.append(
                {
                    "selector": f"thead th.col_heading.level{level}.col{position}",
                    "props": [
                        ("background-color", background),
                        ("color", "#ffffff"),
                        ("font-size", "14px" if level == 0 else "13px"),
                        ("padding", "10px 14px"),
                        ("border-bottom", "2px solid #ffffff"),
                    ],
                }
            )
            if is_start and position > 0:
                width = 3 if level == 0 else 2
                styles.append(
                    {
                        "selector": f"thead th.col_heading.level{level}.col{position}",
                        "props": [("border-left", f"{width}px solid #ffffff")],
                    }
                )
                selectors = [
                    f"thead th.col_heading.level{child}.col{position}" for child in range(level + 1, last_level + 1)
                ]
                selectors.append(f"tbody td.col{position}")
                styles.append(
                    {"selector": ", ".join(selectors), "props": [("border-left", f"{width}px solid {theme}")]}
                )
            previous = group
    return styler.set_table_styles(styles, overwrite=False)


def _apply_table_layout(styler, theme_color="2639E9", fill=True):
    """统一展示表和规则表的表头、隔行底色及整行悬停效果。"""
    theme = _table_hex_color(theme_color)
    rgb = tuple(int(theme[i : i + 2], 16) for i in (1, 3, 5))

    def tint(weight):
        return "#" + "".join(f"{round(255 * (1 - weight) + component * weight):02x}" for component in rgb)

    styler = styler.set_properties(
        **{
            "white-space": "nowrap",
            "text-align": "center",
            "font-size": "12px",
            "font-family": "Arial, sans-serif",
        }
    )

    styler = styler.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", theme),
                    ("color", "#ffffff"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("font-size", "12px"),
                    ("padding", "8px 10px"),
                    ("border", "1px solid rgba(255,255,255,0.3)"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody td, tbody th",
                "props": [
                    ("padding", "6px 10px"),
                    ("border", "1px solid #e0e0e0"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [
                    ("background-color", "#ffffff"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [
                    ("background-color", "#f8f9ff" if theme == "#2639E9" else tint(0.03)),
                ],
            },
            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#e8eaff" if theme == "#2639E9" else tint(0.1)),
                ],
            },
            {
                "selector": "tbody tr:hover td, tbody tr:hover th",
                "props": [("box-shadow", f"inset 0 0 0 9999px rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)")],
            },
            {
                "selector": "tr.selected, td.selected",
                "props": [("background-color", "#e3f2fd"), ("color", "#1565C0")],
            },
            # 进度条样式
            {
                "selector": ".pd-bar",
                "props": [
                    ("opacity", "0.75"),
                ],
            },
        ]
    )

    if not fill:
        styler = styler.set_table_styles(
            [style for style in styler.table_styles if "nth-child" not in style["selector"]]
        )
    return _style_table_column_groups(styler, rgb)


class BinTableDisplay:
    """分箱表展示器.

    提供链式调用接口，方便在 Jupyter 中展示美观的分箱表。

    **参数**

    :param df: 分箱统计表
    :type df: pd.DataFrame

    **参考样例**

    >>> table = feature_bin_stats(data, 'score', target='target')
    >>> table.show()
    >>> table.show(compact=True)
    >>> table.show(highlight_iv=False)
    """

    def __init__(self, df: pd.DataFrame):
        """初始化展示器.

        :param df: 分箱统计表
        """
        self._df = df
        self._styler = None

    def show(
        self,
        max_rows: Optional[int] = None,
        highlight_iv: bool = True,
        highlight_bad_rate: bool = True,
        highlight_lift: bool = True,
        highlight_ks: bool = True,
        compact: bool = False,
        precision: Optional[Dict[str, int]] = None,
        index_as_bin: bool = False,
        percent_format: bool = True,
        high_tech_style: bool = False,
        *,
        percent_cols: Optional[List] = None,
        condition_cols: Optional[List] = None,
        color_cols: Optional[List] = None,
        condition_color: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None,
        theme_color: str = "2639E9",
        fill: bool = True,
        **kwargs,
    ) -> "BinTableDisplay":
        """展示美化的分箱表.

        :param max_rows: 默认值为 ``None`` 。最大显示行数
        :type max_rows: Optional[int]
        :param highlight_iv: 默认值为 ``True`` 。是否高亮 IV 值
        :type highlight_iv: bool
        :param highlight_bad_rate: 默认值为 ``True`` 。是否高亮坏样本率
        :type highlight_bad_rate: bool
        :param highlight_lift: 默认值为 ``True`` 。是否高亮 LIFT 值
        :type highlight_lift: bool
        :param highlight_ks: 默认值为 ``True`` 。是否高亮 KS 值
        :type highlight_ks: bool
        :param compact: 默认值为 ``False`` 。是否使用紧凑模式
        :type compact: bool
        :param precision: 默认值为 ``None`` 。自定义小数位数
        :type precision: Optional[Dict[str, int]]
        :param index_as_bin: 默认值为 ``False`` 。是否将分箱作为索引显示
        :type index_as_bin: bool
        :param percent_format: 是否将百分比相关列显示为百分比格式（默认True）
        :type percent_format: bool
        :param high_tech_style: 默认值为 ``False`` 。是否使用高科技/AI风格样式
        :type high_tech_style: bool
        :param kwargs: 其他参数
        :param percent_cols: 百分比列；None 按指标名自动识别，[] 关闭自动百分比，显式名单优先于 percent_format。
        :param condition_cols: 数据条列；None 沿用 ``highlight_*`` 自动识别，[] 关闭数据条，显式名单优先于 ``highlight_*``。
        :param color_cols: 颜色块渐变列，默认不启用；与数据条重叠时颜色块优先。
        :param condition_color: 与 dataframe2excel 同名配置：六位十六进制单色（可带 #）、2/3 色列表，
            或按列配置的字典。完整 tuple 优先，其次从内向外匹配层级名；未匹配沿用默认配色。
            色阶超过三色取首/中/尾，数据条取末色；单色色块以零为白色基准，三色色块以中位数为中值。
        :param theme_color: 表头主题色，默认 "2639E9"，同时影响隔行底色和 hover 配色。
        :param fill: 是否启用隔行底色，默认 True；不影响表头、hover 和显式条件格式。

        列名单使用列名或多级表头完整 tuple，也可按末级名称选择多个标签的同名列。
        显式指定的列在 compact=True 时仍保留；复杂单元格始终按文本显示。

        :return: self，支持链式调用
        """
        self._styler = _style_bin_table(
            self._df,
            max_rows=max_rows,
            highlight_iv=highlight_iv,
            highlight_bad_rate=highlight_bad_rate,
            highlight_lift=highlight_lift,
            highlight_ks=highlight_ks,
            compact=compact,
            precision=precision,
            index_as_bin=index_as_bin,
            percent_format=percent_format,
            high_tech_style=high_tech_style,
            percent_cols=percent_cols,
            condition_cols=condition_cols,
            color_cols=color_cols,
            condition_color=condition_color,
            theme_color=theme_color,
            fill=fill,
        )
        try:
            from IPython.display import display

            display(self._styler)
        except ImportError:
            pass
        return self

    def highlight_bins(self, bins: Union[int, List[int]], color: str = "#e3f2fd") -> "BinTableDisplay":
        """高亮指定的分箱行.

        :param bins: 要高亮的分箱索引或索引列表
        :type bins: Union[int, List[int]]
        :param color: 默认值为 ``'#e3f2fd'`` 。高亮颜色
        :type color: str
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = _style_bin_table(self._df)

        if isinstance(bins, int):
            bins = [bins]

        def highlight_row(row):
            if row.name in bins:
                return ["background-color: #e3f2fd; color: #1565C0"] * len(row)
            return [""] * len(row)

        self._styler = self._styler.apply(highlight_row, axis=1)
        try:
            from IPython.display import display

            display(self._styler)
        except ImportError:
            pass
        return self

    def export_html(self, filename: str) -> "BinTableDisplay":
        """导出为 HTML 文件.

        :param filename: 文件名
        :type filename: str
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = _style_bin_table(self._df)

        html = self._styler.to_html()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>分箱统计表</title>
                <style>
                    .dataframe {{ overflow-x: auto; }}
                    .dataframe th {{ white-space: nowrap; }}
                    .dataframe td {{ white-space: nowrap; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """)
        logger.info("已导出到: %s", filename)
        return self

    def to_excel(self, filename: str, sheet_name: str = "分箱统计") -> "BinTableDisplay":
        """导出为 Excel 文件.

        :param filename: 文件名
        :type filename: str
        :param sheet_name: 默认值为 ``'分箱统计'`` 。工作表名称
        :type sheet_name: str
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = _style_bin_table(self._df)

        self._styler.to_excel(filename, sheet_name=sheet_name, engine="openpyxl")
        logger.info("已导出到: %s", filename)
        return self


def _style_bin_table(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    highlight_iv: bool = True,
    highlight_bad_rate: bool = True,
    highlight_lift: bool = True,
    highlight_ks: bool = True,
    compact: bool = False,
    precision: Optional[Dict[str, int]] = None,
    index_as_bin: bool = False,
    percent_format: bool = True,
    high_tech_style: bool = False,
    *,
    percent_cols: Optional[List] = None,
    condition_cols: Optional[List] = None,
    color_cols: Optional[List] = None,
    condition_color: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None,
    theme_color: str = "2639E9",
    fill: bool = True,
) -> Any:
    """美化 DataFrame 展示，兼容分箱统计及含列表、字典等复杂值的结果表.

    使用 pandas Styler 对分箱表进行格式化和高亮，使其在 Jupyter 中更易读。

    :param df: 分箱统计表 DataFrame
    :param max_rows: 最大显示行数，None 表示显示全部
    :param highlight_iv: 是否高亮 IV 值
    :param highlight_bad_rate: 是否高亮坏样本率（进度条）
    :param highlight_lift: 是否高亮 LIFT 值（进度条）
    :param highlight_ks: 是否高亮 KS 值（进度条）
    :param compact: 是否使用紧凑模式（隐藏部分列）
    :param precision: 自定义小数位数，格式为 {'列名': 位数}
    :param index_as_bin: 是否将分箱作为索引显示
    :param percent_format: 是否将百分比相关列显示为百分比格式（默认True）
    :param high_tech_style: 是否使用高科技/AI风格样式
    :param percent_cols: 百分比列；None 按指标名自动识别，[] 关闭自动百分比，显式名单优先于 percent_format。
    :param condition_cols: 数据条列；None 沿用 ``highlight_*`` 自动识别，[] 关闭数据条，显式名单优先于 ``highlight_*``。
    :param color_cols: 颜色块渐变列，默认不启用；与数据条重叠时颜色块优先。
    :param condition_color: 与 dataframe2excel 同名配置：六位十六进制单色（可带 #）、2/3 色列表，
        或按列配置的字典。完整 tuple 优先，其次从内向外匹配层级名；未匹配沿用默认配色。
        色阶超过三色取首/中/尾，数据条取末色；单色色块以零为白色基准，三色色块以中位数为中值。
    :param theme_color: 表头主题色，默认 "2639E9"，同时影响隔行底色和 hover 配色。
    :param fill: 是否启用隔行底色，默认 True；不影响表头、hover 和显式条件格式。

    列名单使用列名或多级表头完整 tuple，也可按末级名称选择多个标签的同名列。
    显式指定的列在 compact=True 时仍保留；复杂单元格始终按文本显示。

    :return: 格式化后的 Styler 对象
    """
    # 检查是否为多级表头
    is_multi_level = isinstance(df.columns, pd.MultiIndex)

    # 创建副本避免修改原数据
    df_display = df.copy()

    # index_as_bin: 将分箱列设置为索引，并合并指标名称/含义列
    bin_col = None
    if index_as_bin:
        # 查找分箱标签列
        bin_label_col = None
        indicator_name_col = None
        indicator_desc_col = None

        for col in df_display.columns:
            col_name = col[1] if isinstance(col, tuple) else col
            if col_name == "分箱标签":
                bin_label_col = col
            if col_name == "指标名称":
                indicator_name_col = col
            if col_name == "指标含义":
                indicator_desc_col = col

        # 先合并指标名称和指标含义
        if indicator_name_col is not None and indicator_desc_col is not None:
            # 合并为新列
            if is_multi_level:
                df_display["指标名称_合并"] = (
                    df_display[indicator_name_col].astype(str) + " - " + df_display[indicator_desc_col].astype(str)
                )
                df_display = df_display.drop(columns=[indicator_name_col, indicator_desc_col])
                # 重建多级索引列名
                new_columns = []
                for col in df_display.columns:
                    if col == "指标名称_合并":
                        new_columns.append(("基本信息", "指标"))
                    elif isinstance(col, tuple):
                        new_columns.append(col)
                    else:
                        new_columns.append(col)
                df_display.columns = pd.MultiIndex.from_tuples(new_columns)
                is_multi_level = isinstance(df_display.columns, pd.MultiIndex)

        if bin_label_col is not None:
            # 将分箱标签设为索引名称
            df_display = df_display.set_index(bin_label_col)
            # 如果有分箱列，也删除它
            if bin_col is not None:
                try:
                    df_display = df_display.drop(columns=[bin_col])
                except KeyError:
                    pass

        # 重新检查是否为多级表头（因为可能改变了列结构）
        is_multi_level = isinstance(df_display.columns, pd.MultiIndex)

    # 限制行数
    if max_rows is not None and len(df_display) > max_rows:
        df_display = df_display.head(max_rows)

    explicit_columns = set()
    for selection in (percent_cols, condition_cols, color_cols):
        explicit_columns.update(_select_table_columns(df_display, selection))

    # 紧凑模式下隐藏部分列
    if compact:
        # 确定要隐藏的列
        hide_cols = []
        if is_multi_level:
            # 多级表头
            all_cols = df_display.columns.tolist()
            # 保留核心列
            keep_patterns = [
                "指标",
                "分箱标签",
                "样本总数",
                "坏样本率",
                "分档WOE值",
                "分档IV值",
                "指标IV值",
                "LIFT值",
                "分档KS值",
            ]
            for col in all_cols:
                col_name = col[1] if isinstance(col, tuple) else col
                if not any(p in col_name for p in keep_patterns):
                    hide_cols.append(col)
        else:
            # 单层表头
            all_cols = df_display.columns.tolist()
            keep_patterns = [
                "指标名称",
                "指标含义",
                "指标",
                "分箱标签",
                "样本总数",
                "坏样本率",
                "分档WOE值",
                "分档IV值",
                "指标IV值",
                "LIFT值",
                "分档KS值",
            ]
            hide_cols = [c for c in all_cols if not any(p in str(c) for p in keep_patterns)]

        if hide_cols:
            df_display = df_display.drop(columns=[col for col in hide_cols if col not in explicit_columns])

    # 创建 Styler
    styler = df_display.style

    # 定义默认小数位数
    default_precision = {
        "样本总数": 0,
        "好样本数": 0,
        "坏样本数": 0,
        "样本占比": 2,
        "好样本占比": 2,
        "坏样本占比": 2,
        "坏样本率": 2,
        "分档WOE值": 4,
        "分档IV值": 4,
        "指标IV值": 4,
        "LIFT值": 2,
        "坏账改善": 2,
        "累积LIFT值": 2,
        "累积坏账改善": 2,
        "分档KS值": 2,
    }

    # LIFT 为倍数，不按百分比显示。
    percent_columns = {
        "样本占比",
        "好样本占比",
        "坏样本占比",
        "坏样本率",
        "坏账改善",
        "累积坏账改善",
        "分档KS值",
    }
    if precision:
        default_precision.update(precision)

    selected_percent = (
        set(_select_table_columns(df_display, percent_cols))
        if percent_cols is not None
        else {
            col
            for col in df_display.columns
            if percent_format and (col[-1] if isinstance(col, tuple) else str(col)) in percent_columns
        }
    )
    format_dict = {}
    for col in df_display.columns:
        col_name = col[-1] if isinstance(col, tuple) else str(col)
        spec = None
        digits = default_precision.get(col, default_precision.get(col_name))
        if col in selected_percent:
            spec = f".{digits if digits is not None else 2}%"
        elif digits is not None:
            spec = f".{digits}f"
        format_dict[col] = partial(_format_table_value, spec=spec)
    styler = styler.format(format_dict)
    styler = styler.format_index(escape="html", axis=0).format_index(escape="html", axis=1)

    block_columns = _select_table_columns(df_display, color_cols)
    bar_columns = {}
    if condition_cols is None:
        for enabled, pattern, color, floor, ceiling in (
            (highlight_bad_rate, "坏样本率", "5B8FF9", 0.1, 1),
            (highlight_lift, "LIFT值", "5AD8A6", 1.5, 5),
            (highlight_ks, "分档KS值", "F6BD16", 0.1, 1),
        ):
            if enabled:
                for col in _find_columns(df_display, pattern):
                    if _is_numeric_table_column(df_display[col]):
                        values = df_display[col].dropna()
                        values = values[np.isfinite(values)]
                        if not values.empty:
                            vmax = min(max(values.quantile(0.95), floor) * 1.2, ceiling)
                            bar_columns[col] = (color, 0, vmax)
    else:
        bar_columns = {col: ("F76E6C", None, None) for col in _select_table_columns(df_display, condition_cols)}

    for col, (default_color, vmin, vmax) in bar_columns.items():
        if col in block_columns or not _is_numeric_table_column(df_display[col]):
            continue
        color = _table_condition_colors(condition_color, col, default_color)[-1]
        styler = styler.bar(subset=[col], color=color, vmin=vmin, vmax=vmax, axis=0)
    styler = _apply_table_color_blocks(styler, df_display, block_columns, condition_color)
    styler = _apply_table_layout(styler, theme_color=theme_color, fill=fill)

    return styler


def _find_columns(df: pd.DataFrame, pattern: str) -> List:
    """查找匹配的列名（支持多级表头）."""
    if isinstance(df.columns, pd.MultiIndex):
        return [col for col in df.columns if pattern in str(col[1])]
    else:
        return [col for col in df.columns if pattern in str(col)]


def style_bin_table(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    highlight_iv: bool = True,
    highlight_bad_rate: bool = True,
    highlight_lift: bool = True,
    highlight_ks: bool = True,
    compact: bool = False,
    precision: Optional[Dict[str, int]] = None,
    index_as_bin: bool = False,
    percent_format: bool = True,
    high_tech_style: bool = False,
    *,
    percent_cols: Optional[List] = None,
    condition_cols: Optional[List] = None,
    color_cols: Optional[List] = None,
    condition_color: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None,
    theme_color: str = "2639E9",
    fill: bool = True,
) -> Any:
    """美化分箱表展示.

    使用 pandas Styler 对分箱表进行格式化和高亮，使其在 Jupyter 中更易读。

    :param df: 分箱统计表 DataFrame
    :type df: pd.DataFrame
    :param max_rows: 默认值为 ``None`` 。最大显示行数，None 表示显示全部
    :type max_rows: Optional[int]
    :param highlight_iv: 默认值为 ``True`` 。是否高亮 IV 值
    :type highlight_iv: bool
    :param highlight_bad_rate: 默认值为 ``True`` 。是否高亮坏样本率（进度条）
    :type highlight_bad_rate: bool
    :param highlight_lift: 默认值为 ``True`` 。是否高亮 LIFT 值（进度条）
    :type highlight_lift: bool
    :param highlight_ks: 默认值为 ``True`` 。是否高亮 KS 值（进度条）
    :type highlight_ks: bool
    :param compact: 默认值为 ``False`` 。是否使用紧凑模式（隐藏部分列）
    :type compact: bool
    :param precision: 默认值为 ``None`` 。自定义小数位数，格式为 {'列名': 位数}
    :type precision: Optional[Dict[str, int]]
    :param index_as_bin: 默认值为 ``False`` 。是否将分箱作为索引显示
    :type index_as_bin: bool
    :param percent_format: 是否将百分比相关列显示为百分比格式（默认True）
    :type percent_format: bool
    :param high_tech_style: 默认值为 ``False`` 。是否使用高科技/AI风格样式
    :type high_tech_style: bool
    :param percent_cols: 百分比列；None 按指标名自动识别，[] 关闭自动百分比，显式名单优先于 percent_format。
    :param condition_cols: 数据条列；None 沿用 ``highlight_*`` 自动识别，[] 关闭数据条，显式名单优先于 ``highlight_*``。
    :param color_cols: 颜色块渐变列，默认不启用；与数据条重叠时颜色块优先。
    :param condition_color: 与 dataframe2excel 同名配置：六位十六进制单色（可带 #）、2/3 色列表，
        或按列配置的字典。完整 tuple 优先，其次从内向外匹配层级名；未匹配沿用默认配色。
        色阶超过三色取首/中/尾，数据条取末色；单色色块以零为白色基准，三色色块以中位数为中值。
    :param theme_color: 表头主题色，默认 "2639E9"，同时影响隔行底色和 hover 配色。
    :param fill: 是否启用隔行底色，默认 True；不影响表头、hover 和显式条件格式。

    列名单使用列名或多级表头完整 tuple，也可按末级名称选择多个标签的同名列。
    显式指定的列在 compact=True 时仍保留；复杂单元格始终按文本显示。

    :return: 格式化后的 Styler 对象

    **参考样例**

    >>> table = feature_bin_stats(data, 'score', target='target')
    >>> style_bin_table(table, percent_cols=["坏样本率"], color_cols=["LIFT值"])
    >>> table.show(
    ...     percent_cols=["坏样本率"], condition_cols=["样本占比"], color_cols=["LIFT值"],
    ...     condition_color={"样本占比": "5B8FF9", "LIFT值": ["F76E6C", "FFFFFF", "5AD8A6"]},
    ... )
    """
    return _style_bin_table(
        df,
        max_rows=max_rows,
        highlight_iv=highlight_iv,
        highlight_bad_rate=highlight_bad_rate,
        highlight_lift=highlight_lift,
        highlight_ks=highlight_ks,
        compact=compact,
        precision=precision,
        index_as_bin=index_as_bin,
        percent_format=percent_format,
        high_tech_style=high_tech_style,
        percent_cols=percent_cols,
        condition_cols=condition_cols,
        color_cols=color_cols,
        condition_color=condition_color,
        theme_color=theme_color,
        fill=fill,
    )


def style_rule_table(
    df: pd.DataFrame,
    overall_badrate: Optional[float] = None,
    precision: Optional[dict] = None,
) -> Any:
    """美化决策树规则表展示。

    使用 pandas Styler 对规则表进行格式化、高亮和颜色标注，
    使其在 Jupyter 中更易读。

    :param df: 规则表 DataFrame，兼容两种来源：

        - :meth:`ManualTreeExtractor.get_rule_table`  /  :meth:`report`  的报告格式
          （节点编号、是否叶子、指标含义、样本总数、样本占比、坏样本率、LIFT值、…）

        - 旧版精简格式（节点编号、是否叶子、规则表达式、样本数、坏账率、LIFT值、…）

    :type df: pd.DataFrame

    :param overall_badrate: 全局坏样本率（用于计算颜色梯度），默认从数据推断
    :type overall_badrate: Optional[float]
    :param precision: 默认值为 ``None`` 。自定义小数位数，格式为 {'列名': 位数}
    :type precision: Optional[dict]
    :return: 格式化后的 Styler 对象

    **参考样例**

    >>> from hscredit.report.mining import ManualTreeExtractor
    >>> ext = ManualTreeExtractor(target='IS_BAD')
    >>> ext.fit(df, features=['age', 'income'])
    >>> styled = style_rule_table(ext.get_rule_table())
    >>> styled.show()
    """
    prepare_dependency("matplotlib")
    df_d = df.copy()

    # MultiIndex（多标签 overdue/dpds）沿用统一布局，暂不添加指标条件格式。
    if isinstance(df_d.columns, pd.MultiIndex):
        return _apply_table_layout(df_d.style)

    # 兼容报告格式与旧版精简格式的列名
    badrate_col = "坏样本率" if "坏样本率" in df_d.columns else "坏账率"
    count_col = "样本总数" if "样本总数" in df_d.columns else "样本数"
    rule_col = "指标含义" if "指标含义" in df_d.columns else "规则表达式"

    # 报告格式中恒为常量的噪声列，展示时隐藏（指标名称为原始表达式，与指标含义重复）
    hide_cols = [c for c in ("规则分类", "指标名称", "分箱") if c in df_d.columns]

    # 推断全局坏样本率（颜色梯度用）
    if overall_badrate is None:
        if count_col in df_d.columns and "坏样本数" in df_d.columns:
            total_samples = df_d[count_col].sum()
            total_bad = df_d["坏样本数"].sum()
            overall_badrate = total_bad / total_samples if total_samples > 0 else 0.0
        else:
            overall_badrate = 0.0

    # 创建 Styler
    styler = df_d.style
    if hide_cols:
        styler = styler.hide(subset=hide_cols, axis="columns")

    # 默认精度：整数列、百分数列、普通小数列
    _int_cols = ("节点编号", count_col, "好样本数", "坏样本数")
    _pct_cols = (
        "样本占比",
        badrate_col,
        "好样本占比",
        "坏样本占比",
        "坏账改善",
        "风险拒绝比",
        "准确率",
        "精确率",
        "召回率",
        "F1分数",
    )
    default_precision = {c: 0 for c in _int_cols}
    default_precision.update({c: 2 for c in _pct_cols})
    default_precision["LIFT值"] = 2  # LIFT 为倍数，按普通小数显示
    if precision:
        default_precision.update(precision)

    # 格式化为百分比和数字
    format_dict = {}
    for col, prec in default_precision.items():
        if col in df_d.columns:
            if col in _pct_cols:
                format_dict[col] = f"{{:.{max(prec, 2)}%}}"
            elif prec == 0:
                format_dict[col] = "{:.0f}"
            else:
                format_dict[col] = f"{{:.{prec}f}}"
    if format_dict:
        styler = styler.format(format_dict, na_rep="-")

    # --- 坏样本率颜色梯度：低=绿，高=红 ---
    if badrate_col in df_d.columns:
        bad_vals = df_d[badrate_col].fillna(0)
        vmin, vmax = 0.0, max(bad_vals.max() * 1.2, max(overall_badrate * 1.5, 0.3), 0.01)
        styler = styler.background_gradient(
            subset=[badrate_col],
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
        )

    # --- LIFT值颜色梯度：高=深色，低=浅色 ---
    if "LIFT值" in df_d.columns:
        lift_vals = df_d["LIFT值"].fillna(1)
        vmin_l, vmax_l = max(0, lift_vals.min() * 0.8), lift_vals.max() * 1.2
        styler = styler.background_gradient(
            subset=["LIFT值"],
            cmap="RdYlGn",
            vmin=vmin_l,
            vmax=vmax_l,
        )

    # --- 样本占比进度条 ---
    if "样本占比" in df_d.columns:
        styler = styler.bar(
            subset=["样本占比"],
            color="#5B8FF9",
            vmin=0,
            vmax=1,
            align="left",
        )

    # --- 是否叶子列高亮 ---
    if "是否叶子" in df_d.columns:

        def _leaf_color(val):
            if val == "是":
                return "background-color: #e8f5e9; color: #2e7d32; font-weight: bold"
            elif val == "否":
                return "background-color: #fff8e1; color: #f57f17; font-weight: bold"
            return ""

        styler = map_styler(styler, _leaf_color, subset=["是否叶子"])

    styler = _apply_table_layout(styler)

    # --- 规则文本列左对齐 + 等宽字体（按列名定位，避免依赖列顺序）---
    if rule_col in df_d.columns:
        styler = styler.set_properties(
            subset=[rule_col],
            **{
                "text-align": "left",
                "font-family": "monospace",
                "font-size": "11px",
            },
        )

    return styler


def _show_method(self, **kwargs):
    """展示美化的 DataFrame，复杂单元格按文本显示，数值指标保留条件格式.

    :param kwargs: 传递给 BinTableDisplay.show() 的参数
    :return: BinTableDisplay 对象，支持链式调用

    **参考样例**

    >>> table = feature_bin_stats(data, 'score', target='target')
    >>> table.show()
    >>> table.show(compact=True)
    """
    display_obj = BinTableDisplay(self)
    return display_obj.show(**kwargs)


# =============================================================================
# 4. 注册所有扩展方法
# =============================================================================

# 标记是否已注册
_EXTENSIONS_REGISTERED = False


def register_extensions():
    """注册 pandas DataFrame/Series 扩展方法.

    在导入 hscredit 时自动调用，将以下方法添加到 pandas:

    - df.summary(): 综合特征描述统计

    - df.eda_info(): EDA基础信息

    - df.missing_analysis(): 缺失值分析

    - df.show(): 美化展示分箱表

    - df.save(): 保存到Excel

    - s.summary(): 单字段综合特征描述统计

    - s.save(): Series保存到Excel

    - df/s/groupby.hscredit.apply() / hscredit(...).apply(): 严格单次并行 apply

    幂等：重复调用不会重复注册（已存在同名属性时跳过）。导入 hscredit 时已自动执行，
    通常无需手动调用。

    **参考样例**

    >>> import pandas as pd
    >>> import hscredit  # 导入即自动注册扩展
    >>>
    >>> df = pd.DataFrame({'age': [25, 40], 'target': [0, 1]})
    >>> summary = df.summary(y='target')
    >>> df.save("report.xlsx")
    >>> default_result = df.hscredit.apply(func, axis=1)
    >>> result = df.hscredit(n_jobs=-1, bar=False).apply(func, axis=1)
    """
    global _EXTENSIONS_REGISTERED

    if _EXTENSIONS_REGISTERED:
        return

    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

    from .pandas_parallel import HSCreditApplyAccessor

    # 统一并行 apply 配置代理
    for pandas_type in (pd.DataFrame, pd.Series, DataFrameGroupBy, SeriesGroupBy):
        if not hasattr(pandas_type, "hscredit"):
            pandas_type.hscredit = HSCreditApplyAccessor()

    # EDA相关方法
    if not hasattr(pd.DataFrame, "summary"):
        pd.DataFrame.summary = _summary_method

    if not hasattr(pd.DataFrame, "eda_info"):
        pd.DataFrame.eda_info = _eda_info_method

    if not hasattr(pd.DataFrame, "missing_analysis"):
        pd.DataFrame.missing_analysis = _missing_analysis_method

    # Excel保存方法
    if not hasattr(pd.DataFrame, "save"):
        pd.DataFrame.save = _dataframe_save

    if not hasattr(pd.Series, "save"):
        pd.Series.save = _series_save

    if not hasattr(pd.Series, "summary"):
        pd.Series.summary = _series_summary_method

    # 分箱表展示方法
    if not hasattr(pd.DataFrame, "show"):
        pd.DataFrame.show = _show_method

    _EXTENSIONS_REGISTERED = True


# 自动注册扩展
register_extensions()
