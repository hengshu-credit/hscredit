"""目标变量分析模块.

提供目标变量分布、逾期率分析、时间趋势等功能.
专注于金融风控场景.
"""

from hscredit.utils.dates import date_groups
from hscredit.utils.arguments import analysis_inputs

from ...utils.targets import TargetInput, bind_targets

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple

from ...utils.input_utils import normalize_dpd_values
from ...utils.overdue import make_overdue_target, overdue_label, validate_overdue_operator
from .utils import validate_dataframe


def _build_overdue_labels(
    overdue: Union[str, List[str]], dpds: Union[int, List[int]], overdue_operator: str = ">"
) -> List[Tuple[str, int, str]]:
    """构建逾期标签列表.

    :param overdue: 逾期天数字段名或列表
    :param dpds: 逾期天数或列表
    :return: 标签列表 [(标签名, dpd天数, 逾期字段), ...]
    """
    validate_overdue_operator(overdue_operator)
    if isinstance(overdue, str):
        overdue = [overdue]
    dpds = normalize_dpd_values(dpds)
    if not overdue or not dpds:
        raise ValueError("overdue 和 dpds 不能为空")

    labels = []
    for od_field in overdue:
        for dpd in dpds:
            label_name = overdue_label(od_field, dpd, overdue_operator, style="plain")
            labels.append((label_name, dpd, od_field))

    return labels


def _create_binary_target(
    df: pd.DataFrame, overdue_col: str, dpd: int, del_grey: bool = False, overdue_operator: str = ">"
) -> pd.Series:
    """根据逾期天数创建二元目标变量.

    :param df: 输入数据
    :param overdue_col: 逾期天数字段名
    :param dpd: 逾期定义天数
    :param del_grey: 是否按 overdue_operator 对应区间剔除灰样本； ``<`` 、 ``<=`` 下不剔除。
    :return: 二元目标变量 (0/1)，灰样本为NaN
    """
    validate_overdue_operator(overdue_operator)
    del_grey = del_grey and overdue_operator in (">", ">=")
    return make_overdue_target(df[overdue_col], dpd, del_grey, overdue_operator)


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """目标变量分布统计.

    :param df: 输入数据
    :type df: pd.DataFrame
    :param target: 目标变量列名
    :type target: str
    :return: 目标分布DataFrame，列包括[类别, 样本数, 占比, 累计占比]

    **参考样例**

    >>> dist = target_distribution(df, 'fpd15')
    >>> print(dist)
       类别   样本数   占比(%)  累计占比(%)
    0   0    8500   85.00      85.00
    1   1    1500   15.00     100.00
    """

    frame, targets = bind_targets(df, target=target)
    results = {}
    for name, values in targets.y_map(frame).items():
        counts = values.value_counts().sort_index()
        ratios = counts / counts.sum() * 100
        results[name] = pd.DataFrame(
            {
                "类别": counts.index,
                "样本数": counts.values,
                "占比(%)": ratios.round(2).values,
                "累计占比(%)": ratios.cumsum().round(2).values,
            }
        )
    return results if len(results) > 1 else next(iter(results.values()))


def bad_rate_overall(
    df: pd.DataFrame,
    target: Optional[TargetInput] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    del_grey: bool = False,
    *,
    overdue_operator: str = ">",
) -> Union[Dict, pd.DataFrame]:
    """计算整体逾期率.

    支持单标签分析（通过target）或多标签分析（通过overdue+dpds）。

    :param df: 输入数据
    :type df: pd.DataFrame
    :param target: 默认值为 ``None`` 。支持列名、一维 Series/ndarray/list、callable，
        或 ``{标签名: 标签数据或 callable}``；与 ``overdue+dpds`` 互斥。
        callable 接收当前完整 DataFrame；Series 按索引对齐，数组和列表按行位置对应。
    :type target: Optional[TargetInput]
    :param overdue: 默认值为 ``None`` 。逾期天数字段名或字段名序列；与 ``dpds`` 必须同时传入，
        与 ``target`` 互斥，标签方向由 ``overdue_operator`` 控制。
    :type overdue: Optional[Union[str, List[str]]]
    :param dpds: 默认值为 ``None`` 。逾期阈值标量或非空序列，支持整数或小数；
        必须与 ``overdue`` 同时传入。按 ``overdue × dpds`` 组合生成标签。
    :type dpds: Optional[Union[int, List[int]]]
    :param del_grey: 是否逐标签剔除灰客户。``target`` 模式剔除非 0/1 值（含缺失）；
        逾期模式下 ``>`` 剔除 ``(0, DPD]``，``>=`` 剔除 ``(0, DPD)``，
        ``<``、``<=`` 不设置灰区。各标签独立计算有效样本分母。
        默认 False；显式标签中非 0/1 值按 0 计算，逾期标签按比较结果计算。
    :type del_grey: bool
    :return: 单标签返回字典，多标签返回DataFrame

    **参考样例**

    >>> # 单标签分析
    >>> result = bad_rate_overall(df, target='fpd15')
    >>> print(result)
    {'样本总数': 10000, '好样本数': 8500, '坏样本数': 1500, '逾期率(%)': 15.0}

    >>> # 多标签分析
    >>> result = bad_rate_overall(df, overdue=['MOB1', 'MOB3'], dpds=[7, 30])
    >>> print(result)
         标签      样本总数  好样本数  坏样本数  逾期率(%)
    0  MOB1>7     9800    8820     980    10.00
    1  MOB1>30    9800    9400     400     4.08

    :param overdue_operator: 逾期标签比较符，支持 ``>`` 、 ``>=`` 、 ``<`` 、 ``<=`` ，默认 ``>`` 。
        满足比较条件记为坏样本(1)，否则为好样本(0)。
        ``del_grey=True`` 时， ``>`` 剔除 ``(0, dpd]`` ， ``>=`` 剔除 ``(0, dpd)`` ；
        ``<`` 和 ``<=`` 暂无灰客户，保留参数但不剔除样本。

    :type overdue_operator: str
    """

    validate_dataframe(df)
    frame, targets = bind_targets(df, target, overdue, dpds, del_grey, overdue_operator)
    rows = []
    for name, values in targets.y_map(frame, "plain").items():
        valid = values.dropna()
        total, bad = len(valid), int(valid.sum())
        rows.append(
            {
                "标签": name,
                "样本总数": total,
                "好样本数": total - bad,
                "坏样本数": bad,
                "逾期率(%)": round(bad / total * 100, 2) if total else np.nan,
            }
        )
    if targets.mode == "target" and not targets.explicit_multi and len(targets) == 1:
        return {key: value for key, value in rows[0].items() if key != "标签"}
    return pd.DataFrame(rows)


def bad_rate_by_dimension(
    df: pd.DataFrame,
    dim_col: str = None,
    target: Optional[TargetInput] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    del_grey: bool = False,
    sort_by: str = "bad_rate",
    *,
    segment_col: Optional[str] = None,
    overdue_operator: str = ">",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """分维度逾期率分析.

    支持单标签或多标签分析。

    :param df: 输入数据
    :type df: pd.DataFrame
    :param dim_col: 默认值为 ``None`` 。维度列名（如渠道、产品类型）
    :type dim_col: str
    :param target: 默认值为 ``None`` 。支持列名、一维 Series/ndarray/list、callable，
        或 ``{标签名: 标签数据或 callable}``；与 ``overdue+dpds`` 互斥。
        callable 接收当前完整 DataFrame；Series 按索引对齐，数组和列表按行位置对应。
    :type target: Optional[TargetInput]
    :param overdue: 默认值为 ``None`` 。逾期天数字段名或字段名序列；与 ``dpds`` 必须同时传入，
        与 ``target`` 互斥，标签方向由 ``overdue_operator`` 控制。
    :type overdue: Optional[Union[str, List[str]]]
    :param dpds: 默认值为 ``None`` 。逾期阈值标量或非空序列，支持整数或小数；
        必须与 ``overdue`` 同时传入。按 ``overdue × dpds`` 组合生成标签。
    :type dpds: Optional[Union[int, List[int]]]
    :param del_grey: 是否逐标签剔除灰客户。``target`` 模式剔除非 0/1 值（含缺失）；
        逾期模式下 ``>`` 剔除 ``(0, DPD]``，``>=`` 剔除 ``(0, DPD)``，
        ``<``、``<=`` 不设置灰区。各标签独立计算有效样本分母。
        默认 False；显式标签中非 0/1 值按 0 计算，逾期标签按比较结果计算。
    :type del_grey: bool
    :param sort_by: 默认值为 ``'bad_rate'`` 。排序方式，'bad_rate'或'count'
    :type sort_by: str
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典

    **参考样例**

    >>> # 单标签
    >>> result = bad_rate_by_dimension(df, 'channel', target='fpd15')

    >>> # 多标签
    >>> result = bad_rate_by_dimension(df, 'channel', overdue='MOB1', dpds=[7, 30])
    >>> print(result['MOB1>7'])

    :param overdue_operator: 逾期标签比较符，支持 ``>`` 、 ``>=`` 、 ``<`` 、 ``<=`` ，默认 ``>`` 。
        满足比较条件记为坏样本(1)，否则为好样本(0)。
        ``del_grey=True`` 时， ``>`` 剔除 ``(0, dpd]`` ， ``>=`` 剔除 ``(0, dpd)`` ；
        ``<`` 和 ``<=`` 暂无灰客户，保留参数但不剔除样本。

    :type overdue_operator: str

    :param segment_col: 用于分组比较坏率的维度字段名。 默认 ``None``。
    """

    dim_col = segment_col or dim_col
    if dim_col is None:
        raise ValueError("必须指定 dim_col 或 segment_col 参数")
    validate_dataframe(df, required_cols=[dim_col])
    frame, targets = bind_targets(df, target, overdue, dpds, del_grey, overdue_operator)
    results = {}
    for name, definition in zip(targets.names("plain"), targets.labels):
        valid = frame.loc[frame[definition.column].notna()]
        grouped = valid.groupby(dim_col)[definition.column].agg(["count", "sum", "mean"]).reset_index()
        grouped.columns = ["维度值", "样本数", "坏样本数", "逾期率"]
        grouped["好样本数"] = grouped["样本数"] - grouped["坏样本数"]
        grouped["逾期率(%)"] = (grouped.pop("逾期率") * 100).round(2)
        grouped["样本占比(%)"] = (grouped["样本数"] / len(valid) * 100).round(2) if len(valid) else np.nan
        grouped = grouped.sort_values("逾期率(%)" if sort_by == "bad_rate" else "样本数", ascending=False)
        results[name] = grouped.reset_index(drop=True)
    return results if len(results) > 1 else next(iter(results.values()))


@analysis_inputs("df", roles=("date",))
def bad_rate_trend(
    df: pd.DataFrame,
    date: str,
    target: Optional[TargetInput] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    del_grey: bool = False,
    freq: str = "M",
    dimensions: Optional[List[str]] = None,
    *,
    overdue_operator: str = ">",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """逾期率时间趋势分析.

    支持单标签或多标签分析。

    :param df: 输入数据
    :type df: pd.DataFrame
    :param date: 日期列名
    :type date: str
    :param target: 默认值为 ``None`` 。支持列名、一维 Series/ndarray/list、callable，
        或 ``{标签名: 标签数据或 callable}``；与 ``overdue+dpds`` 互斥。
        callable 接收当前完整 DataFrame；Series 按索引对齐，数组和列表按行位置对应。
    :type target: Optional[TargetInput]
    :param overdue: 默认值为 ``None`` 。逾期天数字段名或字段名序列；与 ``dpds`` 必须同时传入，
        与 ``target`` 互斥，标签方向由 ``overdue_operator`` 控制。
    :type overdue: Optional[Union[str, List[str]]]
    :param dpds: 默认值为 ``None`` 。逾期阈值标量或非空序列，支持整数或小数；
        必须与 ``overdue`` 同时传入。按 ``overdue × dpds`` 组合生成标签。
    :type dpds: Optional[Union[int, List[int]]]
    :param del_grey: 是否逐标签剔除灰客户。``target`` 模式剔除非 0/1 值（含缺失）；
        逾期模式下 ``>`` 剔除 ``(0, DPD]``，``>=`` 剔除 ``(0, DPD)``，
        ``<``、``<=`` 不设置灰区。各标签独立计算有效样本分母。
        默认 False；显式标签中非 0/1 值按 0 计算，逾期标签按比较结果计算。
    :type del_grey: bool
    :param freq: 默认值为 ``'M'`` 。时间频率，'D'日/'W'周/'M'月/'Q'季度
    :type freq: str
    :param dimensions: 默认值为 ``None`` 。分维度分析列表
    :type dimensions: Optional[List[str]]
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典

    **参考样例**

    >>> # 单标签
    >>> trend = bad_rate_trend(df, 'apply_date', target='fpd15', freq='M')

    >>> # 多标签
    >>> trend = bad_rate_trend(df, 'apply_date', overdue=['MOB1', 'MOB3'], dpds=30, freq='M')
    >>> print(trend['MOB1>30'])

    :param overdue_operator: 逾期标签比较符，支持 ``>`` 、 ``>=`` 、 ``<`` 、 ``<=`` ，默认 ``>`` 。
        满足比较条件记为坏样本(1)，否则为好样本(0)。
        ``del_grey=True`` 时， ``>`` 剔除 ``(0, dpd]`` ， ``>=`` 剔除 ``(0, dpd)`` ；
        ``<`` 和 ``<=`` 暂无灰客户，保留参数但不剔除样本。

    :type overdue_operator: str
    """

    validate_dataframe(df, required_cols=[date])
    frame, targets = bind_targets(df, target, overdue, dpds, del_grey, overdue_operator)
    dates = pd.to_datetime(frame[date])
    periods = date_groups(dates, freq)
    results = {}
    for name, values in targets.y_map(frame, "plain").items():
        grouped = values.groupby(periods).agg(["count", "sum", "mean"]).reset_index()
        grouped.columns = ["时间周期", "样本数", "坏样本数", "逾期率"]
        grouped["好样本数"] = grouped["样本数"] - grouped["坏样本数"]
        grouped["逾期率(%)"] = (grouped["逾期率"] * 100).round(2)
        grouped["环比变化(%)"] = grouped["逾期率(%)"].diff().round(2)
        results[name] = grouped[["时间周期", "样本数", "好样本数", "坏样本数", "逾期率(%)", "环比变化(%)"]]
    return results if len(results) > 1 else next(iter(results.values()))


def bad_rate_by_bins(
    df: pd.DataFrame,
    feature: str,
    target: Optional[TargetInput] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    del_grey: bool = False,
    max_n_bins: int = 10,
    method: str = "quantile",
    *,
    overdue_operator: str = ">",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """评分分箱逾期率分析.

    支持单标签或多标签分析。

    :param df: 输入数据
    :type df: pd.DataFrame
    :param feature: 评分列名
    :type feature: str
    :param target: 默认值为 ``None`` 。支持列名、一维 Series/ndarray/list、callable，
        或 ``{标签名: 标签数据或 callable}``；与 ``overdue+dpds`` 互斥。
        callable 接收当前完整 DataFrame；Series 按索引对齐，数组和列表按行位置对应。
    :type target: Optional[TargetInput]
    :param overdue: 默认值为 ``None`` 。逾期天数字段名或字段名序列；与 ``dpds`` 必须同时传入，
        与 ``target`` 互斥，标签方向由 ``overdue_operator`` 控制。
    :type overdue: Optional[Union[str, List[str]]]
    :param dpds: 默认值为 ``None`` 。逾期阈值标量或非空序列，支持整数或小数；
        必须与 ``overdue`` 同时传入。按 ``overdue × dpds`` 组合生成标签。
    :type dpds: Optional[Union[int, List[int]]]
    :param del_grey: 是否逐标签剔除灰客户。``target`` 模式剔除非 0/1 值（含缺失）；
        逾期模式下 ``>`` 剔除 ``(0, DPD]``，``>=`` 剔除 ``(0, DPD)``，
        ``<``、``<=`` 不设置灰区。各标签独立计算有效样本分母。
        默认 False；显式标签中非 0/1 值按 0 计算，逾期标签按比较结果计算。
    :type del_grey: bool
    :param max_n_bins: 默认值为 ``10`` 。分箱数
    :type max_n_bins: int
    :param method: 默认值为 ``'quantile'`` 。分箱方法，'quantile'等频/'uniform'等距
    :type method: str
    :return: 单标签返回DataFrame，多标签返回{标签名: DataFrame}字典

    **参考样例**

    >>> # 单标签
    >>> bins = bad_rate_by_bins(df, 'score', target='fpd15', max_n_bins=10)

    >>> # 多标签
    >>> bins = bad_rate_by_bins(df, 'score', overdue=['MOB1', 'MOB3'], dpds=30, max_n_bins=10)
    >>> print(bins['MOB1>30'])

    :param overdue_operator: 逾期标签比较符，支持 ``>`` 、 ``>=`` 、 ``<`` 、 ``<=`` ，默认 ``>`` 。
        满足比较条件记为坏样本(1)，否则为好样本(0)。
        ``del_grey=True`` 时， ``>`` 剔除 ``(0, dpd]`` ， ``>=`` 剔除 ``(0, dpd)`` ；
        ``<`` 和 ``<=`` 暂无灰客户，保留参数但不剔除样本。

    :type overdue_operator: str
    """

    validate_dataframe(df, required_cols=[feature])
    frame, targets = bind_targets(df, target, overdue, dpds, del_grey, overdue_operator)
    if frame[feature].notna().sum() == 0:
        return pd.DataFrame()
    bins = (
        pd.qcut(frame[feature], q=max_n_bins, duplicates="drop")
        if method == "quantile"
        else pd.cut(frame[feature], bins=max_n_bins)
    )
    results = {}
    for name, values in targets.y_map(frame, "plain").items():
        grouped = values.groupby(bins, observed=False).agg(["count", "sum", "mean"]).reset_index()
        grouped.columns = ["分箱区间", "样本数", "坏样本数", "逾期率"]
        grouped["好样本数"] = grouped["样本数"] - grouped["坏样本数"]
        grouped["逾期率(%)"] = (grouped["逾期率"] * 100).round(2)
        total = values.notna().sum()
        grouped["样本占比(%)"] = (grouped["样本数"] / total * 100).round(2) if total else np.nan
        grouped["提升度"] = (grouped["逾期率"] / values.mean()).round(4)
        grouped["分箱"] = range(1, len(grouped) + 1)
        results[name] = grouped[
            ["分箱", "分箱区间", "样本数", "好样本数", "坏样本数", "逾期率(%)", "样本占比(%)", "提升度"]
        ]
    return results if len(results) > 1 else next(iter(results.values()))


@analysis_inputs("df", roles=("date",))
def sample_distribution(
    df: pd.DataFrame, date: str, freq: str = "M", target: Optional[TargetInput] = None
) -> pd.DataFrame:
    """样本时间分布分析.

    :param df: 输入数据
    :type df: pd.DataFrame
    :param date: 日期列名
    :type date: str
    :param freq: 时间聚合频率， ``'D'`` 日 / ``'W'`` 周 / ``'M'`` 月 / ``'Q'`` 季度，默认 ``'M'``
    :type freq: str
    :param target: 默认值为 ``None`` 。目标变量列名（如有）
    :type target: Optional[TargetInput]
    :return: 样本分布DataFrame

    **参考样例**

    >>> dist = sample_distribution(df, 'apply_date', target='fpd15')
    >>> print(dist[['时间周期', '样本数', '坏样本数', '逾期率(%)']])
    """

    validate_dataframe(df, required_cols=[date])
    if target is not None:
        result = bad_rate_trend(df, date, target=target, freq=freq)

        def convert(table):
            table = table.drop(columns=["环比变化(%)"]).copy()
            table["样本环比(%)"] = (table["样本数"].pct_change() * 100).round(2)
            return table

        return {name: convert(table) for name, table in result.items()} if isinstance(result, dict) else convert(result)
    dates = pd.to_datetime(df[date])
    periods = date_groups(dates, freq)
    table = periods.groupby(periods).size().reset_index(name="样本数")
    table.columns = ["时间周期", "样本数"]
    table["样本环比(%)"] = (table["样本数"].pct_change() * 100).round(2)
    return table
