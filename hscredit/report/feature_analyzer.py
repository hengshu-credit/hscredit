"""特征分析模块.

提供特征分箱统计分析与自动化特征分析输出功能，支持多逾期标签、
多逾期天数组合分析以及 Excel 报告生成。
"""

import logging
import os
import traceback
from copy import deepcopy
from typing import Union, List, Dict, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.worksheet.worksheet import Worksheet
from tqdm import tqdm

from ..core.binning import OptimalBinning
from ..core.binning.base import BaseBinning
from ..core.viz import bin_plot, bin_trend_plot, corr_plot, distribution_plot, hist_plot, ks_plot
from ..core.metrics._binning import compute_bin_stats, add_margins
from ..excel import ExcelWriter, dataframe2excel
from ..utils import init_setting
from ._sample_stats import build_group_distribution_table, build_sample_stats_table
from .rule_strategy import GroupOrder, _resolve_group_labels

logger = logging.getLogger(__name__)


_BINNING_SUMMARY_METRICS = ('分档KS值', 'LIFT值', '指标IV值', '坏样本数', '坏样本率')


def _feature_missing_rate(data: pd.DataFrame, feature: str, dropna: Union[bool, float, int, str] = False) -> float:
    """计算变量缺失率，和 auto_feature_analysis 的剔除口径保持一致."""
    missing_mask = data[feature].isna()
    if isinstance(dropna, (float, int, str)):
        missing_mask = missing_mask | data[feature].eq(dropna)
    return float(missing_mask.mean()) if len(data) > 0 else 0.0


def _auto_feature_target_maps(
    data: pd.DataFrame,
    target: str,
    overdue: Optional[List[str]] = None,
    dpds: Optional[List[int]] = None,
) -> Tuple[str, List[str], Dict[str, str], Dict[str, np.ndarray]]:
    """生成自动特征分析使用的目标列、展示标签和标签数组."""
    if overdue:
        if dpds is None:
            raise ValueError("传入 overdue 参数时必须同时传入 dpds")
        primary_target = f"{overdue[0]} {dpds[0]}+"
        label_names: List[str] = []
        display_labels: Dict[str, str] = {}
        y_map: Dict[str, np.ndarray] = {}
        for mob_col in overdue:
            for dpd in dpds:
                label = f"{mob_col}>{dpd}"
                label_names.append(label)
                display_labels[label] = f"{mob_col}@{dpd}"
                y_map[label] = (data[mob_col] > dpd).astype(int).to_numpy()
        return primary_target, label_names, display_labels, y_map

    return target, [target], {target: target}, {target: data[target].astype(int).to_numpy()}


# feature_binning_summary 支持的摘要指标及其跨分箱聚合方式（取自 feature_bin_stats 的输出列）。
# 聚合方式说明：
#   'sum'      —— 对各分箱（不含合计行）求和
#   'max'      —— 取各分箱的最大值
#   'max_abs'  —— 取各分箱绝对值的最大值
#   'bad_rate' —— 按 sum(坏样本数) / sum(样本总数) 重新计算总体坏样本率
_BINNING_SUMMARY_AGG = {
    '样本总数': 'sum',
    '好样本数': 'sum',
    '坏样本数': 'sum',
    '样本占比': 'sum',
    '好样本占比': 'sum',
    '坏样本占比': 'sum',
    '坏样本率': 'bad_rate',
    '分档WOE值': 'max_abs',
    '分档IV值': 'sum',
    '指标IV值': 'max',
    'LIFT值': 'max',
    '坏账改善': 'max',
    '风险拒绝比': 'max',
    '累积LIFT值': 'max',
    '累积坏账改善': 'max',
    '累计风险拒绝比': 'max',
    '累积好样本数': 'max',
    '累积坏样本数': 'max',
    '分档KS值': 'max',
}


def _normalize_binning_summary_metrics(metrics: Union[str, List[str]]) -> List[str]:
    """标准化摘要指标并保持传入顺序（去重、校验是否受支持）。"""
    normalized = [metrics] if isinstance(metrics, str) else list(metrics)
    if not normalized:
        raise ValueError("metrics 不能为空")

    invalid = [metric for metric in normalized if metric not in _BINNING_SUMMARY_AGG]
    if invalid:
        raise ValueError(f"不支持的metrics: {invalid}，可选: {list(_BINNING_SUMMARY_AGG)}")

    seen = set()
    deduped = []
    for metric in normalized:
        if metric not in seen:
            seen.add(metric)
            deduped.append(metric)
    return deduped


# 长格式（逾期标签按行堆叠）分箱表的标准列顺序
_LONG_FORMAT_COLUMNS = (
    '指标名称', '指标含义', '逾期标签', '分箱标签',
    '样本总数', '样本占比', '好样本数', '坏样本数',
    '好样本占比', '坏样本占比', '坏样本率',
    '分档WOE值', '分档IV值', '指标IV值',
    'LIFT值', '坏账改善', '风险拒绝比',
    '累积LIFT值', '累积坏账改善', '累计风险拒绝比',
    '累积好样本数', '累积坏样本数', '分档KS值',
)


def _summary_display_name(target_name: Any) -> Any:
    """将目标名称转换为摘要展示名称（``MOB1_3+`` → ``MOB1@3``）。"""
    if isinstance(target_name, str) and target_name.endswith('+') and '_' in target_name:
        overdue_name, dpd = target_name[:-1].rsplit('_', 1)
        return f'{overdue_name}@{dpd}'
    return target_name


def _reorder_long_format_columns(table: pd.DataFrame) -> pd.DataFrame:
    """按 :data:`_LONG_FORMAT_COLUMNS` 重排长格式分箱表列顺序，保留未列出的额外列。"""
    ordered = [col for col in _LONG_FORMAT_COLUMNS if col in table.columns]
    extras = [col for col in table.columns if col not in _LONG_FORMAT_COLUMNS]
    return table[ordered + extras]


def _normalize_binning_summary_methods(methods: Union[str, List[str]]) -> List[str]:
    """标准化分箱方法并保持传入顺序。"""
    normalized = [methods] if isinstance(methods, str) else list(methods)
    if not normalized:
        raise ValueError("methods 不能为空")

    invalid = [method for method in normalized if method not in OptimalBinning.VALID_METHODS]
    if invalid:
        raise ValueError(f"不支持的methods: {invalid}，可选: {OptimalBinning.VALID_METHODS}")
    if len(normalized) != len(set(normalized)):
        raise ValueError("methods 不能包含重复的分箱方法")
    return normalized


def _normalize_binning_summary_params(
    methods: List[str],
    bin_params: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """将全局或按方法配置的分箱参数展开为方法参数字典。"""
    if bin_params is None:
        return {method: {} for method in methods}
    if not isinstance(bin_params, dict):
        raise TypeError("bin_params 必须是字典")

    method_keys = set(bin_params).intersection(OptimalBinning.VALID_METHODS)
    if method_keys:
        unknown = set(bin_params).difference(OptimalBinning.VALID_METHODS)
        if unknown or any(not isinstance(value, dict) for value in bin_params.values()):
            raise ValueError("多层 bin_params 必须使用 method 作为 key、字典作为 value")
        return {method: dict(bin_params.get(method, {})) for method in methods}

    return {method: dict(bin_params) for method in methods}


def _summary_target_columns(table: pd.DataFrame) -> List[Tuple[str, str]]:
    """返回分箱表中的目标名称及其摘要展示名称。"""
    if not isinstance(table.columns, pd.MultiIndex):
        return [('', '')]

    targets = []
    for target_name in table.columns.get_level_values(0):
        if target_name == '分箱详情' or target_name in [item[0] for item in targets]:
            continue
        targets.append((target_name, _summary_display_name(target_name)))
    return targets


def _aggregate_summary_metrics(valid: pd.DataFrame, resolve, metrics: List[str]) -> Dict[str, float]:
    """按 :data:`_BINNING_SUMMARY_AGG` 定义的聚合方式汇总单个目标的指标。

    :param valid: 已剔除合计行的分箱明细
    :param resolve: 指标名 → 实际列键 的映射函数（兼容单层/多级表头）
    :param metrics: 需要汇总的指标列表
    """
    aggregated: Dict[str, float] = {}
    for metric in metrics:
        kind = _BINNING_SUMMARY_AGG[metric]
        if kind == 'bad_rate':
            bad_count = pd.to_numeric(valid[resolve('坏样本数')], errors='coerce').sum()
            sample_count = pd.to_numeric(valid[resolve('样本总数')], errors='coerce').sum()
            aggregated[metric] = bad_count / sample_count if sample_count else np.nan
            continue
        series = pd.to_numeric(valid[resolve(metric)], errors='coerce')
        if kind == 'sum':
            aggregated[metric] = series.sum()
        elif kind == 'max_abs':
            aggregated[metric] = series.abs().max()
        else:  # 'max'
            aggregated[metric] = series.max()
    return aggregated


def _summarize_binning_table(
    table: pd.DataFrame,
    single_target_name: str = '',
    metrics: Union[str, List[str]] = _BINNING_SUMMARY_METRICS,
) -> Dict[Tuple[str, str], float]:
    """按目标汇总一个特征、一个方法的分箱结果。"""
    metrics = _normalize_binning_summary_metrics(metrics)
    multi_columns = isinstance(table.columns, pd.MultiIndex)
    result: Dict[Tuple[str, str], float] = {}

    # 长格式：逾期标签按行堆叠，按 ``逾期标签`` 分组逐目标汇总
    if not multi_columns and '逾期标签' in table.columns:
        for target_name, group in table.groupby('逾期标签', sort=False):
            valid = group.loc[group['分箱标签'].astype(str) != '合计']
            display_name = _summary_display_name(target_name)
            for metric, value in _aggregate_summary_metrics(valid, lambda m: m, metrics).items():
                result[(metric, display_name)] = value
        return result

    label_column = ('分箱详情', '分箱标签') if multi_columns else '分箱标签'
    valid = table.loc[table[label_column].astype(str) != '合计']

    target_columns = _summary_target_columns(table)
    if not multi_columns:
        target_columns = [('', single_target_name)]

    for target_name, display_name in target_columns:
        def resolve(metric: str, _target=target_name):
            # 单层表头直接取列名；多级表头优先取目标层，样本数等公共列回落到"分箱详情"
            if not multi_columns:
                return metric
            if (_target, metric) in valid.columns:
                return (_target, metric)
            if ('分箱详情', metric) in valid.columns:
                return ('分箱详情', metric)
            return (_target, metric)

        for metric, value in _aggregate_summary_metrics(valid, resolve, metrics).items():
            result[(metric, display_name)] = value
    return result


def feature_binning_summary(
    data: pd.DataFrame,
    feature: Union[str, List[str]],
    methods: Union[str, List[str]] = 'mdlp',
    bin_params: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    desc: Optional[Union[str, Dict[str, str]]] = None,
    max_n_bins: int = 5,
    min_n_bins: int = 2,
    min_bin_size: float = 0.05,
    max_bin_size: Optional[Union[float, int]] = None,
    min_bad_rate: float = 0.0,
    missing_separate: bool = True,
    prebinning: Optional[Union[str, BaseBinning, Dict]] = None,
    prebinning_params: Optional[Dict[str, Any]] = None,
    special_codes: Optional[List] = None,
    cat_cutoff: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    decimal: int = 4,
    woe_clip: Optional[float] = None,
    del_grey: bool = False,
    margins: bool = False,
    amount: Optional[str] = None,
    verbose: int = 0,
    monotonic: Optional[Union[str, bool]] = None,
    long_format: bool = False,
    metrics: Union[str, List[str]] = _BINNING_SUMMARY_METRICS,
    **kwargs,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    """对一个或多个字段执行多种分箱，并生成跨方法摘要。

    ``bin_params`` 支持两种格式：单层参数应用于所有方法；以 method 为 key、
    参数字典为 value 的多层参数仅应用于对应方法。参数优先级为
    ``bin_params > 显式公共参数/kwargs``。

    ``long_format`` 控制 ``binning_tables`` 中每张分箱表的输出格式：默认 ``False``
    沿用原多级表头样式（多目标时按列展开）；设为 ``True`` 时每张表按 ``逾期标签``
    列将各目标纵向堆叠输出（透传给 :func:`feature_bin_stats`）。摘要 ``binning_summary``
    的结构不受影响。

    ``metrics`` 指定 ``binning_summary`` 中按目标汇总的指标及其展示顺序，默认
    ``['分档KS值', 'LIFT值', '指标IV值', '坏样本数', '坏样本率']``。可选值为
    :func:`feature_bin_stats` 输出的指标列，各指标的跨分箱聚合方式如下（合计行不参与）：

    - **求和（sum）**：``样本总数`` / ``好样本数`` / ``坏样本数`` / ``分档IV值`` /
      ``样本占比`` / ``好样本占比`` / ``坏样本占比``
    - **取最大值（max）**：``指标IV值`` / ``LIFT值`` / ``坏账改善`` / ``风险拒绝比`` /
      ``累积LIFT值`` / ``累积坏账改善`` / ``累计风险拒绝比`` / ``累积好样本数`` /
      ``累积坏样本数`` / ``分档KS值``
    - **取绝对值最大值（max_abs）**：``分档WOE值``
    - **总体坏样本率（bad_rate）**：``坏样本率`` 按 ``sum(坏样本数) / sum(样本总数)`` 重新计算

    传入不受支持的指标将抛出 ``ValueError``。

    :return: ``(binning_tables, binning_summary)``。分箱表结构为
        ``{feature: {method: binning_table}}``，摘要使用两级列索引。

    **参考样例**

    >>> tables, summary = feature_binning_summary(
    ...     data, ['score', 'age'], methods=['quantile', 'mdlp'],
    ...     overdue='MOB1', dpds=[3, 1, 0], max_n_bins=5,
    ...     bin_params={'mdlp': {'min_bin_size': 0.1}},
    ... )
    """
    features = [feature] if isinstance(feature, str) else list(feature)
    if not features:
        raise ValueError("feature 不能为空")
    missing_features = [name for name in features if name not in data.columns]
    if missing_features:
        raise KeyError(f"数据中不存在字段: {missing_features}")

    normalized_methods = _normalize_binning_summary_methods(methods)
    normalized_metrics = _normalize_binning_summary_metrics(metrics)
    per_method_params = _normalize_binning_summary_params(normalized_methods, bin_params)
    common_params = {
        'target': target,
        'overdue': overdue,
        'dpds': dpds,
        'desc': desc,
        'max_n_bins': max_n_bins,
        'min_n_bins': min_n_bins,
        'min_bin_size': min_bin_size,
        'max_bin_size': max_bin_size,
        'min_bad_rate': min_bad_rate,
        'missing_separate': missing_separate,
        'prebinning': prebinning,
        'prebinning_params': prebinning_params,
        'special_codes': special_codes,
        'cat_cutoff': cat_cutoff,
        'random_state': random_state,
        'decimal': decimal,
        'woe_clip': woe_clip,
        'del_grey': del_grey,
        'margins': margins,
        'amount': amount,
        'verbose': verbose,
        'monotonic': monotonic,
        'long_format': long_format,
        **kwargs,
    }

    binning_tables = {name: {} for name in features}
    summary_rows = []
    single_target_name = target or ''
    if overdue is not None:
        overdue_values = [overdue] if isinstance(overdue, str) else list(overdue)
        dpd_values = [dpds] if isinstance(dpds, int) else list(dpds or [])
        target_labels = [f'{overdue_name}@{dpd}' for overdue_name in overdue_values for dpd in dpd_values]
        if len(target_labels) == 1:
            single_target_name = target_labels[0]

    for method in normalized_methods:
        method_params = {**common_params, **per_method_params[method]}
        for reserved in ('data', 'feature', 'method', 'return_rules'):
            method_params.pop(reserved, None)

        for name in features:
            table = feature_bin_stats(data=data, feature=name, method=method, **method_params)
            binning_tables[name][method] = table
            row = {('分箱详情', '分箱方法'): method, ('分箱详情', '指标名称'): name}
            row.update(_summarize_binning_table(table, single_target_name=single_target_name, metrics=normalized_metrics))
            summary_rows.append(row)

    binning_summary = pd.DataFrame(summary_rows)
    ordered_columns = [('分箱详情', '分箱方法'), ('分箱详情', '指标名称')]
    target_names = []
    for row in summary_rows:
        for metric, target_name in row:
            if metric in normalized_metrics and target_name not in target_names:
                target_names.append(target_name)
    ordered_columns.extend((metric, target_name) for metric in normalized_metrics for target_name in target_names)
    binning_summary = binning_summary.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    return binning_tables, binning_summary


def _fit_group_summary_binner(
    data: pd.DataFrame,
    feature: str,
    method: str,
    params: Dict[str, Any],
) -> BaseBinning:
    """在全量数据上拟合分组分析使用的统一分箱器。"""
    overdue = params.get('overdue')
    dpds = params.get('dpds')
    target = params.get('target')
    del_grey = bool(params.get('del_grey', False))

    if overdue is not None:
        if dpds is None:
            raise ValueError("传入 overdue 参数时必须同时传入 dpds")
        overdue_col = overdue if isinstance(overdue, str) else list(overdue)[0]
        dpd = dpds if isinstance(dpds, int) else list(dpds)[0]
        train_data = data[[feature, overdue_col]].copy()
        y_train = (train_data[overdue_col] > dpd).astype(int)
        if del_grey:
            mask = (train_data[overdue_col] > dpd) | (train_data[overdue_col] == 0)
            train_data = train_data.loc[mask]
            y_train = y_train.loc[mask]
    elif target is not None:
        train_data = data[[feature, target]].copy()
        y_train = train_data[target]
    else:
        raise ValueError("必须传入 target 或 overdue+dpds 参数")

    binner_param_names = {
        'max_n_bins', 'min_n_bins', 'min_bin_size', 'max_bin_size', 'min_bad_rate',
        'missing_separate', 'prebinning', 'prebinning_params', 'special_codes',
        'cat_cutoff', 'random_state', 'decimal', 'woe_clip', 'verbose', 'monotonic',
    }
    binner_params = {name: params[name] for name in binner_param_names if name in params}
    stats_only_params = {
        'target', 'overdue', 'dpds', 'desc', 'del_grey', 'margins', 'amount',
        'long_format', 'return_cols', 'return_rules', 'binner', 'rules',
    }
    binner_params.update(
        {
            name: value
            for name, value in params.items()
            if name not in binner_param_names and name not in stats_only_params
        }
    )

    if method == 'mdlp':
        binner_params.setdefault('lift_refine', True)
        binner_params.setdefault('lift_focus_weight', 3.0)
        binner_params.setdefault('sample_stability_weight', 0.2)
        binner_params.setdefault('monotonic_bonus_weight', 0.4)
        binner_params.setdefault('lift_refine_max_bins', binner_params.get('max_n_bins', 5))
    elif method == 'quantile':
        binner_params.setdefault('lift_refine', False)
        binner_params.setdefault('min_bin_size', 0)

    binner = OptimalBinning(method=method, **binner_params)
    binner.fit(train_data[[feature]], y_train)
    return binner


def feature_group_binning_summary(
    data: pd.DataFrame,
    feature: Union[str, List[str]],
    methods: Union[str, List[str]] = 'mdlp',
    date_col: Optional[str] = None,
    freq: str = 'M',
    group_col: Optional[str] = None,
    group_order: GroupOrder = 'asc',
    dropna: bool = True,
    bin_params: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    desc: Optional[Union[str, Dict[str, str]]] = None,
    max_n_bins: int = 5,
    min_n_bins: int = 2,
    min_bin_size: float = 0.05,
    max_bin_size: Optional[Union[float, int]] = None,
    min_bad_rate: float = 0.0,
    missing_separate: bool = True,
    prebinning: Optional[Union[str, BaseBinning, Dict]] = None,
    prebinning_params: Optional[Dict[str, Any]] = None,
    special_codes: Optional[List] = None,
    cat_cutoff: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    decimal: int = 4,
    woe_clip: Optional[float] = None,
    del_grey: bool = False,
    margins: bool = False,
    amount: Optional[str] = None,
    verbose: int = 0,
    monotonic: Optional[Union[str, bool]] = None,
    long_format: bool = False,
    metrics: Union[str, List[str]] = _BINNING_SUMMARY_METRICS,
    **kwargs,
) -> Tuple[Dict[str, Dict[str, Dict[str, pd.DataFrame]]], pd.DataFrame]:
    """统计日期周期或类别分组下的特征分箱效果。

    分箱器在全量数据上拟合一次，各分组复用同一套分箱规则，因此不同日期周期或
    类别分组下的坏样本率、LIFT、KS、IV 等指标可以直接横向比较。

    :param data: 原始明细数据
    :param feature: 待分析特征名或特征名列表
    :param methods: 分箱方法或方法列表
    :param date_col: 日期字段，与 ``freq`` 配合生成时间分组；与 ``group_col`` 二选一
    :param freq: 日期频率，支持 ``D`` / ``W`` / ``M`` / ``Q``，默认按月
    :param group_col: 类别分组字段；与 ``date_col`` 二选一
    :param group_order: 分组顺序，支持升序、降序、出现顺序、排序函数或显式列表
    :param dropna: 是否删除分组字段缺失样本；为 False 时归入“缺失”组
    :param bin_params: 全局或按分箱方法配置的参数，规则与
        :func:`feature_binning_summary` 一致
    :param metrics: summary 汇总指标，规则与 :func:`feature_binning_summary` 一致
    :return: ``(binning_tables, binning_summary)``。分箱表结构为
        ``{feature: {method: {group: binning_table}}}``；summary 使用两级列索引。

    **参考样例**

    >>> tables, summary = feature_group_binning_summary(
    ...     data, feature='score', methods=['quantile', 'mdlp'],
    ...     date_col='申请日期', freq='M', overdue='MOB1', dpds=[3, 0],
    ... )
    >>> category_tables, category_summary = feature_group_binning_summary(
    ...     data, feature='score', group_col='商品类别', target='FPD',
    ... )
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data 必须是非空的 DataFrame")

    features = [feature] if isinstance(feature, str) else list(feature)
    if not features:
        raise ValueError("feature 不能为空")
    missing_features = [name for name in features if name not in data.columns]
    if missing_features:
        raise KeyError(f"数据中不存在字段: {missing_features}")

    labels, ordered_groups = _resolve_group_labels(data, date_col, freq, group_col, dropna, group_order)
    if not ordered_groups:
        raise ValueError("根据 date_col/group_col 未能切分出任何有效分组")

    normalized_methods = _normalize_binning_summary_methods(methods)
    normalized_metrics = _normalize_binning_summary_metrics(metrics)
    per_method_params = _normalize_binning_summary_params(normalized_methods, bin_params)
    common_params = {
        'target': target,
        'overdue': overdue,
        'dpds': dpds,
        'desc': desc,
        'max_n_bins': max_n_bins,
        'min_n_bins': min_n_bins,
        'min_bin_size': min_bin_size,
        'max_bin_size': max_bin_size,
        'min_bad_rate': min_bad_rate,
        'missing_separate': missing_separate,
        'prebinning': prebinning,
        'prebinning_params': prebinning_params,
        'special_codes': special_codes,
        'cat_cutoff': cat_cutoff,
        'random_state': random_state,
        'decimal': decimal,
        'woe_clip': woe_clip,
        'del_grey': del_grey,
        'margins': margins,
        'amount': amount,
        'verbose': verbose,
        'monotonic': monotonic,
        'long_format': long_format,
        **kwargs,
    }

    group_field = group_col or f'{date_col}@{freq}'
    binning_tables: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {
        name: {method: {} for method in normalized_methods} for name in features
    }
    summary_rows = []
    single_target_name = target or ''
    if overdue is not None:
        overdue_values = [overdue] if isinstance(overdue, str) else list(overdue)
        dpd_values = [dpds] if isinstance(dpds, int) else list(dpds or [])
        target_labels = [f'{overdue_name}@{dpd}' for overdue_name in overdue_values for dpd in dpd_values]
        if len(target_labels) == 1:
            single_target_name = target_labels[0]

    for method in normalized_methods:
        method_params = {**common_params, **per_method_params[method]}
        for reserved in ('data', 'feature', 'method', 'return_rules', 'binner'):
            method_params.pop(reserved, None)

        for name in features:
            fitted_binner = _fit_group_summary_binner(data, name, method, method_params)
            for group in ordered_groups:
                subset = data.loc[labels == group]
                if subset.empty:
                    continue
                group_name = str(group)
                table = feature_bin_stats(
                    data=subset,
                    feature=name,
                    method=method,
                    binner=fitted_binner,
                    **method_params,
                )
                binning_tables[name][method][group_name] = table
                row = {
                    ('分箱详情', '分箱方法'): method,
                    ('分箱详情', '指标名称'): name,
                    ('分箱详情', '分组字段'): group_field,
                    ('分箱详情', '分组'): group_name,
                }
                row.update(
                    _summarize_binning_table(
                        table,
                        single_target_name=single_target_name,
                        metrics=normalized_metrics,
                    )
                )
                summary_rows.append(row)

    binning_summary = pd.DataFrame(summary_rows)
    ordered_columns = [
        ('分箱详情', '分箱方法'),
        ('分箱详情', '指标名称'),
        ('分箱详情', '分组字段'),
        ('分箱详情', '分组'),
    ]
    target_names = []
    for row in summary_rows:
        for metric, target_name in row:
            if metric in normalized_metrics and target_name not in target_names:
                target_names.append(target_name)
    ordered_columns.extend((metric, target_name) for metric in normalized_metrics for target_name in target_names)
    binning_summary = binning_summary.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    return binning_tables, binning_summary


def _create_bin_table(
    bins: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    desc: str = "",
    splits: Optional[np.ndarray] = None,
    amount: Optional[np.ndarray] = None,
    bin_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """创建分箱统计表。"""
    if bin_labels is None:
        bin_labels = _get_bin_labels(splits, bins)

    if amount is not None:
        stats = compute_bin_stats(
            bins,
            y,
            target_type='amount_weighted',
            amount=amount,
            bin_labels=bin_labels,
        )
    else:
        stats = compute_bin_stats(bins, y, target_type='binary', bin_labels=bin_labels)

    stats.insert(0, '指标含义', desc if desc else feature_name)
    stats.insert(0, '指标名称', feature_name)

    if '分箱' in stats.columns:
        stats = stats.drop(columns=['分箱'])

    return stats


def _get_binner_bin_labels(
    binner: BaseBinning,
    feature: str,
    bins: np.ndarray,
    fallback_splits: Optional[np.ndarray] = None,
) -> List[str]:
    """按 ``bins`` 的唯一值顺序生成分箱标签，优先复用分箱器内部标签。"""
    unique_bins = np.unique(bins)
    label_map: Dict[int, str] = {}

    table = getattr(binner, 'bin_tables_', {}).get(feature)
    if table is not None and '分箱' in table.columns and '分箱标签' in table.columns:
        label_map = dict(zip(table['分箱'].astype(int), table['分箱标签'].astype(str)))

    labels: List[str] = []
    for bin_value in unique_bins:
        bin_int = int(bin_value)
        if bin_int in label_map:
            labels.append(label_map[bin_int])
        elif bin_int == -1:
            labels.append('missing')
        elif bin_int == -2:
            labels.append('special')
        else:
            labels.append(f'bin_{bin_int}')

    if label_map:
        return labels

    return _get_bin_labels(fallback_splits, bins)


def _get_bin_labels(splits: Optional[np.ndarray], bins: np.ndarray) -> List[str]:
    """根据切分点生成分箱标签。"""
    unique_bins = np.unique(bins)

    if splits is None or len(splits) == 0:
        labels = []
        for current_bin in unique_bins:
            bin_value = int(current_bin)
            if bin_value == -1:
                labels.append('missing')
            elif bin_value == -2:
                labels.append('special')
            else:
                labels.append('[-inf, +inf)')
        return labels

    if isinstance(splits, list):
        labels = []
        for current_bin in unique_bins:
            bin_value = int(current_bin)
            if bin_value == -1:
                labels.append('missing')
            elif bin_value == -2:
                labels.append('special')
            elif 0 <= bin_value < len(splits):
                group = splits[bin_value]
                if isinstance(group, list):
                    labels.append(','.join(str(item) for item in group))
                else:
                    labels.append(str(group))
            else:
                labels.append(f'bin_{bin_value}')
        return labels

    # 过滤掉 NaN split points，用真实切分点生成分箱标签
    real_splits = splits[~np.isnan(splits)]
    n_real_splits = len(real_splits)
    labels = []
    for current_bin in unique_bins:
        bin_value = int(current_bin)
        if bin_value == -1:
            labels.append('missing')
        elif bin_value == -2:
            labels.append('special')
        elif bin_value == 0:
            labels.append(f'[-inf, {real_splits[0]})')
        elif bin_value >= n_real_splits:
            if n_real_splits == 0:
                labels.append('[-inf, +inf)')
            else:
                labels.append(f'[{real_splits[-1]}, +inf)')
        else:
            labels.append(f'[{real_splits[bin_value - 1]}, {real_splits[bin_value]})')

    return labels


def _merge_multi_target_tables(
    tables: List[pd.DataFrame],
    target_names: List[str],
    merge_columns: List[str],
) -> pd.DataFrame:
    """合并多目标的分箱表。"""
    if not tables:
        return pd.DataFrame()

    if len(tables) == 1:
        return tables[0]

    base_table = tables[0].copy()
    available_merge_cols = [column for column in merge_columns if column in base_table.columns]
    non_merge_cols = [column for column in base_table.columns if column not in available_merge_cols]
    base_table = base_table[available_merge_cols + non_merge_cols]

    multi_cols = []
    for column in base_table.columns:
        if column in available_merge_cols:
            multi_cols.append(('分箱详情', column))
        else:
            multi_cols.append((target_names[0], column))
    base_table.columns = pd.MultiIndex.from_tuples(multi_cols)

    for table, target_name in zip(tables[1:], target_names[1:]):
        table_multi_cols = []
        for column in table.columns:
            if column in available_merge_cols:
                table_multi_cols.append(('分箱详情', column))
            else:
                table_multi_cols.append((target_name, column))
        table_copy = table.copy()
        table_copy.columns = pd.MultiIndex.from_tuples(table_multi_cols)

        merge_on = [('分箱详情', column) for column in available_merge_cols]
        base_table = base_table.merge(table_copy, on=merge_on)

    return base_table


def feature_bin_stats(
    data: pd.DataFrame,
    feature: Union[str, List[str]],
    target: Optional[str] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, List[int]]] = None,
    rules: Optional[Union[List, Dict[str, List]]] = None,
    method: str = 'mdlp',
    desc: Optional[Union[str, Dict[str, str]]] = None,
    binner: Optional[Union[BaseBinning, Dict[str, BaseBinning]]] = None,
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    prebinning: Optional[Union[str, BaseBinning, Dict]] = None,  # 默认禁用预分箱，保证分位数分箱准确性
    prebinning_params: Optional[Dict[str, Any]] = None,
    return_cols: Optional[List[str]] = None,
    return_rules: bool = False,
    del_grey: bool = False,
    margins: bool = False,
    amount: Optional[str] = None,
    verbose: int = 0,
    monotonic: Optional[Union[str, bool]] = None,
    long_format: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """特征分箱统计表，汇总统计特征每个分箱的各项指标信息.
    
    支持单特征或多特征，支持单目标或多逾期标签+逾期天数组合分析。
    当传入 overdue 和 dpds 时，会生成多级表头展示不同标签组合下的分箱统计。
    
    :param data: 数据集
    :param feature: 特征名称或特征名称列表
    :param target: 目标变量名称，默认 None
    :param overdue: 逾期天数字段名称或列表，如 'MOB1' 或 ['MOB1', 'MOB3']
    :param dpds: 逾期定义天数或列表，如 7 或 [0, 7, 30]
        - 逾期天数 > dpds 为坏样本(1)，其他为好样本(0)
    :param rules: 自定义分箱规则，支持 list（所有特征统一规则）或 dict（按特征名映射规则）。
        对 rules 中未包含的特征，按 method 参数重新训练分箱器。
        优先级: binner > rules > method
    :param method: 分箱方法，可选：
        - 基础方法: 'uniform'(等宽), 'quantile'(等频), 'tree'(决策树), 'chi'(卡方)
        - 优化方法: 'best_ks'(最优KS), 'best_iv'(最优IV), 'mdlp'(信息论)
        - 运筹规划方法: 'or_tools'(OR-Tools整数规划，需安装 ortools)
        - 高级方法: 'cart'(CART), 'monotonic'(单调性), 'genetic'(遗传算法),
                    'smooth'(平滑), 'kernel_density'(核密度), 
                    'best_lift'(Best Lift), 'target_bad_rate'(目标坏样本率)
        - 聚类方法: 'kmeans'
        默认: 'mdlp'
    :param desc: 特征描述，支持 str（单个特征）或 dict（多个特征）
    :param binner: 分箱器，支持以下三种传入方式：
        - BaseBinning（已训练）: 对其中已包含的特征直接使用，未包含的特征按 method 参数重新训练
        - BaseBinning（未训练）: 作为模板，对每个特征 deepcopy 后 fit
        - Dict[str, BaseBinning]: 按特征名映射的已训练分箱器字典，未包含的特征按 method 参数重新训练
        优先级: binner > rules > method
    :param max_n_bins: 最大分箱数，默认 5
    :param min_bin_size: 每箱最小样本占比，默认 0.05
    :param missing_separate: 是否将缺失值单独分箱，默认 True
    :param prebinning: 预分箱配置，参数格式与 OptimalBinning 保持一致，默认 'quantile'。
        - None: 不使用预分箱
        - str: 预分箱方法名（如 'quantile' / 'tree'）
        - BaseBinning: 预分箱器实例
        - Dict: 预分箱配置字典
    :param prebinning_params: 预分箱参数（传给 OptimalBinning.prebinning_params）。
        默认 None，此时会使用 {'max_n_bins': 100}，即先等频100箱再合并。
    :param return_cols: 指定返回的列名列表，默认返回所有列
    :param return_rules: 是否返回分箱规则，默认 False
    :param del_grey: 是否删除逾期天数 (0, dpds] 的灰样本，仅 overdue 起作用时有用
        - True: 剔除灰样本，不同目标下样本数不同，样本数相关列按目标单独显示
        - False: 保留灰样本，不同目标下样本数相同，样本数相关列作为公共列
    :param margins: 是否在分箱表最后添加合计行，默认 False
        - True: 在最后一行显示合计，缺失值和特殊值放在正常分箱之后、合计之前
    :param amount: 金额字段名称，用于金额口径分析。传入后会增加金额总数、金额占比等指标
    :param verbose: 是否输出详细信息，默认 0
    :param monotonic: 单调性约束，控制分箱后坏样本率的单调方向，透传给 OptimalBinning。可选值：
        - None: 不强制单调性约束
        - 'auto_asc': 自动判断并强制单调递增
        - 'auto_desc': 自动判断并强制单调递减
        - 'auto_asc_desc': 自动选择最优方向（递增/递减），默认选项
        - 'peak': 先升后降，适用于评分类特征
        - 'valley': 先降后升
        - bool: True=强制升序，False=强制降序
        注意：需配合 method 参数使用，部分 method 默认已包含单调约束（如 'monotonic' 方法）
    :param long_format: 分箱表输出格式，默认 False
        - False: 沿用原样式。多目标时使用多级表头（``分箱详情`` + 各逾期标签）按列展开
        - True: 长格式输出。各逾期标签纵向堆叠，新增 ``逾期标签`` 列标识目标，
          列顺序为 指标名称/指标含义/逾期标签/分箱标签/样本总数/样本占比/好样本数/坏样本数/...
          单目标时同样会输出 ``逾期标签`` 列。``margins=True`` 时按各逾期标签分组分别追加合计行
    :param kwargs: 其他分箱器参数（如 lift_refine、prebinning 等）
    
    :return: 
        - pd.DataFrame: 特征分箱统计表
        - Tuple[pd.DataFrame, Dict]: 当 return_rules=True 时返回 (统计表, 分箱规则)
    
    **参考样例**
    
    >>> # 单特征单目标分析
    >>> table = feature_bin_stats(data, 'score', target='target', method='mdlp')
    >>> 
    >>> # 单特征多逾期标签分析
    >>> table = feature_bin_stats(data, 'score', overdue=['MOB1', 'MOB3'], dpds=[0, 7])
    >>> 
    >>> # 多特征分析
    >>> table = feature_bin_stats(data, ['score', 'age'], overdue='MOB1', dpds=7)
    >>> 
    >>> # 使用自定义分箱规则
    >>> table = feature_bin_stats(data, 'score', rules=[300, 500, 700])
    >>> 
    >>> # 使用单调性分箱
    >>> table = feature_bin_stats(data, 'score', method='mdlp', monotonic='peak')
    >>>
    >>> # 使用单调性约束 + 强制升序
    >>> table = feature_bin_stats(data, 'score', method='mdlp', monotonic='auto_asc')
    >>>
    >>> # 直接使用 monotonic 方法
    >>> table = feature_bin_stats(data, 'score', method='monotonic', monotonic='peak')
    >>> 
    >>> # 金额口径分析
    >>> table = feature_bin_stats(data, 'score', target='target', amount='loan_amount')
    >>>
    >>> # 长格式输出：多逾期标签纵向堆叠，新增"逾期标签"列
    >>> table = feature_bin_stats(data, 'score', overdue='MOB1', dpds=[15, 0], long_format=True)
    """
    # 统一处理 feature 参数
    if isinstance(feature, str):
        features = [feature]
    else:
        features = feature
    
    # 统一处理 desc 参数
    if desc is None:
        desc_dict = {f: f for f in features}
    elif isinstance(desc, str):
        desc_dict = {f: desc if f == features[0] else f for f in features}
    else:
        desc_dict = desc
    
    # 检查 overdue 和 dpds 参数
    if overdue is not None and dpds is None:
        raise ValueError("传入 overdue 参数时必须同时传入 dpds")
    
    # 构建目标变量列表
    target_configs = []
    if overdue is not None:
        # 逾期分析模式
        if isinstance(overdue, str):
            overdue = [overdue]
        if isinstance(dpds, int):
            dpds = [dpds]
        
        for mob_col in overdue:
            for d in dpds:
                target_name = f"{mob_col}_{d}+"
                target_configs.append({
                    'name': target_name,
                    'mob_col': mob_col,
                    'dpd': d
                })
    elif target is not None:
        # 普通目标模式
        target_configs = [{'name': target, 'mob_col': None, 'dpd': None}]
    else:
        raise ValueError("必须传入 target 或 overdue+dpds 参数")
    
    # 存储所有特征的结果
    all_feature_tables = []
    all_feature_rules = {}
    
    # 构建默认分箱器参数（在循环外，避免重复计算）
    method_for_binner = 'mdlp' if method == 'optimal' else method

    default_binner_params = {
        'method': method_for_binner,
        'max_n_bins': max_n_bins,
        'min_bin_size': min_bin_size,
        'missing_separate': missing_separate,
        'prebinning': prebinning,
        'prebinning_params': prebinning_params,
    }

    # MDLP默认开启后处理微调，用户可通过 kwargs 覆盖
    if method_for_binner == 'mdlp':
        default_binner_params.setdefault('lift_refine', True)
        default_binner_params.setdefault('lift_focus_weight', 3.0)
        default_binner_params.setdefault('sample_stability_weight', 0.2)
        default_binner_params.setdefault('monotonic_bonus_weight', 0.4)
        default_binner_params.setdefault('lift_refine_max_bins', max_n_bins)

    # quantile 方法需禁用所有后处理，保证分位数切分点精确
    if method == 'quantile':
        default_binner_params.setdefault('lift_refine', False)
        default_binner_params.setdefault('min_bin_size', 0)

    # 透传 monotonic 参数（优先级高于 kwargs）
    if monotonic is not None:
        default_binner_params['monotonic'] = monotonic

    # 添加其他额外参数
    default_binner_params.update(kwargs)

    for feat in features:
        # === 确定当前特征的分箱器 ===
        # 优先级: binner(已训练且覆盖该特征) > rules(覆盖该特征) > binner(未训练模板) > method(新建)
        current_binner = None
        need_fit = False

        # 1. 检查 binner 是否覆盖该特征
        if binner is not None:
            if isinstance(binner, dict):
                # 按特征名映射的分箱器字典
                if feat in binner:
                    feat_binner = binner[feat]
                    if getattr(feat_binner, '_is_fitted', False) and hasattr(feat_binner, 'splits_') and feat in feat_binner.splits_:
                        current_binner = feat_binner  # 直接使用已训练的分箱器
            elif isinstance(binner, BaseBinning):
                if getattr(binner, '_is_fitted', False) and hasattr(binner, 'splits_') and feat in binner.splits_:
                    # 已训练的分箱器且包含该特征 → 直接使用
                    current_binner = binner
                elif not getattr(binner, '_is_fitted', False):
                    # 未训练的分箱器 → 作为模板 deepcopy 后训练
                    current_binner = deepcopy(binner)
                    need_fit = True

        # 2. 检查 rules 是否覆盖该特征
        feat_rule = None
        if current_binner is None and rules is not None:
            if isinstance(rules, dict) and feat in rules:
                feat_rule = np.array(rules[feat])
            elif isinstance(rules, list):
                feat_rule = np.array(rules)

        # 3. 如果 binner 和 rules 都没覆盖，创建新的分箱器
        if current_binner is None and feat_rule is None:
            current_binner = OptimalBinning(**default_binner_params)
            need_fit = True

        # 需要训练或应用规则时，准备训练数据
        if need_fit or feat_rule is not None:
            first_target = target_configs[0]

            # 准备训练数据
            if first_target['mob_col'] is not None:
                # 逾期模式
                train_data = data[[feat, first_target['mob_col']]].copy()
                y_train = (train_data[first_target['mob_col']] > first_target['dpd']).astype(int)

                if del_grey:
                    mask = (train_data[first_target['mob_col']] > first_target['dpd']) | (train_data[first_target['mob_col']] == 0)
                    train_data = train_data[mask]
                    y_train = y_train[mask]
            else:
                # 普通目标模式
                train_data = data[[feat, first_target['name']]].copy()
                y_train = train_data[first_target['name']]

            if feat_rule is not None:
                # 从规则生成分箱器
                current_binner = OptimalBinning(method='quantile')
                current_binner.splits_ = {feat: feat_rule}
                current_binner.feature_types_ = {feat: 'numerical'}
                current_binner.n_bins_ = {feat: len(feat_rule) + 1}
                current_binner._is_fitted = True

                # 生成bin_table用于后续的transform
                bins_tmp = np.digitize(train_data[feat].values, feat_rule, right=True)
                temp_stats = compute_bin_stats(bins_tmp, y_train.values, target_type='binary')
                current_binner.bin_tables_ = {feat: temp_stats}
            else:
                # 拟合分箱器
                current_binner.fit(train_data[[feat]], y_train)
        
        # 为每个目标生成分箱表
        feat_tables = []
        target_names = []
        
        # 根据del_grey确定merge_columns
        # merge_columns: 这些列在不同目标下是相同的，放在"分箱详情"层级下
        # 当 del_grey=True 时，不同目标下样本数不同，样本数相关列不应该合并
        # 当 del_grey=False 时，样本数相同，可以合并样本数相关列
        # 注意：样本占比也受 del_grey 影响，因为分母（总样本数）可能不同
        # 列名已统一，无论金额口径还是样本口径都使用相同的列名
        base_merge_cols = ['指标名称', '指标含义', '分箱标签']
        
        if isinstance(del_grey, bool) and del_grey:
            # 剔除灰样本：只保留基础分箱信息作为公共列
            merge_cols = base_merge_cols
        else:
            # 保留灰样本或单目标：样本数和占比也是公共列
            merge_cols = base_merge_cols + ['样本总数', '样本占比']
        
        for target_cfg in target_configs:
            target_name = target_cfg['name']
            target_names.append(target_name)
            
            # 准备数据
            if target_cfg['mob_col'] is not None:
                # 逾期模式：需要包含金额字段（如果有）
                cols_to_select = [feat, target_cfg['mob_col']]
                if amount is not None and amount in data.columns:
                    cols_to_select.append(amount)
                analysis_data = data[cols_to_select].copy()
                y = (analysis_data[target_cfg['mob_col']] > target_cfg['dpd']).astype(int)
                
                # 剔除灰客户：只保留好样本(overdue==0)和坏样本(overdue>dpd)
                # 参考 scp: _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)")
                if isinstance(del_grey, bool) and del_grey:
                    mask = (analysis_data[target_cfg['mob_col']] > target_cfg['dpd']) | (analysis_data[target_cfg['mob_col']] == 0)
                    analysis_data = analysis_data[mask].reset_index(drop=True)
                    y = y[mask].reset_index(drop=True)
            else:
                # 普通目标模式：需要包含金额字段（如果有）
                cols_to_select = [feat, target_name]
                if amount is not None and amount in data.columns:
                    cols_to_select.append(amount)
                analysis_data = data[cols_to_select].copy()
                y = analysis_data[target_name]
            
            # 分箱转换
            X_feat = analysis_data[[feat]]
            splits = current_binner.splits_.get(feat, np.array([]))
            try:
                transformed = current_binner.transform(X_feat, metric='indices')
                bins = transformed[feat].to_numpy(dtype=float)
            except Exception:
                x_values = X_feat[feat].values
                missing_mask = pd.isna(x_values)
                real_splits = splits[~np.isnan(splits)] if isinstance(splits, np.ndarray) and len(splits) > 0 else splits
                bins = np.digitize(x_values, real_splits, right=True)
                bins = bins.astype(float)
                bins[missing_mask] = -1
            bin_labels = _get_binner_bin_labels(current_binner, feat, bins, fallback_splits=splits)
            
            # 准备金额数据（如果有）
            amount_values = analysis_data[amount].values if amount is not None and amount in analysis_data.columns else None
            
            # 创建分箱表
            splits = current_binner.splits_.get(feat)
            bin_table = _create_bin_table(
                bins=bins,
                y=y.values,
                feature_name=feat,
                desc=desc_dict.get(feat, feat),
                splits=splits,
                amount=amount_values,
                bin_labels=bin_labels,
            )
            
            # 长格式：插入"逾期标签"列标识当前目标
            if long_format:
                bin_table.insert(2, '逾期标签', target_name)

            # 筛选指定列
            if return_cols is not None:
                # 确保基础列存在
                base_cols = ['指标名称', '指标含义', '分箱标签']
                if long_format:
                    base_cols.insert(2, '逾期标签')
                available_cols = [c for c in base_cols + return_cols if c in bin_table.columns]
                bin_table = bin_table[available_cols]

            feat_tables.append(bin_table)
            
            if verbose > 0:
                n_samples = len(analysis_data)
                n_bad = y.sum()
                bad_rate = y.mean()
                logger.info(f"特征 {feat} - 目标 {target_name}: 样本数 {n_samples}, 坏样本数 {n_bad}, 坏样本率 {bad_rate:.4f}, 分箱数 {len(bin_table)}")
        
        # 合并多目标表
        if long_format:
            # 长格式：各逾期标签纵向堆叠；margins 时按目标分别追加合计行
            if margins:
                feat_tables = [add_margins(table) for table in feat_tables]
            merged_table = pd.concat(feat_tables, axis=0, ignore_index=True)
        elif len(feat_tables) > 1:
            merged_table = _merge_multi_target_tables(feat_tables, target_names, merge_cols)
        else:
            merged_table = feat_tables[0]

        all_feature_tables.append(merged_table)
        
        # 保存分箱规则
        if return_rules:
            all_feature_rules[feat] = current_binner.splits_.get(feat, np.array([])).tolist()
    
    # 合并多特征表
    if len(all_feature_tables) == 1:
        final_table = all_feature_tables[0]
    else:
        final_table = pd.concat(all_feature_tables, axis=0, ignore_index=True)

    if long_format:
        # 长格式合计行已按目标分组追加，此处仅规范列顺序
        final_table = _reorder_long_format_columns(final_table)
    elif margins:
        # 添加合计行
        final_table = add_margins(final_table)

    if return_rules:
        return final_table, all_feature_rules
    return final_table


def benchmark_binning_methods(
    data: pd.DataFrame,
    feature: str,
    overdue_col: str = 'MOB1',
    dpds: Optional[List[int]] = None,
    max_n_bins: int = 5,
    min_bin_size: float = 0.01,
    monotonic: str = 'auto_asc_desc',
    hscredit_methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """逐方法对比 hscredit 内部分箱效果。

    仅使用 hscredit 内置分箱器，不依赖额外第三方分箱库。
    重点指标：头部/尾部 Lift、头尾差(edge_gap)、是否单调。
    """
    if dpds is None:
        dpds = [3, 0]
    if hscredit_methods is None:
        hscredit_methods = ['mdlp', 'cart', 'chi', 'tree', 'kmeans', 'best_ks', 'best_iv', 'quantile']

    x = pd.to_numeric(data[feature], errors='coerce')

    def _eval_splits(x_s: pd.Series, y_s: pd.Series, splits: Optional[List], model_name: str, dpd: int) -> Dict[str, Any]:
        mask = x_s.notna() & y_s.notna()
        xv = x_s[mask].values.astype(float)
        yv = y_s[mask].values.astype(int)
        if len(xv) == 0:
            return {'method': model_name, 'dpd': dpd, 'error': 'no valid samples'}

        sp = np.array(splits if splits is not None else [], dtype=float)
        bins = np.digitize(xv, sp, right=True)
        n_bins = len(sp) + 1

        counts = np.bincount(bins, minlength=n_bins).astype(float)
        bad = np.bincount(bins, weights=yv, minlength=n_bins).astype(float)
        bad_rate = bad / np.maximum(counts, 1.0)
        overall_bad_rate = max(yv.mean(), 1e-12)
        lift = bad_rate / overall_bad_rate

        diffs = np.diff(bad_rate)
        asc = bool(np.all(diffs >= -1e-12))
        desc = bool(np.all(diffs <= 1e-12))
        nz = np.sign(diffs)
        nz = nz[nz != 0]
        turns = 0 if len(nz) <= 1 else int(np.sum(nz[1:] * nz[:-1] < 0))

        return {
            'method': model_name,
            'dpd': dpd,
            'n_bins': int(n_bins),
            'head_lift': float(lift[0]),
            'tail_lift': float(lift[-1]),
            'edge_gap': float(abs(lift[-1] - lift[0])),
            'max_lift': float(np.max(lift)),
            'min_lift': float(np.min(lift)),
            'monotonic': bool(asc or desc),
            'turns': turns,
            'splits': sp.tolist(),
        }

    rows = []

    for d in dpds:
        y = (data[overdue_col] > d).astype(int)

        for method in hscredit_methods:
            try:
                binner = OptimalBinning(
                    method=method,
                    max_n_bins=max_n_bins,
                    min_bin_size=min_bin_size,
                    monotonic=monotonic,
                    prebinning='quantile',
                    prebinning_params={'max_n_bins': 100},
                    lift_refine=True,
                )
                binner.fit(pd.DataFrame({feature: x}), y)
                rows.append(_eval_splits(x, y, binner.splits_.get(feature, []), f'hscredit-{method}', d))
            except Exception as e:
                rows.append({'method': f'hscredit-{method}', 'dpd': d, 'error': str(e)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if 'error' in result.columns:
        ok = result[result['error'].isna()] if result['error'].notna().any() else result
    else:
        ok = result

    if not ok.empty:
        ok = ok.sort_values(['dpd', 'monotonic', 'edge_gap', 'head_lift'], ascending=[True, False, False, False])
        return ok.reset_index(drop=True)

    return result.reset_index(drop=True)


def _normalize_efficiency_rules(
    feature: str,
    manual_rules: Union[List, Tuple, np.ndarray, Dict[str, List]],
) -> Tuple[List, Dict[str, List]]:
    """标准化手工分箱规则，兼容 list 和 dict 两种输入方式。

    注意：已废弃，请使用 _generate_quantile_rules 生成固定分位数规则。
    """
    if manual_rules is None:
        raise ValueError("manual_rules 不能为空，请传入手工分箱边界列表或 {特征名: 边界列表} 字典")

    if isinstance(manual_rules, dict):
        if feature not in manual_rules:
            raise ValueError(f"manual_rules 中未找到特征 '{feature}' 的分箱边界")
        feature_rules = manual_rules[feature]
    else:
        feature_rules = manual_rules

    if not isinstance(feature_rules, (list, tuple, np.ndarray)):
        raise ValueError("manual_rules 必须是列表、元组、ndarray 或按特征名映射的字典")

    normalized_rules = list(feature_rules)
    if len(normalized_rules) == 0:
        raise ValueError("manual_rules 不能为空列表")

    return normalized_rules, {feature: normalized_rules}


# 固定分位数列表，用于自动生成手工分箱边界
FIXED_QUANTILES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def _generate_quantile_rules(
    data: pd.DataFrame,
    feature: str,
    quantiles: Optional[List[float]] = None,
    decimals: int = 4,
) -> List[float]:
    """根据分位数列表自动生成分箱边界。

    :param data: 数据集
    :param feature: 特征名
    :param quantiles: 分位数列表，默认使用 FIXED_QUANTILES
    :param decimals: 保留小数位数，默认 4
    :return: 分箱边界列表（已去重并保留指定小数位）
    """
    if quantiles is None:
        quantiles = FIXED_QUANTILES

    x = pd.to_numeric(data[feature], errors='coerce')
    valid_mask = ~x.isna()
    x_valid = x[valid_mask]

    if len(x_valid) == 0:
        return []

    # 计算分位数对应的值
    boundaries = []
    for q in quantiles:
        value = x_valid.quantile(q)
        if pd.notna(value):
            # 转换为普通 Python float，避免 numpy 类型
            float_value = round(float(value), decimals)
            boundaries.append(float_value)

    # 去重并保持排序
    boundaries = sorted(set(boundaries))

    return boundaries


def _prepare_efficiency_dataset(
    data: pd.DataFrame,
    feature: str,
    target: str,
    overdue: Optional[Union[str, List[str]]] = None,
    dpd: int = 0,
    del_grey: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """为效率分析准备目标变量和可绘图数据。"""
    if feature not in data.columns:
        raise ValueError(f"数据中不存在特征列 '{feature}'")

    working_data = data.copy()

    if overdue is not None:
        if isinstance(overdue, (list, tuple)):
            if len(overdue) != 1:
                raise ValueError("feature_efficiency_analysis 仅支持单个 overdue 字段，请传入字符串或单元素列表")
            overdue = overdue[0]

        if overdue not in working_data.columns:
            raise ValueError(f"数据中不存在 overdue 字段 '{overdue}'")

        actual_target = f"{overdue} {int(dpd)}+"
        working_data[actual_target] = (working_data[overdue] > int(dpd)).astype(int)

        if del_grey:
            working_data = working_data.loc[
                (working_data[overdue] > int(dpd)) | (working_data[overdue] == 0)
            ].reset_index(drop=True)
    else:
        actual_target = target
        if actual_target not in working_data.columns:
            raise ValueError(f"数据中不存在目标列 '{actual_target}'")

    score_series = pd.to_numeric(working_data[feature], errors='coerce')
    valid_mask = ~(score_series.isna() | pd.isna(working_data[actual_target]))
    plot_data = working_data.loc[valid_mask].copy()
    plot_data[feature] = score_series.loc[valid_mask]

    if plot_data.empty:
        raise ValueError(f"特征 '{feature}' 没有可用于绘制 KS/ROC 曲线的有效数值数据")

    if plot_data[actual_target].nunique(dropna=True) != 2:
        raise ValueError(f"目标列 '{actual_target}' 必须是二分类标签，当前唯一值数量为 {plot_data[actual_target].nunique(dropna=True)}")

    return working_data, plot_data, actual_target


def feature_efficiency_analysis(
    data: pd.DataFrame,
    feature: str,
    manual_rules: Optional[Union[List, Tuple, np.ndarray, Dict[str, List]]] = None,
    target: str = "target",
    overdue: Optional[Union[str, List[str]]] = None,
    dpd: int = 0,
    auto_method: str = "mdlp",
    desc: Optional[str] = None,
    date_col: Optional[str] = None,
    group_cols: Optional[Union[str, List[str]]] = None,
    date_freq: str = "M",
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True,
    prebinning: Optional[Union[str, BaseBinning, Dict]] = "quantile",
    prebinning_params: Optional[Dict[str, Any]] = None,
    del_grey: bool = False,
    margins: bool = False,
    amount: Optional[str] = None,
    figsize: Tuple[float, float] = (24, 5),
    trend_figsize: Optional[Tuple[float, float]] = None,
    comparison_orientation: str = "horizontal",
    auto_kwargs: Optional[Dict[str, Any]] = None,
    trend_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    suffix: str = "",
    quantiles: Optional[List[float]] = None,
    rule_decimals: int = 4,
    save: Optional[str] = None,
) -> Dict[str, Any]:
    """特征效率分析：对比手工分箱与自动分箱效果，并输出趋势图。

    适用于单个数值型指标或评分变量的快速效果评估。函数会：
    1. 自动生成分位数分箱规则（默认使用 [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]）
    2. 生成手工分箱与自动分箱两张分箱表
    3. 输出一行四列的组合图：手工分箱图、自动分箱图、KS 曲线、ROC 曲线
    4. 当传入日期字段或分组字段时，额外输出手工分箱与自动分箱两张 bin_trend_plot 趋势图

    :param data: 输入数据集
    :param feature: 需要分析的特征名，建议为数值型指标/评分
    :param manual_rules: 手工分箱边界，支持 list 或 {feature: list}。默认 None，表示自动使用 quantiles 生成分箱边界
    :param target: 目标变量列名，默认 target
    :param overdue: 逾期列名。传入后会基于 overdue > dpd 自动构造二分类目标
    :param dpd: 逾期阈值，仅在 overdue 模式下使用，默认 0
    :param auto_method: 自动分箱方法，默认 mdlp
    :param desc: 特征中文描述，默认使用 feature
    :param date_col: 日期列，传入后生成按时间分组的趋势图
    :param group_cols: 分组字段，支持单列或多列，传入后生成分组趋势图
    :param date_freq: 日期聚合频率，默认 M
    :param max_n_bins: 自动分箱最大箱数，默认 5
    :param min_bin_size: 自动分箱最小箱占比，默认 0.05
    :param missing_separate: 缺失值是否单独分箱，默认 True
    :param prebinning: 预分箱配置，默认 quantile
    :param prebinning_params: 预分箱参数，默认 None
    :param del_grey: overdue 模式下是否剔除灰样本，默认 False
    :param margins: 是否追加合计行，默认 False
    :param amount: 金额字段，传入后输出金额口径分箱表
    :param figsize: 一行四列组合图尺寸，默认 (24, 5)
    :param trend_figsize: 趋势图尺寸，默认 None（由 bin_trend_plot 自动计算）
    :param comparison_orientation: 两张分箱图的方向，默认 horizontal
    :param auto_kwargs: 额外传给自动分箱 feature_bin_stats 的参数
    :param trend_kwargs: 额外传给 bin_trend_plot 的参数
    :param output_dir: 图片保存目录，默认 None（不落盘）
    :param suffix: 保存文件名后缀，默认空字符串
    :param quantiles: 分位数列表，用于自动生成分箱边界。默认 [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    :param rule_decimals: 分箱边界保留的小数位数，默认 4
    :param save: 比较图片保存路径，如果提供则将生成的比较图保存至指定路径，默认为 None
    :return: dict，包含分箱表、分箱规则、组合图与趋势图

    Example::

        >>> # 自动使用分位数生成手工分箱规则
        >>> result = feature_efficiency_analysis(
        ...     data=df,
        ...     feature='score',
        ...     target='target',
        ...     auto_method='mdlp',
        ...     date_col='apply_date'
        ... )
        >>> result['manual_table']
        >>> result['comparison_figure']

        >>> # 手动指定分箱规则
        >>> result = feature_efficiency_analysis(
        ...     data=df,
        ...     feature='score',
        ...     manual_rules=[450, 520, 600, 680],
        ...     target='target',
        ...     auto_method='mdlp'
        ... )
    """
    feature_desc = desc or feature
    auto_kwargs = auto_kwargs.copy() if auto_kwargs else {}
    trend_kwargs = trend_kwargs.copy() if trend_kwargs else {}
    for reserved_key in ["data", "feature", "target", "dimension_cols", "date_col", "date_freq", "figsize", "title", "rules", "method"]:
        trend_kwargs.pop(reserved_key, None)

    # 如果未提供 manual_rules，自动根据分位数生成分箱边界
    if manual_rules is None:
        manual_rules_list = _generate_quantile_rules(data, feature, quantiles, rule_decimals)
        manual_rules_dict = {feature: manual_rules_list}
    else:
        manual_rules_list, manual_rules_dict = _normalize_efficiency_rules(feature, manual_rules)

    working_data, plot_data, actual_target = _prepare_efficiency_dataset(
        data=data,
        feature=feature,
        target=target,
        overdue=overdue,
        dpd=dpd,
        del_grey=del_grey,
    )

    common_bin_params = dict(
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        missing_separate=missing_separate,
        prebinning=prebinning,
        prebinning_params=prebinning_params,
        del_grey=del_grey,
        margins=margins,
        amount=amount,
        desc=feature_desc,
    )
    target_params = {"target": target} if overdue is None else {"target": target, "overdue": overdue, "dpds": int(dpd)}

    manual_table = feature_bin_stats(
        working_data,
        feature,
        rules=manual_rules_list,
        **target_params,
        **common_bin_params,
    )

    auto_table, auto_rules_map = feature_bin_stats(
        working_data,
        feature,
        method=auto_method,
        return_rules=True,
        **target_params,
        **common_bin_params,
        **auto_kwargs,
    )
    auto_rules = auto_rules_map.get(feature, [])

    comparison_fig, comparison_axes = plt.subplots(1, 4, figsize=figsize)
    comparison_axes = np.atleast_1d(comparison_axes)

    bin_plot(
        manual_table.copy(),
        ax=comparison_axes[0],
        title=f"手工分箱图\n{feature_desc}",
        desc="",
        orientation=comparison_orientation,
    )
    bin_plot(
        auto_table.copy(),
        ax=comparison_axes[1],
        title=f"自动分箱图({auto_method})\n{feature_desc}",
        desc="",
        orientation=comparison_orientation,
    )
    ks_plot(plot_data[feature], plot_data[actual_target], axes=[comparison_axes[2], comparison_axes[3]])
    comparison_axes[2].set_title("KS 曲线")
    comparison_axes[3].set_title("ROC 曲线")
    comparison_fig.suptitle(f"{feature_desc} 分箱效率分析", fontsize=14, fontweight="bold")
    comparison_fig.tight_layout(rect=(0, 0, 1, 0.94))
    
    trend_figures: Dict[str, plt.Figure] = {}
    if date_col is not None or group_cols is not None:
        common_trend_params = dict(
            data=working_data,
            feature=feature,
            target=actual_target,
            dimension_cols=group_cols,
            date_col=date_col,
            date_freq=date_freq,
            figsize=trend_figsize,
            **trend_kwargs,
        )

        trend_figures["manual"] = bin_trend_plot(
            **common_trend_params,
            rules=manual_rules_dict,
            title=f"{feature_desc} 手工分箱趋势图",
        )

        auto_trend_params = common_trend_params.copy()
        auto_trend_params["title"] = f"{feature_desc} 自动分箱趋势图({auto_method})"
        if auto_rules:
            auto_trend_params["rules"] = {feature: auto_rules}
        else:
            auto_trend_params["method"] = auto_method
            auto_trend_params["max_n_bins"] = max_n_bins
            auto_trend_params["min_bin_size"] = min_bin_size
            auto_trend_params.update(auto_kwargs)
        trend_figures["auto"] = bin_trend_plot(**auto_trend_params)

    # Save comparison figure if save path is provided
    _save_fig = None  # Will be set when needed
    saved_paths: Dict[str, str] = {}
    
    if save is not None and save != "":
        # Import save_figure utility for consistency
        from ..core.viz.utils import save_figure
        _save_fig = save_figure
        _save_fig(comparison_fig, save)
        saved_paths["comparison"] = save
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_path = os.path.join(output_dir, f"feature_efficiency_comparison_{feature}{suffix}.png")
        # Only save comparison figure via output_dir if save parameter wasn't provided
        if save is None or save == "":
            if _save_fig is None:
                from ..core.viz.utils import save_figure
                _save_fig = save_figure
            _save_fig(comparison_fig, comparison_path)
            saved_paths["comparison"] = comparison_path
        
        for trend_name, trend_fig in trend_figures.items():
            trend_path = os.path.join(output_dir, f"feature_efficiency_trend_{trend_name}_{feature}{suffix}.png")
            if _save_fig is None:
                from ..core.viz.utils import save_figure
                _save_fig = save_figure
            _save_fig(trend_fig, trend_path)
            saved_paths[f"trend_{trend_name}"] = trend_path

    return {
        "feature": feature,
        "feature_desc": feature_desc,
        "target": actual_target,
        "manual_table": manual_table,
        "auto_table": auto_table,
        "manual_rules": manual_rules_list,
        "auto_rules": auto_rules,
        "comparison_figure": comparison_fig,
        "trend_figures": trend_figures,
        "saved_paths": saved_paths,
    }


def auto_feature_analysis(
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
    output_dir="model_report",
    margins=False,
    amount=None,
    image_table_gap_rows=None,
):
    """自动特征分析.

    用于三方数据评估或自有评分效果评估。生成包含数据集概况、特征分箱统计、
    KS 曲线、分布图等内容的 Excel 分析结果。

    :param data: 需要评估的数据集，需要包含目标变量
    :param features: 需要进行分析的特征名称，支持单个字符串或列表
    :param target: 目标变量名称
    :param overdue: 逾期天数字段名称，传入时会覆盖 target 参数
    :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本
    :param date: 日期列，用于时间维度分布分析
    :param freq: 日期统计粒度，默认按月 "M"
    :param data_summary_comment: 数据备注信息
    :param excel_writer: Excel 文件路径或 ExcelWriter 对象
    :param sheet: 工作表名称
    :param start_col: 起始列
    :param start_row: 起始行
    :param dropna: 是否剔除缺失值
    :param writer_params: Excel 写入器初始化参数
    :param bin_params: 分箱统计参数，支持 feature_bin_stats 的参数
    :param feature_map: 特征名称映射字典
    :param corr: 是否计算特征相关性
    :param pictures: 需要生成的图片列表，支持 ["ks", "hist", "bin"]
    :param suffix: 文件名后缀，避免同名文件被覆盖
    :param output_dir: 图片输出目录
    :param margins: 是否在每个特征分箱表末尾添加合计行，默认 False
    :param amount: 放款金额或余额字段名称。传入后同时生成订单口径和金额口径两张分箱表
    :param image_table_gap_rows: 图片区与分箱表之间的额外空行数
    :return: (end_row, end_col) 分析结束位置

    **参考样例**

    >>> from hscredit.report.feature_analyzer import auto_feature_analysis
    >>> auto_feature_analysis(data, features=['feature1'], target='target', excel_writer='分析结果.xlsx')
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

    if overdue and not isinstance(overdue, (list, tuple, np.ndarray)):
        overdue = [overdue]
    elif overdue is not None:
        overdue = list(overdue)

    if dpds is not None and not isinstance(dpds, (list, tuple, np.ndarray)):
        dpds = [dpds]
    elif dpds is not None:
        dpds = list(dpds)

    target, target_label_names, target_display_labels, target_y_map = _auto_feature_target_maps(
        data, target=target, overdue=overdue, dpds=dpds
    )
    if overdue:
        data[target] = target_y_map[target_label_names[0]]

    if date is not None and date in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date]):
        converted_date = pd.to_datetime(data[date], errors='coerce')
        if converted_date.notna().sum() < len(data) * 0.5 and pd.api.types.is_numeric_dtype(data[date]):
            converted_date_excel = pd.to_datetime(data[date], unit='D', errors='coerce')
            if converted_date_excel.notna().sum() > converted_date.notna().sum():
                converted_date = converted_date_excel
        data[date] = converted_date

    if isinstance(excel_writer, ExcelWriter):
        writer = excel_writer
    else:
        writer = ExcelWriter(**writer_params)

    worksheet = writer.get_sheet_by_name(sheet)

    if image_table_gap_rows is None:
        image_table_gap_rows = 2 if getattr(writer, "system", "windows") == "windows" else 1

    if bin_params and "del_grey" in bin_params and bin_params.get("del_grey"):
        merge_columns = ["指标名称", "指标含义", "分箱标签"]
    else:
        merge_columns = ["指标名称", "指标含义", "分箱标签", "样本总数", "样本占比"]

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

    dataset_labels = ["整体样本"]
    sample_stats, sample_percent_cols = build_sample_stats_table(
        dataset_labels,
        [target_y_map],
        target_label_names,
        display_labels=target_display_labels,
    )
    sample_start_row = end_row + 2
    if isinstance(sample_stats.columns, pd.MultiIndex):
        end_row, end_col = dataframe2excel(
            sample_stats,
            writer,
            worksheet,
            percent_cols=sample_percent_cols,
            start_row=sample_start_row,
            title="样本总体分布情况",
            index=True,
        )
    else:
        end_row, end_col = dataframe2excel(
            sample_stats,
            writer,
            worksheet,
            percent_cols=sample_percent_cols,
            start_row=sample_start_row,
            title="样本总体分布情况",
        )
    end_row += 2

    if date is not None and date in data.columns:
        distribution_plot(
            data, date=date, freq=freq, target=target,
            save=os.path.join(output_dir, f"sample_time_distribution{suffix}.png"), result=True
        )
        time_title_columns_len = (
            len(sample_stats.columns) + sample_stats.index.nlevels
            if isinstance(sample_stats.columns, pd.MultiIndex)
            else len(sample_stats.columns)
        )
        end_row, end_col = writer.insert_value2sheet(
            worksheet, (end_row, start_col), value="样本时间分布情况", style="header",
            end_space=(end_row, start_col + time_title_columns_len - 1)
        )
        end_row, end_col = writer.insert_pic2sheet(
            worksheet, os.path.join(output_dir, f"sample_time_distribution{suffix}.png"),
            (end_row + 1, start_col), figsize=(720, 370)
        )
        dates = pd.to_datetime(data[date], errors="coerce")
        try:
            period_values = dates.dt.to_period(freq).astype(str).values
        except Exception:
            period_values = dates.dt.to_period("M").astype(str).values
        time_distribution, time_percent_cols = build_group_distribution_table(
            dataset_labels,
            [target_y_map],
            [period_values],
            target_label_names,
            display_labels=target_display_labels,
        )
        table_start_row = end_row
        end_row, end_col = dataframe2excel(
            time_distribution,
            writer,
            worksheet,
            percent_cols=time_percent_cols,
            condition_cols=time_percent_cols,
            start_row=table_start_row,
            index=not isinstance(time_distribution.columns, pd.MultiIndex),
        )
        end_row += 2

    feature_summary = data[features].summary(y=data[target])
    if "特征名" not in feature_summary.columns:
        index_name = feature_summary.index.name or "index"
        feature_summary = feature_summary.reset_index().rename(columns={index_name: "特征名"})
    feature_summary_start_row = end_row
    end_row, end_col = dataframe2excel(
        feature_summary,
        writer,
        worksheet,
        start_row=feature_summary_start_row,
        title="变量综合统计",
        right_cols=[0],
    )
    feature_name_col = start_col + feature_summary.columns.get_loc("特征名")
    feature_summary_rows = {
        str(feat): feature_summary_start_row + 2 + feature_summary.columns.nlevels + position
        for position, feat in enumerate(feature_summary["特征名"])
    }
    end_row += 2

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

    use_amount = amount is not None and amount in data.columns

    features_iter = tqdm(features)
    for col in features_iter:
        features_iter.set_postfix(feature=feature_map.get(col, col))
        try:
            if overdue is None:
                cols_needed = [col, target]
            else:
                cols_needed = list(dict.fromkeys([col, target] + overdue))
            if use_amount:
                cols_needed = list(dict.fromkeys(cols_needed + [amount]))
            temp = data[cols_needed]
            missing_rate = _feature_missing_rate(data, col, dropna)

            if isinstance(dropna, bool) and dropna is True:
                temp = temp.dropna(subset=col).reset_index(drop=True)
            elif isinstance(dropna, (float, int, str)):
                temp = temp[temp[col] != dropna].reset_index(drop=True)

            actual_target = target
            if overdue:
                actual_target = f"{overdue[0]} {dpds[0]}+"

            sample_table = feature_bin_stats(
                temp, col, overdue=overdue, dpds=dpds,
                desc=f"{feature_map.get(col, col)}", target=target,
                margins=margins,
                **bin_params
            )

            if use_amount:
                amount_table = feature_bin_stats(
                    temp, col, overdue=overdue, dpds=dpds,
                    desc=f"{feature_map.get(col, col)}", target=target,
                    amount=amount,
                    margins=margins,
                    **bin_params
                )
            else:
                amount_table = None

            sample_title_columns_len = len(sample_table.columns)
            amount_title_columns_len = len(amount_table.columns) if (use_amount and amount_table is not None) else 0

            if return_cols:
                if sample_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                    _merge_cols_for_title = [("分箱详情", c) for c in merge_columns]
                else:
                    _merge_cols_for_title = merge_columns
                sample_title_columns_len = len(
                    _merge_cols_for_title + [
                        c for c in sample_table.columns
                        if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                        or (not isinstance(c, (tuple, list)) and c in return_cols)
                        or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                    ]
                )

                if use_amount and amount_table is not None:
                    if amount_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                        _merge_cols_amt_for_title = [("分箱详情", c) for c in merge_columns]
                    else:
                        _merge_cols_amt_for_title = merge_columns
                    amount_title_columns_len = len(
                        _merge_cols_amt_for_title + [
                            c for c in amount_table.columns
                            if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                            or (not isinstance(c, (tuple, list)) and c in return_cols)
                            or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                        ]
                    )

            if pictures and len(pictures) > 0:
                if "bin" in pictures:
                    if sample_table.columns.nlevels > 1:
                        level1_cols = sample_table.columns.get_level_values(0).unique().tolist()
                        target_col = actual_target if actual_target in level1_cols else level1_cols[-1] if len(level1_cols) > 1 else level1_cols[0]
                        plot_table = sample_table[["分箱详情", target_col]]
                        plot_table.columns = [c[-1] for c in plot_table.columns]
                    else:
                        plot_table = sample_table.copy()

                    if "分箱标签" in plot_table.columns:
                        plot_table.rename(columns={"分箱标签": "分箱"}, inplace=True)

                    bin_plot(
                        plot_table, desc=f"{feature_map.get(col, col)}", figsize=(10, 5),
                        anchor=0.935, save=os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png")
                    )

                if temp[col].dtypes.name not in ['object', 'str', 'category']:
                    if "ks" in pictures:
                        plot_source = temp.dropna().reset_index(drop=True)
                        has_ks = len(plot_source) > 0 and plot_source[col].nunique() > 1 and plot_source[actual_target].nunique() > 1
                        if has_ks:
                            ks_plot(
                                plot_source[col], plot_source[actual_target], figsize=(10, 5),
                                title=f"{feature_map.get(col, col)}",
                                save=os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png")
                            )
                    if "hist" in pictures:
                        plot_source = temp.dropna().reset_index(drop=True)
                        if len(plot_source) > 0:
                            hist_plot(
                                plot_source[col], y_true=plot_source[actual_target], figsize=(10, 6),
                                desc=f"{feature_map.get(col, col)} 好客户 VS 坏客户",
                                bins=30, anchor=1.11, fontsize=14,
                                labels={0: "好客户", 1: "坏客户"},
                                save=os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png")
                            )

            if use_amount and amount_table is not None:
                title_span = sample_title_columns_len + 1 + amount_title_columns_len
            else:
                title_span = sample_title_columns_len

            feature_title_row = end_row + 2
            end_row, end_col = writer.insert_value2sheet(
                worksheet, (feature_title_row, start_col),
                value=f"数据字段: {feature_map.get(col, col)} (缺失率: {round(missing_rate * 100, 2)}%)",
                style="header", end_space=(feature_title_row, start_col + title_span - 1)
            )

            summary_row = feature_summary_rows.get(str(col))
            if summary_row is not None:
                try:
                    writer.insert_hyperlink2sheet(
                        worksheet,
                        (summary_row, feature_name_col),
                        hyperlink=f"#'{worksheet.title}'!{writer.get_cell_space((feature_title_row, start_col))}",
                    )
                    writer.insert_hyperlink2sheet(
                        worksheet,
                        (feature_title_row, start_col),
                        hyperlink=f"#'{worksheet.title}'!{writer.get_cell_space((summary_row, feature_name_col))}",
                    )
                except Exception:
                    pass

            if pictures and len(pictures) > 0:
                chart_row = end_row + 1
                if "bin" in pictures:
                    end_row, end_col = writer.insert_pic2sheet(
                        worksheet, os.path.join(output_dir, f"feature_bins_plot_{col}{suffix}.png"),
                        (chart_row, start_col), figsize=(600, 350)
                    )
                if temp[col].dtypes.name not in ['object', 'str', 'category'] and temp[col].isnull().sum() != len(temp):
                    if "ks" in pictures and has_ks:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_ks_plot_{col}{suffix}.png"),
                            (chart_row, end_col - 1), figsize=(600, 350)
                        )
                    if "hist" in pictures:
                        end_row, end_col = writer.insert_pic2sheet(
                            worksheet, os.path.join(output_dir, f"feature_hist_plot_{col}{suffix}.png"),
                            (chart_row, end_col - 1), figsize=(600, 350)
                        )

            table_start_row = end_row + image_table_gap_rows
            if return_cols:
                if sample_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                    sample_merge_cols = [("分箱详情", c) for c in merge_columns]
                else:
                    sample_merge_cols = merge_columns
                end_row, end_col = dataframe2excel(
                    sample_table[
                        sample_merge_cols + [
                            c for c in sample_table.columns
                            if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                            or (not isinstance(c, (tuple, list)) and c in return_cols)
                            or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                        ]
                    ], writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=table_start_row
                )
            else:
                end_row, end_col = dataframe2excel(
                    sample_table, writer, worksheet,
                    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                    condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                    merge=True, fill=True, start_row=table_start_row
                )

            if use_amount and amount_table is not None:
                amount_start_col = end_col + 1
                if return_cols:
                    if amount_table.columns.nlevels > 1 and not isinstance(merge_columns[0], tuple):
                        amount_merge_cols = [("分箱详情", c) for c in merge_columns]
                    else:
                        amount_merge_cols = merge_columns
                    dataframe2excel(
                        amount_table[
                            amount_merge_cols + [
                                c for c in amount_table.columns
                                if (isinstance(c, (tuple, list)) and c[-1] in return_cols)
                                or (not isinstance(c, (tuple, list)) and c in return_cols)
                                or (isinstance(return_cols[0], (tuple, list)) and isinstance(c, (tuple, list)) and c in return_cols)
                            ]
                        ], writer, worksheet,
                        percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                        condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                        merge=True, fill=True,
                        start_row=table_start_row, start_col=amount_start_col
                    )
                else:
                    dataframe2excel(
                        amount_table, writer, worksheet,
                        percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善"],
                        condition_cols=["坏样本率", "LIFT值"], merge_column=["指标名称", "指标含义"],
                        merge=True, fill=True,
                        start_row=table_start_row, start_col=amount_start_col
                    )

        except Exception:
            logger.warning("数据字段 %s 分析时发生异常，请排查数据中是否存在异常:\n%s", col, traceback.format_exc())

    if not isinstance(excel_writer, ExcelWriter) and not isinstance(sheet, Worksheet):
        writer.save(excel_writer)

    return end_row, end_col
