"""数据概览模块.

提供数据集整体信息统计和特征描述功能.
主要复用 hscredit.utils.describe 的功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Literal, Any, Callable, Tuple

from .utils import infer_feature_types, validate_dataframe


def data_info(df: pd.DataFrame) -> pd.DataFrame:
    """数据集基础信息统计.
    
    :param df: 输入数据
    :return: 数据集信息DataFrame，列包括[信息项, 值]
    
    **参考样例**

    >>> info = data_info(df)
    >>> print(info)
                信息项         值
    0      样本数（行）     10000
    1     特征数（列）        50
    2      数值型特征        30
    3      分类型特征        15
    4      日期型特征         5
    5      缺失值列数        10
    6    内存使用(MB)      15.5
    """
    validate_dataframe(df, check_empty=False)
    
    if df.empty:
        return pd.DataFrame({'信息项': ['样本数（行）', '特征数（列）'], '值': [0, 0]})
    
    # 推断特征类型
    feature_types = infer_feature_types(df)
    type_counts = pd.Series(feature_types).value_counts().to_dict()
    
    # 计算内存使用
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # 缺失值统计
    missing_cols = (df.isnull().sum() > 0).sum()
    total_missing = df.isnull().sum().sum()
    
    info_data = {
        '信息项': [
            '样本数（行）',
            '特征数（列）',
            '数值型特征',
            '分类型特征',
            '日期型特征',
            '常数特征',
            'ID特征',
            '缺失值列数',
            '总缺失值数',
            '内存使用(MB)',
        ],
        '值': [
            len(df),
            len(df.columns),
            type_counts.get('numerical', 0),
            type_counts.get('categorical', 0),
            type_counts.get('datetime', 0),
            type_counts.get('constant', 0),
            type_counts.get('id', 0),
            missing_cols,
            total_missing,
            round(memory_mb, 2),
        ]
    }
    
    return pd.DataFrame(info_data)


def missing_analysis(df: pd.DataFrame,
                    threshold: float = 0.0,
                    features: List[str] = None) -> pd.DataFrame:
    """缺失值分析.
    
    :param df: 输入数据
    :param threshold: 缺失率阈值，仅返回缺失率>=该值的特征
    :param features: 指定分析的特征列表，None则分析全部
    :return: 缺失值分析DataFrame，列包括[特征名, 缺失数, 缺失率, 非空数, 查得率]
    
    **参考样例**

    >>> missing = missing_analysis(df, threshold=0.05)
    >>> print(missing)
          特征名  缺失数   缺失率(%)  非空数  查得率(%)
    0     age      0      0.00  10000    100.0
    1  income    500      5.00   9500     95.0
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.columns.tolist()
    
    total = len(df)
    results = []
    
    for col in features:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / total
        
        if missing_rate >= threshold:
            results.append({
                '特征名': col,
                '缺失数': int(missing_count),
                '缺失率(%)': round(missing_rate * 100, 2),
                '非空数': int(total - missing_count),
                '查得率(%)': round((1 - missing_rate) * 100, 2),
            })
    
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('缺失率(%)', ascending=False).reset_index(drop=True)
    
    return result_df


def feature_summary(
    df: pd.DataFrame,
    features: List[str] = None,
    y: Optional[Union[str, pd.Series, np.ndarray, List, Tuple]] = None,
    val_df: Optional[pd.DataFrame] = None,
    models: Optional[Dict[str, Any]] = None,
    model_type: Optional[Literal['xgboost', 'lightgbm', 'catboost', 'randomforest']] = None,
    model_params: Optional[Dict] = None,
    max_n_bins: int = 10,
    psi_method: Literal['random_split', 'group_col', 'date_col'] = 'random_split',
    psi_group_col: Optional[str] = None,
    psi_date_col: Optional[str] = None,
    psi_freq: str = 'M',
    psi_test_size: float = 0.3,
    percentiles: List[float] = None,
    random_state: int = 42,
    numeric_as_categorical: Optional[List[str]] = None,
    force_numeric: Optional[List[str]] = None,
    n_jobs: int = -1,
    show_progress: bool = False,
    binning_method: str = 'quantile',
    binning_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """综合特征描述统计.

    整合特征基础统计、IV、KS、PSI和模型特征重要性，快速获取数据集特征详情。
    参考 toad.detect + IV + KS + PSI 的功能组合。

    无论字段是字符还是数字，返回列名都是一致的：
    - 数值型特征：分位数返回对应分位点的数值
    - 类别型特征：分位数返回按类别样本数逆序排列后对应分位点的类别值

    :param df: 训练/基准数据集
    :param features: 特征列表，None则分析全部
    :param y: 目标变量，支持列名、数组、列表、元组或Series，不传则不计算IV/KS/特征重要性
    :param val_df: 验证集，用于计算PSI，不传则使用psi_method指定的方式
    :param models: 已训练好的模型字典，格式{'模型名': model}，用于获取特征重要性
        - model需要支持feature_importances_属性或get_feature_importance()方法
    :param model_type: 模型类型，用于自动训练模型提取特征重要性，可选'xgboost'/'lightgbm'/'catboost'/'randomforest'
    :param model_params: 模型参数，配合model_type使用
    :param max_n_bins: IV、趋势和PSI共用的最大分箱数，默认10
    :param psi_method: PSI计算方式
        - 'random_split': 随机拆分两份数据计算PSI（默认）
        - 'group_col': 按psi_group_col指定的分组列计算PSI
        - 'date_col': 按psi_date_col指定的日期列分组计算PSI
    :param psi_group_col: 分组列名（当psi_method='group_col'时使用）
    :param psi_date_col: 日期列名（当psi_method='date_col'时使用）
    :param psi_freq: 时间频率，'D'/'W'/'M'/'Q'，默认'M'
    :param psi_test_size: 随机拆分比例（当psi_method='random_split'时使用），默认0.3
    :param percentiles: 分位数点，默认[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    :param random_state: 随机种子
    :param numeric_as_categorical: 强制视为分类变量的数值列名列表（仅当指定时才生效）
    :param force_numeric: 强制视为数值变量的列名列表（仅当指定时才生效）
    :param n_jobs: 并行工作数。-1根据数据规模保守推断，1为串行，正整数为明确指定
    :param show_progress: 是否显示已处理字段数、总字段数和当前处理字段
    :param binning_method: IV、趋势和PSI共用的分箱方法，默认'quantile'（等频分箱）
    :param binning_params: 传给OptimalBinning的完整参数。其优先级高于binning_method、
        max_n_bins和random_state，重复键直接覆盖；user_splits字典按原字段名配置
    :return: 综合特征描述DataFrame，包含以下列：
        - 基础统计: 特征名、字段类型、样本数、缺失数/率、唯一值数、众数等
        - 分布统计: 最小值、最大值、平均值、标准差、各分位数
        - 质量指标: 零值率、负值率、重复率
        - 预测指标（传入y时）: IV、KS、趋势
        - 稳定性指标: PSI
        - 重要性指标（传入models时）: 各模型特征重要性

    所有比例字段返回0~1原始小数，所有数值指标均保留底层计算完整精度；展示格式由
    pandas Styler或ExcelWriter的percent_cols等外部配置负责。

    趋势列说明:
        - ascending: 坏样本率单调递增
        - descending: 坏样本率单调递减
        - peak: 倒U型趋势（先增后减）
        - valley: U型趋势（先减后增）
        - categorical: 类别型特征
        - unknown: 无法确定趋势

    **参考样例**

    >>> # 基础统计（无目标变量）
    >>> summary = feature_summary(df)

    >>> # 包含IV、KS、趋势（传入目标变量）
    >>> summary = feature_summary(df, y='target')

    >>> # 包含PSI（随机拆分）
    >>> summary = feature_summary(df, y='target', psi_method='random_split')

    >>> # 按日期分组计算PSI
    >>> summary = feature_summary(df, y='target', psi_method='date_col', psi_date_col='apply_date')

    >>> # 传入已训练模型获取特征重要性
    >>> models = {'XGB': xgb_model, 'LGB': lgb_model}
    >>> summary = feature_summary(df, y='target', models=models)

    >>> # 自动训练模型获取特征重要性
    >>> summary = feature_summary(df, y='target', model_type='xgboost')

    >>> # 指定数值列视为分类变量（如年龄分段编码）
    >>> summary = feature_summary(df, y='target', numeric_as_categorical=['age_group'])

    >>> # 三项指标共用显式分箱；内层参数覆盖外层同名参数
    >>> summary = feature_summary(
    ...     df,
    ...     y='target',
    ...     max_n_bins=10,
    ...     binning_params={
    ...         'method': 'uniform',
    ...         'max_n_bins': 5,
    ...         'user_splits': {'age': [25, 35, 45]},
    ...         'strict_user_splits': True,
    ...     },
    ... )
    """
    validate_dataframe(df)
    
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    if features is None:
        if isinstance(y, str):
            features = [c for c in df.columns if c != y]
        else:
            features = df.columns.tolist()
        if len(features) == 0:
            features = df.columns.tolist()
    
    # 获取目标变量。外部数组型目标按位置匹配，不受双方索引标签影响。
    from ._feature_summary import _normalize_target

    y_series = _normalize_target(df, y)
    if isinstance(y, str) and y in features:
        features = [f for f in features if f != y]
    
    valid_features = [feature for feature in features if feature in df.columns]
    from ._feature_summary import build_feature_summary_fields

    results_df = build_feature_summary_fields(
        df=df,
        features=valid_features,
        percentiles=percentiles,
        numeric_as_categorical=numeric_as_categorical,
        force_numeric=force_numeric,
        y_series=y_series,
        val_df=val_df,
        max_n_bins=max_n_bins,
        psi_method=psi_method,
        psi_group_col=psi_group_col,
        psi_date_col=psi_date_col,
        psi_freq=psi_freq,
        psi_test_size=psi_test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        show_progress=show_progress,
        binning_method=binning_method,
        binning_params=binning_params,
    )
    
    # 特征重要性（传入已训练模型）
    if models is not None:
        for model_name, model in models.items():
            # 统一使用 BaseRiskModel 的 get_feature_importances() 方法
            if hasattr(model, 'get_feature_importances'):
                try:
                    importances = model.get_feature_importances()
                    if isinstance(importances, pd.Series):
                        # Series索引为特征名，值为重要性
                        results_df[f'{model_name}重要性'] = importances.reindex(features)
                except Exception:
                    pass
    
    # 自动训练模型获取特征重要性（使用hscredit统一封装模型）
    if model_type is not None and y_series is not None:
        try:
            # 过滤有效特征：存在于df中、且有非空值
            valid_features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]

            # 只保留数值型特征（模型无法处理object/datetime等类型）
            numeric_features = [
                f for f in valid_features
                if pd.api.types.is_numeric_dtype(df[f])
            ]

            if len(numeric_features) == 0:
                raise ValueError("没有数值型特征可用于训练模型")

            X_train = df[numeric_features].fillna(df[numeric_features].median())

            # 默认参数
            default_params = {'random_state': random_state}
            if model_params:
                default_params.update(model_params)

            # 根据模型类型使用hscredit统一封装的模型类
            model_class = None
            if model_type == 'xgboost':
                try:
                    from ..models import XGBoostRiskModel
                    model_class = XGBoostRiskModel
                except ImportError:
                    pass
            elif model_type == 'lightgbm':
                try:
                    from ..models import LightGBMRiskModel
                    model_class = LightGBMRiskModel
                except ImportError:
                    pass
            elif model_type == 'catboost':
                try:
                    from ..models import CatBoostRiskModel
                    model_class = CatBoostRiskModel
                except ImportError:
                    pass
            elif model_type in ('randomforest', 'rf'):
                try:
                    from ..models import RandomForestRiskModel
                    model_class = RandomForestRiskModel
                except ImportError:
                    pass
            elif model_type == 'logistic':
                try:
                    from ..models import LogisticRegression
                    model_class = LogisticRegression
                except ImportError:
                    pass

            # 使用统一接口训练并获取特征重要性
            if model_class is None:
                raise ImportError(f"无法导入模型类: {model_type}")

            model = model_class(**default_params)
            model.fit(X_train, y_series)
            importances = model.get_feature_importances()
            if isinstance(importances, pd.Series):
                results_df[f'{model_type}重要性'] = importances.reindex(numeric_features)
        except Exception:
            # 训练失败不中断
            pass
    
    # 重置索引，使特征名成为列
    results_df = results_df.reset_index()

    # 将特征效果指标（特征重要性、KS、IV、PSI、趋势）调整到靠前位置
    # 基础统计列之后的合理位置：放在缺失率之后，唯一值数之前
    base_cols = ['特征名', '字段类型', '样本数', '缺失数', '缺失率']
    effect_cols = [c for c in results_df.columns if c not in base_cols]
    # 特征重要性列（包含"重要性"字样）放最前，KS/IV/趋势/PSI紧随其后，其他放最后
    importance_cols = []
    metric_cols = []
    other_effect = []
    for c in effect_cols:
        if '重要性' in c:
            importance_cols.append(c)
        elif c in ('KS', 'IV', 'PSI', '趋势'):
            metric_cols.append(c)
        else:
            other_effect.append(c)
    # 保持原有顺序：特征重要性 -> KS/IV/PSI/趋势 -> 其他
    results_df = results_df[base_cols + importance_cols + metric_cols + other_effect]

    return results_df


def numeric_summary(df: pd.DataFrame,
                   features: List[str] = None,
                   percentiles: List[float] = None) -> pd.DataFrame:
    """数值特征详细统计.
    
    :param df: 输入数据
    :param features: 指定分析的特征列表，None则分析全部数值型特征
    :param percentiles: 额外分位数，默认[0.01, 0.05, 0.95, 0.99]
    :return: 数值特征统计DataFrame
    
    **参考样例**

    >>> num_stats = numeric_summary(df)
    >>> print(num_stats[['特征名', '均值', '标准差', '偏度', '峰度']])
    """
    validate_dataframe(df)
    
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.95, 0.99]
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    results = []
    
    for col in features:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        # 基础统计
        result = {
            '特征名': col,
            '样本数': len(series),
            '均值': round(series.mean(), 4),
            '标准差': round(series.std(), 4),
            '最小值': round(series.min(), 4),
            '最大值': round(series.max(), 4),
            '中位数': round(series.median(), 4),
            '偏度': round(series.skew(), 4),
            '峰度': round(series.kurtosis(), 4),
        }
        
        # 分位数
        for p in percentiles:
            col_name = f'{int(p*100)}%'
            result[col_name] = round(series.quantile(p), 4)
        
        # 零值和负值统计
        result['零值数'] = (series == 0).sum()
        result['零值率(%)'] = round((series == 0).sum() / len(series) * 100, 2)
        result['负值数'] = (series < 0).sum()
        result['负值率(%)'] = round((series < 0).sum() / len(series) * 100, 2)
        
        results.append(result)
    
    return pd.DataFrame(results)


def category_summary(df: pd.DataFrame,
                    features: List[str] = None,
                    max_categories: int = 10) -> pd.DataFrame:
    """类别特征统计.
    
    :param df: 输入数据
    :param features: 指定分析的特征列表，None则分析全部分类别特征
    :param max_categories: 显示最常见的N个类别
    :return: 类别特征统计DataFrame
    
    **参考样例**

    >>> cat_stats = category_summary(df)
    >>> print(cat_stats[['特征名', '类别数', '最常见类别', '最常见占比(%)']])
    """
    validate_dataframe(df)
    
    if features is None:
        # 选择object类型和类别少的数值类型
        feature_types = infer_feature_types(df)
        features = [f for f, t in feature_types.items() if t == 'categorical']
    
    results = []
    
    for col in features:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        value_counts = series.value_counts()
        
        result = {
            '特征名': col,
            '样本数': len(series),
            '类别数': len(value_counts),
            '最常见类别': value_counts.index[0] if len(value_counts) > 0 else None,
            '最常见占比(%)': round(value_counts.iloc[0] / len(series) * 100, 2) if len(value_counts) > 0 else 0,
        }
        
        # 前N个类别的分布
        for i in range(min(max_categories, len(value_counts))):
            cat_name = value_counts.index[i]
            cat_count = value_counts.iloc[i]
            result[f'类别{i+1}'] = cat_name
            result[f'类别{i+1}占比(%)'] = round(cat_count / len(series) * 100, 2)
        
        results.append(result)
    
    return pd.DataFrame(results)


def data_quality_report(df: pd.DataFrame,
                       features: List[str] = None,
                       missing_threshold: float = 0.5,
                       constant_threshold: float = 0.95) -> pd.DataFrame:
    """数据质量综合报告.
    
    :param df: 输入数据
    :param features: 指定分析的特征列表，None则分析全部
    :param missing_threshold: 缺失率阈值，超过视为质量问题
    :param constant_threshold: 常数比例阈值，超过视为准常数特征
    :return: 数据质量报告DataFrame，列包括[特征名, 问题类型, 严重程度, 建议处理]
    
    **参考样例**

    >>> quality = data_quality_report(df)
    >>> print(quality)
          特征名      问题类型  严重程度         建议处理
    0    phone      高缺失率      高    考虑删除或填充
    1   status    准常数特征      中  检查业务意义
    """
    validate_dataframe(df)
    
    if features is None:
        features = df.columns.tolist()
    
    issues = []
    total = len(df)
    
    for col in features:
        if col not in df.columns:
            continue
        
        series = df[col]
        
        # 检查缺失率
        missing_rate = series.isnull().sum() / total
        if missing_rate >= missing_threshold:
            issues.append({
                '特征名': col,
                '问题类型': '高缺失率',
                '严重程度': '高' if missing_rate > 0.7 else '中',
                '问题值': f'{missing_rate*100:.1f}%',
                '建议处理': '考虑删除或业务填充',
            })
        
        # 检查准常数特征
        if series.nunique() > 0:
            mode_ratio = series.value_counts().iloc[0] / total
            if mode_ratio >= constant_threshold:
                issues.append({
                    '特征名': col,
                    '问题类型': '准常数特征',
                    '严重程度': '中',
                    '问题值': f'{mode_ratio*100:.1f}%',
                    '建议处理': '检查业务意义，考虑删除',
                })
        
        # 检查数据类型问题
        if series.dtype == 'object':
            # 尝试转换为数值
            try:
                pd.to_numeric(series.dropna().iloc[:100])
                issues.append({
                    '特征名': col,
                    '问题类型': '数值型存储为字符串',
                    '严重程度': '低',
                    '问题值': 'object',
                    '建议处理': '转换为数值类型',
                })
            except Exception:
                pass
    
    if not issues:
        return pd.DataFrame({'信息': ['未发现明显数据质量问题']})
    
    return pd.DataFrame(issues)


def population_stability_monitor(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    segment_cols: Union[str, List[str]],
    binning_method: str = 'quantile',
    n_bins: int = 5,
    bin_edges: Optional[Dict[str, List]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    group_col: Optional[str] = None,
    metrics: Union[str, List[str]] = '占比',
    sort_by: Optional[str] = None,
    sort_order: Literal['desc', 'asc'] = 'desc'
) -> pd.DataFrame:
    """群体稳定性监控分析（Population Stability Monitor）.
    
    金融风控中监控客群分布变化的核心方法，基于期望样本（expected）构建客群分层标准，
    分析实际样本（actual）在各维度下的客群占比变化。
    
    适用场景:
    - 模型上线后监控生产客群是否偏离训练样本分布
    - 对比不同时间段（如各月、各季度）的客群结构变化
    - 分析不同渠道、产品线的客群构成差异
    - 快速识别客群漂移（Population Drift）风险
    
    指标说明:
    - 占比: 该分层在总样本中的比例（默认）
    - 样本数: 该分层的样本数量
    - 绝对变化率: actual占比 - expected占比（百分点差值）
    - 相对变化率: (actual占比 - expected占比) / expected占比 × 100%
    
    :param expected: 期望/基准数据集（如训练集、历史样本），用于构建分层标准
    :param actual: 实际/监控数据集（如生产数据、近期样本），用于分析客群变化
    :param segment_cols: 客群分层变量，支持单变量或多变量交叉
        - 字符串: 单一变量（如'收入'）
        - 列表: 多变量交叉（如['收入层级', '信用等级']）
    :param binning_method: 分箱方法（当传入bin_edges时此参数忽略）
        - 'quantile': 等频分箱（默认，客群规模相近）
        - 'uniform': 等距分箱
        - 'tree': 决策树分箱
    :param n_bins: 分箱数，默认5层
    :param bin_edges: 自定义分箱边界，格式{变量名: [边界列表]}
        - 传入后直接使用，无需expected数据集计算分箱
    :param date_col: 时间列，用于时间维度监控（如各月对比）
    :param date_freq: 时间频率，'D'日/'W'周/'M'月/'Q'季，默认'M'
    :param group_col: 分类列，用于分类维度监控（如各渠道对比）
    :param metrics: 统计指标，支持单选或多选
        - '占比': 占比百分比（默认）
        - '样本数': 样本数量
        - '绝对变化率': 占比差值（actual - expected）
        - '相对变化率': 占比变化百分比
        - 列表: 多指标同时显示，如['占比', '样本数', '绝对变化率']
    :param sort_by: 排序依据，可选'expected'/'actual_首列'/'actual_末列'
    :param sort_order: 排序方式，'desc'降序/'asc'升序
    :return: 客群监控结果DataFrame
        - Index: 分层标签（单级或多级索引）
        - Columns: (维度, 指标) 多级列，维度包含expected和各actual维度
    
    **参考样例**

    >>> # 1. 基础用法：监控生产客群相对训练集的变化
    >>> result = population_stability_monitor(
    ...     expected=train_df,              # 训练集作为基准
    ...     actual=prod_df,                 # 生产数据
    ...     segment_cols='income'           # 收入分层监控
    ... )
    >>> 
    >>> # 2. 时间维度：监控各月客群变化趋势
    >>> result = population_stability_monitor(
    ...     expected=historical_df,
    ...     actual=prod_df,
    ...     segment_cols='risk_score',
    ...     date_col='apply_date',
    ...     date_freq='M'
    ... )
    >>> 
    >>> # 3. 多维度交叉：收入+信用等级
    >>> result = population_stability_monitor(
    ...     expected=baseline_df,
    ...     actual=current_df,
    ...     segment_cols=['income_level', 'credit_grade']
    ... )
    >>> 
    >>> # 4. 自定义分箱边界（expected可传可不传）
    >>> bin_edges = {'age': [0, 25, 35, 45, 55, 100]}
    >>> result = population_stability_monitor(
    ...     expected=None,                  # 传入bin_edges后可不传
    ...     actual=prod_df,
    ...     segment_cols='age',
    ...     bin_edges=bin_edges
    ... )
    >>> 
    >>> # 5. 多指标显示
    >>> result = population_stability_monitor(
    ...     expected=train_df,
    ...     actual=prod_df,
    ...     segment_cols='score',
    ...     metrics=['占比', '样本数', '绝对变化率', '相对变化率']
    ... )
    >>> 
    >>> # 6. 渠道维度对比
    >>> result = population_stability_monitor(
    ...     expected=train_df,
    ...     actual=current_df,
    ...     segment_cols='customer_type',
    ...     group_col='channel'
    ... )
    """
    from ..binning import QuantileBinning, UniformBinning, TreeBinning
    
    validate_dataframe(actual)
    
    # 处理metrics参数
    if isinstance(metrics, str):
        metrics = [metrics]
    available_metrics = ['占比', '样本数', '绝对变化率', '相对变化率']
    metrics = [m for m in metrics if m in available_metrics]
    if not metrics:
        metrics = ['占比']
    
    # 标准化segment_cols
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]
    
    # 检查变量存在性
    for col in segment_cols:
        if col not in actual.columns:
            raise ValueError(f"客群分层变量 '{col}' 不存在于actual数据集")
        if expected is not None and col not in expected.columns:
            raise ValueError(f"客群分层变量 '{col}' 不存在于expected数据集")
    
    # 确定分箱标准来源
    use_custom_edges = bin_edges is not None and any(col in bin_edges for col in segment_cols)
    
    if not use_custom_edges and expected is None:
        raise ValueError("必须传入expected数据集或bin_edges分箱边界")
    
    # 构建分箱标准
    segment_labels = {}
    actual_copy = actual.copy()
    
    for col in segment_cols:
        # 确定分箱边界
        if use_custom_edges and col in bin_edges:
            edges = bin_edges[col]
        elif expected is not None and pd.api.types.is_numeric_dtype(expected[col]):
            # 从expected计算分箱
            if binning_method == 'uniform':
                binner = UniformBinning(n_bins=n_bins)
            elif binning_method == 'tree':
                binner = TreeBinning(max_depth=int(np.log2(n_bins)) + 1)
            else:
                binner = QuantileBinning(max_n_bins=n_bins)
            
            binner.fit(expected[[col]])
            edges = binner.edges_.get(col, [])
        else:
            # 类别型，不分箱
            edges = []
        
        # 生成分层标签并应用
        if len(edges) >= 2:
            labels = []
            for i in range(len(edges) - 1):
                if i == 0:
                    label = f"≤{edges[i+1]:.2f}".rstrip('0').rstrip('.')
                elif i == len(edges) - 2:
                    label = f">{edges[i]:.2f}".rstrip('0').rstrip('.')
                else:
                    left = f"{edges[i]:.2f}".rstrip('0').rstrip('.')
                    right = f"{edges[i+1]:.2f}".rstrip('0').rstrip('.')
                    label = f"({left}, {right}]"
                labels.append(label)
            
            segment_labels[col] = labels
            actual_copy[f'{col}_分层'] = pd.cut(actual_copy[col], bins=edges, labels=labels, include_lowest=True)
        else:
            # 类别型
            ref_data = expected[col] if expected is not None else actual[col]
            segment_labels[col] = ref_data.dropna().unique().tolist()
            actual_copy[f'{col}_分层'] = actual_copy[col]
    
    # 构建监控维度
    monitor_dims = []
    
    if date_col and date_col in actual.columns:
        actual_copy[date_col] = pd.to_datetime(actual_copy[date_col])
        if date_freq == 'M':
            actual_copy['_period'] = actual_copy[date_col].dt.to_period('M').astype(str)
        elif date_freq == 'W':
            actual_copy['_period'] = actual_copy[date_col].dt.to_period('W').astype(str)
        elif date_freq == 'Q':
            actual_copy['_period'] = actual_copy[date_col].dt.to_period('Q').astype(str)
        elif date_freq == 'D':
            actual_copy['_period'] = actual_copy[date_col].dt.date.astype(str)
        else:
            actual_copy['_period'] = actual_copy[date_col].dt.to_period('M').astype(str)
        monitor_dims.append(('date', '_period'))
    
    if group_col and group_col in actual.columns:
        monitor_dims.append(('group', group_col))
    
    # 计算统计
    results = []
    segment_hierarchy_cols = [f'{col}_分层' for col in segment_cols]
    
    # Expected统计
    if expected is not None:
        expected_copy = expected.copy()
        # 对expected应用相同的分箱
        for col in segment_cols:
            if f'{col}_分层' in actual_copy.columns and col in expected_copy.columns:
                if pd.api.types.is_numeric_dtype(expected_copy[col]) and len(edges) >= 2:
                    expected_copy[f'{col}_分层'] = pd.cut(expected_copy[col], bins=edges, labels=segment_labels[col], include_lowest=True)
                else:
                    expected_copy[f'{col}_分层'] = expected_copy[col]
        
        exp_stats = _calc_psm_stats(
            expected_copy, segment_hierarchy_cols, segment_labels, 
            'expected', metrics
        )
        results.extend(exp_stats)
    
    # Actual各维度统计
    if monitor_dims:
        dim_col = monitor_dims[0][1]  # 主维度
        dim_values = sorted(actual_copy[dim_col].dropna().unique())
        
        for val in dim_values:
            subset = actual_copy[actual_copy[dim_col] == val]
            subset_stats = _calc_psm_stats(
                subset, segment_hierarchy_cols, segment_labels,
                str(val), metrics
            )
            results.extend(subset_stats)
    else:
        # 无维度划分，整体actual
        act_stats = _calc_psm_stats(
            actual_copy, segment_hierarchy_cols, segment_labels,
            'actual', metrics
        )
        results.extend(act_stats)
    
    # 构建DataFrame
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return pd.DataFrame()
    
    # 排序处理（在计算变化率之前）
    if sort_by is not None:
        ascending = (sort_order == 'asc')
        
        # 获取排序依据列的值
        if sort_by == 'expected' and 'expected' in result_df['维度'].values:
            sort_values = result_df[result_df['维度'] == 'expected'].set_index('分层标签')['占比'].to_dict()
        elif sort_by.startswith('actual_'):
            # actual_首列或actual_末列
            actual_dims = [d for d in result_df['维度'].unique() if d != 'expected']
            if actual_dims:
                target_dim = actual_dims[0] if '首列' in sort_by else actual_dims[-1]
                sort_values = result_df[result_df['维度'] == target_dim].set_index('分层标签')['占比'].to_dict()
            else:
                sort_values = {}
        else:
            sort_values = {}
        
        if sort_values:
            result_df['_sort_key'] = result_df['分层标签'].map(sort_values)
            result_df = result_df.sort_values('分层标签', key=lambda x: result_df['_sort_key'], ascending=ascending)
            result_df = result_df.drop(columns=['_sort_key'])
    
    # 计算变化率（排序后）
    if '绝对变化率' in metrics or '相对变化率' in metrics:
        if expected is not None and 'expected' in result_df['维度'].values:
            exp_ratios = result_df[result_df['维度'] == 'expected'].set_index('分层标签')['占比'].to_dict()
            
            def calc_rate_change(row):
                if row['维度'] == 'expected':
                    return 0.0, 0.0
                exp_val = exp_ratios.get(row['分层标签'], 0)
                act_val = row['占比']
                abs_change = act_val - exp_val
                rel_change = (abs_change / exp_val * 100) if exp_val > 0 else 0.0
                return round(abs_change, 2), round(rel_change, 2)
            
            changes = result_df.apply(calc_rate_change, axis=1)
            result_df['绝对变化率'] = [c[0] for c in changes]
            result_df['相对变化率'] = [c[1] for c in changes]
    
    # 构建透视表
    pivot_results = []
    for metric in metrics:
        if metric in result_df.columns:
            pivot_table = result_df.pivot_table(
                index='分层标签',
                columns='维度',
                values=metric,
                aggfunc='first'
            )
            # 调整列顺序：expected在前，其他按时间/分类排序
            cols = pivot_table.columns.tolist()
            if 'expected' in cols:
                other_cols = [c for c in sorted(cols) if c != 'expected']
                pivot_table = pivot_table[['expected'] + other_cols]
            pivot_table['指标'] = metric
            pivot_results.append(pivot_table.reset_index().set_index(['分层标签', '指标']))
    
    if pivot_results:
        final_result = pd.concat(pivot_results).unstack('指标')
        # 列顺序调整
        final_result.columns = final_result.columns.swaplevel(0, 1)
        final_result = final_result.sort_index(axis=1)
        return final_result
    else:
        return result_df


def _calc_psm_stats(df, segment_cols, segment_labels, dim_name, metrics):
    """计算PSM单个维度的统计."""
    results = []
    total = len(df)
    
    if total == 0:
        return results
    
    if len(segment_cols) == 1:
        col = segment_cols[0]
        value_counts = df[col].value_counts()
        ref_col = col.replace('_分层', '')
        
        for label in segment_labels.get(ref_col, []):
            count = value_counts.get(label, 0)
            ratio = count / total * 100 if total > 0 else 0
            
            result = {'分层标签': label, '维度': dim_name}
            if '样本数' in metrics:
                result['样本数'] = int(count)
            if '占比' in metrics or '绝对变化率' in metrics or '相对变化率' in metrics:
                result['占比'] = round(ratio, 2)
            
            results.append(result)
    else:
        grouped = df.groupby(segment_cols)
        for group_values, group_df in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            label = ' × '.join(str(v) for v in group_values)
            count = len(group_df)
            ratio = count / total * 100 if total > 0 else 0
            
            result = {'分层标签': label, '维度': dim_name}
            if '样本数' in metrics:
                result['样本数'] = int(count)
            if '占比' in metrics or '绝对变化率' in metrics or '相对变化率' in metrics:
                result['占比'] = round(ratio, 2)
            
            results.append(result)
    
    return results


def feature_group_analysis(
    df: pd.DataFrame,
    features: List[str] = None,
    group_cols: Union[str, List[str]] = None,
    date_col: Optional[str] = None,
    date_freq: str = 'M',
    stats: Union[str, List[str], Callable, Dict[str, Callable]] = 'default',
    y: Optional[Union[str, pd.Series]] = None,
    y_stats: List[str] = None,
    sort_by: Optional[Union[str, Tuple[str, str]]] = None,
    sort_order: Literal['desc', 'asc', 'custom'] = 'desc',
    custom_sort_func: Optional[Callable] = None,
    feature_order: Optional[List[str]] = None,
    include_overall: bool = True,
    pivot: bool = True
) -> pd.DataFrame:
    """分组特征分布分析.
    
    分析生产环境中特定时间段、客群或两者交叉组合下的特征分布，
    适用于监控不同维度下的特征表现、对比分析等场景。
    
    :param df: 输入数据
    :param features: 分析的特征列表，None则分析全部数值型特征
    :param group_cols: 客群分组列（单级或多级），如 'segment' 或 ['segment', 'channel']
    :param date_col: 时间列，用于时间分组分析，支持日期格式
    :param date_freq: 时间频率，'D'日/'W'周/'M'月/'Q'季/'Y'年，默认'M'
    :param stats: 统计指标
        - 'default': 默认统计[均值、中位数、标准差、缺失率、唯一值数]
        - 'all': 全部基础统计指标
        - 列表: 指定统计指标，如['均值', '标准差', 'IV']
        - 函数: 自定义统计函数，接收Series返回标量
        - 字典: 指标名到函数的映射，如{'变异系数': lambda x: x.std()/x.mean()}
    :param y: 目标变量，用于计算y相关统计（如逾期率、IV等）
    :param y_stats: y相关统计指标，如['逾期率', 'IV', 'KS']，需传入y才生效
    :param sort_by: 排序依据，可以是'特征名'、'统计项'或(特征名, 统计项)元组
    :param sort_order: 排序方式，'desc'降序/'asc'升序/'custom'自定义
    :param custom_sort_func: 自定义排序函数，当sort_order='custom'时使用
        接收DataFrame，返回排序后的DataFrame
    :param feature_order: 特征顺序列表，用于指定特征展示顺序
    :param include_overall: 是否包含总体统计列
    :param pivot: 是否透视结果（True: 特征×统计项为行，分组为列；False: 长格式）
    :return: 分组分析结果DataFrame
        - pivot=True时：多级索引(特征名, 统计项)，列为分组
        - pivot=False时：长格式，包含[特征名, 统计项, 分组, 值]等列
    
    **参考样例**

    >>> # 单维度客群分组分析
    >>> result = feature_group_analysis(
    ...     df, 
    ...     features=['age', 'income'],
    ...     group_cols='customer_segment',
    ...     stats=['均值', '中位数', '缺失率']
    ... )
    >>> 
    >>> # 时间维度分析（按月）
    >>> result = feature_group_analysis(
    ...     df,
    ...     features=['age', 'income'],
    ...     date_col='apply_date',
    ...     date_freq='M',
    ...     stats='default'
    ... )
    >>> 
    >>> # 客群+时间交叉分析（多级列）
    >>> result = feature_group_analysis(
    ...     df,
    ...     features=['age', 'income'],
    ...     group_cols='segment',
    ...     date_col='apply_date',
    ...     date_freq='M'
    ... )
    >>> 
    >>> # 自定义统计指标
    >>> custom_stats = {
    ...     '变异系数': lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan,
    ...     '10%分位数': lambda x: x.quantile(0.1),
    ...     '90%分位数': lambda x: x.quantile(0.9),
    ... }
    >>> result = feature_group_analysis(df, features=['age'], stats=custom_stats)
    >>> 
    >>> # 包含目标变量分析（如逾期率）
    >>> result = feature_group_analysis(
    ...     df,
    ...     features=['age', 'income'],
    ...     group_cols='segment',
    ...     y='fpd15',
    ...     y_stats=['逾期率', '样本数']
    ... )
    >>> 
    >>> # 排序展示
    >>> result = feature_group_analysis(
    ...     df,
    ...     features=['age', 'income'],
    ...     group_cols='segment',
    ...     sort_by=('age', '均值'),
    ...     sort_order='desc'
    ... )
    """
    from ..metrics import iv as iv_metric, ks as ks_metric
    
    validate_dataframe(df)
    
    # 确定分析的特征
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in df.columns]
    
    if len(features) == 0:
        raise ValueError("没有有效的特征可供分析")
    
    # 处理目标变量
    y_series = None
    if y is not None:
        if isinstance(y, str):
            if y in df.columns:
                y_series = df[y]
            else:
                raise ValueError(f"目标列 '{y}' 不存在")
        else:
            y_series = pd.Series(y)
            if len(y_series) != len(df):
                raise ValueError("目标变量长度与数据不匹配")
    
    # 构建分组键
    group_keys = []
    
    # 处理客群分组
    if group_cols is not None:
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        for col in group_cols:
            if col in df.columns:
                group_keys.append(col)
    
    # 处理时间分组
    if date_col is not None and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if date_freq == 'M':
            df['_time_group'] = df[date_col].dt.to_period('M').astype(str)
        elif date_freq == 'W':
            df['_time_group'] = df[date_col].dt.to_period('W').astype(str)
        elif date_freq == 'Q':
            df['_time_group'] = df[date_col].dt.to_period('Q').astype(str)
        elif date_freq == 'Y':
            df['_time_group'] = df[date_col].dt.to_period('Y').astype(str)
        elif date_freq == 'D':
            df['_time_group'] = df[date_col].dt.date.astype(str)
        else:
            df['_time_group'] = df[date_col].dt.to_period('M').astype(str)
        
        group_keys.append('_time_group')
    
    # 定义统计指标
    default_stats = {
        '均值': lambda x: round(x.mean(), 4) if len(x) > 0 else np.nan,
        '中位数': lambda x: round(x.median(), 4) if len(x) > 0 else np.nan,
        '标准差': lambda x: round(x.std(), 4) if len(x) > 0 else np.nan,
        '最小值': lambda x: round(x.min(), 4) if len(x) > 0 else np.nan,
        '最大值': lambda x: round(x.max(), 4) if len(x) > 0 else np.nan,
        '缺失率': lambda x: round(x.isnull().sum() / len(x) * 100, 2) if len(x) > 0 else 0,
        '唯一值数': lambda x: x.nunique(),
        '样本数': lambda x: len(x),
    }
    
    all_stats = {
        **default_stats,
        '零值率': lambda x: round((x == 0).sum() / len(x) * 100, 2) if len(x) > 0 else 0,
        '负值率': lambda x: round((x < 0).sum() / len(x) * 100, 2) if len(x) > 0 else 0,
        '偏度': lambda x: round(x.skew(), 4) if len(x) > 0 else np.nan,
        '峰度': lambda x: round(x.kurtosis(), 4) if len(x) > 0 else np.nan,
        '25%': lambda x: round(x.quantile(0.25), 4) if len(x) > 0 else np.nan,
        '75%': lambda x: round(x.quantile(0.75), 4) if len(x) > 0 else np.nan,
    }
    
    # 处理stats参数
    if stats == 'default':
        stat_funcs = default_stats
    elif stats == 'all':
        stat_funcs = all_stats
    elif isinstance(stats, list):
        stat_funcs = {k: all_stats.get(k, default_stats.get(k)) for k in stats if k in all_stats or k in default_stats}
        if not stat_funcs:
            stat_funcs = default_stats
    elif isinstance(stats, dict):
        stat_funcs = stats
    elif callable(stats):
        stat_funcs = {'自定义统计': stats}
    else:
        stat_funcs = default_stats
    
    # 处理y相关统计
    y_stat_funcs = {}
    if y_series is not None and y_stats:
        if '逾期率' in y_stats or 'bad_rate' in y_stats:
            y_stat_funcs['逾期率'] = lambda x, y: round(y.mean() * 100, 2) if len(y) > 0 else np.nan
        if '样本数' in y_stats or 'count' in y_stats:
            y_stat_funcs['样本数'] = lambda x, y: len(y)
        if '坏样本数' in y_stats or 'bad_count' in y_stats:
            y_stat_funcs['坏样本数'] = lambda x, y: int(y.sum())
    
    # 执行分组统计
    results = []
    
    # 总体统计
    if include_overall:
        for feat in features:
            series = df[feat]
            for stat_name, stat_func in stat_funcs.items():
                try:
                    value = stat_func(series)
                except Exception:
                    value = np.nan
                
                result = {
                    '特征名': feat,
                    '统计项': stat_name,
                    '分组': '总体',
                }
                if group_keys:
                    for gk in group_keys:
                        result[gk] = '总体'
                result['值'] = value
                results.append(result)
            
            # y相关统计
            for stat_name, stat_func in y_stat_funcs.items():
                try:
                    value = stat_func(series, y_series)
                except Exception:
                    value = np.nan
                
                result = {
                    '特征名': feat,
                    '统计项': stat_name,
                    '分组': '总体',
                }
                if group_keys:
                    for gk in group_keys:
                        result[gk] = '总体'
                result['值'] = value
                results.append(result)
    
    # 分组统计
    if group_keys:
        grouped = df.groupby(group_keys)
        
        for group_values, group_df in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            
            group_label = '_'.join(str(v) for v in group_values)
            
            for feat in features:
                series = group_df[feat]
                y_group = y_series[group_df.index] if y_series is not None else None
                
                for stat_name, stat_func in stat_funcs.items():
                    try:
                        value = stat_func(series)
                    except Exception:
                        value = np.nan
                    
                    result = {
                        '特征名': feat,
                        '统计项': stat_name,
                        '分组': group_label,
                    }
                    for i, gk in enumerate(group_keys):
                        col_name = gk.replace('_time_group', '时间分组') if '_time_group' in gk else gk
                        result[col_name] = group_values[i]
                    result['值'] = value
                    results.append(result)
                
                # y相关统计
                for stat_name, stat_func in y_stat_funcs.items():
                    try:
                        value = stat_func(series, y_group)
                    except Exception:
                        value = np.nan
                    
                    result = {
                        '特征名': feat,
                        '统计项': stat_name,
                        '分组': group_label,
                    }
                    for i, gk in enumerate(group_keys):
                        col_name = gk.replace('_time_group', '时间分组') if '_time_group' in gk else gk
                        result[col_name] = group_values[i]
                    result['值'] = value
                    results.append(result)
    
    # 创建DataFrame
    result_df = pd.DataFrame(results)
    
    if result_df.empty:
        return pd.DataFrame()
    
    # 排序处理
    if sort_by is not None:
        if sort_order == 'custom' and custom_sort_func is not None:
            result_df = custom_sort_func(result_df)
        else:
            ascending = (sort_order == 'asc')
            
            if isinstance(sort_by, str):
                # 按统计项排序（所有特征的该统计项）
                mask = result_df['统计项'] == sort_by
                if mask.any():
                    sort_values = result_df[mask].set_index('特征名')['值'].to_dict()
                    result_df['_sort_key'] = result_df['特征名'].map(sort_values)
                    result_df = result_df.sort_values(['特征名', '_sort_key'], ascending=[True, ascending])
                    result_df = result_df.drop(columns=['_sort_key'])
            elif isinstance(sort_by, tuple) and len(sort_by) == 2:
                feat, stat = sort_by
                mask = (result_df['特征名'] == feat) & (result_df['统计项'] == stat)
                if mask.any():
                    # 获取该特征-统计项在各分组的值
                    sort_df = result_df[mask][['分组', '值']].copy()
                    sort_order_dict = sort_df.set_index('分组')['值'].to_dict()
                    result_df['_sort_key'] = result_df['分组'].map(sort_order_dict)
                    result_df = result_df.sort_values(['特征名', '_sort_key'], ascending=[True, ascending])
                    result_df = result_df.drop(columns=['_sort_key'])
    
    # 特征顺序处理
    if feature_order is not None:
        result_df['_feat_order'] = result_df['特征名'].apply(lambda x: feature_order.index(x) if x in feature_order else 9999)
        result_df = result_df.sort_values(['_feat_order', '统计项']).drop(columns=['_feat_order'])
    
    # 透视处理
    if pivot:
        # 构建透视表
        index_cols = ['特征名', '统计项']
        
        # 确定分组列名（排除'分组'汇总列）
        group_col_names = []
        for gk in group_keys:
            col_name = gk.replace('_time_group', '时间分组') if '_time_group' in gk else gk
            if col_name in result_df.columns and col_name not in group_col_names:
                group_col_names.append(col_name)
        
        # 创建多级列
        if group_col_names:
            # 使用实际分组列作为列
            pivot_df = result_df.pivot_table(
                index=index_cols,
                columns=group_col_names,
                values='值',
                aggfunc='first'
            )
        else:
            # 使用分组列
            pivot_df = result_df.pivot_table(
                index=index_cols,
                columns='分组',
                values='值',
                aggfunc='first'
            )
        
        # 确保总体列在最前面
        if '总体' in pivot_df.columns:
            cols = ['总体'] + [c for c in pivot_df.columns if c != '总体']
            pivot_df = pivot_df[cols]
        
        return pivot_df
    else:
        # 长格式返回
        return result_df.reset_index(drop=True)
