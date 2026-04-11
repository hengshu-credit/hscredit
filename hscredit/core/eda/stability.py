"""稳定性分析模块.

提供PSI、CSI、时间稳定性等分析功能.
主要复用 hscredit.core.metrics 的PSI/CSI功能.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

from .utils import validate_dataframe, psi_rating


def psi_analysis(base_df: pd.DataFrame,
                current_df: pd.DataFrame,
                feature: str,
                n_bins: int = 10) -> Dict[str, Union[str, float, pd.DataFrame]]:
    """单变量PSI稳定性分析.
    
    复用 hscredit.core.metrics.psi_table
    
    :param base_df: 基准数据集
    :param current_df: 当前数据集
    :param feature: 特征名
    :param n_bins: 分箱数
    :return: PSI分析结果字典
    
    Example:
        >>> result = psi_analysis(train_df, test_df, 'age')
        >>> print(f"PSI: {result['PSI值']}, 稳定性: {result['稳定性']}")
    """
    from ..metrics import psi_table
    
    # 计算PSI表
    psi_df = psi_table(base_df[feature], current_df[feature], max_n_bins=n_bins)
    
    # 计算总PSI
    psi_value = psi_df['PSI贡献'].sum()
    
    return {
        '特征名': feature,
        'PSI值': round(psi_value, 4),
        '稳定性': psi_rating(psi_value),
        '分箱明细': psi_df,
    }


def batch_psi_analysis(df: pd.DataFrame,
                      features: List[str],
                      date_col: str,
                      base_period: str,
                      compare_periods: List[str],
                      n_bins: int = 10) -> pd.DataFrame:
    """批量PSI时间稳定性分析.
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param base_period: 基准期（如'2023-01'）
    :param compare_periods: 对比期列表
    :param n_bins: 分箱数
    :return: 批量PSI结果DataFrame
    
    Example:
        >>> psi_result = batch_psi_analysis(df, ['age', 'income'], 'apply_month', 
        ...                                 '2023-01', ['2023-02', '2023-03'])
        >>> print(psi_result[['特征名', '对比期', 'PSI值', '稳定性']])
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
    
    # 基准期数据
    base_df = df[df[date_col] == base_period]
    
    results = []
    
    for period in compare_periods:
        period_df = df[df[date_col] == period]
        
        if len(period_df) == 0:
            continue
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            try:
                psi_result = psi_analysis(base_df, period_df, feature, n_bins)
                results.append({
                    '特征名': psi_result['特征名'],
                    '基准期': base_period,
                    '对比期': period,
                    'PSI值': psi_result['PSI值'],
                    '稳定性': psi_result['稳定性'],
                })
            except Exception as e:
                results.append({
                    '特征名': feature,
                    '基准期': base_period,
                    '对比期': period,
                    'PSI值': np.nan,
                    '稳定性': '计算失败',
                })
    
    return pd.DataFrame(results)


def csi_analysis(df: pd.DataFrame,
                feature: str,
                target: str,
                date_col: str,
                base_period: str,
                compare_period: str) -> Dict[str, Union[str, float]]:
    """CSI特征稳定性分析.
    
    复用 hscredit.core.metrics.csi
    
    :param df: 输入数据
    :param feature: 特征名
    :param target: 目标变量名
    :param date_col: 日期列名
    :param base_period: 基准期
    :param compare_period: 对比期
    :return: CSI分析结果
    
    Example:
        >>> result = csi_analysis(df, 'score', 'fpd15', 'apply_month', '2023-01', '2023-02')
        >>> print(f"CSI: {result['CSI值']}")
    """
    from ..metrics import csi
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
    
    base_df = df[df[date_col] == base_period]
    compare_df = df[df[date_col] == compare_period]
    
    # 计算CSI
    csi_value = csi(base_df[feature], compare_df[feature], 
                   base_df[target], compare_df[target])
    
    return {
        '特征名': feature,
        '基准期': base_period,
        '对比期': compare_period,
        'CSI值': round(csi_value, 4),
    }


def time_psi_tracking(df: pd.DataFrame,
                     features: List[str],
                     date_col: str,
                     freq: str = 'M',
                     n_bins: int = 10) -> pd.DataFrame:
    """PSI时序追踪分析.
    
    计算每个特征在不同时间段的PSI值变化趋势
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param freq: 时间频率
    :param n_bins: 分箱数
    :return: PSI时序追踪DataFrame
    
    Example:
        >>> tracking = time_psi_tracking(df, ['age', 'income'], 'apply_date')
        >>> print(tracking.pivot(index='时间周期', columns='特征名', values='PSI值'))
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # 准备数据
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 创建时间周期
    if freq == 'M':
        df['时间周期'] = df[date_col].dt.to_period('M').astype(str)
    elif freq == 'W':
        df['时间周期'] = df[date_col].dt.to_period('W').astype(str)
    elif freq == 'Q':
        df['时间周期'] = df[date_col].dt.to_period('Q').astype(str)
    
    # 获取所有周期
    periods = sorted(df['时间周期'].unique())
    
    if len(periods) < 2:
        return pd.DataFrame({'信息': ['时间周期不足，无法计算PSI']})
    
    # 以第一个周期为基准
    base_period = periods[0]
    base_df = df[df['时间周期'] == base_period]
    
    results = []
    
    for period in periods[1:]:
        period_df = df[df['时间周期'] == period]
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            try:
                psi_result = psi_analysis(base_df, period_df, feature, n_bins)
                results.append({
                    '特征名': psi_result['特征名'],
                    '基准期': base_period,
                    '时间周期': period,
                    'PSI值': psi_result['PSI值'],
                    '稳定性': psi_result['稳定性'],
                })
            except:
                pass
    
    return pd.DataFrame(results)


def stability_report(df: pd.DataFrame,
                    features: List[str],
                    date_col: str,
                    target: str = None,
                    psi_threshold: float = 0.1) -> pd.DataFrame:
    """综合稳定性报告.
    
    包含PSI分析和时间稳定性评估
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名
    :param target: 目标变量名（可选）
    :param psi_threshold: PSI阈值
    :return: 综合稳定性报告DataFrame
    
    Example:
        >>> report = stability_report(df, feature_list, 'apply_date')
        >>> print(report[['特征名', '平均PSI', '最大PSI', '不稳定期数']])
    """
    validate_dataframe(df, required_cols=[date_col])
    
    # PSI时序追踪
    psi_tracking = time_psi_tracking(df, features, date_col)
    
    if psi_tracking.empty or '信息' in psi_tracking.columns:
        return pd.DataFrame({'信息': ['数据不足以生成稳定性报告']})
    
    # 汇总统计
    results = []
    
    for feature in features:
        feature_psi = psi_tracking[psi_tracking['特征名'] == feature]
        
        if len(feature_psi) == 0:
            continue
        
        avg_psi = feature_psi['PSI值'].mean()
        max_psi = feature_psi['PSI值'].max()
        unstable_count = (feature_psi['PSI值'] > psi_threshold).sum()
        
        results.append({
            '特征名': feature,
            '平均PSI': round(avg_psi, 4),
            '最大PSI': round(max_psi, 4),
            '统计期数': len(feature_psi),
            '不稳定期数': int(unstable_count),
            '不稳定期占比(%)': round(unstable_count / len(feature_psi) * 100, 2),
            '稳定性评级': psi_rating(avg_psi),
        })
    
    return pd.DataFrame(results).sort_values('平均PSI', ascending=False)


def psi_cross_analysis(
    df: pd.DataFrame,
    features: List[str],
    date_col: Optional[str] = None,
    group_col: Optional[str] = None,
    freq: str = 'M',
    n_bins: int = 10,
    return_matrix: bool = True
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """PSI交叉分析 - 计算两两组之间的PSI矩阵.
    
    支持两种方式分组：
    1. 自动分组：指定日期列(date_col)和频率(freq)，按时间自动分组
    2. 手工分组：指定分组列(group_col)，使用已有分组
    
    计算每两组之间的PSI值，返回PSI矩阵或长格式DataFrame.
    
    :param df: 输入数据
    :param features: 特征列表
    :param date_col: 日期列名（自动分组模式）
    :param group_col: 分组列名（手工分组模式）
    :param freq: 时间频率，'D'日/'W'周/'M'月/'Q'季度，默认'M'
    :param n_bins: PSI计算分箱数，默认10
    :param return_matrix: 是否返回矩阵格式，True返回方阵矩阵，False返回长格式
    :return: 
        - return_matrix=True: 返回 {特征名: PSI矩阵DataFrame} 字典
        - return_matrix=False: 返回长格式DataFrame [特征名, 组1, 组2, PSI值, 稳定性]
    
    Example:
        >>> # 自动按月份分组
        >>> result = psi_cross_analysis(df, ['age', 'income'], 
        ...                             date_col='apply_date', freq='M')
        >>> print(result['age'])  # 查看age特征的PSI矩阵
        
        >>> # 手工分组
        >>> result = psi_cross_analysis(df, ['score'], group_col='channel')
        
        >>> # 返回长格式
        >>> result = psi_cross_analysis(df, ['age'], date_col='apply_date', 
        ...                             freq='M', return_matrix=False)
    """
    from ..metrics import psi_table
    
    validate_dataframe(df)
    
    # 检查参数
    if date_col is None and group_col is None:
        raise ValueError("必须指定 date_col（自动分组）或 group_col（手工分组）之一")
    
    if date_col is not None and group_col is not None:
        raise ValueError("不能同时指定 date_col 和 group_col，请选择一种分组方式")
    
    df = df.copy()
    
    # 确定分组列
    if date_col is not None:
        # 自动分组模式
        if date_col not in df.columns:
            raise ValueError(f"日期列 '{date_col}' 不存在")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 根据频率创建分组
        if freq == 'D':
            df['_psi_group'] = df[date_col].dt.date.astype(str)
        elif freq == 'W':
            df['_psi_group'] = df[date_col].dt.to_period('W').astype(str)
        elif freq == 'M':
            df['_psi_group'] = df[date_col].dt.to_period('M').astype(str)
        elif freq == 'Q':
            df['_psi_group'] = df[date_col].dt.to_period('Q').astype(str)
        else:
            raise ValueError("freq 必须是 'D'/'W'/'M'/'Q' 之一")
        
        group_col = '_psi_group'
    else:
        # 手工分组模式
        if group_col not in df.columns:
            raise ValueError(f"分组列 '{group_col}' 不存在")
    
    # 获取所有分组
    groups = sorted(df[group_col].unique())
    
    if len(groups) < 2:
        return pd.DataFrame({'信息': ['分组数不足，至少需要2个组']})
    
    results = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        # 为每个特征计算PSI矩阵
        psi_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
        stability_matrix = pd.DataFrame(index=groups, columns=groups, dtype=object)
        
        # 计算两两之间的PSI
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i == j:
                    psi_matrix.loc[group1, group2] = 0.0
                    stability_matrix.loc[group1, group2] = '相同组'
                elif i < j:
                    # 只计算上三角，下三角对称
                    data1 = df[df[group_col] == group1][feature].dropna()
                    data2 = df[df[group_col] == group2][feature].dropna()
                    
                    if len(data1) == 0 or len(data2) == 0:
                        psi_value = np.nan
                        stability = '数据不足'
                    else:
                        try:
                            # 使用 psi_table 计算 PSI
                            psi_df = psi_table(data1, data2, max_n_bins=n_bins)
                            psi_value = psi_df['PSI贡献'].sum()
                            stability = psi_rating(psi_value)
                        except Exception as e:
                            psi_value = np.nan
                            stability = '计算失败'
                    
                    psi_matrix.loc[group1, group2] = psi_value
                    psi_matrix.loc[group2, group1] = psi_value
                    stability_matrix.loc[group1, group2] = stability
                    stability_matrix.loc[group2, group1] = stability
        
        if return_matrix:
            results[feature] = psi_matrix
        else:
            # 转换为长格式
            long_results = []
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if i != j:  # 排除对角线
                        long_results.append({
                            '特征名': feature,
                            '基准组': group1,
                            '对比组': group2,
                            'PSI值': psi_matrix.loc[group1, group2],
                            '稳定性': stability_matrix.loc[group1, group2]
                        })
            results[feature] = pd.DataFrame(long_results)
    
    # 如果只有一个特征，直接返回结果；否则返回字典
    if len(results) == 1:
        return list(results.values())[0]
    return results


# ---------------------------------------------------------------------------
# feature_drift_report
# ---------------------------------------------------------------------------

def feature_drift_report(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    features=None,
    method: str = 'psi',
    psi_bins: int = 10,
) -> pd.DataFrame:
    """特征偏移综合报告.

    对一批特征计算基准集与目标集之间的偏移指标，
    返回含 PSI / 均值变化 / 缺失率变化 / 偏移等级 的汇总表，
    可用于上线前的数据质量检查和模型再训练决策。

    :param df_base: 基准数据集（如训练集）
    :param df_target: 目标数据集（如生产数据）
    :param features: 特征列表，None 时取两个数据集的公共数值列
    :param method: 偏移度量方法，目前仅支持 'psi'
    :param psi_bins: PSI 计算分箱数，默认 10
    :return: 特征偏移报告 DataFrame，含 特征名/PSI/均值变化/缺失率变化/偏移等级

    Example:
        >>> report = feature_drift_report(train_df, prod_df)
        >>> drifted = report[report['偏移等级'] == '显著偏移']
        >>> print(drifted[['特征名', 'PSI', '偏移等级']])
    """
    validate_dataframe(df_base)
    validate_dataframe(df_target)

    if features is None:
        num_cols_base = set(df_base.select_dtypes(include=[np.number]).columns)
        num_cols_tgt = set(df_target.select_dtypes(include=[np.number]).columns)
        features = sorted(num_cols_base & num_cols_tgt)
    else:
        features = list(features)

    rows = []
    for feat in features:
        if feat not in df_base.columns or feat not in df_target.columns:
            continue
        base_col = pd.to_numeric(df_base[feat], errors='coerce')
        tgt_col = pd.to_numeric(df_target[feat], errors='coerce')
        base_arr = base_col.dropna().values
        tgt_arr = tgt_col.dropna().values

        if len(base_arr) == 0 or len(tgt_arr) == 0:
            rows.append({'特征名': feat, 'PSI': np.nan, '偏移等级': '数据不足', '均值变化(%)': np.nan,
                         '缺失率变化(%)': np.nan})
            continue

        try:
            from ..metrics import psi_table as _psi_table
            psi_df = _psi_table(pd.Series(base_arr), pd.Series(tgt_arr), max_n_bins=psi_bins)
            psi_val = float(psi_df['PSI贡献'].sum())
        except Exception:
            psi_val = np.nan

        mean_base = float(base_col.mean())
        mean_tgt = float(tgt_col.mean())
        mean_change = round((mean_tgt - mean_base) / abs(mean_base) * 100, 2) if mean_base != 0 else np.nan

        missing_base = round(df_base[feat].isna().mean() * 100, 2)
        missing_tgt = round(df_target[feat].isna().mean() * 100, 2)

        rows.append({
            '特征名': feat,
            'PSI': round(psi_val, 4) if not np.isnan(psi_val) else np.nan,
            '偏移等级': psi_rating(psi_val) if not np.isnan(psi_val) else '未知',
            '基准均值': round(mean_base, 4),
            '目标均值': round(mean_tgt, 4),
            '均值变化(%)': mean_change,
            '基准缺失率(%)': missing_base,
            '目标缺失率(%)': missing_tgt,
            '缺失率变化(%)': round(missing_tgt - missing_base, 2),
        })

    result = pd.DataFrame(rows)
    if not result.empty and 'PSI' in result.columns:
        result = result.sort_values('PSI', ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# score_drift_report
# ---------------------------------------------------------------------------

def score_drift_report(
    score_base: pd.Series,
    score_target: pd.Series,
    y_base=None,
    y_target=None,
    n_bins: int = 10,
) -> dict:
    """评分分布偏移报告.

    比较基准评分与目标评分分布，输出 PSI、均值/中位数/标准差变化，
    以及可选的 KS/AUC 变化，用于模型监控场景的评分稳定性预警。

    :param score_base: 基准评分 Series
    :param score_target: 目标评分 Series
    :param y_base: 基准真实标签（0/1），可选，提供时计算模型性能变化
    :param y_target: 目标真实标签（0/1），可选
    :param n_bins: PSI 分箱数，默认 10
    :return: 偏移报告字典，含 PSI / 分布统计 / 模型性能变化

    Example:
        >>> report = score_drift_report(train_score, prod_score, y_base=train_y, y_target=prod_y)
        >>> print(report['PSI'], report['偏移等级'])
        >>> print(report['分布统计'])
    """
    s_base = pd.to_numeric(score_base, errors='coerce').dropna()
    s_tgt = pd.to_numeric(score_target, errors='coerce').dropna()

    try:
        from ..metrics import psi_table as _psi_table
        psi_df = _psi_table(s_base, s_tgt, max_n_bins=n_bins)
        psi_val = float(psi_df['PSI贡献'].sum())
    except Exception:
        psi_val = np.nan

    def _stats(s: pd.Series) -> dict:
        return {
            '样本数': len(s),
            '均值': round(float(s.mean()), 4),
            '中位数': round(float(s.median()), 4),
            '标准差': round(float(s.std()), 4),
            'P10': round(float(np.percentile(s, 10)), 4),
            'P90': round(float(np.percentile(s, 90)), 4),
        }

    stats_base = _stats(s_base)
    stats_tgt = _stats(s_tgt)
    stats_base['数据集'] = '基准'
    stats_tgt['数据集'] = '目标'
    dist_df = pd.DataFrame([stats_base, stats_tgt])

    report: dict = {
        'PSI': round(psi_val, 4) if not np.isnan(psi_val) else np.nan,
        '偏移等级': psi_rating(psi_val) if not np.isnan(psi_val) else '未知',
        '分布统计': dist_df,
        '分箱明细': psi_df if not np.isnan(psi_val) else None,
    }

    if y_base is not None and y_target is not None:
        from ..metrics import ks as _ks, auc as _auc  # type: ignore[attr-defined]
        yb = pd.to_numeric(y_base, errors='coerce')
        yt = pd.to_numeric(y_target, errors='coerce')
        mask_b = yb.notna() & pd.to_numeric(score_base, errors='coerce').notna()
        mask_t = yt.notna() & pd.to_numeric(score_target, errors='coerce').notna()
        s_b_aligned = pd.to_numeric(score_base, errors='coerce')[mask_b]
        s_t_aligned = pd.to_numeric(score_target, errors='coerce')[mask_t]
        try:
            ks_base = round(float(_ks(yb[mask_b], s_b_aligned)), 4)
            ks_tgt = round(float(_ks(yt[mask_t], s_t_aligned)), 4)
            auc_base = round(float(_auc(yb[mask_b], s_b_aligned)), 4)
            auc_tgt = round(float(_auc(yt[mask_t], s_t_aligned)), 4)
        except Exception:
            ks_base = ks_tgt = auc_base = auc_tgt = np.nan
        report['模型性能'] = {
            '基准KS': ks_base, '目标KS': ks_tgt,
            'KS变化': round(ks_tgt - ks_base, 4) if not np.isnan(ks_base) else np.nan,
            '基准AUC': auc_base, '目标AUC': auc_tgt,
            'AUC变化': round(auc_tgt - auc_base, 4) if not np.isnan(auc_base) else np.nan,
        }

    return report
