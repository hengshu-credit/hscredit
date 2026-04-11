"""特征评估指标.

提供评估特征预测能力和质量的指标。

主要指标:
- iv: Information Value，衡量特征的预测能力
- iv_table: IV详细统计表
- chi2_test: 卡方独立性检验
- cramers_v: Cramer's V关联强度
- feature_importance: 基于树模型的特征重要性
- bin_stats: 分箱统计计算
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._base import _validate_same_length, _validate_binary_target, _create_bin_edges
from ._binning import compute_bin_stats, _chi2_by_bin


def iv(y_true: Union[np.ndarray, pd.Series],
       feature: Union[np.ndarray, pd.Series],
       method: str = 'quantile',
       max_n_bins: int = 10,
       min_bin_size: float = 0.01,
       **kwargs) -> float:
    """计算Information Value (信息价值).

    IV用于衡量特征的预测能力，值越大表示特征的区分能力越强。

    IV分级标准:
    - IV < 0.02: 无预测能力
    - 0.02 <= IV < 0.1: 弱预测能力
    - 0.1 <= IV < 0.3: 中等预测能力
    - 0.3 <= IV < 0.5: 强预测能力
    - IV >= 0.5: 极强预测能力

    :param y_true: 目标变量 (0/1)
    :param feature: 特征变量
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: IV值
    """
    table = iv_table(y_true, feature, method, max_n_bins, min_bin_size, **kwargs)
    return table['分档IV值'].sum()


def iv_table(y_true: Union[np.ndarray, pd.Series],
             feature: Union[np.ndarray, pd.Series],
             method: str = 'quantile',
             max_n_bins: int = 10,
             min_bin_size: float = 0.01,
             **kwargs) -> pd.DataFrame:
    """计算IV详细统计表.

    :param y_true: 目标变量 (0/1)
    :param feature: 特征变量
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: IV统计表
    """
    y_true = np.asarray(y_true)
    feature = np.asarray(feature)

    _validate_same_length(y_true, feature, ("y_true", "feature"))
    _validate_binary_target(y_true)

    # 移除缺失值
    valid_mask = ~(pd.isna(feature) | pd.isna(y_true))
    feature_clean = feature[valid_mask]
    y_true_clean = y_true[valid_mask]

    if len(feature_clean) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")

    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning

    df = pd.DataFrame({'feature': feature_clean, 'target': y_true_clean})

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df[['feature']], df['target'])
    bins = binner.transform(df[['feature']], metric='indices').values.flatten()

    bin_labels = None
    if 'feature' in binner.bin_tables_:
        bin_table = binner.bin_tables_['feature']
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()

    return compute_bin_stats(bins, y_true_clean, bin_labels=bin_labels)


def chi2_test(x: Union[np.ndarray, pd.Series],
              y: Union[np.ndarray, pd.Series]) -> Tuple[float, float]:
    """计算卡方独立性检验.

    :param x: 特征变量（可以是分类或数值，数值会自动分箱）
    :param y: 目标变量 (0/1)
    :return: (卡方统计量, p值)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_same_length(x, y, ("x", "y"))

    # 如果x是数值型，进行分箱
    if np.issubdtype(x.dtype, np.number):
        bin_edges = _create_bin_edges(x, 10)
        x = np.digitize(x, bin_edges[1:-1])

    contingency = pd.crosstab(x, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 1.0

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    return chi2_stat, p_value


def cramers_v(x: Union[np.ndarray, pd.Series],
              y: Union[np.ndarray, pd.Series]) -> float:
    """计算Cramer's V关联强度.

    Cramer's V是卡方检验的效应量，范围0-1，值越大表示关联越强。

    :param x: 特征变量
    :param y: 目标变量
    :return: Cramer's V值
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_same_length(x, y, ("x", "y"))

    # 如果x是数值型，进行分箱
    if np.issubdtype(x.dtype, np.number):
        bin_edges = _create_bin_edges(x, 10)
        x = np.digitize(x, bin_edges[1:-1])

    contingency = pd.crosstab(x, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1

    if min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def feature_importance(X: Union[pd.DataFrame, np.ndarray],
                       y: Union[np.ndarray, pd.Series],
                       method: str = 'gini',
                       **kwargs) -> pd.Series:
    """计算特征重要性.

    使用树模型计算特征重要性。

    :param X: 特征矩阵
    :param y: 目标变量
    :param method: 计算方法，'gini'或'entropy'
    :param kwargs: 其他传递给模型的参数
    :return: 特征重要性得分
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    if method == 'gini':
        max_depth = kwargs.get('max_depth', 3)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif method == 'entropy':
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 3)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=feature_names)


def feature_summary(feature: Union[np.ndarray, pd.Series],
                   y: Optional[Union[np.ndarray, pd.Series]] = None) -> pd.DataFrame:
    """计算特征描述统计.

    :param feature: 特征变量
    :param y: 目标变量（可选），如果提供会计算目标关系统计
    :return: 描述统计DataFrame
    """
    feature = np.asarray(feature)

    result = {
        '样本数': len(feature),
        '缺失数': np.sum(pd.isna(feature)),
        '缺失率': np.mean(pd.isna(feature)),
        '唯一值数': len(np.unique(feature[~pd.isna(feature)])),
    }

    # 数值型统计
    if np.issubdtype(feature.dtype, np.number):
        valid_feature = feature[~np.isnan(feature)]
        result.update({
            '均值': np.mean(valid_feature),
            '标准差': np.std(valid_feature),
            '最小值': np.min(valid_feature),
            '最大值': np.max(valid_feature),
            '中位数': np.median(valid_feature),
        })

    # 如果有目标变量，计算相关性
    if y is not None:
        y = np.asarray(y)
        valid_mask = ~(pd.isna(feature) | pd.isna(y))
        if valid_mask.sum() > 0:
            # 计算IV
            try:
                iv_value = iv(y[valid_mask], feature[valid_mask])
                result['IV'] = iv_value
            except:
                result['IV'] = np.nan

    return pd.DataFrame([result])
