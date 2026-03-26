"""稳定性指标计算.

提供评估模型稳定性和分布变化的指标.

主要指标:
- psi: Population Stability Index，评估总体分布稳定性
- csi: Characteristic Stability Index，评估特征分布稳定性
- psi_rating: PSI稳定性评级
- batch_psi: 批量计算多特征PSI
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from scipy.stats import chi2_contingency

from ._base import _create_bin_edges


def psi(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        method: str = 'quantile',
        max_n_bins: int = 10,
        min_bin_size: float = 0.01,
        **kwargs) -> float:
    """计算Population Stability Index (PSI).

    PSI用于衡量两个分布之间的差异，评估模型或特征的稳定性。
    PSI值越小表示分布越稳定。

    PSI分级标准:
    - PSI < 0.1: 没有显著变化
    - 0.1 <= PSI < 0.25: 有轻微变化
    - PSI >= 0.25: 有显著变化

    :param expected: 期望分布数据 (通常是训练集或基准数据)
    :param actual: 实际分布数据 (通常是测试集或新数据)
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: PSI值
    """
    table = psi_table(expected, actual, method, max_n_bins, min_bin_size, **kwargs)
    return table['PSI贡献'].sum()


def psi_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              bins: int = None,
              **kwargs) -> pd.DataFrame:
    """计算PSI详细统计表.

    :param expected: 期望分布数据
    :param actual: 实际分布数据
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param bins: 分箱数（兼容参数，等同于max_n_bins）
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: PSI统计表
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # 兼容 bins 参数
    if bins is not None:
        max_n_bins = bins

    # 移除缺失值
    expected_clean = expected[~pd.isna(expected)]
    actual_clean = actual[~pd.isna(actual)]

    # 合并数据确定分箱边界
    combined = np.concatenate([expected_clean, actual_clean])

    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning

    # 构建DataFrame
    df_expected = pd.DataFrame({'value': expected_clean, 'is_expected': 1})
    df_actual = pd.DataFrame({'value': actual_clean, 'is_expected': 0})
    df_combined = pd.concat([df_expected, df_actual], ignore_index=True)

    # 创建临时目标用于分箱
    dummy_target = np.random.randint(0, 2, size=len(df_combined))

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df_combined[['value']], dummy_target)

    # 分别转换expected和actual
    bins_expected = binner.transform(df_expected[['value']], metric='indices').values.flatten()
    bins_actual = binner.transform(df_actual[['value']], metric='indices').values.flatten()

    # 计算每个箱的统计
    unique_bins = sorted(set(bins_expected) | set(bins_actual))

    results = []
    epsilon = 1e-10

    total_expected = len(expected_clean)
    total_actual = len(actual_clean)

    for bin_idx in unique_bins:
        expected_count = np.sum(bins_expected == bin_idx)
        actual_count = np.sum(bins_actual == bin_idx)

        expected_prop = expected_count / total_expected if total_expected > 0 else epsilon
        actual_prop = actual_count / total_actual if total_actual > 0 else epsilon

        # 避免除零
        expected_prop = max(expected_prop, epsilon)
        actual_prop = max(actual_prop, epsilon)

        psi_contrib = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)

        # 获取分箱标签
        bin_label = f"Bin_{bin_idx}"
        if 'value' in binner.bin_tables_:
            bin_table = binner.bin_tables_['value']
            if bin_idx < len(bin_table) and '分箱标签' in bin_table.columns:
                bin_label = bin_table.iloc[bin_idx]['分箱标签']

        results.append({
            '分箱': bin_label,
            '期望样本数': expected_count,
            '实际样本数': actual_count,
            '期望占比': expected_prop,
            '实际占比': actual_prop,
            'PSI贡献': psi_contrib,
        })

    return pd.DataFrame(results)


def psi_rating(psi_value: float) -> str:
    """根据PSI值返回稳定性评级.

    :param psi_value: PSI值
    :return: 稳定性评级描述
    """
    if psi_value < 0.1:
        return "没有显著变化 (PSI < 0.1)"
    elif psi_value < 0.25:
        return "有轻微变化 (0.1 <= PSI < 0.25)"
    else:
        return "有显著变化 (PSI >= 0.25)"


def csi(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        method: str = 'quantile',
        max_n_bins: int = 10,
        min_bin_size: float = 0.01,
        **kwargs) -> float:
    """计算Characteristic Stability Index (CSI).

    CSI是PSI的变体，用于衡量特征分布的稳定性。

    :param expected: 期望分布数据
    :param actual: 实际分布数据
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: CSI值
    """
    return psi(expected, actual, method, max_n_bins, min_bin_size, **kwargs)


def csi_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """计算CSI详细统计表.

    :param expected: 期望分布数据
    :param actual: 实际分布数据
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: CSI统计表
    """
    table = psi_table(expected, actual, method, max_n_bins, min_bin_size, **kwargs)
    table = table.rename(columns={'PSI贡献': 'CSI贡献'})
    return table


def batch_psi(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              features: Optional[List[str]] = None,
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """批量计算多特征的PSI.

    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param features: 需要计算的特征列表，默认全部
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分箱数，默认10
    :param min_bin_size: 每箱最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: PSI结果DataFrame
    """
    if features is None:
        features = list(X_train.columns)

    results = []
    for feature in features:
        if feature in X_train.columns and feature in X_test.columns:
            try:
                psi_value = psi(
                    X_train[feature], X_test[feature],
                    method, max_n_bins, min_bin_size, **kwargs
                )
                rating = psi_rating(psi_value)
                results.append({
                    '特征': feature,
                    'PSI': psi_value,
                    '评级': rating,
                })
            except Exception as e:
                results.append({
                    '特征': feature,
                    'PSI': np.nan,
                    '评级': f'计算失败: {str(e)}',
                })

    return pd.DataFrame(results)


# 向后兼容（Deprecated）
PSI = psi
PSI_table = psi_table
CSI = csi
CSI_table = csi_table
