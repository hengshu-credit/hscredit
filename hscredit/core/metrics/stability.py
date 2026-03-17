"""稳定性指标计算.

提供评估模型稳定性和分布变化的指标。

主要指标:
- PSI (Population Stability Index): 评估总体分布稳定性
- CSI (Characteristic Stability Index): 评估特征分布稳定性
- PSI_table: PSI详细统计表
- CSI_table: CSI详细统计表
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from scipy.stats import chi2_contingency


def PSI(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        bins: Optional[Union[int, list]] = 10) -> float:
    """计算Population Stability Index (PSI).

    PSI用于衡量两个分布之间的差异，评估模型或特征的稳定性。
    PSI值越小表示分布越稳定，越大表示分布变化越大。

    PSI计算公式:
    PSI = Σ [(实际占比 - 期望占比) * ln(实际占比 / 期望占比)]

    PSI分级标准:
    - PSI < 0.1: 没有显著变化
    - 0.1 ≤ PSI < 0.25: 有轻微变化
    - PSI ≥ 0.25: 有显著变化

    :param expected: 期望分布数据 (通常是训练集或基准数据)
    :param actual: 实际分布数据 (通常是测试集或新数据)
    :param bins: 分箱数量或自定义分箱边界
    :return: PSI值
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # 处理分箱
    if isinstance(bins, int):
        # 等频分箱
        combined = np.concatenate([expected, actual])
        if len(np.unique(combined)) <= bins:
            # 如果唯一值少于bins，直接使用唯一值
            bin_edges = np.sort(np.unique(combined))
        else:
            # 使用分位数分箱
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(combined, quantiles)
            bin_edges = np.unique(bin_edges)  # 去重
    else:
        bin_edges = np.array(bins)

    # 计算期望分布的直方图
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # 转换为比例
    expected_prop = expected_hist / len(expected)
    actual_prop = actual_hist / len(actual)

    # 避免除零和log(0)
    epsilon = 1e-10
    expected_prop = np.where(expected_prop == 0, epsilon, expected_prop)
    actual_prop = np.where(actual_prop == 0, epsilon, actual_prop)

    # 计算PSI
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))

    return psi


def CSI(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series]) -> float:
    """计算Characteristic Stability Index (CSI).

    CSI是PSI的变体，用于衡量特征分布的稳定性。
    计算方法与PSI相同，但通常用于特征级别的稳定性评估。

    :param expected: 期望分布数据
    :param actual: 实际分布数据
    :return: CSI值
    """
    # CSI的计算与PSI相同，只是命名不同
    return PSI(expected, actual, bins=10)


def PSI_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              bins: Optional[Union[int, list]] = 10) -> pd.DataFrame:
    """计算PSI详细统计表.

    提供每个分箱的PSI贡献信息。

    :param expected: 期望分布数据 (通常是训练集或基准数据)
    :param actual: 实际分布数据 (通常是测试集或新数据)
    :param bins: 分箱数量或自定义分箱边界，默认为10
    :return: PSI统计表，包含以下列:
        - 分箱区间: 分箱区间
        - 期望样本数: 期望分布样本数
        - 实际样本数: 实际分布样本数
        - 期望占比: 期望分布占比
        - 实际占比: 实际分布占比
        - PSI贡献: 该箱的PSI贡献
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # 处理分箱
    if isinstance(bins, int):
        combined = np.concatenate([expected, actual])
        if len(np.unique(combined)) <= bins:
            bin_edges = np.sort(np.unique(combined))
        else:
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(combined, quantiles)
            bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.array(bins)

    # 计算直方图
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # 转换为比例
    expected_prop = expected_hist / len(expected)
    actual_prop = actual_hist / len(actual)

    # 计算PSI贡献
    epsilon = 1e-10
    psi_contrib = (actual_prop - expected_prop) * np.log(
        np.where(actual_prop == 0, epsilon, actual_prop) /
        np.where(expected_prop == 0, epsilon, expected_prop)
    )

    results = []
    for i in range(len(bin_edges) - 1):
        if i == 0:
            bin_label = f"(-inf, {bin_edges[i + 1]:.3f}]"
        elif i == len(bin_edges) - 2:
            bin_label = f"({bin_edges[i]:.3f}, +inf)"
        else:
            bin_label = f"({bin_edges[i]:.3f}, {bin_edges[i + 1]:.3f}]"

        results.append({
            '分箱区间': bin_label,
            '期望样本数': expected_hist[i],
            '实际样本数': actual_hist[i],
            '期望占比': expected_prop[i],
            '实际占比': actual_prop[i],
            'PSI贡献': psi_contrib[i]
        })

    return pd.DataFrame(results)


def CSI_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              bins: Optional[Union[int, list]] = 10) -> pd.DataFrame:
    """计算CSI详细统计表.

    CSI是PSI的变体，用于衡量特征分布的稳定性。

    :param expected: 期望分布数据
    :param actual: 实际分布数据
    :param bins: 分箱数量或自定义分箱边界，默认为10
    :return: CSI统计表
    """
    # CSI_table与PSI_table相同
    return PSI_table(expected, actual, bins)
