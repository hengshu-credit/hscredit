"""指标计算内部工具函数.

此模块包含所有指标计算的基础工具函数，不对外暴露。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List


def _validate_binary_target(y: np.ndarray, name: str = "y") -> None:
    """验证目标变量是否为二分类."""
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1")


def _validate_same_length(a: np.ndarray, b: np.ndarray, names: Tuple[str, str] = ("a", "b")) -> None:
    """验证两个数组长度相同."""
    if len(a) != len(b):
        raise ValueError(f"{names[0]} and {names[1]} must have same length")


def _handle_missing_values(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """处理缺失值，返回有效数据的掩码."""
    # 计算所有数组的缺失值掩码
    valid_mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        if arr.dtype.kind in 'fc':  # 浮点或复数
            valid_mask &= ~np.isnan(arr)
    return tuple(arr[valid_mask] for arr in arrays)


def _woe_iv_vectorized(
    good_counts: np.ndarray,
    bad_counts: np.ndarray,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """向量化计算WOE和IV.
    
    :param good_counts: 每个箱的好样本数
    :param bad_counts: 每个箱的坏样本数
    :param epsilon: 平滑参数，避免log(0)
    :return: (woe_array, bin_iv_array, total_iv)
    """
    good_counts = np.asarray(good_counts, dtype=np.float64)
    bad_counts = np.asarray(bad_counts, dtype=np.float64)

    total_good = good_counts.sum()
    total_bad = bad_counts.sum()

    if total_good == 0 or total_bad == 0:
        return np.zeros(len(good_counts)), np.zeros(len(good_counts)), 0.0

    # 平滑处理
    good_counts_smooth = np.where(good_counts == 0, epsilon, good_counts)
    bad_counts_smooth = np.where(bad_counts == 0, epsilon, bad_counts)
    
    total_good_smooth = good_counts_smooth.sum()
    total_bad_smooth = bad_counts_smooth.sum()

    # 计算占比
    good_distr = good_counts_smooth / total_good_smooth
    bad_distr = bad_counts_smooth / total_bad_smooth

    # 计算WOE和IV
    woe = np.log(bad_distr / good_distr)
    bin_iv = (bad_distr - good_distr) * woe
    total_iv = bin_iv.sum()

    return woe, bin_iv, total_iv


def _create_bin_edges(
    feature: np.ndarray,
    bins: Union[int, List, np.ndarray]
) -> np.ndarray:
    """创建分箱边界.
    
    :param feature: 特征数组
    :param bins: 分箱数量或自定义边界
    :return: 分箱边界数组
    """
    # 移除缺失值
    feature_clean = feature[~np.isnan(feature)]
    
    if isinstance(bins, int):
        unique_vals = np.unique(feature_clean)
        if len(unique_vals) <= bins:
            # 如果唯一值少于bins，直接使用唯一值
            bin_edges = np.sort(unique_vals)
            if len(bin_edges) > 0:
                bin_edges = np.concatenate([
                    [bin_edges[0] - 1e-10],
                    bin_edges,
                    [bin_edges[-1] + 1e-10]
                ])
        else:
            # 使用分位数分箱
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(feature_clean, quantiles)
            bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.array(bins)

    return bin_edges
