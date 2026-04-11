"""分类模型评估指标.

提供二分类模型评估的核心指标计算功能。

主要指标:
- ks: Kolmogorov-Smirnov统计量
- auc: ROC曲线下面积
- gini: 基尼系数
- accuracy: 准确率
- precision: 精确率
- recall: 召回率
- f1: F1分数
- ks_bucket: 分桶KS统计
- roc_curve: ROC曲线数据
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix as sk_confusion_matrix,
    classification_report as sk_classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

from ._base import _validate_same_length, _validate_binary_target
from ._binning import compute_bin_stats


def ks(y_true: Union[np.ndarray, pd.Series],
       y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算Kolmogorov-Smirnov统计量.

    KS值衡量模型区分正负样本的能力，值越大区分效果越好。
    KS = max(TPR - FPR)，其中TPR为正样本累积率，FPR为负样本累积率。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: KS统计量，取值范围[0, 1]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))
    _validate_binary_target(y_true)

    # 按预测概率降序排序
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    n_total = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg

    return np.max(np.abs(cum_pos - cum_neg))


def auc(y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算ROC曲线下的面积(AUC).

    AUC值在0.5-1.0之间，越接近1.0模型效果越好。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: AUC值，取值范围[0.5, 1.0]
    """
    return roc_auc_score(y_true, y_prob)


def gini(y_true: Union[np.ndarray, pd.Series],
         y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算基尼系数.

    基尼系数 = 2 * AUC - 1
    范围从-1到1，越接近1表示模型区分能力越强。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: 基尼系数，取值范围[-1, 1]
    """
    return 2 * auc(y_true, y_prob) - 1


def accuracy(y_true: Union[np.ndarray, pd.Series],
             y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算准确率.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确率
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true: Union[np.ndarray, pd.Series],
              y_pred: Union[np.ndarray, pd.Series],
              average: str = 'binary') -> float:
    """计算精确率.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，默认'binary'
    :return: 精确率
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true: Union[np.ndarray, pd.Series],
           y_pred: Union[np.ndarray, pd.Series],
           average: str = 'binary') -> float:
    """计算召回率.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，默认'binary'
    :return: 召回率
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_true: Union[np.ndarray, pd.Series],
       y_pred: Union[np.ndarray, pd.Series],
       average: str = 'binary') -> float:
    """计算F1分数.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，默认'binary'
    :return: F1分数
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def ks_bucket(y_true: Union[np.ndarray, pd.Series],
              y_prob: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """计算分桶KS统计表.

    将样本按预测概率分为多个桶，计算每个桶的KS统计信息。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :param method: 分箱方法，默认'quantile'
    :param max_n_bins: 最大分桶数量，默认为10
    :param min_bin_size: 每桶最小样本占比，默认0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 包含桶统计信息的DataFrame
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    _validate_same_length(y_true, y_prob, ("y_true", "y_prob"))

    # 移除缺失值
    valid_mask = ~(pd.isna(y_prob) | pd.isna(y_true))
    y_prob_clean = y_prob[valid_mask]
    y_true_clean = y_true[valid_mask]

    if len(y_prob_clean) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")

    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning

    df = pd.DataFrame({'prob': y_prob_clean, 'target': y_true_clean})

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df[['prob']], df['target'])
    bins = binner.transform(df[['prob']], metric='indices').values.flatten()

    stats_df = compute_bin_stats(bins, y_true_clean, round_digits=False)

    results = []
    for _, row in stats_df.iterrows():
        bin_idx = int(row['分箱'])
        mask = bins == bin_idx

        prob_min = y_prob_clean[mask].min() if mask.sum() > 0 else np.nan
        prob_max = y_prob_clean[mask].max() if mask.sum() > 0 else np.nan

        cum_total = row['累积坏样本数'] + row['累积好样本数']
        results.append({
            '桶编号': bin_idx,
            '最小概率': prob_min,
            '最大概率': prob_max,
            '样本数': int(row['样本总数']),
            '坏样本率': row['坏样本率'],
            '累积坏样本率': row['累积坏样本数'] / cum_total if cum_total > 0 else 0,
            '累积好样本率': row['累积好样本数'] / cum_total if cum_total > 0 else 0,
            'KS贡献': row['分档KS值'],
        })

    return pd.DataFrame(results)


def roc_curve(y_true: Union[np.ndarray, pd.Series],
              y_prob: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算ROC曲线数据.

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_prob)


def confusion_matrix(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series],
                     labels: Optional[list] = None) -> np.ndarray:
    """计算混淆矩阵.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param labels: 类别标签列表，默认为None
    :return: 混淆矩阵
    """
    return sk_confusion_matrix(y_true, y_pred, labels=labels)


def classification_report(y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series],
                          target_names: Optional[list] = None,
                          output_dict: bool = False) -> Union[str, dict]:
    """生成分类报告.

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param target_names: 目标类别名称
    :param output_dict: 如果为True，返回字典格式
    :return: 分类报告
    """
    return sk_classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict
    )
