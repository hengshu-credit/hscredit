"""
分类指标计算

提供二分类模型评估的核心指标计算功能。

主要指标:
- KS (Kolmogorov-Smirnov): 衡量模型区分好坏样本的能力
- AUC (Area Under Curve): ROC曲线下的面积
- Gini: 基于AUC的基尼系数
- KS_bucket: 分桶KS统计
- ROC_curve: ROC曲线数据
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report


def KS(y_true: Union[np.ndarray, pd.Series],
       y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算Kolmogorov-Smirnov统计量。

    KS值衡量模型区分正负样本的能力，值越大区分效果越好。
    KS = max(TPR - FPR)，其中TPR为正样本累积率，FPR为负样本累积率。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: KS统计量，取值范围[0, 1]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    # 按预测概率降序排序
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # 计算累积分布
    n_total = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n_total - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # 计算累积正样本率和累积负样本率
    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg

    # KS值为最大差值
    ks = np.max(np.abs(cum_pos - cum_neg))

    return ks


def AUC(y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算ROC曲线下的面积(AUC)。

    AUC值在0.5-1.0之间，越接近1.0模型效果越好。
    AUC=0.5表示随机猜测，AUC<0.5表示模型效果差于随机。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: AUC值，取值范围[0.5, 1.0]
    """
    return roc_auc_score(y_true, y_prob)


def Gini(y_true: Union[np.ndarray, pd.Series],
         y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算基尼系数。

    基尼系数 = 2 * AUC - 1
    范围从-1到1，越接近1表示模型区分能力越强。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: 基尼系数，取值范围[-1, 1]
    """
    auc = AUC(y_true, y_prob)
    return 2 * auc - 1


def KS_bucket(y_true: Union[np.ndarray, pd.Series],
              y_prob: Union[np.ndarray, pd.Series],
              n_bins: int = 10) -> pd.DataFrame:
    """计算分桶KS统计表。

    将样本按预测概率分为n_bins个桶，计算每个桶的KS统计信息。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :param n_bins: 分桶数量，默认为10
    :return: 包含桶统计信息的DataFrame
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    # 按预测概率降序排序
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    y_prob_sorted = y_prob[desc_indices]

    # 计算每个桶的样本数
    n_total = len(y_true)
    bucket_size = n_total // n_bins
    remainder = n_total % n_bins

    bucket_sizes = [bucket_size + (1 if i < remainder else 0) for i in range(n_bins)]

    results = []
    start_idx = 0

    for bucket in range(n_bins):
        end_idx = start_idx + bucket_sizes[bucket]

        bucket_y_true = y_true_sorted[start_idx:end_idx]
        bucket_y_prob = y_prob_sorted[start_idx:end_idx]

        bucket_count = len(bucket_y_true)
        bucket_bad_count = np.sum(bucket_y_true)
        bucket_bad_rate = bucket_bad_count / bucket_count if bucket_count > 0 else 0

        # 累积统计
        cum_bad_count = np.sum(y_true_sorted[:end_idx])
        cum_good_count = end_idx - cum_bad_count

        total_bad = np.sum(y_true)
        total_good = n_total - total_bad

        cum_bad_rate = cum_bad_count / total_bad if total_bad > 0 else 0
        cum_good_rate = cum_good_count / total_good if total_good > 0 else 0

        ks_contrib = abs(cum_bad_rate - cum_good_rate)

        results.append({
            '桶编号': bucket,
            '最小概率': bucket_y_prob.min() if bucket_count > 0 else np.nan,
            '最大概率': bucket_y_prob.max() if bucket_count > 0 else np.nan,
            '样本数': bucket_count,
            '坏样本率': bucket_bad_rate,
            '累积坏样本率': cum_bad_rate,
            '累积好样本率': cum_good_rate,
            'KS贡献': ks_contrib
        })

        start_idx = end_idx

    return pd.DataFrame(results)


def ROC_curve(y_true: Union[np.ndarray, pd.Series],
              y_prob: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算ROC曲线数据。

    返回FPR, TPR和阈值数组，用于绘制ROC曲线。

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测概率 (正样本概率)
    :return: (fpr, tpr, thresholds)
        - fpr: 假正率数组
        - tpr: 真正率数组
        - thresholds: 阈值数组
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def confusion_matrix(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series],
                     labels: Optional[list] = None) -> np.ndarray:
    """计算混淆矩阵。

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
    """生成分类报告。

    包含精确率、召回率、F1分数等指标。

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param target_names: 目标类别名称，默认为None
    :param output_dict: 如果为True，返回字典格式，否则返回字符串，默认为False
    :return: 分类报告
    """
    return sk_classification_report(y_true, y_pred,
                                   target_names=target_names,
                                   output_dict=output_dict)
