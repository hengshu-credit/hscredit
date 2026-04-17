"""分类模型评估指标.

提供二分类模型评估的核心指标计算功能。

**参数**

所有函数支持numpy.ndarray和pandas.Series两种输入格式，自动进行类型转换。

**参考样例**

>>> from hscredit.core.metrics import ks, auc, gini, accuracy
>>> import numpy as np
>>> np.random.seed(42)
>>> y_true = np.random.randint(0, 2, 1000)  # 模拟真实标签（0=好，1=坏）
>>> y_prob = np.random.uniform(0, 1, 1000)  # 模拟模型预测概率
>>> print(f"KS={ks(y_true, y_prob):.4f}, AUC={auc(y_true, y_prob):.4f}, Gini={gini(y_true, y_prob):.4f}")
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve as sklearn_roc_curve,
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

    **参数**

    :param y_true: 真实标签 (0/1)，0为负样本，1为正样本
    :param y_prob: 预测为正样本的概率值
    :return: KS统计量，取值范围[0, 1]，越接近1区分能力越强
    :raises ValueError: 标签非二值或y_true/y_prob长度不一致时
    :raises ValueError: y_true中全为正样本或全为负样本时

    **参考样例**

    >>> from hscredit.core.metrics import ks
    >>> y_true = [0, 0, 1, 1, 1, 0, 1, 0]  # 真实标签序列
    >>> y_prob = [0.1, 0.3, 0.7, 0.6, 0.8, 0.2, 0.9, 0.4]  # 预测概率（高分对应坏样本）
    >>> ks(y_true, y_prob)
    0.75
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

    AUC值衡量模型在不同分类阈值下区分正负样本的综合能力，
    值在0.5-1.0之间，越接近1.0模型效果越好。

    **参数**

    :param y_true: 真实标签 (0/1)，0为负样本，1为正样本
    :param y_prob: 预测为正样本的概率值
    :return: AUC值，取值范围[0.5, 1.0]
    :raises ValueError: y_true/y_prob长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import auc
    >>> y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    >>> y_prob = [0.1, 0.3, 0.7, 0.6, 0.8, 0.2, 0.9, 0.4]
    >>> auc(y_true, y_prob)
    0.875
    """
    return roc_auc_score(y_true, y_prob)


def gini(y_true: Union[np.ndarray, pd.Series],
         y_prob: Union[np.ndarray, pd.Series]) -> float:
    """计算基尼系数 (Gini Coefficient).

    基尼系数是AUC的线性变换：基尼系数 = 2 * AUC - 1。
    范围从-1到1，越接近1表示模型区分能力越强。

    **参数**

    :param y_true: 真实标签 (0/1)，0为负样本，1为正样本
    :param y_prob: 预测为正样本的概率值
    :return: 基尼系数，取值范围[-1, 1]
    :raises ValueError: y_true/y_prob长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import gini
    >>> y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    >>> y_prob = [0.1, 0.3, 0.7, 0.6, 0.8, 0.2, 0.9, 0.4]
    >>> gini(y_true, y_prob)
    0.75
    """
    return 2 * auc(y_true, y_prob) - 1


def accuracy(y_true: Union[np.ndarray, pd.Series],
             y_pred: Union[np.ndarray, pd.Series]) -> float:
    """计算准确率 (Accuracy).

    准确率 = 预测正确的样本数 / 总样本数。

    **参数**

    :param y_true: 真实标签（任意类型）
    :param y_pred: 预测标签（与y_true类型一致）
    :return: 准确率，取值范围[0, 1]
    :raises ValueError: y_true/y_pred长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import accuracy
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> accuracy(y_true, y_pred)
    0.8
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true: Union[np.ndarray, pd.Series],
              y_pred: Union[np.ndarray, pd.Series],
              average: str = 'binary') -> float:
    """计算精确率 (Precision).

    精确率 = TP / (TP + FP)，即预测为正的样本中真正为正的比例。

    **参数**

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，'binary'（二分类默认只看正类），'micro'（全局TP/FP/FN），
        'macro'（各类分别计算后取平均），'weighted'（加权平均），默认为'binary'
    :return: 精确率，取值范围[0, 1]

    **参考样例**

    >>> from hscredit.core.metrics import precision
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> precision(y_true, y_pred)       # 二分类默认
    1.0
    >>> precision(y_true, y_pred, average='macro')
    0.9
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true: Union[np.ndarray, pd.Series],
           y_pred: Union[np.ndarray, pd.Series],
           average: str = 'binary') -> float:
    """计算召回率 (Recall / Sensitivity).

    召回率 = TP / (TP + FN)，即所有正样本中被正确预测的比例。

    **参数**

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，'binary'（二分类默认只看正类），'micro'，'macro'，'weighted'，
        默认为'binary'
    :return: 召回率，取值范围[0, 1]

    **参考样例**

    >>> from hscredit.core.metrics import recall
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> recall(y_true, y_pred)
    0.666...
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_true: Union[np.ndarray, pd.Series],
       y_pred: Union[np.ndarray, pd.Series],
       average: str = 'binary') -> float:
    """计算F1分数 (F1 Score).

    F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)，是精确率和召回率的调和平均。

    **参数**

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 平均方式，'binary'（二分类默认只看正类），'micro'，'macro'，'weighted'，
        默认为'binary'
    :return: F1分数，取值范围[0, 1]

    **参考样例**

    >>> from hscredit.core.metrics import f1
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> f1(y_true, y_pred)
    0.8
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def ks_bucket(y_true: Union[np.ndarray, pd.Series],
              y_prob: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """计算分桶KS统计表.

    将样本按预测概率分为多个桶，计算每个桶的KS统计信息，
    包括各桶的样本数、坏账率、累积好坏样本率及KS贡献。

    **参数**

    :param y_true: 真实标签 (0/1)，0为负样本（好样本），1为正样本（坏样本）
    :param y_prob: 预测为正样本的概率值
    :param method: 分箱方法，支持的分箱方法同OptimalBinning，默认为'quantile'
        - quantile: 等频分箱
        - tree: 决策树最优分箱
        - chi: 卡方分箱
        - 等其他分箱方法
    :param max_n_bins: 最大分桶数量，默认为10
    :param min_bin_size: 每桶最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 包含桶统计信息的DataFrame，列包括：
        - 桶编号: 分箱序号
        - 最小概率: 该桶内概率最小值
        - 最大概率: 该桶内概率最大值
        - 样本数: 该桶内样本数量
        - 坏样本率: 该桶内坏样本占比
        - 累积坏样本率: 累积坏样本占总坏样本比例
        - 累积好样本率: 累积好样本占总好样本比例
        - KS贡献: 该桶对KS值的贡献
    :raises ValueError: 数据全部为缺失值时

    **参考样例**

    >>> from hscredit.core.metrics import ks_bucket
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y_true = np.random.randint(0, 2, 1000)
    >>> y_prob = np.random.uniform(0, 1, 1000)
    >>> result = ks_bucket(y_true, y_prob, max_n_bins=5)
    >>> print(result[['桶编号', '样本数', '坏样本率']])
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

    返回ROC曲线绘制所需的FPR、TPR和阈值数据。

    **参数**

    :param y_true: 真实标签 (0/1)
    :param y_prob: 预测为正样本的概率值
    :return: 三元组 (fpr, tpr, thresholds)
        - fpr: 假阳性率（False Positive Rate）数组
        - tpr: 真阳性率（True Positive Rate）数组
        - thresholds: 对应的概率阈值数组

    **参考样例**

    >>> from hscredit.core.metrics import roc_curve
    >>> y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    >>> y_prob = [0.1, 0.3, 0.7, 0.6, 0.8, 0.2, 0.9, 0.4]
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    """
    return sklearn_roc_curve(y_true, y_prob)


def confusion_matrix(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series],
                     labels: Optional[list] = None) -> np.ndarray:
    """计算混淆矩阵.

    混淆矩阵展示分类器的预测结果与真实标签的对应关系。

    **参数**

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param labels: 类别标签列表，指定矩阵的行序和列序，默认为None（自动推断）
    :return: 混淆矩阵（2D ndarray）
        - 行表示真实类别，列表示预测类别
        - 二分类时：[[TN, FP], [FN, TP]]

    **参考样例**

    >>> from hscredit.core.metrics import confusion_matrix
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0],
           [1, 2]])
    """
    return sk_confusion_matrix(y_true, y_pred, labels=labels)


def classification_report(y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series],
                          target_names: Optional[list] = None,
                          output_dict: bool = False) -> Union[str, dict]:
    """生成分类报告.

    输出各类的精确率、召回率、F1分数和支持样本数。

    **参数**

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param target_names: 目标类别名称列表，用于报告中替代类别标签显示
    :param output_dict: 如果为True，返回字典格式；否则返回格式化字符串，默认为False
    :return: 当output_dict=False时返回格式化字符串，当output_dict=True时返回字典

    **参考样例**

    >>> from hscredit.core.metrics import classification_report
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> print(classification_report(y_true, y_pred))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       0.67      1.00      0.80         2
               1       1.00      0.67      0.80         3
    <BLANKLINE>
        accuracy                           0.80         5
       macro avg       0.83      0.83      0.80         5
    weighted avg       0.87      0.80      0.80         5
    >>> classification_report(y_true, y_pred, output_dict=True)
    {'0': {'precision': 0.67, 'recall': 1.0, 'f1-score': 0.8, 'support': 2}, ...}
    """
    return sk_classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict
    )
