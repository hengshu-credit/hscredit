"""特征重要性指标计算.

提供评估特征预测能力的指标。

主要指标:
- IV (Information Value): 信息价值，衡量特征的预测能力
- IV_table: IV详细统计表
- gini_importance: 基尼重要性
- entropy_importance: 熵重要性
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def IV(y_true: Union[np.ndarray, pd.Series],
       feature: Union[np.ndarray, pd.Series],
       bins: Optional[Union[int, list]] = None) -> float:
    """计算Information Value (信息价值).

    IV用于衡量特征的预测能力，值越大表示特征的区分能力越强。

    IV计算公式:
    IV = Σ [(好样本占比 - 坏样本占比) * ln(好样本占比 / 坏样本占比)]

    IV分级标准:
    - IV < 0.02: 无预测能力
    - 0.02 ≤ IV < 0.1: 弱预测能力
    - 0.1 ≤ IV < 0.3: 中等预测能力
    - 0.3 ≤ IV < 0.5: 强预测能力
    - IV ≥ 0.5: 极强预测能力

    :param y_true: 目标变量 (0/1)
    :param feature: 特征变量
    :param bins: 分箱数量或自定义分箱边界，如果为None则使用等频分箱
    :return: IV值
    """
    y_true = np.asarray(y_true)
    feature = np.asarray(feature)

    if len(y_true) != len(feature):
        raise ValueError("y_true and feature must have same length")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    # 处理分箱
    if bins is None:
        # 默认使用10箱等频分箱
        bins = 10

    if isinstance(bins, int):
        # 等频分箱
        if len(np.unique(feature)) <= bins:
            # 如果唯一值少于bins，直接使用唯一值
            bin_edges = np.sort(np.unique(feature))
        else:
            # 使用分位数分箱
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(feature, quantiles)
            bin_edges = np.unique(bin_edges)  # 去重
    else:
        bin_edges = np.array(bins)

    # 计算每个箱的统计
    total_good = np.sum(y_true == 0)
    total_bad = np.sum(y_true == 1)

    if total_good == 0 or total_bad == 0:
        return 0.0

    iv = 0.0

    for i in range(len(bin_edges) - 1):
        mask = (feature >= bin_edges[i]) & (feature < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # 最后一个箱包含右边界
            mask = (feature >= bin_edges[i]) & (feature <= bin_edges[i + 1])

        bin_good = np.sum(y_true[mask] == 0)
        bin_bad = np.sum(y_true[mask] == 1)

        if bin_good == 0 or bin_bad == 0:
            continue  # 跳过空箱

        good_pct = bin_good / total_good
        bad_pct = bin_bad / total_bad

        # 平滑处理：将0替换为epsilon，避免log(0)和除零错误
        epsilon = 1e-10
        if good_pct == 0:
            good_pct = epsilon
        if bad_pct == 0:
            bad_pct = epsilon

        # IV公式：(bad_pct - good_pct) * log(bad_pct / good_pct)
        # 该公式理论上总是非负的
        woe = np.log(bad_pct / good_pct)
        iv += (bad_pct - good_pct) * woe

    return iv


def IV_table(y_true: Union[np.ndarray, pd.Series],
             feature: Union[np.ndarray, pd.Series],
             bins: Optional[Union[int, list]] = None,
             feature_name: str = "feature") -> pd.DataFrame:
    """计算IV详细统计表.

    提供每个分箱的WOE和IV贡献信息。

    :param y_true: 目标变量 (0/1)
    :param feature: 特征变量
    :param bins: 分箱数量或自定义分箱边界
    :param feature_name: 特征名称，默认为"feature"
    :return: IV统计表，包含以下列:
        - 分箱区间: 分箱区间
        - 样本数: 样本数
        - 样本占比: 样本占比
        - 好样本数: 好样本数
        - 坏样本数: 坏样本数
        - 坏样本率: 坏样本率
        - WOE: Weight of Evidence
        - IV贡献: 该箱的IV贡献
    """
    y_true = np.asarray(y_true)
    feature = np.asarray(feature)

    if len(y_true) != len(feature):
        raise ValueError("y_true and feature must have same length")

    # 处理分箱
    if bins is None:
        bins = 10

    if isinstance(bins, int):
        if len(np.unique(feature)) <= bins:
            bin_edges = np.sort(np.unique(feature))
        else:
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(feature, quantiles)
            bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.array(bins)

    total_samples = len(y_true)
    total_good = np.sum(y_true == 0)
    total_bad = np.sum(y_true == 1)

    results = []

    for i in range(len(bin_edges) - 1):
        if i == 0:
            mask = feature <= bin_edges[i + 1]
            bin_label = f"(-inf, {bin_edges[i + 1]:.3f}]"
        elif i == len(bin_edges) - 2:
            mask = feature > bin_edges[i]
            bin_label = f"({bin_edges[i]:.3f}, +inf)"
        else:
            mask = (feature > bin_edges[i]) & (feature <= bin_edges[i + 1])
            bin_label = f"({bin_edges[i]:.3f}, {bin_edges[i + 1]:.3f}]"

        bin_count = np.sum(mask)
        bin_good = np.sum(y_true[mask] == 0)
        bin_bad = np.sum(y_true[mask] == 1)

        if bin_count == 0:
            continue

        count_distr = bin_count / total_samples
        bad_rate = bin_bad / bin_count if bin_count > 0 else 0

        # 计算WOE
        good_pct = bin_good / total_good if total_good > 0 else 0
        bad_pct = bin_bad / total_bad if total_bad > 0 else 0

        if good_pct == 0 or bad_pct == 0:
            woe = 0.0
            iv_contrib = 0.0
        else:
            woe = np.log(good_pct / bad_pct)
            iv_contrib = (good_pct - bad_pct) * woe

        results.append({
            '分箱区间': bin_label,
            '样本数': bin_count,
            '样本占比': count_distr,
            '好样本数': bin_good,
            '坏样本数': bin_bad,
            '坏样本率': bad_rate,
            'WOE': woe,
            'IV贡献': iv_contrib
        })

    return pd.DataFrame(results)


def gini_importance(X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    max_depth: int = 3) -> pd.Series:
    """计算基尼重要性.

    使用决策树计算特征的基尼重要性。

    :param X: 特征矩阵
    :param y: 目标变量
    :param max_depth: 决策树最大深度，默认为3
    :return: 特征重要性得分
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    # 训练决策树
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, y)

    # 获取特征重要性
    importance_scores = tree.feature_importances_

    return pd.Series(importance_scores, index=feature_names)


def entropy_importance(X: Union[np.ndarray, pd.DataFrame],
                       y: Union[np.ndarray, pd.Series],
                       n_estimators: int = 100,
                       max_depth: int = 3) -> pd.Series:
    """计算熵重要性.

    使用随机森林计算特征的熵重要性。

    :param X: 特征矩阵
    :param y: 目标变量
    :param n_estimators: 树的数量，默认为100
    :param max_depth: 每棵树的最大深度，默认为3
    :return: 特征重要性得分
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=42)
    rf.fit(X, y)

    # 获取特征重要性
    importance_scores = rf.feature_importances_

    return pd.Series(importance_scores, index=feature_names)
