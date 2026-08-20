"""模型解释的中文 Matplotlib 可视化。"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from hscredit.exceptions import ValidationError


def _finish(fig, show):
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _feature_index(result, feature):
    if isinstance(feature, int):
        if 0 <= feature < len(result.feature_names):
            return feature
    elif feature in result.feature_names:
        return result.feature_names.index(feature)
    raise ValidationError(f"解释特征不存在: {feature}")


def plot_decision(result, *, explainer=None, sample_id=None, position=0, max_display=15, figsize=(9, 6), show=True):
    """绘制单样本贡献决策路径。"""
    if sample_id is not None:
        position = result.position_for(sample_id)
    values = result.values[position]
    order = np.argsort(np.abs(values))[::-1][:max_display][::-1]
    fig, ax = plt.subplots(figsize=figsize)
    colors = np.where(values[order] >= 0, "#D95F59", "#4C78A8")
    ax.barh(np.asarray(result.feature_names)[order], values[order], color=colors)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_title(f"单样本SHAP决策贡献（样本 {result.sample_ids[position]}）")
    ax.set_xlabel(f"SHAP值（{result.model_output}尺度）")
    ax.set_ylabel("特征")
    return _finish(fig, show)


def plot_heatmap(result, *, explainer=None, max_display=20, figsize=(11, 6), show=True):
    """绘制多样本 SHAP 热力图。"""
    order = np.argsort(np.abs(result.values).mean(axis=0))[::-1][:max_display]
    fig, ax = plt.subplots(figsize=figsize)
    limit = np.max(np.abs(result.values[:, order])) or 1.0
    image = ax.imshow(result.values[:, order], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(order)), np.asarray(result.feature_names)[order], rotation=45, ha="right")
    ax.set_ylabel("样本")
    ax.set_title("多样本SHAP贡献热力图")
    fig.colorbar(image, ax=ax, label="SHAP值")
    return _finish(fig, show)


def plot_distribution(result, *, feature, explainer=None, figsize=(8, 5), show=True):
    """绘制特征值与 SHAP 贡献分布。"""
    index = _feature_index(result, feature)
    x = result.data.iloc[:, index]
    y = result.values[:, index]
    fig, ax = plt.subplots(figsize=figsize)
    numeric = pd.to_numeric(x, errors="coerce")
    if numeric.notna().all():
        ax.scatter(numeric, y, c=y, cmap="RdBu_r", alpha=0.75, edgecolor="none")
    else:
        categories = pd.Categorical(x.astype(str))
        ax.scatter(categories.codes, y, c=y, cmap="RdBu_r", alpha=0.75, edgecolor="none")
        ax.set_xticks(range(len(categories.categories)), categories.categories, rotation=45, ha="right")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title(f"{result.feature_names[index]} 的SHAP贡献分布")
    ax.set_xlabel("特征值")
    ax.set_ylabel("SHAP值")
    return _finish(fig, show)


def plot_correlation(result, *, explainer=None, figsize=(8, 7), show=True):
    """绘制 SHAP 贡献相关性热力图。"""
    corr = pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman").fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title("SHAP贡献相关性")
    fig.colorbar(image, ax=ax, label="Spearman相关系数")
    return _finish(fig, show)


def plot_feature_clustering(result, *, explainer=None, figsize=(9, 5), show=True):
    """绘制基于 SHAP 贡献相似性的层次聚类。"""
    fig, ax = plt.subplots(figsize=figsize)
    if len(result.feature_names) == 1:
        ax.text(0.5, 0.5, result.feature_names[0], ha="center", va="center")
        ax.set_axis_off()
    else:
        corr = pd.DataFrame(result.values).corr(method="spearman").fillna(0).to_numpy()
        np.fill_diagonal(corr, 1)
        tree = linkage(squareform(np.clip(1 - np.abs(corr), 0, 1), checks=False), method="average")
        dendrogram(tree, labels=result.feature_names, leaf_rotation=45, ax=ax)
    ax.set_title("SHAP特征层次聚类")
    ax.set_ylabel("贡献距离")
    return _finish(fig, show)


def _interaction_matrix(result, explainer):
    table = explainer.get_feature_interactions(result=result, top_n=max(1, len(result.feature_names) ** 2))
    matrix = pd.DataFrame(0.0, index=result.feature_names, columns=result.feature_names)
    for row in table.itertuples(index=False):
        matrix.loc[row.特征1, row.特征2] = row.交互强度
        matrix.loc[row.特征2, row.特征1] = row.交互强度
    return table, matrix


def plot_interaction_heatmap(result, *, explainer, figsize=(8, 7), show=True):
    table, matrix = _interaction_matrix(result, explainer)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_title("SHAP特征交互强度")
    fig.colorbar(image, ax=ax, label="平均绝对交互值")
    return _finish(fig, show)


def plot_interaction_bubble(result, *, explainer, top_n=20, figsize=(9, 6), show=True):
    table = explainer.get_feature_interactions(result=result, top_n=top_n)
    fig, ax = plt.subplots(figsize=figsize)
    if not table.empty:
        sizes = 100 + 900 * table["交互强度"] / max(table["交互强度"].max(), np.finfo(float).eps)
        ax.scatter(table["特征1"], table["特征2"], s=sizes, c=table["交互强度"], cmap="Blues", alpha=0.75)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("主要SHAP交互气泡图")
    ax.set_xlabel("特征1")
    ax.set_ylabel("特征2")
    return _finish(fig, show)


def plot_importance_overview(result, *, explainer=None, max_display=20, figsize=(12, 7), show=True):
    """组合展示贡献分布和全局重要性。"""
    order = np.argsort(np.abs(result.values).mean(axis=0))[::-1][:max_display]
    names = np.asarray(result.feature_names)[order]
    fig, (distribution_ax, importance_ax) = plt.subplots(1, 2, figsize=figsize)
    distribution_ax.violinplot([result.values[:, index] for index in order], vert=False, showmedians=True)
    distribution_ax.set_yticks(range(1, len(names) + 1), names)
    distribution_ax.axvline(0, color="#666666", linewidth=0.8)
    distribution_ax.set_title("SHAP贡献分布")
    distribution_ax.set_xlabel("SHAP值")
    importance = np.abs(result.values[:, order]).mean(axis=0)
    importance_ax.barh(names[::-1], importance[::-1], color="#4C78A8")
    importance_ax.set_title("SHAP特征重要性")
    importance_ax.set_xlabel("平均绝对SHAP值")
    return _finish(fig, show)


def plot_explanation_overview(result, *, explainer, max_display=10, figsize=(14, 10), show=True):
    """展示重要性、方向、相关性和代表样本的综合总览。"""
    report = explainer.get_global_report(result).head(max_display)
    representative = explainer.select_representative_samples(result)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].barh(report["特征"][::-1], report["平均绝对SHAP值"][::-1], color="#4C78A8")
    axes[0, 0].set_title("全局SHAP重要性")
    axes[0, 1].barh(report["特征"][::-1], report["正向影响占比"][::-1], color="#D95F59")
    axes[0, 1].set_title("提高模型输出的贡献占比")
    corr = pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman").fillna(0)
    axes[1, 0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(len(corr)), corr.index)
    axes[1, 0].set_title("SHAP贡献相关性")
    axes[1, 1].scatter(representative["风险排名"], representative["模型输出"], color="#F28E2B")
    for row in representative.itertuples(index=False):
        axes[1, 1].annotate(str(row.样本索引), (row.风险排名, row.模型输出), fontsize=8)
    axes[1, 1].set_title("代表样本")
    axes[1, 1].set_xlabel("风险排名")
    axes[1, 1].set_ylabel("模型输出")
    fig.suptitle("模型解释综合总览")
    return _finish(fig, show)
