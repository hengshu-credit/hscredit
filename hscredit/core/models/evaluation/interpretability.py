"""模型可解释性模块.

提供SHAP分析和特征重要性可视化功能，支持:
- SHAP值计算
- SHAP摘要图
- SHAP依赖图
- SHAP力图
- 传统特征重要性 + SHAP值组合图

**依赖**
pip install shap

**示例**
>>> from hscredit.core.models import XGBoostRiskModel
>>> from hscredit.core.models.interpretability import ModelExplainer
>>>
>>> model = XGBoostRiskModel()
>>> model.fit(X_train, y_train)
>>>
>>> # 创建解释器
>>> explainer = ModelExplainer(model)
>>>
>>> # 计算SHAP值
>>> shap_values = explainer.compute_shap_values(X_test)
>>>
>>> # 绘制SHAP摘要图
>>> explainer.plot_shap_summary(X_test)
>>>
>>> # 绘制组合特征重要性图
>>> explainer.plot_combined_importance(X_test, top_n=15)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 绘图
import matplotlib.pyplot as plt

# 检查shap是否可用
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from ..base import BaseRiskModel


class ModelExplainer:
    """模型解释器.

    基于SHAP的模型可解释性分析工具。

    **参数**

    :param model: 训练好的风控模型
    :param feature_names: 特征名称列表，可选
    :param background_data: 背景数据用于SHAP计算，可选
    :param explainer_type: SHAP解释器类型，默认'auto'
        - 'auto': 自动选择
        - 'tree': TreeSHAP (树模型)
        - 'kernel': KernelSHAP (通用)
        - 'linear': LinearSHAP (线性模型)

    **示例**

    >>> explainer = ModelExplainer(model)
    >>> shap_values = explainer.compute_shap_values(X_test)
    >>> explainer.plot_shap_summary(X_test)
    """

    def __init__(
        self,
        model: BaseRiskModel,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        explainer_type: str = 'auto'
    ):
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP未安装，请使用 pip install shap 安装"
            )

        self.model = model
        self.explainer_type = explainer_type
        self._explainer = None
        self._shap_values = None
        self._expected_value = None

        # 获取特征名称
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(model, 'feature_names_in_'):
            self.feature_names = model.feature_names_in_
        else:
            self.feature_names = None

        # 背景数据
        self.background_data = background_data

        # 创建解释器
        self._create_explainer()

    def _create_explainer(self):
        """创建SHAP解释器."""
        model = self.model
        model_type = model.__class__.__name__.lower()

        # 自动选择解释器类型
        if self.explainer_type == 'auto':
            if any(x in model_type for x in ['xgboost', 'lightgbm', 'catboost', 'tree', 'forest']):
                self.explainer_type = 'tree'
            elif 'logistic' in model_type or 'linear' in model_type:
                self.explainer_type = 'linear'
            else:
                self.explainer_type = 'kernel'

        # 准备背景数据
        background = self.background_data
        if background is not None:
            if isinstance(background, pd.DataFrame):
                background = background.values

        # 创建解释器
        if self.explainer_type == 'tree':
            self._explainer = shap.TreeExplainer(model._model)
        elif self.explainer_type == 'linear':
            if background is None:
                raise ValueError("线性模型解释器需要提供background_data")
            self._explainer = shap.LinearExplainer(model._model, background)
        elif self.explainer_type == 'kernel':
            if background is None:
                raise ValueError("Kernel解释器需要提供background_data")
            self._explainer = shap.KernelExplainer(model._model.predict_proba, background)
        else:
            raise ValueError(f"不支持的解释器类型: {self.explainer_type}")

    def compute_shap_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        check_additivity: bool = True
    ) -> np.ndarray:
        """计算SHAP值.

        :param X: 特征矩阵
        :param check_additivity: 是否检查可加性，默认True
        :return: SHAP值数组，形状为 (n_samples, n_features) 或 (n_samples, n_features, n_classes)
        """
        # 转换数据
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        # 计算SHAP值
        shap_values = self._explainer.shap_values(
            X_array,
            check_additivity=check_additivity
        )

        # 处理二分类情况
        if isinstance(shap_values, list):
            # 二分类返回两个数组，取正类的SHAP值
            shap_values = shap_values[1]

        self._shap_values = shap_values
        self._expected_value = getattr(self._explainer, 'expected_value', None)

        # 如果是二分类，expected_value也可能是数组
        if isinstance(self._expected_value, (list, np.ndarray)):
            self._expected_value = self._expected_value[1] if len(self._expected_value) > 1 else self._expected_value[0]

        return shap_values

    def get_shap_importance(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> pd.Series:
        """获取基于SHAP值的特征重要性.

        使用SHAP值的平均绝对值作为特征重要性。

        :param X: 特征矩阵，如果之前已计算SHAP值则可选
        :return: 特征重要性Series
        """
        if self._shap_values is None and X is not None:
            self.compute_shap_values(X)
        elif self._shap_values is None:
            raise ValueError("需要提供X或先调用compute_shap_values")

        # 计算平均绝对SHAP值
        mean_shap = np.abs(self._shap_values).mean(axis=0)

        # 获取特征名称
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(mean_shap))]

        importance_series = pd.Series(
            mean_shap,
            index=self.feature_names,
            name='shap_importance'
        ).sort_values(ascending=False)

        return importance_series

    def plot_shap_summary(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_display: int = 20,
        plot_type: str = 'dot',
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """绘制SHAP摘要图.

        显示每个特征对模型输出的影响。

        :param X: 特征矩阵
        :param max_display: 显示的最大特征数，默认20
        :param plot_type: 图表类型，默认'dot'
            - 'dot': 点图 (默认)
            - 'bar': 条形图
            - 'violin': 小提琴图
        :param figsize: 图表大小，默认(10, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :param kwargs: 其他shap.summary_plot参数
        :return: matplotlib Figure对象
        """
        if self._shap_values is None or not np.array_equal(X, getattr(self, '_last_X', None)):
            self.compute_shap_values(X)
            self._last_X = X.copy() if isinstance(X, pd.DataFrame) else X

        # 转换数据
        X_display = X.values if isinstance(X, pd.DataFrame) else X

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制SHAP摘要图
        shap.summary_plot(
            self._shap_values,
            X_display,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
            **kwargs
        )

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('SHAP Summary Plot', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_shap_bar(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        max_display: int = 15,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """绘制SHAP条形图.

        显示特征的平均绝对SHAP值。

        :param X: 特征矩阵，可选（如果已计算SHAP值）
        :param max_display: 显示的最大特征数，默认15
        :param figsize: 图表大小，默认(10, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :return: matplotlib Figure对象
        """
        if self._shap_values is None and X is not None:
            self.compute_shap_values(X)
        elif self._shap_values is None:
            raise ValueError("需要提供X或先调用compute_shap_values")

        # 获取平均绝对SHAP值
        mean_shap = np.abs(self._shap_values).mean(axis=0)

        # 排序并选择top特征
        feature_order = np.argsort(mean_shap)[::-1][:max_display]
        mean_shap_sorted = mean_shap[feature_order]

        # 获取特征名称
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(mean_shap))]
        feature_names_sorted = [self.feature_names[i] for i in feature_order]

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制水平条形图
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(mean_shap_sorted)))
        bars = ax.barh(range(len(mean_shap_sorted)), mean_shap_sorted, color=colors)

        # 设置标签
        ax.set_yticks(range(len(feature_names_sorted)))
        ax.set_yticklabels(feature_names_sorted)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Feature Importance (Mean |SHAP Value|)', fontsize=14, fontweight='bold')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, mean_shap_sorted)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_shap_dependence(
        self,
        feature: Union[str, int],
        X: Union[np.ndarray, pd.DataFrame],
        interaction_feature: Optional[Union[str, int]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """绘制SHAP依赖图.

        显示单个特征值与SHAP值的关系。

        :param feature: 特征名称或索引
        :param X: 特征矩阵
        :param interaction_feature: 交互特征名称或索引，可选
        :param figsize: 图表大小，默认(10, 6)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :param kwargs: 其他shap.dependence_plot参数
        :return: matplotlib Figure对象
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        # 处理特征索引
        if isinstance(feature, str):
            if self.feature_names is None:
                raise ValueError("需要提供feature_names才能使用特征名称")
            feature_idx = self.feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx] if self.feature_names else f'feature_{feature}'

        # 处理交互特征
        interaction_index = interaction_feature
        if isinstance(interaction_feature, str):
            if self.feature_names is None:
                raise ValueError("需要提供feature_names才能使用特征名称")
            interaction_index = self.feature_names.index(interaction_feature)

        # 转换数据
        X_display = X.values if isinstance(X, pd.DataFrame) else X

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制依赖图
        shap.dependence_plot(
            feature_idx,
            self._shap_values,
            X_display,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False,
            ax=ax,
            **kwargs
        )

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_shap_force(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        index: int = 0,
        figsize: Tuple[int, int] = (20, 4),
        show: bool = True,
        **kwargs
    ):
        """绘制SHAP力图.

        显示单个预测的SHAP值分解。

        :param X: 特征矩阵
        :param index: 样本索引，默认0
        :param figsize: 图表大小，默认(20, 4)
        :param show: 是否显示图表，默认True
        :param kwargs: 其他shap.force_plot参数
        :return: matplotlib Figure或JS可视化对象
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        # 转换数据
        if isinstance(X, pd.DataFrame):
            X_display = X.values
        else:
            X_display = X

        # 获取单个样本
        instance = X_display[index]
        shap_values_instance = self._shap_values[index]

        # 创建力图
        fig = plt.figure(figsize=figsize)

        shap.force_plot(
            self._expected_value,
            shap_values_instance,
            instance,
            feature_names=self.feature_names,
            show=show,
            matplotlib=True,
            **kwargs
        )

        return fig

    def plot_combined_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_n: int = 15,
        figsize: Tuple[int, int] = (16, 10),
        importance_type: str = 'gain',
        title: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """绘制组合特征重要性图.

        同时显示传统特征重要性和SHAP值重要性。

        :param X: 特征矩阵
        :param top_n: 显示前N个特征，默认15
        :param figsize: 图表大小，默认(16, 10)
        :param importance_type: 传统重要性类型，默认'gain'
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :return: matplotlib Figure对象
        """
        # 获取传统特征重要性
        traditional_importance = self.model.get_feature_importances(importance_type)

        # 获取SHAP重要性
        shap_importance = self.get_shap_importance(X)

        # 获取共同的top特征
        all_features = set(traditional_importance.index[:top_n]) | set(shap_importance.index[:top_n])
        all_features = list(all_features)

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Traditional': traditional_importance.reindex(all_features),
            'SHAP': shap_importance.reindex(all_features)
        }).fillna(0)

        # 归一化
        importance_df['Traditional_norm'] = importance_df['Traditional'] / importance_df['Traditional'].max()
        importance_df['SHAP_norm'] = importance_df['SHAP'] / importance_df['SHAP'].max()

        # 计算综合排名
        importance_df['Combined'] = (importance_df['Traditional_norm'] + importance_df['SHAP_norm']) / 2
        importance_df = importance_df.sort_values('Combined', ascending=True)

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 左图：传统特征重要性
        ax1 = axes[0]
        colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
        bars1 = ax1.barh(range(len(importance_df)), importance_df['Traditional'], color=colors1)
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df.index)
        ax1.set_xlabel('Traditional Importance', fontsize=12)
        ax1.set_title('Traditional Feature Importance', fontsize=13, fontweight='bold')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, importance_df['Traditional'])):
            if val > 0:
                ax1.text(val, i, f' {val:.2f}', va='center', fontsize=9)

        # 右图：SHAP重要性
        ax2 = axes[1]
        colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(importance_df)))
        bars2 = ax2.barh(range(len(importance_df)), importance_df['SHAP'], color=colors2)
        ax2.set_yticks(range(len(importance_df)))
        ax2.set_yticklabels(importance_df.index)
        ax2.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax2.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars2, importance_df['SHAP'])):
            if val > 0:
                ax2.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        if title:
            fig.suptitle(title, fontsize=15, fontweight='bold')
        else:
            fig.suptitle('Feature Importance Comparison', fontsize=15, fontweight='bold')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_shap_waterfall(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        index: int = 0,
        max_display: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """绘制SHAP瀑布图.

        显示单个样本的SHAP值贡献分解。

        :param X: 特征矩阵
        :param index: 样本索引，默认0
        :param max_display: 显示的最大特征数，默认10
        :param figsize: 图表大小，默认(12, 8)
        :param title: 图表标题，可选
        :param show: 是否显示图表，默认True
        :return: matplotlib Figure对象
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        # 获取单个样本的SHAP值
        shap_values_instance = self._shap_values[index]

        # 转换数据
        if isinstance(X, pd.DataFrame):
            X_display = X.values
        else:
            X_display = X

        # 创建Explanation对象
        explanation = shap.Explanation(
            values=shap_values_instance,
            base_values=self._expected_value,
            data=X_display[index],
            feature_names=self.feature_names
        )

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制瀑布图
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_feature_interactions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_n: int = 10
    ) -> pd.DataFrame:
        """获取特征交互重要性.

        :param X: 特征矩阵
        :param top_n: 返回前N个交互，默认10
        :return: 特征交互DataFrame
        """
        if not isinstance(self._explainer, shap.TreeExplainer):
            warnings.warn("特征交互分析仅支持TreeSHAP")
            return pd.DataFrame()

        # 计算SHAP交互值
        shap_interaction = self._explainer.shap_interaction_values(X)

        if isinstance(shap_interaction, list):
            shap_interaction = shap_interaction[1]

        # 计算交互重要性
        interaction_importance = np.abs(shap_interaction).sum(axis=0)

        # 获取特征名称
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(interaction_importance.shape[0])]

        # 构建交互DataFrame
        interactions = []
        n_features = len(self.feature_names)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append({
                    'Feature_1': self.feature_names[i],
                    'Feature_2': self.feature_names[j],
                    'Interaction_Strength': interaction_importance[i, j]
                })

        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('Interaction_Strength', ascending=False)

        return interactions_df.head(top_n)


def plot_feature_importance(
    model: BaseRiskModel,
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    top_n: int = 20,
    importance_type: str = 'gain',
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    color: str = '#2E86AB',
    show_values: bool = True,
    show: bool = True
) -> plt.Figure:
    """绘制传统特征重要性图.

    统一的特征重要性可视化函数。

    :param model: 训练好的模型
    :param X: 特征矩阵（用于SHAP，可选）
    :param top_n: 显示前N个特征，默认20
    :param importance_type: 重要性类型，默认'gain'
    :param figsize: 图表大小，默认(10, 8)
    :param title: 图表标题，可选
    :param color: 条形颜色，默认'#2E86AB'
    :param show_values: 是否显示数值，默认True
    :param show: 是否显示图表，默认True
    :return: matplotlib Figure对象

    **示例**

    >>> from hscredit.core.models.interpretability import plot_feature_importance
    >>> fig = plot_feature_importance(model, top_n=15)
    >>> fig.savefig('feature_importance.png')
    """
    # 获取特征重要性
    importances = model.get_feature_importances(importance_type)

    # 选择top_n
    importances = importances.head(top_n)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制水平条形图
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))[::-1]
    bars = ax.barh(range(len(importances)), importances.values, color=colors)

    # 设置标签
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Feature Importance ({importance_type})', fontsize=14, fontweight='bold')

    # 添加数值标签
    if show_values:
        for i, (bar, val) in enumerate(zip(bars, importances.values)):
            ax.text(val, i, f' {val:.2f}', va='center', fontsize=9)

    # 添加网格线
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_shap_importance(
    model: BaseRiskModel,
    X: Union[np.ndarray, pd.DataFrame],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """绘制SHAP特征重要性图.

    :param model: 训练好的模型
    :param X: 特征矩阵
    :param top_n: 显示前N个特征，默认20
    :param figsize: 图表大小，默认(10, 8)
    :param title: 图表标题，可选
    :param show: 是否显示图表，默认True
    :return: matplotlib Figure对象

    **示例**

    >>> from hscredit.core.models.interpretability import plot_shap_importance
    >>> fig = plot_shap_importance(model, X_test, top_n=15)
    """
    explainer = ModelExplainer(model)
    return explainer.plot_shap_bar(X, max_display=top_n, figsize=figsize, title=title, show=show)


def plot_importance_comparison(
    model: BaseRiskModel,
    X: Union[np.ndarray, pd.DataFrame],
    top_n: int = 15,
    figsize: Tuple[int, int] = (16, 10),
    importance_type: str = 'gain',
    title: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """绘制特征重要性对比图（传统 vs SHAP）.

    :param model: 训练好的模型
    :param X: 特征矩阵
    :param top_n: 显示前N个特征，默认15
    :param figsize: 图表大小，默认(16, 10)
    :param importance_type: 传统重要性类型，默认'gain'
    :param title: 图表标题，可选
    :param show: 是否显示图表，默认True
    :return: matplotlib Figure对象

    **示例**

    >>> from hscredit.core.models.interpretability import plot_importance_comparison
    >>> fig = plot_importance_comparison(model, X_test, top_n=10)
    """
    explainer = ModelExplainer(model)
    return explainer.plot_combined_importance(X, top_n, figsize, importance_type, title, show)
