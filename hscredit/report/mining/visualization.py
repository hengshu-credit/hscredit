"""决策树可视化模块.

支持使用dtreeviz和graphviz两种方式进行决策树可视化。
使用hscredit主题色和统一样式。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import warnings

# 导入hscredit可视化工具
from ...core.viz.utils import DEFAULT_COLORS, setup_axis_style, save_figure, get_or_create_ax


# hscredit主题色
THEME_COLOR = "#2639E9"  # 主色
SECONDARY_COLORS = ["#F76E6C", "#FE7715", "#5AD8A6", "#F6BD16", "#5B8FF9"]


class TreeVisualizer:
    """决策树可视化器.
    
    支持多种可视化方式：
    - dtreeviz：美观的交互式可视化
    - graphviz：传统的DOT格式可视化
    - matplotlib：使用sklearn内置plot_tree
    
    使用hscredit统一主题色，无需手动设置字体。
    
    **参考样例**

    >>> from hscredit.core.rules.mining import TreeVisualizer
    >>> visualizer = TreeVisualizer()
    >>> viz = visualizer.plot_dtreeviz(tree_model, X, y)
    >>> dot = visualizer.plot_graphviz(tree_model, X.columns)
    >>> dot.render('tree')
    >>> visualizer.plot_matplotlib(tree_model, X.columns, save_path='tree.png')
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        :param feature_names: 特征名称列表
        """
        self.feature_names = feature_names
    
    def plot_dtreeviz(
        self,
        tree_model,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = 'target',
        class_names: Optional[List[str]] = None,
        tree_index: int = 0,
        show_node_labels: bool = True,
        fancy: bool = True
    ):
        """使用dtreeviz可视化决策树.
        
        需要安装dtreeviz: pip install dtreeviz
        
        :param tree_model: 决策树模型或含有estimators_的模型
        :param X: 特征数据
        :param y: 目标变量
        :param target_name: 目标变量名
        :param class_names: 类别名称
        :param tree_index: 树索引（用于随机森林）
        :param show_node_labels: 显示节点标签
        :param fancy: 使用美观样式
        :return: dtreeviz对象
        """
        try:
            import dtreeviz
        except ImportError:
            raise ImportError(
                "需要安装dtreeviz: pip install dtreeviz. "
                "注意：dtreeviz需要graphviz系统库。"
            )
        
        # 获取单棵树
        if hasattr(tree_model, 'estimators_'):
            tree = tree_model.estimators_[tree_index]
        else:
            tree = tree_model
        
        if class_names is None:
            class_names = [str(c) for c in np.unique(y)]
        
        viz = dtreeviz.model(
            tree,
            X,
            y,
            target_name=target_name,
            feature_names=list(X.columns),
            class_names=class_names
        )
        
        return viz
    
    def plot_graphviz(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        filled: bool = True,
        rounded: bool = True,
        special_characters: bool = True,
        max_depth: Optional[int] = None,
        colors: Optional[List[str]] = None
    ):
        """使用graphviz可视化决策树.
        
        :param tree_model: 决策树模型
        :param feature_names: 特征名称
        :param class_names: 类别名称
        :param filled: 填充颜色
        :param rounded: 圆角节点
        :param special_characters: 支持特殊字符
        :param max_depth: 最大深度
        :param colors: 颜色列表，默认使用hscredit主题色
        :return: graphviz.Source对象
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("需要安装graphviz: pip install graphviz")
        
        feature_names = feature_names or self.feature_names
        
        if feature_names is None:
            raise ValueError("需要提供feature_names")
        
        if class_names is None:
            class_names = ['good', 'bad']
        
        # 获取单棵树
        if hasattr(tree_model, 'estimators_'):
            tree = tree_model.estimators_[0]
        else:
            tree = tree_model
        
        # 使用hscredit主题色
        colors = colors or DEFAULT_COLORS[:2]
        
        # 导出DOT数据
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            special_characters=special_characters,
            max_depth=max_depth,
            impurity=True,
            node_ids=True,
            proportion=True
        )
        
        # 添加hscredit样式设置
        dot_data = self._add_theme_settings(dot_data, colors)
        
        return graphviz.Source(dot_data)
    
    def _add_theme_settings(self, dot_data: str, colors: List[str]) -> str:
        """添加hscredit主题设置到DOT数据.
        
        :param dot_data: DOT数据
        :param colors: 颜色列表
        :return: 修改后的DOT数据
        """
        lines = dot_data.split('\n')
        
        # 在第一行后添加主题设置
        new_lines = [
            'digraph Tree {',
            f'    graph [colorscheme="{THEME_COLOR}"];',
            f'    node [color="{THEME_COLOR}"];',
            f'    edge [color="{THEME_COLOR}"];'
        ]
        
        # 跳过原有的digraph行
        for line in lines[1:]:
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def plot_matplotlib(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        filled: bool = True,
        rounded: bool = True,
        figsize: Tuple[int, int] = (20, 10),
        fontsize: int = 10,
        max_depth: Optional[int] = None,
        save_path: Optional[str] = None,
        dpi: int = 240
    ):
        """使用matplotlib可视化决策树.
        
        :param tree_model: 决策树模型
        :param feature_names: 特征名称
        :param class_names: 类别名称
        :param filled: 填充颜色
        :param rounded: 圆角节点
        :param figsize: 图大小
        :param fontsize: 字体大小
        :param max_depth: 最大深度
        :param save_path: 保存路径
        :param dpi: 分辨率
        :return: matplotlib Figure对象
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
        except ImportError:
            raise ImportError("需要安装matplotlib: pip install matplotlib")
        
        feature_names = feature_names or self.feature_names
        
        if class_names is None:
            class_names = ['good', 'bad']
        
        # 获取单棵树
        if hasattr(tree_model, 'estimators_'):
            tree = tree_model.estimators_[0]
        else:
            tree = tree_model
        
        # 创建figure
        fig, ax = get_or_create_ax(figsize=figsize)
        
        # 使用hscredit主题色设置
        setup_axis_style(ax, colors=[THEME_COLOR])
        
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            fontsize=fontsize,
            max_depth=max_depth,
            ax=ax,
            impurity=True,
            node_ids=True,
            proportion=True
        )
        
        ax.set_title('决策树可视化', color=THEME_COLOR)
        
        # 设置标题样式
        for spine in ax.spines.values():
            spine.set_color(THEME_COLOR)
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=dpi)
            print(f"决策树图已保存到: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None
    ):
        """绘制特征重要性图.
        
        :param tree_model: 树模型
        :param feature_names: 特征名称
        :param top_n: 显示前N个特征
        :param figsize: 图大小
        :param save_path: 保存路径
        :param colors: 颜色列表，默认使用hscredit主题色
        :param title: 自定义标题
        :return: matplotlib Figure对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装matplotlib")
        
        if not hasattr(tree_model, 'feature_importances_'):
            raise ValueError("模型没有feature_importances_属性")
        
        feature_names = feature_names or self.feature_names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(tree_model.feature_importances_))]
        
        # 使用hscredit主题色
        colors = colors or DEFAULT_COLORS
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': tree_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = get_or_create_ax(figsize=figsize)
        
        # 绘制条形图
        bars = ax.barh(
            importance_df['feature'][::-1],
            importance_df['importance'][::-1],
            color=colors[0]
        )
        
        # 设置样式
        setup_axis_style(ax, colors=[THEME_COLOR], hide_top_right=True)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center',
                fontsize=9,
                color=THEME_COLOR
            )
        
        plot_title = title or f'Top {top_n} 特征重要性'
        ax.set_title(plot_title, color=THEME_COLOR, fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性', color=THEME_COLOR)
        ax.set_ylabel('特征', color=THEME_COLOR)
        
        # 设置刻度颜色
        ax.tick_params(colors=THEME_COLOR)
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=240)
            print(f"特征重要性图已保存到: {save_path}")
        
        return fig
    
    def plot_tree_comparison(
        self,
        tree_models: List,
        model_names: List[str],
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 15),
        max_depth: int = 3,
        save_path: Optional[str] = None
    ):
        """比较多个决策树.
        
        :param tree_models: 树模型列表
        :param model_names: 模型名称列表
        :param feature_names: 特征名称
        :param figsize: 图大小
        :param max_depth: 最大深度
        :param save_path: 保存路径
        :return: matplotlib Figure对象
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
        except ImportError:
            raise ImportError("需要安装matplotlib")
        
        feature_names = feature_names or self.feature_names
        
        n_models = len(tree_models)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model, name) in enumerate(zip(tree_models, model_names)):
            if hasattr(model, 'estimators_'):
                tree = model.estimators_[0]
            else:
                tree = model
            
            plot_tree(
                tree,
                feature_names=feature_names,
                class_names=['good', 'bad'],
                filled=True,
                rounded=True,
                max_depth=max_depth,
                ax=axes[i],
                fontsize=8
            )
            axes[i].set_title(name, color=THEME_COLOR)
            
            # 设置边框颜色
            for spine in axes[i].spines.values():
                spine.set_color(THEME_COLOR)
        
        # 隐藏多余的子图
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path, dpi=240)
            print(f"对比图已保存到: {save_path}")
        
        return fig


def plot_decision_tree(
    tree_model,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    method: str = 'matplotlib',
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    **kwargs
):
    """便捷函数：绘制决策树.
    
    :param tree_model: 决策树模型
    :param X: 特征数据
    :param y: 目标变量
    :param feature_names: 特征名称
    :param method: 可视化方法，'matplotlib', 'graphviz', 'dtreeviz'
    :param save_path: 保存路径
    :param colors: 颜色列表，默认使用hscredit主题色
    :param kwargs: 其他参数
    :return: 可视化对象
    """
    colors = colors or DEFAULT_COLORS
    visualizer = TreeVisualizer(feature_names=feature_names)
    
    if method == 'matplotlib':
        return visualizer.plot_matplotlib(
            tree_model,
            feature_names=feature_names,
            save_path=save_path,
            **kwargs
        )
    
    elif method == 'graphviz':
        dot = visualizer.plot_graphviz(
            tree_model,
            feature_names=feature_names,
            colors=colors,
            **kwargs
        )
        if save_path:
            dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
            print(f"决策树图已保存到: {save_path}")
        return dot
    
    elif method == 'dtreeviz':
        if X is None or y is None:
            raise ValueError("dtreeviz方法需要提供X和y")
        return visualizer.plot_dtreeviz(
            tree_model,
            X,
            y,
            **kwargs
        )
    
    else:
        raise ValueError(f"不支持的method: {method}，可选: matplotlib, graphviz, dtreeviz")
