"""决策树可视化模块.

支持使用 pydotplus 和 graphviz 两种方式进行决策树可视化，
与原始 manual_tree_extractor.py 的 `tree_plot()` / `export_dot_data()` 逻辑完全一致。
"""

import re
from typing import Optional, List

import numpy as np
import pandas as pd

# 导入 hscredit 可视化工具
from ...core.viz.utils import DEFAULT_COLORS


# ==============================================================================
#  核心 DOT 工具函数（与原始 scripts/manual_tree_extractor.py 一致）
# ==============================================================================


def export_dot_data(
    clf,
    feature_list: List[str],
    class_names: Optional[List[str]] = None,
    out_file: Optional[str] = None,
    max_depth: Optional[int] = None,
    label: str = "all",
    filled: bool = True,
    leaves_parallel: bool = False,
    impurity: bool = True,
    node_ids: bool = True,
    proportion: bool = True,
    rotate: bool = False,
    rounded: bool = True,
    special_characters: bool = True,
    precision: int = 3,
) -> str:
    """导出决策树为 DOT 格式字符串.

    与原始 `export_dot_data()` 完全一致，唯一的额外功能：
    若 `class_names` 未传入，会自动从训练数据中推断。

    :param clf: 决策树分类器（sklearn 或含 tree_ 属性的模拟对象）
    :param feature_list: 特征名列表
    :param class_names: 类别名列表
    :param out_file: 输出 .dot 文件路径（可选）
    :param max_depth: 最大显示深度
    :param filled: 是否填充颜色
    :param rounded: 圆角节点
    :param special_characters: 支持特殊字符
    :param precision: 数值精度
    :return: DOT 格式字符串
    """
    from sklearn.tree import export_graphviz

    # 自动推断 class_names（仅当 clf 是真实 sklearn 对象时）
    if class_names is None and hasattr(clf, "classes_"):
        class_names = [str(c) for c in clf.classes_]

    dot_data = export_graphviz(
        decision_tree=clf,
        feature_names=feature_list,
        class_names=class_names,
        out_file=out_file,
        max_depth=max_depth,
        label=label,
        filled=filled,
        leaves_parallel=leaves_parallel,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rotate=rotate,
        rounded=rounded,
        special_characters=special_characters,
        precision=precision,
    )
    if out_file is not None:
        with open(out_file, "r") as f:
            dot_data = f.read()
    return dot_data


def tree_plot(dot_data: str, method: str = "pydotplus"):
    """绘制决策树.

    与原始 `tree_plot()` 完全一致，支持两种后端：
    - ``pydotplus``：返回 IPython ``Image`` 对象（可直接在 Notebook 中 display）
    - ``graphviz``：返回 ``graphviz.Source`` 对象（支持 .render() 保存）

    :param dot_data: DOT 格式字符串
    :param method: 渲染方式，``pydotplus`` 或 ``graphviz``
    :return: pydotplus.graph_from_dot_data / graphviz.Source 对象
    """
    if method == "pydotplus":
        import pydotplus

        graph = pydotplus.graph_from_dot_data(dot_data)
        # 返回 Image 对象（Notebook 中直接 display）
        try:
            from IPython.display import Image

            return Image(graph.create_png())
        except Exception:
            # 非 IPython 环境返回 graph 对象本身
            return graph
    elif method == "graphviz":
        import graphviz

        return graphviz.Source(dot_data)
    else:
        print("Warning: tree plotting requires pydotplus or graphviz. " "Returning raw dot data.")
        return dot_data


def adjust_dot_colors(
    dot_data: str,
    selected_node_ids: Optional[List[int]] = None,
    leaf_fill: str = "#FFFFFF",
    selected_fill: str = "#4682B4",
    regular_fill: str = "#D3D3D3",
) -> str:
    """修改 DOT 树图中的节点颜色.

    与原始 ``dot_data_adjust_select()`` 一致，用于高亮指定节点。

    :param dot_data: DOT 格式字符串
    :param selected_node_ids: 需要高亮的节点 ID 列表（通常为叶子节点）
    :param leaf_fill: 未选中叶子节点的颜色
    :param selected_fill: 选中节点的高亮颜色
    :param regular_fill: 普通节点的颜色
    :return: 修改颜色后的 DOT 字符串
    """
    if selected_node_ids is None:
        selected_node_ids = []

    dot_select = dot_data
    # 提取所有节点 ID（匹配 "\n123 [" 模式）
    node_pattern = re.compile(r"\n([\d]+)\s+\[")
    total_nodes = [int(m.group(1)) for m in node_pattern.finditer(dot_select)]
    selected_set = set(selected_node_ids)

    for nid in total_nodes:
        if nid in selected_set:
            # 高亮节点：描边加粗 + 高亮填充色
            dot_select = re.sub(
                rf"\n{nid}\s+\[",
                f'\n{nid} [style="filled,rounded", ',
                dot_select,
            )
            dot_select = re.sub(
                rf"(\n{nid}\s+\[.+?)\]\s*;",
                rf'\1, fillcolor="{selected_fill}"] ;',
                dot_select,
                flags=re.DOTALL,
            )
        else:
            # 普通节点：叶子用白色，内部节点用浅灰
            is_leaf = False
            if nid in selected_set:
                is_leaf = False
            dot_select = re.sub(
                rf"\n{nid}\s+\[",
                f'\n{nid} [style="filled,rounded,bold", ',
                dot_select,
            )
            dot_select = re.sub(
                rf"(\n{nid}\s+\[.+?)\]\s*;",
                rf'\1, fillcolor="{leaf_fill if is_leaf else regular_fill}"] ;',
                dot_select,
                flags=re.DOTALL,
            )

    return dot_select


# ==============================================================================
#  TreeVisualizer 封装类
# ==============================================================================

try:
    import graphviz
except ImportError:
    graphviz = None


class TreeVisualizer:
    """决策树可视化器.

    封装 ``export_dot_data`` + ``tree_plot`` + ``adjust_dot_colors`` 三个核心函数，
    提供 sklearn 风格 API，统一使用 hscredit 主题色。

    支持三种可视化方式（按推荐顺序）：

    1. **pydotplus（推荐）**：返回 IPython ``Image``，Notebook 中直接 ``display()``，
       颜色由 sklearn ``filled`` 参数控制（二分类时橙色=好、蓝色=坏）。
    2. **graphviz**：返回 ``graphviz.Source``，支持 ``.render()`` 保存为 PNG/PDF/SVG。
    3. **matplotlib**：使用 sklearn 内置 ``plot_tree``，颜色固定。

    **参考样例**

    >>> from hscredit.report.mining import TreeVisualizer
    >>> viz = TreeVisualizer()
    >>>
    >>> # 方式1：pydotplus（Notebook 直接 display）
    >>> img = viz.plot(tree_model, feature_names, max_depth=3)
    >>> display(img)
    >>>
    >>> # 方式2：graphviz（保存文件）
    >>> src = viz.plot(tree_model, feature_names, method='graphviz')
    >>> src.render('/tmp/tree', format='png', cleanup=True)
    >>>
    >>> # 方式3：matplotlib
    >>> fig = viz.plot_matplotlib(tree_model, feature_names)
    >>> fig.savefig('/tmp/tree.png', dpi=150)
    >>>
    >>> # 高亮指定叶子节点
    >>> img = viz.plot(tree_model, feature_names, method='pydotplus',
    ...                selected_node_ids=[3, 5, 6])
    >>> display(img)
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        :param feature_names: 特征名称列表
        """
        self.feature_names = feature_names

    def plot(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 3,
        proportion: bool = True,
        method: str = "pydotplus",
        selected_node_ids: Optional[List[int]] = None,
        out_file: Optional[str] = None,
    ):
        """绘制决策树（主入口方法）.

        :param tree_model: 决策树模型，或包含 ``tree_`` 属性的模拟对象
        :param feature_names: 特征名列表
        :param class_names: 类别名列表
        :param max_depth: 最大显示深度
        :param filled: 是否填充颜色（sklearn 内部按类别比例着色）
        :param rounded: 圆角节点
        :param precision: 数值精度（小数位）
        :param proportion: 是否显示样本比例
        :param method: ``pydotplus``（返回 Image）或 ``graphviz``（返回 Source）
        :param selected_node_ids: 需要高亮的节点 ID 列表（叶子规则节点）
        :param out_file: 输出 .dot 文件路径（可选）
        :return: pydotplus Image 或 graphviz.Source 对象
        """
        feature_names = feature_names or self.feature_names
        if feature_names is None:
            raise ValueError("需要提供 feature_names 参数或初始化时传入")

        # 获取底层树对象
        tree = self._get_tree(tree_model)

        # 导出 DOT
        dot_data = export_dot_data(
            tree,
            feature_list=feature_names,
            class_names=class_names,
            out_file=out_file,
            max_depth=max_depth,
            filled=filled,
            rounded=rounded,
            precision=precision,
            proportion=proportion,
        )

        # 高亮指定节点
        if selected_node_ids:
            dot_data = adjust_dot_colors(dot_data, selected_node_ids)

        return tree_plot(dot_data, method=method)

    def plot_pydotplus(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 3,
        proportion: bool = True,
        selected_node_ids: Optional[List[int]] = None,
    ):
        """使用 pydotplus 绘制（Notebook 直接 display）。

        参见 ``plot()`` 参数说明。
        """
        return self.plot(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth,
            filled=filled,
            rounded=rounded,
            precision=precision,
            proportion=proportion,
            method="pydotplus",
            selected_node_ids=selected_node_ids,
        )

    def plot_graphviz(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        filled: bool = True,
        rounded: bool = True,
        precision: int = 3,
        proportion: bool = True,
        selected_node_ids: Optional[List[int]] = None,
    ):
        """使用 graphviz 绘制（支持 .render() 保存文件）。

        参见 ``plot()`` 参数说明。
        """
        return self.plot(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth,
            filled=filled,
            rounded=rounded,
            precision=precision,
            proportion=proportion,
            method="graphviz",
            selected_node_ids=selected_node_ids,
        )

    def plot_matplotlib(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        filled: bool = True,
        rounded: bool = True,
        figsize: tuple = (20, 10),
        fontsize: float = 9,
        precision: int = 3,
        proportion: bool = True,
        save: Optional[str] = None,
        dpi: int = 240,
    ):
        """使用 matplotlib/sklearn plot_tree 绘制.

        :param tree_model: 决策树模型
        :param feature_names: 特征名列表
        :param class_names: 类别名列表
        :param max_depth: 最大显示深度
        :param filled: 是否填充颜色
        :param rounded: 圆角节点
        :param figsize: 图大小
        :param fontsize: 字体大小
        :param precision: 数值精度
        :param proportion: 是否显示样本比例
        :param save: 保存路径
        :param dpi: 分辨率
        :return: matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from sklearn.tree import plot_tree

        feature_names = feature_names or self.feature_names
        tree = self._get_tree(tree_model)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
            proportion=proportion,
            precision=precision,
        )
        plt.tight_layout()

        if save:
            import os

            save_dir = os.path.dirname(save)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save, dpi=dpi, format="png", bbox_inches="tight")

        return fig

    def plot_dtreeviz(
        self,
        tree_model,
        X: pd.DataFrame,
        y,
        target_name: str = "target",
        class_names: Optional[List[str]] = None,
        tree_index: int = 0,
        show_node_labels: bool = True,
        fancy: bool = True,
    ):
        """使用 dtreeviz 绘制（需要安装 dtreeviz）。

        :param tree_model: 决策树模型或含 estimators_ 的模型
        :param X: 特征数据（DataFrame）
        :param y: 目标变量
        :param target_name: 目标变量名
        :param class_names: 类别名列表
        :param tree_index: 树索引（用于随机森林）
        :param show_node_labels: 显示节点标签
        :param fancy: 美观样式
        :return: dtreeviz 模型对象
        """
        try:
            import dtreeviz
        except ImportError:
            raise ImportError("需要安装 dtreeviz: pip install dtreeviz。" "注意：dtreeviz 还需要 graphviz 系统库。")

        if hasattr(tree_model, "estimators_"):
            tree = tree_model.estimators_[tree_index]
        else:
            tree = tree_model

        if class_names is None:
            class_names = [str(c) for c in np.unique(y)]

        return dtreeviz.model(
            tree,
            X,
            y,
            target_name=target_name,
            feature_names=list(X.columns),
            class_names=class_names,
        )

    def plot_feature_importance(
        self,
        tree_model,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        figsize: tuple = (10, 6),
        color: str = "#2639E9",
        save: Optional[str] = None,
        dpi: int = 240,
    ):
        """绘制特征重要性图.

        :param tree_model: 树模型
        :param feature_names: 特征名列表
        :param top_n: 显示前 N 个特征
        :param figsize: 图大小
        :param color: 柱状图颜色
        :param save: 保存路径
        :param dpi: 分辨率
        :return: matplotlib Figure 对象
        """
        import matplotlib.pyplot as plt

        feature_names = feature_names or self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(tree_model.feature_importances_))]

        imp = (
            pd.DataFrame({"feature": feature_names, "importance": tree_model.feature_importances_})
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(imp["feature"], imp["importance"], color=color)
        for bar in bars:
            w = bar.get_width()
            ax.text(w, bar.get_y() + bar.get_height() / 2, f" {w:.3f}", va="center", fontsize=9, color=color)
        ax.set_xlabel("Importance", color="#2639E9")
        ax.tick_params(colors="#2639E9")
        ax.set_title(f"Top {top_n} Feature Importance", color="#2639E9", fontweight="bold")
        plt.tight_layout()

        if save:
            import os

            save_dir = os.path.dirname(save)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save, dpi=dpi, format="png", bbox_inches="tight")

        return fig

    # -------------------------------------------------------------------------
    #  内部工具
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_tree(tree_model):
        """从各种树模型中提取底层 sklearn DecisionTreeClassifier."""
        if hasattr(tree_model, "estimators_") and tree_model.estimators_:
            return tree_model.estimators_[0]
        if hasattr(tree_model, "tree_"):
            return tree_model
        raise ValueError(
            "tree_model 需为 sklearn 决策树（Classifier/Regressor）或" "包含 tree_ / estimators_ 属性的对象"
        )


# ==============================================================================
#  顶层便捷函数（与原始 API 完全兼容）
# ==============================================================================


def plot_decision_tree(
    tree_model,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    method: str = "pydotplus",
    filled: bool = True,
    rounded: bool = True,
    precision: int = 3,
    proportion: bool = True,
    selected_node_ids: Optional[List[int]] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """便捷函数：绘制单棵决策树.

    与原始 ``tree_plot(export_dot_data(...))`` 调用链完全等价。

    :param tree_model: 决策树模型
    :param feature_names: 特征名列表
    :param class_names: 类别名列表
    :param max_depth: 最大显示深度
    :param method: ``pydotplus`` 或 ``graphviz``
    :param filled: 是否填充颜色
    :param rounded: 圆角节点
    :param precision: 数值精度
    :param proportion: 是否显示样本比例
    :param selected_node_ids: 高亮节点 ID
    :param save: 保存路径（仅 graphviz 方法支持）
    :return: pydotplus Image 或 graphviz.Source
    """
    viz = TreeVisualizer(feature_names=feature_names)

    if method == "matplotlib":
        fig = viz.plot_matplotlib(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            max_depth=max_depth,
            filled=filled,
            rounded=rounded,
            precision=precision,
            proportion=proportion,
        )
        if save:
            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        return fig

    result = viz.plot(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=filled,
        rounded=rounded,
        precision=precision,
        proportion=proportion,
        method=method,
        selected_node_ids=selected_node_ids,
    )

    if save and method == "graphviz":
        result.render(save.replace(".png", ""), format="png", cleanup=True)

    return result
