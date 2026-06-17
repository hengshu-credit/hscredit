"""决策树可视化模块 — AntV G6 组织结构图风格.

参考 https://ant-design-charts.antgroup.com/examples/relations/organization-chart/#complex-node
的卡片式节点布局，实现一套类 AntV 风格的决策树可视化。

**核心特点（AntV G6 风格）**：
- **卡片节点**：圆角矩形卡片，内含节点标题、分裂条件、统计指标
- **层级布局**：从上到下自动排版，父节点居中，子节点均匀分布
- **平滑连线**：曲线连接父子节点，分支标签（<= / >）清晰标注
- **颜色语义**：蓝/绿=低坏账，红/橙=高坏账（hscredit 风控主题色）
- **双 API 支持**：支持 ManualTreeExtractor 和 sklearn DecisionTreeClassifier

**三种渲染后端**：
1. **matplotlib** — 纯 Python 无外部依赖，适合快速预览
2. **pyecharts** — 交互式 HTML，支持鼠标悬停tooltip、缩放、导出
3. **graphviz** — 高质量矢量图，适合嵌入报告

**参考样例**

>>> from tree_viz import DecisionTreeViz, plot_tree_matplotlib
>>> # matplotlib 快速绘图
>>> plot_tree_matplotlib(ext, save='tree.png')

>>> # pyecharts 交互式图表
>>> viz = DecisionTreeViz(backend='pyecharts')
>>> chart = viz.plot(ext)
>>> chart.render('tree.html')

>>> # graphviz 高质量图
>>> viz = DecisionTreeViz(backend='graphviz')
>>> chart = viz.plot(ext)
>>> chart.render('tree.pdf')
"""

import html
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _tex_label(text: str) -> str:
    """将 ASCII 比较符号转换为 TeX math text，用于 matplotlib 渲染.

    使用 matplotlib 原生 math text 语法（无需 usetex），
    如 "x <= 600" → "x $\leq$ 600"，">" → "$>$"。

    :param text: 原始文本
    :return: TeX math text 格式
    """
    # 使用 matplotlib 原生 math text（无需安装 LaTeX，兼容 Agg 后端）
    return text.replace("<=", r" $\leq$ ")


__all__ = [
    "DecisionTreeViz",
    "plot_tree_matplotlib",
    "plot_tree_pyecharts",
    "plot_tree_graphviz",
    "_AntVNodeStyle",
    "_AntVEdgeStyle",
]

# ============================================================================
# 颜色主题（hscredit 风控主题 + AntV 设计语言）
# ============================================================================

# hscredit 风控主题色
_COLOR_PRIMARY = "#2639E9"  # 主色蓝
_COLOR_SECONDARY = "#F76E6C"  # 副色红
_COLOR_ACCENT = "#FE7715"  # 强调色橙
_COLOR_SUCCESS = "#52C41A"  # 低风险绿
_COLOR_BG = "#FFFFFF"  # 背景白
_COLOR_CARD_BG = "#FAFBFF"  # 卡片背景
_COLOR_BORDER = "#E8ECFF"  # 边框浅蓝
_COLOR_TEXT_DARK = "#1D2129"  # 深色文字
_COLOR_TEXT_MID = "#4B5563"  # 中等文字
_COLOR_TEXT_LIGHT = "#86909C"  # 浅色文字
_COLOR_GRID = "#F2F3F7"  # 网格线

# 节点宽度/高度（以 inch 为单位，转换为点数需乘 dpi）
_NODE_W_INCH = 2.8
_NODE_H_INCH = 1.6
_NODE_GAP_X = 0.6  # 节点间水平间距
_NODE_GAP_Y = 1.2  # 层级间垂直间距
_NODE_STROKE_WIDTH = 1.5


# ============================================================================
# 树结构提取工具函数
# ============================================================================


def _extract_tree_from_mte(mte) -> Dict[str, Any]:
    """从 ManualTreeExtractor 提取树数据字典。"""
    ti = mte._tree_info
    children_left = ti.children_left
    children_right = ti.children_right
    feature = ti.feature
    threshold = ti.threshold
    n_samples = ti.n_node_samples
    values = ti.value
    impurity = ti.impurity
    feat_names = ti.feature_names or []
    n_classes = ti.n_classes or 2
    total_samples = sum(n_samples) if n_samples else 1
    manual_nodes = mte._manual_split_nodes

    return _build_tree_data(
        children_left, children_right, feature, threshold,
        n_samples, values, impurity, feat_names, n_classes,
        total_samples, manual_nodes
    )


def _extract_tree_from_sklearn(clf, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """从 sklearn DecisionTreeClassifier 提取树数据字典。"""
    tree = clf.tree_
    children_left = list(tree.children_left)
    children_right = list(tree.children_right)
    feature = list(tree.feature)
    threshold = list(tree.threshold)
    n_samples = list(tree.n_node_samples)
    values = [list(v) for v in tree.value]
    impurity = list(tree.impurity)
    feat_names = list(feature_names) if feature_names is not None else []
    n_features_in_ = getattr(tree, "n_features_in_", None) or getattr(tree, "n_features", 0)
    if not feat_names:
        feat_names = [f"特征[{i}]" for i in range(n_features_in_)]
    n_classes = tree.n_classes_[0] if hasattr(tree, "n_classes_") else 2
    total_samples = sum(n_samples) if n_samples else 1
    manual_nodes = set()

    return _build_tree_data(
        children_left, children_right, feature, threshold,
        n_samples, values, impurity, feat_names, n_classes,
        total_samples, manual_nodes
    )


def _build_tree_data(
    children_left: List[int],
    children_right: List[int],
    feature: List[int],
    threshold: List[float],
    n_samples: List[int],
    values: List,
    impurity: List[float],
    feat_names: List[str],
    n_classes: int,
    total_samples: int,
    manual_nodes: set,
) -> Dict[str, Any]:
    """构建统一的树数据字典。

    :return: 含 'nodes'（节点列表）和 'edges'（边列表）的字典
    """
    n_nodes = len(feature)
    nodes = []
    edges = []

    # 计算整体坏账率（用于 LIFT 计算）
    total_bad = 0.0
    total_sample_count = 0
    for node_id in range(n_nodes):
        n_s = n_samples[node_id] if node_id < len(n_samples) else 0
        vals_n = values[node_id] if node_id < len(values) else [[0.5] * n_classes]
        if n_classes == 2 and n_s > 0:
            val1 = vals_n[0][1] if vals_n else 0.5
            total_bad += val1 * n_s
            total_sample_count += n_s
    overall_bad_rate = total_bad / total_sample_count if total_sample_count > 0 else 0.0

    # 计算每个节点的层级深度
    depths = _compute_node_depths(n_nodes, children_left, children_right)

    # 预计算节点颜色色阶
    all_node_br: List[float] = []
    for nid in range(n_nodes):
        n_s = n_samples[nid] if nid < len(n_samples) else 0
        v = values[nid] if nid < len(values) else [[0.5] * n_classes]
        if n_classes == 2 and n_s > 0:
            br = v[0][1] / (v[0][0] + v[0][1]) if (v[0][0] + v[0][1]) > 0 else 0.0
        else:
            br = 0.0
        all_node_br.append(br)
    _node_stops = _build_gradient_stops(
        min(all_node_br) if all_node_br else 0.0,
        max(all_node_br) if all_node_br else 1.0,
    )

    for node_id in range(n_nodes):
        vals = values[node_id] if node_id < len(values) else [[0.5] * n_classes]
        feat_idx = feature[node_id]
        is_leaf = feat_idx == -2
        is_manual = node_id in manual_nodes

        # 好/坏样本数
        if n_classes == 2:
            node_total = n_samples[node_id] if node_id < len(n_samples) else 0
            val0 = vals[0][0] if vals else 0.5
            val1 = vals[0][1] if vals else 0.5
            good_count = int(round(val0 * node_total)) if node_total > 0 else 0
            bad_count = int(round(val1 * node_total)) if node_total > 0 else 0
            bad_rate = bad_count / node_total if node_total > 0 else 0.0
        else:
            good_count = 0
            bad_count = 0
            bad_rate = 0.0
            node_total = n_samples[node_id] if node_id < len(n_samples) else 0

        # LIFT = 节点坏账率 / 整体坏账率
        lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0

        # 节点标题（叶子 vs 分裂）
        if is_leaf:
            title_str = f"叶子节点 N{node_id}"
            class_label = "高风险" if bad_rate > 0.3 else ("中风险" if bad_rate > 0.1 else "低风险")
        else:
            title_str = f"分裂节点 N{node_id}"
            class_label = ""

        # 分裂条件文本
        if is_leaf:
            cond_text = "叶子节点"
            split_feat = ""
            th_text = ""
        else:
            feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"x[{feat_idx}]"
            th = threshold[node_id] if node_id < len(threshold) else 0.0
            cond_text = f"{feat_name} <= {th:.4g}"
            split_feat = feat_name
            th_text = f"{th:.4g}"

        imp_val = impurity[node_id] if node_id < len(impurity) else 0.0

        # AntV 风格固定填充色（统一颜色语义）
        fill_color = _compute_fill_color(bad_rate, _node_stops)

        node_data = {
            "node_id": node_id,
            "title": title_str,
            "condition": cond_text,
            "split_feature": split_feat,
            "threshold_text": th_text,
            "is_leaf": is_leaf,
            "is_manual": is_manual,
            "n_samples": node_total,
            "sample_pct": node_total / total_sample_count if total_sample_count > 0 else 0,
            "good_count": good_count,
            "bad_count": bad_count,
            "bad_rate": bad_rate,
            "gini": imp_val,
            "lift": lift,
            "fill_color": fill_color,
            "class_label": class_label,
            "depth": depths.get(node_id, 0),
        }
        nodes.append(node_data)

        # 添加边
        left_child = children_left[node_id] if node_id < len(children_left) else -1
        right_child = children_right[node_id] if node_id < len(children_right) else -1
        if left_child != -1:
            edges.append({
                "source": node_id,
                "target": left_child,
                "label": "<=",
                "label_pos": 0.5,
            })
        if right_child != -1:
            edges.append({
                "source": node_id,
                "target": right_child,
                "label": ">",
                "label_pos": 0.5,
            })

    return {"nodes": nodes, "edges": edges, "total_samples": total_sample_count}


def _compute_node_depths(n_nodes: int, children_left: List[int], children_right: List[int]) -> Dict[int, int]:
    """计算每个节点的深度（根节点=0）。"""
    depths: Dict[int, int] = {}

    def dfs(node: int, depth: int) -> None:
        if node >= n_nodes or node < 0:
            return
        if node in depths:
            return
        depths[node] = depth
        left = children_left[node] if node < len(children_left) else -1
        right = children_right[node] if node < len(children_right) else -1
        if left != -1:
            dfs(left, depth + 1)
        if right != -1:
            dfs(right, depth + 1)

    dfs(0, 0)
    return depths


def _build_gradient_stops(min_br: float, max_br: float) -> List[Tuple[float, Tuple[int, int, int]]]:
    """根据实际坏账率区间生成色阶。

    白 → 淡蓝（整体坏账率附近） → 淡红
    """
    # 柔和色阶
    C_WHITE = (255, 255, 255)  # 白色
    C_LIGHT_BLUE  = (180, 210, 255)  # 淡蓝 #B4D2FF
    C_LIGHT_RED   = (255, 200, 200)  # 淡红 #FFC8C8

    # 确保有区分度
    if max_br <= min_br:
        min_br = 0.0
        max_br = 1.0
    if max_br - min_br < 0.01:
        max_br = min_br + 0.5

    def blend_color(t: float) -> Tuple[int, int, int]:
        """t=0→白, t=0.5→淡蓝, t=1→淡红"""
        if t <= 0.5:
            s = t / 0.5
            r = int(round(C_WHITE[0] + s * (C_LIGHT_BLUE[0] - C_WHITE[0])))
            g = int(round(C_WHITE[1] + s * (C_LIGHT_BLUE[1] - C_WHITE[1])))
            b = int(round(C_WHITE[2] + s * (C_LIGHT_BLUE[2] - C_WHITE[2])))
        else:
            s = (t - 0.5) / 0.5
            r = int(round(C_LIGHT_BLUE[0] + s * (C_LIGHT_RED[0] - C_LIGHT_BLUE[0])))
            g = int(round(C_LIGHT_BLUE[1] + s * (C_LIGHT_RED[1] - C_LIGHT_BLUE[1])))
            b = int(round(C_LIGHT_BLUE[2] + s * (C_LIGHT_RED[2] - C_LIGHT_BLUE[2])))
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    # 生成 11 个采样点
    n = 11
    stops = []
    for i in range(n):
        t = i / (n - 1)
        color = blend_color(t)
        br_val = min_br + t * (max_br - min_br)
        stops.append((br_val, color))
    return stops


def _measure_text_width(text: str, fontsize: float, fontweight: str = "normal",
                        use_tex: bool = False) -> float:
    """测量文本在数据坐标系下的宽度（inch）。

    基于 matplotlib text 渲染器测量，使用当前 figure 的 dpi，
    假设 ax.set_aspect('equal') 后 x 轴 1 unit = 1 inch。

    :param text: 文本内容
    :param fontsize: 字号（points）
    :param fontweight: 粗细
    :param use_tex: 是否将 text 先通过 _tex_label 转换再测量（用于渲染时的精确布局）
    :return: 文本宽度（inch）
    """
    if use_tex:
        text = _tex_label(text)
    fig_tmp = plt.figure(figsize=(1, 1))
    ax_tmp = fig_tmp.add_axes([0, 0, 1, 1])
    # 不指定 fontfamily，使用当前 rcParams 默认字体（与实际渲染时一致），
    # 否则测量字体（如不含中文字形的 sans-serif）与渲染字体不一致会导致宽度严重低估，
    # 是节点文字溢出/与徽章重叠的根本原因
    t = ax_tmp.text(0, 0, text, fontsize=fontsize, fontweight=fontweight,
                    ha='left', va='center')
    renderer = fig_tmp.canvas.get_renderer()
    bb = t.get_window_extent(renderer)
    fig_tmp.clf()
    plt.close(fig_tmp)
    # 宽度 = (right - left) / dpi，即 inch
    return (bb.x1 - bb.x0) / fig_tmp.dpi


def _wrap_condition_text(text: str, max_width: float, fontsize: float) -> List[str]:
    """将切分条件文本智能换行，使其不超过 max_width（inch）。

    换行策略：
    1. 尝试整行
    2. 尝试在 "<=" 处拆分（特征名一行，阈值一行）
    3. 强制在空格处拆分（多行）

    :param text: 条件文本（如 "衡枢鉴真分老客版 <= 600"）
    :param max_width: 最大可用宽度（inch）
    :param fontsize: 字号（points）
    :return: 换行后的文本行列表
    """
    # 测量整行宽度（TeX 渲染后宽度，用于准确布局）
    if _measure_text_width(text, fontsize, use_tex=True) <= max_width:
        return [text]

    # 策略1：在 "<=" 处拆分
    if "<=" in text:
        feat_part, th_part = text.split("<=", 1)
        feat_w = _measure_text_width(feat_part, fontsize, use_tex=True)
        th_w = _measure_text_width(th_part, fontsize, use_tex=True)
        # 如果两部分各自能放下，分两行
        if feat_w <= max_width and th_w <= max_width:
            return [feat_part.strip(), f"<= {th_part.strip()}"]

    # 策略2：强制按空格拆分（单词换行）
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if _measure_text_width(test, fontsize, use_tex=True) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            # 如果单词本身就超宽，直接截断（单字符单词不会太宽）
            if _measure_text_width(word, fontsize, use_tex=True) > max_width:
                # 在单词内部找能放下的前缀
                for i in range(1, len(word) + 1):
                    if _measure_text_width(word[:i] + "-", fontsize, use_tex=True) > max_width:
                        break
                # 放能放下的部分，剩余的继续
                prefix = word[:max(1, i - 1)]
                current = prefix
            else:
                current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def _compute_fill_color(bad_rate, stops):
    """根据预计算的色阶，对给定坏账率返回对应颜色.

    :param bad_rate: 坏账率（0~1）
    :param stops: 由 _build_gradient_stops 生成的色阶列表
    """
    if bad_rate < 0:
        bad_rate = 0.0
    br = min(bad_rate, 1.0)

    if not stops or len(stops) < 2:
        return "#F0F4FF"

    # 线性插值
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= br <= t1:
            alpha = (br - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(round(c0[0] + alpha * (c1[0] - c0[0])))
            g = int(round(c0[1] + alpha * (c1[1] - c0[1])))
            b = int(round(c0[2] + alpha * (c1[2] - c0[2])))
            return f"#{r:02X}{g:02X}{b:02X}"
    # 兜底
    return f"#{stops[-1][1][0]:02X}{stops[-1][1][1]:02X}{stops[-1][1][2]:02X}"


# ============================================================================
# AntV 节点样式定义
# ============================================================================


class _AntVNodeStyle:
    """AntV G6 风格的节点样式生成器。"""

    @staticmethod
    def card_style(node: Dict[str, Any], fill_color: str) -> Dict[str, Any]:
        is_leaf = node["is_leaf"]
        is_manual = node["is_manual"]

        if is_manual:
            border_color = _COLOR_SECONDARY
            stroke_w = 2.5
        elif is_leaf:
            border_color = _COLOR_BORDER
            stroke_w = _NODE_STROKE_WIDTH
        else:
            border_color = _COLOR_PRIMARY
            stroke_w = _NODE_STROKE_WIDTH

        return {
            "width": _NODE_W_INCH,
            "height": _NODE_H_INCH,
            "fill": fill_color,
            "stroke": border_color,
            "linewidth": stroke_w,
        }


# ============================================================================
# 布局算法（Reingold-Tilford 风格，适配 AntV 层级树）
# ============================================================================


def _reingold_tilford_layout(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    node_width_override: Optional[float] = None,
) -> Dict[int, Tuple[float, float]]:
    """Reingold-Tilford 树布局算法，计算每个节点的 (x, y) 坐标。

    AntV G6 风格的垂直布局：根节点在顶部，子节点向下延伸。
    同层节点水平居中对齐，子树均匀分布。

    :param nodes: 节点数据列表
    :param edges: 边数据列表
    :param node_width_override: 全局节点宽度覆盖值（可选，所有节点统一宽度，缺省使用默认宽度）
    :return: 节点ID到坐标的映射 {node_id: (x, y)}
    """
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}

    # 构建父子关系
    children: Dict[int, List[int]] = {n["node_id"]: [] for n in nodes}
    parent: Dict[int, int] = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        children[src].append(tgt)
        parent[tgt] = src

    # 根节点
    root = 0
    for n in nodes:
        nid = n["node_id"]
        if nid not in parent:
            root = nid
            break

    # BFS 计算每个节点的深度
    depth_map: Dict[int, int] = {root: 0}
    queue = [root]
    while queue:
        curr = queue.pop(0)
        for child in children.get(curr, []):
            if child not in depth_map:
                depth_map[child] = depth_map[curr] + 1
                queue.append(child)

    max_depth = max(depth_map.values()) if depth_map else 0

    # 每层的节点列表
    nodes_by_depth: Dict[int, List[int]] = {}
    for nid, d in depth_map.items():
        nodes_by_depth.setdefault(d, []).append(nid)

    # 确定节点宽度：优先使用 node_width_override（全树统一），否则使用默认值
    def get_node_width(nid: int) -> float:
        if node_width_override is not None:
            return node_width_override
        return _NODE_W_INCH

    # 节点宽度（考虑间距）
    height_step = _NODE_H_INCH + _NODE_GAP_Y

    # 为每层节点分配 x 坐标
    coords: Dict[int, Tuple[float, float]] = {}

    for depth in range(max_depth + 1):
        level_nodes = nodes_by_depth.get(depth, [])
        n_at_level = len(level_nodes)
        if n_at_level == 0:
            continue

        # 计算此层所有节点的总宽度（包含间距）
        total_level_width = sum(get_node_width(nid) + _NODE_GAP_X for nid in level_nodes) - _NODE_GAP_X
        start_x = -total_level_width / 2

        # 为此层的每个节点分配 x 坐标
        x_offset = start_x
        for nid in sorted(level_nodes):
            node_w = get_node_width(nid)
            x = x_offset + node_w / 2  # 节点中心
            y = -depth * height_step   # y 向下为负
            coords[nid] = (x, y)
            x_offset += node_w + _NODE_GAP_X

    return coords


# ============================================================================
# matplotlib 渲染器
# ============================================================================


def plot_tree_matplotlib(
    tree_obj: Any,
    figsize: Tuple[float, float] = (18, 12),
    dpi: int = 150,
    save: Optional[str] = None,
    title: str = "",
    show_stats: bool = True,
    show_gini: bool = True,
    node_color_scheme: str = "risk",  # "risk" | "depth"
    feature_names: Optional[List[str]] = None,
) -> plt.Figure:
    """使用 matplotlib 绘制 AntV G6 风格的决策树。

    **AntV G6 风格特点**：
    - 卡片式节点：圆角矩形，内含标题、统计信息
    - 平滑曲线连线：子节点从父节点底部中点出发
    - 颜色语义：按坏账率从浅蓝→浅红渐变

    **参数**

    :param tree_obj: ManualTreeExtractor 或 sklearn DecisionTreeClassifier
    :param figsize: 初始画布大小（宽, 高），单位英寸；实际画布会按节点统一宽度/树形结构
        自动重新计算并覆盖该值，以保证 1 个数据坐标单位严格等于 1 inch（否则节点框
        与文字字号的相对比例会被意外缩放，导致文字溢出节点）
    :param dpi: 图像分辨率
    :param save: 保存路径（如 'tree.png'），如传入路径中有文件夹不存在，会自动创建，默认 None
    :param title: 图表标题
    :param show_stats: 是否显示节点统计信息（样本数、坏账率等）
    :param show_gini: 是否显示 Gini 不纯度
    :param node_color_scheme: 配色方案，'risk'=按坏账率，'depth'=按深度
    :return: matplotlib Figure 对象

    **参考样例**

    >>> fig = plot_tree_matplotlib(ext, figsize=(20, 14), dpi=200)
    >>> plt.show()
    >>> fig.savefig('tree.png', dpi=200, bbox_inches='tight')
    """
    # 提取树数据（feature_names 仅对 sklearn 树有效，ManualTreeExtractor 自行读取）
    tree_data = _extract_tree_data(tree_obj, feature_names=feature_names)
    nodes = tree_data["nodes"]
    edges = tree_data["edges"]

    if not nodes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "树为空或未拟合", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # 创建图形（后续会根据布局更新坐标范围）
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#FAFBFF")
    fig.patch.set_facecolor("#FAFBFF")

    # 动态色阶：从实际节点坏账率区间生成（低→主色蓝透明，高→副色红透明）
    all_br = [n["bad_rate"] for n in nodes]
    min_br = min(all_br)
    max_br = max(all_br)
    gradient_stops = _build_gradient_stops(min_br, max_br)

    # 为每个节点计算填充色
    node_fill_colors: Dict[int, str] = {}
    for n in nodes:
        node_fill_colors[n["node_id"]] = _compute_fill_color(n["bad_rate"], gradient_stops)

    # ============================================================
    # 全树统一节点宽度 + 标题行换行 + padding 计算
    # ============================================================
    # 固定参数
    FONT_TITLE = 10
    FONT_BODY = 9
    CONTENT_LINES = 5  # 内容行数（gini/samples/pct/bad_rate/lift）
    TITLE_H = 0.38  # 单行标题高度（inch）

    # 圆徽章参数（单位：inch）
    BADGE_R = 0.16
    BADGE_DIAM = BADGE_R * 2  # = 0.32

    # 标题行左右 padding 均等于圆徽章直径：
    # 左侧 padding 刚好容纳节点编号圆徽章，避免与条件文本重叠；
    # 右侧采用相同宽度的 padding，使条件文本在节点内整体居中（视觉对称）
    TITLE_PAD_LEFT = BADGE_DIAM
    TITLE_PAD_RIGHT = BADGE_DIAM

    NODE_W_MIN = 2.0  # 节点宽度下限（inch）
    NODE_W_MAX = 5.0  # 节点宽度上限（inch），防止单个超长特征名把节点撑得过宽

    # 第一步：测量全树每个节点标题内容（分裂条件 / "叶子节点"）的单行宽度，
    # 取全树最大值作为统一节点宽度的依据（含上限），保证所有节点等宽、整齐排列
    title_font_size = FONT_TITLE
    node_id_to_idx: Dict[int, int] = {}   # 节点ID到nodes列表索引的映射
    max_content_w = 0.0

    for idx, node in enumerate(nodes):
        node_id_to_idx[node["node_id"]] = idx
        if node["is_leaf"]:
            content_w = _measure_text_width("叶子节点", title_font_size, "bold")
        else:
            content_w = _measure_text_width(node["condition"], title_font_size, "bold", use_tex=True)
        max_content_w = max(max_content_w, content_w)

    # 第二步：全树统一节点宽度 = 最大内容宽度 + 左右 padding，并裁剪到 [下限, 上限]
    uniform_node_w = min(max(max_content_w + TITLE_PAD_LEFT + TITLE_PAD_RIGHT, NODE_W_MIN), NODE_W_MAX)

    # 第三步：基于统一宽度对应的可用文本区域，对每个分裂节点的条件文本换行
    # （仅当条件文本宽度超过上限对应的可用宽度时才会真正换行为多行）
    available_w = uniform_node_w - TITLE_PAD_LEFT - TITLE_PAD_RIGHT
    cond_lines: List[List[str]] = []      # 每个节点换行后的行列表
    for idx, node in enumerate(nodes):
        if node["is_leaf"]:
            cond_lines.append([])
        else:
            wrapped = _wrap_condition_text(node["condition"], available_w, title_font_size)
            cond_lines.append(wrapped)

    # Reingold-Tilford 布局（全树统一节点宽度）
    coords = _reingold_tilford_layout(nodes, edges, node_width_override=uniform_node_w)
    if not coords:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "布局计算失败", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # ============================================================
    # 绘制边和节点的参数预计算
    # ============================================================
    # 内容区起始 y（从标题行底部往上，减去额外标题行高度）
    # 额外标题行高度 = (n_lines - 1) * TITLE_H
    CONTENT_START_Y_OFFSET = TITLE_H + 0.08  # 标题行底部 + padding

    # ============================================================
    # 绘制边（直线连接 + 主题色 + 仅根节点边标注）
    # ============================================================
    ROOT_ID = 0
    for edge in edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        x1, y1 = coords[src_id]
        x2, y2 = coords[tgt_id]

        # 获取节点的高度（需要计算标题行数）
        src_idx = node_id_to_idx[src_id]
        src_n_title_lines = len(cond_lines[src_idx])
        src_extra_title_h = max(0, (src_n_title_lines - 1)) * TITLE_H
        src_total_title_h = TITLE_H + src_extra_title_h
        src_total_node_h = CONTENT_START_Y_OFFSET + CONTENT_LINES * 0.28 + src_extra_title_h

        tgt_idx = node_id_to_idx[tgt_id]
        tgt_n_title_lines = len(cond_lines[tgt_idx])
        tgt_extra_title_h = max(0, (tgt_n_title_lines - 1)) * TITLE_H
        tgt_total_title_h = TITLE_H + tgt_extra_title_h
        tgt_total_node_h = CONTENT_START_Y_OFFSET + CONTENT_LINES * 0.28 + tgt_extra_title_h

        label_text = edge["label"]
        edge_color = "#165DFF" if label_text == "<=" else "#F53F3B"

        # 直线连接：从父节点底边中点 → 子节点顶边中点
        ax.plot(
            [x1, x2],
            [y1 - src_total_node_h / 2, y2 + tgt_total_node_h / 2],
            color=edge_color,
            linewidth=2.0,
            zorder=1,
        )

        # 边标签：放在线段中点，非根节点边字号更小
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        label_bg = "#E8F0FF" if label_text == "<=" else "#FFF1F0"
        is_root_edge = src_id == ROOT_ID
        label_fontsize = 9 if is_root_edge else 8
        label_pad = 0.2 if is_root_edge else 0.12
        bbox_props = dict(
            boxstyle=f"round,pad={label_pad}",
            facecolor=label_bg,
            edgecolor=edge_color,
            linewidth=1.2,
        )
        ax.text(
            mid_x, mid_y,
            f" {_tex_label(label_text)} ",
            ha="center", va="center",
            fontsize=label_fontsize, fontweight="bold", color=edge_color,
            bbox=bbox_props, zorder=3,
        )

    # ============================================================
    # 绘制节点
    # ============================================================
    node_id_to_total_h: Dict[int, float] = {}  # 记录每个节点的实际高度，供后续画布尺寸计算复用
    for idx, node in enumerate(nodes):
        nid = node["node_id"]
        node_w = uniform_node_w
        x, y = coords[nid]
        is_leaf = node["is_leaf"]
        is_manual = node["is_manual"]
        fill_color = node_fill_colors[nid]
        n_title_lines = len(cond_lines[idx])
        extra_title_h = max(0, (n_title_lines - 1)) * TITLE_H
        total_title_h = TITLE_H + extra_title_h

        # 边框颜色
        if is_manual:
            edge_color = _COLOR_SECONDARY
            lw = 2.5
        elif is_leaf:
            edge_color = _COLOR_BORDER
            lw = 1.5
        else:
            edge_color = _COLOR_PRIMARY
            lw = 1.5

        # 节点高度 = 内容区 + 总标题高度
        total_node_h = CONTENT_START_Y_OFFSET + CONTENT_LINES * 0.28 + extra_title_h
        node_id_to_total_h[nid] = total_node_h

        # 画节点矩形（动态宽高）
        rect = plt.Rectangle(
            (x - node_w / 2, y - total_node_h / 2),
            node_w,
            total_node_h,
            linewidth=lw,
            edgecolor=edge_color,
            facecolor=fill_color,
            zorder=2,
        )
        ax.add_patch(rect)

        # ========== 标题行背景 ==========
        title_bar_y_top = y + total_node_h / 2
        title_bar_y_bottom = title_bar_y_top - total_title_h

        title_bg_color = edge_color if is_manual else (edge_color if not is_leaf else "#4B5563")
        title_bar = plt.Rectangle(
            (x - node_w / 2, title_bar_y_bottom),
            node_w,
            total_title_h,
            linewidth=0,
            facecolor=title_bg_color,
            zorder=3,
        )
        ax.add_patch(title_bar)

        # ========== 圆形徽章（节点编号）==========
        # 徽章居中放置在左侧 padding 区域内（宽度恰为 BADGE_DIAM），
        # 徽章右边缘正好与条件文本区域左边缘对齐，不会重叠
        badge_y_center = (title_bar_y_top + title_bar_y_bottom) / 2
        badge_x = x - node_w / 2 + BADGE_R
        badge_bg = "#FFFFFF" if not is_manual else "#FFD700"
        circle = plt.Circle((badge_x, badge_y_center), BADGE_R, color=badge_bg, zorder=4)
        ax.add_patch(circle)
        ax.text(
            badge_x, badge_y_center,
            f"{nid}",
            ha="center", va="center",
            fontsize=FONT_TITLE - 1, fontweight="bold", color=title_bg_color,
            zorder=5,
        )

        # ========== 标题行文字（条件文本，含多行）==========
        # 条件文本区域：从徽章右边到节点右边
        cond_text_left = x - node_w / 2 + TITLE_PAD_LEFT
        cond_text_right = x + node_w / 2 - TITLE_PAD_RIGHT
        cond_text_center_x = (cond_text_left + cond_text_right) / 2
        
        if is_leaf:
            title_lines = ["叶子节点"]
        else:
            title_lines = cond_lines[idx]

        if len(title_lines) == 1:
            # 单行：居中
            ax.text(
                cond_text_center_x, badge_y_center,
                _tex_label(title_lines[0]),
                ha="center", va="center",
                fontsize=FONT_TITLE, fontweight="bold", color="#FFFFFF",
                zorder=5,
            )
        else:
            # 多行：从上到下排列
            line_h = TITLE_H
            top_y = title_bar_y_top - line_h / 2
            for li, line_text in enumerate(title_lines):
                line_y = top_y - li * line_h
                ax.text(
                    cond_text_center_x, line_y,
                    _tex_label(line_text),
                    ha="center", va="center",
                    fontsize=FONT_TITLE, fontweight="bold", color="#FFFFFF",
                    zorder=5,
                )

        # ========== 内容行（居中对齐） ==========
        content_top = title_bar_y_bottom
        content_bottom = y - total_node_h / 2 + 0.08
        row_step = (content_top - content_bottom) / CONTENT_LINES

        rows = [
            (x, content_top - row_step * 0.5, f"GINI: {node['gini']:.4f}"),
            (x, content_top - row_step * 1.5, f"样本总数: {node['n_samples']}"),
            (x, content_top - row_step * 2.5, f"样本占比: {node['sample_pct']:.2%}"),
            (x, content_top - row_step * 3.5, f"坏样本率: {node['bad_rate']:.2%}"),
            (x, content_top - row_step * 4.5, f"LIFT指标: {node['lift']:.2f}"),
        ]

        for rx, ry, text in rows:
            ax.text(
                rx, ry, text,
                ha="center", va="center",
                fontsize=FONT_BODY, color="#1D2129",
                zorder=4,
            )

    # ============================================================
    # 按内容真实尺寸设置画布大小（而非沿用传入的 figsize）
    # ============================================================
    # 整个布局都是按照「1 个数据坐标单位 = 1 inch」计算节点宽高/padding/徽章半径的。
    # 如果直接用传入的 figsize 配合 ax.set_aspect("equal")，
    # 一旦树的实际数据宽高比与 figsize 的宽高比不一致，matplotlib 会整体缩放
    # 数据坐标系以适配画布（即 1 单位 != 1 inch 了）；但文字字号是物理绝对大小，
    # 不会跟着缩放，于是节点框（随之缩小）就可能比文字还窄，导致文字溢出/与徽章重叠。
    # 因此这里改为：以内容的真实数据范围反推画布尺寸，强制保证 1 单位 = 1 inch。
    x_pad = uniform_node_w * 1.5
    y_pad = 2.0
    x_min = min(coords[nid][0] - uniform_node_w / 2 for nid in coords) - x_pad
    x_max = max(coords[nid][0] + uniform_node_w / 2 for nid in coords) + x_pad
    y_min = min(coords[nid][1] - node_id_to_total_h[nid] / 2 for nid in coords) - y_pad
    y_max = max(coords[nid][1] + node_id_to_total_h[nid] / 2 for nid in coords) + y_pad

    data_w = x_max - x_min
    data_h = y_max - y_min

    # 标题预留高度（图形坐标，不占用数据区域，因此不影响 1 单位 = 1 inch 的换算）
    title_margin = 0.7 if title else 0.0
    fig.set_size_inches(data_w, data_h + title_margin)

    # 主坐标区铺满整张画布的数据区域部分（顶部留给标题），从而让坐标轴的物理尺寸
    # 与 (data_w, data_h) 精确一致，set_aspect("equal") 不再需要做任何缩放
    axes_h_frac = data_h / (data_h + title_margin)
    ax.set_position([0.0, 0.0, 1.0, axes_h_frac])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")
    ax.set_aspect("equal")

    # 标题：用 figure 级别坐标绘制，与数据坐标轴的缩放彼此独立
    if title:
        fig.text(
            0.5, axes_h_frac + (1 - axes_h_frac) * 0.45,
            title, ha="center", va="center",
            fontsize=18, fontweight="bold", color=_COLOR_TEXT_DARK,
        )

    if save:
        save_dir = os.path.dirname(save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


# ============================================================================
# pyecharts 渲染器（交互式 HTML）
# ============================================================================


def plot_tree_pyecharts(
    tree_obj: Any,
    title: str = "",
    width: str = "1400px",
    height: str = "900px",
    save: Optional[str] = None,
    page_title: str = "决策树可视化",
    feature_names: Optional[List[str]] = None,
) -> Any:
    """使用 pyecharts 绘制 AntV G6 风格的交互式决策树。

    **交互功能**：
    - 鼠标悬停 tooltip 显示节点详细信息
    - 支持缩放和平移
    - 可导出为 HTML

    **参数**

    :param tree_obj: ManualTreeExtractor 或 sklearn DecisionTreeClassifier
    :param title: 图表标题
    :param width: 画布宽度（CSS 格式，如 '1400px'）
    :param height: 画布高度
    :param save: 保存路径（如 'tree.html'），如传入路径中有文件夹不存在，会自动创建，默认 None
    :param page_title: HTML 页面标题
    :return: pyecharts Graph 对象

    **参考样例**

    >>> chart = plot_tree_pyecharts(ext)
    >>> chart.render('tree.html')
    >>> chart.render_notebook()  # 在 Jupyter 中直接显示
    """
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Graph, Page
    except ImportError:
        raise ImportError(
            "需要安装 pyecharts: pip install pyecharts\n"
            "pyecharts 用于生成交互式 HTML 决策树图"
        )

    # 提取树数据（feature_names 仅对 sklearn 树有效，ManualTreeExtractor 自行读取）
    tree_data = _extract_tree_data(tree_obj, feature_names=feature_names)
    nodes = tree_data["nodes"]
    edges = tree_data["edges"]

    # 动态色阶：从实际节点坏账率区间生成
    all_br = [n["bad_rate"] for n in nodes]
    min_br = min(all_br)
    max_br = max(all_br)
    gradient_stops = _build_gradient_stops(min_br, max_br)
    node_fill_colors: Dict[int, str] = {
        n["node_id"]: _compute_fill_color(n["bad_rate"], gradient_stops) for n in nodes
    }

    if not nodes:
        from pyecharts.charts import Bar
        bar = Bar()
        bar.set_global_opts(title_opts=opts.TitleOpts(title="树为空或未拟合"))
        return bar

    # Reingold-Tilford 布局（根节点 y=0，子节点 y<0）
    # pyecharts y轴向上，negate y 使根节点位于顶部
    coords = _reingold_tilford_layout(nodes, edges)

    # 构建 pyecharts 节点
    graph_nodes = []
    for node in nodes:
        nid = node["node_id"]
        x, y = coords.get(nid, (0, 0))
        # negate y：根节点(0) → 顶部，子节点(负) → 底部
        x_float, y_float = float(x), float(-y)

        is_leaf = node["is_leaf"]
        is_manual = node["is_manual"]
        fill = node_fill_colors[nid]
        bad_rate = node["bad_rate"]
        n_samples = node["n_samples"]
        good = node["good_count"]
        bad = node["bad_count"]

        # AntV 风格颜色
        if is_manual:
            border_color = _COLOR_SECONDARY   # #F76E6C
        elif is_leaf:
            if bad_rate < 0.1:
                border_color = "#52C41A"
            elif bad_rate < 0.3:
                border_color = "#FE7715"
            else:
                border_color = "#FF4D4F"
        else:
            border_color = "#165DFF"

        # 节点标题
        title_text = node["title"]
        if is_leaf and node["class_label"]:
            title_text += f" [{node['class_label']}]"

        # tooltip 内容（AntV 风格）
        _tip_color = "#FF4D4F" if bad_rate > 0.3 else ("#FE7715" if bad_rate > 0.1 else "#52C41A")
        _gini_line = f"<b>Gini:</b> {node['gini']:.4f}<br/>" if not is_leaf else ""
        _manual_line = "<span style='color:#F76E6C'>★ 人工分裂节点</span>" if is_manual else ""
        tooltip = (
            f"<div style='font-family:Arial,sans-serif;font-size:12px;'>"
            f"<b style='color:#1D2129'>" + title_text + "</b><br/>"
            f"<hr style='margin:4px 0'/>"
            f"<b>条件:</b> " + node["condition"] + "<br/>"
            f"<b>样本总数:</b> " + f"{n_samples:,}" + " (" + f"{node['sample_pct']:.1%}" + ")<br/>"
            f"<b>好样本数:</b> " + f"{good:,}" + "<br/>"
            f"<b>坏样本数:</b> " + f"{bad:,}" + "<br/>"
            f"<b>坏样本率:</b> <span style='color:" + _tip_color + ";font-weight:bold'>" + f"{bad_rate:.2%}" + "</span><br/>"
            + _gini_line
            + _manual_line
            + "</div>"
        )

        # 节点大小（叶子节点稍大）
        node_size = 60 if is_leaf else 50

        graph_nodes.append(
            opts.GraphNode(
                name=str(nid),
                x=x_float,
                y=y_float,
                symbol_size=node_size,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=fill,
                    border_color=border_color,
                    border_width=2.5 if is_manual else 1.5,
                ),
                label_opts=opts.LabelOpts(
                    formatter=(
                        "{{b}}\n"
                        "{" + (node["condition"] if node["condition"] else "叶子") + "|\n}\n"
                        "GINI:" + f"{node['gini']:.4f}\n"
                        "样本总数:" + str(n_samples) + "\n"
                        "样本占比:" + f"{node['sample_pct']:.2%}\n"
                        "坏样本率:" + f"{bad_rate:.2%}\n"
                        "LIFT指标:" + f"{node['lift']:.2f}"
                    ),
                    font_size=7,
                    color="#1D2129",
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger_on="mousemove",
                    background_color="#FFFFFF",
                    border_color="#E8ECFF",
                    border_width=1,
                    textstyle_opts=opts.TextStyleOpts(color="#1D2129"),
                    formatter=tooltip,
                ),
            )
        )

    # 构建 pyecharts 边（贝塞尔曲线 + 主题色 + 所有边显示分支标签）
    graph_edges = []
    ROOT_ID = "0"
    for edge in edges:
        label_text = edge["label"]
        edge_color = "#165DFF" if label_text == "<=" else "#F53F3B"
        is_root_edge = str(edge["source"]) == ROOT_ID

        graph_edges.append(
            opts.GraphLink(
                source=str(edge["source"]),
                target=str(edge["target"]),
                linestyle_opts=opts.LineStyleOpts(
                    color=edge_color,
                    width=2.5,
                    opacity=0.85,
                    curve=0.0,  # 直线连接
                ),
                label_opts=opts.LabelOpts(
                    formatter=edge["label"],
                    font_size=9 if is_root_edge else 8,
                    font_weight="bold",
                    color=edge_color,
                    background_color=("#E8F0FF" if label_text == "<=" else "#FFF1F0"),
                    border_color=edge_color,
                    border_width=1.0,
                    border_radius=3,
                    padding=2,
                    position="middle",
                    is_show=True,
                ),
            )
        )

    # 构建 Graph（标题可有可无）
    base_opts = opts.InitOpts(
        width=width, height=height, page_title=page_title, renderer="canvas",
    )
    if title:
        graph = Graph(base_opts)
        graph.add(
            series_name="决策树",
            nodes=graph_nodes,
            links=graph_edges,
            layout="none",
            is_roam=True,
            edge_symbol=["circle", "arrow"],
            edge_symbol_size=6,
        )
        graph.set_colors(["#165DFF", "#F53F3B", "#36CBCB", "#F53F3B"])
        graph.set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                subtitle="AntV G6 风格 · 卡片式决策树可视化",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold", color="#1D2129"),
                subtitle_textstyle_opts=opts.TextStyleOpts(font_size=11, color="#86909C"),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger_on="mousemove", background_color="#FFFFFF", border_color="#E8ECFF",
                textstyle_opts=opts.TextStyleOpts(color="#1D2129"),
            ),
            legend_opts=opts.LegendOpts(
                is_show=True, pos_left="right", pos_top="top", orient="vertical",
                textstyle_opts=opts.TextStyleOpts(color="#4B5563", font_size=10),
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True, pos_left="right", pos_bottom="bottom",
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        is_show=True, type_="png", name="决策树", pixel_ratio=2,
                    ),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=True),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=True),
                ),
            ),
            xaxis_opts=opts.AxisOpts(is_show=False),
            yaxis_opts=opts.AxisOpts(is_show=False),
        )
    else:
        graph = Graph(base_opts)
        graph.add(
            series_name="决策树",
            nodes=graph_nodes,
            links=graph_edges,
            layout="none",
            is_roam=True,
            edge_symbol=["circle", "arrow"],
            edge_symbol_size=6,
        )
        graph.set_colors(["#165DFF", "#F53F3B", "#36CBCB", "#F53F3B"])
        graph.set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger_on="mousemove", background_color="#FFFFFF", border_color="#E8ECFF",
                textstyle_opts=opts.TextStyleOpts(color="#1D2129"),
            ),
            legend_opts=opts.LegendOpts(
                is_show=True, pos_left="right", pos_top="top", orient="vertical",
                textstyle_opts=opts.TextStyleOpts(color="#4B5563", font_size=10),
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True, pos_left="right", pos_bottom="bottom",
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        is_show=True, type_="png", name="决策树", pixel_ratio=2,
                    ),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=True),
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=True),
                ),
            ),
            xaxis_opts=opts.AxisOpts(is_show=False),
            yaxis_opts=opts.AxisOpts(is_show=False),
        )
    chart = graph

    if save:
        save_dir = os.path.dirname(save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        chart.render(save)

    return chart


# ============================================================================
# graphviz 渲染器（高质量矢量图）
# ============================================================================


def plot_tree_graphviz(
    tree_obj: Any,
    save: Optional[str] = None,
    format: str = "png",
    title: str = "",
    feature_names: Optional[List[str]] = None,
) -> Any:
    """使用 graphviz 绘制 AntV G6 风格的高质量决策树。

    **特点**：
    - 高质量矢量图（SVG/PDF/PNG）
    - 支持中文
    - 适合嵌入报告

    **参数**

    :param tree_obj: ManualTreeExtractor 或 sklearn DecisionTreeClassifier
    :param save: 保存路径（如 '/tmp/tree.png'），如传入路径中有文件夹不存在，会自动创建，默认 None
    :param format: 渲染格式，默认 'png'。可选 'pdf', 'svg', 'dot' 等
    :param title: 图表标题
    :param feature_names: 特征名列表（sklearn clf 推荐传入）
    :return: graphviz.Source 对象

    **参考样例**

    >>> src = plot_tree_graphviz(ext, save='/tmp/tree', format='pdf')
    >>> src.render('/tmp/tree', cleanup=True)
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("需要安装 graphviz: pip install graphviz")

    # 提取树数据
    tree_data = _extract_tree_data(tree_obj, feature_names)
    nodes = tree_data["nodes"]
    edges = tree_data["edges"]

    if not nodes:
        dot = graphviz.Digraph(comment=title)
        dot.node("empty", "树为空或未拟合", shape="box")
        return dot

    # 动态色阶 + 节点颜色
    all_br = [n["bad_rate"] for n in nodes]
    min_br = min(all_br)
    max_br = max(all_br)
    gradient_stops = _build_gradient_stops(min_br, max_br)
    node_fill_colors: Dict[int, str] = {
        n["node_id"]: _compute_fill_color(n["bad_rate"], gradient_stops) for n in nodes
    }

    # graphviz 使用自动树布局
    _reingold_tilford_layout(nodes, edges)

    # 构建 DOT 图
    dot_lines: List[str] = []
    dot_lines.append('digraph Tree {')
    dot_lines.append(f'    // {title}')
    dot_lines.append('    graph [ranksep=0.6, nodesep=0.35, splines=line, bgcolor="#FAFBFF", pad=0.5, dpi=150, concentrate=false];')
    dot_lines.append('    node [shape=box, fontname=helvetica, margin="0.15,0.1", style="filled,rounded", width=0, height=0];')
    dot_lines.append('    edge [fontname=helvetica, arrowsize=0, penwidth=1.5, arrowhead=none];')

    for node in nodes:
        nid = node["node_id"]
        fill_color = node_fill_colors[nid]
        border_color = "#165DFF" if not node["is_leaf"] else _get_border_for_leaf(node["bad_rate"])

        # AntV 卡片风格节点内容：统一字体大小，全部居中
        # 行1: node #0
        # 行2: 切分条件
        # 行3~7: 指标（Gini / 样本 / 占比 / 坏账率 / LIFT）
        # 切分条件含 "<=" 等比较符号，label=<...> 是 graphviz 的 HTML-like 标签语法，
        # "<" 会被当成标签起始符解析，必须转义动态内容，否则会产生 DOT 语法错误
        condition_html = html.escape(node["condition"])
        label_html = (
            f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2" WIDTH="140" HEIGHT="120">'
            # 行1: 节点编号
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="18">'
            f'<FONT POINT-SIZE="9" FACE="SimHei,Microsoft YaHei,Arial" COLOR="#165DFF"><B>NODE #{nid}</B></FONT>'
            f'</TD></TR>'
            # 行2: 切分条件
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="18">'
            f'<FONT POINT-SIZE="8" FACE="SimHei,Microsoft YaHei,Arial" COLOR="#1D2129">{condition_html}</FONT>'
            f'</TD></TR>'
            # 分隔线
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="4">'
            f'<FONT POINT-SIZE="4"><BR/></FONT>'
            f'</TD></TR>'
            # 行3: Gini
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="14">'
            f'<FONT POINT-SIZE="8" FACE="Arial" COLOR="#4B5563">GINI: {node["gini"]:.4f}</FONT>'
            f'</TD></TR>'
            # 行4: 样本数
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="14">'
            f'<FONT POINT-SIZE="8" FACE="Arial" COLOR="#4B5563">样本总数: {node["n_samples"]:,}</FONT>'
            f'</TD></TR>'
            # 行5: 占比
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="14">'
            f'<FONT POINT-SIZE="8" FACE="Arial" COLOR="#4B5563">样本占比: {node["sample_pct"]:.2%}</FONT>'
            f'</TD></TR>'
            # 行6: 坏账率
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="14">'
            f'<FONT POINT-SIZE="8" FACE="Arial" COLOR="#4B5563">坏样本率: {node["bad_rate"]:.2%}</FONT>'
            f'</TD></TR>'
            # 行7: LIFT
            f'<TR><TD ALIGN="CENTER" VALIGN="MIDDLE" HEIGHT="14">'
            f'<FONT POINT-SIZE="8" FACE="Arial" COLOR="#4B5563">LIFT指标: {node["lift"]:.2f}</FONT>'
            f'</TD></TR>'
            f'</TABLE>'
        )

        dot_lines.append(
            f'    {nid} [label=<{label_html}>, '
            f'fillcolor="{fill_color}", color="{border_color}", '
            f'penwidth=1.5, tooltip="{node["title"]} | {node["condition"]}"] ;'
        )

    # 添加边（直线 + 无箭头 + 所有边显示分支标签）
    ROOT_ID = 0
    for edge in edges:
        label_text = edge["label"]
        is_root_edge = edge["source"] == ROOT_ID
        edge_color = "#165DFF" if label_text == "<=" else "#F53F3B"
        font_size = "10" if is_root_edge else "8"

        dot_lines.append(
            f'    {edge["source"]} -> {edge["target"]} '
            f'[label="  {label_text}  ", fontcolor="{edge_color}", '
            f'color="{edge_color}", penwidth=1.5, style=solid, fontsize={font_size}] ;'
        )

    dot_lines.append("}")

    dot_src = "\n".join(dot_lines)
    src = graphviz.Source(dot_src, engine="dot")

    if save:
        save_dir = os.path.dirname(save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # graphviz 的 render() 会自动在文件名后追加 ".{format}"；
        # 若 save 已经带有该后缀（如 'tree.png'，与文档约定一致），需先去掉，
        # 否则会生成 "tree.png.png" 这种重复后缀的文件
        save_base = save
        suffix = f".{format}"
        if save_base.lower().endswith(suffix.lower()):
            save_base = save_base[: -len(suffix)]
        src.render(save_base, format=format, cleanup=True)

    return src


def _get_border_for_leaf(bad_rate: float) -> str:
    """根据叶子节点坏账率返回边框颜色。"""
    if bad_rate < 0.1:
        return "#52C41A"
    elif bad_rate < 0.3:
        return "#FE7715"
    else:
        return "#FF4D4F"


# ============================================================================
# 统一 API：DecisionTreeViz
# ============================================================================


class DecisionTreeViz:
    """AntV G6 风格决策树可视化器。

    支持 matplotlib / pyecharts / graphviz 三种渲染后端，
    统一 API 设计，按需切换。

    **参数**

    :param backend: 渲染后端，可选 'matplotlib' | 'pyecharts' | 'graphviz'
        - 'matplotlib': 纯 Python，无需额外依赖，适合快速预览
        - 'pyecharts': 交互式 HTML，支持 tooltip、缩放
        - 'graphviz': 高质量矢量图，适合报告嵌入
    :param feature_names: 特征名列表（当 tree_obj 为 sklearn clf 时需要）
    :param title: 图表标题
    :param figsize: matplotlib 画布大小
    :param dpi: matplotlib 分辨率

    **参考样例**

    >>> # matplotlib 快速预览
    >>> viz = DecisionTreeViz(backend='matplotlib')
    >>> fig = viz.plot(ext, save='tree.png')
    >>> plt.show()

    >>> # pyecharts 交互式
    >>> viz = DecisionTreeViz(backend='pyecharts')
    >>> chart = viz.plot(ext)
    >>> chart.render('tree.html')

    >>> # graphviz 高质量
    >>> viz = DecisionTreeViz(backend='graphviz')
    >>> src = viz.plot(ext, save='tree', format='pdf')
    """

    SUPPORTED_BACKENDS = ["matplotlib", "pyecharts", "graphviz"]

    def __init__(
        self,
        backend: str = "matplotlib",
        feature_names: Optional[List[str]] = None,
        title: str = "",
        figsize: Tuple[float, float] = (18, 12),
        dpi: int = 240,
        **kwargs,
    ):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"不支持的后端 '{backend}'，可选: {self.SUPPORTED_BACKENDS}"
            )
        self.backend = backend
        self.feature_names = feature_names
        self.title = title
        self.figsize = figsize
        self.dpi = dpi
        # 透传其他参数
        self._kwargs = kwargs
        self._last_tree_obj = None
        self._last_result = None

    def plot(
        self,
        tree_obj: Any,
        save: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """绘制决策树。

        :param tree_obj: ManualTreeExtractor 或 sklearn DecisionTreeClassifier
        :param save: 保存路径
        :param title: 图表标题（覆盖构造时的 title）
        :return: 渲染结果（matplotlib Figure / pyecharts Chart / graphviz Source）
        """
        self._last_tree_obj = tree_obj
        kw = {**self._kwargs, **kwargs}
        _title = title if title is not None else self.title

        if self.backend == "matplotlib":
            result = plot_tree_matplotlib(
                tree_obj,
                figsize=kw.pop("figsize", self.figsize),
                dpi=kw.pop("dpi", self.dpi),
                title=kw.pop("title", _title),
                save=save,
                show_stats=kw.pop("show_stats", True),
                show_gini=kw.pop("show_gini", True),
                feature_names=kw.pop("feature_names", self.feature_names),
            )
            self._last_result = result
            return result

        elif self.backend == "pyecharts":
            result = plot_tree_pyecharts(
                tree_obj,
                title=kw.pop("title", _title),
                save=save,
                page_title=kw.pop("page_title", _title),
                width=kw.pop("width", "1400px"),
                height=kw.pop("height", "900px"),
                feature_names=kw.pop("feature_names", self.feature_names),
            )
            self._last_result = result
            return result

        elif self.backend == "graphviz":
            result = plot_tree_graphviz(
                tree_obj,
                save=save,
                format=kw.pop("format", "png"),
                title=kw.pop("title", _title),
                feature_names=kw.pop("feature_names", self.feature_names),
            )
            self._last_result = result
            return result

    def render(self, path: str) -> Any:
        """保存/渲染当前树图到文件。

        :param path: 保存路径
        :return: 渲染结果
        """
        return self.plot(self._last_tree_obj, save=path)



# ============================================================================
# 便捷函数
# ============================================================================


def _extract_tree_data(tree_obj: Any, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """统一提取接口：支持 ManualTreeExtractor / sklearn clf / AutoTreeFitter。

    :param tree_obj: 树对象
    :param feature_names: 特征名列表（sklearn clf 推荐传入，避免 x[0] 占位符）
    """
    # ManualTreeExtractor 或 AutoTreeFitter（两者都有 _tree_info）
    if hasattr(tree_obj, "_tree_info"):
        return _extract_tree_from_mte(tree_obj)
    # sklearn DecisionTreeClassifier
    elif hasattr(tree_obj, "tree_"):
        # 优先用传入的 feature_names，否则用 clf 的 feature_names_in_
        fn = feature_names if feature_names is not None else getattr(tree_obj, "feature_names_in_", None)
        return _extract_tree_from_sklearn(tree_obj, fn)
    else:
        raise TypeError(
            f"不支持的树对象类型: {type(tree_obj).__name__}\n"
            "请传入 ManualTreeExtractor / sklearn DecisionTreeClassifier / AutoTreeFitter"
        )


def plot_tree(
    tree_obj: Any,
    backend: str = "matplotlib",
    save: Optional[str] = None,
    **kwargs,
) -> Any:
    """便捷函数：一行命令绘制决策树。

    **参数**

    :param tree_obj: ManualTreeExtractor 或 sklearn DecisionTreeClassifier
    :param backend: 渲染后端，默认 'matplotlib'
    :param save: 保存路径
    :param kwargs: 传给 DecisionTreeViz 的参数
    :return: 渲染结果

    **参考样例**

    >>> # matplotlib
    >>> fig = plot_tree(ext, backend='matplotlib', save='tree.png')

    >>> # pyecharts
    >>> chart = plot_tree(ext, backend='pyecharts', save='tree.html')

    >>> # graphviz
    >>> src = plot_tree(ext, backend='graphviz', save='tree', format='pdf')
    """
    viz = DecisionTreeViz(backend=backend, **kwargs)
    return viz.plot(tree_obj, save=save)
