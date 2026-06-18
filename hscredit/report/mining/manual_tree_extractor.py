"""人工决策树提取器.

提供决策树规则挖掘的核心工具集，支持：
- **DecisionTreeAnalyzer**：标准 sklearn 决策树训练、结构导出、规则提取与指标评估
- **ManualTreeExtractor**：人工干预决策树节点分裂（业务经验注入模型）

**参考样例**

>>> # DecisionTreeAnalyzer：训练决策树并评估
>>> from hscredit.report.mining import DecisionTreeAnalyzer
>>> analyzer = DecisionTreeAnalyzer(target='target', feature_list=['age', 'income'])
>>> analyzer.fit(df_train)
>>> metrics = analyzer.evaluate(df_test_list=[('测试', df_test)], metric_type='ks')
>>> print(metrics)

>>> # ManualTreeExtractor：人工分裂
>>> from hscredit.report.mining import ManualTreeExtractor
>>> ext = ManualTreeExtractor(target='target')
>>> ext.fit(df, feature_list=['age', 'income'])
>>> ext.manual_split(df, feature_name='age', threshold=35)
>>> print(ext.get_rule_table())
"""

import copy
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from ...core.rules.rule import Rule
from ...exceptions import InputValidationError
from ...utils.pandas_extensions import style_rule_table

# ============================================================================
# 指标计算 — 优先使用 hscredit.core.metrics 中的统一实现
# ============================================================================

try:
    from ...core.metrics import ks as _ks
    from ...core.metrics import auc as _auc
    from ...core.metrics import badrate as _badrate
except ImportError:
    # 降级：内联最小实现（仅在 metrics 未注册时使用）
    def _ks(y_true, y_prob):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float((tpr - fpr).max())

    def _auc(y_true, y_prob):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_prob))

    def _badrate(y_true, mask):
        if mask.sum() == 0:
            return 0.0
        return float(y_true[mask].mean())


def _lift_local(y_true, y_score, n_bins=10):
    """LIFT 计算：取 top-n% 中的坏样本率相对总体坏样本率的倍数。

    :param y_true: 真实标签数组
    :param y_score: 预测分数（分数越高越"坏"）
    :param n_bins: 取最高分样本的比例分母，默认 10 即 top 10%
    :return: LIFT 值
    """
    df = pd.DataFrame({'y': y_true, 's': y_score}).sort_values('s', ascending=False)
    top_n = max(1, int(len(df) * n_bins / 100))
    top_bad_rate = df.head(top_n)['y'].mean()
    overall_bad_rate = df['y'].mean()
    return float(top_bad_rate / overall_bad_rate) if overall_bad_rate > 0 else 0.0


def _lift_table_local(y_true, y_score, n_bins=10):
    """LIFT 表格：按分数分箱计算各箱的坏账率和 LIFT 值。

    :param y_true: 真实标签数组
    :param y_score: 预测分数
    :param n_bins: 分箱数，默认 10
    :return: 含 LIFT 值的 DataFrame
    """
    df = pd.DataFrame({'y': y_true, 's': y_score})
    df['bin'] = pd.qcut(df['s'], n_bins, labels=False, duplicates='drop')
    result = df.groupby('bin').agg(y=('y', 'mean'), count=('y', 'count'))
    result['lift'] = result['y'] / df['y'].mean()
    return result.reset_index()


# ============================================================================
# 决策树工具函数
# ============================================================================

def _rule_generator(clf, feature_name_list: List[str]) -> pd.DataFrame:
    """从训练好的决策树（或模拟树对象）提取规则 DataFrame。

    遍历每个非根节点，根据其父节点路径构建分裂条件。

    :param clf: 已训练的 sklearn 决策树分类器，或包含 tree_ 属性的模拟对象
    :param feature_name_list: 特征名列表
    :return: 规则 DataFrame，含列：
        node / if_leaf / rule_list / node_path / node_samples / node_value / impurity
    """
    children_left = list(clf.tree_.children_left)
    children_right = list(clf.tree_.children_right)
    feature = list(clf.tree_.feature)
    threshold = list(clf.tree_.threshold)
    node_samples = list(clf.tree_.n_node_samples)
    node_values = list(clf.tree_.value)
    node_impurity = list(clf.tree_.impurity)

    def _find_father_path(node: int, father_path: List[str] = None) -> List[str]:
        """递归查找从根节点到目标节点的路径描述。"""
        if father_path is None:
            father_path = []
        if node in children_left:
            father_node = children_left.index(node)
            node_path = f"{father_node},<="
        elif node in children_right:
            father_node = children_right.index(node)
            node_path = f"{father_node},>"
        else:
            father_node = 0
            node_path = "None"
        path = copy.copy(father_path)
        path.append(node_path)
        if father_node > 0:
            return _find_father_path(node=father_node, father_path=path)
        return path

    def _father_path_to_rule(father_path: List[str]) -> Tuple[List[str], List]:
        """将路径描述转换为特征分裂规则列表。

        对同一特征的多个条件取交集（合并 max/min 阈值）。
        返回 (路径ID列表, 规则列表[(特征名, 操作符, 阈值), ...])
        """
        rule_list = []
        for node_tmp in father_path:
            node = int(node_tmp.split(",")[0])
            operator = node_tmp.split(",")[1]
            rule_list.append([feature_name_list[feature[node]], operator, threshold[node]])

        # 按特征聚合：同一特征的多个条件取交集
        rule_df = pd.DataFrame(rule_list, columns=["feature_name", "operator", "threshold"])
        grouped = rule_df.groupby(["feature_name", "operator"], observed=True).agg(
            {"threshold": ["max", "min"]}
        )
        final_rule = []
        for idx in grouped.index:
            thres = (
                grouped.loc[idx, ("threshold", "min")]
                if idx[1] == "<="
                else grouped.loc[idx, ("threshold", "max")]
            )
            final_rule.append(list(idx) + [thres])
        return father_path, final_rule

    result = {
        "node": [],
        "if_leaf": [],
        "rule_list": [],
        "node_path": [],
        "node_samples": [],
        "node_value": [],
        "impurity": [],
    }

    for i in range(1, len(feature)):
        result["node"].append(i)
        result["if_leaf"].append(True if feature[i] == -2 else False)
        father_path = _find_father_path(i)
        rule_path, final_rule = _father_path_to_rule(father_path)
        result["rule_list"].append(final_rule)
        result["node_path"].append(rule_path)
        result["node_samples"].append(node_samples[i])

        # node_value 格式：(n_samples, n_classes) 的比例值
        # 兼容 numpy array（sklearn 原始）和 Python list（手动构造）
        n = node_samples[i]
        raw_val = node_values[i]
        if hasattr(raw_val[0], "tolist"):
            vals = [round(v * n) for v in raw_val[0].tolist()]
        else:
            vals = [round(v * n) for v in raw_val[0]]
        result["node_value"].append(vals)
        result["impurity"].append(node_impurity[i])

    return pd.DataFrame(result)


def _export_dot_data(
    clf,
    feature_list: List[str],
    class_names: Optional[List[str]] = None,
    out_file: Optional[str] = None,
    max_depth: Optional[int] = None,
    filled: bool = True,
    node_ids: bool = True,
    proportion: bool = True,
    precision: int = 3,
) -> str:
    """导出决策树为 DOT 格式字符串。

    :param clf: 决策树分类器
    :param feature_list: 特征名列表
    :param class_names: 类别名列表
    :param out_file: 输出 .dot 文件路径（可选）
    :param max_depth: 最大显示深度
    :param filled: 是否填充颜色
    :param node_ids: 是否显示节点 ID
    :param proportion: 是否显示样本比例
    :param precision: 数值精度
    :return: DOT 格式字符串
    """
    dot_data = export_graphviz(
        decision_tree=clf,
        feature_names=feature_list,
        class_names=class_names,
        out_file=out_file,
        max_depth=max_depth,
        label="all",
        filled=filled,
        leaves_parallel=True,
        impurity=True,
        node_ids=node_ids,
        proportion=proportion,
        rotate=False,
        rounded=True,
        special_characters=True,
        precision=precision,
    )
    if out_file is not None:
        with open(out_file, "r") as f:
            return f.read()
    return dot_data


# ============================================================================
# 树结构节点操作
# ============================================================================

def _add_nodes_to_tree(
    node: int,
    split_list_left: List[int],
    split_list_right: List[int],
    feature: List[int],
    threshold: List[float],
    node_samples: List[int],
    node_values: List,
    node_impurity: List[float],
    split_list_left_new: List[int],
    split_list_right_new: List[int],
    feature_new: List[int],
    threshold_new: List[float],
    node_samples_new: List[int],
    node_values_new: List,
    node_impurity_new: List[float],
) -> Tuple[List, List, List, List, List, List, List]:
    """向现有树结构的指定节点插入一棵子树。

    用于在决策树中指定节点处插入新的分裂分支。
    新子树的节点 ID 会自动偏移以避免与原树冲突。

    :param node: 目标节点 ID（插入位置）
    :param split_list_left: 原树左子节点列表
    :param split_list_right: 原树右子节点列表
    :param feature: 原树分裂特征列表
    :param threshold: 原树分裂阈值列表
    :param node_samples: 原树节点样本数列表
    :param node_values: 原树节点值列表
    :param node_impurity: 原树节点不纯度列表
    :param split_list_left_new: 新子树左子节点列表
    :param split_list_right_new: 新子树右子节点列表
    :param feature_new: 新子树分裂特征列表
    :param threshold_new: 新子树分裂阈值列表
    :param node_samples_new: 新子树节点样本数列表
    :param node_values_new: 新子树节点值列表
    :param node_impurity_new: 新子树节点不纯度列表
    :return: 更新后的树结构元组
    """
    split_list_left_new = list(split_list_left_new)
    split_list_right_new = list(split_list_right_new)
    feature_new = list(feature_new)
    threshold_new = list(threshold_new)
    node_samples_new = list(node_samples_new)
    node_values_new = list(node_values_new)
    node_impurity_new = list(node_impurity_new)

    if node == 0:
        return (
            split_list_left_new,
            split_list_right_new,
            feature_new,
            threshold_new,
            node_samples_new,
            node_values_new,
            node_impurity_new,
        )

    # 为避免节点 ID 冲突，将新子树节点 ID 偏移
    add_n = len(feature) - 1
    split_list_left_new = [i + add_n if i != -1 else i for i in split_list_left_new]
    split_list_right_new = [i + add_n if i != -1 else i for i in split_list_right_new]

    # 替换目标节点
    split_list_left[node] = split_list_left_new[0]
    split_list_right[node] = split_list_right_new[0]
    feature[node] = feature_new[0]
    threshold[node] = threshold_new[0]
    node_samples[node] = node_samples_new[0]
    node_values[node] = node_values_new[0]
    node_impurity[node] = node_impurity_new[0]

    # 追加新子树剩余节点
    split_list_left += split_list_left_new[1:]
    split_list_right += split_list_right_new[1:]
    feature += feature_new[1:]
    threshold += threshold_new[1:]
    node_samples += node_samples_new[1:]
    node_values += node_values_new[1:]
    node_impurity += node_impurity_new[1:]

    return (
        split_list_left,
        split_list_right,
        feature,
        threshold,
        node_samples,
        node_values,
        node_impurity,
    )


def _delete_nodes(
    node: int,
    split_list_left: List[int],
    split_list_right: List[int],
    feature: List[int],
    threshold: List[float],
    node_samples: List[int],
    node_values: List,
    node_impurity: List[float],
) -> Tuple[List, List, List, List, List, List, List]:
    """删除树中指定节点及其所有子节点，将该节点变为叶子节点。

    :param node: 待删除节点 ID
    :param split_list_left: 左子节点列表
    :param split_list_right: 右子节点列表
    :param feature: 分裂特征列表
    :param threshold: 分裂阈值列表
    :param node_samples: 节点样本数列表
    :param node_values: 节点值列表
    :param node_impurity: 节点不纯度列表
    :return: 更新后的树结构元组
    """
    snd = node

    def _del_children_iter(
        start_node: int, left_list: List[int], right_list: List[int]
    ) -> Tuple[List[int], List[int]]:
        """递归删除子树：将左右子节点指针置为 -1。"""
        left_list = list(left_list)
        right_list = list(right_list)

        next_node = left_list[start_node]
        if next_node != -1:
            if start_node == snd:
                left_list[start_node] = -1
            else:
                left_list[start_node] = 0
            left_list, right_list = _del_children_iter(next_node, left_list, right_list)
        else:
            if start_node == snd:
                left_list[start_node] = -1
            else:
                left_list[start_node] = 0

        next_node = right_list[start_node]
        if next_node != -1:
            if start_node == snd:
                right_list[start_node] = -1
            else:
                right_list[start_node] = 0
            left_list, right_list = _del_children_iter(next_node, left_list, right_list)
        else:
            if start_node == snd:
                right_list[start_node] = -1
            else:
                right_list[start_node] = 0

        return left_list, right_list

    def _remap_nodes(node_list: List[int]) -> Tuple[Dict[int, int], List[int]]:
        """重新编号节点，构建节点 ID 映射。

        返回 (ID映射字典, 被删除节点列表)。
        """
        remap: Dict[int, int] = {i: i for i in range(len(node_list))}
        remap[-1] = -1
        removed: List[int] = []
        for i, v in enumerate(node_list):
            if v == 0:
                removed.append(i)
                for j in remap:
                    if i < j:
                        remap[j] = remap[j] - 1
        return remap, removed

    split_list_left, split_list_right = _del_children_iter(node, split_list_left, split_list_right)
    node_map, _ = _remap_nodes(split_list_left)

    split_left_new: List[int] = []
    split_right_new: List[int] = []
    feat_new: List[int] = []
    thresh_new: List[float] = []
    n_samples_new: List[int] = []
    val_new: List = []
    impur_new: List[float] = []

    for a, b, c, d, f, g, h, m in zip(
        split_list_left,
        split_list_right,
        feature,
        threshold,
        node_samples,
        node_values,
        node_impurity,
        list(range(len(split_list_left))),
    ):
        if a == 0 and b == 0:
            continue
        elif a * b == 0:
            raise ValueError(f"树结构异常，请检查节点 {m} 的输入")
        split_left_new.append(node_map[a])
        split_right_new.append(node_map[b])
        if m == node:
            feat_new.append(-2)
            thresh_new.append(-2.0)
        else:
            feat_new.append(c)
            thresh_new.append(d)
        n_samples_new.append(f)
        val_new.append(g)
        impur_new.append(h)

    return (
        split_left_new,
        split_right_new,
        feat_new,
        thresh_new,
        n_samples_new,
        val_new,
        impur_new,
    )


# ============================================================================
# 辅助函数
# ============================================================================


def _find_subtree_node_ids(tree_info, root_node: int) -> List[int]:
    """返回以 root_node 为根的子树中所有节点 ID（包括根节点及所有后代）。

    :param tree_info: 树信息对象（_TreeInfo 或含 children_left/children_right 的对象）
    :param root_node: 子树根节点 ID
    :return: 节点 ID 列表
    """
    result = [root_node]
    stack = [root_node]
    while stack:
        node = stack.pop()
        left = tree_info.children_left[node]
        right = tree_info.children_right[node]
        for child in (left, right):
            if child != -1:
                result.append(child)
                stack.append(child)
    return result


def _node_hit_report(
    df_rules: pd.DataFrame,
    format_rule,
    data: pd.DataFrame,
    target: str,
    overdue: Optional[Union[str, List[str]]],
    dpds: Optional[Union[int, List[int]]],
    del_grey: bool,
    leaf_only: bool,
    **kwargs: Any,
) -> pd.DataFrame:
    """对单个数据集，汇总每个节点规则 :meth:`Rule.report` 中"命中"分箱的结果行。

    供 :class:`DecisionTreeAnalyzer` 和 :class:`ManualTreeExtractor` 的 ``report()`` 方法共用。

    :param df_rules: 规则 DataFrame（含 node / if_leaf / rule_list 列）
    :param format_rule: 将 rule_list 解析为 :class:`Rule` 对象的函数（空规则返回 None）
    :param data: 待评估数据集
    :param target: 目标变量列名
    :param overdue: 逾期天数字段名，参考 :meth:`Rule.report`
    :param dpds: 逾期定义方式，参考 :meth:`Rule.report`
    :param del_grey: 是否删除灰度样本
    :param leaf_only: 是否仅评估叶子节点
    :return: 各节点效果评估 DataFrame
    """
    rules_df = df_rules[df_rules["if_leaf"]] if leaf_only else df_rules

    hit_frames: List[pd.DataFrame] = []
    for _, row in rules_df.iterrows():
        node_id = int(row["node"])
        is_leaf = bool(row["if_leaf"])
        rule_list = row["rule_list"]

        rule = format_rule(rule_list)
        if rule is None:
            rule = Rule(expr="True", name="空规则", description="空规则")

        table = rule.report(
            data,
            target=target,
            overdue=overdue,
            dpds=dpds,
            del_grey=del_grey,
            desc=rule.description,
            **kwargs,
        )

        is_multi = isinstance(table.columns, pd.MultiIndex)
        group = "分箱详情" if is_multi else None
        bin_col = (group, "分箱") if is_multi else "分箱"
        node_col = (group, "节点编号") if is_multi else "节点编号"
        leaf_col = (group, "是否叶子") if is_multi else "是否叶子"

        hit = table[table[bin_col] == "命中"].copy()
        hit[node_col] = node_id
        hit[leaf_col] = "是" if is_leaf else "否"

        front_cols = [node_col, leaf_col]
        other_cols = [c for c in hit.columns if c not in front_cols]
        hit_frames.append(hit[front_cols + other_cols])

    if not hit_frames:
        return pd.DataFrame()

    result = pd.concat(hit_frames, ignore_index=True)
    sort_col = ("分箱详情", "节点编号") if isinstance(result.columns, pd.MultiIndex) else "节点编号"
    return result.sort_values(sort_col).reset_index(drop=True)


# ============================================================================
# 树信息内部类（封装 sklearn 树结构的原始数组）
# ============================================================================


class _TreeInfo:
    """封装决策树原始结构数据的内部类。

    模拟 sklearn 的 tree_ 属性，方便手动构建和操作树结构，
    同时兼容 sklearn 原生的 tree_ 数组格式。
    """

    def __init__(self, feature_list: List[str], n_classes: int):
        self.children_left: List[int] = []
        self.children_right: List[int] = []
        self.feature: List[int] = []
        self.threshold: List[float] = []
        self.feature_names = list(feature_list)
        self.n_features_in_ = len(feature_list)
        self.n_node_samples: List[int] = []
        self.value: List = []
        self.impurity: List[float] = []
        self.n_outputs = 1
        self.n_classes = n_classes


class _SimTree:
    """模拟 sklearn tree_ 对象的简易封装。

    用于将 _TreeInfo 结构适配为 _rule_generator 可识别的格式。
    """

    def __init__(
        self,
        children_left: List[int],
        children_right: List[int],
        feature: List[int],
        threshold: List[float],
        n_node_samples: List[int],
        value: List,
        impurity: List[float],
    ):
        # 将自身作为 .tree_ 属性暴露，兼容 _rule_generator(clf.tree_) 的访问方式
        self.tree_ = self
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.n_node_samples = n_node_samples
        self.value = value
        self.impurity = impurity


# ============================================================================
# DecisionTreeAnalyzer：标准 sklearn 决策树分析器
# ============================================================================


class DecisionTreeAnalyzer:
    """sklearn 决策树分析器。

    在标准 sklearn DecisionTreeClassifier 基础上，提供决策树训练、
    结构导出、规则提取、节点规则评估和 AUC / KS / LIFT 等模型指标计算。

    **参数**

    :param target: 目标变量列名（0=好样本，1=坏样本）
    :param feature_list: 特征名列表（默认自动从数据中推断数值列）
    :param tree_params: 决策树参数字典，默认值如下：

        ============ ================================
        参数          默认值
        ============ ================================
        criterion     'gini'
        splitter      'best'
        max_depth     2
        min_samples_split  2
        min_samples_leaf   1
        random_state  0
        ============ ================================

    **参考样例**

    >>> from hscredit.report.mining import DecisionTreeAnalyzer
    >>> analyzer = DecisionTreeAnalyzer(target='target', feature_list=['age', 'income'])
    >>> analyzer.fit(df_train)
    >>> # 在测试集上评估
    >>> metrics = analyzer.evaluate([('测试集', df_test)], metric_type='ks')
    >>> print(metrics)
    >>> # 获取规则表
    >>> rules = analyzer.get_rules()
    >>> print(rules)
    >>> # 导出树图
    >>> analyzer.export_tree('tree.dot')
    """

    def __init__(
        self,
        target: str = "target",
        feature_list: Optional[List[str]] = None,
        tree_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """初始化决策树训练器。

        :param target: 目标变量列名（0=好样本，1=坏样本）
        :param feature_list: 特征名列表（默认自动从数据中推断数值列）
        :param tree_params: 决策树参数字典，默认值如下：

            ============ ================================
            参数          默认值
            ============ ================================
            criterion     'gini'
            splitter      'best'
            max_depth     2
            min_samples_split  2
            min_samples_leaf   1
            random_state  0
            ============ ================================

        :param kwargs: sklearn DecisionTreeClassifier 的其他参数，直接透传给底层分类器。
            例如：`ccp_alpha=0.01`、`class_weight='balanced'`、`min_weight_fraction_leaf=0.1` 等。
        """
        self.target = target
        self.feature_list = feature_list or []
        self.tree_params = tree_params or {}
        self._sklearn_kwargs: Dict[str, Any] = kwargs

        # 默认树参数
        self._default_params = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": None,
            "random_state": 0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "class_weight": None,
            "ccp_alpha": 0.0,
        }

        # 内部状态
        self.clf: Optional[DecisionTreeClassifier] = None
        self._data: Optional[pd.DataFrame] = None
        self._df_rules: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._dot_data: Optional[str] = None
        self.__tree_info_cache: Optional[_TreeInfo] = None
        self._manual_split_nodes: set = set()  # 兼容 tree_viz._extract_tree_from_mte()

    @property
    def _tree_info(self) -> _TreeInfo:
        """提供 _TreeInfo 接口代理，兼容 tree_viz 等工具的 _tree_info 访问模式。

        注意：DecisionTreeAnalyzer 底层是 sklearn 树，不支持 manual_split 干预。
        """
        if self.__tree_info_cache is None:
            self.__tree_info_cache = _TreeInfo(
                self.feature_list,
                int(self.clf.tree_.n_classes_[0]) if hasattr(self.clf.tree_, "n_classes_") else 2,
            )
            tree = self.clf.tree_
            self.__tree_info_cache.children_left = list(tree.children_left)
            self.__tree_info_cache.children_right = list(tree.children_right)
            self.__tree_info_cache.feature = list(tree.feature)
            self.__tree_info_cache.threshold = list(tree.threshold)
            self.__tree_info_cache.n_node_samples = list(tree.n_node_samples)
            self.__tree_info_cache.value = [list(v) for v in tree.value]
            self.__tree_info_cache.impurity = list(tree.impurity)
        return self.__tree_info_cache

    # -------------------------------------------------------------------------
    # 训练
    # -------------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
        tree_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "DecisionTreeAnalyzer":
        """训练决策树。

        :param df: 包含特征和目标变量的 DataFrame
        :param feature_list: 特征名列表（默认使用除 target 外的所有数值列）
        :param tree_params: 决策树参数字典（与构造参数合并，覆盖默认参数）
        :param kwargs: sklearn DecisionTreeClassifier 的其他参数，直接透传给底层分类器。
            优先级最高，会覆盖默认参数、tree_params 和构造参数中的同名值。
        :return: self

        **参考样例**

        >>> analyzer = DecisionTreeAnalyzer(target='target')
        >>> analyzer.fit(df_train, feature_list=['age', 'income', 'loan'])
        >>> # 使用 ccp_alpha 后剪枝
        >>> analyzer2 = DecisionTreeAnalyzer(target='target')
        >>> analyzer2.fit(df_train, ccp_alpha=0.01)
        """
        # 特征列表
        if feature_list is not None:
            self.feature_list = list(feature_list)
        elif not self.feature_list:
            self.feature_list = [
                c
                for c in df.columns
                if c != self.target and pd.api.types.is_numeric_dtype(df[c])
            ]

        # 过滤缺失数据
        self._data = df.loc[df[self.target].notna(), self.feature_list + [self.target]].copy()

        # 合并参数：默认参数 → 构造参数 → 调用参数 → kwargs（优先级最高）
        params = {**self._default_params, **self.tree_params}
        if tree_params:
            params = {**params, **tree_params}
        params = {**params, **self._sklearn_kwargs, **kwargs}

        # 训练
        self.clf = DecisionTreeClassifier(**params)
        X = self._data[self.feature_list].values
        y = self._data[self.target].values
        self.clf.fit(X, y)

        self._is_fitted = True
        self._df_rules = _rule_generator(self.clf, self.feature_list)
        return self

    # -------------------------------------------------------------------------
    # 预测
    # -------------------------------------------------------------------------

    def predict(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """预测类别标签。

        :param df: 待预测数据（默认使用训练数据）
        :return: 预测结果数组
        """
        self._check_fitted()
        data = df if df is not None else self._data
        return self.clf.predict(data[self.feature_list].values)

    def predict_proba(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """预测类别概率。

        :param df: 待预测数据（默认使用训练数据）
        :return: 类别概率数组，形状 (n_samples, n_classes)
        """
        self._check_fitted()
        data = df if df is not None else self._data
        return self.clf.predict_proba(data[self.feature_list].values)

    def apply(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """返回每个样本所属叶子节点的编号。

        :param df: 待评估数据（默认使用训练数据）
        :return: 叶子节点编号数组

        **参考样例**

        >>> leaf_ids = fitter.apply(df_test)
        >>> print(f"测试集样本分布在 {len(set(leaf_ids))} 个叶子节点")
        """
        self._check_fitted()
        data = df if df is not None else self._data
        return self.clf.apply(data[self.feature_list].values)

    # -------------------------------------------------------------------------
    # 评估
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        test_data_list: List[Tuple[str, pd.DataFrame]],
        metric_type: str = "auc",
        top_rate: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """评估模型性能。

        支持多种评估指标，计算训练集及多个测试集的指标值。

        :param test_data_list: 测试数据集列表，元素为 (数据集名称, DataFrame)
        :param metric_type: 评估指标类型

            ========= ==========================================
            类型       说明
            ========= ==========================================
            'auc'     ROC AUC 分数（使用 predict_proba 的正类概率）
            'ks'      KS 统计量
            'lift'    top 客群的 LIFT 值
            'top'     top客群坏样本率（与 lift 等价）
            ========= ==========================================

        :param top_rate: lift/top 指标计算时取 top 的比例（默认 10%）
        :return: 评估结果列表，元素为 (数据集名称, 指标值)

        **参考样例**

        >>> metrics = fitter.evaluate([('测试集', df_test)], metric_type='ks')
        >>> for name, value in metrics:
        ...     print(f'{name}: {value:.4f}')
        """
        self._check_fitted()
        if metric_type not in ("auc", "ks", "lift", "top"):
            raise ValueError(f"不支持的指标类型: {metric_type}，可选值: auc/ks/lift/top")

        results: List[Tuple[str, float]] = []

        # 训练集评估
        train_prob = self.predict_proba()[:, 1]
        train_y = self._data[self.target].values
        train_metric = self._calc_metric(train_prob, train_y, metric_type, top_rate)
        results.append(("训练集", train_metric))

        # 各测试集评估
        for name, test_df in test_data_list:
            if self.target not in test_df.columns:
                raise ValueError(f"测试集 '{name}' 缺少目标列: {self.target}")
            test_prob = self.predict_proba(test_df)[:, 1]
            test_y = test_df[self.target].values
            test_metric = self._calc_metric(test_prob, test_y, metric_type, top_rate)
            results.append((name, test_metric))

        return results

    def _calc_metric(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        metric_type: str,
        top_rate: float,
    ) -> float:
        """计算单条数据的指定指标。"""
        if metric_type == "auc":
            return _auc(y_true, y_prob)
        elif metric_type == "ks":
            return _ks(y_true, y_prob)
        elif metric_type in ("lift", "top"):
            return _lift_local(y_true, y_prob, n_bins=int(top_rate * 100))

    def report(
        self,
        datasets: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]],
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        del_grey: bool = False,
        leaf_only: bool = False,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]]:
        """在新数据集上评估决策树各节点规则的效果。

        对每个节点（含分裂节点和叶子节点），将其规则路径解析为 :class:`Rule` 对象，
        直接调用 :meth:`Rule.report` 计算统计指标，取其中"命中"分箱对应的结果行汇总，
        得到各节点在新数据上的效果报告。

        **参数**

        :param datasets: 待评估数据集，支持单个 DataFrame、DataFrame 列表、
            或 ``{名称: DataFrame}`` 字典；返回结果类型与输入保持一致
        :param target: 目标变量列名，默认为None（使用拟合时的 ``self.target``）
        :param overdue: 逾期天数字段名（可选，传入时以逾期天数>DPD定义坏样本，
            支持多标签多DPD联合分析），参考 :meth:`Rule.report`
        :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本，默认为0；
            传入列表时支持多DPD联合分析，参考 :meth:`Rule.report`
        :param del_grey: 是否删除逾期天数在(0, DPD]区间内的灰度样本，默认为False
        :param leaf_only: 是否仅评估叶子节点，默认 False（评估所有节点）
        :param kwargs: 其余传递给 :meth:`Rule.report` 的参数（如 ``amount``、``margins``）
        :return: 各节点效果评估表（DataFrame），结构与输入一致：
            单个 DataFrame 输入返回单个 DataFrame，列表输入返回 DataFrame 列表，
            字典输入返回 ``{名称: DataFrame}`` 字典

        **参考样例**

        >>> result = fitter.report(df_test, target='target')
        >>> results = fitter.report([df_test1, df_test2], target='target')
        >>> results = fitter.report({'测试集1': df_test1, '测试集2': df_test2}, target='target')
        >>> # 多标签（逾期天数）联合分析
        >>> result = fitter.report(df_test, overdue=['MOB1'], dpds=[7, 3, 0])
        """
        self._check_fitted()

        def _report_one(data: pd.DataFrame) -> pd.DataFrame:
            return _node_hit_report(
                self._df_rules,
                self._format_rule,
                data,
                target=target or self.target,
                overdue=overdue,
                dpds=dpds,
                del_grey=del_grey,
                leaf_only=leaf_only,
                **kwargs,
            )

        if isinstance(datasets, dict):
            return {name: _report_one(data) for name, data in datasets.items()}
        elif isinstance(datasets, list):
            return [_report_one(data) for data in datasets]
        else:
            return _report_one(datasets)

    def get_leaf_node_ids(self) -> List[int]:
        """获取所有叶子节点的 ID 列表。"""
        self._check_fitted()
        return self._df_rules[self._df_rules["if_leaf"]]["node"].tolist()

    # -------------------------------------------------------------------------
    # 规则提取
    # -------------------------------------------------------------------------

    def get_rules(self) -> List[Rule]:
        """将树的叶子节点转换为 Rule 对象列表。

        :return: Rule 对象列表，每个 Rule 对应一个叶子节点

        **参考样例**

        >>> rules = fitter.get_rules()
        >>> for rule in rules:
        ...     report = rule.report(df_test, target='target')
        """
        self._check_fitted()
        rules: List[Rule] = []
        leaf_rules = self._df_rules[self._df_rules["if_leaf"]]
        for _, row in leaf_rules.iterrows():
            rule = self._format_rule(row["rule_list"])
            if rule is None:
                # 叶子节点必然带有规则路径，空规则（根节点）理论上不会出现，稳妥跳过
                continue
            rule.name = f"DecisionTree_N{int(row['node'])}"
            rules.append(rule)
        return rules

    def get_rule_table(
        self,
        datasets: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        del_grey: bool = False,
        leaf_only: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """获取决策树所有节点（分裂节点+叶子节点）的规则效果表。

        从每个节点提取规则路径，解析为 :class:`Rule` 对象后调用 :meth:`Rule.report`
        在数据集上计算命中效果，取其中"命中"分箱对应的结果行汇总。各节点统计口径
        与 :meth:`report` 完全一致（均基于规则在数据集上的真实命中情况计算，而非
        读取决策树内部存储的节点样本统计），便于与人工分裂结果横向对比。

        **参数**

        :param datasets: 用于计算规则效果的数据集，默认 None（使用 fit 时的训练数据）
        :param target: 目标变量列名，默认 None（使用拟合时的 ``self.target``）
        :param overdue: 逾期天数字段名，参考 :meth:`Rule.report`
        :param dpds: 逾期定义方式，参考 :meth:`Rule.report`
        :param del_grey: 是否删除灰度样本，默认 False
        :param leaf_only: 是否仅返回叶子节点，默认 False
        :param kwargs: 其余传递给 :meth:`Rule.report` 的参数（如 ``amount``、``margins``）
        :return: 规则效果表，列结构与 :meth:`report` 一致（节点编号、是否叶子、
            规则分类、指标名称、指标含义、分箱、样本总数、样本占比、好样本数、
            坏样本数、坏样本率、LIFT值、坏账改善、风险拒绝比等）

        **参考样例**

        >>> table = fitter.get_rule_table()            # 在训练数据上评估
        >>> table = fitter.get_rule_table(df_test)     # 在新数据上评估
        """
        self._check_fitted()
        data = datasets if datasets is not None else self._data
        if data is None:
            raise InputValidationError(
                "get_rule_table 需要数据集计算规则效果：请传入 datasets 参数，"
                "或先通过 fit() 提供训练数据"
            )
        return self.report(
            data,
            target=target,
            overdue=overdue,
            dpds=dpds,
            del_grey=del_grey,
            leaf_only=leaf_only,
            **kwargs,
        )

    @staticmethod
    def _rule_list_to_text(rule_list: List) -> str:
        """将规则列表格式化为可读的中文规则串（如 "age <= 35.0000 且 income > 5000"）。"""
        if not rule_list:
            return "空规则"
        parts = []
        for item in rule_list:
            feat = item[0]
            op = item[1]
            thres = f"{item[2]:.4f}" if isinstance(item[2], float) else str(item[2])
            parts.append(f"{feat} {op} {thres}")
        return " 且 ".join(parts)

    def _format_rule(self, rule_list: List) -> Optional[Rule]:
        """将规则列表解析为 :class:`Rule` 对象。

        - ``expr``：pandas eval 表达式（用于 predict / report 等规则评估）
        - ``name`` / ``description``：可读的中文规则串（用于表格展示）

        根节点对应的空规则返回 None。

        :param rule_list: 规则列表，元素为 ``[特征名, 操作符, 阈值]``
        :return: 解析得到的 Rule 对象；空规则返回 None
        """
        if not rule_list:
            return None
        text = self._rule_list_to_text(rule_list)
        expr = self._rule_to_expr(rule_list)
        return Rule(expr=expr, name=text, description=text)

    def _rule_to_expr(self, rule_list: List) -> str:
        """将规则列表转换为 pandas eval 表达式。"""
        if not rule_list:
            return "True"
        parts = []
        for feat, op, thres in rule_list:
            feat_esc = f"`{feat}`" if not str(feat).isidentifier() else str(feat)
            parts.append(f"({feat_esc} {op} {repr(float(thres))})")
        return " & ".join(parts)

    # -------------------------------------------------------------------------
    # 可视化与导出
    # -------------------------------------------------------------------------

    def export_tree(
        self,
        out_file: Optional[str] = None,
        max_depth: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """导出决策树为 DOT 格式。

        :param out_file: 输出 .dot 文件路径（可选，指定时同时写入文件）
        :param max_depth: 最大显示深度（None=全部显示）
        :param class_names: 类别名列表，默认 ['好', '坏']
        :return: DOT 格式字符串

        **参考样例**

        >>> dot = fitter.export_tree('tree.dot')
        >>> with open('tree.dot') as f:
        ...     print(f.read())
        """
        self._check_fitted()
        if class_names is None:
            class_names = ["好", "坏"]
        return _export_dot_data(
            self.clf,
            self.feature_list,
            class_names=class_names,
            out_file=out_file,
            max_depth=max_depth,
        )

    def save(
        self,
        file_path: str,
        include_data: bool = True,
    ) -> None:
        """将决策树保存为 pickle 文件。

        :param file_path: 保存路径
        :param include_data: 是否包含训练数据（默认 True，保存后可直接 load 并 evaluate）

        **参考样例**

        >>> fitter.save('dt_model.pkl')
        """
        self._check_fitted()
        payload = {
            "clf": self.clf,
            "feature_list": self.feature_list,
            "target": self.target,
            "tree_params": self.tree_params,
        }
        if include_data and self._data is not None:
            payload["_data"] = self._data
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file_path: str) -> "DecisionTreeAnalyzer":
        """从 pickle 文件加载决策树。

        :param file_path: 模型文件路径
        :return: 加载后的 DecisionTreeAnalyzer 实例

        **参考样例**

        >>> analyzer2 = DecisionTreeAnalyzer.load('dt_model.pkl')
        """
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(
            target=payload["target"],
            feature_list=payload["feature_list"],
            tree_params=payload["tree_params"],
        )
        instance.clf = payload["clf"]
        instance._is_fitted = True
        if "_data" in payload:
            instance._data = payload["_data"]
        instance._df_rules = _rule_generator(instance.clf, instance.feature_list)
        return instance

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """检查是否已训练。"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法训练决策树")

    def __repr__(self) -> str:
        if self._is_fitted:
            n_leaves = int(self._df_rules["if_leaf"].sum()) if self._df_rules is not None else 0
            return (
                f"DecisionTreeAnalyzer(target='{self.target}', "
                f"features={self.feature_list}, "
                f"leaves={n_leaves})"
            )
        return "DecisionTreeAnalyzer(not fitted)"

# ============================================================================
# ManualTreeExtractor：人工干预决策树节点分裂
# ============================================================================


class ManualTreeExtractor:
    """人工决策树提取器。

    支持对 sklearn 决策树进行**人工指定分裂节点**后重新训练，
    适合将业务经验注入数据驱动模型。

    核心流程：
    1. 用数据训练一棵基础决策树（或直接指定特征/阈值）
    2. 人工在指定节点分裂（manual_split），指定特征和阈值
    3. 获取规则表或在新数据集上评估效果

    **参数**

    :param target: 目标变量列名（坏样本标签，0=好，1=坏），默认 'target'
    :param max_depth: 树的最大深度，默认 2
    :param min_samples_split: 分裂节点最小样本数，默认 10
    :param min_samples_leaf: 叶子节点最小样本数，默认 5
    :param random_state: 随机种子，默认 0

    **参考样例**

    >>> from hscredit.report.mining import ManualTreeExtractor
    >>> ext = ManualTreeExtractor(target='target', max_depth=2)
    >>> ext.fit(df, feature_list=['age', 'income'])
    >>> # 人工分裂：指定在某节点用某特征+阈值分裂
    >>> ext.manual_split(df_sub, feature_name='age', threshold=35, node=1)
    >>> # 获取规则表
    >>> print(ext.get_rule_table())
    >>> # 在新数据上评估
    >>> print(ext.report(df_test))
    >>> # 获取 Rule 对象
    >>> rules = ext.get_rules()
    """

    def __init__(
        self,
        target: str = "target",
        max_depth: int = 2,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 0,
        **kwargs: Any,
    ):
        """初始化人工决策树提取器。

        :param target: 目标变量列名（坏样本标签，0=好，1=坏），默认 'target'
        :param max_depth: 树的最大深度，默认 2
        :param min_samples_split: 分裂节点最小样本数，默认 10
        :param min_samples_leaf: 叶子节点最小样本数，默认 5
        :param random_state: 随机种子，默认 0
        :param kwargs: sklearn DecisionTreeClassifier 的其他参数，直接透传给底层分类器。
            例如：`ccp_alpha=0.01`、`class_weight='balanced'`、`criterion='entropy'` 等。
        """
        self.target = target
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._sklearn_kwargs: Dict[str, Any] = kwargs

        # 内部状态
        self._data: Optional[pd.DataFrame] = None
        self._feature_list: List[str] = []
        self._n_total_samples: int = 0
        self._overall_badrate: float = 0.0
        self._tree_info: Optional[_TreeInfo] = None
        self._df_rules: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False
        self._sklearn_clf: Optional[DecisionTreeClassifier] = None
        # 追踪经 manual_split 人工修改过的节点 ID
        self._manual_split_nodes: set = set()

    @classmethod
    def from_sklearn(
        cls,
        clf: DecisionTreeClassifier,
        feature_names: Optional[List[str]] = None,
        target: str = "target",
    ) -> "ManualTreeExtractor":
        """从已训练的 sklearn DecisionTreeClassifier 创建 ManualTreeExtractor。

        适用于想先训练好树结构，再进行人工干预分裂的场景。

        **参数**

        :param clf: 已训练的 sklearn DecisionTreeClassifier
        :param feature_names: 特征名列表（默认从 clf.feature_names_in_ 推断）
        :param target: 目标变量名（仅用于存储，不参与训练），默认 'target'
        :return: ManualTreeExtractor 实例（已 fitted 状态）

        **参考样例**

        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from hscredit.report.mining import ManualTreeExtractor
        >>> clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        >>> clf.fit(X, y)
        >>> mte = ManualTreeExtractor.from_sklearn(clf, feature_names=feature_names)
        >>> mte.manual_split(df, feature_name='age', threshold=35, node=1)
        """
        if feature_names is None:
            feature_names = list(getattr(clf, "feature_names_in_", []))

        n_classes = clf.tree_.n_classes_[0] if hasattr(clf.tree_, "n_classes_") else 2
        instance = cls(target=target, max_depth=clf.max_depth, random_state=getattr(clf, "random_state", 0))
        instance._feature_list = list(feature_names)
        instance._n_total_samples = int(clf.tree_.n_node_samples[0]) if clf.tree_.n_node_samples.size > 0 else 0
        instance._data = None
        instance._overall_badrate = 0.0
        instance._tree_info = _TreeInfo(feature_names, n_classes)
        instance._manual_split_nodes = set()
        instance._is_fitted = True
        instance._sklearn_clf = clf
        instance._sync_from_sklearn()
        instance._generate_rules()
        return instance

    # -------------------------------------------------------------------------
    # fit / 自动建树
    # -------------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        **kwargs: Any,
    ) -> "ManualTreeExtractor":
        """训练基础决策树。

        使用数据训练一棵标准 sklearn 决策树。后续可通过 manual_split
        在指定节点人工干预分裂。

        **参数**

        :param df: 包含特征和目标变量的 DataFrame
        :param feature_list: 特征名列表（默认使用除 target 外的所有数值列）
        :param max_depth: 树最大深度（覆盖构造参数）
        :param min_samples_split: 分裂最小样本数（覆盖构造参数）
        :param min_samples_leaf: 叶子最小样本数（覆盖构造参数）
        :param kwargs: sklearn DecisionTreeClassifier 的其他参数，直接透传给底层分类器。
            优先级最高，会覆盖默认参数和构造参数中的同名值。
        :return: self

        **参考样例**

        >>> ext = ManualTreeExtractor(target='target')
        >>> ext.fit(df, feature_list=['age', 'income', 'loan_amount'])
        >>> # 使用熵作为分裂准则 + ccp_alpha 后剪枝
        >>> ext2 = ManualTreeExtractor(target='target')
        >>> ext2.fit(df, feature_list=['age', 'income'], criterion='entropy', ccp_alpha=0.01)
        """
        if feature_list is None:
            feature_list = [
                c
                for c in df.columns
                if c != self.target and pd.api.types.is_numeric_dtype(df[c])
            ]

        self._feature_list = list(feature_list)

        # 过滤缺失数据
        self._data = df.loc[df[self.target].notna(), self._feature_list + [self.target]].copy()
        self._n_total_samples = len(self._data)
        self._overall_badrate = self._data[self.target].mean()

        # 初始化树结构
        n_classes = int(self._data[self.target].nunique())
        self._tree_info = _TreeInfo(self._feature_list, n_classes)

        # 重置人工修改节点记录
        self._manual_split_nodes = set()

        # 训练 sklearn 决策树（优先级：kwargs > self._sklearn_kwargs > 显式参数 > 默认值）
        tree_params: Dict[str, Any] = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": max_depth if max_depth is not None else self.max_depth,
            "min_samples_split": (
                min_samples_split if min_samples_split is not None else self.min_samples_split
            ),
            "min_samples_leaf": (
                min_samples_leaf if min_samples_leaf is not None else self.min_samples_leaf
            ),
            "random_state": self.random_state,
            "class_weight": None,
            "ccp_alpha": 0.0,
        }
        # 合并 kwargs（构造参数 > 调用参数）
        tree_params = {**tree_params, **self._sklearn_kwargs, **kwargs}
        self._sklearn_clf = DecisionTreeClassifier(**tree_params)
        X = self._data[self._feature_list].values
        y = self._data[self.target].values
        self._sklearn_clf.fit(X, y)

        # 同步树结构到 _tree_info
        self._sync_from_sklearn()

        self._is_fitted = True
        self._generate_rules()
        return self

    def _sync_from_sklearn(self) -> None:
        """将 sklearn 训练结果同步到内部 _tree_info。"""
        if self._sklearn_clf is None:
            return
        tree = self._sklearn_clf.tree_
        self._tree_info.children_left = list(tree.children_left)
        self._tree_info.children_right = list(tree.children_right)
        self._tree_info.feature = list(tree.feature)
        self._tree_info.threshold = list(tree.threshold)
        self._tree_info.n_node_samples = list(tree.n_node_samples)
        self._tree_info.value = [list(v) for v in tree.value]
        self._tree_info.impurity = list(tree.impurity)

    # -------------------------------------------------------------------------
    # 树操作：人工分裂 / 删除节点
    # -------------------------------------------------------------------------

    def manual_split(
        self,
        df: pd.DataFrame,
        feature_name: str,
        threshold: Optional[float] = None,
        node: int = 0,
    ) -> "ManualTreeExtractor":
        """在指定节点人工分裂。

        在 node 位置按 feature_name 和 threshold 进行分裂。
        若 threshold 为 None，则用决策树自动计算最优分裂点。
        支持链式调用。

        **参数**

        :param df: 用于计算分裂阈值的数据子集
        :param feature_name: 分裂特征名
        :param threshold: 分裂阈值（None=自动计算最优阈值）
        :param node: 分裂的节点 ID，默认 0（根节点）
        :return: self

        **参考样例**

        >>> # 人工指定阈值
        >>> ext.manual_split(df_sub, feature_name='age', threshold=35, node=1)
        >>> # 自动找最优阈值
        >>> ext.manual_split(df_sub, feature_name='income', threshold=None, node=2)
        >>> # 链式调用
        >>> ext.manual_split(df1, 'f1', 30).manual_split(df2, 'f2', 20)
        """
        self._check_fitted()
        if feature_name not in self._feature_list:
            raise ValueError(f"特征 '{feature_name}' 不在特征列表中")

        # 先删除该节点的旧子树
        self.delete_node(node)

        df_work = df.copy()

        # 若未指定阈值，用单变量决策树找最优切分点
        if threshold is None:
            tmp_params = {
                "max_depth": 1,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            }
            clf_tmp = DecisionTreeClassifier(**tmp_params)
            X_tmp = df_work[[feature_name]].values
            y_tmp = df_work[self.target].values
            clf_tmp.fit(X_tmp, y_tmp)
            threshold = float(clf_tmp.tree_.threshold[0])
            node_values = [list(v) for v in clf_tmp.tree_.value]
        else:
            # 手动阈值：按指定阈值计算实际样本统计
            left_mask = df_work[feature_name] <= threshold
            right_mask = ~left_mask
            parent_total = len(df_work)
            left_n = int(left_mask.sum())
            right_n = parent_total - left_n
            parent_good = float((df_work[self.target] == 0).sum())
            parent_bad = float((df_work[self.target] == 1).sum())
            left_good = float(((df_work[self.target] == 0) & left_mask).sum())
            left_bad = float(((df_work[self.target] == 1) & left_mask).sum())
            right_good = float(((df_work[self.target] == 0) & right_mask).sum())
            right_bad = float(((df_work[self.target] == 1) & right_mask).sum())
            node_values = [
                [[parent_good / parent_total, parent_bad / parent_total]],
                [[left_good / left_n if left_n > 0 else 0, left_bad / left_n if left_n > 0 else 0]],
                [[right_good / right_n if right_n > 0 else 0, right_bad / right_n if right_n > 0 else 0]],
            ]

        feat_idx = self._feature_list.index(feature_name)
        if threshold is None:
            raise RuntimeError("threshold 未能自动计算")
        left_n = int((df_work[feature_name] <= threshold).sum())
        right_n = len(df_work) - left_n

        # 新子树结构：[parent, left_leaf, right_leaf]
        children_left_new = [1, -1, -1]
        children_right_new = [2, -1, -1]
        feature_new = [feat_idx, -2, -2]
        threshold_new = np.array([threshold, -2.0, -2.0])
        n_node_samples_new = [len(df_work), left_n, right_n]
        node_impurity_new = [1.0, 1.0, 1.0]

        # 插入到指定节点
        (
            self._tree_info.children_left,
            self._tree_info.children_right,
            self._tree_info.feature,
            self._tree_info.threshold,
            self._tree_info.n_node_samples,
            self._tree_info.value,
            self._tree_info.impurity,
        ) = _add_nodes_to_tree(
            node=node,
            split_list_left=self._tree_info.children_left,
            split_list_right=self._tree_info.children_right,
            feature=self._tree_info.feature,
            threshold=self._tree_info.threshold,
            node_samples=self._tree_info.n_node_samples,
            node_values=self._tree_info.value,
            node_impurity=self._tree_info.impurity,
            split_list_left_new=children_left_new,
            split_list_right_new=children_right_new,
            feature_new=feature_new,
            threshold_new=list(threshold_new),
            node_samples_new=n_node_samples_new,
            node_values_new=node_values,
            node_impurity_new=node_impurity_new,
        )

        # 记录人工修改过的节点（分裂节点 + 它的两个新子节点）
        new_node_ids = _find_subtree_node_ids(self._tree_info, node)
        self._manual_split_nodes.update(new_node_ids)

        self._generate_rules()
        return self

    def delete_node(self, node: int) -> "ManualTreeExtractor":
        """删除指定节点及其所有子节点，将该节点变为叶子。

        **参数**

        :param node: 待删除的节点 ID
        :return: self

        **参考样例**

        >>> ext.delete_node(node=3)
        """
        self._check_fitted()

        (
            self._tree_info.children_left,
            self._tree_info.children_right,
            self._tree_info.feature,
            self._tree_info.threshold,
            self._tree_info.n_node_samples,
            self._tree_info.value,
            self._tree_info.impurity,
        ) = _delete_nodes(
            node=node,
            split_list_left=self._tree_info.children_left,
            split_list_right=self._tree_info.children_right,
            feature=self._tree_info.feature,
            threshold=self._tree_info.threshold,
            node_samples=self._tree_info.n_node_samples,
            node_values=self._tree_info.value,
            node_impurity=self._tree_info.impurity,
        )

        self._generate_rules()
        return self

    # -------------------------------------------------------------------------
    # 规则生成
    # -------------------------------------------------------------------------

    def _generate_rules(self) -> None:
        """从当前树结构生成规则 DataFrame。"""
        if self._tree_info is None:
            return

        sim = _SimTree(
            children_left=self._tree_info.children_left,
            children_right=self._tree_info.children_right,
            feature=self._tree_info.feature,
            threshold=self._tree_info.threshold,
            n_node_samples=self._tree_info.n_node_samples,
            value=self._tree_info.value,
            impurity=self._tree_info.impurity,
        )
        # sim.tree_ == sim 自身，_rule_generator 内部访问 clf.tree_ → sim
        self._df_rules = _rule_generator(sim, self._feature_list)

    @staticmethod
    def _rule_list_to_text(rule_list: List) -> str:
        """将规则列表格式化为可读的中文规则串（如 "age <= 35.0000 且 income > 5000"）。"""
        if not rule_list:
            return "空规则"
        parts = []
        for item in rule_list:
            feat = item[0]
            op = item[1]
            thres = f"{item[2]:.4f}" if isinstance(item[2], float) else str(item[2])
            parts.append(f"{feat} {op} {thres}")
        return " 且 ".join(parts)

    def _format_rule(self, rule_list: List) -> Optional[Rule]:
        """将规则列表解析为 :class:`Rule` 对象。

        - ``expr``：pandas eval 表达式（用于 predict / report 等规则评估）
        - ``name`` / ``description``：可读的中文规则串（用于表格展示）

        根节点对应的空规则返回 None。

        :param rule_list: 规则列表，元素为 ``[特征名, 操作符, 阈值]``
        :return: 解析得到的 Rule 对象；空规则返回 None
        """
        if not rule_list:
            return None
        text = self._rule_list_to_text(rule_list)
        expr = self._rule_to_expr(rule_list)
        return Rule(expr=expr, name=text, description=text)

    # -------------------------------------------------------------------------
    # 评估与报告
    # -------------------------------------------------------------------------

    def get_rule_table(
        self,
        datasets: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        del_grey: bool = False,
        leaf_only: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """获取当前树各节点规则在数据集上的效果表。

        从树的每个节点提取规则路径，解析为 :class:`Rule` 对象后调用 :meth:`Rule.report`
        在数据集上计算命中效果，取其中"命中"分箱对应的结果行汇总。这样人工分裂节点与
        自动分裂节点的统计口径完全一致（均基于规则在数据集上的真实命中情况计算，而非
        读取决策树内部存储的节点样本统计），与 :meth:`report` 输出格式保持一致。

        **参数**

        :param datasets: 用于计算规则效果的数据集，默认 None（使用 fit 时的训练数据）
        :param target: 目标变量列名，默认 None（使用拟合时的 ``self.target``）
        :param overdue: 逾期天数字段名，参考 :meth:`Rule.report`
        :param dpds: 逾期定义方式，参考 :meth:`Rule.report`
        :param del_grey: 是否删除灰度样本，默认 False
        :param leaf_only: 是否仅返回叶子节点，默认 False
        :param kwargs: 其余传递给 :meth:`Rule.report` 的参数（如 ``amount``、``margins``）
        :return: 规则效果表，列结构与 :meth:`report` 一致（节点编号、是否叶子、
            规则分类、指标名称、指标含义、分箱、样本总数、样本占比、好样本数、
            坏样本数、坏样本率、LIFT值、坏账改善、风险拒绝比等）。其中（参考
            :meth:`Rule.report` 的口径，将命中节点的样本视为"拒绝"客群）：

            - **坏账改善** = (整体坏账率 - 拒绝后剩余客群坏账率) / 整体坏账率
            - **风险拒绝比** = 坏账改善 / 样本占比

        **参考样例**

        >>> ext.manual_split(df, feature_name='age', threshold=35)
        >>> print(ext.get_rule_table())          # 在训练数据上评估
        >>> print(ext.get_rule_table(df_test))   # 在新数据上评估
        """
        self._check_fitted()
        data = datasets if datasets is not None else self._data
        if data is None:
            raise InputValidationError(
                "get_rule_table 需要数据集计算规则效果：请传入 datasets 参数，"
                "或先通过 fit() 提供训练数据"
            )
        return self.report(
            data,
            target=target,
            overdue=overdue,
            dpds=dpds,
            del_grey=del_grey,
            leaf_only=leaf_only,
            **kwargs,
        )

    def display(self) -> "ManualTreeExtractor":
        """在 Jupyter Notebook 中展示决策树图和规则表。

        每次调用都会根据当前树结构重新生成，确保 manual_split / delete_node 后显示最新状态。

        决策树图使用 :func:`hscredit.core.viz.plot_tree_matplotlib`（AntV G6 卡片式风格）
        绘制：卡片节点 + 主题色标题栏 + 节点指标，按坏账率从浅蓝（低风险）到浅红（高风险）
        着色，人工修改节点（manual_split）使用副主题色边框标记。规则表使用
        :func:`style_rule_table` 美化展示。

        **参考样例**

        >>> ext = ManualTreeExtractor(target='target')
        >>> ext.fit(df, feature_list=['age', 'income'])
        >>> ext.display()   # 在 Jupyter 中展示树图和规则表
        >>> ext.manual_split(df, 'income', 5000, node=1).display()
        """
        try:
            from IPython.display import display as ipy_display
            import matplotlib.pyplot as plt

            from ...core.viz.tree_plots import plot_tree_matplotlib

            self._check_fitted()

            # 用 plot_tree_matplotlib 绘制当前树结构（反映最新的人工分裂结果）
            fig = plot_tree_matplotlib(self)
            ipy_display(fig)
            plt.close(fig)

            # 渲染美化后的规则表
            rule_table = self.get_rule_table()
            if rule_table is not None and len(rule_table) > 0:
                styler = style_rule_table(rule_table, overall_badrate=self._overall_badrate)
                ipy_display(styler)

        except ImportError:
            # IPython / matplotlib 未安装
            pass
        except Exception:
            # 非 Jupyter 环境或其他错误：静默跳过
            pass

        return self

    def report(
        self,
        datasets: Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]],
        target: Optional[str] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, List[int]]] = None,
        del_grey: bool = False,
        leaf_only: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, List[pd.DataFrame], Dict[Any, pd.DataFrame]]:
        """在新数据集上评估当前树各节点规则的效果。

        对每个节点（含分裂节点和叶子节点），将其规则路径解析为 :class:`Rule` 对象，
        直接调用 :meth:`Rule.report` 计算统计指标，取其中"命中"分箱对应的结果行汇总，
        得到各节点在新数据上的效果报告。

        **参数**

        :param datasets: 待评估数据集，支持单个 DataFrame、DataFrame 列表、
            或 ``{名称: DataFrame}`` 字典；返回结果类型与输入保持一致
        :param target: 目标变量列名，默认为None（使用拟合时的 ``self.target``）
        :param overdue: 逾期天数字段名（可选，传入时以逾期天数>DPD定义坏样本，
            支持多标签多DPD联合分析），参考 :meth:`Rule.report`
        :param dpds: 逾期定义方式，逾期天数 > DPD 为坏样本，默认为0；
            传入列表时支持多DPD联合分析，参考 :meth:`Rule.report`
        :param del_grey: 是否删除逾期天数在(0, DPD]区间内的灰度样本，默认为False
        :param leaf_only: 是否仅评估叶子节点，默认 False（评估所有节点）
        :param kwargs: 其余传递给 :meth:`Rule.report` 的参数（如 ``amount``、``margins``）
        :return: 各节点效果评估表（DataFrame），结构与输入一致：
            单个 DataFrame 输入返回单个 DataFrame，列表输入返回 DataFrame 列表，
            字典输入返回 ``{名称: DataFrame}`` 字典

        **参考样例**

        >>> result = ext.report(df_test, target='target')
        >>> results = ext.report([df_test1, df_test2], target='target')
        >>> results = ext.report({'测试集1': df_test1, '测试集2': df_test2}, target='target')
        >>> # 多标签（逾期天数）联合分析
        >>> result = ext.report(df_test, overdue=['MOB1'], dpds=[7, 3, 0])
        """
        self._check_fitted()

        def _report_one(data: pd.DataFrame) -> pd.DataFrame:
            return _node_hit_report(
                self._df_rules,
                self._format_rule,
                data,
                target=target or self.target,
                overdue=overdue,
                dpds=dpds,
                del_grey=del_grey,
                leaf_only=leaf_only,
                **kwargs,
            )

        if isinstance(datasets, dict):
            return {name: _report_one(data) for name, data in datasets.items()}
        elif isinstance(datasets, list):
            return [_report_one(data) for data in datasets]
        else:
            return _report_one(datasets)

    def get_rules(self) -> List[Rule]:
        """将当前树的叶子节点规则转换为 Rule 对象列表。

        :return: Rule 对象列表

        **参考样例**

        >>> rules = ext.get_rules()
        >>> for r in rules:
        ...     report = r.report(df, target='target')
        """
        self._check_fitted()
        rules: List[Rule] = []
        for _, row in self._df_rules[self._df_rules["if_leaf"]].iterrows():
            rule = self._format_rule(row["rule_list"])
            if rule is None:
                # 叶子节点必然带有规则路径，空规则（根节点）理论上不会出现，稳妥跳过
                continue
            rule.name = f"TreeNode_{int(row['node'])}"
            rules.append(rule)
        return rules

    def _rule_to_expr(self, rule_list: List) -> str:
        """将规则列表转换为 pandas eval 表达式。"""
        if not rule_list:
            return "True"
        parts = []
        for feat, op, thres in rule_list:
            feat_esc = f"`{feat}`" if not str(feat).isidentifier() else str(feat)
            parts.append(f"({feat_esc} {op} {repr(float(thres))})")
        return " & ".join(parts)

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """检查是否已拟合。"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法训练决策树")

    def __repr__(self) -> str:
        if self._is_fitted:
            n_leaves = int(self._df_rules["if_leaf"].sum()) if self._df_rules is not None else 0
            return (
                f"ManualTreeExtractor(target='{self.target}', "
                f"features={self._feature_list}, "
                f"samples={self._n_total_samples}, "
                f"leaves={n_leaves})"
            )
        return "ManualTreeExtractor(not fitted)"
