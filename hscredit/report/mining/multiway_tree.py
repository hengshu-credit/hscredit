"""可自动生长和人工编辑的多叉分类树，复用分箱语义、Rule 和树图后端。"""

import copy
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import gammaln
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from ...core.binning import QuantileBinning
from ...core.binning._contracts import is_missing_marker, parse_numerical_user_splits
from ...core.binning._categorical import normalize_user_groups
from ...core.binning.spec import BinSpec, field_expression, value_expression
from ...core.metrics import ks
from ...core.models._contracts import extract_target
from ...core.rules import Rule
from ...exceptions import InputValidationError, NotFittedError
from ...utils.parallel import ParallelizableMixin
from ...utils.serialization import ArtifactSerializableMixin
from ...utils.overdue import validate_overdue_operator
from .base import format_feature_context, normalize_feature_names, resolve_feature_map
from .manual_tree_extractor import _node_hit_report


@dataclass
class MultiwayNode:
    """多叉节点状态；children 与 branches 一一对应，节点编号在局部编辑后保持稳定。"""

    node_id: int
    parent: object
    depth: int
    counts: np.ndarray
    path: tuple = ()
    feature: object = None
    thresholds: tuple = ()
    category_groups: tuple = ()
    missing_branch: int = 0
    missing_separate: bool = False
    unknown_branch: object = None
    children: list = field(default_factory=list)
    branches: tuple = ()
    gain: float = 0.0
    manual: bool = False


def _split_statistics(counts, criterion):
    """由原始类别计数计算 Pearson 卡方、自由度和加权不纯度下降。"""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts.sum(axis=1) > 0]
    counts = counts[:, counts.sum(axis=0) > 0]
    total = counts.sum()
    if total == 0 or min(counts.shape) < 2:
        return 0.0, 1.0, 0.0
    sizes = counts.sum(axis=1)
    parent = counts.sum(axis=0) / total
    probabilities = counts / sizes[:, None]
    expected = sizes[:, None] * parent
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    pvalue = float(chi2.sf(statistic, (len(sizes) - 1) * (len(parent) - 1)))
    if criterion == "entropy":
        parent_impurity = -np.sum(parent * np.log2(parent))
        positive = np.maximum(probabilities, np.finfo(float).tiny)
        children_impurity = -np.sum(probabilities * np.log2(positive), axis=1)
    else:
        parent_impurity = 1 - np.sum(parent**2)
        children_impurity = 1 - np.sum(probabilities**2, axis=1)
    gain = float(parent_impurity - np.dot(sizes / total, children_impurity))
    return statistic, pvalue, max(0.0, gain)


def _chi2_log_pvalue(statistic, degrees):
    """保留极小尾概率的对数，避免大样本的多个分裂都被截成同一价值。"""
    if degrees <= 0 or statistic <= 0:
        return 0.0
    probability = chi2.sf(statistic, degrees)
    if probability > 0:
        return float(np.log(probability))
    # 上不完全 Gamma 的大 x 渐近展开；仅在原生尾概率已下溢时使用。
    x, a = statistic / 2, degrees / 2
    term = total = 1.0
    for order in range(1, 256):
        following = term * (a - order) / x
        if abs(following) > abs(term):
            break
        total += following
        term = following
        if abs(term) <= abs(total) * np.finfo(float).eps:
            break
    return float(-x + (a - 1) * np.log(x) - gammaln(a) + np.log(total))


class MultiwayTreeExtractor(ClassifierMixin, ParallelizableMixin, ArtifactSerializableMixin, BaseEstimator):
    """支持自动分裂、人工改阈值和剪枝的多叉分类树。

    **参数**

    :param target: 标签列名，默认 ``'target'``；``fit(X, y)`` 中显式 y 优先，目标列不参与分裂。
    :type target: str
    :param features: 特征列名或列表；默认 ``None`` 使用除标签外的全部列，顺序保留在 ``features_``。
        可在 ``fit`` 中覆盖。金额、日期和逾期字段只用于报告时，应通过此参数限制分裂特征。
    :type features: str or list[str] or None
    :param max_depth: 非负整数，根节点深度为 0，默认 3；0 表示只保留根节点。
    :type max_depth: int
    :param max_branches: 每次分裂最多分支数（含独立缺失分支），至少 2，默认 4。
    :type max_branches: int
    :param min_samples_split: 自动分裂的最小节点样本数，默认 20。
    :type min_samples_split: int
    :param min_samples_leaf: 自动分裂的最小普通叶节点样本数，默认 10；独立缺失分支不受此限制。
    :type min_samples_leaf: int
    :param criterion: ``'chi2'``、``'gini'`` 或 ``'entropy'``，默认 chi2。
        卡方准则按调整后 P 值的 -log10 排序，参考 SAS 的 LogWorth。
    :type criterion: str
    :param alpha: 卡方自动分裂的显著性上限，取值在 (0, 1]，默认 0.05；其他准则不使用此门槛。
    :type alpha: float
    :param adjust_pvalue: 是否按实际搜索方案数作 Bonferroni 调整，默认 True。
        这是明确的多重比较控制，不声称复刻 SAS 的 Kass/深度调整。
    :type adjust_pvalue: bool
    :param max_prebins: 数值候选池的等频预分箱数，默认 12。
    :type max_prebins: int
    :param max_candidates: 穷举方案上限，默认 5000；超过时采用逐步增加切点的贪心搜索。
    :type max_candidates: int
    :param min_impurity_decrease: 自动生长时的最小加权不纯度下降，默认 0.0；
        比较量为节点样本数占全部训练样本的比例乘以该节点分裂增益。
    :type min_impurity_decrease: float
    :param feature_map: 字段名到中文含义的字典，默认 ``None``；用于图形标题及规则报告的字段说明。
    :type feature_map: dict[str, str] or None
    :param n_jobs: 节点规则报告的并行预算，默认 -1 自动调度；树搜索按确定顺序串行执行。
        1 表示串行；其他取值遵循 hscredit 统一并行策略。
    :type n_jobs: int or float or None
    :param parallel_backend: 规则报告的 joblib 后端，默认 ``None`` 使用报告默认后端。
    :type parallel_backend: str or None
    :param parallel_config: 规则报告的并行配置字典，默认 ``None``；支持项见
        :func:`hscredit.utils.parallel.parallel_execute`，不能重复声明 n_jobs 或 backend。
    :type parallel_config: dict or None
    :param missing_separate: 默认 False，自动学习缺失值合并方向；True 时保留独立缺失分支。
        节点训练样本无缺失时默认走最左分支；显式 missing_branch 或规则中的缺失标记优先。
    :type missing_separate: bool

    **属性**

    - ``nodes_``：节点编号到 ``MultiwayNode`` 的映射，根节点编号为 0。
      ``children`` 按从左到右的分支顺序存储子节点编号；``missing_branch`` 是其中的位置，
      不是节点编号。通过编辑方法修改树，避免直接修改节点破坏规则与路由的一致性。
    - ``classes_``：训练标签的排序后唯一值，与 ``predict_proba`` 的列顺序一致。
    - ``feature_importances_``：与 ``features_`` 对齐的归一化加权不纯度下降；无分裂时全为 0。
    - ``features_`` / ``feature_names_in_``：拟合字段列表及对应的 NumPy 数组。
    - ``n_features_in_``：拟合特征数。

    **参考样例**

    >>> tree = MultiwayTreeExtractor(target='FPD', features=['评分'], max_branches=4).fit(df)
    >>> tree.manual_split(feature='评分', thresholds=[500, 650, 750], node=0)
    >>> tree.set_thresholds(0, [520, 660, 760]).grow(node=tree.get_leaf_node_ids()[0])
    >>> tree.prune(node=tree.get_leaf_node_ids()[0])
    >>> rules = tree.get_rules()
    >>> effects = tree.evaluate({'测试集': df_test})
    """

    # 旧版持久化对象没有此参数时，继续沿用原有的缺失合并策略。
    missing_separate = False

    def __init__(
        self,
        target="target",
        features=None,
        max_depth=3,
        max_branches=4,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion="chi2",
        alpha=0.05,
        adjust_pvalue=True,
        max_prebins=12,
        max_candidates=5000,
        min_impurity_decrease=0.0,
        feature_map=None,
        n_jobs=-1,
        parallel_backend=None,
        parallel_config=None,
        missing_separate=False,
    ):
        self.target = target
        self.features = features
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.alpha = alpha
        self.adjust_pvalue = adjust_pvalue
        self.max_prebins = max_prebins
        self.max_candidates = max_candidates
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_map = feature_map
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config
        self.missing_separate = missing_separate

    def _validate_parameters(self):
        self._validate_missing_options(getattr(self, "missing_separate", False), None)
        if getattr(self, "missing_separate", False) is None:
            raise InputValidationError("构造参数 missing_separate 必须是布尔值")
        for name, minimum in (
            ("max_depth", 0),
            ("max_branches", 2),
            ("min_samples_split", 2),
            ("min_samples_leaf", 1),
            ("max_prebins", 2),
            ("max_candidates", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise InputValidationError(f"{name} 必须是大于等于 {minimum} 的整数")
        if self.criterion not in ("chi2", "gini", "entropy"):
            raise InputValidationError("criterion 必须是 chi2、gini 或 entropy")
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise InputValidationError("alpha 必须在 (0, 1] 内")
        if not np.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0:
            raise InputValidationError("min_impurity_decrease 必须是非负有限数值")
        if not isinstance(self.target, str):
            raise InputValidationError("target 必须是标签列名")

    def fit(self, X, y=None, features=None):
        """从根节点重新训练数值/类别多叉树，支持二分类和多分类。

        数值 dtype（不含布尔型）按连续变量处理，其他 dtype 按类别处理。
        保存输入数据副本用于后续节点编辑和训练集报告；再次成功调用会替换整棵树，
        节点重新从 0 编号，训练失败则保留原模型。

        :param X: 训练 DataFrame 或二维数组。DataFrame 可含金额、日期等报告字段；
            用 ``features`` 指定参与分裂的列。数组按给定特征名命名，未提供时使用
            ``特征_0``、``特征_1`` 等，通常需同时提供 y。
        :type X: pandas.DataFrame or array-like
        :param y: 一维标签，默认 ``None`` 从 X 的 ``target`` 列读取；显式传入时优先。
            标签按行位置对应，长度须与 X 一致且不能缺失。支持单一类别，此时不自动分裂。
        :type y: array-like or None
        :param features: 本次使用的特征列名或列表，默认 ``None`` 沿用构造配置；
            两处均为 ``None`` 时使用除目标列外的全部列。数组输入时名称数须等于列数。
        :type features: str or list[str] or None
        :return: 训练后的当前实例，支持链式调用。
        :rtype: MultiwayTreeExtractor
        :raises InputValidationError: 模型参数、特征或训练数据不符合约束。
        :raises ValueError: 未提供标签、标签维度或长度错误等输入校验失败。

        **参考样例**

        >>> tree = MultiwayTreeExtractor(target='FPD', features=['评分']).fit(df_train)
        >>> tree = MultiwayTreeExtractor().fit(df_train[['评分']], y=df_train['FPD'])
        """
        working = copy.copy(self)
        working._validate_parameters()
        selected = normalize_feature_names(features if features is not None else self.features, "features")
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not isinstance(X, pd.DataFrame):
            frame.columns = selected or [f"特征_{i}" for i in range(frame.shape[1])]
        frame, labels = extract_target(frame, y, self.target, require_y=True)
        selected = selected if selected is not None else list(frame.columns)
        if not selected or self.target in selected or len(set(selected)) != len(selected):
            raise InputValidationError("特征不能为空、重复或包含目标列")
        if not frame.columns.is_unique or any(name not in frame for name in selected):
            raise InputValidationError("特征列不存在或输入列名重复")
        for name in selected:
            if not isinstance(name, str):
                raise InputValidationError("特征列名必须是字符串")
            field_expression(name)
        if len(frame) == 0 or pd.isna(labels).any():
            raise InputValidationError("训练数据不能为空且目标不能缺失")
        working.features_ = list(selected)
        working.feature_names_in_ = np.asarray(selected, dtype=object)
        working.n_features_in_ = len(selected)
        working.classes_, working._y = np.unique(np.asarray(labels), return_inverse=True)
        # 保留金额、日期和逾期字段供训练集报告使用；搜索/预测仅访问 features_。
        working._data = frame.reset_index(drop=True).copy()
        working._data[self.target] = np.asarray(labels)
        working._numeric = {
            name: pd.api.types.is_numeric_dtype(frame[name]) and not pd.api.types.is_bool_dtype(frame[name])
            for name in selected
        }
        working.nodes_ = {}
        working._next_id = 0
        working._new_node(None, 0, (), np.arange(len(frame)))
        working._grow(0)
        working._refresh_importances()
        self.__dict__.update(working.__dict__)
        return self

    def _check_fitted(self):
        if not hasattr(self, "nodes_"):
            raise NotFittedError("请先调用 fit() 训练多叉树")

    def _node(self, node):
        self._check_fitted()
        if isinstance(node, bool) or not isinstance(node, Integral) or node not in self.nodes_:
            raise InputValidationError(f"节点 {node} 不存在")
        return self.nodes_[node]

    def _new_node(self, parent, depth, path, positions):
        node_id = self._next_id
        self._next_id += 1
        self.nodes_[node_id] = MultiwayNode(
            node_id, parent, depth, np.bincount(self._y[positions], minlength=len(self.classes_)), path
        )
        return node_id

    def _positions(self, node):
        mask = np.ones(len(self._data), dtype=bool)
        for branch in self.nodes_[node].path:
            mask &= branch.mask(self._data).to_numpy(dtype=bool)
        return np.flatnonzero(mask)

    def _numeric_pool(self, values):
        finite = values[np.isfinite(values)]
        if len(np.unique(finite)) < 2:
            return ()
        # 复用等频分箱器的无监督切点计算；不构造虚假的二分类分箱统计表。
        values = pd.Series(finite)
        binner = QuantileBinning(
            max_n_bins=self.max_prebins,
            min_n_bins=1,
            min_bin_size=None,
            force_numerical=True,
            decimal=16,
            n_jobs=1,
        )
        cuts = binner._fit_numerical(values, pd.Series(np.zeros(len(values))))
        return tuple(float(v) for v in cuts if finite.min() < v < finite.max())

    @staticmethod
    def _validate_missing_options(missing_separate, missing_branch):
        """缺失配置使用明确布尔值和从 0 开始的分支序号。"""
        if missing_separate is not None and not isinstance(missing_separate, (bool, np.bool_)):
            raise InputValidationError("missing_separate 必须是布尔值或 None")
        if missing_branch is not None and (
            isinstance(missing_branch, (bool, np.bool_))
            or not isinstance(missing_branch, Integral)
            or missing_branch < 0
        ):
            raise InputValidationError("missing_branch 必须是非负整数分支序号")

    @staticmethod
    def _parse_thresholds(feature, thresholds):
        """与 binner 共用数值规则解析；缺失标记的位置表示合并目标分支。"""
        try:
            cuts, missing = parse_numerical_user_splits(feature, thresholds)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        if cuts.ndim != 1:
            raise InputValidationError("thresholds 必须是一维数值切点列表，可包含一个缺失标记")
        return cuts, missing

    def _candidate(
        self,
        values,
        labels,
        feature,
        thresholds=(),
        groups=(),
        missing_branch=None,
        enforce=True,
        missing_separate=False,
    ):
        missing = pd.isna(values)
        if self._numeric[feature]:
            branches = np.searchsorted(thresholds, np.asarray(values, dtype=float), side="left")
            count = len(thresholds) + 1
            if missing_separate:
                missing_branch = count
                count += 1
        else:
            branches = np.full(len(values), -1, dtype=int)
            for i, group in enumerate(groups):
                branches[pd.Series(values).isin(group).to_numpy()] = i
            count = len(groups)
            missing = missing | (branches < 0)
            if missing_separate and missing_branch is None:
                missing_branch = count
                groups = tuple(groups) + ((),)
                count += 1
        if missing_branch is not None and missing_branch >= count:
            return None
        choices = range(count) if missing_branch is None and missing.any() else [missing_branch]
        if not missing.any() and missing_branch is None:
            # 节点内没有缺失样本时无法学习方向，为未来缺失保留稳定的最左路由。
            choices = [0]
        best = None
        for destination in choices:
            routed = branches.copy()
            routed[missing] = destination
            counts = np.bincount(routed * len(self.classes_) + labels, minlength=count * len(self.classes_))
            counts = counts.reshape(count, len(self.classes_))
            ordinary = np.arange(count) != destination if missing_separate else np.ones(count, dtype=bool)
            if enforce and np.any(counts.sum(axis=1)[ordinary] < self.min_samples_leaf):
                continue
            statistic, pvalue, gain = _split_statistics(counts, self.criterion)
            degrees = (np.count_nonzero(counts.sum(axis=0)) - 1) * (np.count_nonzero(counts.sum(axis=1)) - 1)
            worth = -_chi2_log_pvalue(statistic, degrees) / np.log(10) if self.criterion == "chi2" else gain
            candidate = dict(
                feature=feature,
                thresholds=tuple(thresholds),
                groups=tuple(groups),
                missing_branch=int(destination),
                missing_separate=bool(missing_separate),
                unknown_branch=(
                    None
                    if self._numeric[feature]
                    else (
                        int(np.argmax(np.where(ordinary, counts.sum(axis=1), -1)))
                        if missing_separate
                        else int(destination)
                    )
                ),
                statistic=statistic,
                pvalue=pvalue,
                gain=gain,
                worth=worth,
                branches=count,
            )
            if best is None or candidate["worth"] > best["worth"] + 1e-14:
                best = candidate
        return best

    def _search(self, node, missing_separate=None, missing_branch=None):
        separate = getattr(self, "missing_separate", False) if missing_separate is None else missing_separate
        separate = bool(separate and missing_branch is None)
        positions = self._positions(node)
        if len(positions) < self.min_samples_split or np.unique(self._y[positions]).size < 2:
            return []
        labels = self._y[positions]
        candidates = []
        for feature in self.features_:
            series = self._data.iloc[positions][feature]
            values = series.to_numpy(dtype=float, na_value=np.nan) if self._numeric[feature] else series.to_numpy()
            feature_candidates = []
            trials = 0

            def consider(thresholds=(), groups=()):
                nonlocal trials
                normal_count = len(thresholds) + 1 if self._numeric[feature] else len(groups)
                trials += normal_count if series.isna().any() and not separate and missing_branch is None else 1
                result = self._candidate(
                    values,
                    labels,
                    feature,
                    thresholds,
                    groups,
                    missing_branch,
                    missing_separate=separate,
                )
                if result is not None:
                    feature_candidates.append(result)
                return result

            if self._numeric[feature]:
                pool = self._numeric_pool(values)
                limit = min(self.max_branches - 1 - int(separate), len(pool))
                if separate:
                    consider()
                if sum(comb(len(pool), k) for k in range(1, limit + 1)) <= self.max_candidates:
                    for k in range(1, limit + 1):
                        for cuts in combinations(pool, k):
                            consider(thresholds=cuts)
                else:
                    cuts = ()
                    for _ in range(limit):
                        choices = [consider(thresholds=tuple(sorted(cuts + (v,)))) for v in pool if v not in cuts]
                        choices = [choice for choice in choices if choice is not None]
                        if not choices:
                            break
                        cuts = max(choices, key=lambda item: item["worth"])["thresholds"]
            else:
                # 类别从单组开始，以 Pearson 相似度合并最接近的组；不强加类别顺序。
                groups = [(value,) for value in pd.unique(series.dropna())]
                for group in groups:
                    value_expression(group[0])
                while len(groups) >= (1 if separate else 2):
                    if len(groups) + int(separate) <= self.max_branches:
                        consider(groups=tuple(groups))
                    if len(groups) == (1 if separate else 2):
                        break
                    counts = [
                        np.bincount(labels[series.isin(group).to_numpy()], minlength=len(self.classes_))
                        for group in groups
                    ]
                    pair = max(
                        combinations(range(len(groups)), 2),
                        key=lambda ij: _split_statistics([counts[ij[0]], counts[ij[1]]], "gini")[1],
                    )
                    i, j = pair
                    groups[i] += groups[j]
                    del groups[j]
            for candidate in feature_candidates:
                candidate["adjusted_pvalue"] = min(1.0, candidate["pvalue"] * (trials if self.adjust_pvalue else 1))
                if self.criterion == "chi2":
                    candidate["worth"] = max(
                        0.0, candidate["worth"] - (np.log10(trials) if self.adjust_pvalue else 0.0)
                    )
            if feature_candidates:
                candidates.append(max(feature_candidates, key=lambda item: (item["worth"], -item["branches"])))
        return sorted(candidates, key=lambda item: item["worth"], reverse=True)

    def get_split_candidates(self, node=0):
        """查看指定节点各特征的最佳分裂方案，不修改模型。

        每个可用特征最多一行，按分裂价值降序排列。搜索执行样本数和分支数约束，
        但不按节点深度、``alpha`` 或 ``min_impurity_decrease`` 过滤返回结果；
        因此有候选不代表 ``grow`` 一定会采用。缺失策略使用当前构造参数。

        :param node: 现有节点编号，默认 0，可查询叶节点或内部节点。
        :type node: int
        :return: 中文候选表。列为分裂特征、切分点、类别分组、分支数、缺失分支、
            缺失独立分支、卡方值、P值、调整后P值、分裂增益、分裂价值。
            缺失分支为从 0 开始的分支位置；无可用候选时返回保留这些列的空表。
            ``chi2`` 的分裂价值为调整后的 LogWorth，其他准则为不纯度下降。
        :rtype: pandas.DataFrame
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点编号无效。

        **参考样例**

        >>> candidates = tree.get_split_candidates(node=0)
        """
        self._node(node)
        columns = [
            "分裂特征",
            "切分点",
            "类别分组",
            "分支数",
            "缺失分支",
            "缺失独立分支",
            "卡方值",
            "P值",
            "调整后P值",
            "分裂增益",
            "分裂价值",
        ]
        return pd.DataFrame(
            [
                [
                    c["feature"],
                    list(c["thresholds"]),
                    [list(g) for g in c["groups"]],
                    c["branches"],
                    c["missing_branch"],
                    "是" if c["missing_separate"] else "否",
                    c["statistic"],
                    c["pvalue"],
                    c["adjusted_pvalue"],
                    c["gain"],
                    c["worth"],
                ]
                for c in self._search(node)
            ],
            columns=columns,
        )

    def _allowed(self, candidate, size):
        return (
            candidate["gain"] > 1e-14
            and candidate["gain"] * size / len(self._data) >= self.min_impurity_decrease
            and (self.criterion != "chi2" or candidate["adjusted_pvalue"] <= self.alpha)
        )

    def _grow(self, node):
        current = self.nodes_[node]
        if current.depth >= self.max_depth:
            return
        if not current.children:
            candidates = self._search(node)
            if not candidates or not self._allowed(candidates[0], current.counts.sum()):
                return
            self._install_split(node, candidates[0], manual=False)
        for child in list(current.children):
            self._grow(child)

    def _install_split(self, node, candidate, manual):
        current = self.nodes_[node]
        self._prune(node)
        current.feature = candidate["feature"]
        current.thresholds = candidate["thresholds"]
        current.category_groups = candidate["groups"]
        current.missing_branch = candidate["missing_branch"]
        current.missing_separate = candidate.get("missing_separate", False)
        current.unknown_branch = candidate.get("unknown_branch", current.missing_branch)
        current.gain = candidate["gain"]
        current.manual = manual
        specs = []
        known = tuple(v for group in current.category_groups for v in group)
        for i in range(candidate["branches"]):
            kwargs = dict(include_missing=i == current.missing_branch)
            if current.missing_separate and i == current.missing_branch:
                spec = BinSpec(current.feature, i, kind="missing")
            elif self._numeric[current.feature]:
                spec = BinSpec(
                    current.feature,
                    i,
                    lower=None if i == 0 else current.thresholds[i - 1],
                    upper=None if i == len(current.thresholds) else current.thresholds[i],
                    closed_left=False,
                    closed_right=True,
                    **kwargs,
                )
            else:
                spec = BinSpec(
                    current.feature,
                    i,
                    kind="categories",
                    values=current.category_groups[i],
                    known_values=known,
                    unknown=i == current.unknown_branch,
                    **kwargs,
                )
            specs.append(spec)
            path = current.path + (spec,)
            positions = self._positions(node)
            positions = positions[spec.mask(self._data.iloc[positions]).to_numpy(dtype=bool)]
            current.children.append(self._new_node(node, current.depth + 1, path, positions))
        current.branches = tuple(specs)

    def _prune(self, node):
        current = self.nodes_[node]
        for child in list(current.children):
            self._prune(child)
            del self.nodes_[child]
        current.children = []
        current.branches = ()
        current.feature = None
        current.thresholds = ()
        current.category_groups = ()
        current.missing_branch = 0
        current.missing_separate = False
        current.unknown_branch = None
        current.gain = 0.0

    def _edit(self, operation):
        self._check_fitted()
        working = copy.copy(self)
        working.nodes_ = copy.deepcopy(self.nodes_)
        operation(working)
        working._refresh_importances()
        self.nodes_ = working.nodes_
        self._next_id = working._next_id
        self.feature_importances_ = working.feature_importances_
        return self

    def manual_split(
        self,
        data=None,
        feature=None,
        thresholds=None,
        node=0,
        *,
        category_groups=None,
        missing_branch=None,
        missing_separate=None,
        grow=False,
    ):
        """人工指定或选择候选分裂，替换所选节点的全部后代。

        指定节点及其他子树的编号保持不变，新后代使用新编号；失败时保留原树。
        显式切点/分组允许空分支，不受 ``min_samples_split``、``min_samples_leaf``、
        ``alpha`` 和 ``min_impurity_decrease`` 限制，但仍受最大深度及总分支数限制。
        空分支计数为 0，预测概率继承最近的非空祖先。

        :param data: 默认 ``None`` 使用拟合时保存的数据。若提供 DataFrame，其特征和目标列
            重置索引后必须与拟合数据一致；此参数不能用来替换训练集，更换样本需重新 ``fit``。
        :type data: pandas.DataFrame or None
        :param feature: 分裂字段名，默认 ``None``。给定切点或分组时必须指定，且须属于 ``features_``。
            两类规则均未提供时，可用此参数限制自动候选的字段；不指定则在所有特征中选择。
        :type feature: str or None
        :param thresholds: 数值特征的严格递增有限切点序列，默认 ``None``。
            k 个有限切点产生 k+1 个普通分支，区间左开右闭，等于切点走左侧区间。
            可插入一个 ``np.nan``、``None`` 或 ``pd.NA``，其位置表示缺失合并分支，
            例如 ``[0.12, np.nan, 0.18]`` 合并到分支 1。空列表配合独立缺失可仅分成非缺失/缺失两支。
        :type thresholds: sequence or None
        :param node: 要替换子树的现有节点编号，默认 0；既可为叶节点，也可为内部节点。
        :type node: int
        :param category_groups: 类别特征分组，默认 ``None``。使用非空、互不重复且完整覆盖该节点
            训练类别的嵌套列表，顺序即分支顺序。``[["A"], [np.nan], ["B", "C"]]``
            声明独立缺失分支；``[["A", np.nan], ["B", "C"]]`` 将缺失并入分支 0。
            与 thresholds 不可同时指定。两者均为 ``None`` 时选择满足样本约束的最佳候选，
            无需通过自动生长的显著性和增益门槛。
        :type category_groups: list[list] or None
        :param missing_branch: 缺失合并到的普通分支位置，从 0 开始，不是子节点编号。
            默认 ``None``：有合并标记则采用标记位置；否则按独立缺失配置或学习合并方向处理。
            无缺失训练样本且未指定策略时走分支 0。显式值优先于合并标记及 missing_separate；
            与指向另一位置的独立类别缺失组冲突时会报错。
        :type missing_branch: int or None
        :param missing_separate: 默认 ``None`` 沿用构造配置；``True`` 保留独立缺失分支，
            数值型追加在末尾，类别独立组保持声明位置。计入 max_branches，训练无缺失也保留空分支。
            显式 missing_branch 或合并标记可覆盖此配置。未知类别在合并模式下与缺失同路，
            在独立模式下走样本最多的普通类别分支。
        :type missing_separate: bool or None
        :param grow: 默认 ``False`` 仅创建本次分裂；``True`` 在新子树上按自动规则继续生长。
        :type grow: bool
        :return: 修改后的当前实例，支持链式调用。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点/特征无效、规则或缺失配置冲突、超出深度/分支限制，或无可用候选。

        **参考样例**

        >>> _ = tree.manual_split(feature='评分', thresholds=[500, 650, 750], missing_branch=0)
        >>> _ = tree.manual_split(feature='评分', thresholds=[500, np.nan, 650, 750])
        >>> _ = tree.manual_split(feature='评分', node=0, grow=True)
        """
        self._validate_parameters()
        self._validate_missing_options(missing_separate, missing_branch)
        separate = getattr(self, "missing_separate", False) if missing_separate is None else missing_separate
        current = self._node(node)
        if current.depth >= self.max_depth:
            raise InputValidationError("该节点已达到 max_depth，请提高深度上限后再分裂")
        if data is not None and not data[self.features_ + [self.target]].reset_index(drop=True).equals(
            self._data[self.features_ + [self.target]]
        ):
            raise InputValidationError("节点编辑使用拟合数据；如需更换训练样本，请重新 fit")
        if feature is not None and feature not in self.features_:
            raise InputValidationError(f"特征 {feature} 不在拟合字段中")
        if thresholds is None and category_groups is None:
            if missing_branch is not None and missing_branch >= self.max_branches:
                raise InputValidationError("missing_branch 超出 max_branches 允许的分支范围")
            candidates = [
                c for c in self._search(node, separate, missing_branch) if feature is None or c["feature"] == feature
            ]
            if not candidates:
                raise InputValidationError("该节点没有满足样本约束的可用分裂")
            return self._edit(lambda tree: tree._split_and_grow(node, candidates[0], grow))
        if feature is None:
            raise InputValidationError("手工分裂必须指定 feature")
        positions = self._positions(node)
        if len(positions) == 0:
            raise InputValidationError("空节点无法继续分裂")
        series = self._data.iloc[positions][feature]
        if self._numeric[feature]:
            if category_groups is not None or thresholds is None:
                raise InputValidationError("数值特征请提供 thresholds，不能提供 category_groups")
            cuts, marker = self._parse_thresholds(feature, thresholds)
            if missing_branch is None:
                missing_branch = marker
            separate = bool(separate and missing_branch is None)
            groups = ()
            count = len(cuts) + 1 + int(separate)
            values = series.to_numpy(dtype=float, na_value=np.nan)
        else:
            if thresholds is not None or category_groups is None:
                raise InputValidationError("类别特征请提供 category_groups，不能提供 thresholds")
            try:
                normalized = normalize_user_groups(feature, category_groups, observed=series)
            except ValueError as exc:
                raise InputValidationError(f"类别分组必须非空、互不重复且完整覆盖训练类别：{exc}") from exc
            marker = next((i for i, group in enumerate(normalized) if any(is_missing_marker(v) for v in group)), None)
            groups = tuple(tuple(value for value in group if not is_missing_marker(value)) for group in normalized)
            flat = [value for group in groups for value in group]
            if len(set(flat)) != len(flat):
                raise InputValidationError("类别分组不能包含相等或重复的类别")
            for value in flat:
                value_expression(value)
            if marker is not None and not groups[marker]:
                if missing_branch is not None and missing_branch != marker:
                    raise InputValidationError("独立缺失分组与 missing_branch 冲突；合并时请将缺失标记放入普通组")
                separate = True
                missing_branch = marker
            elif missing_branch is not None or marker is not None:
                separate = False
                missing_branch = marker if missing_branch is None else missing_branch
            elif separate:
                missing_branch = len(groups)
                groups += ((),)
            cuts = ()
            count = len(groups)
            values = series.to_numpy()
        if count > self.max_branches:
            raise InputValidationError("总分支数（含独立缺失分支）超过 max_branches，请增加上限或减少普通切分点")
        if count < 2:
            raise InputValidationError("分裂至少需要两个分支；仅区分缺失与非缺失可指定 missing_separate=True")
        if missing_branch is not None and (
            isinstance(missing_branch, bool)
            or not isinstance(missing_branch, Integral)
            or not 0 <= missing_branch < count
        ):
            raise InputValidationError("missing_branch 必须是有效的分支序号")
        candidate = self._candidate(
            values,
            self._y[positions],
            feature,
            tuple(cuts),
            groups,
            missing_branch,
            enforce=False,
            missing_separate=separate,
        )
        return self._edit(lambda tree: tree._split_and_grow(node, candidate, grow))

    def _split_and_grow(self, node, candidate, grow):
        self._install_split(node, candidate, manual=True)
        if grow:
            self._grow(node)

    def set_thresholds(self, node, thresholds, *, missing_branch=None, missing_separate=None, grow=False):
        """调整已有数值分裂节点的切点并重建其后代，保留节点自身编号。

        分裂字段保持不变；更换字段或编辑类别分组请使用 ``manual_split``。
        未提供缺失参数且切点中无缺失标记时，保留该节点原有路由；若原来为独立缺失，
        独立分支会随普通分支数移动到末尾。显式传入 missing_separate 会重新解析策略，
        ``False`` 且没有合并目标时重新学习方向，节点训练无缺失则选择分支 0。

        :param node: 已有数值分裂节点编号，无默认值；叶节点或类别分裂节点不适用。
        :type node: int
        :param thresholds: 新的严格递增有限切点序列，可含一个缺失标记；规则与 ``manual_split`` 相同。
        :type thresholds: sequence
        :param missing_branch: 默认 ``None``；显式值指定新的合并分支位置，并优先于切点中的标记。
            减少切点后，若保留的原分支位置越界，必须重新指定有效位置或独立缺失策略。
        :type missing_branch: int or None
        :param missing_separate: 默认 ``None``；没有新合并目标时保留节点原策略。
            显式布尔值设置新的独立/合并策略，含义见 ``manual_split``。
        :type missing_separate: bool or None
        :param grow: 默认 ``False``；``True`` 在重建后继续自动生长新子树。
        :type grow: bool
        :return: 修改后的当前实例；失败时保留原树。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点类型不适用、切点/缺失配置无效，或分裂约束不满足。

        **参考样例**

        >>> _ = tree.set_thresholds(0, [520, 660, 760], missing_branch=0)
        """
        current = self._node(node)
        if current.feature is None or not self._numeric[current.feature]:
            raise InputValidationError("set_thresholds 仅适用于已有数值分裂节点")
        cuts, marker = self._parse_thresholds(current.feature, thresholds)
        if missing_branch is None:
            missing_branch = marker
        if missing_branch is None and missing_separate is None:
            missing_separate = getattr(current, "missing_separate", False)
            if not missing_separate:
                missing_branch = current.missing_branch
        return self.manual_split(
            feature=current.feature,
            thresholds=cuts,
            node=node,
            missing_branch=missing_branch,
            missing_separate=missing_separate,
            grow=grow,
        )

    def prune(self, node=0):
        """将指定节点剪为叶子，删除其全部后代并更新特征重要性。

        保留节点编号、训练类别计数和到达该节点的路径；清除其分裂字段及分支。
        不影响其他子树。剪枝后的预测概率由该节点的训练计数决定，空节点继承非空祖先概率。

        :param node: 现有节点编号，默认 0；对根节点剪枝后整棵树只剩一个叶子。
            已是叶节点时结构不变，仍将该节点标记为人工编辑。
        :type node: int
        :return: 剪枝后的当前实例。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点编号无效。

        **参考样例**

        >>> child = tree.nodes_[0].children[0]
        >>> _ = tree.prune(node=child)
        """
        self._node(node)

        def operation(tree):
            tree._prune(node)
            tree.nodes_[node].manual = True

        return self._edit(operation)

    def delete_node(self, node=0):
        """``prune`` 的兼容入口：删除后代，保留指定节点并将其变为叶子。

        :param node: 现有节点编号，默认 0。
        :type node: int
        :return: 剪枝后的当前实例，与 ``prune(node)`` 相同。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点编号无效。
        """
        return self.prune(node)

    def grow(self, node=0):
        """使用拟合时保存的数据和当前参数，继续生长指定节点下的叶子。

        保留已有分裂、节点编号及缺失路由；对内部节点递归处理后代叶子。
        新分裂遵守最大深度、分支数、样本量、显著性及增益约束，并使用当前构造参数的缺失策略。
        无合格候选、空节点、纯类别节点或达到深度上限时不新增分裂。

        :param node: 生长起点的现有节点编号，默认 0，可为内部节点或叶节点。
        :type node: int
        :return: 更新后的当前实例，重新计算特征重要性；失败时保留原树。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 节点编号或当前模型参数无效。

        **参考样例**

        >>> _ = tree.set_params(max_depth=4).grow(node=0)
        """
        self._node(node)
        self._validate_parameters()
        return self._edit(lambda tree: tree._grow(node))

    def _refresh_importances(self):
        importance = np.zeros(len(self.features_))
        for current in self.nodes_.values():
            if current.children:
                importance[self.features_.index(current.feature)] += current.counts.sum() * current.gain
        self.feature_importances_ = importance / importance.sum() if importance.sum() else importance

    def get_leaf_node_ids(self):
        """取得当前所有叶节点编号，包含训练计数为 0 的叶节点。

        :return: 按节点创建顺序返回的编号列表，与 ``get_rules(leaf_only=True)`` 一一对应。
            编号可能不连续，也不保证按图上的左右顺序排列；局部编辑后应重新获取。
        :rtype: list[int]
        :raises NotFittedError: 尚未训练。
        """
        self._check_fitted()
        return [node for node, current in self.nodes_.items() if not current.children]

    def _frame(self, X):
        self._check_fitted()
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.features_)
        if any(name not in frame for name in self.features_):
            raise InputValidationError("预测数据缺少拟合特征")
        return frame

    def apply(self, X):
        """将每行样本路由到一个叶节点，使用模型保存的缺失/未知类别规则。

        :param X: 待预测 DataFrame 或二维数组。DataFrame 必须包含全部 ``features_``，
            列顺序可不同且允许额外列；数组须按 ``features_`` 顺序排列，不需要目标列。
        :type X: pandas.DataFrame or array-like
        :return: 形状为 ``(n_samples,)`` 的整数数组，顺序与输入行一致，值为叶节点编号。
            返回值不是分支位置；输入索引不会附加到数组中。
        :rtype: numpy.ndarray
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 输入缺少拟合特征，或存在未被分支覆盖的样本。

        **参考样例**

        >>> leaf_ids = tree.apply(df_test)
        >>> selected = df_test.loc[leaf_ids == tree.get_leaf_node_ids()[0]]
        """
        frame = self._frame(X)
        result = np.full(len(frame), -1, dtype=int)
        stack = [(0, np.arange(len(frame)))]
        while stack:
            node, positions = stack.pop()
            current = self.nodes_[node]
            if not current.children:
                result[positions] = node
                continue
            for child, spec in zip(current.children, current.branches):
                mask = spec.mask(frame.iloc[positions]).to_numpy(dtype=bool)
                stack.append((child, positions[mask]))
        if np.any(result < 0):
            raise InputValidationError("存在未被树分支覆盖的样本")
        return result

    def _probabilities(self, node):
        current = self.nodes_[node]
        while current.counts.sum() == 0:
            current = self.nodes_[current.parent]
        return current.counts / current.counts.sum()

    def predict_proba(self, X):
        """返回叶节点训练类别占比作为预测概率。

        普通叶子按训练计数归一化；人工空分支或预留的空缺失分支继承最近非空祖先的概率。
        修改切点、分裂或剪枝后，概率随新的节点训练计数更新。

        :param X: 待预测 DataFrame 或二维数组，特征要求同 ``apply``，不需要目标列。
        :type X: pandas.DataFrame or array-like
        :return: 形状为 ``(n_samples, n_classes)`` 的浮点数组，第 j 列对应 ``classes_[j]``。
            对训练类别为 0/1 的模型，第 1 列是坏样本概率；其他标签需先确认类别顺序。
        :rtype: numpy.ndarray
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 预测输入不符合特征/路由要求。

        **参考样例**

        >>> probabilities = tree.predict_proba(df_test)
        >>> probability_table = pd.DataFrame(probabilities, columns=tree.classes_, index=df_test.index)
        """
        leaves = self.apply(X)
        return np.asarray([self._probabilities(node) for node in leaves]).reshape(-1, len(self.classes_))

    def predict(self, X):
        """返回每行预测概率最大的类别，概率并列时选择 ``classes_`` 中靠前的类别。

        :param X: 待预测 DataFrame 或二维数组，特征要求同 ``apply``。
        :type X: pandas.DataFrame or array-like
        :return: 形状为 ``(n_samples,)`` 的标签数组，值域和 dtype 与 ``classes_`` 一致。
        :rtype: numpy.ndarray
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 预测输入不符合特征/路由要求。
        """
        probability = self.predict_proba(X)
        return self.classes_[probability.argmax(axis=1)]

    @staticmethod
    def _format_rule(path):
        expression = " & ".join(f"({spec.to_expression()})" for spec in path) or "True"
        description = (
            " 且 ".join(
                f"{spec.feature}: {spec.label(8)}"
                + (" 或缺失" if spec.include_missing else "")
                + (" 或未知类别" if spec.unknown else "")
                for spec in path
            )
            or "全部样本"
        )
        return Rule(expression, description=description)

    def get_rules(self, leaf_only=True):
        """将根到节点的路径导出为可执行的 ``Rule`` 对象。

        叶节点规则在模型输入域内互斥且完整覆盖样本，包括缺失和未知类别路由。
        全部节点规则含父子重叠路径，不能按互斥分区汇总。规则表达式使用精确切点，
        展示标签的舍入不会改变执行边界。

        :param leaf_only: 默认 ``True`` 仅返回叶节点；``False`` 包含根节点及所有内部节点。
        :type leaf_only: bool
        :return: 按节点创建顺序排列的 Rule 列表，名称为 ``MultiwayTree_N{节点编号}``。
            仅叶节点时与 ``get_leaf_node_ids()`` 对齐；根节点规则表达式为 ``True``。
        :rtype: list[Rule]
        :raises NotFittedError: 尚未训练。

        **参考样例**

        >>> rules = tree.get_rules()
        >>> hit = rules[0].predict(df_test)
        """
        self._check_fitted()
        rules = []
        for node, current in self.nodes_.items():
            if leaf_only and current.children:
                continue
            rule = self._format_rule(current.path)
            rule.name = f"MultiwayTree_N{node}"
            rules.append(rule)
        return rules

    def to_ruleset(self, nodes=None):
        """将指定节点的路径规则合成为 OR ``RuleSet``，用于命中筛选。

        :param nodes: 节点编号序列，默认 ``None`` 使用当前全部叶节点。
            可显式选择叶节点或内部节点，内部节点规则覆盖其整个子树；按传入顺序保留规则。
            默认全叶 OR 会命中全部样本，作为拒绝策略时应明确选取风险节点。
        :type nodes: sequence[int] or None
        :return: 名为“多叉树规则集”、逻辑为 OR 的 RuleSet，并沿用当前 ``n_jobs``。
            可调用其 ``evaluate(data)`` 得到整体命中掩码和逐规则命中明细。
        :rtype: hscredit.core.models.rules.RuleSet
        :raises NotFittedError: 尚未训练且需要读取节点。
        :raises InputValidationError: 指定节点编号无效。

        **参考样例**

        >>> strategy = tree.to_ruleset(nodes=[tree.get_leaf_node_ids()[0]])
        >>> matched, details = strategy.evaluate(df_test)
        """
        from ...core.models.rules import RuleSet

        selected = self.get_leaf_node_ids() if nodes is None else list(nodes)
        rules = []
        for node in selected:
            current = self._node(node)
            rule = self._format_rule(current.path)
            rule.name = f"MultiwayTree_N{node}"
            rules.append(rule)
        return RuleSet(name="多叉树规则集", logic="or", rules=rules, n_jobs=self.n_jobs)

    def report(
        self,
        datasets,
        target=None,
        overdue=None,
        dpds=None,
        del_grey=False,
        leaf_only=False,
        *,
        overdue_operator=">",
        **kwargs,
    ):
        """用 ``Rule.report`` 逐节点计算命中效果，支持金额、多标签和多数据集。

        每个节点仅保留其规则报告中的“命中”行，按节点编号排序；不会改变树结构或训练计数。
        0/1 训练标签默认直接作为坏样本标签；其他训练标签默认按各类别对其余类别生成多个二分类标签。

        :param datasets: 待评估 DataFrame、DataFrame 列表/元组，或名称到 DataFrame 的字典。
            需包含路径规则使用的特征，以及目标、逾期、金额等本次报告所需字段。
        :type datasets: pandas.DataFrame or list or tuple or dict
        :param target: 默认 ``None`` 使用数据中的构造参数 target 列及上述标签规则。
            支持列名、一维 Series/ndarray/list、callable 或标签映射；Series 按索引对齐，
            数组/列表按行位置对应，callable 接收当前完整数据集。与 overdue+dpds 模式互斥。
            多数据集调用时对每个数据集应用相同参数，可用列名或 callable 分别取得对应标签。
        :type target: str or array-like or callable or dict or None
        :param overdue: 默认 ``None``；逾期天数字段或字段列表，必须与 dpds 同时提供。
        :type overdue: str or list[str] or None
        :param dpds: 默认 ``None``；逾期阈值标量或非空序列，支持整数或小数，
            按 ``overdue × dpds`` 组合生成标签。
        :type dpds: int or float or sequence or None
        :param del_grey: 默认 ``False``，不剔除灰样本；显式标签中非 0/1 值按 0 计算。
            ``True`` 时逐标签保留有效人群：显式标签只保留 0/1，逾期标签在 ``>`` 时剔除
            ``(0, dpd]``，``>=`` 时剔除 ``(0, dpd)``，``<`` 和 ``<=`` 不额外剔灰。
        :type del_grey: bool
        :param leaf_only: 默认 ``False`` 评估全部节点；``True`` 仅评估叶节点。
            全部节点包含重叠的父子路径，样本数不能直接相加作为数据集总人数。
        :type leaf_only: bool
        :param overdue_operator: 逾期比较符，支持 ``>``、``>=``、``<``、``<=``，默认 ``>``。
            满足比较条件记为坏样本 1，主要在 overdue+dpds 模式下使用。
        :type overdue_operator: str
        :param kwargs: 传给 ``Rule.report`` 的其他参数，例如 ``amount='放款金额'``、
            ``round_digits=False``、``prior_rules``；``feature_map`` 可覆盖构造时的字段含义映射。
            金额口径及指标定义遵循 :meth:`hscredit.core.rules.Rule.report`。
        :return: 单个 DataFrame 输入返回节点效果 DataFrame；列表/元组输入返回表格列表；
            字典输入保留键并返回表格字典。单标签为单层列，多标签为 MultiIndex 列，
            包含节点编号、是否叶子、样本总数、坏样本数、坏样本率及规则效果指标。
        :rtype: pandas.DataFrame or list[pandas.DataFrame] or dict
        :raises NotFittedError: 尚未训练。
        :raises ValueError: 标签、逾期参数或金额数据不符合规则报告的输入约束。

        **参考样例**

        >>> leaf_report = tree.report(df_test, target='FPD', leaf_only=True)
        >>> overdue_report = tree.report(
        ...     df_test, overdue=['MOB1'], dpds=[7, 3, 0], amount='放款金额', leaf_only=True
        ... )
        >>> reports = tree.report({'训练集': df_train, '测试集': df_test}, leaf_only=True)
        """
        self._check_fitted()
        validate_overdue_operator(overdue_operator)
        kwargs["overdue_operator"] = overdue_operator
        if isinstance(datasets, dict):
            return {
                key: self.report(value, target, overdue, dpds, del_grey, leaf_only, **kwargs)
                for key, value in datasets.items()
            }
        if isinstance(datasets, (list, tuple)):
            return [self.report(value, target, overdue, dpds, del_grey, leaf_only, **kwargs) for value in datasets]
        if target is None and overdue is None:
            if set(self.classes_).issubset({0, 1}):
                target = self.target
            else:
                target = {f"类别_{value}": datasets[self.target].eq(value).astype(int) for value in self.classes_}
        rules = pd.DataFrame(
            [
                dict(node=node, if_leaf=not current.children, rule_list=current.path)
                for node, current in self.nodes_.items()
            ]
        )
        feature_map = resolve_feature_map(kwargs.pop("feature_map", self.feature_map))
        result = _node_hit_report(
            rules,
            self._format_rule,
            datasets,
            target,
            overdue,
            dpds,
            del_grey,
            leaf_only,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
            parallel_config=self.parallel_config,
            **kwargs,
        )
        if feature_map is not None:
            multi = isinstance(result.columns, pd.MultiIndex)
            node_col = ("分箱详情", "节点编号") if multi else "节点编号"
            context = {
                node: format_feature_context([spec.feature for spec in current.path], feature_map)
                for node, current in self.nodes_.items()
            }
            for index, name in enumerate(("入参字段", "字段含义")):
                column = ("分箱详情", name) if multi else name
                result[column] = result[node_col].map(lambda node: context[int(node)][index])
        return result

    def get_rule_table(self, data=None, leaf_only=False, **kwargs):
        """``report`` 的便捷入口，默认使用拟合时保存的训练数据。

        :param data: 默认 ``None`` 使用训练数据；也可传入 ``report`` 接受的 DataFrame、
            表格列表/元组或名称到表格的字典。
        :type data: pandas.DataFrame or list or tuple or dict or None
        :param leaf_only: 默认 ``False`` 包含全部节点；``True`` 仅输出叶节点。
        :type leaf_only: bool
        :param kwargs: 传给 ``report`` 的参数，例如 target、overdue、dpds、del_grey、
            overdue_operator、amount、feature_map 和 round_digits。
        :return: 与 ``report`` 相同的中文效果表或表格容器；默认返回训练集全部节点的 DataFrame。
        :rtype: pandas.DataFrame or list[pandas.DataFrame] or dict
        :raises NotFittedError: 尚未训练。

        **参考样例**

        >>> leaf_table = tree.get_rule_table(leaf_only=True)
        >>> test_table = tree.get_rule_table(df_test, leaf_only=True)
        """
        self._check_fitted()
        return self.report(self._data if data is None else data, leaf_only=leaf_only, **kwargs)

    def get_node_table(self):
        """查看当前树结构和训练计数，不重新评估外部数据。

        :return: 每节点一行的中文 DataFrame，按节点创建顺序排列。包含节点编号、父节点、深度、
            是否叶子、分裂特征、切分点、类别分组、子节点、缺失分支、缺失独立分支、样本总数、
            规则表达式，以及按 ``classes_`` 展开的“类别_{标签}样本数”列。
            根节点无父节点；叶节点无分裂特征、切点或子节点，其缺失分支字段不用于路由。
            内部节点的缺失分支是 ``子节点`` 列表中的位置；阈值为未舍入的数值。
        :rtype: pandas.DataFrame
        :raises NotFittedError: 尚未训练。

        **参考样例**

        >>> nodes = tree.get_node_table().set_index('节点编号')
        """
        self._check_fitted()
        rows = []
        for node, current in self.nodes_.items():
            row = {
                "节点编号": node,
                "父节点": current.parent,
                "深度": current.depth,
                "是否叶子": "否" if current.children else "是",
                "分裂特征": current.feature,
                "切分点": list(current.thresholds),
                "类别分组": [list(g) for g in current.category_groups],
                "子节点": list(current.children),
                "缺失分支": current.missing_branch,
                "缺失独立分支": "是" if getattr(current, "missing_separate", False) else "否",
                "样本总数": int(current.counts.sum()),
                "规则表达式": self._format_rule(current.path).expr,
            }
            row.update({f"类别_{label}样本数": int(count) for label, count in zip(self.classes_, current.counts)})
            rows.append(row)
        return pd.DataFrame(rows)

    def evaluate(self, datasets=None, y=None):
        """评估整棵树的分类效果，每个数据集输出一行。

        :param datasets: 默认 ``None`` 评估保存的训练数据；可传入 DataFrame，或非空的
            ``{数据集名: DataFrame}`` 字典。单数据集也支持二维特征数组，但须另传 y。
            外部标签使用 ``evaluate(X, y)`` 两个参数传入，不接受把 ``(X, y)`` 作为一个元组。
        :type datasets: pandas.DataFrame or array-like or dict or None
        :param y: 单数据集的一维真实标签，默认 ``None`` 从数据的 target 列读取。
            按行位置对应，须非空、无缺失、与样本等长且均属于 ``classes_``。
            datasets 为字典时此参数不参与评估，每张表必须包含 target 列。
        :type y: array-like or None
        :return: 中文 DataFrame，列包括数据集、样本总数、准确率、交叉熵、AUC；
            训练类别数为 2 时另含 KS，正类为 ``classes_[1]``。字典输入按键的顺序汇总。
            二分类评估标签只有一类时 AUC 为 NaN、KS 为 0；多分类 AUC 为宏平均 OVR，
            评估数据未覆盖全部训练类别时为 NaN。单类别训练模型的交叉熵为 0、AUC 为 NaN，无 KS 列。
        :rtype: pandas.DataFrame
        :raises NotFittedError: 尚未训练。
        :raises InputValidationError: 标签或预测特征不满足输入要求。

        **参考样例**

        >>> training_effects = tree.evaluate()
        >>> test_effects = tree.evaluate(df_test[tree.features_], y=df_test['FPD'])
        >>> effects = tree.evaluate({'训练集': df_train, '测试集': df_test})
        """
        self._check_fitted()
        if isinstance(datasets, dict):
            return pd.concat(
                [self.evaluate(data).assign(数据集=name) for name, data in datasets.items()], ignore_index=True
            )
        frame = self._data if datasets is None else datasets
        labels = np.asarray(frame[self.target] if y is None else y)
        if (
            len(labels) != len(frame)
            or len(labels) == 0
            or pd.isna(labels).any()
            or not pd.Series(labels).isin(self.classes_).all()
        ):
            raise InputValidationError("评估标签长度必须匹配、非空且属于训练类别")
        probability = self.predict_proba(frame)
        prediction = self.classes_[probability.argmax(axis=1)]
        row = {
            "数据集": "训练集" if datasets is None else "评估集",
            "样本总数": len(frame),
            "准确率": accuracy_score(labels, prediction),
            "交叉熵": 0.0,
            "AUC": np.nan,
        }
        if len(self.classes_) >= 2:
            row["交叉熵"] = log_loss(labels, probability, labels=self.classes_)
            if len(self.classes_) == 2:
                binary = (labels == self.classes_[1]).astype(int)
                row["KS"] = ks(binary, probability[:, 1])
                if np.unique(binary).size == 2:
                    row["AUC"] = roc_auc_score(binary, probability[:, 1])
            elif np.unique(labels).size == len(self.classes_):
                row["AUC"] = roc_auc_score(labels, probability, labels=self.classes_, multi_class="ovr")
        return pd.DataFrame([row])

    def plot(self, backend="matplotlib", save=None, **kwargs):
        """复用 ``plot_tree`` 绘制全部多叉分支和节点类别分布。

        :param backend: 默认 ``'matplotlib'``；也可选择 ``'pyecharts'`` 或 ``'graphviz'``。
        :type backend: str
        :param save: 默认 ``None`` 只返回图形；字符串路径按后端保存为图片、交互式 HTML 或 Graphviz 文件。
            Graphviz 导出图片/PDF 时需要系统安装 Graphviz 程序。
        :type save: str or None
        :param kwargs: 传给 :class:`hscredit.core.viz.tree_plots.DecisionTreeViz` 的绘图配置。
            常用 title、dpi；matplotlib 支持 figsize、show_stats、show_gini，
            pyecharts 支持 width、height、page_title。具体参数以所选后端为准。
        :return: matplotlib Figure、pyecharts 图表对象或 graphviz Source，取决于 backend。
        :rtype: object
        :raises NotFittedError: 尚未训练。
        :raises ValueError: backend 不受支持。

        **参考样例**

        >>> figure = tree.plot(save='多叉树.png', title='规则挖掘')
        >>> chart = tree.plot(backend='pyecharts', save='多叉树.html')
        """
        from ...core.viz import plot_tree

        return plot_tree(self, backend=backend, save=save, **kwargs)

    def display(self):
        """在 Notebook 中展示 matplotlib 树图和训练集全部节点的样式化规则表。

        通过 IPython display 输出图形与表格，图形展示后关闭以释放资源。
        如需指定绘图后端、筛选叶节点或评估外部数据，请分别调用 ``plot`` 和 ``get_rule_table``。

        :return: 当前实例，便于链式调用；图形和表格作为 Notebook 展示输出。
        :rtype: MultiwayTreeExtractor
        :raises NotFittedError: 尚未训练。
        :raises ImportError: 环境缺少 IPython 等展示依赖。

        **参考样例**

        >>> _ = tree.display()
        """
        import matplotlib.pyplot as plt
        from IPython.display import display
        from ...utils import style_rule_table

        figure = self.plot()
        try:
            display(figure)
        finally:
            plt.close(figure)
        display(style_rule_table(self.get_rule_table()))
        return self

    def save(self, path):
        """以 hscredit 制品格式保存完整模型，默认使用 joblib 引擎。

        包含参数、人工修改后的节点、类别顺序、精确阈值、缺失路由，以及拟合时保留的数据，
        加载后可继续编辑并生成训练集报告。父目录不存在时自动创建，通过临时文件替换目标文件。

        :param path: 输出文件路径，支持字符串或 pathlib.Path；已存在文件会被替换。
        :type path: str or pathlib.Path
        :return: 保存路径字符串，不返回模型实例。需要其他引擎或压缩配置时可调用继承的 ``save_artifact``。
        :rtype: str
        :raises NotFittedError: 尚未训练。
        :raises OSError: 无法创建目录或写入文件。

        **参考样例**

        >>> saved_path = tree.save('模型/多叉树.pkl')
        """
        self._check_fitted()
        return self.save_artifact(path)

    @classmethod
    def load(cls, path):
        """从可信来源的模型文件恢复多叉树及其编辑所需状态。

        自动识别序列化引擎并校验对象类型；兼容同类型的旧裸对象。
        默认遇到依赖版本等元数据差异时发出警告，模型仍保留保存时的节点和缺失路由，
        不会按新版本默认值重新训练。加载后可继续 predict、report、manual_split、grow 或 prune。

        :param path: 可信的模型文件路径，与 ``save`` 的输出对应。
        :type path: str or pathlib.Path
        :return: 恢复的 MultiwayTreeExtractor 实例。需要严格元数据校验或 SHA256 校验时，
            使用继承的 ``load_artifact`` 并设置 metadata_policy 或 expected_sha256。
        :rtype: MultiwayTreeExtractor
        :raises SerializationError: 制品协议或对象类型不符合要求。
        :raises OSError: 文件不存在或无法读取。

        **参考样例**

        >>> restored = MultiwayTreeExtractor.load('模型/多叉树.pkl')
        >>> probabilities = restored.predict_proba(df_test)
        """
        return cls.load_artifact(path)
