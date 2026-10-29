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

    :param target: 标签列名，默认 ``'target'``；fit(X, y) 中 y 优先。
    :param features: 特征列名或列表；默认使用除标签外的全部列。
    :param max_depth: 最大深度，根节点深度为 0，默认 3。
    :param max_branches: 每次分裂最多分支数（含独立缺失分支），默认 4。
    :param min_samples_split: 自动分裂的最小节点样本数，默认 20。
    :param min_samples_leaf: 自动分裂的最小普通叶节点样本数，默认 10；独立缺失分支不受此限制。
    :param criterion: ``'chi2'``、``'gini'`` 或 ``'entropy'``，默认 chi2。
        卡方准则按调整后 P 值的 -log10 排序，参考 SAS 的 LogWorth。
    :param alpha: 卡方分裂的显著性上限，默认 0.05。
    :param adjust_pvalue: 是否按实际搜索方案数作 Bonferroni 调整，默认 True。
        这是明确的多重比较控制，不声称复刻 SAS 的 Kass/深度调整。
    :param max_prebins: 数值候选池的等频预分箱数，默认 12。
    :param max_candidates: 穷举方案上限，默认 5000；超过时采用逐步增加切点的贪心搜索。
    :param min_impurity_decrease: 最小节点加权不纯度下降，默认 0.0。
    :param feature_map: 可选字段中文含义映射。
    :param n_jobs: 节点规则报告的并行预算，默认 -1 自动调度；树搜索按确定顺序串行执行。
    :param parallel_backend: 规则报告的 joblib 后端。
    :param parallel_config: 规则报告的并行配置。
    :param missing_separate: 默认 False，自动学习缺失值合并方向；True 时保留独立缺失分支。
        显式 missing_branch 或规则中的缺失标记优先于此默认配置。

    **属性**

    - nodes_: 节点编号到 MultiwayNode 的映射；请通过编辑方法修改树。
    - classes_: 类别顺序，与 predict_proba 的列顺序一致。
    - feature_importances_: 由节点加权不纯度下降归一化的特征重要性。
    - features_: 拟合使用的字段列表。

    **参考样例**

    >>> tree = MultiwayTreeExtractor(target='FPD', max_branches=4).fit(df)
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
        """拟合数值/类别特征；支持二分类、多分类及外部 y，失败时保留原模型。"""
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
            choices = [int(np.argmax(np.bincount(branches, minlength=count)))]
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
        """返回该节点每个特征的最佳候选及原始/调整后 P 值，按分裂价值排序。"""
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
        """替换一个节点的子树；数值用 thresholds，类别用 category_groups。

        未提供规则时自动选择候选（feature 可限制字段）。默认不继续生长。
        手工分裂允许空分支，其预测概率继承最近的非空祖先；其他子树编号保持不变。
        数值 thresholds 与 binner 共用规则：np.nan/None/pd.NA 的位置表示缺失合并的普通分支。
        类别 category_groups 中 [np.nan] 表示独立分支，[类别, np.nan] 表示合并。
        显式 missing_branch 优先于合并标记，缺失规则标记优先于 missing_separate 默认配置。
        未知类别在缺失合并模式下同路；独立缺失模式下走样本最多的普通类别分支。
        data 若传入，必须与拟合数据一致。

        :param missing_branch: 从 0 开始的缺失合并分支序号；默认学习最优方向。
        :param missing_separate: None 沿用构造配置，True 保留独立缺失分支（数值型追加在末尾）。
            独立分支计入 max_branches；显式分组中的 [np.nan] 保持声明位置。
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
        """重设数值节点阈值，替换其后代；未指定缺失配置/标记时保留当前路由方式。"""
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
        """删除指定节点的全部后代，将其变为叶子；不重新编号其他节点。"""
        self._node(node)

        def operation(tree):
            tree._prune(node)
            tree.nodes_[node].manual = True

        return self._edit(operation)

    def delete_node(self, node=0):
        """与 ManualTreeExtractor 一致，删除节点的后代而保留该节点。"""
        return self.prune(node)

    def grow(self, node=0):
        """按当前参数继续自动生长指定子树，不替换已有分裂。"""
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
        """返回当前叶节点编号，包含人工创建的空分支。"""
        self._check_fitted()
        return [node for node, current in self.nodes_.items() if not current.children]

    def _frame(self, X):
        self._check_fitted()
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.features_)
        if any(name not in frame for name in self.features_):
            raise InputValidationError("预测数据缺少拟合特征")
        return frame

    def apply(self, X):
        """返回每行所属叶节点编号；缺失、未知类别遵循保存的分支语义。"""
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
        """输出各类别概率，列顺序为 classes_；空叶继承最近非空祖先。"""
        leaves = self.apply(X)
        return np.asarray([self._probabilities(node) for node in leaves]).reshape(-1, len(self.classes_))

    def predict(self, X):
        """返回最大概率类别。"""
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
        """输出互斥且覆盖全部输入的叶节点 Rule 列表，或全部节点路径。"""
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
        """将选定叶节点合成为 OR RuleSet；默认全叶覆盖，风险筛选需明确选择节点。"""
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
        """用 Rule.report 输出节点效果，支持金额、多逾期标签和多数据集。

        多分类默认对每个类别分别计算一对其余类别的规则指标。
        显式 target/overdue 参数按 Rule.report 既有口径处理。

        :param datasets: DataFrame、DataFrame 列表或名称到 DataFrame 的字典。
        :param target: 标签列名、一维 Series/ndarray/list、callable 或标签映射；
            callable 接收当前完整数据集。与 overdue+dpds 模式互斥；
            未指定时沿用拟合标签，多分类按每个类别对其余类别生成标签。
        :param overdue: 逾期天数字段或字段列表，必须与 dpds 同时提供。
        :param dpds: 逾期阈值或阈值列表，按照 overdue × dpds 生成标签。
        :param del_grey: 是否按标签剔除灰样本；显式标签仅保留 0/1。
            逾期标签在 > 时剔除 (0, dpd]，>= 时剔除 (0, dpd)，< 和 <= 不额外剔灰。
        :param leaf_only: 是否只评估叶节点，默认 False。
        :param overdue_operator: 逾期比较符，支持 >、>=、<、<=，默认 >。
        :param kwargs: 传给 Rule.report 的其他参数，例如 amount、round_digits。
        :return: 节点效果表；列表/字典输入返回对应的列表/字典。
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
        """返回训练集或指定数据集的中文规则效果表。"""
        self._check_fitted()
        return self.report(self._data if data is None else data, leaf_only=leaf_only, **kwargs)

    def get_node_table(self):
        """返回结构、精确切点、类别分组及各类别原始计数。"""
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
        """输出模型效果：样本数、准确率、交叉熵、AUC；二分类另含 KS。

        支持 DataFrame、(X, y) 或 {数据集名: DataFrame}；未传入时评估训练集。
        测试集单一类别时 AUC 为 NaN，二分类 KS 沿用库内单类为 0 的约定。
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
        """复用 plot_tree 的 matplotlib、pyecharts 和 graphviz 后端。"""
        from ...core.viz import plot_tree

        return plot_tree(self, backend=backend, save=save, **kwargs)

    def display(self):
        """在 Notebook 中展示当前树和规则效果表。"""
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
        """保存包含人工修改、类别顺序和规则语义的完整模型。"""
        self._check_fitted()
        return self.save_artifact(path)

    @classmethod
    def load(cls, path):
        """加载可信来源的多叉树模型。"""
        return cls.load_artifact(path)
