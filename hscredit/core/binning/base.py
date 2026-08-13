"""分箱算法基类.

定义分箱算法的统一接口和通用方法。
所有分箱算法都继承此类，确保API的一致性。

设计原则:
1. 参数命名统一，与其他库保持一致
2. 支持高度自定义，但提供合理默认值
3. 遵循sklearn API风格
4. 使用 core.metrics 中的指标计算方法
"""

from abc import ABC, abstractmethod
from collections import deque
from copy import copy, deepcopy
from functools import wraps
from typing import Union, List, Dict, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from ...exceptions import FeatureNotFoundError, NotFittedError, ParallelExecutionError
from ...utils.misc import round_float
from ...utils.parallel import ParallelizableMixin, ParallelWorkload
from ...utils.serialization import ArtifactSerializableMixin
from ._categorical import (
    CategoryOrder,
    assign_category_groups,
    encode_ordered_categories,
    is_missing_marker,
    normalize_user_groups,
    resolve_category_order,
    restore_category_groups,
)
from ._contracts import (
    HandleUnknown,
    MISSING_BIN,
    SPECIAL_BIN,
    UNKNOWN_BIN,
    UserSplitsFixed,
    parse_numerical_user_splits,
    resolve_user_splits_fixed,
    validate_handle_unknown,
)

from ..metrics._binning import (
    compute_bin_stats,
    _fit_monotone_quadratic,
)


class BaseBinning(ParallelizableMixin, ArtifactSerializableMixin, BaseEstimator, TransformerMixin, ABC):
    """分箱算法基类.

    所有分箱算法都继承此类，实现统一的fit/transform接口。
    支持16种分箱方法，适用于风控评分卡开发场景。

    **参数**

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param min_n_bins: 最小分箱数，默认为2
    :param max_n_bins: 最大分箱数，默认为5
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，用于避免极端情况，默认为0.0
    :param monotonic: 坏样本率单调性约束，默认为False
        - False: 不要求单调性
        - True 或 'auto': 自动检测最佳趋势（允许单增、单减、正U、倒U）
        - 'auto_asc_desc': 自动检测，但只允许单增或单减（不允许U型）
        - 'auto_heuristic': 使用启发式方法自动确定单调方向
        - 'ascending': 强制坏样本率递增(分箱索引增大时坏样本率增大)
        - 'descending': 强制坏样本率递减(分箱索引增大时坏样本率减小)
        - 'peak': 倒U型/峰值（先增后降）
        - 'valley': U型/谷值（先降后增）
        - 'peak_heuristic': 使用启发式方法检测峰值
        - 'valley_heuristic': 使用启发式方法检测谷值
    :param special_codes: 特殊值列表，这些值会被单独分箱，例如[-99, -98, 'missing']
    :param cat_cutoff: 类别型变量处理阈值，默认为None
        - 如果 < 1, 表示保留占比超过该值的类别
        - 如果 >= 1, 表示保留频率最高的N个类别
    :param user_splits: 用户自定义分箱规则，例如{'feature': [0, 10, 20, 30]}
    :param user_splits_fixed: 用户切分点固定配置，可按字段或节点选择性固定
    :param random_state: 随机种子，用于可复现性，默认为None
    :param n_jobs: 并行工作数，默认为-1；None沿用旧串行行为
    :param parallel_backend: joblib并行后端，默认为None
    :param parallel_config: joblib扩展配置，默认为None
    :param verbose: 是否输出详细信息，默认为False
    :param decimal: 数值型切分点小数点保留精度，默认为4
    :param woe_clip: WOE值截断阈值，默认为None
        当某个分箱无坏样本或无好样本时，WOE可能变得极大（如±10以上），
        这会导致评分卡中对应分箱的分数异常。
        设置此参数可将WOE限制在[-woe_clip, woe_clip]范围内。
        例如 woe_clip=5.0 可将WOE限制在[-5, 5]之间。

    **属性**

    - splits_: 每个特征的分箱切分点，数值型特征为numpy数组，类别型特征为列表
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表，包含中文列名:
        - 分箱: 分箱索引
        - 分箱标签: 分箱区间标签
        - 样本总数: 样本数
        - 样本占比: 样本占比
        - 好样本数: 好样本数
        - 坏样本数: 坏样本数
        - 坏样本率: 坏样本率
        - 分档WOE值: WOE值
        - 分档IV值: IV值
        - 指标IV值: 总IV值
        - LIFT值: Lift值
        - 坏账改善: 坏账改善
        - 累积LIFT值: 累积Lift值
        - 累积坏账改善: 累积坏账改善
        - 累积好样本数: 累积好样本数
        - 累积坏样本数: 累积坏样本数
        - 分档KS值: KS值
    - feature_types_: 每个特征的类型 ('numerical' 或 'categorical')

    **支持的分箱方法**

    | 方法 | 类名 | 说明 |
    |------|------|------|
    | uniform | UniformBinning | 等宽分箱，将数值范围等分 |
    | quantile | QuantileBinning | 等频分箱，每箱样本数相等 |
    | tree | TreeBinning | 决策树分箱，基于信息增益 |
    | chi | ChiMergeBinning | 卡方分箱，基于卡方统计量合并 |
    | best_ks | BestKSBinning | 最优KS分箱，最大化KS统计量 |
    | best_iv | BestIVBinning | 最优IV分箱，最大化IV值(推荐) |
    | mdlp | MDLPBinning | MDLP分箱，信息论方法 |
    | or_tools | ORBinning | 运筹规划分箱（基于Google OR-Tools） |
    | cart | CartBinning | CART分箱，参考optbinning实现 |
    | monotonic | MonotonicBinning | 单调性约束分箱，支持U型/倒U型 |
    | genetic | GeneticBinning | 遗传算法分箱，全局优化 |
    | smooth | SmoothBinning | 平滑分箱，正则化方法 |
    | kernel_density | KernelDensityBinning | 核密度分箱，密度估计 |
    | best_lift | BestLiftBinning | Best Lift分箱，提升度优化 |
    | target_bad_rate | TargetBadRateBinning | 目标坏样本率分箱 |
    | kmeans | KMeansBinning | K-Means聚类分箱 |
    | optimal | OptimalBinning | 统一接口，支持上述所有方法 |

    **参考样例**

    基本使用 (sklearn风格)::

        >>> from hscredit.core.binning import OptimalBinning
        >>> binner = OptimalBinning(method='best_iv', max_n_bins=5)
        >>> binner.fit(X, y)
        >>> X_binned = binner.transform(X)
        >>> bin_table = binner.get_bin_table('feature_name')

    scorecardpipeline风格 (目标列在DataFrame中)::

        >>> from hscredit.core.binning import OptimalBinning
        >>> # 初始化时指定目标列名，fit时传入完整DataFrame
        >>> binner = OptimalBinning(target='target', method='best_iv', max_n_bins=5)
        >>> binner.fit(df)
        >>> X_binned = binner.transform(df.drop(columns=['target']))
        >>> bin_table = binner.get_bin_table('feature_name')

    混合风格 (y参数优先)::

        >>> # 即使初始化时指定了target，fit时传入y会优先使用y
        >>> binner = OptimalBinning(target='target', method='best_iv')
        >>> binner.fit(df, y=external_y)

    设置切分点精度::

        >>> # 默认4位小数
        >>> binner = OptimalBinning(method='best_iv', decimal=4)
        >>> # 设置为2位小数
        >>> binner = OptimalBinning(method='best_iv', decimal=2)

    单调性约束::

        >>> binner = OptimalBinning(method='best_iv', monotonic='descending')
        >>> binner.fit(X, y)

    使用独立分箱类::

        >>> from hscredit.core.binning import ChiMergeBinning, BestIVBinning
        >>> chi_binner = ChiMergeBinning(max_n_bins=5)
        >>> chi_binner.fit(X, y)

    **注意**

    分箱算法的一般流程:
    1. fit(): 训练分箱模型
       - 数据预处理 (缺失值处理、特殊值处理)
       - 检测特征类型 (数值型/类别型)
       - 计算最优分箱切分点
       - 对数值型切分点进行四舍五入(精度由decimal参数控制)
       - 生成分箱统计表

    2. transform(): 应用分箱
       - 根据切分点对数据进行分箱
       - 支持多种输出格式: 'indices'(分箱索引), 'labels'(分箱标签),
         'woe'(WOE值), 'bin_code'(分箱编码)
    """

    artifact_kind = "分箱器"

    _FEATURE_DICT_STATE = (
        "splits_",
        "n_bins_",
        "bin_tables_",
        "feature_types_",
        "_cat_bins_",
        "_category_orders_",
        "_category_code_maps_",
        "_categorical_numeric_splits_",
        "_categorical_fit_context_",
        "_user_splits_fixed_masks_",
        "_user_missing_bin_targets_",
        "_missing_bin_targets_",
        "_woe_maps_",
        "_recorded_bins_",
        # 具体算法的按特征状态；浅拷贝 worker 必须与主对象隔离。
        "tree_models_",
        "monotonic_trend_",
        "_actual_rates",
        "clip_bounds_",
    )
    _FEATURE_SET_STATE = ("_categorical_encoded_features_", "_reserved_bins_finalized_")
    _IMPORTED_RULE_DICT_STATE = (
        "splits_",
        "n_bins_",
        "feature_types_",
        "_cat_bins_",
        "_category_orders_",
        "_category_code_maps_",
        "_user_missing_bin_targets_",
        "_missing_bin_targets_",
        "_woe_maps_",
        "_recorded_bins_",
    )

    def __init_subclass__(cls, **kwargs):
        """将具体分箱器的 ``fit`` 包装为估计器级事务。"""
        super().__init_subclass__(**kwargs)
        fit_method = cls.__dict__.get("fit")
        if fit_method is None or getattr(fit_method, "_hscredit_transactional_fit", False):
            return

        @wraps(fit_method)
        def transactional_fit(self, *args, **kwargs):
            # 子类 fit 可能委托 super().fit，此时由外层事务持有候选和提交边界。
            if getattr(self, "_fit_transaction_active", False):
                return fit_method(self, *args, **kwargs)

            # 常规拟合始终从构造参数创建空候选，避免复制或保留历史大状态。
            # 若刚导入规则，仅显式恢复本次规则快照与拟合列的交集。
            candidate = self._make_fit_transaction_candidate(args, kwargs)
            candidate._fit_transaction_active = True
            try:
                result = fit_method(candidate, *args, **kwargs)
            finally:
                candidate.__dict__.pop("_fit_transaction_active", None)

            if result is not candidate:
                raise TypeError("分箱器 fit 必须返回自身")
            self.__dict__.clear()
            self.__dict__.update(candidate.__dict__)
            return self

        transactional_fit._hscredit_transactional_fit = True
        cls.fit = transactional_fit

    def _make_fit_transaction_candidate(self, fit_args=(), fit_kwargs=None) -> "BaseBinning":
        """创建不携带历史拟合状态的事务候选对象。"""
        candidate = clone(self)
        # ``**kwargs`` 不在 sklearn 的参数签名中；OptimalBinning 用它保存
        # 求解器和后处理选项，需要显式复制到事务候选。
        if hasattr(self, "kwargs"):
            candidate.kwargs = clone(self.kwargs, safe=False)
        if hasattr(self, "_fit_control_options"):
            candidate._fit_control_options = clone(self._fit_control_options, safe=False)
        self._restore_imported_rule_snapshot(candidate, fit_args, fit_kwargs or {})
        return candidate

    @staticmethod
    def _fit_input_features(fit_args, fit_kwargs) -> Optional[List[str]]:
        """从 fit 调用参数中提取本轮输入特征名。"""
        X = fit_args[0] if fit_args else fit_kwargs.get("X")
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        if isinstance(X, pd.Series):
            return [X.name if X.name is not None else "feature"]
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return ["feature"]
            return [f"feature_{index}" for index in range(X.shape[1])]
        return None

    def _capture_imported_rule_snapshot(self, features) -> None:
        """记录最近一次 import_rules 的规则输入，不混入历史拟合输出。"""
        ordered_features = tuple(dict.fromkeys(features))
        snapshot = {}
        for state_name in self._IMPORTED_RULE_DICT_STATE:
            values = getattr(self, state_name, {})
            snapshot[state_name] = {
                feature: deepcopy(values[feature]) for feature in ordered_features if feature in values
            }
        self._imported_rule_features_ = ordered_features
        self._imported_rule_snapshot_ = snapshot

    def _restore_imported_rule_snapshot(self, candidate, fit_args, fit_kwargs) -> None:
        """仅向干净候选注入最近导入且属于本轮 X 的规则。"""
        snapshot = getattr(self, "_imported_rule_snapshot_", None)
        if not getattr(self, "_rules_imported_", False) or snapshot is None:
            return

        input_features = self._fit_input_features(fit_args, fit_kwargs)
        imported_features = getattr(self, "_imported_rule_features_", ())
        selected = [feature for feature in imported_features if input_features is None or feature in input_features]
        if not selected:
            return
        filtered_snapshot = {}
        for state_name in self._IMPORTED_RULE_DICT_STATE:
            values = snapshot.get(state_name, {})
            filtered = {feature: deepcopy(values[feature]) for feature in selected if feature in values}
            setattr(candidate, state_name, filtered)
            filtered_snapshot[state_name] = deepcopy(filtered)

        candidate._rules_imported_ = True
        candidate._imported_rule_features_ = tuple(selected)
        candidate._imported_rule_snapshot_ = filtered_snapshot
        # “存在导入规则”是按特征的输入状态，不能提升为整个估计器已经拟合。
        # OptimalBinning 会据此分别处理固定规则特征和仍需训练的普通特征。
        candidate._is_fitted = False

    def __init__(
        self,
        target: str = "target",
        missing_separate: bool = True,
        min_n_bins: int = 2,
        max_n_bins: int = 5,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        cat_cutoff: Optional[Union[float, int]] = None,
        user_splits: Optional[Union[Dict[str, List], Callable]] = None,
        user_splits_fixed: UserSplitsFixed = None,
        category_order: CategoryOrder = None,
        handle_unknown: HandleUnknown = UNKNOWN_BIN,
        random_state: Optional[int] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        woe_clip: Optional[float] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        self.target = target
        self.missing_separate = missing_separate
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bad_rate = min_bad_rate
        self.monotonic = monotonic
        self.special_codes = special_codes
        self.cat_cutoff = cat_cutoff
        self.user_splits = user_splits
        self.user_splits_fixed = user_splits_fixed
        self.category_order = category_order
        self.handle_unknown = validate_handle_unknown(handle_unknown)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config
        self.verbose = verbose
        if isinstance(decimal, (bool, np.bool_)) or not isinstance(decimal, (int, np.integer)) or int(decimal) < 0:
            raise ValueError("decimal 必须是大于等于 0 的整数")
        self.decimal = int(decimal)
        self.woe_clip = woe_clip
        self._validate_common_parameters()

        # 拟合后的属性
        self.splits_ = {}
        self.n_bins_ = {}
        self.bin_tables_ = {}
        self.feature_types_ = {}
        self._cat_bins_ = {}  # 类别型变量的分组信息，格式: {'feature': [['A', 'B'], ['C'], [np.nan]]}
        self._category_orders_ = {}
        self._category_code_maps_ = {}
        self._categorical_numeric_splits_ = {}
        self._categorical_fit_context_ = {}
        self._categorical_encoded_features_ = set()
        self._user_splits_fixed_masks_ = {}
        self._user_missing_bin_targets_ = {}
        self._missing_bin_targets_ = {}
        self._woe_maps_ = {}
        self._recorded_bins_ = {}
        self._reserved_bins_finalized_ = set()
        self._is_fitted = False

    def set_params(self, **params) -> "BaseBinning":
        """设置 sklearn 参数，并立即规范化未知类别策略。"""
        if "handle_unknown" in params:
            params = dict(params)
            params["handle_unknown"] = validate_handle_unknown(params["handle_unknown"])
        return super().set_params(**params)

    def _validate_common_parameters(self) -> None:
        """统一校验所有分箱器共享的公共参数。"""
        for name, value in (("min_n_bins", self.min_n_bins), ("max_n_bins", self.max_n_bins)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} 必须是大于等于 1 的整数")
        if self.min_n_bins > self.max_n_bins:
            raise ValueError("min_n_bins 不能大于 max_n_bins")

        for name, value in (("min_bin_size", self.min_bin_size), ("max_bin_size", self.max_bin_size)):
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} 必须是正数")
            if not np.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} 必须是正数")

        if (
            isinstance(self.min_bad_rate, (bool, np.bool_))
            or not isinstance(self.min_bad_rate, (int, float, np.integer, np.floating))
            or not np.isfinite(self.min_bad_rate)
            or not 0 <= float(self.min_bad_rate) <= 1
        ):
            raise ValueError("min_bad_rate 必须位于 [0, 1] 范围内")

        if self.cat_cutoff is not None:
            if (
                isinstance(self.cat_cutoff, (bool, np.bool_))
                or not isinstance(self.cat_cutoff, (int, float, np.integer, np.floating))
                or not np.isfinite(self.cat_cutoff)
                or float(self.cat_cutoff) <= 0
            ):
                raise ValueError("cat_cutoff 必须是正数")

        if self.woe_clip is not None:
            if (
                isinstance(self.woe_clip, (bool, np.bool_))
                or not isinstance(self.woe_clip, (int, float, np.integer, np.floating))
                or not np.isfinite(self.woe_clip)
                or float(self.woe_clip) <= 0
            ):
                raise ValueError("woe_clip 必须是正数")

        valid_monotonic = {
            None,
            False,
            True,
            "auto",
            "auto_asc_desc",
            "auto_heuristic",
            "ascending",
            "descending",
            "peak",
            "valley",
            "peak_heuristic",
            "valley_heuristic",
            "convex",
            "concave",
        }
        if self.monotonic not in valid_monotonic:
            raise ValueError(f"monotonic 不支持: {self.monotonic}")
        validate_handle_unknown(self.handle_unknown)
        if self.user_splits is not None and not isinstance(self.user_splits, dict) and not callable(self.user_splits):
            raise ValueError("user_splits 必须是字段规则字典、可调用对象或 None")
        if isinstance(self.user_splits, dict):
            resolve_user_splits_fixed(self.user_splits, self.user_splits_fixed)
        elif self.user_splits_fixed not in (None, False, True) and not isinstance(self.user_splits_fixed, dict):
            raise ValueError("user_splits_fixed 必须是布尔值、字段配置字典或 None")
        if (
            self.category_order is not None
            and not isinstance(self.category_order, dict)
            and not callable(self.category_order)
        ):
            raise ValueError("category_order 必须是特征顺序字典、排序函数或 None")

    def _record_category_orders(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        excluded_features: Optional[set] = None,
    ) -> None:
        """记录训练类别顺序，供有序编码和规则导出使用。"""
        excluded_features = excluded_features or set()
        for feature in X.columns:
            if feature in excluded_features:
                continue
            if self._detect_feature_type(X[feature]) != "categorical":
                continue
            self._category_orders_[feature] = resolve_category_order(
                feature,
                X[feature],
                y,
                category_order=self.category_order,
                special_codes=self.special_codes,
            )

    def _prepare_categorical_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """将类别列转换为有序数值编码，供具体分箱器复用数值算法。"""
        if getattr(self, "_defer_categorical_adapter", False):
            return X

        # 这些容器只描述本轮拟合输入，编码前重置以免成功重拟合保留旧类别列。
        self._category_orders_ = {}
        self._category_code_maps_ = {}
        self._categorical_fit_context_ = {}
        self._categorical_encoded_features_ = set()
        self.__dict__.pop("_categorical_fit_y_", None)
        if getattr(self, "force_numerical", False):
            return X

        X_fit = X.copy()
        for feature in X.columns:
            # 显式用户规则必须在原始类别值上校验和应用，不能先被
            # 通用类别适配器编码。规则 worker 与普通字段 worker 仍在同一批并行。
            if self._has_explicit_user_rule(feature):
                continue
            if self._detect_feature_type(X[feature]) != "categorical":
                continue
            order = resolve_category_order(
                feature,
                X[feature],
                y,
                category_order=self.category_order,
                special_codes=self.special_codes,
            )
            encoded = encode_ordered_categories(X[feature], order, self.special_codes)
            self._category_orders_[feature] = order
            self._category_code_maps_[feature] = [(category, code) for code, category in enumerate(order)]
            self._categorical_fit_context_[feature] = X[feature].copy()
            self._categorical_encoded_features_.add(feature)
            X_fit[feature] = encoded
        if self._categorical_fit_context_:
            self._categorical_fit_y_ = y.copy()
        return X_fit

    def _assign_categorical_bins(self, feature: str, x: pd.Series) -> np.ndarray:
        """使用拟合后的类别组进行类型安全分箱。"""
        return assign_category_groups(
            feature,
            x,
            self._cat_bins_.get(feature, []),
            special_codes=self.special_codes,
            missing_separate=self.missing_separate,
            handle_unknown=self.handle_unknown,
        )

    def _finalize_categorical_fit(self) -> None:
        """把具体方法产生的数值切分点还原为类别规则和统计表。"""
        if not self._categorical_fit_context_:
            return
        y = self._categorical_fit_y_
        for feature, original in self._categorical_fit_context_.items():
            numeric_splits = np.asarray(self.splits_.get(feature, np.array([])), dtype=float)
            numeric_splits = self._ensure_categorical_minimum_bins(feature, original, y, numeric_splits)
            order = self._category_orders_.get(feature, [])
            encoded = encode_ordered_categories(original, order, self.special_codes)
            numeric_splits = self._adjust_splits_for_bin_size_constraints(
                encoded,
                y,
                numeric_splits,
                BaseBinning._get_min_samples(self, len(y)),
                BaseBinning._get_max_samples(self, len(y)),
                allow_empty_splits=True,
                allow_boundary_relocation=True,
            )
            numeric_splits = self._round_splits(numeric_splits)
            self.splits_[feature] = numeric_splits
            self._categorical_numeric_splits_[feature] = numeric_splits.copy()
            groups = restore_category_groups(order, numeric_splits)
            self._cat_bins_[feature] = groups
            self.splits_[feature] = groups
            self.n_bins_[feature] = len(groups)
            self.feature_types_[feature] = "categorical"
            bins = self._assign_categorical_bins(feature, original)
            self.bin_tables_[feature] = self._compute_bin_stats(feature, original, y, bins)
            self._validate_categorical_constraints(feature, y)
        # 原始类别值保留到统一保留箱收口完成，避免用内部编码重算统计表。

    def _ensure_categorical_minimum_bins(
        self,
        feature: str,
        original: pd.Series,
        y: pd.Series,
        numeric_splits: np.ndarray,
    ) -> np.ndarray:
        """在原生方法结果不足时补足可行的最小类别箱数。"""
        order = self._category_orders_.get(feature, [])
        required = min(self.min_n_bins, self.max_n_bins, len(order))
        if required <= 1:
            return np.asarray(numeric_splits, dtype=float)

        splits = np.asarray(numeric_splits, dtype=float)
        splits = np.unique(np.sort(splits[np.isfinite(splits)]))
        codes = np.arange(len(order), dtype=float)
        assignments = np.digitize(codes, splits)
        # 类别只关心相邻编码是否跨箱，把任意浮点边界规范为无空箱的半整数边界。
        splits = np.asarray(
            [index + 0.5 for index in range(len(order) - 1) if assignments[index] != assignments[index + 1]],
            dtype=float,
        )

        encoded = encode_ordered_categories(original, order, self.special_codes)
        rates = []
        counts = []
        for code in range(len(order)):
            mask = encoded == float(code)
            counts.append(int(mask.sum()))
            rates.append(float(y.loc[mask].mean()) if mask.any() else 0.0)

        while len(splits) + 1 < required:
            assignments = np.digitize(codes, splits)
            candidates = [index for index in range(len(order) - 1) if assignments[index] == assignments[index + 1]]
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda index: (
                    abs(rates[index + 1] - rates[index]),
                    counts[index] + counts[index + 1],
                    -index,
                ),
            )
            splits = np.unique(np.sort(np.append(splits, best + 0.5)))
        return splits

    def _validate_categorical_constraints(self, feature: str, y: pd.Series) -> None:
        """验证类别还原后的最终分箱是否满足公共硬约束。"""
        table = self.bin_tables_.get(feature)
        if table is None or table.empty:
            return
        ordinary = table[table["分箱"] >= 0].sort_values("分箱").reset_index(drop=True)
        if ordinary.empty:
            return

        n_bins = len(ordinary)
        available_categories = len(self._category_orders_.get(feature, []))
        required_min_bins = min(self.min_n_bins, available_categories) if available_categories else 1
        if n_bins < required_min_bins:
            raise ValueError(
                f"特征 '{feature}' 的类别分箱无法满足 min_n_bins={self.min_n_bins}；"
                f"最终普通箱数为 {n_bins}，可用类别数为 {available_categories}"
            )
        if n_bins > self.max_n_bins:
            raise ValueError(f"特征 '{feature}' 的类别分箱无法满足 max_n_bins={self.max_n_bins}；" f"最终普通箱数为 {n_bins}")

        counts = ordinary["样本总数"].to_numpy(dtype=int)
        min_samples = BaseBinning._get_min_samples(self, len(y))
        if np.any(counts < min_samples):
            observed = int(counts.min())
            raise ValueError(
                f"特征 '{feature}' 的类别分箱无法满足 min_bin_size={self.min_bin_size}；" f"最小箱样本数为 {observed}，要求至少 {min_samples}"
            )

        max_samples = BaseBinning._get_max_samples(self, len(y))
        if max_samples is not None and np.any(counts > max_samples):
            observed = int(counts.max())
            raise ValueError(
                f"特征 '{feature}' 的类别分箱无法满足 max_bin_size={self.max_bin_size}；"
                f"最大箱样本数为 {observed}，要求至多 {max_samples}；"
                "单个类别或用户类别组是不可拆分的原子单位"
            )

        bad_rates = ordinary["坏样本率"].to_numpy(dtype=float)
        if self.min_bad_rate > 0 and np.any(bad_rates < self.min_bad_rate - 1e-12):
            observed = float(bad_rates.min())
            raise ValueError(f"特征 '{feature}' 的类别分箱无法满足 min_bad_rate={self.min_bad_rate}；" f"最小坏样本率为 {observed:.6f}")

        if self.monotonic in {"ascending", "descending"} and len(bad_rates) > 1:
            differences = np.diff(bad_rates)
            valid = np.all(differences >= -1e-12) if self.monotonic == "ascending" else np.all(differences <= 1e-12)
            if not valid:
                raise ValueError(
                    f"特征 '{feature}' 的类别分箱无法满足 monotonic={self.monotonic}；" f"最终坏样本率为 {bad_rates.tolist()}"
                )

    def _set_input_feature_attributes(self, X: pd.DataFrame) -> None:
        """记录 sklearn 兼容的输入特征元数据。"""
        feature_names = list(X.columns)
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.feature_names_ = feature_names

    @abstractmethod
    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "BaseBinning":
        """拟合分箱。

        支持两种API风格：
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级规则：如果y不是None，直接使用y（优先）；否则从X中提取target列。

        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)，可以是数值型或类别型特征
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
            - 支持DataFrame或numpy数组
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量，必须是二分类 (0/1 或 False/True)
            - scorecardpipeline风格: 不传，从X中提取
            - 如果传入y，优先使用y而忽略X中的target列
        :param kwargs: 其他参数，传递给具体的分箱算法
        :return: 拟合后的分箱器

        **注意**

        fit方法会进行以下操作:
        1. 数据验证和预处理（通过_check_input方法）
        2. 识别特征类型 (数值型/类别型)
        3. 处理缺失值和特殊值
        4. 计算最优分箱切分点
        5. 生成分箱统计表

        **使用示例**

        sklearn风格::

            >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [5000, 6000, 7000]})
            >>> y = pd.Series([0, 1, 0])
            >>> binner.fit(X, y)

        scorecardpipeline风格::

            >>> df = pd.DataFrame({
            ...     'age': [25, 30, 35],
            ...     'income': [5000, 6000, 7000],
            ...     'target': [0, 1, 0]
            ... })
            >>> binner = OptimalBinning(target='target')
            >>> binner.fit(df)  # 自动从df中提取'target'列

        混合风格（y参数优先）::

            >>> binner = OptimalBinning(target='target')
            >>> binner.fit(df, y=external_y)
        """
        pass

    @abstractmethod
    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], metric: str = "indices", **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.

        将原始特征值转换为分箱索引、分箱标签或WOE值。
        这是分箱器的核心方法，用于将新数据应用到已训练的分箱规则。

        :param X: 待转换的数据，shape (n_samples, n_features)
            - 支持DataFrame或numpy数组
            - 列名必须与fit时的特征名一致
        :param metric: 转换类型，默认为'indices'
            - 'indices': 返回分箱索引 (0, 1, 2, ...)
                * 用途: 后续处理、特征工程
                * 示例: [0, 1, 2, 0, 1, ...]
            - 'bins': 返回分箱标签字符串
                * 用途: 可视化、报告展示
                * 示例: ['(-inf, 25]', '(25, 35]', '(35, 45]', ...]
                * 类别型: ['北京,上海', '广州,深圳', ...]
            - 'woe': 返回WOE值
                * 用途: 逻辑回归建模
                * 示例: [0.234, -0.456, 0.123, ...]
        :param kwargs: 其他参数
        :return: 转换后的数据，返回类型与输入类型一致

        **重要说明**

        1. metric参数是枚举值，只能使用以下3个值之一:
           - 'indices' (不是'分箱'、'索引'等中文)
           - 'bins' (不是'分箱标签'等中文)
           - 'woe' (不是'分档WOE值'等中文)

        2. 中文列名出现在分箱结果表中:
           - 使用 binner.get_bin_table(feature) 查看
           - 列名: '分箱', '样本总数', '坏样本率', '分档WOE值'等

        **参考样例**

        >>> binner = OptimalBinning()
        >>> binner.fit(X_train, y_train)
        >>>
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> print(X_binned.head())
        >>>
        >>> # 获取分箱标签
        >>> X_labels = binner.transform(X_test, metric='bins')
        >>> print(X_labels.head())
        >>>
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        >>> print(X_woe.head())
        >>>
        >>> # 错误示例 - 不要使用中文
        >>> # X_error = binner.transform(X_test, metric='分档WOE值')  # ❌ ValueError!

        **处理特殊值**

        transform方法会自动处理:
        1. 缺失值: 如果missing_separate=True，分配到专门的缺失箱 (索引=-1)
        2. 特殊值: 如果指定了special_codes，分配到专门的特殊值箱 (索引=-2)
        3. 超出范围的值: 分配到最近的分箱
        """
        pass

    @staticmethod
    def _enrich_woe_map(woe_map: dict, bin_table) -> None:
        """为 woe_map 补充缺失值/特殊值箱的 WOE 映射.

        ``_apply_splits`` 对缺失值返回 -1、特殊值返回 -2、未知类别返回 -3，而 ``woe_map``
        默认只包含 0..n-1 的映射。本方法从 bin_table 的 missing/special
        行中提取真实 WOE 值写入 woe_map[-1] / woe_map[-2]。
        """
        if "分箱标签" not in bin_table.columns:
            woe_map.setdefault(-1, 0.0)
            woe_map.setdefault(-2, 0.0)
            woe_map.setdefault(-3, 0.0)
            return
        for idx in range(len(bin_table)):
            lbl = str(bin_table.iloc[idx].get("分箱标签", "")).lower()
            if lbl in ("missing", "缺失值", "缺失"):
                woe_map[-1] = float(bin_table.iloc[idx]["分档WOE值"])
            elif lbl in ("special", "特殊值", "特殊"):
                woe_map[-2] = float(bin_table.iloc[idx]["分档WOE值"])
        woe_map.setdefault(-1, 0.0)
        woe_map.setdefault(-2, 0.0)
        woe_map.setdefault(-3, 0.0)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        metric: str = "indices",
        **kwargs,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并应用分箱。

        支持两种API风格：
        1. sklearn风格: fit_transform(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit_transform(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取
        :param metric: 返回值的类型，默认为'indices'
            - 'indices': 返回分箱索引
            - 'bins': 返回分箱标签
            - 'woe': 返回WOE值
        :return: 分箱后的数据

        **使用示例**

        sklearn风格::

            >>> X_binned = binner.fit_transform(X, y, metric='woe')

        scorecardpipeline风格::

            >>> binner = OptimalBinning(target='target')
            >>> X_binned = binner.fit_transform(df, metric='woe')
        """
        fitted = self.fit(X, y, **kwargs)
        transform_input = X
        if isinstance(X, pd.DataFrame):
            feature_names = list(fitted.feature_names_in_)
            missing = [feature for feature in feature_names if feature not in X.columns]
            if missing:
                raise KeyError(f"转换数据缺少拟合特征: {missing}")
            transform_input = X.loc[:, feature_names]
        return fitted.transform(transform_input, metric=metric, **kwargs)

    def _check_input(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """检查并准备输入数据，支持sklearn和scorecardpipeline两种API风格。

        该方法统一处理两种风格的输入：
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级规则：
        - 如果y不是None，直接使用y（优先）
        - 如果y是None且X是DataFrame，从X中提取target列

        :param X: 输入特征或完整数据框
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征和目标列
            - 支持DataFrame或numpy数组
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取
            - 支持pd.Series或numpy数组
        :return: (特征DataFrame, 目标Series)
            - 特征DataFrame: 纯特征数据，不含目标列
            - 目标Series: 二分类目标变量
        :raises ValueError: 如果输入数据格式不正确或目标变量不符合要求

        **使用示例**

        sklearn风格::

            >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [5000, 6000, 7000]})
            >>> y = pd.Series([0, 1, 0])
            >>> X_processed, y_processed = binner._check_input(X, y)

        scorecardpipeline风格::

            >>> df = pd.DataFrame({
            ...     'age': [25, 30, 35],
            ...     'income': [5000, 6000, 7000],
            ...     'target': [0, 1, 0]
            ... })
            >>> binner = OptimalBinning(target='target')
            >>> X_processed, y_processed = binner._check_input(df)

        numpy数组输入::

            >>> X = np.array([[25, 5000], [30, 6000], [35, 7000]])
            >>> y = np.array([0, 1, 0])
            >>> X_processed, y_processed = binner._check_input(X, y)
        """
        # 保存原始索引，用于后续对齐
        original_index = None

        # 转换为DataFrame
        if isinstance(X, pd.Series):
            # 将Series转换为DataFrame，保留Series的名称作为列名
            col_name = X.name if X.name is not None else "feature"
            X = X.to_frame(name=col_name)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = pd.DataFrame(X, columns=["feature"])
            else:
                # 为numpy数组生成默认列名 feature_0, feature_1, ...
                n_cols = X.shape[1]
                columns = [f"feature_{i}" for i in range(n_cols)]
                X = pd.DataFrame(X, columns=columns)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is not None and self.target in X.columns:
            X = X.drop(columns=[self.target])

        original_index = X.index

        # 获取目标变量
        if y is not None:
            # sklearn风格: 使用传入的y（优先）
            if isinstance(y, np.ndarray):
                if y.ndim != 1:
                    raise ValueError(f"目标变量y必须是一维数组，但得到 {y.ndim} 维")
                if len(y) != len(original_index):
                    raise ValueError(f"特征和标签数量不匹配: {len(original_index)} != {len(y)}")
                y = pd.Series(y, index=original_index, name=self.target)
            elif isinstance(y, pd.Series):
                y = y.copy()
                if y.index.equals(original_index):
                    pass
                elif len(y) == len(original_index):
                    # 长度一致但索引不同：按位置对齐
                    y = y.reset_index(drop=True)
                    y.index = original_index
                else:
                    # 长度不一致：尝试按索引交集对齐（常见于调用方先对y做过滤）
                    common_index = original_index.intersection(y.index)
                    if len(common_index) == 0:
                        raise ValueError(f"特征和标签数量不匹配且无公共索引: {len(original_index)} != {len(y)}")
                    X = X.loc[common_index].copy()
                    original_index = X.index
                    y = y.loc[common_index].copy()
                y.name = self.target
            else:
                # 其他可迭代类型
                if len(y) != len(original_index):
                    raise ValueError(f"特征和标签数量不匹配: {len(original_index)} != {len(y)}")
                y = pd.Series(y, index=original_index, name=self.target)
        else:
            # scorecardpipeline风格: 从X中提取target列
            if self.target in X.columns:
                y = X[self.target].copy()
                y.name = self.target
                X = X.drop(columns=[self.target])
            else:
                raise ValueError(
                    f"目标变量 '{self.target}' 未在数据中找到。"
                    f"请提供y参数（sklearn风格）或在数据中包含 '{self.target}' 列（scorecardpipeline风格）。"
                    f"可用列: {list(X.columns)}"
                )

        # 验证数据长度
        if len(X) != len(y):
            raise ValueError(f"特征和标签数量不匹配: {len(X)} != {len(y)}")

        # 验证目标变量
        if y.isna().any():
            raise ValueError("目标变量包含缺失值，请在拟合前完成处理")
        unique_values = y.dropna().unique()
        if len(unique_values) != 2:
            raise ValueError(f"目标变量必须是二分类，但发现 {len(unique_values)} 个唯一值: {unique_values}")

        if not set(unique_values).issubset({0, 1, False, True}):
            raise ValueError(f"目标变量必须是 0/1 或 False/True，但发现 {unique_values}")

        self._set_input_feature_attributes(X)
        return self._prepare_categorical_fit(X, y), y

    def _fit_feature_transaction(self, task):
        """在隔离的估计器浅拷贝上完成一个特征的拟合并返回状态快照。"""
        feature, x, y, method_name = task
        encoded_categorical = feature in getattr(self, "_categorical_encoded_features_", set())
        categorical_input = {}
        for state_name in ("_category_orders_", "_category_code_maps_", "_categorical_fit_context_"):
            values = getattr(self, state_name, {})
            if feature in values:
                categorical_input[state_name] = deepcopy(values[feature])

        worker = copy(self)
        for state_name in self._FEATURE_DICT_STATE:
            setattr(worker, state_name, {})
        for state_name in self._FEATURE_SET_STATE:
            setattr(worker, state_name, set())
        for state_name, value in categorical_input.items():
            getattr(worker, state_name)[feature] = value
        if encoded_categorical:
            # 这是 worker 的输入元数据，用于让内部数值编码继续走数值算法；
            # 容器仍与主对象隔离，并会作为该特征状态随快照返回。
            worker._categorical_encoded_features_.add(feature)

        fit_one = getattr(worker, method_name)
        fit_one(feature, x, y)

        state = {}
        for state_name in self._FEATURE_DICT_STATE:
            values = getattr(worker, state_name, {})
            if not isinstance(values, dict):
                raise TypeError(f"特征状态 {state_name} 必须为字典")
            state[state_name] = {feature: values[feature]} if feature in values else {}
        for state_name in self._FEATURE_SET_STATE:
            values = getattr(worker, state_name, set())
            if not isinstance(values, set):
                raise TypeError(f"特征状态 {state_name} 必须为集合")
            state[state_name] = {feature} if feature in values else set()
        return feature, state

    def _has_explicit_user_rule(self, feature: str) -> bool:
        """判断字段是否有显式用户分箱规则。"""
        if callable(self.user_splits):
            return True
        return isinstance(self.user_splits, dict) and feature in self.user_splits

    def _get_explicit_user_rule(self, feature: str, x: pd.Series):
        """获取字段正式用户规则。"""
        if callable(self.user_splits):
            return self.user_splits(x)
        if isinstance(self.user_splits, dict) and feature in self.user_splits:
            return self.user_splits[feature]
        raise KeyError(f"特征 '{feature}' 没有对应的用户分箱规则")

    def _resolve_feature_fixed_mask(self, feature: str, rule_values: List[Any]) -> List[bool]:
        """解析当前字段的固定节点掩码。"""
        fixed = self.user_splits_fixed
        if isinstance(fixed, dict):
            fixed = {feature: fixed.get(feature, False)}
        masks = resolve_user_splits_fixed({feature: rule_values}, fixed)
        return masks[feature]

    def _is_user_rule_fully_fixed(self, feature: str) -> bool:
        """判断当前字段的全部有效用户节点是否固定。"""
        mask = self._user_splits_fixed_masks_.get(feature, [])
        return bool(mask) and all(mask)

    def _fit_common_user_split_feature(self, feature: str, x: pd.Series, y: pd.Series) -> None:
        """在隔离 worker 中应用具体分箱器共享的用户规则。"""
        feature_type = "numerical" if getattr(self, "force_numerical", False) else self._detect_feature_type(x)
        self.feature_types_[feature] = feature_type
        rule = self._get_explicit_user_rule(feature, x)
        if rule is None or isinstance(rule, (str, bytes)):
            raise ValueError(f"特征 '{feature}' 的用户分箱规则必须是非字符串序列")

        try:
            rule_values = list(rule)
        except TypeError as exc:
            raise ValueError(f"特征 '{feature}' 的用户分箱规则必须是可迭代序列") from exc

        if feature_type == "numerical":
            splits, missing_bin = parse_numerical_user_splits(feature, rule_values)
            fixed_mask = self._resolve_feature_fixed_mask(feature, rule_values)
            retained_splits = []
            retained_fixed = []
            if not all(fixed_mask):
                valid = pd.to_numeric(x, errors="coerce")
                valid = valid[np.isfinite(valid)]
                if self.special_codes:
                    valid = valid[~x.loc[valid.index].isin(self.special_codes)]
                for split, is_fixed in zip(splits, fixed_mask):
                    if is_fixed:
                        retained_splits.append(float(split))
                        retained_fixed.append(True)
                    elif len(valid) > 0 and valid.min() < split < valid.max():
                        retained_splits.append(float(self._round_splits([split])[0]))
                        retained_fixed.append(False)
                splits = np.asarray(retained_splits, dtype=float)
                fixed_mask = retained_fixed
            self._user_splits_fixed_masks_[feature] = list(fixed_mask)
            if sum(bool(value) for value in fixed_mask) + 1 > self.max_n_bins:
                raise ValueError(f"特征 '{feature}' 的固定切分点数量无法满足 max_n_bins={self.max_n_bins}")
            if missing_bin is not None:
                self._user_missing_bin_targets_[feature] = int(missing_bin)
            self.splits_[feature] = splits
            self.n_bins_[feature] = len(splits) + 1
            bins = self._get_feature_bins(feature, x, splits)
        else:
            groups = self._normalize_common_user_category_groups(feature, x, rule_values)
            fixed_mask = self._resolve_feature_fixed_mask(feature, rule_values)
            self._user_splits_fixed_masks_[feature] = fixed_mask
            if any(fixed_mask) and not all(fixed_mask) and len(groups) > 1:
                groups = self._merge_selectively_fixed_category_groups(feature, x, y, groups, fixed_mask)
            elif not any(fixed_mask) and len(groups) > 1:
                groups = self._merge_common_user_category_groups_with_method(feature, x, y, groups)
            self._cat_bins_[feature] = groups
            self.splits_[feature] = groups
            self.n_bins_[feature] = len(groups)
            ordered = [value for group in groups for value in group if not is_missing_marker(value)]
            self._category_orders_[feature] = ordered
            self._category_code_maps_[feature] = [(category, code) for code, category in enumerate(ordered)]
            bins = self._assign_categorical_bins(feature, x)

        self.bin_tables_[feature] = self._compute_bin_stats(feature, x, y, bins)
        if feature_type == "categorical":
            self._validate_categorical_constraints(feature, y)

    def _normalize_common_user_category_groups(
        self, feature: str, x: pd.Series, rule_values: List[Any]
    ) -> List[List[Any]]:
        """规范化严格的类别 List[List] 规则。"""
        if not rule_values:
            raise ValueError(f"特征 '{feature}' 的自定义类别分箱不能为空")
        if not all(isinstance(value, list) for value in rule_values):
            raise ValueError(f"特征 '{feature}' 的自定义类别分箱必须是非空 List[List]")
        groups = [list(value) for value in rule_values]
        return normalize_user_groups(
            feature,
            groups,
            x,
            special_codes=self.special_codes,
            missing_separate=self.missing_separate,
        )

    def _merge_selectively_fixed_category_groups(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        groups: List[List[Any]],
        fixed_mask: List[bool],
    ) -> List[List[Any]]:
        """只合并相邻可变类别组，固定组始终保持原始成员边界。"""
        work = [{"values": list(group), "fixed": bool(fixed)} for group, fixed in zip(groups, fixed_mask)]

        def group_mask(values: List[Any]) -> np.ndarray:
            mask = np.zeros(len(x), dtype=bool)
            for value in values:
                if is_missing_marker(value):
                    mask |= x.isna().to_numpy(dtype=bool)
                else:
                    mask |= x.eq(value).fillna(False).to_numpy(dtype=bool)
            return mask

        def stats(item) -> Tuple[int, float]:
            mask = group_mask(item["values"])
            count = int(mask.sum())
            return count, float(y.to_numpy(dtype=float)[mask].mean()) if count else 0.0

        min_samples = self._get_min_samples(len(y))
        while True:
            group_stats = [stats(item) for item in work]
            mutable_violations = {
                index
                for index, (item, (count, bad_rate)) in enumerate(zip(work, group_stats))
                if not item["fixed"]
                and (count < min_samples or (self.min_bad_rate > 0 and bad_rate < self.min_bad_rate))
            }
            must_reduce = len(work) > self.max_n_bins
            if not must_reduce and not mutable_violations:
                break

            candidates = [
                index for index in range(len(work) - 1) if not work[index]["fixed"] and not work[index + 1]["fixed"]
            ]
            if mutable_violations:
                violating_pairs = [
                    index for index in candidates if index in mutable_violations or index + 1 in mutable_violations
                ]
                if violating_pairs:
                    candidates = violating_pairs
            if not candidates:
                raise ValueError(f"特征 '{feature}' 的 user_splits_fixed 无法同时满足 max_n_bins/箱样本约束")

            merge_index = min(
                candidates,
                key=lambda index: (
                    abs(group_stats[index][1] - group_stats[index + 1][1]),
                    index,
                ),
            )
            work[merge_index]["values"].extend(work[merge_index + 1]["values"])
            del work[merge_index + 1]

        return [item["values"] for item in work]

    def _merge_common_user_category_groups_with_method(
        self,
        feature: str,
        x: pd.Series,
        y: pd.Series,
        groups: List[List[Any]],
    ) -> List[List[Any]]:
        """把用户类别组当作原子箱，由当前具体方法决定是否合并。"""
        group_codes = assign_category_groups(
            feature,
            x,
            groups,
            special_codes=self.special_codes,
            missing_separate=self.missing_separate,
            handle_unknown=UNKNOWN_BIN,
        )
        encoded = pd.Series(group_codes, index=x.index, name=feature, dtype=float)
        encoded.loc[encoded < 0] = np.nan

        # 当前函数已经处于按特征并行的 worker 内。内部只有一个编码字段，
        # 再启动 joblib 不会增加并行度，反而容易形成嵌套超额调度。
        method_worker = copy(self)
        for state_name in self._FEATURE_DICT_STATE:
            setattr(method_worker, state_name, {})
        for state_name in self._FEATURE_SET_STATE:
            setattr(method_worker, state_name, set())
        method_worker.user_splits = None
        method_worker.user_splits_fixed = None
        method_worker.n_jobs = 1
        method_worker._fit_feature(feature, encoded, y)

        numeric_splits = np.asarray(method_worker.splits_.get(feature, np.array([])), dtype=float)
        atomic_groups = restore_category_groups(list(range(len(groups))), numeric_splits)
        return [
            [value for group_index in atomic_group for value in groups[group_index]] for atomic_group in atomic_groups
        ]

    def _fit_features(self, X: pd.DataFrame, y: pd.Series, method_name: str) -> None:
        """事务式拟合所有特征，并按输入列顺序一次性提交状态。"""
        features = list(X.columns)
        task_methods = [
            "_fit_common_user_split_feature"
            if method_name == "_fit_feature" and self._has_explicit_user_rule(feature)
            else method_name
            for feature in features
        ]
        tasks = ((feature, X[feature], y, feature_method) for feature, feature_method in zip(features, task_methods))
        algorithm = self.__class__.__name__.lower()
        process_algorithms = ("genetic", "orbinning", "cpsat", "bestiv", "bestks", "mdlp")
        is_process_safe = any(name in algorithm for name in process_algorithms)
        is_quantile = "quantile" in algorithm
        is_uniform = "uniform" in algorithm
        cost = 20.0 if is_process_safe else 1.0 if is_quantile or is_uniform else 3.0
        auto_max_workers = 8 if is_quantile else 4 if is_uniform else None
        has_children = "genetic" in algorithm
        data_bytes = int(X.memory_usage(deep=True).sum())
        if hasattr(y, "memory_usage"):
            data_bytes += int(y.memory_usage(deep=True))
        workload = ParallelWorkload(
            task_count=len(features),
            rows=len(X),
            columns=len(features),
            data_bytes=data_bytes,
            cost_per_item=cost,
            capability="process_safe" if is_process_safe else "thread_safe",
            releases_gil=not is_process_safe,
            has_parallel_children=has_children,
            auto_max_workers=auto_max_workers,
            operation=f"{self.__class__.__name__}字段拟合",
        )
        results = self._parallel_execute(
            self._fit_feature_transaction,
            tasks,
            default_backend="loky" if is_process_safe else "threading",
            task_labels=features,
            has_parallel_children=has_children,
            workload=workload,
        )

        snapshots = {}
        for expected_feature, result in zip(features, results):
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(f"特征 '{expected_feature}' 的拟合结果格式无效")
            feature, state = result
            if feature != expected_feature or not isinstance(state, dict):
                raise TypeError(f"特征 '{expected_feature}' 的拟合结果无效")
            snapshots[feature] = state

        # 只有全部 worker 成功且结果通过校验后，才在主线程一次性替换本轮特征状态。
        for state_name in self._FEATURE_DICT_STATE:
            merged = {}
            for feature in features:
                feature_state = snapshots[feature].get(state_name, {})
                if feature in feature_state:
                    merged[feature] = feature_state[feature]
            if hasattr(self, state_name) or merged:
                setattr(self, state_name, merged)

        for state_name in self._FEATURE_SET_STATE:
            merged = set()
            for feature in features:
                if feature in snapshots[feature].get(state_name, set()):
                    merged.add(feature)
            setattr(self, state_name, merged)

    def _transform_features(self, X: pd.DataFrame, transform_one) -> pd.DataFrame:
        """并行只读转换各特征，并按输入列顺序拼接结果。"""
        features = list(X.columns)

        def _transform(feature):
            return feature, transform_one(feature)

        try:
            results = self._parallel_execute(
                _transform,
                features,
                default_backend="threading",
                task_labels=features,
                workload=ParallelWorkload(
                    task_count=len(features),
                    rows=len(X),
                    columns=len(features),
                    data_bytes=int(X.memory_usage(deep=True).sum()),
                    cost_per_item=1.0,
                    capability="thread_safe",
                    releases_gil=True,
                    operation=f"{self.__class__.__name__}字段转换",
                ),
            )
        except ParallelExecutionError as exc:
            # transform 历史上直接抛出未知类别等原始校验错误，保持该公共语义。
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise

        blocks = []
        for expected_feature, result in zip(features, results):
            if not isinstance(result, tuple) or len(result) != 2 or result[0] != expected_feature:
                raise TypeError(f"特征 '{expected_feature}' 的转换结果无效")
            values = result[1]
            if isinstance(values, pd.DataFrame):
                block = values.copy()
                block.index = X.index
            elif isinstance(values, pd.Series):
                block = values.copy()
                block.index = X.index
                block.name = expected_feature
            else:
                block = pd.Series(values, index=X.index, name=expected_feature)
            blocks.append(block)
        if not blocks:
            return pd.DataFrame(index=X.index)
        return pd.concat(blocks, axis=1)

    def _transform_binning_features(
        self,
        X: pd.DataFrame,
        metric: str,
        assign_bins,
        *,
        missing_feature: str = "passthrough",
        woe_default: Optional[float] = None,
        extra_metric=None,
    ) -> pd.DataFrame:
        """统一生成单列分箱结果，并交由有序只读转换 helper 执行。"""

        def _transform_one(feature):
            if feature not in self.splits_:
                if missing_feature == "error":
                    raise KeyError(f"特征 '{feature}' 未在训练数据中找到")
                return X[feature].copy()

            bins = self._apply_reserved_bin_policy(feature, X[feature], assign_bins(feature))
            if metric == "indices":
                return bins
            if metric == "bins":
                return self._assign_bin_labels(feature, bins)
            if metric == "woe":
                if hasattr(self, "_woe_maps_") and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(bin_table["分箱"].astype(int), bin_table["分档WOE值"].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                if woe_default is None:
                    return pd.Series(bins, index=X.index).map(woe_map).to_numpy()
                return np.asarray([woe_map.get(value, woe_default) for value in bins])
            if extra_metric is not None:
                values = extra_metric(feature, bins, metric)
                if values is not None:
                    return values
            raise ValueError(f"不支持的metric: {metric}")

        return self._transform_features(X, _transform_one)

    def _transform_lift_metric(self, feature: str, bins: np.ndarray, metric: str):
        """从当前分箱统计表生成只读 LIFT 映射结果。"""
        if metric != "lift":
            return None
        bin_table = self._get_lift_bin_table(feature)
        lift_map = {}
        for _, row in bin_table.iterrows():
            bin_index = int(row["分箱"])
            lift_map[bin_index] = np.nan if bin_index < 0 else row["LIFT值"]
        return np.asarray([lift_map.get(value, np.nan) for value in bins])

    def _get_lift_bin_table(self, feature: str) -> pd.DataFrame:
        """按 BestLift 的公开口径从当前统计状态计算 LIFT 表。"""
        if feature not in self.bin_tables_:
            raise KeyError(f"特征 '{feature}' 未找到")
        bin_table = self.bin_tables_[feature].copy()
        valid_mask = ~bin_table["分箱标签"].isin(["缺失", "special"])
        if valid_mask.any():
            valid_bad_rates = bin_table.loc[valid_mask, "坏样本率"]
            valid_counts = bin_table.loc[valid_mask, "样本总数"]
            total_bad = (valid_bad_rates * valid_counts).sum()
            total_count = valid_counts.sum()
            total_bad_rate = total_bad / total_count if total_count > 0 else 0
        else:
            total_bad_rate = bin_table["坏样本率"].mean()
        bin_table["LIFT值"] = [
            np.nan if row["分箱标签"] in ["缺失", "special"] else row["坏样本率"] / total_bad_rate if total_bad_rate > 0 else 1.0
            for _, row in bin_table.iterrows()
        ]
        return bin_table

    def _get_min_samples(self, n_samples: int) -> int:
        """计算最小样本数.

        根据min_bin_size参数计算每个分箱的最小样本数。

        :param n_samples: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size < 1:
            # 比例值，如0.05表示5%
            return max(1, int(n_samples * self.min_bin_size))
        else:
            # 绝对数量
            return max(1, int(self.min_bin_size))

    def _get_max_samples(self, n_samples: int) -> Optional[int]:
        """计算最大样本数。"""
        if self.max_bin_size is None:
            return None
        if self.max_bin_size < 1:
            return max(1, int(np.ceil(n_samples * self.max_bin_size)))
        return max(1, int(self.max_bin_size))

    def _choose_merge_split_index(
        self,
        counts: np.ndarray,
        bad_counts: np.ndarray,
        bin_idx: int,
    ) -> Optional[int]:
        """为样本量不足的分箱选择要删除的切分点索引。"""
        n_bins = len(counts)
        if n_bins <= max(1, self.min_n_bins):
            return None
        if bin_idx <= 0:
            return 0
        if bin_idx >= n_bins - 1:
            return n_bins - 2

        bad_rates = bad_counts / np.maximum(counts, 1.0)
        left_score = abs(bad_rates[bin_idx] - bad_rates[bin_idx - 1])
        right_score = abs(bad_rates[bin_idx] - bad_rates[bin_idx + 1])
        return bin_idx - 1 if left_score <= right_score else bin_idx

    def _choose_split_point_within_bin(
        self, x: pd.Series, bins: np.ndarray, bin_idx: int, min_samples: int
    ) -> Optional[float]:
        """为样本量过大的分箱选择新的切分点。"""
        values = np.sort(pd.to_numeric(x[bins == bin_idx], errors="coerce").dropna().to_numpy(dtype=float))
        if len(values) < max(2, min_samples * 2):
            return None

        center = len(values) // 2
        candidate_positions = sorted(
            range(min_samples, len(values) - min_samples + 1), key=lambda pos: abs(pos - center)
        )

        for pos in candidate_positions:
            left_value = values[pos - 1]
            right_value = values[pos]
            if np.isclose(left_value, right_value, atol=1e-12, rtol=0):
                continue
            return float((left_value + right_value) / 2.0)
        return None

    def _find_feasible_categorical_size_splits(
        self,
        x: pd.Series,
        current: np.ndarray,
        min_samples: int,
        max_samples: Optional[int],
    ) -> Optional[np.ndarray]:
        """为有序类别原子组搜索满足样本量约束的连续分区。"""
        values, atomic_counts = np.unique(x.to_numpy(dtype=float), return_counts=True)
        n_categories = len(values)
        if n_categories == 0:
            return current

        min_bins = min(max(1, self.min_n_bins), n_categories)
        max_bins = min(max(1, self.max_n_bins), n_categories)
        if min_bins > max_bins:
            return None

        current_bins = np.digitize(x, current) if len(current) > 0 else np.zeros(len(x), dtype=int)
        current_counts = np.bincount(current_bins, minlength=len(current) + 1).astype(int)
        current_is_feasible = (
            min_bins <= len(current_counts) <= max_bins
            and np.all(current_counts >= min_samples)
            and (max_samples is None or np.all(current_counts <= max_samples))
        )
        if current_is_feasible:
            return current

        prefix_counts = np.concatenate(([0], np.cumsum(atomic_counts, dtype=int)))
        total_samples = int(prefix_counts[-1])
        current_positions = tuple(
            sorted(
                {
                    int(np.searchsorted(values, split, side="right"))
                    for split in current
                    if values[0] < split < values[-1]
                }
            )
        )
        current_position_set = set(current_positions)
        preferred_bins = len(current_positions) + 1
        movement_costs = {
            index: (float(min(abs(index - position) for position in current_positions)) if current_positions else 0.0)
            for index in range(1, n_categories)
        }
        best_key = None
        best_cuts = None

        states = {0: ((0, 0.0), tuple())}
        for n_bins in range(1, max_bins + 1):
            next_states = {}
            window = deque()
            next_start = 0

            for end in range(1, n_categories + 1):
                max_prefix = int(prefix_counts[end] - min_samples)
                last_start = min(
                    end - 1,
                    int(np.searchsorted(prefix_counts, max_prefix, side="right")) - 1,
                )
                while next_start <= last_start:
                    previous = states.get(next_start)
                    if previous is not None:
                        while window and previous < window[-1][0]:
                            window.pop()
                        window.append((previous, next_start))
                    next_start += 1

                first_start = 0
                if max_samples is not None:
                    min_prefix = int(prefix_counts[end] - max_samples)
                    first_start = int(np.searchsorted(prefix_counts, min_prefix, side="left"))
                while window and window[0][1] < first_start:
                    window.popleft()
                if not window:
                    continue

                cost, cuts = window[0][0]
                if end < n_categories:
                    is_new_boundary = int(end not in current_position_set)
                    cost = (cost[0] + is_new_boundary, cost[1] + movement_costs[end])
                    cuts = cuts + (end,)
                next_states[end] = (cost, cuts)

            states = next_states
            if not states:
                break
            if n_bins < min_bins:
                continue

            solution = states.get(n_categories)
            if solution is None:
                continue

            cost, cuts = solution
            shared_boundaries = (n_bins - 1) - cost[0]
            boundary_changes = len(current_positions) + (n_bins - 1) - 2 * shared_boundaries
            balance = sum(
                abs(float(prefix_counts[index]) - total_samples * position / n_bins)
                for position, index in enumerate(cuts, start=1)
            )
            candidate_key = (
                abs(n_bins - preferred_bins),
                boundary_changes,
                cost[1],
                balance,
                cuts,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_cuts = cuts

        if best_cuts is None:
            return None
        return np.asarray([(values[index - 1] + values[index]) / 2.0 for index in best_cuts], dtype=float)

    def _adjust_splits_for_bin_size_constraints(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: Union[np.ndarray, list],
        min_samples: int,
        max_samples: Optional[int],
        *,
        allow_empty_splits: bool = False,
        allow_boundary_relocation: bool = False,
    ) -> np.ndarray:
        """调整切分点以满足最小/最大样本量约束。"""
        if (splits is None or len(splits) == 0) and not allow_empty_splits:
            return np.array([])

        current = np.array([], dtype=float) if splits is None else np.unique(np.sort(np.asarray(splits, dtype=float)))
        x_numeric = pd.to_numeric(x, errors="coerce")
        valid_mask = x_numeric.notna()
        if self.special_codes:
            for code in self.special_codes:
                valid_mask &= x_numeric != code
        x_valid = x_numeric[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) == 0:
            return current

        if allow_boundary_relocation:
            feasible = self._find_feasible_categorical_size_splits(
                x_valid,
                current,
                min_samples,
                max_samples,
            )
            return current if feasible is None else feasible

        max_splits_allowed = max(0, self.max_n_bins - 1)

        for _ in range(200):
            bins = np.digitize(x_valid, current) if len(current) > 0 else np.zeros(len(x_valid), dtype=int)
            counts = np.bincount(bins, minlength=len(current) + 1).astype(int)
            bad_counts = np.bincount(bins, weights=y_valid, minlength=len(current) + 1).astype(float)

            small_bins = np.where(counts < min_samples)[0]
            large_bins = np.array([], dtype=int) if max_samples is None else np.where(counts > max_samples)[0]

            changed = False

            if len(small_bins) > 0:
                merge_bin = int(small_bins[np.argmin(counts[small_bins])])
                split_idx = self._choose_merge_split_index(counts, bad_counts, merge_bin)
                if split_idx is not None and 0 <= split_idx < len(current):
                    current = np.delete(current, split_idx)
                    changed = True

            if changed:
                continue

            if len(large_bins) > 0 and len(current) < max_splits_allowed:
                split_bin = int(large_bins[np.argmax(counts[large_bins])])
                new_split = self._choose_split_point_within_bin(x_valid, bins, split_bin, min_samples)
                if new_split is not None and np.isfinite(new_split):
                    current = np.unique(np.sort(np.append(current, new_split)))
                    changed = True

            if not changed:
                break

        return current

    def _enforce_bin_size_constraints(self, X: pd.DataFrame, y: pd.Series) -> None:
        """统一收口最小/最大分箱样本量约束。"""
        for feature in X.columns:
            if feature not in self.splits_:
                continue
            if self.feature_types_.get(feature) != "numerical":
                continue

            splits = self.splits_.get(feature)
            if splits is None:
                continue

            min_samples = self._get_min_samples(len(y))
            max_samples = self._get_max_samples(len(y))
            adjusted = self._adjust_splits_for_bin_size_constraints(X[feature], y, splits, min_samples, max_samples)
            adjusted = self._round_splits(adjusted)

            old_splits = np.asarray(splits, dtype=float) if len(splits) > 0 else np.array([])
            if np.array_equal(adjusted, old_splits):
                continue

            self.splits_[feature] = adjusted
            self.n_bins_[feature] = len(adjusted) + 1

            apply_bins = getattr(self, "_apply_bins", None)
            if callable(apply_bins):
                try:
                    bins = apply_bins(X[feature], adjusted, "numerical", feature)
                except TypeError:
                    try:
                        bins = apply_bins(X[feature], adjusted, feature)
                    except TypeError:
                        bins = apply_bins(X[feature], adjusted)
            else:
                values = X[feature]
                bins = np.zeros(len(values), dtype=int)
                if self.missing_separate:
                    bins[pd.isna(values)] = -1
                mask = pd.notna(values)
                if self.special_codes:
                    for code in self.special_codes:
                        bins[values == code] = -2
                        mask &= values != code
                if len(adjusted) > 0:
                    bins[mask] = np.digitize(pd.to_numeric(values[mask], errors="coerce"), adjusted)
                else:
                    bins[mask] = 0
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)

    def _enforce_max_n_bins_hard_cap(self, X: pd.DataFrame, y: pd.Series) -> None:
        """硬性限制最大分箱数，不超过 max_n_bins。

        当其他约束调整后切分点仍超出限制时，按相邻箱坏样本率差异最小的
        优先合并策略进行截断。
        """
        max_splits = max(0, self.max_n_bins - 1)
        for feature in X.columns:
            if feature not in self.splits_:
                continue
            if self.feature_types_.get(feature) != "numerical":
                continue
            splits = self.splits_[feature]
            if splits is None or len(splits) <= max_splits:
                continue

            current = np.unique(np.sort(np.asarray(splits, dtype=float)))
            x = X[feature]
            while len(current) > max_splits:
                bins = self._get_feature_bins(feature, x, current)
                bin_table = self._compute_bin_stats(feature, x, y, bins)
                valid = bin_table[bin_table["分箱"] >= 0].reset_index(drop=True)
                bad_rates = valid["坏样本率"].to_numpy(dtype=float)
                if len(bad_rates) <= 2:
                    current = current[:max_splits]
                    break
                diffs = np.abs(np.diff(bad_rates))
                merge_idx = int(np.argmin(diffs))
                if merge_idx < 0 or merge_idx >= len(current):
                    current = current[:max_splits]
                    break
                current = np.delete(current, merge_idx)

            self.splits_[feature] = self._round_splits(current)
            self.n_bins_[feature] = len(current) + 1
            bins = self._get_feature_bins(feature, x, current)
            self.bin_tables_[feature] = self._compute_bin_stats(feature, x, y, bins)

    def _enforce_bad_rate_constraints(self, X: pd.DataFrame, y: pd.Series) -> None:
        """合并退化分箱与坏样本率不达标的分箱。

        - **退化分箱**：坏样本数为 0 或好样本数为 0（坏样本率为 0 或 1）。这类分箱的
          WOE 会趋于 ±∞（受平滑参数影响表现为极端值，如 ±23），既使 IV 虚高、又让
          评分卡分数异常，始终予以合并。
        - **坏样本率不达标分箱**：坏样本率低于 ``min_bad_rate`` 的分箱（仅当
          ``min_bad_rate > 0`` 时生效）。

        合并策略：将违规分箱与相邻坏样本率最接近的分箱合并（删除对应切分点），
        在不低于 ``min_n_bins`` 的前提下迭代进行。数值型特征生效。
        """
        min_bad_rate = float(getattr(self, "min_bad_rate", 0.0) or 0.0)

        for feature in X.columns:
            if feature not in self.splits_:
                continue
            if self.feature_types_.get(feature) != "numerical":
                continue
            splits = self.splits_.get(feature)
            if splits is None or len(splits) == 0:
                continue

            current = np.unique(np.sort(np.asarray(splits, dtype=float)))
            x = X[feature]
            changed = False

            for _ in range(200):
                if len(current) == 0:
                    break
                bins = self._get_feature_bins(feature, x, current)
                bin_table = self._compute_bin_stats(feature, x, y, bins)
                valid = bin_table[bin_table["分箱"] >= 0].reset_index(drop=True)
                n_bins = len(valid)
                if n_bins <= max(1, self.min_n_bins):
                    break

                counts = valid["样本总数"].to_numpy(dtype=float)
                bad_counts = valid["坏样本数"].to_numpy(dtype=float)
                good_counts = counts - bad_counts
                bad_rates = valid["坏样本率"].to_numpy(dtype=float)

                # 退化箱（无坏/无好）始终合并；坏样本率低于阈值的箱按需合并
                degenerate = (bad_counts <= 0) | (good_counts <= 0)
                below = (bad_rates < min_bad_rate) if min_bad_rate > 0 else np.zeros(n_bins, dtype=bool)
                violating = np.where(degenerate | below)[0]
                if len(violating) == 0:
                    break

                split_idx = self._choose_merge_split_index(counts.astype(int), bad_counts, int(violating[0]))
                if split_idx is None or split_idx < 0 or split_idx >= len(current):
                    break
                current = np.delete(current, split_idx)
                changed = True

            if not changed:
                continue

            self.splits_[feature] = self._round_splits(current)
            self.n_bins_[feature] = len(current) + 1
            bins = self._get_feature_bins(feature, x, self.splits_[feature])
            self.bin_tables_[feature] = self._compute_bin_stats(feature, x, y, bins)

    def _apply_post_fit_constraints(
        self, X: pd.DataFrame, y: pd.Series, enforce_monotonic: bool = True, enforce_bad_rate: bool = True
    ) -> None:
        """拟合后统一收口约束。

        :param enforce_bad_rate: 是否合并退化分箱/坏样本率不达标分箱，默认 True。
            等宽(uniform)/等频(quantile)等机械分箱以"精确切分结构"为契约，应传 False。
        """
        # 严格用户规则是固定分箱，后处理不得移动或合并其边界。
        # 非严格规则仍走各具体分箱器原有的约束收口逻辑。
        if any(self._is_user_rule_fully_fixed(feature) for feature in X.columns):
            mutable_features = [feature for feature in X.columns if not self._is_user_rule_fully_fixed(feature)]
            if not mutable_features:
                self._restore_fixed_user_splits(X, y)
                return
            X = X.loc[:, mutable_features]

        self._enforce_bin_size_constraints(X, y)

        monotonic_adjuster = getattr(self, "_apply_monotonic_adjustment", None)
        if enforce_monotonic and self.monotonic and callable(monotonic_adjuster):
            monotonic_adjuster(X, y)
            self._enforce_bin_size_constraints(X, y)

        # 合并退化分箱（坏样本率 0/1）及坏样本率低于 min_bad_rate 的分箱
        if enforce_bad_rate:
            self._enforce_bad_rate_constraints(X, y)

        # 最终硬性限制：确保不超过 max_n_bins
        self._enforce_max_n_bins_hard_cap(X, y)
        self._restore_fixed_user_splits(X, y)

    def _restore_fixed_user_splits(self, X: pd.DataFrame, y: pd.Series) -> None:
        """约束优化后恢复选择性固定的数值节点，并只裁剪可变节点。"""
        for feature in X.columns:
            mask = self._user_splits_fixed_masks_.get(feature, [])
            if not any(mask) or self.feature_types_.get(feature) != "numerical":
                continue
            rule = list(self._get_explicit_user_rule(feature, X[feature]))
            original, _ = parse_numerical_user_splits(feature, rule)
            protected = [float(value) for value, fixed in zip(original, mask) if fixed]
            if all(mask):
                current = np.asarray(original, dtype=float)
            else:
                current = np.unique(
                    np.sort(np.append(np.asarray(self.splits_.get(feature, []), dtype=float), protected))
                )
            max_splits = max(0, self.max_n_bins - 1)
            while len(current) > max_splits:
                removable = [
                    index
                    for index, value in enumerate(current)
                    if not any(np.isclose(value, fixed, rtol=0, atol=1e-12) for fixed in protected)
                ]
                if not removable:
                    raise ValueError(f"特征 '{feature}' 的固定切分点数量无法满足 max_n_bins={self.max_n_bins}")
                bins = np.digitize(pd.to_numeric(X[feature], errors="coerce"), current)
                table = self._compute_bin_stats(feature, X[feature], y, bins)
                rates = table[table["分箱"] >= 0].sort_values("分箱")["坏样本率"].to_numpy(dtype=float)
                scores = {}
                for index in removable:
                    adjacent = []
                    if index < len(rates) - 1:
                        adjacent.append(abs(rates[index] - rates[index + 1]))
                    if index > 0:
                        adjacent.append(abs(rates[index] - rates[index - 1]))
                    scores[index] = min(adjacent) if adjacent else 0.0
                current = np.delete(current, min(removable, key=lambda index: (scores[index], index)))

            self.splits_[feature] = current
            self.n_bins_[feature] = len(current) + 1
            bins = self._assign_base_feature_bins(feature, X[feature])
            bins = self._apply_reserved_bin_policy(feature, X[feature], bins)
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)

    def _get_feature_bins(self, feature: str, x: pd.Series, splits: Union[np.ndarray, list]) -> np.ndarray:
        """获取指定特征切分点对应的分箱索引。"""
        apply_bins = getattr(self, "_apply_bins", None)
        if callable(apply_bins):
            try:
                return apply_bins(x, splits, "numerical", feature)
            except TypeError:
                try:
                    return apply_bins(x, splits, feature)
                except TypeError:
                    return apply_bins(x, splits)

        bins = np.zeros(len(x), dtype=int)
        if self.missing_separate:
            bins[pd.isna(x)] = -1

        mask = pd.notna(x)
        if self.special_codes:
            for code in self.special_codes:
                bins[x == code] = -2
                mask &= x != code

        if len(splits) > 0:
            bins[mask] = np.digitize(pd.to_numeric(x[mask], errors="coerce"), splits)
        else:
            bins[mask] = 0
        return bins

    def _special_mask(self, x: pd.Series) -> np.ndarray:
        """返回特殊值掩码，并正确识别特殊配置中的缺失标记。"""
        mask = np.zeros(len(x), dtype=bool)
        for code in self.special_codes or []:
            if is_missing_marker(code):
                mask |= x.isna().to_numpy(dtype=bool)
            else:
                mask |= x.eq(code).fillna(False).to_numpy(dtype=bool)
        return mask

    def _categorical_group_target(self, feature: str, predicate) -> Optional[int]:
        """查找类别规则中由 predicate 命中的用户箱。"""
        for bin_index, group in enumerate(self._cat_bins_.get(feature, [])):
            if any(predicate(value) for value in group):
                return bin_index
        return None

    def _assign_base_feature_bins(self, feature: str, x: pd.Series) -> np.ndarray:
        """仅根据训练规则生成基础箱号，保留箱优先级随后统一收口。"""
        if self.feature_types_.get(feature) == "categorical" and feature in self._cat_bins_:
            return self._assign_categorical_bins(feature, x)

        splits = np.asarray(self.splits_.get(feature, np.array([])), dtype=float)
        numeric = pd.to_numeric(x, errors="coerce")
        missing = x.isna().to_numpy(dtype=bool)
        valid = numeric.notna().to_numpy(dtype=bool) & ~missing
        unknown_bin = UNKNOWN_BIN if self.handle_unknown == "raise" else self.handle_unknown
        bins = np.full(len(x), unknown_bin, dtype=int)
        bins[missing] = MISSING_BIN
        if len(splits) > 0:
            bins[valid] = np.digitize(numeric.to_numpy(dtype=float)[valid], splits)
        else:
            bins[valid] = 0
        return bins

    def _apply_reserved_bin_policy(self, feature: str, x: pd.Series, bins: np.ndarray) -> np.ndarray:
        """按 user_splits > special_codes > missing_separate/学习映射收口箱号。"""
        result = np.asarray(bins, dtype=int).copy()
        missing = x.isna().to_numpy(dtype=bool)
        special = self._special_mask(x)

        user_missing_target = self._user_missing_bin_targets_.get(feature)
        if user_missing_target is None and self.feature_types_.get(feature) == "categorical":
            user_missing_target = self._categorical_group_target(feature, is_missing_marker)

        if user_missing_target is not None:
            result[missing] = int(user_missing_target)
            special &= ~missing
        else:
            result[special] = SPECIAL_BIN
            unresolved_missing = missing & ~special
            learned = self._missing_bin_targets_.get(feature)
            result[unresolved_missing] = MISSING_BIN if learned is None else int(learned)

        if self.feature_types_.get(feature) == "categorical" and self.special_codes:
            for code in self.special_codes:
                claimed_target = self._categorical_group_target(
                    feature,
                    lambda value, expected=code: is_missing_marker(value)
                    if is_missing_marker(expected)
                    else (not is_missing_marker(value) and type(value) is type(expected) and value == expected),
                )
                if claimed_target is None:
                    continue
                code_mask = (
                    x.isna().to_numpy(dtype=bool)
                    if is_missing_marker(code)
                    else x.eq(code).fillna(False).to_numpy(dtype=bool)
                )
                result[code_mask] = claimed_target
        if self.handle_unknown == "raise":
            unresolved_unknown = (result == UNKNOWN_BIN) & ~missing & ~special
            if unresolved_unknown.any():
                unknown_values = x.loc[unresolved_unknown].drop_duplicates().tolist()
                raise ValueError(f"特征 '{feature}' 在 transform 中出现训练期未知类别: {unknown_values}")
        return result

    def _learn_missing_bin_target(self, feature: str, x: pd.Series, y: pd.Series, bins: np.ndarray) -> np.ndarray:
        """将临时缺失箱合并到坏样本率最接近的已有普通箱。"""
        missing = x.isna().to_numpy(dtype=bool)
        if not missing.any() or self.missing_separate:
            return bins
        if feature in self._user_missing_bin_targets_:
            return bins
        if (
            self.feature_types_.get(feature) == "categorical"
            and self._categorical_group_target(feature, is_missing_marker) is not None
        ):
            return bins
        if np.any(self._special_mask(x) & missing):
            return bins

        ordinary = sorted(int(value) for value in np.unique(bins[~missing]) if int(value) >= 0)
        if not ordinary:
            return bins
        y_values = y.to_numpy(dtype=float)
        missing_rate = float(y_values[missing].mean())
        target = min(
            ordinary,
            key=lambda bin_index: (
                abs(float(y_values[(bins == bin_index) & ~missing].mean()) - missing_rate),
                bin_index,
            ),
        )
        self._missing_bin_targets_[feature] = target
        learned = bins.copy()
        learned[missing] = target
        return learned

    def _materialize_unknown_bin(self, feature: str, table: pd.DataFrame) -> pd.DataFrame:
        """记录默认未知箱，并校验用户指定未知箱是否真实存在。"""
        recorded = {int(value) for value in table["分箱"].tolist()}
        if self.handle_unknown == "raise":
            self._recorded_bins_[feature] = recorded
            return table
        if self.handle_unknown != UNKNOWN_BIN:
            if self.handle_unknown not in recorded:
                raise ValueError(f"特征 '{feature}' 的 handle_unknown={self.handle_unknown} 在训练结果中无记录")
            self._recorded_bins_[feature] = recorded
            return table
        recorded.add(UNKNOWN_BIN)
        self._recorded_bins_[feature] = recorded
        return table

    def _finalize_reserved_bins(self, X: pd.DataFrame, y: pd.Series) -> None:
        """统一学习缺失归属、重算训练统计并验证未知箱配置。"""
        self._missing_bin_targets_ = {}
        self._reserved_bins_finalized_ = set()
        for feature in self.splits_:
            if feature not in X.columns:
                continue
            x = self._categorical_fit_context_.get(feature, X[feature])
            bins = self._assign_base_feature_bins(feature, x)
            bins = self._apply_reserved_bin_policy(feature, x, bins)
            bins = self._learn_missing_bin_target(feature, x, y, bins)
            table = self._compute_bin_stats(feature, x, y, bins)
            table = self._materialize_unknown_bin(feature, table)
            self.bin_tables_[feature] = table
            imported_snapshot = getattr(self, "_imported_rule_snapshot_", {})
            imported_woe = imported_snapshot.get("_woe_maps_", {}).get(feature)
            if imported_woe is not None:
                self._woe_maps_[feature] = {int(key): float(value) for key, value in imported_woe.items()}
            else:
                self._woe_maps_[feature] = {
                    int(row["分箱"]): float(row["分档WOE值"]) for _, row in table.iterrows() if "分档WOE值" in table.columns
                }
            if self.handle_unknown == UNKNOWN_BIN:
                self._woe_maps_[feature].setdefault(UNKNOWN_BIN, 0.0)
            self._reserved_bins_finalized_.add(feature)
        self._categorical_fit_context_ = {}
        self._categorical_encoded_features_ = set()

    def _resolve_monotonic_target_mode(self, bad_rates: np.ndarray, target_mode: Union[bool, str]) -> str:
        """为自动单调模式选择目标趋势。"""
        if target_mode in ["ascending", "descending", "peak", "valley"]:
            return target_mode

        if target_mode in ["auto_asc_desc"]:
            candidates = ["ascending", "descending"]
        else:
            candidates = ["ascending", "descending", "peak", "valley"]

        best_mode = candidates[0]
        best_score = None
        for mode in candidates:
            violations = self._count_monotonic_violations(bad_rates, mode)
            score = (violations, 0 if mode in ["ascending", "descending"] else 1)
            if best_score is None or score < best_score:
                best_score = score
                best_mode = mode
        return best_mode

    def _count_monotonic_violations(self, bad_rates: np.ndarray, mode: str) -> int:
        """计算指定趋势下的违例数量。"""
        tol = 1e-10
        diffs = np.diff(bad_rates)
        if len(diffs) == 0:
            return 0
        if mode == "ascending":
            return int(np.sum(diffs < -tol))
        if mode == "descending":
            return int(np.sum(diffs > tol))
        if mode == "peak":
            if len(bad_rates) < 3:
                return self._count_monotonic_violations(bad_rates, "descending")
            return min(
                self._count_monotonic_violations(bad_rates[: pivot + 1], "ascending")
                + self._count_monotonic_violations(bad_rates[pivot:], "descending")
                for pivot in range(1, len(bad_rates) - 1)
            )
        if mode == "valley":
            if len(bad_rates) < 3:
                return self._count_monotonic_violations(bad_rates, "ascending")
            return min(
                self._count_monotonic_violations(bad_rates[: pivot + 1], "descending")
                + self._count_monotonic_violations(bad_rates[pivot:], "ascending")
                for pivot in range(1, len(bad_rates) - 1)
            )
        return 0

    def _choose_monotonic_merge_index(self, bad_rates: np.ndarray, mode: str) -> int:
        """选择需要移除的切分点索引。"""
        tol = 1e-10
        diffs = np.diff(bad_rates)
        if len(diffs) == 0:
            return 0
        if mode == "ascending":
            violations = np.where(diffs < -tol)[0]
            return int(violations[np.argmin(diffs[violations])]) if len(violations) > 0 else 0
        if mode == "descending":
            violations = np.where(diffs > tol)[0]
            return int(violations[np.argmax(diffs[violations])]) if len(violations) > 0 else 0
        if mode == "peak":
            best_pivot = min(
                range(1, len(bad_rates) - 1),
                key=lambda pivot: self._count_monotonic_violations(bad_rates[: pivot + 1], "ascending")
                + self._count_monotonic_violations(bad_rates[pivot:], "descending"),
            )
            left_diffs = diffs[:best_pivot]
            right_diffs = diffs[best_pivot:]
            left_idx = np.where(left_diffs < -tol)[0]
            right_idx = np.where(right_diffs > tol)[0]
            left_choice = None if len(left_idx) == 0 else int(left_idx[np.argmin(left_diffs[left_idx])])
            right_choice = (
                None if len(right_idx) == 0 else int(best_pivot + right_idx[np.argmax(right_diffs[right_idx])])
            )
            if left_choice is None:
                return right_choice if right_choice is not None else 0
            if right_choice is None:
                return left_choice
            return left_choice if abs(diffs[left_choice]) >= abs(diffs[right_choice]) else right_choice
        if mode == "valley":
            best_pivot = min(
                range(1, len(bad_rates) - 1),
                key=lambda pivot: self._count_monotonic_violations(bad_rates[: pivot + 1], "descending")
                + self._count_monotonic_violations(bad_rates[pivot:], "ascending"),
            )
            left_diffs = diffs[:best_pivot]
            right_diffs = diffs[best_pivot:]
            left_idx = np.where(left_diffs > tol)[0]
            right_idx = np.where(right_diffs < -tol)[0]
            left_choice = None if len(left_idx) == 0 else int(left_idx[np.argmax(left_diffs[left_idx])])
            right_choice = (
                None if len(right_idx) == 0 else int(best_pivot + right_idx[np.argmin(right_diffs[right_idx])])
            )
            if left_choice is None:
                return right_choice if right_choice is not None else 0
            if right_choice is None:
                return left_choice
            return left_choice if abs(diffs[left_choice]) >= abs(diffs[right_choice]) else right_choice
        return 0

    def _merge_splits_for_monotonicity(
        self, feature: str, x: pd.Series, y: pd.Series, splits: Union[np.ndarray, list], target_mode: Union[bool, str]
    ) -> Tuple[np.ndarray, str]:
        """基于当前切分点，通过相邻合并满足单调约束。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        if len(current) == 0:
            return current, "unknown"

        min_splits_allowed = max(0, self.min_n_bins - 1)
        final_mode = str(target_mode)

        for _ in range(200):
            bins = self._get_feature_bins(feature, x, current)
            bin_table = self._compute_bin_stats(feature, x, y, bins)
            valid_bins = bin_table[bin_table["分箱"] >= 0]
            bad_rates = valid_bins["坏样本率"].to_numpy(dtype=float)
            if len(bad_rates) <= 1:
                return current, final_mode

            final_mode = self._resolve_monotonic_target_mode(bad_rates, target_mode)
            violations = self._count_monotonic_violations(bad_rates, final_mode)
            max_splits_allowed = max(0, self.max_n_bins - 1)
            if violations == 0 and len(current) <= max_splits_allowed:
                return current, final_mode
            if len(current) <= min_splits_allowed:
                return current, final_mode

            if violations == 0:
                diffs = np.abs(np.diff(bad_rates))
                merge_idx = int(np.argmin(diffs)) if len(diffs) > 0 else len(current) - 1
            else:
                merge_idx = self._choose_monotonic_merge_index(bad_rates, final_mode)
            if merge_idx < 0 or merge_idx >= len(current):
                break
            current = np.delete(current, merge_idx)

        return current, final_mode

    def _count_adjacent_zero_bad_rate_pairs(self, bad_rates: np.ndarray) -> int:
        """统计相邻全零坏样本率箱对数。"""
        arr = np.asarray(bad_rates, dtype=float)
        if len(arr) <= 1:
            return 0
        zero_mask = arr <= 1e-12
        return int(np.sum(zero_mask[:-1] & zero_mask[1:]))

    def _merge_adjacent_zero_bad_rate_bins(
        self, feature: str, x: pd.Series, y: pd.Series, splits: Union[np.ndarray, list]
    ) -> np.ndarray:
        """合并相邻坏样本率全为 0 的分箱。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        if len(current) == 0:
            return current

        for _ in range(200):
            bins = self._get_feature_bins(feature, x, current)
            bin_table = self._compute_bin_stats(feature, x, y, bins)
            valid = bin_table[bin_table["分箱"] >= 0].reset_index(drop=True)
            bad_rates = valid["坏样本率"].to_numpy(dtype=float)
            zero_pairs = (
                np.where((bad_rates[:-1] <= 1e-12) & (bad_rates[1:] <= 1e-12))[0]
                if len(bad_rates) > 1
                else np.array([])
            )
            if len(zero_pairs) == 0:
                break
            if len(current) == 0:
                break
            merge_idx = int(zero_pairs[0])
            if merge_idx < 0 or merge_idx >= len(current):
                break
            current = np.delete(current, merge_idx)

        return current

    def _apply_monotonic_adjustment(self, X: pd.DataFrame, y: pd.Series) -> None:
        """基于当前方法的切分点执行单调性收口。"""
        needs_adjustment_modes = [
            "auto",
            "auto_asc_desc",
            "auto_heuristic",
            "ascending",
            "descending",
            "peak",
            "valley",
            "peak_heuristic",
            "valley_heuristic",
            True,
        ]
        if self.monotonic not in needs_adjustment_modes:
            return

        monotonic_trend = getattr(self, "monotonic_trend_", None)
        if monotonic_trend is None:
            self.monotonic_trend_ = {}

        for feature in X.columns:
            if feature not in self.splits_:
                continue
            if self.feature_types_.get(feature) != "numerical":
                continue
            splits = self.splits_.get(feature)
            if splits is None or len(splits) == 0:
                continue

            bins = self._get_feature_bins(feature, X[feature], splits)
            bin_table = self._compute_bin_stats(feature, X[feature], y, bins)
            valid_bins = bin_table[bin_table["分箱"] >= 0]
            bad_rates = valid_bins["坏样本率"].to_numpy(dtype=float)
            if len(bad_rates) <= 1:
                continue

            target_mode = self._resolve_monotonic_target_mode(bad_rates, self.monotonic)
            if self._count_monotonic_violations(bad_rates, target_mode) == 0:
                adjusted_splits = np.unique(np.sort(np.asarray(splits, dtype=float)))
                final_mode = target_mode
            else:
                adjusted_splits, final_mode = self._merge_splits_for_monotonicity(
                    feature, X[feature], y, splits, target_mode
                )

            adjusted_splits = self._expand_splits_with_monotonicity(feature, X[feature], y, adjusted_splits, final_mode)
            adjusted_splits = self._merge_adjacent_zero_bad_rate_bins(feature, X[feature], y, adjusted_splits)
            self.splits_[feature] = self._round_splits(adjusted_splits)
            self.n_bins_[feature] = len(self.splits_[feature]) + 1
            bins = self._get_feature_bins(feature, X[feature], self.splits_[feature])
            self.bin_tables_[feature] = self._compute_bin_stats(feature, X[feature], y, bins)
            self.monotonic_trend_[feature] = final_mode

    def _quadratic_curve_coefficient(self, values: np.ndarray, trend: Optional[str] = None) -> float:
        """计算曲线二次拟合系数。

        ``trend`` 为 ``'ascending'``/``'descending'`` 时使用带单调约束的二次拟合，
        保证拟合曲线在区间内单调（抛物线顶点落在区间之外）；
        其他情况（如 U 形/倒 U 形）使用无约束二次拟合。
        """
        arr = np.asarray(values, dtype=float)
        if len(arr) < 3 or np.allclose(arr, arr[0], atol=1e-12, rtol=0):
            return 0.0
        x = np.linspace(-1.0, 1.0, len(arr), dtype=float)
        if trend in ("ascending", "descending"):
            return float(_fit_monotone_quadratic(x, arr, trend)[0])
        coeffs = np.polyfit(x, arr, 2)
        return float(coeffs[0])

    def _quadratic_curve_score(self, values: np.ndarray, mode: str) -> float:
        """根据目标趋势解释二次拟合系数方向。"""
        coef = self._quadratic_curve_coefficient(values, trend=mode)
        if mode == "peak":
            return -coef
        return coef

    def _evaluate_split_scheme(
        self, feature: str, x: pd.Series, y: pd.Series, splits: np.ndarray, mode: str
    ) -> Tuple[bool, float, np.ndarray, np.ndarray]:
        """评估切分方案是否满足单调与样本约束，并给出 lift 导向评分。"""
        bins = self._get_feature_bins(feature, x, splits)
        bin_table = self._compute_bin_stats(feature, x, y, bins)
        valid = bin_table[bin_table["分箱"] >= 0].reset_index(drop=True)
        bad_rates = valid["坏样本率"].to_numpy(dtype=float)
        counts = valid["样本总数"].to_numpy(dtype=int)
        min_samples = self._get_min_samples(len(y))
        max_samples = self._get_max_samples(len(y))

        is_valid = len(valid) >= self.min_n_bins
        if len(bad_rates) > 1:
            is_valid = is_valid and self._count_monotonic_violations(bad_rates, mode) == 0
        if len(bad_rates) > 2:
            is_valid = is_valid and self._count_adjacent_zero_bad_rate_pairs(bad_rates) == 0
        is_valid = is_valid and np.all(counts >= min_samples)
        if max_samples is not None:
            is_valid = is_valid and np.all(counts <= max_samples)

        curve_values = valid["LIFT值"].to_numpy(dtype=float) if "LIFT值" in valid.columns else bad_rates
        curve_spread = float(np.max(curve_values) - np.min(curve_values)) if len(curve_values) > 0 else 0.0
        curve_step_sum = float(np.sum(np.abs(np.diff(curve_values)))) if len(curve_values) > 1 else 0.0
        quad_score = self._quadratic_curve_score(curve_values, mode)
        flat_penalty = float(np.sum(np.abs(np.diff(curve_values)) < 1e-8)) if len(curve_values) > 1 else 0.0
        iv_value = float(valid["分档IV值"].sum()) if "分档IV值" in valid.columns else 0.0
        score = (
            quad_score * 1000.0
            + curve_spread * 100.0
            + curve_step_sum * 10.0
            + len(valid) * 80.0
            - flat_penalty * 200.0
            + iv_value * 1e-3
        )
        return is_valid, score, bad_rates, counts

    def _expand_splits_with_monotonicity(
        self, feature: str, x: pd.Series, y: pd.Series, splits: Union[np.ndarray, list], mode: str
    ) -> np.ndarray:
        """在保持单调的前提下，尽量补足到允许的分箱预算。"""
        current = np.unique(np.sort(np.asarray(splits, dtype=float))) if len(splits) > 0 else np.array([])
        max_splits_allowed = max(0, self.max_n_bins - 1)
        if len(current) >= max_splits_allowed:
            return current

        x_numeric = pd.to_numeric(x, errors="coerce")
        valid_mask = x_numeric.notna()
        if self.special_codes:
            for code in self.special_codes:
                valid_mask &= x_numeric != code
        x_valid = x_numeric[valid_mask]
        y_valid = y[valid_mask]
        if len(x_valid) == 0:
            return current

        base_ok, base_score, _, _ = self._evaluate_split_scheme(feature, x, y, current, mode)
        if not base_ok:
            return current

        min_samples = self._get_min_samples(len(y_valid))
        quantiles = [0.25, 0.5, 0.75]

        for _ in range(max_splits_allowed - len(current)):
            bins = np.digitize(x_valid, current) if len(current) > 0 else np.zeros(len(x_valid), dtype=int)
            best_candidate = None
            best_score = None

            for bin_idx in range(len(current) + 1):
                bin_values = np.sort(x_valid[bins == bin_idx].to_numpy(dtype=float))
                if len(bin_values) < max(2 * min_samples, 8):
                    continue

                candidate_positions = set()
                for q in quantiles:
                    pos = int(round(q * (len(bin_values) - 1)))
                    pos = min(max(pos, min_samples - 1), len(bin_values) - min_samples - 1)
                    if 0 <= pos < len(bin_values) - 1:
                        candidate_positions.add(pos)

                for pos in sorted(candidate_positions):
                    left_value = bin_values[pos]
                    right_value = bin_values[pos + 1]
                    if np.isclose(left_value, right_value, atol=1e-12, rtol=0):
                        continue

                    candidate = float((left_value + right_value) / 2.0)
                    if len(current) > 0 and np.any(np.isclose(current, candidate, atol=1e-12, rtol=0)):
                        continue

                    trial = np.unique(np.sort(np.append(current, candidate)))
                    ok, score, _, _ = self._evaluate_split_scheme(feature, x, y, trial, mode)
                    if not ok:
                        continue
                    if best_score is None or score > best_score + 1e-9:
                        best_score = score
                        best_candidate = candidate

            if best_candidate is None:
                break

            current = np.unique(np.sort(np.append(current, best_candidate)))

        return current

    def _round_splits(self, splits: Union[np.ndarray, list]) -> np.ndarray:
        """对数值型切分点进行四舍五入.

        使用 round_float 函数保留指定位数的小数进行四舍五入。

        :param splits: 切分点数组
        :return: 四舍五入后的切分点数组
        """
        if splits is None or len(splits) == 0:
            return np.array([]) if not isinstance(splits, np.ndarray) else splits

        # 转换为 numpy 数组
        if not isinstance(splits, np.ndarray):
            splits = np.array(splits)

        # 对每个切分点进行四舍五入，使用 self.decimal 指定精度
        rounded_splits = np.array([round_float(s, decimal=self.decimal) for s in splits], dtype=float)
        return rounded_splits

    def _detect_feature_type(self, data: Union[pd.Series, np.ndarray]) -> str:
        """检测特征类型.

        参考 scorecardpipeline (scp) 的 Combiner 实现:
        根据 dtype 判断特征类型，优先使用 str(dtype) 判断

        :param data: 特征数据，支持 pd.Series 或 np.ndarray
        :return: 特征类型，'numerical' 或 'categorical'

        **判断逻辑**

        1. 将输入转换为 pd.Series（如果是 np.ndarray）
        2. 如果 str(dtype) 是 "object", "string", "category"，认为是类别型
        3. 如果 dtype 是数值型，先判断是否为布尔型（bool 视为类别型）
        4. 数值型特征只有在明确设置 cat_cutoff 且满足条件时，才视为类别型
        5. 否则认为是数值型
        """
        # 有序编码是类别变量的内部数值表示，必须强制走具体方法的数值算法。
        if isinstance(data, pd.Series) and data.name in self._categorical_encoded_features_:
            return "numerical"

        # 统一转换为 pd.Series 处理
        if isinstance(data, np.ndarray):
            series = pd.Series(data)
        elif isinstance(data, pd.Series):
            series = data
        else:
            # 其他类型（如 list），转换为 pd.Series
            series = pd.Series(data)

        # 缺失值不计入
        series_valid = series.dropna()

        # 获取 dtype 字符串表示
        dtype_str = str(series.dtype)

        # 参考 scp: 如果 dtype 是 object/string/category，直接认为是类别型
        if dtype_str in ["object", "string", "category"]:
            return "categorical"

        # 布尔型视为类别型
        if pd.api.types.is_bool_dtype(series.dtype):
            return "categorical"

        # 如果是数值型
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series_valid.nunique()
            n_valid = len(series_valid)
            unique_ratio = n_unique / n_valid if n_valid > 0 else 0

            # 只有在明确设置 cat_cutoff 时，才根据唯一值判断类型
            if self.cat_cutoff is not None:
                if self.cat_cutoff < 1:
                    # 如果唯一值比例小于等于阈值，认为是类别型
                    if unique_ratio <= self.cat_cutoff:
                        return "categorical"
                else:
                    # 如果唯一值数量小于等于阈值，认为是类别型
                    if n_unique <= int(self.cat_cutoff):
                        return "categorical"

            # 默认：数值型保持为数值型
            return "numerical"

        # 默认认为是类别型
        return "categorical"

    def _compute_bin_stats(self, feature: str, X: pd.Series, y: pd.Series, bins: np.ndarray) -> pd.DataFrame:
        """计算分箱统计信息（使用metrics模块优化版本，中文列名）.

        :param feature: 特征名
        :param X: 特征数据
        :param y: 目标变量
        :param bins: 分箱索引
        :return: 分箱统计表，包含中文列名
        """
        # 生成分箱标签
        bin_labels = None
        if feature in self.splits_:
            splits = self.splits_[feature]
            feature_type = self.feature_types_.get(feature, "numerical")

            if (
                feature_type == "categorical"
                and isinstance(splits, list)
                and len(splits) > 0
                and isinstance(splits[0], list)
            ):
                unique_bins = np.unique(bins)
                bin_labels = []
                for bin_idx in unique_bins:
                    if bin_idx == -1:
                        bin_labels.append("missing")
                    elif bin_idx == -2:
                        bin_labels.append("special")
                    elif bin_idx == -3:
                        bin_labels.append("unknown")
                    elif 0 <= bin_idx < len(splits):
                        group = splits[bin_idx]
                        if isinstance(group, list):
                            # 将np.nan转换为字符串"nan"
                            group_str = [str(v) if not (isinstance(v, float) and np.isnan(v)) else "nan" for v in group]
                            bin_labels.append(",".join(group_str))
                        else:
                            bin_labels.append(str(group))
                    else:
                        bin_labels.append(f"bin_{bin_idx}")
            else:
                bin_labels = self._get_bin_labels(splits, bins)

        # 使用 metrics 模块的向量化计算，传入分箱标签和WOE截断参数
        bin_stats = compute_bin_stats(bins, y.values, bin_labels=bin_labels, woe_clip=self.woe_clip)

        # 如果没有分箱标签，生成默认标签
        if "分箱标签" not in bin_stats.columns:
            bin_stats["分箱标签"] = bin_stats["分箱"].apply(
                lambda x: f"bin_{x}" if x >= 0 else ("缺失" if x == -1 else ("special" if x == -2 else "unknown"))
            )

        # 调整列顺序（将分箱标签放在分箱后面）
        chinese_columns = [
            "分箱",
            "分箱标签",
            "样本总数",
            "好样本数",
            "坏样本数",
            "样本占比",
            "好样本占比",
            "坏样本占比",
            "坏样本率",
            "分档WOE值",
            "分档IV值",
            "指标IV值",
            "LIFT值",
            "坏账改善",
            "累积LIFT值",
            "累积坏账改善",
            "累积好样本数",
            "累积坏样本数",
            "分档KS值",
        ]

        # 确保所有列都存在
        available_cols = [c for c in chinese_columns if c in bin_stats.columns]
        bin_stats = bin_stats[available_cols]

        return bin_stats

    def _detect_decimal_precision(self, values: np.ndarray) -> int:
        """检测数值的小数精度，智能决定保留位数.

        :param values: 数值数组
        :return: 推荐的小数位数
        """
        if values is None or len(values) == 0:
            return 2

        # 过滤掉inf/nan
        valid_values = values[np.isfinite(values)]
        if len(valid_values) == 0:
            return 2

        # 检查是否所有值都是整数
        all_integer = np.all(np.equal(np.mod(valid_values, 1), 0))
        if all_integer:
            return 0

        # 计算最大小数位数
        max_decimals = 0
        for val in valid_values:
            # 转换为字符串计算小数位数
            s = f"{val:.10f}".rstrip("0").rstrip(".") if "." in f"{val:.10f}" else f"{val:.10f}"
            if "." in s:
                decimals = len(s.split(".")[1])
                max_decimals = max(max_decimals, decimals)

        # 智能截断：如果小数位超过4位，统一保留4位；否则保留实际位数
        return min(max_decimals, 4)

    def _get_bin_labels(self, splits: np.ndarray, bins: Optional[np.ndarray] = None) -> List[str]:
        """根据切分点生成分箱标签（左闭右开 [a, b) 风格，与 toad/scorecardpipeline 一致）.

        :param splits: 切分点数组
        :param bins: 分箱索引，用于处理缺失值和特殊值
        :return: 分箱标签列表
        """
        labels = []
        n_splits = len(splits) if splits is not None else 0

        # 检测切分点的小数精度
        decimal_precision = self._detect_decimal_precision(splits) if splits is not None else 2
        format_str = f".{decimal_precision}f"

        # 如果有分箱索引，需要处理缺失值等特殊箱
        if bins is not None:
            unique_bins = np.unique(bins)
            n_bins = n_splits + 1

            for i in unique_bins:
                if i == -1:  # 缺失值箱
                    labels.append("missing")
                elif i == -2:  # 特殊值箱
                    labels.append("special")
                elif n_splits == 0:
                    # 没有切分点时，所有正常值在一个箱
                    labels.append("[-inf, +inf)")
                elif i < n_bins:
                    if i == 0:
                        val = splits[i]
                        labels.append(f"[-inf, {val:{format_str}})")
                    elif i == n_bins - 1:
                        val = splits[i - 1]
                        labels.append(f"[{val:{format_str}}, +inf)")
                    else:
                        val1 = splits[i - 1]
                        val2 = splits[i]
                        labels.append(f"[{val1:{format_str}}, {val2:{format_str}})")
                else:
                    labels.append(f"bin_{i}")
        else:
            # 只根据切分点生成标签
            if n_splits == 0:
                # 没有切分点时，只有一个箱
                labels.append("[-inf, +inf)")
            else:
                for i in range(n_splits + 1):
                    if i == 0:
                        val = splits[i]
                        labels.append(f"[-inf, {val:{format_str}})")
                    elif i == n_splits:
                        val = splits[i - 1]
                        labels.append(f"[{val:{format_str}}, +inf)")
                    else:
                        val1 = splits[i - 1]
                        val2 = splits[i]
                        labels.append(f"[{val1:{format_str}}, {val2:{format_str}})")

        return labels

    def _assign_bin_labels(self, feature: str, bins: np.ndarray) -> List[str]:
        """将分箱索引数组映射为每个样本的分箱标签（逐样本，长度与 bins 一致）.

        以 ``bin_tables_`` 的 '分箱' → '分箱标签' 为唯一映射来源，保证 ``transform(metric='bins')``
        的标签与 ``get_bin_table`` 完全一致（左闭右开 [a, b) 风格），并正确处理缺失值(-1)/
        特殊值(-2)箱。当某分箱索引未出现在分箱表中时（如测试集落入训练集空箱），
        数值型回退到由切分点生成的标签，否则回退为 ``bin_{i}``。

        :param feature: 特征名
        :param bins: 分箱索引数组，shape (n_samples,)
        :return: 逐样本分箱标签列表，长度为 n_samples
        """
        bins = np.asarray(bins)
        label_map: Dict[int, str] = {}
        bin_table = self.bin_tables_.get(feature)
        if bin_table is not None and "分箱" in bin_table.columns and "分箱标签" in bin_table.columns:
            label_map = dict(zip(bin_table["分箱"].astype(int), bin_table["分箱标签"].astype(str)))

        fallback_list = None
        if self.feature_types_.get(feature, "numerical") == "numerical":
            splits = self.splits_.get(feature)
            if splits is not None:
                fallback_list = self._get_bin_labels(np.asarray(splits, dtype=float))

        def _label(b: int) -> str:
            b = int(b)
            if b in label_map:
                return label_map[b]
            if b == -1:
                return "missing"
            if b == -2:
                return "special"
            if b == -3:
                return "unknown"
            if self.feature_types_.get(feature) == "categorical" and 0 <= b < len(self._cat_bins_.get(feature, [])):
                return ", ".join(str(value) for value in self._cat_bins_[feature][b])
            if fallback_list is not None and 0 <= b < len(fallback_list):
                return fallback_list[b]
            return f"bin_{b}"

        return [_label(b) for b in bins]

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """获取指定特征的分箱表.

        :param feature: 特征名
        :return: 分箱统计表（返回副本，修改不会影响分箱器内部数据）
        :raises NotFittedError: 如果分箱器尚未拟合
        :raises FeatureNotFoundError: 如果特征不存在
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.bin_tables_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self.bin_tables_[feature].copy()

    def _splits_with_nan(self, feature: str) -> Union[np.ndarray, list]:
        """返回可直接作为 ``user_splits`` 使用的当前规则.

        数值型规则中的 np.nan 位置表示缺失值归属的普通箱；独立 -1 缺失箱
        由 ``missing_separate`` 表达，因此不会向规则中追加歧义标记。
        """
        if self.feature_types_.get(feature) == "categorical":
            if feature in self._cat_bins_:
                return self._cat_bins_[feature]
            return self.splits_.get(feature, [])

        splits = self.splits_.get(feature, np.array([]))
        arr = splits.tolist() if isinstance(splits, np.ndarray) else list(splits)
        target = self._user_missing_bin_targets_.get(feature)
        if target is None:
            target = self._missing_bin_targets_.get(feature)
        if target is not None:
            if not 0 <= int(target) <= len(arr):
                raise ValueError(f"特征 '{feature}' 的缺失值目标箱 {target} 超出当前普通箱范围")
            arr.insert(int(target), np.nan)
        return arr

    def __getitem__(self, feature: str):
        """通过 `binner['feature']` 获取分箱规则（toad/scorecardpipeline风格）.

        数值型特征返回切分点列表，np.nan 的位置表示缺失值归属的普通箱。
        类别型特征返回 List[List] 分组列表。
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.splits_ and feature not in self._cat_bins_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self._splits_with_nan(feature)

    def get_splits(self, feature: str) -> Union[np.ndarray, list]:
        """获取指定特征的切分点（scorecardpipeline 格式）.

        数值型特征：np.nan 的位置表示缺失值归属的普通箱。
        类别型特征：返回 List[List] 分组列表。

        :param feature: 特征名
        :return: 切分点列表
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        if feature not in self.splits_:
            raise FeatureNotFoundError(f"特征 '{feature}' 未找到")

        return self._splits_with_nan(feature)

    def export_rules(self) -> Dict[str, Union[List, List[List]]]:
        """导出分箱规则.

        数值型变量返回切分点列表，类别型变量返回分组列表。

        :return: 分箱规则字典
            - 数值型: key为特征名，value为切分点列表，如 [25, 35, 45, 55]
            - 类别型: key为特征名，value为分组列表，如 [['A', 'B'], ['C'], [np.nan]]

        **参考样例**

        >>> binner = OptimalBinning()
        >>> binner.fit(X, y)
        >>> rules = binner.export_rules()
        >>>
        >>> # 数值型变量
        >>> print(rules['age'])  # [25, 35, 45, 55]
        >>>
        >>> # 类别型变量
        >>> print(rules['city'])  # [['北京', '上海'], ['广州', '深圳'], [np.nan]]
        >>>
        >>> # 保存规则
        >>> import json
        >>> import numpy as np
        >>>
        >>> # 处理np.nan以便JSON序列化
        >>> def convert_nan(obj):
        ...     if isinstance(obj, dict):
        ...         return {k: convert_nan(v) for k, v in obj.items()}
        ...     elif isinstance(obj, list):
        ...         return [convert_nan(item) for item in obj]
        ...     elif isinstance(obj, float) and np.isnan(obj):
        ...         return "NaN"
        ...     return obj
        >>>
        >>> with open('binning_rules.json', 'w') as f:
        ...     json.dump(convert_nan(rules), f, indent=2)
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

        rules = {}
        for feature in self.splits_:
            if self.feature_types_.get(feature) == "categorical":
                # 类别型变量：返回分组列表
                if feature in self._cat_bins_:
                    # 将numpy数组转换为列表
                    bins = self._cat_bins_[feature]
                    rules[feature] = [
                        [item if not (isinstance(item, float) and np.isnan(item)) else np.nan for item in group]
                        if isinstance(group, (list, np.ndarray))
                        else group
                        for group in bins
                    ]
                else:
                    # 如果没有分组信息，返回空列表
                    rules[feature] = []
            else:
                rules[feature] = self._splits_with_nan(feature)

        return rules

    def import_rules(self, rules: Dict[str, Union[List, List[List]]]):
        """导入分箱规则.

        支持数值型切分点和类别型分组列表。

        :param rules: 分箱规则字典
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}

        调用本方法时沿用增量覆盖语义，导入后可立即 transform。随后调用 fit 时，
        仅把最近一次导入规则中同时属于本轮 X 的特征作为候选输入；普通分箱器仍
        重新运行其算法，OptimalBinning 则保留这些导入切点并补充分箱统计。

        **参考样例**

        >>> # 导入数值型规则
        >>> rules = {'age': [25, 35, 45, 55]}
        >>> binner.import_rules(rules)
        >>>
        >>> # 导入类别型规则
        >>> rules = {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}
        >>> binner.import_rules(rules)
        >>>
        >>> # 从JSON文件导入
        >>> import json
        >>> import numpy as np
        >>>
        >>> def convert_nan_back(obj):
        ...     if isinstance(obj, dict):
        ...         return {k: convert_nan_back(v) for k, v in obj.items()}
        ...     elif isinstance(obj, list):
        ...         return [convert_nan_back(item) for item in obj]
        ...     elif obj == "NaN":
        ...         return np.nan
        ...     return obj
        >>>
        >>> with open('binning_rules.json', 'r') as f:
        ...     rules = json.load(f)
        >>>     rules = convert_nan_back(rules)
        >>> binner.import_rules(rules)
        """
        self._woe_maps_ = getattr(self, "_woe_maps_", {})

        for feature, rule in rules.items():
            # 当前规则没有携带 WOE；同名旧映射属于上一版切点，必须在抓取规则快照前失效。
            # 其他特征仍保留 import_rules 的即时增量语义。load() 若显式携带新 WOE，
            # 会在导入切点后写回新映射并重新抓取快照。
            self._woe_maps_.pop(feature, None)
            # 判断是否为类别型变量
            if isinstance(rule, list) and len(rule) > 0 and isinstance(rule[0], list):
                # 类别型变量：List[List]格式
                normalized = normalize_user_groups(
                    feature,
                    rule,
                    observed=None,
                    special_codes=self.special_codes,
                    missing_separate=self.missing_separate,
                )
                self._cat_bins_[feature] = normalized
                self.feature_types_[feature] = "categorical"
                self.splits_[feature] = normalized  # 保持List[List]格式
                self.n_bins_[feature] = len(normalized)
                order = [value for group in normalized for value in group if not is_missing_marker(value)]
                self._category_orders_[feature] = order
                self._category_code_maps_[feature] = [(value, index) for index, value in enumerate(order)]
                self._woe_maps_[feature] = {
                    **{index: 0.0 for index in range(len(normalized))},
                    -1: 0.0,
                    -2: 0.0,
                    -3: 0.0,
                }
            else:
                # 数值型变量：NaN/None 所在位置显式指定缺失值归属的普通箱。
                clean, missing_target = parse_numerical_user_splits(feature, rule)
                self.splits_[feature] = self._round_splits(clean)
                self.feature_types_[feature] = "numerical"
                self.n_bins_[feature] = len(self.splits_[feature]) + 1
                if missing_target is None:
                    self._user_missing_bin_targets_.pop(feature, None)
                else:
                    self._user_missing_bin_targets_[feature] = int(missing_target)
                self._missing_bin_targets_.pop(feature, None)
                self._recorded_bins_[feature] = set(range(self.n_bins_[feature])) | {UNKNOWN_BIN}

        self._rules_imported_ = True
        self._capture_imported_rule_snapshot(rules.keys())
        self._is_fitted = True
        return self

    def update(
        self,
        splits_dict: Dict[str, Union[List, List[List]]],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "BaseBinning":
        """手动更新特征的切分点并重新计算相关属性.

        参考 toad.Combiner.update 方法，允许在分箱器训练完成后手工修改切分点。
        更新后会自动重新计算 n_bins_、feature_types_、_cat_bins_ 等属性。
        如果提供 X 和 y，还会重新计算 bin_tables_ 分箱统计表。

        **参数**

        :param splits_dict: 新的切分点字典，格式与 export_rules() 返回的相同
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}
        :param X: 可选，训练数据。如果提供，会重新计算分箱统计表
        :param y: 可选，目标变量

        **返回**

        :return: self，支持链式调用

        **参考样例**

        >>> # 只更新切分点（不重新计算统计表）
        >>> binner.update({'age': [20, 30, 40, 50]})

        >>> # 更新切分点并重新计算统计表
        >>> binner.update({'age': [20, 30, 40, 50]}, X=X_train, y=y_train)

        >>> # 批量更新多个特征
        >>> binner.update({
        ...     'age': [20, 30, 40],
        ...     'income': [5000, 10000, 20000],
        ...     'city': [['北京', '上海'], ['广州', '深圳']]
        ... })

        >>> # 链式调用
        >>> binner.update({'age': [20, 30, 40]}).transform(X_test)
        """
        # 检查分箱器是否已拟合
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")

        # 处理数据（如果提供）
        if X is not None:
            X, y = self._check_input(X, y)

        # 更新每个特征的切分点
        for feature, new_splits in splits_dict.items():
            # 判断特征类型并更新
            if isinstance(new_splits, list) and len(new_splits) > 0 and isinstance(new_splits[0], list):
                # 类别型变量：List[List] 格式
                observed = X[feature] if X is not None and feature in X.columns else None
                normalized = normalize_user_groups(
                    feature,
                    new_splits,
                    observed=observed,
                    special_codes=self.special_codes,
                    missing_separate=self.missing_separate,
                )
                self._cat_bins_[feature] = normalized
                self.feature_types_[feature] = "categorical"
                self.splits_[feature] = normalized
                self.n_bins_[feature] = len(normalized)
                order = [value for group in normalized for value in group if not is_missing_marker(value)]
                self._category_orders_[feature] = order
                self._category_code_maps_[feature] = [(value, index) for index, value in enumerate(order)]
                self._user_missing_bin_targets_.pop(feature, None)
                self._missing_bin_targets_.pop(feature, None)
            else:
                # 数值型变量：切分点列表
                clean, missing_target = parse_numerical_user_splits(feature, new_splits)
                self.splits_[feature] = self._round_splits(clean)
                self.feature_types_[feature] = "numerical"
                self.n_bins_[feature] = len(self.splits_[feature]) + 1
                if missing_target is None:
                    self._user_missing_bin_targets_.pop(feature, None)
                else:
                    self._user_missing_bin_targets_[feature] = int(missing_target)
                self._missing_bin_targets_.pop(feature, None)
                # 清除可能存在的旧类别型数据
                if feature in self._cat_bins_:
                    del self._cat_bins_[feature]

            # 如果提供了 X 和 y，重新计算该特征的分箱统计表
            if X is not None and y is not None and feature in X.columns:
                feature_type = self.feature_types_[feature]

                bins = self._assign_base_feature_bins(feature, X[feature])
                bins = self._apply_reserved_bin_policy(feature, X[feature], bins)
                bins = self._learn_missing_bin_target(feature, X[feature], y, bins)
                table = self._compute_bin_stats(feature, X[feature], y, bins)
                self.bin_tables_[feature] = self._materialize_unknown_bin(feature, table)
                self._woe_maps_[feature] = {
                    int(row["分箱"]): float(row["分档WOE值"]) for _, row in self.bin_tables_[feature].iterrows()
                }
                if self.handle_unknown == UNKNOWN_BIN:
                    self._woe_maps_[feature].setdefault(UNKNOWN_BIN, 0.0)
                self._reserved_bins_finalized_.add(feature)
                if feature_type == "categorical":
                    self._validate_categorical_constraints(feature, y)

        return self

    def plot(self, feature: str, save: Optional[str] = None, **kwargs):
        """绘制分箱图.

        :param feature: 特征名
        :param save: 图片保存路径，默认为None
        :param kwargs: 其他绘图参数

        **注意**

        绘制内容包括:
        1. 各分箱的样本数分布
        2. 各分箱的坏样本率
        3. 各分箱的WOE值

        :return: matplotlib Figure 对象
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用 fit 方法")
        if feature not in self.bin_tables_:
            raise FeatureNotFoundError(f"特征 '{feature}' 不存在，可选特征: {list(self.bin_tables_)}")

        from ..viz import bin_plot

        kwargs.setdefault("title", f"{feature}分箱图")
        return bin_plot(self.bin_tables_[feature], save=save, **kwargs)

    def export(self, to_json: Optional[str] = None) -> Dict[str, Union[List, List[List]]]:
        """导出 hscredit 严格分箱规则.

        数值型变量返回切分点列表，类别型变量返回分组列表。
        同时导出WOE映射信息，支持加载后直接进行WOE转换。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :return: 分箱规则字典
            - 数值型: {'age': [25, 35, 45, 55]}
            - 类别型: {'city': [['北京', '上海'], ['广州', '深圳'], [np.nan]]}
            - WOE映射: {'_woe_maps_': {'age': {0: 0.5, 1: -0.3, ...}}}

        **参考样例**

        >>> binner = OptimalBinning()
        >>> binner.fit(X, y)
        >>>
        >>> # 导出为字典
        >>> rules = binner.export()
        >>>
        >>> # 导出并保存到 JSON 文件
        >>> rules = binner.export(to_json='binning_rules.json')

        **WOE转换支持**

        导出的规则包含WOE映射信息，加载后可直接进行WOE转换:

        >>> binner = OptimalBinning()
        >>> binner.load('binning_rules.json')
        >>> X_woe = binner.transform(X_test, metric='woe')  # 直接使用，无需重新fit
        """
        import json

        rules = self.export_rules()

        # 导出WOE映射信息，支持加载后直接进行WOE转换
        woe_maps = {}
        for feature in self.splits_:
            if feature in self.bin_tables_:
                bin_table = self.bin_tables_[feature]
                woe_map = {}
                for _, row in bin_table.iterrows():
                    woe_map[int(row["分箱"])] = float(row["分档WOE值"])
                # 添加缺失值和特殊值的WOE
                self._enrich_woe_map(woe_map, bin_table)
                woe_maps[feature] = woe_map

        if woe_maps:
            rules["_woe_maps_"] = woe_maps

        # 处理 numpy 类型和 np.nan，使其可 JSON 序列化
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_for_json(obj.tolist())
            elif isinstance(obj, float) and np.isnan(obj):
                return None  # JSON 中 null
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        rules_json = convert_for_json(rules)

        if to_json is not None:
            # 确保目录存在
            import os

            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(to_json, "w", encoding="utf-8") as f:
                json.dump(rules_json, f, ensure_ascii=False, indent=2)

        return rules

    def load(self, from_json: Union[str, Dict], update: bool = False) -> "BaseBinning":
        """加载 hscredit 严格分箱规则.

        从字典或 JSON 文件加载数值规则或严格 ``List[List]`` 类别规则。
        同时加载WOE映射信息，支持加载后直接进行WOE转换。

        :param from_json: 分箱规则字典或 JSON 文件路径
            - 字典: {'age': [25, 35, 45, 55]}
            - 文件路径: 'binning_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :return: self，支持链式调用

        **参考样例**

        >>> binner = OptimalBinning()
        >>>
        >>> # 从字典加载
        >>> rules = {'age': [25, 35, 45, 55], 'gender': [['M'], ['F']]}
        >>> binner.load(rules)
        >>>
        >>> # 从 JSON 文件加载
        >>> binner.load('binning_rules.json')
        >>>
        >>> # 更新现有规则
        >>> binner.load({'new_feature': [1, 2, 3]}, update=True)

        **WOE转换支持**

        加载包含WOE映射信息的规则后，可直接进行WOE转换:

        >>> binner.load('binning_rules.json')
        >>> X_woe = binner.transform(X_test, metric='woe')  # 直接使用，无需重新fit
        """
        import json

        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, "r", encoding="utf-8") as f:
                rules = json.load(f)
        else:
            # 直接使用字典
            rules = deepcopy(from_json)

        # 提取WOE映射信息（如果存在）
        woe_maps = rules.pop("_woe_maps_", None)

        # 处理 JSON 中的 null 转换为 np.nan
        def convert_from_json(obj):
            if isinstance(obj, dict):
                return {k: convert_from_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_from_json(item) for item in obj]
            elif obj is None:
                return np.nan
            return obj

        rules = convert_from_json(rules)

        if not update:
            for state_name in self._FEATURE_DICT_STATE:
                setattr(self, state_name, {})
            for state_name in self._FEATURE_SET_STATE:
                setattr(self, state_name, set())
        self.import_rules(rules)

        # 恢复WOE映射信息，支持直接WOE转换
        if woe_maps is not None:
            self._woe_maps_ = {}
            for feature, woe_map in woe_maps.items():
                # 将字符串键转换为整数键
                self._woe_maps_[feature] = {int(k): float(v) for k, v in woe_map.items()}
            if getattr(self, "_rules_imported_", False):
                self._capture_imported_rule_snapshot(self._imported_rule_features_)

        return self

    def __repr__(self):
        if self._is_fitted:
            n_features = len(self.splits_)
            return f"{self.__class__.__name__}(fitted=True, n_features={n_features})"
        else:
            return f"{self.__class__.__name__}(fitted=False)"
