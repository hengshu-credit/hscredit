"""LIFT筛选器.

使用LIFT@ratio值进行特征筛选，支持自定义覆盖率和方向。

**参考样例**

>>> from hscredit.core.selectors import LiftSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = LiftSelector(threshold=0.5, ratio=0.10)  # 筛选LIFT>0.5且覆盖率10%的特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Literal, Tuple, Dict, Any
import warnings
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions, record_counts, validate_binary_target, validate_real
from ...utils.parallel import ParallelWorkload


def _compute_lift_single(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.10,
    ascending: bool = False,
    tie_policy: str = "fractional",
    missing_policy: str = "exclude",
) -> float:
    """计算单个特征在指定排序方向下的LIFT@ratio值.

    将样本按特征值排序后，取头部 ratio 比例的样本，
    计算该子群的坏样本率与整体坏样本率的比值。

    :param x: 特征值数组
    :param y: 目标变量数组
    :param ratio: 覆盖率，默认0.10（LIFT@10%）
    :param ascending: 排序方向，默认False（降序，取最大值头部）
    :return: LIFT值
    """
    if tie_policy != "legacy":
        bad, good, _, _ = _compute_lift_pair(x, y, ratio, tie_policy, missing_policy)
        return good if ascending else bad
    x, y = np.asarray(x), np.asarray(y)
    if missing_policy == "exclude":
        valid = ~pd.isna(x)
        x, y = x[valid], y[valid]
    n = len(x)
    if n == 0:
        return 1.0

    # 特征无变异，无区分能力
    if len(np.unique(x)) <= 1:
        return 1.0

    base_bad_rate = np.mean(y)
    if base_bad_rate == 0 or base_bad_rate == 1:
        return 1.0

    # 头部样本数量（至少1个）
    k = max(1, int(np.ceil(n * ratio)))

    # 按特征值排序
    if ascending:
        order = np.argsort(x, kind="stable")  # 升序：最小值在前
    else:
        order = np.argsort(x, kind="stable")[::-1]  # 降序：最大值在前

    # 取头部 k 个样本
    top_idx = order[:k]
    top_bad_rate = np.mean(y[top_idx])

    lift = top_bad_rate / base_bad_rate
    return float(lift)


def _compute_lift_pair(x, y, ratio, tie_policy="fractional", missing_policy="exclude"):
    """一次排序计算双向 LIFT；边界同值按组坏率比例分摊，不依赖行顺序。"""
    x, y = np.asarray(x), np.asarray(y)
    if missing_policy == "exclude":
        valid = ~pd.isna(x)
        x, y = x[valid], y[valid]
    n = len(x)
    k = max(1, int(np.ceil(n * ratio))) if n else 0
    if tie_policy == "legacy":
        bad = _compute_lift_single(x, y, ratio, False, "legacy", "legacy")
        good = _compute_lift_single(x, y, ratio, True, "legacy", "legacy")
        return bad, good, n, k
    if missing_policy == "legacy" and pd.isna(x).any():
        # 旧缺失排序不定义可比较的同值边界，要求显式同时选择旧并列策略。
        raise ValueError("missing_policy='legacy' 且存在缺失时，必须同时指定 tie_policy='legacy'")
    if n == 0 or pd.Series(x).nunique(dropna=False) <= 1:
        return 1.0, 1.0, n, k
    base_bad_rate = float(np.mean(y))
    if base_bad_rate in (0.0, 1.0):
        return 1.0, 1.0, n, k
    try:
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        cumulative_bad = np.r_[0.0, np.cumsum(y[order], dtype=float)]
        lower = ordered[k - 1]
        lower_left = int(np.searchsorted(ordered, lower, side="left"))
        lower_right = int(np.searchsorted(ordered, lower, side="right"))
        lower_bad = cumulative_bad[lower_left] + (cumulative_bad[lower_right] - cumulative_bad[lower_left]) * (
            k - lower_left
        ) / (lower_right - lower_left)
        upper = ordered[n - k]
        upper_left = int(np.searchsorted(ordered, upper, side="left"))
        upper_right = int(np.searchsorted(ordered, upper, side="right"))
        upper_bad = (
            cumulative_bad[n]
            - cumulative_bad[upper_right]
            + (cumulative_bad[upper_right] - cumulative_bad[upper_left])
            * (k - (n - upper_right))
            / (upper_right - upper_left)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("LIFT 特征必须具有一致、可比较的取值，请先编码混合类别") from exc
    return float(upper_bad / k / base_bad_rate), float(lower_bad / k / base_bad_rate), n, k


def _compute_lift_with_direction(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.10,
    direction: str = "auto",
    tie_policy: str = "fractional",
    missing_policy: str = "exclude",
) -> Tuple[float, float, float, str]:
    """计算单个特征的LIFT得分（支持方向判断）.

    :param x: 特征值数组
    :param y: 目标变量数组
    :param ratio: 覆盖率
    :param direction: 方向模式
        - 'auto': 同时计算两个方向，只比较各自目标方向上的改善
        - 'bad': 仅计算找坏人的LIFT（降序取头部，LIFT越高越好）
        - 'good': 仅计算找好人的LIFT（升序取头部，LIFT越低越好）
    :return: (score, lift_bad, lift_good, best_direction)
        - score: 指定方向相对 LIFT=1 的有效改善，越大区分力越强
        - lift_bad: 降序LIFT值（找坏人方向）
        - lift_good: 升序LIFT值（找好人方向）
        - best_direction: 最优方向 'bad' 或 'good'
    """
    lift_bad, lift_good, _, _ = _compute_lift_pair(x, y, ratio, tie_policy, missing_policy)
    if direction == "bad":
        score = max(lift_bad - 1.0, 0.0)
        return score, lift_bad, np.nan, "bad"

    if direction == "good":
        score = max(1.0 - lift_good, 0.0)
        return score, np.nan, lift_good, "good"

    # auto: 同时计算两个方向，只奖励方向正确的改善。

    dist_bad = max(lift_bad - 1.0, 0.0)
    dist_good = max(1.0 - lift_good, 0.0)

    if dist_bad >= dist_good:
        return dist_bad, lift_bad, lift_good, "bad"
    else:
        return dist_good, lift_bad, lift_good, "good"


def _compute_lift_feature(task):
    """计算单个特征的 LIFT 详情。"""
    feature, values, y, ratio, direction, *policies = task
    return (feature,) + _compute_lift_with_direction(values, y, ratio, direction, *policies)


class LiftSelector(BaseFeatureSelector):
    """LIFT筛选器.

    使用LIFT@ratio值筛选特征，支持找坏人、找好人、自动三种方向模式。
    LIFT衡量特征在头部覆盖率下对目标群体的提升程度。

    **LIFT@ratio% 计算方式**

    1. 将样本按特征值排序
    2. 取头部 ratio 比例的样本
    3. LIFT = 该子群坏样本率 / 整体坏样本率

    **方向模式**

    | direction | 含义 | 评分方式 |
    |-----------|------|----------|
    | auto | 自动选择最优方向（默认） | score = max(LIFT_bad-1, 1-LIFT_good, 0) |
    | bad | 仅评估找坏人能力 | score = max(LIFT_bad - 1, 0) |
    | good | 仅评估找好人能力 | score = max(1 - LIFT_good, 0) |

    **评分含义**

    score 只度量目标方向相对基准 LIFT=1 的改善，反向偏离按 0 计:

    - score = 0: 无区分能力（LIFT = 1）
    - score = 4.0: 强找坏人能力（LIFT_bad=5.0）；找好人得分上限为 1（LIFT_good=0）
    - 内部经验: score >= 0.5 通常认为有一定区分力

    **参数**

    :param threshold: 目标方向的改善得分阈值，默认0.5
        - 仅保留 score >= threshold 的特征
        - threshold=0.5 等价于旧版 LIFT >= 1.5（找坏人方向）
        - 内部经验: 风控场景常用 0.5~1.0
    :param ratio: LIFT计算的覆盖率，默认0.10（即LIFT@10%）
        - 内部经验: 风控场景常用 lift@5% 或 lift@10%
    :param direction: 方向模式，默认'auto'
        - 'auto': 同时计算两个方向，取最优（推荐）
        - 'bad': 仅评估找坏人能力（降序取头部，LIFT > 1）
        - 'good': 仅评估找好人能力（升序取头部，LIFT < 1）
    :param target: 目标变量列名，默认为'target'
    :param include: 强制保留的特征列表
    :param exclude: 强制剔除的特征列表
    :param n_jobs: 并行计算的任务数
    :param tie_policy: 默认 'fractional'，头部边界同值样本按组坏率比例分摊，保持
        ceil(有效样本数 × ratio) 的加权头部数量；'legacy' 恢复依赖样本顺序的截断。
    :param missing_policy: 默认 'exclude'，每列独立排除缺失并在有效样本中计算基准坏率；
        'legacy' 恢复旧缺失排序，存在缺失时需同时指定 tie_policy='legacy'。

    **属性**

    - scores\_: 各特征在目标方向上的改善得分，pd.Series
    - lift_detail\_: 各特征的LIFT详情表，pd.DataFrame
        包含列: 找坏人LIFT、找好人LIFT、最优方向、方向改善得分、有效样本数、头部样本量、实际覆盖率

    **参考样例**

    >>> from hscredit.core.selectors import LiftSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = pd.Series(np.random.randint(0, 2, 1000))
    >>>
    >>> # 自动模式（推荐）: 同时检测找坏人和找好人能力
    >>> selector = LiftSelector(threshold=0.5, ratio=0.10)
    >>> selector.fit(X, y)
    >>> print(selector.lift_detail_)  # 查看各特征两个方向的LIFT
    >>>
    >>> # 仅评估找坏人能力
    >>> selector = LiftSelector(direction='bad', threshold=0.5)
    >>> selector.fit(X, y)
    >>>
    >>> # 仅评估找好人能力
    >>> selector = LiftSelector(direction='good', threshold=0.5)
    >>> selector.fit(X, y)

    **引用**

    LIFT@k%（头部覆盖率下的提升度）是响应/风险模型的标准评估口径，参见 lift chart
    https://en.wikipedia.org/wiki/Lift_(data_mining) 及 Siddiqi, N. (2006).
    *Credit Risk Scorecards.* Wiley。
    """

    method_name = "LIFT筛选"

    def __init__(
        self,
        threshold: float = 0.5,
        ratio: float = 0.10,
        direction: Literal["auto", "bad", "good"] = "auto",
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
        tie_policy: str = "fractional",
        missing_policy: str = "exclude",
    ):
        """初始化筛选器；默认透传已有目标列，仅target_rm=True移除。"""
        super().__init__(
            target=target,
            threshold=threshold,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )
        self.ratio = ratio
        self.direction = direction
        self.tie_policy = tie_policy
        self.missing_policy = missing_policy

    def __setstate__(self, state):
        old_ties, old_missing = "tie_policy" not in state, "missing_policy" not in state
        super().__setstate__(state)
        if old_ties:
            self.tie_policy = "legacy"
        if old_missing:
            self.missing_policy = "legacy"
        if old_ties or old_missing:
            warnings.warn(
                "旧LIFT筛选器保留旧并列截断和缺失排序；新训练建议 tie_policy='fractional', missing_policy='exclude'",
                UserWarning,
                stacklevel=2,
            )

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "LIFT改善阈值", allow_infinite=True)
        validate_real(self.ratio, "ratio", minimum=0, maximum=1)
        if self.ratio == 0:
            raise ValueError("ratio 必须在 (0, 1] 范围内")
        if self.direction not in {"auto", "bad", "good"}:
            raise ValueError("direction 必须是 'auto'、'bad' 或 'good'")
        if self.tie_policy not in {"fractional", "legacy"}:
            raise ValueError("tie_policy 必须是 'fractional' 或 'legacy'")
        if self.missing_policy not in {"exclude", "legacy"}:
            raise ValueError("missing_policy 必须是 'exclude' 或 'legacy'")
        X, y = super()._check_input(X, y)
        return X, validate_binary_target(y, "LiftSelector")

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合LIFT筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "LIFT方向改善得分", "越大越好"

        if y is None:
            raise ValueError("LiftSelector 需要目标变量 y")
        if not 0 < float(self.ratio) <= 1:
            raise ValueError("ratio 必须在 (0, 1] 范围内")
        if self.direction not in {"auto", "bad", "good"}:
            raise ValueError("direction 必须是 'auto'、'bad' 或 'good'")

        y = np.asarray(y)

        results = self._parallel_execute(
            _compute_lift_feature,
            (
                (col, X[col].values, y, self.ratio, self.direction, self.tie_policy, self.missing_policy)
                for col in X.columns
            ),
            task_labels=X.columns,
            default_backend="threading",
            workload=ParallelWorkload(
                task_count=X.shape[1],
                rows=X.shape[0],
                columns=X.shape[1],
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=8.0,
                capability="thread_safe",
                releases_gil=True,
                operation="LIFT字段排序",
            ),
        )

        # 解包结果
        scores = np.array([r[1] for r in results])
        lift_bad = np.array([r[2] for r in results])
        lift_good = np.array([r[3] for r in results])
        best_dirs = [r[4] for r in results]

        # 评分只奖励目标方向上的改善
        self.scores_ = pd.Series(scores, index=X.columns)
        self.effective_counts_ = (
            self.valid_counts_.copy() if self.missing_policy == "exclude" else self.total_counts_.copy()
        )
        self.head_counts_ = np.ceil(self.effective_counts_ * self.ratio).astype(np.int64)
        self.actual_coverage_ = self.head_counts_ / self.effective_counts_.replace(0, np.nan)

        # LIFT详情表
        self.lift_detail_ = pd.DataFrame(
            {
                "找坏人LIFT": lift_bad,
                "找好人LIFT": lift_good,
                "最优方向": [{"bad": "找坏人", "good": "找好人"}[value] for value in best_dirs],
                "方向改善得分": scores,
                "有效样本数": self.effective_counts_,
                "头部样本量": self.head_counts_,
                "实际覆盖率": self.actual_coverage_,
            },
            index=X.columns,
        )

        # 选择 score >= threshold 的特征
        selected_mask = scores >= self.threshold
        record_conditions(self, X.columns, LIFT改善达标=selected_mask)
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 生成剔除原因
        dir_label = {"auto": "自动", "bad": "找坏人", "good": "找好人"}
        self._drop_reason = (
            f"LIFT@{self.ratio:.0%} 方向改善得分 < {self.threshold}"
            f"（方向: {dir_label.get(self.direction, self.direction)}）"
        )
