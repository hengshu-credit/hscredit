"""LIFT筛选器.

使用LIFT@ratio值进行特征筛选，支持自定义覆盖率和方向。

LIFT@ratio% 计算方式:
    1. 将样本按特征值排序
    2. 取头部 ratio 比例的样本
    3. LIFT = 该子群坏样本率 / 整体坏样本率

**方向说明**

| direction | 含义 | LIFT 表现 | 典型场景 |
|-----------|------|-----------|----------|
| bad | 找坏人能力 | LIFT >> 1，越高越强 | 尾部高风险客户识别 |
| good | 找好人能力 | LIFT << 1，越低越强 | 头部优质客户识别 |
| auto | 自动取最优方向 | 取 |LIFT-1| 最大的方向 | 通用特征筛选（默认） |

- 内部经验: 风控场景常用 lift@5% 或 lift@10%
"""

from typing import Union, List, Optional, Literal, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_lift_single(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.10,
    ascending: bool = False,
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
        order = np.argsort(x, kind='stable')        # 升序：最小值在前
    else:
        order = np.argsort(x, kind='stable')[::-1]  # 降序：最大值在前

    # 取头部 k 个样本
    top_idx = order[:k]
    top_bad_rate = np.mean(y[top_idx])

    lift = top_bad_rate / base_bad_rate
    return float(lift)


def _compute_lift_with_direction(
    x: np.ndarray,
    y: np.ndarray,
    ratio: float = 0.10,
    direction: str = 'auto',
) -> Tuple[float, float, float, str]:
    """计算单个特征的LIFT得分（支持方向判断）.

    :param x: 特征值数组
    :param y: 目标变量数组
    :param ratio: 覆盖率
    :param direction: 方向模式
        - 'auto': 同时计算找坏人和找好人的LIFT，取 |LIFT-1| 最大的方向
        - 'bad': 仅计算找坏人的LIFT（降序取头部，LIFT越高越好）
        - 'good': 仅计算找好人的LIFT（升序取头部，LIFT越低越好）
    :return: (score, lift_bad, lift_good, best_direction)
        - score: |LIFT - 1|，与1的距离，越大区分力越强
        - lift_bad: 降序LIFT值（找坏人方向）
        - lift_good: 升序LIFT值（找好人方向）
        - best_direction: 最优方向 'bad' 或 'good'
    """
    if direction == 'bad':
        lift_bad = _compute_lift_single(x, y, ratio, ascending=False)
        score = abs(lift_bad - 1.0)
        return score, lift_bad, np.nan, 'bad'

    if direction == 'good':
        lift_good = _compute_lift_single(x, y, ratio, ascending=True)
        score = abs(lift_good - 1.0)
        return score, np.nan, lift_good, 'good'

    # auto: 同时计算两个方向，取 |LIFT - 1| 更大的
    lift_bad = _compute_lift_single(x, y, ratio, ascending=False)
    lift_good = _compute_lift_single(x, y, ratio, ascending=True)

    dist_bad = abs(lift_bad - 1.0)
    dist_good = abs(lift_good - 1.0)

    if dist_bad >= dist_good:
        return dist_bad, lift_bad, lift_good, 'bad'
    else:
        return dist_good, lift_bad, lift_good, 'good'


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
    | auto | 自动选择最优方向（默认） | score = max(|LIFT_bad-1|, |LIFT_good-1|) |
    | bad | 仅评估找坏人能力 | score = |LIFT_bad - 1|，LIFT >> 1 越强 |
    | good | 仅评估找好人能力 | score = |LIFT_good - 1|，LIFT << 1 越强 |

    **评分含义**

    score = |LIFT - 1|，即与基准LIFT=1的距离:

    - score = 0: 无区分能力（LIFT = 1）
    - score = 4.0: 强区分力（如 LIFT=5.0 找坏人，或 LIFT=0.0 找好人）
    - 内部经验: score >= 0.5 通常认为有一定区分力

    **参数**

    :param threshold: 得分阈值（|LIFT-1|），默认0.5
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

    **属性**

    - scores\_: 各特征的区分力得分（|LIFT-1|），pd.Series
    - lift_detail\_: 各特征的LIFT详情表，pd.DataFrame
        包含列: LIFT_bad, LIFT_good, best_direction, score

    **示例**

    ::

        >>> from hscredit.core.selectors import LiftSelector
        >>> import pandas as pd
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
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ratio: float = 0.10,
        direction: Literal['auto', 'bad', 'good'] = 'auto',
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target, threshold=threshold, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
        )
        self.ratio = ratio
        self.direction = direction
        self.method_name = 'LIFT筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合LIFT筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        y = np.asarray(y)

        # 计算每个特征的LIFT得分（含方向判断）
        if self.n_jobs == 1:
            results = [
                _compute_lift_with_direction(X[col].values, y, self.ratio, self.direction)
                for col in X.columns
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_lift_with_direction)(X[col].values, y, self.ratio, self.direction)
                for col in X.columns
            )

        # 解包结果
        scores = np.array([r[0] for r in results])
        lift_bad = np.array([r[1] for r in results])
        lift_good = np.array([r[2] for r in results])
        best_dirs = [r[3] for r in results]

        # 评分 = |LIFT - 1|
        self.scores_ = pd.Series(scores, index=X.columns)

        # LIFT详情表
        self.lift_detail_ = pd.DataFrame({
            'LIFT_bad': lift_bad,
            'LIFT_good': lift_good,
            'best_direction': best_dirs,
            'score': scores,
        }, index=X.columns)

        # 选择 score >= threshold 的特征
        selected_mask = scores >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 生成剔除原因
        dir_label = {'auto': '自动', 'bad': '找坏人', 'good': '找好人'}
        self._drop_reason = (
            f'LIFT@{self.ratio:.0%} |LIFT-1| < {self.threshold}'
            f'（方向: {dir_label.get(self.direction, self.direction)}）'
        )
