# -*- coding: utf-8 -*-
"""
稳定性感知特征筛选器.

同时考虑特征有效性（IV/KS）和稳定性（PSI），通过综合评分筛选特征，
避免选出区分力强但分布不稳定的特征。

**参考样例**

>>> from hscredit.core.selectors import StabilityAwareSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 训练集特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> oot_data = pd.DataFrame(np.random.randn(500, 5), columns=[f'f{i}' for i in range(5)])  # OOT验证集
>>> selector = StabilityAwareSelector(
...     iv_threshold=0.02,  # IV>0.02表示有预测能力
...     psi_threshold=0.25,  # PSI>0.25表示分布不稳定
...     oot_df=oot_data,    # 传入OOT数据计算PSI
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from .iv_selector import _compute_iv_feature, _compute_iv_single
from .psi_selector import _compute_psi_single
from ...exceptions import ValidationError
from ...utils.parallel import ParallelWorkload


def _compute_iv(x: np.ndarray, y: np.ndarray, regularization: float = 1.0) -> float:
    """计算单个特征的IV值.

    使用信息价值(Information Value)评估特征的区分能力，支持正则化平滑处理。

    :param x: 特征值数组
    :param y: 目标变量数组（0/1）
    :param regularization: 正则化系数，默认1.0，用于平滑分箱占比避免零除
    :return: IV值，类型为float
    """
    return _compute_iv_single(x, y, regularization)


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """计算单个特征的PSI.

    使用群体稳定性指数(Population Stability Index)评估特征在期望分布与实际分布间的差异。

    :param expected: 期望分布数组（训练集/基准期数据）
    :param actual: 实际分布数组（验证集/新数据）
    :param n_bins: 分箱数量，默认10，用于计算各分箱的占比差异
    :return: PSI值，类型为float
    """
    return _compute_psi_single(expected, actual, n_bins=n_bins)


def _compute_stability_feature(task):
    """计算单个特征的 IV 与 PSI。"""
    feature, iv_series, y, expected_values, oot_values, psi_bins = task
    _, iv = _compute_iv_feature((feature, iv_series, y, 1.0))
    psi = _compute_psi(expected_values, oot_values, psi_bins)
    return feature, iv, psi


class StabilityAwareSelector(BaseFeatureSelector):
    """稳定性感知筛选器.

    综合考虑特征区分力（IV）和分布稳定性（PSI），通过加权评分公式选择特征：

        score = iv_weight × IV_normalized − psi_weight × PSI_normalized

    筛选条件（同时满足）:
    1. IV >= iv_threshold
    2. PSI <= psi_threshold
    3. 综合评分 >= score_threshold

    **适用场景:**
    - 模型变量筛选阶段，避免选入"高区分但不稳定"的特征
    - 跨时间段 / OOT 验证时的特征稳健性筛选

    :param iv_threshold: IV下限，默认 0.02
    :param psi_threshold: PSI上限，默认 0.25
    :param score_threshold: 综合评分下限，默认 0.0
    :param iv_weight: IV 在综合评分中的权重，默认 0.6
    :param psi_weight: PSI 在综合评分中的权重，默认 0.4
    :param oot_df: 用于计算 PSI 的 OOT 数据集（DataFrame），
        若不传则使用 fit 时的 X 进行随机对半拆分
    :param psi_bins: PSI 分箱数，默认 10
    :param target: 目标变量列名（当通过DataFrame入参时使用）
    :param n_jobs: 并行数

    **参考样例**

    >>> from hscredit.core.selectors import StabilityAwareSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = pd.Series(np.random.randint(0, 2, 1000))
    >>> oot_data = pd.DataFrame(np.random.randn(500, 5), columns=[f'f{i}' for i in range(5)])
    >>> selector = StabilityAwareSelector(
    ...     iv_threshold=0.02,
    ...     psi_threshold=0.25,
    ...     oot_df=oot_data,
    ... )
    >>> selector.fit(X, y)
    >>> print(selector.selected_features_)

    **注意**

    本筛选器是"稳定性感知"的复合筛选（IV 区分力 + PSI 稳定性加权评分），并非 Meinshausen
    & Bühlmann 的 stability selection（重采样频率法）。IV 与 PSI 的定义与阈值分别见
    :class:`~hscredit.core.selectors.IVSelector` 与
    :class:`~hscredit.core.selectors.PSISelector`。

    **引用**

    IV / PSI 用于评分卡变量筛选与稳定性监控见 Siddiqi, N. (2006).
    *Credit Risk Scorecards.* Wiley。
    """

    method_name = "稳定性感知筛选"

    def __init__(
        self,
        iv_threshold: float = 0.02,
        psi_threshold: float = 0.25,
        score_threshold: float = 0.0,
        iv_weight: float = 0.6,
        psi_weight: float = 0.4,
        oot_df: Optional[pd.DataFrame] = None,
        psi_bins: int = 10,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        random_state: Optional[int] = None,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target=target,
            threshold=iv_threshold,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.iv_threshold = iv_threshold
        self.psi_threshold = psi_threshold
        self.score_threshold = score_threshold
        self.iv_weight = iv_weight
        self.psi_weight = psi_weight
        self.oot_df = oot_df
        self.psi_bins = psi_bins
        self.random_state = random_state

    # ----------------------------------------------------------
    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        if self.iv_threshold < 0 or self.psi_threshold < 0:
            raise ValueError("iv_threshold 和 psi_threshold 不能小于 0")
        if self.iv_weight < 0 or self.psi_weight < 0 or self.iv_weight + self.psi_weight <= 0:
            raise ValueError("iv_weight 和 psi_weight 必须非负且至少一个大于 0")
        if isinstance(self.psi_bins, (bool, np.bool_)) or not isinstance(self.psi_bins, (int, np.integer)) or int(self.psi_bins) < 2:
            raise ValueError("psi_bins 必须是大于等于 2 的整数")
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入 y 或 X 中包含 '{self.target}' 列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        y = np.asarray(y)

        iv_source = X

        # --- PSI ---
        if self.oot_df is not None:
            if not isinstance(self.oot_df, pd.DataFrame):
                raise ValidationError("oot_df 必须是 DataFrame")
            oot = self.oot_df
            if self.target in oot.columns:
                oot = oot.drop(columns=self.target)
            missing = [column for column in X.columns if column not in oot.columns]
            if missing:
                raise ValidationError(f"OOT 数据缺少拟合字段: {missing}")
            oot_source = oot.loc[:, X.columns]
            if self._binner_instance is not None:
                oot_source = self._transform_with_fitted_binner(oot_source)
            expected_source = X
        else:
            # 随机对半拆分
            n = len(X)
            if n < 2:
                raise ValidationError("稳定性筛选至少需要 2 条样本")
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(n)
            expected_source = X.iloc[idx[: n // 2]]
            oot_source = X.iloc[idx[n // 2 :]]

        def iter_tasks():
            for col in X.columns:
                yield col, iv_source[col], y, expected_source[col].values, oot_source[col].values, self.psi_bins

        results = self._parallel_execute(
            _compute_stability_feature,
            iter_tasks(),
            task_labels=X.columns,
            default_backend="loky",
            workload=ParallelWorkload(
                task_count=X.shape[1],
                rows=X.shape[0],
                columns=X.shape[1],
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=10.0,
                capability="process_safe",
                operation="IV与PSI稳定性计算",
            ),
        )
        iv_vals = np.array([iv for _, iv, _ in results])
        psi_vals = np.array([psi for _, _, psi in results])

        # --- 综合评分 ---
        iv_norm = iv_vals / max(iv_vals.max(), 1e-10)
        psi_norm = psi_vals / max(psi_vals.max(), 1e-10)
        combined = self.iv_weight * iv_norm - self.psi_weight * psi_norm

        self.iv_scores_ = pd.Series(iv_vals, index=X.columns)
        self.psi_scores_ = pd.Series(psi_vals, index=X.columns)
        self.combined_scores_ = pd.Series(combined, index=X.columns)
        self.scores_ = self.combined_scores_

        # --- 筛选 ---
        mask = (iv_vals >= self.iv_threshold) & (psi_vals <= self.psi_threshold) & (combined >= self.score_threshold)
        self.selected_features_ = X.columns[mask].tolist()

        dropped_cols = X.columns[~mask].tolist()
        if dropped_cols:
            reasons = []
            for c in dropped_cols:
                parts = []
                if self.iv_scores_[c] < self.iv_threshold:
                    parts.append(f"IV({self.iv_scores_[c]:.4f})<{self.iv_threshold}")
                if self.psi_scores_[c] > self.psi_threshold:
                    parts.append(f"PSI({self.psi_scores_[c]:.4f})>{self.psi_threshold}")
                if self.combined_scores_[c] < self.score_threshold:
                    parts.append(f"综合分({self.combined_scores_[c]:.4f})<{self.score_threshold}")
                reasons.append("; ".join(parts) if parts else "综合不达标")
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": reasons,
                    "IV值": [self.iv_scores_[c] for c in dropped_cols],
                    "PSI值": [self.psi_scores_[c] for c in dropped_cols],
                    "综合评分": [self.combined_scores_[c] for c in dropped_cols],
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "IV值", "PSI值", "综合评分"])

    def get_detail(self) -> pd.DataFrame:
        """获取所有特征的 IV / PSI / 综合评分明细.

        :return: DataFrame，含 IV、PSI、综合评分、是否入选
        """
        if not hasattr(self, "iv_scores_"):
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "IV": self.iv_scores_,
                "PSI": self.psi_scores_,
                "综合评分": self.combined_scores_,
                "入选": [c in self.selected_features_ for c in self.iv_scores_.index],
            }
        )
        return df.sort_values("综合评分", ascending=False)
