# -*- coding: utf-8 -*-
"""
稳定性感知特征筛选器.

同时考虑特征有效性（IV/KS）和稳定性（PSI），通过综合评分筛选特征，
避免选出区分力强但分布不稳定的特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_iv(x: np.ndarray, y: np.ndarray, regularization: float = 1.0) -> float:
    """计算单个特征的IV值."""
    try:
        s = pd.Series(x)
        valid = ~s.isnull().values
    except Exception:
        valid = ~pd.isnull(x)
    x_v, y_v = x[valid], y[valid]
    if len(x_v) == 0:
        return 0.0
    uniques = np.unique(x_v)
    if len(uniques) <= 1:
        return 0.0

    event = y_v == 1
    nonevent = ~event
    e_tot = np.count_nonzero(event) + 2 * regularization
    ne_tot = np.count_nonzero(nonevent) + 2 * regularization

    iv = 0.0
    for cat in uniques:
        mask = x_v == cat
        e_r = (np.count_nonzero(mask & event) + regularization) / e_tot
        ne_r = (np.count_nonzero(mask & nonevent) + regularization) / ne_tot
        iv += (e_r - ne_r) * np.log(max(e_r, 1e-10) / max(ne_r, 1e-10))
    return float(iv)


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """计算单个特征的PSI."""
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    exp_c = np.maximum(np.histogram(expected, bins=bins)[0], 1)
    act_c = np.maximum(np.histogram(actual, bins=bins)[0], 1)
    exp_r = exp_c / exp_c.sum()
    act_r = act_c / act_c.sum()
    return float(np.sum((act_r - exp_r) * np.log(act_r / exp_r)))


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

    示例::

        >>> selector = StabilityAwareSelector(
        ...     iv_threshold=0.02,
        ...     psi_threshold=0.25,
        ...     oot_df=oot_data,
        ... )
        >>> selector.fit(train_df, y_train)
        >>> X_selected = selector.transform(train_df)
        >>> print(selector.get_detail())
    """

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
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=iv_threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.iv_threshold = iv_threshold
        self.psi_threshold = psi_threshold
        self.score_threshold = score_threshold
        self.iv_weight = iv_weight
        self.psi_weight = psi_weight
        self.oot_df = oot_df
        self.psi_bins = psi_bins
        self.method_name = "稳定性感知筛选"

    # ----------------------------------------------------------
    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入 y 或 X 中包含 '{self.target}' 列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        y = np.asarray(y)

        # 编码类别变量
        X_enc = X.copy()
        for col in X.columns:
            if X[col].dtype.name in ("object", "category"):
                X_enc[col] = pd.factorize(X[col])[0]

        # --- IV ---
        if self.n_jobs == 1:
            iv_vals = np.array([_compute_iv(X_enc[c].values, y) for c in X_enc.columns])
        else:
            iv_vals = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_iv)(X_enc[c].values, y) for c in X_enc.columns
                )
            )

        # --- PSI ---
        if self.oot_df is not None:
            oot = self.oot_df
            if self.target in oot.columns:
                oot = oot.drop(columns=self.target)
            oot_enc = oot.copy()
            for col in oot.columns:
                if oot[col].dtype.name in ("object", "category"):
                    oot_enc[col] = pd.factorize(oot[col])[0]
        else:
            # 随机对半拆分
            n = len(X_enc)
            idx = np.random.permutation(n)
            oot_enc = X_enc.iloc[idx[n // 2:]]
            X_enc_half = X_enc.iloc[idx[: n // 2]]
            # 使用前半作为 expected
            X_enc = X_enc_half

        psi_vals = np.array([
            _compute_psi(
                X_enc[c].dropna().values.astype(float),
                oot_enc[c].dropna().values.astype(float) if c in oot_enc.columns else X_enc[c].dropna().values.astype(float),
                self.psi_bins,
            )
            for c in X.columns
        ])

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
            self.dropped_ = pd.DataFrame({
                "特征": dropped_cols,
                "剔除原因": reasons,
                "IV值": [self.iv_scores_[c] for c in dropped_cols],
                "PSI值": [self.psi_scores_[c] for c in dropped_cols],
                "综合评分": [self.combined_scores_[c] for c in dropped_cols],
            })
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "IV值", "PSI值", "综合评分"])

    def get_detail(self) -> pd.DataFrame:
        """获取所有特征的 IV / PSI / 综合评分明细.

        :return: DataFrame，含 IV、PSI、综合评分、是否入选
        """
        if not hasattr(self, "iv_scores_"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "IV": self.iv_scores_,
            "PSI": self.psi_scores_,
            "综合评分": self.combined_scores_,
            "入选": [c in self.selected_features_ for c in self.iv_scores_.index],
        })
        return df.sort_values("综合评分", ascending=False)
