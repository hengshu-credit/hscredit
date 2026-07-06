"""IV值筛选器.

使用信息价值（IV）进行特征筛选，是金融风控场景的核心筛选方法。
"""

from typing import Union, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_iv_single(
    x: np.ndarray,
    y: np.ndarray,
    regularization: float = 1.0,
) -> float:
    """计算单个特征的 IV 值。

    :param x: 特征值数组
    :param y: 目标变量数组（0=好，1=坏）
    :param regularization: 加性平滑系数，避免除零
    :return: IV 值
    """
    # 处理缺失值，兼容 category / object / numpy 类型
    if isinstance(x, pd.Series):
        has_missing = x.isnull().values
    else:
        try:
            has_missing = pd.Series(x).isnull().values
        except Exception:
            has_missing = pd.isnull(x)

    valid = ~has_missing
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) == 0:
        return 0.0

    uniques = np.unique(x_valid)
    n_cats = len(uniques)
    if n_cats <= 1:
        return 0.0

    event_mask = y_valid == 1
    nonevent_mask = ~event_mask

    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization

    event_rates = np.zeros(n_cats, dtype=np.float64)
    nonevent_rates = np.zeros(n_cats, dtype=np.float64)

    for i, cat in enumerate(uniques):
        mask = x_valid == cat
        event_rates[i] = np.count_nonzero(mask & event_mask) + regularization
        nonevent_rates[i] = np.count_nonzero(mask & nonevent_mask) + regularization

    # 单样本分箱不参与 IV，避免高基数连续变量虚高
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot

    ivs = (event_rates - nonevent_rates) * np.log(
        np.maximum(event_rates, 1e-10) / np.maximum(nonevent_rates, 1e-10)
    )
    ivs[bad_pos] = 0.0

    return np.sum(ivs).item()


class IVSelector(BaseFeatureSelector):
    """IV值筛选器。

    使用信息价值（Information Value）筛选特征。默认直接按原始取值计算 IV；
    当传入 ``binning_params``（空字典也会启用）时，先使用
    :class:`~hscredit.core.binning.OptimalBinning` 分箱，再基于分箱索引计算 IV，
    更适用于连续型变量。

    IV 值解释：

    - < 0.02: 无预测能力
    - 0.02 - 0.1: 弱预测能力
    - 0.1 - 0.3: 中等预测能力
    - 0.3 - 0.5: 强预测能力
    - > 0.5: 极强预测能力（可能过拟合）

    **参数**

    :param threshold: IV 保留阈值，``IV >= threshold`` 的特征被保留，默认为 ``0.02``
    :param target: 目标变量列名，默认为 ``'target'``
    :param regularization: 计算 WOE/IV 时的加性平滑系数，默认为 ``1.0``
    :param binning_params: 可选，传入时在计算 IV 前执行 OptimalBinning 分箱。
        默认参数为 ``{'method': 'mdlp', 'max_n_bins': 10,
        'min_bin_size': 0.01, 'missing_separate': True}``；传入字典中的同名
        配置会覆盖默认值。例如 ``{'method': 'cart', 'max_n_bins': 8}``。
    :param include: 强制保留的特征列表
    :param exclude: 强制剔除的特征列表
    :param force_drop: 强制剔除的特征列表，与 ``exclude`` 合并
    :param n_jobs: IV 计算的并行任务数

    **参考样例**

    >>> selector = IVSelector(threshold=0.02)
    >>> selector.fit(X, y)
    >>> selector = IVSelector(
    ...     threshold=0.02,
    ...     binning_params={'method': 'mdlp', 'max_n_bins': 10}
    ... )
    >>> selector.fit(X, y)

    **引用**

    Information Value 用于变量筛选见 Siddiqi, N. (2006). *Credit Risk
    Scorecards.* Wiley；阈值经验区间（0.02/0.1/0.3/0.5）为业界通行标准。
    """

    _DEFAULT_BINNING_PARAMS = {
        'method': 'mdlp',
        'max_n_bins': 10,
        'min_bin_size': 0.01,
        'missing_separate': True,
    }

    def __init__(
        self,
        threshold: float = 0.02,
        target: str = 'target',
        regularization: float = 1.0,
        binning_params: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target,
            threshold=threshold,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
        )
        self.regularization = regularization
        self.binning_params = binning_params
        self.method_name = 'IV值筛选'

    def _bin_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """用 OptimalBinning 将特征转换为分箱索引。"""
        from ..binning import OptimalBinning

        params = self._DEFAULT_BINNING_PARAMS.copy()
        if self.binning_params:
            params.update(self.binning_params)

        self.binner_ = OptimalBinning(**params)
        self.binner_.fit(X, y)
        X_binned = self.binner_.transform(X, metric='indices')

        if isinstance(X_binned, np.ndarray):
            X_binned = pd.DataFrame(X_binned, columns=X.columns, index=X.index)
        else:
            X_binned = X_binned.reindex(index=X.index, columns=X.columns)

        return X_binned

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合 IV 值筛选器。"""
        self._get_feature_names(X)

        if y is None:
            raise ValueError('IVSelector 计算 IV 时需要传入目标变量 y，或在 X 中包含 target 列。')

        # 传入分箱配置时，先将数据转换为离散的分箱索引。
        X_for_iv = self._bin_features(X, y) if self.binning_params is not None else X

        # 对未分箱的类别型变量进行编码；分箱后的数据一般为数值索引，但保留该逻辑
        # 以兼容分箱结果中的 object/category 列。
        X_encoded = X_for_iv.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype.name in ['object', 'category']:
                X_encoded[col] = pd.factorize(X_encoded[col])[0]

        y_array = np.asarray(y)

        if self.n_jobs == 1:
            iv_values = np.array([
                _compute_iv_single(X_encoded[col].values, y_array, self.regularization)
                for col in X_encoded.columns
            ])
        else:
            iv_values = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_iv_single)(
                        X_encoded[col].values,
                        y_array,
                        self.regularization,
                    )
                    for col in X_encoded.columns
                )
            )

        self.scores_ = pd.Series(iv_values, index=X.columns)

        selected_mask = iv_values >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()

        dropped_cols = X.columns[~selected_mask].tolist()
        if dropped_cols:
            self.dropped_ = pd.DataFrame({
                '特征': dropped_cols,
                '剔除原因': [
                    f'IV值({self.scores_[col]:.4f}) < 阈值({self.threshold})'
                    for col in dropped_cols
                ],
                'IV值': [self.scores_[col] for col in dropped_cols],
                '阈值': [self.threshold] * len(dropped_cols),
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', 'IV值', '阈值'])

    def get_iv_interpretation(self) -> pd.DataFrame:
        """获取 IV 值的中文解释。"""
        if not hasattr(self, 'scores_'):
            return pd.DataFrame()

        def interpret_iv(iv: float) -> str:
            if iv < 0.02:
                return '无预测能力'
            if iv < 0.1:
                return '弱预测能力'
            if iv < 0.3:
                return '中等预测能力'
            if iv < 0.5:
                return '强预测能力'
            return '极强预测能力（可能过拟合）'

        df = pd.DataFrame({
            '特征': self.scores_.index,
            'IV值': self.scores_.values,
            '预测能力': [interpret_iv(iv) for iv in self.scores_.values],
        })
        return df.sort_values('IV值', ascending=False)
